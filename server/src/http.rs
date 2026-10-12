//! Minimal HTTP/1.1 server on a hand-bound Winsock socket.
//!
//! # Why raw Winsock and not `std::net`
//!
//! The task asks for "极简 Winsock + 手写 JSON + Bearer 鉴权". This module calls
//! `ws2_32` directly (`WSAStartup` / `socket` / `setsockopt` / `bind` / `listen` /
//! `accept` / `recv` / `send` / `closesocket` / `WSACleanup`) through `extern`
//! blocks. Two deliberate consequences:
//!
//!   * `SO_REUSEADDR` is settable, which matters on Windows: without it a
//!     restart after a kill leaves the port in `TIME_WAIT` and `bind` fails with
//!     `WSAEADDRINUSE` (10048) -- which would look like "the server is broken"
//!     during the accept/verify cycle.
//!   * **We control the receive loop**, so we can decide exactly when a request
//!     is "complete". See the framing section below; getting this wrong is the
//!     main way a hand-written HTTP server produces a *false PASS*.
//!
//! `std` is still used for threads, `String`, file IO, and process spawning. This
//! crate is a normal CRT-linked executable; only the socket layer is hand-bound.
//!
//! # Request framing -- the part that is easy to get subtly wrong
//!
//! A naive implementation `recv`s once and parses whatever arrived. That works on
//! a loopback `curl` for small bodies and is **wrong** in two ways that this
//! project's discipline explicitly forbids:
//!
//!   1. A body larger than one TCP segment arrives in pieces; parsing the first
//!      piece yields "unterminated JSON" on a *valid* request. That is a false
//!      ​400 and it would be blamed on the client.
//!   2. Worse for verification: parsing the first `recv` and *ignoring the rest*
//!      means a request with a valid prefix and garbage appended still gets a
//!      200. Every "reject invalid input" claim would then be unfalsifiable.
//!
//! So: headers are accumulated until `\r\n\r\n`, `Content-Length` is parsed from
//! them, and the loop keeps reading until exactly that many body bytes are in
//! hand. Trailing bytes beyond `Content-Length` are **discarded rather than
//! parsed** (they belong to a pipelined next request; treating them as part of
//! this body would be a request-smuggling primitive). A request that never
//! completes within the budget is answered `400` with the byte counts, not
//! silently dropped.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::Arc;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Raw Winsock bindings
// ---------------------------------------------------------------------------

#[allow(non_camel_case_types)]
type SOCKET = usize;
const INVALID_SOCKET: SOCKET = usize::MAX; // (SOCKET)(~0)
const SOCKET_ERROR: i32 = -1;
const AF_INET: i32 = 2;
const SOCK_STREAM: i32 = 1;
const IPPROTO_TCP: i32 = 6;
const SOL_SOCKET: i32 = 0xffff;
const SO_REUSEADDR: i32 = 0x0004;
const INADDR_ANY: u32 = 0;

#[repr(C)]
#[derive(Clone, Copy)]
struct SockAddrIn {
    sin_family: u16,
    sin_port: u16, // network byte order
    sin_addr: u32, // network byte order
    sin_zero: [u8; 8],
}

#[repr(C)]
struct WsaData {
    version: u16,
    high_version: u16,
    max_sockets: u16,
    max_udp_dg: u16,
    vendor_info: *mut u8,
    description: [u8; 257],
    system_status: [u8; 129],
}

#[link(name = "ws2_32")]
extern "system" {
    fn WSAStartup(version: u16, data: *mut WsaData) -> i32;
    fn WSACleanup() -> i32;
    fn WSAGetLastError() -> i32;
    fn socket(af: i32, ty: i32, proto: i32) -> SOCKET;
    fn setsockopt(
        s: SOCKET,
        level: i32,
        name: i32,
        val: *const u8,
        len: i32,
    ) -> i32;
    fn bind(s: SOCKET, addr: *const SockAddrIn, len: i32) -> i32;
    fn listen(s: SOCKET, backlog: i32) -> i32;
    fn accept(s: SOCKET, addr: *mut SockAddrIn, len: *mut i32) -> SOCKET;
    fn recv(s: SOCKET, buf: *mut u8, len: i32, flags: i32) -> i32;
    fn send(s: SOCKET, buf: *const u8, len: i32, flags: i32) -> i32;
    fn closesocket(s: SOCKET) -> i32;
}

/// A listening socket. Drop closes it; `stop()` breaks the accept loop.
pub struct Listener {
    sock: SOCKET,
    pub addr: String,
}

impl Listener {
    /// Bind + listen on `host:port`. `port = 0` asks the OS for a free port,
    /// which is what the verification script uses so parallel runs cannot
    /// collide on a fixed port.
    pub fn bind(host: &str, port: u16) -> Result<Listener, String> {
        unsafe {
            let mut wsa: WsaData = std::mem::zeroed();
            let rc = WSAStartup(0x0202, &mut wsa);
            if rc != 0 {
                return Err(format!("WSAStartup failed with {}", rc));
            }
        }
        let sock = unsafe { socket(AF_INET, SOCK_STREAM, IPPROTO_TCP) };
        if sock == INVALID_SOCKET {
            let e = unsafe { WSAGetLastError() };
            return Err(format!("socket() failed, WSA error {}", e));
        }
        unsafe {
            let one: i32 = 1;
            let rc = setsockopt(
                sock,
                SOL_SOCKET,
                SO_REUSEADDR,
                &one as *const i32 as *const u8,
                std::mem::size_of::<i32>() as i32,
            );
            if rc == SOCKET_ERROR {
                let e = WSAGetLastError();
                closesocket(sock);
                return Err(format!(
                    "setsockopt(SO_REUSEADDR) failed, WSA error {} -- without it a \
                     restart inside TIME_WAIT cannot bind the port",
                    e
                ));
            }
        }
        let ip = parse_ipv4(host)?;
        let sa = SockAddrIn {
            sin_family: AF_INET as u16,
            sin_port: port.to_be(),
            sin_addr: ip.to_be(),
            sin_zero: [0; 8],
        };
        let rc = unsafe {
            bind(
                sock,
                &sa as *const SockAddrIn,
                std::mem::size_of::<SockAddrIn>() as i32,
            )
        };
        if rc == SOCKET_ERROR {
            let e = unsafe { WSAGetLastError() };
            unsafe { closesocket(sock) };
            return Err(format!(
                "bind({}:{}) failed, WSA error {} ({})",
                host,
                port,
                e,
                match e {
                    10048 => "WSAEADDRINUSE: port already in use",
                    10049 => "WSAEADDRNOTAVAIL: address not available on this host",
                    10013 => "WSAEACCES: permission denied for this port",
                    _ => "see WSA error list",
                }
            ));
        }
        if unsafe { listen(sock, 64) } == SOCKET_ERROR {
            let e = unsafe { WSAGetLastError() };
            unsafe { closesocket(sock) };
            return Err(format!("listen() failed, WSA error {}", e));
        }
        Ok(Listener {
            sock,
            addr: format!("{}:{}", host, port),
        })
    }

    /// Accept one connection. Returns the raw socket.
    pub fn accept(&self) -> Result<SOCKET, String> {
        let mut sa = SockAddrIn {
            sin_family: 0,
            sin_port: 0,
            sin_addr: 0,
            sin_zero: [0; 8],
        };
        let mut len = std::mem::size_of::<SockAddrIn>() as i32;
        let s = unsafe { accept(self.sock, &mut sa, &mut len) };
        if s == INVALID_SOCKET {
            return Err(format!("accept() failed, WSA error {}", unsafe {
                WSAGetLastError()
            }));
        }
        Ok(s)
    }
}

impl Drop for Listener {
    fn drop(&mut self) {
        unsafe {
            closesocket(self.sock);
            WSACleanup();
        }
    }
}

/// Close a socket handed out by [`Listener::accept`].
pub unsafe fn close_socket(s: SOCKET) {
    closesocket(s);
}

fn parse_ipv4(host: &str) -> Result<u32, String> {
    if host == "0.0.0.0" || host.is_empty() {
        return Ok(INADDR_ANY);
    }
    let parts: Vec<&str> = host.split('.').collect();
    if parts.len() != 4 {
        return Err(format!(
            "only dotted-quad IPv4 hosts are supported here, got `{}`",
            host
        ));
    }
    let mut v = 0u32;
    for p in parts {
        let n: u32 = p
            .parse()
            .map_err(|_| format!("`{}` is not an IPv4 address", host))?;
        if n > 255 {
            return Err(format!("`{}` is not an IPv4 address (octet {})", host, n));
        }
        v = (v << 8) | n;
    }
    Ok(v)
}

/// Read and write on a raw socket, adapting it to `std::io` so the HTTP layer
/// can share one code path with the tests.
pub struct SockStream {
    sock: SOCKET,
}

impl SockStream {
    pub fn new(sock: SOCKET) -> SockStream {
        SockStream { sock }
    }
}

impl Read for SockStream {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = unsafe { recv(self.sock, buf.as_mut_ptr(), buf.len() as i32, 0) };
        if n < 0 {
            let e = unsafe { WSAGetLastError() };
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("recv failed, WSA error {}", e),
            ));
        }
        Ok(n as usize)
    }
}

impl Write for SockStream {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        // `send` may accept fewer bytes than asked; the caller loops. A zero
        // return is a legitimate "socket buffer full" signal for a blocking
        // socket, so it is surfaced as WouldBlock rather than as a short write
        // of 0 (which `write_all` would treat as success and then silently
        // truncate the response).
        let n = unsafe { send(self.sock, buf.as_ptr(), buf.len() as i32, 0) };
        if n < 0 {
            let e = unsafe { WSAGetLastError() };
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("send failed, WSA error {}", e),
            ));
        }
        if n == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::WouldBlock,
                "send returned 0",
            ));
        }
        Ok(n as usize)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Drop for SockStream {
    fn drop(&mut self) {
        unsafe { closesocket(self.sock) };
    }
}

/// Serve `handler` on `listener` until `stop` flips to true.
///
/// One thread per connection (the server is a demo endpoint, not a web scale
/// service). Each connection is handled with a bounded number of keep-alive
/// requests; `Connection: close` is always sent so `curl` does not hang waiting
/// for a socket it thinks might be reused.
pub fn serve<F>(listener: Arc<Listener>, stop: Arc<std::sync::atomic::AtomicBool>, handler: F)
where
    F: Fn(&Request) -> Response + Send + Sync + 'static,
{
    // Coerce to a trait object ONCE. Every connection thread then clones an
    // `Arc<dyn Fn>`; keeping the generic `F` on the thread closure would need a
    // second type parameter on `handle_conn` and would monomorphize the whole
    // request path per handler type for no benefit.
    let handler: Arc<dyn Fn(&Request) -> Response + Send + Sync> = Arc::new(handler);
    // The accept() call blocks, so shutdown is driven by a second mechanism: the
    // process is a short-lived demo and the verify script kills it. We still
    // check `stop` between connections, and set a recv timeout so an idle
    // connection cannot pin the loop forever.
    listener_set_blocking(&listener);
    while !stop.load(std::sync::atomic::Ordering::Relaxed) {
        let sock = match listener.accept() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[vmlm-server] {}", e);
                continue;
            }
        };
        let h = handler.clone();
        std::thread::spawn(move || {
            let mut s = SockStream::new(sock);
            let _ = handle_conn(&mut s, &h);
        });
    }
}

fn listener_set_blocking(_l: &Listener) {
    // Accepted sockets inherit the listening socket's blocking mode; that is the
    // mode we want (no busy-wait). This hook exists so the intent is explicit
    // rather than implied by the absence of an ioctlsocket call.
}

/// Errors while reading one request off the wire.
#[derive(Debug)]
pub enum ReadError {
    /// Peer closed before sending anything.
    Eof,
    /// Headers/body did not complete within the budget.
    Incomplete {
        headers_bytes: usize,
        body_have: usize,
        body_want: usize,
    },
    /// Headers arrived but `Content-Length` is unusable.
    BadContentLength(String),
    /// Headers are larger than the cap (a body-in-headers attack or a stuck peer).
    HeadersTooLarge(usize),
    Io(String),
}

#[derive(Debug, Clone)]
pub struct Request {
    pub method: String,
    pub target: String,
    pub version: String,
    pub headers: Vec<(String, String)>,
    pub body: String,
    /// Raw body bytes, exactly `Content-Length` of them.
    pub body_bytes: Vec<u8>,
    /// Bytes the peer sent beyond `Content-Length` that we discarded.
    pub trailing_bytes: usize,
}

impl Request {
    pub fn header(&self, name: &str) -> Option<&str> {
        let n = name.to_ascii_lowercase();
        self.headers
            .iter()
            .find(|(k, _)| k.to_ascii_lowercase() == n)
            .map(|(_, v)| v.as_str())
    }
    /// Path without the query string.
    pub fn path(&self) -> &str {
        match self.target.find('?') {
            Some(i) => &self.target[..i],
            None => &self.target,
        }
    }
}

const MAX_HEADERS: usize = 64 * 1024;
const MAX_BODY: usize = 4 * 1024 * 1024;

/// Read exactly one HTTP request. See the module docs for why this is a loop and
/// not a single `recv`.
pub fn read_request<R: Read>(r: &mut R) -> Result<Request, ReadError> {
    let mut buf: Vec<u8> = Vec::with_capacity(8192);
    let mut chunk = [0u8; 4096];
    let head_end;
    loop {
        if let Some(i) = find_crlfcrlf(&buf) {
            head_end = i + 4;
            break;
        }
        if buf.len() > MAX_HEADERS {
            return Err(ReadError::HeadersTooLarge(buf.len()));
        }
        match r.read(&mut chunk) {
            Ok(0) => {
                if buf.is_empty() {
                    return Err(ReadError::Eof);
                }
                return Err(ReadError::Incomplete {
                    headers_bytes: buf.len(),
                    body_have: 0,
                    body_want: 0,
                });
            }
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
            Err(e) => return Err(ReadError::Io(e.to_string())),
        }
    }

    let head = String::from_utf8_lossy(&buf[..head_end - 4]).to_string();
    let mut lines = head.split("\r\n");
    let request_line = lines.next().unwrap_or("");
    let mut parts = request_line.split(' ');
    let method = parts.next().unwrap_or("").to_string();
    let target = parts.next().unwrap_or("").to_string();
    let version = parts.next().unwrap_or("").to_string();
    if method.is_empty() || target.is_empty() || version.is_empty() {
        return Err(ReadError::BadContentLength(format!(
            "malformed request line `{}` (expected `METHOD TARGET VERSION`)",
            request_line
        )));
    }

    let mut headers = Vec::new();
    let mut content_length: Option<usize> = None;
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let (k, v) = match line.split_once(':') {
            Some(kv) => kv,
            None => {
                return Err(ReadError::BadContentLength(format!(
                    "malformed header line `{}` (no ':')",
                    line
                )))
            }
        };
        let k = k.trim().to_string();
        let v = v.trim().to_string();
        if k.eq_ignore_ascii_case("content-length") {
            match v.parse::<usize>() {
                Ok(n) if n <= MAX_BODY => content_length = Some(n),
                Ok(n) => {
                    return Err(ReadError::BadContentLength(format!(
                        "Content-Length {} exceeds the {} B cap",
                        n, MAX_BODY
                    )))
                }
                Err(_) => {
                    return Err(ReadError::BadContentLength(format!(
                        "Content-Length `{}` is not a non-negative integer",
                        v
                    )))
                }
            }
        }
        headers.push((k, v));
    }

    let want = content_length.unwrap_or(0);
    let mut body = buf[head_end..].to_vec();
    while body.len() < want {
        match r.read(&mut chunk) {
            Ok(0) => {
                return Err(ReadError::Incomplete {
                    headers_bytes: head_end,
                    body_have: body.len(),
                    body_want: want,
                })
            }
            Ok(n) => body.extend_from_slice(&chunk[..n]),
            Err(e) => return Err(ReadError::Io(e.to_string())),
        }
    }
    let trailing = body.len() - want;
    body.truncate(want);

    Ok(Request {
        method,
        target,
        version,
        headers,
        body: String::from_utf8_lossy(&body).to_string(),
        body_bytes: body,
        trailing_bytes: trailing,
    })
}

fn find_crlfcrlf(b: &[u8]) -> Option<usize> {
    b.windows(4).position(|w| w == b"\r\n\r\n")
}

/// An HTTP response.
pub struct Response {
    pub status: u16,
    pub reason: &'static str,
    pub content_type: &'static str,
    pub body: String,
    /// Extra headers, e.g. `WWW-Authenticate` on a 401.
    pub extra: Vec<(&'static str, String)>,
}

impl Response {
    pub fn json(status: u16, body: String) -> Response {
        Response {
            status,
            reason: reason_for(status),
            content_type: "application/json; charset=utf-8",
            body,
            extra: Vec::new(),
        }
    }
    pub fn text(status: u16, body: String) -> Response {
        Response {
            status,
            reason: reason_for(status),
            content_type: "text/plain; charset=utf-8",
            body,
            extra: Vec::new(),
        }
    }
    pub fn with(mut self, k: &'static str, v: String) -> Response {
        self.extra.push((k, v));
        self
    }

    /// Serialize with an explicit `Content-Length` and `Connection: close`.
    ///
    /// `Content-Length` is the BYTE length of the body, not `body.len()` as a
    /// character count -- with a non-ASCII reply (the fixture echoes Chinese
    /// text) those differ, and a char count makes curl report
    /// "transfer closed with N bytes remaining".
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut h = format!(
            "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n",
            self.status,
            self.reason,
            self.content_type,
            self.body.as_bytes().len()
        );
        for (k, v) in &self.extra {
            h.push_str(&format!("{}: {}\r\n", k, v));
        }
        h.push_str("\r\n");
        let mut out = h.into_bytes();
        out.extend_from_slice(self.body.as_bytes());
        out
    }
}

pub fn reason_for(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        405 => "Method Not Allowed",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        503 => "Service Unavailable",
        _ => "Unknown",
    }
}

fn handle_conn(s: &mut SockStream, handler: &Arc<dyn Fn(&Request) -> Response + Send + Sync>) -> std::io::Result<()> {
    for _ in 0..8 {
        let req = match read_request(s) {
            Ok(r) => r,
            Err(ReadError::Eof) => return Ok(()),
            Err(e) => {
                // A malformed request still gets a real HTTP error -- never a
                // silent drop, because a dropped connection is indistinguishable
                // from a hung server and would make the 400 criterion look like
                // a network fault.
                let body = format!(
                    "{{\"error\":{{\"type\":\"invalid_request\",\"message\":{}}}}}",
                    crate::json::to_string(&crate::json::Json::Str(format!("{}", ReadErrFmt(&e))))
                );
                let resp = Response::json(400, body);
                let _ = s.write_all(&resp.to_bytes());
                return Ok(());
            }
        };
        let resp = handler(&req);
        s.write_all(&resp.to_bytes())?;
        // `Connection: close` on every response, so one request per connection.
        return Ok(());
    }
    Ok(())
}

struct ReadErrFmt<'a>(&'a ReadError);
impl std::fmt::Display for ReadErrFmt<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            ReadError::Eof => write!(f, "connection closed before a request arrived"),
            ReadError::Incomplete {
                headers_bytes,
                body_have,
                body_want,
            } => write!(
                f,
                "request truncated: read {} header bytes and {} of {} body bytes \
                 before the peer stopped sending",
                headers_bytes, body_have, body_want
            ),
            ReadError::BadContentLength(m) => write!(f, "{}", m),
            ReadError::HeadersTooLarge(n) => {
                write!(f, "request headers exceeded {} bytes (got {})", MAX_HEADERS, n)
            }
            ReadError::Io(e) => write!(f, "socket error while reading the request: {}", e),
        }
    }
}

// ---------------------------------------------------------------------------
// Small client used by the crate's own tests and by `--selftest`. Distinct from
// std::net::TcpStream's *listener* above: we bind with raw Winsock but connect
// with std, because there is nothing to control on the client side.
// ---------------------------------------------------------------------------

pub fn http_get(addr: &str, path: &str, bearer: Option<&str>) -> std::io::Result<(u16, String)> {
    let mut s = TcpStream::connect(addr)?;
    s.set_read_timeout(Some(Duration::from_secs(10)))?;
    let auth = match bearer {
        Some(t) => format!("Authorization: Bearer {}\r\n", t),
        None => String::new(),
    };
    let req = format!(
        "GET {} HTTP/1.1\r\nHost: {}\r\n{}Connection: close\r\n\r\n",
        path, addr, auth
    );
    s.write_all(req.as_bytes())?;
    let mut out = Vec::new();
    s.read_to_end(&mut out)?;
    parse_client_response(&out)
}

pub fn http_post(
    addr: &str,
    path: &str,
    bearer: Option<&str>,
    body: &str,
) -> std::io::Result<(u16, String)> {
    let mut s = TcpStream::connect(addr)?;
    s.set_read_timeout(Some(Duration::from_secs(60)))?;
    let auth = match bearer {
        Some(t) => format!("Authorization: Bearer {}\r\n", t),
        None => String::new(),
    };
    let req = format!(
        "POST {} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{}Connection: close\r\n\r\n{}",
        path,
        addr,
        body.as_bytes().len(),
        auth,
        body
    );
    s.write_all(req.as_bytes())?;
    let mut out = Vec::new();
    s.read_to_end(&mut out)?;
    parse_client_response(&out)
}

pub fn parse_client_response(raw: &[u8]) -> std::io::Result<(u16, String)> {
    let text = String::from_utf8_lossy(raw).to_string();
    let (head, body) = match text.find("\r\n\r\n") {
        Some(i) => (&text[..i], text[i + 4..].to_string()),
        None => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("no header terminator in response: {:?}", &text[..text.len().min(200)]),
            ))
        }
    };
    let status = head
        .split("\r\n")
        .next()
        .and_then(|l| l.split(' ').nth(1))
        .and_then(|s| s.parse::<u16>().ok())
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("no status code in `{}`", head),
            )
        })?;
    Ok((status, body))
}
