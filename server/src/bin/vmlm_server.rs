//! `vmlm-server` -- the P17-P4 HTTP service.
//!
//! Usage:
//!
//! ```text
//! vmlm-server --token <TOKEN> [--port 8080] [--host 127.0.0.1]
//!             [--engine <vmlm-runtime.exe>] [--model <model.vmb>]
//!             [--allow-anonymous]
//! ```
//!
//! Endpoints:
//!
//! ```text
//!   GET  /health                     -> 200 always (no auth): liveness + engine status
//!                                      This is the only unauthenticated route, and it
//!                                      exists so the verify script can wait for the
//!                                      socket to be up WITHOUT needing a token -- a
//!                                      readiness probe that itself required auth would
//!                                      make "401 for bad auth" untestable.
//!   GET  /v1/models                  -> 200, auth required
//!   POST /v1/chat/completions        -> 200 / 400 / 401 / 503
//!   anything else                    -> 404
//! ```
//!
//! Exit codes: 0 clean shutdown; 2 bad arguments; 3 refused to start without a
//! token; 4 could not bind the port. Argument errors print WHAT was wrong, per the
//! project rule.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use vmlm_server::auth;
use vmlm_server::chat;
use vmlm_server::engine::{Engine, EngineConfig};
use vmlm_server::http::{self, Listener, Request, Response};
use vmlm_server::json::{Json, Obj};

struct Args {
    host: String,
    port: u16,
    token: Option<String>,
    allow_anonymous: bool,
    engine: Option<String>,
    model: Option<String>,
    verbose: bool,
}

fn usage() -> String {
    "\
vmlm-server -- VMLM HTTP service (P17-P4)

USAGE:
    vmlm-server --token <TOKEN> [OPTIONS]

REQUIRED:
    --token <TOKEN>        Bearer token every /v1 request must present.
                           Or set VMLM_TOKEN. There is no default: see
                           --allow-anonymous for why.

OPTIONS:
    --host <IP>            Bind address (default 127.0.0.1)
    --port <N>             Bind port (default 8080; 0 = let the OS choose)
    --engine <PATH>        vmlm-runtime.exe to invoke (default: the bench build)
    --model <PATH>         .vmb to run (default examples/p17_http/model.vmb)
    --allow-anonymous      Explicitly disable bearer auth (for the can-fail
                           control in tools/p17_http_verify.py)
    --verbose              Log every request to stderr
    -h, --help             This text

EXIT CODES:
    0  clean shutdown      2  bad arguments
    3  no token configured 4  could not bind
"
    .to_string()
}

fn parse_args() -> Result<Args, (String, i32)> {
    let mut a = Args {
        host: "127.0.0.1".into(),
        port: 8080,
        token: std::env::var("VMLM_TOKEN").ok().filter(|s| !s.is_empty()),
        allow_anonymous: false,
        engine: None,
        model: None,
        verbose: false,
    };
    let v: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < v.len() {
        let k = v[i].as_str();
        let mut next = |what: &str| -> Result<String, (String, i32)> {
            if i + 1 >= v.len() {
                Err((
                    format!("`{}` requires a value ({})", k, what),
                    2,
                ))
            } else {
                i += 1;
                Ok(v[i].clone())
            }
        };
        match k {
            "-h" | "--help" => {
                print!("{}", usage());
                std::process::exit(0);
            }
            "--host" => a.host = next("an IPv4 address")?,
            "--port" => {
                let p = next("a number in 0..=65535")?;
                a.port = p.parse().map_err(|_| {
                    (
                        format!("--port `{}` is not an integer in 0..=65535", p),
                        2,
                    )
                })?;
            }
            "--token" => {
                let t = next("the bearer token")?;
                if t.is_empty() {
                    return Err((
                        "--token must not be empty (an empty token would authenticate \
                         `Authorization: Bearer ` for every client)"
                            .into(),
                        2,
                    ));
                }
                a.token = Some(t);
            }
            "--engine" => a.engine = Some(next("a path to vmlm-runtime.exe")?),
            "--model" => a.model = Some(next("a path to a .vmb file")?),
            "--allow-anonymous" => a.allow_anonymous = true,
            "--verbose" => a.verbose = true,
            other => {
                return Err((
                    format!(
                        "unknown argument `{}`\n\n{}",
                        other,
                        usage()
                    ),
                    2,
                ))
            }
        }
        i += 1;
    }
    if a.token.is_none() && !a.allow_anonymous {
        return Err((
            "refusing to start without a bearer token.\n  \
             Pass --token <TOKEN> (or set VMLM_TOKEN), or pass --allow-anonymous \
             explicitly if you really want an open server.\n  \
             There is deliberately no implicit default: a server that silently \
             runs without auth makes the 401 criterion untestable."
                .into(),
            3,
        ));
    }
    if a.token.is_some() && a.allow_anonymous {
        eprintln!(
            "[vmlm-server] WARNING: --allow-anonymous overrides --token; \
             no request will be authenticated"
        );
        // ⚠️ THE TOKEN MUST ACTUALLY BE CLEARED, not merely warned about.
        // Measured: the first version warned and then kept `token = Some(..)`, so
        // `auth::check` still compared against it and the server returned 401 for
        // every request without a header. The verify script's "auth disabled"
        // control caught it (expected 200, got 401) -- which is exactly what that
        // control exists for. A warning that does not change behaviour is worse
        // than no warning: it asserts something untrue about the running process.
        a.token = None;
    }
    Ok(a)
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err((msg, code)) => {
            eprintln!("[vmlm-server] error: {}\n", msg);
            std::process::exit(code);
        }
    };

    let root = vmlm_server::find_root();
    let cfg = EngineConfig::resolve(args.engine.as_deref(), args.model.as_deref(), &root);
    let engine = Arc::new(std::sync::Mutex::new(Engine::load(cfg)));
    {
        let e = engine.lock().unwrap();
        eprintln!(
            "[vmlm-server] engine exe:  {}  ({})",
            e.cfg.exe.display(),
            if e.cfg.exe.exists() { "present" } else { "MISSING" }
        );
        eprintln!(
            "[vmlm-server] model:       {}  ({})",
            e.cfg.model.display(),
            if e.cfg.model.exists() { "present" } else { "MISSING" }
        );
        match e.status() {
            vmlm_server::engine::EngineStatus::Unavailable { reason, .. } => {
                eprintln!("[vmlm-server] engine state: UNPROVEN -- {}", reason);
            }
            _ => eprintln!("[vmlm-server] engine state: ready"),
        }
    }

    let listener = match Listener::bind(&args.host, args.port) {
        Ok(l) => Arc::new(l),
        Err(e) => {
            eprintln!("[vmlm-server] error: {}", e);
            std::process::exit(4);
        }
    };
    // 0 means "OS chose one", so report the real port.
    let real_port = {
        if args.port == 0 {
            // We do not getsockname here; the verify script always passes a
            // concrete port. Say so rather than printing a wrong number.
            eprintln!(
                "[vmlm-server] note: --port 0 is not supported for reporting; \
                 pass a concrete port"
            );
        }
        args.port
    };
    let base = format!("http://{}:{}", args.host, real_port);
    eprintln!("[vmlm-server] listening on {}", base);
    eprintln!(
        "[vmlm-server] auth: {}",
        if args.token.is_some() {
            "bearer required on /v1/*"
        } else {
            "DISABLED (--allow-anonymous)"
        }
    );

    let token = Arc::new(args.token.clone());
    let verbose = args.verbose;
    let base_for_handler = base.clone();
    let counter = Arc::new(AtomicU64::new(1));

    let handler = {
        let token = token.clone();
        let engine = engine.clone();
        let base = base_for_handler.clone();
        let counter = counter.clone();
        move |req: &Request| -> Response {
            let id = format!("{:016x}", counter.fetch_add(1, Ordering::Relaxed));
            let resp = route(req, &token, &engine, &base, &id);
            if verbose {
                eprintln!(
                    "[vmlm-server] {} {} -> {} ({} B)",
                    req.method,
                    req.target,
                    resp.status,
                    resp.body.as_bytes().len()
                );
            }
            resp
        }
    };

    let stop = Arc::new(AtomicBool::new(false));
    // Ctrl-C: flip `stop` so the accept loop exits between connections.
    {
        let stop = stop.clone();
        let _ = std::thread::spawn(move || {
            // No signal crate; a blocking accept is interrupted by Ctrl-C at the
            // process level, which is how the verify script terminates us (it
            // sends a taskkill). Keeping this thread documents the intent.
            std::thread::sleep(std::time::Duration::from_secs(3600));
            stop.store(true, Ordering::Relaxed);
        });
    }

    http::serve(listener.clone(), stop, handler);
}

fn route(
    req: &Request,
    token: &Option<String>,
    engine: &Arc<std::sync::Mutex<Engine>>,
    base: &str,
    id: &str,
) -> Response {
    let path = req.path();

    // ---- unauthenticated liveness -----------------------------------------
    if path == "/health" {
        let e = engine.lock().unwrap();
        let mut o = Obj::new();
        o.insert("status", Json::Str("ok".into()));
        o.insert("auth", Json::Str(if token.is_some() { "bearer" } else { "anonymous" }.into()));
        o.insert(
            "engine_status",
            Json::Str(e.status().tag().into()),
        );
        if let vmlm_server::engine::EngineStatus::Unavailable { reason, .. } = e.status() {
            o.insert("engine_reason", Json::Str(reason.clone()));
        }
        o.insert("vmb_bytes", Json::Num(e.vmb.bytes.to_string()));
        o.insert("vmb_slots", Json::Num(e.vmb.n_slots.to_string()));
        return Response::json(200, vmlm_server::json::to_string(&Json::Obj(o)));
    }

    // ---- auth on everything under /v1 -------------------------------------
    if path.starts_with("/v1/") || path == "/v1" {
        let hdr = req.header("authorization");
        // NOTE: only a rejection is an error. `NotConfigured` (i.e.
        // --allow-anonymous) is an OK outcome and must NOT fall into the 401 arm
        // below -- a server that says "anonymous access is enabled" in the error
        // body while returning 401 is refusing exactly what it permits. The
        // verify script's auth-disabled control caught that.
        match auth::check(hdr, token.as_deref()) {
            Ok(_) => {}
            Err(err) => {
                let mut e = Obj::new();
                e.insert("type", Json::Str("invalid_request_error".into()));
                e.insert("code", Json::Str("invalid_api_key".into()));
                e.insert("message", Json::Str(err.message()));
                if matches!(err, auth::AuthError::WrongToken) {
                    e.insert(
                        "hint",
                        Json::Str(
                            "the token itself is compared, not its hash; check for \
                             trailing whitespace or a shell-mangled value"
                                .into(),
                        ),
                    );
                }
                let mut root = Obj::new();
                root.insert("error", Json::Obj(e));
                return Response::json(401, vmlm_server::json::to_string(&Json::Obj(root))).with(
                    "WWW-Authenticate",
                    "Bearer realm=\"vmlm\"".to_string(),
                );
            }
        }
    } else {
        let body = format!(
            "{{\"error\":{{\"type\":\"not_found\",\"message\":{},\"path\":{}}}}}",
            Json::Str(format!("no route for {} {}", req.method, path)).to_string_quoted(),
            Json::Str(path.to_string()).to_string_quoted()
        );
        return Response::json(404, body);
    }

    match (req.method.as_str(), path) {
        ("GET", "/v1/models") => {
            let e = engine.lock().unwrap();
            let mut m = Obj::new();
            m.insert("id", Json::Str("vmlm-sparse".into()));
            m.insert("object", Json::Str("model".into()));
            m.insert("owned_by", Json::Str("vmlm".into()));
            let mut root = Obj::new();
            root.insert("object", Json::Str("list".into()));
            root.insert("data", Json::Arr(vec![Json::Obj(m)]));
            root.insert(
                "vmlm",
                Json::Obj({
                    let mut j = Obj::new();
                    j.insert("engine_status", Json::Str(e.status().tag().into()));
                    j.insert("vmb_bytes", Json::Num(e.vmb.bytes.to_string()));
                    j.insert("n_slots", Json::Num(e.vmb.n_slots.to_string()));
                    j.insert("n_const", Json::Num(e.vmb.n_const.to_string()));
                    j
                }),
            );
            Response::json(200, vmlm_server::json::to_string(&Json::Obj(root)))
        }
        ("POST", "/v1/chat/completions") => {
            // NOTE ON FRAMING: `req.body` holds exactly `Content-Length` bytes.
            // Anything the peer sent past that was counted into
            // `req.trailing_bytes` and DISCARDED by the framer -- it must not be
            // folded into the body, because then a valid JSON body followed by
            // garbage would be parsed as valid and the 400 criterion could not
            // fail. That behaviour is asserted by
            // `tools/p17_http_verify.py --case trailing`.
            match chat::validate(&req.body) {
                Err(rej) => Response::json(400, chat::reject_body(&rej)),
                Ok(creq) => {
                    let mut e = engine.lock().unwrap();
                    let (status, body) = chat::complete(&creq, &mut e, base, id);
                    let mut r = Response::json(status, body);
                    if status == 503 {
                        r = r.with("Retry-After", "1".to_string());
                    }
                    r
                }
            }
        }
        ("POST", p) if p.starts_with("/v1/") => {
            let body = format!(
                "{{\"error\":{{\"type\":\"not_found\",\"message\":{}}}}}",
                Json::Str(format!("no POST route for {}", p)).to_string_quoted()
            );
            Response::json(404, body)
        }
        (m, p) if p.starts_with("/v1/") => {
            let body = format!(
                "{{\"error\":{{\"type\":\"method_not_allowed\",\"message\":{}}}}}",
                Json::Str(format!("{} is not allowed on {}", m, p)).to_string_quoted()
            );
            Response::json(405, body).with("Allow", "POST, GET".to_string())
        }
        _ => Response::json(
            404,
            format!(
                "{{\"error\":{{\"type\":\"not_found\",\"message\":{}}}}}",
                Json::Str(format!("no route for {} {}", req.method, path)).to_string_quoted()
            ),
        ),
    }
}
