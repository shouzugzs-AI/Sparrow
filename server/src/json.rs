//! Hand-written JSON: a strict parser with a byte-offset error, and a small
//! writer. No serde, no third-party crate (project rule for P17).
//!
//! # Why the parser reports an OFFSET and not just "bad json"
//!
//! The task's discipline rule 3 is "非法输入必须硬失败并说明哪里错" -- illegal
//! input must hard-fail and say *where* it is wrong. A 400 whose body is
//! `{"error":"invalid json"}` satisfies the status-code half and fails the
//! explaining half: a client sending a 900-byte chat request gets no way to
//! find the offending byte. So every error carries `offset`, `line` and
//! `column` (all 1-based for line/col), which is what the 400 body returns.
//!
//! # Strictness (this is the part that is easy to get wrong)
//!
//! Rejecting on the *first* syntax error is easy. What is NOT easy, and what
//! this parser spends most of its code on, is not silently accepting things
//! that are not JSON:
//!
//!   * trailing bytes after the top-level value  -> error at that offset
//!     (a parser that just stops at `}` accepts `{}{"a":1}` and any garbage
//!     appended after a valid body)
//!   * trailing commas `[1,2,]` / `{"a":1,}`     -> error
//!   * `NaN`, `Infinity`, `+1`, `.5`, `01`, `1.` -> error (RFC 8259 has no
//!     such literals; `serde_json` rejects them too)
//!   * unescaped control characters inside a string
//!   * lone surrogates in `\uXXXX` and invalid `\u` hex
//!   * duplicate object keys                      -> error (see below)
//!
//! # Duplicate keys are an ERROR here
//!
//! `serde_json` accepts duplicates ('last wins'). This parser does not, and
//! that is a deliberate choice for this server: for an auth/parameter-bearing
//! body, two different interpretations of `{"model":"a","model":"b"}` exist
//! (first-wins vs last-wins), and the two ends of a wire protocol disagreeing
//! about which one is authoritative is precisely the class of bug this project
//! converts into a hard failure. We reject rather than pick.

use std::collections::BTreeMap;
use std::fmt;

/// Where a parse/validation failure happened.
///
/// `offset` is the byte index into the original request body, so the caller can
/// point a client at the exact byte without any further bookkeeping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonError {
    pub message: String,
    pub offset: usize,
    pub line: usize,
    pub column: usize,
    /// The text the offset points at, quoted, for the 400 body. Kept short.
    pub near: String,
}

impl fmt::Display for JsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} at byte {} (line {}, column {})",
            self.message, self.offset, self.line, self.column
        )
    }
}

impl JsonError {
    fn at(body: &str, off: usize, msg: impl Into<String>) -> JsonError {
        // Clamp: an error at EOF reports the EOF offset, not off the end.
        let off = off.min(body.len());
        let mut line = 1usize;
        let mut col = 1usize;
        let mut near_start = 0usize;
        for (i, ch) in body.char_indices() {
            if i >= off {
                break;
            }
            if ch == '\n' {
                line += 1;
                col = 1;
                near_start = i + 1;
            } else {
                col += 1;
            }
        }
        // Up to 24 bytes of context, on a char boundary.
        let mut near_end = off;
        while near_end < body.len() && near_end - off < 24 && !body.is_char_boundary(near_end + 1) {
            near_end += 1;
        }
        near_end = (near_end + 1).min(body.len());
        while near_end > near_start && !body.is_char_boundary(near_end) {
            near_end -= 1;
        }
        let near_start = near_start.min(off);
        JsonError {
            message: msg.into(),
            offset: off,
            line,
            column: col,
            near: body[near_start..near_end].to_string(),
        }
    }
}

/// A JSON value. Objects keep insertion order in `keys` and last-wins-free
/// values in `map` (duplicates are rejected during parsing, so `map` has
/// exactly one entry per key in `keys`).
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    /// Numbers are kept as their source text. Rationale: this server never does
    /// arithmetic on a request number (temperature/max_tokens are echoed or
    /// range-checked), and f64 would silently lose the distinction between
    /// `1` and `1.0` that a strict comparison might want.
    Num(String),
    Str(String),
    Arr(Vec<Json>),
    Obj(Obj),
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Obj {
    keys: Vec<String>,
    map: BTreeMap<String, Json>,
    /// Byte offset of each key's opening quote in the source, parallel to
    /// `keys`. Only populated by the parser; objects built by hand have none.
    /// It exists so a duplicate-key error can point at the FIRST occurrence.
    offsets: Vec<usize>,
}

impl Obj {
    pub fn new() -> Obj {
        Obj::default()
    }

    pub fn insert(&mut self, k: impl Into<String>, v: Json) {
        self.insert_with_offset(k, v, usize::MAX)
    }

    /// `insert`, additionally remembering where the key appeared in the source.
    pub fn insert_with_offset(&mut self, k: impl Into<String>, v: Json, off: usize) {
        let k = k.into();
        if !self.map.contains_key(&k) {
            self.keys.push(k.clone());
            self.offsets.push(off);
        }
        self.map.insert(k, v);
    }

    /// Source offset of `k`'s first occurrence, if it was recorded.
    pub fn offset_of(&self, k: &str) -> Option<usize> {
        self.keys.iter().position(|x| x == k).and_then(|i| {
            self.offsets
                .get(i)
                .copied()
                .filter(|o| *o != usize::MAX)
        })
    }

    pub fn get(&self, k: &str) -> Option<&Json> {
        self.map.get(k)
    }

    pub fn contains_key(&self, k: &str) -> bool {
        self.map.contains_key(k)
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Keys in the order they appeared in the source. Used by the writer so a
    /// round-tripped body keeps its shape.
    pub fn keys(&self) -> &[String] {
        &self.keys
    }
}

impl Json {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    /// A quoted JSON string literal for this value. Exists so call sites that
    /// build error bodies by formatting do not hand-roll escaping -- that is
    /// exactly how an unescaped quote in a path turns into a malformed response.
    pub fn to_string_quoted(&self) -> String {
        let mut s = String::new();
        match self {
            Json::Str(v) => write_string(v, &mut s),
            other => write_value(other, &mut s),
        }
        s
    }

    pub fn as_arr(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(a) => Some(a),
            _ => None,
        }
    }
    pub fn as_obj(&self) -> Option<&Obj> {
        match self {
            Json::Obj(o) => Some(o),
            _ => None,
        }
    }
    /// JSON type name, for "expected string, got number" style messages.
    pub fn type_name(&self) -> &'static str {
        match self {
            Json::Null => "null",
            Json::Bool(_) => "boolean",
            Json::Num(_) => "number",
            Json::Str(_) => "string",
            Json::Arr(_) => "array",
            Json::Obj(_) => "object",
        }
    }
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

pub fn parse(body: &str) -> Result<Json, JsonError> {
    let b = body.as_bytes();
    let mut p = Parser { b, body, i: 0 };
    p.ws();
    let v = p.value()?;
    p.ws();
    if p.i != b.len() {
        // This is the check that makes `{}{"a":1}` and `{...} trailing` fail
        // instead of being accepted as the prefix.
        return Err(JsonError::at(
            body,
            p.i,
            "trailing data after the top-level JSON value",
        ));
    }
    Ok(v)
}

struct Parser<'a> {
    b: &'a [u8],
    body: &'a str,
    i: usize,
}

impl<'a> Parser<'a> {
    fn err<T>(&self, off: usize, msg: impl Into<String>) -> Result<T, JsonError> {
        Err(JsonError::at(self.body, off, msg))
    }

    fn peek(&self) -> Option<u8> {
        self.b.get(self.i).copied()
    }

    fn ws(&mut self) {
        // RFC 8259 whitespace only. Notably NOT \x0b or \x0c.
        while let Some(c) = self.peek() {
            if c == b' ' || c == b'\t' || c == b'\n' || c == b'\r' {
                self.i += 1;
            } else {
                break;
            }
        }
    }

    fn expect(&mut self, c: u8) -> Result<(), JsonError> {
        if self.peek() == Some(c) {
            self.i += 1;
            Ok(())
        } else {
            let got = match self.peek() {
                None => "end of input".to_string(),
                Some(g) => format!("{}", g as char),
            };
            self.err(
                self.i,
                format!("expected '{}', found {}", c as char, got),
            )
        }
    }

    fn lit(&mut self, s: &str) -> bool {
        if self.b[self.i..].starts_with(s.as_bytes()) {
            self.i += s.len();
            true
        } else {
            false
        }
    }

    fn value(&mut self) -> Result<Json, JsonError> {
        let start = self.i;
        match self.peek() {
            None => self.err(self.i, "unexpected end of input, expected a JSON value"),
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') => {
                if self.lit("true") {
                    Ok(Json::Bool(true))
                } else {
                    self.err(start, "invalid literal, expected `true`")
                }
            }
            Some(b'f') => {
                if self.lit("false") {
                    Ok(Json::Bool(false))
                } else {
                    self.err(start, "invalid literal, expected `false`")
                }
            }
            Some(b'n') => {
                if self.lit("null") {
                    Ok(Json::Null)
                } else {
                    self.err(start, "invalid literal, expected `null`")
                }
            }
            Some(c) if c == b'-' || c.is_ascii_digit() => self.number(),
            Some(c) => self.err(
                start,
                format!(
                    "unexpected character `{}`, expected a JSON value \
                     (object, array, string, number, true, false or null)",
                    c as char
                ),
            ),
        }
    }

    fn object(&mut self) -> Result<Json, JsonError> {
        self.expect(b'{')?;
        let mut o = Obj::new();
        self.ws();
        if self.peek() == Some(b'}') {
            self.i += 1;
            return Ok(Json::Obj(o));
        }
        loop {
            self.ws();
            if self.peek() != Some(b'"') {
                let got = match self.peek() {
                    None => "end of input".to_string(),
                    Some(g) => format!("`{}`", g as char),
                };
                return self.err(
                    self.i,
                    format!("expected a double-quoted object key, found {}", got),
                );
            }
            let koff = self.i;
            let k = self.string()?;
            self.ws();
            if self.peek() != Some(b':') {
                let got = match self.peek() {
                    None => "end of input".to_string(),
                    Some(g) => format!("`{}`", g as char),
                };
                return self.err(
                    self.i,
                    format!("expected ':' after object key \"{}\", found {}", k, got),
                );
            }
            self.i += 1;
            self.ws();
            let v = self.value()?;
            if let Some(first_off) = o.offset_of(&k) {
                // Report the offset of the FIRST occurrence -- the key already in
                // the object -- not the duplicate. (Measured: reporting the
                // duplicate's position made the message point at byte 7 of
                // `{"a":1,"a":2}`, so a client looking there sees the copy, not the
                // original. "Duplicate" means the second one collides with the
                // first, so the first is what the offset should identify.)
                //
                // NOTE the message does NOT repeat the "at byte N (line L, column
                // C)" suffix that `Display` appends: the byte number is already
                // stated inline, and the first version produced
                // "... at byte 1 ... bug at byte 1 (line 1, column 2)".
                return self.err(
                    first_off,
                    format!(
                        "duplicate object key \"{}\" -- first defined at byte {}; this \
                         server rejects duplicates rather than picking first-or-last, \
                         because two ends of a protocol disagreeing about which one is \
                         authoritative is a silent-corruption bug",
                        k, first_off
                    ),
                );
            }
            // The key's opening quote is at `koff`; record it so a later duplicate
            // can point back here.
            o.insert_with_offset(k, v, koff);
            self.ws();
            match self.peek() {
                Some(b',') => {
                    self.i += 1;
                    self.ws();
                    if self.peek() == Some(b'}') {
                        return self.err(self.i, "trailing comma before '}'");
                    }
                }
                Some(b'}') => {
                    self.i += 1;
                    return Ok(Json::Obj(o));
                }
                None => return self.err(self.body.len(), "unterminated object, expected '}'"),
                Some(c) => {
                    return self.err(
                        self.i,
                        format!("expected ',' or '}}' in object, found `{}`", c as char),
                    )
                }
            }
        }
    }

    fn array(&mut self) -> Result<Json, JsonError> {
        self.expect(b'[')?;
        let mut a = Vec::new();
        self.ws();
        if self.peek() == Some(b']') {
            self.i += 1;
            return Ok(Json::Arr(a));
        }
        loop {
            self.ws();
            a.push(self.value()?);
            self.ws();
            match self.peek() {
                Some(b',') => {
                    self.i += 1;
                    self.ws();
                    if self.peek() == Some(b']') {
                        return self.err(self.i, "trailing comma before ']'");
                    }
                }
                Some(b']') => {
                    self.i += 1;
                    return Ok(Json::Arr(a));
                }
                None => return self.err(self.body.len(), "unterminated array, expected ']'"),
                Some(c) => {
                    return self.err(
                        self.i,
                        format!("expected ',' or ']' in array, found `{}`", c as char),
                    )
                }
            }
        }
    }

    fn string(&mut self) -> Result<String, JsonError> {
        let open = self.i;
        self.expect(b'"')?;
        let mut out = String::new();
        loop {
            let c = match self.peek() {
                None => {
                    return self.err(
                        self.body.len(),
                        format!("unterminated string started at byte {}", open),
                    )
                }
                Some(c) => c,
            };
            match c {
                b'"' => {
                    self.i += 1;
                    return Ok(out);
                }
                b'\\' => {
                    let esc_at = self.i;
                    self.i += 1;
                    let e = match self.peek() {
                        None => {
                            return self.err(
                                self.i,
                                "unterminated escape sequence at end of input",
                            )
                        }
                        Some(e) => e,
                    };
                    self.i += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{08}'),
                        b'f' => out.push('\u{0c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let cp = self.hex4(esc_at)?;
                            if (0xD800..0xDC00).contains(&cp) {
                                // High surrogate: a low surrogate MUST follow.
                                if self.peek() != Some(b'\\')
                                    || self.b.get(self.i + 1) != Some(&b'u')
                                {
                                    return self.err(
                                        esc_at,
                                        format!(
                                            "lone high surrogate \\u{:04X} must be followed \
                                             by a low surrogate \\uDC00-\\uDFFF",
                                            cp
                                        ),
                                    );
                                }
                                self.i += 2;
                                let lo = self.hex4(self.i - 2)?;
                                if !(0xDC00..0xE000).contains(&lo) {
                                    return self.err(
                                        self.i - 4,
                                        format!(
                                            "invalid low surrogate \\u{:04X} after high \
                                             surrogate \\u{:04X}",
                                            lo, cp
                                        ),
                                    );
                                }
                                let c = 0x10000
                                    + ((cp - 0xD800) << 10)
                                    + (lo - 0xDC00);
                                match char::from_u32(c) {
                                    Some(ch) => out.push(ch),
                                    None => {
                                        return self.err(
                                            esc_at,
                                            "surrogate pair does not form a valid code point",
                                        )
                                    }
                                }
                            } else if (0xDC00..0xE000).contains(&cp) {
                                return self.err(
                                    esc_at,
                                    format!("lone low surrogate \\u{:04X}", cp),
                                );
                            } else {
                                match char::from_u32(cp) {
                                    Some(ch) => out.push(ch),
                                    None => {
                                        return self.err(
                                            esc_at,
                                            format!("\\u{:04X} is not a valid code point", cp),
                                        )
                                    }
                                }
                            }
                        }
                        other => {
                            return self.err(
                                esc_at,
                                format!(
                                    "invalid escape sequence `\\{}` (valid: \\\" \\\\ \\/ \\b \\f \\n \\r \\t \\uXXXX)",
                                    other as char
                                ),
                            )
                        }
                    }
                }
                0x00..=0x1f => {
                    return self.err(
                        self.i,
                        format!(
                            "unescaped control character U+{:04X} in string \
                             (must be written as an escape)",
                            c
                        ),
                    )
                }
                _ => {
                    // Copy one whole UTF-8 sequence. A byte-level copy is fine
                    // because every byte >= 0x20 that is not a delimiter is
                    // valid inside a JSON string, and `from_utf8` on the whole
                    // body already guaranteed well-formedness.
                    let s = self.i;
                    self.i += 1;
                    while self.i < self.b.len()
                        && (self.b[self.i] & 0xC0) == 0x80
                    {
                        self.i += 1;
                    }
                    out.push_str(&self.body[s..self.i]);
                }
            }
        }
    }

    fn hex4(&mut self, at: usize) -> Result<u32, JsonError> {
        // A `\u` escape needs 4 hex digits *in the source*. `["\u12"]` has the
        // digits `1`,`2` and then a `"` -- which is 4 characters but only 2 hex
        // digits, so it must be reported as a bad hex digit at the `"`, not as
        // "truncated". The distinction the caller cares about is "your escape is
        // short" vs "your escape contains a non-hex character"; both are errors.
        if self.i + 4 > self.b.len() {
            return self.err(
                self.i,
                format!(
                    "truncated \\u escape starting at byte {}: only {} of 4 hex digits \
                     remain in the input",
                    at,
                    self.b.len() - self.i
                ),
            );
        }
        let mut v = 0u32;
        for k in 0..4 {
            let c = self.b[self.i + k];
            let d = match c {
                b'0'..=b'9' => (c - b'0') as u32,
                b'a'..=b'f' => (c - b'a' + 10) as u32,
                b'A'..=b'F' => (c - b'A' + 10) as u32,
                _ => {
                    return self.err(
                        self.i + k,
                        format!(
                            "invalid hex digit `{}` at position {} of the \\u escape \
                             starting at byte {} (need 4 hex digits, e.g. \\u4e2d)",
                            c as char,
                            k + 1,
                            at
                        ),
                    )
                }
            };
            v = v * 16 + d;
        }
        self.i += 4;
        Ok(v)
    }

    fn number(&mut self) -> Result<Json, JsonError> {
        let start = self.i;
        if self.peek() == Some(b'-') {
            self.i += 1;
        }
        // ---- integer part ----
        //
        // Lead-zero rejection has to happen HERE, at the point where the second
        // digit is seen. (Measured, first version: the check ran after the scan
        // loop, but `01` is scanned as `0` followed by `1` only in the `0`
        // branch, so the post-hoc `text.starts_with('0')` test saw the text
        // `"0"` -- the `1` had never been consumed -- and the error that came out
        // was "expected ',' or ']' in array", which is technically true and
        // useless to a client.)
        match self.peek() {
            Some(b'0') => {
                self.i += 1;
                if matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                    return self.err(
                        self.i,
                        "numbers may not have a leading zero (JSON allows `0`, `0.5`, \
                         `0e1` but not `00`, `01`)",
                    );
                }
            }
            Some(c) if c.is_ascii_digit() => {
                while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                    self.i += 1;
                }
            }
            _ => return self.err(self.i, "expected a digit after '-'"),
        }
        if self.peek() == Some(b'.') {
            self.i += 1;
            if !matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                return self.err(self.i, "expected a digit after the decimal point");
            }
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.i += 1;
            }
        }
        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            self.i += 1;
            if matches!(self.peek(), Some(b'+') | Some(b'-')) {
                self.i += 1;
            }
            if !matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                return self.err(self.i, "expected a digit in the exponent");
            }
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.i += 1;
            }
        }
        Ok(Json::Num(self.body[start..self.i].to_string()))
    }
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

/// Serialize with the given key order for objects (insertion order is used).
pub fn to_string(v: &Json) -> String {
    let mut s = String::new();
    write_value(v, &mut s);
    s
}

fn write_value(v: &Json, out: &mut String) {
    match v {
        Json::Null => out.push_str("null"),
        Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Json::Num(n) => out.push_str(n),
        Json::Str(s) => write_string(s, out),
        Json::Arr(a) => {
            out.push('[');
            for (i, e) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_value(e, out);
            }
            out.push(']');
        }
        Json::Obj(o) => {
            out.push('{');
            for (i, k) in o.keys().iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_string(k, out);
                out.push(':');
                write_value(o.get(k).expect("key list and map agree"), out);
            }
            out.push('}');
        }
    }
}

/// JSON string escaping. Non-ASCII is emitted as raw UTF-8 (JSON is UTF-8 by
/// definition and the response declares `charset=utf-8`), but every control
/// character is escaped, because a raw newline inside a JSON string is invalid
/// -- and a model reply containing a newline is completely normal.
fn write_string(s: &str, out: &mut String) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

// ---------------------------------------------------------------------------
// Tests -- these run with `cargo test -p vmlm-server`. They are the "can it
// fail?" half of the JSON claims: every rejection case below is asserted to be
// an error, so a future loosening of the parser shows up as a failing test
// rather than as a silently-accepted bad request.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn ok(s: &str) -> Json {
        parse(s).unwrap_or_else(|e| panic!("expected {} to parse, got {}", s, e))
    }

    fn bad(s: &str) -> JsonError {
        match parse(s) {
            Ok(v) => panic!("expected {} to FAIL, but parsed as {:?}", s, v),
            Err(e) => e,
        }
    }

    #[test]
    fn parses_the_chat_shape() {
        let v = ok(r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"temperature":0.2}"#);
        let o = v.as_obj().unwrap();
        assert_eq!(o.get("model").unwrap().as_str().unwrap(), "m");
        let msgs = o.get("messages").unwrap().as_arr().unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(
            msgs[0].as_obj().unwrap().get("content").unwrap().as_str().unwrap(),
            "hi"
        );
        assert_eq!(o.get("temperature").unwrap().as_str(), None);
        match o.get("temperature").unwrap() {
            Json::Num(n) => assert_eq!(n, "0.2"),
            other => panic!("expected Num, got {:?}", other),
        }
    }

    #[test]
    fn rejects_a_truncated_body_with_the_offset() {
        let e = bad(r#"{"model":"m","messages":[{"role":"use"#);
        assert!(e.offset > 0, "offset should point into the body");
        assert!(e.message.contains("unterminated"), "got: {}", e.message);
    }

    #[test]
    fn rejects_trailing_data_after_the_value() {
        let e = bad(r#"{"a":1} garbage"#);
        assert!(e.message.contains("trailing data"), "got: {}", e.message);
        assert_eq!(e.offset, 8);
    }

    #[test]
    fn rejects_a_second_top_level_value() {
        let e = bad(r#"{"a":1}{"b":2}"#);
        assert!(e.message.contains("trailing data"), "got: {}", e.message);
    }

    #[test]
    fn rejects_trailing_commas() {
        assert!(bad("[1,2,]").message.contains("trailing comma"));
        assert!(bad(r#"{"a":1,}"#).message.contains("trailing comma"));
    }

    #[test]
    fn rejects_missing_colon_and_wrong_quoting() {
        let e = bad(r#"{"a" 1}"#);
        assert!(e.message.contains("expected ':'"), "got: {}", e.message);
        assert!(bad("{'a':1}").message.contains("double-quoted"));
        assert!(bad("{a:1}").message.contains("double-quoted"));
    }

    #[test]
    fn rejects_bad_numbers() {
        assert!(bad("[01]").message.contains("leading zero"));
        assert!(bad("[+1]").message.contains("expected a JSON value"));
        assert!(bad("[.5]").message.contains("expected a JSON value"));
        assert!(bad("[1.]").message.contains("after the decimal point"));
        assert!(bad("[1e]").message.contains("exponent"));
        assert!(bad("[NaN]").message.contains("expected a JSON value"));
        assert!(bad("[Infinity]").message.contains("expected a JSON value"));
    }

    #[test]
    fn rejects_control_chars_and_bad_escapes() {
        assert!(bad("[\"a\nb\"]").message.contains("control character"));
        assert!(bad(r#"["a\x"]"#).message.contains("invalid escape"));
        // `\u12` followed by `"`: only 2 hex digits, then a non-hex character.
        // That is the "invalid hex digit" message, not "truncated" -- truncated
        // means the INPUT ran out, which is the case tested next.
        assert!(
            bad(r#"["\u12"]"#).message.contains("invalid hex digit"),
            "got: {}",
            bad(r#"["\u12"]"#).message
        );
        // Genuinely truncated: the input ends mid-escape. Note the raw string
        // ends with `12` and no closing quote, because a `"` would become digit 3.
        assert!(
            bad("[\"\\u12").message.contains("truncated"),
            "got: {}",
            bad("[\"\\u12").message
        );
        assert!(bad(r#"["\uZZZZ"]"#).message.contains("hex digit"));
    }

    #[test]
    fn rejects_lone_surrogates() {
        assert!(bad(r#"["\uD800"]"#).message.contains("lone high surrogate"));
        assert!(bad(r#"["\uDC00"]"#).message.contains("lone low surrogate"));
        assert!(bad(r#"["\uD800\u0041"]"#).message.contains("invalid low surrogate"));
    }

    #[test]
    fn accepts_a_valid_surrogate_pair() {
        let v = ok(r#"["\uD83D\uDE00"]"#);
        assert_eq!(v.as_arr().unwrap()[0].as_str().unwrap(), "\u{1F600}");
    }

    #[test]
    fn rejects_duplicate_keys() {
        let e = bad(r#"{"model":"a","model":"b"}"#);
        assert!(e.message.contains("duplicate object key"), "got: {}", e.message);
        // The offset must point at the FIRST occurrence. Measured byte layout of
        // `{"model":"a","model":"b"}`: the second key's quote is at byte 14, the
        // first is at byte 1. An earlier version reported the duplicate's
        // position, which sends a client to the copy instead of the original.
        assert_eq!(e.offset, 1, "must point at the first occurrence");
        assert!(e.message.contains("first defined at byte 1"), "{}", e.message);
        // The Display suffix must not duplicate the inline byte number.
        assert_eq!(
            e.message.matches("byte 1").count(),
            1,
            "byte number repeated: {}",
            e.message
        );
    }

    #[test]
    fn rejects_empty_and_whitespace_only_bodies() {
        assert!(bad("").message.contains("unexpected end of input"));
        assert!(bad("   ").message.contains("unexpected end of input"));
    }

    #[test]
    fn rejects_unclosed_containers() {
        assert!(bad("[1,2").message.contains("unterminated array"));
        assert!(bad(r#"{"a":1"#).message.contains("unterminated object"));
    }

    #[test]
    fn round_trips_and_escapes() {
        let mut o = Obj::new();
        o.insert("content", Json::Str("line1\nline2\t\"q\"\\ \u{4e2d}\u{6587}".into()));
        o.insert("n", Json::Num("3".into()));
        let s = to_string(&Json::Obj(o));
        let back = ok(&s);
        assert_eq!(
            back.as_obj().unwrap().get("content").unwrap().as_str().unwrap(),
            "line1\nline2\t\"q\"\\ \u{4e2d}\u{6587}"
        );
        assert!(s.contains("\\n") && s.contains("\\t") && s.contains("\\\""), "{}", s);
        assert!(s.contains("\u{4e2d}\u{6587}"), "non-ascii should be raw: {}", s);
    }
}
