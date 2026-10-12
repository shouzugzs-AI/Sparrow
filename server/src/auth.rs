//! Bearer-token authentication.
//!
//! # The decision that matters: an unconfigured token is a STARTUP ERROR
//!
//! The tempting default is "if no token is configured, allow everything" --
//! convenient for a demo. It is refused here, because it makes the 401 criterion
//! unfalsifiable: run the server without `--token`, get 200 for everything, and
//! you cannot tell "auth works" from "auth is disabled". Worse, it is the classic
//! way an auth check ships switched off.
//!
//! So the server **refuses to start** without a token (`--token`, or `VMLM_TOKEN`),
//! unless `--allow-anonymous` is passed explicitly. The verification script always
//! passes a token, and runs one extra case with `--allow-anonymous` so that
//! "auth is on" and "auth is off" produce *different* observed status codes.
//!
//! # Comparison
//!
//! `==` on `str` short-circuits at the first differing byte, which leaks the
//! token's prefix length through timing. That is not a realistic attack over
//! loopback against a demo server, but the constant-time comparison is four lines
//! and needs no allocation, so there is no reason to keep the weaker one. (It is
//! also *tested* -- a comparison that always returns true would pass a naive
//! "correct token works" test.)

/// Why a request was not authenticated.
///
/// # `NotConfigured` is NOT an authentication failure
///
/// The first version of this module had a variant `AnonymousAllowed` that
/// `check` returned when the server had no token. The HTTP layer treated every
/// `Err` as a 401, so `--allow-anonymous` produced
///
/// ```text
///   {"error":{...,"message":"anonymous access is enabled"}}   with status 401
/// ```
///
/// which is self-contradictory: the server announces that anonymous access is
/// enabled and simultaneously refuses the request. The verify script's
/// auth-disabled control caught it (expected 200, got 401).
///
/// The bug was a modelling error: "this server has no token configured" is a
/// property of the SERVER, not a rejection of the REQUEST. So it is now a
/// separate variant and the caller must decide -- here, it means "allow", and the
/// 401 path only ever sees the three genuine rejections.
#[derive(Debug, PartialEq, Eq)]
pub enum AuthOutcome {
    /// The request presented a valid bearer credential.
    Authenticated,
    /// The server has no token configured (explicit `--allow-anonymous`).
    NotConfigured,
}

/// Why a request was rejected.
#[derive(Debug, PartialEq, Eq)]
pub enum AuthError {
    /// No `Authorization` header at all.
    Missing,
    /// Header present but not a `Bearer` credential.
    NotBearer(String),
    /// Bearer credential present but wrong.
    WrongToken,
}

impl AuthError {
    pub fn message(&self) -> String {
        match self {
            AuthError::Missing => "no `Authorization` header was sent; expected \
                 `Authorization: Bearer <token>`"
                .into(),
            AuthError::NotBearer(h) => format!(
                "the `Authorization` header must use the Bearer scheme, got `{}`",
                h
            ),
            AuthError::WrongToken => {
                "the bearer token does not match the server's configured token".into()
            }
        }
    }
}

/// Constant-time byte comparison.
pub fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        // Length is not secret (it is the configured token's length), and
        // returning here is what every real implementation does.
        return false;
    }
    let mut diff = 0u8;
    for i in 0..a.len() {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

/// Check the `Authorization` header value against `expected`.
///
/// `expected == None` means the server was explicitly started with
/// `--allow-anonymous`: the request is allowed, and the outcome says WHY, so the
/// server can log/report the distinction instead of pretending a credential was
/// verified.
pub fn check(header: Option<&str>, expected: Option<&str>) -> Result<AuthOutcome, AuthError> {
    let expected = match expected {
        Some(t) => t,
        None => return Ok(AuthOutcome::NotConfigured),
    };
    let h = match header {
        None => return Err(AuthError::Missing),
        Some(h) => h.trim(),
    };
    if h.is_empty() {
        return Err(AuthError::Missing);
    }
    // Scheme match is case-insensitive (RFC 7235 says the scheme is), the
    // credential is not.
    let (scheme, cred) = match h.split_once(' ') {
        Some((s, c)) => (s, c.trim()),
        None => (h, ""),
    };
    if !scheme.eq_ignore_ascii_case("bearer") {
        return Err(AuthError::NotBearer(h.to_string()));
    }
    if ct_eq(cred.as_bytes(), expected.as_bytes()) {
        Ok(AuthOutcome::Authenticated)
    } else {
        Err(AuthError::WrongToken)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_the_right_token() {
        let ae = Ok(AuthOutcome::Authenticated);
        assert_eq!(check(Some("Bearer s3cret"), Some("s3cret")), ae);
        assert_eq!(check(Some("bearer s3cret"), Some("s3cret")), ae);
        assert_eq!(check(Some("BEARER s3cret"), Some("s3cret")), ae);
        assert_eq!(check(Some("  Bearer s3cret  "), Some("s3cret")), ae);
    }

    #[test]
    fn rejects_missing_short_and_wrong() {
        assert_eq!(check(None, Some("s3cret")), Err(AuthError::Missing));
        assert_eq!(check(Some(""), Some("s3cret")), Err(AuthError::Missing));
        assert_eq!(
            check(Some("Basic s3cret"), Some("s3cret")),
            Err(AuthError::NotBearer("Basic s3cret".into()))
        );
        assert_eq!(check(Some("s3cret"), Some("s3cret")), Err(AuthError::NotBearer("s3cret".into())));
        assert_eq!(
            check(Some("Bearer s3cre"), Some("s3cret")),
            Err(AuthError::WrongToken)
        );
        assert_eq!(
            check(Some("Bearer s3cretx"), Some("s3cret")),
            Err(AuthError::WrongToken)
        );
        assert_eq!(
            check(Some("Bearer S3CRET"), Some("s3cret")),
            Err(AuthError::WrongToken)
        );
        // An empty configured token must not mean "everything works".
        assert_eq!(check(Some("Bearer x"), Some("")), Err(AuthError::WrongToken));
    }

    #[test]
    fn ct_eq_is_a_real_comparison() {
        assert!(ct_eq(b"abc", b"abc"));
        assert!(!ct_eq(b"abc", b"abd"));
        assert!(!ct_eq(b"abc", b"ab"));
        assert!(!ct_eq(b"", b"a"));
        assert!(ct_eq(b"", b""));
        // The can-fail control for a broken `ct_eq` that always returned true:
        // differ in the LAST byte, which is where a truncated loop would show up.
        assert!(!ct_eq(b"aaaa", b"aaab"));
    }

    #[test]
    fn anonymous_mode_is_not_the_same_as_a_pass() {
        // The distinction the module docs promise. `NotConfigured` is an OK
        // outcome, NOT an error -- and it is a DIFFERENT value from
        // `Authenticated`, so a caller that wants to log "no credential was
        // verified" can. (An earlier version returned Err here, and the HTTP
        // layer -- which maps Err to 401 -- then refused the very requests that
        // `--allow-anonymous` was supposed to permit.)
        assert_eq!(
            check(Some("Bearer anything"), None),
            Ok(AuthOutcome::NotConfigured)
        );
        assert_eq!(check(None, None), Ok(AuthOutcome::NotConfigured));
        assert_ne!(
            check(None, None),
            Ok(AuthOutcome::Authenticated),
            "an unconfigured server must not claim a credential was verified"
        );
    }
}
