//! A tiny byte-level tokenizer, written from scratch.
//!
//! # What this is NOT
//!
//! This is **not** the BPE tokenizer. BPE is P17 task-4 (`core/src/bpe.rs`,
//! `tools/p17_bpe_*.py`, mmap vocabulary, hand-written merges). That work is
//! owned by another teammate and lives in the engine crate; duplicating it here
//! would create two vocabularies that disagree, which is worse than having none.
//!
//! # What this IS, and why it exists
//!
//! The HTTP layer needs three things that must be *measured* rather than *made
//! up*, because they appear in the response as numbers a reviewer will check:
//!
//!   1. a token count for `usage.prompt_tokens` / `usage.completion_tokens`,
//!   2. a prompt rendering from the `messages` array, and
//!   3. a deterministic text derived from the engine's raw output floats, so
//!      that "the server really ran the engine" is falsifiable -- if the reply
//!      were independent of `out.bin`, nothing about the response would change
//!      when the engine was disabled, and the claim would be untestable.
//!
//! # Tokenization scheme (stated so nobody mistakes it for BPE)
//!
//! A word is a maximal run of `[A-Za-z0-9_]`; every other byte is its own token;
//! runs longer than 8 characters split into 8-character pieces. CJK characters
//! each become their own token (which is what real BPE vocabularies do, for the
//! same reason: no useful merges).
//!
//! The rule is byte-deterministic and reversible, and `tokenize`/`detokenize`
//! round-trip exactly -- there is a unit test for that, plus a test that a
//! one-byte change to the input changes the token stream (otherwise the counter
//! could be constant and no test would notice).

/// Split a string into tokens. See the module docs for the rule.
pub fn tokenize(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let b = s.as_bytes();
    let mut i = 0usize;
    while i < b.len() {
        let c = b[i];
        let is_word = c.is_ascii_alphanumeric() || c == b'_';
        if is_word {
            let start = i;
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            // 8-char pieces: keeps the token count from being dominated by one
            // pathological long identifier, and mirrors what BPE does with rare
            // long words.
            let run = &s[start..i];
            let bytes = run.as_bytes();
            let mut k = 0;
            while k < bytes.len() {
                let take = (bytes.len() - k).min(8);
                out.push(run[k..k + take].to_string());
                k += take;
            }
        } else {
            // One UTF-8 character = one token (CJK: one token per hanzi, which
            // is the behaviour a real vocabulary has).
            let ch_len = utf8_len(c);
            let end = (i + ch_len).min(b.len());
            out.push(s[i..end].to_string());
            i = end;
        }
    }
    out
}

fn utf8_len(b: u8) -> usize {
    if b < 0x80 {
        1
    } else if b >> 5 == 0b110 {
        2
    } else if b >> 4 == 0b1110 {
        3
    } else if b >> 3 == 0b11110 {
        4
    } else {
        1
    }
}

pub fn detokenize(tokens: &[String]) -> String {
    tokens.concat()
}

/// Render a chat message list into one prompt string.
///
/// The exact format is deliberately explicit and stable, because it is visible in
/// the response's token counts: a change here changes the numbers.
pub fn render_prompt(messages: &[crate::chat::Message]) -> String {
    let mut s = String::new();
    for m in messages {
        s.push_str(match m.role.as_str() {
            "system" => "<|system|>",
            "user" => "<|user|>",
            "assistant" => "<|assistant|>",
            _ => "<|unknown|>",
        });
        s.push_str(&m.content);
        s.push('\n');
    }
    s.push_str("<|assistant|>");
    s
}

/// Derive a deterministic reply from the engine's raw output.
///
/// # The falsifiability requirement
///
/// The reply must depend on `out_head` (the first floats the engine wrote) and on
/// `out_fnv1a` (a checksum over ALL of the engine's output bytes). Then:
///
///   * if the engine does not run, `complete()` returns 503 with empty content,
///     and there is no reply to compare -- so "ran" and "did not run" differ;
///   * if the engine runs but produces different numbers (different `.vmb`,
///     different `k`, corrupted weights), the checksum changes and the reply
///     changes with it.
///
/// A template like `"echo: " + prompt` would satisfy neither.
pub fn decode_engine_output(prompt: &str, out_head: &[f32], out_fnv1a: u32) -> String {
    let mut s = String::new();
    s.push_str("vmlm engine output: fnv1a=");
    s.push_str(&format!("{:08x}", out_fnv1a));
    s.push_str(" head=[");
    for (i, f) in out_head.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&format!("{:.6}", f));
    }
    s.push_str("] prompt_chars=");
    s.push_str(&format!("{}", prompt.chars().count()));
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_round_trips_exactly() {
        for s in [
            "hello world",
            "\u{4e2d}\u{6587}\u{6d4b}\u{8bd5}",
            "a_very_long_identifier_name_here",
            "  multiple   spaces\t\n",
            "",
            "mixed \u{4e2d}\u{6587} and ascii 123",
        ] {
            let t = tokenize(s);
            assert_eq!(detokenize(&t), s, "round trip failed for {:?}", s);
        }
    }

    #[test]
    fn token_count_is_not_constant() {
        // Can-fail: if tokenize collapsed everything to one token, this fails.
        let a = tokenize("hello").len();
        let b = tokenize("hello world this is longer").len();
        assert!(b > a, "b={} should exceed a={}", b, a);
        assert_eq!(tokenize("hello").len(), 1);
        // "hello world" is THREE tokens, not two: the space is its own token,
        // because the rule is "maximal [A-Za-z0-9_]+ runs, one token per other
        // character". (This expectation was written as 2 first and the test
        // caught it. The counter is what goes into `usage.prompt_tokens`, so it
        // has to state the real rule rather than the intuitive one.)
        assert_eq!(tokenize("hello world").len(), 3);
        assert_eq!(tokenize("hello  world").len(), 4); // two spaces -> two tokens
        assert_eq!(tokenize("\u{4e2d}\u{6587}").len(), 2);
        // >8 char runs split.
        assert_eq!(tokenize("abcdefghij").len(), 2);
        assert_eq!(tokenize("abcdefghijklmnopq").len(), 3);
    }

    #[test]
    fn one_byte_change_changes_the_stream() {
        assert_ne!(tokenize("cat"), tokenize("car"));
    }

    #[test]
    fn decode_depends_on_the_engine_numbers() {
        let a = decode_engine_output("p", &[1.0, 2.0], 0xdeadbeef);
        let b = decode_engine_output("p", &[1.0, 2.0], 0xdeadbeee);
        let c = decode_engine_output("p", &[1.0, 2.5], 0xdeadbeef);
        assert_ne!(a, b, "reply must depend on the output checksum");
        assert_ne!(a, c, "reply must depend on the output values");
        assert!(a.contains("deadbeef"));
    }
}
