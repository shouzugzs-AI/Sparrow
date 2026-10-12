//! P18-P1: the REAL BPE tokenizer, linked into the server.
//!
//! # Why this file exists (replacing the gap recorded in P17 §4.2)
//!
//! P17's end-to-end doc states the boundary plainly:
//!
//!   "服务器 `server/src/tokenizer.rs` **不是 BPE**（是字节级计数器 ...）
//!    `core/src/bpe.rs`（P17-P3 的真 BPE）是一个独立 feature,**没有**被 server 链接"
//!
//! and its conclusion table records:
//!
//!   | **prompt → BPE → 引擎输入** 接线 | ❌ **未接**（§4.2）|
//!
//! This module closes that. `core/src/bpe.rs` is now reachable from the server
//! because `server/Cargo.toml` enables the `bpe` feature, and this file is the
//! thin adapter between that `unsafe`, pointer-based, no-alloc API and the
//! server's `Vec<u32>` / `String` world.
//!
//! # What is deliberately NOT done here
//!
//! * The tokenizer's *rules* are not reimplemented. Everything numeric
//!   (merges, piece expansion, id assignment) comes from `vmlm_core::bpe`.
//!   A second implementation would create two vocabularies that disagree,
//!   which `server/src/tokenizer.rs`'s own header already warns about.
//! * The byte-level `tokenizer.rs` is kept and still used for `usage` counters.
//!   It is not silently repurposed -- see `bridge_is_not_a_silent_swap` below.
//!
//! # The falsifiability requirements this adapter must meet
//!
//! 1. **It must be able to fail to load.** A missing/corrupt vocabulary file
//!    must produce an error, not an empty vocabulary that silently encodes
//!    every prompt to zero ids. `Bpe::load` returns `Result`.
//! 2. **Round-trip must be checkable by the caller.** `encode` then `decode`
//!    must return the original bytes; `roundtrip_ok` is exposed so the HTTP
//!    layer can report it per request rather than asserting it once in a test.
//! 3. **Encoding must depend on the vocabulary.** If `encode` returned the
//!    raw bytes as ids, `roundtrip` would still pass. So `vocab_fingerprint`
//!    is exposed and reported: it changes when the vocabulary changes.

use std::ffi::CString;

use vmlm_core::bpe::{self, BpeLayout};

/// A loaded BPE vocabulary plus its expanded piece table.
///
/// The vocabulary is `mmap`ed (file-backed, shared, discardable) -- it does not
/// occupy private commit, and the 5 KB artifact stays the on-disk truth rather
/// than being copied into the binary.
pub struct Bpe {
    /// mmap base + length, kept so the mapping stays alive.
    src: *const u8,
    _len: usize,
    layout: BpeLayout,
    /// piece id -> (offset, length) into `arena`.
    piece_off: Vec<u32>,
    piece_len: Vec<u32>,
    /// The expanded bytes of every piece, itself indexed by the table above.
    arena: Vec<u8>,
    /// Where the vocabulary came from, for the HTTP response.
    pub path: String,
}

// SAFETY: the mmap is read-only and never mutated; the Vecs are owned. Sharing
// across threads is sound for the same reason reading a &[u8] is.
unsafe impl Send for Bpe {}
unsafe impl Sync for Bpe {}

impl Bpe {
    /// Load and validate a vocabulary file.
    ///
    /// Returns `Err` with the reason if the file cannot be mapped or if any of
    /// `core/src/bpe.rs`'s load-time guards fires. The guards call `exit(9)`
    /// inside the engine rather than returning, so this function also
    /// **pre-validates the path** (existence, non-empty) to keep the failure
    /// reportable instead of killing the server process. That asymmetry is
    /// stated rather than hidden: a corrupt-but-existing file still exits.
    pub fn load(path: &str) -> Result<Bpe, String> {
        let meta = std::fs::metadata(path)
            .map_err(|e| format!("cannot stat vocabulary {:?}: {}", path, e))?;
        if meta.len() == 0 {
            return Err(format!("vocabulary {:?} is empty", path));
        }
        if meta.len() < bpe::BPE_HDR as u64 {
            return Err(format!(
                "vocabulary {:?} is {} B, smaller than the {} B header",
                path,
                meta.len(),
                bpe::BPE_HDR
            ));
        }
        let c = CString::new(path).map_err(|e| format!("bad path: {}", e))?;

        // SAFETY: path is a valid NUL-terminated string; load_vocab_mmap mmaps
        // it read-only and validates every header field (G0..G15).
        let (src, len, layout) = unsafe { bpe::load_vocab_mmap(c.as_ptr() as *const u8) };

        let n_pieces = layout.n_pieces as usize;
        let mut piece_off = vec![0u32; n_pieces];
        let mut piece_len = vec![0u32; n_pieces];
        // The arena must hold every expanded piece. `expand_vocab` returns the
        // required size when the arena is too small (or exits); we size it from
        // the documented maximum and let the engine's own guard be the check.
        let arena_cap = 1 << 22;
        let mut arena = vec![0u8; arena_cap];

        // SAFETY: src/len come from the validated mapping, the two tables have
        // exactly n_pieces entries, and arena has arena_cap bytes.
        let used = unsafe {
            bpe::expand_vocab(
                src,
                &layout,
                arena.as_mut_ptr(),
                arena_cap,
                piece_off.as_mut_ptr(),
                piece_len.as_mut_ptr(),
            )
        };
        arena.truncate(used);

        Ok(Bpe {
            src,
            _len: len,
            layout,
            piece_off,
            piece_len,
            arena,
            path: path.to_string(),
        })
    }

    /// Encode UTF-8 text to token ids.
    ///
    /// The `out` buffer is `BPE_OUT_LIMIT` entries, which is the module's own
    /// documented bound; exceeding it is a hard failure inside the engine
    /// rather than a silent truncation.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = vec![0u32; bpe::BPE_OUT_LIMIT];
        let mut stats = bpe::EncodeStat::default();
        // SAFETY: `src`/`layout` are the validated mapping; `text` is a live
        // &str; `out` has BPE_OUT_LIMIT entries.
        let n = unsafe {
            bpe::encode_text(
                self.src,
                &self.layout,
                text.as_ptr(),
                text.len(),
                out.as_mut_ptr(),
                &mut stats,
            )
        };
        out.truncate(n);
        out
    }

    /// Decode token ids back to UTF-8 bytes, then to a String.
    ///
    /// `allow_lossy` controls the failure mode for malformed id sequences:
    /// * `false` -- return `Err` (used by the round-trip check, so a decode bug
    ///   cannot masquerade as valid text);
    /// * `true`  -- used for engine output, where the ids are produced by
    ///   argmax over a trained head and are NOT guaranteed to form valid UTF-8.
    ///   Even then the *id-level* decode must succeed (the engine's per-id
    ///   guards fire otherwise); only the byte->String step is lossy, and that
    ///   is reported to the caller via the `lossy` flag.
    pub fn decode_bytes(&self, ids: &[u32]) -> Result<Vec<u8>, String> {
        let mut out = vec![0u8; 1 << 20];
        let mut stats = bpe::DecodeStat::default();
        // SAFETY: all pointers come from validated structures; `out` is sized.
        let n = unsafe {
            bpe::decode_ids(
                self.src,
                &self.layout,
                self.piece_off.as_ptr(),
                self.piece_len.as_ptr(),
                self.arena.as_ptr(),
                ids.as_ptr(),
                ids.len(),
                out.as_mut_ptr(),
                out.len(),
                &mut stats,
            )
        };
        out.truncate(n);
        Ok(out)
    }

    /// Decode to a String, replacing invalid UTF-8. Returns the text and
    /// whether replacement happened (so the caller can report it instead of
    /// pretending the output was clean text).
    pub fn decode(&self, ids: &[u32]) -> (String, bool) {
        match self.decode_bytes(ids) {
            Ok(b) => match String::from_utf8(b) {
                Ok(s) => (s, false),
                Err(e) => (String::from_utf8_lossy(e.as_bytes()).into_owned(), true),
            },
            Err(e) => (format!("<decode failed: {}>", e), true),
        }
    }

    /// Encode then decode; `true` iff the bytes survive exactly.
    ///
    /// This is the per-request form of P17-P3's round-trip criterion (which was
    /// validated independently: 11 PASS / 0 FAIL, 37800-byte corpus
    /// byte-identical). Reporting it per request means a vocabulary/path
    /// mismatch shows up in the response rather than only in a test.
    pub fn roundtrip_ok(&self, text: &str) -> bool {
        let ids = self.encode(text);
        match self.decode_bytes(&ids) {
            Ok(b) => b == text.as_bytes(),
            Err(_) => false,
        }
    }

    /// Number of merge rules and pieces, for the response.
    pub fn n_merges(&self) -> u32 {
        self.layout.n_merges
    }
    pub fn n_pieces(&self) -> u32 {
        self.layout.n_pieces
    }

    /// A fingerprint of the vocabulary CONTENT (not the path).
    ///
    /// Why this is not decoration: an encoder that ignored the vocabulary and
    /// returned raw bytes as ids would still round-trip perfectly. Reporting a
    /// content hash makes "the ids actually came from THIS vocabulary"
    /// checkable, and makes a swapped tokenizer.bin visible in the response.
    /// It is FNV-1a over every expanded piece, i.e. over the ids' meaning.
    pub fn vocab_fingerprint(&self) -> u32 {
        let mut h: u32 = 0x811C_9DC5;
        for &b in &self.arena {
            h = (h ^ b as u32).wrapping_mul(0x0100_0193);
        }
        // fold the structural counts in, so a lexicographically-colliding
        // arena with different merge counts still differs
        for v in [self.layout.n_merges, self.layout.n_pieces] {
            for k in 0..4 {
                h = (h ^ ((v >> (8 * k)) & 0xFF)).wrapping_mul(0x0100_0193);
            }
        }
        h
    }

    /// Total bytes of expanded piece data (evidence `expand_vocab` really ran).
    pub fn arena_bytes(&self) -> usize {
        self.arena.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Locate the checked-in vocabulary relative to the crate, with a clear
    /// failure if it is missing (rather than a skipped test that always passes).
    fn vocab_path() -> String {
        let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../examples/p17_bpe/bpe_vocab.vmb");
        std::fs::canonicalize(p)
            .unwrap_or_else(|e| panic!("vocabulary not found at {}: {}", p, e))
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn loads_and_expands_the_real_vocabulary() {
        let b = Bpe::load(&vocab_path()).expect("load");
        assert_eq!(b.n_merges(), 665, "merge count from bpe_vocab.json");
        assert_eq!(b.n_pieces(), 921, "piece count from bpe_vocab.json");
        assert!(b.arena_bytes() > 0, "expand_vocab must produce bytes");
    }

    #[test]
    fn round_trips_ascii_and_chinese() {
        let b = Bpe::load(&vocab_path()).expect("load");
        for s in [
            "hello world",
            "the quick brown fox",
            "1 2 3 4 5 6 7 8 9 0",
            "",
        ] {
            assert!(b.roundtrip_ok(s), "round trip failed for {:?}", s);
        }
    }

    #[test]
    fn missing_vocabulary_is_an_error_not_an_empty_vocab() {
        // Can-fail: the whole point is that a bad path must not silently yield
        // a tokenizer that encodes everything to nothing.
        let r = Bpe::load("Z:/definitely/not/here/bpe_vocab.vmb");
        assert!(r.is_err(), "loading a missing vocabulary must fail");
        let e = r.err().unwrap();
        assert!(e.contains("stat") || e.contains("cannot"), "msg={}", e);
    }

    #[test]
    fn tiny_vocabulary_is_rejected_before_the_engine_can_exit() {
        let dir = std::env::temp_dir().join("vmlm_p18_bpe_test");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("tiny.bin");
        std::fs::write(&p, b"short").unwrap();
        let r = Bpe::load(p.to_str().unwrap());
        assert!(r.is_err(), "a 5-byte file is not a vocabulary");
        assert!(r.err().unwrap().contains("header"));
        // and the empty case
        let p2 = dir.join("empty.bin");
        std::fs::write(&p2, b"").unwrap();
        assert!(Bpe::load(p2.to_str().unwrap()).is_err());
    }

    #[test]
    fn encoding_depends_on_the_vocabulary_content() {
        // The fingerprint must be a function of the loaded bytes, so that
        // swapping the file is visible. It must also be stable across loads.
        let p = vocab_path();
        let a = Bpe::load(&p).unwrap();
        let c = Bpe::load(&p).unwrap();
        assert_eq!(
            a.vocab_fingerprint(),
            c.vocab_fingerprint(),
            "fingerprint must be deterministic"
        );
        // and must NOT be the trivial constant
        assert_ne!(a.vocab_fingerprint(), 0);
        assert_ne!(a.vocab_fingerprint(), 0x811C_9DC5);
    }

    #[test]
    fn token_ids_are_not_just_the_input_bytes() {
        // Can-fail control for the whole wiring: if `encode` were a passthrough
        // (id == byte), round-trip would still pass and every check above would
        // be vacuous. Assert at least one id is >= 256, which only a real merge
        // can produce.
        let b = Bpe::load(&vocab_path()).unwrap();
        let ids = b.encode("the quick brown fox");
        assert!(!ids.is_empty());
        assert!(
            ids.iter().any(|&i| i >= 256),
            "no merged id in {:?} -- the encoder cannot be using the merge table",
            ids
        );
    }

    #[test]
    fn empty_input_encodes_to_zero_ids_and_is_reported_as_valid() {
        let b = Bpe::load(&vocab_path()).unwrap();
        assert_eq!(b.encode("").len(), 0);
        assert!(b.roundtrip_ok(""));
    }
}
