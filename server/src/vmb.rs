//! A minimal, read-only `.vmb` reader -- written from the layout documented in
//! `core/src/format.rs`, used so that this crate does NOT have to depend on
//! `vmlm-core` (see `engine.rs` for why that dependency is forbidden).
//!
//! # Scope: exactly what the server needs, and nothing more
//!
//! The server needs three things from a model file:
//!
//!   1. the slot-name -> slot-index mapping (the engine's `out.bin` protocol is
//!      addressed by slot INDEX: `vmlm-runtime model.vmb reps:slot`), and
//!   2. each slot's shape, so the response can say how many elements came back, and
//!   3. the constant table, so the response can report the model's real footprint.
//!
//! It does NOT need the instruction stream, the constant bytes, or the JIT.
//! Deliberately so: this reader must not be able to disagree with the engine about
//! *execution*, only about *metadata*, and metadata disagreements are checkable
//! (see `p17_http_verify.py`'s cross-check against the Python side).
//!
//! # Layout (from `core/src/format.rs`)
//!
//! ```text
//! header, 32 B:
//!   [0]  u32 magic       0x4D4C4D56 "VMLM"
//!   [4]  u16 version
//!   [6]  u16 reserved
//!   [8]  u32 text_off    [12] u32 text_len
//!   [16] u32 const_off   [20] u32 const_len
//!   [24] u32 meta_off
//!   (text_off + text_len == const_off; const_off + const_len == meta_off)
//!
//! META at meta_off:
//!   [0]  u32 n_const     [4]  u32 n_slot
//!   [8]  u32 slots_off   (relative to META)   [12] u32 act_bytes
//!   [16] u32 state_bytes [20] u32 grad_bytes
//!   [24] u32 names_off   (relative to META, n_slot x u32)
//!   [28] const_table: n_const x 24 B
//!   slot table: n_slot x 16 B { kind:u32, idx_or_off:u32, rows:u32, cols:u32 }
//! ```
//!
//! # Hard failure, not silent guessing
//!
//! Every inconsistency is an `Err` with the numbers in it: bad magic, truncated
//! file, a slot table that runs past EOF, a meta offset outside the file. A reader
//! that clamped those would hand the server a slot index computed from garbage --
//! and the server would then run the engine on the wrong slot and return plausible
//! numbers, which is the worst possible outcome. (Same failure mode the engine's
//! own `interp.rs` guards against with `exit(9)`.)
//!
//! Note the FNV-1a hash MUST match `format::hash32` bit for bit; the two
//! implementations are cross-checked by `tools/p17_http_verify.py`, which
//! hashes the slot names with an independent Python FNV-1a and compares against
//! the values in the file. Two implementations agreeing on a constant is weak
//! evidence; agreeing on a *table* of hashes is not.

use std::path::Path;

pub const MAGIC: u32 = 0x4D4C_4D56;
pub const META_HDR: usize = 28;
pub const CONST_ENT: usize = 24;
pub const SLOT_ENT: usize = 16;

/// FNV-1a 32-bit, identical to `vmlm_core::format::hash32`.
pub fn hash32(name: &str) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for b in name.as_bytes() {
        h ^= *b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

#[derive(Debug, Clone)]
pub struct SlotMeta {
    pub name: String,
    pub kind: u32,
    pub idx: u32,
    pub rows: u32,
    pub cols: u32,
    pub hash: u32,
}

#[derive(Debug, Clone)]
pub struct ConstMeta {
    pub off: u32,
    pub len: u32,
    pub rows: u32,
    pub cols: u32,
    pub dtype: u8,
}

impl ConstMeta {
    pub fn dtype_name(&self) -> &'static str {
        match self.dtype {
            0 => "f32",
            1 => "f16",
            2 => "i8",
            3 => "i32",
            4 => "bf16",
            _ => "unknown",
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Vmb {
    pub path: String,
    pub bytes: usize,
    pub text_len: u32,
    pub const_len: u32,
    pub n_const: u32,
    pub n_slots: u32,
    pub act_bytes: u32,
    pub state_bytes: u32,
    pub grad_bytes: u32,
    pub slots: Vec<SlotMeta>,
    pub consts: Vec<ConstMeta>,
    /// Slot name -> index, in file order.
    pub names: Vec<(String, usize)>,
}

fn u16_at(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

impl Vmb {
    pub fn empty() -> Vmb {
        Vmb::default()
    }

    pub fn load(path: &Path) -> Result<Vmb, String> {
        let b = std::fs::read(path).map_err(|e| format!("{}: {}", path.display(), e))?;
        Vmb::parse(&b).map_err(|e| format!("{}: {}", path.display(), e))
    }

    pub fn parse(b: &[u8]) -> Result<Vmb, String> {
        if b.len() < 32 {
            return Err(format!(
                "file is {} B, shorter than the 32 B header",
                b.len()
            ));
        }
        let magic = u32_at(b, 0);
        if magic != MAGIC {
            return Err(format!(
                "bad magic {:#010x}, expected {:#010x} (\"VMLM\")",
                magic, MAGIC
            ));
        }
        let version = u16_at(b, 4);
        if version != 1 {
            return Err(format!("unsupported version {}, expected 1", version));
        }
        let text_off = u32_at(b, 8) as usize;
        let text_len = u32_at(b, 12) as usize;
        let const_off = u32_at(b, 16) as usize;
        let const_len = u32_at(b, 20) as usize;
        let meta_off = u32_at(b, 24) as usize;

        // Section adjacency is an invariant of the writer; a file violating it
        // would make every offset below meaningless, so check it up front and
        // report the numbers.
        if text_off + text_len != const_off {
            return Err(format!(
                "TEXT does not abut CONST: text_off({}) + text_len({}) = {} != const_off({})",
                text_off,
                text_len,
                text_off + text_len,
                const_off
            ));
        }
        if const_off + const_len != meta_off {
            return Err(format!(
                "CONST does not abut META: const_off({}) + const_len({}) = {} != meta_off({})",
                const_off,
                const_len,
                const_off + const_len,
                meta_off
            ));
        }
        if meta_off + META_HDR > b.len() {
            return Err(format!(
                "META offset {} + header {} exceeds file size {}",
                meta_off,
                META_HDR,
                b.len()
            ));
        }

        let n_const = u32_at(b, meta_off);
        let n_slots = u32_at(b, meta_off + 4);
        let slots_off = u32_at(b, meta_off + 8) as usize;
        let act_bytes = u32_at(b, meta_off + 12);
        let state_bytes = u32_at(b, meta_off + 16);
        let grad_bytes = u32_at(b, meta_off + 20);
        let names_off = u32_at(b, meta_off + 24) as usize;

        let ct = meta_off + META_HDR;
        let ct_end = ct + (n_const as usize) * CONST_ENT;
        let st = meta_off + slots_off;
        let st_end = st + (n_slots as usize) * SLOT_ENT;
        let nt = meta_off + names_off;
        let nt_end = nt + (n_slots as usize) * 4;
        for (what, end) in [
            ("const table", ct_end),
            ("slot table", st_end),
            ("names table", nt_end),
        ] {
            if end > b.len() {
                return Err(format!(
                    "{} ends at {} but the file is {} B (n_const={} n_slots={} \
                     slots_off={} names_off={})",
                    what,
                    end,
                    b.len(),
                    n_const,
                    n_slots,
                    slots_off,
                    names_off
                ));
            }
        }

        let mut consts = Vec::with_capacity(n_const as usize);
        for i in 0..n_const as usize {
            let o = ct + i * CONST_ENT;
            let c = ConstMeta {
                off: u32_at(b, o),
                len: u32_at(b, o + 4),
                rows: u32_at(b, o + 8),
                cols: u32_at(b, o + 12),
                dtype: b[o + 16],
            };
            if c.off as usize + c.len as usize > const_len {
                return Err(format!(
                    "const[{}] occupies [{}, {}) but CONST is only {} B",
                    i,
                    c.off,
                    c.off as usize + c.len as usize,
                    const_len
                ));
            }
            consts.push(c);
        }

        // The names table is `n_slot x u32` of hashes. We recover the actual
        // strings by hash-matching against the candidate names the fixture
        // declares... except we have no such list. So: the hash table gives us
        // the ORDER, and the caller supplies names to look up. What we can do
        // here without guessing is record the hash per index and let
        // `slot_index(name)` hash-match. That is exactly the engine's own
        // `find_slot` algorithm.
        let mut slots = Vec::with_capacity(n_slots as usize);
        let mut names = Vec::with_capacity(n_slots as usize);
        for i in 0..n_slots as usize {
            let e = st + i * SLOT_ENT;
            let hash = u32_at(b, nt + i * 4);
            slots.push(SlotMeta {
                name: String::new(),
                kind: u32_at(b, e),
                idx: u32_at(b, e + 4),
                rows: u32_at(b, e + 8),
                cols: u32_at(b, e + 12),
                hash,
            });
            names.push((String::new(), i));
        }

        Ok(Vmb {
            path: String::new(),
            bytes: b.len(),
            text_len: text_len as u32,
            const_len: const_len as u32,
            n_const,
            n_slots,
            act_bytes,
            state_bytes,
            grad_bytes,
            slots,
            consts,
            names,
        })
    }

    /// Slot index for `name`, by FNV-1a hash -- the same lookup the engine does.
    ///
    /// Returns `None` when absent. (The engine `exit(7)`s here; a server must not
    /// kill itself over one bad request, so this is an `Option` and the caller
    /// turns it into a 400/UNPROVEN.)
    pub fn slot_index(&self, name: &str) -> Option<usize> {
        let h = hash32(name);
        self.slots.iter().position(|s| s.hash == h)
    }

    pub fn has_slot(&self, name: &str) -> bool {
        self.slot_index(name).is_some()
    }

    pub fn slot_shape(&self, name: &str) -> Option<(u32, u32)> {
        self.slot_index(name)
            .map(|i| (self.slots[i].rows, self.slots[i].cols))
    }

    /// Total constant-pool bytes actually used (sum of per-const lengths).
    pub fn const_used_bytes(&self) -> u64 {
        self.consts.iter().map(|c| c.len as u64).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tiny but *valid* `.vmb` and check the reader agrees with the
    /// engine's documented layout. This is the "can the reader fail?" control:
    /// the same builder with one byte corrupted must be rejected.
    fn build() -> Vec<u8> {
        let text = vec![0u8; 16]; // one instruction's worth
        let cons = vec![0u8; 64];
        let meta_off = 32 + text.len() + cons.len();
        let n_const = 1u32;
        let n_slots = 2u32;
        let slots_off = META_HDR + CONST_ENT; // right after the const table
        let names_off = slots_off + SLOT_ENT * 2;
        let mut m = vec![0u8; names_off + 4 * 2];
        let put = |m: &mut Vec<u8>, o: usize, v: u32| {
            m[o..o + 4].copy_from_slice(&v.to_le_bytes())
        };
        put(&mut m, 0, n_const);
        put(&mut m, 4, n_slots);
        put(&mut m, 8, slots_off as u32);
        put(&mut m, 12, 64); // act_bytes
        put(&mut m, 16, 0);
        put(&mut m, 20, 0);
        put(&mut m, 24, names_off as u32);
        // const[0]: off 0, len 64, rows 4, cols 4, dtype f32
        put(&mut m, META_HDR, 0);
        put(&mut m, META_HDR + 4, 64);
        put(&mut m, META_HDR + 8, 4);
        put(&mut m, META_HDR + 12, 4);
        m[META_HDR + 16] = 0;
        // slot[0] = activation "x" (4x4), slot[1] = const "w"
        put(&mut m, slots_off, 1);
        put(&mut m, slots_off + 4, 0);
        put(&mut m, slots_off + 8, 4);
        put(&mut m, slots_off + 12, 4);
        put(&mut m, slots_off + 16, 0);
        put(&mut m, slots_off + 20, 0);
        put(&mut m, slots_off + 24, 4);
        put(&mut m, slots_off + 28, 4);
        put(&mut m, names_off, hash32("x"));
        put(&mut m, names_off + 4, hash32("w"));

        let mut b = vec![0u8; 32];
        put(&mut b, 0, MAGIC);
        b[4..6].copy_from_slice(&1u16.to_le_bytes());
        put(&mut b, 8, 32);
        put(&mut b, 12, text.len() as u32);
        put(&mut b, 16, (32 + text.len()) as u32);
        put(&mut b, 20, cons.len() as u32);
        put(&mut b, 24, meta_off as u32);
        b.extend_from_slice(&text);
        b.extend_from_slice(&cons);
        b.extend_from_slice(&m);
        b
    }

    #[test]
    fn reads_a_valid_file() {
        let v = Vmb::parse(&build()).unwrap();
        assert_eq!(v.n_slots, 2);
        assert_eq!(v.n_const, 1);
        assert_eq!(v.slot_index("x"), Some(0));
        assert_eq!(v.slot_index("w"), Some(1));
        assert_eq!(v.slot_index("nope"), None);
        assert_eq!(v.slot_shape("x"), Some((4, 4)));
        assert_eq!(v.const_used_bytes(), 64);
        assert_eq!(v.consts[0].dtype_name(), "f32");
        // And the hash really is FNV-1a of the name. The expected values come
        // from an INDEPENDENT Python implementation (tools/p17_http_verify.py
        // prints them), not from this function -- otherwise the assertion would
        // only prove the function equals itself.
        assert_eq!(hash32("x"), 0xfd0c_5087);
        assert_eq!(hash32("w"), 0xf20c_3f36);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut b = build();
        b[0] = 0;
        let e = Vmb::parse(&b).unwrap_err();
        assert!(e.contains("bad magic"), "{}", e);
    }

    #[test]
    fn rejects_truncation() {
        let b = build();
        for cut in [8usize, 31, 33, b.len() - 1] {
            let e = Vmb::parse(&b[..cut]).unwrap_err();
            assert!(
                e.contains("shorter") || e.contains("exceeds") || e.contains("but the file"),
                "cut {}: {}",
                cut,
                e
            );
        }
    }

    #[test]
    fn rejects_non_adjacent_sections() {
        // Break text_off + text_len == const_off by one byte.
        let mut b = build();
        b[12..16].copy_from_slice(&15u32.to_le_bytes());
        let e = Vmb::parse(&b).unwrap_err();
        assert!(e.contains("TEXT does not abut CONST"), "{}", e);
    }

    #[test]
    fn rejects_a_const_reaching_past_the_pool() {
        let mut b = build();
        let meta_off = u32::from_le_bytes([b[24], b[25], b[26], b[27]]) as usize;
        b[meta_off + META_HDR + 4..meta_off + META_HDR + 8]
            .copy_from_slice(&999u32.to_le_bytes());
        let e = Vmb::parse(&b).unwrap_err();
        assert!(e.contains("CONST is only"), "{}", e);
    }
}
