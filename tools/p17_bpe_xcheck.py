#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P17-P3: cross-check core/src/bpe.rs against the INDEPENDENT Python reference.

=============================================================================
WHAT THIS MEASURES, AND WHY IT IS NOT A TAUTOLOGY
=============================================================================
There are two implementations here and they are genuinely different:

  * Python  -- `tools/p17_bpe_vocab.py`: a from-scratch BPE trainer, an encoder
    written as the classic O(n^2) formulation plus a fast one, and a decoder
    that expands ids through the merge table.
  * Rust    -- `core/src/bpe.rs`: a no_std module with hard-fail guards, a
    bottom-up arena expansion, and an encoder with its own merge loop.

If only ONE of them existed, "text -> ids -> text is byte-identical" would be a
weak claim: a decoder that is wrong twice agrees with itself. So the claim is
measured as AGREEMENT BETWEEN THE TWO, on a corpus neither was hand-fitted to.

The harness drives Rust from the outside through its own CLI, so what is
measured is the SHIPPED artifact behaviour, not an internal helper.

=============================================================================
CAN-FAIL CONTROLS (this project's iron law)
=============================================================================
Every criterion below is paired with a mutation that MUST make it fail. A
criterion with no working control is reported UNPROVEN, never PASS.

    (a) truncated file            -> G1 exit 9 + numbers
    (b) over-declared n_merges    -> G1/G3 exit 9 + numbers
    (c) wrong magic               -> G8 exit 9 + numbers
    (d) wrong version             -> G9 exit 9 + numbers
    (e) reserved0 != 0            -> G14 exit 9 + numbers
    (f) merged_id off by one      -> G11 exit 9 + numbers
    (g) operand referencing future id -> G10 exit 9 + numbers
    (h) n_pieces mismatch         -> G5 exit 9 + numbers
    (i) special_bytes mismatch    -> G15 exit 9 + numbers
    (j) out-of-range token id     -> G12 exit 9 + numbers
    (k) mutate ONE token id       -> decode output MUST change
    (l) mutate ONE merge operand  -> decode of an affected id MUST change
    (m) Python emitter == Rust writer, byte for byte

Usage:
    python tools/p17_bpe_xcheck.py
    python tools/p17_bpe_xcheck.py --keep     # keep the temp build tree
"""
import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p17_bpe_vocab as V   # noqa: E402

ROOT = str(Path(__file__).resolve().parents[1])
EX = os.path.join(ROOT, "examples", "p17_bpe")
VOCAB = os.path.join(EX, "bpe_vocab.vmb")
CORPUS = os.path.join(EX, "corpus.txt")

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print("  [%s] %-52s %s" % ("PASS" if ok else "FAIL", name, detail))
    return ok


def unproven(name, detail=""):
    RESULTS.append((name, None, detail))
    print("  [UNPROVEN] %-48s %s" % (name, detail))


# ---------------------------------------------------------------------------
# Rust harness source. Deliberately tiny: it exercises the SHIPPED module and
# prints counts, so every judgement stays in Python where the reference lives.
# ---------------------------------------------------------------------------
RUST_MAIN = r'''
// Cross-check driver for core/src/bpe.rs. Std binary (NOT the engine) so the
// guards' exit(9) is observable as a process exit code.
use std::io::Write;

use vmlm_core::bpe as bpe;

unsafe fn read_all(p: &str) -> Vec<u8> {
    std::fs::read(p).unwrap_or_else(|e| { eprintln!("read {}: {}", p, e); std::process::exit(2) })
}

unsafe fn out(s: &str) { let _ = std::io::stdout().write_all(s.as_bytes()); }

// Load + validate + expand, in the order the module requires:
//   read_vocab -> expand_vocab_measure (fills piece_len, returns the EXACT
//   arena size) -> allocate -> expand_vocab.
// The measure/allocate/expand split is not decorative: the first cut of this
// harness used an upper bound for the arena, and expanding past it produced an
// access violation (0xC0000005) instead of a diagnosable failure.
struct Voc {
    file: Vec<u8>,
    l: vmlm_core::bpe::BpeLayout,
    arena: Vec<u8>,
    poff: Vec<u32>,
    plen: Vec<u32>,
    exact: usize,
    used: usize,
}

unsafe fn load(vpath: &str) -> Voc {
    let file = read_all(vpath);
    let l = bpe::read_vocab(file.as_ptr(), file.len());
    let mut plen = vec![0u32; l.n_pieces as usize + 1];
    let exact = bpe::expand_vocab_measure(file.as_ptr(), &l, plen.as_mut_ptr());
    let mut arena = vec![0u8; exact + 64];
    let mut poff = vec![0u32; l.n_pieces as usize + 1];
    let used = bpe::expand_vocab(file.as_ptr(), &l, arena.as_mut_ptr(), exact + 64,
                                 poff.as_mut_ptr(), plen.as_mut_ptr());
    Voc { file, l, arena, poff, plen, exact, used }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 { eprintln!("usage: xcheck <cmd> <vocab> [args]"); std::process::exit(2); }
    let cmd = a[1].as_str();
    let vpath = a[2].clone();
    unsafe {
        match cmd {
            // Parse + validate + expand, report the layout. Any guard firing
            // ends the process with code 9 before this prints.
            "validate" => {
                let v = load(&vpath);
                out(&format!("n_merges={}\nn_pieces={}\nn_special={}\nmerge_bytes={}\nspecial_bytes={}\nsection_bytes={}\narena_exact={}\narena_used={}\n",
                    v.l.n_merges, v.l.n_pieces, v.l.n_special, v.l.merge_bytes,
                    v.l.special_bytes, v.l.section_bytes(), v.exact, v.used));
            }
            // Decode a comma-separated id list, print the bytes as hex (so NUL
            // and invalid UTF-8 survive the transport).
            "decode" => {
                let v = load(&vpath);
                let mut ids: Vec<u32> = Vec::new();
                if a.len() > 3 {
                    for t in a[3].split(',') {
                        if t.is_empty() { continue; }
                        ids.push(t.parse::<u32>().unwrap_or_else(|_| {
                            eprintln!("bad id {}", t); std::process::exit(2) }));
                    }
                }
                let mut ob = vec![0u8; 1 << 20];
                let got = bpe::decode_ids(v.file.as_ptr(), &v.l, v.poff.as_ptr(),
                                          v.plen.as_ptr(), v.arena.as_ptr(),
                                          ids.as_ptr(), ids.len(), ob.as_mut_ptr(),
                                          ob.len(), core::ptr::null_mut());
                let mut hex = String::with_capacity(got * 2);
                for i in 0..got { hex.push_str(&format!("{:02x}", ob[i])); }
                out(&format!("n_bytes={}\nhex={}\n", got, hex));
            }
            // BATCHED decode: read one id-list per line from a file and print
            // one hex line per input line. WHY THIS EXISTS: the first version
            // of this harness spawned the process once per corpus word, i.e.
            // thousands of process creations, and did not finish in 10 minutes.
            // Process creation, not the decoder, was the bottleneck -- so the
            // measurement loop was batched rather than the criterion weakened.
            "decodebatch" => {
                let v = load(&vpath);
                let inb = read_all(&a[3]);
                let mut ob = vec![0u8; 1 << 20];
                let mut ids: Vec<u32> = Vec::new();
                let mut hex = String::new();
                let mut acc = String::new();
                // Lines are separated by '\n' and EMPTY LINES ARE MEANINGFUL:
                // they mean "an empty id list", i.e. decode nothing and emit an
                // empty result. Skipping them (as the first version did) broke
                // the one-line-per-input correspondence and desynchronised the
                // whole batch -- the harness then reported 3881 "mismatches"
                // that were entirely an artefact of the transport, not of the
                // decoder. The transport therefore encodes emptiness
                // EXPLICITLY: a line consisting of the single character '-' is
                // the empty id list.
                for line in inb.split(|c| *c == b'\n') {
                    ids.clear();
                    if line == b"-" {
                        // explicit empty list
                    } else if !line.is_empty() {
                        for t in line.split(|c| *c == b',') {
                            if t.is_empty() { continue; }
                            let mut vv: u32 = 0;
                            for d in t { vv = vv * 10 + (*d - b'0') as u32; }
                            ids.push(vv);
                        }
                    } else {
                        continue;   // trailing newline after the last record
                    }
                    let got = bpe::decode_ids(v.file.as_ptr(), &v.l, v.poff.as_ptr(),
                                              v.plen.as_ptr(), v.arena.as_ptr(),
                                              ids.as_ptr(), ids.len(), ob.as_mut_ptr(),
                                              ob.len(), core::ptr::null_mut());
                    hex.clear();
                    for i in 0..got { hex.push_str(&format!("{:02x}", ob[i])); }
                    acc.push_str(&hex);
                    acc.push('\n');
                }
                out(&acc);
            }
            // Encode a whole file's bytes as text; print ids + a decode of
            // those ids, so the round trip is done entirely inside Rust too.
            "roundtrip" => {
                let v = load(&vpath);
                let text = read_all(&a[3]);
                let mut ids = vec![0u32; bpe::BPE_OUT_LIMIT];
                let n = bpe::encode_text(v.file.as_ptr(), &v.l, text.as_ptr(),
                                         text.len(), ids.as_mut_ptr(),
                                         core::ptr::null_mut());
                ids.truncate(n);
                let mut ob = vec![0u8; text.len() + (1 << 16)];
                let got = bpe::decode_ids(v.file.as_ptr(), &v.l, v.poff.as_ptr(),
                                          v.plen.as_ptr(), v.arena.as_ptr(),
                                          ids.as_ptr(), ids.len(), ob.as_mut_ptr(),
                                          ob.len(), core::ptr::null_mut());
                ob.truncate(got);
                let mut ih = String::new();
                for x in ids.iter() { ih.push_str(&format!("{},", x)); }
                // The decoded BYTES travel back as hex, not only as a digest:
                // the caller must be able to compare the actual bytes, so that
                // a miscomputed digest cannot make a failed round trip look
                // exact. (The digest is a convenience; the bytes are evidence.)
                let mut oh = String::with_capacity(got * 2);
                for i in 0..got { oh.push_str(&format!("{:02x}", ob[i])); }
                out(&format!("n_ids={}\nn_bytes={}\nids={}\nhex={}\nsha_in={}\nsha_out={}\nexact={}\n",
                    n, got, ih, oh,
                    sha_hex(&text), sha_hex(&ob),
                    if text == ob { 1 } else { 0 }));
            }
            _ => { eprintln!("unknown cmd {}", cmd); std::process::exit(2); }
        }
    }
}

fn sha_hex(b: &[u8]) -> String {
    // FNV-1a 64, printed as 16 hex digits. Deliberately NOT a real hash: this
    // only needs to be a cheap, dependency-free digest for the harness to
    // compare two byte strings, and using a named algorithm here would invite
    // reading it as a security property.
    let mut h: u64 = 0xcbf29ce484222325;
    for &x in b { h ^= x as u64; h = h.wrapping_mul(0x100000001b3); }
    format!("{:016x}", h)
}
'''


class Harness(object):
    """Builds core/ in an ISOLATED temp copy with `--features bpe` only.

    Isolated on purpose: this crate must compile with default-features = false
    plus ONLY `bpe`, which is the measured form of the claim 'the bpe feature
    is self-sufficient / orthogonal to train and conn-sparse'.
    """

    def __init__(self):
        self.dir = None
        self.bin = None
        self.log = []

    def build(self):
        cargo = shutil.which("cargo")
        if cargo is None:
            return False, "cargo not on PATH"
        self.dir = tempfile.mkdtemp(prefix="p17_bpe_xcheck_")
        core_dir = os.path.join(ROOT, "core").replace("\\", "/")
        src = os.path.join(self.dir, "src")
        os.makedirs(src)
        with open(os.path.join(self.dir, "Cargo.toml"), "w", encoding="utf-8") as f:
            f.write("[package]\n"
                    'name = "p17_bpe_xcheck"\n'
                    'version = "0.0.0"\n'
                    'edition = "2021"\n\n'
                    "[dependencies]\n"
                    'vmlm-core = { path = "%s", default-features = false, '
                    'features = ["bpe"] }\n\n'
                    "[workspace]\n" % core_dir)
        with open(os.path.join(src, "main.rs"), "w", encoding="utf-8") as f:
            f.write(RUST_MAIN)
        p = subprocess.run([cargo, "build", "--release", "--quiet"],
                           cwd=self.dir, capture_output=True, text=True,
                           errors="replace", timeout=1200)
        self.log.append(p.stdout + p.stderr)
        if p.returncode != 0:
            return False, "cargo rc=%d: %s" % (
                p.returncode, " | ".join((p.stderr or "").strip().splitlines()[-10:]))
        self.bin = os.path.join(self.dir, "target", "release", "p17_bpe_xcheck.exe")
        if not os.path.isfile(self.bin):
            return False, "binary missing at %s" % self.bin
        return True, "ok"

    def run(self, *args):
        p = subprocess.run([self.bin] + list(args), capture_output=True,
                           text=True, errors="replace", timeout=600)
        return p.returncode, p.stdout or "", p.stderr or ""

    def close(self):
        if self.dir:
            shutil.rmtree(self.dir, ignore_errors=True)


def kv(out):
    d = {}
    for line in out.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            d[k.strip()] = v.strip()
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    print("=" * 78)
    print("P17-P3 cross-check: core/src/bpe.rs  vs  tools/p17_bpe_vocab.py")
    print("=" * 78)

    if not os.path.isfile(VOCAB):
        print("missing %s -- run tools/p17_bpe_vocab.py first" % VOCAB)
        return 1
    blob = open(VOCAB, "rb").read()
    corpus = open(CORPUS, "rb").read()

    # ---- reference rebuild (must match the shipped artifact) -------------
    b = V.build(4096)
    merges, vocab, specials = b["merges"], b["vocab"], b["specials"]
    ref = V.BpeRef(merges, specials, vocab)

    print()
    print("--- 1. Python emitter vs the shipped artifact ---")
    reemit = V.write_vocab(len(merges), len(specials), merges, specials)
    check("(m) Python emitter reproduces the artifact byte for byte",
          reemit == blob,
          "%d B vs %d B, sha %s" % (len(reemit), len(blob),
                                    hashlib.sha256(blob).hexdigest()[:16]))
    if reemit != blob:
        print("     refusing to continue on a drifted fixture")
        return 1

    # ---- build the isolated Rust harness --------------------------------
    print()
    print("--- 2. building the isolated Rust harness (only --features bpe) ---")
    h = Harness()
    t0 = time.time()
    ok, msg = h.build()
    dt = time.time() - t0
    if not ok:
        print("  BUILD FAILED (%.1fs): %s" % (dt, msg))
        print("  A failure here is itself a finding: it means `--features bpe`")
        print("  is NOT self-sufficient (it would mean the feature graph is")
        print("  coupled to train/conn-sparse).")
        check("bpe feature compiles alone (default-features = false)", False, msg)
        return 1
    check("bpe feature compiles alone (default-features = false)", True,
          "%.1fs" % dt)

    try:
        # ---- layout agreement -------------------------------------------
        print()
        print("--- 3. Rust layout == Python layout ---")
        rc, so, se = h.run("validate", VOCAB)
        if rc != 0:
            check("Rust read_vocab accepts the artifact", False,
                  "rc=%d out=%r err=%r" % (rc, so[-200:], se[-200:]))
            return 1
        d = kv(so)
        want = {
            "n_merges": len(merges),
            "n_pieces": V.N_BYTE_PIECES + len(merges),
            "n_special": len(specials),
            "merge_bytes": len(merges) * V.MERGE_ENTRY,
            "special_bytes": len(specials) * V.SPECIAL_ENTRY,
            "section_bytes": V.section_bytes(len(merges), len(specials)),
        }
        allok = True
        for k, v in want.items():
            got = int(d.get(k, "-1"))
            good = (got == v)
            allok &= good
            print("      %-14s rust=%-10d python=%-10d %s"
                  % (k, got, v, "OK" if good else "MISMATCH"))
        check("Rust and Python agree on every header field", allok)
        # The arena size Rust computed must equal the exact expansion, which
        # Python can compute independently by summing its own piece table.
        # This is the criterion that caught the first version's UPPER BOUND
        # being too small: it reported 10896 where 38189 was needed, and the
        # shortfall showed up as an access violation (0xC0000005) during
        # expansion rather than as a diagnosable failure.
        exact = sum(len(vocab[i]) for i in range(V.N_BYTE_PIECES + len(merges)))
        r_exact = int(d.get("arena_exact", "-1"))
        r_used = int(d.get("arena_used", "-1"))
        check("Rust arena size == Python's exact expansion size",
              r_exact == exact and r_used == exact,
              "rust exact=%d used=%d python exact=%d"
              % (r_exact, r_used, exact))

        # ---- decode agreement on the whole corpus ------------------------
        print()
        print("--- 4. Rust decode == Python decode (independent implementations) ---")
        r = V.BpeRef(merges, specials, vocab)
        # Batch the corpus into chunks and decode each chunk in ONE process.
        # See the Rust comment on "decodebatch": spawning per word did not
        # finish in 10 minutes, and the bottleneck was process creation, not
        # the decoder -- so the loop was batched rather than the criterion
        # weakened to a sample.
        words = corpus.decode("utf-8").split(" ")
        probe_words = [w for w in words if w]
        batches = 0
        mismatches = 0
        first_bad = None
        CHUNK = 400
        for base in range(0, len(probe_words), CHUNK):
            part = probe_words[base:base + CHUNK]
            idlines = []
            pys = []
            for w in part:
                ids, _ = r.encode(w)
                idlines.append(",".join(str(x) for x in ids) if ids else "-")
                pys.append(r.decode_ids(ids))
            inp = os.path.join(h.dir, "batch_ids.txt")
            # BYTES, NOT TEXT MODE. `open(..., "w")` on Windows translates
            # "\n" to "\r\n", and the Rust reader splits on '\n' only, so the
            # trailing '\r' (0x0D) stayed glued to the last id: "415\r" parsed
            # as 415*10+13 = 4163, and the whole batch died with
            #     BPE: token id out of vocabulary range ... id=4371
            # Reported as 3881 decode "mismatches" -- an artefact of the
            # transport, not of the decoder. Binary mode is the fix.
            with open(inp, "wb") as f:
                f.write(("\n".join(idlines) + "\n").encode("utf-8"))
            rc, so, se = h.run("decodebatch", VOCAB, inp)
            if rc != 0:
                mismatches += len(part)
                if first_bad is None:
                    first_bad = (base, part[0], "rust rc=%d %s" % (rc, se[-200:]))
                continue
            got = so.splitlines()
            if len(got) != len(part):
                mismatches += len(part)
                if first_bad is None:
                    first_bad = (base, part[0],
                                 "rust returned %d lines for %d words"
                                 % (len(got), len(part)))
                continue
            for k, w in enumerate(part):
                batches += 1
                if bytes.fromhex(got[k]) != pys[k]:
                    mismatches += 1
                    if first_bad is None:
                        first_bad = (base + k, w,
                                     "rust=%r python=%r"
                                     % (bytes.fromhex(got[k])[:24], pys[k][:24]))
        check("Rust decode == Python decode on all %d corpus words" % batches,
              mismatches == 0,
              "mismatches=%d" % mismatches)
        if first_bad:
            print("      first mismatch: word#%d %r -> %s" % first_bad)

        # ---- Rust-internal round trip on the corpus ----------------------
        print()
        print("--- 5. round trip, measured inside Rust (text -> ids -> text) ---")
        rc, so, se = h.run("roundtrip", VOCAB, CORPUS)
        if rc != 0:
            check("Rust round trip over the %d-byte corpus" % len(corpus), False,
                  "rc=%d err=%r" % (rc, se[-300:]))
        else:
            d = kv(so)
            check("Rust round trip over the %d-byte corpus" % len(corpus),
                  d.get("exact") == "1",
                  "n_ids=%s n_bytes=%s sha_in=%s sha_out=%s"
                  % (d.get("n_ids"), d.get("n_bytes"),
                     d.get("sha_in"), d.get("sha_out")))
            # Python must agree on the ID COUNT for the same text, else the
            # two encoders segment differently even if both round trip.
            py_ids, _ = r.encode(corpus.decode("utf-8"))
            check("Rust and Python agree on the corpus id count",
                  int(d.get("n_ids", "-1")) == len(py_ids),
                  "rust=%s python=%d" % (d.get("n_ids"), len(py_ids)))

        # ---- MUST-FAIL controls ------------------------------------------
        print()
        print("--- 6. must-fail controls (each MUST exit 9 with numbers) ---")

        def guard(name, mutant_blob, expect_msg=None, argv=None):
            """Write the mutant and require exit 9."""
            p = os.path.join(h.dir, "mutant.vmb")
            with open(p, "wb") as f:
                f.write(mutant_blob)
            rc, so, se = h.run(*(argv or ("validate", p)))
            text = so + se
            ok = (rc == 9)
            extra = "rc=%d" % rc
            if ok and expect_msg:
                has = expect_msg in text
                ok = ok and has
                extra += " msg=%s" % ("OK" if has else "MISSING")
            # Every guard message must carry a NUMBER, per the project rule.
            hasnum = any(ch.isdigit() for ch in text)
            ok = ok and hasnum
            extra += " numbers=%s" % ("yes" if hasnum else "NO")
            if ok:
                line = [l for l in text.splitlines() if l.strip()]
                extra += " | " + (line[0][:60] if line else "")
            check(name, ok, extra)
            return text

        def patched(off, data):
            m = bytearray(blob)
            m[off:off + len(data)] = data
            return bytes(m)

        # (a) truncation: drop the last 3 bytes -> G1
        guard("(a) truncated file -> G1", blob[:-3], V.DIAG["G1_LEN"])
        # (b) over-declared n_merges -> G5 (n_pieces no longer matches)
        guard("(b) n_merges +1 -> G5/G3", patched(V.OFF_N_MERGES,
              struct.pack("<I", len(merges) + 1)), V.DIAG["G5_N_PIECES"])
        # (b2) n_merges -1. NOTE THE EXPECTED GUARD: lowering n_merges breaks
        # BOTH the n_pieces relation (921 vs 256+664=920) AND merge_bytes, and
        # G5 runs first, so the message that fires is G5, not G3. The first
        # version of this control asserted G3 and reported FAIL -- the guard
        # fired correctly and the CONTROL was wrong. Fixed by asserting the
        # guard that is actually reachable for this mutation, and G3 gets its
        # own mutation below where it is the FIRST thing to break.
        guard("(b2) n_merges -1 -> G5 (n_pieces breaks first)",
              patched(V.OFF_N_MERGES, struct.pack("<I", len(merges) - 1)),
              V.DIAG["G5_N_PIECES"])
        # (b3) G3 (merge_bytes != n_merges * 8). Same lesson as (b2): when two
        # header fields are mutually constrained, whichever guard runs FIRST is
        # the one that fires, and a control that asserts the other one reports a
        # false FAIL. G1 (exact length) is checked after validate(), and any
        # change to merge_bytes also changes section_bytes(), so G1 would
        # pre-empt G3 for a real file.
        #
        # G3 is therefore only reachable in isolation when the FILE LENGTH is
        # co-mutated to match the new section_bytes() -- i.e. when the file is
        # re-emitted consistently with a wrong merge_bytes, which is exactly
        # what a buggy WRITER produces. That is the realistic failure this guard
        # exists for, so the control builds it that way: patch merge_bytes AND
        # pad/trim the file so G1 stays satisfied, leaving G3 as the only
        # violated constraint.
        mb_bad = len(merges) * V.MERGE_ENTRY - 8
        sec_bad = V.BPE_HDR + mb_bad
        mut = bytearray(blob[:sec_bad])
        mut[V.OFF_MERGE_BYTES:V.OFF_MERGE_BYTES + 4] = struct.pack("<I", mb_bad)
        guard("(b3) merge_bytes -8 with a matching length -> G3", bytes(mut),
              V.DIAG["G3_MERGE_BYTES"])
        # (c) wrong magic
        guard("(c) wrong magic -> G8", patched(V.OFF_MAGIC,
              struct.pack("<I", 0xDEADBEEF)), V.DIAG["G8_MAGIC"])
        # (d) wrong version
        guard("(d) version 2 -> G9", patched(V.OFF_VERSION,
              struct.pack("<I", 2)), V.DIAG["G9_VERSION"])
        # (e) reserved0 != 0
        guard("(e) reserved0 != 0 -> G14", patched(V.OFF_RESERVED0,
              struct.pack("<I", 1)), V.DIAG["G14_RESERVED0"])
        # (f) merged_id off by one on rule 0
        guard("(f) merged_id off by one -> G11",
              patched(V.BPE_HDR + 4, struct.pack("<I", V.N_BYTE_PIECES + 1)),
              V.DIAG["G11_MERGED_ID"])
        # (g) rule 0 operand referring to a FUTURE id (301 > 256)
        guard("(g) operand references a future id -> G10",
              patched(V.BPE_HDR, struct.pack("<H", 301)), V.DIAG["G10_REFS"])
        # (h) n_pieces mismatch
        guard("(h) n_pieces +1 -> G5",
              patched(V.OFF_N_PIECES, struct.pack("<I", V.N_BYTE_PIECES + len(merges) + 1)),
              V.DIAG["G5_N_PIECES"])
        # (i) special_bytes != 0 while n_special == 0
        guard("(i) special_bytes != n_special*5 -> G15",
              patched(V.OFF_SPECIAL_BYTES, struct.pack("<I", 5)),
              V.DIAG["G15_SPECIAL_BYTES"])
        # (j) header below BPE_HDR: truncate to 16 bytes
        guard("(j) file shorter than header -> G0", blob[:16],
              V.DIAG["G0_AVAIL_HDR"])

        # (k) out-of-range token id at DECODE time -> G12.
        # This one is NOT a file mutation: the file is fine, the ID STREAM is
        # bad, which is the shape a real caller hits when the ids come from a
        # model whose vocabulary is a different artifact.
        bad_ids = "0,%d" % (V.N_BYTE_PIECES + len(merges) + 5)
        p = os.path.join(h.dir, "good.vmb")
        with open(p, "wb") as f:
            f.write(blob)
        rc, so, se = h.run("decode", p, bad_ids)
        text = so + se
        check("(j2) out-of-range token id at decode -> G12",
              rc == 9 and V.DIAG["G12_ID_RANGE"] in text
              and any(c.isdigit() for c in text),
              "rc=%d | %s" % (rc, (text.splitlines() or [""])[0][:60]))

        # ---- ROUND-TRIP CAN-FAIL CONTROLS --------------------------------
        # The round trip is the headline criterion, so it needs its own
        # mutation controls. A round trip that cannot fail is not evidence.
        print()
        print("--- 7. round-trip can-fail controls ---")
        base_ids, _ = r.encode("The engine is no_std and links against kernel32")
        base_py = r.decode(base_ids)
        # (k) mutate ONE token id -> decode MUST change
        mut = list(base_ids)
        mut[0] = 65 if mut[0] != 65 else 66
        mut_py = r.decode(mut)
        check("(k) mutating ONE token id changes the output",
              mut_py != base_py,
              "orig[0]=%d mut[0]=%d changed=%s"
              % (base_ids[0], mut[0], mut_py != base_py))
        # same via RUST, so the control covers the shipped decoder
        rc1, so1, _ = h.run("decode", VOCAB, ",".join(str(x) for x in base_ids))
        rc2, so2, _ = h.run("decode", VOCAB, ",".join(str(x) for x in mut))
        h1 = kv(so1).get("hex", "")
        h2 = kv(so2).get("hex", "")
        check("(k2) same control through the RUST decoder",
              rc1 == 0 and rc2 == 0 and h1 != h2,
              "rust hex differs=%s" % (h1 != h2))
        # (l) mutate ONE merge operand -> the affected id's bytes MUST change.
        #
        # ⚠️ THE FIRST VERSION OF THIS CONTROL WAS VACUOUS, and finding that out
        # is the point of running it. It mutated the LAST rule and asserted the
        # bytes changed -- but the last rule's operands are (919, 919), i.e. the
        # SAME id twice, so rewriting left from 919 to 919 (and even to anything
        # that still spells the same tree) leaves the expansion identical. The
        # control reported "mutant == orig" and looked like a decoder bug; it was
        # a badly-chosen mutation that could not fail.
        #
        # Fix: search for a rule whose two operands DIFFER, so flipping one of
        # them is guaranteed to change the expanded bytes. Assert the mutation
        # is non-trivial BEFORE using it -- a mutation that does not apply must
        # abort the control, never silently pass it.
        idx = None
        for k in range(len(merges) - 1, -1, -1):
            if merges[k][0] != merges[k][1]:
                idx = k
                break
        if idx is None:
            unproven("(l) mutating ONE merge operand changes that id's bytes",
                     "every rule has identical operands in this vocabulary, so "
                     "no single-operand mutation is guaranteed to change the "
                     "expansion -- control cannot be constructed")
        else:
            l0, r0 = merges[idx]
            # Flip left to a DIFFERENT already-existing id. Both 0 and l0 exist
            # (G10 guarantees l0 < 256+idx), so the mutant is format-legal and
            # exercises the real expansion path rather than being rejected by a
            # guard -- a mutation killed by a guard would prove nothing about
            # the decoder.
            lnew = 0 if l0 != 0 else 1
            mut_blob = patched(V.BPE_HDR + idx * V.MERGE_ENTRY,
                               struct.pack("<H", lnew))
            pm = os.path.join(h.dir, "mut_merge.vmb")
            with open(pm, "wb") as f:
                f.write(mut_blob)
            tid = V.N_BYTE_PIECES + idx
            rc, so, se = h.run("decode", pm, str(tid))
            got_hex = kv(so).get("hex", "") if rc == 0 else ""
            # Independent expectation: the mutant's bytes must equal
            # bytes(lnew) ++ bytes(r0), computed from the PYTHON piece table.
            want = vocab[lnew] + vocab[r0]
            check("(l) mutating ONE merge operand changes that id's bytes",
                  rc == 0 and bytes.fromhex(got_hex) == want
                  and bytes.fromhex(got_hex) != vocab[tid],
                  "rule=%d id=%d left %d->%d: rust=%d B, python-expect=%d B, "
                  "differs-from-orig=%s"
                  % (idx, tid, l0, lnew,
                     len(bytes.fromhex(got_hex)) if got_hex else -1, len(want),
                     bytes.fromhex(got_hex) != vocab[tid] if got_hex else False))

        # ---- what is NOT proven -----------------------------------------
        print()
        print("--- 8. honest gaps ---")
        unproven("special (byte-fallback) table path on a REAL file",
                 "trained vocab has n_special=0: every piece is in the merge "
                 "tree (665/665 rules fire). G6/G7 are therefore exercised "
                 "only by the malformed-file controls above, never by the "
                 "production artifact.")
        unproven("encoder == a third-party BPE implementation",
                 "there is no reference tokenizer in this repo and no pip "
                 "allowed; agreement is between two implementations that were "
                 "both written here.")
    finally:
        if args.keep:
            print()
            print("kept build tree: %s" % h.dir)
        else:
            h.close()

    npass = sum(1 for _, o, _ in RESULTS if o is True)
    nfail = sum(1 for _, o, _ in RESULTS if o is False)
    nun = sum(1 for _, o, _ in RESULTS if o is None)
    print()
    print("=" * 78)
    print("PASS=%d FAIL=%d UNPROVEN=%d" % (npass, nfail, nun))
    print("=" * 78)

    log = os.path.join(EX, "bpe_xcheck.json")
    with open(log, "w", encoding="utf-8") as f:
        json.dump({"pass": npass, "fail": nfail, "unproven": nun,
                   "results": [{"name": n, "ok": o, "detail": d}
                               for n, o, d in RESULTS]},
                  f, ensure_ascii=False, indent=1)
    print("wrote %s" % log)
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
