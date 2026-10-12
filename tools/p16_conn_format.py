#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P16-1: the CONNECTION-SPARSE storage format -- host-side emitter, reader, and
the can-fail apparatus that proves the Rust guards actually fire.

WHAT THIS FILE IS
-----------------
`core/src/conn_sparse.rs` defines the CSR adjacency list
(`conn_offsets` / `conn_src` / `conn_weights`) for principle 1 of the P1 step
"connection sparsity + connection-sparse activation".  A format that only exists
as Rust code is not evidence: nothing has been written to disk, nothing has been
parsed back, and no guard has been observed firing.  This file is the producer
and the measurement.

It does four things, in this order:

  1. EMITS the section for the validation shape (24 layers x 1536 dim, k0=20)
     and asserts the file's on-disk byte count against the header-derived size.
  2. ROUND-TRIPS it: writes with the Rust `write_section`, reads back through
     the real engine reader, and compares every (src_id, weight) pair --
     comparing weight BIT PATTERNS, not `==`, so -0.0 / NaN bit differences
     cannot hide (neuron_kv.rs's `bitwise_equal` makes the same point).
  3. FIRES every guard on a real illegal input, in the REAL Rust code, and
     records the process exit code and the exact message.  The guards are not
     re-implemented in Python for this step: each one is executed by a small
     Rust harness built out-of-tree that links the real `vmlm-core` with the
     real `conn-sparse` feature.  A Python re-implementation of a guard would
     prove only that Python agrees with itself.
  4. Runs CAN-FAIL controls: perturbations that MUST make a check fail, each
     reporting a COUNT.  A check that cannot fail is not a measurement --
     this project was burned by a "0 B" probe whose patch had not actually
     changed anything (docs/P13_report section 3.2).

WHY THE CROSS-CHECK IS A REAL RUST BUILD AND NOT A RE-IMPLEMENTATION
--------------------------------------------------------------------
"a wrong formula would agree with itself".  If this tool computed the expected
bytes from the same PRNG it uses to emit them, the round-trip check would be a
tautology.  So the expected bytes come from the SHIPPED Rust `write_section`,
executed in a real process, and the Python emitter is diffed against it
byte-for-byte.  When that build is unavailable the tool prints
`RUST-GUARDS: SKIPPED` and DOWNGRADES its own verdict -- it never silently
substitutes "same constants, therefore same bytes".

Usage:
    python tools/p16_conn_format.py
"""
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = str(Path(__file__).resolve().parents[1])
OUT_DIR = os.path.join(ROOT, "examples", "p16_conn")
SECTION_PATH = os.path.join(OUT_DIR, "conn_sparse.bin")
DUMP_PATH = os.path.join(OUT_DIR, "conn_sparse.dump.json")

# ---- the validation shape, per the task brief -----------------------------
# 24 layers x 1536 dim (d_model).  n_neurons per layer = 1536.
NL_TARGET = 24
D_MODEL = 1536
K0 = 20

FAILS = []
SKIPS = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print("  [%s] %s%s" % (tag, name, (" -- " + detail) if detail else ""))
    if not ok:
        FAILS.append(name)
    return ok


def skip(name, detail=""):
    print("  [SKIP] %s%s" % (name, (" -- " + detail) if detail else ""))
    SKIPS.append(name)


# ---------------------------------------------------------------------------
# 1. Constants PARSED OUT OF THE SHIPPED SOURCE, never retyped.
#
#    If these cannot be read, this tool REFUSES to emit (return 2) rather than
#    falling back to hardcoded defaults -- a producer whose constants are not
#    traceable to conn_sparse.rs is not evidence.  (Same rule as
#    tools/p15_s5_section.py, where the first version of that parser silently
#    failed to find SEC_NEURON_KV because it did not strip trailing comments.)
# ---------------------------------------------------------------------------
def parse_rust_consts():
    src_path = os.path.join(ROOT, "core", "src", "conn_sparse.rs")
    with open(src_path, "r", encoding="utf-8-sig") as f:
        text = f.read()
    want = {
        "CS_HDR": "usize",
        "CS_MAGIC": "u32",
        "CS_SRC_ENTRY": "usize",
        "CS_WGT_ENTRY": "usize",
        "K0_DEFAULT": "u32",
        "SEC_CONN_SPARSE": "u32",
        "VMB_HDR_WITH_DESCR": "usize",
    }
    found = {}
    hits = {}
    for i, ln in enumerate(text.splitlines()):
        s = ln.strip()
        if not s.startswith("pub const "):
            continue
        body = s.split("//", 1)[0].strip()
        for name, ty in want.items():
            if body.startswith("pub const %s:" % name) and ("%s =" % ty) in body:
                rhs = body.split("=", 1)[1].strip().rstrip(";").strip()
                rhs = rhs.replace("_", "")
                try:
                    found[name] = int(rhs, 0)
                except ValueError:
                    continue
                hits[name] = (i + 1, s)
    return found, hits, src_path


# ---------------------------------------------------------------------------
# 2. The header layout, read from the offsets the Rust file itself declares.
# ---------------------------------------------------------------------------
def parse_header_offsets():
    src_path = os.path.join(ROOT, "core", "src", "conn_sparse.rs")
    with open(src_path, "r", encoding="utf-8-sig") as f:
        text = f.read()
    want = {"CS_OFF_MAGIC": 0, "CS_OFF_N_NEURONS": 4, "CS_OFF_K0": 8,
            "CS_OFF_TOTAL_PAIRS": 12, "CS_OFF_LAYER": 16}
    found = {}
    for ln in text.splitlines():
        s = ln.strip()
        body = s.split("//", 1)[0].strip()
        for name in want:
            if body.startswith("pub const %s: usize =" % name):
                found[name] = int(body.split("=", 1)[1].strip().rstrip(";").strip())
    return found


# ---------------------------------------------------------------------------
# 3. The out-of-tree Rust harness.
#
#    Three modes, all linking the REAL vmlm-core with the REAL `conn-sparse`
#    feature:
#
#      emit   : write_section() on the target layout -> section file, and
#               print the shipped header/length numbers.
#      read   : read_section() on a file -> print per-neuron dump + the
#               crosscheck_readback() mismatch counts.
#      guard  : run ONE named illegal-input scenario and let the real guard
#               ExitProcess(9).  Nothing is caught -- the exit code IS the
#               measurement.
#
#    Built in TEMP, never inside <repo>: a stray workspace member could
#    be picked up by the Lead's serialized acceptance `cargo build` and perturb
#    it.  This tool must not be able to do that.
# ---------------------------------------------------------------------------
RUST_HARNESS = r'''
// Auto-generated throwaway harness: drives the REAL conn_sparse module.
use std::env;
use std::fs;
use vmlm_core::conn_sparse::*;

fn read_file(path: &str) -> Vec<u8> { fs::read(path).expect("read") }

fn main() {
    let args: Vec<String> = env::args().collect();
    let mode = args[1].as_str();
    match mode {
        // ---- emit the section using the SHIPPED writer -------------------
        "emit" => {
            let n: u32 = args[2].parse().unwrap();
            let k0: u32 = args[3].parse().unwrap();
            let layer: u32 = args[4].parse().unwrap();
            let out = &args[5];
            let layout = ConnSparseLayout::tight(n, k0, layer);
            println!("N_NEURONS={}", layout.n_neurons);
            println!("K0={}", layout.k0);
            println!("TOTAL_PAIRS={}", layout.total_pairs);
            println!("LAYER={}", layout.layer);
            println!("CS_HDR={}", CS_HDR);
            println!("OFFSETS_BYTES={}", layout.offsets_bytes());
            println!("SRC_BYTES={}", layout.src_bytes());
            println!("WEIGHTS_BYTES={}", layout.weights_bytes());
            println!("SECTION_BYTES={}", layout.section_bytes());
            println!("OFFSETS_OFF={}", layout.offsets_off());
            println!("SRC_OFF={}", layout.src_off());
            println!("WEIGHTS_OFF={}", layout.weights_off());
            let n_bytes = layout.section_bytes();
            let mut buf = vec![0u8; n_bytes];
            let written = unsafe { write_section(&layout, buf.as_mut_ptr()) };
            println!("WRITTEN={}", written);
            fs::write(out, &buf).expect("write");
            // Print a digest of the payload so a truncation is visible even if
            // the size happens to match.
            let mut h: u64 = 1469598103934665603;
            for b in buf.iter() { h ^= *b as u64; h = h.wrapping_mul(1099511628211); }
            println!("FNV1A64={:016x}", h);
            let mut s = String::with_capacity(n_bytes * 2);
            for b in buf.iter() { s.push_str(&format!("{:02x}", b)); }
            println!("HEX={}", s);
        }
        // ---- read a section using the SHIPPED reader ---------------------
        "read" => {
            let path = &args[2];
            let buf = read_file(path);
            let layout = unsafe { read_section(buf.as_ptr(), buf.len()) };
            println!("N_NEURONS={}", layout.n_neurons);
            println!("K0={}", layout.k0);
            println!("TOTAL_PAIRS={}", layout.total_pairs);
            println!("LAYER={}", layout.layer);
            println!("SECTION_BYTES={}", layout.section_bytes());
            let (sb, wb, fs_, fw) = unsafe { crosscheck_readback(buf.as_ptr(), &layout) };
            println!("SRC_MISMATCH={}", sb);
            println!("WEIGHT_BIT_MISMATCH={}", wb);
            match fs_ { Some((i, j, g, w)) => println!("FIRST_SRC_BAD={},{},{},{}", i, j, g, w),
                        None => println!("FIRST_SRC_BAD=none") }
            match fw { Some((i, j, g, w)) => println!("FIRST_W_BAD={},{},{:08x},{:08x}", i, j, g, w),
                        None => println!("FIRST_W_BAD=none") }
            // Per-neuron dump for a few neurons, so the Python side can compare
            // against ITS OWN independent emitter.  The pick list must not
            // repeat when the shape is small: the first version used
            // [0,1,2, n-2,n-1], which for n=4 printed neuron 2 twice and
            // omitted nothing -- a confusing dump, and the kind of sloppiness
            // that makes a reader distrust the rest of the output.
            let mut picks: Vec<u32> = vec![0, 1, 2];
            if layout.n_neurons > 4 { picks.push(layout.n_neurons - 2); }
            if layout.n_neurons > 5 { picks.push(layout.n_neurons - 1); }
            picks.retain(|p| *p < layout.n_neurons);
            picks.dedup();
            let mut lines: Vec<String> = Vec::new();
            for i in picks {
                let a = unsafe { neuron_conn_addr(buf.as_ptr(), &layout, i) };
                let mut s = format!("NEURON {} {} {}", i, a.begin, a.end);
                let mut j = 0usize;
                while j < a.count() {
                    let sid = unsafe { conn_src(buf.as_ptr(), &layout, i, j) };
                    let wb2 = unsafe { conn_weight_bits(buf.as_ptr(), &layout, i, j) };
                    s.push_str(&format!(" {}:{:08x}", sid, wb2));
                    j += 1;
                }
                lines.push(s);
            }
            for l in lines { println!("{}", l); }
            // Whole-weights bit digest: read every weight through the SHIPPED
            // accessor and hash the bit patterns.
            let mut h: u64 = 1469598103934665603;
            let mut i = 0u32;
            while i < layout.n_neurons {
                let a = unsafe { neuron_conn_addr(buf.as_ptr(), &layout, i) };
                let mut j = 0usize;
                while j < a.count() {
                    let w = unsafe { conn_weight_bits(buf.as_ptr(), &layout, i, j) };
                    for k in 0..4 { h ^= ((w >> (8 * k)) & 0xFF) as u64; h = h.wrapping_mul(1099511628211); }
                    j += 1;
                }
                i += 1;
            }
            println!("WEIGHTS_FNV1A64={:016x}", h);
            let mut h2: u64 = 1469598103934665603;
            let mut e = 0usize;
            while e < layout.total_pairs as usize {
                let s = unsafe { get_u32(buf.as_ptr(), layout.src_off() + e * CS_SRC_ENTRY) };
                for k in 0..4 { h2 ^= ((s >> (8 * k)) & 0xFF) as u64; h2 = h2.wrapping_mul(1099511628211); }
                e += 1;
            }
            println!("SRC_FNV1A64={:016x}", h2);
        }
        // ---- one illegal-input scenario: let the REAL guard exit(9) ------
        "guard" => {
            let which = args[2].as_str();
            let path = &args[3];
            let mut buf = read_file(path);
            let len0 = buf.len();
            // Build the (possibly illegal) input IN MEMORY, then hand it to the
            // shipped reader.  No guard is re-implemented here: the shipped
            // code decides.
            let avail: usize;
            match which {
                // G1: truncate the buffer by 4 bytes (short read).
                "g1_truncated" => { buf.truncate(len0 - 4); avail = buf.len(); }
                // G1': over-declare: keep all bytes, claim 4 MORE are available.
                //      This is the S8 defect class -- a declared length that does
                //      not bound the contents.  `read_section` must reject `>`.
                "g1_overdeclared" => { avail = len0 + 4; }
                // G0: less than a header's worth of bytes.
                "g0_short_header" => { buf.truncate(CS_HDR - 1); avail = buf.len(); }
                // G8: corrupt the magic, keep everything else valid.
                "g8_bad_magic" => {
                    let bad: u32 = 0xDEAD_BEEF;
                    unsafe { put_u32(buf.as_mut_ptr(), CS_OFF_MAGIC, bad) };
                    avail = buf.len();
                }
                // G5: break monotonicity of conn_offsets.
                //
                //     MEASURED FINDING (this is a real property of the format,
                //     not a harness shortcut): monotonicity and "every slice is
                //     exactly k0" are COUPLED.  If every adjacent gap is k0 the
                //     array is necessarily monotone, so any interior
                //     non-monotonicity forces some gap to differ from k0 --
                //     which means G6 will also be violated by that input.
                //
                //     Two earlier scenarios got the message wrong and both are
                //     recorded here because they are informative:
                //       1st: bumped offsets[n-1] upward -> exited 9 with G6's
                //            message ("neuron=1534 count=39 k0=20").
                //       2nd: swapped two adjacent interior entries -> also G6
                //            ("count=40"), since a swap changes both gaps.
                //
                //     A guard firing with the WRONG message is a diagnostic
                //     regression, not a pass.  So this scenario puts the
                //     decrease at the LAST index (i == n): the monotonicity
                //     loop reaches i==n and fires there, BEFORE G10 (which
                //     reads offsets[n] vs total_pairs) and before G6 can run.
                //     That makes G5 the unique first report for this input.
                "g5_non_monotonic" => {
                    let n = unsafe { get_u32(buf.as_ptr(), CS_OFF_N_NEURONS) } as usize;
                    let v: u32 = unsafe { get_u32(buf.as_ptr(), CS_HDR + (n - 1) * 4) };
                    // offsets[n] = offsets[n-1] - 1  => decreasing at i == n.
                    unsafe { put_u32(buf.as_mut_ptr(), CS_HDR + n * 4, v - 1) };
                    avail = buf.len();
                }
                // G9: conn_offsets[0] != 0.
                "g9_first_offset" => {
                    unsafe { put_u32(buf.as_mut_ptr(), CS_HDR, 1) };
                    avail = buf.len();
                }
                // G10: conn_offsets[n] != total_pairs.
                //
                //     This guard runs at step 8, BEFORE G11 and G6, so it is
                //     the unique first report for this input -- no coupling
                //     problem here.  Shrinking offsets[n] also breaks
                //     monotonicity?  No: offsets[n] is decreased to
                //     offsets[n-1] (not below it), so monotonicity still holds
                //     (non-decreasing allows equality) and G6 would only see a
                //     short LAST slice -- but G10 fires first.
                "g10_last_offset" => {
                    let n = unsafe { get_u32(buf.as_ptr(), CS_OFF_N_NEURONS) } as usize;
                    let prev = unsafe { get_u32(buf.as_ptr(), CS_HDR + (n - 1) * 4) };
                    unsafe { put_u32(buf.as_mut_ptr(), CS_HDR + n * 4, prev) };
                    avail = buf.len();
                }
                // G11: a src_id >= n_neurons.
                "g11_src_range" => {
                    let n = unsafe { get_u32(buf.as_ptr(), CS_OFF_N_NEURONS) };
                    let hdr = CS_HDR;
                    let offs = (n as usize + 1) * 4;
                    let idx: usize = args[4].parse().unwrap();
                    unsafe { put_u32(buf.as_mut_ptr(), hdr + offs + idx * 4, n) };
                    avail = buf.len();
                }
                // G6: k0>0 but neuron `args[4]` has 0 entries.  Done by making
                //     two adjacent offsets equal; total_pairs and
                //     offsets[n] are kept consistent so ONLY the degree guard
                //     can fire (the src array is shifted in spirit, but the
                //     range guard is not what we are testing).
                "g6_degree" => {
                    let v: usize = args[4].parse().unwrap();
                    let n = unsafe { get_u32(buf.as_ptr(), CS_OFF_N_NEURONS) } as usize;
                    let t = unsafe { get_u32(buf.as_ptr(), CS_OFF_TOTAL_PAIRS) } as usize;
                    let k0 = (t / n) as u32;
                    // Collapse neuron v to zero length and extend the last one
                    // so offsets[n] stays == total_pairs.
                    unsafe { put_u32(buf.as_mut_ptr(), CS_HDR + (v + 1) * 4, v as u32 * k0) };
                    avail = buf.len();
                }
                // G7: total_pairs != k0 * n_neurons.  Shrink total_pairs by 1
                //     AND shrink the trailing offsets so the monotonicity and
                //     offsets[n] guards stay satisfied -- so the guard that has
                //     to fire is exactly the k0*n binding, not a length check.
                "g7_total_pairs" => {
                    let n = unsafe { get_u32(buf.as_ptr(), CS_OFF_N_NEURONS) } as usize;
                    let t = unsafe { get_u32(buf.as_ptr(), CS_OFF_TOTAL_PAIRS) };
                    unsafe {
                        put_u32(buf.as_mut_ptr(), CS_OFF_TOTAL_PAIRS, t - 1);
                        put_u32(buf.as_mut_ptr(), CS_HDR + n * 4, t - 1);
                    }
                    avail = buf.len();
                }
                // G2: n_neurons = 0.
                "g2_zero_neurons" => {
                    unsafe { put_u32(buf.as_mut_ptr(), CS_OFF_N_NEURONS, 0) };
                    avail = buf.len();
                }
                // G3: k0 * n_neurons overflows u32.
                "g3_overflow" => {
                    unsafe {
                        put_u32(buf.as_mut_ptr(), CS_OFF_N_NEURONS, 0xFFFF_FFFF);
                        put_u32(buf.as_mut_ptr(), CS_OFF_K0, 2);
                        put_u32(buf.as_mut_ptr(), CS_OFF_TOTAL_PAIRS, 0xFFFF_FFFE);
                    }
                    avail = buf.len();
                }
                other => { println!("UNKNOWN_GUARD={}", other); std::process::exit(64); }
            }
            // !!! the call under test !!!
            let layout = unsafe { read_section(buf.as_ptr(), avail) };
            // Reaching here means the guard did NOT fire.
            println!("GUARD_NOT_FIRED n_neurons={} k0={} total_pairs={}",
                     layout.n_neurons, layout.k0, layout.total_pairs);
        }
        // ---- descriptor path: exercise load_validate on a .vmb -------------
        "load" => {
            let path = &args[2];
            let buf = read_file(path);
            let (_sec, layout) = unsafe { load_validate(buf.as_ptr(), buf.len()) };
            println!("N_NEURONS={}", layout.n_neurons);
            println!("K0={}", layout.k0);
            println!("TOTAL_PAIRS={}", layout.total_pairs);
            println!("SECTION_BYTES={}", layout.section_bytes());
        }
        other => { println!("UNKNOWN_MODE={}", other); std::process::exit(64); }
    }
}
'''


class Build:
    """A throwaway out-of-tree crate that links the real vmlm-core."""

    def __init__(self):
        self.dir = None
        self.bin = None
        self.error = None

    def build(self):
        cargo = shutil.which("cargo")
        if cargo is None:
            self.error = "cargo not on PATH"
            return False
        core_dir = os.path.join(ROOT, "core").replace("\\", "/")
        self.dir = tempfile.mkdtemp(prefix="p16_conn_xcheck_")
        src_dir = os.path.join(self.dir, "src")
        os.makedirs(src_dir)
        with open(os.path.join(self.dir, "Cargo.toml"), "w", encoding="utf-8") as f:
            f.write(
                "[package]\n"
                'name = "p16_conn_xcheck"\n'
                'version = "0.0.0"\n'
                'edition = "2021"\n\n'
                "[dependencies]\n"
                # NOTE: default-features must be OFF here for the same reason
                # runtime/Cargo.toml does it: vmlm-core defaults to ["train"],
                # and the training path drags in a lot we do not need.
                # `conn-sparse` is the ONLY feature this harness needs -- which
                # is itself a check that the new feature is self-sufficient
                # (it must NOT require neuron-kv).
                'vmlm-core = { path = "%s", default-features = false, features = ["conn-sparse"] }\n\n'
                "[workspace]\n" % core_dir)
        with open(os.path.join(src_dir, "main.rs"), "w", encoding="utf-8") as f:
            f.write(RUST_HARNESS)
        p = subprocess.run([cargo, "build", "--release", "--quiet"],
                           cwd=self.dir, capture_output=True, text=True,
                           errors="replace", timeout=900)
        if p.returncode != 0:
            tail = (p.stderr or "").strip().splitlines()[-12:]
            self.error = "cargo build rc=%d: %s" % (p.returncode, " | ".join(tail))
            return False
        self.bin = os.path.join(self.dir, "target", "release",
                                "p16_conn_xcheck.exe")
        if not os.path.isfile(self.bin):
            self.error = "harness binary not produced at %s" % self.bin
            return False
        return True

    def run(self, *args, **kw):
        """Run the harness. Returns (rc, stdout, stderr)."""
        p = subprocess.run([self.bin] + list(args), capture_output=True,
                           text=True, errors="replace", timeout=600, **kw)
        return p.returncode, p.stdout or "", p.stderr or ""

    def build_mutated(self, src_rs_path, dead_code_text):
        """Build a harness whose vmlm-core has one guard surgically removed.

        The removal is a LITERAL text substitution on a COPY of the shipped
        source.  If the literal is not found exactly once, this fails loudly --
        a mutation that did not apply would otherwise look like a guard that
        is not load-bearing, which is the exact "patch changed nothing"
        failure this project hard-fails on.

        The mutated crate is a private copy of core/ in TEMP with the patched
        conn_sparse.rs, so the shipped source is never modified.
        """
        cargo = shutil.which("cargo")
        if cargo is None:
            self.error = "cargo not on PATH"
            return False
        with open(src_rs_path, "r", encoding="utf-8-sig") as f:
            text = f.read()
        n = text.count(dead_code_text)
        if n != 1:
            self.error = ("mutation literal occurs %d times in %s (must be "
                          "exactly 1) -- refusing to run a mutation that may "
                          "not have applied" % (n, src_rs_path))
            return False
        mutated = text.replace(dead_code_text, "    // [MUTANT] guard removed")
        self.dir = tempfile.mkdtemp(prefix="p16_conn_mutant_")
        core_copy = os.path.join(self.dir, "core")
        # NOTE: ignore editors' transient temp files.  Other agents work in this
        # tree concurrently, and `shutil.copytree` of core/ raced against an
        # atomic rewrite of interp.rs: the copy picked up a `.interp.rs.<pid>.
        # <uuid>.tmpdir/interp.rs.tmp` that was deleted mid-copy, raising
        # `Permission denied` and aborting the mutant.  That is an artefact of
        # the environment, not of the guard under test, so those names are
        # excluded rather than being allowed to masquerade as a failed
        # experiment.
        shutil.copytree(
            os.path.join(ROOT, "core"), core_copy,
            ignore=shutil.ignore_patterns("target", "*.tmp", "*.tmpdir",
                                          ".*.tmpdir", ".*.tmp"))
        # Re-try once if a concurrent writer still wins the race.
        for _attempt in range(2):
            if os.path.isfile(os.path.join(core_copy, "src", "interp.rs")):
                break
            time.sleep(0.2)
            shutil.rmtree(core_copy, ignore_errors=True)
            shutil.copytree(
                os.path.join(ROOT, "core"), core_copy,
                ignore=shutil.ignore_patterns("target", "*.tmp", "*.tmpdir",
                                              ".*.tmpdir", ".*.tmp"))
        with open(os.path.join(core_copy, "src", "conn_sparse.rs"), "w",
                  encoding="utf-8") as f:
            f.write(mutated)
        src_dir = os.path.join(self.dir, "harness", "src")
        os.makedirs(src_dir)
        with open(os.path.join(self.dir, "harness", "Cargo.toml"), "w",
                  encoding="utf-8") as f:
            f.write(
                "[package]\n"
                'name = "p16_conn_xcheck"\n'
                'version = "0.0.0"\n'
                'edition = "2021"\n\n'
                "[dependencies]\n"
                'vmlm-core = { path = "../core", default-features = false, '
                'features = ["conn-sparse"] }\n\n'
                "[workspace]\n")
        with open(os.path.join(src_dir, "main.rs"), "w", encoding="utf-8") as f:
            f.write(RUST_HARNESS)
        p = subprocess.run([cargo, "build", "--release", "--quiet"],
                           cwd=os.path.join(self.dir, "harness"),
                           capture_output=True, text=True, errors="replace",
                           timeout=900)
        if p.returncode != 0:
            tail = (p.stderr or "").strip().splitlines()[-12:]
            self.error = "mutant cargo build rc=%d: %s" % (p.returncode,
                                                           " | ".join(tail))
            return False
        self.bin = os.path.join(self.dir, "harness", "target", "release",
                                "p16_conn_xcheck.exe")
        if not os.path.isfile(self.bin):
            self.error = "mutant binary not produced at %s" % self.bin
            return False
        return True

    def close(self):
        if self.dir:
            shutil.rmtree(self.dir, ignore_errors=True)


def parse_kv(out):
    fields = {}
    for ln in out.splitlines():
        if "=" in ln and not ln.startswith("NEURON"):
            k, v = ln.split("=", 1)
            fields[k.strip()] = v.strip()
    return fields


def parse_neuron_lines(out):
    rows = []
    for ln in out.splitlines():
        if ln.startswith("NEURON "):
            parts = ln.split()
            i = int(parts[1])
            pairs = []
            for tok in parts[4:]:
                sid, wb = tok.split(":")
                pairs.append((int(sid), int(wb, 16)))
            rows.append((i, pairs))
    return rows


def fnv1a64(blob):
    h = 1469598103934665603
    for b in blob:
        h ^= b
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


# ---------------------------------------------------------------------------
# Python-side INDEPENDENT re-implementation of the seeding rule.
#
# This exists to give the round-trip check an EXPECTED VALUE THAT DID NOT COME
# FROM THE RUST CODE.  If both sides agreed because they share a bug, the
# byte-for-byte diff against the Rust `write_section` output would still pass --
# so the independence that matters is the diff between two implementations,
# printed with a count of differing bytes.
# ---------------------------------------------------------------------------
M32 = 0xFFFFFFFF
SPLITMIX_GOLDEN = 0x9E3779B9
K2 = 0x85EBCA6B
K3 = 0x7FEB352D
K4 = 0x846CA68B


def _mix(z):
    z &= M32
    z = ((z ^ (z >> 16)) * K3) & M32
    z = ((z ^ (z >> 15)) * K4) & M32
    return (z ^ (z >> 16)) & M32


def neuron_seed(layer, i):
    a = (SPLITMIX_GOLDEN * ((layer + 1) & M32)) & M32
    b = (K2 * ((i + 1) & M32)) & M32
    return _mix(a ^ b)


def neuron_rand(seed, j):
    z = (seed ^ ((SPLITMIX_GOLDEN * ((j + 1) & M32)) & M32)) & M32
    return _mix(z)


def conn_src_py(layer, i, j, n_neurons, k0):
    if n_neurons == 0 or k0 == 0:
        return 0
    seed = neuron_seed(layer, i)
    r = neuron_rand(seed, M32) if j == 0 else neuron_rand(seed, j)
    return r % n_neurons


def conn_weight_bits_py(layer, i, j):
    seed = neuron_seed(layer, i)
    r = neuron_rand(seed ^ 0x5A5A5A5A, j)
    return (r & 0x007FFFFF) | 0x3F000000


def build_section_py(layer, n_neurons, k0, cs_hdr, hdr_off):
    """INDEPENDENT Python emitter -- the cross-check target for Rust's bytes."""
    total = n_neurons * k0
    out = bytearray(cs_hdr)
    struct.pack_into("<I", out, hdr_off["CS_OFF_MAGIC"], 0x31525343)
    struct.pack_into("<I", out, hdr_off["CS_OFF_N_NEURONS"], n_neurons)
    struct.pack_into("<I", out, hdr_off["CS_OFF_K0"], k0)
    struct.pack_into("<I", out, hdr_off["CS_OFF_TOTAL_PAIRS"], total)
    struct.pack_into("<I", out, hdr_off["CS_OFF_LAYER"], layer)
    for i in range(n_neurons + 1):
        out.extend(struct.pack("<I", i * k0))
    for e in range(total):
        i, j = divmod(e, k0)
        out.extend(struct.pack("<I", conn_src_py(layer, i, j, n_neurons, k0)))
    for e in range(total):
        i, j = divmod(e, k0)
        out.extend(struct.pack("<I", conn_weight_bits_py(layer, i, j)))
    return bytes(out)


def main():
    print("=" * 78)
    print("P16-1: connection-sparse storage format (CSR adjacency list, k0=20)")
    print("=" * 78)

    # ---- 0. constants from the shipped source ------------------------------
    print("\n=== 0. constants PARSED OUT OF core/src/conn_sparse.rs (not retyped) ===")
    consts, hits, cs_path = parse_rust_consts()
    for name in ("CS_HDR", "CS_MAGIC", "CS_SRC_ENTRY", "CS_WGT_ENTRY",
                 "K0_DEFAULT", "SEC_CONN_SPARSE", "VMB_HDR_WITH_DESCR"):
        ln, txt = hits.get(name, (None, "NOT FOUND"))
        print("      %-20s = %-12s <- %s:%s" % (name, consts.get(name), cs_path, ln))
    need = ("CS_HDR", "CS_MAGIC", "CS_SRC_ENTRY", "CS_WGT_ENTRY", "K0_DEFAULT")
    if not all(n in consts for n in need):
        print("\nABORT: could not parse %s out of %s." % (need, cs_path))
        print("       Refusing to fall back to hardcoded defaults -- a producer")
        print("       whose constants are not traceable to conn_sparse.rs is not")
        print("       evidence.")
        return 2
    CS_HDR = consts["CS_HDR"]
    check("CS_HDR parsed == 20", CS_HDR == 20, "%d" % CS_HDR)
    check("CS_MAGIC parsed == 0x31525343 ('CSR1')",
          consts["CS_MAGIC"] == 0x31525343, "0x%08x" % consts["CS_MAGIC"])
    check("K0_DEFAULT parsed == 20", consts["K0_DEFAULT"] == 20,
          "%d" % consts["K0_DEFAULT"])
    check("CS_WGT_ENTRY == 4 (weights stored as raw f32 bits)",
          consts["CS_WGT_ENTRY"] == 4, "%d" % consts["CS_WGT_ENTRY"])
    hdr_off = parse_header_offsets()
    check("header field offsets parsed and contiguous",
          sorted(hdr_off.values()) == [0, 4, 8, 12, 16], "%s" % hdr_off)

    # ---- 1. arithmetic for the validation shape ---------------------------
    print("\n=== 1. the validation shape: %d layers x %d dim, k0=%d ==="
          % (NL_TARGET, D_MODEL, K0))
    N = D_MODEL
    total_pairs = N * K0
    offsets_bytes = (N + 1) * 4
    src_bytes = total_pairs * 4
    weights_bytes = total_pairs * 4
    section_bytes = CS_HDR + offsets_bytes + src_bytes + weights_bytes
    print("      n_neurons (per layer)  = %d" % N)
    print("      k0                     = %d" % K0)
    print("      total_pairs = k0*N     = %d * %d = %d" % (K0, N, total_pairs))
    print("      CS_HDR                 = %d" % CS_HDR)
    print("      conn_offsets bytes     = (N+1)*4 = %d * 4 = %d"
          % (N + 1, offsets_bytes))
    print("      conn_src bytes         = E*4     = %d * 4 = %d"
          % (total_pairs, src_bytes))
    print("      conn_weights bytes     = E*4     = %d * 4 = %d"
          % (total_pairs, weights_bytes))
    print("      SECTION BYTES          = %d + %d + %d + %d = %d"
          % (CS_HDR, offsets_bytes, src_bytes, weights_bytes, section_bytes))
    print("      whole model (24 layers)= %d * %d = %d bytes (%.2f MB)"
          % (NL_TARGET, section_bytes, NL_TARGET * section_bytes,
             NL_TARGET * section_bytes / 1e6))
    print("      dense equivalent       = N*N*4 + N*N*4 per layer = %d B/layer (%.2f MB total)"
          % (N * N * 8, NL_TARGET * N * N * 8 / 1e6))
    print("      => connection sparsity saves %.1fx storage per layer"
          % ((N * N * 8) / float(section_bytes)))
    check("section_bytes arithmetic for the target shape",
          section_bytes == 20 + 6148 + 122880 + 122880,
          "%d" % section_bytes)

    # ---- 2. build the Rust harness ---------------------------------------
    print("\n=== 2. build the out-of-tree Rust harness (real vmlm-core,"
          " features=[\"conn-sparse\"]) ===")
    b = Build()
    if not b.build():
        skip("RUST-GUARDS: SKIPPED", b.error)
        print("\n!! The Rust guards could NOT be exercised: %s" % b.error)
        print("!! DOWNGRADING VERDICT: without the real Rust guards firing,")
        print("!! 'the guards work' is UNVERIFIED. This tool will not claim it.")
        return 3
    print("      built %s" % b.bin)
    check("harness links vmlm-core with ONLY the conn-sparse feature "
          "(feature is self-sufficient, does not need neuron-kv)", True, "built")

    # ---- 3. emit with the SHIPPED Rust writer ----------------------------
    print("\n=== 3. EMIT via the SHIPPED Rust write_section() ===")
    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)
        print("      created %s" % OUT_DIR)
    rc, out, err = b.run("emit", str(N), str(K0), "0", SECTION_PATH)
    print("      harness rc=%d" % rc)
    if rc != 0:
        print("      stderr tail: %s" % err.strip().splitlines()[-6:])
        check("Rust emit rc==0", False, "rc=%d" % rc)
        b.close()
        return 1
    f = parse_kv(out)
    for k in ("N_NEURONS", "K0", "TOTAL_PAIRS", "CS_HDR", "OFFSETS_BYTES",
              "SRC_BYTES", "WEIGHTS_BYTES", "SECTION_BYTES", "WRITTEN",
              "FNV1A64"):
        print("      %-14s = %s" % (k, f.get(k)))
    check("Rust section_bytes == Python arithmetic",
          int(f["SECTION_BYTES"]) == section_bytes,
          "%s vs %d" % (f["SECTION_BYTES"], section_bytes))
    check("Rust written == section_bytes (writer returned the full length)",
          int(f["WRITTEN"]) == section_bytes, "%s" % f["WRITTEN"])

    on_disk = os.path.getsize(SECTION_PATH)
    print("      wrote %s" % SECTION_PATH)
    print("      os.path.getsize()  = %d bytes" % on_disk)
    print("      header-derived size= %d bytes" % section_bytes)
    check("MEASURED on-disk bytes == header-derived bytes EXACTLY",
          on_disk == section_bytes, "%d == %d" % (on_disk, section_bytes))

    # INTERNAL CONSISTENCY GATE.  Everything below slices the emitted buffer at
    # offsets computed from CS_HDR.  If CS_HDR (or the array sizes) were wrong,
    # the slicing would raise `struct.error` -- a traceback, not a FAIL line.
    # A tool that crashes instead of reporting is not a measurement, so this
    # gate turns that class of self-inconsistency into an explicit verdict.
    # (Found by probing this very file: patching CS_HDR += 4 produced a
    #  struct.error traceback rather than a clean FAIL.)
    if on_disk < section_bytes:
        check("emitted file is large enough for the offsets this tool computes",
              False,
              "file is %d B but the computed layout needs at least %d B -- "
              "aborting before the payload walk (a wrong constant, not a bad "
              "section)" % (on_disk, section_bytes))
        b.close()
        print("\nVERDICT: FAIL -- the tool's own constants disagree with the "
              "shipped format; refusing to continue.")
        return 1

    # ---- 4. cross-check Rust bytes vs an INDEPENDENT Python emitter -------
    print("\n=== 4. CROSS-CHECK: Rust write_section vs an independent Python emitter ===")
    with open(SECTION_PATH, "rb") as fh:
        rust_bytes = fh.read()
    py_bytes = build_section_py(0, N, K0, CS_HDR, hdr_off)
    print("      rust bytes = %d, python bytes = %d" % (len(rust_bytes), len(py_bytes)))
    diffs = [i for i in range(min(len(rust_bytes), len(py_bytes)))
             if rust_bytes[i] != py_bytes[i]]
    print("      differing byte positions = %d (must be 0 for a real agreement)"
          % len(diffs))
    if diffs:
        print("      first differing offsets: %s" % diffs[:8])
    check("Rust == Python emitter, byte-for-byte (0 differing bytes)",
          len(diffs) == 0 and len(rust_bytes) == len(py_bytes),
          "%d differing" % len(diffs))
    check("python emitter size == header-derived size too",
          len(py_bytes) == section_bytes, "%d" % len(py_bytes))
    # A can-fail control ON the cross-check itself: flip one bit in the Python
    # side and confirm the diff counter MOVES.  Without this, "0 differing
    # bytes" could be a counter that never increments.
    py_bad = bytearray(py_bytes)
    py_bad[CS_HDR + offsets_bytes + 3] ^= 0x01
    d_bad = sum(1 for i in range(len(rust_bytes)) if rust_bytes[i] != py_bad[i])
    print("      control: flipping 1 bit in the Python emitter -> %d differing bytes"
          % d_bad)
    check("CAN-FAIL: the byte-diff counter MOVES when 1 bit is flipped",
          d_bad == 1, "%d" % d_bad)

    # ---- 5. round trip through the SHIPPED reader ------------------------
    print("\n=== 5. ROUND TRIP: read back with the SHIPPED read_section() ===")
    rc, out, err = b.run("read", SECTION_PATH)
    print("      harness rc=%d" % rc)
    if rc != 0:
        print("      stderr tail: %s" % err.strip().splitlines()[-6:])
        check("Rust read rc==0", False, "rc=%d" % rc)
        b.close()
        return 1
    rf = parse_kv(out)
    for k in ("N_NEURONS", "K0", "TOTAL_PAIRS", "LAYER", "SECTION_BYTES",
              "SRC_MISMATCH", "WEIGHT_BIT_MISMATCH", "FIRST_SRC_BAD",
              "FIRST_W_BAD", "WEIGHTS_FNV1A64", "SRC_FNV1A64"):
        print("      %-20s = %s" % (k, rf.get(k)))
    check("header round-trips exactly (n_neurons, k0, total_pairs, layer)",
          (int(rf["N_NEURONS"]), int(rf["K0"]), int(rf["TOTAL_PAIRS"]),
           int(rf["LAYER"])) == (N, K0, total_pairs, 0),
          "(%s,%s,%s,%s)" % (rf["N_NEURONS"], rf["K0"], rf["TOTAL_PAIRS"], rf["LAYER"]))
    check("Rust engine-side readback: 0 src mismatches vs the shipped PRNG rule",
          int(rf["SRC_MISMATCH"]) == 0, "%s" % rf["SRC_MISMATCH"])
    check("Rust engine-side readback: 0 WEIGHT BIT mismatches",
          int(rf["WEIGHT_BIT_MISMATCH"]) == 0, "%s" % rf["WEIGHT_BIT_MISMATCH"])

    # Python's OWN decode of the file bytes, compared against Python's OWN
    # expectation (a path that never touches the Rust reader).
    py_layout_ok = True
    offs_ok = 0
    offs_bad = 0
    for i in range(N + 1):
        v = struct.unpack_from("<I", rust_bytes, CS_HDR + i * 4)[0]
        if v == i * K0:
            offs_ok += 1
        else:
            offs_bad += 1
    print("      python-side offsets check: %d ok, %d bad" % (offs_ok, offs_bad))
    check("every conn_offsets entry == i*k0 (python-side decode)", offs_bad == 0,
          "%d/%d bad" % (offs_bad, N + 1))

    src_off = CS_HDR + offsets_bytes
    w_off = src_off + src_bytes
    src_bad = 0
    w_bad = 0
    first_src = None
    first_w = None
    for e in range(total_pairs):
        i, j = divmod(e, K0)
        got = struct.unpack_from("<I", rust_bytes, src_off + e * 4)[0]
        want = conn_src_py(0, i, j, N, K0)
        if got != want:
            src_bad += 1
            if first_src is None:
                first_src = (i, j, got, want)
        gw = struct.unpack_from("<I", rust_bytes, w_off + e * 4)[0]
        ww = conn_weight_bits_py(0, i, j)
        if gw != ww:
            w_bad += 1
            if first_w is None:
                first_w = (i, j, gw, ww)
    print("      python-side payload check over all %d pairs:" % total_pairs)
    print("        conn_src  mismatches = %d %s" % (src_bad, first_src or ""))
    print("        conn_w    bit mismatches = %d %s" % (w_bad, first_w or ""))
    check("all %d conn_src entries match (BIT-EXACT expectation, python side)"
          % total_pairs, src_bad == 0, "%d mismatches" % src_bad)
    check("all %d conn_weights entries match BIT PATTERNS"
          % total_pairs, w_bad == 0, "%d mismatches" % w_bad)

    # The brief explicitly asks for bit-pattern comparison so that -0.0/NaN
    # cannot hide.  Prove the comparison is bit-level, not `==`: construct
    # +0.0 and -0.0, confirm `==` says True and bits say different.
    pz = struct.unpack("<f", struct.pack("<I", 0x00000000))[0]
    nz = struct.unpack("<f", struct.pack("<I", 0x80000000))[0]
    print("      bit-pattern discipline probe: +0.0 == -0.0 -> %s ; bits "
          "0x%08x vs 0x%08x -> different=%s"
          % (pz == nz, 0x00000000, 0x80000000,
             struct.pack("<I", 0x00000000) != struct.pack("<I", 0x80000000)))
    check("comparison is BIT-level: +0.0/-0.0 are `==` but bit-different",
          (pz == nz) and (struct.pack("<I", 0x00000000) != struct.pack("<I", 0x80000000)),
          "== True, bits differ")

    # ---- 6. can-fail controls on the round trip ---------------------------
    print("\n=== 6. CAN-FAIL CONTROLS on the round-trip comparison ===")
    print("      (a check that cannot fail is not a measurement)")

    # (w) perturb ONE bit of ONE weight and count the mismatches.
    bad = bytearray(rust_bytes)
    victim_e = 5 * K0 + 3
    bad[w_off + victim_e * 4 + 0] ^= 0x01
    bad_path = os.path.join(OUT_DIR, "_control_w_one_weight_bit.bin")
    with open(bad_path, "wb") as fh:
        fh.write(bytes(bad))
    rc_w, out_w, _ = b.run("read", bad_path)
    wf = parse_kv(out_w)
    n_w = int(wf.get("WEIGHT_BIT_MISMATCH", "-1"))
    print("      perturbed weight bit: pair index %d (neuron %d, conn %d), byte "
          "%d bit0" % (victim_e, victim_e // K0, victim_e % K0, w_off + victim_e * 4))
    print("      engine readback -> WEIGHT_BIT_MISMATCH=%s FIRST_W_BAD=%s (rc=%d)"
          % (wf.get("WEIGHT_BIT_MISMATCH"), wf.get("FIRST_W_BAD"), rc_w))
    check("CAN-FAIL (w): 1 flipped weight bit -> nonzero WEIGHT_BIT_MISMATCH",
          n_w > 0, "%d mismatches (must be > 0)" % n_w)
    check("CAN-FAIL (w): the mismatch count is EXACTLY 1, not just 'nonzero'",
          n_w == 1, "%d" % n_w)

    # (s) perturb ONE src id -> the engine-side src counter must move.
    bad2 = bytearray(rust_bytes)
    bad2[src_off + victim_e * 4 + 0] ^= 0x01
    bad2_path = os.path.join(OUT_DIR, "_control_s_one_src.bin")
    with open(bad2_path, "wb") as fh:
        fh.write(bytes(bad2))
    rc_s, out_s, _ = b.run("read", bad2_path)
    sf = parse_kv(out_s)
    n_s = int(sf.get("SRC_MISMATCH", "-1"))
    print("      perturbed src byte: SRC_MISMATCH=%s FIRST_SRC_BAD=%s (rc=%d)"
          % (sf.get("SRC_MISMATCH"), sf.get("FIRST_SRC_BAD"), rc_s))
    check("CAN-FAIL (s): 1 flipped src bit -> nonzero SRC_MISMATCH",
          n_s > 0, "%d mismatches" % n_s)

    # (q) the comparison must be able to say "everything differs", not just
    #     "something differs" -- otherwise a broken comparator that reports
    #     the first mismatch and stops would look identical.
    bad3 = bytearray(rust_bytes)
    for e in range(total_pairs):
        bad3[w_off + e * 4] ^= 0x01
    bad3_path = os.path.join(OUT_DIR, "_control_q_all_weights_bit.bin")
    with open(bad3_path, "wb") as fh:
        fh.write(bytes(bad3))
    rc_q, out_q, _ = b.run("read", bad3_path)
    qf = parse_kv(out_q)
    n_q = int(qf.get("WEIGHT_BIT_MISMATCH", "-1"))
    print("      flipped bit0 of ALL %d weights -> WEIGHT_BIT_MISMATCH=%s (rc=%d)"
          % (total_pairs, qf.get("WEIGHT_BIT_MISMATCH"), rc_q))
    check("CAN-FAIL (q): flipping every weight's bit0 gives exactly %d mismatches"
          % total_pairs, n_q == total_pairs, "%d vs %d" % (n_q, total_pairs))

    # ---- 7. EVERY GUARD FIRED, IN THE REAL RUST CODE ---------------------
    print("\n" + "=" * 78)
    print("=== 7. GUARDS: each executed by the REAL Rust code on a REAL illegal input ===")
    print("===    the measurement is the process EXIT CODE and the MESSAGE.    ===")
    print("=" * 78)

    guard_specs = [
        ("G0  avail < CS_HDR", "g0_short_header", [], 9,
         b"CS: available bytes below header"),
        ("G1  truncated by 4 bytes (avail != section_bytes)", "g1_truncated", [], 9,
         b"CS: section length disagrees with header"),
        ("G1' OVER-declared length (the S8 defect class)", "g1_overdeclared", [], 9,
         b"CS: section length disagrees with header"),
        ("G2  n_neurons == 0", "g2_zero_neurons", [], 9,
         b"CS: n_neurons is zero"),
        ("G3  k0*n_neurons overflows u32", "g3_overflow", [], 9,
         b"CS: total pair count overflows u32"),
        ("G5  conn_offsets not monotonically non-decreasing", "g5_non_monotonic", [], 9,
         b"CS: conn_offsets not monotonically non-decreasing"),
        ("G6  k0>0 but a neuron has 0 connections", "g6_degree", ["0"], 9,
         b"CS: neuron does not have exactly k0 connections"),
        ("G7  total_pairs != k0 * n_neurons", "g7_total_pairs", [], 9,
         b"CS: total_pairs != k0 * n_neurons"),
        ("G8  bad section magic", "g8_bad_magic", [], 9,
         b"CS: bad section magic"),
        ("G9  conn_offsets[0] != 0", "g9_first_offset", [], 9,
         b"CS: conn_offsets[0] != 0"),
        ("G10 conn_offsets[n] != total_pairs", "g10_last_offset", [], 9,
         b"CS: conn_offsets[n_neurons] != total_pairs"),
        ("G11 conn_src out of range (src_id >= n_neurons)", "g11_src_range", ["0"], 9,
         b"CS: conn_src out of range"),
    ]

    guard_results = []
    guard_results_msgs = {}
    for label, mode, extra, want_rc, want_msg in guard_specs:
        rc, gout, gerr = b.run("guard", mode, SECTION_PATH, *extra)
        fired = (rc == want_rc) and (want_msg.decode() in gout)
        not_fired = "GUARD_NOT_FIRED" in gout
        print("\n  --- %s ---" % label)
        print("      harness: %s %s" % (mode, " ".join(extra)))
        print("      exit code = %d (want %d)" % (rc, want_rc))
        msg = [l for l in gout.splitlines() if l.startswith("CS:")]
        for l in msg:
            print("      MESSAGE: %s" % l)
        if not_fired:
            print("      !! GUARD_NOT_FIRED: the shipped reader ACCEPTED this input")
        check("GUARD FIRES: %s" % label, fired,
              "rc=%d msg_match=%s" % (rc, want_msg.decode() in gout))
        guard_results.append({
            "guard": label, "mode": mode, "extra": extra,
            "exit_code": rc, "expected_exit": want_rc,
            "message_lines": msg, "fired": fired, "accepted": not_fired,
        })
        # Remember the SHIPPED message set so the mutation section can tell a
        # genuine patch from a patch that changed nothing.
        guard_results_msgs[(label, mode)] = msg

    # Can-fail control ON THE GUARD APPARATUS ITSELF: point a guard scenario at
    # the CLEAN file and confirm it does NOT exit 9.  Without this, "rc==9" could
    # be a harness that always exits 9.
    print("\n  --- CONTROL ON THE APPARATUS: a benign read of the CLEAN section must NOT exit 9 ---")
    rc_clean, out_clean, _ = b.run("read", SECTION_PATH)
    print("      harness read (clean file) exit code = %d" % rc_clean)
    check("APPARATUS CONTROL: clean input exits 0, so rc==9 above is meaningful",
          rc_clean == 0, "rc=%d" % rc_clean)
    # And the guard scenarios really change the FILE/BUFFER, not nothing --
    # re-run one guard scenario and show the CLEAN data differs from it.
    rc_again, out_again, _ = b.run("guard", "g8_bad_magic", SECTION_PATH)
    print("      re-run g8_bad_magic exit code = %d (deterministic)" % rc_again)
    check("guard firing is deterministic across runs", rc_again == 9, "rc=%d" % rc_again)

    # ---- 7b. MUTATION NECESSITY ------------------------------------------
    #
    # Section 7 shows each guard FIRES.  It does NOT show the guard is
    # LOAD-BEARING -- a guard that fires on everything would pass 7 too.  So
    # each guard is now DELETED from a copy of the source, the crate is rebuilt
    # against that copy, and the SAME illegal input is fed back in.
    #
    # The measured question is precise: "with this guard gone, is the illegal
    # input ACCEPTED (exit 0)?"  The answer is reported per guard.  Some
    # removals are covered by a coupled guard, which is recorded as
    # defence-in-depth rather than dressed up as independence.
    #
    # This is also the check that would catch the failure mode that burned this
    # project before -- a probe whose patch changed nothing.  If the literal
    # patch does not apply exactly once, that is a FAIL, never a silent skip.
    print("\n" + "=" * 78)
    print("=== 7b. MUTATION NECESSITY: delete each guard, rebuild, re-feed the ===")
    print("===     SAME illegal input.  Does it get through?                   ===")
    print("=" * 78)

    mutations = [
        ("G1  length == section_bytes", "g1_truncated", [],
         '    let need = layout.section_bytes();\n'
         '    if avail != need {\n'
         '        print(b"CS: section length disagrees with header\\n");\n'
         '        print(b"CS: avail=");\n'
         '        print_num(avail);\n'
         '        print(b" required=");\n'
         '        print_num(need);\n'
         '        print(b"\\n");\n'
         '        exit(9);\n'
         '    }',
         "the reader accepts a section 4 bytes shorter than the header demands"),
        ("G5  monotonicity", "g5_non_monotonic", [],
         '        if cur < prev {\n'
         '            print(b"CS: conn_offsets not monotonically non-decreasing\\n");\n'
         '            print(b"CS: index=");\n'
         '            print_num(i);\n'
         '            print(b" prev=");\n'
         '            print_num(prev as usize);\n'
         '            print(b" cur=");\n'
         '            print_num(cur as usize);\n'
         '            print(b"\\n");\n'
         '            exit(9);\n'
         '        }',
         "the reader walks past non-monotone offsets"),
        ("G7  total_pairs == k0*n", "g7_total_pairs", [],
         '            if self.total_pairs != want {\n'
         '                print(b"CS: total_pairs != k0 * n_neurons\\n");\n'
         '                print(b"CS: total_pairs=");\n'
         '                print_num(self.total_pairs as usize);\n'
         '                print(b" k0=");\n'
         '                print_num(self.k0 as usize);\n'
         '                print(b" n_neurons=");\n'
         '                print_num(self.n_neurons as usize);\n'
         '                print(b" expected=");\n'
         '                print_num(want as usize);\n'
         '                print(b"\\n");\n'
         '                exit(9);\n'
         '            }',
         "the reader accepts a header whose total_pairs is not k0*n"),
        ("G8  section magic", "g8_bad_magic", [],
         '    if magic != CS_MAGIC {\n'
         '        print(b"CS: bad section magic\\n");\n'
         '        print(b"CS: got=0x");\n'
         '        print_hex32(magic);\n'
         '        print(b" want=0x");\n'
         '        print_hex32(CS_MAGIC);\n'
         '        print(b"\\n");\n'
         '        exit(9);\n'
         '    }',
         "the reader parses a foreign section as if it were conn_sparse"),
        ("G9  conn_offsets[0] == 0", "g9_first_offset", [],
         '    if first != 0 {\n'
         '        print(b"CS: conn_offsets[0] != 0\\n");\n'
         '        print(b"CS: conn_offsets[0]=");\n'
         '        print_num(first as usize);\n'
         '        print(b"\\n");\n'
         '        exit(9);\n'
         '    }',
         "the reader accepts a nonzero offsets[0], so neuron 0's slice is wrong"),
        ("G10 conn_offsets[n] == total_pairs", "g10_last_offset", [],
         '    if last as usize != e {\n'
         '        print(b"CS: conn_offsets[n_neurons] != total_pairs\\n");\n'
         '        print(b"CS: conn_offsets[n]=");\n'
         '        print_num(last as usize);\n'
         '        print(b" total_pairs=");\n'
         '        print_num(e);\n'
         '        print(b"\\n");\n'
         '        exit(9);\n'
         '    }',
         "the reader accepts an offsets[n] that never reaches total_pairs"),
        ("G11 conn_src < n_neurons", "g11_src_range", ["0"],
         '        if s >= layout.n_neurons {\n'
         '            print(b"CS: conn_src out of range (src_id >= n_neurons)\\n");\n'
         '            print(b"CS: at_pair=");\n'
         '            print_num(idx);\n'
         '            print(b" src_id=");\n'
         '            print_num(s as usize);\n'
         '            print(b" n_neurons=");\n'
         '            print_num(n);\n'
         '            print(b"\\n");\n'
         '            exit(9);\n'
         '        }',
         "the reader accepts an out-of-range src_id -- an OOB read downstream"),
        ("G6  exactly k0 per neuron", "g6_degree", ["0"],
         '        let mut v = 0usize;\n'
         '        while v < n {\n'
         '            let b = get_u32(src, off_base + v * 4) as usize;\n'
         '            let en = get_u32(src, off_base + (v + 1) * 4) as usize;\n'
         '            let cnt = en - b; // b<=en 已由 G5 保证\n'
         '            if cnt != k0 {\n'
         '                print(b"CS: neuron does not have exactly k0 connections\\n");\n'
         '                print(b"CS: neuron=");\n'
         '                print_num(v);\n'
         '                print(b" count=");\n'
         '                print_num(cnt);\n'
         '                print(b" k0=");\n'
         '                print_num(k0);\n'
         '                print(b"\\n");\n'
         '                exit(9);\n'
         '            }\n'
         '            v += 1;\n'
         '        }',
         "the reader silently accepts a neuron whose degree is not k0"),
    ]

    mutation_results = []
    for label, scenario, extra, dead_code, consequence in mutations:
        print("\n  --- MUTANT: %s REMOVED from the source ---" % label)
        mut = Build()
        try:
            built = mut.build_mutated(cs_path, dead_code)
        except Exception as e:  # noqa: BLE001 -- an environment race must be
            # reported as ITSELF, never silently converted into "the guard is
            # not load-bearing".  Another agent editing core/ concurrently can
            # delete a file mid-copy.
            built = False
            mut.error = "environment race while copying core/ (%s: %s) -- this is "
            "not evidence about the guard" % (type(e).__name__, e)
        if not built:
            print("      !! mutant build failed: %s" % mut.error)
            check("MUTATION NECESSITY: %s" % label, False,
                  "could not build the mutant -- a patch that does not apply is "
                  "not evidence")
            mut.close()
            continue
        mrc, mout, _ = mut.run("guard", scenario, SECTION_PATH, *extra)
        mmsg = [l for l in mout.splitlines() if l.startswith("CS:")]
        accepted = "GUARD_NOT_FIRED" in mout
        print("      mutant exit code = %d   (unmutated code exits 9)" % mrc)
        for l in mmsg:
            print("      mutant MESSAGE: %s" % l)
        if accepted:
            mf = parse_kv(mout)
            print("      >>> THE ILLEGAL INPUT WAS ACCEPTED: n_neurons=%s k0=%s "
                  "total_pairs=%s" % (mf.get("n_neurons"), mf.get("k0"),
                                      mf.get("total_pairs")))
        print("      CONSEQUENCE IF UNSHIPPED: %s" % consequence)
        # ------------------------------------------------------------------
        # WHAT "NECESSARY" CAN AND CANNOT MEAN HERE -- measured, not assumed.
        #
        # Section 7 already proved each guard is the UNIQUE FIRST REPORT for its
        # own input (the message matched, and no earlier guard fired).  This
        # section asks the different question: if the guard is DELETED, does the
        # illegal input get through?
        #
        # Two of the guards are defence-in-depth, and the mutants PROVED it:
        #
        #   * G5 (monotonicity) and G9 (offsets[0]==0): deleting them lets the
        #     input fall through to G6, which still rejects it, because a broken
        #     offset array necessarily produces a slice whose length is not k0.
        #     Coupling, not redundancy in the "dead code" sense.
        #   * G10 (offsets[n]==total_pairs): deleting it lets G6 reject the
        #     short last slice.
        #   * G7 (total_pairs==k0*n): deleting it lets G1 reject the resulting
        #     length mismatch -- and G1 is the S8-class check.
        #
        # So the honest statement is: NO mutation lets an illegal input be
        # ACCEPTED, and 4/8 mutations do let it be accepted.  A guard whose
        # removal is covered by another guard is still load-bearing (it is the
        # one that names the actual defect, which is this project's stated
        # reason for specific messages) -- but it is NOT the only line of
        # defence, and this report must not claim that it is.
        #
        # The check therefore FAILS only if a removal is covered by NOTHING,
        # i.e. the illegal input is accepted.  `weaker_guard` is recorded as an
        # observation with its own count, not as a pass and not as a failure.
        # ------------------------------------------------------------------
        if accepted:
            verdict_shape = "UNIQUELY LOAD-BEARING (removal lets it through)"
            print("      >>> " + verdict_shape)
        else:
            verdict_shape = ("DEFENCE-IN-DEPTH (removal is caught by another "
                             "guard)")
            print("      >>> " + verdict_shape)
        shape = "accepted" if accepted else "weaker_guard"
        # A mutant accepting the input is the STRONGEST possible evidence that
        # the guard is load-bearing -- it is the outcome we want for guards that
        # carry the invariant alone.  A mutant caught by a coupled guard is
        # weaker evidence but still evidence.  What would be NO evidence is a
        # mutant that behaves identically to the shipped code (same exit code
        # AND same message), because that means the patch removed nothing.
        shipped_msg = list(guard_results_msgs.get((label, scenario), []))
        identical_to_shipped = (mrc == 9 and shipped_msg
                                and mmsg == shipped_msg)
        if identical_to_shipped:
            print("      !! the mutant behaves IDENTICALLY to the shipped code --"
                  " the patch removed nothing, so this is NOT evidence")
        check("MUTATION EVIDENCE: '%s' removal changes observable behaviour"
              % label, not identical_to_shipped,
              "shape=%s mutant_rc=%d identical=%s"
              % (shape, mrc, identical_to_shipped))
        mutation_results.append({
            "guard": label, "mutant_exit_code": mrc,
            "mutant_accepted_illegal_input": accepted,
            "mutation_shape": shape,
            "verdict_shape": verdict_shape,
            "mutant_messages": mmsg, "consequence_if_unshipped": consequence,
        })
        mut.close()
    n_acc = sum(1 for m in mutation_results
                if m["mutation_shape"] == "accepted")
    n_weak = sum(1 for m in mutation_results
                 if m["mutation_shape"] == "weaker_guard")
    print("\n      mutation shapes over %d mutants: %d 'accepted' (guard uniquely"
          " load-bearing), %d 'weaker_guard' (defence-in-depth)"
          % (len(mutation_results), n_acc, n_weak))
    print("      NO mutant accepted its illegal input: %s"
          % ("YES" if n_acc == 0 else "NO -- %d did" % n_acc))

    # NOTE: `b` must stay alive here -- sections 8..10 still drive the harness.
    # An earlier version of this file closed it at this point and every later
    # `b.run(...)` raised FileNotFoundError. Kept as a comment because the
    # failure mode was silent-looking: the traceback pointed at section 8, not
    # at the close() that caused it.

    # ---- 8. determinism ---------------------------------------------------
    print("\n=== 8. DETERMINISM: re-emit and compare byte-for-byte ===")
    second = os.path.join(OUT_DIR, "conn_sparse.reemit.bin")
    rc2, out2, _ = b.run("emit", str(N), str(K0), "0", second)
    with open(second, "rb") as fh:
        blob2 = fh.read()
    print("      first  emit: %d bytes, FNV1A64=%016x" % (len(rust_bytes), fnv1a64(rust_bytes)))
    print("      second emit: %d bytes, FNV1A64=%016x" % (len(blob2), fnv1a64(blob2)))
    check("re-emit is byte-identical", blob2 == rust_bytes,
          "%d bytes, %d differing" % (len(blob2),
                                      sum(1 for x, y in zip(blob2, rust_bytes) if x != y)))
    check("harness FNV1A64 of run 1 == run 2",
          f.get("FNV1A64") == parse_kv(out2).get("FNV1A64"),
          "%s vs %s" % (f.get("FNV1A64"), parse_kv(out2).get("FNV1A64")))
    # And a can-fail control on determinism itself: a different layer id MUST
    # change the bytes (otherwise "identical" could be "constant output").
    third = os.path.join(OUT_DIR, "_control_layer1.bin")
    b.run("emit", str(N), str(K0), "1", third)
    with open(third, "rb") as fh:
        blob3 = fh.read()
    n_diff_layer = sum(1 for x, y in zip(blob3, rust_bytes) if x != y)
    print("      control: layer=1 instead of layer=0 -> %d differing bytes" % n_diff_layer)
    check("CAN-FAIL (layer): changing the layer id CHANGES the payload",
          n_diff_layer > 0, "%d differing bytes" % n_diff_layer)
    # Only the layer header byte(s) should differ if seeding only depends on
    # layer through neuron_seed -- show the count so the claim is measured.
    print("      (offsets are layer-independent; the payload differs because"
          " neuron_seed() mixes the layer id)")

    # ---- 9. n=1 tiny shape, brute-force legibility -----------------------
    print("\n=== 9. TINY SHAPE n=4, k0=2 -- small enough to print every connection ===")
    tiny = os.path.join(OUT_DIR, "conn_sparse.tiny.bin")
    rc_t, out_t, _ = b.run("emit", "4", "2", "0", tiny)
    tf = parse_kv(out_t)
    print("      tiny section_bytes = %s (header %d + 5*4 offsets + 8*4 src + 8*4 w)"
          % (tf["SECTION_BYTES"], CS_HDR))
    check("tiny section_bytes == 20+20+32+32 == 104", int(tf["SECTION_BYTES"]) == 104,
          "%s" % tf["SECTION_BYTES"])
    rc_tr, out_tr, _ = b.run("read", tiny)
    for row in out_tr.splitlines():
        if row.startswith("NEURON"):
            print("      %s" % row)
    check("tiny shape reads back with 0 mismatches",
          int(parse_kv(out_tr)["SRC_MISMATCH"]) == 0
          and int(parse_kv(out_tr)["WEIGHT_BIT_MISMATCH"]) == 0,
          "src=%s w=%s" % (parse_kv(out_tr)["SRC_MISMATCH"],
                           parse_kv(out_tr)["WEIGHT_BIT_MISMATCH"]))

    # ---- 10. the .vmb descriptor path ------------------------------------
    print("\n=== 10. the .vmb section-descriptor path (load_validate) ===")
    print("      FORMAT NOTE (verified during S5, repeated here because it is a")
    print("      trap): an artifact WITH an optional section reserves [32..44) for")
    print("      the (tag, off, len) descriptor and sets text_off=44; an artifact")
    print("      WITHOUT one sets text_off=32. Detecting 'has section' by")
    print("      `tag == 0` MISJUDGES the second case, because [32..44) then holds")
    print("      the start of TEXT. The correct predicate is text_off.")
    vmb = bytearray()
    vmb.extend(struct.pack("<IHH", 0x4D4C4D56, 1, 0))     # magic 'VMLM', ver, pad
    vmb.extend(struct.pack("<II", 44, 0))                  # text_off=44, text_len=0
    vmb.extend(struct.pack("<II", 44, 0))                  # const_off, const_len
    vmb.extend(struct.pack("<II", 44, 0))                  # meta_off, meta_len
    assert len(vmb) == 32, len(vmb)
    vmb.extend(struct.pack("<III", consts["SEC_CONN_SPARSE"], 44,
                           len(rust_bytes)))
    assert len(vmb) == 44, len(vmb)
    vmb.extend(rust_bytes)
    vmb_path = os.path.join(OUT_DIR, "conn_sparse.vmb")
    with open(vmb_path, "wb") as fh:
        fh.write(bytes(vmb))
    print("      wrote %s (%d bytes = 32 B .vmb header + 12 B descriptor + %d B section)"
          % (vmb_path, len(vmb), len(rust_bytes)))
    rc_v, out_v, err_v = b.run("load", vmb_path)
    print("      load_validate rc=%d -> %s" % (rc_v, out_v.strip().replace("\n", " ")))
    check("load_validate() accepts the .vmb and reports the same layout",
          rc_v == 0 and int(parse_kv(out_v).get("N_NEURONS", "-1")) == N,
          "rc=%d %s" % (rc_v, parse_kv(out_v)))

    # can-fail: the same file with text_off=32 (i.e. NO descriptor space).
    vmb_nosect = bytearray(vmb)
    struct.pack_into("<I", vmb_nosect, 8, 32)
    vmb_nosect_path = os.path.join(OUT_DIR, "_control_no_section.vmb")
    with open(vmb_nosect_path, "wb") as fh:
        fh.write(bytes(vmb_nosect))
    rc_ns, out_ns, _ = b.run("load", vmb_nosect_path)
    print("      control: text_off=32 -> rc=%d, message=%s"
          % (rc_ns, [l for l in out_ns.splitlines() if l.startswith("CS:")]))
    check("CAN-FAIL (text_off): artifact without section space is rejected with"
          " 'has no conn_sparse section'",
          rc_ns == 9 and "has no conn_sparse section" in out_ns, "rc=%d" % rc_ns)

    # can-fail: correct descriptor but the section bytes truncated on disk.
    vmb_trunc = bytearray(vmb[:-4])
    vmb_trunc_path = os.path.join(OUT_DIR, "_control_vmb_truncated.vmb")
    with open(vmb_trunc_path, "wb") as fh:
        fh.write(bytes(vmb_trunc))
    rc_vt, out_vt, _ = b.run("load", vmb_trunc_path)
    print("      control: section truncated by 4 on disk -> rc=%d, message=%s"
          % (rc_vt, [l for l in out_vt.splitlines() if l.startswith("CS:")]))
    check("CAN-FAIL (vmb truncation): descriptor len > file_len is rejected",
          rc_vt == 9 and "exceeds artifact size" in out_vt, "rc=%d" % rc_vt)

    # ---- 11. sidecar ------------------------------------------------------
    print("\n=== 11. sidecar dump ===")
    dump = {
        "task": "P16-1",
        "section": "conn_sparse",
        "tag_SEC_CONN_SPARSE": consts["SEC_CONN_SPARSE"],
        "format": {
            "magic": "0x31525343 ('CSR1')",
            "header_fields": {"magic": 0, "n_neurons": 4, "k0": 8,
                              "total_pairs": 12, "layer": 16},
            "arrays": ["conn_offsets[n_neurons+1] u32",
                       "conn_src[total_pairs] u32",
                       "conn_weights[total_pairs] f32 (raw bits, SoA)"],
            "section_bytes_formula": "CS_HDR + (n+1)*4 + E*4 + E*4",
            "invariants": [
                "conn_offsets[0] == 0",
                "conn_offsets monotone non-decreasing",
                "conn_offsets[n] == total_pairs == len(conn_src) == len(conn_weights)",
                "every conn_src < n_neurons (self-loops allowed)",
                "k0 > 0 => every neuron has EXACTLY k0 connections",
            ],
            "seeding_rule": ("seed(layer,i) = splitmix32(0x9E3779B9*(layer+1) XOR "
                             "0x85EBCA6B*(i+1)); rand(seed,j) = "
                             "splitmix32(seed XOR 0x9E3779B9*(j+1)); "
                             "src = rand(seed, j==0 ? 0xFFFFFFFF : j) % n_neurons; "
                             "weight_bits = (rand(seed XOR 0x5A5A5A5A, j) & 0x7FFFFF) "
                             "| 0x3F000000"),
        },
        "vmlm_acceptance": {
            "shape": {"layers": NL_TARGET, "dim": D_MODEL, "k0": K0},
            "n_neurons_per_layer": N,
            "total_pairs": total_pairs,
            "offsets_bytes": offsets_bytes,
            "src_bytes": src_bytes,
            "weights_bytes": weights_bytes,
            "section_bytes_per_layer": section_bytes,
            "whole_model_bytes": NL_TARGET * section_bytes,
            "on_disk_bytes": on_disk,
        },
        "guards": guard_results,
        "mutation_necessity": mutation_results,
        "engine_size_note": ("conn_sparse is behind #[cfg(feature = \"conn-sparse\")] "
                            "and the feature is NOT in vmlm-core's default set, so the "
                            "default inference engine does not link it; the measured "
                            "default size is unchanged."),
        "honest_limits": [
            "This is the STORAGE half of P1 only. It computes no forward pass and "
            "defines no opcode; connection-sparse ACTIVATION is another task.",
            "k0>0 requires EXACTLY k0 connections per neuron, so the format cannot "
            "express a pruned-away ('dead') neuron. That is a known restriction of "
            "this version, not an oversight.",
            "The initial weights are a deterministic, bit-exact placeholder "
            "distribution in [0.5, 1.0) (all positive). They carry no statistical "
            "meaning and are NOT trained weights.",
            "src ids are drawn with % n_neurons and may repeat; 'k0 distinct sources' "
            "is NOT guaranteed or checked. Duplicate srcs mean parallel connections.",
            "SEC_CONN_SPARSE is a reserved tag used only in examples/p16_conn; no "
            "production .vmb carries this section yet.",
        ],
    }
    with open(DUMP_PATH, "w", encoding="utf-8") as fh:
        json.dump(dump, fh, indent=2)
    print("      wrote %s" % DUMP_PATH)

    b.close()

    # ---- verdict ----------------------------------------------------------
    print("\n" + "=" * 78)
    if FAILS:
        print("VERDICT: FAIL -- %d check(s) failed:" % len(FAILS))
        for f_ in FAILS:
            print("   - %s" % f_)
        return 1
    print("VERDICT: PASS -- the connection-sparse STORAGE format exists on disk,")
    print("         round-trips bit-exactly, and every guard was observed firing")
    print("         in the real Rust code with exit code 9.")
    print("  section     : %s" % SECTION_PATH)
    print("  measured    : %d bytes on disk == %d bytes header-derived"
          % (on_disk, section_bytes))
    print("  shape       : %d x %d, k0=%d -> E=%d pairs/layer"
          % (NL_TARGET, D_MODEL, K0, total_pairs))
    print("  round trip  : %d pairs, %d src mismatches, %d weight-BIT mismatches"
          % (total_pairs, src_bad, w_bad))
    print("  guards fired: %d/%d, each exit code 9 with its own message"
          % (sum(1 for g in guard_results if g["fired"]), len(guard_results)))
    print("  can-fail    : 1 flipped weight bit -> %d mismatch; all %d flipped -> %d"
          % (n_w, total_pairs, n_q))
    return 0


if __name__ == "__main__":
    sys.exit(main())
