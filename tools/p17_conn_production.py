# -*- coding: utf-8 -*-
"""
P17 prerequisite: PROVE that SEC_CONN_SPARSE is now WIRED IN, not merely defined.

WHY THIS FILE EXISTS (the gap it closes)
----------------------------------------
P16 built the conn-sparse STORAGE format (core/src/conn_sparse.rs) and shipped
exactly ONE artifact carrying a conn section:
`examples/p16_conn/conn_sparse.vmb` -- hand-built by tools/p16_conn_format.py.
P16_report.md section 8 names the missing step verbatim:

    "生产 .vmb 携带 conn 段 ... 还缺一次接线"

and tools/p16_conn_format.py's own sidecar (honest_limits) admitted it:

    "SEC_CONN_SPARSE is a reserved tag used only in examples/p16_conn; no
     production .vmb carries this section yet."

The tag lived ONLY in core/src/conn_sparse.rs. core/src/format.rs -- the file
that owns the section tag table -- did not have it. A tag that exists in one
module's private constant is not a wiring: nothing outside that module can
legitimately write section tag 3 into an artifact.

WHAT CHANGED (Lead's edit -- verified here, not assumed)
--------------------------------------------------------
core/src/format.rs now declares `pub const SEC_CONN_SPARSE: u32 = 3;` next to
`SEC_NEURON_KV`. This tool MEASURES that the two declarations agree, that the
load path recognises tag 3, and that it HARD-FAILS on 2 (unassigned) and 4
(unknown) -- because "recognises the good value" is worthless unless the bad
values are rejected.

WHAT IS *STILL* NOT DONE (stated up front, so no one over-reads a PASS)
-----------------------------------------------------------------------
This tool wires the TAG. It does not make the runtime ENGINE auto-load a conn
section at startup: no caller inside runtime/ consumes SEC_CONN_SPARSE yet.
What is proven is that a production-shaped .vmb can carry tag 3 and that the
shipped loader accepts/rejects it correctly. Stated again in section 6 below
with the exact file/line evidence, and in the report's UNPROVEN list.

HOW IT MEASURES (and why not by pattern-matching the source)
------------------------------------------------------------
The decisive claims are made by compiling a throwaway harness against the REAL
vmlm-core with `--features conn-sparse`, in TEMP (never in <repo>, so a
stray workspace member cannot perturb a concurrent `cargo build`). The harness:
  * `format_tags`  -- prints format::SEC_NEURON_KV / format::SEC_CONN_SPARSE /
                      MAGIC, i.e. the SHIPPED values. A source grep would only
                      prove some text exists.
  * `prod_emit`    -- builds a REAL production .vmb whose descriptor really
                      carries tag 3, using the SHIPPED write_section().
  * `prod_load`    -- runs the SHIPPED load_find_section()/load_validate() on
                      the artifact and prints the parsed layout.
  * `prod_mutate`  -- rewrites ONLY the descriptor tag (nothing else) and hands
                      the result to the SHIPPED loader. exit(9) is the answer.

Every guard is one process: `exit(9)` calls ExitProcess, so the exit code IS
the measurement and nothing in the harness can trap or reinterpret it.

ASCII-only: this project has been bitten twice by console codepage corruption
of non-ASCII in .ps1/.py tooling. Chinese prose lives in docs/, not here.

Usage:
    python tools/p17_conn_production.py
Exit code 0 = every check passed. Non-zero = at least one FAIL.
"""

import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE = os.path.join(ROOT, "core")
FORMAT_RS = os.path.join(CORE, "src", "format.rs")
CONN_RS = os.path.join(CORE, "src", "conn_sparse.rs")
OUT_DIR = os.path.join(ROOT, "examples", "p17_conn")
REPORT_DIR = os.path.join(ROOT, "docs")

# Shape of the production artifact. Deliberately NOT the P16 example shape
# (1536 x k0=20): a DIFFERENT shape means a PASS here cannot be an artefact of
# the P16 example's particular numbers. 96 neurons x 7 connections is also
# small enough that the whole section is byte-inspectable.
PROD_N = 96
PROD_K0 = 7
PROD_LAYER = 0

FAILS = []
CHECKS = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print("  [%s] %s%s" % (tag, name, (" -- " + detail) if detail else ""))
    CHECKS.append({"name": name, "pass": bool(ok), "detail": detail})
    if not ok:
        FAILS.append(name)
    return bool(ok)


def parse_kv(out):
    """Parse the harness KEY=VALUE lines. Unknown keys are kept, not dropped."""
    kv = {}
    for line in out.splitlines():
        if "=" in line and not line.startswith(" "):
            k, _, v = line.partition("=")
            if re.fullmatch(r"[A-Z0-9_]+", k):
                kv[k] = v.strip()
    return kv


def drift_guard_agrees(fmt_value, module_value):
    """The DRIFT GUARD predicate, in ONE place.

    Both the real run (section 1) and the can-fail control (section 5) call
    THIS function. That is deliberate: if the control exercised a private copy
    of the logic, demonstrating the control fails would prove nothing about the
    real check. Accepts ints or numeric strings, because the control reads its
    values out of harness stdout.
    """
    try:
        return int(fmt_value) == int(module_value)
    except (TypeError, ValueError):
        # An unparseable value is NOT agreement. Silently returning True here
        # would be the classic "predicate that cannot fail".
        return False


# ---------------------------------------------------------------------------
# The harness. It only ever CALLS shipped functions; it never re-implements a
# guard. That property is what makes "the real Rust code exited 9" a
# measurement rather than a restatement of the Python's own logic.
# ---------------------------------------------------------------------------
RUST_HARNESS = r'''
use std::env;
use std::fs;

use vmlm_core::conn_sparse::*;

fn main() {
    let args: Vec<String> = env::args().collect();
    let mode = args[1].as_str();
    match mode {
        // ---- the SHIPPED tag table (measured, not grepped) ---------------
        "format_tags" => {
            println!("FORMAT_SEC_NEURON_KV={}", vmlm_core::format::SEC_NEURON_KV);
            println!("FORMAT_SEC_CONN_SPARSE={}", vmlm_core::format::SEC_CONN_SPARSE);
            println!("MODULE_SEC_CONN_SPARSE={}", SEC_CONN_SPARSE);
            println!("FORMAT_MAGIC={}", vmlm_core::format::MAGIC);
            println!("DESCR_OFF={}", DESCR_OFF);
            println!("VMB_HDR={}", VMB_HDR);
            println!("VMB_HDR_WITH_DESCR={}", VMB_HDR_WITH_DESCR);
            println!("CS_HDR={}", CS_HDR);
            println!("CS_MAGIC=0x{:08x}", CS_MAGIC);
        }
        // ---- emit a REAL production .vmb around a real section -----------
        //
        // The .vmb layout is the one P16 established and S5 measured:
        //   [0..32)  .vmb header: magic,ver,pad | text_off,text_len
        //                         | const_off,const_len | meta_off,meta_len
        //   [32..44) section descriptor: (tag, off, len)
        //   [44..)   the section itself
        // text_off = 44 is what tells the loader "a descriptor is present".
        // NOTE (learned the hard way in S5, recorded in p16_conn_format.py):
        // "has a section" must NOT be tested by tag != 0, because with no
        // descriptor [32..44) holds the START OF TEXT.
        "prod_emit" => {
            let n: u32 = args[2].parse().unwrap();
            let k0: u32 = args[3].parse().unwrap();
            let layer: u32 = args[4].parse().unwrap();
            let tag: u32 = args[5].parse().unwrap();
            let out = &args[6];

            let layout = ConnSparseLayout::tight(n, k0, layer);
            let sec_bytes = layout.section_bytes();
            let mut sec = vec![0u8; sec_bytes];
            let written = unsafe { write_section(&layout, sec.as_mut_ptr()) };
            if written != sec_bytes {
                eprintln!("write_section wrote {} but layout says {}",
                          written, sec_bytes);
                std::process::exit(7);
            }

            let mut vmb = vec![0u8; VMB_HDR_WITH_DESCR];
            unsafe {
                put_u32(vmb.as_mut_ptr(), 0, vmlm_core::format::MAGIC);
                put_u32(vmb.as_mut_ptr(), 4, (vmlm_core::format::VERSION as u32) | (0u32 << 16));
                put_u32(vmb.as_mut_ptr(), 8, VMB_HDR_WITH_DESCR as u32); // text_off
                put_u32(vmb.as_mut_ptr(), 12, 0);                        // text_len
                put_u32(vmb.as_mut_ptr(), 16, VMB_HDR_WITH_DESCR as u32);
                put_u32(vmb.as_mut_ptr(), 20, 0);
                put_u32(vmb.as_mut_ptr(), 24, VMB_HDR_WITH_DESCR as u32);
                put_u32(vmb.as_mut_ptr(), 28, 0);
                put_u32(vmb.as_mut_ptr(), DESCR_OFF, tag);
                put_u32(vmb.as_mut_ptr(), DESCR_OFF + 4, VMB_HDR_WITH_DESCR as u32);
                put_u32(vmb.as_mut_ptr(), DESCR_OFF + 8, sec_bytes as u32);
            }
            vmb.extend_from_slice(&sec);
            fs::write(out, &vmb).expect("write vmb");

            let mut h: u64 = 1469598103934665603;
            for b in vmb.iter() { h ^= *b as u64; h = h.wrapping_mul(1099511628211); }
            println!("VMB_BYTES={}", vmb.len());
            println!("SECTION_BYTES={}", sec_bytes);
            println!("DESCRIPTOR_TAG={}", tag);
            println!("DESCRIPTOR_OFF={}", VMB_HDR_WITH_DESCR);
            println!("DESCRIPTOR_LEN={}", sec_bytes);
            println!("VMB_FNV1A64={:016x}", h);
            println!("N_NEURONS={}", layout.n_neurons);
            println!("K0={}", layout.k0);
            println!("LAYER={}", layout.layer);
            println!("TOTAL_PAIRS={}", layout.total_pairs);
        }
        // ---- load it with the SHIPPED loader ----------------------------
        "prod_load" => {
            let path = &args[2];
            let d = fs::read(path).expect("read");
            let (sec, layout) = unsafe { load_validate(d.as_ptr(), d.len()) };
            println!("LOAD_OK=1");
            println!("SEC_OFF_FROM_BASE={}", (sec as usize) - (d.as_ptr() as usize));
            println!("N_NEURONS={}", layout.n_neurons);
            println!("K0={}", layout.k0);
            println!("LAYER={}", layout.layer);
            println!("TOTAL_PAIRS={}", layout.total_pairs);
            println!("SECTION_BYTES={}", layout.section_bytes());
            println!("OFFSETS_OFF={}", layout.offsets_off());
            println!("SRC_OFF={}", layout.src_off());
            println!("WEIGHTS_OFF={}", layout.weights_off());
            // Round-trip through the SHIPPED accessors: the section found via
            // the DESCRIPTOR path must read back exactly like the writer meant.
            let (sb, wb, fs_, fw) = unsafe { crosscheck_readback(sec, &layout) };
            println!("SRC_MISMATCH={}", sb);
            println!("WEIGHT_BIT_MISMATCH={}", wb);
            match fs_ { Some((i, j, g, w)) => println!("FIRST_SRC_BAD={},{},{},{}", i, j, g, w),
                        None => println!("FIRST_SRC_BAD=none") }
            match fw { Some((i, j, g, w)) => println!("FIRST_W_BAD={},{},{:08x},{:08x}", i, j, g, w),
                        None => println!("FIRST_W_BAD=none") }
            // Independent re-read of the descriptor bytes straight off disk, so
            // "the loader saw tag 3" is not the only witness to what is stored.
            println!("DISK_TAG={}", unsafe { get_u32(d.as_ptr(), DESCR_OFF) });
            println!("DISK_OFF={}", unsafe { get_u32(d.as_ptr(), DESCR_OFF + 4) });
            println!("DISK_LEN={}", unsafe { get_u32(d.as_ptr(), DESCR_OFF + 8) });
            println!("DISK_TEXT_OFF={}", unsafe { get_u32(d.as_ptr(), 8) });
        }
        // ---- mutate ONLY the descriptor; the SHIPPED loader must exit 9 --
        //
        // Rewriting the on-disk descriptor (not an in-memory buffer) matters:
        // it proves the value the loader reacts to is the one a PRODUCER would
        // have written, i.e. exactly what a mis-wired emitter would emit.
        "prod_mutate" => {
            let path = &args[2];
            let tag: u32 = args[3].parse().unwrap();
            let mut d = fs::read(path).expect("read");
            unsafe { put_u32(d.as_mut_ptr(), DESCR_OFF, tag) };
            // hand it straight to the shipped loader -- exit(9) is the answer
            let (_sec, layout) = unsafe { load_validate(d.as_ptr(), d.len()) };
            println!("UNEXPECTEDLY_LOADED=1");
            println!("N_NEURONS={}", layout.n_neurons);
        }
        // ---- can-fail control on the TAG TABLE itself --------------------
        //
        // The interesting failure mode for a wiring check is "the check cannot
        // fail". So we build a MUTANT vmlm-core whose format.rs maps the tag to
        // a DIFFERENT number, and show that our own equality assertion would
        // have caught it. Without this, "format.rs has SEC_CONN_SPARSE = 3" is
        // unfalsifiable from this tool's own output.
        "tag_probe" => {
            println!("PROBE_FORMAT_SEC_CONN_SPARSE={}", vmlm_core::format::SEC_CONN_SPARSE);
            println!("PROBE_MODULE_SEC_CONN_SPARSE={}", SEC_CONN_SPARSE);
        }
        other => { eprintln!("unknown mode {}", other); std::process::exit(2); }
    }
}
'''


class Harness:
    """Throwaway out-of-tree crate linking the REAL vmlm-core."""

    def __init__(self):
        self.dir = None
        self.bin = None
        self.error = None

    def build(self, core_dir=None, tag_override=None):
        cargo = shutil.which("cargo")
        if cargo is None:
            self.error = "cargo not on PATH"
            return False
        if core_dir is None:
            core_dir = CORE
        elif tag_override is not None:
            # Private copy of core/ in TEMP, with format.rs's tag rewritten.
            # The literal must occur EXACTLY once: a mutation that silently did
            # not apply would look like "the check is not load-bearing", which
            # is the failure mode this whole control exists to rule out.
            self.dir = tempfile.mkdtemp(prefix="p17_conn_mutant_")
            dst = os.path.join(self.dir, "core")
            shutil.copytree(
                CORE, dst,
                ignore=shutil.ignore_patterns("target", "*.tmp", "*.tmpdir",
                                              ".*.tmpdir", ".*.tmp"))
            with open(FORMAT_RS, "r", encoding="utf-8-sig") as f:
                text = f.read()
            literal = "pub const SEC_CONN_SPARSE: u32 = 3;"
            n = text.count(literal)
            if n != 1:
                self.error = ("tag literal occurs %d times in format.rs (must be "
                              "exactly 1) -- refusing to run a mutation that may "
                              "not have applied" % n)
                return False
            text = text.replace(
                literal,
                "pub const SEC_CONN_SPARSE: u32 = %d; // [MUTANT]" % tag_override)
            with open(os.path.join(dst, "src", "format.rs"), "w",
                      encoding="utf-8") as f:
                f.write(text)
            core_dir = dst

        if self.dir is None:
            self.dir = tempfile.mkdtemp(prefix="p17_conn_xcheck_")
        work = os.path.join(self.dir, "harness")
        src_dir = os.path.join(work, "src")
        os.makedirs(src_dir, exist_ok=True)
        core_path = core_dir.replace("\\", "/")
        with open(os.path.join(work, "Cargo.toml"), "w", encoding="utf-8") as f:
            f.write(
                "[package]\n"
                'name = "p17_conn_xcheck"\n'
                'version = "0.0.0"\n'
                'edition = "2021"\n\n'
                "[dependencies]\n"
                # default-features = false for the same reason runtime/Cargo.toml
                # does it (vmlm-core defaults to ["train"]). `conn-sparse` being
                # the ONLY feature is itself a check that the module is
                # self-sufficient and does not secretly need neuron-kv.
                'vmlm-core = { path = "%s", default-features = false, '
                'features = ["conn-sparse"] }\n\n'
                "[workspace]\n" % core_path)
        with open(os.path.join(src_dir, "main.rs"), "w", encoding="utf-8") as f:
            f.write(RUST_HARNESS)
        p = subprocess.run([cargo, "build", "--release", "--quiet"],
                           cwd=work, capture_output=True, text=True,
                           errors="replace", timeout=900)
        if p.returncode != 0:
            tail = (p.stderr or "").strip().splitlines()[-12:]
            self.error = "cargo build rc=%d: %s" % (p.returncode, " | ".join(tail))
            return False
        self.bin = os.path.join(work, "target", "release", "p17_conn_xcheck.exe")
        if not os.path.isfile(self.bin):
            self.error = "harness binary not produced at %s" % self.bin
            return False
        return True

    def run(self, *args):
        p = subprocess.run([self.bin] + [str(a) for a in args],
                           capture_output=True, text=True, errors="replace",
                           timeout=600)
        return p.returncode, p.stdout or "", p.stderr or ""

    def close(self):
        if self.dir:
            shutil.rmtree(self.dir, ignore_errors=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 78)
    print("P17 PREREQUISITE -- is SEC_CONN_SPARSE actually WIRED IN?")
    print("=" * 78)
    print("  root   : %s" % ROOT)
    print("  format : %s" % FORMAT_RS)
    print("  outcome dir: %s" % OUT_DIR)

    b = Harness()
    if not b.build():
        print("\nFATAL: could not build the harness: %s" % b.error)
        return 2

    # =====================================================================
    # 1. The tag table, read out of the SHIPPED crate.
    # =====================================================================
    print("\n=== 1. the tag table as the SHIPPED crate declares it ===")
    rc, out, err = b.run("format_tags")
    print("      format_tags rc=%d" % rc)
    for line in out.strip().splitlines():
        print("      %s" % line)
    if rc != 0:
        print("      stderr: %s" % err.strip())
        return 2
    kv = parse_kv(out)
    fmt_nkv = int(kv["FORMAT_SEC_NEURON_KV"])
    fmt_cs = int(kv["FORMAT_SEC_CONN_SPARSE"])
    mod_cs = int(kv["MODULE_SEC_CONN_SPARSE"])
    fmt_magic = int(kv["FORMAT_MAGIC"])

    check("format::SEC_NEURON_KV == 1 (S5 section tag unchanged)",
          fmt_nkv == 1, "measured %d" % fmt_nkv)
    check("format::SEC_CONN_SPARSE == 3 (the P17 wiring)",
          fmt_cs == 3, "measured %d" % fmt_cs)
    check("format::MAGIC == 0x4D4C4D56 ('VMLM') -- this crate IS vmlm-core",
          fmt_magic == 0x4D4C4D56, "measured 0x%08x" % fmt_magic)
    check("conn_sparse::SEC_CONN_SPARSE == format::SEC_CONN_SPARSE "
          "(the two declarations AGREE -- the P16 gap was that only one existed)",
          mod_cs == fmt_cs, "module=%d format=%d" % (mod_cs, fmt_cs))

    # ---- DRIFT GUARD (Lead's request; this is a TIME BOMB, not a style nit) --
    #
    # MEASURED FACT: conn_sparse.rs's loader guard at line 866 compares against
    # its OWN module constant -- `if tag != SEC_CONN_SPARSE` -- and NOT against
    # format::SEC_CONN_SPARSE. So the two constants are today equal by hand, not
    # by construction. If someone later edits ONLY format.rs (which is the file
    # the project calls "the single source of truth for the tag table"), the
    # loader keeps accepting 3 while the tag TABLE claims a different value --
    # and every artifact emits a tag the loader then rejects, with no test
    # failing at the edit site.
    #
    # The check two lines above already compares them, so why restate it here?
    # Because "the two agree" and "a future edit is caught" are different
    # claims, and only the second one is the guard. What makes THIS a guard is
    # the can-fail control in section 5: a mutant whose format.rs says 9 makes
    # this exact assertion fail. That control is what converts this from a
    # coincidence into a load-bearing check.
    print("      DRIFT GUARD: format.rs = %d, conn_sparse.rs = %d" % (fmt_cs, mod_cs))
    check("DRIFT GUARD: the tag table's value and the loader's comparison "
          "constant are identical -- asserted so that a future edit to "
          "format.rs ALONE cannot silently desynchronise them",
          drift_guard_agrees(fmt_cs, mod_cs),
          "format=%d module=%d (delta=%d)"
          % (fmt_cs, mod_cs, fmt_cs - mod_cs))
    # The guard is only meaningful if it would have fired on a divergence. That
    # is measured in section 5 with the mutation; record the expectation here so
    # the two halves are explicitly linked rather than merely adjacent.
    print("      -> falsifiability comes from section 5: the mutant crate "
          "(format.rs=9, module=3) makes this assertion FAIL.")
    # Unit-check the predicate on hand-picked inputs BEFORE trusting it on the
    # real values. A predicate that returned True unconditionally would make
    # every DRIFT GUARD check above vacuous, and that is exactly the
    # "cannot-fail judgement" this project hard-fails on.
    _self = [
        (3, 3, True, "identical values agree"),
        (3, 9, False, "divergent values must NOT agree"),
        (9, 3, False, "divergence is symmetric"),
        ("3", "3", True, "string values parse"),
        ("", "3", False, "unparseable is NOT agreement"),
        (None, 3, False, "None is NOT agreement"),
    ]
    bad = [(a, b, w, got) for (a, b, w, d) in _self
           for got in [drift_guard_agrees(a, b)] if got is not w]
    check("DRIFT GUARD predicate self-test: 6 hand-picked cases incl. the "
          "unparseable-is-not-agreement case",
          not bad, "failures=%s" % (bad if bad else "none"))
    check("descriptor contract: DESCR_OFF == VMB_HDR == 32, "
          "VMB_HDR_WITH_DESCR == 44",
          int(kv["DESCR_OFF"]) == 32 and int(kv["VMB_HDR"]) == 32
          and int(kv["VMB_HDR_WITH_DESCR"]) == 44,
          "DESCR_OFF=%s VMB_HDR=%s WITH_DESCR=%s"
          % (kv["DESCR_OFF"], kv["VMB_HDR"], kv["VMB_HDR_WITH_DESCR"]))

    # tag 0 and 2 are the values that MUST stay free. We cannot ask the crate
    # for a constant that deliberately does not exist, so this is stated as a
    # property of the numbers we DID measure: the table is 0=reserved,
    # 1=neuron_kv, 2=unassigned, 3=conn_sparse. Rejecting 2 and 4 is measured
    # in section 3; here we only assert no *collision* with the reserved value.
    check("no tag collides with 0 (reserved = 'artifact has no such section')",
          fmt_nkv != 0 and fmt_cs != 0, "neuron_kv=%d conn_sparse=%d"
          % (fmt_nkv, fmt_cs))
    check("tag 2 is NOT taken by conn_sparse (it stays unassigned)",
          fmt_cs != 2, "conn_sparse=%d" % fmt_cs)

    # =====================================================================
    # 2. Emit a REAL production .vmb carrying tag 3.
    # =====================================================================
    print("\n=== 2. emit a PRODUCTION .vmb whose descriptor carries tag 3 ===")
    print("      shape n=%d k0=%d layer=%d (deliberately NOT the P16 example's"
          % (PROD_N, PROD_K0, PROD_LAYER))
    print("      1536 x k0=20, so a PASS cannot be an artefact of that shape)")
    prod = os.path.join(OUT_DIR, "prod_conn.vmb")
    rc, out, err = b.run("prod_emit", PROD_N, PROD_K0, PROD_LAYER, fmt_cs, prod)
    print("      prod_emit rc=%d" % rc)
    for line in out.strip().splitlines():
        print("      %s" % line)
    if rc != 0:
        print("      stderr: %s" % err.strip())
        return 2
    ekv = parse_kv(out)
    vmb_bytes = int(ekv["VMB_BYTES"])
    sec_bytes = int(ekv["SECTION_BYTES"])
    on_disk = os.path.getsize(prod)
    print("      on disk: %s (%d bytes)" % (prod, on_disk))
    check("artifact size == 44 B header+descriptor + section_bytes",
          on_disk == 44 + sec_bytes == vmb_bytes,
          "on_disk=%d 44+sec=%d reported=%d" % (on_disk, 44 + sec_bytes, vmb_bytes))

    with open(prod, "rb") as fh:
        raw = fh.read()
    disk_tag, disk_off, disk_len = struct.unpack_from("<III", raw, 32)
    disk_text = struct.unpack_from("<I", raw, 8)[0]
    disk_magic = struct.unpack_from("<I", raw, 0)[0]
    check("descriptor on disk really says tag=3 (read by an INDEPENDENT "
          "Python struct.unpack, not by the Rust loader)",
          disk_tag == 3, "bytes[32..36] = %d" % disk_tag)
    check("descriptor off == 44 and len == section_bytes",
          disk_off == 44 and disk_len == sec_bytes,
          "off=%d len=%d section_bytes=%d" % (disk_off, disk_len, sec_bytes))
    check("text_off == 44 -- the ONLY correct 'a descriptor is present' "
          "predicate (tag==0 misjudges the no-section case)",
          disk_text == 44, "text_off=%d" % disk_text)
    check("artifact magic on disk == 0x4D4C4D56",
          disk_magic == 0x4D4C4D56, "0x%08x" % disk_magic)
    # Independent section-magic witness: the writer must have produced 'CSR1'.
    sec_stored = struct.unpack_from("<I", raw, disk_off)[0]
    check("section magic at off=44 is 0x31525343 ('CSR1')",
          sec_stored == 0x31525343, "0x%08x" % sec_stored)

    # =====================================================================
    # 3. The SHIPPED LOADER must accept it.
    # =====================================================================
    print("\n=== 3. the SHIPPED loader (load_find_section/load_validate) "
          "accepts it ===")
    rc, out, err = b.run("prod_load", prod)
    print("      prod_load rc=%d" % rc)
    for line in out.strip().splitlines():
        print("      %s" % line)
    if rc != 0:
        print("      stderr: %s" % err.strip())
    lkv = parse_kv(out)
    check("load_validate() exited 0 on the production artifact", rc == 0,
          "rc=%d" % rc)
    check("loader reports N_NEURONS=%d" % PROD_N,
          lkv.get("N_NEURONS") == str(PROD_N), "got %s" % lkv.get("N_NEURONS"))
    check("loader reports K0=%d" % PROD_K0,
          lkv.get("K0") == str(PROD_K0), "got %s" % lkv.get("K0"))
    check("loader reports TOTAL_PAIRS == n*k0 == %d"
          % (PROD_N * PROD_K0),
          lkv.get("TOTAL_PAIRS") == str(PROD_N * PROD_K0),
          "got %s" % lkv.get("TOTAL_PAIRS"))
    check("loader resolved the section to offset 44 in the file",
          lkv.get("SEC_OFF_FROM_BASE") == "44",
          "got %s" % lkv.get("SEC_OFF_FROM_BASE"))
    check("loader AGREES with the descriptor bytes read independently "
          "(DISK_TAG=%s DISK_LEN=%s)"
          % (lkv.get("DISK_TAG"), lkv.get("DISK_LEN")),
          lkv.get("DISK_TAG") == str(fmt_cs)
          and lkv.get("DISK_LEN") == str(sec_bytes),
          "DISK_TAG=%s DISK_LEN=%s (want %d, %d)"
          % (lkv.get("DISK_TAG"), lkv.get("DISK_LEN"), fmt_cs, sec_bytes))
    check("round-trip through the descriptor path is bit-exact: "
          "SRC_MISMATCH=0 and WEIGHT_BIT_MISMATCH=0",
          lkv.get("SRC_MISMATCH") == "0" and lkv.get("WEIGHT_BIT_MISMATCH") == "0",
          "src=%s w=%s" % (lkv.get("SRC_MISMATCH"),
                           lkv.get("WEIGHT_BIT_MISMATCH")))

    # =====================================================================
    # 4. MUST-FAIL: tag 2 (unassigned) and tag 4 (unknown) => exit(9).
    # =====================================================================
    print("\n=== 4. MUST-FAIL controls: a mis-wired emitter MUST hard-fail ===")
    print("      Each case rewrites ONLY the 4 descriptor bytes at [32..36)")
    print("      and hands the result to the SHIPPED loader. exit(9) is the")
    print("      answer -- no guard is re-implemented on the Python side.")
    mutate_cases = [
        (2, "tag 2 = UNASSIGNED (the 'someone shipped a tag nobody claimed' case)",
         "unexpected section tag"),
        (4, "tag 4 = UNKNOWN/next (the 'off-by-one in an emitter' case)",
         "unexpected section tag"),
        (1, "tag 1 = SEC_NEURON_KV (a REAL tag, but the WRONG kind of section)",
         "unexpected section tag"),
        (0, "tag 0 = 'no section' while text_off still says 44 "
            "(contradictory artifact)",
         "has no conn_sparse section"),
    ]
    mutation_results = []
    for tag, label, want_msg in mutate_cases:
        p = os.path.join(OUT_DIR, "_control_tag%d.vmb" % tag)
        with open(p, "wb") as fh:
            fh.write(raw)
        with open(p, "r+b") as fh:
            fh.seek(32)
            fh.write(struct.pack("<I", tag))
        assert struct.unpack_from("<I", open(p, "rb").read(), 32)[0] == tag
        rc_m, out_m, err_m = b.run("prod_mutate", p, tag)
        msgs = [l for l in out_m.splitlines() if l.startswith("CS:")]
        detail = "exit=%d msg=%s" % (rc_m, msgs)
        print("      %s" % label)
        print("        -> exit=%d  %s" % (rc_m, msgs))
        mutation_results.append({"tag": tag, "label": label, "exit": rc_m,
                                 "messages": msgs})
        check("MUST-FAIL tag=%d exits 9 with the specific message %r"
              % (tag, want_msg),
              rc_m == 9 and want_msg in out_m, detail)
        check("MUST-FAIL tag=%d produced NO layout (loader did not silently "
              "accept it)" % tag, "UNEXPECTEDLY_LOADED" not in out_m,
              "rc=%d" % rc_m)

    # Determinism: the same illegal input must fail the same way twice.
    rc_a, out_a, _ = b.run("prod_mutate",
                           os.path.join(OUT_DIR, "_control_tag2.vmb"), 2)
    rc_b, out_b, _ = b.run("prod_mutate",
                           os.path.join(OUT_DIR, "_control_tag2.vmb"), 2)
    check("MUST-FAIL is deterministic: tag=2 -> exit %d then exit %d, "
          "identical messages" % (rc_a, rc_b),
          rc_a == rc_b == 9 and out_a == out_b, "rc_a=%d rc_b=%d" % (rc_a, rc_b))

    # =====================================================================
    # 5. CAN-FAIL control: would this tool actually notice a BROKEN wiring?
    # =====================================================================
    print("\n=== 5. CAN-FAIL control: mutate format.rs and show the check FIRES ===")
    print("      A wiring check you cannot demonstrate failing is UNPROVEN.")
    print("      So: copy core/ to TEMP, rewrite the ONE literal")
    print("      'pub const SEC_CONN_SPARSE: u32 = 3;' to a different value,")
    print("      rebuild, and show our own equality assertion catches it.")
    mut = Harness()
    try:
        ok = mut.build(core_dir="__mutant__", tag_override=9)
        if not ok:
            check("CAN-FAIL mutant harness builds (mutation actually applied)",
                  False, mut.error or "unknown build failure")
        else:
            rc_p, out_p, _ = mut.run("tag_probe")
            pkv = parse_kv(out_p)
            got_fmt = pkv.get("PROBE_FORMAT_SEC_CONN_SPARSE")
            got_mod = pkv.get("PROBE_MODULE_SEC_CONN_SPARSE")
            print("      mutant crate reports: format=%s module=%s"
                  % (got_fmt, got_mod))
            check("CAN-FAIL: with format.rs = 9 the two declarations MISMATCH "
                  "(so the real-tree PASS on this same assertion is load-bearing)",
                  got_fmt == "9" and got_mod == "3",
                  "format=%s module=%s" % (got_fmt, got_mod))
            # MEASURED FINDING, and it corrected this tool's own first version.
            #
            # The mutant crate's format::SEC_CONN_SPARSE is 9, but its
            # conn_sparse::SEC_CONN_SPARSE is still 3 -- because
            # conn_sparse.rs's loader guard compares against its OWN module
            # constant (`if tag != SEC_CONN_SPARSE`, line 866), NOT against
            # format::SEC_CONN_SPARSE. So a mutant-loader case would reject a
            # tag-9 artifact and say nothing interesting about the wiring.
            #
            # The first version of this control asserted "the mutant loader
            # accepts 9" and FAILED. That was the control doing its job: the
            # assertion encoded an assumption about the code that is FALSE.
            # The honest, load-bearing claims are the two below.
            mp = os.path.join(OUT_DIR, "_control_mutant_tag9.vmb")
            rc_e, out_e, _ = mut.run("prod_emit", 8, 2, 0, 9, mp)
            if rc_e == 0:
                with open(mp, "rb") as fh:
                    mraw = fh.read()
                mtag = struct.unpack_from("<I", mraw, 32)[0]
                check("CAN-FAIL: the mutant crate really EMITS tag=9 (so the "
                      "next two checks are not vacuous)",
                      mtag == 9, "descriptor tag on disk = %d" % mtag)
                rc_w, out_w, _ = b.run("prod_load", mp)
                msgs_w = [l for l in out_w.splitlines() if l.startswith("CS:")]
                print("      REAL loader on the tag=9 artifact -> exit=%d %s"
                      % (rc_w, msgs_w))
                check("CAN-FAIL: the REAL loader REJECTS a tag-9 artifact with "
                      "exit 9 'unexpected section tag'",
                      rc_w == 9 and "unexpected section tag" in out_w,
                      "exit=%d" % rc_w)
                # And the converse, which is the load-bearing half: the mutant
                # crate -- whose format.rs says 9 -- still self-consistently
                # round-trips its OWN tag, proving the emitter/loader pair is
                # driven by the same constant and not by environment accident.
                rc_ml, out_ml, _ = mut.run("prod_load", mp)
                print("      mutant loader on the tag=9 artifact -> exit=%d "
                      "(its own module constant is still 3, so 9 is foreign)"
                      % rc_ml)
                check("CAN-FAIL: the tag the loader accepts is the MODULE "
                      "constant (mutant format=9 therefore does NOT change "
                      "what the loader accepts) -- measured, not assumed",
                      rc_ml == 9, "mutant load exit=%d" % rc_ml)
                # ---- DRIFT GUARD falsifiability, measured through the SAME
                # predicate the real run uses. The guard is only a guard if it
                # goes red on divergence; this is that demonstration.
                check("CAN-FAIL (DRIFT GUARD): the mutant tree makes the "
                      "format-vs-module drift predicate return NOT-AGREE, i.e. "
                      "the guard in section 1 would have FAILED here",
                      not drift_guard_agrees(got_fmt, got_mod),
                      "format=%s module=%s -> agrees=%s"
                      % (got_fmt, got_mod,
                         drift_guard_agrees(got_fmt, got_mod)))
                # And the positive half, on the same predicate: it says AGREE
                # for the shipped tree. Without this, a predicate that is
                # hardwired to False would also satisfy the check above.
                check("CAN-FAIL (DRIFT GUARD): the same predicate says AGREE for "
                      "the shipped tree, so it is not hardwired to False",
                      drift_guard_agrees(fmt_cs, mod_cs),
                      "shipped format=%d module=%d" % (fmt_cs, mod_cs))
            else:
                check("CAN-FAIL: mutant emitter produced its artifact",
                      False, "emit rc=%d" % rc_e)
    finally:
        mut.close()

    # =====================================================================
    # 6. HONEST BOUNDARY: what is wired, and what is still not.
    # =====================================================================
    print("\n=== 6. honest boundary -- what this does NOT prove ===")
    print("      What must be measured is: does ANY file outside the two that")
    print("      DEFINE the conn tag actually USE it? Matching the bare names")
    print("      'load_find_section'/'load_validate' is NOT enough -- neuron_kv")
    print("      has same-named functions of its own (a first version of this")
    print("      tool counted those three files and would have reported")
    print("      'callers exist', which is exactly backwards). The qualified")
    print("      path `conn_sparse::<fn>` and the tag name are the real test.")
    users = []
    skip_rel = ("core/src/format.rs", "core/src/conn_sparse.rs")
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames
                       if d not in ("target", ".git", "node_modules")]
        for fn in filenames:
            if not fn.endswith(".rs"):
                continue
            fp = os.path.join(dirpath, fn)
            rel = os.path.relpath(fp, ROOT).replace("\\", "/")
            if rel in skip_rel:
                continue
            try:
                with open(fp, "r", encoding="utf-8-sig", errors="replace") as f:
                    txt = f.read()
            except OSError:
                continue
            hits = [l.strip() for l in txt.splitlines()
                    if "SEC_CONN_SPARSE" in l or "conn_sparse::load_" in l
                    or "conn_sparse::read_section" in l
                    or "conn_sparse::crosscheck" in l]
            if hits:
                users.append({"file": rel, "lines": hits[:4]})
    print("      Rust files that USE conn_sparse's tag/loader OUTSIDE the two")
    print("      that define it (qualified path or tag name): %d" % len(users))
    for u in users:
        print("        %s" % u["file"])
        for l in u["lines"]:
            print("            %s" % l)
    if not users:
        print("        (none)")
    # This is the measured LIMIT. It is reported as a limit, and the check
    # records the count so a future reader can re-measure rather than trust.
    check("MEASURED LIMIT: no caller outside core/src/{format,conn_sparse}.rs "
          "consumes SEC_CONN_SPARSE or conn_sparse's load path",
          len(users) == 0,
          "qualified users found = %d %s"
          % (len(users), [u["file"] for u in users]))
    print("      => the tag is WIRED (declared in the format table at the value")
    print("         the module already used; accepted for 3 and hard-failed for")
    print("         0/1/2/4 by the shipped loader); the ENGINE still does not")
    print("         auto-load a conn section at startup, and no runtime/ caller")
    print("         exists. That remaining step is UNPROVEN here, not claimed.")

    # =====================================================================
    # 7. sidecar
    # =====================================================================
    print("\n=== 7. sidecar dump ===")
    dump = {
        "task": "P17 prerequisite",
        "claim": ("SEC_CONN_SPARSE is wired into core/src/format.rs's tag table; "
                  "a production-shaped .vmb can carry it and the shipped loader "
                  "accepts tag 3 / hard-fails tag 0,1,2,4."),
        "tag_table": {
            "0": "reserved -- artifact carries no such section",
            "1": "SEC_NEURON_KV (S5)",
            "2": "UNASSIGNED -- must be rejected",
            "3": "SEC_CONN_SPARSE (P17 wiring)",
            "4": "UNKNOWN -- must be rejected",
        },
        "measured_shipped_values": {
            "format::SEC_NEURON_KV": fmt_nkv,
            "format::SEC_CONN_SPARSE": fmt_cs,
            "conn_sparse::SEC_CONN_SPARSE": mod_cs,
            "format::MAGIC": "0x%08x" % fmt_magic,
            "DESCR_OFF": int(kv["DESCR_OFF"]),
            "VMB_HDR": int(kv["VMB_HDR"]),
            "VMB_HDR_WITH_DESCR": int(kv["VMB_HDR_WITH_DESCR"]),
            "CS_MAGIC": kv["CS_MAGIC"],
        },
        "production_artifact": {
            "path": os.path.relpath(prod, ROOT).replace("\\", "/"),
            "shape": {"n_neurons": PROD_N, "k0": PROD_K0, "layer": PROD_LAYER},
            "vmb_bytes": on_disk,
            "descriptor": {"tag": disk_tag, "off": disk_off, "len": disk_len},
            "text_off": disk_text,
            "fnv1a64": ekv.get("VMB_FNV1A64"),
            "load_roundtrip": {"src_mismatch": lkv.get("SRC_MISMATCH"),
                               "weight_bit_mismatch": lkv.get("WEIGHT_BIT_MISMATCH")},
        },
        "must_fail_controls": mutation_results,
        "checks": CHECKS,
        "honest_limits": [
            "This wires the TAG only. No caller outside "
            "core/src/{format,conn_sparse}.rs consumes SEC_CONN_SPARSE or the "
            "conn_sparse load path (measured: %d qualified user files), so the "
            "engine does not auto-load a conn section at startup."
            % len(users),
            "MEASURED STRUCTURAL FACT: conn_sparse.rs's own loader guard "
            "compares against its MODULE constant (conn_sparse.rs line 866, "
            "`if tag != SEC_CONN_SPARSE`), not against format::SEC_CONN_SPARSE. "
            "The two now hold the same value, so nothing is currently wrong; "
            "but a future edit to format.rs alone would NOT change what the "
            "loader accepts. Reported because it is a real coupling gap, and "
            "this tool's first can-fail assertion wrongly assumed otherwise.",
            "The section content is P16's deterministic placeholder "
            "distribution, NOT trained weights.",
            "The artifact is a production-SHAPED .vmb (real header, real "
            "descriptor, real section) but it is not emitted by a model "
            "converter; no converter emits conn sections yet.",
        ],
    }
    dump_path = os.path.join(OUT_DIR, "prod_conn.dump.json")
    with open(dump_path, "w", encoding="utf-8") as fh:
        json.dump(dump, fh, indent=2)
    print("      wrote %s" % dump_path)

    b.close()

    # =====================================================================
    # verdict
    # =====================================================================
    print("\n" + "=" * 78)
    if FAILS:
        print("VERDICT: FAIL -- %d check(s) failed:" % len(FAILS))
        for f_ in FAILS:
            print("   - %s" % f_)
        return 1
    print("VERDICT: PASS -- SEC_CONN_SPARSE is wired into the format tag table.")
    print("  tag table  : 0 reserved, 1 neuron_kv, 2 unassigned, 3 conn_sparse")
    print("  agreement  : format.rs=%d == conn_sparse.rs=%d" % (fmt_cs, mod_cs))
    print("  production : %s (%d B = 44 B hdr+descr + %d B section)"
          % (os.path.relpath(prod, ROOT).replace("\\", "/"), on_disk, sec_bytes))
    print("  loader     : load_validate() rc=0, %d pairs, 0 src mismatch, "
          "0 weight-BIT mismatch" % (PROD_N * PROD_K0))
    print("  must-fail  : tags 0,1,2,4 all exit 9 with their own message")
    print("  can-fail   : mutated format.rs (tag=9) is detected by this tool")
    print("  LIMIT      : no engine caller yet -- see section 6")
    return 0


if __name__ == "__main__":
    sys.exit(main())
