#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
META-CHECK: is the Rust cross-check able to FAIL?

The strongest claim in tools/p15_s5_section.py is:
    "the Python emitter's bytes are byte-identical to the shipped Rust
     neuron_kv::write_section()"
If that cross-check could not fail, it would be a rubber stamp, and the project
has been burned by exactly that (docs/P14_verification.md section 0.1: a probe
whose patch was EMPTY reported "0 B" for a change that cost 3584 B).

So this script builds a COPY of the workspace's core crate in TEMP, injects the
historical S5-A bug into the COPY's write_section (entry = i*stride*head_dim),
and re-runs the SAME cross-check procedure against the patched crate. If the
cross-check is real, it must now report MISMATCH.

The real <repo>/core is never touched. That matters: it is outside this
task's write scope and another agent may hold it.

Run: python tools/_p15_crosscheck_meta.py
"""
import os
import shutil
import struct
import subprocess
import sys
import tempfile

ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, os.path.join(ROOT, "tools"))
import p15_s5_section as T  # noqa: E402  (reuse the exact emitter under test)

fails = []


def ck(name, ok, detail=""):
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name,
                           (" -- " + detail) if detail else ""))
    if not ok:
        fails.append(name)


def run_xcheck(core_path, label):
    """Replicates section 6 of the tool, pointed at an arbitrary core path."""
    tmp = tempfile.mkdtemp(prefix="p15_meta_")
    try:
        os.makedirs(os.path.join(tmp, "src"))
        with open(os.path.join(tmp, "Cargo.toml"), "w", encoding="utf-8") as f:
            f.write('[package]\nname = "p15_meta"\nversion = "0.0.0"\n'
                    'edition = "2021"\n\n[dependencies]\n'
                    'vmlm-core = { path = "%s", features = ["neuron-kv"] }\n\n'
                    "[workspace]\n" % core_path.replace("\\", "/"))
        with open(os.path.join(tmp, "src", "main.rs"), "w", encoding="utf-8") as f:
            f.write(T.RUST_HARNESS)
        p = subprocess.run(["cargo", "run", "--release", "--quiet"], cwd=tmp,
                           capture_output=True, text=True, errors="replace",
                           timeout=900)
        if p.returncode != 0:
            return None, "cargo rc=%d: %s" % (
                p.returncode, (p.stderr or "").strip().splitlines()[-3:])
        for ln in (p.stdout or "").splitlines():
            if ln.startswith("HEX="):
                return bytes.fromhex(ln.split("=", 1)[1]), "ok"
        return None, "no HEX"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# The emitter output for the real 24-layer geometry.
good, n_neurons = T.build_section(24, 2, 128, 1024, 1024)
print("=== reference: emitter output, %d bytes, %d neurons ===" % (len(good), n_neurons))
print("      sha256(first 32 hex) = %s" % __import__("hashlib").sha256(good).hexdigest()[:32])

print("\n=== 1. cross-check vs the REAL core (must MATCH) ===")
real_hex, note = run_xcheck(os.path.join(ROOT, "core"), "real")
if real_hex is None:
    print("      SKIPPED: %s" % note)
    ck("cross-check ran against real core", False, note)
else:
    same = real_hex == good
    print("      rust bytes=%d  python bytes=%d  identical=%s"
          % (len(real_hex), len(good), same))
    ck("real core write_section == emitter (this is the positive control)",
       same, "identical" if same else "DIFFERS")

print("\n=== 2. negative control: inject the S5-A bug into a COPY of core ===")
tmpws = tempfile.mkdtemp(prefix="p15_metaws_")
try:
    core_copy = os.path.join(tmpws, "core")
    shutil.copytree(os.path.join(ROOT, "core"), core_copy)
    nk = os.path.join(core_copy, "src", "neuron_kv.rs")
    src = open(nk, "r", encoding="utf-8-sig").read()
    needle = "put(NKV_HDR + i * NKV_ENTRY, (i as u64 * layout.stride as u64) as u32);"
    assert needle in src, "write_section line not found -- cannot build the control"
    patched = src.replace(
        needle,
        "put(NKV_HDR + i * NKV_ENTRY, "
        "(i as u64 * layout.stride as u64 * layout.head_dim as u64) as u32);")
    open(nk, "w", encoding="utf-8").write(patched)
    print("      copied core -> %s" % core_copy)
    print("      injected: entry[i] = i*stride*head_dim  (the historical 128x bug)")
    print("      the real <repo>/core is NOT touched")

    bug_hex, note2 = run_xcheck(core_copy, "bugged")
    if bug_hex is None:
        ck("bugged-core cross-check ran", False, note2)
    else:
        differs = bug_hex != good
        first = next((i for i in range(min(len(bug_hex), len(good)))
                      if bug_hex[i] != good[i]), None)
        ndiff_bytes = sum(1 for a, b in zip(bug_hex, good) if a != b)
        print("      bugged rust bytes=%d  identical to emitter=%s"
              % (len(bug_hex), not differs))
        print("      first differing byte offset = %s  (%d/%d bytes differ)"
              % (first, ndiff_bytes, len(good)))
        ck("META: the cross-check DOES report MISMATCH on the injected bug",
           differs, "first diff at offset %s" % first)
        # And show the injected value at entry 1 is exactly the 128x number.
        v1 = struct.unpack_from("<I", bug_hex, 20 + 4)[0]
        ck("injected entry[1] is 131072 = head_dim*stride (the 128x number)",
           v1 == 128 * 1024, "entry[1]=%d" % v1)
        ck("emitter's own entry[1] is 1024", 
           struct.unpack_from("<I", good, 20 + 4)[0] == 1024, "entry[1]=1024")
finally:
    shutil.rmtree(tmpws, ignore_errors=True)
    print("      cleaned up %s" % tmpws)

print("\n" + "=" * 62)
if fails:
    print("META-CHECK: FAIL -- %d" % len(fails))
    for f in fails:
        print("   - %s" % f)
    sys.exit(1)
print("META-CHECK: PASS -- the Rust cross-check is falsifiable:")
print("  it MATCHES the real write_section and MISMATCHES a deliberately")
print("  broken one. It is a measurement, not a rubber stamp.")
