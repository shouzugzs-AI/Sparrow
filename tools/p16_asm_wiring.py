#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P16-2 ASSEMBLER WIRING PROOF + NEGATIVE CONTROL.

    python tools/p16_asm_wiring.py --asm <vmlm-asm.exe>

THE TRAP THIS GUARDS AGAINST
    This project has been bitten by names that exist in core/src/format.rs but
    are wired NOWHERE. `OP_MAX` and `OP_TOPK_IDX` sat defined-but-unimplemented
    for many phases; the assembler PANICKED on them (they were listed in
    `UNIMPLEMENTED_OPS`), so the engine had no argmax and the O6 generation
    loop was unwritable -- while the white paper's opcode table listed both as
    available. The inverse trap is worse and quieter: a name wired in the
    ASSEMBLER but with no ENGINE arm produces a .vmb that loads fine and dies
    mid-run at exit(5) with a bare `unimpl op`.

    So "wired at both ends" needs to be shown, not asserted:

      POSITIVE: a program using TOPK_BY_STRENGTH and SPARSE_MATVEC assembles
                (exit 0) AND the emitted opcode bytes are 93 and 94 -- not just
                "the assembler accepted the text", which could happen with any
                mapping.
      NEGATIVE: a name that is STILL unimplemented still panics. Without this,
                a broken UNIMPLEMENTED_OPS check (e.g. one that never matches)
                would make the positive result meaningless -- it would pass for
                a name that is not wired either.

    The negative control deliberately uses a name from the CURRENT list rather
    than a hardcoded expectation, so the control tracks the list instead of
    going stale. It also asserts the list is non-empty, so the control cannot
    silently become a no-op if every opcode gets implemented some day.
"""
import argparse
import os
import re
import struct
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p16_ops_probe as P  # noqa: E402

WORK = os.path.join(ROOT, "examples", "p16_ops")


def unimplemented_names():
    """Read the CURRENT UNIMPLEMENTED_OPS list out of asm/src/main.rs.

    Read from source rather than duplicated here: a copy would drift, and a
    stale copy would make this negative control test a name that no longer
    exists in the list -- i.e. it would pass for the wrong reason.
    """
    src = open(os.path.join(ROOT, "asm", "src", "main.rs"), encoding="utf-8").read()
    m = re.search(r"const UNIMPLEMENTED_OPS[^=]*=\s*&\[(.*?)\n\];", src, re.S)
    if not m:
        raise SystemExit("could not locate UNIMPLEMENTED_OPS in asm/src/main.rs")
    names = re.findall(r'\(\s*"([A-Z_0-9]+)"', m.group(1))
    return names


def opcodes_of(vmb):
    """Return the list of opcode bytes in a .vmb's TEXT section."""
    raw = open(vmb, "rb").read()
    to = struct.unpack("<I", raw[8:12])[0]
    tl = struct.unpack("<I", raw[12:16])[0]
    return [raw[to + i * 16] for i in range(tl // 16)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asm", required=True)
    a = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)
    npass = nfail = 0

    def rep(ok, name, detail=""):
        nonlocal npass, nfail
        if ok:
            npass += 1
        else:
            nfail += 1
        print("  [%s] %-52s %s" % ("PASS" if ok else "FAIL", name, detail))
        return ok

    print("=" * 78)
    print("ASSEMBLER WIRING PROOF")
    print("=" * 78)

    # ---------- POSITIVE: both names assemble ----------------------------
    src = ("SHAPE y, 3, 1\nSHAPE off, 3, 1\nSHAPE srci, 2, 1\n"
           "SHAPE wv, 2, 1\nSHAPE xv, 3, 1\nSHAPE str, 2, 1\nSHAPE kk, 1, 1\n"
           "SHAPE sel, 2, 2\n"
           "    TOPK_BY_STRENGTH sel, str, kk\n"
           "    SPARSE_MATVEC y, off, srci, wv, xv\n"
           "    HALT\n")
    vml = os.path.join(WORK, "wiring.vml")
    vmb = os.path.join(WORK, "wiring.vmb")
    open(vml, "w", encoding="ascii").write(src)
    rc, out, err = P.run_asm(a.asm, vml, vmb, (), cwd=ROOT)
    rep(rc == 0, "program using BOTH new names assembles",
        "asm rc=%d" % rc)
    if rc != 0:
        print("    %s" % (out + err)[-300:])
        return 1

    ops = opcodes_of(vmb)
    rep(93 in ops, "TOPK_BY_STRENGTH emitted opcode 93",
        "opcodes=%s" % ops)
    rep(94 in ops, "SPARSE_MATVEC emitted opcode 94",
        "opcodes=%s" % ops)
    # And prove the numbers are the ones format.rs declares, by reading it.
    fmt = open(os.path.join(ROOT, "core", "src", "format.rs"), encoding="utf-8").read()
    m1 = re.search(r"pub const OP_TOPK_BY_STRENGTH: u8 = (\d+);", fmt)
    m2 = re.search(r"pub const OP_SPARSE_MATVEC: u8 = (\d+);", fmt)
    rep(m1 is not None and int(m1.group(1)) == 93,
        "format.rs declares OP_TOPK_BY_STRENGTH = 93",
        "found %s" % (m1.group(1) if m1 else "NOT FOUND"))
    rep(m2 is not None and int(m2.group(1)) == 94,
        "format.rs declares OP_SPARSE_MATVEC = 94",
        "found %s" % (m2.group(1) if m2 else "NOT FOUND"))

    # ---------- NEGATIVE CONTROL -----------------------------------------
    names = unimplemented_names()
    rep(len(names) > 0, "UNIMPLEMENTED_OPS is non-empty (control is live)",
        "%d names: %s" % (len(names), names[:4]))
    rep("TOPK_BY_STRENGTH" not in names,
        "TOPK_BY_STRENGTH is NOT on UNIMPLEMENTED_OPS", "")
    rep("SPARSE_MATVEC" not in names,
        "SPARSE_MATVEC is NOT on UNIMPLEMENTED_OPS", "")

    if names:
        bad = names[0]
        nsrc = "SHAPE z, 2, 1\nSHAPE a, 2, 1\n    %s z, a\n    HALT\n" % bad
        nvml = os.path.join(WORK, "negctl.vml")
        nvmb = os.path.join(WORK, "negctl.vmb")
        open(nvml, "w", encoding="ascii").write(nsrc)
        rc2, out2, err2 = P.run_asm(a.asm, nvml, nvmb, (), cwd=ROOT)
        msg = (out2 + err2)
        named = bad in msg
        rep(rc2 != 0, "still-unimplemented name %r STILL panics" % bad,
            "asm rc=%d" % rc2)
        rep(named, "the panic message NAMES %r" % bad,
            "" if named else "message did not mention the name")
        rep(not os.path.exists(nvmb),
            "no .vmb is produced for the unimplemented name", "")

    print("\n" + "=" * 78)
    print("assembler wiring: %d pass, %d fail" % (npass, nfail))
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
