#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P13: the load-time dtype audit must STILL reject illegal input.

WHY THIS MATTERS MORE THAN THE HAPPY PATH
    Task A added a named exception to `audit_const_dtypes` so OP_EMB may take a
    bf16 TABLE. The risk of any such exception is that it quietly turns a
    precise guard into a blanket hole. The project's rule is that illegal input
    fails loudly, so the exception must be shown to be exactly as wide as
    intended: EMB's table may be bf16; everything else still fails.

CASES
    1. bf16 const used by RMSNORM  -> must exit 9 (the guard P12 added)
    2. bf16 const used by COPY     -> must exit 9
    3. EMB with a bf16 table       -> must run (exit 0)   [task A]
    4. EMB with an f32 table       -> must run (exit 0)   [no regression]
    5. three cases above must name the opcode/slot in the message, since a
       bare "bf16 const" cost a bisect cycle once (P12's note).

Usage: python tools/p13_audit_check.py
"""
import os
import subprocess
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = str(Path(__file__).resolve().parents[1])
REL = os.path.join(ROOT, "target", "release")
ASM = os.path.join(REL, "vmlm-asm.exe")
ENG = os.path.join(REL, "vmlm-runtime.exe")
DIR = os.path.join(ROOT, "examples", "p13_audit")

CASES = [
    # (name, vml, expect_exit, expect_in_msg)
    #
    # The engine reports the opcode NUMBER and slot ("op 31 uses bf16 const at
    # slot 2"), not the opcode name -- 31 is OP_RMSNORM in format.rs and 2 is
    # OP_COPY. Asserting on the number keeps this check honest about what the
    # engine actually prints; the first version of this file asserted the NAME
    # and reported a false FAIL against correct engine behaviour.
    ("bf16 const + RMSNORM (must fail)",
     "SHAPE x, 1, 4\nSHAPE eps, 1, 1\nRMSNORM x, small_bf16, eps\nPRINT x\nHALT\n",
     9, b"op 31 uses bf16 const"),
    ("bf16 const + COPY (must fail)",
     "SHAPE x, 1, 4\nCOPY x, small_bf16\nPRINT x\nHALT\n",
     9, b"op 2 uses bf16 const"),
    ("bf16 table + EMB (must run)",
     "SHAPE x, 1, 4\nSHAPE t, 1, 1\nSHAPE tb, 4, 4\nEMB x, tb, t\nPRINT x\nHALT\n",
     0, None),
    ("f32 table + EMB (must run)",
     "SHAPE x, 1, 4\nSHAPE t, 1, 1\nSHAPE tb, 4, 4\nEMB x, tb_f32, t\nPRINT x\nHALT\n",
     0, None),
]


def main():
    import numpy as np
    os.makedirs(DIR, exist_ok=True)
    # bf16 constant for the guard cases, f32 + bf16 tables for the EMB cases
    small = np.arange(1, 5, dtype=np.float32)
    small.view(np.uint32).astype(np.uint64)
    u = small.view(np.uint32).astype(np.uint64)
    rnd = np.uint64(0x7FFF) + ((u >> 16) & 1)
    ((u + rnd) >> 16).astype("<u2").tofile(os.path.join(DIR, "small_bf16.bf16"))
    small.tofile(os.path.join(DIR, "small_f32.f32"))
    u2 = np.arange(16, dtype=np.float32)
    uu = u2.view(np.uint32).astype(np.uint64)
    rr = np.uint64(0x7FFF) + ((uu >> 16) & 1)
    ((uu + rr) >> 16).astype("<u2").tofile(os.path.join(DIR, "table_bf16.bf16"))
    u2.tofile(os.path.join(DIR, "table_f32.f32"))

    fails = 0
    for i, (name, src, want, needle) in enumerate(CASES):
        vml = "a%d.vml" % i
        vmb = "a%d.vmb" % i
        with open(os.path.join(DIR, vml), "w", encoding="ascii", newline="\n") as f:
            f.write(src)
        p = subprocess.run(
            [ASM, vml, vmb,
             "small_bf16:small_bf16.bf16:1:4:bf16",
             "tb:table_bf16.bf16:4:4:bf16",
             "tb_f32:table_f32.f32:4:4:f32"],
            cwd=DIR, capture_output=True)
        if p.returncode != 0:
            print("  %-34s ASSEMBLE FAILED" % name)
            fails += 1
            continue
        import shutil
        shutil.copyfile(os.path.join(DIR, vmb), os.path.join(REL, "model.vmb"))
        tb = os.path.join(REL, "tokens.bin")
        if os.path.exists(tb):
            os.remove(tb)
        q = subprocess.run([ENG], cwd=REL, capture_output=True)
        msg = q.stdout + q.stderr
        ok = q.returncode == want
        if ok and needle is not None:
            ok = needle in msg
        print("  %-34s exit=%d (want %d) %s" % (name, q.returncode, want,
                                                "PASS" if ok else "FAIL"))
        if msg.strip():
            print("      msg: %s" % msg.decode("utf-8", "replace").strip()[:90])
        if not ok:
            fails += 1

    print()
    print("VERDICT: %s" % ("PASS -- audit rejects illegal input, allows bf16 EMB"
                           if fails == 0 else "FAIL (%d)" % fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
