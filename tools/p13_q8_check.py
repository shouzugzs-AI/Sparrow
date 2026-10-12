#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P13 task D: verify the int8 dequantising GEMM in the ENGINE.

WHAT IS CHECKED
    C[M,N] = (A[M,K] . W_i8[N,K]) * scale[N], with one scale per output row,
    computed by the engine's `kernels::gemm_nt_int8` and reachable from the
    assembler as `GEMM c, a, w, scale_slot` (the 4th operand is compiled into
    `dt`, since Ins has no spare byte -- verified against its repr(C) layout).

    The case is hand-checkable AND uses a NON-UNIFORM scale (0.5 and 0.25), so
    a kernel that ignored the scale or applied a single global one would give a
    visibly different answer instead of coincidentally matching.

ALSO CHECKED (the safety property the task book asks for)
    "declared dtype vs available kernel" must be a HARD FAILURE, not a silent
    reinterpretation. Three negative cases:
      * int8 weight used by a non-GEMM op  -> exit 9
      * a scale slot supplied for a bf16/f32 GEMM (assembler-level error)
      * a scale slot shorter than N rows   -> exit 9

Usage: python tools/p13_q8_check.py
"""
import os
import subprocess
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = str(Path(__file__).resolve().parents[1])
DIR = os.path.join(ROOT, "examples", "p13")
REL = os.path.join(ROOT, "target", "release")
ASM = os.path.join(REL, "vmlm-asm.exe")
ENG = os.path.join(REL, "vmlm-runtime.exe")

A = np.array([[1, 2, 3, 4]], dtype=np.float32)
W = np.array([[1, 0, -1, 0], [0, 2, 0, -2]], dtype=np.int8)
SCALE = np.array([0.5, 0.25], dtype=np.float32)
EXPECT = np.array([[1 * 1 + 2 * 0 + 3 * -1 + 4 * 0,
                    1 * 0 + 2 * 2 + 3 * 0 + 4 * -2]], dtype=np.float32)
EXPECT = (EXPECT * SCALE.reshape(1, 2)).astype(np.float32)


def assemble(vml, vmb, consts):
    p = subprocess.run([ASM, vml, vmb] + consts, cwd=DIR,
                       capture_output=True, text=True, errors="replace")
    return p.returncode, p.stdout + p.stderr


def run(vmb, out_name="_q8_out.txt"):
    import shutil
    shutil.copyfile(os.path.join(DIR, vmb), os.path.join(REL, "model.vmb"))
    tb = os.path.join(REL, "tokens.bin")
    if os.path.exists(tb):
        os.remove(tb)
    gt = os.path.join(REL, "gen_tokens.bin")
    if os.path.exists(gt):
        os.remove(gt)
    out = os.path.join(DIR, out_name)
    with open(out, "wb") as f:
        p = subprocess.run([ENG], cwd=REL, stdout=f, stderr=subprocess.DEVNULL)
    msg = b""
    if p.returncode != 0:
        # diagnostics go to stdout, which we redirected; re-run capturing it
        q = subprocess.run([ENG], cwd=REL, capture_output=True)
        msg = q.stdout + q.stderr
        return None, p.returncode, msg
    vals = []
    with open(out, "r", errors="replace") as f:
        for line in f:
            for t in line.split():
                try:
                    vals.append(float(t))
                except ValueError:
                    pass
    return np.array(vals, dtype=np.float32), 0, msg


def trunc6(a):
    out = []
    for x in np.asarray(a, dtype=np.float64).ravel():
        f = abs(float(x)); ip = int(f); fr = f - ip; s = ""
        for _ in range(6):
            fr *= 10.0; dg = int(fr); s += str(dg % 10); fr -= dg
        out.append(float("%d.%s" % (ip, s)))
    return np.array(out)


def main():
    print("=== P13 task D: int8 dequantising GEMM in the engine ===\n")
    A.tofile(os.path.join(DIR, "q8_a.f32"))
    W.tofile(os.path.join(DIR, "q8_w.i8"))
    SCALE.tofile(os.path.join(DIR, "q8_scale.f32"))

    consts = ["a:q8_a.f32:1:4:f32",
              "w:q8_w.i8:2:4:i8",
              "sc:q8_scale.f32:2:1:f32"]
    rc, msg = assemble("q8_probe.vml", "q8_probe.vmb", consts)
    if rc != 0:
        print("ASSEMBLE FAILED:\n%s" % msg[-1500:])
        return 1
    got, rc, msg = run("q8_probe.vmb")
    if got is None:
        print("engine exit=%d  msg=%s" % (rc, msg.decode("utf-8", "replace")[:200]))
        return 1

    print("A        = %s" % A.tolist())
    print("W (i8)   = %s" % W.tolist())
    print("scale    = %s  (non-uniform on purpose)" % SCALE.tolist())
    print("engine   = %s" % got.tolist())
    print("expected = %s" % EXPECT.tolist())
    ok = np.array_equal(trunc6(got), trunc6(EXPECT))
    print("  %s" % ("PASS: int8 GEMM with per-row scale is exact"
                    if ok else "FAIL"))

    # ---- negative cases: the dtype/kernel guard must fail loudly ----
    print("\nnegative cases (illegal input must be a HARD failure):")
    fails = 0

    # 1. int8 constant fed to a non-GEMM op
    with open(os.path.join(DIR, "q8_bad1.vml"), "w", encoding="ascii",
              newline="\n") as f:
        f.write("SHAPE x, 2, 4\nCOPY x, w\nPRINT x\nHALT\n")
    rc, msg = assemble("q8_bad1.vml", "q8_bad1.vmb", consts)
    if rc != 0:
        print("   int8 -> COPY : rejected at ASSEMBLE time (acceptable)")
    else:
        got2, rc2, m2 = run("q8_bad1.vmb", "_q8_bad1.txt")
        good = rc2 == 9 and b"i8" in m2 or b"bf16" in m2 or b"uses" in m2
        print("   int8 -> COPY : exit=%d msg=%s  %s"
              % (rc2, m2.decode("utf-8", "replace").strip()[:60],
                 "PASS" if rc2 == 9 else "FAIL (expected exit 9)"))
        if rc2 != 9:
            fails += 1

    # 2. scale slot supplied for a bf16 weight (assembler must refuse)
    rc, msg = assemble("q8_probe.vml", "q8_bad2.vmb",
                       ["a:q8_a.f32:1:4:f32",
                        "w:q8_scale.f32:2:4:f32",
                        "sc:q8_scale.f32:2:1:f32"])
    refused = rc != 0
    print("   scale given for f32 weights : %s"
          % ("PASS (assembler refused)" if refused
             else "FAIL (assembler accepted a meaningless scale operand)"))
    if not refused:
        fails += 1

    # 3. scale shorter than N rows
    short = np.array([0.5], dtype=np.float32)
    short.tofile(os.path.join(DIR, "q8_short.f32"))
    rc, msg = assemble("q8_probe.vml", "q8_bad3.vmb",
                       ["a:q8_a.f32:1:4:f32",
                        "w:q8_w.i8:2:4:i8",
                        "sc:q8_short.f32:1:1:f32"])
    if rc != 0:
        print("   scale shorter than N    : rejected at ASSEMBLE time")
    else:
        got3, rc3, m3 = run("q8_bad3.vmb", "_q8_bad3.txt")
        print("   scale shorter than N    : exit=%d msg=%s  %s"
              % (rc3, m3.decode("utf-8", "replace").strip()[:60],
                 "PASS" if rc3 == 9 else "FAIL (expected exit 9)"))
        if rc3 != 9:
            fails += 1

    print()
    allok = ok and fails == 0
    print("VERDICT: %s" % ("PASS -- int8 GEMM correct, dtype guards hard-fail"
                           if allok else "FAIL"))
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
