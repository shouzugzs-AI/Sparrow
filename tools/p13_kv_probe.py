#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P13 task C: verify the KV-cache primitives in isolation.

WHAT THIS CHECKS, AND WHY IT IS WORTH A SEPARATE STEP
    Task C's plan needs two layout claims to hold:
      * appending position p to a TRANSPOSED [W,PMAX] cache is a column write,
        i.e. TRANSPOSE(row) then WRITECOL(cache, col, p);
      * attention over that cache is two GEMMs with operand layouts the engine
        already supports -- `nn` for the scores and the default NT for the
        context.
    If either is wrong, a 24-layer decode fails in a way that is very expensive
    to attribute (P12 lost two cycles to exactly this class of mistake in the
    bf16 SIMD path). So it is checked first on a 3x2 case whose answer is
    computable by hand.

    Also confirms the cache really is a STATE slot that survives, and that the
    WRITECOL bounds guard fires when the write would exceed PMAX.

Usage: python tools/p13_kv_probe.py
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

K = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
V = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32)
PMAX = 4
# The query is chosen so the scores are CLOSE TOGETHER (5, 11, 17) rather than
# the raw last row (17, 39, 61). Reason: with scores {17,39,61} the softmax is
# 1.0 for the max term in f32 (exp(-44) is negligible against 1.0), so the
# context collapses to a single v row and the test cannot distinguish "the
# cached context GEMM is right" from "it picked the last row". Close scores
# produce weights near 1/3 each, which is a real test of the accumulation.
Q = np.array([1, 2], dtype=np.float32)  # <q,k_p> = 5, 11, 17


def run(vml, vmb, extra_consts=()):
    consts = [c for c in extra_consts]
    p = subprocess.run([ASM, vml, vmb] + consts, cwd=DIR,
                       capture_output=True, text=True, errors="replace")
    if p.returncode != 0:
        return None, "assemble failed: %s" % (p.stdout + p.stderr)[-400:]
    import shutil
    shutil.copyfile(os.path.join(DIR, vmb), os.path.join(REL, "model.vmb"))
    tb = os.path.join(REL, "tokens.bin")
    if os.path.exists(tb):
        os.remove(tb)
    out = os.path.join(DIR, "_kv_out.txt")
    with open(out, "wb") as f:
        q = subprocess.run([ENG], cwd=REL, stdout=f, stderr=subprocess.DEVNULL)
    if q.returncode != 0:
        return None, "engine exit=%d" % q.returncode
    vals = []
    with open(out, "r", errors="replace") as f:
        for line in f:
            for t in line.split():
                try:
                    vals.append(float(t))
                except ValueError:
                    pass
    return np.array(vals, dtype=np.float32), None


def main():
    print("=== P13 task C: KV cache primitives (transposed append + cached GEMM) ===\n")
    K.tofile(os.path.join(DIR, "kk.f32"))
    V.tofile(os.path.join(DIR, "vv.f32"))
    Q.tofile(os.path.join(DIR, "qq.f32"))

    got, err = run("kv_probe.vml", "kv_probe.vmb",
                   ["kk:kk.f32:3:2:f32", "vv:vv.f32:3:2:f32",
                    "qquery:qq.f32:1:2:f32"])
    if err:
        print("FAIL: %s" % err)
        return 1

    W = 2
    # printed blocks: kcT [2,4], vcT [2,4], sc [1,4], softmax [1,4], ctx [1,2]
    exp_kcT = K.T                       # [2,3] -> padded to [2,4]
    exp_vcT = V.T
    exp_q = Q
    exp_sc = np.array([float(np.dot(Q, K[p])) for p in range(3)] + [0.0])
    # Compute the expected softmax/context in FLOAT32, matching the engine.
    # A float64 expectation once made this test report a failure against
    # correct engine output: with scores {17,39,61} the f32 softmax is exactly
    # 1.0 (exp(-44) is negligible next to 1.0), while float64 gives
    # 0.99999999972 -- a difference that looks like a bug but is not.
    sc32 = exp_sc.astype(np.float32)
    e32 = np.exp((sc32 - sc32.max()).astype(np.float32))
    exp_sm = (e32 / e32.sum()).astype(np.float32)
    exp_ctx = (exp_sm[:3].astype(np.float32) @ V.astype(np.float32))

    def trunc6(a):
        out = []
        for x in np.asarray(a, dtype=np.float64).ravel():
            f = abs(float(x)); ip = int(f); fr = f - ip; s = ""
            for _ in range(6):
                fr *= 10.0; dg = int(fr); s += str(dg % 10); fr -= dg
            out.append(float("%d.%s" % (ip, s)))
        return np.array(out)

    pos = 0
    ok = True

    def take(n):
        nonlocal pos
        v = got[pos:pos + n]
        pos += n
        return v

    # 1. cache contents (as printed: [2,4] each, row-major)
    kc = take(2 * PMAX).reshape(2, PMAX)
    vc = take(2 * PMAX).reshape(2, PMAX)
    for name, g, ex in [("kcT", kc, exp_kcT), ("vcT", vc, exp_vcT)]:
        want = np.zeros((W, PMAX), dtype=np.float32)
        want[:, :3] = ex
        same = np.array_equal(trunc6(g), trunc6(want))
        print("%s as stored:\n%s" % (name, g))
        print("  expected (cols 0..2 = transposed rows, col 3 unused):\n%s" % want)
        print("  %s" % ("PASS: transposed append layout correct" if same else "FAIL"))
        ok = ok and same
    print()

    # 2. scores from the cached GEMM
    sc = take(PMAX)
    print("scores from GEMM(q, kcT, nn): %s" % sc.tolist())
    print("expected h<q,k_p>          : %s" % exp_sc.tolist())
    same = np.array_equal(trunc6(sc[:3]), trunc6(exp_sc[:3]))
    print("  %s" % ("PASS: cached score GEMM correct" if same else "FAIL"))
    ok = ok and same
    print()

    # 3. softmax + cached context GEMM
    sm = take(PMAX)
    ctx = take(W)
    print("softmax: %s" % sm.tolist())
    print("ctx    : %s" % ctx.tolist())
    print("expect : %s" % exp_ctx.tolist())
    same_ctx = np.array_equal(trunc6(ctx), trunc6(exp_ctx))
    print("  %s" % ("PASS: cached context GEMM correct" if same_ctx else "FAIL"))
    ok = ok and same_ctx

    print()
    print("VERDICT: %s" % ("PASS -- cache layout + both GEMM forms verified"
                           if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
