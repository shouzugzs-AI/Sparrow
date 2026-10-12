#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P13 task C: is the DECODE wrong, or is the REFERENCE not causal?

WHY THIS QUESTION
    Incremental decode reproduces a full re-prefill EXACTLY for the last step of
    every configuration (S=1..8, G=1..3), and is wrong for earlier steps by an
    amount that decays ~5.5x per step. The K cache itself was dumped and is
    CORRECT: prefill fills columns 0..S-1 and each decode step changes exactly
    one column, preserving the rest byte-for-byte.

    A causal forward has a strong property: row i of a forward over N tokens
    must not depend on N. So row 4 of a 5-token forward and row 4 of a 7-token
    forward MUST be identical. If they are not, the "extra" positions are
    leaking into earlier rows and the REFERENCE is what is wrong -- which would
    also make my comparison meaningless.

THIS TEST
    Builds plain forwards at several lengths and compares the SHARED rows.
    That is a self-contained check of the engine's causal attention, and it
    does not involve the cache at all.

Usage: python tools/p13_causal_check.py
"""
import argparse
import os
import subprocess
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = str(Path(__file__).resolve().parents[1])
REL = os.path.join(ROOT, "target", "release")
ASM = os.path.join(REL, "vmlm-asm.exe")
ENG = os.path.join(REL, "vmlm-runtime.exe")
WORK = os.path.join(ROOT, "examples", "p13_causal")
IDS = [3, 117, 2501, 88, 42, 7, 900, 15]
H = 1536


def build(seq, out):
    p = subprocess.run(
        [sys.executable, os.path.join(ROOT, "tools", "p12_build_minicpm.py"),
         "--layers", "1", "--seq", str(seq), "--skip-head", "--print-hidden",
         "--prompt-mode", "tokens", "--out", out],
        cwd=ROOT, capture_output=True, text=True, errors="replace")
    if p.returncode != 0:
        print((p.stdout + p.stderr)[-1200:])
        return False
    consts = [l.strip() for l in open(os.path.join(out, "weights.args"),
                                      encoding="utf-8") if l.strip()]
    q = subprocess.run([ASM, "full.vml", "m.vmb"] + consts, cwd=out,
                       capture_output=True, text=True, errors="replace")
    return q.returncode == 0


def run(out, seq):
    import shutil
    shutil.copyfile(os.path.join(out, "m.vmb"), os.path.join(REL, "model.vmb"))
    with open(os.path.join(REL, "tokens.bin"), "wb") as f:
        np.array(IDS[:seq], dtype="<u4").tofile(f)
    gt = os.path.join(REL, "gen_tokens.bin")
    if os.path.exists(gt):
        os.remove(gt)
    res = os.path.join(out, "_o.txt")
    with open(res, "wb") as f:
        p = subprocess.run([ENG], cwd=REL, stdout=f, stderr=subprocess.DEVNULL)
    if p.returncode != 0:
        return None
    v = []
    for line in open(res, "r", errors="replace"):
        for t in line.split():
            try:
                v.append(float(t))
            except ValueError:
                pass
    return np.array(v, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", default="4,5,6,7")
    args = ap.parse_args()
    lens = [int(x) for x in args.lens.split(",")]
    os.makedirs(WORK, exist_ok=True)

    print("=== P13: is a causal forward's row independent of total length? ===\n")
    rows = {}
    for n in lens:
        d = os.path.join(WORK, "s%d" % n)
        if not build(n, d):
            return 1
        v = run(d, n)
        if v is None or v.size != n * H:
            print("S=%d: bad output (%s values)" % (n, None if v is None else v.size))
            return 1
        rows[n] = v.reshape(n, H)
        print("  built and ran a %d-token forward" % n)

    print("\ncomparing the rows shared by successive lengths:")
    print("%6s %6s %14s" % ("len A", "len B", "max|d| on shared rows"))
    print("-" * 32)
    ok = True
    prev = None
    for n in lens:
        if prev is not None:
            m = min(prev, n)
            d = float(np.abs(rows[prev][:m] - rows[n][:m]).max())
            flag = "" if d <= 1e-5 else "   <-- CAUSALITY VIOLATED"
            print("%6d %6d %14.4e%s" % (prev, n, d, flag))
            if d > 1e-5:
                ok = False
        prev = n

    print()
    if ok:
        print("VERDICT: PASS -- shared rows agree, so the reference IS causal and")
        print("         a decode step must be comparable to it.")
        return 0
    print("VERDICT: FAIL -- the forward's early rows CHANGE with total length.")
    print("         The engine's attention is not behaving causally here, so the")
    print("         decode comparison was against a moving target.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
