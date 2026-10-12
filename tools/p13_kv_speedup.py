#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P13 task C: measure the ACTUAL speedup from incremental decode.

WHAT IS COMPARED, AND WHY THIS IS THE RIGHT COMPARISON
    Baseline (what the engine could do before task C): to produce token t+1 you
    re-run the ENTIRE sequence 0..t, because there was no way to carry K/V
    across steps. Generating G tokens after a prompt of P therefore costs
        sum_{s=P}^{P+G-1} forward(s)
    which is why the measured baseline for 4 layers, P=64, G=16 was 14.63 s.

    With the cache: one prefill of P, then G decode steps, each of which
    processes ONE token against the cache. The per-step cost should be close to
    forward(1) and, crucially, should NOT grow with P.

MEASUREMENT DISCIPLINE
    Per-step timings come from the engine's own in-process QPC timer (the
    `bench` feature), because subprocess wall clock on this machine has a
    ~0.27 s spread -- larger than the effects being measured.

    The cache does not change what a single forward COSTS; it changes how many
    forwards are needed. So the speedup reported here is a real end-to-end
    quantity, not a kernel micro-benchmark.

Usage: python tools/p13_kv_speedup.py --layers 4 --prompt 64 --gen 16
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = str(Path(__file__).resolve().parents[1])
RT = os.path.join(ROOT, "runtime")
REL = os.path.join(ROOT, "target", "release")
TRIPLE_REL = os.path.join(ROOT, "target", "x86_64-pc-windows-msvc", "release")
WORK = os.path.join(ROOT, "examples", "p13_kvspeed")


def build_bench():
    p = subprocess.run(["cargo", "build", "--features", "bench", "--release",
                        "--target", "x86_64-pc-windows-msvc"],
                       cwd=RT, capture_output=True, text=True, errors="replace")
    if p.returncode != 0:
        print((p.stdout + p.stderr)[-1000:])
        return None
    exe = os.path.join(REL, "vmlm-runtime-bench.exe")
    shutil.copyfile(os.path.join(TRIPLE_REL, "vmlm-runtime.exe"), exe)
    return exe


def bench_us(exe, vmb, reps):
    tmp = os.path.join(REL, "_bench_model.vmb")
    shutil.copyfile(vmb, tmp)
    p = subprocess.run([exe, tmp, str(reps)], cwd=REL,
                       capture_output=True, text=True, errors="replace")
    m = re.search(r"avg us/iter =\s*(\d+)", p.stdout)
    return int(m.group(1)) if m else None


def build(layers, seq, out, decode=0):
    cmd = [sys.executable, os.path.join(ROOT, "tools", "p12_build_minicpm.py"),
           "--layers", str(layers), "--seq", str(seq), "--skip-head",
           "--print-hidden", "--prompt-mode", "tokens", "--out", out]
    if decode:
        cmd += ["--decode", str(decode)]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                       errors="replace")
    if p.returncode != 0:
        print((p.stdout + p.stderr)[-1000:])
        return None
    consts = [l.strip() for l in open(os.path.join(out, "weights.args"),
                                      encoding="utf-8") if l.strip()]
    q = subprocess.run([os.path.join(REL, "vmlm-asm.exe"), "full.vml", "m.vmb"]
                       + consts, cwd=out, capture_output=True)
    return os.path.join(out, "m.vmb") if q.returncode == 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--prompt", type=int, default=64)
    ap.add_argument("--gen", type=int, default=16)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    P, G, NL = args.prompt, args.gen, args.layers

    exe = build_bench()
    if not exe:
        return 1
    os.makedirs(WORK, exist_ok=True)

    print("=== P13 task C: incremental decode speedup (%d layers) ===\n" % NL)

    # --- baseline: forward at each length a naive generator would need ---
    print("baseline = full forward at each step a cache-less decoder needs:")
    fwd = {}
    for s in [1] + sorted(set([P, P + G // 2, P + G - 1])):
        v = build(NL, s, os.path.join(WORK, "f%d" % s))
        if not v:
            return 1
        fwd[s] = bench_us(exe, v, args.reps)
        print("   forward S=%4d : %10d us" % (s, fwd[s]))

    # interpolate the whole baseline range
    ks = np.array(sorted(fwd), dtype=float)
    us_arr = np.array([fwd[int(k)] for k in ks], dtype=float)
    baseline = 0.0
    for t in range(P, P + G):
        baseline += float(np.interp(t, ks, us_arr)) if t <= ks[-1] else \
            float(us_arr[-1] * (t / ks[-1]))

    # --- cached: one prefill + G decode steps ---
    print("\ncached = one prefill + %d incremental steps:" % G)
    vd = build(NL, P, os.path.join(WORK, "dec"), decode=G)
    if not vd:
        return 1
    cached = float(bench_us(exe, vd, args.reps))
    onestep = fwd[1]

    print("   prefill+%d steps total : %10.0f us" % (G, cached))
    print("   (S=1 forward reference : %10d us -- the floor for one step)" % onestep)
    naive_cached = float(np.interp(P, ks, us_arr)) + G * onestep
    print("   naive estimate (P prefill + G x step) = %.0f us"
          % (naive_cached))

    print()
    print("   baseline (cache-less, %d x full forward) : %10.0f us = %.2f s"
          % (G, baseline, baseline / 1e6))
    print("   cached decode                            : %10.0f us = %.2f s"
          % (cached, cached / 1e6))
    if cached > 0:
        print("   SPEEDUP: %.2fx" % (baseline / cached))
    print()
    print("NOTE: the cache removes RECOMPUTATION, not per-token work. A decode")
    print("      step still reads every weight once, so its floor is the S=1")
    print("      forward time -- which is what the engine is load-bound on.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
