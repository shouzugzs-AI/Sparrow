#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P13 task C: baseline cost of GENERATION (not prefill), before building a cache.

WHY THIS IS THE RIGHT BASELINE
    The engine has no way to carry K/V across calls, so the only way to produce
    token t+1 today is to re-run the ENTIRE sequence 0..t. Generation is
    therefore quadratic in the number of generated tokens even though a single
    forward is not. The task book framed task C as "attention is O(S^2) in
    prefill"; the measurement in tools/p13_bench_attn.py showed that is only
    1.2% of a forward at S<=64 (MLP projections dominate ~74% of MACs). The
    real cost of a KV cache here is avoiding RECOMPUTATION across steps.

WHAT IS MEASURED
    1. Full forward time at a range of S (the unit of recomputation).
    2. The cumulative cost of generating G tokens the way the engine must do it
       today: sum over s in [P, P+G) of forward(s).

    Forward times come from inside-process QPC timing (bench feature), because
    subprocess wall clock has a 0.27 s spread on this machine -- larger than the
    effects being measured.

Usage:
    python tools/p13_gen_baseline.py --layers 4 --prompt 64 --gen 16
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = str(Path(__file__).resolve().parents[1])
RT = os.path.join(ROOT, "runtime")
REL = os.path.join(ROOT, "target", "release")
TRIPLE_REL = os.path.join(ROOT, "target", "x86_64-pc-windows-msvc", "release")
WORK = os.path.join(ROOT, "examples", "p13_attn")


def build_bench():
    p = subprocess.run(["cargo", "build", "--features", "bench", "--release",
                        "--target", "x86_64-pc-windows-msvc"],
                       cwd=RT, capture_output=True, text=True, errors="replace")
    if p.returncode != 0:
        print((p.stdout + p.stderr)[-1200:])
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


def build_model(layers, seq, out):
    p = subprocess.run(
        [sys.executable, os.path.join(ROOT, "tools", "p12_build_minicpm.py"),
         "--layers", str(layers), "--seq", str(seq), "--skip-head",
         "--print-hidden", "--prompt-mode", "tokens", "--out", out],
        cwd=ROOT, capture_output=True, text=True, errors="replace")
    if p.returncode != 0:
        print((p.stdout + p.stderr)[-1200:])
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
    ap.add_argument("--seqs", default="1,8,16,32,64,128")
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args()

    exe = build_bench()
    if not exe:
        return 1

    seqs = [int(x) for x in args.seqs.split(",")]
    print("=== P13 task C: generation baseline (%d layers) ===\n" % args.layers)
    times = {}
    for s in seqs:
        d = os.path.join(WORK, "b%d" % s)
        vmb = build_model(args.layers, s, d)
        if not vmb:
            print("build failed at S=%d" % s)
            return 1
        us = bench_us(exe, vmb, args.reps)
        if us is None:
            print("bench failed at S=%d" % s)
            return 1
        times[s] = us
        print("  forward S=%4d : %10.0f us  (%.1f ms/token)" % (s, us, us / 1000.0))

    S = np.array(sorted(times), dtype=float)
    T = np.array([times[int(x)] for x in S], dtype=float)

    P, G = args.prompt, args.gen
    print("\n--- generation baseline: prompt=%d, generate=%d ---" % (P, G))
    tot = 0.0
    rows = []
    for t in range(P, P + G):
        # interpolate/extrapolate forward cost at length t
        us = float(np.interp(t, S, T)) if t <= S[-1] else \
            float(T[-1] * (t / S[-1]))
        tot += us
        rows.append((t, us))
    for t, us in rows[:4]:
        print("   token %4d : forward over %4d positions = %10.0f us" % (t, t, us))
    if len(rows) > 4:
        print("   ...")
        t, us = rows[-1]
        print("   token %4d : forward over %4d positions = %10.0f us" % (t, t, us))
    print("\n  TOTAL to generate %d tokens after a %d prompt: %.2f s"
          % (G, P, tot / 1e6))
    print("  A KV cache would instead do one prefill of %d (~%.2f s) plus %d"
          % (P, float(np.interp(P, S, T)) / 1e6, G))
    one = float(times[1]) if 1 in times else None
    if one:
        print("  incremental steps (~%d x %.1f ms) = %.2f s"
              % (G, one / 1000.0, G * one / 1e6))
        print("  => potential total %.2f s, i.e. %.1fx faster"
              % ((float(np.interp(P, S, T)) + G * one) / 1e6,
                 tot / (float(np.interp(P, S, T)) + G * one)))
    print()
    print("NOTE: this is the CEILING for a KV cache here; it assumes a decode")
    print("      step costs the same as the S=1 forward measured above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
