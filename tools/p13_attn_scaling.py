#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P13 task C0: is the attention cost actually O(S^2)?

WHY MEASURE FIRST
    The handover note is explicit: "先测 S=1/4/16/64 的耗时曲线，确认平方增长是真的再动手".
    There is a concrete reason to be suspicious. The per-head attention GEMM is
    [S,hd] x [S,hd]^T, which is genuinely S^2 -- but the projections around it
    (q/k/v/o/gate/up/down) are [S,H] x [H,K], i.e. LINEAR in S and, at H=1536,
    far larger in absolute terms than the attention at small S. So the naive
    expectation "attention dominates" may well be false for S<=64, in which case
    a KV cache buys little and the real win is elsewhere.

WHAT IS MEASURED
    A real 24-layer MiniCPM forward at several S, timed end to end with stdout
    going to a REAL FILE. Draining the engine through a PowerShell pipe once made
    a 1.14 s run look like 20 minutes, so no pipes here.

    Also measured separately: the lm_head projection, which is [S,V] with
    V=130560 and does NOT depend on attention at all. If lm_head dominates the
    curve, no amount of KV caching will move the number.

Usage:
    python tools/p13_attn_scaling.py [--layers 24] [--seqs 1,4,16,64] [--reuse]
"""
import argparse
import json
import os
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
REL = os.path.join(ROOT, "target", "release")
ENG = os.path.join(REL, "vmlm-runtime.exe")
WORK = os.path.join(ROOT, "examples", "p13_scaling")


def build(layers, seq, out, skip_head=False):
    cmd = [sys.executable, os.path.join(ROOT, "tools", "p12_build_minicpm.py"),
           "--layers", str(layers), "--seq", str(seq),
           "--prompt-mode", "tokens", "--out", out]
    if skip_head:
        cmd += ["--skip-head", "--print-hidden"]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                       errors="replace")
    if p.returncode != 0:
        print("BUILD FAILED seq=%d" % seq)
        print((p.stdout + p.stderr)[-1500:])
        return False
    return True


def assemble(d):
    consts = [l.strip() for l in open(os.path.join(d, "weights.args"),
                                      encoding="utf-8") if l.strip()]
    p = subprocess.run([os.path.join(REL, "vmlm-asm.exe"), "full.vml", "m.vmb"]
                       + consts, cwd=d, capture_output=True)
    return p.returncode == 0 and os.path.exists(os.path.join(d, "m.vmb"))


def run_timed(vmb, seq, reps=1):
    shutil.copyfile(vmb, os.path.join(REL, "model.vmb"))
    with open(os.path.join(REL, "tokens.bin"), "wb") as f:
        np.arange(3, 3 + seq, dtype="<u4").tofile(f)
    out = os.path.join(WORK, "_t.txt")
    best = None
    for _ in range(reps):
        t0 = time.time()
        with open(out, "wb") as f:
            p = subprocess.run([ENG], cwd=REL, stdout=f, stderr=subprocess.DEVNULL)
        dt = time.time() - t0
        if p.returncode != 0:
            return None, p.returncode
        best = dt if best is None else min(best, dt)
    return best, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--seqs", default="1,4,16,64")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--head", action="store_true",
                    help="use --skip-head (measures the layer stack WITHOUT the "
                         "[S,V] lm_head projection)")
    args = ap.parse_args()
    seqs = [int(x) for x in args.seqs.split(",")]
    os.makedirs(WORK, exist_ok=True)

    mode = "layer stack only (--skip-head)" if args.head else "full (incl. lm_head)"
    print("=== P13 task C0: attention scaling, %d layers, %s ===\n"
          % (args.layers, mode))
    print("%5s %10s %12s %12s" % ("S", "seconds", "s/S", "growth vs prev"))
    print("-" * 46)
    rows = []
    for s in seqs:
        d = os.path.join(WORK, "s%d" % s)
        if not build(args.layers, s, d, skip_head=args.head):
            return 1
        if not assemble(d):
            print("ASSEMBLE FAILED seq=%d" % s)
            return 1
        dt, rc = run_timed(os.path.join(d, "m.vmb"), s, reps=args.reps)
        if dt is None:
            print("engine exit=%d at S=%d" % (rc, s))
            return 1
        rows.append((s, dt))
        prev = rows[-2][1] if len(rows) > 1 else None
        growth = ("%.2fx (S ratio %.1fx)" % (dt / prev, s / rows[-2][0])
                  if prev else "-")
        print("%5d %9.3fs %11.4f %12s" % (s, dt, dt / s, growth))

    # Fit the exponent: if attention dominates, t ~ S^2 (and t/S grows ~S).
    if len(rows) >= 3:
        s = np.array([r[0] for r in rows], dtype=np.float64)
        t = np.array([r[1] for r in rows], dtype=np.float64)
        # subtract the S=min intercept so a large constant (process start,
        # 401 MB of page faults for the embedding table) does not fake the slope
        t0 = t[0]
        mask = t > t0
        if mask.sum() >= 2:
            k = np.polyfit(np.log(s[mask]), np.log(t[mask] - t0), 1)[0]
            print("\nscaling exponent above the S=%d baseline: %.2f" % (seqs[0], k))
            print("  ~1.0 => linear in S (attention not dominant at these sizes)")
            print("  ~2.0 => quadratic (attention dominates)")
    print()
    print("NOTE: every run pays a fixed cost (loading 2.16 GB through mmap and")
    print("      faulting in the embedding/lm_head pages), which does not scale")
    print("      with S. That is why the exponent is fitted above a baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
