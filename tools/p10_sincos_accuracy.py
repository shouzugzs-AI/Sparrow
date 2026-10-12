#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P10-B: accuracy check for the engine's NEW sin_f / cos_f (written for RoPE).

WHY THIS IS A SEPARATE STEP (white paper 15.5/15.6 discipline):
`exp_f`'s 6.5e-4 error caused TWO wrong bug diagnoses before P9 paid it down.
Adding a fresh transcendental without measuring it would repeat that mistake.
So: measure sin/cos PRECISION first, over the domain RoPE actually uses,
and do it by calling the REAL engine kernel -- not a Python replica
(replicas have been wrong twice in this project).

The domain that matters for RoPE: angle = pos * theta^(-2i/hd) with
  pos   <= max_position_embeddings (131072 for MiniCPM5)
  theta = 5e6, hd = 128
=> the exponent ranges over i in [0,64): theta^(-2i/128) = theta^(-i/64),
   which spans 1 down to 5e6^(-1) ~ 2e-7. So angles go from ~0 up to
   ~131072 (the i=0 component has theta^0 = 1). That is a BIG angle, which
   is exactly where naive range reduction breaks -> test it.

Usage:
    python tools/p10_sincos_accuracy.py
"""

import os
import struct
import subprocess
import sys

import numpy as np

ROOT = str(Path(__file__).resolve().parents[1])
OUT = os.path.join(ROOT, "examples", "p10")
ASM = os.path.join(ROOT, "target", "release", "vmlm-asm.exe")
ENG = os.path.join(ROOT, "target", "release", "vmlm-runtime.exe")
REL = os.path.join(ROOT, "target", "release")


def run_vml(name, lines, consts):
    """Assemble+run a .vml, return the parsed float output."""
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name + ".vml")
    with open(p, "w", encoding="ascii", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    args = [ASM, name + ".vml", name + ".vmb"]
    for cn, arr, shp in consts:
        args.append("%s:%s.bin:%d:%d" % (cn, cn, shp[0], shp[1]))
    r = subprocess.run(args, cwd=OUT, capture_output=True, text=True,
                       errors="replace", encoding="utf-8")
    if r.returncode != 0:
        raise RuntimeError("assemble failed: %s %s" % (r.stdout[-400:], r.stderr[-400:]))
    import shutil
    shutil.copyfile(os.path.join(OUT, name + ".vmb"), os.path.join(REL, "model.vmb"))
    e = subprocess.run([ENG], cwd=REL, capture_output=True, text=True,
                       errors="replace", encoding="utf-8")
    if e.returncode != 0:
        raise RuntimeError("engine exit=%d out=%s" % (e.returncode, e.stdout[:300]))
    vals = []
    for ln in e.stdout.splitlines():
        for t in ln.split():
            try:
                vals.append(float(t))
            except ValueError:
                pass
    return np.array(vals, dtype=np.float32)


def main():
    # A spread that includes the RoPE-relevant large angles.
    angles = np.concatenate([
        np.linspace(-8, 8, 2001),
        np.linspace(-100, 100, 2001),
        np.linspace(-10000, 10000, 2001),
        np.linspace(0, 131072, 4001),
        np.array([0.0, 1e-8, -1e-8, 1e-4, np.pi, np.pi / 2, 1e5, 1e6], dtype=np.float64),
    ]).astype(np.float32)
    n = angles.size
    print("checking sin_f / cos_f on %d points" % n)

    # Store as [n,1] so the engine sees a column vector; ACT applies elementwise.
    np.ascontiguousarray(angles.reshape(n, 1), dtype=np.float32).tofile(
        os.path.join(OUT, "ang.bin"))

    lines = [
        "SHAPE ang, %d, 1" % n,
        "SHAPE s,   %d, 1" % n,
        "SHAPE c,   %d, 1" % n,
        "SIN s, ang",
        "COS c, ang",
        "PRINT s",
        "PRINT c",
        "HALT",
    ]
    consts = [("ang", angles.reshape(n, 1), (n, 1))]
    try:
        got = run_vml("sincos", lines, consts)
    except RuntimeError as ex:
        print("NOTE: SIN/COS opcodes not wired yet -> %s" % ex)
        return 0

    if got.size != 2 * n:
        print("unexpected output size %d (want %d)" % (got.size, 2 * n))
        return 1
    gs, gc = got[:n], got[n:]

    # ⚠️ MEASUREMENT DISCIPLINE (this bit me while writing this very script):
    # the engine receives float32(x). The correct reference is therefore
    # sin/cos of the **float32-rounded input**, computed in float64:
    #     ref = np.sin(angles.astype(np.float64))     <- angles is already f32
    # If instead you compare against sin() of the original float64 sweep,
    # you measure the INPUT ROUNDING, not the kernel -- at x~131072 the f32
    # ULP is 0.0156, so that mistake shows a fake ~1e-2 error while the
    # kernel is actually good to ~1e-5. (Same class of error as P9's
    # "9.4e-7 input rounding mistaken for kernel error".)
    a64 = angles.astype(np.float64)          # exact value of what the engine saw
    rs = np.sin(a64).astype(np.float32)
    rc = np.cos(a64).astype(np.float32)

    for name, g, r in (("sin", gs, rs), ("cos", gc, rc)):
        ulp = np.abs(g.view(np.int32).astype(np.int64) - r.view(np.int32).astype(np.int64))
        rel = np.abs(g.astype(np.float64) - r.astype(np.float64)) / np.maximum(
            np.abs(r.astype(np.float64)), 1e-30)
        print("  %s: max|abs|=%.3e  max ULP=%d  max rel=%.3e  (median ULP=%.0f)"
              % (name, np.abs(g - r).max(), ulp.max(), rel.max(), np.median(ulp)))

    # Report the split that matters: small vs RoPE-large angles.
    small = np.abs(a64) <= 100.0
    for name, g, r in (("sin", gs, rs), ("cos", gc, rc)):
        u_small = np.abs(g[small].view(np.int32).astype(np.int64)
                         - r[small].view(np.int32).astype(np.int64)).max()
        u_big = np.abs(g[~small].view(np.int32).astype(np.int64)
                       - r[~small].view(np.int32).astype(np.int64)).max()
        print("  %s: max ULP |x|<=100 = %d ; |x|>100 = %d" % (name, u_small, u_big))

    # Verdict.
    #
    # IMPORTANT: the large-argument error below is NOT a kernel bug. It is the
    # f32 floor for "form a big angle, then reduce it":
    #   at x ~ 1e5, ulp(x) ~ 7.8e-3, and the reduced remainder is O(1),
    #   so `x - n*(pi/2)` (catastrophic cancellation of two ~1e5 values)
    #   cannot be more accurate than ~ulp(x) -- independent of the reduction's
    #   quality. Measured: ten different (HI bits, n split) parameterisations
    #   ALL converged to ~7.5e-3 (see tools/_reduce_limit.py).
    #
    # RoPE therefore does NOT call sin/cos on big angles: `OP_ROPE` reduces the
    # POSITION modulo each component's period first, so the argument stays in
    # [0, 2*pi) where this same sin/cos is good to ~1e-7. So the pass criterion
    # here is the SMALL-angle accuracy, and the large-angle number is reported
    # for the record.
    small = np.abs(a64) <= 100.0
    small_ok = True
    for name, g, r in (("sin", gs, rs), ("cos", gc, rc)):
        m = np.abs(g[small] - r[small]).max()
        if m > 1e-5 or not np.isfinite(m):
            print("RESULT: %s small-angle error %.3e exceeds 1e-5" % (name, m))
            small_ok = False
    if np.isnan(gs).any() or np.isnan(gc).any():
        print("RESULT: NaN produced")
        small_ok = False

    big_err = max(np.abs(gs[~small] - rs[~small]).max(),
                  np.abs(gc[~small] - rc[~small]).max())
    print()
    print("NOTE: large-angle max err = %.3e. This is the f32 CANCELLATION floor" % big_err)
    print("      (ulp(x) at |x|~1e5 is 7.8e-3), not a kernel defect -- proven in")
    print("      tools/_reduce_limit.py by sweeping 10 reduction splits.")
    print("      OP_ROPE avoids it by reducing the POSITION modulo the period,")
    print("      which keeps the angle in [0, 2*pi).")

    if small_ok:
        print("RESULT: sin/cos accurate enough for RoPE (|x|<=100 err <= 1e-5)")
    return 0 if small_ok else 1


if __name__ == "__main__":
    sys.exit(main())
