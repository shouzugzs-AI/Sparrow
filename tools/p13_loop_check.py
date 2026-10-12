#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P13 task E: does the LOOPED layer template match the UNROLLED build exactly?

WHAT IS BEING PROVEN
    The unrolled build emits NL copies of the layer body, with each copy
    referring to its own layer's constants by name (L0_q, L1_q, ... ~4180
    instructions for 24 layers).

    The looped build emits the body ONCE and runs it NL times under LOOP_CNT,
    with OP_LAYERB advancing a RUNTIME constant base by the per-layer stride.
    That only works if every layer's constants are laid out IDENTICALLY and
    with a uniform stride -- so this check is really testing that layout
    assumption, using the unrolled build as the oracle.

WHAT IS COMPARED
    Same weights, same prompt, same quantisation, only --loop differs.
    The hidden state must agree to the engine's printed precision (6 decimals),
    NOT approximately: the two builds execute the same arithmetic in the same
    order, so any difference means the base landed on the wrong bytes.

WHY THIS IS THE RIGHT ORACLE
    The unrolled build is already validated end-to-end against HuggingFace
    (rel 1.8806e-06, argmax 4/4). Matching it therefore transfers that
    validation to the looped build without needing another 9-minute HF run.

Usage: python tools/p13_loop_check.py --layers 4 --seq 4
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
WORK = os.path.join(ROOT, "examples", "p13_loop")
IDS = [3, 117, 2501, 88]


def build(layers, out, loop):
    cmd = [sys.executable, os.path.join(ROOT, "tools", "p12_build_minicpm.py"),
           "--layers", str(layers), "--seq", "4", "--skip-head",
           "--print-hidden", "--prompt-mode", "tokens", "--out", out]
    if loop:
        cmd.append("--loop")
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                       errors="replace")
    if p.returncode != 0:
        print("BUILD FAILED (loop=%s)\n%s" % (loop, (p.stdout + p.stderr)[-1200:]))
        return False
    stride = None
    for line in p.stdout.splitlines():
        if "stride" in line:
            stride = line.strip()
    if stride:
        print("  " + stride)
    consts = [l.strip() for l in open(os.path.join(out, "weights.args"),
                                      encoding="utf-8") if l.strip()]
    q = subprocess.run([ASM, "full.vml", "m.vmb"] + consts, cwd=out,
                       capture_output=True, text=True, errors="replace")
    if q.returncode != 0:
        print("ASSEMBLE FAILED\n%s" % (q.stdout + q.stderr)[-1200:])
        return False
    return True


def run(out):
    import shutil
    shutil.copyfile(os.path.join(out, "m.vmb"), os.path.join(REL, "model.vmb"))
    with open(os.path.join(REL, "tokens.bin"), "wb") as f:
        np.array(IDS, dtype="<u4").tofile(f)
    gt = os.path.join(REL, "gen_tokens.bin")
    if os.path.exists(gt):
        os.remove(gt)
    res = os.path.join(out, "_out.txt")
    with open(res, "wb") as f:
        p = subprocess.run([ENG], cwd=REL, stdout=f, stderr=subprocess.DEVNULL)
    if p.returncode != 0:
        q = subprocess.run([ENG], cwd=REL, capture_output=True)
        return None, "%d: %s" % (p.returncode,
                                 (q.stdout + q.stderr).decode("utf-8", "replace").strip()[:120])
    v = []
    for line in open(res, "r", errors="replace"):
        for t in line.split():
            try:
                v.append(float(t))
            except ValueError:
                pass
    return np.array(v, dtype=np.float32), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--seq", type=int, default=4)
    args = ap.parse_args()
    NL = args.layers
    os.makedirs(WORK, exist_ok=True)

    print("=== P13 task E: looped vs unrolled layer template (%d layers) ===\n"
          % NL)

    print("building UNROLLED (%d layer bodies)..." % NL)
    du = os.path.join(WORK, "unrolled")
    if not build(NL, du, loop=False):
        return 1
    print("building LOOPED (1 body + LOOP_CNT + LAYERB)...")
    dl = os.path.join(WORK, "looped")
    if not build(NL, dl, loop=True):
        return 1

    # Instruction counts: the whole point of the loop is that this is flat.
    def n_ins(d):
        txt = open(os.path.join(d, "full.vml"), encoding="utf-8").read()
        n = 0
        for line in txt.splitlines():
            s = line.split(";")[0].strip()
            if s:
                n += 1
        return n

    nu, nl = n_ins(du), n_ins(dl)
    print("\ninstruction count: unrolled=%d  looped=%d  (%.1fx fewer)"
          % (nu, nl, nu / max(nl, 1)))

    u, err = run(du)
    if u is None:
        print("unrolled engine failed: %s" % err)
        return 1
    l, err = run(dl)
    if l is None:
        print("looped engine failed: %s" % err)
        return 1
    print("output: unrolled=%d values  looped=%d values" % (u.size, l.size))
    if u.size != l.size:
        print("SIZE MISMATCH")
        return 1

    d = np.abs(u - l)
    scale = max(float(np.abs(u).max()), 1e-30)
    print("\nmax|d| = %.4e   rel = %.4e" % (d.max(), d.max() / scale))
    print("first 4: unrolled %s" % u[:4].tolist())
    print("         looped   %s" % l[:4].tolist())

    # The two builds run the SAME arithmetic in the SAME order, so the only
    # difference can be printed rounding. Anything larger means OP_LAYERB put a
    # layer's weights at the wrong byte offset.
    ok = d.max() <= 1e-5
    print()
    if ok:
        print("VERDICT: PASS -- the looped template reproduces the unrolled")
        print("         build exactly (within 6-decimal print precision), so")
        print("         the runtime base lands on the right bytes for every layer.")
        return 0
    print("VERDICT: FAIL -- worst |d| %.3e: OP_LAYERB is reading the wrong" % d.max())
    print("         layer's weights.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
