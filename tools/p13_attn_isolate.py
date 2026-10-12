#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P13 task C: measure the attention term in ISOLATION, not by inference.

WHY THIS EXISTS
    The scalar timing curve (tools/p13_attn_scaling.py) cannot separate the
    attention cost from everything else, for two measured reasons:
      1. A fixed ~0.20 s startup (mmap + page-fault of the 2.16 GB artifact)
         dominates at small S.
      2. The projection GEMMs switch kernel on shape: m<16 takes gemm_nt_avx2,
         m>=16 takes the packed path. So S=4 and S=64 are not running the same
         code, and a fitted exponent over them is meaningless.

WHAT THIS DOES INSTEAD
    Builds the SAME program twice, differing ONLY in whether the attention
    score/context GEMMs are present, and times the difference. That is a
    controlled comparison: identical weights, identical everything else.

    Concretely, `p12_build_minicpm.py` emits per-head
        GEMM sc, qrot, krot        <- [S,S]  (S^2)
        SOFTMAX sc, sc, causal
        GEMM ctx, sc, vh, nn       <- [S,S]  (S^2)
    This script patches the generated .vml to replace those with a single
    zero-cost line, so the layer stack still runs but the S^2 terms are gone.

WHAT TO CONCLUDE
    If (full - no_attention) is a small fraction of the layer time at S=64,
    then a KV cache -- which only removes the S^2 part -- cannot be the
    headline win, and the honest recommendation is to say so.

Usage: python tools/p13_attn_isolate.py --layers 4 --seqs 16,64,256
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
REL = os.path.join(ROOT, "target", "release")
ENG = os.path.join(REL, "vmlm-runtime.exe")
ASM = os.path.join(REL, "vmlm-asm.exe")
WORK = os.path.join(ROOT, "examples", "p13_attn")


def build(seq, out):
    p = subprocess.run(
        [sys.executable, os.path.join(ROOT, "tools", "p12_build_minicpm.py"),
         "--layers", str(LAYERS), "--seq", str(seq), "--skip-head",
         "--print-hidden", "--prompt-mode", "tokens", "--out", out],
        cwd=ROOT, capture_output=True, text=True, errors="replace")
    if p.returncode != 0:
        print("BUILD FAILED")
        print((p.stdout + p.stderr)[-1200:])
        return False
    return True


ATTN_RE = re.compile(
    r"^(GEMM sc,.*|SCALE sc,.*|SOFTMAX sc,.*|GEMM ctx,.*|WRITECOL att,.*)$",
    re.MULTILINE)


def strip_attention(vml_text, seq):
    """Remove the S^2 attention ops, keeping the rest of the layer intact.

    `att` must still be a defined [S, NH*HD] activation or the o_proj GEMM that
    consumes it has nothing to read -- so instead of deleting outright, the
    score/context GEMMs are replaced by a zero-fill of `ctx` and one WRITECOL
    of that zero into each head slot. The o_proj then runs on zeros. Its cost
    is unchanged (same shape), which is what makes this a clean subtraction:
    only the S^2 work disappears.
    """
    lines = vml_text.split("\n")
    out = []
    i = 0
    removed = 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("GEMM sc,") or s.startswith("SCALE sc,") or \
           s.startswith("SOFTMAX sc,"):
            removed += 1
            # Keep the FIRST occurrence as a zero-fill so ctx is initialised.
            #
            # NOTE: `ZERO ctx` does NOT work here -- OP_ZERO (66) lives in the
            # train-only group in interp.rs, so the INFERENCE build exits 6
            # ("train feature disabled"). Measured. Use EWMUL with a zero
            # scalar instead: inference-safe, same effect.
            if removed == 1:
                out.append("; P13: zero-fill ctx instead of the S^2 attention")
                out.append("SHAPE zero1, 1, 1")
                out.append("EWMUL ctx, ctx, zero1")
            i += 1
            continue
        if s.startswith("GEMM ctx,"):
            removed += 1
            i += 1
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out), removed


def assemble(d, vml, vmb, with_zero=False):
    consts = [l.strip() for l in open(os.path.join(d, "weights.args"),
                                      encoding="utf-8") if l.strip()]
    if with_zero:
        # EWMUL needs a real constant slot holding 0.0 (the assembler requires
        # every referenced name to resolve; a SHAPE-only slot has no contents).
        z = os.path.join(d, "zero1.f32")
        np.zeros(1, dtype=np.float32).tofile(z)
        consts = consts + ["zero1:zero1.f32:1:1:f32"]
    p = subprocess.run([ASM, vml, vmb] + consts, cwd=d,
                       capture_output=True, text=True, errors="replace")
    if p.returncode != 0:
        print("  assemble failed: %s" % (p.stdout + p.stderr)[-400:])
        return False
    return True


def run_timed(vmb, seq, reps=3):
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
            return None
        best = dt if best is None else min(best, dt)
    return best


LAYERS = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--seqs", default="16,64,256")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    global LAYERS
    LAYERS = args.layers

    os.makedirs(WORK, exist_ok=True)
    print("=== P13 task C: isolate the S^2 attention term (%d layers) ===\n"
          % LAYERS)
    print("%5s %11s %11s %11s %9s" %
          ("S", "full", "no-attn", "attn cost", "share"))
    print("-" * 54)
    for seq in [int(x) for x in args.seqs.split(",")]:
        d = os.path.join(WORK, "s%d" % seq)
        if not build(seq, d):
            return 1
        vml = open(os.path.join(d, "full.vml"), encoding="utf-8").read()
        if "ZERO ctx" in vml:
            print("  (already stripped)")
        stripped, n = strip_attention(vml, seq)
        with open(os.path.join(d, "noattn.vml"), "w", encoding="ascii",
                  newline="\n") as f:
            f.write(stripped)
        if n == 0:
            print("  S=%d: attention pattern not found -- nothing stripped" % seq)
            return 1
        if not assemble(d, "full.vml", "full.vmb"):
            return 1
        if not assemble(d, "noattn.vml", "noattn.vmb", with_zero=True):
            return 1
        tf = run_timed(os.path.join(d, "full.vmb"), seq, args.reps)
        tn = run_timed(os.path.join(d, "noattn.vmb"), seq, args.reps)
        if tf is None or tn is None:
            print("  S=%d: engine failed" % seq)
            return 1
        print("%5d %10.3fs %10.3fs %10.3fs %8.1f%%"
              % (seq, tf, tn, tf - tn, 100.0 * (tf - tn) / tf))
    print()
    print("attn cost = full - no_attention (same weights, same shapes elsewhere).")
    print("If the share stays small, the KV cache is NOT the headline win here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
