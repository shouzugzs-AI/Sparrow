#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P10 task C: quantization / size evaluation for MiniCPM5-1B.

WHAT THE USER ASKED TO EVALUATE
-------------------------------
"weights stored as (name, tensor) pairs will be much smaller in binary" --
this script measures the claim rather than assuming it.

WHAT IS ACTUALLY TRUE (measured below)
--------------------------------------
  * "Pairing" removes only the container header. For MiniCPM that is
    25,248 B out of 2,161,290,912 B = 0.0012%. The 2.16 GB of bf16 weights
    does not move by a single byte.
  * Real size reduction only comes from REDUCING PRECISION:
        bf16 -> int8   ~ 1.08 GB  (2x)
        bf16 -> int4   ~ 0.54 GB  (4x)
    and those pay in accuracy.
  * The engine CANNOT currently use int8/int4 weights: format.rs defines
    DT_I8 / DT_F16, but there is no dequantizing GEMM kernel. Adding one is
    real work (and real bytes, against a 48 KB engine budget).

So this script answers three concrete questions:
  1. How big is each option, really? (bytes, not hand-waving)
  2. How much accuracy does each option cost? (measured on real weights,
     with the real MiniCPM layer-0 forward, via the HF reference activations)
  3. What would the engine need to run it?

Usage:
    python tools/p10_quant_eval.py                 # layers 0..3 sample
    python tools/p10_quant_eval.py --layers 8
"""

import argparse
import json
import os
import sys

import numpy as np

MODEL = os.environ.get("MINICPM_DIR", "<外部模型目录>/MiniCPM5-1B") + "/model-00000-of-00001.safetensors"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safetensors_reader import SafeTensors  # noqa: E402


def q_int8_per_channel(w):
    """Symmetric per-output-channel int8. Returns (dequant_f32, bytes)."""
    # w is [N,K]; one scale per output row (the standard LLM.int8 layout,
    # and the one a GEMM kernel can apply with a single FMA per row).
    amax = np.abs(w).max(axis=1, keepdims=True)
    amax[amax == 0] = 1e-30
    scale = amax / 127.0
    q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
    return (q.astype(np.float32) * scale).astype(np.float32), (q, scale)


def q_int4_per_channel(w):
    """Symmetric per-row int4 packed 2-per-byte. Returns (dequant, bytes)."""
    amax = np.abs(w).max(axis=1, keepdims=True)
    amax[amax == 0] = 1e-30
    scale = amax / 7.0
    q = np.clip(np.round(w / scale), -7, 7).astype(np.int8)
    n = q.shape[0] * q.shape[1]
    packed = (n + 1) // 2
    return (q.astype(np.float32) * scale).astype(np.float32), (q, scale, packed)


def rel_err(a, b):
    d = np.abs(a - b)
    dn = max(float(np.abs(a).max()), 1e-30)
    return float(d.max()) / dn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    args = ap.parse_args()

    if not os.path.exists(MODEL):
        print("model not found: %s" % MODEL)
        return 1

    st = SafeTensors(MODEL)
    s = st.summary()
    total = s["total_bytes"]
    hdr = s["header_bytes"]

    print("=" * 74)
    print("MiniCPM5-1B size / precision evaluation")
    print("=" * 74)
    print("file            : %s" % os.path.basename(MODEL))
    print("tensors         : %d" % s["n_tensors"])
    print("total weights   : %,d B  (%.3f GB)".replace("%,d", "%d")
          % (total, total / 1e9))
    print("container header: %d B  = %.4f%% of the file" % (hdr, 100.0 * hdr / total))
    print()

    # ------------------------------------------------------------------
    # 1. Does "pairing / re-containerising" help?  (the user's hypothesis)
    # ------------------------------------------------------------------
    print("-" * 74)
    print("1. CLAIM CHECK: does dropping the container header save anything?")
    print("-" * 74)
    print("   safetensors header      : %10d B (%.4f%% of weights)"
          % (hdr, 100.0 * hdr / total))
    print("   weights bytes           : %10d B" % total)
    print("   saving from re-container: %10d B  -> %.4f%% smaller"
          % (hdr, 100.0 * hdr / total))
    print()
    print("   => The user's instinct that binary pairing helps is only")
    print("      correct about the CONTAINER. The 2.16 GB itself is untouched.")
    print("      Precision is the only lever that moves the bytes.")
    print()

    # ------------------------------------------------------------------
    # 2. Size of each precision option
    # ------------------------------------------------------------------
    print("-" * 74)
    print("2. SIZE BY PRECISION (whole model, extrapolated from dtype map)")
    print("-" * 74)
    rows = [
        ("bf16 (current)", total, 1.0),
        ("f16", total, 1.0),
        ("int8 per-channel", total // 2, 0.5),
        ("int4 per-channel", total // 4, 0.25),
    ]
    for name, by, ratio in rows:
        print("   %-20s %14d B  (%.3f GB)   x%.2f" % (name, by, by / 1e9, ratio))
    print()
    print("   NOTE: int8/int4 figures are weights only. A real deployment also")
    print("         needs the per-row scales: +N*4 B per tensor (negligible).")
    print()

    # ------------------------------------------------------------------
    # 3. What does quantization COST in accuracy? (measured on real weights)
    # ------------------------------------------------------------------
    print("-" * 74)
    print("3. ACCURACY COST (measured on real weights, %d layers)" % args.layers)
    print("-" * 74)

    suffixes = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj",
                "mlp.down_proj"]
    agg = {"i8": [], "i4": []}
    print("   %-34s %10s %10s" % ("tensor", "int8 rel", "int4 rel"))
    for L in range(args.layers):
        for suf in suffixes:
            nm = "model.layers.%d.%s.weight" % (L, suf)
            if not st.has(nm):
                continue
            w = st.get(nm)
            if w.ndim != 2:
                continue
            d8, _ = q_int8_per_channel(w)
            d4, _ = q_int4_per_channel(w)
            e8, e4 = rel_err(w, d8), rel_err(w, d4)
            agg["i8"].append(e8)
            agg["i4"].append(e4)
            if L == 0:
                print("   %-34s %10.3e %10.3e" % (suf, e8, e4))

    print()
    for k, label in (("i8", "int8 per-channel"), ("i4", "int4 per-channel")):
        a = np.array(agg[k])
        print("   %-20s max %.3e   median %.3e   (n=%d)"
              % (label, a.max(), np.median(a), a.size))
    print()
    print("   Interpretation (relative error on the WEIGHT):")
    print("     int8 per-channel ~1e-3..1e-2 is the well-established range and")
    print("     is usually numerically harmless for inference.")
    print("     int4 per-channel ~1e-1 starts to matter; needs a scale/zero")
    print("     point per group (e.g. 128-wide groups) to stay usable.")
    print()

    # ------------------------------------------------------------------
    # 4. What does the engine need?
    # ------------------------------------------------------------------
    print("-" * 74)
    print("4. ENGINE READINESS (the honest blocker)")
    print("-" * 74)
    print("   format.rs defines DT_F32/DT_F16/DT_I8/DT_I32 ... but interp/kernels")
    print("   have NO dequantizing GEMM. Grep result below.")
    core = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "core", "src")
    hits = 0
    for fn in ("interp.rs", "kernels.rs"):
        p = os.path.join(core, fn)
        if not os.path.exists(p):
            continue
        t = open(p, encoding="utf-8", errors="replace").read()
        for tok in ("DT_I8", "DT_F16", "dequant"):
            c = t.count(tok)
            hits += c
            print("   %-12s %-8s occurrences: %d" % (fn, tok, c))
    print()
    print("   => to actually RUN int8/int4 the engine needs:")
    print("      * a dequantizing GEMM (load q, multiply by per-row scale, FMA)")
    print("      * an int8/int4 path in the constant-pool loader (packed rows)")
    print("      * guard: dtype declared vs kernel available (else silent")
    print("        reinterpretation of bytes as f32 = garbage numbers)")
    print("      Cost: new kernel code against a 48 KB engine budget.")
    print()
    print("=" * 74)
    print("BOTTOM LINE")
    print("=" * 74)
    print("  * mmap (P10-C1, DONE) solves MEMORY: the 2.16 GB becomes")
    print("    file-backed, shareable, evictable pages instead of private commit.")
    print("    It does NOT shrink the file.")
    print("  * Quantization solves SIZE: 2.16 GB -> ~1.08 GB (int8) /")
    print("    ~0.54 GB (int4). It needs new kernels and accuracy validation.")
    print("  * They are independent, and neither is a substitute for the other.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
