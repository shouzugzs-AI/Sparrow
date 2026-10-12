#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P16-4 (task-10) STEP 4: WHY DOES THE 4-LAYER HF TRUNCATION NOT MATCH P14's ANCHOR?

MEASURED PROBLEM
    HF truncated to 4 layers gives prompt argmax [791, 29885, 104, 91794].
    The P12/P13/P14 record -- and HF at the FULL 24 layers on the same prompt --
    gives [5, 242, 90, 82591].

    So either:
      (a) truncating cfg.num_hidden_layers in transformers 5.14.1 does NOT give
          the same model as the VML builder's `layers 0..N-1` truncation, or
      (b) the 4-layer artifacts are built from DIFFERENT weights than layers 0..3.

    This decides whether my 4-layer HF oracle is a valid oracle at all. If the
    layer subsets differ, the generation-position agreement I measured in step 1
    is still meaningful (it was done on the SAME sequence), but the PREFILL
    comparison is not comparable and must be marked UNPROVEN rather than FAIL.

METHOD (no assertion -- measure the actual tensors)
    1. Load the safetensors directly and print the per-layer weight fingerprints
       (a cheap checksum over a slice of each layer's q_proj).
    2. Load HF truncated to 4 and print its weight fingerprints for the same
       tensors. If layer 0's fingerprint matches, the truncation keeps the
       correct early layers; then compare layer 3 as well.
    3. Load the ENGINE's bf16 weight files (L0_q.bf16 .. L3_q.bf16) and
       fingerprint them too. Now all three sources are directly comparable and
       the answer is a measurement, not an argument.

Usage:
    python tools/p16_o6_weight_provenance.py
"""
import hashlib
import json
import os
import struct
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = str(Path(__file__).resolve().parents[1])
MODEL = os.environ.get("MINICPM_DIR", "<外部模型目录>/MiniCPM5-1B")
ART = os.path.join(ROOT, "examples", "p14_gen")


def fp(a):
    """Stable fingerprint of a tensor: shape + first 8 floats + norm + md5."""
    a = np.asarray(a, dtype=np.float32).ravel()
    h = hashlib.md5(a[:4096].tobytes()).hexdigest()[:16]
    return {
        "shape": list(np.asarray(a).shape) if a.ndim else [int(a.size)],
        "first8": [round(float(x), 6) for x in a[:8]],
        "norm": round(float(np.linalg.norm(a.astype(np.float64))), 4),
        "md5_first4k": h,
    }


def read_bf16(path):
    """Read a raw bf16 file as float32 (bf16 -> f32 is a 16-bit left shift)."""
    raw = np.fromfile(path, dtype=np.uint16)
    u32 = raw.astype(np.uint32) << 16
    return u32.view(np.float32)


def main():
    print("=== P16-4 STEP 4: weight provenance of the 4-layer artifact ===\n")

    # ---------- 1. safetensors ground truth ----------
    st_path = os.path.join(MODEL, "model-00000-of-00001.safetensors")
    print("[1] safetensors ground truth: %s" % st_path)
    with open(st_path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n).decode("utf-8"))
    base = 8 + n
    print("    header entries: %d" % len(hdr))

    def st_tensor(name):
        e = hdr[name]
        o0, o1 = e["data_offsets"]
        dt = e["dtype"]
        npdt = {"BF16": np.uint16, "F32": np.float32, "F16": np.float16}[dt]
        with open(st_path, "rb") as f:
            f.seek(base + o0)
            raw = f.read(o1 - o0)
        a = np.frombuffer(raw, dtype=npdt)
        if dt == "BF16":
            a = (a.astype(np.uint32) << 16).view(np.float32)
        return a.astype(np.float32)

    st_names = [k for k in hdr if k.startswith("model.layers.")]
    layers_present = sorted({int(k.split(".")[2]) for k in st_names})
    print("    layer indices present: min=%d max=%d count=%d"
          % (min(layers_present), max(layers_present), len(layers_present)))

    st_fp = {}
    for L in range(4):
        nm = "model.layers.%d.self_attn.q_proj.weight" % L
        if nm in hdr:
            st_fp[L] = fp(st_tensor(nm))
    print("    safetensors q_proj fingerprints (layers 0..3):")
    for L, v in st_fp.items():
        print("      L%d shape=%s norm=%.4f md5=%s" % (L, v["shape"], v["norm"], v["md5_first4k"]))

    # ---------- 2. HF truncated to 4 ----------
    print("\n[2] HF (transformers) truncated to 4 layers")
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(MODEL)
    cfg.num_hidden_layers = 4
    cfg.use_cache = False
    m = AutoModelForCausalLM.from_pretrained(MODEL, config=cfg,
                                             torch_dtype=torch.float32,
                                             low_cpu_mem_usage=True)
    m.eval()
    print("    model.config.num_hidden_layers = %d" % m.config.num_hidden_layers)
    print("    len(model.model.layers)        = %d" % len(m.model.layers))
    found = [n for n, _ in m.named_parameters() if "layers." in n]
    li = sorted({int(n.split("layers.")[1].split(".")[0]) for n in found})
    print("    layer indices in HF model      = %s (min=%d max=%d)"
          % (li, min(li), max(li)))
    print("    HF total parameter tensors     = %d" % len(list(m.named_parameters())))

    hf_fp = {}
    for L in range(4):
        w = None
        for n, p in m.named_parameters():
            if n == "model.layers.%d.self_attn.q_proj.weight" % L:
                w = p.detach().numpy().astype(np.float32)
                break
        if w is not None:
            hf_fp[L] = fp(w)
    print("    HF q_proj fingerprints (layers 0..3):")
    for L, v in hf_fp.items():
        print("      L%d shape=%s norm=%.4f md5=%s" % (L, v["shape"], v["norm"], v["md5_first4k"]))

    # ---------- 3. engine bf16 files ----------
    print("\n[3] engine artifact weight files: %s" % ART)
    eng_fp = {}
    for L in range(4):
        p = os.path.join(ART, "weights", "L%d_q.bf16" % L)
        if os.path.exists(p):
            a = read_bf16(p)
            eng_fp[L] = fp(a)
    print("    engine L*_q.bf16 fingerprints (layers 0..3):")
    for L, v in eng_fp.items():
        print("      L%d elems=%d first8=%s" % (L, v["shape"][0], v["first8"]))

    # engine files are [1536 (out), 2048 (in)] flattened row-major; the HF tensor
    # is [2048, 1536]. Same numbers, transposed layout -- compare SETS of norms
    # only, and print enough raw values for a human check.
    print("\n[4] DIRECT VALUE COMPARISON (engine bf16 vs safetensors, layer 0)")
    p0 = os.path.join(ART, "weights", "L0_q.bf16")
    if os.path.exists(p0):
        e0 = read_bf16(p0)
        s0 = st_tensor("model.layers.0.self_attn.q_proj.weight")
        print("    engine L0_q   size=%d  first12=%s" % (e0.size, np.round(e0[:12], 6).tolist()))
        print("    st     L0_q   size=%d  first12=%s" % (s0.size, np.round(s0[:12], 6).tolist()))
        # The VML builder writes q as [out=2048, in=1536]? check both orders.
        for label, arr in [("engine-as-is", e0), ("engine-transposed", e0.reshape(-1, 1536).T.ravel())]:
            if arr.size == s0.size:
                d = float(np.max(np.abs(arr - s0)))
                print("    max|engine - st| (%s) = %.6e" % (label, d))
        # also compare sorted multisets (layout-agnostic)
        if e0.size == s0.size:
            print("    sorted-multiset max|diff| = %.6e"
                  % float(np.max(np.abs(np.sort(e0) - np.sort(s0)))))

    out = {
        "safetensors_layers": [min(layers_present), max(layers_present), len(layers_present)],
        "safetensors_fp": {str(k): v for k, v in st_fp.items()},
        "hf_fp": {str(k): v for k, v in hf_fp.items()},
        "engine_fp": {str(k): v for k, v in eng_fp.items()},
        "hf_num_hidden_layers_effective": len(m.model.layers),
        "hf_param_tensor_count": len(list(m.named_parameters())),
    }
    p = os.path.join(ROOT, "examples", "p16_o6", "weight_provenance.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nwrote %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
