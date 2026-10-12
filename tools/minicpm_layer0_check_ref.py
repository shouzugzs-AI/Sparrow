#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P10 B2 step 2: validate the numpy re-implementation of MiniCPM layer 0
against the HuggingFace capture -- *before* touching the engine.

WHY THIS STEP EXISTS (methodology, from white paper 15.6)
--------------------------------------------------------
If we go straight to "engine vs HF", a mismatch has two possible causes:
the ENGINE is wrong, or MY UNDERSTANDING of the layer is wrong. Those need
very different fixes, and mixing them cost this project real time before
(P9: "measurement scripts are themselves buggy -- twice").

So: first prove a pure-numpy layer 0 reproduces the HF capture stage by
stage. Whatever error remains after THAT is genuinely the engine's.
The numpy path here is written to mirror exactly the ops the .vml will use
(GEMM / RMSNORM / RoPE / softmax / silu), so agreement also proves the
.op-level decomposition is right.

Usage: python tools/minicpm_layer0_check_ref.py
"""

import json
import os
import sys

import numpy as np

REF = r"<repo>/examples/minicpm/layer0_ref"


def L(name):
    return np.load(os.path.join(REF, name + ".npy")).astype(np.float32)


def report(tag, got, want):
    got = np.asarray(got, dtype=np.float32)
    want = np.asarray(want, dtype=np.float32)
    if got.shape != want.shape:
        print("  %-22s SHAPE MISMATCH got %s want %s" % (tag, got.shape, want.shape))
        return None
    d = np.abs(got - want)
    denom = max(float(np.abs(want).max()), 1e-30)
    rel = float(d.max()) / denom
    print("  %-22s absmax=%.4e  refmax=%.4e  rel=%.3e  %s"
          % (tag, d.max(), denom, rel, "OK" if rel < 2e-5 else "<-- CHECK"))
    return rel


def rmsnorm(x, g, eps):
    """Mirror of kernels::rmsnorm: no mean subtraction, no bias."""
    ss = (x.astype(np.float64) ** 2).mean(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(ss + eps)
    return (x * inv * g).astype(np.float32)


def linear(x, w):
    """y = x @ w.T  (nn.Linear)"""
    return (x @ w.T).astype(np.float32)


def silu(x):
    return (x / (1.0 + np.exp(-x.astype(np.float64)))).astype(np.float32)


def apply_rope(t, cos, sin):
    """
    t: [b, h, s, hd], cos/sin: [s, hd]

    Llama/transformers 'rotate_half' convention:
        rotate_half(x) = cat(-x[..., hd/2:], x[..., :hd/2])
        out = x*cos + rotate_half(x)*sin
    which is EQUIVALENT to the paired form below (x1*cos - x2*sin,
    x2*cos + x1*sin) for the first/second halves. We use the explicit
    half-split form because that is what the .vml will implement.
    """
    hd = t.shape[-1]
    t1 = t[..., : hd // 2]
    t2 = t[..., hd // 2:]
    c1 = cos[:, : hd // 2]
    s1 = sin[:, : hd // 2]
    o1 = t1 * c1 - t2 * s1
    o2 = t1 * s1 + t2 * c1
    return np.concatenate([o1, o2], axis=-1).astype(np.float32)


def main():
    meta = json.load(open(os.path.join(REF, "meta.json"), encoding="utf-8"))
    cfg = meta["config"]
    H = cfg["hidden_size"]
    nh, nkv, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    eps = cfg["rms_norm_eps"]
    I = cfg["intermediate_size"]
    print("layer0 reference check: H=%d nh=%d nkv=%d hd=%d I=%d eps=%g"
          % (H, nh, nkv, hd, I, eps))

    h_in = L("hidden_in")          # [1,4,1536]
    b, s, _ = h_in.shape
    print("hidden_in %s   (seq=%d)" % (h_in.shape, s))

    # ---------------------------------------------------- stage 1: input norm
    print("\n[1] RMSNorm (input_layernorm)")
    g_in = L("input_layernorm_weight")
    x = rmsnorm(h_in, g_in, eps)
    report("in_ln", x, L("in_ln_out"))

    # -------------------------------------------------------- stage 2: qkv
    print("[2] q/k/v projections (GQA: %d kv heads)" % nkv)
    Wq = L("self_attn_q_proj_weight")   # [2048,1536]
    Wk = L("self_attn_k_proj_weight")   # [256,1536]
    Wv = L("self_attn_v_proj_weight")   # [256,1536]
    Wo = L("self_attn_o_proj_weight")   # [1536,2048]
    q = linear(x, Wq)
    k = linear(x, Wk)
    v = linear(x, Wv)
    report("q_proj", q, L("q_proj_out"))
    report("k_proj", k, L("k_proj_out"))
    report("v_proj", v, L("v_proj_out"))

    # --------------------------------------------------------- stage 3: RoPE
    print("[3] RoPE (theta=%s, head_dim=%d)" % (cfg["rope_theta"], hd))
    cos, sin = L("rope_cos"), L("rope_sin")
    qh = q.reshape(b, s, nh, hd).transpose(0, 2, 1, 3)
    kh = k.reshape(b, s, nkv, hd).transpose(0, 2, 1, 3)
    vh = v.reshape(b, s, nkv, hd).transpose(0, 2, 1, 3)
    q_r = apply_rope(qh, cos, sin)
    k_r = apply_rope(kh, cos, sin)
    report("q_roped", q_r, L("q_roped"))
    report("k_roped", k_r, L("k_roped"))

    # ------------------------------------------------ stage 4: GQA attention
    print("[4] GQA attention (repeat_kv %d->%d)" % (nkv, nh))
    rep = nh // nkv
    k_e = np.repeat(k_r, rep, axis=1)
    v_e = np.repeat(vh, rep, axis=1)
    sc = (q_r @ k_e.transpose(0, 1, 3, 2)) / np.sqrt(hd).astype(np.float32)
    # causal mask
    mask = np.triu(np.ones((s, s), dtype=np.float32), 1) * np.float32(-1e30)
    sc = (sc + mask).astype(np.float32)
    mx = sc.max(axis=-1, keepdims=True)
    ex = np.exp((sc - mx).astype(np.float64)).astype(np.float32)
    att = (ex / ex.sum(axis=-1, keepdims=True)).astype(np.float32)
    report("attn_weights", att, L("attn_weights"))
    ctx = (att @ v_e).astype(np.float32)          # [b,nh,s,hd]
    report("attn_ctx", ctx, L("attn_ctx"))
    ctx_cat = ctx.transpose(0, 2, 1, 3).reshape(b, s, nh * hd)
    report("attn_ctx_cat", ctx_cat, L("attn_ctx_cat"))

    # ------------------------------------------------------ stage 5: o_proj
    print("[5] o_proj + residual")
    o = linear(ctx_cat, Wo)
    report("o_proj", o, L("o_proj_out"))
    h1 = (h_in + o).astype(np.float32)

    # --------------------------------------------- stage 6: post norm + MLP
    print("[6] post_attention_layernorm + SwiGLU MLP")
    g_post = L("post_attention_layernorm_weight")
    x2 = rmsnorm(h1, g_post, eps)
    report("post_ln", x2, L("post_ln_out"))

    Wg = L("mlp_gate_proj_weight")   # [4608,1536]
    Wu = L("mlp_up_proj_weight")     # [4608,1536]
    Wd = L("mlp_down_proj_weight")   # [1536,4608]
    gate = linear(x2, Wg)
    up = linear(x2, Wu)
    report("gate_proj", gate, L("gate_proj_out"))
    report("up_proj", up, L("up_proj_out"))
    act = (silu(gate) * up).astype(np.float32)
    report("mlp_act (silu*up)", act, L("mlp_act"))
    down = linear(act, Wd)
    report("down_proj", down, L("down_proj_out"))

    h_out = (h1 + down).astype(np.float32)

    print("\n[7] layer 0 output (end to end)")
    rel = report("hidden_out", h_out, L("hidden_out"))

    print()
    if rel is not None and rel < 2e-5:
        print("RESULT: numpy layer 0 reproduces HuggingFace. The decomposition")
        print("        (RMSNorm / GEMM / RoPE / GQA / softmax / SiLU) is CORRECT.")
        print("        => any engine mismatch from here is an ENGINE issue.")
        return 0
    print("RESULT: numpy layer 0 does NOT match HF -- fix the oracle/"
          "understanding BEFORE blaming the engine.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
