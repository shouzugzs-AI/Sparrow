#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P14 / O6 REFERENCE BUILDER: one-shot full causal forward over N positions.

WHY THIS IS A SEPARATE PROGRAM FROM THE LOOP
    The O6 headline claim is that the closed loop's 16-token chain equals what a
    ONE-SHOT FULL FORWARD produces over the same positions. To test that, the
    reference must be a genuinely different program SHAPE:

      loop      : one token at a time, KV cache, attention over the live range
      one-shot  : all N positions in a single pass, full [N,N] causal attention

    They share weights and positions but not summation structure. That is the
    point: if both produce the same argmax at positions S..S+G-1, the incremental
    cache + in-engine argmax feedback has not changed the answer.

    This is deliberately NOT derived from the loop emitter -- it is the plain
    P12/P13-style full forward, which is already independently validated against
    HuggingFace (P13 measured 4/4 argmax and rel 1.88e-06). Reusing the validated
    path as the reference is stronger than writing a second cache-based emitter.

DETERMINISM NOTE
    The layer body emitted here is the SAME instruction sequence the closed
    loop's per-step body uses (RMSNORM/GEMM/ROPE/SOFTMAX/GEMM/SILU/EWMUL), with
    the same operand order, so the two programs exercise the same kernels in the
    same order. Any residual difference is therefore attributable to the
    attention structure (full causal vs live-range slice), which is exactly what
    the comparison is meant to expose.

Usage:
    python tools/p14_oneshot_build.py --dir examples/p14_gen --out examples/p14_oneshot
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safetensors_reader import SafeTensors  # noqa: E402

MODEL = os.environ.get("MINICPM_DIR", "<外部模型目录>/MiniCPM5-1B")

LAYER_SUFFIXES = [
    ("in_ln", "input_layernorm.weight", "f32"),
    ("q", "self_attn.q_proj.weight", "bf16"),
    ("k", "self_attn.k_proj.weight", "bf16"),
    ("v", "self_attn.v_proj.weight", "bf16"),
    ("o", "self_attn.o_proj.weight", "bf16"),
    ("post_ln", "post_attention_layernorm.weight", "f32"),
    ("gate", "mlp.gate_proj.weight", "bf16"),
    ("up", "mlp.up_proj.weight", "bf16"),
    ("down", "mlp.down_proj.weight", "bf16"),
]


def f32_to_bf16(a: np.ndarray) -> np.ndarray:
    u = a.astype(np.float32).view(np.uint32).astype(np.uint64)
    rnd = np.uint64(0x7FFF) + ((u >> 16) & 1)
    return ((u + rnd) >> 16).astype("<u2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="examples/p14_gen",
                    help="the LOOP artifact dir: supplies the token chain to run "
                         "the full forward over, and the layer count")
    ap.add_argument("--out", default="examples/p14_oneshot")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--tokens", default="",
                    help="explicit comma-separated ids; default = prompt + the "
                         "closed loop's generated chain from o6_headline.json")
    args = ap.parse_args()

    ROOT = str(Path(__file__).resolve().parents[1])
    ldir = args.dir if os.path.isabs(args.dir) else os.path.join(ROOT, args.dir)
    odir = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    lmeta = json.load(open(os.path.join(ldir, "meta.json")))
    NL = lmeta["layers"]
    S = lmeta["seq"]
    G = lmeta["gen"]

    # The token sequence to run the full forward over: the prompt, then the
    # closed loop's own generated tokens. NOTE the closed loop's output is
    # chain[k] = argmax at position S+k; feeding them back as the INPUT at
    # position S+k+1 is exactly what the loop did, so this reproduces the same
    # positions.
    if args.tokens:
        toks = [int(t) for t in args.tokens.split(",") if t.strip()]
    else:
        hpath = os.path.join(ldir, "o6_headline.json")
        cpath = os.path.join(ldir, "_chain.json")
        if os.path.exists(cpath):
            chain = json.load(open(cpath))["chain"]
        elif os.path.exists(hpath):
            chain = json.load(open(hpath))["closed_loop_chain"]
        else:
            print("need %s (from p14_gen_check) to know the chain" % cpath)
            return 1
        # inputs = prompt + seed ... then the generated chain drives positions
        # S+1.. ; position S's input is the SEED.
        toks = list(lmeta["prompt_ids"]) + [lmeta["seed_token"]] + list(chain)
        # N positions total: prompt (S) + generated (G). The input at position
        # S+k is the seed for k=0 and chain[k-1] after. So the id sequence is:
        toks = list(lmeta["prompt_ids"]) + [lmeta["seed_token"]] + list(chain)[:G - 1]

    N = len(toks)
    print("one-shot full forward: N=%d tokens, layers=%d" % (N, NL))
    print("  tokens = %s" % toks)

    st = SafeTensors(os.path.join(args.model, "model-00000-of-00001.safetensors"))
    with open(os.path.join(args.model, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    H = cfg["hidden_size"]
    NH = cfg["num_attention_heads"]
    NKV = cfg["num_key_value_heads"]
    HD = cfg["head_dim"]
    INTER = cfg["intermediate_size"]
    V = cfg["vocab_size"]
    EPS = cfg["rms_norm_eps"]
    THETA = cfg["rope_theta"]
    rep = NH // NKV

    wdir = os.path.join(odir, "weights")
    os.makedirs(wdir, exist_ok=True)
    consts = []

    def emit_const(name, arr, dtype_tag):
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        r, c = arr.shape
        ext = {"bf16": "bf16", "i8": "i8"}.get(dtype_tag, "f32")
        p = os.path.join(wdir, name + "." + ext)
        if dtype_tag == "bf16":
            f32_to_bf16(arr).tofile(p)
        else:
            np.ascontiguousarray(arr, dtype=np.float32).tofile(p)
        consts.append((name, os.path.join("weights", name + "." + ext), r, c,
                       dtype_tag))
        return r, c

    def emit_scalar(name, val):
        p = os.path.join(wdir, name + ".f32")
        np.array([val], dtype=np.float32).tofile(p)
        consts.append((name, os.path.join("weights", name + ".f32"), 1, 1, "f32"))

    emit_const("embed", st.get("model.embed_tokens.weight"), "bf16")
    emit_const("lm_head", st.get("lm_head.weight"), "bf16")
    emit_const("final_norm", st.get("model.norm.weight"), "f32")
    emit_scalar("eps1", EPS)
    emit_scalar("theta1", THETA)
    emit_scalar("scale1", 1.0 / np.sqrt(HD))
    for L in range(NL):
        for short, suffix, dt in LAYER_SUFFIXES:
            nm = "model.layers.%d.%s" % (L, suffix)
            if not st.has(nm):
                print("missing tensor: %s" % nm)
                return 1
            emit_const("L%d_%s" % (L, short), st.get(nm), dt)

    A = []

    def w(s):
        A.append(s)

    w("; O6 REFERENCE: one-shot full causal forward over %d positions" % N)
    w("; generated by tools/p14_oneshot_build.py")
    w("; This is the program the closed loop is compared against: same weights,")
    w("; same positions, but a SINGLE pass with causal masking instead of an")
    w("; incremental KV cache. The argmax at positions %d..%d must match the"
      % (S, N - 1))
    w("; closed loop's chain if the incremental path is faithful.")
    w("")
    w("; ---- shapes ----")
    w("SHAPE st_tok, 1, %d" % N)
    w("SHAPE h,      %d, %d" % (N, H))
    w("SHAPE xn,     %d, %d" % (N, H))
    w("SHAPE q,      %d, %d" % (N, NH * HD))
    w("SHAPE k,      %d, %d" % (N, NKV * HD))
    w("SHAPE v,      %d, %d" % (N, NKV * HD))
    w("SHAPE qh,     %d, %d" % (N, HD))
    w("SHAPE kh,     %d, %d" % (N, HD))
    w("SHAPE vh,     %d, %d" % (N, HD))
    w("SHAPE qrot,   %d, %d" % (N, HD))
    w("SHAPE krot,   %d, %d" % (N, HD))
    w("SHAPE sc,     %d, %d" % (N, N))
    w("SHAPE ctx,    %d, %d" % (N, HD))
    w("SHAPE att,    %d, %d" % (N, NH * HD))
    w("SHAPE oproj,  %d, %d" % (N, H))
    w("SHAPE h2,     %d, %d" % (N, H))
    w("SHAPE gate,   %d, %d" % (N, INTER))
    w("SHAPE up,     %d, %d" % (N, INTER))
    w("SHAPE acts,   %d, %d" % (N, INTER))
    w("SHAPE down,   %d, %d" % (N, H))
    w("SHAPE logits, %d, %d" % (N, V))
    w("SHAPE eps1,   1, 1")
    w("SHAPE theta1, 1, 1")
    w("SHAPE scale1, 1, 1")
    w("")
    w("; ---- embedding ----")
    for i in range(N):
        w("SLICE tok%d, st_tok, %d, 1" % (i, i))
        w("SHAPE e%d, 1, %d" % (i, H))
        w("EMB e%d, embed, tok%d" % (i, i))
    w("SHAPE hcat0, 1, %d" % H)
    w("COPY hcat0, e0")
    prev, total = "hcat0", H
    for i in range(1, N):
        total += H
        cur = "hcat%d" % i
        w("SHAPE %s, 1, %d" % (cur, total))
        w("CONCAT %s, %s, e%d" % (cur, prev, i))
        prev = cur
    w("SHAPE hflat, %d, %d" % (N, H))
    w("COPY hflat, %s" % prev)
    w("COPY h, hflat")
    w("")
    for L in range(NL):
        w("; ---- layer %d ----" % L)
        w("RMSNORM xn, h, L%d_in_ln, eps1" % L)
        w("GEMM q, xn, L%d_q" % L)
        w("GEMM k, xn, L%d_k" % L)
        w("GEMM v, xn, L%d_v" % L)
        for h_i in range(NH):
            j = h_i // rep
            w("SLICE qh, q, %d, %d" % (h_i * HD, HD))
            w("SLICE kh, k, %d, %d" % (j * HD, HD))
            w("SLICE vh, v, %d, %d" % (j * HD, HD))
            w("ROPE qrot, qh, theta1, 0")
            w("ROPE krot, kh, theta1, 0")
            w("GEMM sc, qrot, krot")
            w("SCALE sc, sc, scale1")
            w("SOFTMAX sc, sc, causal")
            w("GEMM ctx, sc, vh, nn")
            w("WRITECOL att, ctx, %d, %d" % (h_i * HD, HD))
        w("GEMM oproj, att, L%d_o" % L)
        w("EWADD h2, h, oproj")
        w("RMSNORM xn, h2, L%d_post_ln, eps1" % L)
        w("GEMM gate, xn, L%d_gate" % L)
        w("GEMM up, xn, L%d_up" % L)
        w("ACT acts, gate, silu")
        w("EWMUL acts, acts, up")
        w("GEMM down, acts, L%d_down" % L)
        w("EWADD h, h2, down")
    w("")
    w("RMSNORM xn, h, final_norm, eps1")
    w("GEMM logits, xn, lm_head")
    w("PRINT logits")
    w("HALT")

    vmlp = os.path.join(odir, "full.vml")
    with open(vmlp, "w", encoding="ascii", newline="\n") as f:
        f.write("\n".join(A) + "\n")
    argp = os.path.join(odir, "weights.args")
    with open(argp, "w", encoding="ascii", newline="\n") as f:
        for name, path, r, c, dt in consts:
            f.write("%s:%s:%d:%d:%s\n" % (name, path.replace("\\", "/"), r, c, dt))
    np.array(toks, dtype="<u4").tofile(os.path.join(odir, "prompt_tok.bin"))
    with open(os.path.join(odir, "gen_rows.json"), "w", encoding="utf-8") as f:
        json.dump({"blocks": [{"name": "logits", "rows": N, "cols": V}]}, f, indent=2)
    with open(os.path.join(odir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"n_pos": N, "layers": NL, "seq": S, "gen": G, "vocab": V,
                   "tokens": toks}, f, indent=2)
    print("wrote %s (%d lines), %d constants, prompt_tok.bin=%d ids"
          % (vmlp, len(A), len(consts), N))
    return 0


if __name__ == "__main__":
    sys.exit(main())
