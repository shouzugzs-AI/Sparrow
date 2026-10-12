#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P13 task D, step 0: does plain per-row int8 survive on REAL MiniCPM weights?

WHY THIS PRECEDES ANY ENGINE WORK
    P11 measured int8 at the KERNEL level and reported two things:
      * on uniform weights:       rel err 7.2e-04, 3.62x  -> PASS
      * on outlier-bearing weights: rel err 4.0e-02, 3.44x -> FAIL vs 1e-2
    and concluded (its own conclusion 4) that "accuracy depends strongly on the
    weight distribution, so quantization must be validated on real model
    weights".
    That is exactly what this does. The engine-side outlier machinery is not
    free (a 5th operand does not fit in the 16-byte Ins -- checked against
    interp.rs, which has op/fl/dt/dst/a/b and NO spare byte), so before paying
    for it the honest question is whether plain int8 actually breaks the P13
    acceptance that matters: argmax agreement on real MiniCPM logits.

WHAT IS SIMULATED
    A full 24-layer forward in numpy, with the weight matrices quantized
    per-output-row to int8, compared against the same forward in the bf16 that
    the engine actually consumes. This isolates the quantization error from
    every other error source, and it is cheap (no engine changes yet).

    The comparison that decides: how many of the 4 argmax positions still
    agree, and what is the relative logit error.

Usage: python tools/p13_quant_plan.py --layers 24 --seq 4
"""
import argparse
import json
import os
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

MODEL = os.environ.get("MINICPM_DIR", "<外部模型目录>/MiniCPM5-1B")
sys.path.insert(0, os.path.join(ROOT, "tools"))


def bf16(a):
    u = np.asarray(a, dtype=np.float32).view(np.uint32).astype(np.uint64)
    rnd = np.uint64(0x7FFF) + ((u >> 16) & 1)
    return (((u + rnd) >> 16).astype(np.uint32) << 16).astype(np.uint32).view(np.float32)


def quant_i8(w, per_row=True):
    """Symmetric int8 with one scale per output row (LLM.int8 layout)."""
    if not per_row:
        s = np.abs(w).max() / 127.0
        return np.round(w / s).clip(-127, 127).astype(np.int8), np.full(1, s, np.float32)
    amax = np.abs(w).max(axis=1, keepdims=True)
    amax = np.where(amax == 0, 1e-30, amax)
    s = (amax / 127.0).astype(np.float32)
    q = np.round(w / s).clip(-127, 127).astype(np.int8)
    return q, s


def quant_i8_outlier(w, thresh=3.0):
    """LLM.int8-style mixed precision, the P11 design.

    Column j is an OUTLIER CHANNEL when its RMS exceeds `thresh` x the mean
    column RMS. Those columns keep full f32 precision and are EXCLUDED from the
    per-row absmax, so they no longer inflate the int8 scale (which was the
    measured cause of the 4.0e-02 error in P11).

    Returns (int8 matrix with zeroed outlier columns, scale, outlier mask).
    This is the numpy analogue of what the engine would need: a quantised GEMM
    plus a second f32 GEMM restricted to the outlier columns.
    """
    rms = np.sqrt((w.astype(np.float64) ** 2).mean(axis=0))
    mask = rms > thresh * rms.mean()
    wq = w.copy()
    wq[:, mask] = 0.0                       # f32 path handles these
    amax = np.abs(wq).max(axis=1, keepdims=True)
    amax = np.where(amax == 0, 1e-30, amax)
    s = (amax / 127.0).astype(np.float32)
    q = np.round(wq / s).clip(-127, 127).astype(np.int8)
    return q, s, mask


def deq(q, s):
    return (q.astype(np.float32) * s.astype(np.float32)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--seq", type=int, default=4)
    ap.add_argument("--ids", default="3,117,2501,88")
    args = ap.parse_args()

    from safetensors_reader import SafeTensors
    st = SafeTensors(os.path.join(MODEL, "model-00000-of-00001.safetensors"))
    with open(os.path.join(MODEL, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    H, NL, NH, NKV = cfg["hidden_size"], args.layers, cfg["num_attention_heads"], cfg["num_key_value_heads"]
    HD, INTER, V = cfg["head_dim"], cfg["intermediate_size"], cfg["vocab_size"]
    EPS, THETA = cfg["rms_norm_eps"], cfg["rope_theta"]
    S = args.seq
    ids = [int(x) for x in args.ids.split(",")][:S]
    rep = NH // NKV

    def get(n):
        return bf16(st.get(n))

    embed = get("model.embed_tokens.weight")
    final_norm = np.asarray(st.get("model.norm.weight"), np.float32)
    head = get("lm_head.weight")

    # Collect the per-layer weight set the engine consumes.
    W = []
    for L in range(NL):
        W.append(dict(
            q=get("model.layers.%d.self_attn.q_proj.weight" % L),
            k=get("model.layers.%d.self_attn.k_proj.weight" % L),
            v=get("model.layers.%d.self_attn.v_proj.weight" % L),
            o=get("model.layers.%d.self_attn.o_proj.weight" % L),
            gate=get("model.layers.%d.mlp.gate_proj.weight" % L),
            up=get("model.layers.%d.mlp.up_proj.weight" % L),
            down=get("model.layers.%d.mlp.down_proj.weight" % L),
            in_ln=np.asarray(st.get("model.layers.%d.input_layernorm.weight" % L), np.float32),
            post_ln=np.asarray(st.get("model.layers.%d.post_attention_layernorm.weight" % L), np.float32),
        ))

    def rmsnorm(x, g):
        m = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
        return (x / np.sqrt(m + EPS) * g).astype(np.float32)

    def rope(x, pos0):
        """x: [S, HD], rotate_half pairing, engine convention."""
        out = np.empty_like(x)
        half = HD // 2
        i = np.arange(half, dtype=np.float32)
        inv = (np.float32(THETA) ** (-2.0 * i / HD)).astype(np.float32)
        for r in range(x.shape[0]):
            pos = np.float32(pos0 + r)
            period = (np.float32(2 * np.pi) / inv).astype(np.float32)
            pr = np.where(period >= 1e6, pos, pos - period * np.floor(pos / period)).astype(np.float32)
            ang = (pr * inv).astype(np.float32)
            co, si = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
            x1, x2 = x[r, :half], x[r, half:]
            out[r, :half] = x1 * co - x2 * si
            out[r, half:] = x1 * si + x2 * co
        return out

    def softmax_causal(sc):
        n = sc.shape[-1]
        mask = np.triu(np.full((n, n), -1e30, np.float32), 1)
        z = sc + mask
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z.astype(np.float32))
        return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)

    def forward(quant, keep_head_f32=False, only=None):
        """quant: None | 'i8' | 'i8out'.

        'i8'    : per-row int8, every weight.
        'i8out' : P11's LLM.int8 mixed precision -- outlier CHANNELS keep f32.
        only    : set of matrix names to quantise; others stay bf16. Used to
                  find the cheapest subset that meets the <1e-2 criterion.
        """
        outl = {}

        def use(w, tag):
            if quant is None:
                return w
            # `only` restricts quantisation to named matrices; a tag like
            # "L3_gate" is matched by its suffix so the set can be written once.
            if only is not None and not any(tag.endswith(s) for s in only):
                return w
            if quant == "i8":
                q, s = quant_i8(w)
                return deq(q, s)
            q, s, mask = quant_i8_outlier(w)
            outl[tag] = (mask, w)
            return deq(q, s)

        def useA(x, tag):
            """Apply the outlier f32 correction to the activation side.

            C = A x W^T, with W = Q*s + W_out (on outlier cols).
            If the int8 part already carries zeroed outlier columns, the total
            is A x (Q*s)^T + A[:, mask] x W[:, mask]^T.
            """
            if quant != "i8out" or tag not in outl:
                return None
            mask, w = outl[tag]
            if not mask.any():
                return np.zeros((x.shape[0], w.shape[0]), np.float32)
            return (x[:, mask].astype(np.float32) @ w[:, mask].T.astype(np.float32)).astype(np.float32)

        def proj(x, w, tag):
            base = (x @ use(w, tag).T).astype(np.float32)
            corr = useA(x, tag)
            return base if corr is None else (base + corr).astype(np.float32)

        h = embed[ids].astype(np.float32)         # [S,H]
        for L in range(NL):
            lw = W[L]
            xn = rmsnorm(h, lw["in_ln"])
            q = proj(xn, lw["q"], "L%d_q" % L)
            k = proj(xn, lw["k"], "L%d_k" % L)
            v = proj(xn, lw["v"], "L%d_v" % L)
            att = np.zeros((S, NH * HD), np.float32)
            for h_i in range(NH):
                j = h_i // rep
                qh = q[:, h_i * HD:(h_i + 1) * HD]
                kh = k[:, j * HD:(j + 1) * HD]
                vh = v[:, j * HD:(j + 1) * HD]
                qr, kr = rope(qh, 0), rope(kh, 0)
                sc = (qr @ kr.T) * np.float32(1.0 / np.sqrt(HD))
                ctx = softmax_causal(sc.astype(np.float32)) @ vh
                att[:, h_i * HD:(h_i + 1) * HD] = ctx
            h2 = h + proj(att, lw["o"], "L%d_o" % L)
            xn2 = rmsnorm(h2, lw["post_ln"])
            gate = proj(xn2, lw["gate"], "L%d_gate" % L)
            up = proj(xn2, lw["up"], "L%d_up" % L)
            acts = (gate / (1.0 + np.exp(-gate.astype(np.float32)))).astype(np.float32) * up
            h = h2 + proj(acts, lw["down"], "L%d_down" % L)
        xn = rmsnorm(h, final_norm)
        if keep_head_f32:
            return (xn @ head.T).astype(np.float32)
        # proj() returns C = X x W^T (with the outlier correction folded in),
        # which for the head IS the logits.
        return proj(xn, head, "head")

    print("=== P13 task D step 0: int8 on REAL MiniCPM weights (%d layers) ===\n" % NL)
    ref = forward(None)
    print("bf16 reference logits: shape=%s absmax=%.4f" % (ref.shape, float(np.abs(ref).max())))

    # ------------------------------------------------------------------
    # VALIDATE THE SIMULATION before trusting any verdict from it.
    # If this numpy forward disagreed with the engine, a "FAIL" below could be
    # my bug rather than int8's. The HF oracle (f32 weights) is an independent
    # reference, and the ENGINE running bf16 weights scored rel 1.8806e-06
    # against it. So the numpy bf16 forward must land in the same place.
    # ------------------------------------------------------------------
    oracle = os.path.join(ROOT, "examples", "minicpm_full", "hf_logits.npy")
    if NL == 24 and S == 4 and os.path.exists(oracle):
        hf = np.load(oracle).astype(np.float32)
        if hf.shape == ref.shape:
            dv = np.abs(ref - hf)
            relv = float(dv.max()) / max(float(np.abs(hf).max()), 1e-30)
            mv = int((hf.argmax(1) == ref.argmax(1)).sum())
            print("\nSIMULATION VALIDATION vs the HuggingFace f32 oracle:")
            print("   numpy bf16 forward: rel = %.4e   argmax %d/%d" % (relv, mv, S))
            print("   engine bf16 (P13) : rel = 1.8806e-06   argmax 4/4")
            if relv < 1e-4 and mv == S:
                print("   => the simulation reproduces the engine's agreement, so")
                print("      the quantisation numbers below are trustworthy.")
            else:
                print("   => MISMATCH: the simulation does NOT reproduce the")
                print("      engine's result. Fix the simulation before drawing")
                print("      any conclusion from the quantisation numbers.")
                return 2
    else:
        print("\n(validation skipped: needs --layers 24 --seq 4 and the HF oracle)")

    q8 = forward("i8")
    d = np.abs(q8 - ref)
    rel = float(d.max()) / max(float(np.abs(ref).max()), 1e-30)
    am_ref, am_q = ref.argmax(1), q8.argmax(1)
    match = int((am_ref == am_q).sum())
    print("\nper-row int8 (no outlier handling):")
    print("   max_abs_err = %.4e   rel = %.4e" % (d.max(), rel))
    print("   argmax: int8 %s  bf16 %s  -> %d/%d match"
          % (am_q.tolist(), am_ref.tolist(), match, S))

    # Per-tensor stats: which matrices actually carry the outlier channels?
    print("\nper-matrix outlier check (max|x| / mean|x| of the ABSOLUTE column RMS):")
    for name in ("q", "k", "v", "o", "gate", "up", "down"):
        w = W[0][name]
        rms = np.sqrt((w.astype(np.float64) ** 2).mean(axis=0))
        print("   L0_%-5s colRMS ratio max/mean = %8.2f" % (name, rms.max() / rms.mean()))

    print()
    # ---- attribution: is the error from the layers, or from lm_head? ----
    # lm_head is [130560,1536] and its quantisation error lands DIRECTLY on the
    # logits with no averaging, so it is the prime suspect. It is also used only
    # ONCE per forward, so keeping it bf16 costs almost nothing in bytes
    # compared with the 24 layers.
    q8h = forward("i8", keep_head_f32=True)
    dh = np.abs(q8h - ref)
    relh = float(dh.max()) / max(float(np.abs(ref).max()), 1e-30)
    matchh = int((ref.argmax(1) == q8h.argmax(1)).sum())
    print("ATTRIBUTION -- int8 everywhere EXCEPT lm_head (kept bf16):")
    print("   max_abs_err = %.4e   rel = %.4e   argmax %d/%d"
          % (dh.max(), relh, matchh, S))
    print("   => lm_head contributes %.1f%% of the error, the 24 layers %.1f%%"
          % (100.0 * (1 - relh / max(rel, 1e-30)), 100.0 * relh / max(rel, 1e-30)))

    # ---- P11's actual prescription: outlier channels in f32 ----
    print()
    print("P11 prescription -- outlier CHANNELS keep f32 (LLM.int8 mixed precision):")
    q8o = forward("i8out")
    do = np.abs(q8o - ref)
    relo = float(do.max()) / max(float(np.abs(ref).max()), 1e-30)
    matcho = int((ref.argmax(1) == q8o.argmax(1)).sum())
    print("   max_abs_err = %.4e   rel = %.4e   argmax %d/%d"
          % (do.max(), relo, matcho, S))
    print("   outlier columns kept in f32: ", end="")
    rms = np.sqrt((W[0]["q"].astype(np.float64) ** 2).mean(axis=0))
    m = rms > 3.0 * rms.mean()
    print("%d / %d (%.2f%%) of a representative matrix"
          % (int(m.sum()), m.size, 100.0 * m.sum() / m.size))

    print()
    # ---- which SUBSET of matrices can be quantised and still pass? ----
    # This decides the engine design: if only the big MLP matrices need int8 to
    # get most of the byte saving, the attention path can stay bf16 (simpler).
    print("SUBSET TEST -- quantise only some matrices (criterion: rel <= 1e-2):")
    subsets = [
        ("MLP only (gate/up/down)", ("gate", "up", "down")),
        ("attention only (q/k/v/o)", ("_q", "_k", "_v", "_o")),
        ("all but lm_head", ("q", "k", "v", "o", "gate", "up", "down")),
    ]
    for name, sel in subsets:
        g = forward("i8", only=sel)
        dg = np.abs(g - ref)
        rg = float(dg.max()) / max(float(np.abs(ref).max()), 1e-30)
        mg = int((ref.argmax(1) == g.argmax(1)).sum())
        print("   %-26s rel = %.4e  argmax %d/%d  %s"
              % (name, rg, mg, S, "PASS" if rg <= 1e-2 else "FAIL"))

    print()
    ok = match == S and rel <= 1e-2
    ok_head = matchh == S and relh <= 1e-2
    ok_out = matcho == S and relo <= 1e-2
    if ok_out:
        print("VERDICT: PASS with the P11 outlier design (argmax %d/%d, rel %.2e)."
              % (matcho, S, relo))
        print("         Plain int8 alone gives %.2e (FAIL vs 1e-2), so the outlier" % rel)
        print("         path is REQUIRED -- and P11 already measured that it costs")
        print("         almost nothing in speed (3.18x vs 3.44x kernel-level).")
    elif ok:
        print("VERDICT: PASS -- plain per-row int8 suffices on real weights.")
    elif ok_head:
        print("VERDICT: PASS only with a bf16 lm_head (argmax %d/%d, rel %.2e)."
              % (matchh, S, relh))
    else:
        print("VERDICT: FAIL -- none of the three variants holds")
        print("         (plain %.2e, bf16-head %.2e, outlier %.2e)" % (rel, relh, relo))
    # The decision that matters for the engine: is the outlier path needed?
    print()
    print("DECISION INPUT: outlier path required = %s"
          % ("YES" if (not ok and ok_out) else ("NO" if ok else "unclear")))
    return 0 if (ok or ok_head or ok_out) else 1


if __name__ == "__main__":
    sys.exit(main())
