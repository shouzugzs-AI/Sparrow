#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P10 B2 step 1: HuggingFace oracle for MiniCPM5-1B layer 0.

GOAL OF THIS SCRIPT (per the task brief): NOT to "make it run", but to
**expose every real gap** by comparing a single layer against the true
HuggingFace forward pass. A numpy reimplementation would only reproduce my
own misunderstanding of Llama's layer; the HF model is the ground truth
(and both torch and transformers are installed here).

It dumps, for a fixed random-but-seeded input:
  * hidden_states entering layer 0
  * every intermediate inside layer 0 (norm / q / k / v / rope / attn / o /
    post-norm / gate / up / act / down)
  * hidden_states leaving layer 0
so the engine slice can be diffed stage by stage.

IMPORTANT SUBTLETY (this is where naive oracle scripts go wrong):
`model.layers.0` is called by LlamaModel with `position_embeddings=(cos,sin)`
produced ONCE for the whole model, and with `attention_mask`. Hooking the
block directly with hand-made cos/sin is a great way to introduce a bug in
the ORACLE. So we use forward hooks on the real modules and call the real
`model(...)`, then read out what actually happened.

Usage:
    python tools/minicpm_layer0_oracle.py --out examples/minicpm/layer0_ref
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

MODEL_DIR = os.environ.get("MINICPM_DIR", "<外部模型目录>/MiniCPM5-1B")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DIR)
    ap.add_argument("--out", default=r"<repo>/examples/minicpm/layer0_ref")
    ap.add_argument("--seq", type=int, default=4, help="sequence length")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--use-cache", action="store_true",
                    help="reuse _capture.npz from a previous run (skips the "
                         "several-minute f32 weight load)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ---------------------------------------------------------------- config
    with open(os.path.join(args.model, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    print("config: hidden=%d layers=%d heads=%d kv_heads=%d head_dim=%d"
          % (cfg["hidden_size"], cfg["num_hidden_layers"], cfg["num_attention_heads"],
             cfg["num_key_value_heads"], cfg["head_dim"]))
    print("        rope_theta=%s rms_eps=%s act=%s tie=%s"
          % (cfg["rope_theta"], cfg["rms_norm_eps"], cfg["hidden_act"],
             cfg["tie_word_embeddings"]))

    from transformers import AutoModelForCausalLM, AutoConfig

    hf_cfg = AutoConfig.from_pretrained(args.model)
    # Load in float32 on CPU: the oracle must be HIGHER precision than the
    # engine (f32), otherwise we cannot tell engine error from oracle noise.
    model = AutoModelForCausalLM.from_pretrained(
        args.model, config=hf_cfg, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    model.eval()
    print("loaded. dtype=%s" % next(model.parameters()).dtype)

    layer0 = model.model.layers[0]

    # ------------------------------------------------------------- hook setup
    # Capture inputs AND outputs of the interesting submodules. Keyed by a
    # stable name so the engine comparison can look them up.
    cap = {}

    def mk(name, keep_out=True, keep_in=True):
        def hook(mod, inp, out):
            if keep_in and len(inp):
                cap[name + ":in"] = inp[0].detach().to(torch.float32).numpy().copy()
            if keep_out:
                o = out[0] if isinstance(out, (tuple, list)) else out
                cap[name + ":out"] = o.detach().to(torch.float32).numpy().copy()
        return hook

    hooks = []
    hooks.append(layer0.input_layernorm.register_forward_hook(mk("in_ln")))
    hooks.append(layer0.self_attn.q_proj.register_forward_hook(mk("q_proj")))
    hooks.append(layer0.self_attn.k_proj.register_forward_hook(mk("k_proj")))
    hooks.append(layer0.self_attn.v_proj.register_forward_hook(mk("v_proj")))
    hooks.append(layer0.self_attn.o_proj.register_forward_hook(mk("o_proj")))
    hooks.append(layer0.post_attention_layernorm.register_forward_hook(mk("post_ln")))
    hooks.append(layer0.mlp.gate_proj.register_forward_hook(mk("gate_proj")))
    hooks.append(layer0.mlp.up_proj.register_forward_hook(mk("up_proj")))
    hooks.append(layer0.mlp.down_proj.register_forward_hook(mk("down_proj")))
    hooks.append(layer0.register_forward_hook(mk("layer0", keep_in=True)))

    # ---------------------------------------------------------------- forward
    torch.manual_seed(args.seed)
    V = cfg["vocab_size"]
    # Fixed token ids (arbitrary but deterministic, all < V)
    ids = torch.tensor([[3, 117, 2501, 88][: args.seq]], dtype=torch.long)
    print("input ids: %s" % ids.tolist())

    # NOTE: loading 0.99B params in f32 takes several minutes. Cache the
    # capture so re-running the *analysis* (which is where the iteration
    # happens) does not pay for it again.
    cache = os.path.join(args.out, "_capture.npz")
    from_cache = args.use_cache and os.path.exists(cache)
    if from_cache:
        print("loading cached capture from %s" % cache)
        z = np.load(cache, allow_pickle=False)
        cap = {k: z[k] for k in z.files if k.startswith("cap__")}
        cap = {k[len("cap__"):]: v for k, v in cap.items()}
        hs0 = z["hs0"]
        hs1 = z["hs1"]
        c = z["c"]
        sn = z["sn"]
        print("cache: %d captured tensors, cos %s" % (len(cap), (c.shape,)))
    else:
        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=True, use_cache=False)

        for h in hooks:
            h.remove()

        hs = out.hidden_states
        hs0 = hs[0].numpy()
        hs1 = hs[1].numpy()

        # cos/sin: recompute from the same rotary embedding the model used.
        # These are model-level: they depend on rope_theta AND head_dim, and
        # are the most common source of "engine looks close but not equal".
        rotary = model.model.rotary_emb
        pos = torch.arange(args.seq, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            try:
                cos, sin = rotary(hs[0], pos)
            except TypeError:
                cos, sin = rotary(pos, dtype=torch.float32)
        print("rope cos/sin raw shape: %s %s" % (tuple(cos.shape), tuple(sin.shape)))
        c = cos.numpy()
        sn = sin.numpy()
        # Normalize to [s, hd]: HF versions differ ([b,s,hd] vs [b,1,s,hd]).
        if c.ndim == 4:
            c = c[0, 0] if c.shape[1] == 1 else c[0]
            sn = sn[0, 0] if sn.shape[1] == 1 else sn[0]
        elif c.ndim == 3:
            c = c[0]
            sn = sn[0]
        c = np.ascontiguousarray(c, dtype=np.float32)
        sn = np.ascontiguousarray(sn, dtype=np.float32)

        np.savez_compressed(
            cache,
            **{"cap__" + k: v for k, v in cap.items()},
            hs0=hs0, hs1=hs1, c=c, sn=sn,
        )
        print("cached capture -> %s" % cache)

    print("cos normalized to %s" % (c.shape,))

    # ------------------------------------------------------------- dump refs
    def save(name, arr):
        p = os.path.join(args.out, name + ".npy")
        np.save(p, arr.astype(np.float32))
        return arr

    saved = {}
    for k, v in sorted(cap.items()):
        nm = k.replace(":", "_")
        save(nm, v)
        saved[nm] = list(v.shape)

    # Embedding output = hidden_states[0] (input to layer 0)
    save("hidden_in", hs0)
    save("hidden_out", hs1)
    saved["hidden_in"] = list(hs0.shape)
    saved["hidden_out"] = list(hs1.shape)

    # Also dump the layer-0 weights AND the exact cos/sin that were used.
    # cos/sin are model-level: they depend on rope_theta AND head_dim, and
    # are the single most common source of "engine looks close but not equal".
    from_cache = args.use_cache and os.path.exists(cache)
    W = model.model.layers[0]
    wdump = {}
    if not from_cache:
        wdump = {
            "input_layernorm.weight": W.input_layernorm.weight,
            "self_attn.q_proj.weight": W.self_attn.q_proj.weight,
            "self_attn.k_proj.weight": W.self_attn.k_proj.weight,
            "self_attn.v_proj.weight": W.self_attn.v_proj.weight,
            "self_attn.o_proj.weight": W.self_attn.o_proj.weight,
            "post_attention_layernorm.weight": W.post_attention_layernorm.weight,
            "mlp.gate_proj.weight": W.mlp.gate_proj.weight,
            "mlp.up_proj.weight": W.mlp.up_proj.weight,
            "mlp.down_proj.weight": W.mlp.down_proj.weight,
            "model.embed_tokens.weight": model.model.embed_tokens.weight,
        }
    for k, t in wdump.items():
        if k == "model.embed_tokens.weight":
            # only the rows we actually use (id 3,117,2501,88) to keep it small
            t = t[ids[0]]
            k = "embed_rows"
        save(k.replace(".", "_"), t.detach().to(torch.float32).numpy())

    save("rope_cos", c)
    save("rope_sin", sn)

    # ------------------------------------------------- attention internals
    # Reproduce the attention math explicitly so the engine has a stagewise
    # oracle for q/k/v AFTER rope and for the softmax weights.
    cfgd = cfg
    nh, nkv, hd = cfgd["num_attention_heads"], cfgd["num_key_value_heads"], cfgd["head_dim"]
    b, s, _ = cap["q_proj:out"].shape
    print("q_proj out %s -> reshape (b=%d, s=%d, nh=%d, hd=%d)"
          % (cap["q_proj:out"].shape, b, s, nh, hd))
    print("k_proj out %s (nkv*hd = %d)" % (cap["k_proj:out"].shape, nkv * hd))
    q = cap["q_proj:out"].reshape(b, s, nh, hd).transpose(0, 2, 1, 3)   # [b,nh,s,hd]
    k = cap["k_proj:out"].reshape(b, s, nkv, hd).transpose(0, 2, 1, 3)  # [b,nkv,s,hd]
    v = cap["v_proj:out"].reshape(b, s, nkv, hd).transpose(0, 2, 1, 3)
    c = np.ascontiguousarray(c, dtype=np.float32)
    sn = np.ascontiguousarray(sn, dtype=np.float32)
    print("cos normalized to %s" % (c.shape,))

    def rope(t, c, s_):
        # t: [b,h,s,hd]; c/s: [s,hd] (transformers applies to full head_dim)
        t1 = t[..., : hd // 2]
        t2 = t[..., hd // 2:]
        cc = c[:, : hd // 2]
        ss = s_[:, : hd // 2]
        o1 = t1 * cc - t2 * ss
        o2 = t1 * ss + t2 * cc
        return np.concatenate([o1, o2], axis=-1)

    q_r = rope(q, c, sn)
    k_r = rope(k, c, sn)
    save("q_roped", q_r)
    save("k_roped", k_r)
    save("v_heads", v)

    # GQA expand + causal attention
    rep = nh // nkv
    k_e = np.repeat(k_r, rep, axis=1)
    v_e = np.repeat(v, rep, axis=1)
    sc = np.einsum("bhsd,bhtd->bhst", q_r, k_e) / np.sqrt(hd)
    mask = np.triu(np.ones((s, s), dtype=np.float32), 1) * -1e30
    sc = sc + mask
    mx = sc.max(axis=-1, keepdims=True)
    ex = np.exp(sc - mx)
    att = ex / ex.sum(axis=-1, keepdims=True)
    save("attn_weights", att)
    ctx = np.einsum("bhst,bhtd->bhsd", att, v_e)   # [b,nh,s,hd]
    save("attn_ctx", ctx)
    ctx_cat = ctx.transpose(0, 2, 1, 3).reshape(b, s, nh * hd)
    save("attn_ctx_cat", ctx_cat)

    # MLP activation (silu)
    g = cap["gate_proj:out"]
    u = cap["up_proj:out"]
    act = (g / (1.0 + np.exp(-g))) * u
    save("mlp_act", act)

    meta = {
        "config": cfg,
        "input_ids": ids.tolist(),
        "seq": args.seq,
        "seed": args.seed,
        "saved": saved,
        "norms": {
            "hidden_in_absmax": float(np.abs(hs0).max()),
            "hidden_out_absmax": float(np.abs(hs1).max()),
        },
    }
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print()
    print("saved %d arrays to %s" % (len(saved), args.out))
    for k in sorted(saved):
        print("  %-28s %s" % (k, saved[k]))


if __name__ == "__main__":
    sys.exit(main())
