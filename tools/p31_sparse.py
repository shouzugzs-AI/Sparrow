#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P31: sparsify MiniCPM5-1B -- quality vs keep-ratio, several criteria.

WHAT THIS MEASURES (and what it does not)
    This is a QUALITY experiment: masks are applied ZERO-SHOT to a pretrained
    checkpoint, then mean CE is measured on held-out wiki_zh. It is not a
    healing/retraining run (that is a separate, much larger budget) and it is
    not a kernel benchmark (speed implications are derived analytically from
    the FLOP ratio and B1's measured cost per live connection).

    Arithmetic is f32 on bf16 weights, i.e. exactly what the VMLM engine does
    (P12/P13), so a loss delta here is a delta the engine would also see.

SPARSITY SHAPES
    A pretrained Llama block gives three different ways to spend a 1% budget,
    and they are NOT the same thing:

      conn   per-output-row top-k0 connections (|B1's shape|): each row of W
             keeps k0 of its K inputs. Needs a gather kernel to be fast, but
             compresses to keep-ratio * weights.
      in     per-matrix top-k INPUT channels, shared by every row: W keeps k
             columns. A dense GEMM on the sliced matrix is already fast (K
             shrinks), so this buys speed with NO sparse kernel.
      neuron FFN intermediate-unit pruning: keep M of 4608 hidden units,
             i.e. rows of gate/up and columns of down. Also dense-friendly
             (N of gate/up shrinks, K of down shrinks), and it is the shape a
             stock GEMM likes best.

    Reporting all three on one axis (share of the model's GEMM FLOPs kept)
    answers the design question for B2: is a connection mask worth a gather
    kernel, or does a structured cut buy more quality per FLOP?

CRITERIA (what decides which connections survive)
    mag     |W_ij|                     -- user's algorithm 1
    wanda   |W_ij| * rms(x_j)          -- user's algorithm 2, MSE-optimal form:
                                           per-row output error of dropping j is
                                           w_ij^2 E[x_j^2], so top-k by this
                                           minimises E[(y-y~)^2] when inputs
                                           are independent.
    l1      |W_ij| * E|x_j|            -- user's algorithm 2, L1 form
    rand    uniform per element        -- CONTROL: if random matches magnitude
                                           at low keep-ratio, saliency is not
                                           where the sparsity budget is spent.
    rescale bool: after masking, multiply each kept row by
            sqrt(sum_j w^2 E[x^2] / sum_{kept} w^2 E[x^2]) -- variance
            matching, the cheap half of SparseGPT/AWQ-style compensation.

CALIBRATION STATS
    rms(x_j), E|x_j| per input channel come from a forward pass over a
    DIFFERENT document range than the eval set (never reuse eval text).

USAGE
    python tools/p31_sparse.py --preset smoke
    python tools/p31_sparse.py --preset A --eval-tokens 1024
"""
import argparse
import json
import os
import sys
import time
import zlib

import numpy as np
import torch

BASE = r"D:\czsai\MiniCPM5-1B-Base"
WIKI = r"D:\czsai\data\wiki_zh.jsonl"
OUT = r"D:\czsai\vmlm\temp\p31"

MATS = ["q", "k", "v", "o", "gate", "up", "down"]
ATTR = {
    "q": ("self_attn", "q_proj"), "k": ("self_attn", "k_proj"),
    "v": ("self_attn", "v_proj"), "o": ("self_attn", "o_proj"),
    "gate": ("mlp", "gate_proj"), "up": ("mlp", "up_proj"),
    "down": ("mlp", "down_proj"),
}
MLP = ["gate", "up", "down"]
ATTN = ["q", "k", "v", "o"]


# ------------------------------------------------------- checkpoint (mmap)
class Ckpt(object):
    """Read-only mmap view of the safetensors file.

    WHY NOT KEEP A PRISTINE COPY IN RAM: the f32 model already needs 4.3 GB;
    a second bf16 copy would add 2.2 GB for something the OS page cache can
    serve at 4.5 GB/s on demand. Every config recomputes its scores from THIS
    file, so "start from the checkpoint" is true by construction.
    """

    def __init__(self, path):
        import struct
        self.f = open(path, "rb")
        (n,) = struct.unpack("<Q", self.f.read(8))
        self.meta = json.loads(self.f.read(n).decode("utf-8"))
        self.meta.pop("__metadata__", None)
        self.off = 8 + n
        self.mm = __import__("mmap").mmap(self.f.fileno(), 0, access=__import__("mmap").ACCESS_READ)

    def f32(self, name):
        m = self.meta[name]
        beg, end = m["data_offsets"]
        n = int(np.prod(m["shape"]))
        a = np.frombuffer(self.mm, dtype="<u2", count=n, offset=self.off + beg)
        return (a.astype(np.uint32) << 16).view(np.float32).reshape(m["shape"])


def param_name(layer, mk):
    a, b = ATTR[mk]
    return "model.layers.%d.%s.%s.weight" % (layer, a, b)



# --------------------------------------------------------------------- data
def read_docs(path, start, count, min_chars=200):
    out, seen = [], 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seen += 1
            if seen <= start:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            t = d.get("completion") or d.get("text") or ""
            if len(t) >= min_chars:
                out.append(t)
            if len(out) >= count:
                break
    return out


def build_seqs(tok, docs, seq_len):
    seqs = []
    for t in docs:
        ids = tok.encode(t, add_special_tokens=False)
        for i in range(0, len(ids) - seq_len + 1, seq_len):
            seqs.append(ids[i:i + seq_len])
    return seqs


# ------------------------------------------------------------------- model
def module_of(model, layer, mk):
    a, b = ATTR[mk]
    return getattr(getattr(model.model.layers[layer], a), b)


def flop_table(model):
    """Per-token GEMM FLOPs for every matrix, per layer, plus lm_head."""
    cfg = model.config
    H, I, V = cfg.hidden_size, cfg.intermediate_size, cfg.vocab_size
    hd, nh = cfg.head_dim, cfg.num_attention_heads
    nkv = cfg.num_key_value_heads
    shapes = {"q": (nh * hd, H), "k": (nkv * hd, H), "v": (nkv * hd, H),
              "o": (H, nh * hd), "gate": (I, H), "up": (I, H), "down": (H, I)}
    per_layer = {m: 2 * n * k for m, (n, k) in shapes.items()}
    return shapes, per_layer, 2 * V * H


# ------------------------------------------------------------------- masks
def score_of(W, crit, st, rng):
    """W: f32 [N,K]; st: dict(sq, ab, n[, g]) for its input channels."""
    if crit == "rand":
        return rng.random(W.shape).astype(np.float32)
    if crit == "grad":
        return np.abs(st["g"])
    a = np.abs(W)
    if crit == "snip":
        # |w * dL/dw| -- first-order loss sensitivity of THIS connection.
        # Needs one calibration backward pass; no retraining.
        return a * np.abs(st["g"])
    if crit == "mag":
        return a
    if crit == "wanda":
        rms = np.sqrt(st["sq"] / st["n"]).astype(np.float32)
        return a * rms[None, :]
    if crit == "l1":
        mabs = (st["ab"] / st["n"]).astype(np.float32)
        return a * mabs[None, :]
    raise ValueError(crit)


def keep_conn(score, k0):
    N, K = score.shape
    k0 = max(1, min(k0, K))
    idx = np.argpartition(-score, k0 - 1, axis=1)[:, :k0]
    m = np.zeros(score.shape, dtype=bool)
    np.put_along_axis(m, idx, True, axis=1)
    return m


def keep_in(score, k):
    """Columns: keep the k columns with the largest activation-weighted norm."""
    N, K = score.shape
    k = max(1, min(k, K))
    cs = np.sqrt((score.astype(np.float64) ** 2).sum(axis=0))
    idx = np.argpartition(-cs, k - 1)[:k]
    m = np.zeros((1, K), dtype=bool)
    m[0, idx] = True
    return np.broadcast_to(m, (1, K))


def neuron_importance(Wg, Wu, sg, su):
    """Hidden-unit importance for SwiGLU: activation-weighted row norms of
    gate and up. `up` is gated by silu(gate), `gate` by up, so both write;
    the read side (down) only scales, so it does not change the ranking."""
    rg = np.sqrt(sg["sq"] / sg["n"]).astype(np.float32)
    ru = np.sqrt(su["sq"] / su["n"]).astype(np.float32)
    ig = (np.abs(Wg) * rg[None, :]).astype(np.float64)
    iu = (np.abs(Wu) * ru[None, :]).astype(np.float64)
    return np.sqrt((ig ** 2).sum(axis=1) + (iu ** 2).sum(axis=1))


# -------------------------------------------------------------------- plans
def preset(name, ev=1024):
    """-> (list of config dicts)."""
    P = []

    def add(tag, mats, mode, crit, rho, **kw):
        d = dict(name=tag, mats=mats, mode=mode, crit=crit, rho=rho)
        d.update(kw)
        P.append(d)

    if name == "smoke":
        add("canary_dense", MLP, "conn", "mag", 1.0)
        add("canary_restore", MLP, "conn", "mag", 1.0)
        add("zero_all", MLP, "conn", "mag", 0.0)
        add("mlp_wanda_50", MLP, "conn", "wanda", 0.5)
    elif name == "A":       # criterion x keep-ratio on the MLP (conn shape)
        for crit in ["mag", "wanda", "l1", "rand"]:
            for rho in [0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]:
                add("mlp_%s_%d" % (crit, round(rho * 100)), MLP, "conn", crit, rho)
    elif name == "A2":      # finer near the knee, best two criteria
        for crit in ["mag", "wanda"]:
            for rho in [0.005, 0.01, 0.03, 0.07, 0.15, 0.3, 0.5, 0.7, 0.9]:
                add("mlp_%s_%d" % (crit, round(rho * 100)), MLP, "conn", crit, rho)
    elif name == "B":       # rescale + structured shapes at matched FLOPs
        for rho in [0.05, 0.1, 0.2, 0.4]:
            add("mlp_wanda_rs_%d" % round(rho * 100), MLP, "conn", "wanda", rho, rescale=True)
        for rho in [0.05, 0.1, 0.2, 0.4]:
            add("mlp_neuron_%d" % round(rho * 100), MLP, "neuron", "wanda", rho)
        for rho in [0.05, 0.1, 0.2, 0.4]:
            add("mlp_in_%d" % round(rho * 100), MLP, "in", "wanda", rho)
    elif name == "C":       # where is the damage? matrix scope at fixed rho
        for rho in [0.1, 0.3]:
            add("only_down_%d" % round(rho * 100), ["down"], "conn", "wanda", rho)
            add("only_gateup_%d" % round(rho * 100), ["gate", "up"], "conn", "wanda", rho)
            add("only_attn_%d" % round(rho * 100), ATTN, "conn", "wanda", rho)
            add("all_%d" % round(rho * 100), MLP + ATTN, "conn", "wanda", rho)
    elif name == "F":       # held-out validation: different docs, 2x longer seqs
        add("val_canary_1.0", MLP, "conn", "mag", 1.0)
        for crit in ["mag", "wanda"]:
            for rho in [0.6, 0.7, 0.8]:
                add("val_%s_%d" % (crit, round(rho * 100)), MLP, "conn", crit, rho)
    elif name == "G":       # does wanda's win come from down (outlier channels)?
        for crit in ["mag", "wanda"]:
            for rho in [0.3, 0.4]:
                add("g_down_%s_%d" % (crit, round(rho * 100)),
                    ["down"], "conn", crit, rho)
                add("g_gateup_%s_%d" % (crit, round(rho * 100)),
                    ["gate", "up"], "conn", crit, rho)
    elif name == "knee":
        P = preset("E") + preset("E2") + preset("E3") + preset("E4")
        # canary for the restore path: after a WIDE mask, a NARROW no-op scope
        # must still give the dense loss (this is what caught the 11.3718 bug)
        P.append(dict(name="wide_mlp_20", mats=MLP, mode="conn", crit="wanda", rho=0.2))
        P.append(dict(name="canary_narrow_1.0", mats=["down"], mode="conn",
                      crit="mag", rho=1.0))
    elif name == "E":       # knee-focused: only the points A/A2 do not cover
        for rho in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            add("k_snip_%d" % round(rho * 100), MLP, "conn", "snip", rho)
        for rho in [0.4, 0.6, 0.8, 0.9]:
            add("k_rand_%d" % round(rho * 100), MLP, "conn", "rand", rho)
    elif name == "E2":      # compensation + structured shapes AT THE KNEE
        for rho in [0.5, 0.6, 0.7, 0.8]:
            add("rs_wanda_%d" % round(rho * 100), MLP, "conn", "wanda", rho, rescale=True)
            add("rs_snip_%d" % round(rho * 100), MLP, "conn", "snip", rho, rescale=True)
        for rho in [0.5, 0.6, 0.7, 0.8]:
            add("neuron_%d" % round(rho * 100), MLP, "neuron", "wanda", rho)
        for rho in [0.5, 0.6, 0.7, 0.8]:
            add("in_%d" % round(rho * 100), MLP, "in", "wanda", rho)
    elif name == "E3":      # where in the model is the damage, at the knee
        for rho in [0.5, 0.7]:
            add("only_down_%d" % round(rho * 100), ["down"], "conn", "wanda", rho)
            add("only_gateup_%d" % round(rho * 100), ["gate", "up"], "conn", "wanda", rho)
            add("only_attn_%d" % round(rho * 100), ATTN, "conn", "wanda", rho)
            add("only_gate_%d" % round(rho * 100), ["gate"], "conn", "wanda", rho)
            add("only_up_%d" % round(rho * 100), ["up"], "conn", "wanda", rho)
            add("all_%d" % round(rho * 100), MLP + ATTN, "conn", "wanda", rho)
    elif name == "E4":      # per-layer sensitivity at the knee
        for l in [0, 4, 8, 12, 16, 20, 23]:
            add("L%d_mlp_70" % l, MLP, "conn", "wanda", 0.7, layers=[l])
    elif name == "D":       # per-layer sensitivity: only one layer's MLP masked
        for l in [0, 4, 8, 12, 16, 20, 23]:
            add("L%d_mlp_30" % l, MLP, "conn", "wanda", 0.3, layers=[l])
    elif name == "full":
        P = preset("canaries") + preset("A") + preset("A2") + preset("B") + preset("C") + preset("D")
    elif name == "canaries":
        add("canary_mag_1.0", MLP, "conn", "mag", 1.0)
        add("canary_after", MLP, "conn", "mag", 1.0)
        add("wipe_all", MLP, "conn", "mag", 0.0)
    else:
        raise SystemExit("unknown preset %s" % name)
    return P


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=BASE)
    ap.add_argument("--preset", default="smoke")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--eval-tokens", type=int, default=1024)
    ap.add_argument("--calib-tokens", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=0,
                    help="0 = benchmark 1/2/4 and pick the fastest")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default="")
    ap.add_argument("--calib-start", type=int, default=0)
    ap.add_argument("--eval-start", type=int, default=64)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.makedirs(args.out, exist_ok=True)
    tag = args.tag or args.preset
    res_path = os.path.join(args.out, "results_%s.jsonl" % tag)
    t_start = time.time()

    def log(*a):
        print(*a, flush=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(args.model)
    n_seq = max(1, args.eval_tokens // args.seq_len)
    log("[tok] %s vocab=%d" % (args.model, tok.vocab_size))

    ev_docs = read_docs(WIKI, args.eval_start, 256)
    ev_seqs = build_seqs(tok, ev_docs, args.seq_len)[:n_seq]
    n_cal = max(1, args.calib_tokens // args.seq_len)
    ca_docs = read_docs(WIKI, args.calib_start, 256)
    ca_seqs = build_seqs(tok, ca_docs, args.seq_len)[:n_cal]
    log("[data] eval %d seqs (%d tok)  calib %d seqs (%d tok)  seq_len=%d"
        % (len(ev_seqs), len(ev_seqs) * args.seq_len, len(ca_seqs),
           len(ca_seqs) * args.seq_len, args.seq_len))

    def batches(seqs, B):
        return [torch.tensor(seqs[i:i + B], dtype=torch.long)
                for i in range(0, len(seqs), B)]

    t0 = time.time()
    kw = dict(dtype=torch.float32)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, **kw)
    except TypeError:
        kw["torch_dtype"] = kw.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(args.model, **kw)
    model.eval()
    log("[model] loaded %.1fs params=%.4fB"
        % (time.time() - t0, sum(p.numel() for p in model.parameters()) / 1e9))

    shapes, per_layer, head_flops = flop_table(model)
    nl = model.config.num_hidden_layers
    log("[cfg] L=%d H=%d I=%d  head_flops/token=%.3f GFLOP"
        % (nl, model.config.hidden_size, model.config.intermediate_size, head_flops / 1e9))

    # ---- read-only mmap of the checkpoint: every config recomputes its scores
    #      from the TRUE weights and restores from them (no RAM copy needed).
    t0 = time.time()
    ck = Ckpt(os.path.join(args.model, "model-00000-of-00001.safetensors"))
    log("[ckpt] mmap %.1fs  %d tensors  %.2f GB on disk"
        % (time.time() - t0, len(ck.meta),
           os.path.getsize(os.path.join(args.model, "model-00000-of-00001.safetensors")) / 1e9))

    # ---- pick the fastest eval batch size (M=B*S; this machine's GEMM is
    #      load-port bound, so the best B is not obvious a priori)
    if args.batch == 0:
        best = None
        for B in (1, 2, 4):
            bx = batches(ev_seqs[:2 * B], B)
            t0 = time.time()
            with torch.inference_mode():
                for x in bx:
                    model(input_ids=x, labels=x)
            dt = time.time() - t0
            tps = sum(x.numel() for x in bx) / dt
            log("[bench] batch=%d  %.1f tok/s" % (B, tps))
            if best is None or tps > best[1]:
                best = (B, tps)
        args.batch = best[0]
        log("[bench] using batch=%d" % args.batch)
    ev_b = batches(ev_seqs, args.batch)
    ca_b = batches(ca_seqs, args.batch)

    # ---- calibration stats over a DIFFERENT document range
    plan = preset(args.preset, args.eval_tokens)
    log("[plan] %d configs" % len(plan))
    stats = {}
    hooks = []

    def mk_hook(key):
        def fn(mod, inp, out):
            x = inp[0].detach()
            s = stats[key]
            xd = x.double()
            s["sq"] += (xd * xd).sum(dim=(0, 1)).numpy()
            s["ab"] += xd.abs().sum(dim=(0, 1)).numpy()
            s["n"] += x.shape[0] * x.shape[1]
        return fn

    tgt_mats = set()
    for cfgd in plan:
        tgt_mats.update(cfgd["mats"])
    for mk in tgt_mats:
        for l in range(nl):
            key = (l, mk)
            stats[key] = dict(sq=np.zeros(shapes[mk][1]), ab=np.zeros(shapes[mk][1]), n=0)
            hooks.append(module_of(model, l, mk).register_forward_hook(mk_hook(key)))
    t0 = time.time()
    with torch.inference_mode():
        for x in ca_b:
            model(input_ids=x)
    for h in hooks:
        h.remove()
    log("[calib] %.1fs  %d matrices, %d tok" % (time.time() - t0, len(stats), len(ca_seqs) * args.seq_len))

    # ---- optional: one calibration backward pass -> |dL/dw| (snip, grad)
    need_grad = sorted({c["mats"][i] for c in plan
                        if c["crit"] in ("snip", "grad")
                        for i in range(len(c["mats"]))})
    if need_grad:
        t0 = time.time()
        acc = {}
        for x in ca_b:
            model.zero_grad(set_to_none=True)
            out = model(input_ids=x, labels=x)
            out.loss.backward()
            for mk in need_grad:
                for l in range(nl):
                    key = (l, mk)
                    g = module_of(model, l, mk).weight.grad.detach().numpy()
                    a = np.abs(g)
                    acc[key] = a.astype(np.float32) if key not in acc else acc[key] + a
            model.zero_grad(set_to_none=True)
        nb = len(ca_b)
        for key, v in acc.items():
            stats[key]["g"] = (v / nb).astype(np.float32)
        del acc
        log("[grad] %.1fs  mean |dL/dw| over %d matrices, %d tok"
            % (time.time() - t0, len(need_grad) * nl, len(ca_seqs) * args.seq_len))

    # ---- eval helper
    def eval_loss():
        tl, tn = 0.0, 0
        with torch.inference_mode():
            for x in ev_b:
                out = model(input_ids=x, labels=x)
                n = x.shape[0] * (x.shape[1] - 1)
                tl += float(out.loss) * n
                tn += n
        return tl / tn, tn

    def restore(mats, layers):
        for mk in mats:
            for l in layers:
                p = module_of(model, l, mk).weight
                p.data.copy_(torch.from_numpy(ck.f32(param_name(l, mk))))

    def all_layers(cfg):
        return cfg.get("layers") or list(range(nl))

    # ---- dense baseline + determinism canary
    loss0, ntok = eval_loss()
    loss0b, _ = eval_loss()
    log("[dense] loss=%.6f ppl=%.3f tokens=%d  (repeat %.6f, delta=%.2e)"
        % (loss0, float(np.exp(loss0)), ntok, loss0b, loss0b - loss0))
    dense_flops = sum(per_layer.values()) * nl + head_flops
    rows = []
    with open(res_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(dict(name="dense", loss=loss0, ppl=float(np.exp(loss0)),
                                tokens=ntok, flops=dense_flops,
                                secs=time.time() - t_start,
                                eval_tokens=len(ev_seqs) * args.seq_len,
                                calib_tokens=len(ca_seqs) * args.seq_len,
                                batch=args.batch)) + "\n")

    # ---- sweep
    dirty = set()          # matrices currently masked in the live model
    for ci, cfg in enumerate(plan):
        t0 = time.time()
        layers = all_layers(cfg)
        mats = cfg["mats"]
        mode, crit, rho = cfg["mode"], cfg["crit"], cfg["rho"]
        rng = np.random.default_rng(zlib.crc32(cfg["name"].encode()) % (2 ** 32))
        kept_conn, dense_conn = 0, 0
        k0_of = {}

        # Restore EVERY matrix left masked by an earlier config. Restoring only
        # this config's own scope is wrong: a narrow config (one layer, or one
        # matrix group) leaves the previous config's damage in place and the
        # number looks plausible -- measured, that bug made every per-layer row
        # read exactly the all-layers loss (11.3718).
        cur = {(l, mk) for mk in mats for l in layers}
        for (l, mk) in dirty - cur:
            module_of(model, l, mk).weight.data.copy_(
                torch.from_numpy(ck.f32(param_name(l, mk))))
        dirty = set(cur)
        if mode == "neuron":
            # one joint hidden-unit mask per layer for gate/up/down
            M_keep = max(1, int(round(rho * shapes["gate"][0])))
            for l in layers:
                Wg = ck.f32(param_name(l, "gate"))
                Wu = ck.f32(param_name(l, "up"))
                Wd = ck.f32(param_name(l, "down"))
                imp = neuron_importance(Wg, Wu, stats[(l, "gate")], stats[(l, "up")])
                idx = np.argpartition(-imp, M_keep - 1)[:M_keep]
                rowm = np.zeros(Wg.shape[0], dtype=bool)
                rowm[idx] = True
                Wg[~rowm, :] = 0.0
                Wu[~rowm, :] = 0.0
                Wd[:, ~rowm] = 0.0
                module_of(model, l, "gate").weight.data.copy_(torch.from_numpy(Wg))
                module_of(model, l, "up").weight.data.copy_(torch.from_numpy(Wu))
                module_of(model, l, "down").weight.data.copy_(torch.from_numpy(Wd))
                kept_conn += M_keep * (Wg.shape[1] + Wu.shape[1]) + M_keep * Wd.shape[0]
                dense_conn += (Wg.size + Wu.size + Wd.size)
            k0_of["hidden_units"] = M_keep
        else:
            for mk in mats:
                k_this = max(1, int(round(rho * shapes[mk][1])))
                k0_of[mk] = k_this
                for l in layers:
                    W0 = ck.f32(param_name(l, mk))
                    N, K = W0.shape
                    sc = score_of(W0, crit, stats[(l, mk)], rng)
                    if mode == "conn":
                        keep = keep_conn(sc, k_this)
                        kept_conn += N * k_this
                    elif mode == "in":
                        keep = keep_in(sc, k_this)
                        kept_conn += N * k_this
                    else:
                        raise SystemExit("bad mode %s" % mode)
                    dense_conn += N * K
                    Wn = W0 * keep
                    if cfg.get("rescale") and mode == "conn":
                        s2 = sc.astype(np.float64) ** 2
                        num = s2.sum(axis=1)
                        den = (s2 * keep).sum(axis=1)
                        alpha = np.sqrt(np.where(den > 0, num / np.maximum(den, 1e-30), 1.0))
                        Wn = Wn * alpha[:, None].astype(np.float32)
                    module_of(model, l, mk).weight.data.copy_(torch.from_numpy(Wn))

        # how much of the saliency mask is just magnitude? (mask identity, not loss)
        overlap = None
        if mode == "conn" and crit != "mag":
            js = []
            for l in (0, nl // 2):
                for mk in mats:
                    W0 = ck.f32(param_name(l, mk))
                    k = max(1, int(round(rho * shapes[mk][1])))
                    ka = keep_conn(np.abs(W0), k)
                    kb = keep_conn(score_of(W0, crit, stats[(l, mk)],
                                            np.random.default_rng(0)), k)
                    js.append(float((ka & kb).sum()) / float((ka | kb).sum()))
            overlap = float(np.mean(js))

        loss, ntok = eval_loss()
        dt = time.time() - t0
        if rho >= 1.0 and abs(loss - loss0) > 1e-6:
            log("[FATAL] canary broken: rho=1.0 gives %.8f, dense is %.8f"
                % (loss, loss0))
            return 2
        scope_dense = sum(per_layer[mk] for mk in mats) * len(layers)
        scope_kept = scope_dense * (kept_conn / max(1, dense_conn))
        flops_now = dense_flops - scope_dense + scope_kept
        row = dict(name=cfg["name"], mode=mode, crit=crit, rho=rho, mats=mats,
                   layers=("all" if len(layers) == nl else layers),
                   rescale=bool(cfg.get("rescale")), k0=k0_of,
                   mask_overlap_mag=overlap,
                   conn_ratio=kept_conn / max(1, dense_conn),
                   flop_ratio=flops_now / dense_flops,
                   loss=loss, ppl=float(np.exp(loss)), d_loss=loss - loss0,
                   tokens=ntok, secs=dt)
        rows.append(row)
        with open(res_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        log("[%2d/%2d] %-18s %-6s %-5s rho=%.3f flop=%.3f loss=%.4f (d=%+.3f) %.1fs"
            % (ci + 1, len(plan), cfg["name"], mode, crit, rho,
               row["flop_ratio"], loss, loss - loss0, dt))

    restore(list(tgt_mats), list(range(nl)))
    log("[done] %.1f min  -> %s" % ((time.time() - t_start) / 60, res_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
