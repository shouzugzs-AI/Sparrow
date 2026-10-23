#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P35: dense weights + DYNAMIC activation mask (input-side K truncation).

THE IDEA (user's design, from scratch so the basis can be made square)
    A single per-token channel mask is chosen from the ATTENTION sublayer and
    reused across the whole layer.  The mask lives in the residual-channel
    space (H = d), so every GEMM that *reads* h can drop the masked columns and
    every GEMM that *writes* h could drop masked rows (we only do the input
    side here).  Because q_proj is square (heads*head_dim == d) the Q channel
    index is the same index as the residual channel -- "Q channels are directly
    connected to the FFN", the mask can be reused with no projection.

    mask m_in for layer l  -> qkv K dim      (which h channels attention reads)
    same mask              -> gate/up K dim  (which h channels the FFN reads)
    FFN mid mask (own)     -> down K dim     (which neurons the FFN writes with)

WHICH SIGNAL (--mask-src)
    none    dense baseline, no mask at all
    h       mask = top-m of |rmsnorm(x)| at the layer input; same-layer use for
            qkv and FFN (zero pipeline delay, exact, but no learning signal)
    q       mask from the ATTENTION sublayer: per-channel |Q| of layer l is used
            by layer l's FFN and by layer l+1's qkv (one-layer pipeline, exactly
            the "attention picks the mask, the layer reuses it" reading)
    q-ste   same as q but the top-k is passed through a straight-through softmax
            surrogate so the selection gradient reaches Wq ("the mask
            computation itself is learnable")

WHY MASKED-DENSE GEMM IS AN EXACT SIMULATION
    x * m then dense GEMM == gather(x, m) @ W[:, m] exactly (masked entries are
    multiplied by 0).  So the trained function is the sparse one; only the FLOPs
    are wasted here.  We therefore count FLOPs analytically and NEVER quote
    wall-clock speedups from this script -- the fast layout (channel frequency
    sort + per-token-group union slice) is a separate Rust job.

DATA PROTOCOL: identical to p32_dense.py (same files, same batch cursor
    (seed*977) % (N-seq-1), += batch*seq, same fixed val batches, same Adam +
    per-tensor clip + warmup/cosine) so arms are comparable.

USAGE
    python tools/p35_dyn.py --d 256 --layers 12 --heads 4 --hidden 1024 \
        --mask-src q --keep 0.5625 --steps 488 --tag C_q
    python tools/p35_dyn.py --d 256 --layers 12 --heads 4 --hidden 1024 \
        --mask-src none --steps 488 --tag A_dense
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = r"D:\czsai\vmlm\artifacts\zh_wiki"
OUT = r"D:\czsai\vmlm\artifacts\p35"


def load_ids(path):
    return np.fromfile(path, dtype="<u4").astype(np.int64)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        r = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * r * self.w


def mask_from_score(score, keep, ste=False):
    """score [B,T,d] -> (mask, p).  Forward is hard top-k, backward (ste) = softmax.

    sorted=False matters: on this CPU torch.topk(...,sorted=True) costs 1.7x more
    than sorted=False (measured 63.5 vs 25.6 ms for 32x128x1024), and the order
    is never used.
    """
    B, T, D = score.shape
    m = max(1, int(round(keep * D)))
    idx = torch.topk(score, m, dim=-1, sorted=False).indices
    hard = torch.zeros_like(score)
    hard.scatter_(-1, idx, 1.0)
    if not ste:
        return hard, None
    p = torch.softmax(score, dim=-1)
    return hard + p - p.detach(), p


def norm_entropy(t):
    """Normalized entropy of the per-row |value| distribution (0=one value, 1=uniform).

    This is the "数值差异熵": how concentrated the activation magnitudes are inside
    one token.  Reported separately for the residual input (h) and the FFN
    intermediate, because those are the two spaces the masks act on.
    """
    D = t.shape[-1]
    a = t.detach().abs()
    p = a / (a.sum(-1, keepdim=True) + 1e-12)
    H = -(p * (p + 1e-12).log()).sum(-1).mean()
    return float(H / math.log(D))


class Block(nn.Module):
    def __init__(self, d, heads, hidden):
        super().__init__()
        self.d, self.h, self.hd = d, heads, d // heads
        self.ln1 = RMSNorm(d)
        self.ln2 = RMSNorm(d)
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(d, hidden, bias=False)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)
        self.last_mask = None      # detached m_ffn, for diagnostics
        # --- P35 metrics (user request) ---
        # cumulative per-channel activation counts over the WHOLE run:
        #   usage_h   [d]  how often each residual channel was selected
        #   usage_mid [I]  how often each FFN neuron was selected
        self.usage_h = None
        self.usage_mid = None
        self.last_H = None         # value-distribution entropies on the last forward

    def _rope(self, x, cos, sin):
        # x: [B, heads, T, hd]
        T = x.shape[2]
        c = cos[:T].view(1, 1, T, self.hd)
        s = sin[:T].view(1, 1, T, self.hd)
        h2 = self.hd // 2
        return x * c + torch.cat([-x[..., h2:], x[..., :h2]], -1) * s

    def forward(self, x, cos, sin, m_in, cfg, aux):
        B, T, D = x.shape
        h1 = self.ln1(x)
        m_qkv = m_in
        if cfg.src == "h":
            m_qkv, _ = mask_from_score(h1.detach().abs(), cfg.keep)
        h1m = h1 if m_qkv is None else h1 * m_qkv

        q = self.q(h1m).view(B, T, self.h, self.hd).transpose(1, 2)
        k = self.k(h1m).view(B, T, self.h, self.hd).transpose(1, 2)
        v = self.v(h1m).view(B, T, self.h, self.hd).transpose(1, 2)
        if cfg.src in ("q", "q-ste"):
            # per-channel |Q| in the SAME index space as the residual (square q_proj)
            qc = q.transpose(1, 2).reshape(B, T, D).detach().abs()
            m_ffn, p = mask_from_score(qc, cfg.keep, ste=(cfg.src == "q-ste"))
            if p is not None:
                aux.append((p, m_ffn - p + p.detach()))   # (soft prob, hard mask)
        else:
            m_ffn = m_qkv          # src=h: one mask, same layer, reused by FFN
        self.last_mask = None if m_ffn is None else m_ffn.detach()
        if cfg.track and m_ffn is not None:
            if self.usage_h is None:
                self.usage_h = torch.zeros(D)
            self.usage_h += m_ffn.detach().sum(dim=(0, 1)).cpu()

        q = self._rope(q, cos, sin)
        k = self._rope(k, cos, sin)
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        xa = x + self.o(att.transpose(1, 2).reshape(B, T, D))

        h2 = self.ln2(xa)
        if m_ffn is not None:
            h2 = h2 * m_ffn
        act = F.silu(self.gate(h2)) * self.up(h2)
        if cfg.src != "none":
            mid, _ = mask_from_score(act.detach().abs(), cfg.keep_mid)
            if cfg.track:
                if self.usage_mid is None:
                    self.usage_mid = torch.zeros(mid.shape[-1])
                self.usage_mid += mid.detach().sum(dim=(0, 1)).cpu()
            if cfg.track_values:
                self.last_H = {"h": norm_entropy(h1), "mid": norm_entropy(act)}
            act = act * mid
        elif cfg.track_values:
            self.last_H = {"h": norm_entropy(h1), "mid": norm_entropy(act)}
        return xa + self.down(act), m_ffn


class DynLM(nn.Module):
    def __init__(self, vocab, d, layers, heads, hidden, rope_theta=10000.0):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.layers = nn.ModuleList([Block(d, heads, hidden) for _ in range(layers)])
        self.norm = RMSNorm(d)
        self.head_mask = None
        hd = d // heads
        inv = rope_theta ** (-torch.arange(0, hd, 2).float() / hd)
        pos = torch.arange(4096).float()[:, None]
        ang = pos * inv[None, :]
        emb = torch.cat([ang, ang], -1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, idx, cfg, collect=False):
        aux = []
        x = self.emb(idx)
        masks = []
        m = None
        for b in self.layers:
            x, m = b(x, self.cos, self.sin, m, cfg, aux)
            if collect:
                masks.append(b.last_mask)
        xn = self.norm(x)
        if cfg.src != "none":
            if cfg.src == "h":
                hm, _ = mask_from_score(xn.detach().abs(), cfg.keep)
            else:
                hm = m
            if hm is not None:
                xn = xn * hm
            self.head_mask = None if hm is None else hm
            if collect:
                masks.append(self.head_mask)
        logits = xn @ self.emb.weight.t()
        if not collect:
            return logits, aux
        return logits, aux, masks


def macs_per_token(vocab, d, layers, hidden, seq, keep_h=1.0, keep_mid=1.0):
    per_layer = 4 * d * d + 3 * d * hidden
    core = 2 * (seq // 2) * d
    base = layers * (per_layer + core) + vocab * d
    saved = (1 - keep_h) * (layers * (3 * d * d + 2 * d * hidden) + vocab * d)
    saved += (1 - keep_mid) * layers * d * hidden
    return base, base - saved


def get_batch(train, cursor, batch, seq):
    n = len(train) - seq - 1
    out = np.empty((batch, seq), dtype=np.int64)
    for b in range(batch):
        base = (cursor + b * seq) % n
        out[b] = train[base:base + seq]
    return torch.from_numpy(out), (cursor + batch * seq) % n


def get_val_batches(val, batch, seq, eval_batches=4):
    n = len(val) - seq - 1
    out = np.empty((eval_batches, batch, seq), dtype=np.int64)
    for eb in range(eval_batches):
        for b in range(batch):
            base = (eb * 977 + b * seq * 13) % n
            out[eb, b] = val[base:base + seq]
    return torch.from_numpy(out.reshape(eval_batches, batch, seq))


@torch.no_grad()
def eval_val(model, vb, seq, cfg):
    tot, n = 0.0, 0
    for i in range(vb.shape[0]):
        x = vb[i]
        logits, _ = model(x, cfg)
        loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.shape[-1]),
                               x[:, 1:].reshape(-1), reduction="sum")
        tot += float(loss)
        n += x.shape[0] * (seq - 1)
    return tot / n


def _mask_patterns(m):
    """[N,D] mask -> (bool ndarray, distinct patterns, counts)."""
    xb = m.detach().cpu().numpy().astype(np.bool_).reshape(-1, m.shape[-1])
    packed = np.packbits(xb, axis=1)
    uniq, counts = np.unique(packed, axis=0, return_counts=True)
    return xb, uniq, counts


def _jac_pairs(xb, n_pairs=200, chunk=128, seed=0):
    """采样 token 对：同块(同一序列内) vs 跨块 的掩码 Jaccard。"""
    N = xb.shape[0]
    rng = np.random.default_rng(seed)
    nch = max(1, N // chunk)
    c = rng.integers(0, nch, n_pairs)
    i = c * chunk + rng.integers(0, chunk, n_pairs)
    j = c * chunk + rng.integers(0, chunk, n_pairs)
    c1 = rng.integers(0, nch, n_pairs)
    c2 = (c1 + rng.integers(1, nch, n_pairs)) % nch
    i2 = c1 * chunk + rng.integers(0, chunk, n_pairs)
    j2 = c2 * chunk + rng.integers(0, chunk, n_pairs)

    def jac(ii, jj):
        a, b = xb[ii], xb[jj]
        inter = (a & b).sum(1).astype(np.float64)
        union = np.maximum((a | b).sum(1).astype(np.float64), 1.0)
        return float((inter / union).mean())

    return jac(i, j), jac(i2, j2)


def mask_structure(m):
    """Distinguish 'domain-specific combinations' from 'global collapse'.

    n_pat      how many DISTINCT masks the batch uses.  If tokens in a domain
               share one combination, n_pat is small but > 1 and top_share is
               high; if every token picks its own set, n_pat ~ N.
    pat_H      entropy of the pattern distribution / log(N)  (1 = all unique)
    top_share  share of tokens using the single most common pattern
    jac_w/jac_c  mean mask Jaccard of token pairs INSIDE one sequence vs ACROSS
               sequences.  Domain specialization predicts jac_w >> jac_c;
               global collapse predicts jac_w ~= jac_c ~= high.
    """
    xb, uniq, counts = _mask_patterns(m)
    N = xb.shape[0]
    p = counts.astype(np.float64) / N
    pat_H = float(-(p * np.log(p)).sum() / np.log(N))
    top = float(p.max())
    jw, jc = _jac_pairs(xb)
    return dict(n_pat=int(len(uniq)), pat_H=pat_H, top_share=top,
                jac_w=jw, jac_c=jc, jac_ratio=jw / (jc + 1e-9))


@torch.no_grad()
def mask_stats(model, x, cfg):
    """Metrics for dynamic activation sparsity.

    H_mask   normalized entropy of the cumulative per-channel activation counts,
             -sum p log p / log(D).  1.0 = every channel used equally often.
    dead_*   fraction of channels/neurons NEVER selected over the whole run.
    H_val_*  "数值差异熵": normalized entropy of the per-token |value| distribution
             at the two mask inputs; low = values themselves are concentrated.
             Also reported as EFFECTIVE DIMS = D^H (how many dimensions actually
             carry non-trivial energy) so it can be compared with the kept m.
    struct_* mask-pattern statistics: see mask_structure(); a random-mask baseline
             is measured with the same keep so 'differentiation' is normalised.
    """
    if cfg.src == "none":
        return {}
    prev = cfg.track_values
    cfg.track_values = True
    _, _, masks = model(x, cfg, collect=True)
    cfg.track_values = prev

    def usage_stats(u):
        u = u.float()
        p = u / u.sum()
        p = p[p > 0]
        H = float(-(p * p.log()).sum() / math.log(u.numel()))
        return H, float((u == 0).float().mean())

    out = {"per_layer_union": [], "per_layer_used": [], "head_union": 0.0}
    for m in masks:
        if m is None:
            out["per_layer_union"].append(0.0)
            out["per_layer_used"].append(0.0)
            continue
        B, T, D = m.shape
        out["per_layer_union"].append((m.reshape(-1, D).sum(0) > 0).float().mean().item())
        out["per_layer_used"].append(float(m.mean()))
    out["head_union"] = out["per_layer_union"][-1]
    out["union_mean"] = (float(np.mean(out["per_layer_union"][:-1]))
                         if len(out["per_layer_union"]) > 1 else 0.0)
    out["per_layer_union"] = [round(v, 4) for v in out["per_layer_union"]]
    out["per_layer_used"] = [round(v, 4) for v in out["per_layer_used"]]

    # ---- mask-pattern structure（每层最后两个 mask 是 mid / head，跳过头掩码）----
    st = [mask_structure(m) for m in masks if m is not None]
    # 随机掩码基线（同 keep、同形状），用于区分「分化」与「本来就随机」
    rnd = None
    if st:
        D = masks[0].shape[-1]
        keep = cfg.keep
        rs = torch.rand(x.shape[0], x.shape[1], D)
        rm, _ = mask_from_score(rs, keep)
        rnd = mask_structure(rm)

    Hs, dead_h, Hmid, dead_mid, Hv_h, Hv_mid = [], [], [], [], [], []
    for b in model.layers:
        if b.usage_h is not None:
            h, d = usage_stats(b.usage_h)
            Hs.append(h)
            dead_h.append(d)
        if b.usage_mid is not None:
            h, d = usage_stats(b.usage_mid)
            Hmid.append(h)
            dead_mid.append(d)
        if b.last_H:
            Hv_h.append(b.last_H.get("h", 0.0))
            Hv_mid.append(b.last_H.get("mid", 0.0))

    def mean(v):
        return float(np.mean(v)) if v else 0.0
    Dh = model.layers[0].d
    Ih = model.layers[0].gate.weight.shape[0]
    out.update(dict(
        H_mask=mean(Hs), dead_h=mean(dead_h),
        H_mid=mean(Hmid), dead_mid=mean(dead_mid),
        H_val_h=mean(Hv_h), H_val_mid=mean(Hv_mid),
        eff_dim_h=float(Dh ** mean(Hv_h)) if Hv_h else 0.0,
        eff_dim_mid=float(Ih ** mean(Hv_mid)) if Hv_mid else 0.0,
        kept_h=int(round(cfg.keep * Dh)), kept_mid=int(round(cfg.keep_mid * Ih)),
        n_pat=mean([s["n_pat"] for s in st]),
        pat_H=mean([s["pat_H"] for s in st]),
        top_share=mean([s["top_share"] for s in st]),
        jac_w=mean([s["jac_w"] for s in st]),
        jac_c=mean([s["jac_c"] for s in st]),
        jac_ratio=mean([s["jac_ratio"] for s in st]),
        rnd_n_pat=(rnd["n_pat"] if rnd else 0),
        rnd_pat_H=(rnd["pat_H"] if rnd else 0.0),
        rnd_top_share=(rnd["top_share"] if rnd else 0.0),
        rnd_jac_w=(rnd["jac_w"] if rnd else 0.0),
        rnd_jac_c=(rnd["jac_c"] if rnd else 0.0),
        rnd_jac_ratio=(rnd["jac_ratio"] if rnd else 0.0),
        H_mask_layers=[round(v, 4) for v in Hs],
        dead_h_layers=[round(v, 4) for v in dead_h],
        H_mid_layers=[round(v, 4) for v in Hmid],
        dead_mid_layers=[round(v, 4) for v in dead_mid],
        H_val_h_layers=[round(v, 4) for v in Hv_h],
        H_val_mid_layers=[round(v, 4) for v in Hv_mid],
        n_pat_layers=[s["n_pat"] for s in st],
        jac_ratio_layers=[round(s["jac_ratio"], 3) for s in st],
    ))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=0)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=488)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--mask-src", default="none", choices=["none", "h", "q", "q-ste"])
    ap.add_argument("--keep", type=float, default=1.0)
    ap.add_argument("--keep-mid", type=float, default=0.0)
    ap.add_argument("--aux-coef", type=float, default=0.01)
    ap.add_argument("--tag", default="")
    ap.add_argument("--bench", type=int, default=0)
    args = ap.parse_args()
    args.hidden = args.hidden or 4 * args.d
    args.keep_mid = args.keep_mid or args.keep
    if args.mask_src == "none":
        args.keep = 1.0
        args.keep_mid = 1.0

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    meta = json.load(open(os.path.join(DATA, "meta.json"), encoding="utf-8"))
    vocab = int(meta["vocab_size"])
    train = load_ids(os.path.join(DATA, "train_ids.bin"))
    val = load_ids(os.path.join(DATA, "val_ids.bin"))

    model = DynLM(vocab, args.d, args.layers, args.heads, args.hidden)
    nparam = sum(p.numel() for p in model.parameters())
    mac_dense, mac_eff = macs_per_token(vocab, args.d, args.layers, args.hidden, args.seq,
                                        args.keep, args.keep_mid)

    class C:
        pass
    cfg = C()
    cfg.src, cfg.keep, cfg.keep_mid = args.mask_src, args.keep, args.keep_mid
    cfg.track = args.mask_src != "none"    # accumulate usage counts every step
    cfg.track_values = False               # value entropies only on the eval forward

    print("[cfg] V=%d d=%d L=%d heads=%d hidden=%d src=%s keep=%.4f keep_mid=%.4f"
          % (vocab, args.d, args.layers, args.heads, args.hidden, args.mask_src,
             args.keep, args.keep_mid))
    print("[cfg] params=%.3fM MAC/tok dense=%.3fM effective=%.3fM (%.2fx) seq=%d batch=%d"
          % (nparam / 1e6, mac_dense / 1e6, mac_eff / 1e6, mac_dense / mac_eff,
             args.seq, args.batch))

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    vb = get_val_batches(val, args.batch, args.seq)
    n = len(train) - args.seq - 1
    cursor = (args.seed * 977) % n

    def lr_at(t, total):
        if t < args.warmup:
            return args.lr * (t + 1) / args.warmup
        p = (t - args.warmup) / max(1, total - args.warmup)
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + np.cos(np.pi * p))

    def step_loss(x):
        logits, aux = model(x, cfg)
        ce = F.cross_entropy(logits[:, :-1, :].reshape(-1, vocab), x[:, 1:].reshape(-1))
        al = torch.zeros(())
        if aux and args.aux_coef > 0:
            D = args.d
            terms = []
            for p, hard in aux:
                f = hard.mean(dim=(0, 1))
                q = p.mean(dim=(0, 1))
                terms.append(D * (f * q).sum())
            al = torch.stack(terms).mean() * args.aux_coef
        return ce, al, logits

    if args.bench:
        t0 = time.time()
        for _ in range(args.bench):
            x, cursor = get_batch(train, cursor, args.batch, args.seq)
            ce, al, _ = step_loss(x)
            opt.zero_grad(set_to_none=True)
            (ce + al).backward()
            for p in model.parameters():
                if p.grad is not None:
                    g = p.grad
                    nrm = float(g.norm())
                    if nrm > args.clip:
                        g.mul_(args.clip / (nrm + 1e-6))
            opt.step()
        dt = time.time() - t0
        toks = args.bench * args.batch * args.seq
        print("[bench] %d steps %.2fs = %.3f s/step, %.1f tok/s, eff %.1f GMAC/s, ce=%.4f"
              % (args.bench, dt, dt / args.bench, toks / dt,
                 toks * mac_eff / dt / 1e9, float(ce)))
        return 0

    tag = args.tag or ("%s_k%.3f_d%d" % (args.mask_src, args.keep, args.d))
    os.makedirs(OUT, exist_ok=True)
    mpath = os.path.join(OUT, tag + ".jsonl")
    t_start = time.time()
    with open(mpath, "w", encoding="utf-8") as f:
        f.write(json.dumps(dict(kind="cfg", tag=tag, vocab=vocab, d=args.d,
                                layers=args.layers, heads=args.heads, hidden=args.hidden,
                                params=nparam, macs_dense=mac_dense, macs_eff=mac_eff,
                                mask_src=args.mask_src, keep=args.keep,
                                keep_mid=args.keep_mid, seq=args.seq, batch=args.batch,
                                steps=args.steps, lr=args.lr, min_lr=args.min_lr,
                                warmup=args.warmup, clip=args.clip, seed=args.seed,
                                aux_coef=args.aux_coef)) + "\n")
    print("[run] tag=%s steps=%d -> %s" % (tag, args.steps, mpath), flush=True)
    val_loss = float("nan")
    for st in range(1, args.steps + 1):
        lr = lr_at(st - 1, args.steps)
        for g in opt.param_groups:
            g["lr"] = lr
        x, cursor = get_batch(train, cursor, args.batch, args.seq)
        ce, al, _ = step_loss(x)
        opt.zero_grad(set_to_none=True)
        (ce + al).backward()
        for p in model.parameters():
            if p.grad is not None:
                g = p.grad
                nrm = float(g.norm())
                if nrm > args.clip:
                    g.mul_(args.clip / (nrm + 1e-6))
        opt.step()
        if st % 50 == 0 or st == args.steps:
            print("  step %5d lr %.2e train %.4f wall %.1fs"
                  % (st, lr, float(ce), time.time() - t_start), flush=True)
        if st % args.eval_every == 0 or st == args.steps:
            t_e = time.time()
            val_loss = eval_val(model, vb, args.seq, cfg)
            stats = mask_stats(model, vb[0], cfg)
            rec = dict(kind="eval", step=st, train_loss=float(ce), val_loss=val_loss,
                       val_ppl=float(np.exp(val_loss)), wall_s=time.time() - t_start,
                       eval_s=time.time() - t_e, mask=stats,
                       tok_per_s=(st * args.batch * args.seq) / (time.time() - t_start))
            with open(mpath, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print("  [val] step %d val_loss=%.4f ppl=%.2f %.1f tok/s%s"
                  % (st, val_loss, float(np.exp(val_loss)), rec["tok_per_s"],
                     "" if not stats else
                     "  H=%.3f dead=%.3f | Hmid=%.3f | Hval=%.3f/%.3f eff=%.0f/%.0f kept=%d/%d"
                     " | pat=%.3f npat=%.0f top=%.3f jac=%.3f(w%.3f/c%.3f) union=%.3f"
                     % (stats.get("H_mask", 0), stats.get("dead_h", 0),
                        stats.get("H_mid", 0),
                        stats.get("H_val_h", 0), stats.get("H_val_mid", 0),
                        stats.get("eff_dim_h", 0), stats.get("eff_dim_mid", 0),
                        stats.get("kept_h", 0), stats.get("kept_mid", 0),
                        stats.get("pat_H", 0), stats.get("n_pat", 0),
                        stats.get("top_share", 0), stats.get("jac_ratio", 0),
                        stats.get("jac_w", 0), stats.get("jac_c", 0),
                        stats.get("union_mean", 0))),
                  flush=True)
    wall = time.time() - t_start
    with open(mpath, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(kind="final", step=args.steps, final_val=val_loss,
                                wall_s=wall,
                                tok_per_s=args.steps * args.batch * args.seq / wall,
                                tokens_seen=args.steps * args.batch * args.seq)) + "\n")
    print("[done] %s %.1f min %.1f tok/s final_val=%.6f"
          % (tag, wall / 60, args.steps * args.batch * args.seq / wall, val_loss))
    return 0


if __name__ == "__main__":
    sys.exit(main())
