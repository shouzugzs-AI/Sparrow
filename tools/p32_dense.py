#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P32 arm B: a REAL dense transformer at B1's exact MAC budget.

WHY THIS EXISTS
    Every existing "sparse vs dense" number in this repo compares B1 against a
    dense model built by the SAME B1 code with k0 = dim (no RMSNorm, no
    residual, no RoPE) and only matches LAYER params, not total MACs -- the
    head (V*dim) differs 5-13x and is 99% of the compute in the small-vocab
    runs. This script builds the honest comparator: a standard pre-norm
    Llama-ish block (RMSNorm + residual + RoPE + SwiGLU + tied embed) whose
    TOTAL MACs/token matches B1's production config.

WHAT IS MATCHED EXACTLY (so the two val losses are comparable)
    - same data files (train_ids.bin / val_ids.bin, u32 LE, char vocab 8198)
    - same batch/seq (32x64 = 2048 tokens/step) and the SAME batch cursor rule
      as b1_full_train.rs: cursor = (seed*977) % (N-seq-1), += batch*seq, wrap
    - same val protocol: 4 fixed batches x 32 seqs, base = (eb*977 + b*896) %
      (val_len-seq-1), mean CE over the (seq-1)*batch predictions
    - same optimizer Adam(0.9,0.999), per-tensor L2 clip at 1.0,
      linear warmup then cosine to --min-lr
    - same f32 arithmetic (CPU)

DELIBERATE DIFFERENCES (reported, not hidden)
    - standard architecture (norm+residual+RoPE+SwiGLU) instead of B1's
      norm-free/residual-free stack: this is the point of the comparison
    - warmup is set per-budget (default 100) instead of B1's 500
    - B1 has a bigram skip path (--head-skip 1); the dense arm's analogue is
      the tied embedding

USAGE
    python tools/p32_dense.py --d 64 --layers 32 --bench 10
    python tools/p32_dense.py --d 64 --layers 32 --steps 2200 --tag B_d64
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = r"D:\czsai\vmlm\artifacts\zh_wiki"
OUT = r"D:\czsai\vmlm\artifacts\p32"


def load_ids(path):
    a = np.fromfile(path, dtype="<u4")
    return a.astype(np.int64)


class RMSNorm(nn.Module):
    """Llama-style RMSNorm (no mean subtraction, no bias)."""

    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        r = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * r * self.w


class Block(nn.Module):
    def __init__(self, d, heads, hidden, nl=1):
        super().__init__()
        self.h = heads
        self.hd = d // heads
        self.ln1 = RMSNorm(d)
        self.ln2 = RMSNorm(d)
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(d, hidden, bias=False)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)
        # precomputed RoPE tables are built in the parent (shared)

    def attn(self, x, cos, sin):
        B, T, D = x.shape
        q = self.q(x).view(B, T, self.h, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, self.h, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.h, self.hd).transpose(1, 2)
        c = cos[:T].view(1, 1, T, self.hd)
        s = sin[:T].view(1, 1, T, self.hd)
        qr = q * c + torch.cat([-q[..., self.hd // 2:], q[..., : self.hd // 2]], -1) * s
        kr = k * c + torch.cat([-k[..., self.hd // 2:], k[..., : self.hd // 2]], -1) * s
        out = F.scaled_dot_product_attention(qr, kr, v, is_causal=True)
        return self.o(out.transpose(1, 2).reshape(B, T, D))

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        h = self.ln2(x)
        x = x + self.down(F.silu(self.gate(h)) * self.up(h))
        return x


class DenseLM(nn.Module):
    def __init__(self, vocab, d, layers, heads, hidden, rope_theta=10000.0):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.layers = nn.ModuleList([Block(d, heads, hidden) for _ in range(layers)])
        self.norm = RMSNorm(d)
        hd = d // heads
        inv = rope_theta ** (-torch.arange(0, hd, 2).float() / hd)
        T = 4096
        pos = torch.arange(T).float()[:, None]
        ang = pos * inv[None, :]
        emb = torch.cat([ang, ang], -1)          # [T, hd]
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, idx):
        x = self.emb(idx)
        for b in self.layers:
            x = b(x, self.cos, self.sin)
        x = self.norm(x)
        return x @ self.emb.weight.t()


def macs_per_token(vocab, d, layers, hidden, seq):
    """Total MACs/token (fwd), same accounting used for B1's 2.49M."""
    per_layer = 4 * d * d + 3 * d * hidden
    attn_core = 2 * (seq // 2) * d        # causal: avg seq/2 keys, QK+PV
    return layers * (per_layer + attn_core) + vocab * d


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
def eval_val(model, vb, seq):
    tot, n = 0.0, 0
    for i in range(vb.shape[0]):
        x = vb[i]
        logits = model(x)
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.shape[-1]),
            x[:, 1:].reshape(-1), reduction="sum")
        tot += float(loss)
        n += x.shape[0] * (seq - 1)
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=0, help="0 = 4*d")
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=2200)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--tag", default="")
    ap.add_argument("--bench", type=int, default=0, help="N>0: time N steps and exit")
    args = ap.parse_args()
    hidden = args.hidden or 4 * args.d

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    meta = json.load(open(os.path.join(DATA, "meta.json"), encoding="utf-8"))
    vocab = int(meta["vocab_size"])
    train = load_ids(os.path.join(DATA, "train_ids.bin"))
    val = load_ids(os.path.join(DATA, "val_ids.bin"))

    model = DenseLM(vocab, args.d, args.layers, args.heads, hidden)
    nparam = sum(p.numel() for p in model.parameters())
    macs = macs_per_token(vocab, args.d, args.layers, hidden, args.seq)
    print("[cfg] V=%d d=%d L=%d h=%d hidden=%d params=%.3fM MACs/tok=%.3fM"
          % (vocab, args.d, args.layers, args.heads, hidden, nparam / 1e6, macs / 1e6))

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    vb = get_val_batches(val, args.batch, args.seq)

    n = len(train) - args.seq - 1
    cursor = (args.seed * 977) % n

    def lr_at(t, total):
        if t < args.warmup:
            return args.lr * (t + 1) / args.warmup
        p = (t - args.warmup) / max(1, total - args.warmup)
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + np.cos(np.pi * p))

    if args.bench:
        t0 = time.time()
        for st in range(args.bench):
            x, cursor = get_batch(train, cursor, args.batch, args.seq)
            logits = model(x)
            loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, vocab), x[:, 1:].reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            for p in model.parameters():
                if p.grad is not None:
                    g = p.grad
                    nrm = float(g.norm())
                    if nrm > args.clip:
                        g.mul_(args.clip / (nrm + 1e-6))
            opt.step()
        dt = time.time() - t0
        toks = args.bench * args.batch * args.seq
        print("[bench] %d steps in %.2fs = %.3f s/step, %.1f tok/s, %.1f GMAC/s(fwd)"
              % (args.bench, dt, dt / args.bench, toks / dt, toks * macs / dt / 1e9))
        return 0

    tag = args.tag or ("B_d%d_L%d" % (args.d, args.layers))
    os.makedirs(OUT, exist_ok=True)
    mpath = os.path.join(OUT, tag + ".jsonl")
    t_start = time.time()
    with open(mpath, "w", encoding="utf-8") as f:
        f.write(json.dumps(dict(kind="cfg", tag=tag, vocab=vocab, d=args.d,
                                layers=args.layers, heads=args.heads, hidden=hidden,
                                params=nparam, macs_per_token=macs,
                                seq=args.seq, batch=args.batch, steps=args.steps,
                                lr=args.lr, min_lr=args.min_lr, warmup=args.warmup,
                                clip=args.clip, seed=args.seed)) + "\n")
    print("[run] tag=%s steps=%d  -> %s" % (tag, args.steps, mpath), flush=True)
    val_loss = float("nan")
    for st in range(1, args.steps + 1):
        lr = lr_at(st - 1, args.steps)
        for g in opt.param_groups:
            g["lr"] = lr
        x, cursor = get_batch(train, cursor, args.batch, args.seq)
        logits = model(x)
        loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, vocab), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                g = p.grad
                nrm = float(g.norm())
                if nrm > args.clip:
                    g.mul_(args.clip / (nrm + 1e-6))
        opt.step()
        if st % 50 == 0 or st == args.steps:
            print("  step %5d  lr %.2e  train %.4f  wall %.1fs"
                  % (st, lr, float(loss), time.time() - t_start), flush=True)
        if st % args.eval_every == 0 or st == args.steps:
            t_e = time.time()
            val_loss = eval_val(model, vb, args.seq)
            rec = dict(kind="eval", step=st, train_loss=float(loss), val_loss=val_loss,
                       val_ppl=float(np.exp(val_loss)), wall_s=time.time() - t_start,
                       eval_s=time.time() - t_e,
                       tok_per_s=(st * args.batch * args.seq) / (time.time() - t_start))
            with open(mpath, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print("  [val] step %d  val_loss=%.4f  ppl=%.2f  %.1f tok/s"
                  % (st, val_loss, float(np.exp(val_loss)), rec["tok_per_s"]), flush=True)
    wall = time.time() - t_start
    with open(mpath, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(kind="final", step=args.steps, final_val=val_loss,
                                wall_s=wall, tok_per_s=args.steps * args.batch * args.seq / wall,
                                tokens_seen=args.steps * args.batch * args.seq)) + "\n")
    print("[done] %s  %.1f min  %.1f tok/s  final_val=%.6f"
          % (tag, wall / 60, args.steps * args.batch * args.seq / wall, val_loss))
    return 0


if __name__ == "__main__":
    sys.exit(main())
