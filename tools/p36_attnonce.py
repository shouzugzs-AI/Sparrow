"""P36: "attention-once" 1-D stack.

Design under test (user proposal, 2026-10-21):
  * the whole body is a 1-D / per-token network: gated MLP, RMSNorm, residual;
    no q/k/v, no softmax, no causal mask, no positional encoding inside a layer;
  * token mixing happens in ONE shared causal attention that is computed every
    `attn_every` layers and injected as an external context vector into every
    layer (inject=add: added to the residual stream at refresh, so the residual
    carries it downstream; inject=gain: per-channel learnable gain per layer);
  * the output is projected with the input embedding matrix (weight-tied head).

So `--attn-every 1` reproduces the Transformer body exactly (same FLOPs as
p35's dense arm) and `--attn-every 12` is the pure "one attention + 12 MLPs"
limit.  Everything else (data, batching, eval, schedule) is the p35 protocol,
so losses are directly comparable with artifacts/p35.

  python tools\p36_attnonce.py --attn-every 12 --steps 244 --tag b_attonce
  python tools\p36_attnonce.py --attn-every 4  --steps 244 --tag b_every4
  python tools\p36_attnonce.py --attn-every 1  --steps 244 --tag b_tra
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
OUT = r"D:\czsai\vmlm\artifacts\p36"


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


class CausalAttn(nn.Module):
    """The only token-mixing component. Weights are SHARED across refreshes."""

    def __init__(self, d, heads):
        super().__init__()
        self.d, self.h, self.hd = d, heads, d // heads
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def _rope(self, x, cos, sin):
        T = x.shape[2]
        c = cos[:T].view(1, 1, T, self.hd)
        s = sin[:T].view(1, 1, T, self.hd)
        h2 = self.hd // 2
        return x * c + torch.cat([-x[..., h2:], x[..., :h2]], -1) * s

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        q = self.q(x).view(B, T, self.h, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, self.h, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.h, self.hd).transpose(1, 2)
        q = self._rope(q, cos, sin)
        k = self._rope(k, cos, sin)
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(att.transpose(1, 2).reshape(B, T, D))

    def matrix(self, x, cos, sin):
        """The shared N*N routing matrix, built ONCE from the model input.

        Row-stochastic (causal softmax, averaged over heads) so it can be reused
        as a token-mixing operator at every layer: y = M @ h.
        """
        B, T, D = x.shape
        q = self.q(x).view(B, T, self.h, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, self.h, self.hd).transpose(1, 2)
        q = self._rope(q, cos, sin)
        k = self._rope(k, cos, sin)
        s = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.hd)
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        s = s.masked_fill(~mask, float("-inf"))
        return torch.softmax(s, dim=-1).mean(1)      # [B,T,T]


class Block(nn.Module):
    """Pure per-token processor: x -> x + MLP(RMSNorm(x)).

    mlp=gated: silu(gate) * up          (3 d*I matrices, like p35)
    mlp=plain: gelu(up)                (2 d*I matrices, the "直接全连接" variant)
    """

    def __init__(self, d, hidden, mlp="gated", route_gate="channel"):
        super().__init__()
        self.d = d
        self.mlp = mlp
        self.ln1 = RMSNorm(d)
        self.ln2 = RMSNorm(d)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)
        if mlp == "gated":
            self.gate = nn.Linear(d, hidden, bias=False)
        self.inj = nn.Parameter(torch.zeros(d))   # context re-injection gain
        # gate of the shared N*N routing operator (per-channel by default)
        if route_gate == "channel":
            self.gmix = nn.Parameter(torch.full((d,), 0.1))
        else:
            self.gmix = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, A, M, cfg):
        if cfg.inject == "gain" and A is not None:
            x = x + self.inj * A
        h = self.ln1(x)
        if cfg.mlp == "gated":
            act = F.silu(self.gate(h)) * self.up(h)
        else:
            act = F.gelu(self.up(h))
        x = x + self.down(act)
        if M is not None:
            x = x + self.gmix * torch.matmul(M, self.ln2(x))
        return x


class DynLM(nn.Module):
    def __init__(self, vocab, d, layers, heads, hidden, attn_every=1, share_attn=1,
                 mlp="gated", route="none", route_gate="channel", rope_theta=10000.0):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.ln0 = RMSNorm(d)
        n_attn = -(-layers // attn_every)
        # share_attn=1 -> ONE attention block whose weights are reused at every
        # refresh ("一个 attn 对接每一层"); share_attn=0 -> one per refresh,
        # which is the ordinary Transformer when attn_every=1.
        if share_attn:
            self.attns = nn.ModuleList([CausalAttn(d, heads)])
        else:
            self.attns = nn.ModuleList([CausalAttn(d, heads) for _ in range(n_attn)])
        self.layers = nn.ModuleList([Block(d, hidden, mlp=mlp, route_gate=route_gate)
                                     for _ in range(layers)])
        self.norm = RMSNorm(d)
        hd = d // heads
        inv = rope_theta ** (-torch.arange(0, hd, 2).float() / hd)
        pos = torch.arange(4096).float()[:, None]
        ang = pos * inv[None, :]
        emb = torch.cat([ang, ang], -1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, idx, cfg):
        x = self.emb(idx)
        A = None
        M = None
        if cfg.route == "full":
            # one N*N routing matrix for the whole sequence, built once ...
            blk = self.attns[0]
            M = blk.matrix(self.ln0(x), self.cos, self.sin)
        ai = 0
        for li, b in enumerate(self.layers):
            if li % cfg.attn_every == 0:
                blk = self.attns[0] if cfg.share_attn else self.attns[ai]
                ai += 1
                A = blk(self.ln0(x), self.cos, self.sin)
                if cfg.inject == "add":
                    x = x + A
            x = b(x, A, M, cfg)      # ... and re-read by every layer
        return self.norm(x) @ self.emb.weight.t()


def macs_per_token(vocab, d, layers, hidden, seq, attn_every=1, mlp="gated",
                   route="none"):
    """(full-Transformer MAC, this-model MAC) per token, causal core = seq*d."""
    core = seq * d
    per_attn = 4 * d * d + core
    per_mlp = (3 if mlp == "gated" else 2) * d * hidden
    n_attn = -(-layers // attn_every)
    total = n_attn * per_attn + layers * per_mlp + vocab * d
    if route == "full":          # build M once (seq*d/token) + M@h per layer
        total += (1 + layers) * seq * d
    tra = layers * (per_attn + per_mlp) + vocab * d
    return tra, total


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
        logits = model(x, cfg)
        loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.shape[-1]),
                               x[:, 1:].reshape(-1), reduction="sum")
        tot += float(loss)
        n += x.shape[0] * (seq - 1)
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=0)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=244)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--attn-every", type=int, default=12)
    ap.add_argument("--share-attn", type=int, default=1)
    ap.add_argument("--inject", default="add", choices=["add", "gain"])
    ap.add_argument("--mlp", default="gated", choices=["gated", "plain"])
    ap.add_argument("--route", default="none", choices=["none", "full"])
    ap.add_argument("--route-gate", default="channel", choices=["channel", "scalar"])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    args.hidden = args.hidden or 4 * args.d
    args.attn_every = max(1, min(args.attn_every, args.layers))

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    meta = json.load(open(os.path.join(DATA, "meta.json"), encoding="utf-8"))
    vocab = int(meta["vocab_size"])
    train = load_ids(os.path.join(DATA, "train_ids.bin"))
    val = load_ids(os.path.join(DATA, "val_ids.bin"))

    model = DynLM(vocab, args.d, args.layers, args.heads, args.hidden,
                  attn_every=args.attn_every, share_attn=args.share_attn,
                  mlp=args.mlp, route=args.route, route_gate=args.route_gate)
    nparam = sum(p.numel() for p in model.parameters())
    mac_tra, mac_eff = macs_per_token(vocab, args.d, args.layers, args.hidden,
                                      args.seq, args.attn_every, args.mlp, args.route)

    class C:
        pass
    cfg = C()
    cfg.attn_every, cfg.inject = args.attn_every, args.inject
    cfg.share_attn = int(bool(args.share_attn))
    cfg.mlp = args.mlp
    cfg.route = args.route

    print("[cfg] V=%d d=%d L=%d heads=%d hidden=%d attn_every=%d share=%d inject=%s"
          " mlp=%s route=%s/%s"
          % (vocab, args.d, args.layers, args.heads, args.hidden,
             args.attn_every, args.share_attn, args.inject, args.mlp,
             args.route, args.route_gate))
    print("[cfg] params=%.3fM MAC/tok tra=%.3fM this=%.3fM (%.2fx) seq=%d batch=%d"
          % (nparam / 1e6, mac_tra / 1e6, mac_eff / 1e6, mac_tra / mac_eff,
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

    tag = args.tag or ("a%02d_%s_s%d" % (args.attn_every, args.inject, args.seed))
    os.makedirs(OUT, exist_ok=True)
    mpath = os.path.join(OUT, tag + ".jsonl")
    t_start = time.time()
    with open(mpath, "w", encoding="utf-8") as f:
        f.write(json.dumps(dict(kind="cfg", tag=tag, vocab=vocab, d=args.d,
                                layers=args.layers, heads=args.heads, hidden=args.hidden,
                                params=nparam, macs_tra=mac_tra, macs_eff=mac_eff,
                                attn_every=args.attn_every, inject=args.inject,
                                share_attn=args.share_attn, mlp=args.mlp,
                                route=args.route, route_gate=args.route_gate,
                                seq=args.seq, batch=args.batch, steps=args.steps,
                                lr=args.lr, min_lr=args.min_lr, warmup=args.warmup,
                                clip=args.clip, seed=args.seed)) + "\n")
    print("[run] tag=%s steps=%d -> %s" % (tag, args.steps, mpath), flush=True)
    val_loss = float("nan")
    for st in range(1, args.steps + 1):
        lr = lr_at(st - 1, args.steps)
        for g in opt.param_groups:
            g["lr"] = lr
        x, cursor = get_batch(train, cursor, args.batch, args.seq)
        logits = model(x, cfg)
        ce = F.cross_entropy(logits[:, :-1, :].reshape(-1, vocab), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        ce.backward()
        for p in model.parameters():
            if p.grad is not None:
                g = p.grad
                nrm = float(g.norm())
                if nrm > args.clip:
                    g.mul_(args.clip / (nrm + 1e-6))
        opt.step()
        if st % 50 == 0 or st == args.steps:
            print("  step %5d lr %.2e train %.4f wall %.1fs"
                  % (st, lr, float(ce.detach()), time.time() - t_start), flush=True)
        if st % args.eval_every == 0 or st == args.steps:
            val_loss = eval_val(model, vb, args.seq, cfg)
            rec = dict(kind="eval", step=st, train_loss=float(ce.detach()),
                       val_loss=val_loss, val_ppl=float(np.exp(val_loss)),
                       wall_s=time.time() - t_start,
                       tok_per_s=(st * args.batch * args.seq) / (time.time() - t_start))
            with open(mpath, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print("  [val] step %d val_loss=%.4f ppl=%.2f %.1f tok/s"
                  % (st, val_loss, float(np.exp(val_loss)), rec["tok_per_s"]), flush=True)
    wall = time.time() - t_start
    with open(mpath, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(kind="final", step=args.steps, final_val=val_loss,
                                wall_s=wall,
                                tok_per_s=args.steps * args.batch * args.seq / wall)) + "\n")
    print("[done] %s %.1f min %.1f tok/s final_val=%.6f"
          % (tag, wall / 60, args.steps * args.batch * args.seq / wall, val_loss))
    return 0


if __name__ == "__main__":
    sys.exit(main())
