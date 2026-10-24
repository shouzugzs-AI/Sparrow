"""P36C: 上下文接口（context interface）—— 把「注意力深度」从主干宽度里拆出来。

设计（用户提案，2026-10-23）：
  主干是纯逐 token 的 FFN 栈（没有 q/k/v）；token 混合全部交给**前置的一叠窄注意力**：
  嵌入 -> (d -> c) -> N 层串行因果注意力（每层可选 FFN）-> 每个主干隐藏层一份 c->d 的读出
  -> 加到该层残差流上。也就是"注意力是一个对接口，主干是纯全连接"，注意力深度可以远大于
  主干层数而参数极小（c=128 时每层注意力只有 4c^2=65k 参数）。

  这和 p36（attention-once）的区别：p36 只在前段/间隔处做一次全宽注意力，这里做 N 次窄注意力，
  且读到的是同一个 ctx 状态；主干层数只影响读出（L*c*d）不影响注意力参数。

  python tools\p36_ctx.py --ctx-layers 6 --ctx-dim 128 --ctx-ffn 1 --ctx-out perlayer          --attn-every 12 --no-backbone-attn 1 --steps 244 --tag ctx6

【原始 attention-once 说明】

P36: "attention-once" 1-D stack.

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


class CtxInterface(nn.Module):
    """一叠窄的串行因果注意力，作为「上下文接口」。

    x [B,T,d] -> inp [B,T,c] -> N x (causal attn + 可选 FFN) -> ctx [B,T,c]
    然后由 outs[li] 把 ctx 读到第 li 个主干层的宽度 d 上（perlayer）或共用一份（shared）。
    参数：d*c + N*(4c^2 [+ 2*c*hc]) + L*c*d(或 c*d)
    """

    def __init__(self, d, c, n_layers, heads, ffn=1, n_out=0, out_mode="perlayer",
                 rope_theta=10000.0, mix="attn", out_kind="linear"):
        super().__init__()
        assert c % heads == 0, "ctx_dim 必须能被 ctx_heads 整除"
        self.d, self.c, self.h, self.hd = d, c, heads, c // heads
        self.inp = nn.Linear(d, c, bias=False)
        self.mix = mix
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict(dict(
                ln1=RMSNorm(c),
                attn=CausalAttn(c, heads) if mix == "attn" else StateMix(c),
                ln2=RMSNorm(c) if ffn else None,
                up=nn.Linear(c, 4 * c, bias=False) if ffn else None,
                down=nn.Linear(4 * c, c, bias=False) if ffn else None,
            )))
        self.out_mode = out_mode
        def make_out(kind):
            if kind == "mlp":
                return nn.Sequential(nn.Linear(c, 2 * c, bias=False), nn.SiLU(),
                                     nn.Linear(2 * c, d, bias=False))
            return nn.Linear(c, d, bias=False)

        if out_mode == "perlayer":
            self.outs = nn.ModuleList([make_out(out_kind) for _ in range(n_out)])
        else:
            self.outs = nn.ModuleList([make_out(out_kind)])
        hd = c // heads
        inv = rope_theta ** (-torch.arange(0, hd, 2).float() / hd)
        pos = torch.arange(4096).float()[:, None]
        ang = pos * inv[None, :]
        emb = torch.cat([ang, ang], -1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, x):
        h = self.inp(x)
        for L in self.layers:
            if self.mix == "attn":
                h = h + L["attn"](L["ln1"](h), self.cos, self.sin)
            else:
                h = h + L["attn"](L["ln1"](h))
            if L["down"] is not None:
                h = h + L["down"](F.silu(L["up"](L["ln2"](h))))
        return h

    def read(self, li):
        return self.outs[li] if self.out_mode == "perlayer" else self.outs[0]


class StateMix(nn.Module):
    """用「状态向量」代替注意力做 token 混合（用户提案 2 / 兼容 SSM-SNN 的那条路）。

    h_t = a ⊙ h_{t-1} + (1-a) ⊙ W x_t,    a = sigmoid(p) ∈ (0,1) 每通道一个
    p 按 log 间隔初始化 → 同时存在很多时间尺度（短记忆/长记忆通道）。
    成本 c^2 + c^2 + c per token，与序列长度 T 无关；推理只需保存 h（常数内存），
    不像注意力要 KV cache。
    """

    def __init__(self, c):
        super().__init__()
        self.c = c
        self.w = nn.Linear(c, c, bias=False)
        self.o = nn.Linear(c, c, bias=False)
        self.p = nn.Parameter(torch.linspace(-4.0, 2.5, c))

    def forward(self, x):
        a = torch.sigmoid(self.p)                       # [c]
        xs = self.w(x)
        h = torch.zeros_like(xs[:, 0])
        outs = []
        for t in range(x.shape[1]):                     # 因果扫描（T 很小，够用）
            h = a * h + (1.0 - a) * xs[:, t]
            outs.append(h)
        return self.o(torch.stack(outs, 1))


class ReadBack(nn.Module):
    """回读：主干状态 x 作为 query，接口状态 ctx 作为 key/value（在 c 维里做）。

    这一步让「上下文接口」看得到主干演化中的表示，而不只是嵌入。
    权重跨层共享（只有每层的门不同），所以参数只有 d*c + 2c^2 + c*d。
    """

    def __init__(self, d, c, heads, rope_theta=10000.0):
        super().__init__()
        self.d, self.c, self.h, self.hd = d, c, heads, c // heads
        self.ln_x = RMSNorm(d)
        self.ln_c = RMSNorm(c)
        self.q = nn.Linear(d, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.o = nn.Linear(c, d, bias=False)
        hd = c // heads
        inv = rope_theta ** (-torch.arange(0, hd, 2).float() / hd)
        pos = torch.arange(4096).float()[:, None]
        ang = pos * inv[None, :]
        emb = torch.cat([ang, ang], -1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, x, ctx):
        B, T, _ = x.shape
        q = self.q(self.ln_x(x)).view(B, T, self.h, self.hd).transpose(1, 2)
        c = self.ln_c(ctx)
        k = self.k(c).view(B, T, self.h, self.hd).transpose(1, 2)
        v = self.v(c).view(B, T, self.h, self.hd).transpose(1, 2)
        q = CausalAttn._rope(self, q, self.cos, self.sin)
        k = CausalAttn._rope(self, k, self.cos, self.sin)
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(att.transpose(1, 2).reshape(B, T, self.c))


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
                 mlp="gated", route="none", route_gate="channel", rope_theta=10000.0,
                 ctx_layers=0, ctx_dim=128, ctx_heads=4, ctx_ffn=1,
                 ctx_out="perlayer", ctx_gate_init=0.05, attn_enable=1,
                 readback_every=0, rb_gate_init=0.05, ctx_mix="attn",
                 ctx_out_kind="linear"):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.ln0 = RMSNorm(d)
        n_attn = -(-layers // attn_every)
        # share_attn=1 -> ONE attention block whose weights are reused at every
        # refresh ("一个 attn 对接每一层"); share_attn=0 -> one per refresh,
        # which is the ordinary Transformer when attn_every=1.
        if not attn_enable:
            self.attns = nn.ModuleList([])          # 主干无注意力：token 混合只在 ctx 里
        elif share_attn:
            self.attns = nn.ModuleList([CausalAttn(d, heads)])
        else:
            self.attns = nn.ModuleList([CausalAttn(d, heads) for _ in range(n_attn)])
        self.layers = nn.ModuleList([Block(d, hidden, mlp=mlp, route_gate=route_gate)
                                     for _ in range(layers)])
        self.norm = RMSNorm(d)
        # --- 上下文接口：一叠窄注意力，输出接到每个主干层 ---
        self.ctx = None
        self.ctx_gate = None
        if ctx_layers > 0:
            self.ctx = CtxInterface(d, ctx_dim, ctx_layers, ctx_heads, ffn=ctx_ffn,
                                    n_out=layers, out_mode=ctx_out, mix=ctx_mix,
                                    out_kind=ctx_out_kind)
            # 每层每通道一个门，初值 small：接口一开始几乎不起作用，必须自己学出贡献
            self.ctx_gate = nn.Parameter(torch.full((layers, d), ctx_gate_init))
        self.rb = None
        self.rb_gate = None
        if ctx_layers > 0 and readback_every > 0:
            self.rb = ReadBack(d, ctx_dim, ctx_heads)
            self.rb_gate = nn.Parameter(torch.full((layers, d), rb_gate_init))
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
        C = self.ctx(self.ln0(x)) if self.ctx is not None else None
        ai = 0
        for li, b in enumerate(self.layers):
            if C is not None:
                x = x + self.ctx_gate[li] * self.ctx.read(li)(C)
            if self.rb is not None and li % cfg.rb_every == 0:
                x = x + self.rb_gate[li] * self.rb(x, C)
            if cfg.attn_enable and li % cfg.attn_every == 0:
                blk = self.attns[0] if cfg.share_attn else self.attns[ai]
                ai += 1
                A = blk(self.ln0(x), self.cos, self.sin)
                if cfg.inject == "add":
                    x = x + A
            x = b(x, A, M, cfg)      # ... and re-read by every layer
        return self.norm(x) @ self.emb.weight.t()


def macs_per_token(vocab, d, layers, hidden, seq, attn_every=1, mlp="gated",
                   route="none", ctx_layers=0, ctx_dim=0, ctx_ffn=1,
                   ctx_out="perlayer", attn_enable=1, readback_every=0,
                   ctx_mix="attn", ctx_out_kind="linear"):
    """(full-Transformer MAC, this-model MAC) per token, causal core = seq*d."""
    core = seq * d
    per_attn = 4 * d * d + core
    per_mlp = (3 if mlp == "gated" else 2) * d * hidden
    n_attn = -(-layers // attn_every) if attn_enable else 0
    total = n_attn * per_attn + layers * per_mlp + vocab * d
    if route == "full":          # build M once (seq*d/token) + M@h per layer
        total += (1 + layers) * seq * d
    if ctx_layers > 0:
        c = ctx_dim
        total += d * c                                    # 嵌入 -> ctx 维
        if ctx_mix == "attn":
            total += ctx_layers * (4 * c * c + seq * c)   # 串行窄注意力
        else:
            total += ctx_layers * 2 * c * c               # 状态向量（与 T 无关）
        if ctx_ffn:
            total += ctx_layers * 2 * c * (4 * c)         # ctx 内 FFN
        n_read = layers if ctx_out == "perlayer" else 1
        if ctx_out_kind == "mlp":
            total += n_read * (2 * c * c + 2 * c * d)     # 非线性适配器
        else:
            total += n_read * c * d                       # 线性读出
        if readback_every > 0:                                     # 回读
            n_rb = -(-layers // readback_every)
            total += n_rb * (2 * d * c + 2 * c * c + seq * c)
    tra = layers * (4 * d * d + core + per_mlp) + vocab * d
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
    ap.add_argument("--ctx-layers", type=int, default=0, help="上下文接口里串行窄注意力的层数")
    ap.add_argument("--ctx-dim", type=int, default=128, help="接口宽度 c")
    ap.add_argument("--ctx-heads", type=int, default=4)
    ap.add_argument("--ctx-ffn", type=int, default=1, help="1 = 每层接口内含 128->512->128 的 FFN")
    ap.add_argument("--ctx-out", default="perlayer", choices=["perlayer", "shared"],
                    help="最后一个接口层怎么接到主干：每层一份 c->d 读出，或共用一份")
    ap.add_argument("--ctx-gate-init", type=float, default=0.05)
    ap.add_argument("--ctx-mix", default="attn", choices=["attn", "state"],
                    help="接口里做 token 混合的算子：注意力 / 多时间尺度状态向量")
    ap.add_argument("--ctx-out-kind", default="linear", choices=["linear", "mlp"],
                    help="读出形式：线性 / 128->256->d 的非线性适配器")
    ap.add_argument("--readback-every", type=int, default=0,
                    help="每几层做一次「主干回读 ctx」的交叉注意力，0=关")
    ap.add_argument("--rb-gate-init", type=float, default=0.05)
    ap.add_argument("--no-backbone-attn", type=int, default=0,
                    help="1 = 主干完全不放注意力（纯逐 token FFN 栈）")
    ap.add_argument("--tag", default="")
    ap.add_argument("--ckpt", default="", help="缺省 <OUT>/<tag>.pt")
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--resume", type=int, default=0)
    ap.add_argument("--stop-at", type=int, default=0)
    args = ap.parse_args()
    args.hidden = args.hidden or 4 * args.d
    args.attn_every = max(1, min(args.attn_every, args.layers))

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    meta = json.load(open(os.path.join(DATA, "meta.json"), encoding="utf-8"))
    vocab = int(meta["vocab_size"])
    train = load_ids(os.path.join(DATA, "train_ids.bin"))
    val = load_ids(os.path.join(DATA, "val_ids.bin"))

    attn_enable = 0 if args.no_backbone_attn else 1
    if args.ctx_layers > 0 and args.ctx_dim % args.ctx_heads != 0:
        raise SystemExit("--ctx-dim 必须能被 --ctx-heads 整除")
    model = DynLM(vocab, args.d, args.layers, args.heads, args.hidden,
                  attn_every=args.attn_every, share_attn=args.share_attn,
                  mlp=args.mlp, route=args.route, route_gate=args.route_gate,
                  ctx_layers=args.ctx_layers, ctx_dim=args.ctx_dim,
                  ctx_heads=args.ctx_heads, ctx_ffn=args.ctx_ffn,
                  ctx_out=args.ctx_out, ctx_gate_init=args.ctx_gate_init,
                  attn_enable=attn_enable, readback_every=args.readback_every,
                  rb_gate_init=args.rb_gate_init, ctx_mix=args.ctx_mix,
                  ctx_out_kind=args.ctx_out_kind)
    nparam = sum(p.numel() for p in model.parameters())
    mac_tra, mac_eff = macs_per_token(vocab, args.d, args.layers, args.hidden,
                                      args.seq, args.attn_every, args.mlp, args.route,
                                      ctx_layers=args.ctx_layers, ctx_dim=args.ctx_dim,
                                      ctx_ffn=args.ctx_ffn, ctx_out=args.ctx_out,
                                      attn_enable=attn_enable,
                                      readback_every=args.readback_every,
                                      ctx_mix=args.ctx_mix,
                                      ctx_out_kind=args.ctx_out_kind)

    class C:
        pass
    cfg = C()
    cfg.attn_every, cfg.inject = args.attn_every, args.inject
    cfg.attn_enable = attn_enable
    cfg.rb_every = args.readback_every
    cfg.share_attn = int(bool(args.share_attn))
    cfg.mlp = args.mlp
    cfg.route = args.route

    print("[cfg] V=%d d=%d L=%d heads=%d hidden=%d attn_every=%d share=%d inject=%s"
          " mlp=%s route=%s/%s backbone_attn=%d"
          % (vocab, args.d, args.layers, args.heads, args.hidden,
             args.attn_every, args.share_attn, args.inject, args.mlp,
             args.route, args.route_gate, attn_enable))
    if args.ctx_layers:
        print("[cfg] ctx: layers=%d c=%d heads=%d ffn=%d out=%s gate_init=%.3f readback=%d"
              % (args.ctx_layers, args.ctx_dim, args.ctx_heads, args.ctx_ffn,
                 args.ctx_out, args.ctx_gate_init, args.readback_every))
        print("[cfg] ctx mix=%s out_kind=%s" % (args.ctx_mix, args.ctx_out_kind))
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
    ck_path = args.ckpt or os.path.join(OUT, tag + ".pt")
    start_st = 1
    mode = "w"
    if args.resume and os.path.exists(ck_path):
        sd = torch.load(ck_path, map_location="cpu", weights_only=False)
        model.load_state_dict(sd["model"])
        opt.load_state_dict(sd["opt"])
        start_st = int(sd["step"]) + 1
        cursor = int(sd["cursor"])
        mode = "a"
        print("[resume] %s step=%d cursor=%d" % (ck_path, start_st - 1, cursor), flush=True)
    t_start = time.time()
    with open(mpath, mode, encoding="utf-8") as f:
        if mode == "w":
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
    for st in range(start_st, args.steps + 1):
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
        if args.ckpt_every and (st % args.ckpt_every == 0 or st == args.steps):
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": st, "cursor": cursor}, ck_path)
        if args.stop_at and st >= args.stop_at:
            print("[stop] 到达 --stop-at %d" % args.stop_at, flush=True)
            break
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
