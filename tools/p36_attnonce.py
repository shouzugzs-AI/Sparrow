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


def topk_mask(score, keep):
    """每 token 取 |score| 最大的 keep 比例通道，返回 0/1 掩码（不可导，与 p35 同做法）。"""
    n = score.shape[-1]
    k = max(1, int(round(n * float(keep))))
    if k >= n:
        return None
    idx = score.topk(k, dim=-1).indices
    m = torch.zeros_like(score)
    return m.scatter_(-1, idx, 1.0)


def eff_keep(n, keep, group):
    """组稀疏下的实际保留比例（组的整数化会让它偏离名义 keep）。"""
    group = max(1, int(group))
    if group == 1 or n % group != 0:
        return max(1, int(round(n * float(keep)))) / n
    G = n // group
    kg = max(1, int(round(keep * G)))
    return min(kg, G) * group / n


def topk_mask_grouped(score, keep, group):
    """按连续 group 个通道成组选择（生产动机见 p35）：组内 max 打分，选 top-k 组整组开。"""
    n = score.shape[-1]
    group = max(1, int(group))
    if group == 1 or n % group != 0:
        return topk_mask(score, keep)
    G = n // group
    kg = max(1, int(round(keep * G)))
    if kg >= G:
        return None
    sg = score.reshape(*score.shape[:-1], G, group).amax(dim=-1)      # [...,G]
    idx = sg.topk(kg, dim=-1).indices
    mg = torch.zeros_like(sg).scatter_(-1, idx, 1.0)
    return mg.unsqueeze(-1).expand(*sg.shape, group).reshape(score.shape)


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


class StateMix(nn.Module):
    """用「状态向量」代替 KV cache 做 token 混合（seq 线性 → 常数）。

    lin  : 线性注意力 = 把 KV 压缩成状态矩阵
           S_t = S_{t-1} + phi(k_t) v_t^T          [h, hd, hd]
           o_t = phi(q_t)^T S_t / (phi(q_t)^T z_t)  z_t = z_{t-1} + phi(k_t)
           KV cache 是 O(T*d)，状态是 O(d*hd)，与 T 无关；训练用分块（块内因果注意力 + 块间状态）。
    diag : 多时间尺度对角状态（SSM 味道）
           h_t = a⊙h_{t-1} + (1-a)⊙W x_t,  a = sigmoid(p)，p 按对数间隔初始化
           成本 3*c²/token，与 T 无关，推理只存 h。
    """

    def __init__(self, d, heads, kind="lin", chunk=128, p_lo=-4.0, p_hi=2.5,
                 block=1):
        super().__init__()
        self.d, self.h, self.hd = d, heads, d // heads
        self.kind, self.chunk = kind, chunk
        self.block = max(1, int(block))
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        if kind == "diag":
            # 对角（或块对角）状态：写入矩阵 + 每通道一个时间常数
            del self.k, self.v
            self.w = nn.Linear(d, d, bias=False)
            self.p = nn.Parameter(torch.linspace(p_lo, p_hi, d))
            if self.block > 1 and d % self.block == 0:
                nb, bs = d // self.block, self.block
                a0 = torch.sigmoid(self.p).view(nb, bs)
                A = torch.zeros(nb, bs, bs)
                A[:, torch.arange(bs), torch.arange(bs)] = a0
                self.A = nn.Parameter(A)
        elif kind in ("gla", "gla_dd"):
            # 矩阵状态 + 衰减：S_t = γ ⊙ S_{t-1} + φ(k_t) v_t^T
            self.p = nn.Parameter(torch.linspace(p_lo, p_hi, self.hd))
            if kind == "gla_dd":
                # 数据相关的门：γ_t = σ(W_g x_t)
                self.wg = nn.Linear(d, d, bias=True)
        elif kind in ("ssm", "ssm_g"):
            # 选择性 SSM（Mamba/S6 的简化版，全对角）
            #   a_t = exp(-softplus(Δ_t) ⊙ A)      输入决定步长，A 决定时间尺度
            #   h_t = a_t ⊙ h_{t-1} + (1-a_t) ⊙ W x_t
            del self.k, self.v
            self.w = nn.Linear(d, d, bias=False)
            self.wd = nn.Linear(d, d, bias=True)
            self.logA = nn.Parameter(torch.linspace(p_lo, p_hi, d))   # 多时间尺度 A
            nn.init.constant_(self.wd.bias, -2.0)                     # 初始 Δ 不太大
            if kind == "ssm_g":
                self.wg = nn.Linear(d, d, bias=False)                 # 输出门
        elif kind == "rnn":
            # 经典 RNN：h_t = tanh(A h_{t-1} + W x_t)，状态是完整 d×d 矩阵
            del self.k, self.v
            self.w = nn.Linear(d, d, bias=False)
            self.A = nn.Parameter(torch.empty(d, d))
            nn.init.uniform_(self.A, -1.0 / math.sqrt(d), 1.0 / math.sqrt(d))

    # ---------- 线性注意力（分块） ----------
    @staticmethod
    def phi(x):
        return F.elu(x) + 1.0

    def _lin(self, x):
        B, T, D = x.shape
        q = self.phi(self.q(x)).view(B, T, self.h, self.hd).transpose(1, 2)   # [B,h,T,hd]
        k = self.phi(self.k(x)).view(B, T, self.h, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.h, self.hd).transpose(1, 2)
        C = min(self.chunk, T)
        S = x.new_zeros(B, self.h, self.hd, self.hd)          # 状态矩阵
        z = x.new_zeros(B, self.h, self.hd)                   # 归一化累加
        outs = []
        for s in range(0, T, C):
            qc, kc, vc = q[:, :, s:s + C], k[:, :, s:s + C], v[:, :, s:s + C]
            n = qc.shape[2]
            intra = torch.matmul(qc, kc.transpose(-1, -2))     # [B,h,n,n]
            causal = torch.ones(n, n, dtype=torch.bool, device=x.device).tril()
            intra = intra.masked_fill(~causal, 0.0)
            num = torch.matmul(intra, vc) + torch.matmul(qc, S)
            den = intra.sum(-1) + torch.matmul(qc, z.unsqueeze(-1)).squeeze(-1)
            outs.append(num / (den.unsqueeze(-1) + 1e-6))
            S = S + torch.matmul(kc.transpose(-1, -2), vc)
            z = z + kc.sum(2)
        out = torch.cat(outs, 2).transpose(1, 2).reshape(B, T, D)
        return self.o(out)

    # ---------- 多时间尺度对角 / 块对角状态 ----------
    def _diag(self, x):
        B, T, D = x.shape
        a = torch.sigmoid(self.p)                              # [D]，log 间隔初始化
        xs = self.w(x) * (1.0 - a)                             # 写入量
        outs = []
        if self.block > 1 and hasattr(self, "A"):
            nb, bs = D // self.block, self.block
            h = x.new_zeros(B, nb, bs)
            A = self.A
            for t in range(T):
                h = torch.einsum("nij,bnj->bni", A, h) + xs[:, t].view(B, nb, bs)
                outs.append(h.reshape(B, D))
        else:
            h = x.new_zeros(B, D)
            for t in range(T):
                h = a * h + xs[:, t]
                outs.append(h)
        return self.o(torch.stack(outs, 1))

    # ---------- 矩阵状态 + 衰减（GLA 形态） ----------
    def _gla(self, x):
        B, T, D = x.shape
        q = self.phi(self.q(x)).view(B, T, self.h, self.hd).transpose(1, 2)
        k = self.phi(self.k(x)).view(B, T, self.h, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.h, self.hd).transpose(1, 2)
        if self.kind == "gla_dd":
            g = torch.sigmoid(self.wg(x)).view(B, T, self.h, self.hd).transpose(1, 2)
        g0 = torch.sigmoid(self.p).view(1, 1, self.hd, 1)      # 衰减作用在 key 维
        S = x.new_zeros(B, self.h, self.hd, self.hd)           # [B,h,kd,vd]
        z = x.new_zeros(B, self.h, self.hd)
        outs = []
        for t in range(T):
            kt, vt, qt = k[:, :, t], v[:, :, t], q[:, :, t]
            if self.kind == "gla_dd":
                gt = g[:, :, t].unsqueeze(-1)                  # [B,h,hd,1]
                S = S * (0.5 + 0.5 * gt) + kt.unsqueeze(-1) * vt.unsqueeze(-2)
                z = z * (0.5 + 0.5 * gt.squeeze(-1)) + kt
            else:
                S = S * g0 + kt.unsqueeze(-1) * vt.unsqueeze(-2)
                z = z * g0.squeeze(-1) + kt
            num = (qt.unsqueeze(-2) @ S).squeeze(-2)
            den = (qt * z).sum(-1, keepdim=True)
            outs.append(num / (den + 1e-6))
        out = torch.stack(outs, 1).transpose(1, 2).reshape(B, T, D)
        return self.o(out)

    # ---------- 选择性 SSM（Mamba 简化版） ----------
    def _ssm(self, x):
        B, T, D = x.shape
        A = F.softplus(self.logA) + 1e-4                       # [D] 时间尺度
        dt = F.softplus(self.wd(x))                            # [B,T,D] 输入决定步长
        a = torch.exp(-dt * A)                                 # (0,1)
        b = (1.0 - a) * self.w(x)
        gate = torch.sigmoid(self.wg(x)) if self.kind == "ssm_g" else None
        h = x.new_zeros(B, D)
        outs = []
        for t in range(T):
            h = a[:, t] * h + b[:, t]
            outs.append(h * gate[:, t] if gate is not None else h)
        return self.o(torch.stack(outs, 1))

    # ---------- 经典 RNN（tanh，全矩阵递推） ----------
    def _rnn(self, x):
        B, T, D = x.shape
        xs = self.w(x)
        h = x.new_zeros(B, D)
        outs = []
        for t in range(T):
            h = torch.tanh(xs[:, t] + h @ self.A.t())
            outs.append(h)
        return self.o(torch.stack(outs, 1))

    def forward(self, x):
        if self.kind == "lin":
            return self._lin(x)
        if self.kind in ("gla", "gla_dd"):
            return self._gla(x)
        if self.kind in ("ssm", "ssm_g"):
            return self._ssm(x)
        if self.kind == "rnn":
            return self._rnn(x)
        return self._diag(x)


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
        if getattr(cfg, "mask_src", "none") == "h":
            # C 线的动态掩码：每 token 对 |h| 取 top-k，h 侧与中间维各自一个稀疏度
            mh = topk_mask_grouped(h.detach().abs(), cfg.keep,
                                   getattr(cfg, "group_h", 1))
            if mh is not None:
                h = h * mh
        else:
            mh = None
        if cfg.mlp == "gated":
            act = F.silu(self.gate(h)) * self.up(h)
        else:
            act = F.gelu(self.up(h))
        if mh is not None:
            mm = topk_mask_grouped(act.detach().abs(), cfg.keep_mid,
                                   getattr(cfg, "group_mid", 1))
            if mm is not None:
                act = act * mm
        x = x + self.down(act)
        if M is not None:
            x = x + self.gmix * torch.matmul(M, self.ln2(x))
        return x


class DynLM(nn.Module):
    def __init__(self, vocab, d, layers, heads, hidden, attn_every=1, share_attn=1,
                 mlp="gated", route="none", route_gate="channel", rope_theta=10000.0,
                 mix="attn", state_kind="lin", state_p_lo=-4.0, state_p_hi=2.5,
                 state_block=1):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.ln0 = RMSNorm(d)
        n_attn = -(-layers // attn_every)
        self.mix = mix
        # share_attn=1 -> ONE mixing block whose weights are reused at every
        # refresh ("一个 attn 对接每一层"); share_attn=0 -> one per refresh,
        # which is the ordinary Transformer when attn_every=1.
        # mix=state -> 同一位置换成状态向量（没有 KV cache）
        mk = (lambda: StateMix(d, heads, kind=state_kind, p_lo=state_p_lo,
                               p_hi=state_p_hi, block=state_block)) \
            if mix == "state" else (lambda: CausalAttn(d, heads))
        if share_attn:
            self.attns = nn.ModuleList([mk()])
        else:
            self.attns = nn.ModuleList([mk() for _ in range(n_attn)])
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
                if cfg.mix == "state":
                    A = blk(self.ln0(x))            # 状态向量，无 KV cache
                else:
                    A = blk(self.ln0(x), self.cos, self.sin)
                if cfg.inject == "add":
                    x = x + A
            x = b(x, A, M, cfg)      # ... and re-read by every layer
        xn = self.norm(x)
        if getattr(cfg, "mask_src", "none") == "h":
            hm = topk_mask_grouped(xn.detach().abs(), cfg.keep,
                                   getattr(cfg, "group_h", 1))
            if hm is not None:
                xn = xn * hm
        return xn @ self.emb.weight.t()


def macs_per_token(vocab, d, layers, hidden, seq, attn_every=1, mlp="gated",
                   route="none", mask_src="none", keep=1.0, keep_mid=1.0,
                   mix="attn", heads=4, state_kind="lin", state_block=1):
    """(full-Transformer MAC, this-model MAC) per token, causal core = seq*d."""
    core = seq * d
    per_attn = 4 * d * d + core
    if mix == "state":
        hd = d // heads
        if state_kind == "lin":
            # 状态写入 (hd²/头) + 状态读出 (hd²/头) + 归一化 (hd/头)
            per_attn = 4 * d * d + 2 * d * hd + d
        elif state_kind in ("gla", "gla_dd"):
            per_attn = 4 * d * d + 2 * d * hd + d
            if state_kind == "gla_dd":
                per_attn += d * d                    # 门 W_g x
        elif state_kind in ("ssm", "ssm_g"):
            per_attn = 3 * d * d + d                 # W x + Δ 投影 + 输出投影
            if state_kind == "ssm_g":
                per_attn += d * d                    # 输出门
        elif state_kind == "rnn":
            per_attn = 2 * d * d + d * d             # W x + 全矩阵递推 + 输出投影
        else:
            # 对角/块对角状态：写入 d²，读出 d²，混合 d 或 d*block
            per_attn = 2 * d * d + d * max(1, state_block) + d * d
    per_mlp = (3 if mlp == "gated" else 2) * d * hidden
    n_attn = -(-layers // attn_every)
    total = n_attn * per_attn + layers * per_mlp + vocab * d
    if route == "full":          # build M once (seq*d/token) + M@h per layer
        total += (1 + layers) * seq * d
    tra = layers * (per_attn + per_mlp) + vocab * d
    if mask_src == "h":
        # h 侧稀疏：gated 的 gate+up 是 2*d*I，plain 只有 up 的 1*d*I；再加词表 head
        n_up = 2 if mlp == "gated" else 1
        total -= (1 - keep) * (layers * n_up * d * hidden + vocab * d)
        total -= (1 - keep_mid) * layers * d * hidden
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
    ap.add_argument("--ckpt", default="", help="缺省 <OUT>/<tag>.pt")
    ap.add_argument("--ckpt-every", type=int, default=0)
    ap.add_argument("--resume", type=int, default=0)
    ap.add_argument("--stop-at", type=int, default=0)
    ap.add_argument("--mask-src", default="none", choices=["none", "h"],
                    help="h = 借 C 线的动态掩码：每 token 对 |h| 取 top-k（默认 none）")
    ap.add_argument("--keep", type=float, default=1.0)
    ap.add_argument("--keep-mid", type=float, default=-1.0,
                    help="负值 = 与 keep 相同")
    ap.add_argument("--group-h", type=int, default=1,
                    help="h 侧掩码的成组宽度：1=逐通道，>1=连续 group 个通道整组开")
    ap.add_argument("--group-mid", type=int, default=1,
                    help="中间维掩码的成组宽度（1=逐通道）")
    ap.add_argument("--mix", default="attn", choices=["attn", "state"],
                    help="token 混合算子：attn=共享注意力（要 KV cache）/ state=状态向量（无 KV）")
    ap.add_argument("--state-kind", default="lin",
                    choices=["lin", "diag", "gla", "gla_dd", "ssm", "ssm_g", "rnn"],
                    help="lin=KV 压成状态矩阵；diag=多时间尺度对角状态；gla=矩阵状态+衰减；"
                         "gla_dd=gla 的门由输入决定；ssm=选择性 SSM（Mamba 简化）；"
                         "ssm_g=ssm+输出门；rnn=tanh 全矩阵递推")
    ap.add_argument("--state-p-lo", type=float, default=-4.0,
                    help="衰减 logit 初始化下界（越小=记忆越长）")
    ap.add_argument("--state-p-hi", type=float, default=2.5,
                    help="衰减 logit 初始化上界（越大=记忆越短）")
    ap.add_argument("--state-block", type=int, default=1,
                    help="diag 的块对角宽度：1=纯对角，>1=块内可学耦合（d 的因数）")
    args = ap.parse_args()
    args.hidden = args.hidden or 4 * args.d
    args.attn_every = max(1, min(args.attn_every, args.layers))
    if args.keep_mid < 0:
        args.keep_mid = args.keep

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    meta = json.load(open(os.path.join(DATA, "meta.json"), encoding="utf-8"))
    vocab = int(meta["vocab_size"])
    train = load_ids(os.path.join(DATA, "train_ids.bin"))
    val = load_ids(os.path.join(DATA, "val_ids.bin"))

    model = DynLM(vocab, args.d, args.layers, args.heads, args.hidden,
                  attn_every=args.attn_every, share_attn=args.share_attn,
                  mlp=args.mlp, route=args.route, route_gate=args.route_gate,
                  mix=args.mix, state_kind=args.state_kind,
                  state_p_lo=args.state_p_lo, state_p_hi=args.state_p_hi,
                  state_block=args.state_block)
    nparam = sum(p.numel() for p in model.parameters())
    k_h = args.keep if args.mask_src != "h" else eff_keep(args.d, args.keep, args.group_h)
    k_mid = args.keep if args.mask_src != "h" else eff_keep(args.hidden, args.keep_mid,
                                                            args.group_mid)
    mac_tra, mac_eff = macs_per_token(vocab, args.d, args.layers, args.hidden,
                                      args.seq, args.attn_every, args.mlp, args.route,
                                      args.mask_src, k_h, k_mid,
                                      mix=args.mix, heads=args.heads,
                                      state_kind=args.state_kind,
                                      state_block=args.state_block)

    class C:
        pass
    cfg = C()
    cfg.attn_every, cfg.inject = args.attn_every, args.inject
    cfg.share_attn = int(bool(args.share_attn))
    cfg.mlp = args.mlp
    cfg.route = args.route
    cfg.mask_src, cfg.keep, cfg.keep_mid = args.mask_src, args.keep, args.keep_mid
    cfg.group_h, cfg.group_mid = args.group_h, args.group_mid
    cfg.mix, cfg.state_kind = args.mix, args.state_kind

    print("[cfg] V=%d d=%d L=%d heads=%d hidden=%d attn_every=%d share=%d inject=%s"
          " mlp=%s route=%s/%s mix=%s/%s"
          % (vocab, args.d, args.layers, args.heads, args.hidden,
             args.attn_every, args.share_attn, args.inject, args.mlp,
             args.route, args.route_gate, args.mix, args.state_kind))
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
    last_st = start_st - 1
    with open(mpath, mode, encoding="utf-8") as f:
        if mode == "w":
            f.write(json.dumps(dict(kind="cfg", tag=tag, vocab=vocab, d=args.d,
                                    layers=args.layers, heads=args.heads, hidden=args.hidden,
                                    params=nparam, macs_tra=mac_tra, macs_eff=mac_eff,
                                    attn_every=args.attn_every, inject=args.inject,
                                    share_attn=args.share_attn, mlp=args.mlp,
                                    route=args.route, route_gate=args.route_gate,
                                    mix=args.mix, state_kind=args.state_kind,
                                    state_p_lo=args.state_p_lo,
                                    state_p_hi=args.state_p_hi,
                                    state_block=args.state_block,
                                    mask_src=args.mask_src, keep=args.keep,
                                    keep_mid=args.keep_mid, keep_eff_h=k_h,
                                    keep_eff_mid=k_mid,
                                    group_h=args.group_h, group_mid=args.group_mid,
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
        last_st = st
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
        f.write(json.dumps(dict(kind="final", step=last_st, final_val=val_loss,
                                wall_s=wall,
                                tok_per_s=last_st * args.batch * args.seq / wall)) + "\n")
    print("[done] %s %.1f min %.1f tok/s final_val=%.6f"
          % (tag, wall / 60, last_st * args.batch * args.seq / wall, val_loss))
    return 0


if __name__ == "__main__":
    sys.exit(main())
