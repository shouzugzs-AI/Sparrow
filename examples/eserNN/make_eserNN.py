#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P7-C: 真实 ESERNN 单步前向 → VMLM 迁移。

本脚本做三件事（成员 C 的独占写作用域）：
  1. 用**真实 ESERNN 源码语义**（<外部 ESERNN 源码目录>/core/ese_rnn.py 的
     TimeMixing / ChannelMixing / ESERNNBlock / forward_one）生成 mini 配置权重；
  2. 用 numpy **逐步**复算同一个 mini 配置，落盘参考结果（ref_logits / ref_S / ref_x）；
  3. 导出所有张量为原始 f32 小端 .bin，供汇编器 `名字:文件.bin:rows:cols` 注入。

与 P4 的 examples/rnn/ese_step.vml 的关键差别（这是本任务的重点）：
  * 真实 ESERNN 是 **多头** 的：state[H, D, D]，H=heads, D=head_dim
      - 递归态衰减是 **per-head 标量**：decay[B,T,H,D] → 每个 head 的 D 个通道共享同一 decay
      - OUTER 写入按 head 切片：S[h] += v[h] ⊗ k[h]
      - 读出 o[h] = S[h] @ r[h]，再把 H*D 个通道 **reshape 回 H*D 维** 做 LayerNorm
  * 真实 ESERNN 是 **多层** 的：每层有独立的 x_prev 与 kv
  * 真实 ESERNN 有 **通道混合（FFN）** 分支：
      g = x_k⊙x_prev + (1-x_k)⊙LN2(x)   （注意是 LN2(x)，**不是** x）
      x += value_w(relu(key_w(g))**2) * ln   （ffn_norm_mode='scale'，gamma 恒等初始）
  * x_prev 不参与梯度且**每层独立**：层内跨 step 更新为 x_prev ← 本层输出 x

约定（与 VMLM 对齐）：
  所有向量都是「行向量」[1, N]；矩阵是 [rows, cols]；GEMM 语义 C[M,N] = A[M,K]·B[N,K]^T，
  即权重按 nn.Linear 的 [out_features, in_features] 存放。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))

# ======================================================================
# 0. mini 配置（任务要求：先用 mini 跑通，再说明放大路径）
# ======================================================================
CFG = dict(
    vocab=32,
    hidden=64,      # H = heads * head_dim
    heads=8,        # n_heads —— **必须等于 hidden/head_dim**，否则源码里 reshape(H_,D_) 会炸
    head_dim=8,     # D
    inter=128,      # intermediate（真实 1536/512 = 3x，这里 128/64 = 2x 省脚本量）
    decay_hidden=16,
    eps=1e-6,       # cfg.rms_eps 默认 1e-6（**不是** torch 默认的 1e-5！）
    T=6,            # 跑 6 个 step（≥4 满足验收）
)

HD = CFG["hidden"]            # 64
NC = CFG["heads"]             # 8
D = CFG["head_dim"]           # 8
F = CFG["inter"]              # 128
DH = CFG["decay_hidden"]      # 16
V = CFG["vocab"]              # 32
T = CFG["T"]
EPS = CFG["eps"]
#: 引擎 `OP_LN` 把 eps **硬编码成 1e-5**（core/src/interp.rs:505），
#: 而真实 ESERNN 用 cfg.rms_eps = 1e-6。两者在 mini 配置下会差到 2.8e-2
#: （因为 ln0 的输入方差只有 ~4e-4，eps 直接主导 sqrt(var+eps)）。
#: 我不越权改引擎，所以**同时**产出两份参考：
#:   ref_*      用 EPS_REAL(1e-6) —— 真实 ESERNN 的权威语义
#:   ref_eps5_* 用 1e-5          —— 引擎实际语义（用来证明指令流本身没写错）
EPS_ENGINE = 1e-5
assert NC * D == HD, "真实 ESERNN 要求 hidden == n_heads * head_dim"

SEED = 20240927
rng = np.random.default_rng(SEED)


def rnd(*shape, scale=1.0):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def orth(rows, cols, gain, rng_):
    """等价于 torch.nn.init.orthogonal_(W) 后乘 gain。

    torch 的 orthogonal_ 对 [rows, cols] 张量：若 rows < cols 先转置，对
    (rows, cols) 做 QR 取 Q，再转置回去。这里只在 rows>=cols 的情形用到
    （TimeMixing 的四个方阵），所以直接用 QR。
    """
    a = rng_.standard_normal((rows, cols))
    q, r = np.linalg.qr(a)
    q = q * np.sign(np.diag(r))          # 符号规范化，与 torch 一致
    return (q * gain).astype(np.float32)


# ======================================================================
# 1. 权重（命名严格照抄真实 ESERNN 的 state_dict 路径，把 '.' 换成 '_'）
#    **初始化必须照抄 ese_rnn.py**，否则递归态会数值发散：
#      - TimeMixing.w1 = randn(H, decay_hidden) * 0.02   （注意是 [H,DH] 布局！）
#      - TimeMixing.w2 = zeros(decay_hidden, H)
#      - r_w/k_w/v_w/o_w : orthogonal_(weight) 后 * 0.1
#      - embed_tokens  : N(0, 0.02)
#    真实 w2=0 会让 decay 逐通道恒为 exp(-exp(w0))，mini 下递归态**不衰减**，
#    几十步后必然 e+27 量级 —— 上面这些真实初始化保证 6 步内数值健康。
# ======================================================================
W: dict[str, np.ndarray] = {}

# --- ESERNN.embed_tokens.weight : [V, H]，N(0, 0.02) ---
W["E"] = rnd(V, HD, scale=0.02)
# --- ESERNN.ln0 : LayerNorm(H) ---
W["ln0_g"] = np.ones((1, HD), dtype=np.float32)
W["ln0_b"] = np.zeros((1, HD), dtype=np.float32)
# --- ESERNN.ln_out : LayerNorm(H) ---
W["lnout_g"] = np.ones((1, HD), dtype=np.float32)
W["lnout_b"] = np.zeros((1, HD), dtype=np.float32)

# --- ESERNNBlock.ln1 / ln2 : LayerNorm(H) ---
W["ln1_g"] = np.ones((1, HD), dtype=np.float32)
W["ln1_b"] = np.zeros((1, HD), dtype=np.float32)
W["ln2_g"] = np.ones((1, HD), dtype=np.float32)
W["ln2_b"] = np.zeros((1, HD), dtype=np.float32)

# --- TimeMixing : 逐通道门控（都是 [H] 的行向量），真实是 zeros 初始化 ---
W["tm_xr"] = np.zeros((1, HD), dtype=np.float32)
W["tm_xk"] = np.zeros((1, HD), dtype=np.float32)
W["tm_xv"] = np.zeros((1, HD), dtype=np.float32)
W["tm_xw"] = np.zeros((1, HD), dtype=np.float32)
# --- TimeMixing.w0 : 真实默认 decay_init_w0 = -2.5（decay ≈ 0.921）---
#     w0 以「列向量 [H,1]」导出，便于 EWMUL 行广播（见 step.vml 注释）。
W["tm_w0col"] = np.full((HD, 1), -2.5, dtype=np.float32)
# --- TimeMixing.w1 : nn.Parameter(randn(H, decay_hidden) * 0.02)，布局 [H, DH] ---
#     ⚠ 注意 w1/w2 是 **nn.Parameter 而不是 nn.Linear**，所以它们的布局与
#     nn.Linear 的 [out,in] **相反**。numpy 参考里用 `@` 直接乘（A·W），
#     而 VMLM 的 GEMM 默认是 NT（C=A·B^T，B 按 [N,K]）。
#     为了让 VMLM 用默认 NT 直接吃这块权重，这里导出**转置后的** [DH,H]：
#       numpy:  tanh(g4 @ w1p)      ，w1p 是 [H,DH]
#       VMLM :  GEMM(g4, w1pT)      ，w1pT 是 [DH,H] ⇒ C = g4·w1pT^T = g4·w1p ✓
W["tm_w1p"] = rnd(HD, DH, scale=0.02)          # 训练侧的权威布局 [H,DH]
W["tm_w1pT"] = np.ascontiguousarray(W["tm_w1p"].T)   # [DH,H]，给 VMLM 的 NT 用
# --- TimeMixing.w2 : nn.Parameter(zeros(decay_hidden, H))，布局 [DH, H] ---
W["tm_w2p"] = np.zeros((DH, HD), dtype=np.float32)   # 训练侧 [DH,H]
W["tm_w2pT"] = np.ascontiguousarray(W["tm_w2p"].T)   # [H,DH]，给 VMLM 的 NT 用
# --- TimeMixing.r_w / k_w / v_w / o_w : nn.Linear(H,H)，orthogonal_ * 0.1 ---
W["tm_Wr"] = orth(HD, HD, 0.1, rng)
W["tm_Wk"] = orth(HD, HD, 0.1, rng)
W["tm_Wv"] = orth(HD, HD, 0.1, rng)
W["tm_Wo"] = orth(HD, HD, 0.1, rng)
# --- TimeMixing.ln : LayerNorm(H*D) ---
W["tmln_g"] = np.ones((1, HD), dtype=np.float32)
W["tmln_b"] = np.zeros((1, HD), dtype=np.float32)

# --- ChannelMixing.x_k : [H]，真实是 zeros 初始化 ---
W["cm_xk"] = np.zeros((1, HD), dtype=np.float32)
# --- ChannelMixing.key_w : nn.Linear(H, inter)，orthogonal_ * 0.05 → 权重 [F, H] ---
W["cm_Wkey"] = orth(F, HD, 0.05, rng)
# --- ChannelMixing.value_w : nn.Linear(inter, H)，orthogonal_ * 0.05 → 权重 [H, F] ---
#     注意 Inter >= H 时 torch 的 orthogonal_ 会**先转置**再 QR 再转回来，
#     直接对 [H,F] 做 QR 得到的不是同一个矩阵。这里显式复刻。
W["cm_Wval"] = orth(F, HD, 0.05, rng).T.copy()               # [F,H] → [H,F]
# --- ChannelMixing.ln : 'scale' 模式的逐通道 gamma [H]，恒等初始化 ---
W["cmln_g"] = np.ones((1, HD), dtype=np.float32)

# --- 输入序列（逐 step 喂一个 token 的**嵌入**，与 P4 的 rnn_run 驱动一致）---
TOKENS = rng.integers(0, V, size=T).astype(np.int32)
X = W["E"][TOKENS]                      # [T, H]：第 t 步的输入 x
W["tok"] = TOKENS.astype(np.float32)


# ======================================================================
# 2. numpy 参考实现（严格照抄 ese_rnn.py 的 forward_one 单步数学）
# ======================================================================
def ln(x, g, b, eps=EPS):
    """LayerNorm：沿最后一维。x [..., C]，g/b [1, C]。"""
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)       # 有偏方差（torch 默认 unbiased=False）
    return (x - mu) / np.sqrt(var + eps) * g + b

def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def relu2(x):
    r = np.maximum(x, 0.0)
    return r * r


def reference(eps=EPS):
    """逐步复算：返回 (logits[T,V], S_final, x_final, xprev_final, x_steps)。

    eps 可切换：默认 EPS(1e-6) 是真实 ESERNN 语义；传 EPS_ENGINE(1e-5)
    得到「引擎 OP_LN 实际语义」下的参考，用来单独验证指令流本身。

    完全对应 ESERNN.forward_one(token, state)：
        x = ln0(embed(token))                      # [1,1,H]
        for blk in layers:                          # 这里只有 1 层
            x, kv = blk.forward_train(x, prev_x, kv, H, D)
            state.x_prev = x[0,-1]
    """
    S = np.zeros((NC, D, D), dtype=np.float32)   # state[i][1] : kv
    x_prev = np.zeros(HD, dtype=np.float32)      # state[i][0] : 上一 step 的本层输入
    logits_all = []
    x_all = []

    for t in range(T):
        xt = X[t]                                 # 本步输入（已是嵌入，对应 ln0 的输入）
        # ---------- ESERNN.forward_one: x = self.ln0(embed(token)) ----------
        x = ln(xt[None, :], W["ln0_g"], W["ln0_b"], eps)      # [1,H]
        xp = x_prev[None, :]                                 # [1,H]
        # 真实实现里 H_ 是 ESERNNBlock.forward_train 的形参 H = cfg.n_heads，
        # 而 hidden = n_heads * head_dim 必须成立 —— 所以 heads = hidden/head_dim。
        # mini 配置 hidden=64, head_dim=8 → heads=8（不是我把 NC 写成 2 就成立）。
        H_, D_ = HD // D, D

        # ========== ESERNNBlock.forward_train ==========
        # --- x, kv = time_mixing.forward_train(ln1(x), x_prev, kv, H, D) ---
        xin = ln(x, W["ln1_g"], W["ln1_b"], eps)              # [1,H]  ← ln1(x)

        # TimeMixing.forward_train：
        #   xx = x_prev - x        （x 是 ln1 之后的；x_prev 是**上一 step 的
        #                            ESERNNBlock 级 x_prev**，未过 ln1）
        xx = xp - xin                                        # [1,H]
        r = sig(W["tm_Wr"] @ (xin + xx * W["tm_xr"]).T).T    # [1,H]
        k = (W["tm_Wk"] @ (xin + xx * W["tm_xk"]).T).T       # [1,H]
        v = (W["tm_Wv"] @ (xin + xx * W["tm_xv"]).T).T       # [1,H]
        # w1/w2 是 nn.Parameter（不是 nn.Linear），布局 [H,DH] / [DH,H]
        # → 数学上等价于 nn.Linear(H,DH) 与 nn.Linear(DH,H)
        w = (W["tm_w0col"].T
             + np.tanh((xin + xx * W["tm_xw"]) @ W["tm_w1p"]) @ W["tm_w2p"])  # [1,H]
        decay = np.exp(-np.exp(w))                           # [1,H] ∈ (0,1)
        # view(B,T,H,D)：t 维已经单步，等价于 reshape 到 [H,D]
        r_h = r.reshape(H_, D_)     # H_ = n_heads = 2, D_ = head_dim = 8
        k_h = k.reshape(H_, D_)
        v_h = v.reshape(H_, D_)
        d_h = decay.reshape(H_, D_)
        # state[H,D,D] = state * decay + v⊗k  (decay 在 v 方向 → S[h,:,:] *= d_h[h,0])
        S = S * d_h[:, 0][:, None, None] + v_h[:, :, None] * k_h[:, None, :]
        # outs[H,D] = state @ r  → reshape(B,T,H*D) = [1,H*D]
        outs = np.einsum("hij,hj->hi", S, r_h).reshape(1, H_ * D_)
        out = ln(outs, W["tmln_g"], W["tmln_b"], eps)
        tm_out = out @ W["tm_Wo"].T                          # o_w(out) : [1,H]

        x = x + tm_out                                       # 残差
        # --- x = x + channel_mixing.forward_train(ln2(x), x_prev) ---
        xin2 = ln(x, W["ln2_g"], W["ln2_b"], eps)
        # ChannelMixing: k = x + (x_prev - x) * x_k  ← 注意用的是 **未 LN 的 x_prev**
        kk = xin2 + (xp - xin2) * W["cm_xk"]
        kk = relu2(kk @ W["cm_Wkey"].T)                      # [1,F]
        cm_out = kk @ W["cm_Wval"].T                         # [1,H]
        cm_out = cm_out * W["cmln_g"]                        # ffn_norm_mode='scale'
        x = x + cm_out
        # ==============================================================

        x_prev = x[0].copy()                                 # state[i][0].copy_(x[0,-1])
        x_all.append(x[0].copy())
        # ---------- logits = F.linear(ln_out(x), E) ----------
        lg = ln(x, W["lnout_g"], W["lnout_b"], eps) @ W["E"].T    # [1,V]
        logits_all.append(lg[0].copy())

    return (np.array(logits_all, dtype=np.float32), S, x_prev,
            x[0].copy(), np.array(x_all, dtype=np.float32))


# ======================================================================
# 3. 通过真实 ESERNN 类做「源码级交叉验证」（可选，需要 torch）
# ======================================================================
def _build_real_model():
    """构造真实 ESERNN 实例并把 mini 权重灌进去。

    返回 (model, torch)；取不到 torch / ESERNN 包时返回 (None, 原因字符串)。
    """
    eser_dir = r"<外部 ESERNN 源码目录>"
    if not os.path.isdir(eser_dir):
        return None, "ESERNN 目录不存在"
    try:
        import torch
    except ImportError:
        return None, "无 torch"
    sys.path.insert(0, eser_dir)
    try:
        from core.ese_rnn import ESERNN, ESERNNConfig
    except Exception as e:                                # noqa: BLE001
        return None, f"导入 ESERNN 失败: {e}"

    torch.set_num_threads(1)
    cfg = ESERNNConfig(vocab_size=V, hidden=HD, n_layers=1,
                       n_heads=NC, head_dim=D, intermediate=F,
                       decay_hidden=DH, rms_eps=EPS, ffn_norm=True,
                       ffn_norm_mode="scale", ffn_act="relu2")
    m = ESERNN(cfg)
    sd = m.state_dict()

    def put(name, arr):
        t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
        assert sd[name].numel() == t.numel(), \
            f"{name}: 形状不符 {tuple(sd[name].shape)} vs {tuple(t.shape)}"
        sd[name] = t.reshape(sd[name].shape)

    put("embed_tokens.weight", W["E"])
    for nm, pre in (("ln0", "ln0"), ("ln_out", "lnout")):
        put(f"{nm}.weight", W[f"{pre}_g"].ravel())
        put(f"{nm}.bias", W[f"{pre}_b"].ravel())
    p = "layers.0."
    put(p + "ln1.weight", W["ln1_g"].ravel()); put(p + "ln1.bias", W["ln1_b"].ravel())
    put(p + "ln2.weight", W["ln2_g"].ravel()); put(p + "ln2.bias", W["ln2_b"].ravel())
    q = p + "time_mixing."
    put(q + "x_r", W["tm_xr"].ravel()); put(q + "x_k", W["tm_xk"].ravel())
    put(q + "x_v", W["tm_xv"].ravel()); put(q + "x_w", W["tm_xw"].ravel())
    put(q + "w0", W["tm_w0col"].ravel())
    put(q + "w1", W["tm_w1p"]); put(q + "w2", W["tm_w2p"])
    put(q + "r_w.weight", W["tm_Wr"]); put(q + "k_w.weight", W["tm_Wk"])
    put(q + "v_w.weight", W["tm_Wv"]); put(q + "o_w.weight", W["tm_Wo"])
    put(q + "ln.weight", W["tmln_g"].ravel()); put(q + "ln.bias", W["tmln_b"].ravel())
    q = p + "channel_mixing."
    put(q + "x_k", W["cm_xk"].ravel())
    put(q + "key_w.weight", W["cm_Wkey"]); put(q + "value_w.weight", W["cm_Wval"])
    put(q + "ln", W["cmln_g"].ravel())
    m.load_state_dict(sd, strict=True)
    m.eval()
    return m, torch


def crosscheck_with_real_eserNN():
    """把上面的权重灌进真实的 ESERNN 类，用 forward_one 复算，逐位对比 numpy 参考。

    这一层是**最有说服力的证据**：证明我的 numpy 参考 == 真实源码语义，
    也证明我导出的权重确实就是真实模型的那一份。
    """
    m, torch = _build_real_model()
    if m is None:
        return None, torch
    st = m.new_state(1)
    outs = []
    with torch.no_grad():
        for t in range(T):
            lg, st = m.forward_one(int(TOKENS[t]), st)
            outs.append(lg.numpy().copy())
    return np.array(outs, dtype=np.float32), (
        st[0][1].numpy().copy(), st[0][0].numpy().copy())


def check_stepwise_vs_block():
    """钉死 x_prev 语义：forward_one（逐步）必须与 forward_train（整块）一致。

    这一步专门用来防一手容易搞错的语义：**TimeMixing 里的 x 是 ln1(x) 之后的，
    而 x_prev 是块级的、未过 ln1 的 x_prev**。若误把 x_prev 也过一遍 ln1，
    逐步路径就会和整块路径对不上。
    """
    m, torch = _build_real_model()
    if m is None:
        return None, torch
    st = m.new_state(1)
    step = []
    with torch.no_grad():
        for t in range(T):
            lg, st = m.forward_one(int(TOKENS[t]), st)
            step.append(lg.numpy().copy())
        st2 = m.new_state(1)
        blk, _ = m.forward_train(
            torch.tensor([TOKENS.tolist()], dtype=torch.long), st2,
            truncate_state=True)
    # blk 形状是 [B,T,V]；逐步路径的第 t 步对应 blk[0, t]
    return np.array(step, dtype=np.float32), blk[0].numpy().copy()


def main():
    ref_logits, S_final, x_final, xpre_final, x_steps = reference()

    # 输入序列：既导出「整段嵌入」给 batch 校验，也导出逐 step 单行给驱动
    for t in range(T):
        W[f"x_t{t}"] = X[t:t + 1].copy()          # [1,H] 行向量
    W["x_all"] = X.copy()                         # [T,H]

    # 汇编用辅助常量：one_d1 是 [D,1] 全 1 列向量。
    # 用途一：GEMM(w0col[H,1], one_d1[D,1]) ⇒ [H,D]，把每 head 的 w0 复制 D 份，
    #         从而满足 EWADD 的 [1,H] op [H,D] 行广播（[1,H] op [H,1] 不是合法分支）。
    # 用途二：GEMM(decay_slice[1,1], one_d1[D,1]) ⇒ [D,1]，
    #         把 per-head 的 decay 标量复制成列向量，配合 [D,D] op [D,1] 行广播。
    # 汇编用辅助常量（名字故意与训练侧权重区分：
    #   xxxT 后缀 = 训练侧权重的转置副本，专供 VMLM 的默认 NT GEMM；
    #   one_* = 全 1 常量，用 GEMM 做「标量 → 定长向量」的广播复制，
    #          因为 [1,1] op [H,D] 之类的逐元素广播在引擎里没有对应分支。
    W["one_d1"] = np.ones((D, 1), dtype=np.float32)
    W["one_r8"] = np.ones((D, 1), dtype=np.float32)
    # eps1e6：喂给 `LN_EPS` 指令，把引擎的 LayerNorm eps 从默认 1e-5 改成
    # 真实 ESERNN 的 cfg.rms_eps = 1e-6。
    W["eps1e6"] = np.array([[EPS]], dtype=np.float32)

    # 落盘所有 .bin（原始 f32 小端）
    manifest = {}
    for name, arr in W.items():
        a = np.ascontiguousarray(arr, dtype=np.float32)
        a.tofile(os.path.join(DIR, f"{name}.bin"))
        manifest[name] = list(a.shape)

    np.savetxt(os.path.join(DIR, "ref_logits.txt"), ref_logits, fmt="%.8f")
    S_final.astype(np.float32).tofile(os.path.join(DIR, "ref_S.bin"))
    x_final.astype(np.float32).tofile(os.path.join(DIR, "ref_x.bin"))
    xpre_final.astype(np.float32).tofile(os.path.join(DIR, "ref_xprev.bin"))
    # 逐步的块输出 x（第 t 步的 x 就是第 t+1 步的 x_prev，也是 logits 的来源）
    x_steps.astype(np.float32).tofile(os.path.join(DIR, "ref_x_steps.bin"))

    # ---- 第二份参考：eps = 1e-5（引擎 OP_LN 的硬编码值）----
    # 用途：单独证明「.vml 指令流本身」没有写错。真实 ESERNN 用 1e-6，
    # 而引擎固定 1e-5，两者在 mini 配置下差 ~2.8e-2（见 README 「已知语义偏差」）。
    (lg5, S5, _x5, xp5, xs5) = reference(eps=EPS_ENGINE)
    np.savetxt(os.path.join(DIR, "ref_eps5_logits.txt"), lg5, fmt="%.8f")
    S5.astype(np.float32).tofile(os.path.join(DIR, "ref_eps5_S.bin"))
    xp5.astype(np.float32).tofile(os.path.join(DIR, "ref_eps5_xprev.bin"))
    xs5.astype(np.float32).tofile(os.path.join(DIR, "ref_eps5_x_steps.bin"))
    with open(os.path.join(DIR, "cfg.json"), "w", encoding="utf-8") as f:
        json.dump({**CFG, "tokens": TOKENS.tolist(), "seed": SEED,
                   "shapes": manifest}, f, indent=2)

    print("=== mini 配置 ===")
    print(f"  vocab={V} hidden={HD} heads={NC} head_dim={D} inter={F} "
          f"decay_hidden={DH} layers=1 eps={EPS} T={T}")
    print(f"  tokens = {TOKENS.tolist()}")
    print(f"  权重张量 {len(W)} 个 → {DIR}")
    print("\n=== 末步参考 logits[0:8] ===")
    print("  " + " ".join(f"{v:+.6f}" for v in ref_logits[-1][:8]))
    print(f"\n末步递归态 S[0,0,:4] = {S_final[0, 0, :4]}")
    print(f"末步递归态 S[1,0,:4] = {S_final[1, 0, :4]}")
    print(f"|S| max = {np.abs(S_final).max():.6f}")
    print(f"\n[eps 对照] 真实 ESERNN eps={EPS} vs 引擎硬编码 eps={EPS_ENGINE}")
    print(f"  两者末步 logits 最大差 = "
          f"{np.abs(lg5 - ref_logits).max():.3e}  ← 这是引擎语义偏差，不是指令流错误")

    # ---- eps 敏感性曲线：说明「放大到真实配置后该偏差为何会消失」----
    # LayerNorm 输出 = (x-μ)/sqrt(var+eps)。eps 的影响取决于 var：
    # var 越小分母越被 eps 主导，偏差越大。mini 配置的 ln0 输入方差只有 ~4e-4，
    # 正好落在最坏区间；真实 6 层模型残差累积后方差是 O(1)，影响掉到 1e-5 以下。
    print("\n[eps 敏感性] LayerNorm 在 eps=1e-6 vs 1e-5 下的输出最大差：")
    eps_rows = []
    for var in (4e-4, 1e-2, 0.1, 1.0, 10.0):
        r = np.random.default_rng(1)
        xx = r.standard_normal(HD).astype(np.float32) * np.sqrt(var)
        xx = xx - xx.mean()
        aa = xx / np.sqrt(xx.var() + EPS)
        bb = xx / np.sqrt(xx.var() + EPS_ENGINE)
        d = float(np.abs(aa - bb).max())
        eps_rows.append((var, d))
        print(f"    var={var:8.4g}  max|ΔLN| = {d:.3e}"
              f"{'   ← mini 的 ln0 落在这里' if abs(var - 4e-4) < 1e-9 else ''}")
    with open(os.path.join(DIR, "eps_sensitivity.txt"), "w", encoding="utf-8") as f:
        f.write("var\tmax_abs_dLN\teps_real\teps_engine\n")
        for var, d in eps_rows:
            f.write(f"{var:.6g}\t{d:.6e}\t{EPS:g}\t{EPS_ENGINE:g}\n")

    # 真实 ESERNN 交叉验证
    cc, info = crosscheck_with_real_eserNN()
    if cc is None:
        print(f"\n[交叉验证] 跳过（{info}）")
    else:
        d_lg = float(np.abs(cc - ref_logits).max())
        d_S = float(np.abs(info[0] - S_final).max())
        d_xp = float(np.abs(info[1] - xpre_final).max())
        print("\n[交叉验证 A] numpy 参考 vs 真实 ESERNN.forward_one")
        print(f"  logits 最大绝对差 = {d_lg:.3e}")
        print(f"  S      最大绝对差 = {d_S:.3e}")
        print(f"  x_prev 最大绝对差 = {d_xp:.3e}")
        worst = max(d_lg, d_S, d_xp)
        # 必须显式判 isfinite：NaN > 1e-5 是 False，会把发散静默判成 PASS
        ok = bool(np.isfinite(worst)) and worst < 1e-5
        print(f"  判定: {'PASS' if ok else 'FAIL'}"
              "（numpy 参考 == 真实源码语义）")
        if not ok:
            sys.exit(1)

    # forward_one（逐步）vs forward_train（整块）：钉死 x_prev 锁存语义
    bs, block_lg = check_stepwise_vs_block()
    if bs is None:
        print(f"[交叉验证 B] 跳过（{block_lg}）")
    else:
        db = float(np.abs(bs - block_lg).max())     # 逐步 vs 整块，逐步比全部 T 个时间步
        print("[交叉验证 B] forward_one(逐步) vs forward_train(整块)，T 步全部对比")
        print(f"  最大绝对差 = {db:.3e}")
        okb = bool(np.isfinite(db)) and db < 1e-5
        print(f"  判定: {'PASS' if okb else 'FAIL'}（x_prev 锁存语义正确）")
        if not okb:
            sys.exit(1)

    # 生成汇编器的权重注入参数（名字:文件.bin:rows:cols）
    # ⚠ 必须用**相对文件名**：汇编器用 ':' 切分参数，而 Windows 绝对路径
    #   形如 D:\... 自带冒号，会把参数切碎导致「读不到权重文件」。
    #   所以调用方必须先 cd 到本目录再汇编。
    lines = []
    for name, arr in W.items():
        a = np.asarray(arr)
        r = a.shape[0] if a.ndim >= 1 else 1
        c = a.shape[1] if a.ndim == 2 else 1
        lines.append(f"{name}:{name}.bin:{r}:{c}")
    with open(os.path.join(DIR, "weights.args"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n权重注入参数已写入 weights.args（{len(lines)} 项，相对路径）")


if __name__ == "__main__":
    main()
