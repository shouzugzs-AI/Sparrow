#!/usr/bin/env python3
"""
P5: 小 Transformer 的权重 + Python 参考实现
（多头注意力 + 因果掩码 + FFN，验证 VMLM 的 BMM/SOFTMAX/TRANSPOSE）

结构（sparse_dyn README 的简化版，去掉动态稀疏）：
  x = tok_emb(idx) + pos_emb
  for l in layers:
     x = x + Attn(LN(x))
     x = x + FFN(LN(x))
  logits = LN_f(x) @ E^T

维度：T=8（序列长）, D=16（模型维）, H=2（头数）, hd=8（头维）, F=32（FFN）, V=4
"""
import os
import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(2025)

T, D, NH, HD, F, V = 8, 16, 2, 8, 32, 4


def rnd(*s, scale=0.3):
    return (rng.standard_normal(s) * scale).astype(np.float32)


W = {
    # 注意力：一次大 GEMM 出 QKV（sparse_dyn 的做法）
    "Wqkv": rnd(3 * D, D),      # [48, 16]
    "bqkv": rnd(1, 3 * D),      # [1, 48]
    "Wproj": rnd(D, D),         # [16, 16]
    "bproj": rnd(1, D),
    # FFN
    "W1": rnd(F, D),            # [32, 16]
    "b1": rnd(1, F),
    "W2": rnd(D, F),            # [16, 32]
    "b2": rnd(1, D),
    # LayerNorm
    "ln1_g": np.ones((1, D), dtype=np.float32), "ln1_b": np.zeros((1, D), dtype=np.float32),
    "ln2_g": np.ones((1, D), dtype=np.float32), "ln2_b": np.zeros((1, D), dtype=np.float32),
    "lnf_g": np.ones((1, D), dtype=np.float32), "lnf_b": np.zeros((1, D), dtype=np.float32),
    # 嵌入 & 输出头（权重共享）
    "E": rnd(V, D),
    "pos": rnd(T, D),
    # 输入 token id
    "tok": np.array([1, 2, 3, 0, 2, 1, 3, 0], dtype=np.float32),
}

for k, v in W.items():
    v.astype(np.float32).tofile(os.path.join(DIR, f"{k}.bin"))


# ---------------- Python 参考 ----------------
def ln(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


def softmax_causal(s):
    T_ = s.shape[-1]
    mask = np.tril(np.ones((T_, T_), dtype=bool))
    s = np.where(mask, s, -1e30)
    m = s.max(-1, keepdims=True)
    e = np.exp(s - m)
    return e / e.sum(-1, keepdims=True)


tok = W["tok"].astype(int)
x = W["E"][tok] + W["pos"]          # [T, D]

xd = ln(x, W["ln1_g"], W["ln1_b"])
qkv = xd @ W["Wqkv"].T + W["bqkv"]  # [T, 3D]
q, k, v = np.split(qkv, 3, axis=-1)
# 拆头：[T, NH, HD] -> [NH, T, HD]
q = q.reshape(T, NH, HD).transpose(1, 0, 2)
k = k.reshape(T, NH, HD).transpose(1, 0, 2)
v = v.reshape(T, NH, HD).transpose(1, 0, 2)

att = q @ k.transpose(0, 2, 1) / np.sqrt(HD)   # [NH, T, T]
att = softmax_causal(att)
y = att @ v                                     # [NH, T, HD]
y = y.transpose(1, 0, 2).reshape(T, D)
attn_out = y @ W["Wproj"].T + W["bproj"]
x = x + attn_out

xh = ln(x, W["ln2_g"], W["ln2_b"])
h = np.maximum(xh @ W["W1"].T + W["b1"], 0)     # ReLU FFN
ffn_out = h @ W["W2"].T + W["b2"]
x = x + ffn_out

xf = ln(x, W["lnf_g"], W["lnf_b"])
logits = xf @ W["E"].T                          # [T, V]

np.savetxt(os.path.join(DIR, "ref_logits.txt"), logits, fmt="%.6f")
np.savetxt(os.path.join(DIR, "ref_att.txt"), att.reshape(-1, T), fmt="%.6f")
print(f"Python 参考: logits shape={logits.shape}")
print("末位 logits =", logits[-1])
print("att[0] 因果性检查（上三角应为 0）:")
print(np.round(att[0], 4))
