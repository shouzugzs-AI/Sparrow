#!/usr/bin/env python3
"""P4: ESERNN 风格 RNN 的权重 + Python 参考实现（验证 VMLM 递归态正确性）"""
import numpy as np
import os

DIR = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(7)

H, D, V = 32, 16, 4

def rnd(*s):
    return (rng.standard_normal(s) * 0.3).astype(np.float32)

# 每层一组权重（全部按 VMLM 的 [rows, cols] 约定，向量用 1×N 表示）
W = {
    # 门控投影：把 H 维映射到 head-dim D（ESERNN 里 r/k/v 都在 head-dim 上）
    "W_r": rnd(D, H), "W_k": rnd(D, H), "W_v": rnd(D, H), "W_o": rnd(H, D),
    "W1d": rnd(32, H), "W2d": rnd(D, 32), "w0": rnd(1, D),
    "xr": rnd(1, H), "xk": rnd(1, H), "xv": rnd(1, H), "xw": rnd(1, H),
    "ln1_g": np.ones((1, H), dtype=np.float32),
    "ln_t_g": np.ones((1, D), dtype=np.float32),
    "ln_out_g": np.ones((1, H), dtype=np.float32),
    "E": rnd(V, H),
    "r_ln_g": np.ones((1, D), dtype=np.float32),
}
# x: 输入序列（逐 step 喂一个 token 的嵌入）
T = 8
X = rnd(T, H)
W["x"] = X.copy()

for k, v in W.items():
    v.astype(np.float32).tofile(os.path.join(DIR, f"{k}.bin"))

# ---------- Python 参考：逐步跑 T 步 ----------
def ln(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b

def sig(x): return 1 / (1 + np.exp(-x))

S = np.zeros((D, D), dtype=np.float32)
x_prev = np.zeros(H, dtype=np.float32)
outs = []


def lnv(x, g, eps=1e-5):
    """1×N 向量的 LayerNorm，gamma 是 1×N"""
    mu = x.mean()
    var = x.var()
    return (x - mu) / np.sqrt(var + eps) * g.ravel()


for t in range(T):
    x = X[t]
    u = lnv(x, W["ln1_g"])
    d = x_prev - u
    r = sig(W["W_r"] @ (u + d * W["xr"].ravel()))
    k = W["W_k"] @ (u + d * W["xk"].ravel())
    v = W["W_v"] @ (u + d * W["xv"].ravel())
    om = np.tanh(W["W1d"] @ (u + d * W["xw"].ravel()))
    om = W["W2d"] @ om + W["w0"].ravel()
    decay = np.exp(-np.exp(om))
    S = S * decay[:, None] + np.outer(v, k)
    o = S @ r
    o = lnv(o, W["ln_t_g"])
    x = x_prev + W["W_o"] @ o
    x_prev = x
    logits = lnv(x, W["ln_out_g"]) @ W["E"].T
    outs.append(logits.copy())

np.savetxt(os.path.join(DIR, "ref_logits.txt"), np.array(outs), fmt="%.6f")
np.save(os.path.join(DIR, "ref_S.npy"), S)
print("Python 参考: 末步 logits =", outs[-1])
print("末步递归态 S[0,:4] =", S[0, :4])
