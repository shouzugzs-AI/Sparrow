#!/usr/bin/env python3
"""
P6: 动态稀疏验证数据（sparse_dyn 的神经元级 top-k 门控）

核心机制（sparse_dyn/neuron_state.py）：
  h = x @ W1^T + b1        # [T, F]
  strength = |h| 的某种聚合（这里逐行取 |h|）
  k = floor(F * top_ratio)
  tau = 第 k 大的 strength
  mask = (strength >= tau)
  h = h * mask             # 稀疏
  h = GELU(h)
  y = h @ W2^T + b2

关键不变量：**每行恰好保留 k 个非零**（稀疏率恒等于 top_ratio）
"""
import os
import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(7)

T, D, F = 4, 16, 32
TOP_RATIO = 0.25
K = int(F * TOP_RATIO)  # = 8

W1 = (rng.standard_normal((F, D)) * 0.5).astype(np.float32)
b1 = (rng.standard_normal(F) * 0.2).astype(np.float32)
W2 = (rng.standard_normal((D, F)) * 0.5).astype(np.float32)
b2 = (rng.standard_normal(D) * 0.2).astype(np.float32)
x = (rng.standard_normal((T, D)) * 0.8).astype(np.float32)

for n, a in [("W1", W1), ("b1", b1), ("W2", W2), ("b2", b2), ("x", x)]:
    a.tofile(os.path.join(DIR, f"{n}.bin"))

# ---------------- Python 参考 ----------------
h = x @ W1.T + b1                    # [T, F]

# TOPK_THRESH：逐行取第 k 大的值作为阈值
tau = np.sort(np.abs(h), axis=-1)[:, -K]           # [T]
mask = (np.abs(h) >= tau[:, None]).astype(np.float32)

# 注意：用 >= 可能选出多于 k 个（并列）。Python 参考和 VMLM 必须用同一规则。
# 这里如实记录每行实际选中的个数。
counts = mask.sum(-1).astype(int)

hs = h * mask
np.savetxt(os.path.join(DIR, "ref_hs.txt"), hs, fmt="%.6f")
# GELU（tanh 近似，和引擎里的 act_scalar 一致）
g = 0.5 * hs * (1 + np.tanh(0.7978845608 * (hs + 0.044715 * hs**3)))
y = g @ W2.T + b2

np.savetxt(os.path.join(DIR, "ref_mask.txt"), mask, fmt="%d")
np.savetxt(os.path.join(DIR, "ref_y.txt"), y, fmt="%.6f")

print(f"配置: T={T} D={D} F={F} top_ratio={TOP_RATIO} k={K}")
print(f"阈值 tau = {np.round(tau, 4)}")
print(f"每行选中个数 = {counts}  (期望 {K})")
print(f"实际稀疏率 = {mask.mean():.4f}  (期望 {TOP_RATIO})")
print(f"参考 y[0,:4] = {y[0,:4]}")
