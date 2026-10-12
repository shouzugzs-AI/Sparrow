#!/usr/bin/env python3
"""
生成 P0 测试权重 + 参考实现（工具层，体积不受限）
产出: W1.bin W2.bin x.bin  以及 expected.txt
"""
import numpy as np
import struct

rng = np.random.default_rng(1234)

IN, H, OUT = 64, 32, 10

W1 = rng.standard_normal((H, IN)).astype(np.float32) * 0.1
b1 = rng.standard_normal(H).astype(np.float32) * 0.1
W2 = rng.standard_normal((OUT, H)).astype(np.float32) * 0.1
x = rng.standard_normal((1, IN)).astype(np.float32)

# 参考实现
h = x @ W1.T + b1
h = np.maximum(h, 0)
y = h @ W2.T

for name, arr in [("W1", W1), ("b1", b1), ("W2", W2), ("x", x)]:
    arr.tofile(f"W_{name}.bin" if False else f"{name}.bin")

np.savetxt("expected.txt", y, fmt="%.8f")
print("W1", W1.shape, "W2", W2.shape, "x", x.shape)
print("expected y[0][:5] =", y[0][:5])
