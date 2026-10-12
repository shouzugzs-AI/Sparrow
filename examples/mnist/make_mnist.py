#!/usr/bin/env python3
"""
P3 训练用数据/权重准备（工具层，体积不受限）

生成 MNIST 风格的小数据集（4 类，14x14 下采样），以及初始权重。
用法: python make_mnist.py
"""
import numpy as np
import os

OUT = os.path.dirname(os.path.abspath(__file__))

try:
    from torchvision import datasets, transforms
    HAS_TV = True
except ImportError:
    HAS_TV = False

rng = np.random.default_rng(2024)

IN, H, NCLS = 196, 64, 4   # 14x14 -> 64 -> 4 类


def synth():
    """没有 torchvision 时用可分的合成数据（保证能学会）"""
    n = 2000
    X = rng.standard_normal((n, IN)).astype(np.float32) * 0.5
    # 4 类：在 4 个不同维度区间上加固定 pattern
    Y = rng.integers(0, NCLS, n)
    for k in range(NCLS):
        m = Y == k
        X[m, k * 8:(k + 1) * 8] += 2.0
    return X, Y


def load_mnist():
    if not HAS_TV:
        print("未装 torchvision，使用合成数据")
        return synth()
    tr = datasets.MNIST(root=os.path.join(OUT, "mnist_data"), train=True,
                        download=True, transform=transforms.ToTensor())
    X = tr.data.numpy().astype(np.float32) / 255.0
    Y = tr.targets.numpy()
    # 下采样 28->14（2x2 平均）
    X = X.reshape(-1, 14, 2, 14, 2).mean(axis=(2, 4)).reshape(-1, 196)
    # 取 4 类
    m = Y < NCLS
    return X[m], Y[m]


X, Y = load_mnist()
n = len(X)
print(f"数据: X={X.shape} Y={Y.shape} 类别={sorted(set(Y.tolist()))}")

# 划分
idx = rng.permutation(n)
ntr = int(n * 0.9)
tr, te = idx[:ntr], idx[ntr:]

# 初始权重（He 初始化）
W1 = (rng.standard_normal((H, IN)) * np.sqrt(2.0 / IN)).astype(np.float32)
b1 = np.zeros(H, dtype=np.float32)
W2 = (rng.standard_normal((NCLS, H)) * np.sqrt(2.0 / H)).astype(np.float32)
b2 = np.zeros(NCLS, dtype=np.float32)

for name, arr in [("X", X[tr]), ("Y", Y[tr].astype(np.float32)),
                  ("Xte", X[te]), ("Yte", Y[te].astype(np.float32)),
                  ("W1", W1), ("b1", b1), ("W2", W2), ("b2", b2)]:
    arr.tofile(os.path.join(OUT, f"{name}.bin"))

with open(os.path.join(OUT, "meta.txt"), "w") as f:
    f.write(f"{len(tr)} {len(te)} {IN} {H} {NCLS}\n")

# PyTorch 参考（同样初始化，验证梯度/收敛对齐）
import torch
torch.set_num_threads(1)
tw1 = torch.tensor(W1, requires_grad=True)
tb1 = torch.tensor(b1, requires_grad=True)
tw2 = torch.tensor(W2, requires_grad=True)
tb2 = torch.tensor(b2, requires_grad=True)
opt = torch.optim.Adam([tw1, tb1, tw2, tb2], lr=1e-3)
tX = torch.tensor(X[tr]); tY = torch.tensor(Y[tr], dtype=torch.long)
losses = []
for step in range(200):
    opt.zero_grad()
    h = torch.relu(tX @ tw1.T + tb1)
    lo = h @ tw2.T + tb2
    loss = torch.nn.functional.cross_entropy(lo, tY)
    loss.backward()
    opt.step()
    losses.append(loss.item())
with open(os.path.join(OUT, "ref_loss.txt"), "w") as f:
    for l in losses:
        f.write(f"{l:.6f}\n")
print(f"PyTorch 参考: loss {losses[0]:.4f} -> {losses[-1]:.4f}")
print(f"训练集 {len(tr)} 测试集 {len(te)}")
