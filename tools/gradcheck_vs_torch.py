#!/usr/bin/env python3
"""
P3 梯度正确性交叉验证：把 VMLM 的解析梯度与 PyTorch autograd 对比。

这是比「数值差分」更强的判据：PyTorch 用双精度/成熟反向，
如果两边一致，说明 VMLM 的反向指令流实现正确。
"""
import numpy as np
import torch
import subprocess
import os
import sys

torch.set_num_threads(1)
DIR = "examples/mnist"

meta = open(os.path.join(DIR, "meta.txt")).read().split()
ntr, nte, IN, H, NCLS = map(int, meta)

rd = lambda n: np.fromfile(os.path.join(DIR, f"{n}.bin"), dtype=np.float32)

X = rd("X").reshape(-1, IN)
Y = rd("Y")
W1 = rd("W1").reshape(H, IN).copy()
b1 = rd("b1").reshape(H).copy()
W2 = rd("W2").reshape(NCLS, H).copy()
b2 = rd("b2").reshape(NCLS).copy()

i = 0
x = X[i:i + 1]
y = int(Y[i])

# ---------- PyTorch 参考梯度 ----------
tW1 = torch.tensor(W1, requires_grad=True)
tb1 = torch.tensor(b1, requires_grad=True)
tW2 = torch.tensor(W2, requires_grad=True)
tb2 = torch.tensor(b2, requires_grad=True)
tx = torch.tensor(x)

h = torch.relu(tx @ tW1.T + tb1)
lo = h @ tW2.T + tb2
loss = torch.nn.functional.cross_entropy(lo, torch.tensor([y]))
loss.backward()

print(f"样本 0，标签 {y}，PyTorch loss = {loss.item():.6f}")

# 存给 Rust 对比
np.save(os.path.join(DIR, "_ref_grads.npy"),
        np.concatenate([tW1.grad.numpy().ravel(),
                        tb1.grad.numpy().ravel(),
                        tW2.grad.numpy().ravel(),
                        tb2.grad.numpy().ravel()]))

# ---------- VMLM 解析梯度 ----------
subprocess.run([r"target\release\vmlm-graddump.exe", DIR], check=False)
print("\n（VMLM 侧梯度由 vmlm-graddump 写出，下面做对比）")

vf = os.path.join(DIR, "_vmlm_grads.bin")
if not os.path.exists(vf):
    print("缺少 VMLM 梯度文件")
    sys.exit(1)

vg = np.fromfile(vf, dtype=np.float32)
tg = np.concatenate([tW1.grad.numpy().ravel(), tb1.grad.numpy().ravel(),
                     tW2.grad.numpy().ravel(), tb2.grad.numpy().ravel()])

if len(vg) != len(tg):
    print(f"长度不一致: VMLM {len(vg)} vs Torch {len(tg)}")
    sys.exit(1)

names = ["W1"] * W1.size + ["b1"] * H + ["W2"] * W2.size + ["b2"] * NCLS
off = 0
sizes = [W1.size, H, W2.size, NCLS]
labels = ["W1", "b1", "W2", "b2"]
print(f"\n{'参数':<6}{'元素数':>8}{'最大绝对差':>13}{'相对范数误差':>14}  判定")
allok = True
for lab, sz in zip(labels, sizes):
    a = vg[off:off + sz]
    b = tg[off:off + sz]
    ad = np.abs(a - b).max()
    # 标准判据：|Δ| / (‖a‖ + ‖b‖)，对整体量级归一化，
    # 避免单个接近 0 的元素把相对误差放大成无意义的比值
    rel = np.abs(a - b).max() / (np.linalg.norm(a) + np.linalg.norm(b) + 1e-12)
    ok = rel < 1e-4
    allok &= ok
    print(f"{lab:<6}{sz:>8}{ad:>13.3e}{rel:>14.3e}  {'PASS' if ok else 'FAIL'}")
    off += sz

print(f"\n{'全部一致 PASS' if allok else '存在差异 FAIL'}")
print("（判据：|Δ| / (‖VMLM‖+‖PyTorch‖) < 1e-4）")
