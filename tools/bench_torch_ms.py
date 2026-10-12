#!/usr/bin/env python3
"""
PyTorch 单核 GEMM 基线的**多次采样**测量（成员 D）。

为什么单独写这个脚本：
  Lead 和我分别测到 PyTorch 单核 512³ 为 122.7 / 111.95 / 125.2 GFLOPS，
  波动幅度 >10%。拿单次采样当基线会导致「相对倍率」这个指标完全不可信。
  本脚本用与 vmlm-bench 相同的统计口径（多次独立进程轮 + 每轮多次迭代取 min），
  并把每一轮的原始数字全部打印出来，便于判断波动来源。

口径说明（重要）：
  * torch.set_num_threads(1)：与白皮书 §5 一致，公平单核对比。
  * 取 min 而不是 mean：噪声只会让耗时变长，min 最接近真实速度。
  * 同时给出 median，以及 min 的离散度，让读者自己判断基线可信区间。

用法:
  python bench_torch_ms.py            # 512³, 5 轮 x 50 次
  python bench_torch_ms.py 512 512 512 5 50
"""
import sys
import time
import statistics
import numpy as np
import torch

torch.set_num_threads(1)

M = int(sys.argv[1]) if len(sys.argv) > 1 else 512
N = int(sys.argv[2]) if len(sys.argv) > 2 else 512
K = int(sys.argv[3]) if len(sys.argv) > 3 else 512
ROUNDS = int(sys.argv[4]) if len(sys.argv) > 4 else 5
REPS = int(sys.argv[5]) if len(sys.argv) > 5 else 50

FL = 2 * M * N * K
rng = np.random.default_rng(7)
A = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)
B = (rng.standard_normal((N, K)) * 0.1).astype(np.float32)
tA, tB = torch.from_numpy(A), torch.from_numpy(B)

print(f"torch {torch.__version__}  threads={torch.get_num_threads()}")
print(f"GEMM {M}x{N}x{K}  FLOPs={FL/1e6:.1f}MFLOP  rounds={ROUNDS} reps={REPS}")

for _ in range(5):
    _ = tA @ tB.T

mins = []
meds = []
for r in range(ROUNDS):
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        _ = tA @ tB.T
        t1 = time.perf_counter()
        ts.append((t1 - t0) * 1e6)
    ts.sort()
    mins.append(ts[0])
    meds.append(ts[len(ts) // 2])
    print(f"  round {r}: min={ts[0]:8.1f}us  med={ts[len(ts)//2]:8.1f}us  "
          f"max={ts[-1]:8.1f}us   {FL/(ts[0]/1e6)/1e9:6.2f} GFLOPS(min)")

best = min(mins)
print()
print(f"min-of-min   = {best:8.1f} us  -> {FL/(best/1e6)/1e9:6.2f} GFLOPS   <- 推荐口径")
print(f"median-of-med= {statistics.median(meds):8.1f} us  -> "
      f"{FL/(statistics.median(meds)/1e6)/1e9:6.2f} GFLOPS")
print(f"各轮 min 的相对离散: {(max(mins)-min(mins))/min(mins)*100:.1f}%  "
      f"({min(mins):.1f} .. {max(mins):.1f} us)")
