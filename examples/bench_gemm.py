#!/usr/bin/env python3
"""
P1 基准：VMLM GEMM vs PyTorch 单核
生成随机矩阵 + 参考结果，并计时对比。
"""
import subprocess
import sys
import time
import numpy as np
import torch

torch.set_num_threads(1)  # 公平单核对比

M, K, N = 512, 512, 512
rng = np.random.default_rng(7)
A = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)
B = (rng.standard_normal((N, K)) * 0.1).astype(np.float32)

A.tofile("A.bin")
B.tofile("B.bin")

ref = A @ B.T
np.savetxt("bench_expected.txt", ref.reshape(-1)[:16], fmt="%.6f")
print(f"shape: A[{M},{K}] x B[{N},{K}]^T -> [{M},{N}]")
print(f"FLOPs = 2*M*N*K = {2*M*N*K/1e6:.1f} MFLOP")

# --- PyTorch 单核计时 ---
tA = torch.from_numpy(A)
tB = torch.from_numpy(B)
for _ in range(3):
    _ = tA @ tB.T
reps = 50
t0 = time.perf_counter()
for _ in range(reps):
    _ = tA @ tB.T
t1 = time.perf_counter()
pt_ms = (t1 - t0) / reps * 1000
pt_gflops = 2 * M * N * K / (pt_ms / 1000) / 1e9
print(f"PyTorch(1线程): {pt_ms:.3f} ms  {pt_gflops:.2f} GFLOPS")

# 存参考供校验
np.savetxt("bench_full.txt", ref, fmt="%.6f")
