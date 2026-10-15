"""e2e_tiny 的确定性权重/常量生成器（Sparrow V0.1.1 端到端验收用）。

产出 7 个 .bin（小端 f32），全部由固定 seed 决定 —— 重跑必然逐字节相同：

    tiny_in.bin    [1,1]  词元 id（=3.0）
    EMBED_E.bin    [8,4]  嵌入表 [vocab, dim]
    SPARSE_off.bin [5,1]  CSR 行偏移（f32 存整数：0,2,4,6,8）
    SPARSE_src.bin [8,1]  CSR 列索引（f32 存整数）
    SPARSE_W.bin   [8,1]  CSR 连接权重
    DENSE_W.bin    [4,4]  稀疏层的**稠密等价**矩阵（W[n,k] = 连接 n←k 的权重）
    LINEAR_W.bin   [4,4]  线性层权重 [out, in]（NT）
    LINEAR_b.bin   [1,4]  线性层偏置

DENSE_W 是关键：手写参考 .vml 用它走 `GEMM`，生成 .vml 走 `SPARSE_MATVEC`(CSR)。
两条**不同的指令路径**算同一个函数，输出逐字节相同才算端到端成立。

连接拓扑（V0.1.1 固定规则）：神经元 j 的第 t 条连接指向输入 (j*k0+t) % dim，
所以行内列号单调递增 —— 与 CSR 的遍历顺序一致，累加顺序相同 ⇒ 可逐位比较。
"""

import os
import numpy as np
import os as _os_pathfix
_HERE = _os_pathfix.path.dirname(_os_pathfix.path.abspath(__file__))
ROOT = _os_pathfix.path.dirname(_HERE)          # 本文件在 <root>/tools/ 下

OUT = _os.path.join(ROOT, "examples", "e2e_bins")
VOCAB, DIM, K0, OUT_DIM = 8, 4, 2, 4
TOKEN = 3.0

os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(7)
f32 = np.float32

E = rng.standard_normal((VOCAB, DIM)).astype(f32)            # [vocab, dim]
off = np.arange(0, DIM * K0 + 1, K0, dtype=f32)              # [dim+1]
src = np.array([(j * K0 + t) % DIM for j in range(DIM) for t in range(K0)], dtype=f32)
w = rng.standard_normal(DIM * K0).astype(f32)
Lw = rng.standard_normal((OUT_DIM, DIM)).astype(f32)         # [out, in]  (NT)
Lb = rng.standard_normal(OUT_DIM).astype(f32)                # [1, out]

# 稀疏层的稠密等价矩阵
Wd = np.zeros((DIM, DIM), dtype=f32)
for j in range(DIM):
    for t in range(K0):
        Wd[j, int(src[j * K0 + t])] += w[j * K0 + t]
assert np.all(np.sort(src.reshape(DIM, K0), axis=1) == src.reshape(DIM, K0)), "行内列号必须递增"


def dump(name, arr, rows, cols):
    a = np.ascontiguousarray(arr, dtype=f32).reshape(rows, cols)
    p = os.path.join(OUT, name)
    a.tofile(p)
    assert os.path.getsize(p) == rows * cols * 4, (name, os.path.getsize(p))
    print(f"{name:<16} [{rows},{cols}]  {rows * cols * 4} B")


dump("tiny_in.bin", np.array([TOKEN], dtype=f32), 1, 1)
dump("EMBED_E.bin", E, VOCAB, DIM)
dump("SPARSE_off.bin", off, DIM + 1, 1)
dump("SPARSE_src.bin", src, DIM * K0, 1)
dump("SPARSE_W.bin", w, DIM * K0, 1)
dump("DENSE_W.bin", Wd, DIM, DIM)
dump("LINEAR_W.bin", Lw, OUT_DIM, DIM)
dump("LINEAR_b.bin", Lb, 1, OUT_DIM)

# 手工参考值（供报告核对：与引擎输出应一致）
h0 = E[int(TOKEN)]                       # [4]
h1 = Wd @ h0                             # [4]
y = Lw @ h1 + Lb                         # [4]
print("\n手工参考（numpy，f32 单精度顺序累加）:")
print(" h0 =", " ".join(f"{v:.6f}" for v in h0))
print(" h1 =", " ".join(f"{v:.6f}" for v in h1))
print(" y  =", " ".join(f"{v:.6f}" for v in y))
