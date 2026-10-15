"""loop24 的确定性权重生成器（Sparrow V0.1.2 P0-2 循环化验收用）。

关键：**24 层的权重必须互不相同**（每层一个不同 seed）。
步长错一位时第 1 层仍然算对，只有从第 2 层起读错权重 —— 若各层权重相同，
"展开版 vs 循环版逐字节相同"这条判据就恒真，等于没验。

产出（小端 f32）：

    loop24_in.bin        [1,1]   词元 id（=3.0）
    EMBED_E.bin          [8,4]   嵌入表
    t__{1..24}_off.bin   [5,1]   CSR 行偏移（D+1 = 5，f32 存整数）
    t__{1..24}_src.bin   [8,1]   CSR 列索引（D*K0 = 8）
    t__{1..24}_W.bin     [8,1]   CSR 连接权重（**每层不同 seed**）
    LINEAR_W.bin         [4,4]   线性层权重（NT）
    LINEAR_b.bin         [1,4]   线性层偏置

连接拓扑与 V0.1.1 一致：神经元 j 的第 t 条连接指向 (j*k0+t) % dim，
行内列号单调递增（与 CSR 遍历顺序一致 ⇒ 累加顺序相同 ⇒ 可逐位比较）。
"""

import os
import numpy as np
import os as _os_pathfix
_HERE = _os_pathfix.path.dirname(_os_pathfix.path.abspath(__file__))
ROOT = _os_pathfix.path.dirname(_HERE)          # 本文件在 <root>/tools/ 下

OUT = _os.path.join(ROOT, "examples", "loop_bins")
VOCAB, DIM, K0, OUT_DIM, N = 8, 4, 2, 4, 24
TOKEN = 3.0

os.makedirs(OUT, exist_ok=True)
f32 = np.float32


def save(name, arr):
    a = np.asarray(arr, dtype=f32)
    p = os.path.join(OUT, name)
    a.astype("<f4").tofile(p)
    return a


# 嵌入表与线性层：固定 seed，与 24 层无关
rng = np.random.default_rng(7)
E = rng.standard_normal((VOCAB, DIM)).astype(f32)
Lw = rng.standard_normal((OUT_DIM, DIM)).astype(f32)
Lb = rng.standard_normal(OUT_DIM).astype(f32)

off = np.arange(0, DIM * K0 + 1, K0, dtype=f32)                 # [dim+1]
src = np.array([(j * K0 + t) % DIM for j in range(DIM) for t in range(K0)], dtype=f32)

save("loop24_in.bin", np.array([TOKEN]))
# 模型输入常量的文件名带**模型名**（`{model}_in.bin`），而 A/B 对照模型
# （examples/loop6.sparrow）用的是同一批权重，所以两个名字都要有。
save("loop6_in.bin", np.array([TOKEN]))
save("EMBED_E.bin", E)
save("LINEAR_W.bin", Lw)
save("LINEAR_b.bin", Lb)

ws = []
for r in range(1, N + 1):
    # 每层一个不同 seed（1000+r 与原 rng 无关联，保证 24 组权重互不相同）
    r2 = np.random.default_rng(1000 + r)
    w = r2.standard_normal(DIM * K0).astype(f32)
    save(f"t__{r}_off.bin", off)
    save(f"t__{r}_src.bin", src)
    save(f"t__{r}_W.bin", w)
    ws.append(w)

# 判别性自检：24 组权重必须两两不同，且不能是"同一组 + 常数"这种伪不同
W = np.stack(ws)
assert len({w.tobytes() for w in ws}) == N, "24 层权重出现了重复 —— 判据会恒真"
d = np.abs(W[:, None, :] - W[None, :, :]).max(axis=2)
np.fill_diagonal(d, np.inf)
print(f"24 层权重互不相同：最小逐元素差 {d.min():.6g}")
assert d.min() > 1e-3, "两层权重过于接近：步长错误可能看不出来"

# 步长自检（与前端预测值对账）：off/src/W 三块的补齐尺寸之和
need = [(DIM + 1) * 4, DIM * K0 * 4, DIM * K0 * 4]
stride = sum((n + 63) & ~63 for n in need)
print(f"预测步长 = {stride} B（每层 3 块：{need} → 补齐 {[(n + 63) & ~63 for n in need]}）")
print(f"24 份常量共 {stride * N} B（不含其它层的常量）")
print(f"产出 {len(os.listdir(OUT))} 个文件到 {OUT}")
