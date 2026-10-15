"""把 f32 权重转换成 bf16（以及"截断成 bf16 精度但仍是 f32"的对照），
用于 Sparrow V0.1.3 的 `--weight-dtype bf16` 验收。

bf16 在这里的定义由**引擎**给出（core/src/interp.rs / format.rs）：
    bf16 = f32 的**高 16 位**；加宽是 `(u32)h << 16` —— **精确、零成本**。
所以：

    widen(truncate_to_bf16(x)) == truncate_to_bf16(x)   （同一个 f32 位模式）

这一条给了验收一个**可判定**的判据：
  * A 组：`--weight-dtype bf16` + bf16 文件
  * B 组：默认 f32 + "已截断成 bf16 精度"的 f32 文件
两组喂给引擎的**位模式完全相同** ⇒ 输出必须相同（不允许"差不多"）。

产出到 examples/bf16_bins/：
    <name>.bin        f32，值 = 截断成 bf16 精度（B 组用）
    <name>.bf16.bin   bf16 原始 2 字节（A 组用）
并把 e2e_tiny 的所有常量都转一份（含必须保持 f32 的：偏置/CSR 索引）。
"""

import os
import numpy as np
import os as _os_pathfix
_HERE = _os_pathfix.path.dirname(_os_pathfix.path.abspath(__file__))
ROOT = _os_pathfix.path.dirname(_HERE)          # 本文件在 <root>/tools/ 下

SRC = _os.path.join(ROOT, "examples", "e2e_bins")
# 两组常量必须放**两个目录**：`.args` 里的文件名是固定的（`LINEAR_W.bin`），
# 汇编器靠 `:bf16` 后缀决定怎么解释**同一个文件名**的字节 —— 一个目录装不下
# "同名但不同编码"的两份数据。名字一样、内容不同，正是这份验收的要点。
OUT_F32 = _os.path.join(ROOT, "examples", "bf16_bins_f32")    # B 组：f32，值已截断成 bf16 精度
OUT_BF16 = _os.path.join(ROOT, "examples", "bf16_bins_bf16")  # A 组：真 bf16（2 字节/元素）

# 哪些常量**可以**用 bf16：只有 GEMM 的 b（线性层权重）与 EMB 的表（嵌入表）
# —— 与前端 `--weight-dtype` 的适用范围一致（引擎装载期还会再审计一次）。
CAN_BF16 = ("LINEAR_W.bin", "EMBED_E.bin")

os.makedirs(OUT_F32, exist_ok=True)
os.makedirs(OUT_BF16, exist_ok=True)
fed = []
for fn in sorted(os.listdir(SRC)):
    if not fn.endswith(".bin"):
        continue
    a = np.fromfile(os.path.join(SRC, fn), dtype="<f4")
    bits = a.view(np.uint32)
    trunc = (bits & np.uint32(0xFFFF0000)).view(np.float32)
    if fn in CAN_BF16:
        # B 组：f32 存储，但数值已经只有 bf16 精度
        # （= A 组里那块 bf16 被引擎加宽回来之后**逐位相同**的值）
        trunc.astype("<f4").tofile(os.path.join(OUT_F32, fn))
        # A 组：真正的 bf16（高 16 位原样写盘）
        (bits >> np.uint32(16)).astype("<u2").tofile(os.path.join(OUT_BF16, fn))
        fed.append(fn)
    else:
        # 其余常量（偏置 / CSR 索引 / 词元 id）两组**都必须保持原值 f32**：
        # 引擎装载期审计会拒绝 bf16 出现在 BIAS/LN/SPARSE_MATVEC 的操作数上（exit 9），
        # 前端也不会给它们标 bf16。早先这里把 B 组也截断了 —— 于是"两组位模式相同"
        # 这个前提被破坏，比较出来的 1.5e-3 差异全是**自己的参考错**，不是引擎的。
        a.astype("<f4").tofile(os.path.join(OUT_F32, fn))
        a.astype("<f4").tofile(os.path.join(OUT_BF16, fn))

# 逐位自检：加宽回来的位模式必须与截断版**完全相同**（否则判据本身不成立）
rng = np.random.default_rng(3)
for fn in fed:
    a = np.fromfile(os.path.join(SRC, fn), dtype="<f4")
    trunc = (a.view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)
    widened = (a.view(np.uint32) >> np.uint32(16)).astype(np.uint32) << np.uint32(16)
    widened = widened.view(np.float32)
    assert np.array_equal(widened.view(np.uint32), trunc.view(np.uint32)), fn
print(f"bf16 转换完成：f32 组 {len(os.listdir(OUT_F32))} 个 / bf16 组 {len(os.listdir(OUT_BF16))} 个文件；可降精度的是 {fed}")
print("自检：widen(bf16(x)) 与 truncate(x) 的位模式逐元素相同")
