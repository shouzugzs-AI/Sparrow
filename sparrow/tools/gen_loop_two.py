"""`examples/loop_two.sparrow` 的确定性权重生成器（Sparrow V0.1.5）。

补的是一个**长期缺口**：V0.1.2 起就有循环化，但"多个循环组串联"一直只有单组实测
（见 REPORT_V012.md 的边界清单）。模型里两个 `重复` 组串联：

    a = 线性 输出=4 重复3次      组 1（3 份）
    b = 线性 输出=4 重复2次      组 2（2 份，接在组 1 后面）
    mg = 线性 输出=4            组外的汇入层

验收用法（tools/lib_ab.ps1）：
  * `.vmb` 循环版输出 == Rust 循环版输出（逐字符）
  * `.vmb` 循环版输出 == `.vmb` **展开版**输出（E7：这是这一族判据的支点）
  * 每份实例的权重要互不相同 —— 否则步长/顺序错一位也看不出来（同 gen_loop24 的纪律）

用法: python tools/gen_loop_two.py
"""

import os

import numpy as np

import os as _os_pathfix
_HERE = _os_pathfix.path.dirname(_os_pathfix.path.abspath(__file__))
ROOT = _os_pathfix.path.dirname(_HERE)          # 本文件在 <root>/tools/ 下
OUT = os.path.join(ROOT, "examples", "loop_two_bins")
VOCAB, DIM = 8, 4
TOKEN = 3.0
GROUPS = (("a", 3), ("b", 2))          # 两个循环组：名字 + 份数
f32 = np.float32


def main() -> None:
    os.makedirs(OUT, exist_ok=True)

    def save(name, arr):
        np.asarray(arr, dtype=f32).astype("<f4").tofile(os.path.join(OUT, name))

    rng = np.random.default_rng(31)
    save("EMBED_E.bin", rng.standard_normal((VOCAB, DIM)))
    save("l2_in.bin", np.array([TOKEN]))

    seen = set()
    for gname, n in GROUPS:
        for k in range(1, n + 1):
            # 固定整数 seed（**不要**用 Python 的 hash()：它每个进程都不一样）
            r = np.random.default_rng(5000 + 100 * ord(gname) + k)
            w = r.standard_normal((DIM, DIM)).astype(f32)
            b = r.standard_normal(DIM).astype(f32)
            save(f"{gname}__{k}_W.bin", w)
            save(f"{gname}__{k}_b.bin", b)
            seen.add(w.tobytes())
    assert len(seen) == sum(n for _, n in GROUPS), "每份实例的权重必须互不相同"

    r = np.random.default_rng(6001)
    save("mg_W.bin", r.standard_normal((DIM, DIM)).astype(f32))
    save("mg_b.bin", r.standard_normal(DIM).astype(f32))

    print(f"  产出 {len(os.listdir(OUT))} 个权重文件到 {OUT}")
    print(f"  两个循环组：{' · '.join(f'{g}×{n}' for g, n in GROUPS)}，"
          f"每份权重互不相同（{len(seen)} 份）")


if __name__ == "__main__":
    main()
