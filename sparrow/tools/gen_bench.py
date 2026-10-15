"""AOT（--backend lib）vs 解释器（.vmb）基准的确定性模型生成器（Sparrow V0.1.5）。

模型 `chain<N>`：嵌入(词表 8, 维度 D) → l1 … lN（线性 D→D）
  * 每层权重用**不同 seed**（与 gen_loop24 同一条纪律：层间权重必须可区分）
  * **每行权重做 L2 归一化** —— 这条不是审美：不归一化时每层增益 ≈ 2（随机矩阵
    谱半径），96 层后 ≈ 2^96 ⇒ 全 NaN；而引擎的 `print_f32` 是**截断**实现，
    `NaN as u64 = 0`、`frac` 也是 NaN ⇒ **NaN 被打成 `0.000000`**。于是
    「AOT 与引擎逐字符相同」和「改权重输出必须变」**两条判据同时恒真** ——
    实测就是这样被 can-fail 抓住的（见 REPORT_V015.md §2.5）。行归一化后每层
    近似保范，激活量级恒定。
  * 前向只有 N 次 GEMM + BIAS ⇒ 每层成本 ≈ D² MAC，规模可用 N 线性放大

本脚本自带**判别性自检**（用 numpy 走一遍 f32 前向）：
  1. 输出必须全有限（无 NaN/Inf）；
  2. 输出取值不得退化（不同值个数 > D/2，且不是全零）；
  3. 扰动 `l1_W[0] += 1.0` 必须让输出变化 > 1e-3（否则驱动里的 can-fail 恒真）。

用法: python tools/gen_bench.py <tag> <N> [D]
产出: out/aot_bench/model_<tag>.sparrow
      out/aot_bench/bins_<tag>/   <tag>_in.bin, EMBED_E.bin, l{i}_W.bin, l{i}_b.bin
"""

import os
import sys

import numpy as np

import os as _os_pathfix
_HERE = _os_pathfix.path.dirname(_os_pathfix.path.abspath(__file__))
ROOT = _os_pathfix.path.dirname(_HERE)          # 本文件在 <root>/tools/ 下
BASE = os.path.join(ROOT, "out", "aot_bench")
VOCAB = 8
TOKEN = 3
f32 = np.float32


def main() -> None:
    tag = sys.argv[1]
    n = int(sys.argv[2])
    d = int(sys.argv[3]) if len(sys.argv) > 3 else 256

    bindir = os.path.join(BASE, f"bins_{tag}")
    os.makedirs(bindir, exist_ok=True)

    rng = np.random.default_rng(4242)
    e = rng.standard_normal((VOCAB, d)).astype(f32)

    ws, bs = [], []
    for i in range(1, n + 1):
        r = np.random.default_rng(90000 + i)                 # 每层一个不同 seed
        w = r.standard_normal((d, d))
        # 行归一化：每行单位 L2 范数 ⇒ 每层近似保范（见模块 docstring 的实测理由）
        w /= np.linalg.norm(w, axis=1, keepdims=True)
        ws.append(w.astype(f32))
        bs.append(r.standard_normal(d).astype(f32))

    # ---- 判别性自检（numpy f32 前向，不调引擎、不读 .vml） ----
    assert len({w.tobytes() for w in ws}) == n, "各层权重要互不相同"

    def forward(mut_first=None):
        x = e[TOKEN].copy()
        for i, (w, b) in enumerate(zip(ws, bs)):
            wm = mut_first if (mut_first is not None and i == 0) else w
            x = (wm @ x).astype(f32) + b
        return x

    y = forward()
    assert np.isfinite(y).all(), "前向出现 NaN/Inf ⇒ 判据会恒真（引擎把 NaN 打成 0.000000）"
    assert float(np.abs(y).max()) < 1e6, f"输出量级失控：max|y|={float(np.abs(y).max()):.3g}"
    txt = ["%.6f" % v for v in y]
    n_distinct = len(set(txt))
    assert n_distinct > d // 2, f"输出取值退化：只有 {n_distinct}/{d} 个不同值"
    assert not all(t == "0.000000" for t in txt), "输出全零 ⇒ 判据恒真"

    mut = ws[0].copy()
    mut[0, 0] += f32(1.0)
    gap = float(np.abs(forward(mut) - y).max())
    assert gap > 1e-3, f"扰动 l1_W[0] 只让输出变 {gap:.3g} ⇒ 驱动的 can-fail 判据太弱"
    print(f"  判别性自检：{d} 个数里 {n_distinct} 个不同，max|y|={float(np.abs(y).max()):.3f}，"
          f"扰动 l1_W[0] 最大变化 {gap:.4g}")

    def save(name, arr):
        a = np.asarray(arr, dtype=f32)
        a.astype("<f4").tofile(os.path.join(bindir, name))

    save("EMBED_E.bin", e)
    save(f"{tag}_in.bin", np.array([float(TOKEN)]))
    for i, (w, b) in enumerate(zip(ws, bs), start=1):
        save(f"l{i}_W.bin", w)
        save(f"l{i}_b.bin", b)

    lines = [
        f"# 由 tools/gen_bench.py 生成：chain{n}，D={d}（基准用，别手改）",
        f"模型 {tag}:",
        f"  嵌入 词表={VOCAB} 维度={d}",
    ]
    lines += [f"  l{i} = 线性 输出={d}" for i in range(1, n + 1)]
    lines.append("  连接 嵌入 l1")
    lines += [f"  连接 l{i - 1} l{i}" for i in range(2, n + 1)]
    lines.append(f"  前向 嵌入 l{n}")
    src = os.path.join(BASE, f"model_{tag}.sparrow")
    with open(src, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    macs = n * d * d
    print(f"  {tag}: N={n} D={d} 权重 {len(os.listdir(bindir))} 个文件，"
          f"前向 {macs} MAC（{2 * macs / 1e6:.1f} MFLOP）")
    print(f"  模型源 {src}")


if __name__ == "__main__":
    main()
