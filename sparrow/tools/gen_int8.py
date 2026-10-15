"""⑥ int8（per-row scale）的确定性权重 + A/B 素材（Sparrow V0.1.5）。

内核事实（`vmlm/ops/src/kernels.rs::gemm_nt_int8`）：

    s = Σ_k a[k] · f32(q[k])        （8 宽 FMA 累加 + 尾部标量）
    out[j] = s · scale[j]           （**整行共用一个 scale**，乘在内积之外）

所以判据要分两档，这是本脚本存在的理由：

* **k=1（单元素内积）**：`out = (a·q)·scale`。取 `scale = 2^-6` 且 `w = q/64`
  ⇒ `w` 在 f32 里**精确可表示**，且"先乘 q 再乘 2^-6"与"直接乘 w"在实数上相等
  （乘 2 的幂与舍入可交换）⇒ f32 路径与 int8 路径必须**逐字符相同**。
  这条是强判据：行列布局、scale 的**行**下标、bias 位置错一个都会立刻红。

* **k>1**：两条路的**累加顺序不同**（`gemm_nt` 是 AVX2 8×8 分块；int8 核是
  8 宽单行 FMA + 尾部标量）⇒ 即使权重精确可表示，和也可能差 ~1 ULP。
  所以判据只能是"差值 ≤ 实测界"，而**can-fail**（动一行 scale）必须远超该界。

产出（三个目录，都是小文件）：

    examples/int8_bins/        精确可表示（w = q/64），模型 i8o(k=1) 与 i8t(k=4)
    examples/int8_naive_bins/  朴素量化（scale = max|row|/127）⇒ 看真实量化损失
    examples/int8_bad_bins/    精确集 + 把 a_scale[0] 动 2^-6 ⇒ can-fail

用法: python tools/gen_int8.py
"""

import os
import shutil
import sys

import numpy as np

# 打印里有 `⇒` 这类字符：调用方没设 PYTHONIOENCODING 时，Windows 的 GBK 控制台会
# UnicodeEncodeError（实测）。这里自己钉死 UTF-8，不依赖调用方。
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import os as _os_pathfix
_HERE = _os_pathfix.path.dirname(_os_pathfix.path.abspath(__file__))
ROOT = _os_pathfix.path.dirname(_HERE)          # 本文件在 <root>/tools/ 下
EX = os.path.join(ROOT, "examples")
DIM, OUT_DIM = 4, 4
Q_MAX = 127
SCALE = 64.0          # scale = 1/64（2 的幂 ⇒ w = q/64 精确可表示）
f32 = np.float32


def save(d, name, arr, dtype="<f4"):
    os.makedirs(d, exist_ok=True)
    np.asarray(arr).astype(dtype).tofile(os.path.join(d, name))


def main() -> None:
    exact = os.path.join(EX, "int8_bins")
    naive = os.path.join(EX, "int8_naive_bins")
    bad = os.path.join(EX, "int8_bad_bins")
    for d in (exact, naive, bad):
        shutil.rmtree(d, ignore_errors=True)

    rng = np.random.default_rng(626)

    # ---- i8o：k=1（单元素内积）----
    q1 = rng.integers(-Q_MAX, Q_MAX + 1, size=(OUT_DIM, 1)).astype(np.int8)
    w1 = (q1.astype(f32) / f32(SCALE)).astype(f32)          # 精确可表示
    assert np.array_equal(w1, q1.astype(f32) / f32(SCALE)), "q/64 必须精确"
    save(exact, "o_W.bin", w1)
    save(exact, "o_Wq.bin", q1, "<i1")
    save(exact, "o_scale.bin", np.full(OUT_DIM, 1.0 / SCALE, dtype=f32))
    save(exact, "o_b.bin", rng.standard_normal(OUT_DIM).astype(f32))
    save(exact, "i8o_in.bin", np.array([[1.25]], dtype=f32))

    # ---- i8t：k=4（两层线性）----
    for tag in ("a", "b"):
        q = rng.integers(-Q_MAX, Q_MAX + 1, size=(OUT_DIM, DIM)).astype(np.int8)
        w = (q.astype(f32) / f32(SCALE)).astype(f32)
        save(exact, f"{tag}_W.bin", w)
        save(exact, f"{tag}_Wq.bin", q, "<i1")
        save(exact, f"{tag}_scale.bin", np.full(OUT_DIM, 1.0 / SCALE, dtype=f32))
        save(exact, f"{tag}_b.bin", rng.standard_normal(OUT_DIM).astype(f32))
    save(exact, "i8t_in.bin", rng.standard_normal((1, DIM)).astype(f32))

    # ---- 朴素量化：f32 权重 + 逐行 max|w|/127 的 scale ----
    for tag in ("a", "b"):
        w = rng.standard_normal((OUT_DIM, DIM)).astype(f32)
        sc = (np.abs(w).max(axis=1) / Q_MAX).astype(f32)
        q = np.clip(np.rint(w / sc[:, None]), -Q_MAX, Q_MAX).astype(np.int8)
        save(naive, f"{tag}_W.bin", w)                      # f32 参照（**未量化**）
        save(naive, f"{tag}_Wq.bin", q, "<i1")
        save(naive, f"{tag}_scale.bin", sc)
        save(naive, f"{tag}_b.bin", rng.standard_normal(OUT_DIM).astype(f32))
    save(naive, "i8t_in.bin", np.fromfile(os.path.join(exact, "i8t_in.bin"), dtype="<f4"))

    # ---- can-fail：精确集里把 a_scale[0] 动 2^-6 ----
    shutil.copytree(exact, bad)
    sc = np.fromfile(os.path.join(bad, "a_scale.bin"), dtype="<f4").copy()
    sc[0] = f32(sc[0] + f32(1.0 / SCALE))
    sc.astype("<f4").tofile(os.path.join(bad, "a_scale.bin"))

    # 判别性自检
    assert not np.array_equal(q1, np.zeros_like(q1)), "量化权重不能全零"
    print(f"  精确集：w = q/64（q∈[-127,127]）⇒ f32 权重与反量化结果**逐位相同**")
    print(f"  朴素集：scale = max|row|/127 ⇒ 有真实量化损失（用于对照）")
    print(f"  can-fail：a_scale[0] += 1/64（整行输出必须跟着变）")
    for d in (exact, naive, bad):
        print(f"  {d}: {len(os.listdir(d))} 个文件")


if __name__ == "__main__":
    main()
