"""训练路径（`损失 均方` + `反向`）的确定性素材 + numpy 独立参考（Sparrow V0.1.6）。

模型（examples/train_mse.sparrow）：

    tmse: 线性(4→4) → 激活(tanh) → 线性(4→2)，损失 = 均方误差

产出（小端 f32）：

    examples/train_bins/tmse_in.bin   [1,4]   模型输入
    examples/train_bins/tmse_tgt.bin  [1,2]   损失目标
    examples/train_bins/x_W.bin / x_b.bin     第一层线性（[4,4] / [1,4]）
    examples/train_bins/y_W.bin / y_b.bin     第二层线性（[2,4] / [1,2]）

另外写出 out/train/ref.txt：**两份**参考，用途不同，缺一不可 ——

  1. `fd <参数> <值…>`：**中心差分**（纯数值，不含任何反向推导）。
     它只依赖前向 —— 所以"我的解析梯度算对了"这句话才有独立依据。
  2. `an <参数> <值…>`：numpy 自己按链式法则写的解析梯度。
     与引擎打印的值比（判据 A），再与 FD 比（判据 B）：
       A 证明引擎与 numpy 实现一致，B 证明这份解析梯度真的是梯度。
     只有 A 会被"两边都抄同一份错代码"骗过；只有 B 会在引擎侧写错时通过。

约定（逐条对齐引擎 vmlm/core/src/{interp,train}.rs）：
  * `GEMM dst, a, b` 是 NT：`y[1,N] = x[1,K] · W[N,K]^T`，即 `W` 是 [out, in] 行主序；
  * `MSE_BWD` 的梯度是 `2*(y-t)/n`，loss 是 `Σ(y-t)²/n`；
  * `GEMM_BWD_W`：`dW[N,K] += dy^T x`；`GEMM_BWD_X`：`dx = dy·W`。

判别性自检（防判据恒真）：梯度必须非退化 —— 至少一个 |∂L/∂w| > 1e-3，
且逐个参数扰动都要真的改变 loss（否则 FD 全是 0，"相同"就恒真）。
"""
import os
import struct
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BINS = os.path.join(ROOT, "examples", "train_bins")
OUT = os.path.join(ROOT, "out", "train")

D_IN, D_MID, D_OUT = 4, 4, 2
H = 1e-3  # 中心差分步长


def write_f32(path, arr):
    a = np.asarray(arr, dtype="<f4").ravel()
    with open(path, "wb") as f:
        f.write(a.tobytes())
    return a.size


def forward(P, x, t):
    """复刻引擎的前向：NT 线性 + 偏置 + tanh + MSE。"""
    Wx, bx, Wy, by = P
    h1 = x @ Wx.T + bx
    h2 = np.tanh(h1)
    y = h2 @ Wy.T + by
    n = y.size
    d = y - t
    return float((d * d).sum() / n)


def analytic(P, x, t):
    """numpy 侧按链式法则写的解析梯度（独立于 Rust 实现）。"""
    Wx, bx, Wy, by = P
    h1 = x @ Wx.T + bx
    h2 = np.tanh(h1)
    y = h2 @ Wy.T + by
    n = y.size
    dy = 2.0 * (y - t) / n
    dWy = dy.T @ h2
    dby = dy.sum(axis=0)
    dh2 = dy @ Wy
    dh1 = dh2 * (1.0 - h2 * h2)
    dWx = dh1.T @ x
    dbx = dh1.sum(axis=0)
    return [dWx, dbx, dWy, dby]


def fd_grad(P, x, t, idx):
    """对第 idx 个参数数组做中心差分（逐元素）。"""
    base = [p.copy() for p in P]
    g = np.zeros_like(base[idx])
    flat = g.reshape(-1)
    src = base[idx]
    for i in range(src.size):
        old = src.reshape(-1)[i]
        src.reshape(-1)[i] = old + H
        lp = forward(base, x, t)
        src.reshape(-1)[i] = old - H
        lm = forward(base, x, t)
        src.reshape(-1)[i] = old
        flat[i] = (lp - lm) / (2 * H)
    return g


def main():
    os.makedirs(BINS, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.RandomState(20261014)
    # 幅度刻意放小：tanh 不能饱和（饱和处导数≈0，梯度检查会退化成"两边都 0"）
    Wx = (rng.randn(D_MID, D_IN) * 0.5).astype(np.float64)
    bx = (rng.randn(D_MID) * 0.1).astype(np.float64)
    Wy = (rng.randn(D_OUT, D_MID) * 0.5).astype(np.float64)
    by = (rng.randn(D_OUT) * 0.1).astype(np.float64)
    x = (rng.randn(1, D_IN) * 0.7).astype(np.float64)
    t = (rng.randn(1, D_OUT) * 0.5).astype(np.float64)
    P = [Wx, bx, Wy, by]

    names = ["l1_W", "l1_b", "l2_W", "l2_b"]
    for nm, arr in zip(names, P):
        write_f32(os.path.join(BINS, nm + ".bin"), arr)
    write_f32(os.path.join(BINS, "tmse_in.bin"), x)
    write_f32(os.path.join(BINS, "tmse_tgt.bin"), t)

    loss = forward(P, x, t)
    an = analytic(P, x, t)
    fd = [fd_grad(P, x, t, i) for i in range(4)]

    # --- 判别性自检 ---
    scale = max(float(np.abs(g).max()) for g in an)
    assert scale > 1e-3, f"解析梯度退化（max={scale}）⇒ 判据会恒真"
    for nm, a, f in zip(names, an, fd):
        rel = np.abs(a - f) / np.maximum(np.abs(f), 1e-6)
        assert rel.max() < 1e-3, f"{nm}: numpy 解析 vs 中心差分 相对误差 {rel.max()} —— 参考自身不可信"
    # 逐个参数扰动必须改变 loss
    for i, nm in enumerate(names):
        Q = [p.copy() for p in P]
        Q[i].reshape(-1)[0] += H
        assert abs(forward(Q, x, t) - loss) > 1e-9, f"{nm} 扰动不改变 loss ⇒ can-fail 无意义"

    lines = []
    for nm, g in zip(names, an):
        lines.append("an " + nm + " " + " ".join(f"{v:.9g}" for v in np.asarray(g).ravel()))
    for nm, g in zip(names, fd):
        lines.append("fd " + nm + " " + " ".join(f"{v:.9g}" for v in np.asarray(g).ravel()))
    with open(os.path.join(OUT, "ref.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"素材 → {BINS}")
    print(f"loss = {loss:.9g}")
    print(f"参考 → {os.path.join(OUT, 'ref.txt')}")
    print("解析梯度最大绝对误差 vs 中心差分："
          + ", ".join(f"{nm}={np.abs(a - f).max():.3e}" for nm, a, f in zip(names, an, fd)))
    print("解析梯度量级：" + ", ".join(f"{nm}={np.abs(a).max():.6g}" for nm, a in zip(names, an)))


if __name__ == "__main__":
    main()
