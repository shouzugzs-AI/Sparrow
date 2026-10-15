"""int8 铺到**注意力投影**的确定性素材（Sparrow V0.1.6）。

模型（examples/int8_attn.sparrow）：

    i8attn: 线性(4→4) → 注意力(维度=4 头=2)

两个素材目录（都是**精确可表示**的：`w = q/64`，q 为整数且 |q| ≤ 127）：

    examples/int8_attn_bins/       f32 权重（对照组）
        i8attn_in.bin   [4,4]（序列=4）
        h1_W.bin       [4,4]    h1_b.bin  [1,4]
        ATTN_Wq/Wk/Wv/Wo.bin  [4,4]     ATTN_bWq/bWk/bWv/bWo.bin  [1,4]
        ATTN_scale.bin  [1,1]    softmax 的 1/sqrt(hd)

    examples/int8_attn_i8_bins/    int8（`--weight-dtype int8`）
        i8attn_in.bin   [4,4]（序列=4）（同 f32）
        h1_Wq.bin (i8)  h1_scale.bin (f32[4])  h1_b.bin
        ATTN_W*.bin (i8) ATTN_sW*.bin (f32[4])  ATTN_bW*.bin  ATTN_scale.bin

`--weight-dtype int8` 对**线性层与注意力投影同时生效**，所以两个目录的差别就是
"权重哪来的"：一个是 f32 原值，一个是 `q`（i8）+ 每输出行一个 `1/64`。

为什么精确可表示：`scale = 1/64` 是 2 的幂 ⇒ `q/64` 在 f32 里**无误差**，
于是"两条路的差值"只反映**内核累加顺序**（`gemm_nt` 是 AVX2 分块，`gemm_nt_int8`
是 8 宽 FMA），不掺杂量化损失 —— 这是能断言"逐字符相同/差值有界"的前提。

判别性自检（防判据恒真）：
  * 每个权重行至少要有一个非零元素（否则"相同"可能来自"两边都是 0"）；
  * `q` 必须能还原出原值（`q/64 == w` 逐元素），否则这个素材集就不是精确可表示的。
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
F32 = os.path.join(ROOT, "examples", "int8_attn_bins")
I8 = os.path.join(ROOT, "examples", "int8_attn_i8_bins")

D = 4
HEADS = 2


def write_f32(path, arr):
    with open(path, "wb") as f:
        f.write(np.asarray(arr, dtype="<f4").ravel().tobytes())


def write_i8(path, q):
    with open(path, "wb") as f:
        f.write(np.asarray(q, dtype=np.int8).ravel().tobytes())


def quantize_exact(w):
    """w = q/64 的精确量化；同时返回还原值用于自检。"""
    q = np.rint(w * 64.0).astype(np.int32)
    assert np.abs(q).max() <= 127, "超出 int8 范围"
    back = q.astype(np.float64) / 64.0
    assert np.array_equal(back, w.astype(np.float64)), "素材不是精确可表示的（w != q/64）"
    return q.astype(np.int8), back


def main():
    os.makedirs(F32, exist_ok=True)
    os.makedirs(I8, exist_ok=True)
    rng = np.random.RandomState(20261015)

    def w_mat(rows, cols):
        # 元素取 k/8 ⇒ 乘 8 就是整数，配合 scale=1/64 精确可表示
        m = rng.randint(-4, 5, size=(rows, cols)).astype(np.float64) / 8.0
        assert (m != 0).any(axis=1).all(), "每行都要有非零元素"
        return m

    x = (rng.randn(4, D) * 0.7).astype(np.float64)   # 序列=4 ⇒ 输入 [T,D]
    h1_W = w_mat(D, D)
    h1_b = (rng.randn(D) * 0.1).astype(np.float64)
    proj = {k: w_mat(D, D) for k in ["Wq", "Wk", "Wv", "Wo"]}
    bias = {k: (rng.randn(D) * 0.1).astype(np.float64) for k in ["bWq", "bWk", "bWv", "bWo"]}
    attn_scale = np.array([1.0 / np.sqrt(D / HEADS)], dtype=np.float64)

    # --- f32 目录 ---
    write_f32(os.path.join(F32, "i8attn_in.bin"), x)
    write_f32(os.path.join(F32, "h1_W.bin"), h1_W)
    write_f32(os.path.join(F32, "h1_b.bin"), h1_b)
    for k, v in proj.items():
        write_f32(os.path.join(F32, f"ATTN_{k}.bin"), v)
    for k, v in bias.items():
        write_f32(os.path.join(F32, f"ATTN_{k}.bin"), v)
    write_f32(os.path.join(F32, "ATTN_scale.bin"), attn_scale)

    # --- int8 目录 ---
    write_f32(os.path.join(I8, "i8attn_in.bin"), x)
    q, _ = quantize_exact(h1_W)
    write_i8(os.path.join(I8, "h1_Wq.bin"), q)
    write_f32(os.path.join(I8, "h1_scale.bin"), np.full(D, 1.0 / 64.0))
    write_f32(os.path.join(I8, "h1_b.bin"), h1_b)
    for k, v in proj.items():
        q, _ = quantize_exact(v)
        write_i8(os.path.join(I8, f"ATTN_{k}.bin"), q)
        write_f32(os.path.join(I8, f"ATTN_s{k}.bin"), np.full(D, 1.0 / 64.0))
    for k, v in bias.items():
        write_f32(os.path.join(I8, f"ATTN_{k}.bin"), v)
    write_f32(os.path.join(I8, "ATTN_scale.bin"), attn_scale)

    # 体积：4 个投影 [4,4]（线性层另算，第 7 节只比投影）
    b_f32 = sum(os.path.getsize(os.path.join(F32, f"ATTN_{k}.bin")) for k in proj)
    b_i8 = sum(
        os.path.getsize(os.path.join(I8, f"ATTN_{k}.bin"))
        + os.path.getsize(os.path.join(I8, f"ATTN_s{k}.bin"))
        for k in proj
    )
    print(f"素材 → {F32}")
    print(f"素材 → {I8}")
    print(f"注意力四个投影：f32 {b_f32} B → int8(q+scale) {b_i8} B（比值 {b_f32 / b_i8:.3f}，公式 4/(1+4/4) = 2.0）")


if __name__ == "__main__":
    main()
