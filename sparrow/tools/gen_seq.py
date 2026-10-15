"""`序列=T`（嵌入链首）的确定性权重 + numpy 参考（Sparrow V0.1.5 ⑤）。

模型（exaples/seq_embed.sparrow / seq_one.sparrow）：

    se3: 嵌入(词表 8, 维度 4) 序列=3 → 线性 输出=4     ← T>1，链首是嵌入
    se1: 嵌入(词表 8, 维度 4)         → 线性 输出=4     ← T=1 的对照

验收思路（tools/seq_ab.ps1）：**逐行独立性**。`嵌入 → 线性` 这条链里每一行互不影响，
所以 `se3` 输出的第 k 行必须与"只喂第 k 个词元"的 `se1` 输出**逐字符相同**。
这是一条不依赖任何第三方库的判据（两侧都是引擎自己）。

产出（小端 f32，examples/seq_bins/）：

    EMBED_E.bin   [8,4]   嵌入表
    se3_in.bin    [1,3]   一行 3 个词元 id（T>1 时输入是"一行 T 个 id"）
    se1_in.bin    [1,1]   单个词元 id（驱动会按词元逐个改写它）
    LINEAR_W.bin  [4,4] / LINEAR_b.bin [1,4]

另外写出 numpy 独立参考（复刻引擎的**截断**式打印，见 TRAPS A7）：
    out/seq/tok_<id>.ref.txt

判别性自检（防判据恒真）：
  * 3 个词元的嵌入行必须两两不同（否则"第 k 行 == 第 k 个词元的 T=1 输出"会恒真）；
  * 3 行的线性输出必须两两不同。

用法: python tools/gen_seq.py
"""

import os

import numpy as np

import os as _os_pathfix
_HERE = _os_pathfix.path.dirname(_os_pathfix.path.abspath(__file__))
ROOT = _os_pathfix.path.dirname(_HERE)          # 本文件在 <root>/tools/ 下
OUT = os.path.join(ROOT, "examples", "seq_bins")
REFOUT = os.path.join(ROOT, "out", "seq")
VOCAB, DIM, OUT_DIM = 8, 4, 4
TOKENS = [3, 5, 1]          # se3 的三个词元（互不相同、且嵌入行互不相同）
f32 = np.float32


def save(name, arr):
    np.asarray(arr, dtype=f32).astype("<f4").tofile(os.path.join(OUT, name))


def eng_fmt(v):
    """逐位复刻 vmlm 的 print_f32（f32 运算 + 截断，6 位小数）—— 见 TRAPS A7。"""
    x = f32(v)
    neg = bool(x < f32(0))
    av = f32(-x) if neg else x
    ip = int(av)
    frac = f32(av - f32(ip))
    s = ("-" if neg else "") + str(ip) + "."
    for _ in range(6):
        frac = f32(frac * f32(10.0))
        dig = int(frac)
        s += str(dig % 10)
        frac = f32(frac - f32(dig))
    return s


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(REFOUT, exist_ok=True)

    rng = np.random.default_rng(77)
    e = rng.standard_normal((VOCAB, DIM)).astype(f32)
    w = rng.standard_normal((OUT_DIM, DIM)).astype(f32)
    b = rng.standard_normal(OUT_DIM).astype(f32)

    save("EMBED_E.bin", e)
    save("se3_in.bin", np.array(TOKENS, dtype=f32))
    save("se1_in.bin", np.array([TOKENS[0]], dtype=f32))
    save("LINEAR_W.bin", w)
    save("LINEAR_b.bin", b)

    # `seq_loop.sparrow`：序列=3 + 一个 `重复2次` 的线性组。
    # 用途：**T>1 与循环组同时出现**时，"循环版 == 展开版"必须逐字符成立
    #（V0.1.4 这里是把进位槽行数写死成 1 ⇒ 只算出第一行，静默错答案）。
    lp_in = np.random.default_rng(88).standard_normal((3, DIM)).astype(f32)
    save("lp_in.bin", lp_in)
    for tag, seed in (("a", 11), ("t__1", 21), ("t__2", 22)):
        r = np.random.default_rng(seed)
        save(f"{tag}_W.bin", r.standard_normal((OUT_DIM, DIM)).astype(f32))
        save(f"{tag}_b.bin", r.standard_normal(OUT_DIM).astype(f32))

    # ---- 判别性自检 + numpy 参考 ----
    assert len(set(TOKENS)) == len(TOKENS), "词元必须互不相同"
    rows = []
    for t in TOKENS:
        rows.append(((e[t] @ w.T).astype(f32) + b).astype(f32))
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            assert not np.array_equal(rows[i], rows[j]), f"词元 {TOKENS[i]}/{TOKENS[j]} 的输出相同 ⇒ 判据恒真"
            assert not np.array_equal(e[TOKENS[i]], e[TOKENS[j]]), "嵌入行相同 ⇒ 判据恒真"

    print(f"  {len(TOKENS)} 个词元 {TOKENS}：嵌入行与输出行都两两不同（判据有判别力）")
    for t, y in zip(TOKENS, rows):
        txt = " ".join(eng_fmt(v) for v in y)
        with open(os.path.join(REFOUT, f"tok_{t}.ref.txt"), "w", encoding="ascii") as f:
            f.write(txt + "\n")
        print(f"  tok={t}  numpy 参考 = {txt}")
    print(f"  产出 {len(os.listdir(OUT))} 个权重文件到 {OUT}")


if __name__ == "__main__":
    main()
