"""residual 系列（V0.1.3 汇入）的确定性权重 + numpy 参考输出。

三个模型共用 examples/res_bins 一套权重（层名各不相同，互不撞车）：

    resadd  汇入=相加（等宽 4）        l_a  [4,4]
    rescat  汇入=拼接（4+4=8）         l_c  [4,8]
    res3    三个前驱相加（左折叠两次） l_b  [4,4] + l_d [4,4]

参考输出用 numpy **独立算一遍**（不调用引擎、不读 .vml）：
  嵌入 e = E[token]
  稀疏 s = CSR 矩阵乘 e
  resadd: y = (e + s) @ la_W^T + la_b
  rescat: y = [e ; s] @ lc_W^T + lc_b     （列拼接）
  res3  : y = ((s + lb) + e) @ ld_W^T + ld_b
每个模型的输出写在 out/residual_<模型>.ref.txt（6 位小数，与引擎打印格式一致）。
"""

import functools
import os
import numpy as np

import os as _os_pathfix
_HERE = _os_pathfix.path.dirname(_os_pathfix.path.abspath(__file__))
ROOT = _os_pathfix.path.dirname(_HERE)          # 本文件在 <root>/tools/ 下
OUT = os.path.join(ROOT, "examples", "res_bins")
REFOUT = os.path.join(ROOT, "out", "residual")
VOCAB, DIM, K0, OUT_DIM = 8, 4, 2, 4
TOKEN = 3.0

os.makedirs(OUT, exist_ok=True)
os.makedirs(REFOUT, exist_ok=True)
f32 = np.float32


def save(name, arr):
    a = np.asarray(arr, dtype=f32)
    a.astype("<f4").tofile(os.path.join(OUT, name))
    return a


rng = np.random.default_rng(11)
E = rng.standard_normal((VOCAB, DIM)).astype(f32)
off = np.arange(0, DIM * K0 + 1, K0, dtype=f32)
src = np.array([(j * K0 + t) % DIM for j in range(DIM) for t in range(K0)], dtype=f32)
w = rng.standard_normal(DIM * K0).astype(f32)
la_W = rng.standard_normal((OUT_DIM, DIM)).astype(f32)
la_b = rng.standard_normal(OUT_DIM).astype(f32)
lc_W = rng.standard_normal((OUT_DIM, 2 * DIM)).astype(f32)
lc_b = rng.standard_normal(OUT_DIM).astype(f32)
lb_W = rng.standard_normal((OUT_DIM, DIM)).astype(f32)
lb_b = rng.standard_normal(OUT_DIM).astype(f32)
ld_W = rng.standard_normal((OUT_DIM, DIM)).astype(f32)
ld_b = rng.standard_normal(OUT_DIM).astype(f32)

for m in ("resadd", "rescat", "res3", "resadd2", "rescat2", "resdup", "resloop", "resloop2"):
    save(f"{m}_in.bin", np.array([TOKEN]))
save("EMBED_E.bin", E)
save("t_off.bin", off)
save("t_src.bin", src)
save("t_W.bin", w)
for tag, W, b in (("la", la_W, la_b), ("lc", lc_W, lc_b), ("lb", lb_W, lb_b), ("ld", ld_W, ld_b)):
    save(f"{tag}_W.bin", W)
    save(f"{tag}_b.bin", b)

# resloop / resloop2 的重量：重复组的 3 份（**必须互不相同**，否则
# "循环版 == 展开版"这条判据对步长/顺序类错误不敏感）+ 组外的两层
tW, tb = [], []
for r in range(12):
    r3 = np.random.default_rng(2000 + r)
    tW.append(r3.standard_normal((OUT_DIM, DIM)).astype(f32))
    tb.append(r3.standard_normal(OUT_DIM).astype(f32))
    save(f"t__{r + 1}_W.bin", tW[-1])
    save(f"t__{r + 1}_b.bin", tb[-1])
mg_W = np.random.default_rng(2100).standard_normal((OUT_DIM, DIM)).astype(f32)
mg_b = np.random.default_rng(2101).standard_normal(OUT_DIM).astype(f32)
aa_W = np.random.default_rng(2200).standard_normal((OUT_DIM, DIM)).astype(f32)
aa_b = np.random.default_rng(2201).standard_normal(OUT_DIM).astype(f32)
save("mg_W.bin", mg_W)
save("mg_b.bin", mg_b)
save("aa_W.bin", aa_W)
save("aa_b.bin", aa_b)
assert len({w.tobytes() for w in tW}) == 12, "重复组的 12 份权重必须互不相同"

# ---- 参考输出（纯 numpy） ----
e = E[int(TOKEN)].astype(f32)                        # [4]
Wd = np.zeros((DIM, DIM), dtype=f32)                 # CSR → 稠密
for j in range(DIM):
    for t in range(K0):
        Wd[j, int(src[j * K0 + t])] = w[j * K0 + t]
s = (Wd @ e).astype(f32)                             # [4]


def gemm(x, W, b):
    return (W @ x).astype(f32) + b                   # NT：W 是 [out, in]


refs = {
    # 相加：**顺序必须与发射的折叠顺序一致** —— f32 加法可交换但**不可结合**，
    # 前端按 `连接` 语句出现的顺序收前驱（嵌入, lb, t），左折叠 ⇒ (e + lb) + t
    "resadd": gemm((e + s).astype(f32), la_W, la_b),
    # 拼接：顺序**不是**可交换的。发射的是 `CONCAT lc_in, t_out, EMBED_out`
    # ⇒ [s ; e]（稀疏在前），不是 [e ; s]
    "rescat": gemm(np.concatenate([s, e]).astype(f32), lc_W, lc_b),
    "res3": gemm(((e + gemm(s, lb_W, lb_b)).astype(f32) + s).astype(f32), ld_W, ld_b),
    # 顺序对照：相加可交换 ⇒ 与 resadd 同一个参考值
    "resadd2": gemm((e + s).astype(f32), la_W, la_b),
    # 顺序对照：拼接不可交换 ⇒ 前驱反过来是 [e ; s]，与 rescat 的 [s ; e] 不同
    "rescat2": gemm(np.concatenate([e, s]).astype(f32), lc_W, lc_b),
    # 逐位验收：relu(e)+relu(e) == 2*relu(e)（f32 里加自己无舍入）⇒ 精确可算
    "resdup": (2 * np.maximum(e, f32(0))).astype(f32),
    # 残差块 + 重复：x0=e, x_k = t_k(x_{k-1}), m = x3 + e, y = mg(m)
    "resloop": gemm(
        (
            functools.reduce(lambda acc, k: gemm(acc, tW[k], tb[k]), range(12), e) + e
        ).astype(f32),
        mg_W,
        mg_b,
    ),
    # 重复组**内部**的汇入：x0 = aa(e)，m_k = x_{k-1} + e，x_k = t_k(m_k)
    "resloop2": functools.reduce(
        lambda acc, k: gemm((acc + e).astype(f32), tW[k], tb[k]),
        range(12),
        gemm(e, aa_W, aa_b),
    ),
}


# 格式化后的参考文本（与引擎**逐字符**比较用）
#
# 关键：引擎的 print_f32（core/src/interp.rs:2416）是**截断**，不是四舍五入 ——
# 它用 `frac *= 10; dig = frac as u32` 逐位取数，从不进位。所以 numpy 参考
# 必须复刻这个格式，否则每个数都会在末位差 1（这正是 V0.1.1 里
# "numpy 0.975112 / 引擎 0.975111" 那个一直没解释的差）。
def eng_fmt(v):
    """逐位复刻 vmlm 的 print_f32（f32 运算 + 截断，6 位小数）。"""
    x = f32(v)
    neg = bool(x < f32(0))
    av = f32(-x) if neg else x
    ip = int(av)                       # `av as u64`：向零截断
    frac = f32(av - f32(ip))
    s = ("-" if neg else "") + str(ip) + "."
    for _ in range(6):
        frac = f32(frac * f32(10.0))
        dig = int(frac)                # `frac as u32`
        s += str(dig % 10)
        frac = f32(frac - f32(dig))
    return s


def fmt(v):
    return " ".join(eng_fmt(x) for x in v)


outs = {k: fmt(v) for k, v in refs.items()}


# can-fail 自检：**故意写错**的汇入必须与正确参考值差得足够远，
# 否则"和参考值一致"这条判据就是恒真的（比如合并写成减法、拼接顺序反了）。
wrong = {
    "resadd": gemm((e - s).astype(f32), la_W, la_b),
    "rescat": gemm(np.concatenate([e, s]).astype(f32), lc_W, lc_b),
    "res3": gemm(((e + gemm(s, lb_W, lb_b)).astype(f32) * s).astype(f32), ld_W, ld_b),
}
for k, bad in wrong.items():
    gap = float(np.abs(bad - refs[k]).max())
    assert gap > 1e-3, f"{k} 的 can-fail 差距只有 {gap:.3g} —— 判据不够判别性"
    print(f"  {k:7s} can-fail 与正确值的最大差 {gap:.6g}（远超 1e-6 容差）")
assert np.abs(refs["rescat"] - refs["rescat2"]).max() > 1e-3, "拼接顺序不敏感？"


for k, v in refs.items():
    p = os.path.join(REFOUT, f"{k}.ref.txt")
    with open(p, "w", encoding="ascii") as f:
        f.write(fmt(v) + "\n")
    print(f"  {k:7s} ref = {fmt(v)}")

# 判别性自检：三个"形状不同"的模型参考输出必须互不相同（否则判据可能恒真）；
# 顺序对照则**按设计**：相加可交换（resadd2 == resadd）、拼接不可交换（rescat2 != rescat）。
assert len({outs["resadd"], outs["rescat"], outs["res3"]}) == 3, "三个模型参考输出出现重复"
assert outs["resadd2"] == outs["resadd"], "相加必须可交换（否则说明前驱顺序被写进了结果）"
assert outs["rescat2"] != outs["rescat"], "拼接必须不可交换（否则顺序没进结果）"
print("判别性自检：3 个形状互不相同；相加顺序无关、拼接顺序有关")
print(f"产出 {len(os.listdir(OUT))} 个权重文件到 {OUT}")
