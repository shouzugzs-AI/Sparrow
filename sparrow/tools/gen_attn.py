"""注意力块（V0.1.3）的确定性权重 + 两条验收模型的输入。

验收思路（都能判定，不靠"看起来对"）：

*A. T=1 的**恒等式**：*
    序列=1 时 softmax 只有一个元素参与 ⇒ 恒等于 1 ⇒ 注意力输出
    = Wo·(Wv·x + bWv) + bWo
    这正好是"两个线性层串联"。拆头/转置/拼接都是**纯数据搬运**，
    所以两条路必须输出**逐字符相同** —— 一个切片偏移写错、头顺序反了、
    转置接错，都会立刻打破它。
    所以 `ATTN_Wv` 的字节与参考模型的 `rv_W` **完全相同**（Wo/bWo 同理），
    两个模型的输入也完全相同。

*B. T=4 的**因果掩码**：*
    `因果=是` 时第 0 行只能看第 0 个 token ⇒ 输出第 0 行必须与 T=1 的结果**逐位相同**；
    `因果=否` 时第 0 行看全部 4 个 token ⇒ 必须**明显不同**（这是 can-fail：
    证明"逐位相同"这条判据真的能被打破）。
"""

import os
import numpy as np

import os as _os_pathfix
_HERE = _os_pathfix.path.dirname(_os_pathfix.path.abspath(__file__))
ROOT = _os_pathfix.path.dirname(_HERE)          # 本文件在 <root>/tools/ 下
OUT = os.path.join(ROOT, "examples", "attn_bins")
D, H, T = 4, 2, 4
os.makedirs(OUT, exist_ok=True)
f32 = np.float32
rng = np.random.default_rng(23)


def save(name, arr):
    a = np.asarray(arr, dtype=f32)
    a.astype("<f4").tofile(os.path.join(OUT, name))
    return a


x1 = rng.standard_normal(D).astype(f32)                      # [1,D] 单 token 输入
x4 = rng.standard_normal((T, D)).astype(f32)                 # [4,D] 四 token 输入
x4[0] = x1                                                   # 第 0 行与单 token 相同
l1_W = rng.standard_normal((D, D)).astype(f32)
l1_b = rng.standard_normal(D).astype(f32)
Wq = rng.standard_normal((D, D)).astype(f32)
Wk = rng.standard_normal((D, D)).astype(f32)
Wv = rng.standard_normal((D, D)).astype(f32)
Wo = rng.standard_normal((D, D)).astype(f32)
bq = rng.standard_normal(D).astype(f32)
bk = rng.standard_normal(D).astype(f32)
bv = rng.standard_normal(D).astype(f32)
bo = rng.standard_normal(D).astype(f32)

save("attnid_in.bin", x1)
save("attnref_in.bin", x1)                      # 与 attnid 的输入**逐位相同**
save("attncausal_in.bin", x4)
save("attnnocausal_in.bin", x4)
save("attnloop_in.bin", x1)
# 循环化验收：`重复2次` 的注意力组，每份实例一套常量（含每份自己的缩放常量
# —— 前端会把 `a__1_scale.bin` 按实例名复制成 `a__2_scale.bin`）
for r in (1, 2):
    for tag, M in (("Wq", Wq), ("Wk", Wk), ("Wv", Wv), ("Wo", Wo)):
        save(f"a__{r}_{tag}.bin", M)
    for tag, bb in (("bWq", bq), ("bWk", bk), ("bWv", bv), ("bWo", bo)):
        save(f"a__{r}_{tag}.bin", bb)
save("LINEAR_W.bin", l1_W)                      # attnid 链首那层线性
save("LINEAR_b.bin", l1_b)
for tag, W in (("Wq", Wq), ("Wk", Wk), ("Wv", Wv), ("Wo", Wo)):
    save(f"ATTN_{tag}.bin", W)
for tag, b in (("bWq", bq), ("bWk", bk), ("bWv", bv), ("bWo", bo)):
    save(f"ATTN_{tag}.bin", b)
# 参考模型的**三层**：链首那层线性（与 attnid 的链首同一份字节）+ 值投影 + 输出投影。
# 漏掉链首那层是这份验收第一次跑出来的错：注意力读的是**链首的输出**，
# 而参考模型原本直接吃模型输入 —— 两条路根本不是同一个函数。
save("r1_W.bin", l1_W)
save("r1_b.bin", l1_b)
save("rv_W.bin", Wv)
save("rv_b.bin", bv)
save("ro_W.bin", Wo)
save("ro_b.bin", bo)

# 自检：参考模型的各层权重必须与注意力里对应那份**逐字节相同**
for ref, att in (("r1_W.bin", "LINEAR_W.bin"), ("r1_b.bin", "LINEAR_b.bin"),
                 ("rv_W.bin", "ATTN_Wv.bin"), ("ro_W.bin", "ATTN_Wo.bin"),
                 ("rv_b.bin", "ATTN_bWv.bin"), ("ro_b.bin", "ATTN_bWo.bin")):
    a = open(os.path.join(OUT, ref), "rb").read()
    b = open(os.path.join(OUT, att), "rb").read()
    assert a == b, (ref, att)
# 自检：Q/K 权重必须与 V 不同（否则"头顺序/切片错位"这类错误可能看不出来）
assert not np.array_equal(Wq, Wv) and not np.array_equal(Wk, Wv)
print(f"产出 {len(os.listdir(OUT))} 个权重文件到 {OUT}")
print("自检：参考两层与注意力的 V/Wo 权重逐字节相同；Q/K/V 互不相同")
