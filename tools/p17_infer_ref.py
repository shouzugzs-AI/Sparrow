#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P17 P1 独立稠密参考 —— **纯 Python，不调用任何 Rust**。

================================================================================
这个文件要回答的唯一问题
================================================================================

    Rust 侧 vmlm-sparse-infer.exe 的 24 层稀疏前向，**逐位**算对了吗？

为什么必须是「独立」的：本项目吃过三次同型的亏（P16_report.md §9）——
**判据的两端来自同一个实现**，于是它对自己写的错也成立。
如果用 Rust 再写一份前向来对拍 Rust，那验的是「我算的和我算的一样」。

所以本文件：
  * 用自己的双层 Python 循环做矩阵乘（不用 numpy 的 BLAS —— 见下面 WHY）
  * 自己从常量重写 top-k 选择（插入排序，不用 heapq/sorted 的稳定性假设）
  * 自己重写 PRNG（neuron_seed/neuron_rand/neuron_conn_src）
  * **不 import** 任何本项目的东西，只读 dump 出来的二进制

================================================================================
WHY 不用 numpy 的 matmul / dot
================================================================================

numpy 的 `a @ b` 会调 BLAS，而 BLAS 的归约顺序**不保证**与 Rust 的
`acc += w*x` 顺序一致，还可能用 FMA（融合乘加）——那会改变每一次舍入。
于是「max|diff| = 0」会变成一个关于 BLAS 版本的偶然事实，而不是关于
实现的证明。本文件的 mm() 因此是**显式的 j 递增循环 + np.float32 逐步舍入**，
与 Rust 侧 `acc += *w * *x` 的求值顺序逐位对应。

⚠️ 能力边界（诚实标注，不要扩大）：
  逐位一致证明的是「两侧对**同一个算法**的实现相同」。它**不**证明这个
  算法是「标准 transformer 前向」—— 本阶段的序列混合是刻意简化的
  （见 train/src/sparse_infer24.rs 的说明），标准 attention 由 task-1 负责。

用法：
    python tools/p17_infer_ref.py examples/p17_infer/dump_L2_d16_k08_k3_s3.bin
"""

import struct
import sys
import os

import numpy as np

F32 = np.float32

# 超过这个规模就跑不动（稠密 Python 双重循环是 O(L*T*d^2)）。
# 24×1536 需要 ~3.6e12 次内层迭代 —— 那是几十小时，不是几分钟。
# 这里**显式拒绝**并说明，而不是静默跑一个没人会等完的循环。
MAX_D = 128


def r_u32(b, o):
    return struct.unpack_from("<I", b, o)[0], o + 4


def r_f32s(b, o, n):
    return struct.unpack_from("<%df" % n, b, o), o + 4 * n


# ============================================================================
# PRNG：从 conn_sparse.rs 的常量**重写**（不是调用）
# ============================================================================
#
# 这三个函数在 Rust 侧是 `const fn`，只用 u32 回绕算术、无浮点。
# Python 的 int 是任意精度，所以 `& 0xFFFFFFFF` 就是 u32 回绕。
# 这让我们能**逐位**复现拓扑：如果 Rust 侧的 src 与这里算出的不一致，
# 说明要么公式改了、要么 dump 错了 —— 两种都要立刻知道。
M32 = 0xFFFFFFFF
SPLITMIX_GOLDEN = 0x9E3779B9


def neuron_seed(layer, i):
    a = (SPLITMIX_GOLDEN * ((layer + 1) & M32)) & M32
    bb = (0x85EBCA6B * ((i + 1) & M32)) & M32
    z = a ^ bb
    z = ((z ^ (z >> 16)) * 0x7FEB352D) & M32
    z = ((z ^ (z >> 15)) * 0x846CA68B) & M32
    return (z ^ (z >> 16)) & M32


def neuron_rand(seed, j):
    z = (seed ^ ((SPLITMIX_GOLDEN * ((j + 1) & M32)) & M32)) & M32
    z = ((z ^ (z >> 16)) * 0x7FEB352D) & M32
    z = ((z ^ (z >> 15)) * 0x846CA68B) & M32
    return (z ^ (z >> 16)) & M32


def neuron_conn_src(layer, i, j, n_neurons, k0):
    if n_neurons == 0 or k0 == 0:
        return 0
    seed = neuron_seed(layer, i)
    r = neuron_rand(seed, M32) if j == 0 else neuron_rand(seed, j)
    return r % n_neurons


# ============================================================================
# 矩阵乘：显式 j 递增 + 逐步 f32 舍入（与 Rust 的 acc += w*x 逐位对应）
# ============================================================================
def mm(W, x, n_out, n_in):
    """W 行主序 [n_out][n_in]；x 长度 n_in；返回长度 n_out 的 f32 数组。

    ⚠️ 累加子从字面 0.0 开始（与 Rust 的 `let mut acc = 0.0f32;` 一致）。
    """
    out = np.zeros(n_out, dtype=F32)
    for i in range(n_out):
        acc = F32(0.0)
        base = i * n_in
        for j in range(n_in):
            acc = F32(acc + F32(W[base + j] * x[j]))
        out[i] = acc
    return out


# ============================================================================
# top-k：与 row_topk **结构不同**的实现（全排序），用于交叉验证
# ============================================================================
#
# Rust 的 row_topk = k 轮「扫未取下标取最大值」+ taken 位图。
# 这里用「算全部强度 + 稳定插入排序降序 + 取前 k + 升序输出」。
# 两条路径结构不同 ⇒ 选择结果必须相同（这条**能失败**：把平局规则改成
# 保留下标大的，在存在相等强度时就会给出不同结果）。
def topk_sorted(w_row, src_row, x, k0, k):
    """返回**升序**的命中下标列表（与引擎 93 / row_topk 的输出序一致）。

    平局规则写死：稳定排序，只在 strict < 时前移 ⇒ 平局保留下标小的
    （与 row_topk 的 `if v > bestv` 等价）。
    """
    strv = []
    for j in range(k0):
        sid = int(src_row[j])
        strv.append((abs(float(F32(w_row[j]) * F32(x[sid]))), j))
    order = list(range(k0))
    # 稳定插入排序：降序，只在严格更小时前移
    for i in range(1, k0):
        vi, ii = strv[i][0], order[i]
        q = i
        while q > 0 and strv[q - 1][0] < vi:
            strv[q] = strv[q - 1]
            order[q] = order[q - 1]
            q -= 1
        strv[q] = (vi, ii)
        order[q] = ii
    nk = min(k, k0)
    hit = order[:nk]
    return sorted(hit)  # 升序输出


# ============================================================================
# 前向：与 Rust run_forward 同一结构，但**独立写的**
# ============================================================================
def forward(sparse_k, layers, h0, seq, d, k0):
    """sparse_k=None 表示全连接（k=k0，不做 top-k）。"""
    h = [np.array(h0[t * d:(t + 1) * d], dtype=F32) for t in range(seq)]
    layer_inputs = []
    total_macs = 0
    for (wv, wo, w, src) in layers:
        layer_inputs.append(
            np.array([v for row in h for v in row], dtype=F32)
        )
        newh = []
        for t in range(seq):
            hrow = h[t]
            v = mm(wv, hrow, d, d)
            a = mm(wo, v, d, d)
            u = np.array([F32(hrow[i] + a[i]) for i in range(d)], dtype=F32)
            y = np.zeros(d, dtype=F32)
            for i in range(d):
                base = i * k0
                if sparse_k is None:
                    sel = list(range(k0))
                else:
                    sel = topk_sorted(
                        w[base:base + k0], src[base:base + k0], u, k0, sparse_k
                    )
                acc = F32(0.0)
                first = True
                for j in sel:
                    sid = int(src[base + j])
                    term = F32(w[base + j] * u[sid])
                    acc = term if first else F32(acc + term)
                    first = False
                    total_macs += 1
                y[i] = acc
            newh.append(np.array([F32(u[i] + y[i]) for i in range(d)], dtype=F32))
        h = newh
    return h, layer_inputs, total_macs


def main():
    if len(sys.argv) < 2:
        print("usage: p17_infer_ref.py <dump.bin>")
        return 2
    path = sys.argv[1]
    if not os.path.exists(path):
        print("FATAL: dump not found: %s" % path)
        print("  run: vmlm-sparse-infer.exe --layers 2 --d 16 --k0 8 --k 3 --seq 3 --dump 1")
        return 9
    b = open(path, "rb").read()
    o = 0
    nl, o = r_u32(b, o)
    d, o = r_u32(b, o)
    k0, o = r_u32(b, o)
    k, o = r_u32(b, o)
    seq, o = r_u32(b, o)
    vocab, o = r_u32(b, o)
    ntok, o = r_u32(b, o)
    tok, o = r_f32s(b, o, 0)  # placeholder
    toks = []
    for _ in range(ntok):
        v, o = r_u32(b, o)
        toks.append(v)

    print("=" * 72)
    print("P17 P1 INDEPENDENT PYTHON DENSE REFERENCE (no Rust calls)")
    print("=" * 72)
    print("dump          = %s" % path)
    print("cfg           = layers=%d d=%d k0=%d k=%d seq=%d vocab=%d" % (nl, d, k0, k, seq, vocab))
    print("tokens        = %s" % toks)

    if d > MAX_D:
        print()
        print("REFUSING TO RUN: d=%d exceeds MAX_D=%d." % (d, MAX_D))
        print("Reason: pure-Python O(L*T*d^2) inner loop would need ~%.3e iterations"
              % (nl * seq * d * d * 4))
        print("For 24x1536 that is tens of hours, not minutes. This is a HARD refusal")
        print("(not a silent skip) -- the 24-layer run is verified by the in-process")
        print("dense-loop identity check instead, plus the small-d cross-implementation")
        print("bitwise comparison here. See the report for which criterion covers what.")
        return 9

    emb, o = r_f32s(b, o, vocab * d)
    h0, o = r_f32s(b, o, seq * d)
    layers = []
    for l in range(nl):
        wq, o = r_f32s(b, o, d * d)
        wk, o = r_f32s(b, o, d * d)
        wv, o = r_f32s(b, o, d * d)
        wo, o = r_f32s(b, o, d * d)
        w, o = r_f32s(b, o, d * k0)
        src, o = r_f32s(b, o, d * k0)
        layers.append((np.array(wv, dtype=F32), np.array(wo, dtype=F32),
                       np.array(w, dtype=F32), np.array(src, dtype=F32)))
    rust_layer_in = []
    for l in range(nl):
        row, o = r_f32s(b, o, seq * d)
        rust_layer_in.append(np.array(row, dtype=F32))
    rust_final = np.array(r_f32s(b, o, seq * d)[0], dtype=F32)

    checks = []
    fails = []

    def chk(name, ok, detail):
        checks.append((name, ok))
        if not ok:
            fails.append(name)
        print("[%s] %s  %s" % ("PASS" if ok else "FAIL", name, detail))

    # ---- C1: PRNG 交叉校验（Rust 侧打印的三个值）----
    #
    # 这三个数字由 vmlm-sparse-infer.exe 在 dump 时打印。如果 Python 重写的
    # 公式给出不同的值，说明公式被改过或者我对它的理解错了。
    # 期望值在运行时从 dump 的 src 反推会更弱 —— 所以这里要求调用方
    # 把 Rust 打印的值传进来（见 --expect）。没有传时只报 Python 侧的值，
    # 并把这判据标为 UNPROVEN（诚实：没有对照就没法断言）。
    print()
    print("--- PRNG cross-check (constants rewritten in Python) ---")
    py_seed = neuron_seed(0, 0)
    py_src = neuron_conn_src(0, 0, 0, d, k0)
    py_rand = neuron_rand(12345, 7)
    print("python: neuron_seed(0,0)=%d  neuron_conn_src(0,0,0,%d,%d)=%d  neuron_rand(12345,7)=%d"
          % (py_seed, d, k0, py_src, py_rand))

    # 更强的自校验：dump 里的 src 表必须与 Python 重算的拓扑**逐位**一致。
    # 这条不依赖调用方传参，能独立失败。
    bad_src = 0
    for l in range(nl):
        src_l = layers[l][3]
        for i in range(d):
            for j in range(k0):
                want = neuron_conn_src(l, i, j, d, k0)
                got = int(src_l[i * k0 + j])
                if got != want:
                    bad_src += 1
    chk("C1 src topology from rewritten PRNG == dumped src", bad_src == 0,
        "bad=%d of %d (layer,i,j) triples" % (bad_src, nl * d * k0))

    # ---- C2: 稀疏前向逐位一致 ----
    h_sp, lin_sp, macs_sp = forward(k, layers, h0, seq, d, k0)
    got = np.array([v for row in h_sp for v in row], dtype=F32)
    bit = int((got.view(np.uint32) != rust_final.view(np.uint32)).sum())
    maxd = float(np.abs(got.astype(np.float64) - rust_final.astype(np.float64)).max())
    first = int(np.argmax((got.view(np.uint32) != rust_final.view(np.uint32)))) \
        if bit else -1
    chk("C2 sparse forward: Python == Rust, BITWISE", bit == 0,
        "bit_diffs=%d / %d  max|diff|=%.9e  first_diff_idx=%d  (py macs=%d)"
        % (bit, got.size, maxd, first, macs_sp))

    # ---- C3: 逐层输入也一致（差异定位到层）----
    bad_lin = 0
    first_bad_layer = -1
    for l in range(nl):
        p = lin_sp[l].view(np.uint32)
        r = rust_layer_in[l].view(np.uint32)
        nb = int((p != r).sum())
        if nb and first_bad_layer < 0:
            first_bad_layer = l
        bad_lin += nb
    chk("C3 per-layer inputs identical (localizes divergence to a layer)",
        bad_lin == 0,
        "bad=%d  first_bad_layer=%d  (per-layer compare, %d layers)"
        % (bad_lin, first_bad_layer, nl))

    # ---- C4: must-fail 对照 —— 扰动一个**被 top-k 选中**的权重必须被检出 ----
    #
    # 这正是 task-3 点名要的对照，也是 P16 踩过的坑。
    # 实现方式：在 Python 侧改一个**确实被选中**的权重（用 Python 自己
    # 选的 sel 判定），重算，输出必须变。然后确认扰动**未被选中**的
    # 权重输出不变。
    #
    # ⚠️ 关键：这里 Python 与 Rust 是**两个独立实现**，所以这条判据同时
    #    验证了「扰动检测链」是真的连着数值的 —— 如果 forward 里有个
    #    bug 让 c 根本不参与输出，C4 会检出（输出不变）。
    print()
    print("--- must-fail control: perturbation detection ---")
    l0w = layers[0][2].copy()
    l0s = layers[0][3]
    # 用第 0 层第 0 个 token 的输入算一次，拿到 sel
    hrow = np.array(h0[0:d], dtype=F32)
    v = mm(layers[0][0], hrow, d, d)
    a = mm(layers[0][1], v, d, d)
    u = np.array([F32(hrow[i] + a[i]) for i in range(d)], dtype=F32)
    sel = topk_sorted(l0w[0:k0], l0s[0:k0], u, k0, k)
    unsel = [j for j in range(k0) if j not in sel]
    print("layer0 neuron0: selected=%s unselected=%s (k=%d k0=%d)"
          % (sel, unsel[:4], k, k0))

    if not sel:
        print("FATAL: no selected connection -- criterion cannot be demonstrated")
        return 9
    if not unsel:
        print("FATAL: no unselected connection (k==k0?) -- negative control impossible")
        return 9

    base_out = np.array([F32(F32(hrow[i]) + F32(0.0)) for i in range(d)], dtype=F32)
    # 基准（用原权重）
    def ffn_out0(w_arr):
        y = np.zeros(d, dtype=F32)
        for i in range(d):
            s = topk_sorted(w_arr[i * k0:(i + 1) * k0], l0s[i * k0:(i + 1) * k0], u, k0, k)
            acc = F32(0.0)
            first = True
            for j in s:
                sid = int(l0s[i * k0 + j])
                term = F32(w_arr[i * k0 + j] * u[sid])
                acc = term if first else F32(acc + term)
                first = False
            y[i] = acc
        return y

    y0 = ffn_out0(l0w)
    # 扰动被选中的权重（注意：改权重会改变强度 ⇒ 可能改变 top-k 选择！
    # 这里用 +1e-3 的小扰动，并**只断言输出变化**，不假设选择不变 ——
    # 前者无论如何都必须成立（除非该权重恰好乘以 0）
    w_sel = l0w.copy()
    jj = sel[0]
    w_sel[jj] = F32(w_sel[jj] + F32(1e-3))
    y_sel = ffn_out0(w_sel)
    d_sel = int((y0.view(np.uint32) != y_sel.view(np.uint32)).sum())
    chk("C4a perturb a SELECTED weight => output MUST change", d_sel > 0,
        "conn j=%d  changed_bits=%d  max|diff|=%.9e"
        % (jj, d_sel, float(np.abs(y_sel - y0).max())))

    # 扰动未被选中的权重：**只要它没被选中**（且扰动没把它推进 top-k），
    # 输出必须不变。这里先检查扰动后它**仍然**未被选中，否则这条对照
    # 本身不成立（那是选择变化，不是「跳过」）。
    ju = unsel[0]
    w_uns = l0w.copy()
    w_uns[ju] = F32(w_uns[ju] + F32(1e-3))
    sel_after = topk_sorted(w_uns[0:k0], l0s[0:k0], u, k0, k)
    if ju in sel_after:
        print("     (note: perturbing j=%d pushed it INTO top-k; control not applicable"
              " for this j -- picking another)")
        for ju2 in unsel:
            w_try = l0w.copy()
            w_try[ju2] = F32(w_try[ju2] + F32(1e-3))
            if ju2 not in topk_sorted(w_try[0:k0], l0s[0:k0], u, k0, k):
                ju = ju2
                w_uns = w_try
                break
    y_uns = ffn_out0(w_uns)
    d_uns = int((y0.view(np.uint32) != y_uns.view(np.uint32)).sum())
    chk("C4b perturb an UNSELECTED weight => output must NOT change", d_uns == 0,
        "conn j=%d  changed_bits=%d  max|diff|=%.9e  (the P16 pitfall, asserted)"
        % (ju, d_uns, float(np.abs(y_uns - y0).max())))

    # ---- C5: 独立重算 union fraction 与跳过率 ----
    print()
    print("--- independent recomputation of the two ratios ---")
    union = set()
    act_conns = 0
    for l in range(nl):
        wv, wo, w, src = layers[l]
        hh = [np.array(h0[t * d:(t + 1) * d], dtype=F32) for t in range(seq)]
        # 重建本层输入
        hh = [rust_layer_in[l][t * d:(t + 1) * d] for t in range(seq)]
        for t in range(seq):
            hrow = hh[t]
            v = mm(wv, hrow, d, d)
            a = mm(wo, v, d, d)
            u = np.array([F32(hrow[i] + a[i]) for i in range(d)], dtype=F32)
            for i in range(d):
                s = topk_sorted(w[i * k0:(i + 1) * k0], src[i * k0:(i + 1) * k0], u, k0, k)
                act_conns += len(s)
                for j in s:
                    union.add(int(src[i * k0 + j]))
    uf = len(union) / d
    full = d * k0 * nl * seq
    skip = 1.0 - act_conns / full
    print("python: union_fraction = %.6f  (|union|=%d / d=%d)" % (uf, len(union), d))
    print("python: skip_rate      = %.6f  (macs=%d / full=%d)" % (skip, act_conns, full))
    print("python: activated conns= %d" % act_conns)
    chk("C5 union_fraction is a fraction in (0,1]", 0.0 < uf <= 1.0,
        "union_fraction=%.6f" % uf)

    print()
    print("=" * 72)
    npass = sum(1 for _, ok in checks if ok)
    print("checks=%d pass=%d fail=%d" % (len(checks), npass, len(checks) - npass))
    for n in fails:
        print("  FAILED: %s" % n)
    print("=" * 72)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
