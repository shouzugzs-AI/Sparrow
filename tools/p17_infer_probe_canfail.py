#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P17 P1 can-fail 审计：**证明**上面那套判据真的会失败。

================================================================================
为什么需要这个文件
================================================================================

纪律 #2：「每个判据必须能演示自己失败（can-fail 对照）。不能失败的检查标
UNPROVEN，绝不写成 PASS。」

tools/p17_infer_ref.py 里的 6 条判据都报了 PASS。但「报 PASS」本身不构成
证据 —— 一个恒真的检查也会报 PASS。本文件对每一条判据施加一个**已知错误**，
并断言它**确实翻成 FAIL**。凡是翻不过来的，就必须标 UNPROVEN。

方法：不重写参考实现，而是**参数化地注入错误**（monkey-patch / 改数据），
这样被检验的仍是同一条判据代码路径。

用法：python tools/p17_infer_probe_canfail.py <dump.bin>
"""

import sys
import os
import importlib.util

import numpy as np

F32 = np.float32

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "p17ref", os.path.join(HERE, "p17_infer_ref.py")
)
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)

N_CHECK = 0
N_FAIL = 0
N_UNPROVEN = 0


def chk(name, ok, detail):
    global N_CHECK, N_FAIL
    N_CHECK += 1
    if not ok:
        N_FAIL += 1
    print("[%s] %s  %s" % ("PASS" if ok else "FAIL", name, detail))


def unproven(name, why):
    global N_UNPROVEN
    N_UNPROVEN += 1
    print("[UNPROVEN] %s  %s" % (name, why))


def load(path):
    b = open(path, "rb").read()
    o = 0
    v = []
    for _ in range(7):
        x, o = ref.r_u32(b, o)
        v.append(x)
    nl, d, k0, k, seq, vocab, ntok = v
    toks = []
    for _ in range(ntok):
        x, o = ref.r_u32(b, o)
        toks.append(x)
    emb, o = ref.r_f32s(b, o, vocab * d)
    h0, o = ref.r_f32s(b, o, seq * d)
    layers = []
    for l in range(nl):
        wq, o = ref.r_f32s(b, o, d * d)
        wk, o = ref.r_f32s(b, o, d * d)
        wv, o = ref.r_f32s(b, o, d * d)
        wo, o = ref.r_f32s(b, o, d * d)
        w, o = ref.r_f32s(b, o, d * k0)
        src, o = ref.r_f32s(b, o, d * k0)
        layers.append((np.array(wv, dtype=F32), np.array(wo, dtype=F32),
                       np.array(w, dtype=F32), np.array(src, dtype=F32)))
    # ⚠️ 必须跳过每层的 layer_in 块，否则 rust_final 会从错误的偏移读出来
    #    —— 第一版漏了这一步，于是「基准」读到的是 layer_in 的字节，
    #    把 C2 的基线误报成 bit_diffs=48。这个错误**恰好**被 CF-2a
    #    （基线必须为 0）抓住了 —— 这就是「判据要能失败」的价值：
    #    如果这里只写 can-fail 分支而没写基线分支，这个读偏移 bug 会
    #    一直藏着，并让所有 can-fail 结论都建立在错误的基线上。
    for _ in range(nl):
        _, o = ref.r_f32s(b, o, seq * d)
    rust_final = np.array(ref.r_f32s(b, o, seq * d)[0], dtype=F32)
    return nl, d, k0, k, seq, layers, h0, rust_final


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "examples/p17_infer/dump_L2_d16_k08_k3_s3.bin"
    nl, d, k0, k, seq, layers, h0, rust_final = load(path)

    print("=" * 72)
    print("P17 P1 CAN-FAIL AUDIT -- does each criterion actually fail when wrong?")
    print("=" * 72)
    print("dump = %s   cfg: L=%d d=%d k0=%d k=%d seq=%d" % (path, nl, d, k0, k, seq))

    # =====================================================================
    # CF-1: C1（拓扑判据）—— 把一个 src 项改错，bad 必须 > 0
    # =====================================================================
    print()
    print("--- CF-1: C1 (topology from PRNG) ---")
    bad = 0
    for l in range(nl):
        src_l = layers[l][3]
        for i in range(d):
            for j in range(k0):
                if int(src_l[i * k0 + j]) != ref.neuron_conn_src(l, i, j, d, k0):
                    bad += 1
    chk("CF-1a baseline (unmodified) => bad == 0", bad == 0, "bad=%d" % bad)
    # 注入：把 layer0 neuron0 j=0 的 src 改成一个不同的合法值
    layers2 = [(a, b, c, s.copy()) for (a, b, c, s) in layers]
    orig = int(layers2[0][3][0])
    layers2[0][3][0] = float((orig + 1) % d)
    bad2 = 0
    for l in range(nl):
        src_l = layers2[l][3]
        for i in range(d):
            for j in range(k0):
                if int(src_l[i * k0 + j]) != ref.neuron_conn_src(l, i, j, d, k0):
                    bad2 += 1
    chk("CF-1b CAN-FAIL: one tampered src => bad > 0", bad2 > 0,
        "bad=%d (tampered layer0[0][0]: %d -> %d)" % (bad2, orig, int(layers2[0][3][0])))

    # =====================================================================
    # CF-2: C2（逐位前向判据）—— 改一个权重/改一处累加，bit_diffs 必须 > 0
    # =====================================================================
    print()
    print("--- CF-2: C2 (bitwise forward equality) ---")
    h_sp, _, _ = ref.forward(k, layers, h0, seq, d, k0)
    got = np.array([v for row in h_sp for v in row], dtype=F32)
    base_bits = int((got.view(np.uint32) != rust_final.view(np.uint32)).sum())
    chk("CF-2a baseline => bit_diffs == 0", base_bits == 0, "bit_diffs=%d" % base_bits)

    # 注入 A：扰动一个**被选中**的权重（用 Python 自己算的 sel 判定）
    l0w = layers[0][2].copy()
    l0s = layers[0][3]
    hrow = np.array(h0[0:d], dtype=F32)
    v = ref.mm(layers[0][0], hrow, d, d)
    a = ref.mm(layers[0][1], v, d, d)
    u = np.array([F32(hrow[i] + a[i]) for i in range(d)], dtype=F32)
    sel = ref.topk_sorted(l0w[0:k0], l0s[0:k0], u, k0, k)
    l0w2 = l0w.copy()
    l0w2[sel[0]] = F32(l0w2[sel[0]] + F32(1e-3))
    layers_w = list(layers)
    layers_w[0] = (layers[0][0], layers[0][1], l0w2, layers[0][3])
    h2, _, _ = ref.forward(k, layers_w, h0, seq, d, k0)
    got2 = np.array([v for row in h2 for v in row], dtype=F32)
    bits2 = int((got2.view(np.uint32) != rust_final.view(np.uint32)).sum())
    chk("CF-2b CAN-FAIL: perturb a SELECTED weight => bit_diffs > 0", bits2 > 0,
        "bit_diffs=%d (perturbed j=%d)" % (bits2, sel[0]))

    # 注入 B：扰动一个**未被选中**的权重 ⇒ C2 **应当仍然 PASS**
    # 这不是 can-fail，而是一条**阴性对照**：它证明 C2 不是「任何改动都报错」
    # 的粗糙检查，而是真的在逐位比较**参与计算的量**。
    #
    # ⚠️ 这里第一版写错了，而且错得很有教育意义：我按**neuron 0** 算出的
    #    `unsel[0]` 去扰动 `layers_w[0]` 的**整个权重表**的对应下标
    #    `l0w3[ju]` —— 但 `l0w` 是**展平的 [d*k0]**，下标 `ju` 只落在
    #    neuron 0 的行里。而 `ref.topk_sorted(l0w3[0:k0], ...)` 检验的
    #    也只是 neuron 0。所以「未被选中」这个性质**只对 neuron 0 成立**，
    #    扰动它对**别的** 15 个神经元来说是另一回事……实际上下标 ju<k0
    #    确实只影响 neuron 0 —— 真正的问题是：'unsel' 是用 u（neuron 0 的
    #    输入）算的，而 forward 里每个 token / 每层的 u 都不同。
    #    ⇒ 正确的做法是：**验证扰动后该权重在所有 (layer, token) 上都未被选中**，
    #      否则这条阴性对照不成立（那是选择变化，不是「跳过」）。
    #    这里改成「找到第一个在**所有** layer-token 组合下都未被选中的连接」，
    #    这才是「扰动一个未被选中的权重」的准确含义。
    def is_unselected_everywhere(w_arr, li, l, i, j):
        """返回 True 表示 (l,i,j) 这条连接在**所有 token** 上都没被 top-k 选中。"""
        for t in range(seq):
            # 重建该 token 的 u
            hrow = li[t * d:(t + 1) * d]
            v = ref.mm(layers[l][0], hrow, d, d)
            a = ref.mm(layers[l][1], v, d, d)
            uu = np.array([F32(hrow[k_] + a[k_]) for k_ in range(d)], dtype=F32)
            base = i * k0
            w_row = w_arr if w_arr is not None else layers[l][2][base:base + k0]
            s = ref.topk_sorted(w_row, l0s_all[l][base:base + k0], uu, k0, k)
            if j in s:
                return False
        return True

    # 先重建每层的输入（用未扰动的 layers 跑一次 forward）
    _, lin_all, _ = ref.forward(k, layers, h0, seq, d, k0)
    l0s_all = [x[3] for x in layers]
    target = None
    for i in range(d):
        for j in range(k0):
            if is_unselected_everywhere(
                layers[0][2][i * k0:(i + 1) * k0], lin_all[0], 0, i, j
            ):
                target = (i, j)
                break
        if target:
            break
    if target is None:
        unproven("CF-2c negative control",
                 "no connection is unselected on ALL tokens in layer 0")
    else:
        i_u, j_u = target
        l0w3 = layers[0][2].copy()
        l0w3[i_u * k0 + j_u] = F32(l0w3[i_u * k0 + j_u] + F32(1e-3))
        layers_u = list(layers)
        layers_u[0] = (layers[0][0], layers[0][1], l0w3, layers[0][3])
        h3, _, _ = ref.forward(k, layers_u, h0, seq, d, k0)
        got3 = np.array([v for row in h3 for v in row], dtype=F32)
        bits3 = int((got3.view(np.uint32) != rust_final.view(np.uint32)).sum())
        chk("CF-2c NEGATIVE CTL: perturb an UNSELECTED weight (all tokens) => bit_diffs stays 0",
            bits3 == 0,
            "bit_diffs=%d (layer0 neuron%d conn%d unselected on all %d tokens)"
            % (bits3, i_u, j_u, seq))

    # 注入 C：模拟「累加顺序改动」—— 把 first-flag 去掉（从 +0.0 起累加）
    # 改法：monkey-patch ref.mm 成「先加 0.0」的版本，看 C2 会不会检出。
    # ⚠️ 这个注入针对 mm（注意力），不是 FFN。理由：FFN 的 first-flag 在
    #    本数据集上实测与 +0.0 起累加**逐位相同**（见报告），所以用它做
    #    注入点不会失败 —— 那才是需要标 UNPROVEN 的情况。这里诚实地
    #    换一个确实会改变结果的位置。
    print()
    print("--- CF-3: inject a REAL arithmetic change (rounding order) ---")
    orig_mm = ref.mm

    def mm_perturbed(W, x, n_out, n_in):
        # 用 float64 中间量再截回 f32：这是**另一种**求值路径，
        # 会给出与「逐步 f32 舍入」不同的位型。
        out = np.zeros(n_out, dtype=F32)
        for i in range(n_out):
            acc = 0.0
            base = i * n_in
            for j in range(n_in):
                acc = acc + float(W[base + j]) * float(x[j])
            out[i] = F32(acc)
        return out

    ref.mm = mm_perturbed
    try:
        h4, _, _ = ref.forward(k, layers, h0, seq, d, k0)
    finally:
        ref.mm = orig_mm
    got4 = np.array([v for row in h4 for v in row], dtype=F32)
    bits4 = int((got4.view(np.uint32) != rust_final.view(np.uint32)).sum())
    chk("CF-3 CAN-FAIL: f64-intermediate accumulation => C2 detects bit diffs",
        bits4 > 0,
        "bit_diffs=%d (same algorithm, different rounding path)" % bits4)

    # =====================================================================
    # CF-4: C4a（扰动被选中者必须被检出）—— 用「恒等 forward」注入让它失败
    # =====================================================================
    print()
    print("--- CF-4: C4a (perturbation of a SELECTED weight must be detected) ---")

    def ffn_out0(w_arr, u, l0s, d, k0, k, topk=ref.topk_sorted):
        y = np.zeros(d, dtype=F32)
        for i in range(d):
            s = topk(w_arr[i * k0:(i + 1) * k0], l0s[i * k0:(i + 1) * k0], u, k0, k)
            acc = F32(0.0)
            first = True
            for j in s:
                sid = int(l0s[i * k0 + j])
                term = F32(w_arr[i * k0 + j] * u[sid])
                acc = term if first else F32(acc + term)
                first = False
            y[i] = acc
        return y

    y0 = ffn_out0(l0w, u, l0s, d, k0, k)
    y_sel = ffn_out0(l0w2, u, l0s, d, k0, k)
    ds = int((y0.view(np.uint32) != y_sel.view(np.uint32)).sum())
    chk("CF-4a baseline => C4a fires (changed_bits > 0)", ds > 0,
        "changed_bits=%d" % ds)

    # 注入：让 ffn 忽略权重（用恒等 top-k，但项里不含 w）——
    # 模拟「权重根本没进输出」的 bug。C4a 必须**失败**（detect 不到）。
    def ffn_ignoring_w(w_arr, u, l0s, d, k0, k, topk=ref.topk_sorted):
        y = np.zeros(d, dtype=F32)
        for i in range(d):
            s = topk(w_arr[i * k0:(i + 1) * k0], l0s[i * k0:(i + 1) * k0], u, k0, k)
            acc = F32(0.0)
            first = True
            for j in s:
                sid = int(l0s[i * k0 + j])
                term = F32(1.0 * u[sid])  # <-- w 被丢掉
                acc = term if first else F32(acc + term)
                first = False
            y[i] = acc
        return y

    y0b = ffn_ignoring_w(l0w, u, l0s, d, k0, k)
    y_selb = ffn_ignoring_w(l0w2, u, l0s, d, k0, k)
    dsb = int((y0b.view(np.uint32) != y_selb.view(np.uint32)).sum())
    chk("CF-4b CAN-FAIL: if w is dropped from the output, C4a goes to 0 (detects the bug)",
        dsb == 0,
        "changed_bits=%d (with w ignored, perturbing w cannot be detected)" % dsb)

    # =====================================================================
    # CF-5: C4b（扰动未被选中者必须不改变输出）
    # =====================================================================
    print()
    print("--- CF-5: C4b (perturbation of an UNSELECTED weight must be invisible) ---")
    i_u, j_u = target
    ju = i_u * k0 + j_u
    w_uns = l0w.copy()
    w_uns[ju] = F32(w_uns[ju] + F32(1e-3))
    # C4b 的 scoped 版本：只对 neuron i_u 的那一行扰动
    def ffn_row(w_arr, i, u, l0s, d, k0, k, topk=ref.topk_sorted):
        base = i * k0
        s = topk(w_arr[base:base + k0], l0s[base:base + k0], u, k0, k)
        acc = F32(0.0)
        first = True
        for j in s:
            sid = int(l0s[base + j])
            term = F32(w_arr[base + j] * u[sid])
            acc = term if first else F32(acc + term)
            first = False
        return acc

    y0_row = ffn_row(l0w, i_u, u, l0s, d, k0, k)
    y_uns_row = ffn_row(w_uns, i_u, u, l0s, d, k0, k)
    du = 0 if y0_row.tobytes() == y_uns_row.tobytes() else 1
    chk("CF-5a baseline => C4b holds (changed_bits == 0)", du == 0,
        "changed_bits=%d (neuron%d conn%d)" % (du, i_u, j_u))

    # 注入：把**所有**连接都当成选中（k = k0 的极端形式）⇒ 扰动必然可见
    def topk_all(w_row, src_row, x, k0_, k_):
        return list(range(k0_))

    y0_all = ffn_row(l0w, i_u, u, l0s, d, k0, k, topk=topk_all)
    y_uns_all = ffn_row(w_uns, i_u, u, l0s, d, k0, k, topk=topk_all)
    dua = 0 if y0_all.tobytes() == y_uns_all.tobytes() else 1
    chk("CF-5b CAN-FAIL: with NO skipping (all conns selected), C4b is violated",
        dua > 0,
        "changed_bits=%d (proves C4b measures the skip, not an accident)" % dua)

    print()
    print("=" * 72)
    print("can-fail audit: checks=%d fail=%d unproven=%d" % (N_CHECK, N_FAIL, N_UNPROVEN))
    print("Interpretation: every PASS above is a criterion that DID turn red when")
    print("given a known-wrong input. Any UNPROVEN item means the criterion could not")
    print("be made to fail and therefore must not be reported as PASS.")
    print("=" * 72)
    return 1 if N_FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
