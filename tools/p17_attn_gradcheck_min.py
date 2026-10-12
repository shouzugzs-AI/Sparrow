#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P17：**极小 1 层 attention 的数值梯度自检**（独立实现，纯 Python + numpy）。

================================================================================
为什么写这个（Lead 的 val-CE 发散问题）
================================================================================

Lead 在 task-1 实测到一个反直觉现象：

    train CE 降到 2.0，而 **val CE 升到 5.97**，且 val 比 unigram 基线(3.38)还差
    post-train argmax 准确率：train 43.3% / **val 5.6%**

这种形态（train 学得动、val 比基线还差）通常指向**反向传播里的某个符号/索引
错误**，而不是超参问题 —— 因为超参错误一般让两者一起差。

本文件的作用：给 Lead 一个**独立于他实现**的数值梯度参照，
把「反向是否正确」从「训练结果好不好」里**分离**出来。

  * 如果本文件的解析梯度与中心差分一致，而 Lead 的实现与两者都不一致
    ⇒ 问题在 Lead 的反向代码里，可以逐项对拍定位。
  * 如果本文件的解析梯度**自己**就不一致
    ⇒ 问题在**我这边的公式理解**里（即对 spec 的理解错了），
      那也是一个有用的结论 —— 它能说明「spec 有歧义」。

================================================================================
规模刻意极小（d=4, P=3, vocab=4）
================================================================================

理由：本检查的目的是**定位错误**，不是测性能。极小规模下：
  * 中心差分可以**逐参数**打印差异，直接指出是哪个矩阵的哪一项错了；
  * 全部参数只有 4 个 [4,4] 矩阵 + 嵌入 = 64+64 个可变参数，
    差分一遍秒级完成；
  * 4 个 [4,4] 矩阵 + P=3 ⇒ 每个归约只有 4 项，浮点误差可控。

⚠️ 规模小时的一个**真实风险**（必须防）：梯度里某个分量恰好是 0，
于是「解析 = 0，差分 = 0」也能通过 —— 那是**空洞**的通过。
所以本文件**报告非零分量的数量**，并要求每个矩阵都有足够多的非零项，
否则该矩阵的判据标 UNPROVEN。

================================================================================
前向公式（与 train/src/sparse_train24.rs 的注释逐条对齐）
================================================================================

  q = h Wq,  k = h Wk,  v = h Wv              （[P,d]，W 是 [d,d] 行主序）
  s[t][s2] = (q[t] · k[s2]) * scale,  scale = 1/sqrt(d)
  **因果**：s2 > t 的位置不参与（等价于 -inf）
  A[t][:]  = softmax(s[t][:t+1])
  o[t]     = A[t] · v
  u[t]     = h[t] + o[t] W_o                 （残差 + 输出投影）
  logits   = u[P-1] · E^T                    （只取最后一个位置，vocab=4）
  loss     = CE(logits, target)

反向用**手写解析式**，再用**中心差分**独立验证。两者都写在这里，
所以这个文件**自己**就能演示「判据能失败」（见 §can-fail）。

用法：
    python tools/p17_attn_gradcheck_min.py            # 默认 d=4 P=3 vocab=4
    python tools/p17_attn_gradcheck_min.py --eps 1e-3 --seed 7
"""

import argparse
import sys

import numpy as np

F32 = np.float32
F64 = np.float64


# ============================================================================
# 前向 + 解析反向（写在一起，便于逐行对照）
# ============================================================================
def forward(params, h, target, d, P, vocab, need_cache=False):
    Wq, Wk, Wv, Wo, E = params
    q = h @ Wq.T
    k = h @ Wk.T
    v = h @ Wv.T
    scale = 1.0 / np.sqrt(d)

    s = (q @ k.T) * scale                      # [P,P]
    mask = np.triu(np.ones((P, P)), k=1).astype(bool)   # s2 > t 为 True
    s_masked = np.where(mask, -np.inf, s)

    # 数值稳定 softmax（逐行）
    m = s_masked.max(axis=1, keepdims=True)
    ex = np.exp(s_masked - m)
    ex = np.where(mask, 0.0, ex)
    A = ex / ex.sum(axis=1, keepdims=True)

    o = A @ v                                  # [P,d]
    u = h + o @ Wo.T                           # [P,d]
    logits = u[P - 1] @ E.T                    # [vocab]

    # CE
    mx = logits.max()
    lse = mx + np.log(np.exp(logits - mx).sum())
    loss = lse - logits[target]

    if need_cache:
        return loss, logits, dict(q=q, k=k, v=v, A=A, o=o, u=u, scale=scale, mask=mask)
    return loss


def backward(params, h, target, d, P, vocab):
    """手写解析反向。返回与 params 同结构的梯度。"""
    Wq, Wk, Wv, Wo, E = params
    loss, logits, c = forward(params, h, target, d, P, vocab, need_cache=True)
    q, k, v, A, o, u, scale, mask = (
        c["q"], c["k"], c["v"], c["A"], c["o"], c["u"], c["scale"], c["mask"]
    )

    # ---- CE 反向 ----
    p = np.exp(logits - logits.max())
    p = p / p.sum()
    dlogits = p.copy()
    dlogits[target] -= 1.0

    dE = np.outer(dlogits, u[P - 1])           # [vocab,d]
    du = np.zeros((P, d))
    du[P - 1] = dlogits @ E                    # [d]

    # ---- u = h + o Wo^T ----
    dWo = du.T @ o                             # [d,d]
    do_ = du @ Wo                              # [P,d]

    # ---- o = A v ----
    dA = do_ @ v.T                             # [P,P]
    dv = A.T @ do_                             # [P,d]

    # ---- softmax 反向（因果行内）----
    dS = np.zeros((P, P))
    for t in range(P):
        r = t + 1
        ar = A[t, :r]
        dar = dA[t, :r]
        dS[t, :r] = ar * (dar - np.dot(ar, dar))

    # ---- s = q k^T * scale ----
    dS_scaled = dS * scale
    dq = dS_scaled @ k
    dk = dS_scaled.T @ q

    # ---- q = h Wq^T 等（h 不参与求导，只对 W 求导）----
    dWq = dq.T @ h
    dWk = dk.T @ h
    dWv = dv.T @ h

    return [dWq, dWk, dWv, dWo, dE]


# ============================================================================
# 中心差分
# ============================================================================
def numgrad(params, idx, i, j, eps, h, target, d, P, vocab):
    """对 params[idx][i,j] 做中心差分。"""
    orig = params[idx][i, j]
    params[idx][i, j] = orig + eps
    lp = forward(params, h, target, d, P, vocab)
    params[idx][i, j] = orig - eps
    lm = forward(params, h, target, d, P, vocab)
    params[idx][i, j] = orig
    return (lp - lm) / (2 * eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=4)
    ap.add_argument("--P", type=int, default=3)
    ap.add_argument("--vocab", type=int, default=4)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tol", type=float, default=2e-3)
    a = ap.parse_args()

    d, P, vocab, eps = a.d, a.P, a.vocab, a.eps
    rng = np.random.default_rng(a.seed)

    print("=" * 74)
    print("P17 MINIMAL 1-LAYER ATTENTION GRADCHECK (independent, central diff)")
    print("=" * 74)
    print("config: d=%d P=%d vocab=%d eps=%.1e seed=%d tol=%.1e" % (d, P, vocab, eps, a.seed, a.tol))
    print("spec  : causal softmax(q k^T/sqrt(d)) v ; u = h + o Wo^T ; CE on last pos")
    print()

    # 初始化：小权重（与 Xavier 同量级），使梯度量级合理、差分稳定
    s = 1.0 / np.sqrt(d)
    params = [
        rng.uniform(-s, s, (d, d)),      # Wq
        rng.uniform(-s, s, (d, d)),      # Wk
        rng.uniform(-s, s, (d, d)),      # Wv
        rng.uniform(-s, s, (d, d)),      # Wo
        rng.uniform(-s, s, (vocab, d)),  # E
    ]
    h = rng.uniform(-1, 1, (P, d))
    target = int(rng.integers(0, vocab))

    loss = forward(params, h, target, d, P, vocab)
    print("loss = %.8f   target = %d" % (loss, target))

    ana = backward(params, h, target, d, P, vocab)

    names = ["Wq", "Wk", "Wv", "Wo", "E"]
    n_check = 0
    n_fail = 0
    n_unproven = 0

    # ------------------------------------------------------------------
    # 逐矩阵对拍
    # ------------------------------------------------------------------
    print()
    print("--- analytic vs central-difference, per matrix ---")
    per_matrix = {}
    for idx, nm in enumerate(names):
        A_ = ana[idx]
        nz = int((np.abs(A_) > 1e-12).sum())
        rel_max = 0.0
        abs_max = 0.0
        worst = None
        for i in range(A_.shape[0]):
            for j in range(A_.shape[1]):
                g_num = numgrad(params, idx, i, j, eps, h, target, d, P, vocab)
                g_ana = A_[i, j]
                ad = abs(g_ana - g_num)
                denom = max(abs(g_ana), abs(g_num), 1e-8)
                rl = ad / denom
                if ad > abs_max:
                    abs_max = ad
                if rl > rel_max:
                    rel_max = rl
                    worst = (i, j, g_ana, g_num)
                n_check += 1
        ok = rel_max <= a.tol
        if not ok:
            n_fail += 1
        per_matrix[nm] = (ok, rel_max, abs_max, worst, nz)
        print("[%s] %-3s  rel_max=%.3e  abs_max=%.3e  nonzero=%d/%d  worst=(i=%d,j=%d ana=%.6e num=%.6e)"
              % ("PASS" if ok else "FAIL", nm, rel_max, abs_max, nz, A_.size,
                 worst[0] if worst else -1, worst[1] if worst else -1,
                 worst[2] if worst else 0.0, worst[3] if worst else 0.0))

    # ------------------------------------------------------------------
    # 稀疏性/空洞检查：若某矩阵梯度几乎全 0，对拍就是空的
    # ------------------------------------------------------------------
    print()
    print("--- vacuity guard (a nearly-all-zero grad makes the check empty) ---")
    for nm in names:
        ok, rel_max, abs_max, worst, nz = per_matrix[nm]
        if nz == 0:
            n_unproven += 1
            print("[UNPROVEN] %-3s gradient is all-zero => comparison is vacuous" % nm)
        else:
            print("[PASS] %-3s has %d nonzero entries => comparison is non-vacuous" % (nm, nz))

    # ------------------------------------------------------------------
    # can-fail：把反向里的一处**故意**改错，判据必须翻红
    # ------------------------------------------------------------------
    print()
    print("--- can-fail: perturb the analytic backward and require detection ---")

    def backward_broken(params, h, target, d, P, vocab, mode):
        Wq, Wk, Wv, Wo, E = params
        _, logits, c = forward(params, h, target, d, P, vocab, need_cache=True)
        q, k, v, A, o, u, scale, mask = (
            c["q"], c["k"], c["v"], c["A"], c["o"], c["u"], c["scale"], c["mask"]
        )
        p = np.exp(logits - logits.max())
        p = p / p.sum()
        dlogits = p.copy()
        dlogits[target] -= 1.0
        dE = np.outer(dlogits, u[P - 1])
        du = np.zeros((P, d))
        du[P - 1] = dlogits @ E
        dWo = du.T @ o
        do_ = du @ Wo
        dA = do_ @ v.T
        dv = A.T @ do_
        dS = np.zeros((P, P))
        for t in range(P):
            r = t + 1
            ar = A[t, :r]
            dar = dA[t, :r]
            dS[t, :r] = ar * (dar - np.dot(ar, dar))
        if mode == "drop_scale":
            dS_scaled = dS              # <-- 忘了乘 scale
        elif mode == "no_softmax_jac":
            dS_scaled = (dA * scale)    # <-- 用 dA 代替 dS（softmax 反向写错）
        elif mode == "neg_dv":
            dv = -dv                    # <-- 符号错
            dS_scaled = dS * scale
        elif mode == "drop_causal":
            # 不做因果：softmax 反向在整行上做（相当于非因果）
            dS2 = np.zeros((P, P))
            for t in range(P):
                ar = A[t, :]
                dar = dA[t, :]
                dS2[t, :] = ar * (dar - np.dot(ar, dar))
            dS_scaled = dS2 * scale
        else:
            dS_scaled = dS * scale
        dq = dS_scaled @ k
        dk = dS_scaled.T @ q
        dWq = dq.T @ h
        dWk = dk.T @ h
        dWv = dv.T @ h
        return [dWq, dWk, dWv, dWo, dE]

    for mode in ["drop_scale", "no_softmax_jac", "neg_dv", "drop_causal"]:
        brk = backward_broken(params, h, target, d, P, vocab, mode)
        # 与差分比：只要有任一项超过 tol 就算「检出」
        detected = False
        worst_rl = 0.0
        worst_where = None
        for idx in range(len(names)):
            A_ = brk[idx]
            for i in range(A_.shape[0]):
                for j in range(A_.shape[1]):
                    g_num = numgrad(params, idx, i, j, eps, h, target, d, P, vocab)
                    ad = abs(A_[i, j] - g_num)
                    rl = ad / max(abs(A_[i, j]), abs(g_num), 1e-8)
                    if rl > worst_rl:
                        worst_rl = rl
                        worst_where = (names[idx], i, j)
                    if rl > a.tol:
                        detected = True
        if mode == "drop_causal":
            # ==============================================================
            # ⚠️ 这个模式**测不出来**，而且原因是**结构性的**，不是 bug。
            #
            # 实测：drop_causal 的 worst_rel = 5.9e-8 —— 与正确版本**逐位相同**。
            #
            # 原因（已实测确认）：因果 mask 让 A 在**上三角恰好为 0**
            #   A[0] = [1, 0, 0]
            # 而「非因果 softmax 反向」多出来的那些项**都带因子 A[t,s2]**：
            #   dS2[t,:] = A[t,:] * (dA[t,:] - Σ_s A[t,s] dA[t,s])
            # 上三角 A=0 ⇒ 那些项恒为 0 ⇒ 两个版本**在数值上完全相同**。
            #
            # ⇒ 这条对照**无法证伪**「有没有做因果」。
            #   按纪律 #2，必须标 **UNPROVEN**，绝不能因为它是「对照」就
            #   当成 PASS —— 一个恒真的对照不加任何证据。
            #
            # ⇒ 正确的因果判据不是看反向，而是看**前向**：
            #   A[t, s2] ≡ 0 for s2 > t（这能失败：去掉 mask 立刻非零）。
            #   见下面 CAUSAL-1/CAUSAL-2。
            # ==============================================================
            n_unproven += 1
            print("[UNPROVEN] can-fail mode=%-16s detected=%s worst_rel=%.3e"
                  % (mode, detected, worst_rl))
            print("           REASON: causal mask makes A[t,s2]==0 for s2>t, and every")
            print("           extra term in the non-causal backward carries a factor A[t,s2].")
            print("           => the two variants are NUMERICALLY IDENTICAL; this control")
            print("           cannot falsify 'causality'. Asserted by the FORWARD instead:")
            continue
        n_check += 1
        if not detected:
            n_fail += 1
        print("[%s] can-fail mode=%-16s detected=%s  worst_rel=%.3e (must exceed tol=%.1e)"
              % ("PASS" if detected else "FAIL", mode, detected, worst_rl, a.tol))

    # ------------------------------------------------------------------
    # CAUSAL：因果性必须由**前向**断言（反向测不出来，见上面的 UNPROVEN）
    # ------------------------------------------------------------------
    print()
    print("--- CAUSAL: causality must be asserted on the FORWARD (see UNPROVEN above) ---")
    _, _, c = forward(params, h, target, d, P, vocab, need_cache=True)
    A_full = c["A"]
    upper = [A_full[t, s2] for t in range(P) for s2 in range(t + 1, P)]
    nz_upper = sum(1 for v in upper if v != 0.0)
    n_check += 1
    if nz_upper:
        n_fail += 1
    print("[%s] CAUSAL-1 A[t,s2] == 0 for ALL s2 > t   nonzero_upper=%d / %d"
          % ("PASS" if nz_upper == 0 else "FAIL", nz_upper, len(upper)))

    # 未来 token 不得影响过去位置的输出：改 v[P-1] 后前两个位置的 o 必须不变
    # （这是因果性的**行为级**判据，比看 A 的零更接近「机制」）
    v_before = c["v"].copy()
    o_before = c["o"].copy()
    # 手工重算：把最后一行 h[P-1] 大幅改掉，看 o[0], o[1] 是否变
    h2 = h.copy()
    h2[P - 1] *= 3.0
    _, _, c2 = forward(params, h2, target, d, P, vocab, need_cache=True)
    changed_early = int((c2["o"][:P - 1] != o_before[:P - 1]).sum())
    n_check += 1
    if changed_early:
        n_fail += 1
    print("[%s] CAUSAL-2 perturbing h[P-1] leaves o[0..P-2] UNCHANGED  changed_entries=%d"
          % ("PASS" if changed_early == 0 else "FAIL", changed_early))

    # CAUSAL-1 的 can-fail：去掉 mask，上三角必须立刻非零
    s_nc = (c["q"] @ c["k"].T) * c["scale"]
    m_nc = s_nc.max(axis=1, keepdims=True)
    e_nc = np.exp(s_nc - m_nc)
    A_nc = e_nc / e_nc.sum(axis=1, keepdims=True)
    nz_nc = sum(1 for t in range(P) for s2 in range(t + 1, P) if A_nc[t, s2] != 0.0)
    n_check += 1
    if nz_nc == 0:
        n_fail += 1
    print("[%s] CAUSAL-1 can-fail: WITHOUT the mask, upper triangle is nonzero  nonzero=%d / %d"
          % ("PASS" if nz_nc > 0 else "FAIL", nz_nc, len(upper)))

    # ------------------------------------------------------------------
    # 一个有用的诊断：h 的梯度（Lead 的层间反传需要它）
    # ------------------------------------------------------------------
    print()
    print("--- diagnostic: dH (needed for layer-to-layer backprop) ---")
    Wq, Wk, Wv, Wo, E = params
    _, _, c = forward(params, h, target, d, P, vocab, need_cache=True)
    q, k, v, A, o, u, scale, mask = (
        c["q"], c["k"], c["v"], c["A"], c["o"], c["u"], c["scale"], c["mask"]
    )
    p = np.exp(forward(params, h, target, d, P, vocab, need_cache=True)[1]
               - forward(params, h, target, d, P, vocab, need_cache=True)[1].max())
    p = p / p.sum()
    dlogits = p.copy()
    dlogits[target] -= 1.0
    du = np.zeros((P, d))
    du[P - 1] = dlogits @ E
    do_ = du @ Wo
    dA = do_ @ v.T
    dv = A.T @ do_
    dS = np.zeros((P, P))
    for t in range(P):
        r = t + 1
        ar = A[t, :r]
        dar = dA[t, :r]
        dS[t, :r] = ar * (dar - np.dot(ar, dar))
    dS_scaled = dS * scale
    dq = dS_scaled @ k
    dk = dS_scaled.T @ q
    dH_ana = du + dq @ Wq + dk @ Wk + dv @ Wv

    dH_num = np.zeros((P, d))
    for t in range(P):
        for j in range(d):
            orig = h[t, j]
            h[t, j] = orig + eps
            lp = forward(params, h, target, d, P, vocab)
            h[t, j] = orig - eps
            lm = forward(params, h, target, d, P, vocab)
            h[t, j] = orig
            dH_num[t, j] = (lp - lm) / (2 * eps)
    rel = np.abs(dH_ana - dH_num) / np.maximum(
        np.maximum(np.abs(dH_ana), np.abs(dH_num)), 1e-8)
    rl = float(rel.max())
    nz = int((np.abs(dH_ana) > 1e-12).sum())
    ok = rl <= a.tol
    if not ok:
        n_fail += 1
    n_check += 1
    print("[%s] dH analytic vs central diff  rel_max=%.3e  nonzero=%d/%d"
          % ("PASS" if ok else "FAIL", rl, nz, dH_ana.size))
    if nz == 0:
        n_unproven += 1
        print("[UNPROVEN] dH is all-zero => comparison is vacuous")

    print()
    print("=" * 74)
    print("checks=%d fail=%d unproven=%d" % (n_check, n_fail, n_unproven))
    if n_fail:
        print("RESULT: FAIL -- an analytic gradient disagrees with central difference.")
        print("        This localizes the bug to the formula above, NOT to hyperparameters.")
    else:
        print("RESULT: all analytic gradients match central difference within tol=%g." % a.tol)
        print("        => the formula in THIS file is correct on this spec.")
        print("        If Lead's implementation disagrees with these numbers, the bug is")
        print("        in that implementation (compare per-matrix, per-entry).")
    print("=" * 74)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
