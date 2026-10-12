#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment D: per-layer effective rank of the activations (SVD).

Independent tool: it reads a directory produced by
    vmlm-b1-full.exe ... --dump-acts <dir>
which contains
    meta.json          model/provenance info
    input.f32          [tokens x dim]  the embedding expansion
    layer_00.f32 ...   [tokens x dim]  each layer's OUTPUT (row-major f32)

and writes a markdown report (default docs/B1_effrank.md).

WHY RANK (and what it can and cannot show)
------------------------------------------
Each row is one token's activation vector in R^dim. Stacking `tokens` rows gives a
matrix A [tokens x dim]; rank(A) <= min(tokens, dim). Two things the numbers DO show:

  * The INPUT has rank <= k0_embed by construction (only k0_embed coordinates are
    non-zero), so its effective rank is a ceiling that the stack can only lift by
    mixing coordinates.
  * Layer-by-layer: if the effective rank stays at the input's level, the sparse
    layers are (in this measurable sense) not creating new directions; if it grows,
    they are mixing.

What it does NOT show: that the model "learned knowledge", or that the layer is
"just passing through". Rank is a property of the representation geometry, not of
task performance. The attribution question is answered by experiments A/B/C; this
is their structural complement.

Effective rank definition used here (reported for both thresholds):
    k_eff(p) = min k such that sum_{i<=k} s_i^2 / sum_i s_i^2 >= p      (p = 0.95, 0.99)
also reported: stable rank = sum(s^2) / max(s)^2, and the top-8 singular values.

Usage:
  python tools/b1_effrank.py --dir artifacts/effrank/base
  python tools/b1_effrank.py --dir artifacts/effrank/base --out docs/B1_effrank.md
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = str(Path(__file__).resolve().parents[1])


def read_f32(path, rows, cols):
    a = np.fromfile(path, dtype="<f4")
    if a.size != rows * cols:
        raise SystemExit("FATAL: %s has %d f32, expected %d (%d x %d)" % (path, a.size, rows * cols, rows, cols))
    return a.reshape(rows, cols)


def eff_rank(sv2, p):
    tot = sv2.sum()
    if tot <= 0:
        return 0
    c = np.cumsum(sv2) / tot
    k = int(np.searchsorted(c, p) + 1)
    return min(k, len(sv2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "B1_effrank.md"))
    ap.add_argument("--topk", type=int, default=8)
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.dir, "meta.json"), encoding="utf-8-sig"))
    dim = meta["dim"]
    T = meta["tokens"]
    layers = meta["layers"]

    names = ["input"] + ["layer_%02d" % l for l in range(layers)]
    rows = []
    for nm in names:
        p = os.path.join(args.dir, nm + ".f32")
        if not os.path.isfile(p):
            raise SystemExit("FATAL: missing %s" % p)
        A = read_f32(p, T, dim).astype(np.float64)
        # 中心化：比较"表示用了多少方向"，不把公共均值算成一个方向
        Ac = A - A.mean(axis=0, keepdims=True)
        # SVD：经济型即可
        s = np.linalg.svd(Ac, compute_uv=False)
        s2 = s ** 2
        nz = int((np.abs(A) > 1e-12).sum(axis=1).mean())  # 平均非零坐标数/行
        rows.append(
            {
                "name": nm,
                "nz_per_row": nz,
                "rank_exact": int(np.linalg.matrix_rank(A)),
                "rank_exact_c": int(np.linalg.matrix_rank(Ac)),
                "keff95": eff_rank(s2, 0.95),
                "keff99": eff_rank(s2, 0.99),
                "stable": float(s2.sum() / (s2[0] + 1e-30)),
                "sv": s[: args.topk].tolist(),
                "fro": float(np.sqrt(s2.sum())),
            }
        )
        print(
            "%-10s nz/row=%7.1f  rank=%4d (centered %4d)  keff95=%4d  keff99=%4d  stable=%8.2f"
            % (nm, nz, rows[-1]["rank_exact"], rows[-1]["rank_exact_c"], rows[-1]["keff95"], rows[-1]["keff99"], rows[-1]["stable"])
        )

    # 头部输入（瓶颈度量）：末层输出 与 **原始嵌入展开** 横向拼接后的秩。
    # 这正是 `--head-skip 1`（嵌入→head 跨层残差）要改善的量：
    # head 实际看到的输入有多少个有效方向。
    concat = None
    plast = os.path.join(args.dir, "layer_%02d.f32" % (layers - 1))
    pinput = os.path.join(args.dir, "input.f32")
    if layers >= 1 and os.path.isfile(plast) and os.path.isfile(pinput):
        A_last = read_f32(plast, T, dim).astype(np.float64)
        A_in = read_f32(pinput, T, dim).astype(np.float64)
        # ⚠️ 两个块必须**各自归一化**再拼接：head 对两条通路各有独立权重矩阵，
        #    所以"可用方向数"不该被两块之间的尺度差吃掉。第一版直接 hstack，
        #    结果整块的能量被尺度大的那块（末层）主导，keff95 恒等于末层的 2 —— 那是个假象。
        nl = np.linalg.norm(A_last)
        ni = np.linalg.norm(A_in)
        Al = A_last / (nl + 1e-30)
        Ai = A_in / (ni + 1e-30)
        Ac = np.hstack([Al, Ai])
        Ac = Ac - Ac.mean(axis=0, keepdims=True)
        s = np.linalg.svd(Ac, compute_uv=False)
        s2 = s ** 2
        concat = {
            "keff95": eff_rank(s2, 0.95),
            "keff99": eff_rank(s2, 0.99),
            "stable": float(s2.sum() / (s2[0] + 1e-30)),
            "fro_last": float(nl),
            "fro_in": float(ni),
        }
        print(
            "head-in  (normalized last+input concat, %d cols)  keff95=%4d  keff99=%4d  stable=%8.2f   [fro last=%.3e in=%.3e]"
            % (2 * dim, concat["keff95"], concat["keff99"], concat["stable"], nl, ni)
        )

    out = []
    w = out.append
    w("# 实验 D：逐层激活的有效秩（SVD）")
    w("")
    w("> 由 `tools/b1_effrank.py` 从 `vmlm-b1-full --dump-acts` 的产物算出。**这是可测量量，不是训练结果。**")
    w("")
    w("## 数据来源")
    w("")
    w("| 项 | 值 |")
    w("|---|---|")
    for k in ("dim", "layers", "tokens", "k0", "k0_embed", "k0_head", "io_sparse", "identity_layers", "steps", "seed", "freeze_layers", "freeze_io", "final_val"):
        if k in meta:
            w("| %s | %s |" % (k, meta[k]))
    w("| dump 目录 | `%s` |" % args.dir)
    w("")
    w("## 定义（写清楚，避免「有效秩」被当成结论）")
    w("")
    w("* 矩阵 `A = [tokens × dim]`，**每行是一个 token 的激活**（`input` 是嵌入展开，`layer_ll` 是第 ll 层输出）。")
    w("* **先中心化**（减去每维均值）再 SVD：否则「所有 token 共有的那个均值方向」会被算成第 1 个奇异方向，把有效秩压低。")
    w("* `keff95 / keff99` = 奇异值平方累积能量达到 95% / 99% 所需的方向数。")
    w("* `rank` 用数值秩（阈值 1e-8·max）；`stable` = ‖A‖_F² / σ₁²（稳定秩）。")
    w("* 上界：`rank ≤ min(tokens, dim)`；输入还额外受 `rank ≤ k0_embed` 限制（只有 k0_embed 个坐标非零）。")
    w("")
    w("## 逐层结果")
    w("")
    w("| 矩阵 | 每行非零坐标数 | 精确秩 | 中心化秩 | **keff95** | keff99 | 稳定秩 | ‖A‖_F |")
    w("|---|---|---|---|---|---|---|---|")
    for r in rows:
        w(
            "| %s | %.1f | %d | %d | **%d** | %d | %.1f | %.3e |"
            % (r["name"], r["nz_per_row"], r["rank_exact"], r["rank_exact_c"], r["keff95"], r["keff99"], r["stable"], r["fro"])
        )
    w("")
    w("## 前 %d 个奇异值" % args.topk)
    w("")
    w("| 矩阵 | " + " | ".join("σ%d" % (i + 1) for i in range(args.topk)) + " |")
    w("|" + "---|" * (args.topk + 1))
    for r in rows:
        w("| %s | " % r["name"] + " | ".join("%.3e" % v for v in r["sv"]) + " |")
    w("")
    # 结论性读数（只报数字与判据，不替读者下结论）
    w("## 读数")
    w("")
    ki = rows[0]["keff95"]
    kl = rows[-1]["keff95"]
    w("* 输入 keff95 = **%d**（构造上界 k0_embed = %s，实测每行非零 %.1f 个坐标）" % (ki, meta.get("k0_embed"), rows[0]["nz_per_row"]))
    w("* 最后一层 keff95 = **%d**（tokens=%d、dim=%d ⇒ 上界 %d）" % (kl, T, dim, min(T, dim)))
    if kl <= ki + 1:
        w("* ⇒ **24 层之后有效秩没有扩大**：稀疏层在这套结构里没有（按这个度量）新增方向。")
    else:
        w("* ⇒ **有效秩从 %d 扩到 %d（×%.2f）**：稀疏层确实在混叠/新增方向。" % (ki, kl, kl / max(1, ki)))
    w("* 逐层轨迹：%s" % " → ".join(str(r["keff95"]) for r in rows))
    w("")
    w("⚠️ 边界：秩是**表示几何**的度量，**不能**说明「知识在哪」。要回答归因，看实验 A/B/C（冻结对照）。")
    w("")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
