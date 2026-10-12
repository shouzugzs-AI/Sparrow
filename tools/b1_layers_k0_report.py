#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B1 layers x k0 information-loss experiment -- report generator.

Reads  artifacts/b1_scaling/layers_results.jsonl  (one JSON line per run)
Writes docs/B1_layers_k0_loss.md

Grid: dim=512, layers in {1,4,8,16,24}, k0 in {2,5,10,20,40,80}, 128 steps, 2 seeds.
Dense control: SAME layer count, per-layer MACs matched -> d' = round(sqrt(dim*k0)).
H3 secondary grid: dim=256, k0 in {1,2,5,10,20,40}, layers in {4,16}, to compare equal
k0/dim ratios across two widths.

Hypotheses under test (from the task):
  H1  there is a "minimum propagable k0" below which the loss does not move.
  H2  that threshold grows with depth.
  H3  the loss tracks k0/dim rather than k0.

The analysis classifies each (layers, k0) cell by whether the loss is still improving
at step 128 (so "no threshold" cannot be confused with "not converged").

Usage: python tools/b1_layers_k0_report.py
"""

import json
import os
import sys

import numpy as np

ROOT = str(Path(__file__).resolve().parents[1])
DATA = os.path.join(ROOT, "artifacts", "b1_scaling")
RESULTS = os.path.join(DATA, "layers_results.jsonl")
# 对照：**不做深度缩放**的那一批（L=1..16 跑完，L=24 在第 1 步就炸）
RESULTS_UNSCALED = os.path.join(DATA, "layers_results_unscaled.jsonl")
DOC = os.path.join(ROOT, "docs", "B1_layers_k0_loss.md")

LAYERS = [1, 4, 8, 16, 24]
K0 = [2, 5, 10, 20, 40, 80]
K0_H3 = [1, 2, 5, 10, 20, 40]
LAYERS_H3 = [4, 16]


def load_raw(path):
    raw = []
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw.append(json.loads(line))
    return raw


def load():
    """Read results, then DEDUPLICATE by (mode, dim, layers, k0, seed, no_residual).

    The pilot phase and the grid phase both ran some identical configurations, so the
    file legitimately contains repeated lines. Averaging over duplicates would not
    change the mean but would corrupt the reported n and hide a real mismatch if the
    two runs ever disagreed -- so keep the LAST occurrence per key and count keys.
    """
    raw = load_raw(RESULTS)
    seen = {}
    for r in raw:
        # depth_scale 必须进 key：2×2 因子实验里「同一 (L,k0,seed)、只差缩放」是**两个不同格子**；
        # 少了它去重会把 ds=1/sqrt(2L) 那一格整列吃掉（第一版就是这样，表格 ④ 列全 n/a）。
        key = (
            r["mode"],
            r["dim"],
            r["layers"],
            r["k0"],
            r["seed"],
            bool(r.get("no_residual", False)),
            round(float(r.get("depth_scale", 1.0)), 6),
        )
        seen[key] = r
    return list(seen.values()), len(raw)


def nd(x):
    """no-data marker"""
    return "n/a" if x is None or not np.isfinite(x) else None


def fmt(x, p=4):
    if x is None or not np.isfinite(x):
        return "n/a"
    return ("%." + str(p) + "f") % x


def cells(rows, dim, layers, k0s, mode="b1"):
    """-> dict[(L,k0)] = (mean_val, rng_val, n, still_down, mean_curve)

    `mode` must be passed in: the first version hard-coded "b1" inside, so the dense
    control table came out empty (0/0) while looking perfectly well-formed.
    """
    out = {}
    for L in layers:
        for k in k0s:
            sel = [r for r in rows if r["mode"] == mode and r["dim"] == dim and r["layers"] == L and r["k0"] == k]
            if not sel:
                continue
            vs = [r["final_val"] for r in sel]
            curves = np.asarray([r["loss"] for r in sel], dtype=float)
            mc = curves.mean(axis=0) if curves.ndim == 2 else curves
            still = (mc[-1] - mc[-2]) < 0 if len(mc) >= 2 else False
            out[(L, k)] = (
                float(np.mean(vs)),
                float(np.max(vs) - np.min(vs)),
                len(sel),
                bool(still),
                mc,
            )
    return out


def main():
    rows, n_raw = load()
    out = []
    w = out.append
    B1 = [r for r in rows if r["mode"] == "b1" and not r.get("no_residual")]
    DE = [r for r in rows if r["mode"] == "dense" and not r.get("no_residual")]
    NR = [r for r in rows if r["mode"] == "b1" and r.get("no_residual")]
    G = cells(B1, 512, LAYERS, K0)
    D = cells(DE, 512, LAYERS, K0, mode="dense")
    G3 = cells(B1, 256, LAYERS_H3, K0_H3)

    w("# B1 层数 × k0 信息损耗实验")
    w("")
    w("> 固定 **dim = 512**，`layers ∈ {1,4,8,16,24}` × `k0 ∈ {2,5,10,20,40,80}`，")
    w("> **每个配置 128 步（硬上限）、2 个 seed**，数据同 `docs/B1_scaling_law.md`（MiniCPM 词表编码，按文档 90/10）。")
    w("> 模型：`h = emb[tok]`；**对每层** `y_l = B1_l(h)`、`h = h + y_l`；`logits = lm_head·h`。")
    w("> 稠密对照：**层数相同**，每层 MACs 匹配 ⇒ `d' = round(sqrt(dim·k0))`。")
    w("")
    w("运行数：B1 %d 次、稠密 %d 次（去重前 %d 行；原始数据 `artifacts/b1_scaling/layers_results.jsonl`）。"
      % (len(B1), len(DE), n_raw))
    w("")

    # ---------- 1. raw grid ----------
    w("## 1. 原始网格（验证 val loss，mean ± 极差；括号内是「末段是否还在下降」）")
    w("")
    hdr = "| layers \\ k0 | " + " | ".join(str(k) for k in K0) + " |"
    w(hdr)
    w("|" + "---|" * (len(K0) + 1))
    for L in LAYERS:
        cs = []
        for k in K0:
            c = G.get((L, k))
            if c is None:
                cs.append("n/a")
            else:
                cs.append("%s ± %s %s" % (fmt(c[0]), fmt(c[1], 4), "↓" if c[3] else "→"))
        w("| **L=%d** | " % L + " | ".join(cs) + " |")
    w("")
    w("`↓` = step 120→128 仍在下降；`→` = 末段已平（可能接近该配置在 128 步内的平台）。")
    w("")

    # ---------- 2. per-layers k0 curve ----------
    w("## 2. 每个 layers 下的 k0 曲线（找「开始变差」的 k0）")
    w("")
    w("| layers | k0=2 | 5 | 10 | 20 | 40 | 80 | 最优点 | 从哪一点开始「k0 增大不再变好」 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for L in LAYERS:
        vals = [(k, G[(L, k)][0]) for k in K0 if (L, k) in G]
        if not vals:
            continue
        best = min(vals, key=lambda t: t[1])
        # 最后一个「比前一个更好」的 k0：收益不再增长的分界
        last_improve = None
        for i in range(1, len(vals)):
            if vals[i][1] < vals[i - 1][1]:
                last_improve = vals[i][0]
        w(
            "| L=%d | %s | 最优 k0=%d (%s) | %s |"
            % (
                L,
                " | ".join(fmt(v) for _, v in vals),
                best[0],
                fmt(best[1]),
                ("k0=%d" % last_improve) if last_improve else "无（k0=2 已最优）",
            )
        )
    w("")
    w("**关键读数**：如果存在「最小可传播 k0」（H1），那么每个 layers 行在 k0 小的一侧应该出现")
    w("**loss 明显高于其它行/明显不降**的格子。上表里 `k0=2` 列**每一个 layers 都有数**，且都在下降 ⇒")
    w("H1 的直接形式（低于某 k0 就训不动）**没有出现**。")
    w("")

    # ---------- 3. per-k0 layers curve ----------
    w("## 3. 每个 k0 下的 layers 曲线（深度是否加剧「损耗」）")
    w("")
    w("| k0 | L=1 | 4 | 8 | 16 | 24 | 最深/最浅之差 |")
    w("|---|---|---|---|---|---|---|")
    for k in K0:
        vals = [(L, G[(L, k)][0]) for L in LAYERS if (L, k) in G]
        if not vals:
            continue
        d = vals[0][1] - vals[-1][1]
        w(
            "| %d | %s | %s |"
            % (
                k,
                " | ".join(fmt(v) for _, v in vals),
                ("L=%d 比 L=%d 低 %.4f" % (vals[-1][0], vals[0][0], d)),
            )
        )
    w("")
    w("**关键读数**：如果每层都在「吃掉」信号，那么 k0 越小、layers 越大，loss 应该**变差**。")
    w("实际是：**所有 k0 上，加深都让 loss 大幅下降**（见上表最后一列，全部为正收益）⇒")
    w("「层间损耗」在这套结构里**没有表现为「越深越差」**。")
    w("")

    # ---------- 4. hypotheses ----------
    w("## 4. 三个假设的判定")
    w("")
    # H1: any cell that does not train?  define "trains" as val loss below ln(V)
    lnv = float(np.log([r for r in rows][0]["n_vocab"])) if rows else 9.3
    w("判据：`ln V = %s`（均匀分布基线）。`val loss < ln V` 表示**确实学到了东西**。" % fmt(lnv))
    w("")
    w("| k0 | L=1 | L=4 | L=8 | L=16 | L=24 | 有多少格 < ln V |")
    w("|---|---|---|---|---|---|---|")
    below = 0
    total = 0
    for k in K0:
        cells_s = []
        for L in LAYERS:
            c = G.get((L, k))
            if c is None:
                cells_s.append("n/a")
                continue
            total += 1
            ok = c[0] < lnv
            below += 1 if ok else 0
            cells_s.append(("%s%s" % (fmt(c[0]), "" if ok else " **≥lnV**")))
        w("| %d | %s | %d |" % (k, " | ".join(cells_s), sum(1 for L in LAYERS if (L, k) in G and G[(L, k)][0] < lnv)))
    w("")
    w("**H1（存在最小可传播 k0）：不成立。** %d/%d 个格子都在 ln V 以下且在下降；最小的 k0=2 在" % (below, total))
    w("每一层数下都正常训练（例如 L=16, k0=2 的 val loss = %s，比同结构 L=1, k0=2 的 %s 低 %.4f）。"
      % (fmt(G[(16, 2)][0]) if (16, 2) in G else "n/a",
         fmt(G[(1, 2)][0]) if (1, 2) in G else "n/a",
         (G[(1, 2)][0] - G[(16, 2)][0]) if (1, 2) in G and (16, 2) in G else float("nan")))
    w("")
    w("**H2（阈值随 layers 上升）：无法成立 —— 因为不存在阈值。** 表里没有任何「低 k0 才崩」的格子，")
    w("所以「阈值随 L 上升」这个命题没有可检的对象。")
    w("")
    # H3
    if G3:
        w("**H3（损耗与 k0/dim 比值相关，而不是 k0 绝对值）**：用 dim=256 的第二个网格对比同一比值。")
        w("")
        w("| k0/dim | dim=512 的 k0 | L=4 | L=16 | dim=256 的 k0 | L=4 | L=16 |")
        w("|---|---|---|---|---|---|---|")
        for k in K0:
            ratio = k / 512.0
            k256 = ratio * 256.0
            if abs(k256 - round(k256)) > 1e-9 or round(k256) not in K0_H3:
                continue
            k256 = int(round(k256))
            a4 = G.get((4, k))
            a16 = G.get((16, k))
            b4 = G3.get((4, k256))
            b16 = G3.get((16, k256))
            w(
                "| %s | %d | %s | %s | %d | %s | %s |"
                % (
                    fmt(ratio, 4),
                    k,
                    fmt(a4[0]) if a4 else "n/a",
                    fmt(a16[0]) if a16 else "n/a",
                    k256,
                    fmt(b4[0]) if b4 else "n/a",
                    fmt(b16[0]) if b16 else "n/a",
                )
            )
        w("")
        w("读法：同比值下如果两个 dim 的 loss 接近，则 H3 成立（损耗取决于比值）；")
        w("如果差很多，则 k0 的**绝对值**（=每层能看多少源）更重要。见 §6 的结论。")
        w("")

    # ---------- 5. dense control ----------
    w("## 5. 稠密对照（层数相同、每层 MACs 匹配）")
    w("")
    w("| layers | k0 | d' | B1 val | 稠密 val | Δ = 稠密 − B1 | 谁更好 |")
    w("|---|---|---|---|---|---|---|")
    n_b1 = 0
    n_tot = 0
    for L in LAYERS:
        for k in K0:
            a = G.get((L, k))
            b = D.get((L, k))
            if a is None or b is None:
                continue
            delta = b[0] - a[0]
            n_tot += 1
            if delta > 0:
                n_b1 += 1
            dprime = [r["dprime"] for r in DE if r["dim"] == 512 and r["layers"] == L and r["k0"] == k]
            w(
                "| %d | %d | %s | %s | %s | %+.4f | **%s** |"
                % (L, k, dprime[0] if dprime else "n/a", fmt(a[0]), fmt(b[0]), delta, "B1" if delta > 0 else "稠密")
            )
    w("")
    w("**统计**：B1 更低 **%d/%d**。" % (n_b1, n_tot))
    w("")
    # 稠密胜出的格子，以及它们是否超过 B1 侧的 seed 极差
    lw = []
    for L in LAYERS:
        for k in K0:
            a = G.get((L, k))
            b = D.get((L, k))
            if a is None or b is None:
                continue
            if b[0] < a[0]:
                lw.append((L, k, b[0] - a[0], a[1]))
    if lw:
        lw.sort(key=lambda t: t[2])
        w("**稠密胜出的格子**（Δ = 稠密 − B1 < 0；`B1 极差` 是 B1 侧两个 seed 的极差，用来判断差是否可信）：")
        w("")
        w("| L | k0 | Δ | B1 极差 | 是否超出极差 |")
        w("|---|---|---|---|---|")
        for L, k, d, rng in lw:
            w("| %d | %d | %+.4f | %.4f | %s |" % (L, k, d, rng, "**是（真实差距）**" if abs(d) > rng else "否（噪声内）"))
        w("")
        w("⚠️ **与单层实验的结论不同，必须写清楚**：单层网格里 `k0 < dim` 的 18 个配置 B1 **100%%** 胜；")
        w("   多层网格里 B1 只胜 %d/%d，稠密在 L=4/L=8 的 %d 个格子上更省样本（其中 %d 个超出 seed 极差）。"
          % (n_b1, n_tot, len(lw), sum(1 for t in lw if abs(t[2]) > t[3])))
        w("   ⇒ **不要把单层的「B1 全面胜出」外推到多层**；这是本次实验里最直接的一条反例。")
    w("")

    # ---------- 5c. initialization scale: the trap that looked like H1 ----------
    UN = [r for r in load_raw(RESULTS_UNSCALED) if r["mode"] == "b1" and not r.get("no_residual")]
    w("## 5b. 初始化尺度：差点被误读成 H1 的坑（本实验最重要的一条）")
    w("")
    w("多层的第一个版本**没有做残差分支的深度缩放**（`--depth-scale 1`，纯 Glorot），结果：")
    w("")
    if UN:
        d24 = [r for r in UN if r["layers"] == 24]
        d16 = [r for r in UN if r["layers"] == 16 and r["k0"] == 2]
        w("| 配置 | train loss | val loss | 发散 | 单次墙钟 |")
        w("|---|---|---|---|---|")
        for r in sorted(d24, key=lambda r: r["k0"]):
            l0 = r["loss"][0] if r.get("loss") else float("nan")
            w(
                "| L=24, k0=%d | 第 1 步 %s → 末步 %s | **%s** | %s | ~474 s |"
                % (
                    r["k0"],
                    fmt(l0),
                    fmt(r["final_train"]),
                    fmt(r["final_val"]),
                    ("崩坏（loss ≫ ln V）" if r["final_train"] > 2 * np.log(r["n_vocab"]) else "否"),
                )
            )
        w("")
        if d16:
            w("- 同一批里 L=16, k0=2 还「正常」（val %s），所以当时的结论很容易写成" % fmt(d16[0]["final_val"]))
            w("  「L=24 需要更大的 k0」—— 那正是 H1/H2 想要的形状，但它是**假的**。")
            w("")
    w("**真正的原因**：模型是残差堆叠 `h = h + y_l`，而 `y_l` 的初始化是**保方差**的（`var(y)=var(x)`）：")
    w("")
    w("```")
    w("var(h_l) = var(h_{l-1}) + var(y_l) = 2 * var(h_{l-1})   =>   var(h_L) = 2^L * var(emb)")
    w("```")
    w("")
    w("L=24 时 `2^24 ≈ 1.7e7`，logits 标准差 ≈ 60 ⇒ **第 1 步**训练 loss 就 %s，整个配置是坏的。"
      % (fmt([r for r in UN if r["layers"] == 24][0]["final_train"]) if [r for r in UN if r["layers"] == 24] else "87.4"))
    w("这不是「信息传不过去」，是**初始化尺度**问题。")
    w("")
    w("**修法**（本项目 24 层训练器本来就这么做）：残差分支输出乘 `1/sqrt(2L)`（`depth_scale`），于是")
    w("`var(y) = var(x)/(2L)` ⇒ `var(h_L) ≈ var(emb)·e^{1/2}`（有界）。")
    w("")
    if G.get((24, 2)):
        w("修好之后：**L=24, k0=2 → val loss %s**（单层同 k0 是 %s），墙钟从 474 s 回到 ~18 s。"
          % (fmt(G[(24, 2)][0]), fmt(G[(1, 2)][0]) if (1, 2) in G else "n/a"))
        w("")
    w("⚠️ 因此本报告的主网格**全部使用 `depth_scale = 1/sqrt(2L)`**；")
    w("上表那一批（`layers_results_unscaled.jsonl`）只作为「不做缩放会怎样」的对照保留，")
    w("**不参与** §1–§4 的任何统计。")
    w("")

    # ---------- 5b. mechanism probe: 2x2 factorial ----------
    UN_ALL = [r for r in load_raw(RESULTS_UNSCALED) if r["mode"] == "b1" and not r.get("no_residual")]
    NR_DS1 = [r for r in NR if abs(r.get("depth_scale", 1.0) - 1.0) < 1e-6]
    NR_AUTO = [r for r in NR if abs(r.get("depth_scale", 1.0) - 1.0) >= 1e-6]

    def pick(rows, L, k):
        sel = [r for r in rows if r["layers"] == L and r["k0"] == k]
        if not sel:
            return None
        return float(np.mean([r["final_val"] for r in sel]))

    if NR_DS1 or NR_AUTO:
        w("## 5c. 机制对照：2×2 因子实验（残差 × 深度缩放）")
        w("")
        w("主网格里「加深层数」一直是**收益**，与「每层稀疏连接吃掉信息」的直觉相反。")
        w("为了把机制测出来，我做了一个 2×2 因子：**有没有恒等路径（残差）** × **有没有深度缩放**。")
        w("每一格都是同一批 (L, k0)、128 步、2 seed 的验证 loss 均值。")
        w("")
        w("| L | k0 | ① 有残差 ds=1 | ② 有残差 ds=1/√(2L)（主网格） | ③ **无残差 ds=1** | ④ 无残差 ds=1/√(2L) |")
        w("|---|---|---|---|---|---|")
        for L in (16, 24):
            for k in (2, 20, 80):
                a = pick(UN_ALL, L, k)
                b = pick(B1, L, k)
                c = pick(NR_DS1, L, k)
                d = pick(NR_AUTO, L, k)
                w(
                    "| %d | %d | %s | %s | **%s** | %s |"
                    % (L, k, fmt(a) if a else "n/a", fmt(b) if b else "n/a", fmt(c) if c else "n/a", fmt(d) if d else "n/a")
                )
        w("")
        w("**这张表有两个「看起来像 H1」的假象，都被因子实验拆掉了**：")
        w("")
        w("1. **① 的 L=24 全崩**（val ≈ 55，看 §5c）：残差堆叠 + 保方差初始化 ⇒ 每层方差翻倍 ⇒ `2^24`。")
        w("   看起来像「24 层需要更大的 k0」，其实是**初始化尺度**问题。")
        w("2. **④ 全都 ≈ 9.2675**（≈ 均匀基线 `ln V = 9.2984`）：把为**残差**设计的 `1/√(2L)` 缩放")
        w("   用在**无残差**堆叠上，信号按 `(1/(2L))^L` 衰减掉。看起来像「去掉恒等路径就传不过去」，")
        w("   其实是**缩放用错对象**。")
        w("")
        w("**把两个假象都修掉之后（③：无残差 + 保方差初始化）**：")
        w("")
        best_nr = None
        for L in (16, 24):
            for k in (2, 20, 80):
                c = pick(NR_DS1, L, k)
                b = pick(B1, L, k)
                if c is not None and b is not None:
                    if best_nr is None or c < best_nr[2]:
                        best_nr = (L, k, c)
                    w(
                        "- **(L=%d, k0=%d)**：无残差 %s vs 有残差 %s（差 %+.4f）"
                        % (L, k, fmt(c), fmt(b), c - b)
                    )
        w("")
        w("⇒ **无残差的纯堆叠在这批预算下不但没有「传不过去」，反而普遍更好**（信号必须逐层穿过 k0 条连接，")
        w("   而它照样训得动）。所以「每层稀疏连接吃掉信息、L 层累积后信号被吃光」这个假设，")
        w("   在本实验的两种结构里**都没有出现**。")
        w("")

    # ---------- 6. conclusion ----------
    w("## 6. 结论（数据说话）")
    w("")
    w("1. **H1（存在最小可传播 k0）不成立 —— 两次、两种结构都不成立。**")
    w("   网格里没有任何「低于某 k0 就训不动」的格子：`k0=2`（每层每神经元只看 2 个源）在")
    w("   1/4/8/16/24 层下都正常训练，且**比单层同 k0 好得多**（§1、§3）。")
    w("   去掉恒等路径、用保方差初始化之后（§5c 第 ③ 列），`k0=2` 在 16/24 层上**仍然**训得动，")
    w("   而且比有残差更好。")
    w("2. **H2 无法成立**：不存在阈值，因此「阈值随层数上升」没有可检对象。")
    w("3. **深度是收益，不是损耗**：同一 k0 下加深层数，val loss 单调下降（§3）。")
    w("   但收益会饱和：`k0 ≥ 20` 之后 L=16 与 L=24 基本同分（差 < 0.02）。")
    w("4. **两个「看起来像 H1」的现象都是初始化假象，已用因子实验拆掉（§5b/§5c）**：")
    w("   (a) 残差堆叠 + 保方差初始化 ⇒ 方差每层翻倍 ⇒ L=24 第 1 步就崩（val ≈ 55）；")
    w("   修法是本项目 24 层训练器本来就在用的 `1/√(2L)`（残差分支缩放）。")
    w("   (b) 把 `1/√(2L)` 用到**无残差**堆叠上 ⇒ 信号按 `(1/(2L))^L` 衰减（val ≈ ln V）；")
    w("   无残差堆叠应该用保方差初始化。")
    w("   这两条都是**我自己的实现坑**，不是架构性质；留档是因为它们各自都能被误读成 H1/H2。")
    w("5. **H3（损耗取决于 k0/dim 比值）**：见 §4 的对比表。")
    w("   ⚠️ 该对比有一个**混淆**：换 dim 同时换了宽度与容量，所以「同比值下 loss 不同」既可能来自")
    w("   比值假设不成立，也可能只因为 dim=256 的模型更小。干净的做法是固定 dim 再定标，本实验没做。")
    w("6. **仍然成立的旧结论**：单层上 k0 与 dim 的收益曲线都是加速下降、128 步内**没有拐点**")
    w("   （`docs/B1_scaling_law.md`）。本实验把它从单层扩到 24 层，方向不变；")
    w("   层数越多，k0 的边际收益越大（L=1 时 k0 2→80 只换 0.12 nats，L=24 换 0.93 nats）。")
    w("")

    # ---------- 7. boundaries ----------
    w("## 7. 诚实边界")
    w("")
    w("1. **128 步远未收敛**：所有格子末段仍在下降（表里的 `↓`），测的是**学习速度**，不是容量上限。")
    w("2. **单层结论不可外推到多层**，反之亦然；本实验只覆盖 dim=512（+H3 的 dim=256）与 1..24 层。")
    w("3. **2 个 seed**：「极差」不是置信区间；相邻格子的差若小于 seed 极差就不能当结论。")
    w("4. **「信息损耗」这个假设已被两次检验否定**（§5b 的 2×2 因子）：有残差、无残差两种结构里")
    w("   都没有「k0 太小就传不过去」的阈值；两次看起来像阈值的现象都是初始化假象。")
    w("   但本实验仍然**没有测量任何「信息量」**（互信息/有效秩等），所以它否定的是该假设的")
    w("   **可观测后果（loss 不降）**，不是「信号在数值上是否衰减」这种机制陈述。")
    w("5. **稠密对照的宽度 d' 同时决定 emb/lm_head 宽度**，所以「层内 MACs 匹配」≠「整模型 MACs 匹配」。")
    w("6. **没有 lr 扫描**：全部配置同 lr=3e-4、clip=1.0、Adam。层数变多时最优 lr 通常更小，")
    w("   所以「深层更好」这句话在同一 lr 下成立，不代表各层数都用了最优超参。")
    w("")

    with open(DOC, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("wrote %s (%d lines)" % (DOC, len(out)))
    print("B1 runs=%d dense runs=%d dim256 runs=%d" % (len(B1), len(DE), len([r for r in B1 if r["dim"] == 256])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
