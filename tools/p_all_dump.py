# -*- coding: utf-8 -*-
"""汇总 2026-10-21 21:30 之后跑出来的全部数据 -> docs/实验数据总览_2026-10-22.md"""
import glob
import json
import os

P35 = r"D:\czsai\vmlm\artifacts\p35"
P36 = r"D:\czsai\vmlm\artifacts\p36"
OUT = r"D:\czsai\vmlm\docs\实验数据总览_2026-10-22.md"


def load(d, tag):
    with open(os.path.join(d, tag + ".jsonl"), encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def cfg_of(rs):
    return [r for r in rs if r.get("kind") == "cfg"][0]


def ev_of(rs):
    return [r for r in rs if r.get("kind") == "eval"]


def fin_of(rs):
    return [r for r in rs if r.get("kind") == "final"]


def val(x, n=4):
    return ("%%.%df" % n) % x if isinstance(x, float) else str(x)


def g(m, k, n=3):
    if not m or k not in m or m[k] is None:
        return "—"
    v = m[k]
    return ("%%.%df" % n) % v if isinstance(v, float) else str(v)


L = []
W = L.append

W("# 实验数据总览（2026-10-22，收上次总览之后新增的全部运行）")
W("")
W("覆盖：C 系列长程（C1/C5/C15，1000 步）、稠密 1000 步基线（d=64/80/112/256）、")
W("跨域臂（C1x/C5x/C15x，700 步）、Quill 架构（244 步全套 + 1000 步长程）、")
W("灾难性遗忘实验（预训练→对话微调，150 步与 400 步两档）。")
W("")
W("协议不变：d=256, L=12, I=1024, V=8198, seq=128, batch=32（4096 token/步），")
W("Adam, 裁剪 1.0，warmup 30 + cosine。掩码 = 每 token 对 |RMSNorm(x)| 取 top-k。")
W("MAC 为解析乘加计数。速度数据不参与结论。")
W("")

# ---------- 1. 长程 C 系列 ----------
W("## 1. C 系列长程（1000 步 = 4.1M token）")
W("")
W("| 代号 | keep | 参数 | MAC/token | val@1000 | 选择熵 | 死通道率 | 有效活跃集 | "
  "主掩码 | 模式数 | 数值差异熵 | 等效维数/保留 | 并集覆盖 |")
W("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
rows1 = []
for tag in ["L_h01", "L_h05", "L_h15"]:
    rs = load(P35, tag)
    c, ev, fin = cfg_of(rs), ev_of(rs), fin_of(rs)
    m = ev[-1].get("mask") or {}
    rows1.append((tag, c, ev, fin, m))
    W("| C%d | %.0f%% | %.2fM | %.3fM | %.4f | %s | %s | %s | %s | %s | %s | %s/%s | %s |"
      % (round(c["keep"] * 100), c["keep"] * 100, c["params"] / 1e6, c["macs_eff"] / 1e6,
         fin[-1]["final_val"] if fin else ev[-1]["val_loss"],
         g(m, "H_mask_eval"), g(m, "dead_eval"), g(m, "cover90_eval", 1),
         g(m, "top_share"), g(m, "n_pat", 0), g(m, "H_val_h"),
         g(m, "eff_dim_h", 0), m.get("kept_h", "—"), g(m, "union_mean")))
W("")

W("### 1b. 三条长程轨迹（每 100 步）")
W("")
for tag, c, ev, fin, m in rows1:
    W("**C%d（keep=%.0f%%，MAC=%.3fM）**" % (round(c["keep"] * 100), c["keep"] * 100,
                                            c["macs_eff"] / 1e6))
    W("")
    W("| 步 | 训练loss | 验证loss | 选择熵 | 死通道率 | 半死率 | 有效活跃集 | 主掩码 | 模式数 | "
      "数值差异熵 | 等效维数 |")
    W("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in ev:
        mm = r.get("mask") or {}
        W("| %d | %.3f | %.4f | %s | %s | %s | %s | %s | %s | %s | %s |"
          % (r["step"], r["train_loss"], r["val_loss"], g(mm, "H_mask_eval"),
             g(mm, "dead_eval"), g(mm, "low_eval"), g(mm, "cover90_eval", 1),
             g(mm, "top_share"), g(mm, "n_pat", 0), g(mm, "H_val_h"),
             g(mm, "eff_dim_h", 0)))
    W("")

W("### 1c. 逐层明细（1000 步时的最后一次 eval）")
W("")
for tag, c, ev, fin, m in rows1:
    if "H_mask_eval_layers" not in m:
        continue
    W("**C%d**" % round(c["keep"] * 100))
    W("")
    W("| 层 | 选择熵 | 死通道率 | 有效活跃集 | 主掩码 | 模式数 | 数值差异熵 |")
    W("|---|---|---|---|---|---|---|")
    for i in range(len(m["H_mask_eval_layers"])):
        W("| %d | %.3f | %.3f | %d | %.3f | %d | %.3f |"
          % (i + 1, m["H_mask_eval_layers"][i], m["dead_eval_layers"][i],
             m["cover90_eval_layers"][i], m["top_share_layers"][i],
             m["n_pat_layers"][i], m["H_val_h_layers"][i]))
    W("")

# ---------- 2. 稠密基线 ----------
W("## 2. 稠密 1000 步基线")
W("")
W("| 臂 | d | 参数 | MAC/token | val@1000 |")
W("|---|---|---|---|---|")
dense1000 = []
for tag in ["D64-t1000", "D80-t1000", "D112-t1000", "D256-t1000"]:
    rs = load(P35, tag)
    c, ev, fin = cfg_of(rs), ev_of(rs), fin_of(rs)
    v = fin[-1]["final_val"] if fin else ev[-1]["val_loss"]
    dense1000.append((c["macs_eff"] / 1e6, v))
    W("| %s | %d | %.2fM | %.3fM | %.4f |"
      % (tag, c["d"], c["params"] / 1e6, c["macs_eff"] / 1e6, v))
W("")
W("244 步稠密曲线：d=64 1.409M→6.9599, d=80 2.008M→6.8199, d=92 2.521M→6.7603, "
  "d=112 3.499M→6.7194, d=128 4.392M→6.6979, d=152 5.916M→6.6754, d=176 7.661M→6.6861。")
W("")


def iso_dense(mac):
    pts = sorted(dense1000)
    if mac <= pts[0][0]:
        return pts[0][1]
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        if x0 <= mac <= x1:
            return y0 + (y1 - y0) * (mac - x0) / (x1 - x0)
    x0, y0 = pts[-2]
    x1, y1 = pts[-1]
    return y1 + (y1 - y0) * (mac - x1) / (x1 - x0)


def inv_iso(v):
    """达到损失 v 所需的稠密 MAC（在 1000 步稠密曲线上反插值）。"""
    pts = sorted(dense1000)
    if v > pts[0][1]:
        return None
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        if y1 <= v <= y0:
            return x0 + (x1 - x0) * (y0 - v) / max(1e-9, y0 - y1)
    x0, y0 = pts[-2]
    x1, y1 = pts[-1]
    return x1 + (x1 - x0) * (y1 - v) / max(1e-9, y1 - y0)


W("### 2b. 等 FLOP 判决（1000 步，稠密曲线插值）")
W("")
W("| 臂 | MAC/token | val | 等FLOP稠密损失 | Δ | 达同损失的稠密所需 MAC | 倍数 |")
W("|---|---|---|---|---|---|---|")
for tag, c, ev, fin, m in rows1:
    v = fin[-1]["final_val"] if fin else ev[-1]["val_loss"]
    mac = c["macs_eff"] / 1e6
    ref = iso_dense(mac)
    need = inv_iso(v)
    W("| C%d | %.3fM | %.4f | %.4f | %+.3f | %s | %s |"
      % (round(c["keep"] * 100), mac, v, ref, v - ref,
         ("%.2fM" % need) if need else "—（优于所有稠密基线）",
         ("%.2f×" % (need / mac)) if need else "—"))
W("")
W("等参数对照：稠密 d=256（14.69M 参数、15.075M MAC）1000 步 **5.4299**；"
  "C15 同参数只到 5.7691。→ **C 系列是拿参数换算力，不是同参数更强。**")
W("")

# ---------- 3. 跨域 ----------
W("## 3. 跨域臂 C*x（wiki + fox_sft 对话混合训练，700 步）")
W("")
for tag in ["x_mix01", "x_mix05", "x_mix15"]:
    rs = load(P35, tag)
    c, ev = cfg_of(rs), ev_of(rs)
    W("### C%d x（keep=%.0f%%，MAC=%.3fM）" % (round(c["keep"] * 100), c["keep"] * 100,
                                              c["macs_eff"] / 1e6))
    W("")
    W("| 步 | valA(wiki) | valB(对话) | 死A | 死B | 仅A | 仅B | 共用 | 都没用 | 跨域重叠 | "
      "核心A | 核心B | 核心仅A | 核心仅B | 核心重叠 |")
    W("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in ev:
        d = r.get("mask_dom") or {}
        if not d:
            continue
        W("| %d | %.4f | %.4f | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
          % (r["step"], r["val_loss"], r.get("val_loss_b", 0),
             g(d, "死_A"), g(d, "死_B"), g(d, "仅A"), g(d, "仅B"), g(d, "共用"),
             g(d, "都没用"), g(d, "跨域重叠"), g(d, "核心A", 1), g(d, "核心B", 1),
             g(d, "核心仅A", 1), g(d, "核心仅B", 1), g(d, "核心跨域重叠", 2)))
    W("")

# ---------- 4. Quill ----------
W("## 4. Quill 架构")
W("")
W("### 4a. 244 步（seq=128 / seq=512，4096 token/步）")
W("")
W("| 臂 | 层 | 参数 | MAC/token | val |")
W("|---|---|---|---|---|")
for tag in ["b_a12_s128", "b_tra_s128", "b_a12_s512", "b_tra_s512",
            "c_L24_once", "c_L24_route", "c_L24_tra", "c_L24_route_s512"]:
    rs = load(P36, tag)
    c, ev, fin = cfg_of(rs), ev_of(rs), fin_of(rs)
    v = fin[-1]["final_val"] if fin else ev[-1]["val_loss"]
    W("| %s | %d | %.2fM | %.3fM | %.4f |"
      % (tag, c["layers"], c["params"] / 1e6, c["macs_eff"] / 1e6, v))
W("")
W("### 4b. 1000 步长程")
W("")
for tag in ["y_a12_1000", "y_tra_1000"]:
    rs = load(P36, tag)
    c, ev, fin = cfg_of(rs), ev_of(rs), fin_of(rs)
    W("**%s（%s，MAC=%.3fM，参数 %.2fM）**" % (
        tag, "一个 attn + 12 MLP" if c["attn_every"] > 1 else "每层 attn", 
        c["macs_eff"] / 1e6, c["params"] / 1e6))
    W("")
    W("| 步 | val |")
    W("|---|---|")
    for r in ev:
        W("| %d | %.4f |" % (r["step"], r["val_loss"]))
    W("")
W("Quill-1 5.4552 vs Quill-T 5.5279：差 0.073、少 21% MAC。但 p36 的 Quill-T 与 p35 稠密 "
  "d=256（同一架构）差 0.098 = 初始化噪声地板，故只能写“持平但更省”。")
W("")

# ---------- 5. 遗忘 ----------
W("## 5. 灾难性遗忘实验")
W("")
W("设置：wiki 预训练 400 步（C15b 为 700 步）→ fox_sft 对话微调（lr 1e-4）→ 评 wiki 保留。")
W("")
W("| 臂 | 预训练步 | 起点 wiki | 起点 fox | 微调步 | fox | wiki | Δwiki(遗忘) | Δfox(学习) |")
W("|---|---|---|---|---|---|---|---|---|")
meta = [("dense", "pre_dense", "t0_dense", "ft_dense", "ftl_dense", 400),
        ("C5", "pre_c05", "t0_c05", "ft_c05", "ftl_c05", 400),
        ("C15", "pre_c15", "t0_c15", "ft_c15", None, 400),
        ("C15b", "pre_c15b", "t0_c15b", "ft_c15b", "ftl_c15b", 700)]
for name, pre, t0, ft, ftl, psteps in meta:
    pr = load(P35, pre)
    pv = fin_of(pr)[-1]["final_val"]
    rs = load(P35, t0)
    t0e = ev_of(rs)[-1]
    for tag, steps in [(ft, 150), (ftl, 400)]:
        if not tag:
            continue
        rr = load(P35, tag)
        e2 = ev_of(rr)[-1]
        f2 = fin_of(rr)[-1]
        W("| %s | %d | %.4f | %.4f | %d | %.4f | %.4f | %+.4f | %+.4f |"
          % (name, psteps, pv, t0e["val_loss"], steps, e2["val_loss"],
             e2.get("val_ret", 0), e2.get("val_ret", 0) - pv,
             e2["val_loss"] - t0e["val_loss"]))
W("")
W("等计算对照（微调阶段）：稠密 100 步（15.075M×100）≈ C15b 400 步（3.264M×400）；")
W("稠密@100 fox 6.3388 / wiki 6.8076  vs  C15b@400 fox 6.1975 / wiki 6.7827。")
W("")

W("## 6. 数据完整性 / 必须挂的偏差")
W("")
W("1. 全部运行 seed=1，除 A_dense256_s2 / C_h_s2 两个 seed2 对照；0.1 以内差值不可信，")
W("   实现间噪声地板实测 ≈0.098（p36 Quill-T vs p35 稠密 d=256，同一架构）。")
W("2. 244 步 = 1.05M token 明显欠训练，所有 244 步结论已在长程中修正过一轮。")
W("3. C 系列是“参数换算力”：同参数下稠密更强（5.43 vs 5.77），同算力下 C 系列更强。")
W("4. 遗忘实验里“学得少”受每步计算量影响（稠密 4.6 倍），等计算比较才公平。")
W("5. 旧累积口径 dead_h 恒为 0，无判别力；分化指标只用验证集口径。")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")
print("wrote", OUT, len(L), "lines")
