# -*- coding: utf-8 -*-
"""把 artifacts/p35 下所有 P35 数据汇总成一份中文总览（markdown）。"""
import glob
import json
import os

BASE = r"D:\czsai\vmlm\artifacts\p35"
OUT = r"D:\czsai\vmlm\docs\P35_数据总览.md"


def rows(tag):
    with open(os.path.join(BASE, tag + ".jsonl"), encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def summarize(tag):
    rs = rows(tag)
    cfg = [r for r in rs if r.get("kind") == "cfg"][0]
    ev = [r for r in rs if r.get("kind") == "eval"]
    fin = [r for r in rs if r.get("kind") == "final"]
    return cfg, ev, fin[-1] if fin else (ev[-1] if ev else None)


def val_of(rec):
    if rec is None:
        return float("nan")
    return rec.get("val_loss", rec.get("final_val", float("nan")))


def g(d, k, n=3):
    """取指标，没有就返回 '—'（旧臂没测过这些指标）。"""
    if not d or k not in d or d[k] in (None, 0.0) and k not in ("dead_h",):
        if not d or k not in d:
            return "—"
    v = d.get(k)
    if v is None:
        return "—"
    return ("%%.%df" % n) % v if isinstance(v, float) else str(v)


tags = [os.path.basename(p)[:-6] for p in sorted(glob.glob(os.path.join(BASE, "*.jsonl")))]
tags = [t for t in tags if not t.startswith(("t_", "test_"))]
data = {}
for t in tags:
    try:
        data[t] = summarize(t)
    except Exception as e:
        print("skip", t, e)

L = []
W = L.append

W("# P35 动态激活稀疏：全部已有数据")
W("")
W("协议：d=256, L=12, heads=4, I=1024, V=8198, seq=128, batch=32 (=4096 token/步),")
W("Adam(0.9,0.999), 每张量梯度裁剪 1.0, warmup 30, cosine 3e-4→3e-5，数据 artifacts/zh_wiki")
W("（跨域臂加 artifacts/fox_sft），验证集固定 4×32 批。掩码 = 每 token 对 |RMSNorm(x)|")
W("取 top-k，同层 qkv/gate/up/head 复用同一掩码；FFN 中间维单独一份掩码（keep_mid）。")
W("MAC 为解析乘加计数；掩码 GEMM 用“掩码×稠密权重”精确模拟，不是近似。")
W("")

dense244 = []
for t, (cfg, ev, last) in data.items():
    if cfg.get("mask_src") == "none" and cfg.get("layers") == 12:
        dense244.append((cfg["macs_eff"] / 1e6, val_of(last), t, cfg))
dense244.sort()


def iso_dense(mac):
    pts = [(x, y) for x, y, _, _ in dense244 if _.get("steps") == 244]
    if mac <= pts[0][0]:
        return pts[0][1], "低于最小稠密点"
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        if x0 <= mac <= x1:
            return y0 + (y1 - y0) * (mac - x0) / (x1 - x0), "插值"
    x0, y0 = pts[-2]
    x1, y1 = pts[-1]
    return y1 + (y1 - y0) * (mac - x1) / (x1 - x0), "外推"


W("## 1. 稠密基线尺寸曲线（hidden=4d）")
W("")
W("| 臂 | d | 步数 | MAC/token | val |")
W("|---|---|---|---|---|")
for mac, v, t, cfg in dense244:
    W("| %s | %d | %d | %.3fM | %.4f |" % (t, cfg["d"], cfg["steps"], mac, v))
W("")

W("## 2. 掩码臂（244 步，keep=h 侧保留比例，keep_mid 缺省=keep）")
W("")
W("| 臂 | 源 | keep | keep_mid | MAC/token | val@244 | 等FLOP稠密 | Δ | 备注 |")
W("|---|---|---|---|---|---|---|---|---|")
masked = []
for t, (cfg, ev, last) in sorted(data.items(), key=lambda kv: kv[1][0].get("macs_eff", 0)):
    if cfg.get("steps") != 244 or cfg.get("mask_src") == "none" or cfg.get("layers") != 12:
        continue
    if not ev:
        continue
    mac = cfg["macs_eff"] / 1e6
    val = val_of(last)
    dref, how = iso_dense(mac)
    note = "" if how == "插值" else how
    W("| %s | %s | %.4f | %.4f | %.3fM | %.4f | %.4f | %+.3f | %s |"
      % (t, cfg.get("mask_src"), cfg.get("keep", 1), cfg.get("keep_mid", 1),
         mac, val, dref, val - dref, note))
    masked.append((t, cfg, val, mac, dref))
W("")

W("## 3. 488 步臂（完整跑完的）")
W("")
W("| 臂 | 源 | keep | MAC/token | val@488 | 备注 |")
W("|---|---|---|---|---|---|")
for t, (cfg, ev, last) in sorted(data.items(), key=lambda kv: kv[1][0].get("macs_eff", 0)):
    if cfg.get("steps") != 488 or cfg.get("layers") != 12:
        continue
    mac = cfg["macs_eff"] / 1e6
    W("| %s | %s | %.4f | %.3fM | %.4f | %s |"
      % (t, cfg.get("mask_src"), cfg.get("keep", 1), mac, val_of(last),
         "seed2" if cfg.get("seed") == 2 else ""))
W("")
W("这组里 `A_dense256`/`B_dense192` 的 val 分别是 6.0282 / 6.1217。")
W("")

W("## 3b. 其它步数 / 未跑完的臂")
W("")
W("| 臂 | 源 | keep | 目标步数 | 最后记录步 | MAC/token | val |")
W("|---|---|---|---|---|---|---|")
for t, (cfg, ev, last) in sorted(data.items(), key=lambda kv: kv[1][0].get("mask_src", "")):
    if cfg.get("layers") != 12 or cfg.get("mask_src") == "none":
        continue
    if cfg.get("steps") in (244, 488):
        continue
    if not ev:
        continue
    W("| %s | %s | %.4f | %d | %d | %.3fM | %.4f |"
      % (t, cfg.get("mask_src"), cfg.get("keep", 1), cfg.get("steps", 0),
         ev[-1]["step"], cfg["macs_eff"] / 1e6, val_of(last)))
W("")

W("## 4. 掩码分化指标（各臂最后一次 eval）")
W("")
W("v2 口径（验证集上统计，可信）：`L_*`、`x_*`；旧口径（全程累积）：其余臂。")
W("`—` = 该臂当时还没实现这个指标（不是 0）。")
W("")
W("| 臂 | 步 | keep | 选择熵 | 死通道率 | 半死率 | 有效活跃集 | 主掩码占比 | 模式数 | "
  "数值差异熵 | 等效维数/保留 | 并集覆盖 |")
W("|---|---|---|---|---|---|---|---|---|---|---|---|")
for t, (cfg, ev, last) in sorted(data.items(), key=lambda kv: kv[1][0].get("macs_eff", 0)):
    if cfg.get("mask_src") == "none" or not ev or cfg.get("layers") != 12:
        continue
    m = ev[-1].get("mask") or {}
    if not m:
        continue
    new = "H_mask_eval" in m
    H = m.get("H_mask_eval", m.get("H_mask"))
    dead = m.get("dead_eval", m.get("dead_h"))
    low = m.get("low_eval")
    cov = m.get("cover90_eval")
    W("| %s | %d | %.4f | %s | %s | %s | %s | %s | %s | %s | %s/%s | %s |"
      % (t, ev[-1]["step"], cfg.get("keep", 1), g(m, "H_mask_eval" if new else "H_mask"),
         g(m, "dead_eval" if new else "dead_h"), g(m, "low_eval") if new else "—",
         g(m, "cover90_eval", 1) if new else "—", g(m, "top_share"),
         g(m, "n_pat", 0), g(m, "H_val_h"),
         g(m, "eff_dim_h", 0), m.get("kept_h", "—"), g(m, "union_mean")))
W("")
W("旧口径的死通道率恒为 0（全程累积，3M 次选择早就摸遍 256 个通道），")
W("只有验证集口径（v2）能反映“这个数据上其实只用了一小撮通道”。")
W("")

W("## 5. 长程臂轨迹（1000 步 = 4.1M token，v2 指标）")
W("")
for t in ["L_h01", "L_h05", "L_h15"]:
    if t not in data:
        continue
    cfg, ev, last = data[t]
    W("### %s（keep=%.2f, MAC=%.3fM）" % (t, cfg["keep"], cfg["macs_eff"] / 1e6))
    W("")
    W("| 步 | 训练loss | 验证loss | 选择熵 | 死通道率 | 半死率 | 有效活跃集 | 主掩码 | 模式数 | "
      "数值差异熵 | 等效维数 |")
    W("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in ev:
        m = r.get("mask") or {}
        W("| %d | %.3f | %.4f | %s | %s | %s | %s | %s | %s | %s | %s |"
          % (r["step"], r["train_loss"], r["val_loss"], g(m, "H_mask_eval"),
             g(m, "dead_eval"), g(m, "low_eval"), g(m, "cover90_eval", 1),
             g(m, "top_share"), g(m, "n_pat", 0), g(m, "H_val_h"),
             g(m, "eff_dim_h", 0)))
    W("")

W("## 6. 跨领域臂（wiki + fox_sft 对话混合训练，交替取批）")
W("")
for t in ["x_mix01", "x_mix05", "x_mix15"]:
    if t not in data:
        continue
    cfg, ev, last = data[t]
    W("### %s（keep=%.2f）" % (t, cfg["keep"]))
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

W("## 7. 逐层明细（v2 口径）")
W("")
for t in ["L_h01", "L_h05", "L_h15"]:
    if t not in data:
        continue
    cfg, ev, last = data[t]
    if not ev:
        continue
    m = ev[-1].get("mask") or {}
    if "H_mask_eval_layers" not in m:
        continue
    W("### %s（第 %d 步, keep=%.2f）" % (t, ev[-1]["step"], cfg["keep"]))
    W("")
    W("| 层 | 选择熵 | 死通道率 | 有效活跃集 | 主掩码 | 模式数 | 数值差异熵 |")
    W("|---|---|---|---|---|---|---|")
    for i in range(len(m["H_mask_eval_layers"])):
        W("| %d | %.3f | %.3f | %d | %.3f | %d | %.3f |"
          % (i + 1, m["H_mask_eval_layers"][i], m["dead_eval_layers"][i],
             m["cover90_eval_layers"][i], m["top_share_layers"][i],
             m["n_pat_layers"][i], m["H_val_h_layers"][i]))
    W("")

W("## 8. 已知偏差与注意事项")
W("")
W("1. 244 步 = 1.05M token，明显欠训练；同配置复跑差 0.04（t_v2b 6.8653 vs e_h05 6.9033，")
W("   差异来自调度总步数 250 vs 244，不是噪声结论）。0.05 以内差值一律未验证。")
W("2. 高算力端（>7.66M MAC）的等 FLOP 稠密值由最后两点外推，而稠密曲线在 d=152 后回升")
W("   （6.6754 → 6.6861），说明大稠密模型欠训练，外推会系统性高估稀疏优势。")
W("   可信区间：keep 15–40%、MAC 2–6.7M。")
W("3. 旧口径 `dead_h`（全程累积）恒为 0，不能用来判断塌缩；v2 加了验证集口径。")
W("4. 全部臂只跑 seed=1（seed=2 只做过 A_dense256_s2 6.6206 / C_h_s2 6.7026 两个对照）。")
W("5. 速度数据不参与任何结论（机器有外部负载）。")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")
print("wrote", OUT, len(L), "lines")
