# -*- coding: utf-8 -*-
"""汇总 artifacts/p35 + artifacts/p36 全部运行 -> docs/实验数据总览_2026-10-23.md

与 p_all_dump.py 的区别：不写死臂列表，扫描目录里的每个 *.jsonl，
从 cfg 读超参、从 final（没有就取最后一条 eval）读 val，自动分组出表。
跑中的臂会被标成「跑中」而不是丢掉。
"""
import glob
import json
import os

ROOT = r"D:\czsai\vmlm"
P35 = os.path.join(ROOT, "artifacts", "p35")
P36 = os.path.join(ROOT, "artifacts", "p36")
OUT = os.path.join(ROOT, "docs", "实验数据总览_2026-10-23.md")

# 噪声地板：同架构 p36 Quill-T vs p35 dense d=256 差 0.098（初始化顺序/RNG 不同）
NOISE = 0.098

# 已作废的臂（跑挂了 / 参数写错，结果不可用）
DROP = {
    "t_v2b",
    "mb_L24_k25_244",  # 起跑后被环境杀，只到 step 50，不可用（均衡结论由 mb_h01/mb_h05 给出）
}


def load(path):
    rs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rs


def kind(rs, k):
    return [r for r in rs if r.get("kind") == k]


def fnum(x, n=4):
    if isinstance(x, (int, float)):
        return ("%%.%df" % n) % x
    return "—"


def mac_str(x):
    if not isinstance(x, (int, float)):
        return "—"
    return "%.2fM" % (x / 1e6)


# 不属于主线对照的臂：遗忘/微调、t=0 基线、冒烟调试
AUX_PREFIX = ("ft_", "s_fgt_", "t0_")
# Quill 系列（周期性全注意力 + 共享读），与 C 系列不是同一架构，单列
QUILL_PREFIX = ("q_", "quill", "t_a12", "t_tra", "t_L24", "t_rte")


def is_aux(a):
    return a["tag"].startswith(AUX_PREFIX) or (a["cfg"].get("steps") or 0) < 100


def is_quill(a):
    return a["suite"] == "p36" or a["tag"].startswith(QUILL_PREFIX)


def collect():
    arms = []
    for d in (P35, P36):
        for p in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
            tag = os.path.basename(p)[:-6]
            if tag in DROP:
                continue
            rs = load(p)
            cfg = kind(rs, "cfg")
            if not cfg:
                continue
            cfg = cfg[0]
            fin = kind(rs, "final")
            evs = kind(rs, "eval")
            if fin:
                val = fin[-1].get("final_val")
                state = "完成"
            elif evs:
                val = evs[-1].get("val_loss")
                state = "跑中"
            else:
                continue
            if val is None:
                continue
            arms.append(
                dict(
                    tag=tag,
                    suite=os.path.basename(d),
                    cfg=cfg,
                    val=float(val),
                    state=state,
                    n_ev=len(evs),
                    last_step=evs[-1].get("step") if evs else None,
                )
            )
    return arms


def main():
    arms = collect()
    L = []
    W = L.append

    W("# 实验数据总览（2026-10-23）")
    W("")
    W("本表由 `tools/p42_dump.py` 从 `artifacts/p35` 与 `artifacts/p36` 的 `*.jsonl` 直接生成，")
    W("不手抄数字；跑中的臂取最后一次 eval 并标「跑中」。")
    W("")
    W("协议：d=256, L=12, heads=4, I=4d, V=8198, seq=128, batch=32（4096 token/步），")
    W("Adam(0.9,0.999)、逐张量裁剪 1.0、warmup + cosine 到 1/10 峰值 lr；数据 `artifacts/zh_wiki`；")
    W("验证集为固定 4×32 批。掩码 = 每 token 对 |RMSNorm(x)| 取 top-k（h 侧），FFN 中间独立掩码。")
    W("")
    W("**噪声地板 0.098**：同架构、同协议、只换初始化顺序（p36 Quill-T vs p35 稠密 d=256）的差为 0.098，")
    W("因此**小于 0.1 的差不能当作结论**；单种子运行全部按此口径读。")
    W("")
    W("MAC 为解析乘加计数（每 token），不含选择开销；速度数据不参与任何结论。")
    W("")

    # ---- 1. 全部臂一览 ----
    aux_arms = [a for a in arms if is_aux(a)]
    quill_arms = [a for a in arms if not is_aux(a) and is_quill(a)]
    main_arms = [a for a in arms if not is_aux(a) and not is_quill(a)]
    W("## 1. 主线运行一览（步数 >= 100，不含遗忘/冒烟臂）")
    W("")
    W("| 代号 | 套件 | 步数 | keep | lr | MAC/token | 参数 | val | 状态 |")
    W("|---|---|---|---|---|---|---|---|---|")
    for a in sorted(main_arms, key=lambda x: (x["suite"], x["cfg"].get("steps", 0), x["tag"])):
        c = a["cfg"]
        keep = c.get("keep")
        keep_s = "—" if keep is None or float(keep) >= 1.0 else ("%.3g" % float(keep))
        W(
            "| `%s` | %s | %s | %s | %s | %s | %s | %s | %s |"
            % (
                a["tag"],
                a["suite"],
                c.get("steps", "—"),
                keep_s,
                ("%.1e" % c["lr"]) if c.get("lr") else "—",
                mac_str(c.get("macs_eff")),
                ("%.2fM" % (c["params"] / 1e6)) if c.get("params") else "—",
                fnum(a["val"]),
                a["state"],
            )
        )
    W("")

    # ---- 1a. Quill ----
    W("## 1a. Quill 架构（周期性全注意力 + 跨层共享读）")
    W("")
    W("与 C 系列**不是同一架构**，不能和上面混着比；这里单列。")
    W("")
    W("| 代号 | 步数 | MAC/token | 参数 | val | 状态 |")
    W("|---|---|---|---|---|---|")
    for a in sorted(quill_arms, key=lambda x: (x["cfg"].get("steps", 0), x["tag"])):
        c = a["cfg"]
        W(
            "| `%s` | %s | %s | %s | %s | %s |"
            % (
                a["tag"],
                c.get("steps", "—"),
                mac_str(c.get("macs_eff")),
                ("%.2fM" % (c["params"] / 1e6)) if c.get("params") else "—",
                fnum(a["val"]),
                a["state"],
            )
        )
    W("")

    # ---- 1b. 附录 ----
    W("## 1b. 附录：遗忘/微调、t=0 基线、冒烟臂")
    W("")
    W("`ft_*` 是灾难性遗忘实验的微调臂（val 在对话域上，**不能**与主线表横比）；")
    W("`t0_*` 是微调前 t=0 基线；步数 < 100 的是冒烟/调试臂。")
    W("")
    W("| 代号 | 套件 | 步数 | keep | lr | MAC/token | 参数 | val | 状态 |")
    W("|---|---|---|---|---|---|---|---|---|")
    for a in sorted(aux_arms, key=lambda x: (x["cfg"].get("steps", 0), x["tag"])):
        c = a["cfg"]
        keep = c.get("keep")
        keep_s = "—" if keep is None or float(keep) >= 1.0 else ("%.3g" % float(keep))
        W(
            "| `%s` | %s | %s | %s | %s | %s | %s | %s | %s |"
            % (
                a["tag"],
                a["suite"],
                c.get("steps", "—"),
                keep_s,
                ("%.1e" % c["lr"]) if c.get("lr") else "—",
                mac_str(c.get("macs_eff")),
                ("%.2fM" % (c["params"] / 1e6)) if c.get("params") else "—",
                fnum(a["val"]),
                a["state"],
            )
        )
    W("")

    # ---- 2. 每个步数档位下按 MAC 排名 ----
    W("## 2. 同 token 预算下的有效算力排名")
    W("")
    W("每个步数档位内按 MAC/token 升序排列，便于直接读「同算力谁更好」。")
    W("")
    for steps in sorted({a["cfg"].get("steps") for a in arms if a["cfg"].get("steps")}):
        grp = [
            a
            for a in main_arms
            if a["cfg"].get("steps") == steps and a["state"] == "完成"
        ]
        if len(grp) < 2:
            continue
        W("### %s 步（%.1fM token）" % (steps, steps * 4096 / 1e6))
        W("")
        W("| 代号 | MAC/token | 参数 | keep | lr | val |")
        W("|---|---|---|---|---|---|")
        for a in sorted(grp, key=lambda x: x["cfg"].get("macs_eff") or 0):
            c = a["cfg"]
            keep = c.get("keep")
            keep_s = "稠密" if keep is None or float(keep) >= 1.0 else ("%.3g" % float(keep))
            W(
                "| `%s` | %s | %s | %s | %s | %s |"
                % (
                    a["tag"],
                    mac_str(c.get("macs_eff")),
                    ("%.2fM" % (c["params"] / 1e6)) if c.get("params") else "—",
                    keep_s,
                    ("%.1e" % c["lr"]) if c.get("lr") else "—",
                    fnum(a["val"]),
                )
            )
        W("")

    # ---- 3. 稀疏 vs 稠密 等算力配对 ----
    W("## 3. 等算力配对（稀疏臂 vs MAC 最接近的稠密臂）")
    W("")
    W("对每个完成的长程稀疏臂，取同一步数、MAC 最接近的稠密臂做差；Δ<0 表示稀疏更好。")
    W("")
    W("| 稀疏臂 | MAC | val | 稠密臂 | MAC | val | Δ(稀疏−稠密) | 判读 |")
    W("|---|---|---|---|---|---|---|---|")
    pairs = 0
    for steps in sorted({a["cfg"].get("steps") for a in arms if a["cfg"].get("steps")}):
        sp = [
            a
            for a in main_arms
            if a["cfg"].get("steps") == steps
            and a["state"] == "完成"
            and a["cfg"].get("mask_src") not in (None, "none")
        ]
        de = [
            a
            for a in main_arms
            if a["cfg"].get("steps") == steps
            and a["state"] == "完成"
            and a["cfg"].get("mask_src") in (None, "none")
        ]
        for s in sp:
            if not de:
                continue
            d = min(de, key=lambda x: abs((x["cfg"]["macs_eff"] or 0) - (s["cfg"]["macs_eff"] or 0)))
            delta = s["val"] - d["val"]
            if abs(delta) < NOISE:
                verd = "打平（<噪声地板）"
            elif delta < 0:
                verd = "稀疏更好"
            else:
                verd = "稠密更好"
            pairs += 1
            W(
                "| `%s` | %s | %s | `%s` | %s | %s | %+.4f | %s |"
                % (
                    s["tag"],
                    mac_str(s["cfg"]["macs_eff"]),
                    fnum(s["val"]),
                    d["tag"],
                    mac_str(d["cfg"]["macs_eff"]),
                    fnum(d["val"]),
                    delta,
                    verd,
                )
            )
    if pairs == 0:
        W("| — | — | — | — | — | — | — | 无同档位配对 |")
    W("")

    # ---- 4. 学习率阶梯 ----
    W("## 4. 学习率阶梯（同 keep、同步数，只改 lr）")
    W("")
    W("用于读「最优 lr 随稀疏率怎么移动」。")
    W("")
    W("| 步数 | keep | lr | val |")
    W("|---|---|---|---|")
    ladder = {}
    for a in main_arms:
        c = a["cfg"]
        if a["state"] != "完成":
            continue
        keep = c.get("keep")
        if c.get("mask_src") in (None, "none"):
            key = ("稠密 d=%s" % c.get("d"), c.get("steps"))
        else:
            key = ("keep=%.3g" % float(keep), c.get("steps"))
        ladder.setdefault(key, []).append(a)
    for (name, steps) in sorted(ladder, key=lambda k: (k[1] or 0, k[0])):
        grp = sorted(ladder[(name, steps)], key=lambda x: x["cfg"].get("lr") or 0)
        if len(grp) < 2:
            continue
        best = min(grp, key=lambda x: x["val"])
        for a in grp:
            star = " **←最优**" if a is best else ""
            W("| %s | %s | %.1e | %s%s |" % (steps, name, a["cfg"]["lr"], fnum(a["val"]), star))
    W("")

    # ---- 5. 差分：稠密尺寸曲线 ----
    W("## 5. 稠密尺寸曲线（同协议，仅改 d）")
    W("")
    W("| 步数 | d | 参数 | MAC/token | lr | val |")
    W("|---|---|---|---|---|---|")
    for a in sorted(
        [
            x
            for x in main_arms
            if x["cfg"].get("mask_src") in (None, "none") and x["state"] == "完成"
        ],
        key=lambda x: (x["cfg"].get("steps") or 0, x["cfg"].get("d") or 0, x["cfg"].get("lr") or 0),
    ):
        c = a["cfg"]
        W(
            "| %s | %s | %.2fM | %s | %.1e | %s |"
            % (
                c.get("steps"),
                c.get("d"),
                c["params"] / 1e6,
                mac_str(c.get("macs_eff")),
                c["lr"],
                fnum(a["val"]),
            )
        )
    W("")

    # ---- 6. 提醒 ----
    W("## 6. 读表须知")
    W("")
    W("- 全部单种子（例外：`A_dense256_s2` / `C_h_s2`），小于 0.1 的差不可信。")
    W("- 等算力配对里稠密臂通常参数更多（稀疏臂参数按 d = 256 记），所以「同 MAC 稀疏赢」"
      "不等于「同参数稀疏赢」——后者见 P35 正文的等参数口径。")
    W("- 不同臂的 warmup 不同（30 或 60），长程臂用 60；同档位内比较时注意。")
    W("- 跑中的臂 val 还会降，不要当终值引用。")
    W("")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    txt = "\n".join(L) + "\n"
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(txt)

    print("写入 %s" % OUT)
    print("臂数 %d（完成 %d / 跑中 %d）" % (
        len(arms),
        sum(1 for a in arms if a["state"] == "完成"),
        sum(1 for a in arms if a["state"] == "跑中"),
    ))
    print("行数 %d" % (txt.count("\n")))


if __name__ == "__main__":
    main()
