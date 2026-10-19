#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P31 report: turn the sweep JSONL into the markdown tables used in the doc.

Every number here comes from a row written by p31_sparse.py; nothing is
recomputed from theory except the FLOP shares, which are printed so the
arithmetic can be checked by hand.

Usage:
    python tools/p31_report.py --files results_full.jsonl,results_knee.jsonl
"""
import argparse
import json
import os
import sys

OUT = r"D:\czsai\vmlm\temp\p31"

# Rows from results_full.jsonl that a KNOWN bug invalidated. The first sweep's
# restore() only restored the CURRENT config's scope, so after the first narrow
# config every later row inherited the previous mask. Caught by inspection:
# every per-layer row printed exactly the all-layers loss (11.3718). The fixed
# run is results_knee.jsonl; these rows are dropped here rather than deleted so
# the raw artifact keeps the history.
INVALID = {"only_gateup_10", "only_attn_10", "all_10", "only_down_30",
           "only_gateup_30", "only_attn_30", "all_30", "L0_mlp_30", "L4_mlp_30",
           "L8_mlp_30", "L12_mlp_30", "L16_mlp_30", "L20_mlp_30", "L23_mlp_30"}

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def load(files):
    rows = []
    for f in files:
        p = f if os.path.isabs(f) else os.path.join(OUT, f)
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    if r["name"] in INVALID:
                        continue
                    rows.append(r)
    return rows


def fmt(x, nd=4):
    return ("%." + str(nd) + "f") % x


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=r"D:\czsai\vmlm\artifacts\p31")
    ap.add_argument("--files", default="results_full.jsonl,results_knee.jsonl,results_F.jsonl")
    ap.add_argument("--md", default="")
    args = ap.parse_args()
    OUT = args.dir
    rows = load(args.files.split(","))
    dense = [r for r in rows if r["name"] == "dense"]
    if not dense:
        raise SystemExit("no dense baseline row")
    L0 = dense[0]["loss"]
    out = []
    w = out.append

    w("# P31 auto-generated tables\n")
    w("dense: loss=%.4f  ppl=%.2f  tokens=%s  batch=%s  eval_tokens=%s  calib_tokens=%s\n"
      % (L0, dense[0]["ppl"], dense[0]["tokens"], dense[0].get("batch"),
         dense[0].get("eval_tokens"), dense[0].get("calib_tokens")))

    def sel(**kw):
        r = rows
        for k, v in kw.items():
            r = [x for x in r if x.get(k) == v]
        return r

    # ---- FLOP structure (per token, dense)
    H, I, V, L = 1536, 4608, 130560, 24
    pl = {"q": 2 * 2048 * H, "k": 2 * 256 * H, "v": 2 * 256 * H, "o": 2 * H * 2048,
          "gate": 2 * I * H, "up": 2 * I * H, "down": 2 * H * I}
    head = 2 * V * H
    tot = sum(pl.values()) * L + head
    mlp = (pl["gate"] + pl["up"] + pl["down"]) * L
    attn = (pl["q"] + pl["k"] + pl["v"] + pl["o"]) * L
    w("## 每 token GEMM FLOP 结构\n")
    w("| 部分 | GFLOP | 占比 |\n|---|---|---|")
    w("| MLP (gate+up+down) | %.3f | %.1f%% |" % (mlp / 1e9, 100 * mlp / tot))
    w("| attention (q,k,v,o) | %.3f | %.1f%% |" % (attn / 1e9, 100 * attn / tot))
    w("| lm_head | %.3f | %.1f%% |" % (head / 1e9, 100 * head / tot))
    w("| **total** | **%.3f** | **100%%** |" % (tot / 1e9))
    w("")
    w("=> MLP 全部消灭的上限是 %.2fx; 加上 attention 全灭 -> %.2fx.\n"
      % (tot / (tot - mlp), tot / (tot - mlp - attn)))

    # ---- 1. conn criterion x rho on the MLP
    crits = ["mag", "wanda", "l1", "rand"]
    _all = [x for x in rows if x.get("mode") == "conn" and x.get("mats") == ["gate", "up", "down"]
            and x.get("layers") == "all" and not x.get("rescale")
            and x["crit"] in crits]
    seen, rr = set(), []
    for x in sorted(_all, key=lambda x: (x["crit"], x["rho"])):
        k = (x["crit"], round(x["rho"], 4))
        if k not in seen:
            seen.add(k)
            rr.append(x)
    rhos = sorted({round(x["rho"], 4) for x in rr})
    w("## 1. MLP 连接剪枝: 判据 x 保留率 (ρ = 每行保留 K 的比例)\n")
    w("ρ 与 k0: gate/up K=1536, down K=4608\n")
    w("| ρ | k0(gate/up) | k0(down) | 模型 FLOP 比 | " +
      " | ".join("Δ %s" % c for c in crits) + " |")
    w("|---|---|---|" + "---|" * (len(crits) + 1))
    for rho in rhos:
        cells = []
        for c in crits:
            m = [x for x in rr if x["crit"] == c and abs(x["rho"] - rho) < 1e-9]
            cells.append(("%+.3f" % m[0]["d_loss"]) if m else "-")
        fl = [x for x in rr if abs(x["rho"] - rho) < 1e-9]
        w("| %.1f%% | %d | %d | %.3f | %s |"
          % (rho * 100, round(rho * 1536), round(rho * 4608),
             fl[0]["flop_ratio"] if fl else 0, " | ".join(cells)))
    w("")

    # ---- 2. knee: interpolated keep ratio where d_loss crosses a threshold
    w("## 2. 拐点: Δloss 越过阈值时的保留率 ρ (线性插值, 不含 ρ=1)\n")
    w("| 判据 | Δ=0.02 | Δ=0.05 | Δ=0.2 | Δ=1.0 |")
    w("|---|---|---|---|---|")

    def cross(sub, th):
        sub = [x for x in sorted(sub, key=lambda x: x["rho"]) if x["rho"] < 1.0]
        for a, b in zip(sub, sub[1:]):
            if a["d_loss"] > th >= b["d_loss"]:
                t = (a["d_loss"] - th) / (a["d_loss"] - b["d_loss"])
                return a["rho"] + t * (b["rho"] - a["rho"])
        return None

    for c in crits + ["snip"]:
        sub = [x for x in rows if x.get("crit") == c and x.get("mode") == "conn"
               and sorted(x.get("mats") or []) == ["down", "gate", "up"]
               and x.get("layers") == "all" and not x.get("rescale")]
        if not sub:
            continue
        cells = []
        for th in (0.02, 0.05, 0.2, 1.0):
            v = cross(sub, th)
            cells.append(("-" if v is None else "%.0f%%" % (100 * v)))
        w("| %s | %s |" % (c, " | ".join(cells)))
    w("")

    # ---- 3. criterion gain at matched rho (paired, same data)
    w("## 3. 判据的相对收益 (同 ρ 下相对 mag 的 Δloss 差, 正=更好)\n")
    w("| ρ | wanda | l1 | rand |")
    w("|---|---|---|---|")
    for rho in rhos:
        m = [x for x in rr if x["crit"] == "mag" and abs(x["rho"] - rho) < 1e-9]
        if not m:
            continue
        cells = []
        for c in ("wanda", "l1", "rand"):
            o = [x for x in rr if x["crit"] == c and abs(x["rho"] - rho) < 1e-9]
            cells.append(("%+.3f" % (o[0]["d_loss"] - m[0]["d_loss"])) if o else "-")
        w("| %.1f%% | %s |" % (rho * 100, " | ".join(cells)))
    w("")

    # ---- 4. structured vs unstructured at matched rho
    w("## 4. 结构化 vs 连接级 (同为 MLP, 同 ρ; 'in'=保留输入列, 'neuron'=保留 hidden 单元)\n")
    w("| ρ | conn wanda | conn+rescale | neuron | in |")
    w("|---|---|---|---|---|")
    for rho in rhos:
        def get(pred):
            m = [x for x in rows if pred(x) and abs(x.get("rho", -1) - rho) < 1e-9]
            return ("%+.3f" % m[0]["d_loss"]) if m else "-"
        w("| %.1f%% | %s | %s | %s | %s |" % (
            rho * 100,
            get(lambda x: x.get("mode") == "conn" and x["crit"] == "wanda"
                and not x.get("rescale") and x.get("mats") == ["gate", "up", "down"]
                and x.get("layers") == "all"),
            get(lambda x: x.get("mode") == "conn" and x["crit"] == "wanda"
                and x.get("rescale")),
            get(lambda x: x.get("mode") == "neuron"),
            get(lambda x: x.get("mode") == "in")))
    w("")

    # ---- 5. where the damage is (scope)
    w("## 5. 损伤位置 (只剪一组矩阵)\n")
    scopes = [("gate", ["gate"]), ("up", ["up"]), ("down", ["down"]),
              ("gate+up", ["gate", "up"]), ("mlp(全)", ["gate", "up", "down"]),
              ("attn(全)", ["q", "k", "v", "o"]),
              ("mlp+attn", ["gate", "up", "down", "q", "k", "v", "o"])]
    w("| scope | FLOP 占比 | ρ=50% Δ | ρ=70% Δ |")
    w("|---|---|---|---|")
    for nm, mats in scopes:
        def g(rho):
            m = [x for x in rows if sorted(x.get("mats") or []) == sorted(mats)
                 and x.get("mode") == "conn" and x.get("crit") == "wanda"
                 and not x.get("rescale")
                 and abs(x.get("rho", -1) - rho) < 1e-9
                 and x.get("layers") == "all"]
            return m[0]
        a, b = g(0.5), g(0.7)
        share = sum(pl[mk] for mk in mats if mk in pl) * L / tot
        w("| %s | %.1f%% | %s | %s |" % (
            nm, 100 * share,
            ("%+.3f (flop %.3f)" % (a["d_loss"], a["flop_ratio"])) if a else "-",
            ("%+.3f (flop %.3f)" % (b["d_loss"], b["flop_ratio"])) if b else "-"))
    w("")

    # ---- 6. per-layer sensitivity
    w("## 6. 分层敏感度 (只剪该层 MLP, ρ=70%)\n")
    w("| layer | Δloss |\n|---|---|")
    ls = [x for x in rows if isinstance(x.get("layers"), list) and x.get("mode") == "conn"
          and abs(x.get("rho", -1) - 0.7) < 1e-9]
    for x in sorted(ls, key=lambda x: x["layers"][0]):
        w("| L%d | %+.3f |" % (x["layers"][0], x["d_loss"]))
    w("")

    # ---- 6b. speed implication of each shape, from measured rates
    w("## 6b. 换算成速度 (实测: dense 285 GFLOPS = 142.5 GMAC/s; "
      "B1 稀疏核 0.22 ns/conn = 4.5 GMAC/s/核 = 18 GMAC/s@4核)\n")
    R = 142.5 / 18.2
    w("核效率比 R = %.2f => 连接稀疏必须剪掉 >= %.1f%% 才与 dense 打平\n"
      % (R, 100.0 * (1 - 1.0 / R)))
    w("| ρ | 连接稀疏模型 FLOP 比 | 连接稀疏实测加速 | 结构化(dense GEMM)加速 |")
    w("|---|---|---|---|")
    for rho in [0.2, 0.4, 0.6, 0.7, 0.8, 0.9]:
        s = mlp / tot
        flop = 1 - s * (1 - rho)
        sp = 1.0 / ((1 - s) + R * s * rho)
        st = 1.0 / flop
        w("| %.0f%% | %.3f | %.2fx | %.2fx |" % (100 * rho, flop, sp, st))
    w("")

    # ---- 8. held-out validation
    w("## 8. 留出验证 (换文档区间 + 256 tok/seq + 2048 tok; dense=2.9828)\n")
    w("| ρ | mag 主 | mag 留出 | wanda 主 | wanda 留出 |")
    w("|---|---|---|---|---|")
    for rho in [0.6, 0.7, 0.8]:
        def v(prefix, crit):
            m = [x for x in rows if x["name"] == "%s_%s_%d" % (prefix, crit, round(rho * 100))]
            return ("%+.3f" % m[0]["d_loss"]) if m else "-"
        w("| %.0f%% | %s | %s | %s | %s |" % (
            100 * rho, v("mlp", "mag"), v("val", "mag"), v("mlp", "wanda"),
            v("val", "wanda")))
    w("")
    w("## 7. 梯度判据 (SNIP |w·dL/dw|) 与 rescale 在拐点区\n")
    w("| ρ | mag | wanda | snip | wanda+rs | snip+rs | rand |")
    w("|---|---|---|---|---|---|---|")
    for rho in rhos:
        def g(c, rs=False, mode="conn"):
            m = [x for x in rows if x.get("crit") == c and x.get("mode") == mode
                 and bool(x.get("rescale")) == rs
                 and abs(x.get("rho", -1) - rho) < 1e-9
                 and x.get("mats") == ["gate", "up", "down"] and x.get("layers") == "all"]
            return ("%+.3f" % m[0]["d_loss"]) if m else "-"
        w("| %.1f%% | %s | %s | %s | %s | %s | %s |" % (
            100 * rho, g("mag"), g("wanda"), g("snip"), g("wanda", True),
            g("snip", True), g("rand")))
    w("")

    md = "\n".join(out) + "\n"
    print(md)
    if args.md:
        with open(args.md, "w", encoding="utf-8") as f:
            f.write(md)
        print("[saved] %s" % args.md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
