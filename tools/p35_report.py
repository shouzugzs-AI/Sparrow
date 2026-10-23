#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P35 report: dynamic-mask arms vs dense arms, same token budget.

Reads artifacts/p35/*.jsonl (written by tools/p35_dyn.py) and prints:
  1. per-arm cfg + final val loss
  2. val loss at the common token budget (all arms share steps/batch/seq)
  3. loss per axis: per token, per wall-clock second, per effective MAC
  4. mask diagnostics (union size, used fraction) -- the number that decides
     whether a batched fast implementation is even possible

USAGE
    python tools/p35_report.py
    python tools/p35_report.py --dir artifacts\\p35 --steps 488
"""
import argparse
import glob
import json
import os
import sys

OUT = r"D:\czsai\vmlm\artifacts\p35"


def load(path):
    cfg, evals, final = None, [], None
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r["kind"] == "cfg":
            cfg = r
        elif r["kind"] == "eval":
            evals.append(r)
        elif r["kind"] == "final":
            final = r
    return cfg, evals, final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=OUT)
    ap.add_argument("--steps", type=int, default=0)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.dir, "*.jsonl")))
    rows = []
    for p in files:
        cfg, evals, final = load(p)
        if cfg is None:
            continue
        tag = cfg["tag"]
        step = args.steps or cfg["steps"]
        e = [x for x in evals if x["step"] == step] or (evals[-1:] or [None])
        e = e[0] if e else None
        rows.append(dict(tag=tag, cfg=cfg, e=e, final=final))
    if not rows:
        print("no data in %s" % args.dir)
        return 1

    print("## 1. arms\n")
    print("| arm | src | d | L | params(M) | MAC/tok dense | MAC/tok eff | speedup |"
          " val@%d | tok/s | wall(min) |" % (args.steps or "last"))
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        c = r["cfg"]
        e = r["e"] or {}
        f = r["final"] or {}
        v = e.get("val_loss", f.get("final_val", float("nan")))
        print("| %s | %s | %d | %d | %.3f | %.3fM | %.3fM | %.2fx | %.4f | %.1f | %.1f |"
              % (r["tag"], c.get("mask_src", "?"), c["d"], c["layers"],
                 c["params"] / 1e6, c["macs_dense"] / 1e6, c["macs_eff"] / 1e6,
                 c["macs_dense"] / c["macs_eff"], v,
                 e.get("tok_per_s", f.get("tok_per_s", float("nan"))),
                 f.get("wall_s", float("nan")) / 60))

    print("\n## 2. loss vs budget axis (final eval)\n")
    print("| arm | val loss | tokens | MACs(fwd) | wall s |")
    print("|---|---|---|---|---|")
    for r in rows:
        c = r["cfg"]
        e = r["e"] or {}
        f = r["final"] or {}
        step = e.get("step", 0)
        toks = step * c["batch"] * c["seq"]
        print("| %s | %.4f | %.3fM | %.2fG | %.0f |"
              % (r["tag"], e.get("val_loss", float("nan")), toks / 1e6,
                 toks * c["macs_eff"] / 1e9, e.get("wall_s", float("nan"))))

    print("\n## 3. mask / neuron diagnostics (last eval)\n")
    print("| arm | keep | 稀疏熵 H_mask | 死神经元率 h | FFN中间 H_mid | 死神经元率 mid |"
          " 数值熵 H_val(h) | 数值熵 H_val(mid) | 并集 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        c = r["cfg"]
        e = r["e"] or {}
        m = e.get("mask") or {}
        if not m:
            print("| %s | %.3f | - | - | - | - | - | - | - |" % (r["tag"], c.get("keep", 1.0)))
            continue
        print("| %s | %.3f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.3f |"
              % (r["tag"], c.get("keep", 1.0),
                 m.get("H_mask", 0.0), m.get("dead_h", 0.0),
                 m.get("H_mid", 0.0), m.get("dead_mid", 0.0),
                 m.get("H_val_h", 0.0), m.get("H_val_mid", 0.0),
                 m.get("union_mean", 0.0)))
    print("\n## 3b. 分化 vs 塌缩判别（last eval；随机掩码为同 keep 基线）\n")
    print("| arm | keep | n_pat | pat_H | top_share | jac_w | jac_c | jac比 |"
          " 随机jac比 | eff_dim_h | kept_h | eff_dim_mid | kept_mid |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        c = r["cfg"]
        m = (r["e"] or {}).get("mask") or {}
        if not m:
            print("| %s | %.3f | - | - | - | - | - | - | - | - | - | - | - |"
                  % (r["tag"], c.get("keep", 1.0)))
            continue
        print("| %s | %.3f | %d | %.3f | %.3f | %.3f | %.3f | %.2f | %.2f | %d | %d | %d | %d |"
              % (r["tag"], c.get("keep", 1.0), m.get("n_pat", 0), m.get("pat_H", 0),
                 m.get("top_share", 0), m.get("jac_w", 0), m.get("jac_c", 0),
                 m.get("jac_ratio", 0), m.get("rnd_jac_ratio", 0),
                 m.get("eff_dim_h", 0), m.get("kept_h", 0),
                 m.get("eff_dim_mid", 0), m.get("kept_mid", 0)))
    print("\n判别：n_pat≈4096 且 pat_H≈1 ⇒ 每 token 一套掩码（无模板）；n_pat 小且 top_share 高 ⇒"
          " 少数组合被反复使用；jac_w/jac_c ≫ 1 ⇒ 同一序列（同领域）内共享组合。"
          " eff_dim = D^H_val，即真正承载能量的维数，和 kept 对比即可看出截断是否过头。")
    print("\n定义：H_mask = -Σ p log p / log(D)（p 为全程累计激活频次占比，1=所有通道等频使用）；"
          "dead = 全程从未被选中的通道比例；"
          "H_val = 每 token |值| 分布的归一化熵（低=数值本身已经很集中）。")
    print("\n## 4. per-layer usage entropy / dead rate (last eval)\n")
    for r in rows:
        m = (r["e"] or {}).get("mask") or {}
        if m.get("H_mask_layers"):
            print("- %s  H_mask_layers=%s" % (r["tag"], m["H_mask_layers"]))
            print("  dead_h_layers=%s  H_mid_layers=%s" %
                  (m.get("dead_h_layers"), m.get("H_mid_layers")))
            print("  H_val_h_layers=%s  H_val_mid_layers=%s" %
                  (m.get("H_val_h_layers"), m.get("H_val_mid_layers")))
            print("  n_pat_layers=%s  jac_ratio_layers=%s" %
                  (m.get("n_pat_layers"), m.get("jac_ratio_layers")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
