#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P8-C2：跑完 hidden × layers 全矩阵，收集误差/指令数/体积/耗时。

直接调用已汇编好的 examples/eserNN/gen_m_h{H}_l{L}，把驱动输出里的
综合误差、装载耗时、每 token 耗时抠出来，汇总成 Markdown 表。

用法: python tools/eserNN_matrix.py
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

ROOT = str(Path(__file__).resolve().parents[1])
DRV = os.path.join(ROOT, "target", "release", "vmlm-eserNN.exe")
EX = os.path.join(ROOT, "examples", "eserNN")

HIDDENS = [64, 128, 256, 512]
LAYERS = [1, 2, 3, 6]


def run(tag):
    d = os.path.join(EX, f"gen_{tag}")
    vmb = os.path.join(d, "step.vmb")
    if not os.path.exists(vmb):
        return None
    r = subprocess.run([DRV, d], capture_output=True, text=False)
    # 驱动打印的是 UTF-8（含中文与 ⇒），必须显式按 UTF-8 解码；
    # 用 text=True 会走系统 GBK，正则全部失配（我踩过：误差列全是 nan）。
    out = r.stdout.decode("utf-8", "replace")
    res = {"tag": tag, "vmb_mb": os.path.getsize(vmb) / 1048576.0,
           "exit": r.returncode}
    m = re.search(r"OK:\s+(\d+)\s+", "")  # placeholder
    m = re.search(r"综合\s+([\d.eE+-]+)", out)
    res["err"] = float(m.group(1)) if m else float("nan")
    # 主判据那一段（第一处综合=真实 ESERNN，第二处=引擎 eps 对照）
    ms = re.findall(r"综合\s+([\d.eE+-]+)", out)
    if len(ms) >= 2:
        res["err"], res["err_eps5"] = float(ms[0]), float(ms[1])
    m = re.search(r"Vm::load = ([\d.]+)", out)
    res["load_ms"] = float(m.group(1)) if m else float("nan")
    m = re.search(r"([\d.]+) ms/token", out)
    res["tok_ms"] = float(m.group(1)) if m else float("nan")
    m = re.search(r"配置: vocab=(\d+) hidden=(\d+) heads=(\d+) head_dim=(\d+) layers=(\d+) steps=(\d+)", out)
    if m:
        res["cfg"] = tuple(int(x) for x in m.groups())
    res["verdict"] = "PASS" if r.returncode == 0 else "FAIL"
    # 指令数 / 槽位数：从 _slots.txt 的槽位数推不出指令数，
    # 所以直接数 .vml 里的指令行（非空、非注释、非 .section、非标签）
    vml = os.path.join(d, "step.vml")
    if os.path.exists(vml):
        n = 0
        for raw in open(vml, encoding="utf-8"):
            s = raw.split(";")[0].strip()
            if s and not s.startswith(".") and not s.endswith(":"):
                n += 1
        res["n_ins"] = n
    sl = os.path.join(d, "_slots.txt")
    if os.path.exists(sl):
        txt = open(sl, "rb").read().decode("utf-8", "replace")
        res["n_slots"] = sum(1 for ln in txt.splitlines()
                             if re.match(r"\s*slot\d+", ln))
    return res


def main():
    rows = []
    for h in HIDDENS:
        for l in LAYERS:
            tag = f"m_h{h}_l{l}"
            r = run(tag)
            if r:
                rows.append(r)
            else:
                print(f"  [skip] {tag} 未生成", file=sys.stderr)

    print(f"{'hidden':>7} {'layers':>6} {'指令数':>7} {'槽位':>6} {'vmb(MB)':>9} "
          f"{'装载(ms)':>9} {'ms/token':>9} {'误差(vs真实)':>13} {'判定':>6}")
    print("-" * 92)
    for r in rows:
        cfg = r.get("cfg")
        h, l = (cfg[1], cfg[4]) if cfg else ("?", "?")
        print(f"{h:>7} {l:>6} {r.get('n_ins', '?'):>7} {r.get('n_slots', '?'):>6} "
              f"{r['vmb_mb']:>9.2f} {r['load_ms']:>9.2f} {r['tok_ms']:>9.2f} "
              f"{r['err']:>13.3e} {r['verdict']:>6}")

    worst = max((r["err"] for r in rows if r["verdict"] == "PASS"), default=float("nan"))
    n_pass = sum(1 for r in rows if r["verdict"] == "PASS")
    print(f"\n{n_pass}/{len(rows)} PASS；最大误差 = {worst:.3e}")


if __name__ == "__main__":
    main()
