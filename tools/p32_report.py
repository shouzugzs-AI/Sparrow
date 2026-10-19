#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P32 report: B1 sparse vs real dense at matched MACs / tokens / wall-clock.

Reads the three arms' logs and prints the comparisons that decide the
"cost-performance" question:

  - loss at equal TOKENS        (same data budget)
  - loss at equal WALL-CLOCK    (same money -> who wins?)
  - loss at equal cumulative MACs (is the architecture MAC-efficient?)
  - measured throughput and GMAC/s per arm

Usage: python tools/p32_report.py
"""
import json
import os
import sys

P32 = r"D:\czsai\vmlm\artifacts\p32"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ARMS = [
    dict(key="A_B1", kind="b1", path="A_b1", macs=2.621632e6, params=1.442e6,
         note="B1 sparse dim4096/k8/L32 sparse-IO no-residual"),
    dict(key="B_dense64", kind="dense", path="B_d64_L32.jsonl",
         macs=2.753e6, params=2.626e6, note="dense d=64 L=32 (equal MAC)"),
    dict(key="C_dense176", kind="dense", path="C_d176_L32.jsonl",
         macs=17.663e6, params=17.314e6, note="dense d=176 L=32 (7x MAC)"),
    dict(key="D_B1_res", kind="b1", path="D_b1_res", macs=2.621632e6, params=1.442e6,
         note="B1 sparse + residual (--no-residual 0)"),
]
BATCH_SEQ = 2048.0


def read_b1(d):
    p = os.path.join(P32, d, "metrics.jsonl")
    rows = []
    if not os.path.exists(p):
        return rows
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "val_loss" in r:
                rows.append(dict(step=r["step"], val=r["val_loss"],
                                 wall=r.get("wall_s", float("nan"))))
    return rows


def read_dense(p):
    rows = []
    p = p if os.path.isabs(p) else os.path.join(P32, p)
    if not os.path.exists(p):
        return rows
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("kind") == "eval":
                rows.append(dict(step=r["step"], val=r["val_loss"], wall=r["wall_s"]))
    return rows


def lin(rows, xkey, x, ykey):
    """Linear interpolation of rows[ykey] at x along xkey; None if outside."""
    rs = sorted(rows, key=lambda r: r[xkey])
    if not rs or x < rs[0][xkey] or x > rs[-1][xkey]:
        return None
    for a, b in zip(rs, rs[1:]):
        if a[xkey] <= x <= b[xkey]:
            if b[xkey] == a[xkey]:
                return a[ykey]
            t = (x - a[xkey]) / (b[xkey] - a[xkey])
            return a[ykey] + t * (b[ykey] - a[ykey])
    return rs[-1][ykey]


def main():
    data = {}
    for a in ARMS:
        rows = read_b1(a["path"]) if a["kind"] == "b1" else read_dense(a["path"])
        for r in rows:
            r["tok"] = r["step"] * BATCH_SEQ
            r["macs"] = r["tok"] * a["macs"]
        data[a["key"]] = rows

    print("# P32 raw curves\n")
    for a in ARMS:
        rows = data[a["key"]]
        if not rows:
            print("## %s: (no data)\n" % a["key"])
            continue
        last = rows[-1]
        print("## %s — %s\n" % (a["key"], a["note"]))
        print("MAC/token=%.2fM  params=%.2fM  final: step=%d tokens=%.2fM wall=%.0fs val=%.4f"
              % (a["macs"] / 1e6, a["params"] / 1e6, last["step"], last["tok"] / 1e6,
                 last["wall"], last["val"]))
        print("=> %.0f tok/s, %.1f GMAC/s(fwd) i.e. %.1f GMAC/s training(3 pass)\n"
              % (last["tok"] / max(last["wall"], 1e-9),
                 last["tok"] * a["macs"] / max(last["wall"], 1e-9) / 1e9,
                 3 * last["tok"] * a["macs"] / max(last["wall"], 1e-9) / 1e9))
        print("| step | tokens | wall s | val |")
        print("|---|---|---|---|")
        for r in rows:
            print("| %d | %.2fM | %.0f | %.4f |"
                  % (r["step"], r["tok"] / 1e6, r["wall"], r["val"]))
        print("")

    live = [a for a in ARMS if data[a["key"]]]
    if not live:
        return 1
    axes = [("tok", "equal tokens", 1e6, "%.2fM"),
            ("wall", "equal wall-clock", 1.0, "%.0f s"),
            ("macs", "equal cumulative MACs", 1e12, "%.2f T")]
    for ax, label, disp, fmtv in axes:
        common = min(max(r[ax] for r in data[a["key"]]) for a in live)
        print("## %s = %s\n" % (label, fmtv % (common / disp)))
        print("| arm | val | Δ vs A | tokens | MACs |")
        print("|---|---|---|---|---|")
        base = None
        for a in live:
            rows = data[a["key"]]
            v = lin(rows, ax, common, "val")
            tok = lin(rows, ax, common, "tok")
            if v is None or tok is None:
                print("| %s | out of range | - | - | - |" % a["key"])
                continue
            if a["key"] == "A_B1":
                base = v
            d = "—" if base is None or a["key"] == "A_B1" else "%+.4f" % (v - base)
            print("| %s | %.4f | %s | %.2fM | %.2fT |"
                  % (a["key"], v, d, tok / 1e6, tok * a["macs"] / 1e12))
        print("")
    return 0


if __name__ == "__main__":
    sys.exit(main())
