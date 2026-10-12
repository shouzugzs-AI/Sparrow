#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P10 B2: run the assembled MiniCPM layer-0 .vmb on the REAL engine and diff
every stage against the HuggingFace capture.

The engine's default build reads a fixed `model.vmb` (no argv, by design --
argv lives in the `bench` feature which is excluded from the 48 KB budget).
So we copy the .vmb next to the exe as model.vmb and run it there.

Stage order of the PRINT dumps in layer0.vml:
    xn q k v att_out oproj xn2 gate acts down h_out
"""

import json
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = str(Path(__file__).resolve().parents[1])
REF = os.path.join(ROOT, "examples", "minicpm", "layer0_ref")
OUT = os.path.join(ROOT, "examples", "minicpm")
ENG = os.path.join(ROOT, "target", "release", "vmlm-runtime.exe")
REL = os.path.join(ROOT, "target", "release")

# dump order must match the PRINT sequence in tools/minicpm_layer0_vml.py
ORDER = ["xn", "q", "k", "v", "att_out", "oproj", "xn2", "gate", "acts", "down", "h_out"]
# matching HF reference arrays
REFMAP = {
    "xn": "in_ln_out",
    "q": "q_proj_out",
    "k": "k_proj_out",
    "v": "v_proj_out",
    "att_out": "attn_ctx_cat",
    "oproj": "o_proj_out",
    "xn2": "post_ln_out",
    "gate": "gate_proj_out",
    "acts": "mlp_act",
    "down": "down_proj_out",
    "h_out": "hidden_out",
}


def L(name):
    return np.load(os.path.join(REF, name + ".npy")).astype(np.float32)


def parse_prints(text, meta):
    """
    The engine's PRINT emits one ROW per line, values space-separated
    (see interp.rs print_f32). So parse every whitespace-separated float
    across the whole stdout, in order.
    """
    vals = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        for tok in ln.split():
            try:
                vals.append(float(tok))
            except ValueError:
                pass
    return np.array(vals, dtype=np.float32)


def main():
    vmb_src = os.path.join(OUT, "layer0.vmb")
    if not os.path.exists(vmb_src):
        print("missing %s -- run tools/minicpm_layer0_vml.py first" % vmb_src)
        return 1

    # The default engine build has model.vmb hardcoded (no argv by design).
    dst = os.path.join(REL, "model.vmb")
    shutil.copyfile(vmb_src, dst)

    p = subprocess.run([ENG], cwd=REL, capture_output=True, text=True,
                       errors="replace", encoding="utf-8")
    out = p.stdout
    print("engine exit=%d, stdout %d chars" % (p.returncode, len(out)))
    if p.returncode != 0:
        print("STDERR:", p.stderr[:2000])
        print("STDOUT:", out[:2000])
        return 1

    meta = json.load(open(os.path.join(REF, "meta.json"), encoding="utf-8"))
    S = meta["seq"]
    cfg = meta["config"]
    got = parse_prints(out, meta)

    # Walk the flat output, slicing each stage by its known element count.
    shapes = {
        "xn": (S, cfg["hidden_size"]),
        "q": (S, cfg["num_attention_heads"] * cfg["head_dim"]),
        "k": (S, cfg["num_key_value_heads"] * cfg["head_dim"]),
        "v": (S, cfg["num_key_value_heads"] * cfg["head_dim"]),
        "att_out": (S, cfg["num_attention_heads"] * cfg["head_dim"]),
        "oproj": (S, cfg["hidden_size"]),
        "xn2": (S, cfg["hidden_size"]),
        "gate": (S, cfg["intermediate_size"]),
        "acts": (S, cfg["intermediate_size"]),
        "down": (S, cfg["hidden_size"]),
        "h_out": (S, cfg["hidden_size"]),
    }
    total = sum(int(np.prod(v)) for v in shapes.values())
    print("expected %d values, engine printed %d" % (total, got.size))
    if got.size != total:
        print("!! count mismatch -- cannot align stages")
        return 1

    print()
    print("%-10s %-14s %12s %12s %12s  %s"
          % ("stage", "vs HF", "absmax", "refmax", "rel", "verdict"))
    worst = 0.0
    off = 0
    fails = []
    for k in ORDER:
        n = int(np.prod(shapes[k]))
        g = got[off:off + n]
        off += n
        w = L(REFMAP[k]).reshape(-1)
        if w.size != n:
            print("%-10s SHAPE MISMATCH engine %d ref %d" % (k, n, w.size))
            fails.append(k)
            continue
        d = np.abs(g - w)
        refmax = max(float(np.abs(w).max()), 1e-30)
        rel = float(d.max()) / refmax
        worst = max(worst, rel)
        ok = rel < 1e-3
        if not ok:
            fails.append(k)
        print("%-10s %-14s %12.4e %12.4e %12.3e  %s"
              % (k, REFMAP[k], d.max(), refmax, rel, "OK" if ok else "FAIL <--"))

    print()
    print("worst stage rel = %.3e" % worst)
    if not fails:
        print("RESULT: engine layer 0 matches HuggingFace at every stage.")
        return 0
    print("RESULT: %d stage(s) differ: %s" % (len(fails), ", ".join(fails)))
    print("        (these are ENGINE gaps -- the numpy path already matched HF)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
