#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P10 layer-0 engine-vs-HuggingFace comparison, run in an ISOLATED dir.

WHY THIS IS SEPARATE FROM tools/minicpm_layer0_run.py
    That script is correct, but it must stage into `target/release/` because
    the engine hardcodes `model.vmb` (runtime/src/main.rs:108). We share that
    directory with other teammates, so a concurrent run overwrote model.vmb /
    model.vmbw / tokens.bin and made this suite report:
        engine exit=2, stdout "cannot open file"
    which reads as a code regression but is pure contention. Measured: the
    113 MB model.vmbw sidecar had been wiped, and target/release/tokens.bin
    held (3, 117, 2501, 88) -- another session's test data.

    This script copies the engine + artifact + sidecar into a private temp
    directory and runs there, so nothing can interleave. It then applies the
    SAME stage-by-stage comparison and the SAME acceptance bar as
    tools/minicpm_layer0_run.py, so the verdict is comparable.

    IMPORTANT: this is a MEASUREMENT of the shared tree, not a replacement for
    the authoritative `tools/p10_minicpm_verify.ps1`. It exists to answer "was
    that FAIL contention or a real regression?".

Usage:
    python tools/o6_layer0_isolated.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = str(Path(__file__).resolve().parents[1])
REL = os.path.join(ROOT, "target", "release")
EXAMPLES = os.path.join(ROOT, "examples", "minicpm")
REF = os.path.join(EXAMPLES, "layer0_ref")
ENG = os.path.join(REL, "vmlm-runtime.exe")

# Stage order must match the PRINT sequence in tools/minicpm_layer0_vml.py.
ORDER = ["xn", "q", "k", "v", "att_out", "oproj", "xn2", "gate", "acts",
         "down", "h_out"]
REFMAP = {
    "xn": "in_ln_out", "q": "q_proj_out", "k": "k_proj_out", "v": "v_proj_out",
    "att_out": "attn_ctx_cat", "oproj": "o_proj_out", "xn2": "post_ln_out",
    "gate": "gate_proj_out", "acts": "mlp_act", "down": "down_proj_out",
    "h_out": "hidden_out",
}


def run_isolated():
    vmb = os.path.join(EXAMPLES, "layer0.vmb")
    vmbw = os.path.join(EXAMPLES, "layer0_weights.vmbw")
    for p in (ENG, vmb):
        if not os.path.exists(p):
            raise RuntimeError("MISSING: %s" % p)
    d = tempfile.mkdtemp(prefix="o6_layer0_")
    try:
        shutil.copyfile(ENG, os.path.join(d, "vmlm-runtime.exe"))
        shutil.copyfile(vmb, os.path.join(d, "model.vmb"))
        if os.path.exists(vmbw):
            shutil.copyfile(vmbw, os.path.join(d, "model.vmbw"))
        q = subprocess.run([os.path.join(d, "vmlm-runtime.exe")], cwd=d,
                           capture_output=True)
        return q.returncode, (q.stdout + q.stderr).decode("utf-8", "replace")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    print("=== P10 layer 0: engine vs HuggingFace (ISOLATED dir) ===\n")
    if not os.path.isdir(REF):
        print("SKIP: oracle %s absent" % REF)
        return 0

    code, out = run_isolated()
    print("engine exit=%d, stdout %d chars" % (code, len(out)))
    if code != 0:
        print("STDOUT/ERR: %s" % out[:600])
        return 1

    vals = []
    for ln in out.splitlines():
        for tok in ln.split():
            try:
                vals.append(float(tok))
            except ValueError:
                pass
    got = np.array(vals, dtype=np.float32)

    meta = json.load(open(os.path.join(REF, "meta.json"), encoding="utf-8"))
    S = meta["seq"]
    cfg = meta["config"]
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
    print("  %-10s %12s %12s %12s" % ("stage", "absmax|d|", "refmax", "relative"))
    print("  " + "-" * 50)
    off = 0
    worst = 0.0
    bad = []
    for name in ORDER:
        n = int(np.prod(shapes[name]))
        g = got[off:off + n]
        r = np.load(os.path.join(REF, REFMAP[name] + ".npy")).astype(np.float32).ravel()
        off += n
        if r.size != n:
            print("  %-10s REFERENCE SIZE MISMATCH %d vs %d" % (name, r.size, n))
            bad.append(name)
            continue
        d = float(np.max(np.abs(g - r))) if n else 0.0
        refmax = float(np.max(np.abs(r))) if n else 0.0
        rel = d / refmax if refmax > 0 else 0.0
        worst = max(worst, rel)
        print("  %-10s %12.3e %12.3e %12.3e  %s"
              % (name, d, refmax, rel, "OK" if rel < 1e-3 else "FAIL"))
        if rel >= 1e-3:
            bad.append(name)

    print()
    print("worst relative error: %.3e (bar 1e-3)" % worst)
    if bad:
        print("VERDICT: FAIL on %s" % ", ".join(bad))
        return 1
    print("VERDICT: PASS -- layer 0 matches HuggingFace stage by stage")
    print("         => the P10 layer-0 FAIL in verify_all was CONTENTION, not a regression")
    return 0


if __name__ == "__main__":
    sys.exit(main())
