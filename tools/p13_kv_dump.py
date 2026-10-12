#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P13 task C: dump the K cache and compare against an independently computed
RoPE-rotated key. Hard evidence instead of inference.

WHAT THIS IS FOR
    Incremental decode matched the reference on the LAST step of every
    configuration tried (S=1/4, G=1/2/3: last step 0.0000) while earlier steps
    were wrong by a factor that decayed ~5.5x per step. That pattern is
    consistent with exactly ONE cache column being wrong, and the candidate is
    the column a step appends for ITSELF. So rather than keep reasoning about
    it, print the cache and compare it to a value computed independently here.

HOW
    Builds a 1-layer decode model, then patches the generated .vml to PRINT the
    whole st_kc0_0 cache after prefill and after each decode step. numpy then
    recomputes what each column should be, using the engine's own convention:
    RoPE with rotate_half pairing and theta^(-2i/hd).

Usage: python tools/p13_kv_dump.py --seq 2 --decode 2
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = str(Path(__file__).resolve().parents[1])
REL = os.path.join(ROOT, "target", "release")
ASM = os.path.join(REL, "vmlm-asm.exe")
ENG = os.path.join(REL, "vmlm-runtime.exe")
WORK = os.path.join(ROOT, "examples", "p13_kvdump")
PROMPT_IDS = [3, 117, 2501, 88, 42, 7, 900, 15]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=2)
    ap.add_argument("--decode", type=int, default=2)
    args = ap.parse_args()
    S, G = args.seq, args.decode
    os.makedirs(WORK, exist_ok=True)

    # Build the decode program.
    p = subprocess.run(
        [sys.executable, os.path.join(ROOT, "tools", "p12_build_minicpm.py"),
         "--layers", "1", "--seq", str(S), "--decode", str(G), "--skip-head",
         "--print-hidden", "--prompt-mode", "tokens", "--out", WORK],
        cwd=ROOT, capture_output=True, text=True, errors="replace")
    if p.returncode != 0:
        print((p.stdout + p.stderr)[-1200:])
        return 1

    # Patch: after prefill (just before the decode section) and at the end of
    # each decode step, print the whole kv-head-0 key cache.
    vml = open(os.path.join(WORK, "full.vml"), encoding="ascii").read()
    vml = vml.replace("; ================ P13 task C: incremental decode",
                      "PRINT st_kc0_0\n; ================ P13 task C: incremental decode",
                      1)
    # after each decode step: the step's final norm print is the marker
    parts = vml.split("PRINT dtail")
    # parts[0]=before step0 ... parts[k] ends step k-1
    out = parts[0]
    for i, seg in enumerate(parts[1:]):
        out += "PRINT dtail\nPRINT st_kc0_0\n" + seg
    vml = out
    open(os.path.join(WORK, "dump.vml"), "w", encoding="ascii", newline="\n").write(vml)

    consts = [l.strip() for l in open(os.path.join(WORK, "weights.args"),
                                      encoding="utf-8") if l.strip()]
    q = subprocess.run([ASM, "dump.vml", "dump.vmb"] + consts, cwd=WORK,
                       capture_output=True, text=True, errors="replace")
    if q.returncode != 0:
        print("assemble failed")
        print((q.stdout + q.stderr)[-1200:])
        return 1

    # Run it.
    shutil.copyfile(os.path.join(WORK, "dump.vmb"), os.path.join(REL, "model.vmb"))
    with open(os.path.join(REL, "tokens.bin"), "wb") as f:
        np.array(PROMPT_IDS[:S], dtype="<u4").tofile(f)
    with open(os.path.join(REL, "gen_tokens.bin"), "wb") as f:
        np.array(PROMPT_IDS[S:S + G], dtype="<u4").tofile(f)
    res = os.path.join(WORK, "_out.txt")
    with open(res, "wb") as f:
        r = subprocess.run([ENG], cwd=REL, stdout=f, stderr=subprocess.DEVNULL)
    if r.returncode != 0:
        print("engine exit=%d" % r.returncode)
        return 1
    vals = []
    for line in open(res, "r", errors="replace"):
        for t in line.split():
            try:
                vals.append(float(t))
            except ValueError:
                pass
    print("=== P13 task C: cache dump (S=%d G=%d, layer 0, kv head 0) ===\n" % (S, G))
    print("total printed values: %d" % len(vals))
    HID = 1536                     # the model's hidden size, NOT head_dim
    HD = 128                       # head_dim (row count of the cache)
    PMAX = S + G
    # Print stream, in program order:
    #   [prefill: PRINT xn            -> S*HID]
    #   [PRINT st_kc0_0  (patched)    -> HD*PMAX]
    #   per step: [PRINT dtail (HID)] [PRINT st_kc0_0 (HD*PMAX)]
    pos = 0
    prefill_h = vals[pos:pos + S * HID]
    pos += S * HID
    print("prefill hidden rows: %d values (expected %d)" % (len(prefill_h), S * HID))
    blk = vals[pos:pos + HD * PMAX]
    print("\ncache after PREFILL (%d values, %d x %d):" % (len(blk), HD, PMAX))
    cm = np.array(blk, dtype=np.float32).reshape(HD, PMAX)
    for c in range(PMAX):
        col = cm[:, c]
        print("   col %d: norm2=%12.6f  first3=%s"
              % (c, float(np.linalg.norm(col)), np.round(col[:3], 6).tolist()))
    pos += HD * PMAX

    prev = cm.copy()
    for step in range(G):
        hid = vals[pos:pos + HID]
        pos += HID
        blk = vals[pos:pos + HD * PMAX]
        pos += HD * PMAX
        cm = np.array(blk, dtype=np.float32).reshape(HD, PMAX)
        changed = [c for c in range(PMAX)
                   if not np.allclose(cm[:, c], prev[:, c])]
        print("\ncache after step %d:" % step)
        for c in range(PMAX):
            mark = "  <-- CHANGED" if c in changed else ""
            print("   col %d: norm2=%12.6f  first3=%s%s"
                  % (c, float(np.linalg.norm(cm[:, c])),
                     np.round(cm[:3, c], 6).tolist(), mark))
        print("   columns that changed vs the previous state: %s" % changed)
        print("   (a correct append changes EXACTLY ONE column: index %d)"
              % (S + step))
        prev = cm.copy()
    return 0


if __name__ == "__main__":
    sys.exit(main())
