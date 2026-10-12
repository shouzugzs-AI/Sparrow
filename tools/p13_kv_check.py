#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P13 task C: does incremental decode reproduce a FULL re-prefill?

THE CORRECTNESS QUESTION, STATED PRECISELY
    Decoding token t from a K/V cache must produce the same hidden state as
    running a normal forward over tokens [0..t]. If it does not, the cache is
    wrong -- and the failure will look like a plausible numeric drift rather
    than a crash, which is the dangerous kind.

HOW THE COMPARISON IS SET UP
    Two builds of the SAME weights:
      A) "reference", seq = S+G, prompt-mode tokens  -> the final hidden state
         prints represent the last position of a plain forward over S+G tokens.
         (`PRINT xn` prints h after the final norm at every position; the LAST
          S... actually the whole [S+G, H] block, of which the final row is the
          one the decode path must reproduce.)
      B) "decode", seq = S with --decode G -> prefill S, then G cached steps,
         each printing its post-final-norm hidden state.

    Decode step k (0-based) processes absolute position S+k, so it must equal
    row S+k of the reference. The mask is causal in both, and the reference's
    row S+k sees positions [0, S+k] -- exactly the live range the cached step
    attends over.

WHAT WOULD CATCH EACH CLASS OF BUG
    * wrong RoPE position  -> large error, grows with position
    * wrong cache column   -> large error on every step
    * missing causal mask  -> attention leaks future positions
    * transposed layout    -> dimension-sized garbage, not small drift

Usage:
    python tools/p13_kv_check.py --layers 1 --seq 4 --decode 3
"""
import argparse
import json
import os
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
WORK = os.path.join(ROOT, "examples", "p13_kv")

PROMPT_IDS = [3, 117, 2501, 88, 42, 7, 900, 15]
GEN_IDS = [55, 321, 4, 88, 1234, 9, 77, 20]


def build(layers, seq, out, decode=0):
    cmd = [sys.executable, os.path.join(ROOT, "tools", "p12_build_minicpm.py"),
           "--layers", str(layers), "--seq", str(seq), "--skip-head",
           "--print-hidden", "--prompt-mode", "tokens", "--out", out]
    if decode:
        cmd += ["--decode", str(decode)]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                       errors="replace")
    if p.returncode != 0:
        print("BUILD FAILED")
        print((p.stdout + p.stderr)[-1500:])
        return False
    consts = [l.strip() for l in open(os.path.join(out, "weights.args"),
                                      encoding="utf-8") if l.strip()]
    q = subprocess.run([ASM, "full.vml", "m.vmb"] + consts, cwd=out,
                       capture_output=True, text=True, errors="replace")
    if q.returncode != 0:
        print("ASSEMBLE FAILED")
        print((q.stdout + q.stderr)[-1500:])
        return False
    return True


def run(out, prompt_ids, gen_ids=None):
    shutil.copyfile(os.path.join(out, "m.vmb"), os.path.join(REL, "model.vmb"))
    with open(os.path.join(REL, "tokens.bin"), "wb") as f:
        np.array(prompt_ids, dtype="<u4").tofile(f)
    gt = os.path.join(REL, "gen_tokens.bin")
    if gen_ids is None:
        if os.path.exists(gt):
            os.remove(gt)
    else:
        with open(gt, "wb") as f:
            np.array(gen_ids, dtype="<u4").tofile(f)
    res = os.path.join(out, "_out.txt")
    with open(res, "wb") as f:
        p = subprocess.run([ENG], cwd=REL, stdout=f, stderr=subprocess.DEVNULL)
    if p.returncode != 0:
        return None, p.returncode
    vals = []
    with open(res, "r", errors="replace") as f:
        for line in f:
            for t in line.split():
                try:
                    vals.append(float(t))
                except ValueError:
                    pass
    return np.array(vals, dtype=np.float32), 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--seq", type=int, default=4)
    ap.add_argument("--decode", type=int, default=3)
    args = ap.parse_args()
    S, G, H = args.seq, args.decode, 1536

    os.makedirs(WORK, exist_ok=True)
    print("=== P13 task C: cached decode vs full re-prefill "
          "(layers=%d, S=%d, G=%d) ===\n" % (args.layers, S, G))

    # --- reference: one plain forward over S+G tokens ---
    ref_ids = PROMPT_IDS[:S] + GEN_IDS[:G]
    dref = os.path.join(WORK, "ref")
    if not build(args.layers, S + G, dref):
        return 1
    ref, rc = run(dref, ref_ids)
    if ref is None:
        print("reference engine exit=%d" % rc)
        return 1
    if ref.size != (S + G) * H:
        print("reference printed %d values, expected %d" % (ref.size, (S + G) * H))
        return 1
    ref = ref.reshape(S + G, H)
    print("reference: forward over %d tokens -> %s" % (S + G, ref.shape))

    # --- decode: prefill S, then G cached steps ---
    ddec = os.path.join(WORK, "dec")
    if not build(args.layers, S, ddec, decode=G):
        return 1
    got, rc = run(ddec, PROMPT_IDS[:S], GEN_IDS[:G])
    if got is None:
        print("decode engine exit=%d" % rc)
        return 1
    # stdout is ONE stream, and the prefill's own `PRINT xn` comes first (the
    # builder emits it before the decode steps on purpose -- see the ORDERING
    # NOTE in p12_build_minicpm.py). So the layout is:
    #     [S rows of prefill hidden state][G rows of decoded hidden state]
    # Reading only the trailing G*H values is what isolates the decode output.
    expect_total = (S + G) * H
    if got.size != expect_total:
        print("decode printed %d values, expected %d "
              "(%d prefill rows + %d decoded rows)"
              % (got.size, expect_total, S, G))
        return 1
    allrows = got.reshape(S + G, H)
    got = allrows[S:]          # the decoded rows
    prefill = allrows[:S]
    print("decode   : %d cached steps -> %s (preceded by %d prefill rows)\n"
          % (G, got.shape, S))

    # Sanity: the prefill rows must also match the reference's first S rows.
    # If they do not, the comparison below is meaningless.
    pd = float(np.abs(prefill - ref[:S]).max())
    print("prefill rows vs reference rows 0..%d: max|d| = %.3e %s"
          % (S - 1, pd, "(OK)" if pd <= 1e-5 else "(MISMATCH -- aborting)"))
    if pd > 1e-5:
        print("\nVERDICT: FAIL -- the prefill itself disagrees with the reference")
        return 1
    print()

    # --- compare step k against reference row S+k ---
    print("%6s %8s %14s %14s" % ("step", "abs pos", "max|d|", "rel to absmax"))
    print("-" * 48)
    worst = 0.0
    scale = max(float(np.abs(ref).max()), 1e-30)
    for k in range(G):
        row = S + k
        d = float(np.abs(got[k] - ref[row]).max())
        worst = max(worst, d)
        print("%6d %8d %14.4e %14.4e" % (k, row, d, d / scale))

    # The engine prints 6 decimals, so two computations agreeing to full f32
    # precision still differ by up to ~1e-6 in text. What matters is whether
    # the cache changed the ANSWER by more than that floor.
    floor = 1e-5
    print()
    if worst <= floor:
        print("VERDICT: PASS -- cached decode reproduces the full re-prefill")
        print("         (worst |d| %.2e <= %.0e text-precision floor)" % (worst, floor))
        return 0
    print("VERDICT: FAIL -- worst |d| %.3e exceeds the %.0e floor" % (worst, floor))
    print("         the cache is NOT equivalent to a full forward")
    return 1


if __name__ == "__main__":
    sys.exit(main())
