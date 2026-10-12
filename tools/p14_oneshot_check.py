#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P14 / O6 headline: is the closed loop's logit stream BIT-IDENTICAL to a one-shot
full forward over the same positions?

THE CLAIM UNDER TEST
    Task-2's acceptance says: "Generate 16 tokens at temperature=0; the sequence
    must be BIT-IDENTICAL to a one-shot full forward over the same 16 positions."

WHY THIS NEEDS A CAREFUL TEST RATHER THAN A NAIVE ONE
    The closed loop decodes ONE token at a time against a KV cache, appending to
    it with WRITECOL. A one-shot full forward runs all N positions through the
    layer stack in a single pass with causal masking.

    Those two are different SUMMATION ORDERS in the attention GEMM:
      * decode:   sc[1,live] = q[1,128] x kcache[128,live]  (K = 128)
      * full:     sc[N,N]    = q[N,128] x kT[128,N]         (K = 128, same K!)
    In this engine the reduction over K is 128 in both cases, and both go
    through the same kernel, so the SUM ITSELF has the same order. What differs
    is the extent of the output and the softmax row width (live vs N), and
    whether the causal mask is applied inside softmax or by slicing the live
    range.

    P13 already measured that its cached decode reproduced a full re-prefill
    "bit-exactly (0.0000e+00)" for the HIDDEN STATE. What was never tested is
    the LOGITS at the final position and the ARGMAX derived from them.

WHAT THIS TOOL ACTUALLY DOES
    It builds a second artifact from the SAME weights -- a one-shot full forward
    over prompt + generated chain (N = S + G = 20 positions) -- runs it, and
    compares:
      1. argmax at each of the last G positions vs the closed loop's chain
         (THE ACCEPTANCE CRITERION: 16/16)
      2. max |logit difference| at those positions (the honest magnitude, which
         is reported even when argmax matches, because "bit-identical" is a much
         stronger statement than "same argmax" and we must not conflate them)
      3. exact float equality count, so a bit-identical result is DISTINGUISHABLE
         from a merely close one.

    If the two differ in the last bits (expected, for the summation-order reason
    above), this tool SAYS SO and reports the measured deviation rather than
    claiming bit-identity it did not observe.

Usage: python tools/p14_oneshot_check.py --dir examples/p14_gen --oneshot examples/p14_oneshot
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = str(Path(__file__).resolve().parents[1])
REL = os.path.join(ROOT, "target", "release")
ENG = os.path.join(REL, "vmlm-runtime.exe")

sys.path.insert(0, os.path.join(ROOT, "tools"))
from p14_gen_check import parse_stdout, run_artifact  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="examples/p14_gen")
    ap.add_argument("--oneshot", default="examples/p14_oneshot")
    args = ap.parse_args()

    adir = os.path.join(ROOT, args.dir)
    odir = os.path.join(ROOT, args.oneshot)
    meta = json.load(open(os.path.join(adir, "meta.json")))
    rows = json.load(open(os.path.join(adir, "gen_rows.json")))
    S, G, V = meta["seq"], meta["gen"], meta["vocab"]

    print("=== O6 headline: closed loop vs one-shot full forward ===")
    print("S=%d G=%d vocab=%d layers=%d" % (S, G, V, meta["layers"]))
    print()

    # ---- the closed loop's own logits (already produced) ----
    co = os.path.join(adir, "_check_closed.txt")
    if not os.path.exists(co):
        print("run tools/p14_gen_check.py --dir %s first (need %s)" % (args.dir, co))
        return 1
    cb, cw = parse_stdout(co, rows["blocks"])
    if cb is None:
        print("closed-loop parse failed: %s" % cw)
        return 1
    step_logits = [cb["step%d_logits" % k].reshape(V) for k in range(G)]
    chain = [int(np.argmax(x)) for x in step_logits]
    print("closed-loop ARGMAX chain (the O6 output):")
    print("  %s" % chain)
    print()

    # ---- the one-shot artifact ----
    om = os.path.join(odir, "meta.json")
    if not os.path.exists(om):
        print("one-shot artifact missing at %s" % odir)
        print("Build it with: python tools/p14_oneshot_build.py --dir %s --out %s"
              % (args.dir, args.oneshot))
        return 1
    ometa = json.load(open(om))
    orows = json.load(open(os.path.join(odir, "gen_rows.json")))
    N = ometa["n_pos"]

    out, dt, rc = run_artifact(odir, "oneshot", "prompt_tok.bin", "oneshot")
    if rc != 0:
        print("one-shot engine exit=%d" % rc)
        return 1
    print("one-shot full forward over N=%d positions: ran in %.2f s" % (N, dt))

    ob, ow = parse_stdout(out, orows["blocks"])
    if ob is None:
        print("one-shot parse failed: %s" % ow)
        return 1
    full_logits = ob["logits"].reshape(N, V)
    print()

    # ---- compare the last G positions ----
    # Closed-loop step k is absolute position S+k. The one-shot run's row S+k is
    # the SAME position, computed by a full causal pass instead of incrementally.
    print("--- per-position comparison (closed loop step k vs one-shot row S+k) ---")
    print("  %-5s %-10s %-10s %-12s %-10s" %
          ("pos", "loop-am", "oneshot-am", "max|dlogit|", "exact-eq"))
    am_match = 0
    maxd_all = 0.0
    exact_all = True
    for k in range(G):
        pos = S + k
        a_loop = int(np.argmax(step_logits[k]))
        a_full = int(np.argmax(full_logits[pos]))
        d = np.abs(step_logits[k].astype(np.float64)
                   - full_logits[pos].astype(np.float64))
        md = float(d.max())
        eq = bool(np.array_equal(step_logits[k], full_logits[pos]))
        exact_all = exact_all and eq
        maxd_all = max(maxd_all, md)
        if a_loop == a_full:
            am_match += 1
        print("  %-5d %-10d %-10d %-12.4e %-10s"
              % (pos, a_loop, a_full, md, "YES" if eq else "no"))

    print()
    print("=== RESULT ===")
    print("  argmax agreement : %d/%d" % (am_match, G))
    print("  max |dlogit|     : %.6e" % maxd_all)
    print("  all logits bit-identical : %s" % exact_all)

    # The acceptance criterion is the TOKEN SEQUENCE.
    if am_match == G:
        print("  ACCEPTANCE (bit-identical token sequence): PASS")
        if exact_all:
            print("  and the underlying LOGITS are bit-identical too.")
        else:
            print("  NOTE: the token sequence matches but the LOGITS are NOT")
            print("        bit-identical (max|d|=%.3e). Stating the claim as" % maxd_all)
            print("        'bit-identical token sequence' is supported; stating")
            print("        it as 'bit-identical logits' would NOT be.")
    else:
        print("  ACCEPTANCE: FAIL (%d of %d positions differ in argmax)"
              % (G - am_match, G))
        return 1

    # ---- write a machine-readable record ----
    rec = {
        "S": S, "G": G, "N": N, "vocab": V, "layers": meta["layers"],
        "prompt_ids": meta["prompt_ids"], "seed": meta["seed_token"],
        "closed_loop_chain": chain,
        "oneshot_argmax_tail": [int(np.argmax(full_logits[S + k])) for k in range(G)],
        "argmax_agreement": am_match,
        "max_abs_logit_diff": maxd_all,
        "logits_bit_identical": exact_all,
    }
    with open(os.path.join(adir, "o6_headline.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)
    print()
    print("wrote %s" % os.path.join(adir, "o6_headline.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
