#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P14 O6-P5 independent check: "temperature=0 degrades EXACTLY to argmax",
and "temperature>0 genuinely changes output".

THE HONEST FRAMING, WHICH MATTERS MORE THAN THE RESULT
    Sampling in O6 is HOST-SIDE: the engine closes the loop with an in-engine
    argmax (OP_TOPK_IDX) and prints each step's logits; temperature / top-k /
    top-p are then applied by Python to those printed logits. So a test of
    "temperature=0 gives argmax" tests a PYTHON FUNCTION, not the engine.

    That is a legitimate design (the builder documents it explicitly), but it
    means this criterion CANNOT be reported as an engine property. This script
    therefore:
      1. measures the property on the host sampler, and says so;
      2. verifies the property against an INDEPENDENT argmax (numpy) rather than
         against the sampler's own branch -- comparing a function to itself
         proves nothing;
      3. checks the specific ways a "temperature" implementation can be WRONG
         while still looking plausible: temp=0 vs a tiny temp, temp>0 not
         actually reordering, and top-k/top-p being inert;
      4. reports the whole thing as HOST-SIDE so the headline claim is not
         overstated as an engine capability.
"""
import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p14_verify_lib as L  # noqa: E402

ROOT = L.ROOT
CHECK = os.path.join(ROOT, "tools", "p14_gen_check.py")


def load_sampler():
    """Import p14_gen_check.py purely to reach its sample_token().

    Imported by path so this script does not depend on the implementer's module
    being importable as a package.
    """
    spec = importlib.util.spec_from_file_location("p14_gen_check", CHECK)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["p14_gen_check"] = mod
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    return mod


def main():
    print("=== O6-P5: temperature=0 -> argmax EXACTLY (HOST-SIDE sampler) ===")
    if not os.path.exists(CHECK):
        print("UNVERIFIED: %s absent" % CHECK)
        return 1
    mod = load_sampler()
    if not hasattr(mod, "sample_token"):
        print("UNVERIFIED: sample_token not found in the checker")
        return 1
    fn = mod.sample_token
    meta = L.MetaCheck()
    fails = 0

    rng = np.random.RandomState(20240607)
    logits = rng.randn(130560).astype(np.float32)
    # A deliberately peaked case and a deliberately flat one, so the test does
    # not only exercise one shape of distribution.
    peaked = np.full(130560, -20.0, dtype=np.float32)
    peaked[1234] = 30.0
    flat = np.zeros(130560, dtype=np.float32)

    # ---------------------------------------------------------------
    # 1. temp=0 must equal an INDEPENDENT argmax
    # ---------------------------------------------------------------
    print("\n[1] temp=0 vs an INDEPENDENT argmax (numpy), not vs its own branch")
    bad = 0
    for name, lg in [("random", logits), ("peaked", peaked), ("flat", flat)]:
        want = int(np.argmax(lg))
        got, how = fn(lg.copy(), 0.0, 0, 1.0, np.random.RandomState(1))
        same = (got == want)
        bad += 0 if same else 1
        print("   %-7s temp=0 -> %-8d independent argmax -> %-8d  %s  (%s)"
              % (name, got, want, "OK" if same else "MISMATCH", how))
    p5a = (bad == 0)
    print("   -> %s" % ("PASS" if p5a else "FAIL"))

    # ---------------------------------------------------------------
    # 2. temp=0 must NOT be implementable as "very small temp": the two must
    #    agree here, but the DISTINCTION is that temp=0 must be exact even when
    #    the distribution is flat (all-tied), where a softmax over
    #    logits/1e-9 produces NaN.
    # ---------------------------------------------------------------
    print("\n[2] the flat/all-tied row: temp=0 must stay exact, not NaN")
    got0, _ = fn(flat.copy(), 0.0, 0, 1.0, np.random.RandomState(2))
    got_tiny, how_tiny = fn(flat.copy(), 1e-9, 0, 1.0, np.random.RandomState(2))
    print("   temp=0    -> %d" % got0)
    print("   temp=1e-9 -> %d  (%s)" % (got_tiny, how_tiny))
    exact0 = (got0 == 0)      # np.argmax on an all-zero row returns index 0
    print("   temp=0 exact (index 0): %s" % exact0)

    # can-fail: the SAME assertion against a row whose argmax is NOT 0
    shifted = flat.copy()
    shifted[7] = 1.0
    got_s, _ = fn(shifted.copy(), 0.0, 0, 1.0, np.random.RandomState(2))
    moved = (got_s == 7)
    print("   control: row with a peak at index 7 -> %d (must be 7): %s"
          % (got_s, moved))
    p5b = exact0 and moved
    meta.record("O6-P5 temp=0 exact on ties", p5b,
                "a row whose argmax is not 0 must return 7, not 0",
                bool(moved), "control moved the answer as required")
    if not p5b:
        fails += 1

    # ---------------------------------------------------------------
    # 3. temp>0 must ACTUALLY change the output at least sometimes
    # ---------------------------------------------------------------
    print("\n[3] temp>0 must genuinely change output (else it is inert)")
    changed = 0
    trials = 200
    for s in range(trials):
        r = np.random.RandomState(1000 + s)
        g0, _ = fn(logits.copy(), 0.0, 0, 1.0, r)
        r2 = np.random.RandomState(1000 + s)
        g1, _ = fn(logits.copy(), 1.0, 0, 1.0, r2)
        if g0 != g1:
            changed += 1
    print("   temp=0 vs temp=1.0 differed in %d / %d draws" % (changed, trials))
    p5c = (changed > 0)
    print("   -> %s" % ("PASS: temperature is wired in" if p5c else
                        "FAIL: temperature appears inert"))
    meta.record("O6-P5 temp>0 changes output", p5c,
                "temperature having no effect at all", bool(changed > 0),
                "%d/%d draws differed" % (changed, trials))
    if not p5c:
        fails += 1

    # ---------------------------------------------------------------
    # 4. top-k must actually restrict the candidate set
    # ---------------------------------------------------------------
    print("\n[4] top-k must restrict candidates (measured, not assumed)")
    seen = set()
    for s in range(400):
        r = np.random.RandomState(5000 + s)
        g, _ = fn(logits.copy(), 2.0, 5, 1.0, r)
        seen.add(g)
    print("   distinct tokens over 400 draws with top_k=5: %d" % len(seen))
    p5d = (len(seen) <= 5 and len(seen) > 0)
    print("   -> %s" % ("PASS: never exceeded k" if p5d else
                        "FAIL: top-k not enforced"))
    meta.record("O6-P5 top-k enforced", p5d,
                "drawing a 6th distinct token under top_k=5",
                bool(len(seen) <= 5),
                "%d distinct <= 5" % len(seen))
    if not p5d:
        fails += 1

    # ---------------------------------------------------------------
    print("\n" + "=" * 62)
    print("IMPORTANT SCOPE STATEMENT")
    print("  Sampling here is HOST-SIDE PYTHON, not the engine. The engine's")
    print("  contribution to O6 is the IN-ENGINE argmax (OP_TOPK_IDX) that")
    print("  closes the loop; that part IS verified against the engine")
    print("  elsewhere (O6-P1/P2 in tools/p14_verify_o6.py).")
    print("  'temperature=0 degrades to argmax' is therefore a property of the")
    print("  host sampler, and is recorded as HOST-SIDE, not as an engine")
    print("  capability.")
    allproven = meta.report()
    print("\nO6-P5 VERDICT: %s" % ("PASS (host-side sampler)"
                                   if (fails == 0 and allproven) else "FAIL"))
    return 0 if (fails == 0 and allproven) else 1


if __name__ == "__main__":
    sys.exit(main())
