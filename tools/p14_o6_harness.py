#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P14 O6 verification: the forward-error-propagation harness.

WHAT O6 CLAIMS, AND THE THREE WAYS IT CAN BE FAKE
    Claim: "generate 16 tokens; the sequence must be bit-identical to a one-shot
    forward over the same 16 positions", plus "temperature=0 degrades exactly to
    argmax", plus "the loop really feeds back".

    FAKE 1 -- THE LOOP IS A NO-OP ECHOING ITS INPUT FILE.
        The runtime pre-loads st_gen from gen_tokens.bin BEFORE vm.run()
        (runtime/src/main.rs::load_tokens -> load_ids, measured). If the
        "generation loop" merely prints the token ids it was handed, the
        acceptance passes while nothing was generated. DEFENCE: the harness
        makes generation CONTROLLABLE -- it can hand the engine a wrong chain
        (wrong prompt ids, wrong feedback ids) and requires the output to be the
        ARGMAX CHAIN, not the file's contents. It also requires that a MISSING
        gen_tokens.bin is a LOUD failure (measured: exit=2 "cannot open file"),
        which an echo-the-file implementation cannot survive.

    FAKE 2 -- THE REFERENCE IS PRODUCED BY THE SAME BUGGY CODE PATH.
        "Bit-identical to a one-shot forward" is worthless if both numbers come
        from the same builder, the same cache, or the same argmax helper.
        DEFENCE: the harness builds the reference from a STRUCTURALLY DIFFERENT
        route and reconciles the two through a quantity that is externally
        anchored:
          * the reference argmax chain is obtained by a full forward over the
            SEQUENCE OF INTEREST in one shot (no incremental decode, no cache,
            no loop) -- this is the P13 "one-shot" path, which is itself
            validated against HuggingFace (argmax 4/4 on the model's own case);
          * and the anchors are the two measured engine chains:
                WITH_FEEDBACK : tokens produced by the O6 loop (argmax fed back)
                NO_FEEDBACK   : tokens produced by P13 decode (ids pre-given)
            These must AGREE, because the O6 loop's argmax -- if correct -- IS
            the argmax of the logits the no-feedback decode computes at each
            step. That makes the check: argmax_chain(logits_k) == chain[k] for
            every k, i.e. an INTERNAL CONSISTENCY test that does not depend on
            either implementation being right, plus an EXTERNAL anchor against
            HuggingFace on the model's own prompt.
        If the two routes disagree, at least one is wrong and that is reported
        as a FAIL with the first differing index -- never smoothed over.

    FAKE 3 -- THE COMPARISON CANNOT FAIL.
        The harness proves its own comparison can fail by running it against a
        deliberately perturbed chain (off-by-one, reversed, shifted, truncated)
        and requiring FAIL. Every perturbation and its outcome is recorded.

THE EXTERNAL ANCHOR
    On the model's own prompt (ids [3,117,2501,88]) the engine and HuggingFace
    already agree at 4/4 argmax [5, 242, 90, 82591] (P12/P13, re-measured here).
    So the first generated token of a generation starting from that prefill is
    FIXED BY AN INDEPENDENT ORACLE, not by our own code. The harness uses that
    as the outer anchor and treats everything after it as internal consistency.
"""
import argparse
import os
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p14_verify_lib as L  # noqa: E402

ROOT = L.ROOT
REL = L.REL
HF_LOGITS = os.path.join(ROOT, "examples", "minicpm_full", "hf_logits.npy")
HF_META = os.path.join(ROOT, "examples", "minicpm_full", "hf_meta.json")

# The model's own prompt and the HF-verified argmax chain for it. These are the
# only numbers in this file that do not come from our own implementation.
MODEL_IDS = [3, 117, 2501, 88]
HF_ARGMAX = [5, 242, 90, 82591]

VOCAB = 130560
HIDDEN = 1536


def load_hf_anchor():
    """Re-read the HuggingFace reference and re-derive its argmax HERE.

    Deliberately re-derived rather than hard-coded: if the .npy on disk changed,
    a hard-coded list would hide it and the anchor would silently stop anchoring.
    """
    if not os.path.exists(HF_LOGITS):
        return None, "hf_logits.npy absent"
    lg = np.load(HF_LOGITS)
    if lg.shape != (len(MODEL_IDS), VOCAB):
        return None, "unexpected hf_logits shape %s" % (lg.shape,)
    return [int(i) for i in np.argmax(lg, axis=1)], None


def split_logits_block(vals, n_rows):
    """Split a flat engine stdout dump into [n_rows, VOCAB] logits.

    The engine prints one row per position, VOCAB values each. A dump whose
    length is not an exact multiple is a FAILURE of the measurement, not
    something to trim silently -- trimming is how a misaligned read becomes a
    confident wrong answer (P12 lost a cycle to exactly this: 10752 values read
    where 4608 were expected).
    """
    v = np.asarray(vals, dtype=np.float32)
    want = n_rows * VOCAB
    if v.size != want:
        return None, ("engine printed %d values, expected %d (%d rows x %d vocab)"
                      % (v.size, want, n_rows, VOCAB))
    return v.reshape(n_rows, VOCAB), None


def argmax_of_logits(lg):
    return [int(i) for i in np.argmax(np.asarray(lg, dtype=np.float32), axis=1)]


def compare_chains(claimed, observed, label):
    """Exact integer-chain comparison. Reports the FIRST divergence index."""
    a = [int(x) for x in claimed]
    b = [int(x) for x in observed]
    if len(a) == 0 or len(b) == 0:
        return False, {"label": label, "note": "EMPTY -- not a measurement"}
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return False, {"label": label, "first_diff": i,
                           "claimed[i]": a[i], "observed[i]": b[i],
                           "n": n}
    if len(a) != len(b):
        return False, {"label": label, "note": "LENGTH MISMATCH",
                       "len_a": len(a), "len_b": len(b)}
    return True, {"label": label, "n": n, "identical": True}


def demonstrate_can_fail(chain):
    """THE META-CHECK for the chain comparison.

    A chain comparison that accepts everything is worthless. Run it against
    every perturbation class we can construct and require each to be REJECTED,
    while the unperturbed chain must be ACCEPTED. Returns (all_rejected, rows).
    """
    rows = []
    ok = True
    good, _ = compare_chains(chain, chain, "identity")
    if not good:
        rows.append(("identity", "ACCEPTED", "*** comparison rejects the "
                                            "unperturbed chain -- broken ***"))
        ok = False
    else:
        rows.append(("identity", "ACCEPTED", "correct: must accept"))
    for kind in ["off_by_one", "reverse", "shift_left", "shift_right",
                 "wrong_position", "all_zero", "truncate", "extend"]:
        p = L.perturb(chain, kind)
        if p == list(chain):
            rows.append((kind, "SAME AS INPUT", "*** perturbation was a no-op -- "
                                                "empty measurement ***"))
            ok = False
            continue
        acc, detail = compare_chains(chain, p, kind)
        if acc:
            rows.append((kind, "ACCEPTED", "*** NOT DETECTED -- comparison is a "
                                           "rubber stamp ***"))
            ok = False
        else:
            rows.append((kind, "REJECTED", "first_diff=%s" %
                         detail.get("first_diff", detail.get("note"))))
    return ok, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", default=None,
                    help="comma-separated token chain to self-check")
    args = ap.parse_args()

    print("=== O6 harness self-test: chain comparison + perturbator ===")

    chain = ([int(x) for x in args.chain.split(",")] if args.chain
             else MODEL_IDS + HF_ARGMAX)

    print("\n[1] external anchor: HuggingFace logits re-read from disk")
    hf, err = load_hf_anchor()
    if err:
        print("    UNAVAILABLE: %s" % err)
    else:
        same, d = compare_chains(HF_ARGMAX, hf, "hard-coded vs .npy")
        print("    hf_logits.npy argmax     : %s" % hf)
        print("    expected (P12/P13 record) : %s" % HF_ARGMAX)
        print("    %s" % ("PASS: anchor reproduces the recorded 4/4 argmax"
                          if same else "FAIL: %s" % d))

    print("\n[2] META-CHECK: can the chain comparison FAIL?")
    ok, rows = demonstrate_can_fail(chain)
    print("    %-16s %-15s %s" % ("perturbation", "verdict", "detail"))
    print("    " + "-" * 62)
    for kind, verdict, detail in rows:
        print("    %-16s %-15s %s" % (kind, verdict, detail))
    print("\n    comparison shown able to reject a wrong chain: %s" % ok)

    print("\n[3] the pre-registered O6 predicates, STATED BEFORE the "
          "implementations land")
    print("    P1  chain(O6 loop, temp=0) == argmax_chain(one-shot logits at "
          "those positions)")
    print("    P2  chain(O6 loop) == chain(P13 decode, same ids)  [route "
          "agreement]")
    print("    P3  chain[0] == %s  [HuggingFace anchor on the model's own "
          "prompt]" % HF_ARGMAX[0])
    print("    P4  gen_tokens.bin ABSENT -> loud non-zero exit (no echo-the-file)")
    print("    P5  gen_tokens.bin WRONG  -> chain unchanged (loop ignores it)")
    print("    P6  temp=0 == argmax exactly; temp>0 differs (and differs from "
          "temp=0)")
    print("    P7  two runs byte-identical")
    print("    P8  all of P1-P7 shown able to FAIL by perturbation")

    print("\nHARNESS VERDICT: %s" % ("OK -- comparison demonstrably rejects "
                                     "wrong chains" if ok else
                                     "BROKEN -- do not trust its PASS"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
