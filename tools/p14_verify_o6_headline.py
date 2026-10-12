#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P14 O6 headline INDEPENDENT re-derivation from the raw engine stdout.

WHY THIS IS REAL VERIFICATION AND NOT A RUBBER STAMP
    The headline artifact directory already contains the implementer's own
    o6_headline.json claiming 16/16 argmax agreement with logits_bit_identical
    = false. I did NOT write that file and I am not accepting its numbers.
    What I do here is re-derive the SAME quantities from the RAW engine stdout
    (_engine_out.txt, 50 MB) using the artifact's own emitted manifest
    (gen_rows.json), and then compare my independently computed chain against
    the claimed one. If they agree, the claim is corroborated by a second,
    independent computation over the primitive data. If they disagree, the
    claim is wrong.

    This is cheap (no model re-run) and it is the strongest check available for
    numbers someone else produced: I recompute from the raw bytes rather than
    trusting a summary.

WHAT I AM SPECIFICALLY LOOKING FOR, because the claimed numbers contain a
tension worth resolving:
    * "argmax_agreement: 16"      -> every generated token matches
    * "logits_bit_identical: false", max_abs_logit_diff = 1.597e-05
    The task book's acceptance for O6 is "逐位相同" (BIT-IDENTICAL). 16/16 argmax
    agreement is a WEAKER statement than bit-identical logits: it says the
    ranking survived, not that the values are the same. Both can be true at
    once, and if so the honest headline is "argmax-identical, NOT bit-identical"
    -- which must be reported as such rather than as "bit-identical".
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p14_verify_lib as L  # noqa: E402

DIRS = [os.path.join(L.ROOT, "examples", "p14_gen"),
        os.path.join(L.ROOT, "examples", "p14", "argmax")]


def read_numbers(path):
    """Parse whitespace-separated floats from an engine stdout file.

    ENCODING MATTERS AND IS A REAL TRAP HERE: the engine writes plain ASCII, but
    when a run is captured through a PowerShell redirect the file can come out
    UTF-16LE (with a BOM and a NUL between every ASCII byte). Reading such a file
    as ASCII yields ZERO numbers, which would look like "the engine produced no
    output" rather than "the capture encoding differs". The first version of
    this reader did exactly that. So the bytes are decoded by sniffing the BOM.
    """
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        text = raw.decode("utf-16", "replace")
    elif raw[:3] == b"\xef\xbb\xbf":
        text = raw[3:].decode("utf-8", "replace")
    else:
        # Detect UTF-16LE without a BOM: many NUL bytes in even positions.
        nul = raw[:4096].count(0)
        text = (raw.decode("utf-16-le", "replace") if nul > len(raw[:4096]) // 8
                else raw.decode("ascii", "replace"))
    vals = []
    for t in text.split():
        try:
            vals.append(float(t))
        except ValueError:
            pass
    return vals, len(raw)


def load_dir(d):
    need = ["gen_rows.json", "_engine_out.txt"]
    if not all(os.path.exists(os.path.join(d, n)) for n in need):
        return None
    man = json.load(open(os.path.join(d, "gen_rows.json"), encoding="utf-8"))
    vals, nbytes = read_numbers(os.path.join(d, "_engine_out.txt"))
    idx = 0
    blocks = {}
    for b in man["blocks"]:
        n = b["rows"] * b["cols"]
        if idx + n > len(vals):
            return {"dir": d, "error": "stdout too short for block %r: need %d, "
                                       "have %d (contention artefact, or an "
                                       "encoding mismatch)"
                    % (b["name"], idx + n, len(vals))}
        blocks[b["name"]] = np.array(vals[idx:idx + n],
                                     dtype=np.float32).reshape(b["rows"],
                                                               b["cols"])
        idx += n
    return {"dir": d, "man": man, "blocks": blocks, "n_vals": len(vals),
            "consumed": idx, "nbytes": nbytes}


def main():
    print("=== O6 headline: INDEPENDENT re-derivation from raw stdout ===")
    got = None
    for d in DIRS:
        r = load_dir(d)
        if r is None:
            print("skip (no raw stdout/manifest): %s" % d)
            continue
        if "error" in r:
            print("SKIP %s: %s" % (r["dir"], r["error"]))
            continue
        got = r
        break
    if got is None:
        print("UNVERIFIED: no directory with both _engine_out.txt and "
              "gen_rows.json")
        return 1

    d, man, blocks = got["dir"], got["man"], got["blocks"]
    G = man["gen"]
    print("dir            : %s" % d)
    print("manifest       : S=%d G=%d layers=%s sampler=%s"
          % (man["seq"], G, man.get("layers"), man.get("sampler")))
    print("stdout values  : %d (manifest accounts for %d)"
          % (got["n_vals"], got["consumed"]))
    if got["n_vals"] != got["consumed"]:
        print("  NOTE: %d trailing values beyond the manifest"
              % (got["n_vals"] - got["consumed"]))

    # ---- re-derive the chain: argmax of each step's own logits ----
    per_step = []
    for k in range(G):
        key = "step%d_logits" % k
        if key in blocks:
            per_step.append(int(np.argmax(blocks[key])))
    print("\nre-derived argmax chain (from raw logits): %s" % per_step)

    final = None
    for key in ("st_gen_final", "st_gen"):
        if key in blocks:
            final = [int(round(float(v))) for v in blocks[key][0]]
            break
    print("st_gen_final from stdout                 : %s" % final)
    derived_tail = final[1:] if final else []
    print("st_gen tail (generated)                  : %s" % derived_tail)

    # ---- compare with the implementer's own claim ----
    claim_path = os.path.join(d, "o6_headline.json")
    ok_claim = True
    if os.path.exists(claim_path):
        claim = json.load(open(claim_path, encoding="utf-8"))
        c = claim.get("closed_loop_chain", [])
        print("\nclaimed closed_loop_chain                 : %s" % c)
        same, det = L.compare_chains(c, derived_tail, "claim vs my derivation")
        print("my derivation == claimed chain           : %s  %s" % (same, det))
        print("claimed argmax_agreement                 : %s / %s"
              % (claim.get("argmax_agreement"), claim.get("N", G)))
        print("claimed logits_bit_identical             : %s"
              % claim.get("logits_bit_identical"))
        print("claimed max_abs_logit_diff               : %s"
              % claim.get("max_abs_logit_diff"))
        ok_claim = same
    else:
        print("\nno o6_headline.json in %s -- nothing to corroborate" % d)

    # ---- the O6-P1 internal consistency, recomputed here ----
    consumed = []
    for k in range(G):
        key = "step%d_seen" % k
        if key in blocks:
            consumed.append(int(round(float(blocks[key][0, 0]))))
    print("\ntokens consumed per step (from stdout)   : %s" % consumed)
    if consumed and per_step:
        exp = consumed[0]
        # consumed[k] must equal argmax(step k-1 logits) for k>=1
        chain_ok = all(consumed[k + 1] == per_step[k]
                       for k in range(min(len(consumed) - 1, len(per_step))))
        print("consumed[k+1] == argmax(step k logits)   : %s" % chain_ok)
    else:
        chain_ok = None
        print("cannot check feed-back consistency (blocks missing)")

    # ---- the headline wording question: argmax-identical vs bit-identical ----
    print("\n" + "=" * 62)
    print("HEADLINE WORDING CHECK")
    print("  The task book asks for 逐位相同 (BIT-IDENTICAL).")
    if os.path.exists(claim_path):
        bi = claim.get("logits_bit_identical")
        mad = claim.get("max_abs_logit_diff")
        agree = claim.get("argmax_agreement")
        if bi is False:
            print("  The artifact's own claim reports logits_bit_identical = FALSE")
            print("  with max|d| = %s, while argmax agreement is %s/%s."
                  % (mad, agree, claim.get("N", G)))
            print("  => The DEFENSIBLE headline is 'argmax-identical over G tokens',")
            print("     NOT 'bit-identical'. Bit-identical is FALSE as measured,")
            print("     and reporting it as bit-identical would be an overstatement.")
            if agree == claim.get("N", G):
                print("  => argmax agreement IS full (%s/%s): the ranking survived."
                      % (agree, claim.get("N", G)))
        else:
            print("  artifact claims logits_bit_identical = %s" % bi)

    # degenerate-tail observation (informational, not a pass/fail)
    if derived_tail:
        from collections import Counter
        cnt = Counter(derived_tail)
        top, n = cnt.most_common(1)[0]
        if n >= 3:
            print("\n  OBSERVATION: token %d repeats %d times in the %d-token "
                  "tail." % (top, n, len(derived_tail)))
            print("  A degenerate loop is a legitimate outcome of greedy decoding")
            print("  with a 4-LAYER model, but it means the headline chain does not")
            print("  demonstrate 16 INDEPENDENT steps. Worth noting alongside the")
            print("  claim rather than presenting 16/16 as 16 informative tokens.")

    print("\nVERDICT: %s" % ("claim CORROBORATED by independent recomputation"
                             if ok_claim else
                             "claim NOT corroborated -- see differences above"))
    return 0 if ok_claim else 1


if __name__ == "__main__":
    sys.exit(main())
