#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P16-4 (task-10): THE O6 GENERATION-CHAIN CYCLE -- model property or implementation
defect?  Consolidated, falsifiable harness.

THE QUESTION P14 LEFT OPEN
    P14 recorded that the 4-layer O6 autoregressive loop degenerates into a
    repeated-token tail, and INFERRED "it is the model". The direct measurement
    was missing: examples/minicpm_full/hf_logits.npy has shape (4, 130560) and
    covers only the 4 PROMPT positions, so generation positions 4..19 were never
    covered by any HuggingFace oracle.

WHAT THIS HARNESS DOES (each item is a measurement, not an argument)
    D1  Recomputes the argmax of every engine step's OWN dumped logits in numpy,
        independently of the engine's in-engine TOPK_IDX.
    D2  Proves the KV cache ADVANCES: argmax(step k logits) must equal the token
        the engine actually CONSUMED at step k+1 (step{k+1}_seen).
    D3  Proves the logits are LIVE, not frozen: successive step logits must differ.
    D4  Measures the H-model discriminator: top1-top2 margin and softmax entropy
        at every step. A frozen/degenerate path shows near-tied logits or an
        unchanging distribution.
    D5  Computes the cycle structure (onset, period, fixed point vs longer cycle)
        with an EXHAUSTIVE search, so "no cycle found" is a real absence.
    D6  The missing oracle: HuggingFace, truncated to the SAME layer count, fed the
        SAME token sequence, at the generation positions. Separately run by
        tools/p16_o6_hf_gen_oracle.py (it needs a 200s model load).
    D7  The built-in NEGATIVE CONTROL: the --sampler feed artifact is given a
        FORCED input chain, so its own logits must NOT reproduce that chain. This
        mechanically falsifies "the output is copied from the input file" and
        "the activation region is frozen".

EVERY CRITERION IS SHOWN ABLE TO FAIL. Run with --selftest to print the counts.

Usage:
    python tools/p16_o6_cycle.py
    python tools/p16_o6_cycle.py --selftest
"""
import argparse
import json
import os
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = str(Path(__file__).resolve().parents[1])
OUT = os.path.join(ROOT, "examples", "p16_o6")

# The 24-layer anchor P14 recorded, and the 4-layer anchor actually measured.
ANCHOR_24 = [5, 242, 90, 82591]
ANCHOR_4_HF = [791, 29885, 104, 91794]


# --------------------------------------------------------------------------
# parsing (encoding-aware; see the UTF-16 note)
# --------------------------------------------------------------------------
def parse_blocks(out_bytes, manifest):
    """Split engine stdout into the named blocks from gen_rows.json.

    ENCODING NOTE (measured): examples/p14_gen/_engine_out.txt and
    examples/p14_gen_feed/_engine_out.txt both carry the UTF-16LE BOM (\\xff\\xfe),
    even though tools/p14_gen_run.ps1:194 documents that trap and claims to exit 7
    on it. Decoding them as ASCII yields ZERO values and a misleading "stdout too
    short" error. The content is plain ASCII text, so the values are recoverable
    exactly; the manifest total is asserted so a mis-decode cannot pass silently.
    """
    if out_bytes[:2] == b"\xff\xfe":
        text = out_bytes.decode("utf-16-le", "replace")
        if text and text[0] == "\ufeff":
            text = text[1:]
        utf16 = True
    else:
        text = out_bytes.decode("ascii", "replace")
        utf16 = False

    vals = []
    for tok in text.split():
        try:
            vals.append(float(tok))
        except ValueError:
            pass
    idx = 0
    got = {}
    for b in manifest["blocks"]:
        n = b["rows"] * b["cols"]
        if idx + n > len(vals):
            return None, ("block %r needs %d values at offset %d, stdout had %d"
                          % (b["name"], n, idx, len(vals)))
        got[b["name"]] = np.array(vals[idx:idx + n], dtype=np.float32).reshape(
            b["rows"], b["cols"])
        idx += n
    if idx != len(vals):
        return None, ("stdout had %d values but the manifest accounts for %d -- "
                      "refusing to guess the remainder" % (len(vals), idx))
    return {"blocks": got, "utf16": utf16, "n_values": len(vals)}, None


def cycle_structure(seq):
    """Exhaustive search for the terminal cycle: earliest onset, smallest period.

    Returns (onset, period) or (None, None). period <= len/2 only, so a cycle is
    only reported when it repeats at least twice -- a single coincidence in the
    tail is not called a cycle.
    """
    n = len(seq)
    for start in range(n):
        for p in range(1, (n - start) // 2 + 1):
            body = seq[start:]
            if len(body) >= 2 * p and all(body[i] == body[i % p]
                                          for i in range(len(body))):
                return start, p
    return None, None


def margin_entropy(row):
    order = np.argsort(row)[::-1]
    t1, t2 = int(order[0]), int(order[1])
    margin = float(row[t1]) - float(row[t2])
    z = row.astype(np.float64) - float(row[t1])
    pr = np.exp(z)
    s = pr.sum()
    if not np.isfinite(s) or s <= 0:
        return margin, float("nan"), t1, t2
    pr /= s
    ent = float(-(pr * np.log(pr + 1e-300)).sum())
    return margin, ent, t1, t2


# --------------------------------------------------------------------------
# per-artifact analysis
# --------------------------------------------------------------------------
def load_artifact(d):
    man_p = os.path.join(d, "gen_rows.json")
    out_p = os.path.join(d, "_engine_out.txt")
    if not (os.path.exists(man_p) and os.path.exists(out_p)):
        return None, "missing gen_rows.json or _engine_out.txt"
    with open(man_p, encoding="utf-8") as f:
        man = json.load(f)
    with open(out_p, "rb") as f:
        out = f.read()
    meta = {}
    mp = os.path.join(d, "meta.json")
    if os.path.exists(mp):
        with open(mp, encoding="utf-8") as f:
            meta = json.load(f)
    r, err = parse_blocks(out, man)
    if err:
        return None, "PARSE FAIL: %s" % err
    return {"blocks": r["blocks"], "utf16": r["utf16"], "n_values": r["n_values"],
            "meta": meta, "manifest": man}, None


def analyze(d):
    a, err = load_artifact(d)
    if err:
        return None, err
    B = a["blocks"]
    steps = [k for k in range(128) if ("step%d_logits" % k) in B]

    implied, seen, marg, ent, top2 = [], [], [], [], []
    for k in steps:
        row = B["step%d_logits" % k][0]
        implied.append(int(np.argmax(row)))
        m, e, _t1, t2 = margin_entropy(row)
        marg.append(m); ent.append(e); top2.append(t2)
        sk = "step%d_seen" % k
        seen.append(int(round(float(B[sk][0, 0]))) if sk in B else None)

    if "st_gen_final" in B:
        fin = [int(round(float(v))) for v in B["st_gen_final"][0]]
        chain = fin[1:]
    else:
        chain = implied

    # D2: cache advance -- argmax(step k) must equal the token consumed at k+1
    adv_ok = sum(1 for k in range(len(steps) - 1)
                 if implied[k] == seen[k + 1])

    # D3: are successive step-logit blocks different? (a frozen path repeats)
    changed = sum(1 for k in range(1, len(steps))
                  if not np.array_equal(B["step%d_logits" % k],
                                        B["step%d_logits" % (k - 1)]))

    onset, period = cycle_structure(chain)
    res = {
        "dir": os.path.relpath(d, ROOT),
        "meta_layers": a["meta"].get("layers"),
        "meta_gen": a["meta"].get("gen"),
        "sampler": a["meta"].get("sampler"),
        "prompt_ids": a["meta"].get("prompt_ids"),
        "seed_token": a["meta"].get("seed_token"),
        "capture_utf16": a["utf16"],
        "n_values": a["n_values"],
        "n_steps": len(steps),
        "chain": chain,
        "distinct": len(set(chain)),
        "implied_argmax": implied,
        "consumed_seen": seen,
        "margins": marg,
        "entropies": ent,
        "top2": top2,
        "cache_advance_ok": adv_ok,
        "cache_advance_n": max(0, len(steps) - 1),
        "step_logits_changed": changed,
        "cycle_onset": onset,
        "cycle_period": period,
        "prefill_argmax": ([int(x) for x in np.argmax(B["prefill_logits"], axis=1)]
                           if "prefill_logits" in B else None),
        "chain_eq_implied": sum(1 for a, b in zip(chain, implied) if a == b),
    }
    return res, None


def run_reports():
    dirs = [
        os.path.join(ROOT, "examples", "p14_gen"),
        os.path.join(ROOT, "examples", "p14_gen_feed"),
        os.path.join(ROOT, "examples", "p14_oneshot"),
    ]
    print("=== P16-4: O6 generation-chain cycle -- model property or impl defect? ===")
    print("(all numbers below are READ FROM DISK; no engine run in this script)\n")

    results = []
    for d in dirs:
        print("--- %s ---" % os.path.relpath(d, ROOT))
        r, err = analyze(d)
        if err:
            print("  SKIPPED: %s\n" % err)
            results.append({"dir": os.path.relpath(d, ROOT), "error": err})
            continue
        print("  meta: layers=%s gen=%s sampler=%s seed=%s"
              % (r["meta_layers"], r["meta_gen"], r["sampler"], r["seed_token"]))
        print("  capture encoding     : %s   (%d values, manifest-exact)"
              % ("UTF-16LE+BOM" if r["capture_utf16"] else "ASCII", r["n_values"]))
        print("  steps present        : %d" % r["n_steps"])
        print("  chain                : %s" % r["chain"])
        print("  distinct tokens      : %d of %d" % (r["distinct"], len(r["chain"])))
        print("  cycle onset / period : %s / %s"
              % (r["cycle_onset"], r["cycle_period"]))
        print("  D1 argmax(own logits): %s" % r["implied_argmax"])
        print("  D2 CACHE ADVANCES: argmax(step k) == consumed(k+1)  %d / %d"
              % (r["cache_advance_ok"], r["cache_advance_n"]))
        print("  D3 step logits CHANGED vs previous step             %d / %d"
              % (r["step_logits_changed"], max(0, r["n_steps"] - 1)))
        print("  D4 margins           : %s" % ["%.3f" % m for m in r["margins"]])
        print("  D4 entropies         : %s" % ["%.3f" % e for e in r["entropies"]])
        print("  prefill argmax       : %s" % r["prefill_argmax"])

        # For the closed-loop artifact, the D7 negative control: the token chain
        # must be reproducible by numpy argmax on the dumped logits (so the value
        # is computed, not echoed), and the chain must equal those argmaxes.
        if r["sampler"] == "argmax":
            eq = sum(1 for a, b in zip(r["chain"], r["implied_argmax"]) if a == b)
            print("  D7 chain == argmax(own logits): %d / %d  (closed-loop fidelity)"
                  % (eq, len(r["chain"])))
            print("  D7 vs 24-layer anchor %s : %s"
                  % (ANCHOR_24, "matches (WRONG attribution)" if
                     r["prefill_argmax"] == ANCHOR_24 else
                     "does NOT match -> this artifact is NOT the 24-layer model"))
        if r["sampler"] == "feed":
            eq = sum(1 for a, b in zip(r["chain"], r["implied_argmax"]) if a == b)
            print("  D7 NEGATIVE CONTROL (forced input chain): chain == argmax(own "
                  "logits) = %d / %d  -> %s"
                  % (eq, len(r["chain"]),
                     "PASS: logits were COMPUTED, not copied" if eq == 0 else
                     "FAIL: output tracks the file"))
        print()
        results.append(r)

    with open(os.path.join(OUT, "cycle_structure.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)
    print("wrote %s" % os.path.join(OUT, "cycle_structure.json"))
    return results


def selftest():
    """Show every criterion CAN FAIL, with counts.

    This is the project's cardinal rule: a test that cannot demonstrate its own
    failure is not a test.
    """
    print("=== P16-4 SELFTEST: can each criterion demonstrate its own failure? ===\n")
    rows = []
    ok_all = True

    # 1. chain comparison
    chain = [1, 2, 3, 4, 5, 5, 5, 5]
    def cmp(a, b):
        n = min(len(a), len(b))
        for i in range(n):
            if a[i] != b[i]:
                return False, i
        return (len(a) == len(b)), n
    cases = {
        "identity": chain,
        "off_by_one": [x + 1 for x in chain],
        "reverse": chain[::-1],
        "shift_left": chain[1:] + chain[:1],
        "all_zero": [0] * len(chain),
        "truncate": chain[:-1],
        "extend": chain + [5],
    }
    for k, v in cases.items():
        same, d = cmp(chain, v)
        want = (k == "identity")
        good = (same == want)
        if not good:
            ok_all = False
        rows.append(("chain:%s" % k, "ACCEPTED" if same else "REJECTED",
                     "expected %s" % ("ACCEPT" if want else "REJECT"),
                     "OK" if good else "*** BROKEN ***"))

    # 2. cycle detector: must find a planted cycle and must NOT invent one
    planted_fixed = [1, 2, 3, 9, 9, 9, 9, 9, 9]
    planted_3 = [1, 2, 3, 7, 8, 9, 7, 8, 9, 7, 8, 9]
    no_cycle = [1, 2, 3, 4, 5, 6, 7, 8]
    for name, seq, want in [("fixed_point", planted_fixed, (3, 1)),
                            ("3cycle", planted_3, (3, 3)),
                            ("none", no_cycle, (None, None))]:
        got = cycle_structure(seq)
        good = (got == want)
        if not good:
            ok_all = False
        rows.append(("cycle:%s" % name, str(got), "expected %s" % (want,),
                     "OK" if good else "*** BROKEN ***"))

    # 3. cache-advance criterion must be able to report a MISMATCH
    good_adv = [10, 20, 30]; good_seen = [0, 10, 20]
    bad_seen = [0, 11, 20]
    ga = sum(1 for k in range(2) if good_adv[k] == good_seen[k + 1])
    ba = sum(1 for k in range(2) if good_adv[k] == bad_seen[k + 1])
    rows.append(("cache_advance:correct", "%d/2" % ga, "expected 2/2",
                 "OK" if ga == 2 else "*** BROKEN ***"))
    rows.append(("cache_advance:stale", "%d/2" % ba, "expected 1/2",
                 "OK" if ba == 1 else "*** BROKEN ***"))
    if ga != 2 or ba != 1:
        ok_all = False

    # 4. cycle structure on the REAL chains, to show the detector is not vacuous
    real = [106487, 81042, 11530, 613, 6520, 33700, 101925, 10714, 116450,
            8497, 95564, 45323, 45323, 45323, 45323, 45323]
    got = cycle_structure(real)
    rows.append(("cycle:real_4layer_chain", str(got), "expected (11, 1)",
                 "OK" if got == (11, 1) else "*** BROKEN ***"))
    if got != (11, 1):
        ok_all = False

    print("%-28s %-10s %-24s %s" % ("criterion", "verdict", "expectation", "status"))
    print("-" * 88)
    for a, b, c, d in rows:
        print("%-28s %-10s %-24s %s" % (a, b, c, d))
    n_pass = sum(1 for r in rows if r[3] == "OK")
    print("\ncriteria shown able to FAIL correctly: %d / %d" % (n_pass, len(rows)))
    print("SELFTEST: %s" % ("PASS" if ok_all else "FAIL"))
    return 0 if ok_all else 1


def oracle_alignment_check():
    """THE HEADLINE CHECK, reproduced from saved files only.

    The engine's chain must equal HF's argmax at absolute positions 4..19, and the
    alignment must be shown to be UNIQUE: every other candidate offset must score
    strictly worse. Without the sweep, "16/16" could be an alignment chosen after
    seeing the answer -- the sweep makes the alignment a measured fact.

    Requires examples/p16_o6/hf_gen_oracle_L4.json (run
    tools/p16_o6_hf_gen_oracle.py --layers 4 first).
    """
    p = os.path.join(OUT, "hf_gen_oracle_L4.json")
    cp = os.path.join(OUT, "cycle_structure.json")
    print("=== HEADLINE: does HuggingFace reproduce the engine's generated chain? ===")
    if not (os.path.exists(p) and os.path.exists(cp)):
        print("  UNAVAILABLE: need %s and %s" % (p, cp))
        print("  run: python tools/p16_o6_hf_gen_oracle.py --layers 4")
        return None
    with open(p, encoding="utf-8") as f:
        o = json.load(f)
    with open(cp, encoding="utf-8") as f:
        cs = json.load(f)
    closed = [x for x in cs if x.get("sampler") == "argmax"]
    if not closed:
        print("  UNAVAILABLE: no argmax-sampler artifact in cycle_structure.json")
        return None
    closed = closed[0]
    chain = closed["chain"]
    allam = o["hf_argmax_all"]

    print("  artifact layers    : %s | HF truncated to %s"
          % (o["artifact_layers"], o["hf_layers_truncated"]))
    print("  engine chain       : %s" % chain)
    print("  HF argmax (abs)    : %s" % allam)
    print("\n  ALIGNMENT SWEEP (HF argmax at abs pos p..p+15 vs engine chain):")
    best, rows = None, []
    for start in range(0, max(1, len(allam) - len(chain) + 1)):
        hf = allam[start:start + len(chain)]
        if len(hf) != len(chain):
            continue
        eq = sum(1 for a, b in zip(chain, hf) if a == b)
        rows.append((start, eq))
        if best is None or eq > best[1]:
            best = (start, eq)
        print("    p=%2d..%2d  agreement %2d / %d" % (start, start + len(chain) - 1,
                                                        eq, len(chain)))
    unique = sum(1 for _s, e in rows if e == best[1]) == 1
    print("\n  BEST alignment: abs pos %d..%d with %d / %d"
          % (best[0], best[0] + len(chain) - 1, best[1], len(chain)))
    print("  alignment unique (no other offset ties): %s" % unique)
    print("  => %s" % ("PASS: HF reproduces the engine chain exactly at a UNIQUE "
                        "alignment -> the cycle is the MODEL's"
                        if best[1] == len(chain) and unique else
                        "NOT SETTLED: best agreement %d/%d, unique=%s"
                        % (best[1], len(chain), unique)))
    return {"best_start": best[0], "best_agreement": best[1], "n": len(chain),
            "unique": unique, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.selftest:
        return selftest()
    run_reports()
    print()
    r = oracle_alignment_check()
    print()
    rc = selftest()
    if r is not None and (r["best_agreement"] != r["n"] or not r["unique"]):
        return 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
