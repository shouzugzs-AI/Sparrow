#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P16-4 (task-10) STEP 2: ENGINE-SIDE CYCLE STRUCTURE AND DISCRIMINATORS.

Reads the ALREADY-EXISTING engine stdout dumps (no engine rebuild, no re-run of a
GB-scale artifact) and extracts, for every generation step:

  * the argmax of that step's own logits, recomputed HERE in numpy (independent
    of the engine's in-engine TOPK_IDX),
  * the token the engine actually CONSUMED (step{k}_seen), which proves whether
    the feedback loop advanced or re-used a stale token,
  * top1-top2 MARGIN and softmax ENTROPY at each step (H-model discriminator:
    a genuinely peaked distribution supports the model; near-tied logits or a
    degenerate shape support an implementation defect),
  * the cycle structure: onset step, detected period, whether it is a fixed
    point or a longer cycle.

WHY THIS IS THE RIGHT INSTRUMENT
    The signature "every step identical from some point on" is exactly what a
    STALE cache slot or a frozen activation region produces. But it is ALSO what
    a genuine greedy fixed point produces. The discriminator is whether the
    engine's step-k computation CONSUMES step-k-1's argmax and whether the logits
    CHANGE across steps -- if the logits keep changing while the argmax keeps
    returning the same token, the cache is advancing and the token is a real
    attractor.

Usage:
    python tools/p16_o6_engine_steps.py --dir examples/p14_gen
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
sys.path.insert(0, os.path.join(ROOT, "tools"))


def decode_engine_stdout(out_bytes):
    """Decode an engine stdout capture, tolerating the UTF-16LE capture trap.

    MEASURED, and it matters: examples/p14_gen/_engine_out.txt and
    examples/p14_gen_feed/_engine_out.txt both start with the UTF-16LE BOM
    (\\xff\\xfe). tools/p14_gen_run.ps1 documents this exact trap at line 194 and
    claims to hard-fail with exit 7 if it happens -- yet these two P14 headline
    dumps have the BOM. Decoding them as ASCII yields ZERO values and a
    misleading "stdout too short" error, which is how a whole cycle gets lost.

    The CONTENT is plain ASCII text, just UTF-16LE-encoded, so the byte length is
    ~2x and the values are recoverable exactly. The manifest total is asserted by
    the caller, so a mis-decode cannot pass silently.
    """
    if out_bytes[:2] == b"\xff\xfe":
        txt = out_bytes.decode("utf-16-le", "replace")
        if txt and txt[0] == "\ufeff":
            txt = txt[1:]
        return txt, True
    return out_bytes.decode("ascii", "replace"), False


def parse_blocks(out_bytes, manifest):
    """Reuse of tools/p14_verify_o6.py::parse_blocks, plus the encoding fix.

    Same semantics otherwise: parse from the MANIFEST, never from position
    arithmetic, and refuse to guess an unaccounted remainder.
    """
    text, was_utf16 = decode_engine_stdout(out_bytes)
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
            return None, ("block %r needs %d values at offset %d, but stdout "
                          "only had %d" % (b["name"], n, idx, len(vals)))
        got[b["name"]] = np.array(vals[idx:idx + n], dtype=np.float32).reshape(
            b["rows"], b["cols"])
        idx += n
    if idx != len(vals):
        return None, ("stdout had %d values but the manifest accounts for %d "
                      "-- refusing to guess the remainder" % (len(vals), idx))
    got["_capture_utf16"] = was_utf16
    return got, None


def cycle_structure(seq):
    """Detect onset and period of the terminal cycle of `seq`.

    Returns (onset, period) where seq[onset:] is the eventual cycle, or
    (None, None) if no repetition was found. Searched exhaustively over all
    (onset, period) with period <= len/2 -- no heuristic, so an absence of a
    cycle is a real absence rather than a missed detection.
    """
    n = len(seq)
    for start in range(n):
        for p in range(1, (n - start) // 2 + 1):
            body = seq[start:]
            if all(body[i] == body[i % p] for i in range(len(body))):
                # `start` must be the EARLIEST onset: re-check from 0 is implied
                # by the loop order; require the cycle to be a full repetition.
                if len(body) >= 2 * p:
                    return start, p
    return None, None


def entropy_margin(row):
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


def analyze(d, label):
    man_p = os.path.join(d, "gen_rows.json")
    out_p = os.path.join(d, "_engine_out.txt")
    meta_p = os.path.join(d, "meta.json")
    if not (os.path.exists(man_p) and os.path.exists(out_p)):
        return None, "missing gen_rows.json or _engine_out.txt"

    with open(man_p, encoding="utf-8") as f:
        man = json.load(f)
    with open(out_p, "rb") as f:
        out = f.read()
    meta = {}
    if os.path.exists(meta_p):
        with open(meta_p, encoding="utf-8") as f:
            meta = json.load(f)

    blocks, err = parse_blocks(out, man)
    if err:
        return None, "PARSE FAIL: %s" % err

    steps = [k for k in range(128) if ("step%d_logits" % k) in blocks]
    res = {
        "dir": d, "label": label,
        "capture_utf16": bool(blocks.get("_capture_utf16")),
        "layers": meta.get("layers"),
        "gen": meta.get("gen"),
        "prompt_ids": meta.get("prompt_ids"),
        "seed_token": meta.get("seed_token"),
        "n_steps": len(steps),
        "has_st_gen_final": "st_gen_final" in blocks,
    }

    implied, seen, margins, ents, top1, top2 = [], [], [], [], [], []
    for k in steps:
        lg = blocks["step%d_logits" % k]
        a = int(np.argmax(lg[0]))
        implied.append(a)
        m, e, i1, i2 = entropy_margin(lg[0])
        margins.append(m); ents.append(e); top1.append(i1); top2.append(i2)
        sk = "step%d_seen" % k
        if sk in blocks:
            seen.append(int(round(float(blocks[sk][0, 0]))))
        else:
            seen.append(None)

    res["implied_argmax"] = implied
    res["consumed_seen"] = seen
    res["margins"] = margins
    res["entropies"] = ents
    res["top2"] = top2
    res["top1"] = top1

    if res["has_st_gen_final"]:
        fin = [int(round(float(v))) for v in blocks["st_gen_final"][0]]
        res["st_gen_final"] = fin
        # the generated chain = positions 1.. of st_gen (position 0 is the seed)
        chain = fin[1:]
        res["chain"] = chain
    else:
        res["chain"] = implied

    # ---- independent recompute of argmax vs what the engine consumed ----
    # step k logits predict the token consumed at step k+1.
    ok_advance = 0
    adv_rows = []
    for k in range(len(steps) - 1):
        pred = implied[k]
        nxt = seen[k + 1]
        same = (pred == nxt)
        ok_advance += int(same)
        adv_rows.append((k, pred, nxt, same))
    res["advance_rows"] = adv_rows
    res["advance_ok"] = ok_advance
    res["advance_n"] = max(0, len(steps) - 1)

    onset, period = cycle_structure(res["chain"])
    res["cycle_onset"] = onset
    res["cycle_period"] = period
    res["distinct"] = len(set(res["chain"]))
    return res, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "examples", "p16_o6"))
    args = ap.parse_args()
    dirs = args.dir or [
        os.path.join(ROOT, "examples", "p14_gen"),
        os.path.join(ROOT, "examples", "p14_gen_feed"),
        os.path.join(ROOT, "examples", "p14", "argmax"),
        os.path.join(ROOT, "examples", "p14", "feed"),
    ]
    os.makedirs(args.out, exist_ok=True)

    print("=== P16-4 STEP 2: engine step-by-step analysis (existing dumps) ===\n")
    all_res = []
    for d in dirs:
        rel = os.path.relpath(d, ROOT)
        print("--- %s ---" % rel)
        res, err = analyze(d, rel)
        if err:
            print("  SKIPPED: %s\n" % err)
            all_res.append({"dir": rel, "error": err})
            continue
        print("  layers=%s  gen=%s  prompt=%s  seed=%s"
              % (res["layers"], res["gen"], res["prompt_ids"], res["seed_token"]))
        print("  steps present      : %d" % res["n_steps"])
        print("  chain              : %s" % res["chain"])
        print("  distinct           : %d of %d" % (res["distinct"], len(res["chain"])))
        print("  cycle onset/period : %s / %s" % (res["cycle_onset"], res["cycle_period"]))
        print("  consumed token/step: %s" % res["consumed_seen"])
        print("  argmax(own logits) : %s" % res["implied_argmax"])
        print("  CACHE-ADVANCE CHECK: argmax(step k) == consumed(step k+1): %d / %d"
              % (res["advance_ok"], res["advance_n"]))
        print("  margins            : %s" % ["%.3f" % m for m in res["margins"]])
        print("  entropies          : %s" % ["%.3f" % e for e in res["entropies"]])

        # Are successive step logits IDENTICAL? A stale cache/activation would
        # make them so; a live model moves them.
        steps = [k for k in range(res["n_steps"])]
        blk = None
        man = json.load(open(os.path.join(d, "gen_rows.json"), encoding="utf-8"))
        out = open(os.path.join(d, "_engine_out.txt"), "rb").read()
        blocks, _ = parse_blocks(out, man)
        if blocks:
            prev = None
            ndiff = 0
            for k in steps:
                cur = blocks["step%d_logits" % k].astype(np.float32)
                if prev is not None:
                    if not np.array_equal(cur, prev):
                        ndiff += 1
                prev = cur
            print("  step logits that CHANGE vs previous step: %d / %d"
                  % (ndiff, max(0, len(steps) - 1)))
            res["logits_change_count"] = ndiff
        print()
        all_res.append(res)

    p = os.path.join(args.out, "engine_steps.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(all_res, f, indent=2, default=float)
    print("wrote %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
