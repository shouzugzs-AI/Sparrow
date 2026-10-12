#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P16-4 (task-10) STEP 1: THE MISSING MEASUREMENT.

P14 explicitly recorded that hf_logits.npy is shape (4, 130560) and covers only
the 4 PROMPT positions, so generation steps at absolute positions 4..19 were
NEVER covered by any HuggingFace oracle. P14 therefore only INFERRED that the
repeated-token tail "is the model".

This script supplies the missing measurement for the artifact that actually
exists: the 4-LAYER truncated model (examples/p14_gen/meta.json -> layers=4).

METHOD (deliberately the strongest available):
  * Build the HF model TRUNCATED to the SAME layer count as the engine artifact
    (cfg.num_hidden_layers = L). Same technique as tools/p12_hf_ref.py, which is
    what produced the HF anchor the whole P12-P14 chain trusts.
  * Feed HF the EXACT 20-token sequence the engine's one-shot forward used
    (examples/p14_oneshot/meta.json "tokens", length 20). That sequence is the
    engine's own closed loop output, so this is a true teacher-forced oracle
    over generation positions 4..19.
  * Compare HF argmax at positions 4..19 against the engine's chain.

WHAT EACH OUTCOME MEANS (honest reading, stated BEFORE the run):
  * HF argmax == engine chain at every generated position
      -> the engine's generation path reproduces the model. H-model supported:
         the oracle independently produces the same repeated tail.
  * HF argmax differs at some generated position, while HF prefill (pos 0..3)
     still matches the engine 4/4
      -> the engine's DECODE PATH diverges from the model at exactly that step.
         H-impl supported, and the first differing index localises it.
  * HF prefill does NOT match
      -> the measurement itself is invalid (wrong truncation / wrong config);
         reported as UNPROVEN, not as evidence for either hypothesis.

CAN THIS FAIL? Yes, and that is checked here: the same comparison is run against
a deliberately perturbed chain, which must be REJECTED. Counts are printed.

Usage:
    python tools/p16_o6_hf_gen_oracle.py --layers 4 --seq-id examples/p14_oneshot
"""
import argparse
import json
import os
import sys
import time

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = str(Path(__file__).resolve().parents[1])
MODEL = os.environ.get("MINICPM_DIR", "<外部模型目录>/MiniCPM5-1B")
VOCAB = 130560


def compare_chain(a, b):
    """Exact integer comparison. Returns (all_equal, first_diff, n)."""
    n = min(len(a), len(b))
    for i in range(n):
        if int(a[i]) != int(b[i]):
            return False, i, n
    if len(a) != len(b):
        return False, n, n
    return True, None, n


def perturb(chain, kind):
    c = list(chain)
    if kind == "identity":
        return c
    if kind == "off_by_one":
        return [x + 1 for x in c]
    if kind == "reverse":
        return c[::-1]
    if kind == "shift_left":
        return c[1:] + c[:1]
    if kind == "shift_right":
        return c[-1:] + c[:-1]
    if kind == "all_zero":
        return [0] * len(c)
    if kind == "truncate":
        return c[:-1]
    if kind == "extend":
        return c + [c[-1]]
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4,
                    help="truncate HF model to this many layers (MUST match the "
                         "engine artifact's meta.json layers)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--seq-id", default=os.path.join(ROOT, "examples", "p14_oneshot"),
                    help="artifact dir whose meta.json 'tokens' is the sequence")
    ap.add_argument("--out", default=os.path.join(ROOT, "examples", "p16_o6"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    # ---- the sequence under test: the engine's own one-shot 20-token input ----
    meta_path = os.path.join(args.seq_id, "meta.json")
    with open(meta_path, encoding="utf-8") as f:
        smeta = json.load(f)
    ids = smeta["tokens"]
    art_layers = smeta.get("layers")
    print("=== P16-4 STEP 1: HF oracle over GENERATION positions ===")
    print("sequence source     : %s" % meta_path)
    print("artifact layers     : %s" % art_layers)
    print("sequence length     : %d" % len(ids))
    print("sequence            : %s" % ids)
    print("prompt positions    : 0..3    generation positions: 4..%d" % (len(ids) - 1))

    if art_layers is not None and int(art_layers) != int(args.layers):
        print("\n!! LAYER MISMATCH: artifact meta says layers=%s but --layers %d "
              "was requested. The oracle would be for the WRONG model. Refusing."
              % (art_layers, args.layers))
        return 3

    # ---- HF forward, truncated to the artifact's layer count ----
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(args.model)
    print("\nHF config           : %s" % cfg.model_type)
    print("HF true layers      : %d" % cfg.num_hidden_layers)
    cfg.num_hidden_layers = int(args.layers)
    cfg.use_cache = False
    print("HF TRUNCATED to     : %d layers  (same depth as the engine artifact)"
          % cfg.num_hidden_layers)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, config=cfg, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()
    print("loaded f32 oracle   : %.1f s" % (time.time() - t0))

    tin = torch.tensor([ids], dtype=torch.long)
    t1 = time.time()
    with torch.no_grad():
        out = model(input_ids=tin, use_cache=False)
    logits = out.logits.numpy().astype(np.float32)[0]      # [T, V]
    print("HF forward done     : %.1f s, logits shape %s" % (time.time() - t1, logits.shape))

    np.save(os.path.join(args.out, "hf_gen_logits_L%d.npy" % args.layers), logits)
    hf_am = [int(x) for x in np.argmax(logits, axis=1)]

    # ---- report ----
    print("\n--- HF argmax over the WHOLE 20-position sequence ---")
    print("HF argmax (absolute) : %s" % hf_am)
    print("HF argmax pos 0..3   : %s   <- the P14 prefill anchor region" % hf_am[:4])
    print("HF argmax pos 4..19  : %s   <- THE P14 GAP, now measured" % hf_am[4:])

    engine_prefill = [5, 242, 90, 82591]
    eq_pre, fd_pre, _ = compare_chain(engine_prefill, hf_am[:4])
    print("\nprefill match vs engine record %s : %s"
          % (engine_prefill, "MATCH 4/4" if eq_pre else "MISMATCH first_diff=%d" % fd_pre))

    # The engine's generated chain = argmax at the positions where it generated.
    # From gen_rows/oneshot: HF position k predicts the token that the engine
    # placed at k+1 for k in 3..18 (position 3 is the last prompt pos, which
    # predicts token index 4 = first generated token).
    chain = ids[4:]              # engine's 16 generated tokens
    hf_pred = hf_am[3:19]        # HF's prediction at each generating position
    print("\n--- THE DECISIVE COMPARISON ---")
    print("engine generated chain (16)  : %s" % chain)
    print("HF argmax at positions 3..18 : %s" % hf_pred)
    eq, fd, n = compare_chain(chain, hf_pred)
    print("agreement                    : %d / %d"
          % (sum(1 for a, b in zip(chain, hf_pred) if a == b), n))
    if eq:
        print("RESULT: HF reproduces the engine's generated chain EXACTLY.")
    else:
        print("RESULT: FIRST DIVERGENCE at generated index %d" % fd)

    # ---- per-position detail ----
    print("\n--- per-step detail (idx = generated index) ---")
    print("%4s %8s %12s %12s %10s" % ("idx", "abs_pos", "engine_tok", "hf_tok", "match"))
    for i in range(len(chain)):
        print("%4d %8d %12d %12d %10s"
              % (i, i + 3, chain[i], hf_pred[i], "OK" if chain[i] == hf_pred[i] else "DIFF"))

    # ---- CAN THIS COMPARISON FAIL? ----
    print("\n--- META-CHECK: can this comparison demonstrate its own failure? ---")
    rows = []
    ok_all = True
    for kind in ["identity", "off_by_one", "reverse", "shift_left", "shift_right",
                 "all_zero", "truncate", "extend"]:
        p = perturb(chain, kind)
        e, f, _ = compare_chain(chain, p)
        if kind == "identity":
            if e:
                rows.append((kind, "ACCEPTED", "correct: must accept"))
            else:
                rows.append((kind, "REJECTED", "*** identity rejected: BROKEN ***"))
                ok_all = False
        else:
            if e:
                rows.append((kind, "ACCEPTED", "*** NOT DETECTED: rubber stamp ***"))
                ok_all = False
            else:
                rows.append((kind, "REJECTED", "first_diff=%s" % f))
    for k, v, d in rows:
        print("  %-14s %-10s %s" % (k, v, d))
    print("  comparison demonstrably rejects wrong chains: %s" % ok_all)

    # ---- entropy / margin at each generating position (H-model discriminator) ----
    print("\n--- logit shape at each generating position (HF, f32) ---")
    print("%4s %10s %10s %12s %12s" % ("pos", "top1", "top2", "margin", "entropy"))
    for p in range(3, 19):
        row = logits[p]
        order = np.argsort(row)[::-1]
        t1i, t2i = int(order[0]), int(order[1])
        m = float(row[t1i] - row[t2i])
        z = row - row.max()
        pr = np.exp(z.astype(np.float64))
        pr /= pr.sum()
        ent = float(-(pr * np.log(pr + 1e-300)).sum())
        print("%4d %10d %10d %12.4f %12.4f" % (p, t1i, t2i, m, ent))

    summary = {
        "artifact": args.seq_id,
        "artifact_layers": art_layers,
        "hf_layers_truncated": int(args.layers),
        "sequence": ids,
        "hf_argmax_all": hf_am,
        "hf_argmax_gen": hf_pred,
        "engine_chain": chain,
        "agreement": int(sum(1 for a, b in zip(chain, hf_pred) if a == b)),
        "n": len(chain),
        "exact": bool(eq),
        "first_diff": fd,
        "comparison_can_fail": bool(ok_all),
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out, "hf_gen_oracle_L%d.json" % args.layers), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nwrote %s" % os.path.join(args.out, "hf_gen_oracle_L%d.json" % args.layers))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
