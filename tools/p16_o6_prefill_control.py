#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P16-4 (task-10) STEP 3: THE PREFILL CONTROL + THE POSITION-INDEPENDENCE TEST.

Two independent jobs:

(A) PREFILL CONTROL -- guard against an off-by-one in step 1.
    Step 1 fed HF the 20-token sequence and compared HF positions 0..3 against
    the recorded engine prefill [5,242,90,82591] taken from a SEPARATE 4-token
    run. Those are different contexts (a 20-token sequence's position 0..3 has
    the same causal prefix, so they SHOULD agree -- but "should" is not a
    measurement). This script runs HF on the 4-token prompt ALONE and checks the
    4/4 anchor directly. If it disagrees, the step-1 prefill comparison was
    invalid, not the engine.

(B) POSITION / SEED INDEPENDENCE -- the H-impl discriminator.
    A RoPE/position bug or a stale cache would make the collapse depend on WHERE
    the context sits or WHICH seed is used. A genuine model attractor is a
    property of the (context, token) pair. We test:
      * the 4-layer model fed the SAME prompt but with the generated tail
        EXTENDED (longer context -> if the repeat is position-dependent it will
        move),
      * and we re-run HF on a DIFFERENT seed token in place of 88.

Usage:
    python tools/p16_o6_prefill_control.py --layers 4
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
PROMPT = [3, 117, 2501, 88]
ENGINE_PREFILL_ARGMAX = [5, 242, 90, 82591]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", default=os.path.join(ROOT, "examples", "p16_o6"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(args.model)
    cfg.num_hidden_layers = int(args.layers)
    cfg.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(
        args.model, config=cfg, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()
    print("=== P16-4 STEP 3: prefill control + position/seed independence ===")
    print("HF truncated to %d layers; loaded in %.1f s\n" % (args.layers, time.time() - t0))

    def hf_argmax(ids, label):
        t = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            o = model(input_ids=t, use_cache=False)
        lg = o.logits.numpy().astype(np.float32)[0]
        am = [int(x) for x in np.argmax(lg, axis=1)]
        print("  %-42s -> argmax %s" % (label, am))
        return am, lg

    # ---------- (A) prefill control ----------
    print("(A) PREFILL CONTROL: does HF on the BARE 4-token prompt give 4/4?")
    print("    prompt                : %s" % PROMPT)
    am4, _ = hf_argmax(PROMPT, "HF argmax of bare 4-token prompt")
    print("    engine record (P12/P13): %s" % ENGINE_PREFILL_ARGMAX)
    n_match = sum(1 for a, b in zip(am4, ENGINE_PREFILL_ARGMAX) if a == b)
    print("    match                 : %d / 4" % n_match)
    print("    => %s\n" % ("PASS: the P12/P13 HF anchor reproduces exactly"
                           if n_match == 4 else
                           "FAIL: the recorded anchor does NOT reproduce"))

    # ---------- (B) position / seed independence ----------
    print("(B) POSITION & SEED INDEPENDENCE (H-impl discriminator)")
    print("    If the repeat were caused by a RoPE/position bug, it would")
    print("    change with context length or seed. A model attractor should")
    print("    reproduce under teacher forcing from HF itself.\n")

    # B1: HF teacher-forced on the ENGINE's 20-token sequence (from step 1) --
    #     already measured; re-derive here so this script is self-contained.
    seq20 = [3, 117, 2501, 88, 88, 106487, 81042, 11530, 613, 6520, 33700,
             101925, 10714, 116450, 8497, 95564, 45323, 45323, 45323, 45323]
    am20, lg20 = hf_argmax(seq20, "HF teacher-forced on 20-token seq")
    print("      HF pos 4..19      : %s" % am20[4:])

    # B2: the model's OWN greedy rollout from the bare prompt, no teacher forcing.
    print("\n    B2: HF's own greedy rollout from the bare prompt (no forcing)")
    cur = list(PROMPT)
    roll = []
    for step in range(16):
        t = torch.tensor([cur], dtype=torch.long)
        with torch.no_grad():
            o = model(input_ids=t, use_cache=False)
        nxt = int(np.argmax(o.logits.numpy()[0, -1]))
        roll.append(nxt)
        cur.append(nxt)
    print("      HF greedy rollout : %s" % roll)
    engine_chain = [106487, 81042, 11530, 613, 6520, 33700, 101925, 10714,
                    116450, 8497, 95564, 45323, 45323, 45323, 45323, 45323]
    print("      engine chain      : %s" % engine_chain)
    # NOTE: the engine seeds with 88 (the last prompt token) then generates;
    # HF's rollout continues from the prompt's own argmax (5). So compare the
    # TAIL, and compare HF(forced) vs engine (the like-for-like comparison).
    print("      HF rollout tail   : %s" % roll[-6:])
    print("      engine  tail      : %s" % engine_chain[-6:])
    tail_same = roll[-6:] == engine_chain[-6:]
    print("      tail identical    : %s" % tail_same)

    # B3: different seed in the same slot
    print("\n    B3: replace the seed token 88 (position 4) with other values")
    seed_rows = []
    for seed in [88, 5, 100, 45323, 106487]:
        s = [3, 117, 2501, 88, seed]
        am, _ = hf_argmax(s, "seed=%-7d at position 4" % seed)
        seed_rows.append({"seed": seed, "argmax_last": am[-1]})
    print("      (argmax at the position AFTER the seed)")

    summary = {
        "hf_layers": int(args.layers),
        "prompt": PROMPT,
        "engine_prefill_record": ENGINE_PREFILL_ARGMAX,
        "hf_bare_prompt_argmax": am4,
        "prefill_match": n_match,
        "hf_forced_20_argmax": am20,
        "hf_greedy_rollout": roll,
        "engine_chain": engine_chain,
        "rollout_tail6": roll[-6:],
        "engine_tail6": engine_chain[-6:],
        "tail6_identical": bool(tail_same),
        "seed_sweep": seed_rows,
        "elapsed_s": round(time.time() - t0, 1),
    }
    p = os.path.join(args.out, "hf_prefill_and_seed.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    print("\nwrote %s" % p)
    return 0 if n_match == 4 else 1


if __name__ == "__main__":
    sys.exit(main())
