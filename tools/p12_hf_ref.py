#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P12: HuggingFace reference for the SAME truncated model the engine runs.

Why a separate oracle rather than reusing p10's layer-0 dump:
  P10 validated ONE layer with S=4 using the layer-0 hook. P12 now runs
  N layers (configurable) as a full program, so the reference must be the
  full stack of N layers plus the final norm -- otherwise the comparison
  would be against the wrong thing.

Approach: build a model with num_hidden_layers = N (same weights, just
truncated) and run the real HF forward. That keeps the oracle honest: no
hand-reimplementation of the layer stack.

Usage:
    python tools/p12_hf_ref.py --layers 1 --seq 4
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

MODEL = os.environ.get("MINICPM_DIR", "<外部模型目录>/MiniCPM5-1B")
OUT = r"<repo>/examples/minicpm_full"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--seq", type=int, default=4)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--full", action="store_true",
                    help="use all 24 layers and also dump lm_head logits")
    args = ap.parse_args()

    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(args.model)
    nlayers = cfg.num_hidden_layers if args.full else args.layers
    cfg.num_hidden_layers = nlayers
    print("HF reference: layers=%d seq=%d" % (nlayers, args.seq))

    model = AutoModelForCausalLM.from_pretrained(
        args.model, config=cfg, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()
    print("loaded (f32 oracle; the engine is f32 maths on bf16 weights)")

    ids = torch.tensor([[3, 117, 2501, 88][: args.seq]], dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)

    # hidden_states[-1] is AFTER the final norm in HF's convention; the engine
    # applies `final_norm` explicitly and prints that, so use the last entry.
    h_final = out.hidden_states[-1].numpy().astype(np.float32)   # [1,S,H]
    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "hf_final_hidden.npy"), h_final[0])
    print("saved hf_final_hidden.npy  shape=%s" % (h_final[0].shape,))

    if args.full:
        logits = out.logits.numpy().astype(np.float32)           # [1,S,V]
        np.save(os.path.join(args.out, "hf_logits.npy"), logits[0])
        print("saved hf_logits.npy  shape=%s" % (logits[0].shape,))

    meta = {
        "layers": nlayers, "seq": args.seq, "input_ids": ids.tolist(),
        "full": bool(args.full),
    }
    with open(os.path.join(args.out, "hf_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("saved hf_meta.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
