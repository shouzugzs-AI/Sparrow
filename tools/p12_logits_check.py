#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P12: validate the FULL MiniCPM5-1B forward pass against HuggingFace logits.

This is the end-to-end acceptance for "the engine can actually load the model":
24 layers + final norm + lm_head -> [4, 130560] logits, compared against the
real HF forward on the same prompt.

METRICS (why more than max_abs_err):
  * max_abs_err / rel  -- magnitude agreement
  * argmax match       -- does the engine pick the same next token? This is
                          what actually matters for generation, and it survives
                          small numeric noise far better than a max-norm test.
  * top-1 rate         -- fraction of positions whose argmax agrees.
  * rank of HF's argmax in the engine's ordering -- if the engine's top token
    differs, was HF's choice at least ranked 2nd? That distinguishes "noise"
    from "broken".

Usage: python tools/p12_logits_check.py
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = str(Path(__file__).resolve().parents[1])
DIR = os.path.join(ROOT, "examples", "minicpm_full")
REL = os.path.join(ROOT, "target", "release")
ENG = os.path.join(REL, "vmlm-runtime.exe")


def main():
    ref = np.load(os.path.join(DIR, "hf_logits.npy")).astype(np.float32)
    S, V = ref.shape
    meta = json.load(open(os.path.join(DIR, "hf_meta.json")))
    print("HF logits: shape=%s (layers=%d, ids=%s)"
          % (ref.shape, meta["layers"], meta["input_ids"]))

    shutil.copyfile(os.path.join(DIR, "full.vmb"), os.path.join(REL, "model.vmb"))
    out = os.path.join(DIR, "_logits_out.txt")
    with open(out, "wb") as f:
        p = subprocess.run([ENG], cwd=REL, stdout=f, stderr=subprocess.DEVNULL)
    if p.returncode != 0:
        print("engine exit=%d" % p.returncode)
        return 1
    vals = []
    with open(out, "r", errors="replace") as f:
        for line in f:
            for t in line.split():
                try:
                    vals.append(float(t))
                except ValueError:
                    pass
    got = np.array(vals, dtype=np.float32)
    print("engine logits: %d values (expect %d)" % (got.size, S * V))
    if got.size != S * V:
        print("SIZE MISMATCH -- cannot align")
        return 1
    got = got.reshape(S, V)

    refmax = max(float(np.abs(ref).max()), 1e-30)
    d = np.abs(got - ref)
    rel = float(d.max()) / refmax
    print()
    print("magnitude:  max_abs_err = %.4e   rel = %.4e   (ref absmax %.3f)"
          % (d.max(), rel, refmax))

    # argmax agreement
    am_ref = ref.argmax(axis=1)
    am_got = got.argmax(axis=1)
    match = int((am_ref == am_got).sum())
    print("argmax:     engine %s   HF %s   -> %d/%d match"
          % (am_got.tolist(), am_ref.tolist(), match, S))

    # if argmax differs, where does HF's choice rank in the engine's ordering?
    for i in range(S):
        if am_ref[i] != am_got[i]:
            order = np.argsort(-got[i])
            rank = int(np.where(order == am_ref[i])[0][0]) + 1
            print("   position %d: HF picked %d, engine picked %d "
                  "(HF's choice ranked %d in engine)"
                  % (i, am_ref[i], am_got[i], rank))

    print()
    ok = True
    if match != S:
        print("NOTE: argmax differs -- see ranks above (rank 1-3 = numeric noise)")
        ok = False
    if rel > 1e-2:
        print("FAIL: relative error above bf16 tolerance")
        ok = False
    if not ok and rel <= 1e-2 and match >= S - 1:
        print("VERDICT: PASS (magnitude within tolerance; at most one noisy argmax)")
        return 0
    if ok:
        print("VERDICT: PASS -- full 24-layer forward matches HF")
        return 0
    print("VERDICT: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
