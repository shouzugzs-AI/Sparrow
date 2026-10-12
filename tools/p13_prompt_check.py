#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P13 task A acceptance: the SAME .vmb must serve ANY prompt.

THE CLAIM BEING TESTED
    Before P13 the prompt was baked into the program: `p12_build_minicpm.py`
    extracted the prompt rows host-side and emitted them as a `prompt_emb`
    constant consumed by `COPY h, prompt_emb`. Changing the prompt therefore
    meant reassembling the whole 1.76 GB artifact (~106 s).
    In `--prompt-mode tokens` the program carries no prompt data at all: token
    ids come from `tokens.bin` into the `st_tok` state slot, and OP_EMB widens
    one bf16 row per token.

WHAT IS ASSERTED (in order of strength)
    1. IDENTITY: with tokens.bin holding the SAME ids the const-mode build
       baked in, tokens-mode must reproduce const-mode's hidden state. Both
       run the identical layer stack over identical weights, so any difference
       means the lookup or the row stacking is wrong. Compared exactly --
       a tolerance would hide a transposed row.
    2. PROMPT CHANGE, NO REASSEMBLY: write DIFFERENT ids to tokens.bin, re-run
       the SAME .vmb, and require the output to change. This is the actual
       deliverable: one artifact, many prompts.
    3. CROSS-CHECK vs HF: the const-mode reference was already validated
       against HuggingFace in P12 (rel 1.88e-06, argmax 4/4), so matching it
       transfers that validation to tokens-mode.

Usage:
    python tools/p13_prompt_check.py [--layers 1]
"""
import argparse
import os
import shutil
import subprocess
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = str(Path(__file__).resolve().parents[1])
REL = os.path.join(ROOT, "target", "release")
ENG = os.path.join(REL, "vmlm-runtime.exe")
ASM = os.path.join(REL, "vmlm-asm.exe")
WORK = os.path.join(ROOT, "examples", "p13_prompt")

IDS_A = [3, 117, 2501, 88]
IDS_B = [90, 12, 7777, 401]


def build(layers, mode, out):
    p = subprocess.run(
        [sys.executable, os.path.join(ROOT, "tools", "p12_build_minicpm.py"),
         "--layers", str(layers), "--seq", "4", "--skip-head",
         "--print-hidden", "--prompt-mode", mode, "--out", out],
        cwd=ROOT, capture_output=True, text=True, errors="replace")
    if p.returncode != 0:
        print("BUILD FAILED (%s)" % mode)
        print((p.stdout + p.stderr)[-2500:])
        return None
    return out


def assemble(d, vmb):
    consts = [l.strip() for l in open(os.path.join(d, "weights.args"),
                                      encoding="utf-8") if l.strip()]
    p = subprocess.run([ASM, "full.vml", vmb] + consts,
                       cwd=d, capture_output=True, text=True, errors="replace")
    if p.returncode != 0 or os.path.exists(os.path.join(d, vmb)) is False:
        print("ASSEMBLE FAILED")
        print((p.stdout + p.stderr)[-2000:])
        return False
    return True


def run(vmb_path, tokens=None):
    """Copy the artifact to target/release/model.vmb and run it.

    stdout goes to a REAL FILE: draining it through a PowerShell pipe made a
    1.14 s run look like ~20 minutes in P12, so all timing/IO here is file-based.
    """
    shutil.copyfile(vmb_path, os.path.join(REL, "model.vmb"))
    tb = os.path.join(REL, "tokens.bin")
    if tokens is None:
        if os.path.exists(tb):
            os.remove(tb)
    else:
        with open(tb, "wb") as f:
            np.array(tokens, dtype="<u4").tofile(f)
    out = os.path.join(REL, "_p13_out.txt")
    with open(out, "wb") as f:
        p = subprocess.run([ENG], cwd=REL, stdout=f, stderr=subprocess.DEVNULL)
    if p.returncode != 0:
        return None, p.returncode
    vals = []
    with open(out, "r", errors="replace") as f:
        for line in f:
            for t in line.split():
                try:
                    vals.append(float(t))
                except ValueError:
                    pass
    return np.array(vals, dtype=np.float32), 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(WORK, exist_ok=True)
    print("=== P13 task A: one .vmb, any prompt (layers=%d) ===\n" % args.layers)

    # --- build both modes from identical weights ---
    print("[1/4] building const-mode (pre-P13 path)...")
    d_const = build(args.layers, "const", os.path.join(WORK, "const"))
    if not d_const or not assemble(d_const, "m.vmb"):
        return 1
    print("[2/4] building tokens-mode (P13 path)...")
    d_tok = build(args.layers, "tokens", os.path.join(WORK, "tok"))
    if not d_tok or not assemble(d_tok, "m.vmb"):
        return 1

    # --- get the const-mode reference (its prompt is IDS_A by construction) ---
    ref, rc = run(os.path.join(d_const, "m.vmb"), tokens=None)
    if ref is None:
        print("const-mode engine exit=%d" % rc)
        return 1
    print("    const-mode hidden state: %d values" % ref.size)

    # --- (1) identity: tokens-mode with IDS_A must equal it ---
    got, rc = run(os.path.join(d_tok, "m.vmb"), tokens=IDS_A)
    if got is None:
        print("tokens-mode engine exit=%d" % rc)
        return 1
    print("\n[3/4] identity check (tokens.bin == the ids const-mode baked in)")
    if got.size != ref.size:
        print("  FAIL: size %d vs %d" % (got.size, ref.size))
        return 1
    nbad = int((got != ref).sum())
    md = float(np.abs(got - ref).max())
    print("  tokens-mode: %d values, differing %d, max|d| = %.3e" % (got.size, nbad, md))
    identity_ok = nbad == 0
    print("  %s" % ("PASS: bit-identical to const-mode"
                    if identity_ok else "FAIL: mismatch (lookup/stacking wrong)"))

    # --- (2) the actual deliverable: new prompt, same artifact ---
    print("\n[4/4] prompt change with NO reassembly (same %s)" % os.path.basename(
        os.path.join(d_tok, "m.vmb")))
    other, rc = run(os.path.join(d_tok, "m.vmb"), tokens=IDS_B)
    if other is None:
        print("  engine exit=%d" % rc)
        return 1
    changed = not np.array_equal(other, ref)
    print("  ids A=%s -> first 6: %s" % (IDS_A, got[:6].tolist()))
    print("  ids B=%s -> first 6: %s" % (IDS_B, other[:6].tolist()))
    print("  %s" % ("PASS: output changed with only a 16-byte tokens.bin write"
                    if changed else "FAIL: output identical -- ids ignored"))

    ok = identity_ok and changed
    print()
    print("VERDICT: %s" % ("PASS -- one artifact serves any prompt" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
