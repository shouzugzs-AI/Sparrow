#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P13 task A FINAL acceptance: 24 layers, prompt fed at runtime, vs HuggingFace.

THE CLAIM
    One 2.161 GB .vmb answers ANY prompt. The artifact carries no prompt data:
    token ids are written to `tokens.bin` (16 bytes for S=4) and read into the
    `st_tok` state slot at load. Before P13 the same model required baking the
    prompt rows into the artifact, i.e. a ~106 s / 1.76 GB reassembly PER
    PROMPT -- that is evaluation scaffolding, not inference.

WHAT IS CHECKED
    1. argmax vs the HF oracle, per position    -- the metric that matters for
       generation, and the one P12 used (4/4 with rel 1.88e-06).
    2. rel error vs HF                          -- magnitude agreement.
    3. A SECOND prompt through the SAME .vmb     -- output must differ, proving
       the ids are actually consumed rather than ignored.
    4. The prompt-change step must not touch the .vmb at all (hash before and
       after), which is the literal statement of "no reassembly".

The HF oracle (hf_logits.npy, S=4, ids [3,117,2501,88]) already exists and was
produced by p12_hf_ref.py for this exact prompt.

Usage: python tools/p13_final_check.py
"""
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = str(Path(__file__).resolve().parents[1])
DIR = os.path.join(ROOT, "examples", "minicpm_p13")
REL = os.path.join(ROOT, "target", "release")
ENG = os.path.join(REL, "vmlm-runtime.exe")
REF_DIR = os.path.join(ROOT, "examples", "minicpm_full")
IDS_A = [3, 117, 2501, 88]


def md5(path, chunk=1 << 22):
    m = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            m.update(b)
    return m.hexdigest()


def run_model(vmb, tokens, out_name):
    """Run the engine with stdout to a REAL FILE (never a PowerShell pipe)."""
    import shutil
    shutil.copyfile(vmb, os.path.join(REL, "model.vmb"))
    tb = os.path.join(REL, "tokens.bin")
    if tokens is None:
        if os.path.exists(tb):
            os.remove(tb)
    else:
        with open(tb, "wb") as f:
            np.array(tokens, dtype="<u4").tofile(f)
    out = os.path.join(DIR, out_name)
    t0 = time.time()
    with open(out, "wb") as f:
        p = subprocess.run([ENG], cwd=REL, stdout=f, stderr=subprocess.DEVNULL)
    dt = time.time() - t0
    if p.returncode != 0:
        return None, p.returncode, dt
    vals = []
    with open(out, "r", errors="replace") as f:
        for line in f:
            for t in line.split():
                try:
                    vals.append(float(t))
                except ValueError:
                    pass
    return np.array(vals, dtype=np.float32), 0, dt


def main():
    vmb = os.path.join(DIR, "full.vmb")
    print("=== P13 task A: 24-layer MiniCPM with a RUNTIME prompt ===\n")
    if not os.path.exists(vmb):
        print("no %s" % vmb)
        return 1
    size_gb = os.path.getsize(vmb) / 1e9
    print("artifact: %.3f GB  (contains embed 401 MB + lm_head 401 MB, no prompt)"
          % size_gb)

    ref = np.load(os.path.join(REF_DIR, "hf_logits.npy")).astype(np.float32)
    S, V = ref.shape
    meta = json.load(open(os.path.join(REF_DIR, "hf_meta.json")))
    print("HF oracle: %s  ids=%s" % (ref.shape, meta["input_ids"]))
    print("tokens.bin for this run: %s -> %d bytes\n" % (IDS_A, 4 * len(IDS_A)))

    # --- (1)(2) run with the HF prompt and compare ---
    h_before = md5(vmb)
    got, rc, dt = run_model(vmb, IDS_A, "_p13_logits_a.txt")
    if got is None:
        print("engine exit=%d" % rc)
        return 1
    print("engine: %d values in %.2f s" % (got.size, dt))
    if got.size != S * V:
        print("SIZE MISMATCH: %d vs %d" % (got.size, S * V))
        return 1
    got = got.reshape(S, V)

    relmax = max(float(np.abs(ref).max()), 1e-30)
    d = np.abs(got - ref)
    rel = float(d.max()) / relmax
    am_ref, am_got = ref.argmax(1), got.argmax(1)
    match = int((am_ref == am_got).sum())
    print()
    print("magnitude:  max_abs_err = %.4e   rel = %.4e   (HF absmax %.3f)"
          % (d.max(), rel, relmax))
    print("argmax:     engine %s   HF %s   -> %d/%d match"
          % (am_got.tolist(), am_ref.tolist(), match, S))

    ok = (match == S) and (rel <= 1e-2)

    # --- (3)(4) second prompt through the SAME artifact ---
    print("\nprompt change, NO reassembly:")
    ids_b = [90, 12, 7777, 401]
    got_b, rc, dt_b = run_model(vmb, ids_b, "_p13_logits_b.txt")
    h_after = md5(vmb)
    if got_b is None:
        print("  engine exit=%d" % rc)
        return 1
    changed = not np.array_equal(got_b.reshape(S, V), got)
    print("  ids A=%s  argmax %s" % (IDS_A, got.argmax(1).tolist()))
    print("  ids B=%s  argmax %s" % (ids_b, got_b.reshape(S, V).argmax(1).tolist()))
    print("  artifact md5 unchanged: %s" % (h_before == h_after))
    print("  %s" % ("PASS: same .vmb, different prompt, no reassembly"
                    if (changed and h_before == h_after)
                    else "FAIL: artifact or output did not behave as claimed"))
    ok = ok and changed and (h_before == h_after)

    print()
    print("VERDICT: %s" % ("PASS -- runtime prompt works at 24 layers"
                           if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
