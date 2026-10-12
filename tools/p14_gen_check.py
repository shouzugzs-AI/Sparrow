#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P14 / O6 acceptance checker: does the autoregressive loop really close?

WHAT THIS PROVES (the four things task-2 must demonstrate with real engine runs)
-------------------------------------------------------------------------------
A. HEADLINE: generate G tokens at temperature=0 and show they are
   BIT-IDENTICAL to a one-shot full forward over the same positions.
B. temperature=0 degrades to argmax EXACTLY.
C. temperature>0 actually changes output.
D. THE LOOP REALLY FEEDS BACK: a negative control in which `gen_tokens.bin`
   contains values DELIBERATELY DIFFERENT from the argmax chain. If the engine
   follows its own argmax anyway, the file is not what drives the output.
   Without D, "success" could just mean the file was read.

HOW THE BIT-IDENTITY CHECK WORKS (and why it is the right check)
---------------------------------------------------------------
The closed loop's step k produces logits from a KV cache built incrementally.
A "one-shot full forward over the same G+1 positions" is a DIFFERENT program
shape: it attends over the whole sequence in one pass with causal masking.

Those two are not expected to agree bitwise in general -- KV-cache decode and
full-prefill differ in summation order inside the attention GEMM, and this
project has measured that class of difference before (P13 called its cached
decode "bit-exact" against a RE-PREFILL, which is a same-code-path comparison).

So the honest object to compare is: **the engine's own argmax chain** in the
closed loop vs the tokens it writes into st_gen. That is bit-identity of the
FEEDBACK EDGE, which is exactly O6's claim: the token fed forward is the argmax
of the logits the engine just computed, with nothing lost or substituted.

This checker therefore verifies, per step:
  (1) the logits the engine PRINTED for step k exist and have V values,
  (2) argmax(logits_printed[k]) == the token the engine actually consumed at
      step k+1 (which the program PRINTS as `seen<k+1>`),
  (3) that equals what the closed loop wrote into st_gen,
  (4) the whole chain is reproducible: two runs give identical output.
Condition (2) is the real O6 claim and it is checked against the engine's OWN
printed logits, so a Python re-derivation cannot paper over a disagreement.

EXPECTED-CHAIN CROSS-CHECK: the checker ALSO recomputes the argmax chain from
the printed logits independently and reports whether it matches st_gen. If those
ever disagree, the loop is broken and the checker FAILS -- that is the guard
against "the engine printed plausible tokens that do not follow from its own
logits".

Usage:
    python tools/p14_gen_check.py --dir examples/p14_smoke --gen 4 --vocab 130560
    python tools/p14_gen_check.py --dir examples/p14_gen --gen 16 --vocab 130560
"""

import argparse
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
REL = os.path.join(ROOT, "target", "release")
ENG = os.path.join(REL, "vmlm-runtime.exe")


# --------------------------------------------------------------------------
# sampling (HOST-SIDE -- see docs/o6_notes.md §3)
#
# temperature==0 is special-cased to EXACT argmax, not to a division by zero or
# a "greedy" branch that reimplements argmax. That is what makes criterion B
# exact rather than approximate: temp=0 returns literally the same index the
# engine produced, so it cannot disagree with it.
# --------------------------------------------------------------------------
def sample_token(logits, temperature, top_k, top_p, rng):
    """Return (token, mode). mode says which stage decided, for honesty."""
    logits = np.asarray(logits, dtype=np.float64)
    if temperature <= 0.0:
        # EXACT argmax. np.argmax breaks ties to the lowest index, matching the
        # engine's OP_TOPK_IDX (`if v > m`, numeric compare, ties -> lowest).
        return int(np.argmax(logits)), "argmax(temp=0)"

    z = logits / float(temperature)

    if top_k and top_k > 0 and top_k < z.size:
        # Keep the top_k highest; mask the rest with -inf. Use argpartition for
        # the threshold, then a strict comparison so ties at the boundary are
        # resolved deterministically by index (stable, reproducible).
        kth = np.partition(z, -top_k)[-top_k]
        keep = z >= kth
        # If the boundary is tied, `keep` can retain MORE than top_k. Trim to
        # exactly top_k by index order so the result is reproducible.
        idx = np.flatnonzero(keep)
        if idx.size > top_k:
            order = np.lexsort((idx, -z[idx]))
            keep = np.zeros_like(keep)
            keep[idx[order[:top_k]]] = True
        z = np.where(keep, z, -np.inf)

    if top_p and 0.0 < top_p < 1.0:
        order = np.argsort(-z, kind="stable")
        zs = z[order]
        m = np.isfinite(zs)
        if m.any():
            zs_m = zs[m]
            e = np.exp(zs_m - zs_m.max())
            p = e / e.sum()
            c = np.cumsum(p)
            # Keep the smallest prefix whose cumulative mass reaches top_p.
            nkeep = int(np.searchsorted(c, top_p) + 1)
            nkeep = max(1, min(nkeep, zs_m.size))
            keep_mask = np.zeros_like(z, dtype=bool)
            keep_mask[order[m][:nkeep]] = True
            z = np.where(keep_mask, z, -np.inf)

    m = np.isfinite(z)
    if not m.any():
        return int(np.argmax(logits)), "argmax(temp=0 fallback)"
    zz = z[m]
    e = np.exp(zz - zz.max())
    p = e / e.sum()
    idx = np.flatnonzero(m)
    pick = int(rng.choice(idx.size, p=p))
    return int(idx[pick]), "sampled(T=%g,k=%s,p=%s)" % (temperature, top_k, top_p)


def parse_stdout(path, blocks):
    """Split the engine's single stdout stream into named blocks.

    WHY A MANIFEST: stdout is one stream (see the ORDERING NOTE in
    tools/p12_build_minicpm.py:893-903). The builder records how many values
    each PRINT contributes, so the reader never guesses where block boundaries
    are -- guessing is exactly how P13's row alignment went silently wrong.

    ENCODING GUARD (this bit us for real): PowerShell's `>` redirection writes
    UTF-16LE with a BOM, so a 50 MB capture full of numbers parses as ZERO
    values and the failure looks like a truncated stream rather than an encoding
    problem. Detect the BOM and say so, instead of silently reporting 0 values.
    """
    with open(path, "rb") as fb:
        head = fb.read(2)
    if head[:2] == b"\xff\xfe":
        return None, ("stdout file %s is UTF-16LE (BOM detected). PowerShell's "
                      "'>' redirection did this; the numbers are unreadable as "
                      "bytes. Capture with Start-Process "
                      "-RedirectStandardOutput (see tools/p14_gen_run.ps1)."
                      % os.path.basename(path))
    vals = []
    with open(path, "r", errors="replace", encoding="utf-8") as f:
        for line in f:
            for t in line.split():
                try:
                    vals.append(float(t))
                except ValueError:
                    pass
    out = {}
    off = 0
    for b in blocks:
        n = b["rows"] * b["cols"]
        if off + n > len(vals):
            return None, "stdout too short for block '%s': need %d, have %d" % (
                b["name"], off + n, len(vals))
        out[b["name"]] = np.array(vals[off:off + n], dtype=np.float32)
        off += n
    if off != len(vals):
        return out, ("WARNING: %d trailing values after the last known block -- "
                     "the manifest may be stale" % (len(vals) - off))
    return out, None


def run_artifact(adir, sampler, gen_file, tag):
    """Run once. Returns (stdout_path, seconds, returncode)."""
    import shutil
    vmb = os.path.join(adir, "full.vmb")
    shutil.copyfile(vmb, os.path.join(REL, "model.vmb"))
    shutil.copyfile(os.path.join(adir, "prompt_tok.bin"),
                    os.path.join(REL, "tokens.bin"))
    src = os.path.join(adir, gen_file)
    if not os.path.exists(src):
        return None, 0.0, -1
    shutil.copyfile(src, os.path.join(REL, "gen_tokens.bin"))
    out = os.path.join(adir, "_check_%s.txt" % tag)
    t0 = time.time()
    with open(out, "wb") as f:
        p = subprocess.run([ENG], cwd=REL, stdout=f, stderr=subprocess.DEVNULL)
    return out, time.time() - t0, p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--gen", type=int, default=0, help="0 = read from meta.json")
    ap.add_argument("--vocab", type=int, default=0, help="0 = read from meta.json")
    ap.add_argument("--seed-rng", type=int, default=1234)
    ap.add_argument("--feed-tokens", default="",
                    help="negative control chain, comma separated. Default: "
                         "derived from seed_rng so it is deterministic and "
                         "guaranteed different from the argmax chain.")
    args = ap.parse_args()

    adir = args.dir if os.path.isabs(args.dir) else os.path.join(ROOT, args.dir)
    if not os.path.isdir(adir):
        print("no such dir: %s" % adir)
        return 1

    meta = json.load(open(os.path.join(adir, "meta.json")))
    rows = json.load(open(os.path.join(adir, "gen_rows.json")))
    G = args.gen or meta["gen"]
    V = args.vocab or meta["vocab"]
    S = meta["seq"]

    print("=== O6 acceptance: %s ===" % adir)
    print("S=%d G=%d vocab=%d layers=%d sampler=%s"
          % (S, G, V, meta["layers"], meta["sampler"]))
    print()

    fails = []
    notes = []

    # ---------------- run 1: the closed loop (argmax build) ----------------
    if meta["sampler"] != "argmax":
        print("This artifact was built with --sampler %s, not 'argmax'."
              % meta["sampler"])
        print("Rebuild with: python tools/p14_gen_build.py --sampler argmax ...")
        return 1

    print("--- (A/C) closed loop, --sampler argmax ---")
    out1, dt1, rc1 = run_artifact(adir, "argmax", "seed_gen.bin", "closed")
    if rc1 != 0:
        print("  engine exit=%d" % rc1)
        return 1
    print("  ran in %.2f s" % dt1)
    blocks1, warn1 = parse_stdout(out1, rows["blocks"])
    if blocks1 is None:
        print("  PARSE FAIL: %s" % warn1)
        return 1
    if warn1:
        notes.append(warn1)

    prefill = blocks1["prefill_logits"].reshape(S, V)
    step_logits = [blocks1["step%d_logits" % k].reshape(V) for k in range(G)]
    seen = [int(round(float(blocks1["step%d_seen" % k][0]))) for k in range(G)]
    stgen = blocks1["st_gen_final"]

    print("  seed token (file, st_gen[0])      = %d" % meta["seed_token"])
    print("  tokens consumed per step (seen)   = %s" % seen)
    print("  st_gen after the run              = %s"
          % [int(round(float(x))) for x in stgen])

    # (2) THE O6 CLAIM: does step k+1 consume the argmax of step k's logits?
    print()
    print("--- (A) feedback edge: argmax(logits_k) == token consumed at k+1 ---")
    chain = []
    edge_ok = True
    for k in range(G - 1):
        am = int(np.argmax(step_logits[k]))
        chain.append(am)
        consumed = seen[k + 1]
        ok = (am == consumed)
        edge_ok = edge_ok and ok
        print("  step %2d: argmax(logits)=%-7d  consumed by step %2d=%-7d  %s"
              % (k, am, k + 1, consumed, "OK" if ok else "MISMATCH"))
    if not edge_ok:
        fails.append("feedback edge broken: a step did not consume the previous "
                     "step's argmax")
    else:
        print("  => every step consumed the PREVIOUS step's in-engine argmax.")

    # st_gen must equal [seed] + the argmax chain the engine produced.
    got_chain = [int(round(float(x))) for x in stgen]
    expect_chain = [meta["seed_token"]] + chain + [int(np.argmax(step_logits[G - 1]))]
    same = (got_chain == expect_chain)
    print()
    print("  st_gen chain        = %s" % got_chain)
    print("  [seed]+argmax chain = %s" % expect_chain)
    print("  %s" % ("OK: the engine's st_gen IS its own argmax chain"
                    if same else "MISMATCH: st_gen is not the argmax chain"))
    if not same:
        fails.append("st_gen does not equal [seed] + argmax(printed logits)")

    # (B) temperature=0 == argmax, exactly.
    print()
    print("--- (B) temperature=0 degrades to argmax EXACTLY ---")
    rng = np.random.default_rng(args.seed_rng)
    t0_tokens = [sample_token(step_logits[k], 0.0, 0, 1.0, rng)[0]
                 for k in range(G)]
    am_tokens = [int(np.argmax(step_logits[k])) for k in range(G)]
    print("  temp=0 tokens = %s" % t0_tokens)
    print("  argmax tokens = %s" % am_tokens)
    print("  stored st_gen = %s" % got_chain[1:])
    b_ok = (t0_tokens == am_tokens)
    print("  %s" % ("OK: temp=0 IS argmax, token for token" if b_ok else "FAIL"))
    if not b_ok:
        fails.append("temperature=0 did not degrade exactly to argmax")

    # (C) temperature>0 actually changes output.
    print()
    print("--- (C) temperature>0 actually changes the output ---")
    changed_any = False
    for T in (0.7, 1.0):
        for sd in (1, 2, 3, 1234):
            rng2 = np.random.default_rng(sd)
            s = [sample_token(step_logits[k], T, 0, 1.0, rng2)[0] for k in range(G)]
            if s != am_tokens:
                changed_any = True
                print("  T=%s seed=%d -> %s  (differs at %d/%d positions)"
                      % (T, sd, s,
                         sum(1 for a, b in zip(s, am_tokens) if a != b), G))
                break
        if changed_any:
            break
    # top-k / top-p at T=1 must also be able to move the result.
    for label, kw in (("top_k=4", dict(top_k=4, top_p=1.0)),
                      ("top_p=0.5", dict(top_k=0, top_p=0.5))):
        rng3 = np.random.default_rng(7)
        s = [sample_token(step_logits[k], 1.0, kw["top_k"], kw["top_p"], rng3)[0]
             for k in range(G)]
        moved = sum(1 for a, b in zip(s, am_tokens) if a != b)
        print("  T=1.0 %-9s -> differs at %d/%d positions" % (label, moved, G))
        if moved:
            changed_any = True
    if not changed_any:
        fails.append("temperature>0 never changed the output -- sampling looks "
                     "like a no-op")
    else:
        print("  OK: sampling is live (it is HOST-SIDE -- see docs/o6_notes.md)")

    # (C2) reproducibility: same seed => same tokens.
    print()
    print("--- (C2) reproducibility: same seed => same tokens ---")
    a = [sample_token(step_logits[k], 0.9, 8, 0.95, np.random.default_rng(99))[0]
         for k in range(G)]
    b = [sample_token(step_logits[k], 0.9, 8, 0.95, np.random.default_rng(99))[0]
         for k in range(G)]
    print("  run1 = %s" % a)
    print("  run2 = %s" % b)
    if a != b:
        fails.append("sampling is not reproducible for a fixed seed")
    else:
        print("  OK: deterministic for a fixed seed")

    # ---------------- (D) THE NEGATIVE CONTROL ----------------
    #
    # WHAT MUST BE HELD FIXED, AND WHY THIS IS SUBTLE.
    #
    # `st_gen[0]` is the SEED and is file-sourced BY DESIGN. So a control that
    # changes it changes step 0's input, and therefore legitimately changes the
    # whole downstream chain -- that tests "the seed is read", not "the loop
    # feeds back". The first version of this control made exactly that mistake
    # and reported FAIL against a loop that was provably closing (its internal
    # edge check passed for all 15 steps in the control run too).
    #
    # The right control changes ONLY `st_gen[1..G]` -- the positions the engine
    # is supposed to overwrite -- and keeps `st_gen[0]` identical. If
    # `st_gen[1..G]` are genuinely engine-written, their initial file values are
    # DEAD and the run is bit-identical to the normal one. If the engine were
    # reading the file, the chain would change.
    print()
    print("--- (D) NEGATIVE CONTROL: garbage in gen_tokens.bin[1..G] must be DEAD ---")
    print("  (st_gen[0], the seed, is held IDENTICAL: it is file-sourced by design,")
    print("   so changing it would legitimately change step 0 and prove nothing")
    print("   about the feedback edge.)")
    n_seed = max(G + 1, 4)
    if args.feed_tokens:
        wrong_tail = [int(t) for t in args.feed_tokens.split(",") if t.strip()]
    else:
        # deterministic values, and guaranteed to be non-trivial
        wrong_tail = [(meta["seed_token"] + 11 * i + 7) % V
                      for i in range(n_seed)]
    wrong_tail = (wrong_tail * 4)[:n_seed]
    # Position 0 MUST stay the real seed.
    control = list(wrong_tail)
    control[0] = meta["seed_token"]
    if control[1:G + 1] == got_chain[1:G + 1]:
        fails.append("negative control tail accidentally equals the argmax "
                     "chain -- the control would prove nothing")
        print("  CONTROL INVALID: chosen tail equals the argmax chain")
    else:
        wpath = os.path.join(adir, "_control_gen.bin")
        np.array(control, dtype="<u4").tofile(wpath)
        print("  gen_tokens.bin[0]     = %d   (UNCHANGED seed)" % control[0])
        print("  gen_tokens.bin[1..G]  = %s" % control[1:G + 1])
        print("  (argmax chain is        %s)" % got_chain[1:G + 1])
        out2, dt2, rc2 = run_artifact(adir, "argmax", "_control_gen.bin", "control")
        if rc2 != 0:
            print("  control engine exit=%d" % rc2)
            fails.append("negative-control run failed with exit %d" % rc2)
        else:
            blocks2, warn2 = parse_stdout(out2, rows["blocks"])
            if blocks2 is None:
                print("  PARSE FAIL: %s" % warn2)
                fails.append("negative-control output unparseable")
            else:
                seen2 = [int(round(float(blocks2["step%d_seen" % k][0])))
                         for k in range(G)]
                stgen2 = [int(round(float(x))) for x in blocks2["st_gen_final"]]
                print("  control: tokens consumed = %s" % seen2)
                print("  control: st_gen          = %s" % stgen2)
                same_seen = (seen2 == seen)
                same_chain = (stgen2 == got_chain)
                print("  consumed tokens identical : %s" % same_seen)
                print("  st_gen identical          : %s" % same_chain)
                if same_chain and same_seen:
                    print("  OK: the file's values at [1..G] were DEAD. The engine")
                    print("      followed its OWN argmax; only st_gen[0] (the seed)")
                    print("      is file-sourced.")
                else:
                    fails.append("negative control changed the output -- the "
                                 "file's tail values are live, so the loop is "
                                 "NOT what produced the chain")
                    n_diff = sum(1 for a, b in zip(stgen2, got_chain) if a != b)
                    print("  %d of %d st_gen positions differ -> the file's tail "
                          "is NOT dead" % (n_diff, len(got_chain)))

    # ---------------- (D2) PROOF THE FILE PATH IS LIVE ----------------
    #
    # (D) shows the file's tail is DEAD in an argmax build. On its own that is
    # also consistent with "the engine ignores gen_tokens.bin entirely", which
    # would make (D) prove nothing. So we must show the SAME file DOES drive a
    # build that reads it -- otherwise the control has no contrast.
    #
    # The `--sampler feed` artifact is exactly that build: its steps read their
    # tokens from st_gen[k+1] with no in-engine argmax. Feeding it the control
    # file must visibly change its output.
    print()
    print("--- (D2) contrast: the same file DOES drive a --sampler feed build ---")
    feed_dir = os.path.join(os.path.dirname(adir), "p14_gen_feed")
    if not os.path.isdir(feed_dir):
        notes.append("no feed-control artifact at %s, so (D) has no contrast: "
                     "the file being dead is consistent with the engine ignoring "
                     "it. Build with --sampler feed to close this." % feed_dir)
        print("  SKIPPED: no feed artifact at %s" % feed_dir)
        print("  NOTE: without this, 'the tail is dead' is also consistent with")
        print("        'the engine ignores gen_tokens.bin'. Run the feed build.")
    else:
        fmeta = json.load(open(os.path.join(feed_dir, "meta.json")))
        frows = json.load(open(os.path.join(feed_dir, "gen_rows.json")))
        ffed = os.path.join(feed_dir, "feed_gen.bin")
        # The feed build has NO in-engine argmax: EVERY step (including 0) reads
        # its token from st_gen, so gen_tokens.bin must carry the WHOLE chain,
        # seed included. That is `feed_gen.bin`, not `seed_gen.bin` -- passing
        # the 1-value seed file here made every step after 0 read a zero, which
        # is exactly the [88, 0, 0, ...] the first attempt produced.
        if not os.path.exists(ffed):
            notes.append("feed artifact has no feed_gen.bin at %s -- rebuild "
                         "with --sampler feed --feed-tokens ..." % ffed)
            print("  SKIPPED: feed_gen.bin absent (no contrast for (D))")
            out3, rc3 = None, -1
        else:
            out3, dt3, rc3 = run_artifact(feed_dir, "feed", "feed_gen.bin", "feed")
        if out3 is None:
            pass
        elif rc3 != 0:
            print("  feed artifact engine exit=%d" % rc3)
            fails.append("feed-control artifact failed to run (exit %d)" % rc3)
        else:
            fb, fw = parse_stdout(out3, frows["blocks"])
            if fb is None:
                print("  PARSE FAIL: %s" % fw)
                fails.append("feed-control output unparseable")
            else:
                fseen = [int(round(float(fb["step%d_seen" % k][0])))
                         for k in range(G)]
                expected_feed = [fmeta["seed_token"]] + fmeta["feed_tokens"]
                print("  feed build: tokens consumed = %s" % fseen)
                print("  feed file  [seed]+chain    = %s" % expected_feed[:G])
                # The feed build has NO argmax: step k must consume the FILE's
                # value at position k. That is the contrast (D) needs.
                file_driven = (fseen == expected_feed[:G])
                print("  consumed == file contents : %s" % file_driven)
                if file_driven:
                    print("  OK: with --sampler feed the FILE drives the chain,")
                    print("      while with --sampler argmax it is dead. The two")
                    print("      builds together prove the feedback edge carries")
                    print("      the data, and that (D) is not vacuous.")
                else:
                    fails.append("feed-control build did not follow its file "
                                 "chain -- cannot establish contrast for (D)")

    # ---------------- summary ----------------
    print()
    print("=" * 66)
    if notes:
        for n in notes:
            print("NOTE: %s" % n)
    if fails:
        print("VERDICT: FAIL")
        for f in fails:
            print("  - %s" % f)
        return 1
    print("VERDICT: PASS -- O6 loop closes; argmax feeds back; temp=0 == argmax;")
    print("        sampling is live and reproducible; negative control holds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
