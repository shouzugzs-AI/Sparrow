#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P14 O6 INDEPENDENT verification (verifier-owned, task-4).

This is NOT tools/p14_gen_check.py. That file belongs to the implementer of O6.
A checker written by the implementer shares the implementer's assumptions about
what the artifact does, so it can only confirm intent. This suite is written by
someone who did not write the loop, drives the artifact as a BLACK BOX, and
treats every claim as false until an input perturbation makes it true.

THE CENTRAL ATTACK: "the loop might be a no-op"
------------------------------------------------
tools/p14_gen_build.py offers `--sampler argmax` (true closed loop: in-engine
TOPK_IDX writes st_gen[k+1]) and `--sampler feed` (negative control: tokens come
from a file, no in-engine argmax). Those two are the SAME builder, so the
difference between them is exactly "did the engine compute the argmax or did it
read a file".

The decisive experiment is NOT "do the two runs differ" -- it is:

    run --sampler argmax on an artifact that declares st_gen
    while gen_tokens.bin contains a DELIBERATELY WRONG chain.

If the artifact's output equals the argmax chain, the file is NOT the source of
truth and the loop is real. If the output equals the wrong chain, the loop is a
fake and this suite reports FAIL. Because the runtime loads gen_tokens.bin into
st_gen BEFORE run() (measured), a no-op loop CANNOT pass this test.

WHY A MISSING-FILE TEST IS ALSO DECISIVE
----------------------------------------
Measured on the shipped engine: a program declaring st_gen with no
gen_tokens.bin exits 2 "cannot open file". So:
   * argmax artifact + no file  -> must FAIL LOUDLY. If it succeeds, the
     artifact never reads st_gen, i.e. it is not the loop under test.
   * feed artifact + no file    -> must also FAIL LOUDLY (it needs the file).
Both are recorded, and the difference between them is the evidence.

THE INDEPENDENT REFERENCE
-------------------------
"Bit-identical to a one-shot forward" must not be checked against a reference
produced by the same code. Three anchors are used, in increasing order of
independence:
  A1  the implementer's `--sampler feed` build with the argmax chain handed to
      it: same weights, same positions, NO in-engine argmax. If the argmax loop
      is right, its chain equals the feed build's chain when the feed build is
      given that chain. (Two different control-flow paths, one builder.)
  B1  the logits the artifact itself prints: argmax_chain(step k logits) must
      equal the token step k+1 actually consumed. This is INTERNAL consistency
      -- it does not assume the loop is right, it assumes only that argmax of
      the printed logits is what should be fed back.
  C1  HuggingFace, on the model's own prompt [3,117,2501,88]: the first
      generated token must be the HF argmax 5, re-derived from
      examples/minicpm_full/hf_logits.npy at test time (not hard-coded).
Each anchor is shown able to fail.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p14_verify_lib as L  # noqa: E402

ROOT = L.ROOT
REL = L.REL
ASM = L.ASM
ENG = L.ENG
MODEL_DIR = os.environ.get("MINICPM_DIR", "<外部模型目录>/MiniCPM5-1B")   # the O6 builder's own default
WORK = os.path.join(ROOT, "examples", "p14")


def find_model_dir():
    """Locate the HF model dir the builder needs (config.json + safetensors).

    Read the builder's own MODEL constant rather than hard-coding a second copy:
    if the builder's default changes, this suite follows it instead of silently
    testing a different model. Windows paths are case-insensitive, so the exact
    casing of the builder's literal need not match what is on disk.
    """
    cands = []
    try:
        src = open(os.path.join(ROOT, "tools", "p14_gen_build.py"),
                   encoding="utf-8").read()
        for line in src.splitlines():
            if line.startswith("MODEL") and "=" in line:
                v = line.split("=", 1)[1].strip().strip('"').strip("'")
                if v:
                    cands.append(v)
    except Exception:
        pass
    cands.append(MODEL_DIR)
    for c in cands:
        if os.path.exists(os.path.join(c, "config.json")):
            return c
    return None


def build(sampler, gen, out, feed=None, layers=0, seq=4, model=None):
    """Invoke the O6 builder. `layers=0` means all 24."""
    if os.path.isdir(out):
        shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, os.path.join(ROOT, "tools", "p14_gen_build.py"),
           "--seq", str(seq), "--gen", str(gen), "--out", out,
           "--sampler", sampler]
    if layers:
        cmd += ["--layers", str(layers)]
    if model:
        cmd += ["--model", model]
    if feed is not None:
        cmd += ["--feed-tokens", ",".join(str(int(t)) for t in feed)]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                       errors="replace")
    return p.returncode == 0, (p.stdout or "") + (p.stderr or "")


def assemble(out, vmb="m.vmb"):
    """Assemble `full.vml`. Uses RELATIVE weight specs + cwd=out.

    Measured trap: PowerShell cannot pass a drive-letter spec, but Python can.
    Regardless, relative specs keep the command line short (the real model has
    hundreds of constants) and avoid any arg-length limit.
    """
    args_path = os.path.join(out, "weights.args")
    if not os.path.exists(args_path):
        return False, "no weights.args"
    consts = [l.strip() for l in open(args_path, encoding="utf-8") if l.strip()]
    rel = []
    for c in consts:
        f = c.split(":")
        # name:file:rows:cols[:dt] -> make `file` relative to out dir
        if len(f) >= 2 and os.path.isabs(f[1]):
            f[1] = os.path.relpath(f[1], out).replace("\\", "/")
        rel.append(":".join(f))
    p = subprocess.run([ASM, "full.vml", vmb] + rel, cwd=out,
                       capture_output=True, text=True, errors="replace")
    ok = (p.returncode == 0 and os.path.exists(os.path.join(out, vmb)))
    return ok, (p.stdout or "") + (p.stderr or "")


def deploy(vmb_src, prompt_ids, gen_ids, mode):
    """Deprecated staging helper, kept only for reference.

    The shared-directory staging this used to do is the source of the race the
    team hit repeatedly (artifact overwritten or DELETED mid-run by another
    teammate, for a run lasting 20-30 minutes). All runs now go through
    L.run_engine_hermetic(), which copies the engine and the artifact into a
    PRIVATE temp dir and runs there; the engine's `model.vmb` is a RELATIVE
    name, so cwd is sufficient to isolate it. See deploy_hermetic().
    """
    raise NotImplementedError(
        "use L.run_engine_hermetic() instead -- shared target/release staging "
        "is not safe with concurrent teammates")


def run_hermetic(vmb_src, prompt_ids, mode, seed_tok=None, gen_ids=None,
                 gen_raw=None):
    """Deploy into a PRIVATE temp dir and run. Returns (code, out, err, mode_desc).

    mode:
      'seed'   -> tokens.bin = prompt ; gen_tokens.bin = [seed_tok]
      'gen'    -> tokens.bin = prompt ; gen_tokens.bin = gen_ids (planted chain)
      'raw'    -> tokens.bin = prompt ; gen_tokens.bin = gen_raw bytes verbatim
      'prompt' -> tokens.bin = prompt ; gen_tokens.bin ABSENT (must fail loudly)
      'none'   -> BOTH absent
    There is no artifact-intact flag any more because there is nothing shared to
    clobber: the temp dir contains only our engine and our artifact.
    """
    if mode == "none":
        return L.run_engine_hermetic(vmb_src, tokens=None, omit_tokens=True,
                                    omit_gen=True) + ("none",)
    if mode == "prompt":
        return L.run_engine_hermetic(vmb_src, tokens=prompt_ids,
                                     omit_gen=True) + ("prompt, no gen file",)
    if mode == "seed":
        ids = seed_tok if isinstance(seed_tok, (list, tuple)) else [seed_tok]
        return L.run_engine_hermetic(vmb_src, tokens=prompt_ids,
                                     gen_tokens=ids) + ("seed only",)
    if mode == "gen":
        return L.run_engine_hermetic(vmb_src, tokens=prompt_ids,
                                     gen_tokens=gen_ids) + ("planted chain",)
    if mode == "raw":
        return L.run_engine_hermetic(vmb_src, tokens=prompt_ids,
                                     gen_tokens_raw=gen_raw) + ("raw bytes",)
    raise ValueError(mode)


def parse_blocks(out_bytes, manifest):
    """Split engine stdout into the named blocks recorded in gen_rows.json.

    Parsing from the manifest rather than from position arithmetic is
    deliberate: P12 lost a cycle to a row-alignment bug where the reader's
    assumed layout silently disagreed with what was printed (it read 10752
    values where 4608 were expected). The manifest is written by the same code
    that emits the program, so it cannot drift.
    """
    vals = []
    for tok in out_bytes.decode("ascii", "replace").split():
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
    return got, None


def argmax_rows(logits):
    return [int(i) for i in np.argmax(np.asarray(logits, dtype=np.float32),
                                      axis=1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=int, default=16)
    ap.add_argument("--seq", type=int, default=4)
    ap.add_argument("--layers", type=int, default=0,
                    help="0 = all 24; small values for a fast iteration")
    ap.add_argument("--quick", action="store_true",
                    help="reduce G so the suite runs fast")
    args = ap.parse_args()
    if args.quick:
        args.gen = min(args.gen, 4)

    G, S = args.gen, args.seq
    meta_checks = L.MetaCheck()
    fails = 0
    unverified = []

    print("=== P14 O6 independent verification (S=%d G=%d layers=%s) ==="
          % (S, G, args.layers or "all"))

    mdl = find_model_dir()
    if mdl is None:
        print("UNVERIFIED: cannot locate the HF model dir (config.json + "
              "safetensors) that the O6 builder needs.")
        print("           Nothing below was run. This is NOT a pass.")
        return 1
    print("model dir: %s" % mdl)

    # ------------------------------------------------------------------
    # HuggingFace anchor, re-derived from disk
    # ------------------------------------------------------------------
    hf, err = L.load_hf_anchor() if hasattr(L, "load_hf_anchor") else (
        None, "helper absent")
    if hf is None:
        try:
            lg = np.load(os.path.join(ROOT, "examples", "minicpm_full",
                                      "hf_logits.npy"))
            hf = argmax_rows(lg)
            err = None
        except Exception as e:
            hf, err = None, str(e)
    if err:
        unverified.append(("O6-P3 HuggingFace anchor", "hf_logits.npy: %s" % err))
        print("WARNING: HF anchor unavailable (%s)" % err)
    else:
        print("HF anchor (re-derived from hf_logits.npy): %s" % hf)

    # ------------------------------------------------------------------
    # Build the two artifacts
    # ------------------------------------------------------------------
    d_arg = os.path.join(WORK, "argmax")
    d_feed = os.path.join(WORK, "feed")
    print("\n[1/6] building --sampler argmax ...")
    ok, log = build("argmax", G, d_arg, layers=args.layers, seq=S, model=mdl)
    if not ok:
        print("BUILD FAILED:\n%s" % log[-2500:])
        return 1
    print("      assembling ...")
    ok, log = assemble(d_arg)
    if not ok:
        print("ASSEMBLE FAILED:\n%s" % log[-2500:])
        return 1
    man_arg = json.load(open(os.path.join(d_arg, "gen_rows.json"),
                             encoding="utf-8"))
    seed_tok = man_arg["seed_token"]
    prompt_ids = man_arg["prompt_ids"]
    print("      prompt=%s seed(st_gen[0])=%d" % (prompt_ids, seed_tok))

    # ------------------------------------------------------------------
    # O6-P1: the argmax run, and the token each step ACTUALLY consumed
    # ------------------------------------------------------------------
    print("\n[2/6] running --sampler argmax (seed file only) ...")
    code, out, errb, mode_desc = run_hermetic(os.path.join(d_arg, "m.vmb"),
                                              prompt_ids, "seed",
                                              seed_tok=seed_tok)
    print("      exit=%d  stdout=%d bytes  (hermetic private dir; mode=%s)"
          % (code, len(out), mode_desc))
    if code != 0:
        print("      engine stderr: %s" % errb[:400])
        print("VERDICT: FAIL -- the argmax artifact did not run")
        return 1
    blocks, perr = parse_blocks(out, man_arg)
    if perr:
        print("      PARSE FAIL: %s" % perr)
        return 1
    seen = [int(round(float(blocks["step%d_seen" % k][0, 0])))
            for k in range(G)]
    final = [int(round(float(v))) for v in blocks["st_gen_final"][0]]
    print("      st_gen_final       : %s" % final)
    print("      tokens consumed    : %s" % seen)

    # What the ENGINE's own logits imply the chain should be. This is the
    # internal-consistency anchor: argmax(step k logits) == the token step k+1
    # consumed. It does not assume the loop is correct.
    implied = []
    for k in range(G):
        lg = blocks["step%d_logits" % k]
        implied.append(argmax_rows(lg)[0])
    print("      argmax(own logits) : %s" % implied)

    # The engine's chain = st_gen contents. st_gen[0] is the seed; positions
    # 1..G are the tokens it generated.
    eng_chain = final[1:]
    print("      generated chain    : %s" % eng_chain)

    # O6-P1 (internal form): consumed token k+1 must equal argmax(step k logits)
    exp_consumed = [seed_tok] + implied[:-1]
    ok_p1, d_p1 = L.compare_chains(exp_consumed, seen, "consumed vs implied")
    print("\n      O6-P1 feed-back consistency: consumed[k] == "
          "argmax(step k-1 logits)")
    print("        %s  %s" % ("PASS" if ok_p1 else "FAIL", d_p1))

    # and the generated chain must equal the implied argmax chain
    ok_p1b, d_p1b = L.compare_chains(implied, eng_chain, "chain vs implied")
    print("      O6-P1 chain == argmax(own logits): %s  %s"
          % ("PASS" if ok_p1b else "FAIL", d_p1b))

    # META-CHECK for O6-P1
    canfail_p1, rows = L.demonstrate_can_fail(eng_chain) \
        if hasattr(L, "demonstrate_can_fail") else (None, None)
    if rows is None:
        # local equivalent
        canfail_p1 = True
        rows = []
        for kind in ["off_by_one", "reverse", "shift_left", "truncate"]:
            p = L.perturb(eng_chain, kind)
            acc, det = L.compare_chains(eng_chain, p, kind)
            rows.append((kind, "REJECTED" if not acc else "ACCEPTED", det))
            if acc:
                canfail_p1 = False
    meta_checks.record("O6-P1 feed-back consistency", bool(ok_p1 and ok_p1b),
                       "perturbed chain (off-by-one/reverse/shift/truncate)",
                       bool(canfail_p1),
                       "rejected: %s" % ", ".join(r[0] for r in rows))
    if not (ok_p1 and ok_p1b):
        fails += 1
    if not canfail_p1:
        fails += 1

    # ------------------------------------------------------------------
    # O6-P3: HuggingFace anchor
    # ------------------------------------------------------------------
    print("\n[3/6] O6-P3 external anchor: prefill logits vs HuggingFace")
    if hf is None:
        unverified.append(("O6-P3", "HF logits unavailable"))
        print("      UNVERIFIED")
    else:
        # WHICH POSITION IS THE ANCHOR, AND WHY (this was my own first bug):
        # The artifact PRE-FILLS the S prompt tokens, printing logits for all S
        # positions, and only THEN runs generation. Generation step 0 consumes
        # st_gen[0] -- the SEED -- at absolute position S. So:
        #   * argmax(prefill logits)[S-1]  is the token the model would emit
        #     after the prompt, and is directly comparable to hf[S-1];
        #   * engine_chain[0] is the argmax at position S (prompt+seed), a
        #     position the prompt-only HF dump never covered.
        # An earlier version of this check compared engine_chain[0] to hf[S-1]
        # and reported a spurious FAIL. The correct anchor is the prefill row.
        n_pref = S
        eng_pre = argmax_rows(blocks["prefill_logits"])[:n_pref]
        hf_pre = hf[:n_pref]
        matches = sum(1 for a, b in zip(eng_pre, hf_pre) if a == b)
        print("      prefill argmax (engine): %s" % eng_pre)
        print("      prefill argmax (HF)    : %s" % hf_pre)
        print("      match: %d / %d" % (matches, n_pref))

        # The 24-layer HF dump can only be matched by a 24-layer artifact. A
        # reduced-layer smoke build MUST disagree, so requiring agreement there
        # would be a false failure -- and, worse, silently accepting a smoke
        # build against the full reference would be a false PASS. State which
        # case this run is.
        artifact_layers = man_arg.get("layers", 0)
        # ⚠️ P16 LEAD FIX -- DO NOT HARD-CODE 24 HERE.
        #
        # The original was `full_model = (artifact_layers in (0, 24))`, with 24
        # as a LITERAL. That is correct for this checkpoint but it is exactly
        # how the "24-layer headline" survived: the guard routed 4-layer
        # artifacts to UNVERIFIED, yet the report still said "PASS (24 layers)"
        # -- so the guard never actually caught the mismatch in the write-up.
        # A literal also silently breaks the moment the reference model changes
        # depth: it would keep demanding 24.
        #
        # Derive the expected depth from the HF reference config when it is
        # available, and only fall back to the literal when it is not.
        try:
            _md = find_model_dir()
            with open(os.path.join(_md, "config.json"), encoding="utf-8") as f:
                expected_layers = json.load(f)["num_hidden_layers"]
        except Exception:
            expected_layers = 24
        full_model = (artifact_layers in (0, expected_layers))
        # Make the layer count LOUD, whether or not it matches. The cost of the
        # original bug was that the number was never printed next to the claim.
        print("      artifact declares %d layer(s); HF reference declares %d"
              % (artifact_layers, expected_layers))
        if not full_model:
            print("      artifact has %d layer(s); the HF dump is the %d-LAYER"
                  % (artifact_layers, expected_layers))
            print("      model, so agreement is NOT expected here. This anchor")
            print("      is therefore UNVERIFIED at this layer count and will be")
            print("      settled by the full %d-layer run." % expected_layers)
            unverified.append(
                ("O6-P3 HuggingFace anchor",
                 "smoke artifact has %d layer(s); HF reference is 24-layer"
                 % artifact_layers))
            # Still record a real falsification demonstration so the criterion
            # is not left as an untested assertion.
            meta_checks.record(
                "O6-P3 HF anchor (layer-matched only)", False,
                "wrong-layer artifact must NOT pass this anchor",
                bool(not full_model),
                "smoke build correctly refused the 24-layer anchor")
        else:
            ok_p3 = (matches == n_pref)
            print("      %s" % ("PASS: prefill argmax matches HF at every "
                                "position" if ok_p3 else
                                "FAIL: engine disagrees with HuggingFace"))
            meta_checks.record(
                "O6-P3 HF anchor (first token)", ok_p3,
                "any position whose argmax differs from HF",
                True, "integer inequality at %d/%d positions" % (n_pref -
                                                                 matches,
                                                                 n_pref))
            if not ok_p3:
                fails += 1

    # ------------------------------------------------------------------
    # O6-P2/P4: THE NEGATIVE CONTROL -- wrong gen_tokens.bin, and no file
    # ------------------------------------------------------------------
    print("\n[4/6] O6-P2/P4 negative controls (the anti-no-op attack)")
    wrong = [max(0, (t + 7919) % 130560) for t in eng_chain]
    if wrong == eng_chain:
        wrong = L.perturb(eng_chain, "off_by_one")
    print("      wrong chain planted in gen_tokens.bin: %s" % wrong)

    # (a) argmax artifact + DELIBERATELY WRONG gen_tokens.bin.
    #     st_gen is pre-filled with `wrong` before run(). If the loop is real,
    #     only st_gen[0] (=seed) is read from the file for step 0, and every
    #     later position is OVERWRITTEN by the in-engine argmax. So the output
    #     must STILL be the argmax chain.
    code_w, out_w, _, mode_w = run_hermetic(
        os.path.join(d_arg, "m.vmb"), prompt_ids, "gen",
        gen_ids=[seed_tok] + wrong[:G])
    print("      hermetic run (%s)" % mode_w)
    if code_w != 0:
        # Reading the file succeeded, so any non-zero exit here is informative.
        print("      (wrong-file run exit=%d)" % code_w)
    blk_w, perr_w = parse_blocks(out_w, man_arg) if code_w == 0 else (None, "exit")
    if blk_w is None:
        unverified.append(("O6-P2 wrong-file control",
                           "run failed or unparseable: %s" % perr_w))
        print("      UNVERIFIED: %s" % perr_w)
    else:
        final_w = [int(round(float(v))) for v in blk_w["st_gen_final"][0]]
        eng_w = final_w[1:]
        # THE SHARP TEST (tail invariance): the ENGINE-PRODUCED tail must be
        # unchanged by a wrong file, because st_gen[0] is the only file-sourced
        # position by design. If the tail tracks the file, the loop is a no-op.
        ok_w, d_w = L.compare_chains(eng_chain, eng_w,
                                     "argmax output with wrong file")
        # Did the PLANTED chain leak into the result? Compare position-by-
        # position so a partial leak (e.g. only the last step failing to be
        # written -- a REAL bug the implementer reported finding) is caught.
        leaked_at = [i for i in range(min(len(wrong), len(eng_w)))
                     if eng_w[i] == wrong[i] and (i >= len(eng_chain)
                                                  or eng_chain[i] != wrong[i])]
        leaked = (len(leaked_at) > 0)
        print("      argmax output with WRONG file : %s" % eng_w)
        print("      true argmax chain             : %s" % eng_chain)
        print("      planted wrong chain           : %s" % wrong)
        print("      tail identical to argmax run? : %s" % ok_w)
        print("      planted values leaked into result at positions: %s"
              % (leaked_at if leaked_at else "none"))
        print("      %s" % ("PASS: the engine followed its own argmax, not the "
                            "file" if ok_w and not leaked else
                            "FAIL: the file contents influenced the output"))
        meta_checks.record(
            "O6-P2 loop is not a no-op", bool(ok_w and not leaked),
            "gen_tokens.bin = a chain differing from the argmax chain",
            bool(not leaked),
            "planted %s ; leak positions %s"
            % (wrong[:min(4, len(wrong))], leaked_at if leaked_at else "none"))
        if not (ok_w and not leaked):
            fails += 1

    # (b) argmax artifact + gen_tokens.bin ABSENT -> must fail loudly.
    code_n, out_n, err_n, mode_n = run_hermetic(os.path.join(d_arg, "m.vmb"),
                                                prompt_ids, "prompt")
    print("      argmax artifact with NO gen_tokens.bin: exit=%d (%s)"
          % (code_n, mode_n))
    loud = (code_n != 0)
    print("      %s" % ("PASS: loud failure (loop needs st_gen, as a real "
                        "loop must)" if loud else
                        "FAIL: exited 0 without its token file"))
    meta_checks.record(
        "O6-P4 missing gen_tokens.bin fails loudly", loud,
        "gen_tokens.bin absent", loud,
        "exit=%d (a no-op echo implementation could not survive this)" % code_n)
    if not loud:
        fails += 1

    # ------------------------------------------------------------------
    # O6-P1 (route agreement): feed build given the argmax chain must agree
    # ------------------------------------------------------------------
    print("\n[5/6] O6-P1 route agreement: --sampler feed fed the argmax chain")
    ok, log = build("feed", G, d_feed, feed=eng_chain[:G],
                    layers=args.layers, seq=S, model=mdl)
    if not ok:
        print("      BUILD FAILED:\n%s" % log[-1500:])
        unverified.append(("O6-P1 route agreement", "feed build failed"))
    else:
        ok, log = assemble(d_feed)
        if not ok:
            print("      ASSEMBLE FAILED:\n%s" % log[-1500:])
            unverified.append(("O6-P1 route agreement", "feed assemble failed"))
        else:
            man_feed = json.load(open(os.path.join(d_feed, "gen_rows.json"),
                                      encoding="utf-8"))
            code_f, out_f, _, mode_f = run_hermetic(
                os.path.join(d_feed, "m.vmb"), prompt_ids, "gen",
                gen_ids=[seed_tok] + eng_chain[:G])
            print("      hermetic run (%s)" % mode_f)
            blk_f, perr_f = parse_blocks(out_f, man_feed) if code_f == 0 \
                else (None, "exit %d" % code_f)
            if blk_f is None:
                print("      feed run unusable: %s" % perr_f)
                unverified.append(("O6-P1 route agreement", str(perr_f)))
            else:
                feed_chain = [int(round(float(v)))
                              for v in blk_f["st_gen_final"][0]][1:]
                # The feed build's LOGITS must agree with the argmax build's
                # logits at the same positions: same weights, same input tokens,
                # same positions -- two DIFFERENT control-flow paths.
                ndiff = 0
                maxd = 0.0
                for k in range(min(G, len(eng_chain))):
                    a = blk_f["step%d_logits" % k].astype(np.float32)
                    b = blocks["step%d_logits" % k].astype(np.float32)
                    if a.shape == b.shape:
                        ndiff += int((a != b).sum())
                        maxd = max(maxd, float(np.abs(a - b).max()))
                print("      step logits: argmax path vs feed path -> "
                      "%d differing values, max|d| = %.3e" % (ndiff, maxd))
                ok_ra = (ndiff == 0)
                print("      %s" % ("PASS: the two control-flow paths produce "
                                    "bit-identical logits" if ok_ra else
                                    "FAIL: the paths disagree"))
                meta_checks.record(
                    "O6-P1 route agreement (independent path)",
                    ok_ra, "one ULP / one value perturbed in the logits",
                    True,
                    "exact bit comparison; %d diffs, max|d|=%.2e"
                    % (ndiff, maxd))
                if not ok_ra:
                    fails += 1

    # ------------------------------------------------------------------
    # O6-P6: determinism
    # ------------------------------------------------------------------
    print("\n[6/6] O6-P6 determinism: run twice, compare raw stdout bytes")
    c1, o1, _, m1 = run_hermetic(os.path.join(d_arg, "m.vmb"), prompt_ids,
                                 "seed", seed_tok=seed_tok)
    c2, o2, _, m2 = run_hermetic(os.path.join(d_arg, "m.vmb"), prompt_ids,
                                 "seed", seed_tok=seed_tok)
    same = (c1 == c2 == 0 and o1 == o2 and len(o1) > 0)
    print("      run1 exit=%d %d bytes | run2 exit=%d %d bytes | identical=%s"
          % (c1, len(o1), c2, len(o2), o1 == o2))
    print("      %s" % ("PASS" if same else "FAIL"))
    meta_checks.record("O6-P6 determinism (byte-identical reruns)", same,
                       "an empty or differing output", bool(o1 and o1 == o2),
                       "non-empty output required, else it is not a measurement")
    if not same:
        fails += 1

    # ------------------------------------------------------------------
    allproven = meta_checks.report()
    print("\n" + "=" * 62)
    print("O6 VERDICT: %s" % ("PASS" if (fails == 0 and allproven) else "FAIL"))
    print("  failed criteria   : %d" % fails)
    print("  unproven criteria : %d" % sum(1 for r in meta_checks.rows
                                            if not r["can_fail"]))
    if unverified:
        print("\nUNVERIFIED (NOT passes):")
        for w, why in unverified:
            print("  - %s: %s" % (w, why))
    return 0 if (fails == 0 and allproven) else 1


if __name__ == "__main__":
    sys.exit(main())
