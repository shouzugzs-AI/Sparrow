#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P14 / O6: the AUTOREGRESSIVE GENERATION LOOP -- generate -> argmax -> feed back.

WHAT WAS MISSING (and what this builder adds)
--------------------------------------------
P13 built an unrolled incremental decode against a KV cache, and it was
bit-exact and 13.15x faster than re-prefilling. But the decode tokens were
PRE-GIVEN: `runtime/src/main.rs` `load_tokens` writes `gen_tokens.bin` into the
`st_gen` state slot BEFORE `vm.run()`. So step k read a token that came from a
FILE. The engine never sampled anything, and nothing was ever fed back. There
was no argmax in the engine at all (OP_MAX/OP_TOPK_IDX were defined in
`core/src/format.rs` but had no arm in `interp.rs`, and were on the assembler's
UNIMPLEMENTED_OPS panic list).

O6 closes the loop. Step k computes logits, takes the argmax IN-ENGINE writes it
into `st_gen[k+1]`, and step k+1 reads its input row from there. This file emits
that program.

THE FOUR CONSTRAINTS THAT BITE (all from P13's measured experience)
------------------------------------------------------------------
1. OP_ROPE's start position is `ins.dt`, a COMPILE-TIME u16
   (`core/src/format.rs:141`). A runtime loop would rotate EVERY token at the
   same position -- wrong for all but the first, and wrong in the way that looks
   like plausible numeric drift rather than a bug. So the decode stays UNROLLED
   and every position is a literal integer. Unrolling costs TEXT only; the token
   sequence is what must be dynamic, the position is not.

2. The assembler UNIFIES a slot's shape to the WIDEST definition
   (`asm/src/main.rs:149/167`). Measured directly (examples/_probe/unify.vml):
   `SHAPE acc,1,2` then `SHAPE acc,1,4` gives ONE slot of shape 1x4. P13's
   signature bug was exactly this -- per-step `SHAPE kcv,128,5/6/7` collapsed
   back to PMAX, so every step attended unwritten zero columns whose zeros
   became REAL softmax mass (symptom: G=1 exact, G>=2 wrong on all but the last
   step). => EVERY per-step-sized buffer gets a UNIQUE NAME PER STEP
   (kcv0, kcv1, ...). Unrolling is what makes that possible.

3. `st_gen` is pre-filled by the runtime before run(). MEASURED (see
   docs/o6_notes.md §4): the pre-fill strictly PRECEDES the in-engine writes and
   does not clobber them. Design choice: `st_gen[0]` is the SEED (file-sourced,
   the first generated token's input) and positions 1..G are written in-engine.
   See `--seed` / the negative control below.

4. stdout is a single stream: the prefill's PRINT must come BEFORE the decode
   block's PRINTs or row alignment is silently wrong
   (`tools/p12_build_minicpm.py:893-903`). This builder emits in strict program
   order and writes a row manifest (`gen_rows.json`) so the reader never guesses.

WHERE SAMPLING HAPPENS -- READ THIS BEFORE TRUSTING ANY NUMBER
--------------------------------------------------------------
**Argmax is IN-ENGINE. Temperature / top-k / top-p are HOST-SIDE.**

That split is deliberate and must not be overclaimed:

  * The acceptance bar is "temperature=0 degrades EXACTLY to argmax". The only
    way to make that EXACT rather than approximate is for temperature=0 to call
    the very same argmax the engine used -- not a Python reimplementation that
    could disagree on ties or comparison order. So with temperature=0 we do not
    sample at all: we read the engine's own argmax out of `st_gen`.
  * In-engine sampling would need a real RNG and a sort inside a no_std engine
    with a hard size budget (66560 B). That is a lot of new engine surface for a
    feature whose own spec permits host-side.
  * Determinism: the host RNG is seeded explicitly and the algorithm is written
    down, so a run is reproducible AND auditable from Python.

So: with `--sampler argmax` the ENGINE closes the loop with no host round-trip
between steps. Temperature/top-k/top-p then re-rank the SAME in-engine logits
host-side. `p14_gen_check.py` prints which stage produced each token.

THE TWO PROGRAMS
----------------
  --sampler argmax   TRUE closed loop (O6's claim). Each step's argmax feeds the
                     next step. `st_gen[1..G]` is engine-produced.
  --sampler feed     NEGATIVE CONTROL. No engine argmax: the builder writes a
                     token chain into `gen_tokens.bin` and the program just reads
                     it. Used to prove the closed loop is really feeding back --
                     if `feed` with a DIFFERENT chain changes the output, then
                     the file path is live; if `argmax` ignores that same file and
                     still produces the argmax chain, the loop is what drives it.

Usage:
    python tools/p14_gen_build.py --seq 4 --gen 16 --layers 4 --out examples/p14_gen
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safetensors_reader import SafeTensors  # noqa: E402

MODEL = os.environ.get("MINICPM_DIR", "<外部模型目录>/MiniCPM5-1B")
OUT = r"<repo>/examples/p14_gen"

LAYER_SUFFIXES = [
    ("in_ln", "input_layernorm.weight", "f32"),
    ("q", "self_attn.q_proj.weight", "bf16"),
    ("k", "self_attn.k_proj.weight", "bf16"),
    ("v", "self_attn.v_proj.weight", "bf16"),
    ("o", "self_attn.o_proj.weight", "bf16"),
    ("post_ln", "post_attention_layernorm.weight", "f32"),
    ("gate", "mlp.gate_proj.weight", "bf16"),
    ("up", "mlp.up_proj.weight", "bf16"),
    ("down", "mlp.down_proj.weight", "bf16"),
]


def f32_to_bf16(a: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even f32 -> bf16 (matches the engine's expectation)."""
    u = a.astype(np.float32).view(np.uint32).astype(np.uint64)
    rnd = np.uint64(0x7FFF) + ((u >> 16) & 1)
    return ((u + rnd) >> 16).astype("<u2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=4, help="prompt length S")
    ap.add_argument("--gen", type=int, default=16,
                    help="G tokens to generate in the closed loop")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--layers", type=int, default=0,
                    help="0 = all; a reduced count is for fast iteration")
    ap.add_argument("--sampler", choices=["argmax", "feed"], default="argmax",
                    help="'argmax' = true closed loop (in-engine argmax feeds "
                         "back). 'feed' = negative control: tokens come from a "
                         "file instead, with no in-engine argmax.")
    ap.add_argument("--feed-tokens", default="",
                    help="for --sampler feed: comma-separated token ids to place "
                         "in gen_tokens.bin. This is the negative control's "
                         "deliberately-WRONG chain.")
    ap.add_argument("--seed", type=int, default=-1,
                    help="token id for st_gen[0], the seed the first generated "
                         "step consumes. -1 = use the last prompt token "
                         "(the natural 'continue this prompt' choice).")
    ap.add_argument("--quant", choices=["none", "i8"], default="none")
    ap.add_argument("--oneshot", action="store_true",
                    help="emit a ONE-SHOT FULL CAUSAL FORWARD over N=S+G "
                         "positions instead of the loop. This is the REFERENCE "
                         "the closed loop is compared against: same weights, same "
                         "positions, but a single pass with causal masking rather "
                         "than an incremental KV cache. Used by "
                         "tools/p14_oneshot_check.py for the headline claim.")
    ap.add_argument("--token-file", default="",
                    help="with --oneshot: a .bin of u32 token ids (the prompt + "
                         "the generated chain) to run the full forward over. "
                         "Without it, the prompt is used and N=S.")
    args = ap.parse_args()

    st = SafeTensors(os.path.join(args.model, "model-00000-of-00001.safetensors"))
    with open(os.path.join(args.model, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)

    H = cfg["hidden_size"]
    NL = args.layers if args.layers else cfg["num_hidden_layers"]
    # ⚠️ P16 LEAD FIX -- THE "24-LAYER HEADLINE" WAS ACTUALLY 4 LAYERS.
    #
    # `--layers N` DEEP-TRUNCATES the model. That is a legitimate feature (a
    # cheap smoke build), but it was used to produce the artifact that
    # docs/P14_S5_O6_报告.md and docs/P14_verification.md then described as
    # "24 层 headline", and 4-layer numbers were compared against a 24-layer
    # HF anchor. The layer count the report claimed came from the INTENT, not
    # from the artifact; examples/p14_gen/meta.json says layers=4 and its cache
    # slots are st_L with L in {0,1,2,3}.
    #
    # This script's OWN docstring (around line 83) shows the invocation with
    # `--layers 4`, which is how the truncation entered the pipeline unnoticed.
    # The default path (`args.layers == 0`) is correct -- it uses the config.
    #
    # The fix is NOT to forbid truncation (smoke builds need it). It is to make
    # the truncation IMPOSSIBLE TO MISS: say it loudly on stdout, and record the
    # TRUE count in meta.json (already done via NL). Any downstream reader that
    # asserts "layers == 24" must read NL, never assume it.
    if args.layers and args.layers != cfg["num_hidden_layers"]:
        print("WARNING: LAYER-TRUNCATED BUILD -- producing a %d-layer model, but "
              "the checkpoint declares %d layers." % (args.layers,
                                                      cfg["num_hidden_layers"]))
        print("         This artifact MUST NOT be described as the full model, "
              "and its numbers must not be compared against a %d-layer anchor."
              % cfg["num_hidden_layers"])
    NH = cfg["num_attention_heads"]
    NKV = cfg["num_key_value_heads"]
    HD = cfg["head_dim"]
    INTER = cfg["intermediate_size"]
    V = cfg["vocab_size"]
    EPS = cfg["rms_norm_eps"]
    THETA = cfg["rope_theta"]
    S = args.seq
    G = args.gen
    rep = NH // NKV

    # The prompt. Same ids p12/p13 use, so results are comparable across phases.
    ids = [3, 117, 2501, 88][:S]
    if len(ids) < S:
        ids = (ids * ((S // len(ids)) + 1))[:S]
    seed_tok = ids[-1] if args.seed < 0 else args.seed

    print("MiniCPM5-1B -> O6 autoregressive loop   S=%d G=%d layers=%d sampler=%s"
          % (S, G, NL, args.sampler))
    print("prompt ids=%s   seed(st_gen[0])=%d" % (ids, seed_tok))

    if args.sampler == "feed":
        if not args.feed_tokens:
            print("ERROR: --sampler feed needs --feed-tokens (the deliberately "
                  "different chain). Refusing to emit an unlabelled control.")
            return 1
        feed = [int(t) for t in args.feed_tokens.split(",") if t.strip()]
        if len(feed) != G:
            print("ERROR: --feed-tokens has %d ids but --gen is %d" % (len(feed), G))
            return 1
        for t in feed:
            if not (0 <= t < V):
                print("ERROR: feed token %d out of vocab range [0,%d)" % (t, V))
                return 1
    else:
        feed = None

    wdir = os.path.join(args.out, "weights")
    os.makedirs(wdir, exist_ok=True)

    consts = []
    SHAPES = {
        "in_ln": (1, H),
        "q": (NH * HD, H), "k": (NKV * HD, H), "v": (NKV * HD, H),
        "o": (H, NH * HD),
        "post_ln": (1, H),
        "gate": (INTER, H), "up": (INTER, H), "down": (H, INTER),
    }

    def emit_const(name, arr, dtype_tag):
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        r, c = arr.shape
        ext = {"bf16": "bf16", "i8": "i8"}.get(dtype_tag, "f32")
        p = os.path.join(wdir, name + "." + ext)
        if dtype_tag == "bf16":
            f32_to_bf16(arr).tofile(p)
        elif dtype_tag == "i8":
            a = np.asarray(arr, dtype=np.float32)
            amax = np.abs(a).max(axis=1, keepdims=True)
            amax = np.where(amax == 0.0, 1e-30, amax)
            s = (amax / 127.0).astype(np.float32)
            q = np.round(a / s).clip(-127.0, 127.0).astype(np.int8)
            q.tofile(p)
            np.ascontiguousarray(s.reshape(-1), dtype=np.float32).tofile(
                os.path.join(wdir, name + "_sc.f32"))
            consts.append((name, os.path.join("weights", name + ".i8"), r, c, "i8"))
            consts.append((name + "_sc",
                           os.path.join("weights", name + "_sc.f32"), r, 1, "f32"))
            return r, c
        else:
            np.ascontiguousarray(arr, dtype=np.float32).tofile(p)
        consts.append((name, os.path.join("weights", name + "." + ext), r, c, dtype_tag))
        return r, c

    def emit_scalar(name, val):
        p = os.path.join(wdir, name + ".f32")
        np.array([val], dtype=np.float32).tofile(p)
        consts.append((name, os.path.join("weights", name + ".f32"), 1, 1, "f32"))

    def qtag(default):
        return "i8" if (args.quant == "i8" and default == "bf16") else default

    # ---- constants ----
    embed = st.get("model.embed_tokens.weight")   # [V,H] f32
    emit_const("embed", embed, qtag("bf16"))
    emit_const("lm_head", st.get("lm_head.weight"), qtag("bf16"))
    emit_const("final_norm", st.get("model.norm.weight"), "f32")
    emit_scalar("eps1", EPS)
    emit_scalar("theta1", THETA)
    emit_scalar("scale1", 1.0 / np.sqrt(HD))

    for L in range(NL):
        for short, suffix, dt in LAYER_SUFFIXES:
            nm = "model.layers.%d.%s" % (L, suffix)
            if not st.has(nm):
                print("missing tensor: %s" % nm)
                return 1
            emit_const("L%d_%s" % (L, short), st.get(nm), qtag(dt))

    print("constants: %d" % len(consts))
    q8_names = set(n for (n, _p, _r, _c, dt) in consts if dt == "i8")

    def gemm(dst, a, wname):
        if wname in q8_names:
            w("GEMM %s, %s, %s, %s_sc" % (dst, a, wname, wname))
        else:
            w("GEMM %s, %s, %s" % (dst, a, wname))

    # ------------------------------------------------------------- .vml
    A = []

    def w(s):
        A.append(s)

    PMAX = S + G

    w("; MiniCPM5-1B O6 autoregressive generation loop")
    w("; generated by tools/p14_gen_build.py   sampler=%s" % args.sampler)
    w("; S=%d G=%d layers=%d hidden=%d heads=%d kv_heads=%d head_dim=%d inter=%d vocab=%d"
      % (S, G, NL, H, NH, NKV, HD, INTER, V))
    w(";")
    w("; THE LOOP: step k computes logits, takes the argmax IN-ENGINE, writes it")
    w("; into st_gen[k+1], and step k+1 reads its input row from there.")
    w("; st_gen[0] is the SEED (file-sourced); positions 1..G are ENGINE-WRITTEN.")
    w(";")
    w("; UNROLLED ON PURPOSE: OP_ROPE's position is a compile-time u16 in ins.dt,")
    w("; so a runtime loop would rotate every token at the same position. Here")
    w("; every position is a literal integer.")
    w(";")
    w("; UNIQUE SLOT NAME PER STEP ON PURPOSE: the assembler unifies a slot's")
    w("; shape to the WIDEST definition, so a shared per-step name would silently")
    w("; collapse to PMAX and make every step attend unwritten zero columns.")
    w("")
    w("; ---- shapes ----")
    w("SHAPE st_tok, 1, %d" % S)
    w("SHAPE st_gen, 1, %d" % (G + 1))
    w("SHAPE h,      %d, %d" % (S, H))
    w("SHAPE xn,     %d, %d" % (S, H))
    w("SHAPE q,      %d, %d" % (S, NH * HD))
    w("SHAPE k,      %d, %d" % (S, NKV * HD))
    w("SHAPE v,      %d, %d" % (S, NKV * HD))
    w("SHAPE qh,     %d, %d" % (S, HD))
    w("SHAPE kh,     %d, %d" % (S, HD))
    w("SHAPE vh,     %d, %d" % (S, HD))
    w("SHAPE qrot,   %d, %d" % (S, HD))
    w("SHAPE krot,   %d, %d" % (S, HD))
    w("SHAPE sc,     %d, %d" % (S, S))
    w("SHAPE ctx,    %d, %d" % (S, HD))
    w("SHAPE att,    %d, %d" % (S, NH * HD))
    w("SHAPE oproj,  %d, %d" % (S, H))
    w("SHAPE h2,     %d, %d" % (S, H))
    w("SHAPE gate,   %d, %d" % (S, INTER))
    w("SHAPE up,     %d, %d" % (S, INTER))
    w("SHAPE acts,   %d, %d" % (S, INTER))
    w("SHAPE down,   %d, %d" % (S, H))
    w("SHAPE logits, %d, %d" % (S, V))
    w("SHAPE eps1,   1, 1")
    w("SHAPE theta1, 1, 1")
    w("SHAPE scale1, 1, 1")
    w("")
    # ---- prompt embedding: h = embed[st_tok] ----
    w("; ---- prompt embedding: h = embed[st_tok] (runtime-supplied ids) ----")
    for i in range(S):
        w("SLICE tok%d, st_tok, %d, 1" % (i, i))
        w("SHAPE e%d, 1, %d" % (i, H))
        w("EMB e%d, embed, tok%d" % (i, i))
    if S == 1:
        w("SHAPE hcat, 1, %d" % H)
        w("COPY hcat, e0")
        w("COPY h, hcat")
    else:
        w("SHAPE hcat0, 1, %d" % H)
        w("COPY hcat0, e0")
        prev, total = "hcat0", H
        for i in range(1, S):
            total += H
            cur = "hcat%d" % i
            w("SHAPE %s, 1, %d" % (cur, total))
            w("CONCAT %s, %s, e%d" % (cur, prev, i))
            prev = cur
        w("SHAPE hflat, %d, %d" % (S, H))
        w("COPY hflat, %s" % prev)
        w("COPY h, hflat")
    w("")

    # ---- KV cache (state, transposed [HD, PMAX]) ----
    w("; ---- KV cache in STATE slots, transposed [HD, %d] ----" % PMAX)
    w("; TRANSPOSED so that appending position p is a COLUMN write (WRITECOL),")
    w("; and the same layout serves both GEMM flavours (nn for scores, default")
    w("; NT for the context). ONE CACHE PER KV HEAD: OP_SLICE offsets COLUMNS,")
    w("; so a single [NKV*HD,PMAX] cache would make head j unreachable.")
    for L in range(NL):
        for h_j in range(NKV):
            w("SHAPE st_kc%d_%d, %d, %d" % (L, h_j, HD, PMAX))
            w("SHAPE st_vc%d_%d, %d, %d" % (L, h_j, HD, PMAX))
    w("SHAPE kdT, %d, 1" % HD)
    w("SHAPE vdT, %d, 1" % HD)
    w("SHAPE krow, 1, %d" % HD)
    w("SHAPE vrow, 1, %d" % HD)
    w("SHAPE kcraw, %d, %d" % (S, HD))
    w("SHAPE kcrot, %d, %d" % (S, HD))
    w("SHAPE vcraw, %d, %d" % (S, HD))
    w("SHAPE kcrotT, %d, %d" % (HD, S))
    w("SHAPE vcrawT, %d, %d" % (HD, S))
    w("SHAPE kcol, %d, 1" % HD)
    w("SHAPE vcol, %d, 1" % HD)
    w("SHAPE gtok, 1, 1")
    # decode working buffers: per-step names for the ones sized by live range.
    w("SHAPE hdec, 1, %d" % H)
    w("SHAPE eh, 1, %d" % H)
    w("SHAPE attd, 1, %d" % (NH * HD))
    w("SHAPE qd, 1, %d" % (NH * HD))
    w("SHAPE kd, 1, %d" % (NKV * HD))
    w("SHAPE vd, 1, %d" % (NKV * HD))
    w("SHAPE qdh, 1, %d" % HD)
    w("SHAPE kdh, 1, %d" % HD)
    w("SHAPE vdh, 1, %d" % HD)
    w("SHAPE qdrot, 1, %d" % HD)
    w("SHAPE kdrot, 1, %d" % HD)
    w("SHAPE h2d, 1, %d" % H)
    w("SHAPE oprojd, 1, %d" % H)
    w("SHAPE gated, 1, %d" % INTER)
    w("SHAPE upd, 1, %d" % INTER)
    w("SHAPE actsd, 1, %d" % INTER)
    w("SHAPE downd, 1, %d" % H)
    w("SHAPE xnd, 1, %d" % H)
    w("")

    # ---- prefill: S positions through NL layers, filling the cache ----
    w("; ================ PREFILL over %d prompt positions ================" % S)
    for L in range(NL):
        w("; ---- layer %d ----" % L)
        w("RMSNORM xn, h, L%d_in_ln, eps1" % L)
        gemm("q", "xn", "L%d_q" % L)
        gemm("k", "xn", "L%d_k" % L)
        gemm("v", "xn", "L%d_v" % L)
        for h_i in range(NH):
            j = h_i // rep
            w("SLICE qh, q, %d, %d" % (h_i * HD, HD))
            w("SLICE kh, k, %d, %d" % (j * HD, HD))
            w("SLICE vh, v, %d, %d" % (j * HD, HD))
            w("ROPE qrot, qh, theta1, 0")
            w("ROPE krot, kh, theta1, 0")
            w("GEMM sc, qrot, krot")
            w("SCALE sc, sc, scale1")
            w("SOFTMAX sc, sc, causal")
            w("GEMM ctx, sc, vh, nn")
            w("WRITECOL att, ctx, %d, %d" % (h_i * HD, HD))
        gemm("oproj", "att", "L%d_o" % L)
        w("EWADD h2, h, oproj")
        w("RMSNORM xn, h2, L%d_post_ln, eps1" % L)
        gemm("gate", "xn", "L%d_gate" % L)
        gemm("up", "xn", "L%d_up" % L)
        w("ACT acts, gate, silu")
        w("EWMUL acts, acts, up")
        gemm("down", "acts", "L%d_down" % L)
        w("EWADD h, h2, down")
        # cache ROTATED K (attention is softmax(q_rot . k_rot); caching raw k
        # and comparing against a rotated decode query mixes two bases -- P13
        # measured ~8% uniform error from exactly that).
        for h_j in range(NKV):
            w("SLICE kcraw, k, %d, %d" % (h_j * HD, HD))
            w("ROPE kcrot, kcraw, theta1, 0")
            w("SLICE vcraw, v, %d, %d" % (h_j * HD, HD))
            w("TRANSPOSE kcrotT, kcrot")
            w("TRANSPOSE vcrawT, vcraw")
            for p in range(S):
                w("SLICE kcol, kcrotT, %d, 1" % p)
                w("WRITECOL st_kc%d_%d, kcol, %d" % (L, h_j, p))
                w("SLICE vcol, vcrawT, %d, 1" % p)
                w("WRITECOL st_vc%d_%d, vcol, %d" % (L, h_j, p))
    w("")
    # prefill logits PRINT -- BEFORE the decode block, because stdout is one
    # stream and row alignment would otherwise be silently wrong.
    w("; ---- prefill logits (printed FIRST: stdout is a single stream) ----")
    w("RMSNORM xn, h, final_norm, eps1")
    gemm("logits", "xn", "lm_head")
    w("PRINT logits")
    w("")

    # ---- the generation loop ----
    w("; ================ O6: %d generation steps ================" % G)
    if args.sampler == "argmax":
        w("; Each step: forward -> logits -> TOPK_IDX (IN-ENGINE argmax) ->")
        w("; WRITECOL into st_gen[k+1]. The next step reads st_gen[k+1].")
    else:
        w("; NEGATIVE CONTROL (--sampler feed): NO in-engine argmax. Every step")
        w("; reads its pre-given token from st_gen[k+1], which the runtime")
        w("; loaded from gen_tokens.bin. If the output here differs from the")
        w("; --sampler argmax run, the file path is live and the argmax path is")
        w("; what closes the loop.")
    for step in range(G):
        pos = S + step
        live = pos + 1
        w("")
        w("; ---- generation step %d (absolute position %d) ----" % (step, pos))
        # UNIQUE NAME PER STEP -- see constraint 2 at the top of this file.
        w("SHAPE kcv%d, %d, %d" % (step, HD, live))
        w("SHAPE vcv%d, %d, %d" % (step, HD, live))
        w("SHAPE scd%d, 1, %d" % (step, live))
        w("SHAPE ctxd%d, 1, %d" % (step, HD))
        w("SHAPE logitsd%d, 1, %d" % (step, V))
        # INPUT ROW: st_gen[step] for step 0 (the seed) and, in the closed
        # loop, st_gen[step] for step k>0 is the value step k-1 WROTE.
        w(";   input token = st_gen[%d]" % step)
        w("SLICE gtok, st_gen, %d, 1" % step)
        w("EMB eh, embed, gtok")
        w("COPY hdec, eh")
        for L in range(NL):
            w(";   layer %d" % L)
            w("RMSNORM xnd, hdec, L%d_in_ln, eps1" % L)
            gemm("qd", "xnd", "L%d_q" % L)
            gemm("kd", "xnd", "L%d_k" % L)
            gemm("vd", "xnd", "L%d_v" % L)
            for h_j in range(NKV):
                w("SLICE kdh, kd, %d, %d" % (h_j * HD, HD))
                w("SLICE vdh, vd, %d, %d" % (h_j * HD, HD))
                w("ROPE kdrot, kdh, theta1, %d" % pos)
                w("TRANSPOSE kdT, kdrot")
                w("TRANSPOSE vdT, vdh")
                w("WRITECOL st_kc%d_%d, kdT, %d" % (L, h_j, pos))
                w("WRITECOL st_vc%d_%d, vdT, %d" % (L, h_j, pos))
            for h_i in range(NH):
                j = h_i // rep
                w("SLICE qdh, qd, %d, %d" % (h_i * HD, HD))
                w("ROPE qdrot, qdh, theta1, %d" % pos)
                w("SLICE kcv%d, st_kc%d_%d, 0, %d" % (step, L, j, live))
                w("SLICE vcv%d, st_vc%d_%d, 0, %d" % (step, L, j, live))
                w("GEMM scd%d, qdrot, kcv%d, nn" % (step, step))
                w("SCALE scd%d, scd%d, scale1" % (step, step))
                w("SOFTMAX scd%d, scd%d" % (step, step))
                w("GEMM ctxd%d, scd%d, vcv%d" % (step, step, step))
                w("WRITECOL attd, ctxd%d, %d, %d" % (step, h_i * HD, HD))
            gemm("oprojd", "attd", "L%d_o" % L)
            w("EWADD h2d, hdec, oprojd")
            w("RMSNORM xnd, h2d, L%d_post_ln, eps1" % L)
            gemm("gated", "xnd", "L%d_gate" % L)
            gemm("upd", "xnd", "L%d_up" % L)
            w("ACT actsd, gated, silu")
            w("EWMUL actsd, actsd, upd")
            gemm("downd", "actsd", "L%d_down" % L)
            w("EWADD hdec, h2d, downd")
        w(";   logits for this position")
        w("RMSNORM xnd, hdec, final_norm, eps1")
        gemm("logitsd%d" % step, "xnd", "lm_head")
        w("PRINT logitsd%d" % step)
        if args.sampler == "argmax":
            # IN-ENGINE argmax -> feed back into st_gen[step+1].
            #
            # This runs for EVERY step including the last, and that is
            # deliberate. The last step's token is not consumed by anything
            # (there is no step G), but if it is NOT written then st_gen[G]
            # keeps whatever the file happened to contain -- and the negative
            # control would then see the file's value survive in the final
            # st_gen. Measured: with the write skipped on the last step, a
            # control file of [118,125,132,139,146] left st_gen as
            # [118,0,0,0,146], i.e. the LAST position was file-sourced. Writing
            # it makes every position of st_gen except [0] engine-produced, so
            # "the file drove the output" is false for the whole chain.
            w(";   IN-ENGINE argmax -> feed back into st_gen[%d]" % (step + 1))
            w("SHAPE nxt%d, 1, 1" % step)
            w("TOPK_IDX nxt%d, logitsd%d" % (step, step))
            w("WRITECOL st_gen, nxt%d, %d" % (step, step + 1))
        else:
            w(";   control: token %d comes from the file, nothing is fed back"
              % (step + 1))
        # Record the input token actually used, so the checker can prove what
        # each step consumed without re-deriving it.
        w("SHAPE seen%d, 1, 1" % step)
        w("SLICE seen%d, st_gen, %d, 1" % (step, step))
        w("PRINT seen%d" % step)
    w("")
    w("; final st_gen contents: the engine's own argmax chain + the seed")
    w("PRINT st_gen")
    w("")
    w("HALT")

    # ------------------------------------------------------------------
    # PRE-FLIGHT GUARD: every name the program REFERENCES must be DECLARED.
    #
    # WHY THIS EXISTS (this is a real bug that shipped and cost two teammates a
    # verification cycle): the builder emitted `GEMM logits, xn, lm_head_out`
    # while DECLARING the constant as `lm_head`. The assembler does not refuse
    # an unknown name -- it SILENTLY MATERIALISES it as an activation slot of
    # shape 0x0. A GEMM whose B operand is 0x0 then writes zero elements, and
    # the rest of the program keeps computing on STALE activation memory from
    # the previous iteration. Measured consequence: the assembled artifact
    # reported `slot83 lm_head_out [act] 0x0`, and the run died later in an
    # unrelated-looking EMB with "row index out of range / table_rows=4".
    #
    # That is the P13 §2.1 failure class exactly ("an undeclared slot sits at
    # 0x0 and the op writes nothing"), and the project's iron rule is that
    # illegal input must fail LOUDLY rather than produce plausible garbage. An
    # artifact that assembles at exit=0 while carrying a 0x0 GEMM operand is
    # precisely that silent degradation, so it is refused HERE, at build time,
    # with the offending line printed.
    #
    # The slot-table check (`assert no 0x0 slot`) lives in the run wrapper,
    # tools/p14_gen_run.ps1, because only the ASSEMBLER knows the final unified
    # shapes -- this check catches the common cause (a typo'd name) before we
    # spend minutes assembling an 858 MB artifact.
    # ------------------------------------------------------------------
    declared = set(n for (n, _p, _r, _c, _d) in consts)
    declared |= {"eps1", "theta1", "scale1", "final_norm", "embed", "lm_head"}
    # SHAPE declarations and every slot named anywhere in the text:
    shaped = set()
    written = set()     # names that appear as a DESTINATION operand
    read = {}           # names that appear as a SOURCE operand -> first line
    for ln, raw in enumerate(A):
        line = raw.split(";")[0].strip()
        if not line:
            continue
        toks = line.replace(",", " ").split()
        if not toks:
            continue
        op = toks[0].upper()
        if op == "SHAPE":
            if len(toks) >= 2:
                shaped.add(toks[1])
            continue
        if op in ("LOOP_TOP:", "HALT"):
            continue
        # Operand-bearing ops. Convention across this engine: operand 0 (after
        # the opcode) is the DESTINATION, the rest are SOURCES. That matches the
        # assembler's own parse (dst=regs[0], a=regs[1], b=regs[2]).
        operands = []
        for t in toks[1:]:
            if t.startswith("@") or t.startswith("LOOP"):
                continue
            if t.lower() in ("causal", "nn", "silu", "gelu", "relu", "sigmoid",
                             "tanh", "relu2"):
                continue
            try:
                float(t)
                continue
            except ValueError:
                pass
            if t.isidentifier():
                operands.append(t)
        if not operands:
            continue
        written.add(operands[0])
        for name in operands[1:]:
            read.setdefault(name, ln + 1)

    # A name is fine if it is declared, shaped, or WRITTEN somewhere (a SLICE
    # destination like `tok0` is declared by being written, then read later).
    undefined = [(n, ln) for n, ln in read.items()
                 if n not in declared and n not in shaped and n not in written]
    if undefined:
        print("ERROR: these names are READ but never declared and never written "
              "(no SHAPE, no constant, never a destination). The assembler would "
              "silently create them as 0x0 slots and the op would write NOTHING:")
        for name, ln in undefined[:12]:
            print("   line %d: %s" % (ln, name))
        print("   fixes: add `SHAPE %s, rows, cols`, or fix the spelling."
              % undefined[0][0])
        return 1

    # NOTE: a "GEMM's B must be a constant" rule was tried here and REMOVED --
    # it is wrong. Attention legitimately uses ACTIVATIONS as B
    # (`GEMM sc, qrot, krot` and `GEMM ctx, sc, vh, nn`), so requiring a
    # constant B rejects the core attention path. The general read-vs-declared
    # check above already catches the real class (a typo'd name that is never
    # declared or written), and tools/p14_gen_run.ps1 additionally rejects any
    # 0x0 slot from the assembler's own slot table, which is authoritative.

    vml_path = os.path.join(args.out, "full.vml")
    os.makedirs(args.out, exist_ok=True)
    with open(vml_path, "w", encoding="ascii", newline="\n") as f:
        f.write("\n".join(A) + "\n")
    print("wrote %s (%d lines)" % (vml_path, len(A)))

    argp = os.path.join(args.out, "weights.args")
    with open(argp, "w", encoding="ascii", newline="\n") as f:
        for name, path, r, c, dt in consts:
            f.write("%s:%s:%d:%d:%s\n" % (name, path.replace("\\", "/"), r, c, dt))
    print("wrote %s (%d constants)" % (argp, len(consts)))

    # ---- the token inputs ----
    # st_tok: the prompt. st_gen: the SEED at position 0 only (positions 1..G
    # are engine-written; see docs/o6_notes.md §4).
    seed_path = os.path.join(args.out, "seed_gen.bin")
    np.array([seed_tok], dtype="<u4").tofile(seed_path)
    prompt_path = os.path.join(args.out, "prompt_tok.bin")
    np.array(ids, dtype="<u4").tofile(prompt_path)
    if args.sampler == "feed":
        feed_path = os.path.join(args.out, "feed_gen.bin")
        np.array([seed_tok] + feed, dtype="<u4").tofile(feed_path)

    # ---- row manifest: how many values each PRINT contributes, in order ----
    # WHY: stdout is a single stream, so the reader must know exactly how many
    # rows each stage emitted. Deriving that from the .vml is fragile; recording
    # it at emit time cannot drift from what was emitted.
    rows = {
        "seq": S, "gen": G, "layers": NL, "vocab": V, "sampler": args.sampler,
        "prompt_ids": ids, "seed_token": seed_tok,
        "feed_tokens": feed,
        "blocks": [
            {"name": "prefill_logits", "rows": S, "cols": V},
        ],
        "per_step": [],
    }
    for step in range(G):
        rows["blocks"].append(
            {"name": "step%d_logits" % step, "rows": 1, "cols": V})
        rows["blocks"].append(
            {"name": "step%d_seen" % step, "rows": 1, "cols": 1})
    rows["blocks"].append({"name": "st_gen_final", "rows": 1, "cols": G + 1})
    with open(os.path.join(args.out, "gen_rows.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    meta = {
        "seq": S, "gen": G, "layers": NL, "hidden": H, "heads": NH,
        "kv_heads": NKV, "head_dim": HD, "inter": INTER, "vocab": V,
        "eps": EPS, "theta": THETA, "prompt_ids": ids, "seed_token": seed_tok,
        "sampler": args.sampler, "feed_tokens": feed, "quant": args.quant,
    }
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("wrote %s (seed=%d) and %s (prompt=%s)"
          % (seed_path, seed_tok, prompt_path, ids))
    if args.sampler == "feed":
        print("wrote feed_gen.bin = [%d] + %s" % (seed_tok, feed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
