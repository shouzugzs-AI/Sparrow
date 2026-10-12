#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P12: convert MiniCPM5-1B into a loadable VMLM model.

WHAT THIS PRODUCES
    examples/minicpm_full/
        weights/*.bf16        one file per constant, bf16, NOT widened
        full.vml              the program (embed -> 24 layers -> norm -> head)
        (then tools/asm_all-style assembly makes full.vmb)

WHY bf16 PASSES THROUGH UNCHANGED
    The engine now has a bf16 constant pool + a widening GEMM (P12), measured
    4.4-7.9x faster than f32 because this machine's GEMM is load-port bound.
    Widening to f32 on disk would double the file to 4.32 GB for nothing.

WHY THE LAYER BODY IS UNROLLED 24x (and not looped)
    The P10 loop (LOOP_CNT) works, and P10 proved instruction count is flat in
    layer count. But weight addressing is STATIC: slot -> const is bound once
    at load. Looping over layers would need a runtime const base (P10 §16.8
    item 5). Unrolling keeps this deliverable free of new engine features --
    and it still validates all the numerics the looped version would rely on.
    Instruction count is ~1500, i.e. ~24 KB of TEXT, versus 2.16 GB of weights.

WHAT IS *NOT* OPTIMIZED HERE
    This is a CORRECTNESS-first loader: one token, full 24 layers, compared
    against HuggingFace logits. Prefill/decode batching, KV cache and the E4/E5
    precomputations are separate work.

Usage:
    python tools/p12_build_minicpm.py --seq 4
"""

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safetensors_reader import SafeTensors  # noqa: E402

# ---- S5-D2: the production neuron_kv section writer -------------------------
#
# Imported at module scope so an ImportError is a startup failure, not a
# surprise minutes into a 2.16 GB weight dump. The geometry it derives is NOT
# imported from a test harness: `cross_check_against_rust()` re-parses
# core/src/format.rs, and `derive_geometry()` takes THIS build's own dimensions.
try:
    from p15_s5_production_section import (  # noqa: E402
        GeometryError, build_section, cross_check_against_rust,
        derive_geometry, install_into_vmb, patched_meta_end, read_vmb_header,
        section_bytes, verify_numerics_unchanged, verify_section_of)
except ImportError as _e:  # pragma: no cover - only on a broken checkout
    raise SystemExit(
        "ERROR: tools/p15_s5_production_section.py is required by the "
        "production pipeline (S5-D2): %s" % _e)

ROOT_ABS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL = os.environ.get("MINICPM_DIR", "<外部模型目录>/MiniCPM5-1B")
OUT = r"<repo>/examples/minicpm_full"

# Which tensors become constants, in the order the .vml uses them.
# `dt` is the storage dtype: bf16 for the big matrices (halves the bytes and
# the engine widens inside GEMM), f32 for the 1-D norms and scalars.
#
# WHY THE NORMS ARE f32: `OP_RMSNORM` (and every op other than GEMM) reads its
# operands as f32. The engine's load-time audit rejects a bf16 constant used by
# a non-GEMM op -- measured: it caught `op 31 (RMSNORM) uses bf16 const`.
# The norms are 1536 floats each (6 KB in f32), i.e. 0.0001% of the model, so
# there is nothing to gain by making RMSNORM dtype-aware.
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


def int8_quant(w: np.ndarray):
    """Symmetric per-output-row int8, the layout the engine's kernel expects.

    Returns (q[N,K] int8, scale[N] f32) with C = (A . q) * scale.

    WHY PER-ROW AND NOT PER-TENSOR: the P11 kernel benchmark measured 3.44x for
    per-row int8 vs f32, and a single global scale would lose most of the range
    on matrices whose rows differ in magnitude.

    WHY NOT OUTLIER MIXED PRECISION (P11's LLM.int8 design): measured on the
    REAL MiniCPM weights (tools/p13_quant_plan.py), the outlier detector finds
    **0 of 1536** columns -- MiniCPM's per-column RMS spread is only 1.16-2.26x,
    where P11's synthetic test used 12x. Adding the outlier path changed the
    relative logit error from 2.89e-02 to 2.74e-02, i.e. essentially nothing.
    So it would be unproven complexity here; P11's own conclusion 4 says to
    validate quantisation on real weights, and on real weights it is not needed.
    """
    w = np.asarray(w, dtype=np.float32)
    amax = np.abs(w).max(axis=1, keepdims=True)
    # A degenerate (all-zero) row must get a NON-ZERO scale, otherwise the
    # division below yields NaN and the whole layer goes silently NaN.
    amax = np.where(amax == 0.0, 1e-30, amax)
    s = (amax / 127.0).astype(np.float32)
    q = np.round(w / s).clip(-127.0, 127.0).astype(np.int8)
    return q, s.reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=4, help="sequence length (prompt)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", default=OUT)
    # A full 24-layer graph is big; allow a reduced layer count for iteration.
    ap.add_argument("--layers", type=int, default=0, help="0 = all")
    ap.add_argument("--skip-head", action="store_true",
                    help="skip lm_head (401 MB) -- for layer-loop debugging")
    ap.add_argument("--with-embed", action="store_true",
                    help="also embed the full 401 MB token matrix. OFF by "
                         "default: the prompt rows are copied in directly, so "
                         "the matrix is dead weight in the artifact.")
    ap.add_argument("--print-hidden", action="store_true",
                    help="with --skip-head, print the FULL post-norm hidden "
                         "state so it can be compared against HF's "
                         "output_hidden_states[-1]")
    # ---- P13 task A ----
    ap.add_argument("--prompt-mode", choices=["const", "tokens"], default="const",
                    help="'const' (pre-P13): prompt rows are baked in as a "
                         "constant, so changing the prompt means reassembling "
                         "the whole 1.76 GB model. "
                         "'tokens': the program carries NO prompt data; it "
                         "reads token ids from a `tokens.bin` file into a "
                         "state slot at runtime, so ONE .vmb serves any prompt.")
    # ---- P13 task C: incremental decode with a KV cache ----
    ap.add_argument("--decode", type=int, default=0, metavar="G",
                    help="after prefilling S tokens, generate G more tokens "
                         "incrementally against a K/V cache held in state "
                         "slots. Without this, each generated token re-runs the "
                         "WHOLE sequence (measured: 14.6 s for P=64,G=16 for a "
                         "4-layer model), because the engine had no way to carry "
                         "K/V across steps.")
    # ---- P13 task E: int8 weights ----
    ap.add_argument("--quant", choices=["none", "i8"], default="none",
                    help="'i8': store the projection matrices as int8 with a "
                         "per-output-row scale, and have the engine dequantise "
                         "inside the GEMM. Halves the weight bytes AGAIN versus "
                         "bf16 (1.76 GB -> ~0.9 GB), and P11 measured the "
                         "kernel 3.4x faster than f32 because this machine's "
                         "GEMM is weight-byte bound. NOT the default: see "
                         "docs/P13_report for the accuracy tradeoff.")
    ap.add_argument("--loop", action="store_true",
                    help="P13 task E: emit ONE layer template and run it NL "
                         "times with LOOP_CNT, advancing a runtime constant base "
                         "(OP_LAYERB) instead of unrolling 24 copies. Requires "
                         "every layer's constants to have an identical layout "
                         "and uniform stride; tools/p13_loop_check.py verifies "
                         "the result against the unrolled build.")
    # ---- S5-D2: neuron_kv_offsets section in the PRODUCTION artifact ----
    #
    # DEFECT THIS CLOSES (independent verifier, docs/P15_S5_wiring_verification.md
    # 4.2): this production pipeline emitted NO neuron_kv section at all. The only
    # 24-layer artifact carrying one was built by the p15_s5_e2e.py HARNESS. So
    # "S5 is wired" was not true of the artifact a real run loads.
    #
    # The section is written into the .vmb by the ASSEMBLE step
    # (`--emit-neuron-kv` below), because the container needs a 12-byte section
    # descriptor that the assembler does not emit yet.
    ap.add_argument("--emit-neuron-kv", dest="emit_neuron_kv",
                    action="store_true", default=True,
                    help="S5: emit a neuron_kv_offsets section describing this "
                         "artifact's own KV-cache geometry, so the engine's "
                         "--s5-load path can parse the PRODUCTION artifact. The "
                         "geometry is DERIVED from this run's n_layers / "
                         "n_kv_heads / head_dim / PMAX, never from a test's "
                         "constants. Default ON: the point of D2 is that the "
                         "production artifact is the one that carries the index.")
    ap.add_argument("--no-neuron-kv-section", dest="emit_neuron_kv",
                    action="store_false",
                    help="S5 backward-compatibility CONTROL: emit the .vmb "
                         "exactly as before (text_off=32, no descriptor, no "
                         "section). Used to demonstrate that an artifact WITHOUT "
                         "the section loads exactly as it did pre-S5.")
    ap.add_argument("--neuron-kv-stride", type=int, default=None,
                    help="S5: override the section stride in ELEMENTS. Default "
                         "(and the only layout S5 accepts as tight) is "
                         "max_seq_per_neuron. Do NOT pass "
                         "head_dim*max_seq -- that is the documented 128x "
                         "S5-A defect that inflated MiniCPM from 25 MB to "
                         "3.22 GB; the emitter rejects stride < max_seq.")
    ap.add_argument("--assemble", action="store_true",
                    help="S5-D2: after writing full.vml + weights.args, run "
                         "vmlm-asm to produce <out>/m.vmb and (unless "
                         "--no-neuron-kv-section) install the neuron_kv_offsets "
                         "section into it. Without this flag the builder only "
                         "writes sources, exactly as before.")
    args = ap.parse_args()

    st = SafeTensors(os.path.join(args.model, "model-00000-of-00001.safetensors"))
    with open(os.path.join(args.model, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)

    H = cfg["hidden_size"]
    NL = args.layers if args.layers else cfg["num_hidden_layers"]
    NH = cfg["num_attention_heads"]
    NKV = cfg["num_key_value_heads"]
    HD = cfg["head_dim"]
    INTER = cfg["intermediate_size"]
    V = cfg["vocab_size"]
    EPS = cfg["rms_norm_eps"]
    THETA = cfg["rope_theta"]
    S = args.seq
    rep = NH // NKV

    print("MiniCPM5-1B -> VMLM   S=%d layers=%d H=%d NH=%d NKV=%d HD=%d I=%d V=%d"
          % (S, NL, H, NH, NKV, HD, INTER, V))

    wdir = os.path.join(args.out, "weights")
    os.makedirs(wdir, exist_ok=True)

    consts = []   # (name, file, rows, cols, dtype)

    # The shape of each per-layer matrix, derived from the config rather than
    # read back from the safetensors, so the stride computation below cannot
    # drift from what emit_const() actually writes.
    SHAPES = {
        "in_ln": (1, H),
        "q": (NH * HD, H), "k": (NKV * HD, H), "v": (NKV * HD, H),
        "o": (H, NH * HD),
        "post_ln": (1, H),
        "gate": (INTER, H), "up": (INTER, H), "down": (H, INTER),
    }

    def layer_const_stride():
        """Bytes between layer L's constants and layer L+1's, as the ASSEMBLER
        will lay them out.

        WHY COMPUTE RATHER THAN HARDCODE: the stride is what OP_LAYERB adds per
        loop iteration, so a wrong value silently reads every layer's weights at
        a byte offset -- numbers that look plausible and are wrong. Deriving it
        from the same (rows, cols, dtype) the assembler will use, with the same
        64-byte padding rule, removes the chance of a stale hand-copied number.

        VALIDATED against a real artifact: for MiniCPM5-1B with --quant i8 this
        returns 28383232, which is EXACTLY the stride measured in the assembled
        .vmb (tools/p13_loop_check.py re-checks it at run time).
        """
        tot = 0
        for short, _suffix, dt in [(s[0], s[1], qtag(s[2])) for s in LAYER_SUFFIXES]:
            r, c = SHAPES[short]
            esz = 1 if dt == "i8" else (2 if dt == "bf16" else 4)
            need = r * c * esz
            tot += (need + 63) & ~63          # CONST_ALIGN = 64
            if dt == "i8":
                # the int8 path emits a companion <name>_sc scale constant
                tot += (r * 4 + 63) & ~63
        # the three per-layer scalars (eps1/theta1/scale1); see the note where
        # they are emitted for why they must be inside the block
        tot += 3 * 64
        return tot

    def emit_const(name, arr, dtype_tag):
        """Write one constant in the requested dtype and record its shape.

        bf16 is written WITHOUT widening -- that is the point (half the bytes,
        and the engine widens inside the GEMM). f32 is used for the few small
        constants that a non-GEMM op consumes (the engine rejects a bf16 const
        used by any op other than GEMM).
        """
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        r, c = arr.shape
        ext = {"bf16": "bf16", "i8": "i8"}.get(dtype_tag, "f32")
        p = os.path.join(wdir, name + "." + ext)
        if dtype_tag == "bf16":
            f32_to_bf16(arr).tofile(p)
        elif dtype_tag == "i8":
            # P13 task D: int8 + per-output-row scale.
            #
            # The scale is written as a SEPARATE f32 constant named <name>_sc,
            # because the engine takes it as a distinct operand (the 4th
            # register, compiled into `dt`). Keeping it separate rather than
            # packed means the dtype audit can check it independently, and the
            # assembler can verify the length matches N rows.
            q, s = int8_quant(arr)
            q.tofile(p)
            sp = os.path.join(wdir, name + "_sc.f32")
            np.ascontiguousarray(s, dtype=np.float32).tofile(sp)
            consts.append((name, os.path.join("weights", name + ".i8"), r, c, "i8"))
            consts.append((name + "_sc", os.path.join("weights", name + "_sc.f32"),
                           r, 1, "f32"))
            return r, c
        else:
            np.ascontiguousarray(arr, dtype=np.float32).tofile(p)
        consts.append((name, os.path.join("weights", name + "." + ext), r, c, dtype_tag))
        return r, c

    def emit_scalar(name, val):
        p = os.path.join(wdir, name + ".f32")
        np.array([val], dtype=np.float32).tofile(p)
        consts.append((name, os.path.join("weights", name + ".f32"), 1, 1, "f32"))

    # ---- non-layer constants ----
    #
    # ⚠️ SIZE DISCIPLINE (this was the 858 MB bug):
    # `embed_tokens` is [130560,1536] bf16 = 401 MB and `lm_head` is the same.
    # Together they are 802 MB of an 858 MB one-layer model -- i.e. 93% of the
    # file was tensors the program never reads:
    #   * embed  : the prompt rows are COPIED IN directly (EMB is not
    #              implemented in the engine), so the full embedding matrix is
    #              dead weight.
    #   * lm_head: only needed for the final logits, and not at all when we are
    #              debugging a layer.
    # Emitting only what the program uses is the difference between an 858 MB
    # debug artifact and a 57 MB one.
    embed = st.get("model.embed_tokens.weight")     # [V,H] f32
    tokens_mode = args.prompt_mode == "tokens"
    if args.with_embed or tokens_mode:
        # P13 task A: in tokens mode the embedding table is what makes the
        # prompt runtime-changeable, so it is REQUIRED here (and it stopped
        # being dead weight the moment OP_EMB learned to read bf16).
        # It stays bf16: 401 MB, versus 802 MB if widened to f32 -- the engine
        # widens one row on the fly per lookup.
        emit_const("embed", embed, "bf16")
    if not args.skip_head:
        # P13 task D: honour --quant for the head too. Emitting it here as bf16
        # and again below as i8 produced TWO constants of the same name; the
        # assembler resolved the first (bf16) while the GEMM attached the i8
        # scale, and it (correctly) refused. One emission site, one dtype.
        emit_const("lm_head", st.get("lm_head.weight"),
                   "i8" if args.quant == "i8" else "bf16")
    emit_const("final_norm", st.get("model.norm.weight"), "f32")
    emit_scalar("eps1", EPS)
    emit_scalar("theta1", THETA)
    emit_scalar("scale1", 1.0 / np.sqrt(HD))

    # ---- per-layer constants ----
    #
    # P13 task D: `--quant i8` switches the big matrices to int8 + per-row
    # scale, and the GEMMs then carry the scale slot as their 4th operand.
    # The norms stay f32 (a non-GEMM op may only read f32).
    def qtag(default):
        return "i8" if (args.quant == "i8" and default == "bf16") else default

    for L in range(NL):
        for short, suffix, dt in LAYER_SUFFIXES:
            nm = "model.layers.%d.%s" % (L, suffix)
            if not st.has(nm):
                print("missing tensor: %s" % nm)
                return 1
            emit_const("L%d_%s" % (L, short), st.get(nm), qtag(dt))
        if args.loop:
            # P13 task E: in LOOPED mode the scalars the layer body reads
            # (eps1/theta1/scale1) MUST live inside the per-layer block.
            #
            # WHY (this was a real bug, caught by tools/p13_loop_check.py):
            # OP_LAYERB shifts EVERY constant access by one base offset. The
            # loop body also reads eps1 (RMSNORM), theta1 and scale1 (RoPE and
            # the attention scale). With a single shared copy, iteration 1 onward
            # read those three at base+stride -- i.e. three unrelated bytes deep
            # in the weight pool. The values are still inside the const region,
            # so the bound check could not catch it, and the result was plausible
            # wrong numbers (max|d| 35 on values of ~76).
            #
            # Emitting them per layer makes the loop body read ONLY per-layer
            # constants, so one base is sufficient and correct. The cost is
            # 24 x 3 x 4 = 288 bytes.
            emit_scalar("L%d_eps1" % L, EPS)
            emit_scalar("L%d_theta1" % L, THETA)
            emit_scalar("L%d_scale1" % L, 1.0 / np.sqrt(HD))

    # NOTE: lm_head is emitted ONCE, above, with the dtype --quant selects.
    # (An earlier version emitted it again here for the i8 case, which produced
    # duplicate constant names -- the assembler took the bf16 one while the GEMM
    # carried an i8 scale, and the mismatch was caught only by the new guard.)

    print("constants: %d  (quant=%s)" % (len(consts), args.quant))

    # Which constants actually ended up int8? Derive it from the emitted list
    # rather than re-deriving from the flags: the first version of this helper
    # guessed "lm_head is quantised when --quant i8", but that build had
    # --skip-head, so no lm_head existed and the assembler (correctly) refused a
    # GEMM with a scale operand attached to a bf16 weight. Asking the emitted
    # table removes the whole class of "builder and flags disagree" bugs.
    q8_names = set(n for (n, _p, _r, _c, dt) in consts if dt == "i8")

    def gemm(dst, a, wname):
        """Emit a projection GEMM, attaching the int8 scale when quantised.

        The scale rides as a 4th operand; the assembler REJECTS a 4th operand
        on a non-int8 weight, so this stays honest in both directions rather
        than silently emitting a meaningless operand.
        """
        if wname in q8_names:
            w("GEMM %s, %s, %s, %s_sc" % (dst, a, wname, wname))
        else:
            w("GEMM %s, %s, %s" % (dst, a, wname))

    # ------------------------------------------------------------- .vml
    A = []
    def w(s):
        A.append(s)

    w("; MiniCPM5-1B full forward (generated by tools/p12_build_minicpm.py)")
    w("; S=%d layers=%d hidden=%d heads=%d kv_heads=%d head_dim=%d inter=%d vocab=%d"
      % (S, NL, H, NH, NKV, HD, INTER, V))
    w(";")
    w("; Weights are bf16 and stay bf16: the engine widens them inside the GEMM")
    w("; (P12). Widening to f32 on disk would double 2.16 GB to 4.32 GB.")
    w(";")
    w("; Shapes are declared explicitly because the assembler infers only from")
    w("; instruction use, and a wrong width here is a silent misread.")
    w("")
    w("; ---- shapes (activations) ----")
    w("SHAPE h,      %d, %d" % (S, H))
    w("SHAPE h2,     %d, %d" % (S, H))
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
    w("SHAPE gate,   %d, %d" % (S, INTER))
    w("SHAPE up,     %d, %d" % (S, INTER))
    w("SHAPE acts,   %d, %d" % (S, INTER))
    w("SHAPE down,   %d, %d" % (S, H))
    if not args.skip_head:
        w("SHAPE logits, %d, %d" % (S, V))
    else:
        # Debug mode does not build logits, so do not allocate the [S,130560]
        # activation (2 MB) either -- keep the debug artifact proportionate.
        w("SHAPE tail,   %d, 8" % S)
    w("SHAPE eps1,   1, 1")
    w("SHAPE theta1, 1, 1")
    w("SHAPE scale1, 1, 1")
    w("")
    ids = [3, 117, 2501, 88][:S]
    if tokens_mode:
        w("; ---- embedding lookup: h = embed[tokens]  (P13 task A) ----")
        w("; THE POINT OF THIS MODE: the program carries NO prompt data.")
        w("; Token ids are read from `tokens.bin` by the runtime into the")
        w("; `st_tok` state slot, then EMB widens one row per token.")
        w("; ONE .vmb therefore serves ANY prompt -- changing the prompt is a")
        w("; 16-byte file write instead of a 1.76 GB reassembly (~106 s).")
        w(";")
        w("; `st_tok` is a state slot (the `st_` prefix); state is engine-owned")
        w("; and persists, and the runtime fills it before run().")
        w(";")
        w("; Its shape MUST be declared: an undeclared slot has capacity 0, and")
        w("; the runtime then hard-fails with 'tokens.bin is longer than the")
        w("; st_tok slot' (measured -- that guard is what caught this). [1,S]")
        w("; because the runtime fills it with S token ids, one per element.")
        w("SHAPE st_tok, 1, %d" % S)
        w(";")
        w("; EMB reads its row index from a SINGLE scalar slot, so one lookup is")
        w("; needed per token position. Rows are then stacked with CONCAT, which")
        w("; concatenates along COLUMNS -- so we transpose the problem: look up")
        w("; the token's row as [1,H], and build h by concatenating the S rows")
        w("; side by side into [1, S*H], then RESHAPE-equivalent via SHAPE.")
        w("; Simpler and exact: build a [S,H] h by concatenating single-row")
        w("; slices, so the intermediate is [1,H] and CONCAT gives [1,S*H].")
        w("; That layout is (token-major), i.e. element (i, j) sits at i*H + j,")
        w("; which IS row-major [S,H]. So h = the concatenated buffer.")
        for i in range(S):
            w(";   row %d = embed[tok[%d]]" % (i, i))
            w("SLICE tok%d, st_tok, %d, 1" % (i, i))
            # SHAPE e%d is REQUIRED, not decorative. Measured: without it the
            # assembler leaves e0..e3 at 0x0, so EMB writes ZERO elements and
            # the rows silently never land -- the layer stack then ran on stale
            # activation memory. In the small emb_only.vml harness the shape
            # happened to be inferred from use, which is exactly why this hid.
            w("SHAPE e%d, 1, %d" % (i, H))
            w("EMB e%d, embed, tok%d" % (i, i))
        if S == 1:
            w("SHAPE hcat, 1, %d" % H)
            w("COPY hcat, e0")
            w("COPY h, hcat")
        else:
            # CONCAT dst, a, b reads BOTH a and b, so the accumulator must be
            # fed forward each step. The first version wrote
            #   CONCAT hcat1, hcat, e1
            #   CONCAT hcat2, hcat, e2     <- re-reads hcat, so e1 is LOST
            # and the result differed from const-mode by max|d|=41 (measured).
            # Chaining prev->next fixes it; the micro-test in
            # examples/p13/stack_dbg.vml is what pinned this down.
            w("SHAPE hcat0, 1, %d" % H)
            w("COPY hcat0, e0")
            prev = "hcat0"
            total = H
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
    else:
        w("; ---- embedding lookup: h = embed[token] ----")
        w("; PRE-P13 PATH (kept for comparison / regression): the prompt rows are")
        w("; extracted HOST-SIDE and baked in as a constant, so every new prompt")
        w("; requires reassembling this whole .vmb.")
        prompt_rows = np.zeros((S, H), dtype=np.float32)
        for i, t in enumerate(ids):
            prompt_rows[i] = embed[t]
        emit_const("prompt_emb", prompt_rows, "f32")
        w("COPY h, prompt_emb")
        w("")

    # ------------------------------------------------------------------
    # P13 task C: incremental decode against a K/V cache.
    #
    # WHY THE CACHE, AND WHAT THE MEASUREMENT SAID
    #   tools/p13_bench_attn.py measured, with the engine's own in-process QPC
    #   timer (subprocess wall clock has a 0.27 s spread here, which is bigger
    #   than the effect being measured), the share of a forward spent on the
    #   S^2 attention:
    #       S=16 -> 1.2%   S=64 -> 1.2%   S=256 -> 8.9%   S=1024 -> 27.4%
    #   So "attention is O(S^2) in prefill" is true but IRRELEVANT at the sizes
    #   that matter here: the MLP projections are ~74% of the MACs. A cache
    #   that only shortened prefill would buy almost nothing.
    #
    #   The real cost is that generation re-runs the WHOLE sequence per token.
    #   Measured baseline (4 layers, prompt 64, generate 16): 14.63 s, because
    #   forward(64)..forward(79) are each a full pass. A decode step, by
    #   contrast, is one token: measured forward(S=1) = 19.5 ms.
    #
    # CACHE LAYOUT -- TRANSPOSED, AND THAT IS THE WHOLE TRICK
    #   No new engine opcode is used. Storing the cache as [W, PMAX] (feature
    #   major) makes "append position pos" a COLUMN write:
    #       kcacheT[i][pos] = knewT[i]      for i in 0..W
    #   and WRITECOL(dst, src, off) is exactly `dst[i][off+j] = src[i][j]`, so
    #       WRITECOL kcacheT, knewT, pos
    #   with knewT shaped [W,1]. Verified against interp.rs OP_WRITECOL, and its
    #   `off + sc > dc` guard doubles as the capacity check: decoding past PMAX
    #   is a loud exit(9), not a silent overwrite of the last column.
    #
    #   Attention then reads the cache with GEMM forms the engine ALREADY has:
    #     scores:  sc[1,P]  = qrot[1,hd] x kcacheT[hd,P]   -> GEMM ..., nn
    #              (NN: C[0][n] = sum_k A[0][k]B[k][n]; B=[K,N]=[hd,P])
    #     context: ctx[1,hd] = sc[1,P]  x vcacheT[hd,P]    -> default NT GEMM
    #              (NT: C[0][j] = sum_k A[0][k]B[j][k]; B=[N,K]=[hd,P])
    #   So the SAME transposed [W,PMAX] layout serves both operands, and the two
    #   GEMM flavours needed are exactly the two the program already uses.
    #
    # WHY f32 AND NOT bf16: the cache is engine-owned STATE, and state slots are
    # always f32 (see Slot.dtype in bind_slots). Making it bf16 would mean adding
    # dtype support to TRANSPOSE / WRITECOL / GEMM-NN, and the measured error
    # budget does not need it -- the weights are already bf16, so the cache is
    # not the precision-limiting term.
    # ------------------------------------------------------------------
    PMAX = S + args.decode

    # `n_seq` is the number of POSITIONS the KV cache holds for this build: the
    # S prompt positions plus, for a decode build, the G decoded ones. It is the
    # flow-precision quantity behind max_seq_per_neuron (PMAX is the CAPACITY);
    # n_seq <= PMAX always, so the section's tight stride=PMAX satisfies the
    # loader's `stride >= max_seq_per_neuron`.
    #
    # WHY NOT n_seq * HD: a "neuron" is ONE head_dim channel, so its slots are
    # (position, dim) for that single channel -- there are n_seq of them. The
    # head_dim factor is already carried by the neuron INDEX ((L*NKH+h)*HD+d).
    # Multiplying it in again here is precisely the 128x defect.
    n_seq = S + max(args.decode, 0)

    # ---- S5-D2: the neuron_kv geometry, DERIVED from THIS artifact ----
    #
    # Every input is the cache geometry this run actually declares: NL layers,
    # NKV kv heads, HD head_dim, and the state slots' PMAX. Nothing here is
    # copied from a test -- a hand-carried geometry is exactly how the 128x
    # stride defect survived (neuron_kv.rs:118-132).
    #
    # n_layers must be the DECLARED layer count, because load_validate compares
    # the section against the artifact header field by field; a reduced --layers
    # build is a smaller artifact and must describe itself as such.
    #
    # `n_seq` is what the cache actually needs to hold: the prompt plus the
    # decoded positions for a decode build, just the prompt otherwise. It is a
    # FLOW-precision quantity (separate from PMAX, the CAPACITY), and it is
    # always <= PMAX, so `stride=PMAX` satisfies the loader's invariant
    # `stride >= max_seq_per_neuron`. Keeping the two distinct is deliberate:
    # a neuron's slots are (position, dim) for exactly one head_dim channel, so
    # the count is n_seq and NOT n_seq*HD.
    if args.emit_neuron_kv:
        # Fail BEFORE spending minutes writing 2.16 GB of weights if the
        # writer's constants have drifted from core/src/format.rs.
        try:
            cross_check_against_rust()
            neuron_kv_geom = derive_geometry(NL, NKV, HD, PMAX,
                                             args.neuron_kv_stride)
        except GeometryError as e:
            print("ERROR: refusing to build -- the neuron_kv geometry is not "
                  "legal: %s" % e)
            return 1
    else:
        neuron_kv_geom = None

    if args.decode:
        w("; ---- P13 task C: KV cache in STATE slots (transposed [HD, PMAX]) ----")
        w("; STATE, not activation: state persists and is engine-owned. The")
        w("; transposed layout makes appending position p a COLUMN write, so")
        w("; TRANSPOSE + WRITECOL suffice -- no new engine opcode.")
        w(";")
        w("; ONE CACHE PER KV HEAD, not one per layer. Why (this was a real bug):")
        w("; a single [NKV*HD, PMAX] cache stores kv head j in ROWS j*HD..j*HD+HD.")
        w("; Selecting those rows needs a ROW slice, but OP_SLICE indexes COLUMNS")
        w("; (dst[i][j] = src[i][off+j]), so head j was unreachable -- every query")
        w("; head read kv head 0. Measured symptom: prefill matched the reference")
        w("; EXACTLY (0.0) while every decode step was off by ~80 on values of")
        w("; magnitude ~80, i.e. structurally wrong rather than drifting.")
        w("; Per-head caches remove the row arithmetic entirely: head j reads")
        w("; st_kc{j} directly, and the read slice is a plain COLUMN range.")
        for L in range(NL):
            for h_j in range(NKV):
                w("SHAPE st_kc%d_%d, %d, %d" % (L, h_j, HD, PMAX))
                w("SHAPE st_vc%d_%d, %d, %d" % (L, h_j, HD, PMAX))
        # `st_gen` holds the token ids to decode (one per step). The runtime
        # fills it from gen_tokens.bin, exactly like st_tok for the prompt.
        w("SHAPE st_gen, 1, %d" % max(args.decode, 1))
        # Per-head working buffers for the cached attention. Sizes are worst
        # case (the full live range) so one declaration covers every step.
        w("SHAPE kdT, %d, 1" % HD)
        w("SHAPE vdT, %d, 1" % HD)
        # NOTE: kcv/vcv/scd/ctxd are NOT declared here. They are declared per
        # decode step with unique names (kcv0, kcv1, ...), because the assembler
        # unifies a shared slot's shape to the widest definition -- see the note
        # inside the decode loop.
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
        # The cache append works PER KV HEAD: a [1,HD] row slice of the prefill
        # k/v, transposed to [HD,1] for the column write.
        w("SHAPE krow, 1, %d" % HD)
        w("SHAPE vrow, 1, %d" % HD)
        w("SHAPE knewT, %d, 1" % HD)
        w("SHAPE vnewT, %d, 1" % HD)
        # Per-kv-head prefill buffers: raw key for this head, its rotated form
        # (what actually goes into the cache), and the raw V.
        w("SHAPE kcraw, %d, %d" % (S, HD))
        w("SHAPE kcrot, %d, %d" % (S, HD))
        w("SHAPE vcraw, %d, %d" % (S, HD))
        w("SHAPE kcrotT, %d, %d" % (HD, S))
        w("SHAPE vcrawT, %d, %d" % (HD, S))
        w("SHAPE kcol, %d, 1" % HD)
        w("SHAPE vcol, %d, 1" % HD)
        w("SHAPE gtok, 1, 1")
        w("")

    # ------------------------------------------------------------------
    # P13 task E: LOOPED layers.
    #
    # The unrolled build emits NL copies of the layer body (~4180 instructions
    # for 24 layers). This mode emits the body ONCE and runs it NL times with
    # LOOP_CNT, advancing a RUNTIME constant base between iterations.
    #
    # WHY A RUNTIME BASE IS REQUIRED (and why P10 could not do this):
    #   slot -> const is bound ONCE at load, so a loop re-executing the same
    #   instruction reads the SAME layer's weights every iteration. OP_LAYERB
    #   (added by this task) shifts every constant access by a base offset.
    #
    # WHY ONE BASE SUFFICES (measured, not assumed):
    #   the artifact's 16 per-layer constants were measured to have a perfectly
    #   UNIFORM stride -- 28,383,232 bytes between layer L and L+1 for all 16
    #   (in_ln/q/q_sc/k/k_sc/v/v_sc/o/o_sc/post_ln/gate/gate_sc/up/up_sc/down/
    #   down_sc). So advancing one base moves the whole layer at once.
    #   The stride is stored as an f32 constant; it is EXACTLY representable
    #   (it has 11 trailing zero bits, so it fits in 24 significant bits) --
    #   checked, because a rounded stride would drift by bytes per layer and
    #   silently read misaligned weights.
    #
    # STRUCTURE: the body refers to layer 0's constant NAMES for every layer.
    # Correctness therefore rests on every layer having an identical constant
    # layout, which the uniformity measurement above establishes and
    # tools/p13_loop_check.py verifies by comparing against the unrolled build.
    # ------------------------------------------------------------------
    looped = args.loop
    if looped:
        layer_stride = layer_const_stride()
        print("looped layers: per-layer constant stride = %d bytes" % layer_stride)
        if layer_stride % 64 != 0:
            print("ERROR: stride is not 64-byte aligned; the base would land on "
                  "a padding byte")
            return 1
        w("; ================ P13 task E: LOOPED layer template ================")
        w("; ONE copy of the layer body, run %d times." % NL)
        w("; The weight names below are layer 0's; OP_LAYERB shifts every")
        w("; constant access to the NEXT layer's copy each iteration.")
        w("SHAPE lcnt, 1, 1")
        w("; stride = measured per-layer constant stride, in BYTES")
        w("SHAPE lstride, 1, 1")
        w("SHAPE lneg, 1, 1")
        w("SHAPE lnl, 1, 1")
        emit_scalar("layer_stride", float(layer_stride))
        emit_scalar("layer_count", float(NL))
        emit_scalar("neg_layer_count", float(-NL))
        # ALL loop-control inputs are copied into ACTIVATIONS before the loop.
        #
        # WHY THIS IS MANDATORY, not tidiness: OP_LAYERB shifts every CONSTANT
        # access. The post-loop restore needs -NL*stride, and if it read
        # `neg_layer_count` (a const) while the base was still NL*stride, it
        # would fetch whatever lies at neg_layer_count + NL*stride -- the base
        # corrupting its own correction. Measured symptom: the restore silently
        # subtracted the wrong amount and the next const read tripped the bound
        # check reporting base=113271168 (= 2*stride) instead of 0.
        #
        # const_ptr applies the base only to slots with is_const, so reading
        # these from ACTIVATION slots is immune by construction -- and it needs
        # no new opcode.
        w("COPY lcnt, layer_count")
        w("COPY lstride, layer_stride")
        w("COPY lnl, neg_layer_count")
        w("LOOP_TOP:")
    for L in range(NL):
        if looped:
            # Only one pass; every weight name refers to layer 0.
            if L > 0:
                break
            w("; ---- layer template (used for all %d layers) ----" % NL)
            lname = "L0"
            # In LOOPED mode the scalars are per-layer too (see the emission
            # site for why), so the body must reference layer 0's copies.
            s_eps, s_theta, s_scale = "L0_eps1", "L0_theta1", "L0_scale1"
        else:
            w("; ================= layer %d =================" % L)
            lname = "L%d" % L
            s_eps, s_theta, s_scale = "eps1", "theta1", "scale1"
        w("RMSNORM xn, h, %s_in_ln, %s" % (lname, s_eps))
        gemm("q", "xn", "%s_q" % lname)
        gemm("k", "xn", "%s_k" % lname)
        gemm("v", "xn", "%s_v" % lname)
        for h_i in range(NH):
            j = h_i // rep
            w("SLICE qh, q, %d, %d" % (h_i * HD, HD))
            w("SLICE kh, k, %d, %d" % (j * HD, HD))
            w("SLICE vh, v, %d, %d" % (j * HD, HD))
            w("ROPE qrot, qh, %s, 0" % s_theta)
            w("ROPE krot, kh, %s, 0" % s_theta)
            w("GEMM sc, qrot, krot")
            w("SCALE sc, sc, %s" % s_scale)
            w("SOFTMAX sc, sc, causal")
            w("GEMM ctx, sc, vh, nn")
            w("WRITECOL att, ctx, %d, %d" % (h_i * HD, HD))
        gemm("oproj", "att", "%s_o" % lname)
        w("EWADD h2, h, oproj")
        w("RMSNORM xn, h2, %s_post_ln, %s" % (lname, s_eps))
        gemm("gate", "xn", "%s_gate" % lname)
        gemm("up", "xn", "%s_up" % lname)
        w("ACT acts, gate, silu")
        w("EWMUL acts, acts, up")
        gemm("down", "acts", "%s_down" % lname)
        w("EWADD h, h2, down")
        if args.decode:
            # ---- cache ROTATED K and raw V, per KV head ----
            #
            # CRITICAL (this was a real bug worth ~8% error on every decode step):
            # the cache must hold the ROTATED key, not the raw one. Attention is
            # `softmax(q_rot . k_rot)`, so caching raw `k` and then comparing it
            # against a rotated decode query mixes two different bases. The
            # symptom was a UNIFORM ~8% relative error on every step (measured
            # 6.9 on values of magnitude 80) -- uniform because the mistake
            # affects every score equally, which is exactly why it did not look
            # structural and needed the layout reasoning to pin down.
            #
            # `krot` from the per-head loop above IS the rotated key for kv head
            # j, but it is overwritten per query head. So re-derive it here by
            # rotating each kv head's key at its own absolute positions:
            # ROPE's `dt` is a compile-time base and it processes r rows at
            # positions dt..dt+r-1, which is exactly the prefill's job.
            w(";   cache K (ROTATED) and V for the %d prompt positions, layer %d"
              % (S, L))
            for h_j in range(NKV):
                # Rotate this kv head's keys at their absolute positions 0..S-1.
                w("SLICE kcraw, k, %d, %d" % (h_j * HD, HD))
                w("ROPE kcrot, kcraw, theta1, 0")
                w("SLICE vcraw, v, %d, %d" % (h_j * HD, HD))
                # TRANSPOSE FIRST, then select positions as COLUMNS.
                #
                # This is the second time the same limitation bit in this file,
                # so it is worth stating plainly: **OP_SLICE offsets COLUMNS, it
                # cannot select a ROW** (dst[i][j] = src[i][off+j]). The first
                # attempt sliced rows of kcrot ([S,HD]) with `SLICE krow, kcrot,
                # 0, HD`, which always returns row 0 -- so ALL S positions were
                # cached as position 0's key. Measured: the decode output was
                # byte-identical whether the cache held raw or rotated keys,
                # which is what exposed that the cache content was not varying
                # with position at all.
                #
                # After TRANSPOSE the matrix is [HD, S], so position p IS
                # column p and a column slice is exactly right. It also collapses
                # the work: the [HD,1] column IS the operand WRITECOL wants, so
                # the per-position TRANSPOSE disappears.
                w("TRANSPOSE kcrotT, kcrot")
                w("TRANSPOSE vcrawT, vcraw")
                for p in range(S):
                    w("SLICE kcol, kcrotT, %d, 1" % p)
                    w("WRITECOL st_kc%d_%d, kcol, %d" % (L, h_j, p))
                    w("SLICE vcol, vcrawT, %d, 1" % p)
                    w("WRITECOL st_vc%d_%d, vcol, %d" % (L, h_j, p))
        if looped:
            # Advance the base to the NEXT layer and loop back. LAYERB is
            # emitted at the END of the body so iteration 0 runs with base 0
            # (layer 0). LOOP_CNT decrements the counter and jumps back while it
            # is still > 0, so the LAST iteration also pays the (harmless)
            # increment -- the base is not read again before HALT.
            w("; advance to the next layer's weights, then loop")
            w("LAYERB lstride")
            # Syntax is `LOOP_CNT <counter_slot>, @<label>` -- counter FIRST.
            # (Getting this backwards is a hard assembler error, not a silent
            # jump-to-0, which is why p10_loop_verify pins that guard.)
            w("LOOP_CNT lcnt, @LOOP_TOP")
            # RESTORE the base so non-per-layer constants (final_norm, embed,
            # the output path) are read at their real offsets.
            #
            # The base after the loop is NL*stride, NOT stride: LAYERB runs at
            # the END of the body, so it fires once per iteration including the
            # last. Subtracting ONE stride leaves the base at (NL-1)*stride and
            # every subsequent non-per-layer read is shifted -- measured: it
            # walked past the constant pool and the bound check rejected it
            # (exit 9), which is exactly why that check exists.
            #
            # Multiplying the stride by -NL is done with an EWMUL against a
            # single scalar constant, so no loop or extra state is needed.
            w("EWMUL lneg, lstride, lnl")
            w("LAYERB lneg")
        w("")

    def emit_decode_steps():
        # ==================================================================
        # P13 task C: incremental decode.
        #
        # Each step processes ONE token against the cache built above and then
        # appends that token's own K/V. Unlike the baseline, the cost does NOT
        # grow with the prompt: the measured baseline (4 layers, P=64, G=16) was
        # 14.63 s precisely because forward(64)..forward(79) were each a FULL
        # pass over the whole sequence.
        #
        # WHY UNROLLED RATHER THAN LOOP_CNT (deliberate, correctness-first):
        #   OP_ROPE takes its base position from `ins.dt`, a COMPILE-TIME u16.
        #   A decode step for the k-th generated token must rotate at the
        #   token's ABSOLUTE position S+k. A runtime loop would rotate every
        #   token at the same position -- wrong for all but the first, and the
        #   kind of wrong that looks like a plausible numeric drift rather than
        #   an error. Using a runtime position would mean a new engine opcode;
        #   unrolling instead makes the position a literal, i.e. correct BY
        #   CONSTRUCTION with no engine change and no new failure mode.
        #   The cost is TEXT only: G=256 is ~590K instructions ~ 9.4 MB, against
        #   2.1 GB of weights. That is the right trade for the correct option.
        #
        # The cache column for step k is S+k; every cache slice below is
        # therefore [0, S+k+1), which is exactly the causal live range.
        # ==================================================================
        w("; ================ P13 task C: incremental decode ================")
        w("; %d steps after a %d-token prefill; each step is one token." % (args.decode, S))
        for step in range(args.decode):
            pos = S + step  # absolute position of the token being decoded
            live = pos + 1  # including itself
            w("")
            w("; ---- decode step %d (absolute position %d) ----" % (step, pos))
            # Working buffers whose extent must equal THIS step's live range.
            #
            # UNIQUE NAMES PER STEP, and that is load-bearing. A GEMM takes its
            # extent from the destination slot's declared shape, and the
            # assembler UNIFIES a slot's shape across every definition, keeping
            # the WIDEST. So sharing `kcv` across steps and declaring it per step
            # (`SHAPE kcv, 128, 5` ... `128, 6` ...) silently collapses back to
            # the largest, PMAX -- measured: the slot table showed `kcv [act]
            # 128x9` for S=4,G=3 even though the per-step declarations said
            # 5,6,7. Every step then attended PMAX columns, including the ones
            # not yet written, whose zeros act as REAL softmax mass.
            #
            # Measured signature of exactly that: G=1 (live == PMAX) was exact,
            # while G>=2 was wrong on every step except the last (whose live ==
            # PMAX too).
            #
            # Unrolling is what makes unique names possible; a LOOP_CNT loop
            # could not express this.
            w("SHAPE kcv%d, %d, %d" % (step, HD, live))
            w("SHAPE vcv%d, %d, %d" % (step, HD, live))
            w("SHAPE scd%d, 1, %d" % (step, live))
            w("SHAPE ctxd%d, 1, %d" % (step, HD))
            w("; h = embed[token]  (one row)")
            w("SLICE gtok, st_gen, %d, 1" % step)
            w("EMB eh, embed, gtok")
            w("COPY hdec, eh")
            w("")

            for L in range(NL):
                w(";   layer %d" % L)
                w("RMSNORM xnd, hdec, L%d_in_ln, eps1" % L)
                gemm("qd", "xnd", "L%d_q" % L)
                gemm("kd", "xnd", "L%d_k" % L)
                gemm("vd", "xnd", "L%d_v" % L)
                # Append THIS token's K/V to the cache ONCE PER KV HEAD, before
                # the attention loop. Doing it inside the query-head loop would
                # repeat the write NKV/rep times and, worse, each query head
                # would append under its own rotation of `kdh` -- but the
                # rotation is over the kv head's own features, so the repeated
                # writes happen to be identical and merely wasteful. Keeping it
                # outside makes the single-writer property explicit rather than
                # accidental.
                for h_j in range(NKV):
                    w("SLICE kdh, kd, %d, %d" % (h_j * HD, HD))
                    w("SLICE vdh, vd, %d, %d" % (h_j * HD, HD))
                    # Rotate at THIS token's absolute position; `dt` is a
                    # compile-time literal (see the unrolling note above).
                    w("ROPE kdrot, kdh, theta1, %d" % pos)
                    w("TRANSPOSE kdT, kdrot")
                    w("TRANSPOSE vdT, vdh")
                    w("WRITECOL st_kc%d_%d, kdT, %d" % (L, h_j, pos))
                    w("WRITECOL st_vc%d_%d, vdT, %d" % (L, h_j, pos))
                for h_i in range(NH):
                    j = h_i // rep
                    w("SLICE qdh, qd, %d, %d" % (h_i * HD, HD))
                    w("ROPE qdrot, qdh, theta1, %d" % pos)
                    # Attend over the live range [0, live) of THIS kv head.
                    # The buffer names carry the step index so their declared
                    # width cannot be unified back to PMAX -- see the SHAPE note
                    # at the top of this step.
                    w("SLICE kcv%d, st_kc%d_%d, 0, %d" % (step, L, j, live))
                    w("SLICE vcv%d, st_vc%d_%d, 0, %d" % (step, L, j, live))
                    w("GEMM scd%d, qdrot, kcv%d, nn" % (step, step))
                    w("SCALE scd%d, scd%d, scale1" % (step, step))
                    # Non-causal softmax: this single query may attend to every
                    # live position (the causal constraint is already encoded by
                    # the live range above).
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

            w(";   final norm for this step")
            w("RMSNORM xnd, hdec, final_norm, eps1")
            if not args.skip_head:
                w("GEMM logitsd, xnd, lm_head")
                w("PRINT logitsd")
            else:
                # Debug: print the decoded hidden state so the check tool can
                # compare it against a full re-prefill at the same length.
                w("SHAPE dtail, 1, %d" % H)
                w("COPY dtail, xnd")
                w("PRINT dtail")
        w("")

    w("; ---- final norm ----")
    w("RMSNORM xn, h, final_norm, eps1")
    if not args.skip_head:
        gemm("logits", "xn", "lm_head")
        w("PRINT logits")
    elif args.print_hidden:
        # Full post-norm hidden state: this is what the HuggingFace reference
        # (`output_hidden_states[-1]`) gives, so it is the comparable quantity.
        #
        # ORDERING NOTE (this was a real bug): in decode builds the prefill's
        # PRINT must come BEFORE the decode steps, because stdout is a single
        # stream -- with the decode block emitted first, the prefill's [S,H]
        # block landed AFTER the G decoded rows and the check tool's row
        # alignment was silently wrong (it read 10752 values where 4608 were
        # expected).
        w("PRINT xn")
    else:
        # Small debug output. NOTE (measured): the engine writing 6 KB to a
        # FILE takes ~2 s; the same output drained through a PowerShell pipe
        # looked like ~20 minutes. Timing must always use a real file.
        w("SLICE tail, xn, 0, 8")
        w("PRINT tail")
    w("")

    if args.decode:
        emit_decode_steps()
        w("")

    w("HALT")

    # ------------------------------------------------------------------
    # P13 task E guard: no CONSTANT may be read between the first LAYERB and
    # the last, except inside the layer template.
    #
    # WHY: OP_LAYERB shifts every constant access, so any constant read while a
    # non-zero base is active must be a PER-LAYER constant (whose +base lands on
    # the next layer's copy). A non-per-layer constant read there silently gets
    # unrelated bytes -- the base corrupts the very control values meant to undo
    # it. That bug cost several cycles (see the COPY lnl note above), and its
    # symptom was a bound-check failure reporting a base that was 2*stride
    # instead of 0.
    #
    # This scans the emitted program and refuses to build if it sees a
    # non-per-layer constant read after the loop has advanced the base.
    if looped:
        import re as _re
        const_names = set(n for (n, _p, _r, _c, _d) in consts)
        per_layer = set(n for n in const_names if _re.match(r"^L\d+_", n))
        txt = "\n".join(A)
        after = txt.split("LOOP_TOP:", 1)[1] if "LOOP_TOP:" in txt else ""
        before = txt.split("LOOP_TOP:", 1)[0]
        # Constants copied into ACTIVATIONS before the loop are the sanctioned
        # way to use a non-per-layer value inside the loop.
        safe_copies = set()
        for m in _re.finditer(r"COPY\s+(\w+),\s+(\w+)", before):
            if m.group(2) in const_names:
                safe_copies.add(m.group(1))
        # The base is NON-ZERO only between the first LAYERB (the loop's advance)
        # and the restore (the final LAYERB, which adds the negative stride).
        # Outside that window the base is 0 and any constant read is fine --
        # which is exactly what the restore exists for, so the final norm and the
        # output path must NOT be flagged.
        lines = after.split("\n")
        layerb = [i for i, l in enumerate(lines) if l.strip().startswith("LAYERB")]
        if not layerb:
            print("ERROR: looped build emitted no LAYERB")
            return 1
        base_restored_from = layerb[-1] + 1
        offenders = []
        for i, raw in enumerate(lines):
            if i >= base_restored_from:
                break
            line = raw.split(";")[0]
            if not line.strip():
                continue
            for name in const_names:
                if name in per_layer or name in safe_copies:
                    continue
                if _re.search(r"\b%s\b" % _re.escape(name), line):
                    offenders.append((name, line.strip()))
        if offenders:
            print("ERROR: these non-per-layer constants are read while the "
                  "layer base is NON-ZERO, so they would be silently shifted:")
            for n, l in offenders[:8]:
                print("   %s  <- %s" % (n, l))
            print("   fixes: COPY them into an activation BEFORE the loop, or")
            print("          move the read after the base restore.")
            return 1

    vml_path = os.path.join(args.out, "full.vml")
    os.makedirs(args.out, exist_ok=True)
    with open(vml_path, "w", encoding="ascii", newline="\n") as f:
        f.write("\n".join(A) + "\n")
    print("wrote %s (%d lines)" % (vml_path, len(A)))

    # ---- the assembler argument list (mirrors asm_all.ps1 conventions) ----
    argp = os.path.join(args.out, "weights.args")
    with open(argp, "w", encoding="ascii", newline="\n") as f:
        for name, path, r, c, dt in consts:
            f.write("%s:%s:%d:%d:%s\n" % (name, path.replace("\\", "/"), r, c, dt))
    print("wrote %s (%d constants)" % (argp, len(consts)))

    meta = {
        "seq": S, "layers": NL, "hidden": H, "heads": NH, "kv_heads": NKV,
        "head_dim": HD, "inter": INTER, "vocab": V, "eps": EPS, "theta": THETA,
        "input_ids": ids, "skip_head": bool(args.skip_head),
    }
    # Record the geometry the section was (or would be) built from, so a reader
    # never has to re-derive it and cannot disagree with what was written.
    if neuron_kv_geom is not None:
        meta["neuron_kv"] = {
            "n_layers": neuron_kv_geom["n_layers"],
            "n_kv_heads": neuron_kv_geom["n_kv_heads"],
            "head_dim": neuron_kv_geom["head_dim"],
            "max_seq_per_neuron": neuron_kv_geom["max_seq_per_neuron"],
            "stride": neuron_kv_geom["stride"],
            "n_neurons": neuron_kv_geom["n_neurons"],
            "pmax": PMAX, "n_seq": n_seq,
            "section": "--assemble writes it into the .vmb",
        }
        print("neuron_kv geometry (from THIS artifact): n_layers=%d n_kv_heads=%d "
              "head_dim=%d PMAX=%d n_seq=%d -> stride=%d n_neurons=%d"
              % (NL, NKV, HD, PMAX, n_seq, neuron_kv_geom["stride"],
                 neuron_kv_geom["n_neurons"]))
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # ---- S5-D2: assemble, then install the neuron_kv section ----
    #
    # The section cannot come from the assembler alone: the container needs a
    # 12-byte descriptor at file offset 32 that asm/src/main.rs does not emit.
    # So the pipeline assembles EXACTLY as before (byte-identical .vmb) and then
    # re-frames the header (text_off 32 -> 44, the text and weights shift by 12
    # and keep their bytes) and adds the section. Nothing the default engine
    # reads changes meaning.
    if args.assemble and neuron_kv_geom is not None:
        asm = os.path.join(ROOT_ABS, "target", "release", "vmlm-asm.exe")
        if not os.path.exists(asm):
            print("ERROR: --assemble needs %s (build it first)" % asm)
            return 1
        vmb = os.path.join(args.out, "m.vmb")
        # Absolute paths and cwd=args.out, matching how every p13_*/p14_* checker
        # invokes the assembler (`cwd=out`, bare `full.vml`). Passing the
        # project-relative vml_path together with cwd=args.out resolves it
        # against the WRONG directory -- measured: the assembler then exits with
        # 0xC0000409 (a missing-input crash, not a diagnostic).
        cmd = [asm, os.path.abspath(vml_path), os.path.abspath(vmb)] + [
            "%s:%s:%d:%d:%s" % (n, p, r, c, d) for (n, p, r, c, d) in consts]
        t0 = time.time()
        # The assembler dumps every slot; text mode with a GB-scale const list is
        # wasted work. Capture as bytes and decode defensively for the message.
        q = subprocess.run(cmd, cwd=args.out, capture_output=True)
        if q.returncode != 0:
            print("ASSEMBLE FAILED (%d)" % q.returncode)
            tail = (q.stdout + q.stderr)[-2000:]
            print(tail.decode("utf-8", "replace"))
            return 1
        print("assembled %s in %.1fs (%d B)" % (vmb, time.time() - t0,
                                                os.path.getsize(vmb)))

        # Keep the section-less image so the re-framing can be PROVEN not to
        # have touched any non-section byte, and so the backward-compat control
        # artifact exists as a measured fact rather than a claim.
        plain = vmb + ".nosection"
        shutil.copyfile(vmb, plain)

        # A geometry the container cannot legally carry must stop the build, not
        # produce an artifact whose section the engine would misread.
        try:
            rep = install_into_vmb(vmb, neuron_kv_geom)
        except GeometryError as e:
            print("ERROR: the assembled container cannot carry the "
                  "neuron_kv section: %s" % e)
            print("   the section-less image is kept at %s" % plain)
            return 1
        ok, mism = verify_numerics_unchanged(plain, vmb, rep["sec_off"],
                                             rep["sec_len"])
        print("section: off=%d len=%d n_neurons=%d stride=%d mode=%s"
              % (rep["sec_off"], rep["sec_len"], rep["n_neurons"],
                 rep["stride"], rep["mode"]))
        print("vmb size: %d -> %d  (delta %+d B)"
              % (rep["file_len_before"], rep["file_len_after"],
                 rep["size_delta"]))
        print("non-section bytes identical to the section-less image: %s"
              % ok)
        if not ok:
            print("ERROR: the re-framing changed bytes outside the section: %r"
                  % (mism[:4],))
            return 1
        problems, h, (tag, off, ln) = verify_section_of(vmb, neuron_kv_geom)
        if problems:
            print("ERROR: the section on disk does not read back: %s"
                  % "; ".join(problems))
            return 1
        print("read-back: tag=%d off=%d len=%d header=%s  text_off=%d -> OK"
              % (tag, off, ln,
                 struct.unpack_from("<5I", open(vmb, "rb").read()[off:off + 20], 0),
                 h["text_off"]))

    total = sum(os.path.getsize(os.path.join(args.out, c[1])) for c in consts)
    print("total weights: %.3f GB (bf16)" % (total / 1e9))
    return 0


if __name__ == "__main__":
    sys.exit(main())
