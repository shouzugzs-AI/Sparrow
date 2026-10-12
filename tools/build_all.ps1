# VMLM canonical build script
#
# WHY THIS EXISTS (two independent reasons, both empirically verified):
#
# 1. Cargo only reads .cargo/config.toml from the invocation directory upward.
#    vmlm-runtime is no_std and needs extra LINK flags (/ENTRY, /NODEFAULTLIB, ...)
#    from runtime/.cargo/config.toml. Those flags BREAK the std tool crates if
#    applied globally (verified: vmlm-asm fails with LNK2001 mainCRTStartup).
#    => std crates and the engine MUST be built from different directories.
#
# 2. That same config ALSO breaks a build script inside runtime/ (member G found
#    this; Lead reproduced it). Fix: pass --target x86_64-pc-windows-msvc, which
#    separates host from target so target.*.rustflags no longer hit the host
#    build script. Verified safe for std crates as well.
#
# Side effect of --target: artifacts land in target\<triple>/release\ instead of
# target/release/. This script mirrors them back so existing tools keep working.
#
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File tools/build_all.ps1

$ErrorActionPreference = "Continue"
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$triple = "x86_64-pc-windows-msvc"
$fail = 0

function Step($name, $dir, $cargoArgs) {
    Write-Host "--- $name ---" -ForegroundColor Cyan
    Push-Location $dir
    $out = & cargo build @cargoArgs --release 2>&1 | Out-String
    $code = $LASTEXITCODE
    Pop-Location
    if ($code -ne 0) {
        Write-Host $out
        Write-Host "  FAILED ($name)" -ForegroundColor Red
        $script:fail++
    } else {
        Write-Host "  ok"
    }
}

Write-Host "===== VMLM build =====" -ForegroundColor Yellow

# std crates: build from workspace root
Step "vmlm-asm (assembler)"      $root @("-p", "vmlm-asm")
Step "vmlm-core (library)"       $root @("-p", "vmlm-core")
Step "vmlm-tests (kernel tests)" $root @("-p", "vmlm-tests", "--bins")
Step "vmlm-train (drivers)"      $root @("-p", "vmlm-train", "--bins")

# no_std engine: MUST build from runtime/; --target unlocks build.rs
$rt = Join-Path $root "runtime"
Step "vmlm-runtime (engine core)" $rt @("--target", $triple)
Step "vmlm-runtime (AOT)"         $rt @("--features", "aot", "--target", $triple)
Step "vmlm-runtime (bench)"       $rt @("--features", "bench", "--target", $triple)
Step "vmlm-runtime (engine, final)" $rt @("--target", $triple)

# Mirror the engine back to target/release\ so existing scripts keep working.
$src = Join-Path $root "target\$triple\release\vmlm-runtime.exe"
$dst = Join-Path $root "target/release/vmlm-runtime.exe"
if (Test-Path $src) {
    Copy-Item $src $dst -Force
    Write-Host ""
    Write-Host "Mirrored engine -> target/release/vmlm-runtime.exe"
}

if (Test-Path $dst) {
    $sz = (Get-Item $dst).Length
    # Budget history:
    #   40000 -> original white-paper target
    #   48000 -> P7 re-baseline (user-approved)
    #   48256 -> P10 first re-baseline (RMSNORM + LOOP_CNT + JMP fix)
    #   51200 -> P10 second re-baseline (SIN/COS + ROPE, user pre-approved:
    #            "???????????????????????????????????)
    #   51712 -> P12 (bf16 constant pool + widening GEMM)
    #   53312 -> P12 follow-up (assemble-time dtype audit)
    #   53760 -> P13 task A (runtime prompt feed), user-approved
    #
    # WHY P12 NEEDED +512 B:
    #   MiniCPM/Llama weights ship as bf16. Before this, the engine read every
    #   constant as f32 -- a bf16 weight became garbage with no error, which is
    #   THE blocker for actually loading a real model. The fix is:
    #     * Slot carries a dtype (so the loader stops guessing)
    #     * a widening bf16 GEMM kernel
    #     * a hard-fail guard when any other op gets a non-f32 constant
    #   Measured effect: 2x smaller const pool (2.16 GB stays file-backed
    #   instead of 4.32 GB of private commit) AND 4.4-7.9x faster GEMM, since
    #   this machine's GEMM is load-port bound and bf16 halves the bytes.
    #
    #   53760 -> P13 task A (runtime prompt feed)
    #   64000 -> P13 user re-baseline: +10 KB headroom granted
    #   66560 -> P13 tasks D+E (int8 dequant GEMM, runtime layer base)
    #
    # WHY P12 NEEDED +512 B:
    #   MiniCPM/Llama weights ship as bf16. Before this, the engine read every
    #   constant as f32 -- a bf16 weight became garbage with no error, which is
    #   THE blocker for actually loading a real model. The fix is:
    #     * Slot carries a dtype (so the loader stops guessing)
    #     * a widening bf16 GEMM kernel
    #     * a hard-fail guard when any other op gets a non-f32 constant
    #   Measured effect: 2x smaller const pool (2.16 GB stays file-backed
    #   instead of 4.32 GB of private commit) AND 4.4-7.9x faster GEMM, since
    #   this machine's GEMM is load-port bound and bf16 halves the bytes.
    #
    # WHY P13 NEEDED +448 B (net; P13 also SAVED 1024 B elsewhere):
    #   Task A makes the prompt a runtime input instead of a compiled-in
    #   constant. Before P13 the prompt rows were extracted host-side and baked
    #   into the .vmb as `prompt_emb`, so EVERY new prompt meant reassembling
    #   1.76 GB (~106 s). Now token ids come from tokens.bin into the `st_tok`
    #   state slot and OP_EMB widens one bf16 row per token -- one artifact,
    #   any prompt, verified bit-identical to the old path.
    #
    # WHY P13 D+E NEEDED +2560 B (measured, not estimated):
    #   Task D: an int8 dequantising GEMM + the "declared dtype vs available
    #   kernel" audit. It shrinks the MiniCPM artifact 2.161 GB -> 1.283 GB and
    #   keeps argmax 4/4 (though NOT the <1e-2 relative bar -- see the report).
    #   Task E: OP_LAYERB, a bound-checked runtime constant base, which is what
    #   lets ONE layer template run 24 layers under LOOP_CNT instead of an
    #   unrolled 4180-instruction stream.
    #   The first cut of E cost +12.8 KB because `slot_mut` is the hottest
    #   accessor and its bound check inlined into ~40 call sites; splitting the
    #   cold half into an #[inline(never)] fn recovered 4.6 KB of that (measured
    #   70144 -> 65536). What remains is genuine functionality.
    #
    # THE USER'S STANDING INSTRUCTION (P13): do NOT trade performance or agent
    # time for bytes. This number is HEADROOM, not a target to squeeze against.
    # tools/p13_sizeprobe.py stays available for when a number is genuinely
    # needed -- measure, never estimate -- but do not optimise size for its own
    # sake.
    #
    #   66560 -> 68096 -> P14 O6 (argmax + autoregressive loop), Lead re-baseline
    #
    # WHY P14 O6 NEEDED +1536 B (measured, not estimated):
    #   O6 is the step from "can run a prompt" to "can generate": the decode
    #   argmax is fed BACK as the next token. That needs two ops which were
    #   DEFINED in format.rs but ABSENT from interp.rs -- tools/p10_opcode_gaps.py
    #   listed OP_MAX (34) and OP_TOPK_IDX (42) as "user-visible trap": the
    #   assembler accepted the names, the engine refused them at load. So there
    #   was no argmax anywhere in the engine and the loop could not be written.
    #   Measured delta: engine 66560 -> 68096 B with the two arms + their
    #   hard-failure guards. Verified by rebuilding with core's `neuron-kv`
    #   feature OFF: the S5 module contributes 0 B, so the whole +1536 B is
    #   these two ops.
    #   The guards are NOT padding: OP_EMB had no row-bounds check, so an
    #   out-of-range row silently returned zeros. On the O6 path the row index
    #   is now produced by an in-engine argmax, i.e. it is data, not a literal
    #   an author checked -- a bad index would have produced plausible-looking
    #   garbage forever rather than failing. That is the exact silent
    #   degradation class this project converts to hard failures.
    #   Per the standing instruction the +1536 B is granted as headroom rather
    #   than recovered by weakening a guard.
    #
    #   68096 -> 84480 -> P15 S5 wiring (user-approved re-baseline)
    #   84480 -> 74752 (USED, not a new baseline) -> P16 connection sparsity
    #
    # P16 MEASURED COST (isolated target dirs; the DEFAULT engine, not a feature):
    #   P15 baseline                        : 68096 B
    #   + OP_TOPK_BY_STRENGTH (93)          : 71680 B   delta +3584 B
    #   + OP_SPARSE_MATVEC SEL-GATED mode   : 74752 B   delta +3072 B (cumulative 6656)
    #   --features train positive control   : 96256 B   delta +21504 B
    #   => headroom remaining: 84480 - 74752 = 9728 B
    #   The two opcodes are IN the default engine (they are inference
    #   instructions, not test scaffolding), so unlike S5 they permanently
    #   raise the default size. Budget 84480 still holds.
    #
    #   Known constraint: OP_TOPK_BY_STRENGTH's segmented mode is bounded by
    #   TOPK_MAX_L = 2048. That bound is NOT arbitrary -- it exists because
    #   the engine ships a deliberate no-op __chkstk stub, so a large stack
    #   frame skips the guard page and dies as 0xC0000005 with no diagnostic;
    #   the >2048 case is an explicit exit(9) instead. Do not raise it without
    #   re-measuring that crash mode.
    #
    #   The conn-sparse STORAGE module (core/src/conn_sparse.rs) is behind a
    #   `conn-sparse` feature that is NOT in `default`, and runtime/Cargo.toml
    #   does not forward it, so it contributes 0 B to this figure -- verified
    #   two ways (structural: feature absent + no forwarding + cfg gate;
    #   measured: 72704 B with the module vs 72704 B with it deleted, while
    #   the train control moved the same tree).
    #
    # MEASURED P15 COST (isolated target dirs, one variable changed per build):
    #   default (feature OFF) : 68096 B   <- UNCHANGED, the invariant that matters
    #   --features neuron-kv  : 81408 B   <- delta +13312 B
    #   --features train      : 88576 B   <- +20480 B (positive control: proves
    #                                        the size apparatus moves)
    #   attribution: A +0 B (call site dead -> neither module links; this is the
    #   honest reading of P14's old "S5 costs 0 B" -- it meant UNREACHABLE, not
    #   cheap) / B +512 B (core neuron_kv itself + its guards) / C +12800 B
    #   (the load path: section parse, reorder, bit-exact + vacuity verification).
    #   History of this delta across P15 revisions: +16384, +17408, +10752,
    #   +13312 -- each a DIFFERENT revision of the same code, and the reason
    #   every figure here carries the build it came from.
    #
    # METHOD WARNING (cost a teammate a wrong number): building both ways in the
    # SAME target dir can report a fake delta of 0 -- cargo reuses the cached
    # artifact. Use separate target dirs for any A/B size claim.
    #
    # WHY P15 NEEDED A RE-BASELINE AND NOT A SIZE HUNT:
    #   P14 left the budget at EXACTLY the measured size (68096 == 68096, 0 B
    #   headroom), so it had stopped being a warning. P14's own report flagged
    #   that as a risk (搂5.3) and asked for an explicit decision before the next
    #   opcode rather than a surprise on contact.
    #
    #   S5's step 1 is to give `neuron_kv.rs` its first REAL caller. That is the
    #   one thing that CANNOT be free: P14's "S5 costs 0 B" is true only because
    #   nothing linked the module -- the measurement was honest (a +20480 B
    #   positive control proved the apparatus worked) but the reason for the zero
    #   was unreachability, which is the exact gap this step closes. Wiring it
    #   necessarily links code into the engine.
    #
    #   The user was asked, and chose: re-baseline to +5.5 KB against the
    #   standing P13 instruction ("do NOT trade performance or agent time for
    #   bytes; this number is HEADROOM, not a target to squeeze against").
    #   So this is a deliberate, user-approved headroom grant -- NOT a squeeze
    #   target, and NOT recovered by weakening a guard.
    #
    #   The DEFAULT inference build must still measure 68096 B: the S5 entry is
    #   behind the existing `neuron-kv` feature, so the inference engine pays
    #   nothing. The +5.5 KB headroom exists so that the feature-enabled and
    #   future builds have room without another re-baseline round-trip.
    $budget = 84480
    if ($sz -le $budget) { $verdict = "OK (<= $budget)" } else { $verdict = "OVER BUDGET" }
    Write-Host "Engine core size: $sz bytes  -> $verdict"
}

Write-Host ""
if ($fail -eq 0) {
    Write-Host "All builds OK. Now run: tools/verify_all.ps1" -ForegroundColor Green
} else {
    Write-Host "$fail build step(s) failed" -ForegroundColor Red
}
exit $fail
