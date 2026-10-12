# VMLM unified verification (Lead-maintained).
#
# ASCII-ONLY BY DESIGN: PowerShell 5.1 reads .ps1 as GBK/ANSI unless the file has
# a UTF-8 BOM. Chinese comments written without a BOM get mis-decoded and can
# produce stray quote/brace tokens -> "Unexpected token '}'" parse errors that
# point at innocent lines. Keep this file pure ASCII.
#
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File tools/verify_all.ps1 [-SkipAot]
#   -SkipAot : skip the two AOT suites (they rebuild feature-enabled engines and
#              embed a 19.5MB dataset). Use for everyday regression.

$SkipAot = $args -contains "-SkipAot"

$ErrorActionPreference = "Continue"
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $root

$script:results = @()

function Run-Check {
    # NOTE: do not name a parameter $Args -- it collides with the automatic variable.
    # $ExeArgs is an ARRAY of arguments. Passing a single space-joined string and
    # re-splitting it looks tempting but mangles Windows paths: PowerShell's
    # splatting re-parses "\r" / "\t" as escapes, so "examples/rnn/train" arrives
    # as "e" + CR + "nn" + TAB + "rain". (Cost me a long debug cycle.)
    param([string]$Name, [string]$Exe, [string[]]$ExeArgs = @())
    Write-Host ""
    Write-Host ("--- " + $Name + " ---") -ForegroundColor Cyan
    $path = Join-Path $root ("target/release\" + $Exe + ".exe")
    if (-not (Test-Path $path)) {
        Write-Host ("  MISSING: " + $Exe + ".exe")
        $script:results += [pscustomobject]@{ Item = $Name; Result = "MISSING"; Note = "not built" }
        return
    }
    if ($ExeArgs.Count -gt 0) {
        # Do not pipe straight into Out-String: for large outputs the pipeline can
        # clobber $LASTEXITCODE and mis-record a pass as FAIL.
        $out = & $path @ExeArgs 2>&1
    } else {
        $out = & $path 2>&1
    }
    $code = $LASTEXITCODE
    $out = ($out | Out-String)
    $lines = ($out -split "`n" | Where-Object { $_.Trim() -ne "" })
    $tail = ($lines | Select-Object -Last 2) -join " | "
    if ($code -eq 0) { $res = "PASS" } else { $res = "FAIL(exit=" + $code + ")" }
    $script:results += [pscustomobject]@{ Item = $Name; Result = $res; Note = $tail.Trim() }
    Write-Host ("  " + $tail)
}

# Run an external verification script and record its result.
function Run-Script {
    param([string]$Name, [string]$ScriptName)
    Write-Host ""
    Write-Host ("--- " + $Name + " ---") -ForegroundColor Cyan
    $p = Join-Path $root ("tools\" + $ScriptName)
    if (-not (Test-Path $p)) {
        $script:results += [pscustomobject]@{ Item = $Name; Result = "MISSING"; Note = $ScriptName }
        return
    }
    $o = & powershell -NoProfile -ExecutionPolicy Bypass -File $p 2>&1 | Out-String
    $code = $LASTEXITCODE
    $tailLines = ($o -split "`n" | Where-Object { $_.Trim() -ne "" })
    $tail = ($tailLines | Select-Object -Last 2) -join " | "
    if ($code -eq 0) { $res = "PASS" } else { $res = "FAIL(exit=" + $code + ")" }
    $script:results += [pscustomobject]@{ Item = $Name; Result = $res; Note = $tail.Trim() }
    Write-Host ("  " + $tail)
}

Write-Host "===== VMLM unified verification =====" -ForegroundColor Yellow

# --- P0-P6 regression ---
Run-Check "P1 GEMM kernel regression" "vmlm-tests"
# exp_f accuracy: guards the P9 precision debt paydown (0.66 ULP). Calls the REAL
# engine -- do not replace with a Python replica (replicas have been wrong twice).
Run-Check "P1 exp_f accuracy"         "vmlm-expcheck"
# 20000 steps is the converged configuration; 2000 only covers 1/10 of the samples.
Run-Check "P3 MNIST training"         "vmlm-train"  @("examples/mnist", "20000")
Run-Check "P4 RNN recurrent state"    "vmlm-rnn"    @("examples/rnn")
Run-Check "P5 Transformer attention"  "vmlm-attn"   @("examples/attn")
Run-Check "P6 dynamic sparsity"       "vmlm-sparse" @("examples/sparse")

# --- P8: training loops ---
Run-Check "P8 RNN BPTT training"      "vmlm-rnn-train"  @("examples/rnn/train")
Run-Check "P8 Transformer training"   "vmlm-attn-train"

# --- P7 deliverables ---
Run-Check "P7-A backward ops + AdamW" "vmlm-bwd-check"
Run-Check "P7-A lead cross-check"     "vmlm-lead-verify"
Run-Check "P7-C ESERNN migration"     "vmlm-eserNN"
Run-Check "P7-D GEMM benchmark"       "vmlm-bench"
Run-Check "P7-D packed crash guard"   "vmlm-packbench"
Run-Check "P7-E sparse thin-GEMM"     "sparse_opt_run" @("examples/sparse")

# --- PATH B (sparse-B): speed acceptance with tok/s ---
# The sparse-B (PATH: B1/B2) acceptance drivers are run standalone:
#   vmlm-b1-prod-eps (B1: determinism + epsilon), vmlm-b1-verify (20 criteria),
#   vmlm-b1-canfail (7), vmlm-b2-verify (40, known sole FAIL = A3 PER-ROW).
# This one adds the SPEED side the user asked for: a genuinely TIMED loop over
# 24 layers x 5 sparse matrices x 128 tokens, reported as tok/s, plus the
# bit-exact criteria of the token-blocked kernels (C1..C9).
# --reps 6 keeps it near 15 s (W7 PER-ROW is intentionally slow and is measured
# with only 2 reps); raise it for a headline number.
Run-Check "B sparse-B speed + tok/s"  "vmlm-sparseb-tok" @("--d","1536","--k0","20","--k","8","--layers","24","--mats","5","--tokens","128","--reps","6")

# P0/P1: sparse-B BACKWARD acceptance (dx + fused dw/dx).
# E1 forward epsilon, E2 dw bitwise, E3 dx bitwise, E4 fusion BITWISE-equivalent
# (dw and dx both 0 divergent), E5 can-fail, E6 determinism, E7 epsilon vs the
# masked operator at k=k0, E8 fusion speedup, E9 B2-SHARED dx bitwise.
Run-Check "B sparse-B backward (dx + fusion)" "vmlm-sparseb-bwd" @("--d","1536","--k0","20","--k","8","--reps","200")

# P1-b: source fan-out statistics -- the DECIDING measurement for whether the
# translator should actively reuse sources (task's own bar: <5% savings => skip).
# Reports both the infinite-register bound and the register-constrained bound.
Run-Check "B source fan-out verdict" "vmlm-sparseb-fanout" @("--d","1536","--k0","20","--layers","24","--mats","5")

# P2-a: compute-matched DENSE comparison (DENSE-REF -- a control, not a production path).
# Same data generator/seed as sparse_train24, same step count; d'=175 makes the dense
# per-layer parameters and per-token MACs match B1 (d=1536,k0=20) to 0.3%.
# Its DENSE-0 gradient check is the gate that keeps the comparison meaningful: an
# incorrect backward still lowers the loss, so the curve is only evidence AFTER it.
Run-Check "B dense compute-match + gradcheck" "vmlm-sparseb-dense24" @("--layers","24","--d","175","--steps","200","--batch","1","--seq","8","--vocab","64","--train-seqs","64","--val-seqs","32","--eval-seqs","8")

# B1 scaling-law driver (single-layer, real MiniCPM-tokenized data).
# One small config here as a REGRESSION (the full 128-step matrix, the tables and the
# fits live in docs/B1_scaling_law.md). Its self-check requires a finite final loss
# below ln(V)+5, which is the cheap way to catch a broken backward/optimizer.
Run-Check "B1 scaling driver (one config)" "vmlm-b1-scaling" @("--mode","b1","--dim","128","--k0","20","--seed","1","--steps","128","--seq","32","--batch","4","--vocab-mode","observed","--data-dir","artifacts/b1_scaling")

# --- P10: runtime loop control + real MiniCPM layer 0 ---
# The loop check assembles a one-template loop and asserts it runs exactly
# N iterations (this is the whole point of P10-A: flat instruction count).
Run-Script "P10 runtime loop control" "p10_loop_verify.ps1"
# MiniCPM5-1B layer 0 vs HuggingFace, stage by stage. Requires the
# transformers oracle dump (examples/minicpm/layer0_ref) -- if it is absent
# this reports SKIP rather than failing, so the suite stays runnable
# without a 2.2GB model checkout.
Run-Script "P10 MiniCPM layer0 vs HF" "p10_minicpm_verify.ps1"
# Task C: instruction-stream embed + weight mmap. Builds its own
# feature-enabled engines, so it is grouped with the other AOT suites
# but kept here because it is a P10 deliverable.
Run-Script "P10 AOT embed + weight mmap" "p10_mmap_verify.ps1"
# Task B: RoPE + the sin/cos kernels behind it. Checks accuracy against the
# HuggingFace rotate_half convention at positions up to 131071 (MiniCPM's
# max_position_embeddings), where a naive pos*inv_freq would be ~1e-2 off.
Run-Script "P10 RoPE + sin/cos" "p10_rope_verify.ps1"
# P11: optimization suite. Pins down that the algebraic eliminations are
# numerically correct, that the diagnostic explains why they do not speed
# things up (aux work <1% of GEMM time), and that quantization -- which
# shrinks the ACTUAL bottleneck (weight bytes) -- does, with accuracy
# verified on outlier-bearing weights.
Run-Script "P11 optimization (elim + quant)" "p11_opt_verify.ps1"

# P12: the engine actually LOADS AND RUNS MiniCPM5-1B.
# bf16 weights now compute correctly (before this, the engine read every
# constant as f32, so a bf16 weight was silent garbage -- the blocker for
# loading any real model). A full 24-layer forward + final norm + lm_head
# matches HuggingFace: 4/4 argmax and 1.9e-06 relative error.
Run-Script "P12 MiniCPM5-1B full model" "p12_minicpm_full_verify.ps1"

# P13 task A: the prompt is a RUNTIME input, not a compiled-in constant.
# Two independent checks:
#   * p13_emb_check   -- OP_EMB is dtype-aware (bf16 table widened bit-exactly,
#                        f32 path unregressed) and addressed by a state slot.
#   * p13_prompt_check-- the SAME .vmb produces different output for different
#                        token ids, bit-identical to the old const-mode path
#                        when given the same ids. No reassembly involved.
# Both are small/fast (1 layer), so they run in the default suite.
Run-Script "P13 EMB dtype-aware + state row" "p13_emb_verify.ps1"
Run-Script "P13 one .vmb serves any prompt" "p13_prompt_verify.ps1"
# The audit exception Task A added must not have widened the guard: illegal
# input (bf16 const to a non-EMB/GEMM op) still has to fail loudly with the
# opcode and slot named.
Run-Script "P13 dtype audit still fails loud" "p13_audit_verify.ps1"
# P13 task C: incremental decode against a K/V cache held in STATE slots.
# Before this, generating token t+1 re-ran the whole sequence 0..t (measured
# 13.2 s for 4 layers / P=64 / G=16); with the cache it is 1.01 s (13.15x).
# The check requires a cached step to reproduce a full re-prefill EXACTLY, at
# several prompt lengths and with 2 layers so errors would have to compound.
Run-Script "P13 incremental decode (KV cache)" "p13_kv_verify.ps1"
# P13 task D: int8 weights with dequantisation inside the GEMM, plus the
# "declared dtype vs available kernel" hard check. Shrinks the MiniCPM artifact
# from 2.161 GB (bf16) to 1.283 GB (int8) and keeps argmax 4/4, but does NOT
# meet the <1e-2 relative bar -- see docs/P13_report for the measurement.
Run-Script "P13 int8 weights + dequant GEMM" "p13_q8_verify.ps1"
# P13 task E: ONE layer template run NL times under LOOP_CNT with a runtime
# constant base (OP_LAYERB), instead of 24 unrolled bodies. That is P10's
# unfinished item 5. Verified against the unrolled build, which is already
# validated against HuggingFace; the two must agree EXACTLY.
Run-Script "P13 looped layers (runtime base)" "p13_loop_verify.ps1"

# --- AOT suites (each builds its own feature-enabled engine) ---
Run-Script "P7-B AOT standalone" "aot_verify.ps1"
if (-not $SkipAot) {
    Run-Script "P8-G AOT multi-model" "aot_multi_verify.ps1"
    Run-Script "P8-G AOT train loop"  "aot_train_verify.ps1"
}

Write-Host ""
Write-Host "===== Summary =====" -ForegroundColor Yellow
$script:results | Format-Table -AutoSize -Wrap

# --- Engine size budget (Lead re-baselined to 48000 B for P7) ---
# The AOT suites above rebuild the engine with models embedded, which inflates
# this file. Rebuild the plain engine before reading its size.
$rt = Join-Path $root "runtime"
Push-Location $rt
& cargo build --release --target x86_64-pc-windows-msvc 2>&1 | Out-Null
Pop-Location
$src = Join-Path $root "target/x86_64-pc-windows-msvc/release/vmlm-runtime.exe"
$eng = Join-Path $root "target/release/vmlm-runtime.exe"
if (Test-Path $src) { Copy-Item $src $eng -Force }
if (Test-Path $eng) {
    $sz = (Get-Item $eng).Length
    # P10 re-baseline: 48000 -> 51200. The +3200 B buys, MEASURED:
    #   +128  B : OP_RMSNORM kernel + OP_LOOP_CNT + the new hard-failure guards
    #   +2560 B : OP_SIN/OP_COS (Cody-Waite reduction + minimax) and OP_ROPE
    # The trigonometric kernels are what let RoPE run without a 134 MB
    # precomputed cos/sin constant table at MiniCPM's 131072 max positions.
    # Full rationale and the stub-and-measure method: tools/build_all.ps1.
    #
    # P13 re-baseline: 53312 -> 64000. The user granted a flat +10 KB and asked
    # that future work NOT trade performance (or agent time) for bytes, so this
    # number is now HEADROOM rather than a target to squeeze against: prefer the
    # clearer/faster implementation and let the size land where it lands. The
    # stub-and-measure tooling (tools/p13_sizeprobe.py) remains for when a
    # number is genuinely needed -- measure, never estimate.
    #
    # P14 re-baseline: 64000 -> 68096. Two steps, both measured:
    #   64000 -> 66560 (P13 tasks D+E, by the other workstream: int8 dequant
    #                    GEMM + OP_LAYERB runtime constant base).
    #   66560 -> 68096 (P14 O6: OP_MAX + OP_TOPK_IDX). Both ops were DEFINED in
    #                    format.rs but UNIMPLEMENTED in interp.rs -- merely
    #                    listed by tools/p10_opcode_gaps.py as a "user-visible
    #                    trap". So the engine had NO argmax anywhere, and O6's
    #                    "argmax fed back as the next token" was unwritable.
    #   The S5 neuron-kv module is behind a cargo feature and contributes 0 B
    #   to this figure (verified by rebuilding with the feature OFF), so the
    #   entire +1536 B is these two ops plus their loud-failure guards. Those
    #   guards matter: OP_EMB had no row-bounds check, and on the O6 path the
    #   row index is produced by an in-engine argmax (data, not a literal), so
    #   a bad index would have silently embedded zeros and generated
    #   plausible-looking garbage. Per the user's standing P13 instruction this
    #   is HEADROOM, not a target to squeeze against.
    #   build_all.ps1 carries the identical rationale and value.
    #
    # P15 re-baseline: 68096 -> 84480. USER-APPROVED headroom grant.
    #
    # P16 (connection sparsity) raises the DEFAULT engine, because the two new
    # opcodes are inference instructions rather than test scaffolding:
    #   68096 -> 71680 (+3584, OP_TOPK_BY_STRENGTH) -> 74752 (+3072, OP_SPARSE_MATVEC)
    #   --features train positive control = 96256 (+21504), so the apparatus moves.
    #   Headroom left: 84480 - 74752 = 9728 B.  All measured in ISOLATED target
    #   dirs; the conn-sparse storage module is feature-gated and costs 0 B.
    #
    #   MEASURED (isolated target dirs): default/feature-OFF = 68096 B (UNCHANGED),
    #   --features neuron-kv = 81408 B (delta +13312 B), --features train = 88576 B
    #   (+20480 B positive control proving the apparatus moves).
    #   NOTE: building both ways in the SAME target dir can report a fake delta
    #   of 0 because cargo reuses the cached artifact -- use separate target dirs.
    #
    #   Why this one is a deliberate grant and not a squeeze target: S5 step 1
    #   is to give core/src/neuron_kv.rs its FIRST REAL CALLER, and that is the
    #   one change in this project that cannot be free.
    #
    #   P14 "S5 contributes 0 B" was an honest measurement -- it shipped with a
    #   +20480 B positive control (--features train) proving the apparatus could
    #   detect a real difference, which is what makes a zero trustworthy here.
    #   But the REASON for the zero was unreachability: nothing called the
    #   module. That is exactly the gap docs/P14_S5_O6_report section 9.1 names
    #   as step 1, and exactly what P14 verifier left UNVERIFIED
    #   (docs/P14_verification.md section 3: the 24-layer engine end-to-end was
    #   UNVERIFIED because neuron_kv.rs had no caller). Wiring it links code
    #   into the engine, so the zero cannot survive the wiring.
    #
    #   P14 also flagged the 0 B headroom as a risk (section 5.3): with
    #   68096 == 68096 the budget had stopped working as a warning and the next
    #   byte would trip it. The user chose an explicit re-baseline over a size
    #   hunt, per the standing P13 instruction not to trade performance or agent
    #   time for bytes.
    #
    #   INVARIANT THIS BASELINE MUST NOT BREAK: the DEFAULT inference build is
    #   still 68096 B exactly. The S5 load-time entry sits behind the existing
    #   `neuron-kv` feature, so the inference engine pays nothing. If the default
    #   number moves, the gating leaked -- a defect to fix, not a baseline to accept.
    $budget = 84480
    if ($sz -le $budget) { $verdict = "OK (<= $budget)" } else { $verdict = "OVER BUDGET" }
    Write-Host ("Engine core size: " + $sz + " bytes  -> " + $verdict)
}

$fails = @($script:results | Where-Object { $_.Result -notmatch "^PASS" }).Count
Write-Host ""
Write-Host ("Non-passing items: " + $fails)
exit $fails
