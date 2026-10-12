# P13 task E verification: LOOPED layer template with a runtime constant base.
#
# ASCII-ONLY (see verify_all.ps1 header for why).
#
# WHAT THIS PROVES
#   The unrolled MiniCPM build emits one body per layer (~4180 instructions for
#   24 layers). The looped build emits the body ONCE and runs it NL times with
#   LOOP_CNT, advancing a RUNTIME constant base (OP_LAYERB) by the per-layer
#   stride. This is P10's unfinished item 5, and it is what makes instruction
#   count flat in layer count instead of linear.
#
#   Correctness is checked by comparing against the UNROLLED build, which is
#   already validated against HuggingFace (rel 1.8806e-06, argmax 4/4). Matching
#   it transfers that validation without a second 9-minute HF run. The two
#   builds run the same arithmetic in the same order, so the requirement is
#   EXACT agreement, not a tolerance.
#
#   Also checked: the `--loop` builder REFUSES to build if any non-per-layer
#   constant would be read while the base is non-zero (a real bug class: the
#   base shifts every constant access, so the restore's own control constant
#   must be copied into an activation first).
#
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File tools/p13_loop_verify.ps1

$ErrorActionPreference = "Continue"
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$script:fails = 0

Write-Host "===== P13 looped layer template (runtime const base) =====" -ForegroundColor Yellow

# Two layer counts: 2 catches an off-by-one in the base advance, 4 catches
# accumulation. (1 layer is bit-exact even when the base NEVER advances, so it
# is deliberately not the only case tested -- that exact trap cost a cycle.)
foreach ($n in @(2, 4)) {
    $o = & python (Join-Path $root "tools/p13_loop_check.py") --layers $n --seq 4 2>&1 | Out-String
    $code = $LASTEXITCODE
    $tail = (($o -split "`n") | Where-Object { $_ -match 'VERDICT|max\|d\||instruction count' }) -join " | "
    Write-Host ("  layers=$n : " + $tail.Trim())
    if ($code -ne 0) { $script:fails++ }
}

Write-Host ""
if ($script:fails -eq 0) {
    Write-Host "P13 looped layers: ALL PASS" -ForegroundColor Green
} else {
    Write-Host ("P13 looped layers: " + $script:fails + " FAILED") -ForegroundColor Red
}
exit $script:fails
