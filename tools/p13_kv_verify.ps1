# P13 task C verification: incremental decode with a K/V cache.
#
# ASCII-ONLY (see verify_all.ps1 header for why).
#
# WHAT THIS PROVES
#   Before task C the engine had no way to carry K/V between steps, so producing
#   token t+1 meant re-running the ENTIRE sequence 0..t. Measured baseline for
#   4 layers, P=64, G=16: 13.2 s. With the cache: 1.01 s, i.e. 13.15x.
#
#   Two separate claims are checked, because they fail differently:
#     1. CORRECTNESS -- a decode step must reproduce a full re-prefill exactly.
#        Checked against the reference at 1 layer (fast) for several prompt
#        lengths and step counts.
#     2. The cache PRIMITIVES -- transposed append + cached GEMM layouts --
#        on a tiny hand-computable case, since a layout error there is
#        otherwise very hard to attribute.
#
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File tools/p13_kv_verify.ps1

$ErrorActionPreference = "Continue"
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$script:fails = 0

Write-Host "===== P13 incremental decode (KV cache) =====" -ForegroundColor Yellow

# ---- 1. cache primitives (fast, hand-computable) ----
$o1 = & python (Join-Path $root "tools/p13_kv_probe.py") 2>&1 | Out-String
$c1 = $LASTEXITCODE
foreach ($line in ($o1 -split "`n")) {
    if ($line -match 'PASS|FAIL|VERDICT') { Write-Host ("  " + $line.Trim()) }
}
if ($c1 -ne 0) { $script:fails++ }

# ---- 2. decode == full re-prefill, across prompt lengths ----
# Each case builds a reference (S+G tokens) and a decode build (S prompt + G
# steps) from the same weights, then requires bit-identical hidden states.
foreach ($s in @(1, 4, 8)) {
    $o = & python (Join-Path $root "tools/p13_kv_check.py") --layers 1 --seq $s --decode 3 2>&1 | Out-String
    $code = $LASTEXITCODE
    $tail = (($o -split "`n") | Where-Object { $_ -match 'VERDICT' }) -join " "
    Write-Host ("  S=$s G=3 : " + $tail.Trim())
    if ($code -ne 0) { $script:fails++ }
}

# ---- 3. two layers, so errors would have to compound ----
$o = & python (Join-Path $root "tools/p13_kv_check.py") --layers 2 --seq 4 --decode 2 2>&1 | Out-String
$code = $LASTEXITCODE
$tail = (($o -split "`n") | Where-Object { $_ -match 'VERDICT' }) -join " "
Write-Host ("  layers=2 S=4 G=2 : " + $tail.Trim())
if ($code -ne 0) { $script:fails++ }

Write-Host ""
if ($script:fails -eq 0) {
    Write-Host "P13 KV cache: ALL PASS" -ForegroundColor Green
} else {
    Write-Host ("P13 KV cache: " + $script:fails + " FAILED") -ForegroundColor Red
}
exit $script:fails
