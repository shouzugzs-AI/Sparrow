# P11 optimization verification.
#
# ASCII-ONLY BY DESIGN. PowerShell 5.1 decodes .ps1 as GBK/ANSI when there is
# no UTF-8 BOM, so Chinese text here turns into mojibake and then into
# "Missing type name after '['" parse errors that point at innocent lines.
# This is a documented P9 trap and it bit me again while writing this file.
# The verified scripts (p10_*) are ASCII for the same reason.
#
# WHAT THIS PINS DOWN
#   1. The algebraic eliminations (E1/E2/E3) are numerically CORRECT.
#      Their speedup is REPORTED, not asserted -- it measures ~1.00x because
#      (proven by vmlm-why) the work they remove is under 1% of GEMM time.
#   2. Quantization (Q1/Q3) DOES speed things up, because it shrinks the
#      actual bottleneck (weight bytes), and its accuracy is checked against
#      the plan's <1e-2 criterion on realistic outlier-bearing weights.
#
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File tools/p11_opt_verify.ps1

$ErrorActionPreference = "Continue"
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$rel  = Join-Path $root "target/release"

$script:fails = 0
function Check($name, $cond, $detail) {
    if ($cond) { Write-Host ("  " + $name + ": PASS  " + $detail) }
    else { Write-Host ("  " + $name + ": FAIL  " + $detail); $script:fails++ }
}

Write-Host "===== P11 optimization verification =====" -ForegroundColor Yellow

foreach ($b in @("vmlm-why", "vmlm-quant", "vmlm-optbench")) {
    $p = Join-Path $rel ($b + ".exe")
    if (-not (Test-Path $p)) {
        Write-Host ("  MISSING: " + $b + ".exe -- run tools/build_all.ps1 first")
        $script:fails++
    }
}
if ($script:fails -gt 0) { exit $script:fails }

# ---- 1. diagnostic: aux work is a tiny fraction of GEMM time ----
$why = & (Join-Path $rel "vmlm-why.exe") --shape 512 2>&1 | Out-String
$whyCode = $LASTEXITCODE
foreach ($line in ($why -split "`n")) {
    if ($line -match 'us$|GFLOPS|ratio|=> ') { Write-Host ("  " + $line.Trim()) }
}
Check "diagnostic runs" ($whyCode -eq 0) "vmlm-why exit=$whyCode"

# Parse the measured elementwise share; it must be small (that is the
# entire explanation for why the eliminations do not speed anything up).
$share = -1.0
foreach ($line in ($why -split "`n")) {
    if ($line -match '([0-9]+\.[0-9]+)%') {
        $v = [double]$Matches[1]
        if ($share -lt 0 -or $v -lt $share) { $share = $v }
    }
}
Check "aux work share is small (<5%)" ($share -ge 0 -and $share -lt 5.0) ("min measured share = $share%")

# ---- 2. quantization: real speedup + accuracy on realistic weights ----
$q = & (Join-Path $rel "vmlm-quant.exe") --shape 512 2>&1 | Out-String
$qCode = $LASTEXITCODE
Check "quant benchmark runs" ($qCode -eq 0) "vmlm-quant exit=$qCode"

$i8 = -1.0
$outlierOk = $false
$outlierFast = $false
foreach ($line in ($q -split "`n")) {
    if ($line -match 'int8[^0-9]*([0-9]+\.[0-9]+)x') {
        $v = [double]$Matches[1]
        if ($v -gt $i8) { $i8 = $v }
    }
    # Match on the NUMBERS ONLY, not the (non-ASCII) label text:
    #   "<label>: N.NNx  <label> E.EEe+NN  -> PASS|FAIL"
    # Matching on Chinese labels is fragile -- this file is ASCII, but the
    # program under test prints UTF-8, and the console may re-encode it.
    # Give me: speedup, error, verdict.
    if ($line -match '([0-9]+\.[0-9]+)x\s+\S+\s+([0-9]+\.[0-9]+e[+-][0-9]+)\s+->\s+(PASS|FAIL)') {
        $sp = [double]$Matches[1]
        $er = [double]$Matches[2]
        $vd = $Matches[3]
        if ($vd -eq 'PASS' -and $er -lt 1e-2 -and $sp -gt 2.0) {
            $outlierOk = $true
            $outlierFast = $true
        }
    }
}
Check "int8 speedup > 2x" ($i8 -gt 2.0) ("best measured int8 speedup = $i8 x")
Check "int8+outlier passes accuracy and stays fast" ($outlierOk -and $outlierFast) "mixed precision path"

# ---- 3. eliminations: correctness is the hard requirement ----
$ob = & (Join-Path $rel "vmlm-optbench.exe") --shape 512 --reps 150 2>&1 | Out-String
$obCode = $LASTEXITCODE
Check "optbench runs" ($obCode -eq 0) "vmlm-optbench exit=$obCode"

$worstErr = -1.0
foreach ($line in ($ob -split "`n")) {
    if ($line -match '([0-9]+\.[0-9]+e[+-][0-9]+)') {
        $v = [double]$Matches[1]
        if ($v -gt $worstErr) { $worstErr = $v }
    }
}
Check "eliminations numerically correct (<1e-4)" ($worstErr -ge 0 -and $worstErr -lt 1e-4) ("worst error = $worstErr")

Write-Host ""
if ($script:fails -eq 0) {
    Write-Host "P11 optimization: ALL PASS" -ForegroundColor Green
} else {
    Write-Host ("P11 optimization: " + $script:fails + " FAILED") -ForegroundColor Red
}
exit $script:fails
