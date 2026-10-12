# P16-4 (task-10): runner for the O6 generation-chain cycle investigation.
#
# PURE ASCII (mandatory for .ps1 in this repo).
#
# WHAT THIS RUNS, AND WHY IN THIS ORDER
#   1. The fast, read-only analysis of EXISTING dumps (cycle structure + the five
#      discriminators + the 13-item self-test). Seconds, no engine, no model load.
#   2. The weight-provenance check (needs a ~200 s HF model load).
#   3. The prefill control + seed sweep (same model load).
#   4. The MISSING MEASUREMENT: the HuggingFace oracle over the GENERATION
#      positions. This is the decisive step and the expensive one.
#
# NOT run here on purpose: tools/build_all.ps1 and tools/verify_all.ps1 (Lead-owned
# serialized acceptance gates), and any rebuild of a GB-scale artifact.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools/p16_o6_cycle.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools/p16_o6_cycle.ps1 -Fast
#
# -Fast skips steps 2-4 (no HF model load) and only re-runs the dump analysis.

param(
    [switch]$Fast,
    [int]$Layers = 4
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$out  = Join-Path $root "examples/p16_o6"
$py   = "python"

if (-not (Test-Path $out)) { New-Item -ItemType Directory -Force -Path $out | Out-Null }
Set-Location $root

function Step($n, $desc, $cmd) {
    Write-Host ""
    Write-Host ("=== [{0}] {1} ===" -f $n, $desc) -ForegroundColor Cyan
    # Redirect the native command's stdout to a REAL FILE, never a pipe: P13
    # measured that draining the engine's stdout through a PowerShell pipe made a
    # 1.14 s run look like ~20 minutes. The same discipline is kept here.
    $log = Join-Path $out ("_log_{0}.txt" -f $n)
    & $py @cmd *> $log
    $code = $LASTEXITCODE
    # NOTE: everything printed below must go through Write-Host, NOT the success
    # stream. A bare expression in a PowerShell function is RETURNED, so piping
    # Get-Content into the pipeline would make this function return an ARRAY and
    # the caller would capture the text instead of the exit code.
    Get-Content $log -ErrorAction SilentlyContinue |
        Where-Object { $_ -notmatch "it/s|Loading weights|UNEXPECTED|^Notes:|not ok if you expect" } |
        ForEach-Object { Write-Host $_ }
    Write-Host ("    exit={0}  log={1}" -f $code, $log) -ForegroundColor Green
    return $code
}

Write-Host "P16-4: O6 generation-chain cycle -- model property or implementation defect?"
Write-Host "layers under test: $Layers (the artifact's ACTUAL depth; see meta.json)"

# ---- non-fatal helper: keep going so one failure does not hide the rest ----
$codes = @{}

$codes["1_cycle"] = Step "1_cycle" "cycle structure + discriminators + self-test (no engine)" `
    @("tools/p16_o6_cycle.py")

if (-not $Fast) {
    $codes["2_weights"] = Step "2_weights" "weight provenance: engine vs HF truncation vs safetensors" `
        @("tools/p16_o6_weight_provenance.py")
    $codes["3_prefill"] = Step "3_prefill" "prefill control + seed sweep + position independence" `
        @("tools/p16_o6_prefill_control.py", "--layers", "$Layers", "--out", "examples/p16_o6")
    $codes["4_oracle"] = Step "4_oracle" "THE MISSING MEASUREMENT: HF oracle at generation positions" `
        @("tools/p16_o6_hf_gen_oracle.py", "--layers", "$Layers", "--out", "examples/p16_o6")
} else {
    Write-Host ""
    Write-Host "-Fast: skipped the HF steps (2,3,4)." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== summary ===" -ForegroundColor Cyan
$fail = 0
foreach ($k in ($codes.Keys | Sort-Object)) {
    $c = $codes[$k]
    if ($c -ne 0) { $fail++ }
    Write-Host ("    {0,-14} exit={1}" -f $k, $c)
}
Write-Host ""
Write-Host "Deliverables in $out ; write-up: docs/P16_O6_cycle_finding.md"
if ($fail -gt 0) {
    Write-Host "ATTENTION: $fail step(s) exited non-zero -- read their logs before trusting any PASS." -ForegroundColor Red
}
exit $fail
