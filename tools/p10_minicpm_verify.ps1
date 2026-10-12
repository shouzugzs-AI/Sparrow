# P10-B verification: MiniCPM5-1B layer 0 on the real engine vs HuggingFace.
#
# ASCII-ONLY (see verify_all.ps1 header for why).
#
# Reports SKIP (exit 0) when the HF oracle dump or the model is absent --
# the suite must stay runnable without a 2.2 GB model checkout. When the
# reference IS present this is a hard check, because it is the only test
# that pins RMSNorm / GQA / RoPE / SwiGLU against a real 1536-dim model.
#
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File tools/p10_minicpm_verify.ps1

$ErrorActionPreference = "Continue"
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$ref  = Join-Path $root "examples/minicpm/layer0_ref"
$run  = Join-Path $root "tools/minicpm_layer0_run.py"

Write-Host "===== P10-B MiniCPM5-1B layer 0 vs HuggingFace =====" -ForegroundColor Yellow

if (-not (Test-Path (Join-Path $ref "meta.json"))) {
    Write-Host "  SKIP: no HF oracle at $ref"
    Write-Host "        (generate with: python tools/minicpm_layer0_oracle.py)"
    exit 0
}
if (-not (Test-Path $run)) {
    Write-Host "  SKIP: missing $run"
    exit 0
}
$vmb = Join-Path $root "examples/minicpm/layer0.vmb"
if (-not (Test-Path $vmb)) {
    Write-Host "  building layer0.vmb ..."
    python (Join-Path $root "tools/minicpm_layer0_vml.py") | Out-Null
}

$out = & python $run 2>&1 | Out-String
$code = $LASTEXITCODE
# Show the stage table so a failure is diagnosable from CI output alone.
foreach ($line in ($out -split "`n")) {
    if ($line -match 'rel|worst|RESULT|stage|OK|FAIL') { Write-Host ("  " + $line.TrimEnd()) }
}
Write-Host ""
if ($code -eq 0) {
    Write-Host "MiniCPM layer 0: PASS (engine matches HF at every stage)" -ForegroundColor Green
} else {
    Write-Host "MiniCPM layer 0: FAIL" -ForegroundColor Red
}
exit $code
