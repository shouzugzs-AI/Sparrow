# P13 task A verification: OP_EMB is dtype-aware and state-addressed.
#
# ASCII-ONLY (see verify_all.ps1 header: PowerShell 5.1 decodes .ps1 as GBK
# unless there is a UTF-8 BOM, and mis-decoded Chinese produces parse errors
# that point at innocent lines).
#
# WHAT THIS PROVES
#   1. A bf16 embedding table is widened EXACTLY (bf16 is the top 16 bits of
#      f32, so the round trip must be bit-exact -- a tolerance would hide the
#      stride bug this dtype path exists to prevent).
#   2. The pre-P13 f32 table path is unchanged (no regression).
#   3. The row index is read from a `st_` state slot, which is what lets the
#      same .vmb serve any prompt.
#
# Reports SKIP (exit 0) if the model checkout is absent, matching the suite's
# convention of staying runnable without a 2.2 GB download.
#
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File tools/p13_emb_verify.ps1

$ErrorActionPreference = "Continue"
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

Write-Host "===== P13 EMB dtype-aware + state row =====" -ForegroundColor Yellow

$out = & python (Join-Path $root "tools/p13_emb_check.py") 2>&1 | Out-String
$code = $LASTEXITCODE
foreach ($line in ($out -split "`n")) {
    if ($line -match 'PASS|FAIL|VERDICT|engine printed') {
        Write-Host ("  " + $line.Trim())
    }
}
if ($code -ne 0) {
    Write-Host ("P13 EMB: FAILED (exit=" + $code + ")") -ForegroundColor Red
    Write-Host $out
    exit $code
}
Write-Host "P13 EMB: ALL PASS" -ForegroundColor Green
exit 0
