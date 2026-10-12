# P13 verification: the load-time dtype audit is still a real guard.
#
# ASCII-ONLY (see verify_all.ps1 header for why).
#
# WHY THIS EXISTS
#   Task A added a NAMED exception to audit_const_dtypes so OP_EMB may take a
#   bf16 table. The failure mode of such an exception is that it silently turns
#   a precise guard into a blanket hole, re-opening the P12 bug class (a bf16
#   constant read as f32 = garbage with no error). This asserts the exception
#   is exactly as wide as intended:
#     * bf16 const used by RMSNORM  -> exit 9, naming the op and slot
#     * bf16 const used by COPY     -> exit 9, naming the op and slot
#     * bf16 table fed to EMB       -> runs
#     * f32 table fed to EMB        -> runs (no regression)
#
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File tools/p13_audit_verify.ps1

$ErrorActionPreference = "Continue"
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

Write-Host "===== P13 dtype audit still rejects illegal input =====" -ForegroundColor Yellow

$out = & python (Join-Path $root "tools/p13_audit_check.py") 2>&1 | Out-String
$code = $LASTEXITCODE
foreach ($line in ($out -split "`n")) {
    if ($line -match 'PASS|FAIL|VERDICT|msg:') {
        Write-Host ("  " + $line.Trim())
    }
}
if ($code -ne 0) {
    Write-Host ("P13 audit: FAILED (exit=" + $code + ")") -ForegroundColor Red
    Write-Host $out
    exit $code
}
Write-Host "P13 audit: ALL PASS" -ForegroundColor Green
exit 0
