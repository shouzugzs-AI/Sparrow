# P13 task D verification: int8 weights + dequantising GEMM.
#
# ASCII-ONLY (see verify_all.ps1 header for why).
#
# WHAT THIS PROVES
#   1. The kernel is numerically correct: C = (A . W_i8) * scale, with the
#      per-output-row scale named as the GEMM's 4th operand. Checked on a
#      hand-computable case with a NON-UNIFORM scale, so a kernel that ignored
#      the scale could not coincidentally pass.
#   2. Illegal input still HARD-FAILS rather than being reinterpreted:
#        * i8 constant used by a non-GEMM op        -> exit 9
#        * scale operand on a non-i8 weight         -> assembler refuses
#        * scale shorter than N rows                -> exit 9
#      This is the "declared dtype vs available kernel" check the task book
#      asks for, and it is the same class of guard P12 introduced.
#
# ACCURACY IS REPORTED, NOT ASSERTED HERE
#   Measured on REAL MiniCPM weights (tools/p13_quant_plan.py): int8 keeps
#   argmax 4/4 but gives rel 2.89e-02, which EXCEEDS the project's <1e-2 bar.
#   The 24-layer end-to-end (tools/p13_q8_e2e.py) confirms the engine matches
#   the numpy prediction to 5 significant digits, so the engine is faithful to
#   the model; the criterion itself is what int8 does not meet. Asserting a
#   pass here would be dishonest, so the check asserts CORRECTNESS and the
#   report carries the accuracy verdict.
#
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File tools/p13_q8_verify.ps1

$ErrorActionPreference = "Continue"
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

Write-Host "===== P13 int8 weights + dequantising GEMM =====" -ForegroundColor Yellow

$out = & python (Join-Path $root "tools/p13_q8_check.py") 2>&1 | Out-String
$code = $LASTEXITCODE
foreach ($line in ($out -split "`n")) {
    if ($line -match 'PASS|FAIL|VERDICT|engine   =|expected =') {
        Write-Host ("  " + $line.Trim())
    }
}
if ($code -ne 0) {
    Write-Host ("P13 int8: FAILED (exit=" + $code + ")") -ForegroundColor Red
    Write-Host $out
    exit $code
}
Write-Host "P13 int8: ALL PASS" -ForegroundColor Green
exit 0
