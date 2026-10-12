# P13 task A verification: ONE .vmb serves ANY prompt.
#
# ASCII-ONLY (see verify_all.ps1 header for why).
#
# WHAT THIS PROVES
#   Before P13 the prompt was baked into the artifact: the builder extracted
#   the prompt rows host-side and emitted them as a `prompt_emb` constant, so
#   every new prompt meant reassembling the whole model (~106 s, 1.76 GB).
#   Now token ids are read from tokens.bin into the `st_tok` state slot.
#
#   The check builds BOTH modes at 1 layer (fast) and asserts:
#     * tokens-mode is BIT-IDENTICAL to const-mode for the same ids
#       (bit-identical, not approximate: same weights, same ops, so any
#        difference means the lookup or row stacking is wrong)
#     * a different 16-byte tokens.bin changes the output, with no reassembly
#
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File tools/p13_prompt_verify.ps1

$ErrorActionPreference = "Continue"
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

Write-Host "===== P13 one .vmb serves any prompt =====" -ForegroundColor Yellow

$out = & python (Join-Path $root "tools/p13_prompt_check.py") --layers 1 2>&1 | Out-String
$code = $LASTEXITCODE
foreach ($line in ($out -split "`n")) {
    if ($line -match 'PASS|FAIL|VERDICT|differing') {
        Write-Host ("  " + $line.Trim())
    }
}
if ($code -ne 0) {
    Write-Host ("P13 prompt: FAILED (exit=" + $code + ")") -ForegroundColor Red
    Write-Host $out
    exit $code
}
Write-Host "P13 prompt: ALL PASS" -ForegroundColor Green
exit 0
