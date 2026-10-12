# P12 verification: the engine can actually LOAD AND RUN MiniCPM5-1B.
#
# ASCII-ONLY (see verify_all.ps1 header for why).
#
# WHAT THIS PROVES
#   1. bf16 weights load and compute correctly (the blocker: the engine used
#      to read every constant as f32, so a bf16 weight was silent garbage).
#   2. A full 24-layer forward + final norm + lm_head matches HuggingFace
#      (magnitude within bf16 tolerance AND identical argmax).
#   3. The artifact is proportionate: only tensors the program reads.
#
# Reports SKIP (exit 0) when the generated artifacts or the model are absent,
# so the suite stays runnable without a 2.2 GB checkout.
#
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File tools/p12_minicpm_full_verify.ps1

$ErrorActionPreference = "Continue"
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$dir  = Join-Path $root "examples/minicpm_full"

$script:fails = 0
function Check($name, $cond, $detail) {
    if ($cond) { Write-Host ("  " + $name + ": PASS  " + $detail) }
    else { Write-Host ("  " + $name + ": FAIL  " + $detail); $script:fails++ }
}

Write-Host "===== P12 MiniCPM5-1B full model =====" -ForegroundColor Yellow

if (-not (Test-Path (Join-Path $dir "hf_logits.npy"))) {
    Write-Host "  SKIP: no HF reference (run tools/p12_hf_ref.py --full)"
    exit 0
}
if (-not (Test-Path (Join-Path $dir "full.vmb"))) {
    Write-Host "  SKIP: no assembled model (run tools/p12_build_minicpm.py then assemble)"
    exit 0
}

# ---- 1. bf16 end-to-end path (independent of the big model) ----
$bf = & python (Join-Path $root "tools/p12_bf16_e2e.py") 2>&1 | Out-String
$bfCode = $LASTEXITCODE
foreach ($line in ($bf -split "`n")) {
    if ($line -match 'PASS|FAIL|weights:|vmb:') { Write-Host ("  " + $line.Trim()) }
}
Check "bf16 engine path" ($bfCode -eq 0) "conversion exact, artifact half size"

# ---- 2. full 24-layer logits vs HuggingFace ----
$lg = & python (Join-Path $root "tools/p12_logits_check.py") 2>&1 | Out-String
$lgCode = $LASTEXITCODE
foreach ($line in ($lg -split "`n")) {
    if ($line -match 'magnitude|argmax|VERDICT|logits:') { Write-Host ("  " + $line.Trim()) }
}
Check "24-layer logits vs HF" ($lgCode -eq 0) "argmax match + rel within bf16 tolerance"

# ---- 3. the artifact must not carry dead tensors ----
# P13 NOTE: this check used to assert flatly that `embed` is ABSENT. That was
# correct pre-P13, when the prompt rows were copied in host-side and the 401 MB
# matrix was never read. Task A made OP_EMB dtype-aware, so the matrix is now
# READ whenever the program looks up tokens -- and in --prompt-mode tokens it
# MUST be present, because that lookup is what makes one .vmb serve any prompt.
# The real invariant is therefore "the embedding matrix is shipped iff the
# program actually reads it", which is what both branches below assert.
$argsFile = Join-Path $dir "weights.args"
$vmlFile = Join-Path $dir "full.vml"
if ((Test-Path $argsFile) -and (Test-Path $vmlFile)) {
    $lines = Get-Content $argsFile | Where-Object { $_.Trim() -ne "" }
    $hasEmbed = ($lines | Where-Object { $_ -match '^embed:' }).Count
    $vml = Get-Content $vmlFile -Raw
    $readsEmbed = ($vml -match '(?m)^\s*EMB\s')
    if ($readsEmbed) {
        Check "embed present because program reads it" ($hasEmbed -eq 1) `
              "EMB used, so the table must ship (bf16, 401 MB)"
    } else {
        Check "no dead embed tensor" ($hasEmbed -eq 0) `
              "no EMB in program, so shipping the 401 MB matrix is dead weight"
    }
}

Write-Host ""
if ($script:fails -eq 0) {
    Write-Host "P12 MiniCPM full model: ALL PASS" -ForegroundColor Green
} else {
    Write-Host ("P12 MiniCPM full model: " + $script:fails + " FAILED") -ForegroundColor Red
}
exit $script:fails
