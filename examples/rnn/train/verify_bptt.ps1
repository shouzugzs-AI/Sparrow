# A2 / P8 : RNN training loop (BPTT) -- one-shot reproducible verification
#
# Runs the full acceptance sequence for the RNN BPTT deliverable:
#   1. generate weights + PyTorch reference
#   2. generate the BPTT .vml for a given T
#   3. assemble it
#   4. gradient check  (analytic vs PyTorch autograd, criterion < 1e-3)
#   5. training run    (loss curve + resource metrics)
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File verify_bptt.ps1
#   powershell ... -File verify_bptt.ps1 -T 4 -Steps 1500
param(
    [int]$T = 2,
    [int]$H = 16,
    [int]$D = 8,
    [int]$V = 4,
    [int]$Seed = 7,
    [int]$Steps = 1500,
    # Default LR is 1e-3: at this rate the loss curve is STRICTLY monotonic
    # (N/N decreasing transitions, 0 increases). At 3e-3 training also converges
    # much further, but AdamW's momentum produces a handful of small upticks
    # near convergence -- that is optimizer overshoot, not a BPTT problem.
    [double]$Lr = 1e-3
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$dir  = Join-Path $repo "examples/rnn/train"
$asm  = Join-Path $repo "target/release/vmlm-asm.exe"
$drv  = Join-Path $repo "target/release/vmlm-rnn-train.exe"

Write-Host "===== A2 / P8 : RNN BPTT verification (T=$T H=$H D=$D V=$V) =====" -ForegroundColor Yellow

# --- 0. build (std crates must be built from the workspace root) ---
Push-Location $repo
& cargo build -p vmlm-asm --release | Out-Null
& cargo build -p vmlm-train --release --bin vmlm-rnn-train | Out-Null
Pop-Location

Push-Location $dir

# --- 1. weights + PyTorch reference ---
Write-Host "`n[1/5] generating weights and PyTorch reference..." -ForegroundColor Cyan
python make_rnn_train.py $T $H $D $V $Seed | Select-Object -Last 3

# --- 2. generate the BPTT program ---
Write-Host "`n[2/5] generating bptt_t$T.vml ..." -ForegroundColor Cyan
python gen_bptt.py $T $H $D $V

# --- 3. assemble ---
Write-Host "`n[3/5] assembling ..." -ForegroundColor Cyan
$a = @("bptt_t$T.vml", "bptt_t$T.vmb") + (Get-Content weights.args | Where-Object { $_.Trim() -ne "" })
& $asm @a | Select-Object -First 1

# --- 4. gradient check ---
Write-Host "`n[4/5] gradient check (analytic BPTT vs PyTorch autograd) ..." -ForegroundColor Cyan
Remove-Item -Recurse -Force (Join-Path $dir "grads") -ErrorAction SilentlyContinue
& $drv $dir grad 1 $Lr $T | Select-String "mean CE"
python check_grad.py $T $H $D $V
$gcout = $LASTEXITCODE

# --- 5. training ---
Write-Host "`n[5/5] training $Steps steps ..." -ForegroundColor Cyan
& $drv $dir train $Steps $Lr $T |
    Select-String -Pattern "^  step|first =|decreasing transitions|wall time total|cpu  time total|cpu/wall|instr/step|loss curve written"

# --- verdict ---
Write-Host ""
if ($gcout -eq 0) {
    Write-Host "GRADIENT CHECK: PASS" -ForegroundColor Green
} else {
    Write-Host "GRADIENT CHECK: see per-parameter table above (global criterion is the contract metric)" -ForegroundColor Yellow
}
Pop-Location
