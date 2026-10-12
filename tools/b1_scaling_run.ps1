# B1 scaling law -- run one phase of the matrix. ASCII-only (see verify_all.ps1 header).
#
# Fixed budget per config: 128 steps, seq=32, batch=4 => 128 tok/step, 16384 tok total.
# Two seeds per configuration (as the task requires).
# Results are APPENDED as one JSON line per run to artifacts/b1_scaling/results.jsonl,
# so a killed run loses at most the config in flight and phases can be re-run.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools/b1_scaling_run.ps1 -Phase A
#   -Phase B | C | dense | all

param([string]$Phase = "A")

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $root
$exe = Join-Path $root "target/release/vmlm-b1-scaling.exe"
$data = Join-Path $root "artifacts/b1_scaling"
$out = Join-Path $data "results.jsonl"

# scan A: k0 grows, dim fixed at 512
$A = @(1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
# scan B: dim grows, k0 fixed at 20
$B = @(32, 64, 128, 256, 512)
# scan C: total connections fixed at 10240 = dim * k0
$C = @(@(64, 160), @(128, 80), @(256, 40), @(512, 20), @(1024, 10))

function Run-One([string]$mode, [int]$dim, [int]$k0, [int]$seed, [string]$tag) {
    $args = @("--mode", $mode, "--dim", "$dim", "--k0", "$k0", "--seed", "$seed",
              "--steps", "128", "--seq", "32", "--batch", "4",
              "--vocab-mode", "observed", "--data-dir", $data, "--out", $out)
    $t0 = Get-Date
    $o = & $exe @args 2>&1
    $code = $LASTEXITCODE
    $wall = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
    $line = ($o | Where-Object { $_ -like "RESULT *" } | Select-Object -First 1)
    if (-not $line) {
        Write-Host ("  [{0}] mode={1} dim={2} k0={3} seed={4} -> NO RESULT (exit={5})" -f $tag, $mode, $dim, $k0, $seed, $code)
        return
    }
    $j = ($line.Substring(7) | ConvertFrom-Json)
    Write-Host ("  [{0}] mode={1} dim={2} k0={3} d'={4} seed={5} -> train={6:N4} val={7:N4} ppl_val={8:N1} ({9}s)" -f `
        $tag, $mode, $dim, $k0, $j.dprime, $seed, $j.final_train, $j.final_val, $j.ppl_val, $wall)
}

Write-Host "===== B1 scaling run: phase $Phase ====="
Write-Host "results -> $out"

if ($Phase -eq "A" -or $Phase -eq "all") {
    Write-Host "--- scan A: dim=512, k0 in {$($A -join ',')} ---"
    foreach ($k in $A) { foreach ($s in 1, 2) { Run-One "b1" 512 $k $s "A" } }
}
if ($Phase -eq "B" -or $Phase -eq "all") {
    Write-Host "--- scan B: k0=20, dim in {$($B -join ',')} ---"
    foreach ($dd in $B) { foreach ($s in 1, 2) { Run-One "b1" $dd 20 $s "B" } }
}
if ($Phase -eq "C" -or $Phase -eq "all") {
    Write-Host "--- scan C: dim*k0 = 10240 ---"
    foreach ($pair in $C) { foreach ($s in 1, 2) { Run-One "b1" $pair[0] $pair[1] $s "C" } }
}
if ($Phase -eq "dense" -or $Phase -eq "all") {
    Write-Host "--- dense control: d' = round(sqrt(dim*k0)) for every B1 config ---"
    foreach ($k in $A) { foreach ($s in 1, 2) { Run-One "dense" 512 $k $s "A-dense" } }
    foreach ($dd in $B) { foreach ($s in 1, 2) { Run-One "dense" $dd 20 $s "B-dense" } }
    foreach ($pair in $C) { foreach ($s in 1, 2) { Run-One "dense" $pair[0] $pair[1] $s "C-dense" } }
}

$n = 0
if (Test-Path $out) { $n = (Get-Content $out | Measure-Object -Line).Lines }
Write-Host ("phase {0} done; results.jsonl now has {1} lines" -f $Phase, $n)
