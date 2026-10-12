# B1 layers x k0 information-loss experiment -- run one phase. ASCII-only.
#
# Fixed budget per config: 128 steps, seq=32, batch=4 (128 tok/step), dim=512.
# Grid: layers in {1,4,8,16,24} x k0 in {2,5,10,20,40,80}, 2 seeds each.
# Dense control: same LAYERS, per-layer MACs matched => d' = round(sqrt(dim*k0)).
#
# The task's execution order is explicit, so phases are separate:
#   -Phase pilot   layers in {1,4,16} x 6 k0 x 1 seed   (does the hypothesis hold at all?)
#   -Phase grid    the full 5 x 6 x 2 matrix (+dense)
#   -Phase dense   only the dense controls
#   -Phase h3      dim=256 x k0 in {1,2,5,10,20,40} x layers in {4,16} (+dense), to test
#                  whether the loss tracks k0/dim rather than k0
#
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File tools/b1_layers_k0_run.ps1 -Phase pilot

param([string]$Phase = "pilot")

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $root
$exe = Join-Path $root "target/release/vmlm-b1-scaling.exe"
$data = Join-Path $root "artifacts/b1_scaling"
$out = Join-Path $data "layers_results.jsonl"

$LAYERS = @(1, 4, 8, 16, 24)
$K0 = @(2, 5, 10, 20, 40, 80)
$LAYERS_H3 = @(4, 16)
$K0_H3 = @(1, 2, 5, 10, 20, 40)

function Run-One([string]$mode, [int]$dim, [int]$k0, [int]$layers, [int]$seed, [string]$tag) {
    $a = @("--mode", $mode, "--layers", "$layers", "--dim", "$dim", "--k0", "$k0", "--seed", "$seed",
           "--steps", "128", "--seq", "32", "--batch", "4",
           "--vocab-mode", "observed", "--data-dir", $data, "--out", $out)
    $t0 = Get-Date
    $o = & $exe @a 2>&1
    $code = $LASTEXITCODE
    $wall = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
    $line = ($o | Where-Object { $_ -like "RESULT *" } | Select-Object -First 1)
    if (-not $line) {
        Write-Host ("  [{0}] mode={1} L={2} k0={3} seed={4} -> NO RESULT (exit={5})" -f $tag, $mode, $layers, $k0, $seed, $code)
        return
    }
    $j = ($line.Substring(7) | ConvertFrom-Json)
    Write-Host ("  [{0}] {1} L={2,2} k0={3,3} d'={4,3} seed={5} -> train={6:N4} val={7:N4} ppl={8,9:N1} ({9}s)" -f `
        $tag, $mode, $layers, $k0, $j.dprime, $seed, $j.final_train, $j.final_val, $j.ppl_val, $wall)
}

Write-Host "===== B1 layers x k0: phase $Phase ====="
Write-Host "results -> $out"

if ($Phase -eq "pilot") {
    Write-Host "--- pilot: layers in {1,4,16} x k0 in {$($K0 -join ',')}, seed 1 ---"
    foreach ($L in 1, 4, 16) { foreach ($k in $K0) { Run-One "b1" 512 $k $L 1 "pilot" } }
} elseif ($Phase -eq "grid") {
    Write-Host "--- full grid: layers {$($LAYERS -join ',')} x k0 {$($K0 -join ',')} x 2 seeds ---"
    foreach ($L in $LAYERS) { foreach ($k in $K0) { foreach ($s in 1, 2) { Run-One "b1" 512 $k $L $s "grid" } } }
} elseif ($Phase -eq "dense") {
    Write-Host "--- dense controls: same layers, d' = round(sqrt(512*k0)) ---"
    foreach ($L in $LAYERS) { foreach ($k in $K0) { foreach ($s in 1, 2) { Run-One "dense" 512 $k $L $s "dense" } } }
} elseif ($Phase -eq "h3") {
    Write-Host "--- h3: dim=256, k0 in {$($K0_H3 -join ',')}, layers in {$($LAYERS_H3 -join ',')}, 2 seeds ---"
    foreach ($L in $LAYERS_H3) { foreach ($k in $K0_H3) { foreach ($s in 1, 2) { Run-One "b1" 256 $k $L $s "h3" } } }
    foreach ($L in $LAYERS_H3) { foreach ($k in $K0_H3) { foreach ($s in 1, 2) { Run-One "dense" 256 $k $L $s "h3-dense" } } }
} elseif ($Phase -eq "nores") {
    # 机制对照：去掉残差（h = y_l）。信息必须逐层穿过 k0 条连接。
    Write-Host "--- no-residual probe: layers in {16,24} x k0 in {2,20,80}, 2 seeds ---"
    foreach ($L in 16, 24) { foreach ($k in 2, 20, 80) { foreach ($s in 1, 2) {
        $a = @("--mode", "b1", "--layers", "$L", "--no-residual", "1", "--dim", "512", "--k0", "$k",
               "--seed", "$s", "--steps", "128", "--seq", "32", "--batch", "4",
               "--vocab-mode", "observed", "--data-dir", $data, "--out", $out)
        $o = & $exe @a 2>&1
        $line = ($o | Where-Object { $_ -like "RESULT *" } | Select-Object -First 1)
        if ($line) {
            $j = ($line.Substring(7) | ConvertFrom-Json)
            Write-Host ("  [nores] L={0,2} k0={1,3} seed={2} -> train={3:N4} val={4:N4} diverged={5}@{6}" -f $L, $k, $s, $j.final_train, $j.final_val, $j.diverged, $j.diverged_at)
        } else { Write-Host ("  [nores] L={0} k0={1} seed={2} -> NO RESULT" -f $L, $k, $s) }
    } } }
} elseif ($Phase -eq "nores-ds1") {
    # 无残差 + **不做**深度缩放（每层保方差，普通 MLP 堆叠的标准初始化）。
    # 为什么需要这一组：残差的 1/sqrt(2L) 缩放是为**残差**堆叠设计的；
    # 把它用在无残差堆叠上会让信号按 (3/(2L))^L 速度衰减，那就分不清
    # 「去掉恒等路径」与「缩放用错」两个因素。
    Write-Host "--- no-residual + depth_scale=1: layers in {16,24} x k0 in {2,20,80}, 2 seeds ---"
    foreach ($L in 16, 24) { foreach ($k in 2, 20, 80) { foreach ($s in 1, 2) {
        $a = @("--mode", "b1", "--layers", "$L", "--no-residual", "1", "--depth-scale", "1",
               "--dim", "512", "--k0", "$k", "--seed", "$s", "--steps", "128", "--seq", "32", "--batch", "4",
               "--vocab-mode", "observed", "--data-dir", $data, "--out", $out)
        $t0 = Get-Date
        $o = & $exe @a 2>&1
        $wall = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
        $line = ($o | Where-Object { $_ -like "RESULT *" } | Select-Object -First 1)
        if ($line) {
            $j = ($line.Substring(7) | ConvertFrom-Json)
            Write-Host ("  [nores-ds1] L={0,2} k0={1,3} seed={2} -> train={3:N4} val={4:N4} diverged={5}@{6} ({7}s)" -f $L, $k, $s, $j.final_train, $j.final_val, $j.diverged, $j.diverged_at, $wall)
        } else { Write-Host ("  [nores-ds1] L={0} k0={1} seed={2} -> NO RESULT" -f $L, $k, $s) }
    } } }
} elseif ($Phase -eq "all") {
    foreach ($L in $LAYERS) { foreach ($k in $K0) { foreach ($s in 1, 2) { Run-One "b1" 512 $k $L $s "grid" } } }
    foreach ($L in $LAYERS) { foreach ($k in $K0) { foreach ($s in 1, 2) { Run-One "dense" 512 $k $L $s "dense" } } }
    foreach ($L in $LAYERS_H3) { foreach ($k in $K0_H3) { foreach ($s in 1, 2) { Run-One "b1" 256 $k $L $s "h3" } } }
    foreach ($L in $LAYERS_H3) { foreach ($k in $K0_H3) { foreach ($s in 1, 2) { Run-One "dense" 256 $k $L $s "h3-dense" } } }
    Write-Host "--- no-residual probe ---"
    foreach ($L in 16, 24) { foreach ($k in 2, 20, 80) { foreach ($s in 1, 2) {
        $a = @("--mode", "b1", "--layers", "$L", "--no-residual", "1", "--dim", "512", "--k0", "$k",
               "--seed", "$s", "--steps", "128", "--seq", "32", "--batch", "4",
               "--vocab-mode", "observed", "--data-dir", $data, "--out", $out)
        $o = & $exe @a 2>&1
        $line = ($o | Where-Object { $_ -like "RESULT *" } | Select-Object -First 1)
        if ($line) {
            $j = ($line.Substring(7) | ConvertFrom-Json)
            Write-Host ("  [nores] L={0,2} k0={1,3} seed={2} -> train={3:N4} val={4:N4} diverged={5}@{6}" -f $L, $k, $s, $j.final_train, $j.final_val, $j.diverged, $j.diverged_at)
        } else { Write-Host ("  [nores] L={0} k0={1} seed={2} -> NO RESULT" -f $L, $k, $s) }
    } } }
}

$n = 0
if (Test-Path $out) { $n = (Get-Content $out | Measure-Object -Line).Lines }
Write-Host ("phase {0} done; layers_results.jsonl now has {1} lines" -f $Phase, $n)
