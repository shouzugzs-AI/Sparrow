# 所有示例模型的规范汇编命令（Lead 维护）
#
# 为什么需要这个文件：
#   P8 期间因为 Ins 布局变更，所有 .vmb 必须重新汇编；但各示例的汇编参数
#   (常量名:文件:rows:cols) 此前只存在于命令行历史里，重新汇编时得从 .bin
#   文件大小反推形状，极易搞错（本次实际错了 4 次）。
#   这里把权威命令固化下来。改 .vml 或 .bin 后跑这个脚本重汇编。
#
# 用法: powershell -NoProfile -ExecutionPolicy Bypass -File tools/asm_all.ps1

$ErrorActionPreference = "Continue"
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$asm = Join-Path $root "target/release/vmlm-asm.exe"
$script:fail = 0

function Asm-Model {
    param([string]$Dir, [string]$Vml, [string]$Vmb, [string[]]$Consts)
    Write-Host "--- $Dir/$Vml ---" -ForegroundColor Cyan
    Push-Location (Join-Path $root $Dir)
    $a = @($Vml, $Vmb) + $Consts
    $out = & $asm @a 2>&1 | Out-String
    Pop-Location
    ($out -split "`n" | Where-Object { $_ -match 'OK:|警告' }) | ForEach-Object { Write-Host "  $_" }
    if (-not ($out -match 'OK:')) { $script:fail++ }
}

Write-Host "===== Reassemble all example models =====" -ForegroundColor Yellow

$attn = @(
    "emb:emb.bin:8:16", "pos:pos.bin:8:16",
    "Wqkv:Wqkv.bin:48:16", "bqkv:bqkv.bin:1:48",
    "Wproj:Wproj.bin:16:16", "bproj:bproj.bin:1:16",
    "W1:W1.bin:32:16", "b1:b1.bin:1:32",
    "W2:W2.bin:16:32", "b2:b2.bin:1:16",
    "ln1_g:ln1_g.bin:1:16", "ln1_b:ln1_b.bin:1:16",
    "ln2_g:ln2_g.bin:1:16", "ln2_b:ln2_b.bin:1:16",
    "lnf_g:lnf_g.bin:1:16", "lnf_b:lnf_b.bin:1:16",
    "E:E.bin:4:16", "tok:tok.bin:4:2", "inv_sqrt_hd:inv_sqrt_hd.bin:1:1"
)
Asm-Model "examples/attn" "block.vml" "block.vmb" $attn

# 注意：x 是「驱动按 step 覆盖的常量」，必须按 [1,32] 传入（不是省略）
$rnn = @(
    "x:x.bin:1:32",
    "W_r:W_r.bin:16:32", "W_k:W_k.bin:16:32", "W_v:W_v.bin:16:32", "W_o:W_o.bin:32:16",
    "W1d:W1d.bin:32:32", "W2d:W2d.bin:16:32", "w0:w0.bin:1:16",
    "xr:xr.bin:1:32", "xk:xk.bin:1:32", "xv:xv.bin:1:32", "xw:xw.bin:1:32",
    "ln1_g:ln1_g.bin:1:32", "ln1_b:ln1_b.bin:1:32",
    "ln_t_g:ln_t_g.bin:1:16", "ln_t_b:ln_t_b.bin:1:16",
    "ln_out_g:ln_out_g.bin:1:32", "ln_out_b:ln_out_b.bin:1:32",
    "E:E.bin:4:32"
)
Asm-Model "examples/rnn" "ese_step.vml" "ese_step.vmb" $rnn

$sparse = @(
    "x:x.bin:4:16", "W1:W1.bin:32:16", "b1:b1.bin:1:32",
    "W2:W2.bin:16:32", "b2:b2.bin:1:16"
)
Asm-Model "examples/sparse" "ffn.vml" "ffn.vmb" $sparse

# eserNN 用 weights.args 里的清单
$es = Join-Path $root "examples/eserNN"
if (Test-Path (Join-Path $es "weights.args")) {
    $c = @((Get-Content (Join-Path $es "weights.args")) | Where-Object { $_.Trim() -ne '' })
    Asm-Model "examples/eserNN" "step.vml" "step.vmb" $c
} else {
    Write-Host "  (skip eserNN: no weights.args; use tools/gen_eserNN.py)" -ForegroundColor DarkGray
}

Write-Host ""
if ($script:fail -eq 0) {
    Write-Host "All models assembled OK. Now run: tools/verify_all.ps1" -ForegroundColor Green
} else {
    Write-Host "$($script:fail) model(s) failed to assemble" -ForegroundColor Red
}
exit $script:fail
