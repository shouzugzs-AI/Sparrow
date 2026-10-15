# Sparrow V0.1.3 —— 权重精度（bf16）验收驱动
#
# 引擎的 bf16 定义（core/src/interp.rs 的 DT_BF16 + ops/src/kernels.rs 的
# `gemm_nt_bf16`）：**bf16 = f32 的高 16 位**，核内 `(u32)h << 16` 精确加宽。
# 于是有一条**可判定**的判据：
#
#   A 组：`--weight-dtype bf16` + 真 bf16 文件（2 字节/元素）
#   B 组：默认 f32      + 同样数值但以 f32 存的文件
#   ⇒ 引擎看到的**位模式完全相同** ⇒ 输出必须**逐字符相同**
#
# 判据（每条都从产物读）：
#   1. A/B 输出逐字符相同（这就是"降精度路径接对了"的证明）
#   2. .vmb 更小（bf16 的权重与嵌入表减半）
#   3. can-fail：拿**未截断**的原 f32 权重跑一遍，输出必须**不同**
#      （否则"输出相同"这条判据对精度损失不敏感 = 恒真）
#   4. hard-fail：把 f32 长度的文件标成 bf16 ⇒ 汇编器必须拒绝
#      （元素大小搞错 = 第二个元素起全错，绝不能静默通过）
#
# 用法: pwsh -NoProfile -ExecutionPolicy Bypass -File tools/bf16_ab.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot   # 仓库根（本脚本在 <root>\tools\ 下）
# 引擎路径（V0.1.6 起两种布局都支持）：
#   ① $env:SPARROW_VMLM_ROOT 显式指定
#   ② 合并布局：本目录是引擎仓库的子目录 ⇒ **上一级**就是引擎仓库根
#   ③ 独立仓库：引擎在**同级**目录（默认名 vmlm）
$parentDir = Split-Path -Parent $root
$vmlm = if ($env:SPARROW_VMLM_ROOT) { $env:SPARROW_VMLM_ROOT }
        elseif (Test-Path (Join-Path $parentDir "ops\Cargo.toml")) { $parentDir }
        else { Join-Path $parentDir "vmlm" }
if (-not (Test-Path (Join-Path $vmlm "Cargo.toml"))) {
    Write-Host ('错误：找不到 vmlm 仓库（{0}）；用 $env:SPARROW_VMLM_ROOT 指定，或把它放在本仓库同级目录' -f $vmlm) -ForegroundColor Red
    exit 1
}
# 引擎产物（V0.1.6）：带 --target 的构建落在 target\<triple>\release，老工作区里也可能有
# target\release 的构建 —— 所以**逐个二进制**在两个目录里找（公开树 vmlmd 恰好两边各有一半）。
# 新克隆 / 合并布局下这条必须自己成立（合并时暴露过：只有老残留才找得到）。
function EngineBin([string]$name) {
    foreach ($d in @((Join-Path $vmlm "target\release"), (Join-Path $vmlm "target\x86_64-pc-windows-msvc\release"))) {
        $p = Join-Path $d $name
        if (Test-Path $p) { return $p }
    }
    Write-Host "错误：找不到引擎产物 $name。先在引擎仓库里构建：" -ForegroundColor Red
    Write-Host "  cd <引擎>/runtime ; cargo build --target x86_64-pc-windows-msvc --release" -ForegroundColor Red
    exit 1
}
$asm = EngineBin "vmlm-asm.exe"
$eng = EngineBin "vmlm-runtime.exe"
$rel = Split-Path -Parent $eng
$spar = Join-Path $root "target\debug\sparrow.exe"
$out  = Join-Path $root "out\bf16"
$src  = Join-Path $root "examples\e2e_tiny.sparrow"

New-Item -ItemType Directory -Force -Path $out | Out-Null
$script:fails = 0
function Check($name, $cond, $detail) {
    if ($cond) { Write-Host "  $name : PASS  $detail" -ForegroundColor Green }
    else { Write-Host "  $name : FAIL  $detail" -ForegroundColor Red; $script:fails++ }
}
function RunModel($vmb) {
    Copy-Item $vmb (Join-Path $rel "model.vmb") -Force
    Push-Location $rel
    try { $t = (& $eng 2>&1 | Out-String).Trim(); $c = $LASTEXITCODE } finally { Pop-Location }
    return @($t, $c)
}

Write-Host "===== 0. 生成 bf16 / 截断 f32 两组常量 =====" -ForegroundColor Yellow
python (Join-Path $root "tools\gen_bf16.py") | ForEach-Object { if ($_ -notmatch '^\s*$') { Write-Host "  $_" } }

$binBf16 = Join-Path $root "examples\bf16_bins_bf16"
$binF32  = Join-Path $root "examples\bf16_bins_f32"
$binOrig = Join-Path $root "examples\e2e_bins"
if (-not (Test-Path $binBf16)) { throw "缺 $binBf16（先跑 python tools/gen_bf16.py）" }

Write-Host "`n===== 1/2. A（bf16）vs B（同样数值的 f32） =====" -ForegroundColor Yellow
$vA = Join-Path $out "A_bf16.vmb"; $vB = Join-Path $out "B_f32.vmb"; $vC = Join-Path $out "C_orig.vmb"
$rA = & $spar $src --asm $asm --consts-dir $binBf16 --weight-dtype bf16 -o $vA 2>&1 | Out-String
$rB = & $spar $src --asm $asm --consts-dir $binF32 -o $vB 2>&1 | Out-String
$rC = & $spar $src --asm $asm --consts-dir $binOrig -o $vC 2>&1 | Out-String
if ($rA -notmatch 'OK:' -or $rB -notmatch 'OK:' -or $rC -notmatch 'OK:') { Write-Host $rA $rB $rC; throw "汇编失败" }
$a = RunModel $vA; $b = RunModel $vB; $c = RunModel $vC
Write-Host "  A(bf16)  : $($a[0])   $((Get-Item $vA).Length) B"
Write-Host "  B(f32@bf16精度): $($b[0])   $((Get-Item $vB).Length) B"
Write-Host "  C(原 f32): $($c[0])   $((Get-Item $vC).Length) B"
Check "1 A/B 输出逐字符相同" ($a[0] -ceq $b[0] -and $a[1] -eq 0) "引擎看到的位模式相同 ⇒ 结果相同"
Check "2 A 的 .vmb 更小" ((Get-Item $vA).Length -lt (Get-Item $vB).Length) "$((Get-Item $vB).Length) B → $((Get-Item $vA).Length) B"
# 两份权重文件确实减半
$e1 = (Get-Item (Join-Path $binF32 "EMBED_E.bin")).Length; $e2 = (Get-Item (Join-Path $binBf16 "EMBED_E.bin")).Length
$w1 = (Get-Item (Join-Path $binF32 "LINEAR_W.bin")).Length; $w2 = (Get-Item (Join-Path $binBf16 "LINEAR_W.bin")).Length
Check "2b 权重字节减半" ($e2 * 2 -eq $e1 -and $w2 * 2 -eq $w1) "嵌入表 ${e1}→${e2} B，线性权重 ${w1}→${w2} B"

Write-Host "`n===== 3. can-fail：未截断的原 f32 权重必须算出**不同**结果 =====" -ForegroundColor Yellow
Check "3 精度损失是可见的" ($c[0] -cne $a[0]) "原 f32 输出与 bf16 不同 ⇒ 判据 1 不是恒真"

Write-Host "`n===== 4. hard-fail：把 f32 长度的文件标成 bf16 =====" -ForegroundColor Yellow
$bad = & $spar $src --asm $asm --consts-dir $binF32 --weight-dtype bf16 -o (Join-Path $out "bad.vmb") 2>&1 | Out-String
$rejected = ($bad -match 'E201') -or ($bad -match '长度') -or ($bad -match 'panicked')
Check "4 元素大小搞错被拒绝" $rejected "$([regex]::Match($bad,'(长度|字节|panicked)[^\n]*').Value)"
Check "4b 不留下产物" (-not (Test-Path (Join-Path $out "bad.vmb"))) "拒绝后没有 .vmb"

Write-Host "`n===== 5. 与其它开关叠加 =====" -ForegroundColor Yellow
$r5 = & $spar $src --asm $asm --consts-dir $binBf16 --weight-dtype bf16 --slot-reuse -o (Join-Path $out "D_combo.vmb") 2>&1 | Out-String
if ($r5 -match 'OK:') {
    $d = RunModel (Join-Path $out "D_combo.vmb")
    Check "5 bf16 + 槽位复用 组合" ($d[0] -ceq $a[0]) "输出仍与 A 相同"
} else { Check "5 bf16 + 槽位复用 组合" $false $r5 }

Write-Host "`n===== 结果 =====" -ForegroundColor Yellow
if ($script:fails -eq 0) { Write-Host "  全部 PASS" -ForegroundColor Green }
else { Write-Host "  $($script:fails) 项 FAIL" -ForegroundColor Red }
exit $script:fails
