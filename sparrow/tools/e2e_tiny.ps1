# Sparrow V0.1.2 —— 端到端驱动（.sparrow → .spar → .vml → .vmb → 引擎）
#
# 三条路径全部验到：
#   P1 一条命令（--asm，前端 fork 汇编器）
#   P2 手工两步（--backend vml，再单独跑汇编器）—— 与 P1 必须产出同一个 .vmb
#   P3 手写参考 .vml（稀疏层走稠密 GEMM）—— 引擎输出必须与 P1/P2 逐字节相同
#
# 不改汇编器、不改引擎：本脚本只调用它们。
# 用法: pwsh -NoProfile -ExecutionPolicy Bypass -File tools/e2e_tiny.ps1

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
$bins = Join-Path $root "examples\e2e_bins"
$spar = Join-Path $root "target\debug\sparrow.exe"
$src  = Join-Path $root "examples\e2e_tiny.sparrow"
$outdir = Join-Path $root "out"

New-Item -ItemType Directory -Force -Path $outdir | Out-Null
$script:fails = 0
function Check($name, $cond, $detail) {
    if ($cond) { Write-Host "  $name : PASS  $detail" -ForegroundColor Green }
    else { Write-Host "  $name : FAIL  $detail" -ForegroundColor Red; $script:fails++ }
}

Write-Host "===== 1. 常量（确定性 seed=7） =====" -ForegroundColor Yellow
python (Join-Path $root "tools\gen_e2e_tiny.py") | ForEach-Object { if ($_ -notmatch '^\s*$') { Write-Host "  $_" } }

Write-Host "`n===== P1. 一条命令：--asm 让前端 fork 汇编器 =====" -ForegroundColor Yellow
$p1_vmb = Join-Path $bins "e2e_tiny.vmb"
foreach ($ext in "vmb","vml","spar","args") { Remove-Item (Join-Path $bins "e2e_tiny.$ext") -ErrorAction SilentlyContinue }
$o1 = & $spar $src --asm $asm --consts-dir $bins -o $p1_vmb 2>&1 | Out-String
$c1 = $LASTEXITCODE
$o1.TrimEnd() -split "`n" | ForEach-Object { Write-Host "  $_" }
Check "A1 一条命令退出码 0" ($c1 -eq 0) "exit=$c1"
Check "A1 产出 .vmb" (Test-Path $p1_vmb) "$((Get-Item $p1_vmb).Length) B"
foreach ($ext in "spar","vml","args") {
    Check "A1 中间产物 .$ext 落盘" (Test-Path (Join-Path $bins "e2e_tiny.$ext")) ""
}

Write-Host "`n===== P2. 手工两步（--backend vml + 单独汇编） =====" -ForegroundColor Yellow
$p2_vml = Join-Path $root "examples\e2e_tiny_generated.vml"
$p2_arg = Join-Path $root "examples\e2e_tiny_generated.args"
$p2_vmb = Join-Path $root "examples\e2e_tiny_generated.vmb"
& $spar $src --backend vml -o $p2_vml --args $p2_arg
$genArgs = @((Get-Content $p2_arg) | Where-Object { $_.Trim() -ne '' })

Write-Host "`n===== P3. 手写参考 .vml（稠密等价） =====" -ForegroundColor Yellow
$p3_vml = Join-Path $root "examples\e2e_tiny.vml"
$p3_vmb = Join-Path $root "examples\e2e_tiny_ref.vmb"
$refArgs = $genArgs + @("DENSE_W:DENSE_W.bin:4:4")

Write-Host "`n===== 汇编 P2 / P3（CWD = 常量目录） =====" -ForegroundColor Yellow
Push-Location $bins
try {
    $a2 = & $asm $p2_vml $p2_vmb @genArgs 2>&1 | Out-String
    $a3 = & $asm $p3_vml $p3_vmb @refArgs 2>&1 | Out-String
} finally { Pop-Location }
Write-Host "  P2: $(((($a2 -split "`n") | Where-Object { $_ -match 'OK|panicked|错误' }) | Select-Object -First 1))"
Write-Host "  P3: $(((($a3 -split "`n") | Where-Object { $_ -match 'OK|panicked|错误' }) | Select-Object -First 1))"
Check "A2 P2 汇编成功" ($a2 -match 'OK') ""
Check "A2 P3 汇编成功" ($a3 -match 'OK') ""

Write-Host "`n===== 引擎运行（引擎写死读 model.vmb，CWD 必须是 target\release） =====" -ForegroundColor Yellow
function Run-Model($vmb, $tag) {
    Copy-Item $vmb (Join-Path $rel "model.vmb") -Force
    Push-Location $rel
    try { $text = & $eng 2>&1 | Out-String; $code = $LASTEXITCODE } finally { Pop-Location }
    $p = Join-Path $outdir "e2e_tiny_$tag.txt"
    $text | Set-Content -Path $p -Encoding ascii -NoNewline
    return @{ code = $code; text = $text; path = $p }
}
$r1 = Run-Model $p1_vmb "p1_asm"
$r2 = Run-Model $p2_vmb "p2_manual"
$r3 = Run-Model $p3_vmb "p3_handwritten"
foreach ($r in @($r1, $r2, $r3)) { Write-Host "  exit=$($r.code)  $($r.text.TrimEnd())" }

Write-Host "`n===== 逐字节比较 =====" -ForegroundColor Yellow
$h1 = (Get-FileHash $r1.path -Algorithm SHA256).Hash
$h2 = (Get-FileHash $r2.path -Algorithm SHA256).Hash
$h3 = (Get-FileHash $r3.path -Algorithm SHA256).Hash
$b1 = [System.IO.File]::ReadAllBytes($r1.path)
$b3 = [System.IO.File]::ReadAllBytes($r3.path)
$diff13 = if ($b1.Length -eq $b3.Length) { (@(0..([Math]::Max(0, $b1.Length - 1))) | Where-Object { $b1[$_] -ne $b3[$_] }).Count } else { -1 }
Write-Host "  P1 (--asm)     sha256=$h1  ($($b1.Length) B)"
Write-Host "  P2 (手工两步)  sha256=$h2"
Write-Host "  P3 (手写参考)  sha256=$h3  ($($b3.Length) B)"
Check "B1 三个引擎退出码都是 0" (($r1.code -eq 0) -and ($r2.code -eq 0) -and ($r3.code -eq 0)) "$($r1.code)/$($r2.code)/$($r3.code)"
Check "B2 P1 与 P3 输出逐字节相同" ($diff13 -eq 0) "差异字节数=$diff13"
Check "B3 三者 sha256 全相同" (($h1 -eq $h2) -and ($h1 -eq $h3)) ""

Write-Host "`n===== 汇编器报错时的转发（验收 3） =====" -ForegroundColor Yellow
$bad = Join-Path $outdir "_bad.vml"
"SHAPE x, 1, 4`nFROBNICATE x, y`nHALT" | Set-Content -Path $bad -Encoding ascii
Push-Location $bins
try { $ba = & $asm $bad (Join-Path $outdir "_bad.vmb") 2>&1 | Out-String; $bc = $LASTEXITCODE } finally { Pop-Location }
Write-Host "  汇编器直调 exit=$bc"
Write-Host "  $(((($ba -split "`n") | Where-Object { $_ -match '未知指令' }) | Select-Object -First 1).Trim())"
Check "C1 坏 .vml 的退出码不是 0（且装不进 u8）" ($bc -ne 0) "exit=$bc"
Check "C2 坏 .vml 不产出 .vmb" (-not (Test-Path (Join-Path $outdir "_bad.vmb"))) ""
Write-Host "  （前端侧的转发由 tests/asm_fork.rs 覆盖：同一 asmdriver::run 调用）"

Write-Host ""
if ($script:fails -eq 0) { Write-Host "端到端 PASS（$($b1.Length) 字节输出，P1/P2/P3 三方逐字节一致）" -ForegroundColor Green }
else { Write-Host "端到端 FAIL：$($script:fails) 项" -ForegroundColor Red }
exit $script:fails
