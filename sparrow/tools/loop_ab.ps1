# Sparrow V0.1.2 P0-2 —— 循环化（LOOP_CNT + LAYERB）验收驱动
#
# 验收项（每项都带 can-fail，避免"判据恒真"）：
#   1. O(1) 指令流：24 层的指令数 == 6 层的指令数（循环发射与 N 无关）
#   2. 池不变：循环版的 .args 以展开版的 .args 为**前缀**（多出的 3 条是循环控制常量）
#   3. 逐字节等价：6 层（展开版能汇编的最大规模之一）循环版 vs 展开版，引擎输出**逐字节相同**
#   4. can-fail A：把步长 +64 / +192 后重汇编，引擎输出**必须变化**
#   5. can-fail B：用"假汇编器"产出一个常量池不连续的 .vmb，前端**必须**报 E210 并删产物
#   6. 展开版的天花板：24 层展开版**汇编不出来**（SPARSE_MATVEC 权重槽号 ≤ 63）——
#      这条是"循环化不只是优化，而是 ≥21 层的唯一编译路径"
#
# 不改汇编器、不改引擎：本脚本只调用它们。
# 用法: pwsh -NoProfile -ExecutionPolicy Bypass -File tools/loop_ab.ps1

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
$bins  = Join-Path $root "examples\loop_bins"
$spar  = Join-Path $root "target\debug\sparrow.exe"
$out   = Join-Path $root "out\loop"
$m24   = Join-Path $root "examples\loop24.sparrow"
$m6    = Join-Path $root "examples\loop6.sparrow"

New-Item -ItemType Directory -Force -Path $out | Out-Null
$script:fails = 0
function Check($name, $cond, $detail) {
    if ($cond) { Write-Host "  $name : PASS  $detail" -ForegroundColor Green }
    else { Write-Host "  $name : FAIL  $detail" -ForegroundColor Red; $script:fails++ }
}
function NIns($text) {
    $m = [regex]::Match($text, 'OK:\s*(\d+)\s*条指令')
    if ($m.Success) { return [int]$m.Groups[1].Value } else { return -1 }
}
function RunModel($vmb) {
    Copy-Item $vmb (Join-Path $rel "model.vmb") -Force
    Push-Location $rel
    try { $t = & $eng 2>&1 | Out-String; $c = $LASTEXITCODE } finally { Pop-Location }
    return @($t.Trim(), $c)
}
function Asm1($vml, $vmb, $argfile) {
    $a = (Get-Content $argfile) -join ' '
    Push-Location $bins
    try { $t = & $asm $vml $vmb @($a -split ' ') 2>&1 | Out-String } finally { Pop-Location }
    return $t
}

Write-Host "===== 0. 常量（24 层各一个不同 seed） =====" -ForegroundColor Yellow
python (Join-Path $root "tools\gen_loop24.py") | ForEach-Object { if ($_ -notmatch '^\s*$') { Write-Host "  $_" } }

Write-Host "`n===== 1/2/3. 汇编两条路（循环 / 展开） =====" -ForegroundColor Yellow
$l24 = & $spar $m24 --asm $asm --consts-dir $bins --loop-repeat -o (Join-Path $out "loop24.vmb") 2>&1 | Out-String
$u24 = & $spar $m24 --asm $asm --consts-dir $bins              -o (Join-Path $out "loop24_unrolled.vmb") 2>&1 | Out-String
$l6  = & $spar $m6  --asm $asm --consts-dir $bins --loop-repeat -o (Join-Path $out "loop6.vmb") 2>&1 | Out-String
$u6  = & $spar $m6  --asm $asm --consts-dir $bins              -o (Join-Path $out "loop6_unrolled.vmb") 2>&1 | Out-String
$n24 = NIns $l24; $n6 = NIns $l6; $nu6 = NIns $u6
Write-Host "  循环 24 层: $n24 条指令 | 循环 6 层: $n6 条指令 | 展开 6 层: $nu6 条指令"
Write-Host "  展开 24 层: 汇编失败（预期）" -ForegroundColor DarkGray
Check "1 O(1) 指令流" ($n24 -eq $n6 -and $n24 -gt 0) "24 层 $n24 条 == 6 层 $n6 条（循环发射与 N 无关）"
Check "1b 展开是 O(N)" ($nu6 -gt $n6) "展开 6 层 $nu6 条 > 循环 6 层 $n6 条（每层 3 条）"
Check "1c 24 层循环版汇编成功" ($l24 -match 'OK:') "24 层展开版装不出来（见验收 6）"

# 池不变：循环版 .args 的前缀 == 展开版 .args
& $spar $m24 --backend vml --loop-repeat --args (Join-Path $out "loop24.args") -o (Join-Path $out "loop24.vml") | Out-Null
& $spar $m24 --backend vml            --args (Join-Path $out "loop24_unrolled.args") -o (Join-Path $out "loop24_unrolled.vml") | Out-Null
$la = Get-Content (Join-Path $out "loop24.args")
$ua = Get-Content (Join-Path $out "loop24_unrolled.args")
$prefix_ok = ($la.Count -eq $ua.Count + 3) -and (@(Compare-Object $ua $la[0..($ua.Count-1)] -SyncWindow 0).Count -eq 0)
Check "2 常量池不变" $prefix_ok "循环版 $($la.Count) 条 = 展开版 $($ua.Count) 条 + 3 条循环控制（前缀逐行相同）"

Write-Host "`n===== 3. 逐字节等价（6 层，24 层装不出来） =====" -ForegroundColor Yellow
$rl = RunModel (Join-Path $out "loop6.vmb")
$ru = RunModel (Join-Path $out "loop6_unrolled.vmb")
Write-Host "  循环版: $($rl[0])   (exit $($rl[1]))"
Write-Host "  展开版: $($ru[0])   (exit $($ru[1]))"
Check "3 引擎输出逐字节相同" ($rl[0] -ceq $ru[0] -and $rl[1] -eq 0) "两条指令流同一函数"

Write-Host "`n===== 4. can-fail：步长错一点，输出必须变 =====" -ForegroundColor Yellow
$lstride = Join-Path $bins "t__1_lstride.bin"
function SetStride($v) {
    python -c "import struct,sys; open(sys.argv[1],'wb').write(struct.pack('<f',float(sys.argv[2])))" $lstride $v
}
foreach ($delta in @(64, 192, -64)) {
    $v = 192 + $delta
    SetStride $v
    $bad = Asm1 (Join-Path $out "loop6.vml") (Join-Path $out "loop6_bad.vmb") (Join-Path $out "loop6.args")
    $rb = RunModel (Join-Path $out "loop6_bad.vmb")
    $changed = ($rb[0] -cne $rl[0]) -or ($rb[1] -ne 0)
    Check "4 can-fail 步长 $delta" $changed "步长 $v B → 输出/退出码变化：exit=$($rb[1]) 输出=`"$($rb[0])`""
}
SetStride 192
$rfix = RunModel (Join-Path $out "loop6.vmb")
Check "4d 恢复步长后回到原值" ($rfix[0] -ceq $rl[0]) "步长还原 192 B → 输出回到 `"$($rfix[0])`""

Write-Host "`n===== 5. can-fail E210：常量池不连续必须被拦下 =====" -ForegroundColor Yellow
$fake = Join-Path $root "target\debug\examples\asm_fault_inject.exe"
if (-not (Test-Path $fake)) { & cargo build --example asm_fault_inject --manifest-path (Join-Path $root "Cargo.toml") | Out-Null }
$env:SPARROW_FAULT_SRC = Join-Path $out "loop6.vmb"
$bad_out = Join-Path $out "loop6_e210.vmb"
Remove-Item $bad_out -ErrorAction SilentlyContinue
# 故障 A：把第 5 条常量（t__2_off）的池偏移 +64 ⇒ 步长不再恒定
$env:SPARROW_FAULT = "shift_const:5:64"
$e210 = & $spar $m6 --asm $fake --consts-dir $bins --loop-repeat -o $bad_out 2>&1 | Out-String
Check "5 E210 步长不恒定" ($e210 -match 'E210' -and $e210 -match '步长') "$([regex]::Match($e210,'E210[^\n]*').Value)"
Check "5b 不留下不可信产物" (-not (Test-Path $bad_out)) "对账失败后 .vmb 已删除"
# 故障 B：谎报 n_const ⇒ "池序 = 清单序" 这条也必须是硬的
$env:SPARROW_FAULT = "drop_const:24"
$e210b = & $spar $m6 --asm $fake --consts-dir $bins --loop-repeat -o $bad_out 2>&1 | Out-String
Check "5c E210 条数不符" ($e210b -match 'E210') "$([regex]::Match($e210b,'E210[^\n]*').Value)"
Remove-Item Env:SPARROW_FAULT, Env:SPARROW_FAULT_SRC -ErrorAction SilentlyContinue

Write-Host "`n===== 6. 展开版的天花板（≥21 层装不出来） =====" -ForegroundColor Yellow
Check "6 展开 24 层被汇编器拒绝" ($u24 -notmatch 'OK:' -and $u24 -match '权重槽号') "原因：SPARSE_MATVEC 权重槽压进 fl 低 6 位，槽号 ≤ 63"

Write-Host "`n===== 结果 =====" -ForegroundColor Yellow
if ($script:fails -eq 0) { Write-Host "  全部 PASS" -ForegroundColor Green }
else { Write-Host "  $($script:fails) 项 FAIL" -ForegroundColor Red }
exit $script:fails
