# Sparrow V0.1.3 —— 注意力块验收驱动
#
# 汇编器里**没有** `ATTEN`/`HEAD` 这类助记符，所以注意力是我们自己用
# `GEMM/SLICE/TRANSPOSE/SCALE/SOFTMAX/CONCAT` 拼出来的。判据分两层：
#
#   A. 恒等式（逐字符）：**序列=1** 时 softmax 只有一个元素参与 ⇒ 恒等于 1 ⇒
#      注意力输出 = Wo·(Wv·h + bWv) + bWo，其中 h 是注意力的输入。
#      这正好是"三层线性串联"。拆头/转置/拼接都是纯数据搬运 ⇒ 两条路
#      **必须逐字符相同**。切片偏移错、头顺序反、转置接错都会立刻打破它。
#
#   B. 因果掩码：**序列=4** 时第 0 行只能看第 0 个 token ⇒ 输出的第 0 行
#      必须与 A 的 T=1 结果**逐字符相同**；而 `因果=否` 时必须**明显不同**
#      （can-fail：证明 A 的判据真的能被打破，不是恒真）。
#
# 用法: pwsh -NoProfile -ExecutionPolicy Bypass -File tools/attn_ab.ps1

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
$bins = Join-Path $root "examples\attn_bins"
$out  = Join-Path $root "out\attn"

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
function Build($name, $extra = @()) {
    $vmb = Join-Path $out "$name.vmb"
    $r = & $spar (Join-Path $root "examples\attn_$name.sparrow") --asm $asm --consts-dir $bins @extra -o $vmb 2>&1 | Out-String
    if ($r -notmatch 'OK:') { Write-Host $r; throw "汇编失败: $name" }
    return $vmb
}
function FirstN($s, $n) { return (($s -split '\s+')[0..($n - 1)] -join ' ') }

Write-Host "===== 0. 权重 =====" -ForegroundColor Yellow
python (Join-Path $root "tools\gen_attn.py") | ForEach-Object { if ($_ -notmatch '^\s*$') { Write-Host "  $_" } }

Write-Host "`n===== A. 恒等式：注意力(T=1) == 三层线性串联 =====" -ForegroundColor Yellow
$t1 = RunModel (Build "t1")
$rf = RunModel (Build "ref")
Write-Host "  注意力  : $($t1[0])"
Write-Host "  三层线性: $($rf[0])"
Check "A 逐字符相同" ($t1[0] -ceq $rf[0] -and $t1[1] -eq 0) "拆头/转置/拼接/输出投影全部接对"

Write-Host "`n===== B. 因果掩码（序列=4） =====" -ForegroundColor Yellow
$ca = RunModel (Build "causal")
$nc = RunModel (Build "nocausal")
$ca0 = FirstN $ca[0] 4
$nc0 = FirstN $nc[0] 4
Write-Host "  因果=是 第0行: $ca0"
Write-Host "  因果=否 第0行: $nc0"
Check "B1 因果时第0行 == T=1" ($ca0 -ceq $t1[0]) "只让第 0 个 token 参与 ⇒ 与单 token 完全一致"
Check "B2 can-fail：不掩码时第0行不同" ($nc0 -cne $t1[0]) "4 个 token 都参与 ⇒ 结果不同（判据非恒真）"
# 因果时第 3 行应当与"前 4 个 token 全参与"的第 3 行相同 ⇒ 与不掩码版的第 3 行一致
$ca3 = (($ca[0] -split '\s+')[12..15] -join ' ')
$nc3 = (($nc[0] -split '\s+')[12..15] -join ' ')
Check "B3 因果时最后一行 == 不掩码的最后一行" ($ca3 -ceq $nc3) "第 3 行本来就能看全部 4 个 token ⇒ 掩码对它无影响"

Write-Host "`n===== C. 循环化：注意力放进 `重复2次` =====" -ForegroundColor Yellow
$lp = Build "loop" @("--loop-repeat")
$un = Build "loop"
$a = RunModel $lp; $b = RunModel $un
Write-Host "  循环版: $($a[0])"
Write-Host "  展开版: $($b[0])"
Check "C 循环版 == 展开版" ($a[0] -ceq $b[0]) "每份实例的缩放常量都按实例名复制到位"

Write-Host "`n===== D. 语义边界 =====" -ForegroundColor Yellow
$cases = @(
    @{ tag = "D1 维度不能被头数整除"; src = "模型 bad:`n  线性 输入=6 输出=6`n  注意力 维度=6 头=4`n  连接 线性 注意力`n  前向 线性 注意力`n"; want = "E108" },
    @{ tag = "D2 缺头数";             src = "模型 bad:`n  线性 输入=4 输出=4`n  注意力 维度=4`n  连接 线性 注意力`n  前向 线性 注意力`n"; want = "E105" },
    @{ tag = "D3 因果值不是真假词";   src = "模型 bad:`n  线性 输入=4 输出=4`n  注意力 维度=4 头=2 因果=或许`n  连接 线性 注意力`n  前向 线性 注意力`n"; want = "E108" },
    # D4 原来是「序列>1 且链首是嵌入 ⇒ E107」。**V0.1.5 起这条合法**（逐行查表 + 拼行，
    # 见 REPORT_V015 §5）：这里改成验"能编出来，且链首确实展开成 4 次 EMB"。
    @{ tag = "D5 `稀疏` 在 序列>1 下必须拒绝"; src = "模型 bad:`n  线性 输入=4 输出=4 序列=4`n  稀疏 维度=4 连接=2`n  连接 线性 稀疏`n  前向 线性 稀疏`n"; want = "E107" },
    @{ tag = "D6 链路中间的 `嵌入` 必须拒绝"; src = "模型 bad:`n  线性 输入=4 输出=4 序列=4`n  嵌入 词表=8 维度=4`n  连接 线性 嵌入`n  前向 线性 嵌入`n"; want = "E107" }
)
foreach ($c in $cases) {
    $f = Join-Path $out "_bad.sparrow"
    Set-Content -Path $f -Value $c.src -Encoding utf8
    # `-o` 必须给：否则前端把"自己产出的极小常量"（ATTN_scale.bin 之类）写到**进程 CWD**，
    # 驱动会把工作区弄脏（实测踩过，见 TRAPS A15）。
    $e = & $spar $f --backend vml -o (Join-Path $out "_bad.vml") 2>&1 | Out-String
    Check $c.tag ($e -match $c.want) "$([regex]::Match($e, "$($c.want)\][^\n]*").Value)"
}
# D4（正向）：序列>1 + 嵌入链首 ⇒ 现在应当编得出来，且链首展开成 4 次 EMB
$f4 = Join-Path $out "_seq4.sparrow"
Set-Content -Path $f4 -Value "模型 bad:`n  嵌入 词表=8 维度=4 序列=4`n  注意力 维度=4 头=2 序列=4`n  连接 嵌入 注意力`n  前向 嵌入 注意力`n" -Encoding utf8
$v4 = Join-Path $out "_seq4.vml"
$null = & $spar $f4 --backend vml -o $v4 2>&1 | Out-String
$embCount = if (Test-Path $v4) { ([regex]::Matches((Get-Content $v4 -Raw), '(?m)^\s*EMB\s+')).Count } else { -1 }
Check "D4 序列>1 + 嵌入链首（V0.1.5 起合法）" ($embCount -eq 4) "链首展开成 $embCount 次 EMB"

Write-Host "`n===== 结果 =====" -ForegroundColor Yellow
if ($script:fails -eq 0) { Write-Host "  全部 PASS" -ForegroundColor Green }
else { Write-Host "  $($script:fails) 项 FAIL" -ForegroundColor Red }
exit $script:fails
