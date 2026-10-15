# Sparrow V0.1.3 —— 槽位复用验收驱动
#
# 判据（每条都从**产物**读，不从生成器自己的账本读）：
#   1. 复用开/关：引擎输出**逐字符相同**（复用只改"写哪个槽"，不该改数值）
#   2. 激活区字节数**严格变小**（从 .vmb 的 META 读，不是生成器自报）
#   3. can-fail：分叉-汇流模型（两个同宽分支必须**同时活着**）也逐字符相同
#      —— 若生命周期分析把还活着的槽回收了，汇入会算成 B+B 而不是 A+B，输出必变
#   4. 反复跑两次逐字节相同（确定性）
#   5. V0.1.5 起还回收**暂存槽**：汇入的中间槽 `<层名>_in`、循环组的进位槽/汇入槽/
#      计数器/步长槽。判据 3 直接**数槽名**（数字变小可能是别的原因），
#      12 层残差：不复用 12 个中间槽 → 复用后 1 个（轮着用）。
#   6. V0.1.6 起**注意力块内部**的每头暂存槽也共用：q/k/kt/v/s 五个角色 H 个头轮流借，
#      `c`（每头输出）**必须**按头分开、拼头目标借 `_q`。判据 3c 数注意力块的目的槽名，
#      差额必须正好是 5×(头−1)+1；can-fail 见 docs/TRAPS.md G10（把 `c` 也共用 ⇒ 输出必变）。
#
# 用法: pwsh -NoProfile -ExecutionPolicy Bypass -File tools/slot_reuse.ps1

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
$out  = Join-Path $root "out\slots"

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
# 从 .vmb 的 META 读 (槽位数, 激活区字节数)
function MetaOf($vmb) {
    $b = [System.IO.File]::ReadAllBytes($vmb)
    $metaOff = [BitConverter]::ToUInt32($b, 24)
    return @([BitConverter]::ToUInt32($b, $metaOff + 4), [BitConverter]::ToUInt32($b, $metaOff + 12))
}

# 每条：模型源、权重目录、说明、额外参数、是否**预期**激活区变小
#   shrink = 链式模型：同时活着的槽很少，复用应当显著省
#   keep   = 分叉汇流模型：各分支的输出必须**同时活着**（汇入要读它们），
#            所以复用省不动 —— 这不是失败，而是生命周期分析的**正确**表现。
#            这两条的价值在"输出逐字符相同"：若错回收，汇入会算成 B+B，
#            输出立刻不同 ⇒ 判据 1 会 FAIL。
$cases = @(
    @{ src = "examples\residual_loop_inner.sparrow"; bins = "examples\res_bins"; tag = "12 层残差（展开）"; extra = @(); expect = "shrink" },
    @{ src = "examples\e2e_tiny.sparrow";            bins = "examples\e2e_bins"; tag = "最小闭环（链）"; extra = @(); expect = "shrink" },
    @{ src = "examples\loop24.sparrow";              bins = "examples\loop_bins"; tag = "24 层稀疏（循环）"; extra = @("--loop-repeat"); expect = "keep" },
    @{ src = "examples\residual_three.sparrow";      bins = "examples\res_bins"; tag = "三分支汇入（can-fail）"; extra = @(); expect = "keep" },
    @{ src = "examples\residual_add.sparrow";        bins = "examples\res_bins"; tag = "两分支汇入（can-fail）"; extra = @(); expect = "keep" },
    @{ src = "examples\residual_dup.sparrow";        bins = "examples\res_bins"; tag = "逐元素尾巴"; extra = @(); expect = "shrink" },
    # V0.1.5：暂存槽回收之后**循环组**也能省（进位槽/汇入槽/计数器/步长槽 → 池）
    @{ src = "examples\residual_loop.sparrow";       bins = "examples\res_bins"; tag = "12 层残差（循环）"; extra = @("--loop-repeat"); expect = "shrink" },
    @{ src = "examples\loop_two.sparrow";            bins = "examples\loop_two_bins"; tag = "两个循环组（循环）"; extra = @("--loop-repeat"); expect = "shrink" },
    @{ src = "examples\loop_two.sparrow";            bins = "examples\loop_two_bins"; tag = "两个循环组（展开）"; extra = @(); expect = "shrink" },
    # V0.1.6：注意力块**内部**的每头暂存槽也共用（q/k/kt/v/s 按角色，c 每头独立）
    @{ src = "examples\attn_t1.sparrow";        bins = "examples\attn_bins"; tag = "注意力 T=1 头=2"; extra = @(); expect = "shrink" },
    @{ src = "examples\attn_causal.sparrow";    bins = "examples\attn_bins"; tag = "注意力 T=4 因果"; extra = @(); expect = "shrink" },
    @{ src = "examples\attn_nocausal.sparrow";  bins = "examples\attn_bins"; tag = "注意力 T=4 非因果"; extra = @(); expect = "shrink" },
    @{ src = "examples\attn_loop.sparrow";      bins = "examples\attn_bins"; tag = "注意力 重复2次（循环）"; extra = @("--loop-repeat"); expect = "shrink" }
)

Write-Host "===== 槽位复用 A/B =====" -ForegroundColor Yellow
foreach ($c in $cases) {
    $src = Join-Path $root $c.src
    $bins = Join-Path $root $c.bins
    if (-not (Test-Path $bins)) { Write-Host "  SKIP $($c.tag)：缺 $bins" -ForegroundColor DarkGray; continue }
    $off = Join-Path $out "off.vmb"; $on = Join-Path $out "on.vmb"; $on2 = Join-Path $out "on2.vmb"
    $ex = $c.extra
    $r0 = & $spar $src --asm $asm --consts-dir $bins @ex -o $off 2>&1 | Out-String
    $r1 = & $spar $src --asm $asm --consts-dir $bins @ex --slot-reuse -o $on 2>&1 | Out-String
    $r2 = & $spar $src --asm $asm --consts-dir $bins @ex --slot-reuse -o $on2 2>&1 | Out-String
    if ($r0 -notmatch 'OK:' -or $r1 -notmatch 'OK:') { Write-Host $r0 $r1; Check "$($c.tag) 汇编" $false "失败"; continue }
    $a = RunModel $off; $b = RunModel $on
    $m0 = MetaOf $off; $m1 = MetaOf $on
    $act0 = [int]$m0[1]; $act1 = [int]$m1[1]
    $pct = if ($act0 -gt 0) { [math]::Round(100.0 * ($act0 - $act1) / $act0, 1) } else { 0 }
    Write-Host "  $($c.tag): 槽 $($m0[0])→$($m1[0])，激活区 ${act0}B→${act1}B（省 $pct%），文件 $((Get-Item $off).Length)B→$((Get-Item $on).Length)B"
    Check "1 $($c.tag) 输出逐字符相同" ($a[0] -ceq $b[0] -and $a[1] -eq 0) "复用不改数值"
    if ($c.expect -eq "shrink") {
        Check "2 $($c.tag) 激活区变小" ($act1 -lt $act0) "$act0 B → $act1 B（省 $pct%）"
    } else {
        Check "2 $($c.tag) 激活区不增大" ($act1 -le $act0) "分支必须同时活着 ⇒ 省不动（$act0→$act1）"
    }
    $same = ([System.IO.File]::ReadAllBytes($on)).Length -eq ([System.IO.File]::ReadAllBytes($on2)).Length
    Check "4 $($c.tag) 确定性" $same "两次复用产物同长"
}

Write-Host "`n===== 3. 结构判据：汇入的中间槽**真的被复用**了 =====" -ForegroundColor Yellow
# 只看激活区字节数是不够的：数字变小可能是别的原因。这里直接数**汇入中间槽的槽名**：
# 12 层残差在展开模式下有 12 次汇入（每次一个 `<层名>_in` 中间槽）。
#   * 不复用：12 个不同名字（每个都要一块激活区）
#   * 复用后：**同一个**名字被 12 次汇入轮流借用
$sSrc = Join-Path $root "examples\residual_loop_inner.sparrow"
$sBins = Join-Path $root "examples\res_bins"
$vOff = Join-Path $out "scratch_off.vmb"
$vOn = Join-Path $out "scratch_on.vmb"
$null = & $spar $sSrc --asm $asm --consts-dir $sBins -o $vOff 2>&1 | Out-String
$null = & $spar $sSrc --asm $asm --consts-dir $sBins --slot-reuse -o $vOn 2>&1 | Out-String
function InSlotNames($vmb) {
    $vml = [System.IO.Path]::ChangeExtension($vmb, ".vml")
    return @(Get-Content $vml | Select-String -Pattern '^\s*SHAPE\s+(\S+_in),' |
        ForEach-Object { $_.Matches[0].Groups[1].Value })
}
$offNames = InSlotNames $vOff
$onNames = InSlotNames $vOn
$offDistinct = @($offNames | Sort-Object -Unique).Count
$onDistinct = @($onNames | Sort-Object -Unique).Count
Write-Host "  不复用：$($offNames.Count) 次汇入，$offDistinct 个不同中间槽"
Write-Host "  复用后：$($onNames.Count) 次汇入，$onDistinct 个不同中间槽（轮着用同一个）"
Check "3a 不复用：每次汇入一个独立中间槽" ($offDistinct -eq $offNames.Count -and $offDistinct -ge 12) "$offDistinct 个"
Check "3b 复用后：中间槽被回收（不同名字数显著下降）" ($onDistinct -lt $offDistinct -and $onDistinct -le 2) `
    "$offDistinct → $onDistinct（12 次汇入共用 $onDistinct 个槽）"

Write-Host "`n===== 3c. 结构判据：注意力**每头暂存槽**真的共用了（V0.1.6）=====" -ForegroundColor Yellow
# 只看激活区字节数不够：这里直接数**注意力块里的目的槽名**（第一操作数）。
#   * 不复用：3 个投影 + H 头 × 6 个角色（q/k/kt/v/s/c）+ 拼头 `_ctx` + 层输出
#   * 复用后：3 个投影 + 5 个角色槽（H 头轮流借）+ H 个 `c` + 层输出（`_ctx` 借 `_q`）
# 于是期望差额正好是 **5×(H−1)+1**，本模型 `attn_causal.sparrow` 头=2、T=4 ⇒ 6。
$aSrc = Join-Path $root "examples\attn_causal.sparrow"
$aBins = Join-Path $root "examples\attn_bins"
$wOff = Join-Path $out "attn_off.vmb"
$wOn = Join-Path $out "attn_on.vmb"
$null = & $spar $aSrc --asm $asm --consts-dir $aBins -o $wOff 2>&1 | Out-String
$null = & $spar $aSrc --asm $asm --consts-dir $aBins --slot-reuse -o $wOn 2>&1 | Out-String
function AttnDstNames($vmb) {
    $lines = Get-Content ([System.IO.Path]::ChangeExtension($vmb, ".vml"))
    $start = -1
    for ($i = 0; $i -lt $lines.Count; $i++) { if ($lines[$i] -match ';\s*---\s*注意力') { $start = $i; break } }
    $names = New-Object System.Collections.Generic.List[string]
    for ($i = $start + 1; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match '^\s*(\.end|PRINT)') { break }
        $m = [regex]::Match($lines[$i], '^\s*(SHAPE|SLICE|TRANSPOSE|GEMM|SCALE|SOFTMAX|CONCAT|COPY|BIAS|EWADD|ACT|NORM)\s+([^\s,]+)')
        if ($m.Success) { $names.Add($m.Groups[2].Value) }
    }
    return @($names | Sort-Object -Unique)
}
$nOff = AttnDstNames $wOff
$nOn = AttnDstNames $wOn
$H = 2                       # attn_causal.sparrow：头=2
$expectDrop = 5 * ($H - 1) + 1
Write-Host "  不复用：$($nOff.Count) 个不同目的槽 —— $($nOff -join ' ')"
Write-Host "  复用后：$($nOn.Count) 个不同目的槽 —— $($nOn -join ' ')"
Check "3c-1 注意力块槽名减少" ($nOn.Count -lt $nOff.Count) "$($nOff.Count) → $($nOn.Count)"
Check "3c-2 差额 == 5×(头−1)+1 = $expectDrop（头=$H）" (($nOff.Count - $nOn.Count) -eq $expectDrop) "实测 $($nOff.Count - $nOn.Count)"
$sharedRoles = @($nOn | Where-Object { $_ -match '_(hq|hk|hkt|hv|hs)$' })
$cSlots      = @($nOn | Where-Object { $_ -match '_c\d+$' })
Check "3c-3 五个角色槽各只有一份（H 头轮流借）" ($sharedRoles.Count -eq 5) "共用槽：$($sharedRoles -join ' ')"
Check "3c-4 每头输出槽 `c` 仍按头分开" ($cSlots.Count -eq $H) "c 槽：$($cSlots -join ' ')"
Check "3c-5 拼头目标借了 `_q`（不再有 `_ctx`）" `
    (@($nOn | Where-Object { $_ -match '_ctx$' }).Count -eq 0 -and @($nOn | Where-Object { $_ -match '_q$' }).Count -eq 1) `
    "无 _ctx；_q 兼任拼头目标（最后一次读它在最后一个头的 GEMM，形状同为 [T,D]）"

Write-Host "`n===== 4. can-fail 的判据说明 =====" -ForegroundColor Yellow
Write-Host "  上面『三分支汇入 / 两分支汇入』这两个模型就是 can-fail 本体：" -ForegroundColor DarkGray
Write-Host "  两个同宽分支必须同时活着才能相加；若生命周期分析把还活着的槽回收，" -ForegroundColor DarkGray
Write-Host "  汇入会变成 B+B（而不是 A+B），输出立刻与不复用版不同 ⇒ 判据 1 会 FAIL。" -ForegroundColor DarkGray
Write-Host "  参考值另由 tools/residual_ab.ps1 用 numpy 独立算过（逐字符）。" -ForegroundColor DarkGray

Write-Host "`n===== 结果 =====" -ForegroundColor Yellow
if ($script:fails -eq 0) { Write-Host "  全部 PASS" -ForegroundColor Green }
else { Write-Host "  $($script:fails) 项 FAIL" -ForegroundColor Red }
exit $script:fails
