# Sparrow 全量验收驱动（V0.1.5）
#
# 一条命令跑完：cargo test + 全部验收驱动 + 「验收后工作区干净」不变量。
# 驱动清单就是下面 $drivers 那一行（数量不写死在注释里：写死过一次 8，加到 10 就假失败）。
#
# 为什么最后那条不变量值得存在：
#   examples/ 下的 .spar/.vml/.vmb/.args 与 *_bins/ 常量全是**生成物**，本仓库
#   把它们当「生成物指纹」入库（理由见 REPORT_V015.md §1.3）。于是「跑完全部
#   驱动后 git status 必须为空」成为一条可假（can-fail）的不变量：任何一个
#   生成器变得不确定、或哪个驱动偷偷改了工作区，它都会红。
#
# 用法: pwsh -NoProfile -ExecutionPolicy Bypass -File tools/verify_all.ps1
# 退出码 = 失败项数（0 = 全绿）

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot   # 仓库根（本脚本在 <root>\tools\ 下）
# git / cargo / pwsh 一律从 PATH 取（本机 E:\Git\cmd\git.exe 就在 PATH 上）；
# 需要钉死时用 $env:SPARROW_GIT 覆盖。
$git = if ($env:SPARROW_GIT) { $env:SPARROW_GIT } else { (Get-Command git -ErrorAction SilentlyContinue).Source }
if (-not $git) { Write-Host '错误：找不到 git；用 $env:SPARROW_GIT 指定' -ForegroundColor Red; exit 1 }
$out   = Join-Path $root "out"
$log   = Join-Path $out "_verify_all.log"
New-Item -ItemType Directory -Force -Path $out | Out-Null

$script:fails = 0
function Check($name, $cond, $detail) {
    if ($cond) { Write-Host "  $name : PASS  $detail" -ForegroundColor Green }
    else { Write-Host "  $name : FAIL  $detail" -ForegroundColor Red; $script:fails++ }
}

"===== 0. 前置：工作区在跑之前必须是干净的 =====" | Out-File $log -Encoding utf8
Write-Host "===== 0. 前置：跑之前工作区干净 =====" -ForegroundColor Yellow
# 合并布局（本目录是引擎仓库的子目录）下只对准**本目录**，不要求整个引擎仓库干净；
# 而本目录若不在任何 git 仓库里（私有工作副本），必须**显式 SKIP** ——
# `@(空) .Count -eq 0` 会让判据自动 PASS，那是假通过，本项目不接受。
$inRepo = $false
$null = & $git -C $root rev-parse --show-toplevel 2>&1
if ($LASTEXITCODE -eq 0) { $inRepo = $true }
$pre = if ($inRepo) { @(& $git -C $root status --porcelain -- .) } else { @() }
if (-not $inRepo) { Write-Host "  G0 跑之前工作区干净 : SKIP  本目录不在 git 仓库里（私有工作副本）" -ForegroundColor DarkYellow }
else { Check "G0 跑之前 git status 为空" ($pre.Count -eq 0) "entries=$($pre.Count)" }

Write-Host "`n===== 1. cargo test =====" -ForegroundColor Yellow
Push-Location $root
$t0 = Get-Date
$testOut = (& cargo test 2>&1 | Out-String)
$tTest = ((Get-Date) - $t0).TotalSeconds
Pop-Location
$testOut | Out-File $log -Append -Encoding utf8
$results = @($testOut -split "`n" | Where-Object { $_ -match 'test result:' })
$passed = 0; $failed = 0
foreach ($r in $results) {
    if ($r -match '(\d+) passed') { $passed += [int]$Matches[1] }
    if ($r -match '(\d+) failed') { $failed += [int]$Matches[1] }
}
$nSuites = $results.Count
Check "T1 cargo test 全绿" ($failed -eq 0 -and $passed -gt 0) "$nSuites 个 target，$passed passed / $failed failed，$([math]::Round($tTest,1)) s"

$drivers = @(
    'e2e_tiny', 'loop_ab', 'residual_ab', 'slot_reuse',
    'bf16_ab', 'attn_ab', 'lib_ab', 'perf_frontend', 'seq_ab', 'int8_ab',
    'train_ab'
)
Write-Host "`n===== 2. 验收驱动（$($drivers.Count) 个）=====" -ForegroundColor Yellow
$codes = @{}
foreach ($d in $drivers) {
    $t0 = Get-Date
    $o = (& pwsh -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root "tools\$d.ps1") 2>&1 | Out-String)
    $code = $LASTEXITCODE
    $secs = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
    $codes[$d] = $code
    $o | Out-File $log -Append -Encoding utf8
    $tail = @($o -split "`n" | Where-Object { $_ -match 'PASS|FAIL' } | Select-Object -Last 1)
    Check "D-$d 退出码 0" ($code -eq 0) "exit=$code，$secs s，最后一行：$($tail -join '')"
}

Write-Host "`n===== 3. 不变量：跑完之后工作区仍然干净 =====" -ForegroundColor Yellow
$post = if ($inRepo) { @(& $git -C $root status --porcelain -- .) } else { @() }
if (-not $inRepo) { Write-Host "  G1 跑完之后工作区干净 : SKIP  本目录不在 git 仓库里（私有工作副本）" -ForegroundColor DarkYellow }
else { Check "G1 跑完之后 git status 为空" ($post.Count -eq 0) "entries=$($post.Count)" }
if ($post.Count -gt 0) {
    Write-Host "  被改动/新增的文件（前 20 个）：" -ForegroundColor DarkYellow
    $post | Select-Object -First 20 | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkYellow }
}

"`n最终：fails=$($script:fails)" | Out-File $log -Append -Encoding utf8
Write-Host ""
if ($script:fails -eq 0) { Write-Host "全部 PASS（cargo test + $($drivers.Count) 个驱动 + 工作区干净）" -ForegroundColor Green }
else { Write-Host "$($script:fails) 项 FAIL" -ForegroundColor Red }
Write-Host "日志：$log"
exit $script:fails
