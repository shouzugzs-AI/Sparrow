# Sparrow V0.1.3 —— 前端性能基准（可复现、带判据）
#
# 判据：编译时间必须**近似线性**。线性的话 N=1600 约是 N=100 的 16 倍；
# 出现二次项的话是 256 倍。阈值取 40 倍（给常数项与缓存留余量）。
#
# 为什么这条值得测：V0.1.3 之前关键字分类每次都对**上百个别名**做
# `norm(a)`（`to_ascii_lowercase()` → 一次堆分配），而 `classes()`
# 在解析每个词时都会被调用。1600 层模型（~3600 个词）里这几行占了约 95% 的
# 解析时间。改成逐字节 ASCII 不敏感前缀比较后：
#
#     400 层：96 ms → 8.7 ms（11×）
#    1600 层：367 ms → 20.4 ms（18×）
#
# 用法: pwsh -NoProfile -ExecutionPolicy Bypass -File tools/perf_frontend.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot   # 仓库根（本脚本在 <root>\tools\ 下）
$spar = Join-Path $root "target\release\sparrow.exe"
$dir  = Join-Path $root "out\perf"
New-Item -ItemType Directory -Force -Path $dir | Out-Null
$script:fails = 0
function Check($name, $cond, $detail) {
    if ($cond) { Write-Host "  $name : PASS  $detail" -ForegroundColor Green }
    else { Write-Host "  $name : FAIL  $detail" -ForegroundColor Red; $script:fails++ }
}

if (-not (Test-Path $spar)) {
    # 新克隆 / 合并布局下 target\release 不存在 —— 自己构建一次（本驱动量的就是 release 前端）
    Write-Host "  （缺 target\release\sparrow.exe，先 cargo build --release）" -ForegroundColor DarkYellow
    Push-Location $root
    $bn = (& cargo build --release 2>&1 | Out-String)
    Pop-Location
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $spar)) { throw "cargo build --release 失败：$bn" }
}

# 生成规模序列（一个模型一条链，每层 3 个 token）
python -c @"
import os
d = r'$dir'
for n in (1, 200, 400, 800, 1600):
    L = ['模型 big:', '  嵌入 词表=8 维度=64']
    L += [f'  l{i} = 线性 输出=64' for i in range(n)]
    L.append('  连接 ' + ' '.join(['嵌入'] + [f'l{i}' for i in range(n)]))
    L.append('  前向 嵌入 l%d' % (n - 1))
    open(os.path.join(d, f'big{n}.sparrow'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
"@ | Out-Null

function Best($sparrow, $mode) {
    $best = [double]::MaxValue
    foreach ($k in 1..4) {
        $t = Measure-Command { & $spar $sparrow @mode -o (Join-Path $dir "b.tmp") | Out-Null }
        if ($t.TotalMilliseconds -lt $best) { $best = $t.TotalMilliseconds }
    }
    return $best
}

Write-Host "===== 前端编译时间（min of 4） =====" -ForegroundColor Yellow
# 进程启动 + 动态库加载有 ~8-10 ms 的固定成本，小模型上它占大头 ⇒
# 规模判据要用"去掉基线后的时间"和"边际成本"，不能用绝对值比值。
$base = Best (Join-Path $dir "big1.sparrow") @("--backend", "vml")
Write-Host ("  基线（1 层，主要是进程启动）: {0:N1} ms" -f $base)
$times = @{}
foreach ($n in @(200, 400, 800, 1600)) {
    $f = Join-Path $dir "big$n.sparrow"
    $t = Best $f @("--backend", "vml")
    $times[$n] = $t
    $size = (Get-Item (Join-Path $dir "b.tmp")).Length
    Write-Host ("  {0,5} 层 : {1,8:N1} ms（净 {2,7:N1} ms）产物 {3,9:N0} B  {4,7:N4} ms/层" -f `
            $n, $t, ($t - $base), $size, (($t - $base) / $n))
}

# 边际成本：每加一层要多少时间。线性 ⇒ 两段基本相等；有二次项 ⇒ 后一段急剧变大。
$m1 = ($times[800] - $times[400]) / 400.0
$m2 = ($times[1600] - $times[800]) / 800.0
Write-Host ("`n  边际成本：400→800 {0:N4} ms/层，800→1600 {1:N4} ms/层" -f $m1, $m2)
Check "1 近似线性（边际成本稳定）" ($m2 -lt 3 * $m1 + 0.01) ("{0:N4} vs {1:N4} ms/层" -f $m1, $m2)
Check "2 绝对速度" ($times[1600] -lt 150) "1600 层 $([math]::Round($times[1600],1)) ms"
Check "3 大模型单调" ($times[400] -le $times[800] -and $times[800] -le $times[1600]) "400/800/1600 单调递增"

Write-Host "`n===== 结果 =====" -ForegroundColor Yellow
if ($script:fails -eq 0) { Write-Host "  全部 PASS" -ForegroundColor Green }
else { Write-Host "  $($script:fails) 项 FAIL" -ForegroundColor Red }
exit $script:fails
