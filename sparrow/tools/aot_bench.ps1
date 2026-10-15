# Sparrow V0.1.5 —— ② AOT（`--backend lib`）vs 解释器（`.vmb`）的可测基准
#
# 为什么不能"跑一遍 exe 记 wall time"就完事：
#   * 进程启动（CreateProcess + 页错误 + 装载常量）在**毫秒**量级，而单次前向在
#     **十微秒**量级 —— 直接比 wall time 测的是启动，不是执行；
#   * 引擎一次运行只前向一遍（`vmlm-runtime.exe` 固定读 `model.vmb`）。
# 所以两条路都改成**进程内**基准，口径对齐：
#   解释器：`vmlm-runtime-bench.exe <model.vmb> <reps>`
#           —— vmlm 自带（runtime/src/bench.rs）：预热 1 遍 + QueryPerformanceCounter
#              计时 reps 遍，打印 `avg us/iter = N`；
#   AOT   ：`sparrow --backend lib --bench-iters K` 生成的主程序（本版新增）
#           —— 预热 1 遍 + `std::time::Instant` 计时 K 遍，打印**同一行文本**。
#
# 两条路都先把「打印」排除在计时之外：引擎那条路把 `.vml` 里的 `PRINT` 行去掉再汇编
# （否则每个 iter 都要输出 D 个数，I/O 会把 GEMM 完全盖掉；实测 D=256 时每 iter 256 次
# 写出，量级与计算同阶）。
#
# 隔离承诺：本驱动**不写** vmlm 仓库的任何文件 ——
#   * 官方 plain 引擎**拷到 out/ 里**跑（它只读同目录的 model.vmb）；
#   * bench 引擎构建到 out/ 下的独立 `CARGO_TARGET_DIR`；
#   * 构建前后对 `vmlm-runtime.exe` 与 `Cargo.lock` 取 SHA256 比对（判据 5）。
#
# 判据（每条都能假）：
#   1 正确性 ：plain 引擎输出 == AOT 输出（**逐字符**）
#   1b 非退化 ：输出必须有 >D/2 个互不相同的值 —— 否则"相同"可能只是两边都算出了垃圾
#              （实测：96 层**未归一化**权重的激活是 NaN，而引擎的截断式 print_f32
#               把 NaN 打成 `0.000000`，"逐字符相同"与 can-fail 于是双双恒真）
#   2 等价性 ：隔离构建的 bench 引擎 reps=1 输出 == plain 引擎输出
#   3 可分辨 ：耗时必须随 N 增长（斜率 > 0 且增长量 > 3× 观测抖动），否则报"测不出来"
#   4 can-fail：改一个权重字节 ⇒ AOT 输出必须变化（否则判据 1 可能恒真）
#   5 隔离   ：vmlm 的 vmlm-runtime.exe / Cargo.lock SHA256 前后一致
#
# 用法: pwsh -NoProfile -ExecutionPolicy Bypass -File tools/aot_bench.ps1
# 退出码 = 失败项数（0 = 全绿）

$ErrorActionPreference = "Stop"
# Python 的进度行是中文：不显式统一编码时，子进程按 ANSI/GBK 写、PowerShell 按另一种
# 解码，控制台打成乱码（实测）。把两边都钉成 UTF-8。
$env:PYTHONIOENCODING = "utf-8"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
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
$plain = EngineBin "vmlm-runtime.exe"
$spar  = Join-Path $root "target\debug\sparrow.exe"
$out   = Join-Path $root "out\aot_bench"
$engTgt   = Join-Path $out "engine_target"
$crateTgt = Join-Path $out "crate_target"
$DIM   = 256
$CASES = @(8, 32, 96)
$REPS  = 400
$RUNS  = 5          # 每个 (后端, N) 跑几轮取最小值

New-Item -ItemType Directory -Force -Path $out | Out-Null
$script:fails = 0
function Check($name, $cond, $detail) {
    if ($cond) { Write-Host "  $name : PASS  $detail" -ForegroundColor Green }
    else { Write-Host "  $name : FAIL  $detail" -ForegroundColor Red; $script:fails++ }
}
function Hash($p) { if (Test-Path $p) { (Get-FileHash -Algorithm SHA256 $p).Hash } else { "MISSING" } }
function AvgUs($text) {
    $m = [regex]::Match($text, 'avg us/iter = (\d+)')
    if ($m.Success) { return [int]$m.Groups[1].Value } else { return $null }
}
# 一组计时（取最小 + 抖动）——最小值的意义：排除其它进程/GC 的偶发干扰
function TimeRuns($exe, $argList) {
    $vals = @()
    for ($k = 0; $k -lt $RUNS; $k++) {
        if ($argList) { $t = (& $exe @argList 2>&1 | Out-String) } else { $t = (& $exe 2>&1 | Out-String) }
        $v = AvgUs $t
        if ($null -eq $v) { return $null }
        $vals += $v
    }
    $min = ($vals | Measure-Object -Minimum).Minimum
    $max = ($vals | Measure-Object -Maximum).Maximum
    return @{ min = $min; spread = ($max - $min) }
}
function FitSlope($xs, $ys) {
    $n = $xs.Count
    $mx = ($xs | Measure-Object -Average).Average
    $my = ($ys | Measure-Object -Average).Average
    $num = 0.0; $den = 0.0
    for ($i = 0; $i -lt $n; $i++) { $num += ($xs[$i] - $mx) * ($ys[$i] - $my); $den += ($xs[$i] - $mx) * ($xs[$i] - $mx) }
    $s = $num / $den
    return @{ slope = $s; intercept = $my - $s * $mx }
}

# ===== 0. 隔离承诺：构建前取哈希 =====
Write-Host "===== 0. 隔离：构建前取 vmlm 关键文件哈希 =====" -ForegroundColor Yellow
$hEng0  = Hash $plain
$hLock0 = Hash (Join-Path $vmlm "Cargo.lock")
Write-Host "  vmlm-runtime.exe sha256 = $($hEng0.Substring(0,16))…"
Write-Host "  vmlm/Cargo.lock  sha256 = $($hLock0.Substring(0,16))…"

# 官方 plain 引擎拷进 out/ 跑（它固定读同目录的 model.vmb）
$engDir = Join-Path $out "engine_plain"
New-Item -ItemType Directory -Force -Path $engDir | Out-Null
Copy-Item $plain (Join-Path $engDir "vmlm-runtime.exe") -Force

Write-Host "`n===== 1. 构建 bench 引擎（独立 CARGO_TARGET_DIR，不碰 vmlm 的 target） =====" -ForegroundColor Yellow
$env:CARGO_TARGET_DIR = $engTgt
Push-Location (Join-Path $vmlm "runtime")
$bout = (& cargo build --target x86_64-pc-windows-msvc --release --features bench 2>&1 | Out-String)
$bcode = $LASTEXITCODE
Pop-Location
$bench = Join-Path $out "vmlm-runtime-bench.exe"
$benchBuilt = Join-Path $engTgt "x86_64-pc-windows-msvc\release\vmlm-runtime.exe"
if (Test-Path $benchBuilt) { Copy-Item $benchBuilt $bench -Force }
Check "1a bench 引擎构建成功" ($bcode -eq 0 -and (Test-Path $bench)) "exit=$bcode，$((Get-Item $bench).Length) B"
if (-not (Test-Path $bench)) { Write-Host $bout; exit 1 }
$usage = (& $bench 2>&1 | Out-String)
Check "1b bench 版确实带 argv/计时设施" ($usage -match 'usage: vmlm-runtime') "无参数时打 usage（default 构建没有 argv 解析）"

# ===== 2. 逐个 N：正确性 + 计时 =====
Write-Host "`n===== 2. 每个 N：正确性（逐字符）+ 进程内计时（REPS=$REPS, RUNS=$RUNS） =====" -ForegroundColor Yellow
$rows = @()
foreach ($n in $CASES) {
    $tag = "c$n"
    Write-Host "`n--- N=$n ---" -ForegroundColor DarkCyan
    & python (Join-Path $root "tools\gen_bench.py") $tag $n $DIM | ForEach-Object { Write-Host $_ }
    $model = Join-Path $out "model_$tag.sparrow"
    $bins  = Join-Path $out "bins_$tag"
    $vmb   = Join-Path $out "$tag.vmb"
    $r = (& $spar $model --asm $asm --consts-dir $bins -o $vmb 2>&1 | Out-String)
    if ($r -notmatch 'OK:') { throw "汇编失败 $tag：$r" }

    # 判据 1：plain 引擎（在 out/engine_plain 里跑）
    Copy-Item $vmb (Join-Path $engDir "model.vmb") -Force
    Push-Location $engDir
    $e1 = (& (Join-Path $engDir "vmlm-runtime.exe") 2>&1 | Out-String).Trim()
    $ec = $LASTEXITCODE
    Pop-Location
    Check "2.$n 引擎跑通" ($ec -eq 0) "exit=$ec，输出长 $($e1.Length)"

    # 判据 2：bench 引擎 reps=1（同一份 .vmb，带 PRINT）
    $e2 = (& $bench $vmb 1 2>&1 | Out-String).Trim()
    Check "2.$n bench 引擎与官方二进制同源" ($e2 -ceq $e1) "reps=1 输出逐字符相同"

    # 解释器计时：去掉 PRINT 再汇编（打印不能进计时区）
    $vml  = Join-Path $out "$tag.vml"
    $argsF = Join-Path $out "$tag.args"
    $npVml = Join-Path $out "${tag}_noprint.vml"
    (Get-Content $vml) | Where-Object { $_.Trim() -notmatch '^PRINT' } | Set-Content $npVml -Encoding ascii
    $npVmb = Join-Path $out "${tag}_noprint.vmb"
    $constArgs = @(Get-Content $argsF | Where-Object { $_.Trim() -ne '' })
    Push-Location $bins
    $asmOut = (& $asm $npVml $npVmb @constArgs 2>&1 | Out-String)
    $asmCode = $LASTEXITCODE
    Pop-Location
    Check "2.$n 去 PRINT 版汇编成功" ($asmCode -eq 0 -and (Test-Path $npVmb)) "exit=$asmCode（$($constArgs.Count) 条常量）"

    $interp = TimeRuns $bench @($npVmb, "$REPS")
    if ($null -eq $interp) { throw "解释器计时没拿到 avg us/iter（$tag）" }

    # AOT：一次生成（带 --bench-iters）同时用于正确性与计时
    $crate = Join-Path $out "aot_$tag"
    Remove-Item -Recurse -Force $crate -ErrorAction SilentlyContinue
    & $spar $model --backend lib --lang rust --vmlm-root $vmlm --consts-dir $bins --bench-iters $REPS -o $crate | Out-Null
    $env:CARGO_TARGET_DIR = $crateTgt
    $cb = (& cargo build --release --quiet --manifest-path (Join-Path $crate "Cargo.toml") 2>&1 | Out-String)
    $aotExe = Join-Path $crateTgt "release\aot_$tag.exe"
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $aotExe)) { Write-Host $cb; throw "AOT 构建失败 $tag" }
    $aotOut = (& $aotExe 2>&1 | Out-String)
    $aotVal = @($aotOut -split "`n")[0].Trim()
    Check "2.$n AOT 输出与引擎逐字符相同" ($aotVal -ceq $e1) "N=$n"
    # **退化守卫**（实测踩到）：96 层随机权重会让激活变成 NaN，而引擎的 print_f32 是
    # 截断实现，NaN 被打成 `0.000000` —— 于是"逐字符相同"和 can-fail **双双恒真**。
    # 这条判据保证"相同"不是因为两边都算出了垃圾。
    $valTokens = @($e1 -split '\s+' | Where-Object { $_ -ne '' })
    $distinct = @($valTokens | Sort-Object -Unique).Count
    Check "2.$n 输出非退化（判据不恒真）" ($valTokens.Count -eq $DIM -and $distinct -gt $DIM / 2 -and $distinct -gt 1) `
        "值 $($valTokens.Count) 个，其中 $distinct 个互不相同（max|y| 见上）"
    $aot = TimeRuns $aotExe $null
    if ($null -eq $aot) { throw "AOT 计时没拿到 avg us/iter（$tag）" }

    $insCount = @(Get-Content $vml | Where-Object { $t = $_.Trim(); $t -ne '' -and -not $t.StartsWith(';') -and -not $t.StartsWith('.') }).Count
    $wBytes = (Get-ChildItem $bins -File | Measure-Object -Sum Length).Sum
    $rows += [pscustomobject]@{
        N = $n; ins = $insCount; wB = $wBytes
        vmbB = (Get-Item $vmb).Length; aotB = (Get-Item $aotExe).Length
        it_min = $interp.min; it_sp = $interp.spread
        aot_min = $aot.min; aot_sp = $aot.spread
    }
}

# ===== 3. 判据 3：可分辨 + 拟合每层成本 =====
Write-Host "`n===== 3. 可分辨性与每层成本 =====" -ForegroundColor Yellow
$xs = @($rows | ForEach-Object { $_.N })
$itY = @($rows | ForEach-Object { [double]$_.it_min })
$aoY = @($rows | ForEach-Object { [double]$_.aot_min })
$itFit = FitSlope $xs $itY
$aoFit = FitSlope $xs $aoY
$spreadMax = ($rows | ForEach-Object { [math]::Max($_.it_sp, $_.aot_sp) } | Measure-Object -Maximum).Maximum
if ($null -eq $spreadMax) { $spreadMax = 0 }
$itGrow = $itY[-1] - $itY[0]
$aoGrow = $aoY[-1] - $aoY[0]
Check "3a 解释器耗时随 N 增长（可分辨）" ($itFit.slope -gt 0 -and $itGrow -gt 3 * $spreadMax) "增长 $itGrow µs，抖动上限 $spreadMax µs"
Check "3b AOT 耗时随 N 增长（可分辨）" ($aoFit.slope -gt 0 -and $aoGrow -gt 3 * $spreadMax) "增长 $aoGrow µs，抖动上限 $spreadMax µs"

Write-Host ""
Write-Host ("  {0,-4} {1,-8} {2,-10} {3,-12} {4,-14} {5,-16} {6,-14}" -f 'N','指令数','权重 B','.vmb B','AOT exe B','解释器 µs/iter','AOT µs/iter')
foreach ($r in $rows) {
    Write-Host ("  {0,-4} {1,-8} {2,-10} {3,-12} {4,-14} {5,-16} {6,-14}" -f `
        $r.N, $r.ins, $r.wB, $r.vmbB, $r.aotB, "$($r.it_min) (±$($r.it_sp))", "$($r.aot_min) (±$($r.aot_sp))")
}
Write-Host ""
Write-Host ("  解释器：{0:N3} µs/层 + 固定 {1:N1} µs   （{2:N3} → {3:N3} µs）" -f $itFit.slope, $itFit.intercept, $itY[0], $itY[-1])
Write-Host ("  AOT   ：{0:N3} µs/层 + 固定 {1:N1} µs   （{2:N3} → {3:N3} µs）" -f $aoFit.slope, $aoFit.intercept, $aoY[0], $aoY[-1])
# 为什么**不能**只报上面那一条斜率：实测每层成本不是常数（权重 2.1 MB → 25.3 MB，
# 越过 L3 后边际成本上升），两参数拟合的截距会变成负数 —— 那是模型不合适的信号，
# 不是"固定开销为负"。所以下面按**相邻两点**报边际成本，哪一段是多少一目了然。
if ($itFit.intercept -lt 0 -or $aoFit.intercept -lt 0) {
    Write-Host "  注意：两参数拟合截距为负 ⇒ 线性模型不合适（每层成本随 N 上升，见下面的分段边际）" -ForegroundColor DarkYellow
}
Write-Host ""
Write-Host "  分段边际成本（µs/层）：" -ForegroundColor Cyan
for ($k = 1; $k -lt $rows.Count; $k++) {
    $dn = $rows[$k].N - $rows[$k - 1].N
    $itm = ($itY[$k] - $itY[$k - 1]) / $dn
    $aom = ($aoY[$k] - $aoY[$k - 1]) / $dn
    $seg = $aom / $itm
    $wmb = [math]::Round($rows[$k].wB / 1MB, 1)
    Write-Host ("    N {0,3}→{1,-3}：解释器 {2,7:N2}，AOT {3,7:N2}，比 {4:N3}   （权重 {5} MB）" -f `
        $rows[$k - 1].N, $rows[$k].N, $itm, $aom, $seg, $wmb)
}
$lastSeg = $rows.Count - 1
$ratio = (($aoY[$lastSeg] - $aoY[$lastSeg - 1]) / ($rows[$lastSeg].N - $rows[$lastSeg - 1].N)) /
         (($itY[$lastSeg] - $itY[$lastSeg - 1]) / ($rows[$lastSeg].N - $rows[$lastSeg - 1].N))
Write-Host ("  最后一档每层成本比（AOT / 解释器）= {0:N3}  ⇒ {1}" -f $ratio, $(if ($ratio -lt 1) { "AOT 每层快 $([math]::Round((1-$ratio)*100,1))%" } else { "AOT 每层慢 $([math]::Round(($ratio-1)*100,1))%" })) -ForegroundColor Cyan
$totRatio = $aoY[-1] / $itY[-1]
Write-Host ("  N={0} 整次前向比 = {1:N3}（含各自固定开销）" -f $xs[-1], $totRatio) -ForegroundColor Cyan

# ===== 4. 判据 4：can-fail（改一个权重字节） =====
Write-Host "`n===== 4. can-fail：改一个权重字节 ⇒ AOT 输出必须变 =====" -ForegroundColor Yellow
$last = $rows[-1]
$mut = Join-Path $out "mut_bins"
Remove-Item -Recurse -Force $mut -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $mut | Out-Null
Copy-Item (Join-Path $out "bins_c$($last.N)\*") $mut -Force
& python -c @"
import struct, sys
p = sys.argv[1]
b = bytearray(open(p,'rb').read())
v = struct.unpack_from('<f', b, 0)[0] + 1.0
struct.pack_into('<f', b, 0, v)
open(p,'wb').write(bytes(b))
print(f'  l1_W[0] -> {v:.6f}')
"@ (Join-Path $mut "l1_W.bin")
$mutCrate = Join-Path $out "aot_mut"
Remove-Item -Recurse -Force $mutCrate -ErrorAction SilentlyContinue
& $spar (Join-Path $out "model_c$($last.N).sparrow") --backend lib --lang rust --vmlm-root $vmlm --consts-dir $mut -o $mutCrate | Out-Null
$env:CARGO_TARGET_DIR = $crateTgt
$null = (& cargo build --release --quiet --manifest-path (Join-Path $mutCrate "Cargo.toml") 2>&1 | Out-String)
$mutExe = Join-Path $crateTgt "release\aot_mut.exe"
$mutVal = @((& $mutExe 2>&1 | Out-String) -split "`n")[0].Trim()
$baseVal = @((& (Join-Path $crateTgt "release\aot_c$($last.N).exe") 2>&1 | Out-String) -split "`n")[0].Trim()
Check "4 改权重 ⇒ 输出变化" ($mutVal -cne $baseVal) "base=[$($baseVal.Substring(0,[math]::Min(40,$baseVal.Length)))…] mut=[$($mutVal.Substring(0,[math]::Min(40,$mutVal.Length)))…]"

# ===== 5. 判据 5：隔离 =====
Write-Host "`n===== 5. 隔离：vmlm 的文件未被改动 =====" -ForegroundColor Yellow
$hEng1  = Hash $plain
$hLock1 = Hash (Join-Path $vmlm "Cargo.lock")
Check "5a vmlm-runtime.exe 未改" ($hEng0 -ceq $hEng1) "sha256 前 $($hEng0.Substring(0,12))… 后 $($hEng1.Substring(0,12))…"
Check "5b vmlm/Cargo.lock 未改" ($hLock0 -ceq $hLock1) "sha256 前 $($hLock0.Substring(0,12))… 后 $($hLock1.Substring(0,12))…"

Write-Host "`n===== 结果 =====" -ForegroundColor Yellow
if ($script:fails -eq 0) { Write-Host "  全部 PASS（正确性 + 可分辨 + can-fail + 隔离）" -ForegroundColor Green }
else { Write-Host "  $($script:fails) 项 FAIL" -ForegroundColor Red }
exit $script:fails
