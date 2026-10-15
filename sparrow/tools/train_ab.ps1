# Sparrow V0.1.6 —— 训练路径（`损失 均方` + `反向`）验收驱动
#
# 为什么需要**训练版运行时**：
#   官方 `vmlm-runtime.exe`（75,776 B 不变式那个）**不带** train feature ——
#   它遇到反向指令会 `exit=6` 并打印 "train feature disabled"（实测）。
#   所以本驱动在 `out/` 下**隔离构建**一份 `--features train` 的运行时
#   （独立 CARGO_TARGET_DIR + 显式 --target，与 tools/aot_bench.ps1 同一套隔离办法），
#   并且验证官方二进制的 SHA256 **前后一致**（判据 E）。
#
# 判据：
#   A. 引擎打印的解析梯度 == numpy 独立算的解析梯度（相对误差 < 1e-3）
#   B. numpy 解析梯度 == numpy **中心差分**（由 tools/gen_train.py 自己断言，
#      它只依赖前向 ⇒ 是"这份解析梯度真的是梯度"的独立依据）
#   C. can-fail：扰动一个权重字节 ⇒ 对应的梯度必须变（否则"相同"可能恒真）
#   D. 边界可判定：`--slot-reuse`/`--loop-repeat` + 反向 ⇒ E114；
#      `--backend lib` + 反向 ⇒ E115；`反向` 没有 `损失` ⇒ E113；`训练` ⇒ E111
#   E. 隔离：官方 vmlm-runtime.exe 的 SHA256 前后一致（本驱动不写 vmlm 任何文件）
#
# 用法: pwsh -NoProfile -ExecutionPolicy Bypass -File tools/train_ab.ps1
# 退出码 = 失败项数（0 = 全绿）

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
$offi = EngineBin "vmlm-runtime.exe"
$spar = Join-Path $root "target\debug\sparrow.exe"
$bins = Join-Path $root "examples\train_bins"
$out  = Join-Path $root "out\train"
$src  = Join-Path $root "examples\train_mse.sparrow"

New-Item -ItemType Directory -Force -Path $out | Out-Null
$script:fails = 0
function Check($name, $cond, $detail) {
    if ($cond) { Write-Host "  $name : PASS  $detail" -ForegroundColor Green }
    else { Write-Host "  $name : FAIL  $detail" -ForegroundColor Red; $script:fails++ }
}
function Sha($p) { (Get-FileHash -Algorithm SHA256 $p).Hash }

Write-Host "===== 0. 隔离：记录官方运行时哈希，并在 out/ 下构建训练版 =====" -ForegroundColor Yellow
$h0 = Sha $offi
$engTarget = Join-Path $out "engine"          # 只在 out/ 里
$env:CARGO_TARGET_DIR = $engTarget
Push-Location (Join-Path $vmlm "runtime")
try { $b = (& cargo build --target x86_64-pc-windows-msvc --release --features train 2>&1 | Out-String) } finally { Pop-Location }
$bc = $LASTEXITCODE
Remove-Item Env:CARGO_TARGET_DIR
$train = Join-Path $engTarget "x86_64-pc-windows-msvc\release\vmlm-runtime.exe"
Check "0 训练版运行时构建成功" ($bc -eq 0 -and (Test-Path $train)) "exit=$bc，$([math]::Round((Get-Item $train).Length/1KB,1)) KB（官方推理版 $([math]::Round((Get-Item $offi).Length/1KB,1)) KB）"

Write-Host "`n===== 1. 素材 + numpy 独立参考（中心差分）=====" -ForegroundColor Yellow
Push-Location $root
$py = (& python "tools\gen_train.py" 2>&1 | Out-String)
$pc = $LASTEXITCODE
Pop-Location
Check "1 gen_train.py（含 FD 自检断言）" ($pc -eq 0) "exit=$pc；$(($py -split "`n" | Where-Object { $_ -match 'loss' }) -join '')"

# 读参考：an/fd 两组，每行 "an <名> <值…>"
$ref = @{}
foreach ($line in Get-Content (Join-Path $out "ref.txt")) {
    if ($line.Trim() -eq "") { continue }
    $p = $line.Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries)
    $ref["$($p[0]) $($p[1])"] = @($p[2..($p.Count - 1)] | ForEach-Object { [double]$_ })
}

Write-Host "`n===== 2. 编译成训练步 .vmb =====" -ForegroundColor Yellow
# 引擎写死读同目录下的 `model.vmb`（TRAPS A5），所以产物就叫这个名字。
$vmb = Join-Path $out "pristine.vmb"
$vml = Join-Path $out "pristine.vml"   # 汇编器把 .vml 写在 **.vmb 旁边**（同名）
$sparOut = (& $spar $src --emit-spar 2>&1 | Out-String)
$r = (& $spar $src --asm $asm --consts-dir $bins -o $vmb 2>&1 | Out-String)
$c = $LASTEXITCODE
Check "2 汇编训练步" ($c -eq 0 -and (Test-Path $vmb)) "exit=$c；$(($r -split "`n" | Where-Object { $_ -match 'OK:' }) -join '')"
Copy-Item $vmb (Join-Path $out "model.vmb") -Force
Check "2b .spar 里有训练陈述" ($sparOut -match 'LOSS\s+mse' -and $sparOut -match 'BACKWARD') "LOSS mse + BACKWARD"
$vmlText = Get-Content $vml -Raw
Check "2c .vml 里有反向指令（MSE_BWD + GEMM_BWD_W/X + BIAS_BWD + ACT_BWD + ZERO）" `
    ($vmlText -match 'MSE_BWD' -and $vmlText -match 'GEMM_BWD_W' -and $vmlText -match 'GEMM_BWD_X' -and $vmlText -match 'BIAS_BWD' -and $vmlText -match 'ACT_BWD' -and $vmlText -match 'ZERO') `
    "反向算子齐"

Write-Host "`n===== 3. 跑训练版运行时，取解析梯度 =====" -ForegroundColor Yellow
function RunEngine($dir) {
    Push-Location $dir
    try { $t = (& $train 2>&1 | Out-String); $c = $LASTEXITCODE } finally { Pop-Location }
    return @($t, $c)
}
$o = RunEngine $out
Check "3 训练版运行时退出码 0" ($o[1] -eq 0) "exit=$($o[1])（官方推理版对同一 .vmb 是 exit=6 train feature disabled）"
$lines = @($o[0] -split "`n" | Where-Object { $_.Trim() -match '^[-0-9\. ]+$' } | ForEach-Object { $_.Trim() })
# 打印顺序：模型输出（2 个数）→ g_l2_W(8) → g_l2_b(2) → g_l1_W(16) → g_l1_b(4)
$flat = @()
foreach ($l in $lines) { $flat += @($l -split '\s+' | Where-Object { $_ -ne "" } | ForEach-Object { [double]$_ }) }
Check "3b 打印段数正确" ($flat.Count -eq 2 + 8 + 2 + 16 + 4) "共 $($flat.Count) 个数（期望 32）"

$got = @{
    "l2_W" = @($flat[2..9]);   "l2_b" = @($flat[10..11])
    "l1_W" = @($flat[12..27]); "l1_b" = @($flat[28..31])
}
$worst = 0.0; $worstName = ""
$fdWorst = 0.0
foreach ($k in @("l2_W", "l2_b", "l1_W", "l1_b")) {
    $an = $ref["an $k"]; $fd = $ref["fd $k"]; $g = $got[$k]
    for ($i = 0; $i -lt $an.Count; $i++) {
        $den = [Math]::Max([Math]::Abs($an[$i]), 1e-6)
        $rel = [Math]::Abs($g[$i] - $an[$i]) / $den
        if ($rel -gt $worst) { $worst = $rel; $worstName = "$k[$i]" }
        $relFD = [Math]::Abs($an[$i] - $fd[$i]) / [Math]::Max([Math]::Abs($fd[$i]), 1e-6)
        if ($relFD -gt $fdWorst) { $fdWorst = $relFD }
    }
}
Check "A 引擎解析梯度 == numpy 解析梯度（相对误差 < 1e-3）" ($worst -lt 1e-3) "最差 $worstName 相对误差 $([math]::Round($worst,9))"
Check "B numpy 解析梯度 == numpy 中心差分（相对误差 < 1e-3）" ($fdWorst -lt 1e-3) "最差相对误差 $([math]::Round($fdWorst,9))"

Write-Host "`n===== 4. can-fail：扰动一个权重 ⇒ 对应梯度必须变 =====" -ForegroundColor Yellow
$wy = Join-Path $bins "l2_W.bin"
$orig = [System.IO.File]::ReadAllBytes($wy)
$mut = $orig.Clone()
$v = [BitConverter]::ToSingle($mut, 0); [BitConverter]::GetBytes([single]($v + 0.05)).CopyTo($mut, 0)
[System.IO.File]::WriteAllBytes($wy, $mut)
$vmbMut = Join-Path $out "mut.vmb"
$null = & $spar $src --asm $asm --consts-dir $bins -o $vmbMut 2>&1 | Out-String
Copy-Item $vmbMut (Join-Path $out "model.vmb") -Force
$o2 = RunEngine $out
[System.IO.File]::WriteAllBytes($wy, $orig)   # 立刻还原
Copy-Item $vmb (Join-Path $out "model.vmb") -Force
$lines2 = @($o2[0] -split "`n" | Where-Object { $_.Trim() -match '^[-0-9\. ]+$' } | ForEach-Object { $_.Trim() })
$flat2 = @()
foreach ($l in $lines2) { $flat2 += @($l -split '\s+' | Where-Object { $_ -ne "" } | ForEach-Object { [double]$_ }) }
$dOut = [Math]::Abs($flat2[0] - $flat[0])
$dG = [Math]::Abs($flat2[2] - $flat[2])
Check "C can-fail 输出变" ($dOut -gt 1e-4) "出口第 0 个分量：$($flat[0]) → $($flat2[0])（Δ=$([math]::Round($dOut,6))）"
Check "C can-fail g_l2_W[0] 变" ($dG -gt 1e-4) "g_l2_W[0]：$($flat[2]) → $($flat2[2])（Δ=$([math]::Round($dG,6))）"

Write-Host "`n===== 5. 边界可判定 =====" -ForegroundColor Yellow
$tmp = Join-Path $out "edge.sparrow"
function TryErr($text, $extra = @()) {
    Set-Content -LiteralPath $tmp -Value $text -Encoding utf8
    $argv = @($tmp, '--asm', $asm, '--consts-dir', $bins) + @($extra) + @('-o', (Join-Path $out "edge.vmb"))
    $e = (& $spar @argv 2>&1 | Out-String)
    return @($LASTEXITCODE, $e)
}
$body = "模型 tmse:`n  线性 输入=4 输出=4`n  激活 激活=tanh`n  线性 输出=2`n  连接 线性 激活`n  前向 线性 线性`n"
# D1：直接用**例子本体** + --slot-reuse（自造的简写模型会先撞上 E102「两个链首」）
$e1 = (& $spar $src --asm $asm --consts-dir $bins --slot-reuse -o (Join-Path $out "edge.vmb") 2>&1 | Out-String)
$c1 = $LASTEXITCODE
Check "D1 反向 + --slot-reuse ⇒ E114" ($c1 -ne 0 -and $e1 -match 'E114') "exit=$c1，$((($e1 -split "`n") | Where-Object { $_ -match 'E114' }) -join '')"
$e2 = TryErr "模型 tmse:`n  线性 输入=4 输出=4`n  线性 输出=2`n  连接 线性 线性`n  前向 线性 线性`n  反向`n"
Check "D2 反向 没有损失 ⇒ E113" ($e2[0] -ne 0 -and $e2[1] -match 'E113') "exit=$($e2[0])，$((($e2[1] -split "`n") | Where-Object { $_ -match 'E113' }) -join '')"
$e3 = TryErr "模型 tmse:`n  线性 输入=4 输出=4`n  损失 交叉熵`n  前向 线性 线性`n"
Check "D3 不认识的损失函数 ⇒ E112" ($e3[0] -ne 0 -and $e3[1] -match 'E112') "exit=$($e3[0])，$((($e3[1] -split "`n") | Where-Object { $_ -match 'E112' }) -join '')"
$e4 = TryErr "模型 tmse:`n  线性 输入=4 输出=4`n  训练 步数=10`n  前向 线性 线性`n"
Check "D4 `训练` 仍是命名边界 ⇒ E111" ($e4[0] -ne 0 -and $e4[1] -match 'E111') "exit=$($e4[0])"
$bodyLoop = "模型 tmse:`n  l1 = 线性 输入=4 输出=4`n  a = 激活 激活=tanh 重复2次`n  l2 = 线性 输出=2`n  连接 l1 a`n  连接 a l2`n  前向 l1 l2`n  损失 均方`n  反向`n"
$e5 = TryErr $bodyLoop @('--loop-repeat')
Check "D5 反向 + --loop-repeat ⇒ E114" ($e5[0] -ne 0 -and $e5[1] -match 'E114') "exit=$($e5[0])，$((($e5[1] -split "`n") | Where-Object { $_ -match 'E114' }) -join '')"
$e6 = (& $spar $src --backend lib --lang rust --vmlm-root $vmlm --consts-dir $bins 2>&1 | Out-String)
Check "D6 --backend lib + 反向 ⇒ E115" ($LASTEXITCODE -ne 0 -and $e6 -match 'E115') "exit=$LASTEXITCODE，$((($e6 -split "`n") | Where-Object { $_ -match 'E115' }) -join '')"

Write-Host "`n===== 6. 隔离：官方运行时没被碰过 =====" -ForegroundColor Yellow
Check "E 官方 vmlm-runtime.exe SHA256 前后一致" ((Sha $offi) -eq $h0) "$($h0.Substring(0,16))…"

Write-Host "`n===== 结果 =====" -ForegroundColor Yellow
if ($script:fails -eq 0) { Write-Host "  全部 PASS（A 解析梯度 / B 数值梯度 / C can-fail / D 边界 / E 隔离）" -ForegroundColor Green }
else { Write-Host "  $($script:fails) 项 FAIL" -ForegroundColor Red }
exit $script:fails
