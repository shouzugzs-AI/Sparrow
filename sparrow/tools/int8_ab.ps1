# Sparrow V0.1.5 —— ⑥ int8（per-输出行 scale）验收驱动
#
# 判据（每条都能假）：
#   1 **k=1 精确性**：`i8o`（单层、输入宽度 1，内积只有一项）在
#      `--weight-dtype f32` 与 `int8` 下必须**逐字符相同**。
#      为什么这是强判据：`w = q/64` 精确可表示，且"先乘 q 再乘 2^-6"与"直接乘 w"
#      在实数上相等 ⇒ 只有布局/scale 行下标/bias 位置全对才可能逐字符相同。
#   2 k=4 有界：`i8t` 的两条路差值必须 ≤ 实测界（累加顺序不同 ⇒ 允许 ~1 ULP），
#      并把实测最大差打印出来（不藏）。
#   3 **can-fail**：把 `a_scale[0]` 动一个 2^-6（= 动一行 scale）⇒ 差值必须
#      **远超**第 2 条的界（否则"差值小"可能只是没用上 scale）。
#   4 朴素量化对照：`examples/int8_naive_bins`（scale = max|row|/127）的差值必须
#      **明显大于**精确集 ⇒ 第 2 条测的是量化损失，不是噪声。
#   5 hard-fail：`--asm` / `--backend vml` + `--weight-dtype int8` ⇒ `E109` 说清理由。
#   6 体积：int8 的权重常量字节数 ≈ f32 的 1/4（从生成 crate 引用的 .bin 量）。
#   7. V0.1.6：**注意力投影**（`Wq/Wk/Wv/Wo` 四组 `gemm_nt`）也吃 int8：
#      精确可表示时 f32 与 int8 **逐字符相同**；动一行投影 scale ⇒ 输出必变；
#      体积按同一个公式核对；结构上四个投影都调了 `gemm_nt_int8`。
#
# 用法: pwsh -NoProfile -ExecutionPolicy Bypass -File tools/int8_ab.ps1
# 退出码 = 失败项数（0 = 全绿）

$ErrorActionPreference = "Stop"
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
$spar = Join-Path $root "target\debug\sparrow.exe"
$out  = Join-Path $root "out\int8"
$tgt  = Join-Path $out "target"

New-Item -ItemType Directory -Force -Path $out | Out-Null
$env:CARGO_TARGET_DIR = $tgt
$script:fails = 0
function Check($name, $cond, $detail) {
    if ($cond) { Write-Host "  $name : PASS  $detail" -ForegroundColor Green }
    else { Write-Host "  $name : FAIL  $detail" -ForegroundColor Red; $script:fails++ }
}
# 生成一个 crate（lib 后端）并跑它，返回 stdout（已 Trim）
function BuildRun($src, $bins, $tag, $wd) {
    $dir = Join-Path $out $tag
    Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
    $r = & $spar $src --backend lib --lang rust --vmlm-root $vmlm --consts-dir $bins --weight-dtype $wd -o $dir 2>&1 | Out-String
    if ($r -notmatch '\[lib\]') { throw "生成失败 $tag：$r" }
    $cb = (& cargo build --release --quiet --manifest-path (Join-Path $dir "Cargo.toml") 2>&1 | Out-String)
    $exe = Join-Path $tgt "release\$tag.exe"
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $exe)) { Write-Host $cb; throw "构建失败 $tag" }
    return (& $exe 2>&1 | Out-String).Trim()
}
function MaxDiff($a, $b) {
    $x = @($a -split '\s+' | Where-Object { $_ -ne '' } | ForEach-Object { [double]$_ })
    $y = @($b -split '\s+' | Where-Object { $_ -ne '' } | ForEach-Object { [double]$_ })
    if ($x.Count -ne $y.Count) { return [double]::PositiveInfinity }
    $m = 0.0
    for ($i = 0; $i -lt $x.Count; $i++) { $m = [math]::Max($m, [math]::Abs($x[$i] - $y[$i])) }
    return $m
}

Write-Host "===== 0. 素材（确定性 + 精确可表示） =====" -ForegroundColor Yellow
python (Join-Path $root "tools\gen_int8.py") | ForEach-Object { Write-Host $_ }
$exact = Join-Path $root "examples\int8_bins"
$naiveBins = Join-Path $root "examples\int8_naive_bins"
$badBins = Join-Path $root "examples\int8_bad_bins"
$one = Join-Path $root "examples\int8_one.sparrow"
$two = Join-Path $root "examples\int8_two.sparrow"

Write-Host "`n===== 1. k=1：f32 与 int8 必须**逐字符相同** =====" -ForegroundColor Yellow
$oneF32 = BuildRun $one $exact "i8o_f32" "f32"
$oneI8  = BuildRun $one $exact "i8o_i8" "int8"
Write-Host "  f32 =[$oneF32]"
Write-Host "  i8  =[$oneI8]"
Check "1 单元素内积：两条路逐字符相同" ($oneF32 -ceq $oneI8) "w=q/64 精确可表示 ⇒ 布局/scale/bias 全对才可能相同"

Write-Host "`n===== 2. k=4：差值有界（累加顺序不同 ⇒ 允许 ~1 ULP） =====" -ForegroundColor Yellow
$twoF32 = BuildRun $two $exact "i8t_f32" "f32"
$twoI8  = BuildRun $two $exact "i8t_i8" "int8"
$d2 = MaxDiff $twoF32 $twoI8
Write-Host "  f32 =[$twoF32]"
Write-Host "  i8  =[$twoI8]"
Write-Host "  实测最大差 = $d2"
Check "2 精确集差值 ≤ 1e-4" ($d2 -le 1e-4) "最大差 $d2（两条路只是求和顺序不同）"

Write-Host "`n===== 3. can-fail：动一行 scale ⇒ 差值必须远超第 2 条 =====" -ForegroundColor Yellow
$twoBad = BuildRun $two $badBins "i8t_bad" "int8"
$d3 = MaxDiff $twoF32 $twoBad
Write-Host "  动 a_scale[0] 后最大差 = $d3"
Check "3 动一行 scale ⇒ 差值 ≥ 第 2 条的 100×" ($d3 -gt 100 * $d2 -and $d3 -gt 1e-3) "bad=$d3 vs exact=$d2"

Write-Host "`n===== 4. 朴素量化对照：差值应明显更大 =====" -ForegroundColor Yellow
$naiveF32 = BuildRun $two $naiveBins "i8t_naive_f32" "f32"
$naiveI8  = BuildRun $two $naiveBins "i8t_naive_i8" "int8"
$d4 = MaxDiff $naiveF32 $naiveI8
Write-Host "  朴素集最大差 = $d4（精确集是 $d2）"
Check "4 朴素量化损失 > 精确集 10×" ($d4 -gt 10 * $d2 -and $d4 -gt 1e-3) "naive=$d4 exact=$d2"

Write-Host "`n===== 5. hard-fail：int8 不能走 .vmb/汇编器 =====" -ForegroundColor Yellow
$e1 = & $spar $one --asm $asm --consts-dir $exact --weight-dtype int8 -o (Join-Path $out "_x.vmb") 2>&1 | Out-String
Check "5a --asm + int8 ⇒ E109" ($e1 -match 'E109' -and $e1 -match '反量化') "$([regex]::Match($e1,'错误\[E109\][^\n]*').Value)"
$e2 = & $spar $one --backend vml --weight-dtype int8 -o (Join-Path $out "_x.vml") 2>&1 | Out-String
Check "5b --backend vml + int8 ⇒ E109" ($e2 -match 'E109' -and $e2 -match 'gemm_nt_int8') "$([regex]::Match($e2,'错误\[E109\][^\n]*').Value)"

Write-Host "`n===== 6. 体积：按公式核对（不是拍一个倍数） =====" -ForegroundColor Yellow
# int8 的常量 = q（1 B/元素）+ 每**输出行**一个 f32 scale（4 B/行）：
#     bytes_int8 = N*K + 4N      bytes_f32 = 4*N*K      ⇒ 比值 = 4 / (1 + 4/K)
# 小模型（K=4）比值只有 2×（scale 开销占比大）；K=256 时有 3.94×。
# 所以判据是"实测 == 公式"，而不是拍一个"≥3×"——后者在玩具尺寸下必然假失败。
$K = 4
$expected = 4.0 / (1.0 + 4.0 / $K)
$wF32 = (Get-Item (Join-Path $exact "a_W.bin")).Length + (Get-Item (Join-Path $exact "b_W.bin")).Length
$wI8 = (Get-Item (Join-Path $exact "a_Wq.bin")).Length + (Get-Item (Join-Path $exact "b_Wq.bin")).Length +
       (Get-Item (Join-Path $exact "a_scale.bin")).Length + (Get-Item (Join-Path $exact "b_scale.bin")).Length
$ratio = $wF32 / [double]$wI8
$asym = 4.0 / (1.0 + 4.0 / 256.0)
Write-Host "  f32 $wF32 B，int8(q+scale) $wI8 B ⇒ 比值 $([math]::Round($ratio,3))（公式 4/(1+4/$K) = $([math]::Round($expected,3))）"
Write-Host "  同一公式在 K=256（真实线性层）⇒ $([math]::Round($asym,3))×"
Check "6 实测体积比 == 公式 4/(1+4/K)" ([math]::Abs($ratio - $expected) -le 0.05 * $expected) `
    "$([math]::Round($ratio,3)) vs 期望 $([math]::Round($expected,3))（每输出行 4 B 的 scale 已计入）"

Write-Host "`n===== 7. int8 铺到**注意力投影**（V0.1.6） =====" -ForegroundColor Yellow
# 注意力里的 `Wq/Wk/Wv/Wo` 本来就是四组 `gemm_nt`，所以能吃同一个 `gemm_nt_int8`。
# 判据与线性层同源：**精确可表示**（w = q/64）时两条路必须**逐字符相同** ——
# 只有布局、每输出行 scale 的行下标、偏置位置全对才可能相同。
# （命名的坑：反量化 scale 叫 `ATTN_sWq` 而不是 `ATTN_Wq_scale`，因为 `lower()` 取常量名
#   最后一段，后者会被压成 `scale`，与 softmax 的 `ATTN_scale` 撞成同一个字段。）
$aSrc = Join-Path $root "examples\int8_attn.sparrow"
$aF32 = Join-Path $root "examples\int8_attn_bins"
$aI8 = Join-Path $root "examples\int8_attn_i8_bins"
python (Join-Path $root "tools\gen_int8_attn.py") | ForEach-Object { Write-Host "  $_" }
$atF32 = BuildRun $aSrc $aF32 "i8a_f32" "f32"
$atI8 = BuildRun $aSrc $aI8 "i8a_i8" "int8"
Write-Host "  f32 =[$atF32]"
Write-Host "  i8  =[$atI8]"
Check "7a 注意力投影：f32 与 int8 逐字符相同" ($atF32 -ceq $atI8) "4 个投影 + 线性层都走 int8，输出一字不差"

$bad = Join-Path $out "attn_bad_bins"
Remove-Item -Recurse -Force $bad -ErrorAction SilentlyContinue
Copy-Item $aI8 $bad -Recurse
$sw = Join-Path $bad "ATTN_sWq.bin"
$b = [System.IO.File]::ReadAllBytes($sw)
$v0 = [BitConverter]::ToSingle($b, 0)
[BitConverter]::GetBytes([single]($v0 + 0.015625)).CopyTo($b, 0)   # 动 2^-6
[System.IO.File]::WriteAllBytes($sw, $b)
$atBad = BuildRun $aSrc $bad "i8a_bad" "int8"
$d7 = MaxDiff $atF32 $atBad
Write-Host "  动 ATTN_sWq[0] 一个 2^-6 后最大差 = $d7"
Check "7b can-fail：动一行投影 scale ⇒ 输出必变（≥1e-3）" ($d7 -ge 1e-3) "bad=$d7（精确集是 $d2 = 0）"

$pF32 = 0; $pI8 = 0
foreach ($pk in @("Wq", "Wk", "Wv", "Wo")) {
    $pF32 += (Get-Item (Join-Path $aF32 "ATTN_$pk.bin")).Length
    $pI8 += (Get-Item (Join-Path $aI8 "ATTN_$pk.bin")).Length + (Get-Item (Join-Path $aI8 "ATTN_s$pk.bin")).Length
}
$ratioA = $pF32 / [double]$pI8
$expA = 4.0 / (1.0 + 4.0 / $K)
Check "7c 投影体积比 == 公式 4/(1+4/K)（K=$K）" ([math]::Abs($ratioA - $expA) -le 0.05 * $expA) `
    "f32 $pF32 B → int8(q+scale) $pI8 B，比值 $([math]::Round($ratioA,3))（公式 $([math]::Round($expA,3))）"

$rs = Get-Content (Join-Path $out "i8a_i8\src\main.rs") -Raw
$nCore = ([regex]::Matches($rs, 'gemm_nt_int8')).Count
Check "7d 结构：四个投影的 q/scale 字段都在，且都调了反量化内核" `
    ($nCore -ge 5 -and $rs -match 'wq: Vec<i8>' -and $rs -match 'swq: Vec<f32>' -and $rs -match 'swk: Vec<f32>' -and $rs -match 'swv: Vec<f32>' -and $rs -match 'swo: Vec<f32>') `
    "gemm_nt_int8 出现 $nCore 次；字段 wq: Vec<i8> + swq/swk/swv/swo: Vec<f32> 都在"

Write-Host "`n===== 结果 =====" -ForegroundColor Yellow
if ($script:fails -eq 0) { Write-Host "  全部 PASS" -ForegroundColor Green }
else { Write-Host "  $($script:fails) 项 FAIL" -ForegroundColor Red }
exit $script:fails
