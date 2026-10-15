# Sparrow V0.1.5 —— ⑤ `序列=T>1`（**嵌入链首**）验收驱动
#
# 背景：V0.1.3 起 `序列=` 能进 IR，但链首只能是"逐行"的 `线性`；
# `嵌入` 链首 + `序列>1` 直接报 `E107`（`EMB` 一次只查一个行号）。
# V0.1.5 把 `嵌入` 链首做成**逐行查表**：T 次 `SLICE`+`EMB` → 每行转列 →
# 按列左折叠 `CONCAT` 成 [D,T] → 一次 `TRANSPOSE` 得到 [T,D]。
#
# 判据（每条都能假）：
#   1 **逐行独立性**：`se3`（序列=3）输出的第 k 行 == `se1`（不写序列）只喂第 k 个
#     词元时的输出，**逐字符**相同。两侧都是引擎自己算的 ⇒ 不依赖任何第三方库。
#   2 numpy 独立参考：每行的参考值用 numpy 另算（复刻引擎的**截断**式打印）。
#   3 can-fail：把 `se3_in.bin` 的第 2 个词元换一个 ⇒ **只有第 2 行**变，
#     第 1/3 行逐字符不变（若拼接顺序错位，"只有一行变"这条会立刻红）。
#   4 Rust 后端（`--backend lib`）与 `.vmb` 输出**逐字符**相同（T>1 也走同一条路）。
#   5 结构：T>1 时链首必须展开成 **T 次** `EMB`（不是 1 次）。
#
# 用法: pwsh -NoProfile -ExecutionPolicy Bypass -File tools/seq_ab.ps1
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
$bins = Join-Path $root "examples\seq_bins"
$out  = Join-Path $root "out\seq"
# 自带的引擎运行目录：**不依赖别的驱动留在 out/ 里的残留**。
# 合并布局暴露过这个隐藏依赖：原来指向 out\aot_bench\engine_plain —— 那是 aot_bench
# 的产物，只有跑过一次 aot_bench 才存在，新克隆/新工作区下必然失败。
$engDir = Join-Path $out "engine"

New-Item -ItemType Directory -Force -Path $out | Out-Null
New-Item -ItemType Directory -Force -Path $engDir | Out-Null
Copy-Item (EngineBin "vmlm-runtime.exe") (Join-Path $engDir "vmlm-runtime.exe") -Force
$script:fails = 0
function Check($name, $cond, $detail) {
    if ($cond) { Write-Host "  $name : PASS  $detail" -ForegroundColor Green }
    else { Write-Host "  $name : FAIL  $detail" -ForegroundColor Red; $script:fails++ }
}
function EngineRun($vmb) {
    Copy-Item $vmb (Join-Path $engDir "model.vmb") -Force
    Push-Location $engDir
    try { $t = (& (Join-Path $engDir "vmlm-runtime.exe") 2>&1 | Out-String).Trim(); $c = $LASTEXITCODE } finally { Pop-Location }
    return @($t, $c)
}
function Build($src, $tag, $extra = @()) {
    $vmb = Join-Path $out "$tag.vmb"
    $r = & $spar $src --asm $asm --consts-dir $bins @extra -o $vmb 2>&1 | Out-String
    if ($r -notmatch 'OK:') { throw "汇编失败 $tag：$r" }
    return $vmb
}
# 取第 k 行（`$dim` 个数）。
# 注意：范围运算符 `..` 的优先级**低于** `*` ⇒ 起始下标必须加括号，
# 否则 PowerShell 先去算 `数组 * 整数`（实测报 "does not contain a method named op_Multiply"）。
function RowOf($v, $k, $dim) {
    return (($v[($k * $dim)..(($k + 1) * $dim - 1)]) -join ' ')
}

Write-Host "===== 0. 权重（确定性 + 判别性自检） =====" -ForegroundColor Yellow
python (Join-Path $root "tools\gen_seq.py") | ForEach-Object { Write-Host $_ }

$se3 = Join-Path $root "examples\seq_embed.sparrow"
$se1 = Join-Path $root "examples\seq_one.sparrow"
$tokens = @(3, 5, 1)
$dim = 4

Write-Host "`n===== 1. T=3 vs 逐词元的 T=1（逐行独立性，逐字符） =====" -ForegroundColor Yellow
$v3 = Build $se3 "se3"
$e3 = EngineRun $v3
Check "1a se3 跑通" ($e3[1] -eq 0) "exit=$($e3[1])"
$vals = @($e3[0] -split '\s+' | Where-Object { $_ -ne '' })
Check "1b 输出行数 = 序列数" ($vals.Count -eq 3 * $dim) "T=3 ⇒ $($vals.Count) 个数（$([math]::Round($vals.Count / $dim)) 行）"

$v1 = Build $se1 "se1"
for ($k = 0; $k -lt 3; $k++) {
    # 把 se1 的输入改写成第 k 个词元（同一份权重，只换输入常量）。
    # **必须重新汇编**：输入常量是烧进 `.vmb` 的，改 `.bin` 不会改已有产物
    # （第一次就是这么假失败的：三次都读到词元 3 的 T=1 输出）。
    python -c @"
import struct, sys
open(sys.argv[1],'wb').write(struct.pack('<f', float(sys.argv[2])))
"@ (Join-Path $bins "se1_in.bin") $tokens[$k]
    $v1 = Build $se1 "se1"
    $e1 = EngineRun $v1
    $row = RowOf $vals $k $dim
    Check "1.$k 第 $k 行 == 词元 $($tokens[$k]) 的 T=1 输出" ($row -ceq $e1[0] -and $e1[1] -eq 0) "row=[$row] t1=[$($e1[0])]"
}

Write-Host "`n===== 2. numpy 独立参考（复刻截断式打印） =====" -ForegroundColor Yellow
for ($k = 0; $k -lt 3; $k++) {
    $ref = (Get-Content (Join-Path $out "tok_$($tokens[$k]).ref.txt") -Raw).Trim()
    $row = RowOf $vals $k $dim
    # 引擎的 GEMM 是 AVX2+FMA 8×8 分块，numpy 的 f32 点积顺序不一定相同 ⇒
    # 按数值比较（打印文本必然同格式，只可能在末位差 1）。差距如实打印出来。
    $a = @($row -split ' ' | ForEach-Object { [double]$_ })
    $b = @($ref -split ' ' | ForEach-Object { [double]$_ })
    $gap = 0.0
    for ($j = 0; $j -lt $dim; $j++) { $gap = [math]::Max($gap, [math]::Abs($a[$j] - $b[$j])) }
    Check "2.$k 第 $k 行与 numpy 参考一致（≤1e-5）" ($gap -le 1e-5) "最大差 $gap；ref=[$ref]"
}

Write-Host "`n===== 3. can-fail：只改第 2 个词元 ⇒ 只有第 2 行变 =====" -ForegroundColor Yellow
python -c @"
import struct, sys
p = sys.argv[1]
b = bytearray(open(p,'rb').read())
struct.pack_into('<f', b, 4, 6.0)   # 第 2 个词元 5 → 6
open(p,'wb').write(bytes(b))
print('  se3_in[1] -> 6.0')
"@ (Join-Path $bins "se3_in.bin")
$v3b = Build $se3 "se3_mut"
$e3b = EngineRun $v3b
$vals2 = @($e3b[0] -split '\s+' | Where-Object { $_ -ne '' })
$r0 = RowOf $vals 0 $dim; $r0b = RowOf $vals2 0 $dim
$r1 = RowOf $vals 1 $dim; $r1b = RowOf $vals2 1 $dim
$r2 = RowOf $vals 2 $dim; $r2b = RowOf $vals2 2 $dim
Check "3a 第 1 行不变" ($r0 -ceq $r0b) "改的是第 2 个词元"
Check "3b 第 2 行变了" ($r1 -cne $r1b) "第 2 行必须跟着变（否则说明拼接没按词元走）"
Check "3c 第 3 行不变" ($r2 -ceq $r2b) "第 3 行不该受影响"
# 复原输入，供后续判据用
python -c @"
import struct, sys
open(sys.argv[1],'wb').write(struct.pack('<3f', 3.0, 5.0, 1.0))
"@ (Join-Path $bins "se3_in.bin")

Write-Host "`n===== 4. Rust 后端（--backend lib）与 .vmb 逐字符相同 =====" -ForegroundColor Yellow
$crate = Join-Path $out "rs"
Remove-Item -Recurse -Force $crate -ErrorAction SilentlyContinue
& $spar $se3 --backend lib --lang rust --vmlm-root $vmlm --consts-dir $bins -o $crate | Out-Null
$env:CARGO_TARGET_DIR = Join-Path $out "target"
$cb = (& cargo build --release --quiet --manifest-path (Join-Path $crate "Cargo.toml") 2>&1 | Out-String)
$exe = Join-Path $env:CARGO_TARGET_DIR "release\rs.exe"
if ($LASTEXITCODE -ne 0 -or -not (Test-Path $exe)) { Write-Host $cb; Check "4 Rust 构建" $false "失败" }
else {
    $rust = (& $exe 2>&1 | Out-String).Trim()
    Check "4 Rust 输出 == .vmb 输出（逐字符）" ($rust -ceq $e3[0]) "rust=[$rust]"
}

Write-Host "`n===== 5. 结构：T>1 的链首要展开成 T 次 EMB =====" -ForegroundColor Yellow
$vml3 = Join-Path $out "se3.vml"
$emb = ([regex]::Matches((Get-Content $vml3 -Raw), '(?m)^\s*EMB\s+')).Count
Check "5 链首有 3 次 EMB（不是 1 次）" ($emb -eq 3) "实测 $emb 次"

Write-Host "`n===== 6. 序列=T>1 **与循环组同时出现**：循环版 == 展开版 =====" -ForegroundColor Yellow
# V0.1.4 这里把循环进位槽的行数写死成 1 ⇒ 循环体只按 1 行算，最终只打印 4 个数
# （而不是 12 个），退出码 0 —— 静默错答案。判据就是 E7 那条：循环版 == 展开版。
$sl = Join-Path $root "examples\seq_loop.sparrow"
$vUn = Build $sl "seq_loop_unrolled"
$vLp = Build $sl "seq_loop" @("--loop-repeat")
$eUn = EngineRun $vUn
$eLp = EngineRun $vLp
$cells = @($eLp[0] -split '\s+' | Where-Object { $_ -ne '' }).Count
Check "6a 循环版输出个数 = 3 行 × $dim" ($cells -eq 3 * $dim) "T=3 ⇒ $cells 个数（写死 1 行时只有 $dim 个）"
Check "6b 循环版 == 展开版（逐字符）" ($eUn[0] -ceq $eLp[0] -and $eLp[1] -eq 0) "loop=[$($eLp[0])]"

Write-Host "`n===== 结果 =====" -ForegroundColor Yellow
if ($script:fails -eq 0) { Write-Host "  全部 PASS" -ForegroundColor Green }
else { Write-Host "  $($script:fails) 项 FAIL" -ForegroundColor Red }
exit $script:fails
