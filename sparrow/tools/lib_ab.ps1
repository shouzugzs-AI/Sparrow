# Sparrow V0.1.4 —— 路径 2 后端（`.spar → Rust AOT 源`）验收驱动
#
# 判据：**与 `.vmb` 后端逐字符一致**（同权重、同输入）。
#
# 为什么这条判据能成立（而不是"差不多"）：生成的 Rust 直接
# `use vmlm_ops::kernels::*`，调**同一个函数**（gemm_nt / layernorm / act_avx2 /
# softmax），其余逐元素/搬运算子按 `interp.rs` 的同一条标量规则手写。
# 朴素循环版本实测差 1 ULP —— 过不了"逐字符"，所以必须同源。
#
# 验收项：
#   1. 每个模型：Rust 后端输出 == `.vmb` 后端输出（逐字符）—— 含 3 个**循环化**的
#      `重复` 组（loop24 / resloop / attnloop）；loop24 是展开版根本汇编不出来的例子
#   2. can-fail：把某个权重文件改一个字节 ⇒ Rust 输出必须**变化**
#      （否则"相同"可能是两边都没算对）
#   3. 循环组结构：源码必须是「一份代码 + N 份常量」（长度与 N 无关），
#      且 `LAYERB` 的实现常量不内嵌进 Rust
#   4. hard-fail：参数错误（缺 --vmlm-root / 未知 --lang）报清楚，不静默降级
#
# 用法: pwsh -NoProfile -ExecutionPolicy Bypass -File tools/lib_ab.ps1

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
$out  = Join-Path $root "out\lib"

New-Item -ItemType Directory -Force -Path $out | Out-Null
$script:fails = 0
function Check($name, $cond, $detail) {
    if ($cond) { Write-Host "  $name : PASS  $detail" -ForegroundColor Green }
    else { Write-Host "  $name : FAIL  $detail" -ForegroundColor Red; $script:fails++ }
}
function EngineOut($vmb) {
    Copy-Item $vmb (Join-Path $rel "model.vmb") -Force
    Push-Location $rel
    try { $t = (& $eng 2>&1 | Out-String).Trim(); $c = $LASTEXITCODE } finally { Pop-Location }
    return @($t, $c)
}
function RustOut($src, $bins, $tag, $extra = @()) {
    $dir = Join-Path $out $tag
    Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
    & $spar $src --backend lib --lang rust --vmlm-root $vmlm --consts-dir $bins @extra -o $dir | Out-Null
    if (-not (Test-Path (Join-Path $dir "src\main.rs"))) { throw "没生成 $tag" }
    # 先编译（编译输出丢掉），再**直接跑产物**：否则 build 的 warning 会混进被比较的
    # 字符串，把"逐字符相同"变成永远失败的噪声（第一次就是这么假失败的）。
    $null = cargo build --release --quiet --manifest-path (Join-Path $dir "Cargo.toml") 2>&1
    $exe = Join-Path $env:CARGO_TARGET_DIR ("release\" + (Split-Path $dir -Leaf) + ".exe")
    if (-not (Test-Path $exe)) { throw "没编出 $exe" }
    return (& $exe 2>&1 | Out-String).Trim()
}

# 全驱动共用一个 CARGO_TARGET_DIR：`vmlm-ops` 只编一次（否则 9 个 crate 各编一遍）
$env:CARGO_TARGET_DIR = Join-Path $out "target"

# 每个模型：源、权重目录、说明
$cases = @(
    @{ src = "examples\e2e_tiny.sparrow";       bins = "examples\e2e_bins";  tag = "tiny";     extra = @() },
    @{ src = "examples\residual_add.sparrow";   bins = "examples\res_bins";  tag = "resadd";   extra = @() },
    @{ src = "examples\residual_three.sparrow"; bins = "examples\res_bins";  tag = "res3";     extra = @() },
    @{ src = "examples\residual_dup.sparrow";   bins = "examples\res_bins";  tag = "resdup";   extra = @() },
    @{ src = "examples\attn_t1.sparrow";        bins = "examples\attn_bins"; tag = "attn1";    extra = @() },
    @{ src = "examples\attn_causal.sparrow";    bins = "examples\attn_bins"; tag = "attnc";    extra = @() },
    # V0.1.5：循环化的 `重复` 组也必须逐字符一致。
    # loop24 是**只能**走循环的那个例子：24 层稀疏的权重槽号超过 63（TRAPS B9），
    # 展开版根本汇编不出来 ⇒ 这里顺手证明"循环版是唯一可行路径，而且它是对的"。
    @{ src = "examples\loop24.sparrow";         bins = "examples\loop_bins"; tag = "loop24";   extra = @("--loop-repeat") },
    @{ src = "examples\residual_loop.sparrow";  bins = "examples\res_bins";  tag = "resloop";  extra = @("--loop-repeat") },
    @{ src = "examples\attn_loop.sparrow";      bins = "examples\attn_bins"; tag = "attnloop"; extra = @("--loop-repeat") },
    # 两个循环组**串联**（补 V0.1.2 起就存在的覆盖缺口，见 examples/loop_two.sparrow）
    @{ src = "examples\loop_two.sparrow";       bins = "examples\loop_two_bins"; tag = "loop2"; extra = @("--loop-repeat") }
)

Write-Host "===== 1. Rust 后端 vs .vmb 后端（逐字符） =====" -ForegroundColor Yellow
foreach ($c in $cases) {
    $src = Join-Path $root $c.src
    $bins = Join-Path $root $c.bins
    if (-not (Test-Path $bins)) { Write-Host "  SKIP $($c.tag)：缺 $bins" -ForegroundColor DarkGray; continue }
    $vmb = Join-Path $out "$($c.tag).vmb"
    $r = & $spar $src --asm $asm --consts-dir $bins @($c.extra) -o $vmb 2>&1 | Out-String
    if ($r -notmatch 'OK:') { Write-Host $r; throw "汇编失败 $($c.tag)" }
    $e = EngineOut $vmb
    $u = RustOut $src $bins $c.tag @($c.extra)
    Write-Host "  $($c.tag): .vmb=[$($e[0])]"
    Write-Host "           rust=[$u]" -ForegroundColor DarkGray
    Check "1 $($c.tag) 逐字符相同" ($e[0] -ceq $u -and $e[1] -eq 0) "两条后端同一套内核 + 同一套权重$(if ($c.extra) { '（循环化）' } else { '' })"
}

Write-Host "`n===== 2. can-fail：改一个权重字节必须让输出变化 =====" -ForegroundColor Yellow
# 复制一份权重目录，把 LINEAR_W 的第一个 f32 加 1.0
$mut = Join-Path $out "mut_bins"
Remove-Item -Recurse -Force $mut -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $mut | Out-Null
Copy-Item (Join-Path $root "examples\e2e_bins\*") $mut -Force
python -c @"
import struct, sys
p = sys.argv[1]
b = bytearray(open(p,'rb').read())
v = struct.unpack_from('<f', b, 0)[0] + 1.0
struct.pack_into('<f', b, 0, v)
open(p,'wb').write(bytes(b))
print(f'  LINEAR_W[0]: -> {v:.6f}')
"@ (Join-Path $mut "LINEAR_W.bin")
$base = RustOut (Join-Path $root "examples\e2e_tiny.sparrow") (Join-Path $root "examples\e2e_bins") "tiny"
$changed = RustOut (Join-Path $root "examples\e2e_tiny.sparrow") $mut "tiny_mut"
Check "2 改权重 ⇒ 输出变化" ($changed -cne $base) "base=[$base] mut=[$changed]"

Write-Host "`n===== 3. 循环组：源码必须是「一份代码 + N 份常量」 =====" -ForegroundColor Yellow
# 这一节盯的是**结构**（数值一致性已在上面的 1-loop24 / 1-resloop / 1-attnloop 验过）：
# 循环化的意义就是源码长度与 N 无关 —— 展开 24 层会生成 24 个 struct。
$loopRs = Get-Content (Join-Path $out "loop24\src\main.rs") -Raw
$fwdPat = 'fn forward\(&self, x: &\[f32\], rows: usize\)'
$fwdCount = ([regex]::Matches($loopRs, $fwdPat)).Count
# 注意：Check 的 detail 是**双引号字符串**，里面不能出现反引号 ——
# 「反引号 + 引号」会被 PowerShell 当成转义引号，整段字符串的终止符就丢了
#（本次实测：语法错误落在文件最后一行，报"字符串缺少终止符"，很难定位）。
Check "3a 生成 for 循环（不是展开 N 份）" ($loopRs -match 'for inst in &self\.g0\.items') "loop24 的 Rust 里有 for inst in &self.g0.items"
$inlined = @(1..24 | Where-Object { $loopRs -match "C_t__${_}_W_RAW" }).Count
Check "3b 24 份实例权重全部内嵌" ($inlined -eq 24) "内嵌 $inlined/24 份（每份一个 include_bytes!）"
Check "3c 循环控制常量没进 Rust" (-not ($loopRs -match '_lstride') -and -not ($loopRs -match '_nstride')) `
    "_lstride/_nstride 是 LAYERB 的实现细节，Rust 版直接遍历实例、不需要它们"
Check "3d 代码份数与 N 无关" ($fwdCount -lt 24 -and $fwdCount -ge 2) `
    "24 层模型只有 $fwdCount 份 forward（1 份循环体 + 组外的层）"

# 3e：**两个循环组串联**时，循环版输出必须等于展开版输出（E7 的支点判据）。
# 为什么这条必须存在：Rust 后端与 `.vmb` 后端都是我这边生成的，"两边一致"只说明
# 两个实现互相同意；而"循环 == 展开"是把它们钉在**同一个函数语义**上的那条判据
# （组归一化保证每份实例前驱结构一致 ⇒ 循环与展开算的是同一个函数）。
$l2src = Join-Path $root "examples\loop_two.sparrow"
$l2bins = Join-Path $root "examples\loop_two_bins"
$vmbUn = Join-Path $out "loop2_unrolled.vmb"
$null = & $spar $l2src --asm $asm --consts-dir $l2bins -o $vmbUn 2>&1 | Out-String
$eu = EngineOut $vmbUn
$el = EngineOut (Join-Path $out "loop2.vmb")
Check "3e 两个循环组：循环版 == 展开版（引擎输出）" ($eu[0] -ceq $el[0] -and $eu[1] -eq 0) `
    "loop=[$($el[0])] unrolled=[$($eu[0])]"

Write-Host "`n===== 4. hard-fail：参数错误必须报清楚，不静默降级 =====" -ForegroundColor Yellow
$e2 = & $spar (Join-Path $root "examples\e2e_tiny.sparrow") --backend lib --lang rust -o (Join-Path $out "x") 2>&1 | Out-String
Check "4a 缺 --vmlm-root 报清楚" ($e2 -match 'E109' -and $e2 -match 'vmlm-root') "$([regex]::Match($e2,'错误\[E109\][^\n]*').Value)"
$e3 = & $spar (Join-Path $root "examples\e2e_tiny.sparrow") --backend lib --lang python 2>&1 | Out-String
Check "4b 未知 --lang 报清楚" ($e3 -match 'lang') "$([regex]::Match($e3,'用法错误[^\n]*').Value)"

Write-Host "`n===== 结果 =====" -ForegroundColor Yellow
if ($script:fails -eq 0) { Write-Host "  全部 PASS" -ForegroundColor Green }
else { Write-Host "  $($script:fails) 项 FAIL" -ForegroundColor Red }
exit $script:fails
