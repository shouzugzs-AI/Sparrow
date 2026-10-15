# Sparrow V0.1.3 —— 汇入（多前驱合流）验收驱动
#
# 验收项：
#   1. 相加（等宽，EWADD）：引擎输出与 numpy 独立参考一致（6 位小数）
#   2. 拼接（不等宽，CONCAT）：同上
#   3. 3 个前驱（左折叠两次 EWADD）：同上
#   4. 顺序可交换（相加）：把 `连接` 顺序颠倒后，输出必须**逐位相同**
#   5. 顺序敏感（拼接）：把 `连接` 顺序颠倒后，输出必须**不同**（can-fail：
#      若前端把拼接操作数搞错/搞反，这条会立刻发现）
#   6. can-fail 判别性：故意写错的参考值必须与正确值差 ≫ 容差（由生成器断言）
#
# 容差说明（为什么要 1e-6 而不是逐字节）：
#   引擎的 GEMM 走 AVX2 + **FMA**（乘加只舍入一次），numpy 的 `W @ x` 是
#   先舍入乘积再相加 —— 两者在末位必然差 1 ULP。所以参考值比对用 6 位小数，
#   而"接线是否接对"这条用**逐位**的 4/5 两项来判（那两项不经过 GEMM 的舍入差）。
#
# 用法: pwsh -NoProfile -ExecutionPolicy Bypass -File tools/residual_ab.ps1

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
$bins = Join-Path $root "examples\res_bins"
$spar = Join-Path $root "target\debug\sparrow.exe"
$out  = Join-Path $root "out\residual"

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
function Build($sparrow) {
    $tag = [System.IO.Path]::GetFileNameWithoutExtension($sparrow) -replace '^residual_', ''
    $r = & $spar (Join-Path $root "examples\$sparrow") --asm $asm --consts-dir $bins -o (Join-Path $out "res_$tag.vmb") 2>&1 | Out-String
    if ($r -notmatch 'OK:') { Write-Host $r; throw "汇编失败: $sparrow" }
    return $tag
}
# 参考值由 tools/gen_residual.py 里的 `eng_fmt` 产生 —— 它**逐位复刻**了引擎的
# print_f32（f32 运算 + **截断**，6 位小数，见 core/src/interp.rs:2416）。
# 所以下面的比较是**逐字符**的，不是容差比较：容差比较会放过真实的算子错误。
function NormNum($s) { return (($s -split '\s+') -join ' ') }

Write-Host "===== 0. 权重与 numpy 参考（含判别性自检） =====" -ForegroundColor Yellow
python (Join-Path $root "tools\gen_residual.py") | ForEach-Object { if ($_ -notmatch '^\s*$') { Write-Host "  $_" } }

Write-Host "`n===== 1/2/3. 三种汇入 vs 独立参考（逐字符） =====" -ForegroundColor Yellow
$map = @{ add = "resadd"; concat = "rescat"; three = "res3" }
$results = @{}
foreach ($k in @("add", "concat", "three")) {
    $tag = Build "residual_$k.sparrow"
    $r = RunModel (Join-Path $out "res_$tag.vmb")
    $ref = (Get-Content (Join-Path $out "$($map[$k]).ref.txt")).Trim()
    $results[$k] = $r[0]
    Write-Host "  $k : exit=$($r[1])  engine=$($r[0])"
    Write-Host "         ref=$ref" -ForegroundColor DarkGray
    Check "1-3 $k 与参考逐字符相同" ($r[1] -eq 0 -and (NormNum $r[0]) -ceq (NormNum $ref)) "engine == ref"
}

Write-Host "`n===== 3b. 逐元素尾巴的汇入（relu(e)+relu(e)==2*relu(e)） =====" -ForegroundColor Yellow
$tag = Build "residual_dup.sparrow"
$r = RunModel (Join-Path $out "res_$tag.vmb")
$ref = (Get-Content (Join-Path $out "resdup.ref.txt")).Trim()
Write-Host "  dup : engine=$($r[0])"
Write-Host "        ref   =$ref" -ForegroundColor DarkGray
Check "3b 逐字符相同" (($r[0] -ceq $ref)) "两个前驱逐位相同 ⇒ 相加就是精确的 2x"

Write-Host "`n===== 4. 相加可交换（前驱顺序颠倒 ⇒ 必须逐位相同） =====" -ForegroundColor Yellow
$tag = Build "residual_add_swapped.sparrow"
$r = RunModel (Join-Path $out "res_$tag.vmb")
Check "4 相加顺序无关" ($r[0] -ceq $results["add"]) "swapped=`"$($r[0])`" == add=`"$($results['add'])`""

Write-Host "`n===== 5. 拼接顺序敏感（can-fail） =====" -ForegroundColor Yellow
$tag = Build "residual_concat_swapped.sparrow"
$r = RunModel (Join-Path $out "res_$tag.vmb")
Check "5 拼接顺序有关" ($r[0] -cne $results["concat"]) "swapped=`"$($r[0])`" != concat=`"$($results['concat'])`""
Check "5b 颠倒后等于 [e;s] 的参考" ((NormNum $r[0]) -ceq (NormNum (Get-Content (Join-Path $out "rescat2.ref.txt")).Trim())) "与 rescat2 参考逐字符相同"

Write-Host "`n===== 6. 语义边界（必须有名字地拒绝） =====" -ForegroundColor Yellow
# 6a 不等宽还要相加：宽度 4 与 8 → EWADD 无定义
$bad1 = "模型 bad1:`n  嵌入 词表=8 维度=4`n  t = 稀疏 维度=4 连接=2`n  w8 = 线性 输出=8`n  mg = 线性 输出=4 合并=相加`n  连接 嵌入 t`n  连接 t w8`n  连接 嵌入 mg`n  连接 w8 mg`n  前向 嵌入 mg`n"
$f1 = Join-Path $out "_bad1.sparrow"; Set-Content -Path $f1 -Value $bad1 -Encoding utf8
$e1 = & $spar $f1 --backend vml 2>&1 | Out-String
Check "6a 不等宽相加" ($e1 -match 'E100' -and $e1 -match '相加') "$([regex]::Match($e1,'E100[^\n]*').Value)"
# 6b 该层自己写的 `输入=` 与汇入宽度冲突（拼接后是 8，却声明 4）
$bad2 = "模型 bad2:`n  嵌入 词表=8 维度=4`n  t = 稀疏 维度=4 连接=2`n  mg = 线性 输入=4 输出=4 合并=拼接`n  连接 嵌入 t`n  连接 嵌入 mg`n  连接 t mg`n  前向 嵌入 mg`n"
$f2 = Join-Path $out "_bad2.sparrow"; Set-Content -Path $f2 -Value $bad2 -Encoding utf8
$e2 = & $spar $f2 --backend vml 2>&1 | Out-String
Check "6b 声明输入与汇入宽度冲突" ($e2 -match 'E100') "$([regex]::Match($e2,'E100[^\n]*').Value)"
# 6c `重复` 组内的汇入 + 循环化：V0.1.3 起**支持**（组归一化让每份前驱结构一致），
#    所以这里改成正面判据：能编译，且与展开版逐字符相同（验收 7b）
$ok3 = "模型 ok3:`n  嵌入 词表=8 维度=4`n  aa = 线性 输出=4`n  t = 线性 输出=4 重复3次`n  连接 嵌入 aa`n  连接 aa t`n  连接 嵌入 t`n  前向 嵌入 t`n"
$f3 = Join-Path $out "_ok3.sparrow"; Set-Content -Path $f3 -Value $ok3 -Encoding utf8
$e3 = & $spar $f3 --backend vml --loop-repeat 2>&1 | Out-String
Check "6c 组内汇入 + 循环化可编译" ($e3 -match 'LOOP_CNT' -and $e3 -match 'EWADD') "生成循环 + 循环体内汇入"
# 6d 成环：嵌入→t 与 t→嵌入 同时存在 ⇒ 0 个链首
$bad4 = "模型 bad4:`n  嵌入 词表=8 维度=4`n  t = 稀疏 维度=4 连接=2`n  连接 嵌入 t`n  连接 t 嵌入`n  前向 嵌入 t`n"
$f4 = Join-Path $out "_bad4.sparrow"; Set-Content -Path $f4 -Value $bad4 -Encoding utf8
$e4 = & $spar $f4 --backend vml 2>&1 | Out-String
Check "6d 成环被拒" ($e4 -match 'E102') "$([regex]::Match($e4,'E102[^\n]*').Value)"

Write-Host "`n===== 7. 残差 + 重复组（循环化）A/B =====" -ForegroundColor Yellow
# 7a 残差在重复组**外**（汇入挂在出口层）
# 7b 残差在重复组**内**（组内每一份都有汇入前驱 —— V0.1.2 里报 E106 的那个组合）
foreach ($m in @(@("loop", "resloop", "residual_loop"), @("loopinner", "resloop2", "residual_loop_inner"))) {
    $mod = $m[0]; $refname = $m[1]
    $src = Join-Path $root "examples\$($m[2]).sparrow"
    $lp = Join-Path $out "rl_$mod.vmb"; $up = Join-Path $out "rl_${mod}_un.vmb"
    $r1 = & $spar $src --asm $asm --consts-dir $bins --loop-repeat -o $lp 2>&1 | Out-String
    $r0 = & $spar $src --asm $asm --consts-dir $bins -o $up 2>&1 | Out-String
    if ($r1 -notmatch 'OK:' -or $r0 -notmatch 'OK:') { Write-Host $r1 $r0; throw "汇编失败 $mod" }
    $n1 = [int][regex]::Match($r1, 'OK:\s*(\d+)').Groups[1].Value
    $n0 = [int][regex]::Match($r0, 'OK:\s*(\d+)').Groups[1].Value
    $a = RunModel $lp; $b = RunModel $up
    $ref = (Get-Content (Join-Path $out "$refname.ref.txt")).Trim()
    Write-Host "  $mod : loop=$n1 条 / 展开=$n0 条"
    Write-Host "        loop=$($a[0])" -ForegroundColor DarkGray
    Write-Host "        un  =$($b[0])" -ForegroundColor DarkGray
    # 核心判据：两条指令流在引擎侧**逐字符**相同（不经过参考格式）
    Check "7 $mod 循环版 == 展开版" ($a[0] -ceq $b[0]) "逐字符相同"
    Check "7b $mod 指令数更少" ($n1 -lt $n0) "loop $n1 < 展开 $n0（每层 4 条 ⇒ 48 条）"
    # 与 numpy 独立参考：十几次 GEMM 串联会把 ULP 累积到打印末位，
    # 所以这里只做**相对**比对；逐字符的那条判据是上面「两条指令流之间」。
    $d = 0.0
    $g = $b[0] -split '\s+'; $rr = $ref -split '\s+'
    for ($i = 0; $i -lt $g.Count; $i++) {
        $dd = [math]::Abs(([double]$g[$i] - [double]$rr[$i]) / [double]$rr[$i]); if ($dd -gt $d) { $d = $dd }
    }
    Check "7c $mod 与 numpy 参考一致（相对）" ($d -le 1e-5) "最大相对偏差 $d"
}

Write-Host "`n===== 结果 =====" -ForegroundColor Yellow
if ($script:fails -eq 0) { Write-Host "  全部 PASS" -ForegroundColor Green }
else { Write-Host "  $($script:fails) 项 FAIL" -ForegroundColor Red }
exit $script:fails
