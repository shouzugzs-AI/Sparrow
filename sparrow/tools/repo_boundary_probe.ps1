# Sparrow V0.1.5 —— 仓库边界探针（「附加仓库」与「并入 vmlm」到底差在哪）
#
# 背景：用户允许描述语言「作为附加仓库，或者与 vmlm 合并」。这句话是架构决策，
# 不该靠嘴说 —— 本探针把它变成三条可假（can-fail）的判据，全部在 out/ 下做，
# **不碰 vmlm 和 vmlmd 的任何一个文件**。
#
# 判据 1（并入不会污染父仓库的构建）：
#   造一个**忠实镜像** vmlm workspace 的父仓库（成员表逐个照抄、profile 照抄），
#   把 sparrow 整棵源码拷进 `repo\sparrow\`。父仓库 `cargo metadata` 的包列表里
#   **不得出现 sparrow**，且父仓库 `cargo build --release` 必须成功。
#   ⇒ 并入后 vmlm 的「加 member 必须让 vmlm-runtime.exe 逐字节不变」纪律不受影响。
#
# 判据 2（并入后语言自身仍能独立构建/测试）：
#   在 `repo\sparrow\` 里 `cargo test` 必须与**真实仓库**的数字完全一致（本版 85 passed / 0 failed）。
#   数字**不写死**：写死过一次 81，后来加到 85 就假失败了；判据改成"两边逐项相同"。
#   机制：sparrow/Cargo.toml 自带 `[workspace]`，自己就是 workspace 根；
#   父仓库 members 是显式列表，不会把它吸进去。
#
# 判据 3（can-fail：那一行 `[workspace]` 不是装饰）：
#   把拷贝里的 `[workspace]` 删掉，`cargo test` **必须失败**并提到 workspace。
#   这一条保证判据 2 不是「怎么写都能过」。
#
# 用法: pwsh -NoProfile -ExecutionPolicy Bypass -File tools/repo_boundary_probe.ps1
# 退出码 = 失败项数（0 = 全绿）

$ErrorActionPreference = "Continue"
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
$probe = Join-Path $root "out\_repo_probe"
$repo  = Join-Path $probe "repo"

$script:fails = 0
function Check($name, $cond, $detail) {
    if ($cond) { Write-Host "  $name : PASS  $detail" -ForegroundColor Green }
    else { Write-Host "  $name : FAIL  $detail" -ForegroundColor Red; $script:fails++ }
}

Write-Host "===== 0. 造忠实镜像的父仓库 =====" -ForegroundColor Yellow
Remove-Item -Recurse -Force $probe -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $repo | Out-Null

# 照抄 vmlm 的 workspace：members 列表 + [profile.release]
$vmlmToml = Get-Content (Join-Path $vmlm "Cargo.toml") -Raw
# 注意（实测踩到）：必须显式 @(...) 包一层再取 [0]。
# PowerShell 里 `($x | Where-Object {...})[0]` 在**只匹配到一条**时，结果是单个
# string，[0] 于是取到字符串的**第一个字符** —— 本次就生成了 `members` 位置写成
# 单个字母 `m` 的父仓库，cargo 直接 exit=101，判据 1a/1b/3 全红（假失败）。
$membersLine = @($vmlmToml -split "`n" | Where-Object { $_ -match '^members\s*=' })[0].TrimEnd()
$members = [regex]::Matches($membersLine, '"([^"]+)"') | ForEach-Object { $_.Groups[1].Value }
Write-Host "  照抄 vmlm members：$($members -join ', ')"

$parentToml = @"
[workspace]
$membersLine
resolver = "2"

[profile.release]
panic = "abort"
opt-level = 3
lto = true
codegen-units = 1
strip = true
"@
Set-Content -Path (Join-Path $repo "Cargo.toml") -Value $parentToml -Encoding ascii

# 每个 member 造一个占位 crate（只为让 workspace 能解析；不复制 vmlm 的任何代码）
foreach ($m in $members) {
    $d = Join-Path $repo $m
    New-Item -ItemType Directory -Force -Path (Join-Path $d "src") | Out-Null
    Set-Content -Path (Join-Path $d "Cargo.toml") -Encoding ascii -Value @"
[package]
name = "stub-$m"
version = "0.0.1"
edition = "2021"

[lib]
path = "src/lib.rs"
"@
    Set-Content -Path (Join-Path $d "src\lib.rs") -Encoding ascii -Value "// 占位：只为让 workspace 结构可解析`n"
}

# 把 sparrow 整棵拷进去（源码 + 测试 + examples；不拷 target/out/.git）
$dst = Join-Path $repo "sparrow"
New-Item -ItemType Directory -Force -Path $dst | Out-Null
foreach ($item in 'Cargo.toml', 'src', 'tests', 'examples') {
    Copy-Item -Recurse -Force (Join-Path $root $item) (Join-Path $dst $item)
}
Write-Host "  已拷入：$dst"

Write-Host "`n===== 1. 父仓库不认识 sparrow（并入不会改父仓库构建） =====" -ForegroundColor Yellow
Push-Location $repo
$meta = (& cargo metadata --format-version 1 --no-deps 2>&1 | Out-String)
Pop-Location
$pkgs = @()
try { $pkgs = @(($meta | ConvertFrom-Json).packages | ForEach-Object { $_.name }) } catch { }
Check "1a 父仓库包列表不含 sparrow" ($pkgs.Count -gt 0 -and ($pkgs -notcontains 'sparrow')) "packages=$($pkgs -join ',')"

Push-Location $repo
$null = & cargo build --release 2>&1 | Out-String
$pb = $LASTEXITCODE
Pop-Location
Check "1b 父仓库 cargo build --release 成功" ($pb -eq 0) "exit=$pb（占位 crate × $($members.Count) + vmlm 的 profile）"

Write-Host "`n===== 2. 并入后语言自身仍独立构建/测试 =====" -ForegroundColor Yellow
# 期望的测试数**不写死**：写死过一次 81，后来加到 85 就假失败了。
# 判据改成"与**真实仓库**的测试数一模一样"——这才是要证的东西（撤到别处跑，数量与结果都不变）。
Push-Location $root
$ref = (& cargo test 2>&1 | Out-String)
Pop-Location
$refRes = @($ref -split "`n" | Where-Object { $_ -match 'test result:' })
$refPassed = 0; $refFailed = 0
foreach ($r in $refRes) {
    if ($r -match '(\d+) passed') { $refPassed += [int]$Matches[1] }
    if ($r -match '(\d+) failed') { $refFailed += [int]$Matches[1] }
}
Push-Location $dst
$o = (& cargo test 2>&1 | Out-String)
$ct = $LASTEXITCODE
Pop-Location
$res = @($o -split "`n" | Where-Object { $_ -match 'test result:' })
$passed = 0; $failed = 0
foreach ($r in $res) {
    if ($r -match '(\d+) passed') { $passed += [int]$Matches[1] }
    if ($r -match '(\d+) failed') { $failed += [int]$Matches[1] }
}
Check "2 嵌套 sparrow cargo test 与真实仓库完全一致" `
    ($ct -eq 0 -and $failed -eq 0 -and $passed -eq $refPassed -and $refFailed -eq 0 -and $refPassed -gt 0) `
    "嵌套 $passed passed / $failed failed；真实仓库 $refPassed passed / $refFailed failed"

Write-Host "`n===== 3. can-fail：删掉 [workspace] 那一行必须报错 =====" -ForegroundColor Yellow
$tomlPath = Join-Path $dst "Cargo.toml"
$before = Get-Content $tomlPath -Raw
$after = $before -replace '(?m)^\[workspace\]\s*\r?\n', ""
Set-Content -Path $tomlPath -Value $after -Encoding utf8 -NoNewline
$hasLine = (Get-Content $tomlPath -Raw) -match '(?m)^\[workspace\]'
Push-Location $dst
$o2 = (& cargo test 2>&1 | Out-String)
$ct2 = $LASTEXITCODE
Pop-Location
$hit = $o2 -match 'workspace'
Check "3 去掉 [workspace] ⇒ 构建失败并点名 workspace" (($hasLine -eq $false) -and ($ct2 -ne 0) -and $hit) `
    "exit=$ct2，提到 workspace=$(($o2 -split "`n" | Where-Object { $_ -match 'workspace' } | Select-Object -First 1) -join '')"

Write-Host "`n===== 结果 =====" -ForegroundColor Yellow
if ($script:fails -eq 0) { Write-Host "  全部 PASS（并入 vmlm 目录这条路已实测可行；探针全程只写 out/）" -ForegroundColor Green }
else { Write-Host "  $($script:fails) 项 FAIL" -ForegroundColor Red }
exit $script:fails
