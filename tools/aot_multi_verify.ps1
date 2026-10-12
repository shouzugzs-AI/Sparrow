# VMLM AOT 多模型验收脚本（P8 成员 G）
#
# 用法: pwsh -File <repo>/tools/aot_multi_verify.ps1
#
# 干什么：
#   1) 生成 runtime/src/models_aot.rs（N 个模型）
#   2) cd runtime; cargo build --release --features aot-multi
#   3) 把产物挪到「只有 .exe 的隔离目录」（证明不读任何外部文件）
#   4) 对每个模型：
#        a) 读文件版基线（default 构建 + model.vmb）-> stdout
#        b) AOT 多模型版（--model=<name>）        -> stdout
#        c) 两者字节数 / SHA256 比对
#   5) 报体积：内嵌 1 个 vs N 个
#
# 判据（可判定）：每个模型的 stdout 逐字节一致 **且** SHA256 相同，
#                 且隔离目录里 .vmb 计数必须为 0。

$ErrorActionPreference = "Stop"

# runtime/ 现在有 build.rs，而 config.toml 的 /NODEFAULTLIB 会连 build script
# （std 宿主程序）一起干掉 -> LNK1120。必须显式带 --target 让 cargo 区分
# 宿主与目标，config 里的 target.*.rustflags 才只作用于目标产物。
$Target = "x86_64-pc-windows-msvc"

function Invoke-Cargo {
    param([string]$WorkDir, [string[]]$CargoArgs)
    $outFile = [System.IO.Path]::GetTempFileName()
    $q = [char]34
    $args2 = $CargoArgs
    if ($CargoArgs -notcontains "--target") { $args2 = $CargoArgs + @("--target", $Target) }
    $cmd = "cd /d " + $q + $WorkDir + $q + " && cargo " + ($args2 -join ' ') + " > " + $q + $outFile + $q + " 2>&1"
    $null = & cmd.exe /c $cmd
    $code = $LASTEXITCODE
    $text = Get-Content $outFile -Raw -ErrorAction SilentlyContinue
    Remove-Item $outFile -Force -ErrorAction SilentlyContinue
    if ($code -ne 0) {
        Write-Host $text
        throw ("cargo " + ($args2 -join ' ') + " failed (" + $code + ")")
    }
    return $text
}

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Py   = "python"
$Work = Join-Path $Root "_g_aot_tmp\verify"
New-Item -ItemType Directory -Force -Path $Work | Out-Null

# 产物路径：带 --target 后在 target\<triple>/release\
function Get-EngineExe {
    return (Join-Path $Root "target\$Target\release\vmlm-runtime.exe")
}

# 被测模型表（名字 -> .vmb 相对路径）。默认模型放第一。
$Models = @(
    @{ Name = "linear_relu"; Vmb = "examples/linear_relu.vmb"    },
    @{ Name = "attn";        Vmb = "examples/attn/block.vmb"     },
    @{ Name = "sparse";      Vmb = "examples/sparse/ffn.vmb"     },
    @{ Name = "p8stack";     Vmb = "examples/p8_stack_test.vmb"  },
    @{ Name = "nnreg";       Vmb = "examples/p7/nn_regress.vmb"  }
)

function Get-Sha256([byte[]]$bytes) {
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($sha.ComputeHash($bytes)) -replace '-','') }
    finally { $sha.Dispose() }
}
# 输出必须按**原始字节**取，不能走 PowerShell 的字符串管道（会做编码转换）。
#
# 关键：必须在 `$WorkDir` 里执行 —— 读文件版基线是按相对路径 "model.vmb"
# 打开文件的，cwd 不对就会 "cannot open file"（exit=2）。这里是踩过的坑：
# 早先用 `& $exe` 直接在调用者 cwd 里跑，基线全部打不开，被误报成
# 「逐位不一致」，其实 AOT 侧一直是对的。
function Run-Capture($exe, $argList, $WorkDir) {
    $tmp = [System.IO.Path]::GetTempFileName()
    $q = [char]34
    $argpart = if ($null -eq $argList) { "" } else { $argList }
    $inner = "cd /d $q$WorkDir$q && $q$exe$q $argpart > $q$tmp$q 2>&1"
    $null = & cmd.exe /c $inner
    $code = $LASTEXITCODE
    $bytes = [System.IO.File]::ReadAllBytes($tmp)
    Remove-Item $tmp -Force -ErrorAction SilentlyContinue
    return @{ Code = $code; Bytes = $bytes }
}

$results = @()

# ---------- 1) 读文件基线（default 构建） ----------
Write-Host "=== [1/4] 读文件基线 (default build) ===" -ForegroundColor Cyan
$null = Invoke-Cargo -WorkDir (Join-Path $Root "runtime") -CargoArgs @("build","--release")
$baseExe = Get-EngineExe
$baseSize = (Get-Item $baseExe).Length
$baseCopy = Join-Path $Work "baseline_default.exe"
Copy-Item $baseExe $baseCopy -Force
Write-Host "baseline exe = $baseSize B"

# ---------- 2) 用 N 个模型构建 aot-multi ----------
Write-Host ""
Write-Host "=== [2/4] 内嵌 $($Models.Count) 个模型并构建 aot-multi ===" -ForegroundColor Cyan
$specs = @()
foreach ($m in $Models) { $specs += ("{0}={1}" -f $m.Name, $m.Vmb) }

# 模型集由 VMLM_AOT_MODELS 驱动 build.rs。
$absSpecs = ($specs | ForEach-Object {
    $kv = $_ -split '=', 2
    "{0}={1}" -f $kv[0], (Join-Path $Root $kv[1])
}) -join ';'
$env:VMLM_AOT_MODELS = $absSpecs
try {
    $null = Invoke-Cargo -WorkDir (Join-Path $Root "runtime") -CargoArgs @("build","--release","--features","aot-multi")
} finally {
    Remove-Item Env:\VMLM_AOT_MODELS -ErrorAction SilentlyContinue
}
$multiExe = Get-EngineExe
$multiSize = (Get-Item $multiExe).Length
Write-Host "aot-multi exe = $multiSize B  (delta vs baseline = $($multiSize - $baseSize) B)"

# 隔离目录：只放 exe，绝不能有 .vmb
$iso = Join-Path $Work "iso_multi"
if (Test-Path $iso) { Remove-Item $iso -Recurse -Force }
New-Item -ItemType Directory -Force -Path $iso | Out-Null
Copy-Item $multiExe (Join-Path $iso "multi.exe") -Force
$vmbCount = (Get-ChildItem $iso -File -Filter *.vmb -ErrorAction SilentlyContinue | Measure-Object).Count
Write-Host "isolation dir .vmb count = $vmbCount (must be 0)"
if ($vmbCount -ne 0) { throw "isolation dir not clean" }
Write-Host ("isolation dir files: " + ((Get-ChildItem $iso -File | ForEach-Object { $_.Name }) -join ", "))

# --list 冒烟
$lst = Run-Capture (Join-Path $iso "multi.exe") "--list" $iso
Write-Host ""
Write-Host "--list 输出:"
Write-Host ([System.Text.Encoding]::UTF8.GetString($lst.Bytes))

# ---------- 3) 逐模型比对 ----------
Write-Host "=== [3/4] 逐模型 逐位一致 比对 ===" -ForegroundColor Cyan
foreach ($m in $Models) {
    $name = $m.Name
    $vmb  = Join-Path $Root $m.Vmb
    $vmbSize = (Get-Item $vmb).Length

    # 读文件版：基线 exe + 同名 model.vmb，在同一隔离目录里跑
    $rd = Join-Path $Work "read_$name"
    if (Test-Path $rd) { Remove-Item $rd -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $rd | Out-Null
    Copy-Item $baseCopy (Join-Path $rd "run.exe") -Force
    Copy-Item $vmb (Join-Path $rd "model.vmb") -Force
    $readRes = Run-Capture (Join-Path $rd "run.exe") $null $rd

    # AOT 多模型版：--model=<name>，在无 .vmb 的隔离目录里跑
    $aotRes = Run-Capture (Join-Path $iso "multi.exe") "--model=$name" $iso

    $readSha = Get-Sha256 $readRes.Bytes
    $aotSha  = Get-Sha256 $aotRes.Bytes
    $identical = ($readSha -ceq $aotSha)

    $results += [pscustomobject]@{
        Model        = $name
        VmbBytes     = $vmbSize
        ReadExit     = $readRes.Code
        AotExit      = $aotRes.Code
        ReadBytes    = $readRes.Bytes.Length
        AotBytes     = $aotRes.Bytes.Length
        ReadSha256   = $readSha.Substring(0,16)
        AotSha256    = $aotSha.Substring(0,16)
        BitIdentical = $identical
    }

    Write-Host ""
    Write-Host ("  [{0}] vmb={1} B" -f $name, $vmbSize)
    Write-Host ("    read exit={0} bytes={1} sha256={2}..." -f $readRes.Code, $readRes.Bytes.Length, $readSha.Substring(0,16))
    Write-Host ("    aot  exit={0} bytes={1} sha256={2}..." -f $aotRes.Code, $aotRes.Bytes.Length, $aotSha.Substring(0,16))
    Write-Host ("    bit-identical = {0}" -f $identical) -ForegroundColor $(if($identical){"Green"}else{"Red"})
    if (-not $identical) {
        Write-Host "    --- read ---"; Write-Host ([System.Text.Encoding]::ASCII.GetString($readRes.Bytes))
        Write-Host "    --- aot  ---"; Write-Host ([System.Text.Encoding]::ASCII.GetString($aotRes.Bytes))
    }
}

# ---------- 4) 体积：1 个 vs N 个 ----------
Write-Host ""
Write-Host "=== [4/4] 体积对比：内嵌 1 个 vs $($Models.Count) 个 ===" -ForegroundColor Cyan
# 关键：模型集由 **VMLM_AOT_MODELS 环境变量**驱动 build.rs，不是写 src/models_aot.rs
# （后者只给 --cfg vmlm_aot_offline 用）。早先我改了 src/models_aot.rs 却没改
# 环境变量，于是两次构建的模型集完全一样 -> 体积表报出 "增 0.0000 字节" 的假数据。
function Build-WithModels($specList) {
    # 传绝对路径，避免相对 CARGO_MANIFEST_DIR 的歧义
    $joined = ($specList | ForEach-Object {
        $kv = $_ -split '=', 2
        "{0}={1}" -f $kv[0], (Join-Path $Root $kv[1])
    }) -join ';'
    $env:VMLM_AOT_MODELS = $joined
    try {
        $null = Invoke-Cargo -WorkDir (Join-Path $Root "runtime") -CargoArgs @("build","--release","--features","aot-multi")
    } finally {
        Remove-Item Env:\VMLM_AOT_MODELS -ErrorAction SilentlyContinue
    }
}

$oneSpec = @("{0}={1}" -f $Models[0].Name, $Models[0].Vmb)
Build-WithModels $oneSpec
$oneExe = Get-EngineExe
$oneSize = (Get-Item $oneExe).Length

# 重新用 N 个模型构建，保持后续/仓库状态与主验收结果一致
Build-WithModels $specs
$multiSize = (Get-Item (Get-EngineExe)).Length

$sum1 = (Get-Item (Join-Path $Root $Models[0].Vmb)).Length
$sumN = ($Models | ForEach-Object { (Get-Item (Join-Path $Root $_.Vmb)).Length } | Measure-Object -Sum).Sum

Write-Host ("  1 model  ({0} B payload): exe = {1} B" -f $sum1, $oneSize)
Write-Host ("  {0} models ({1} B payload): exe = {2} B" -f $Models.Count, $sumN, $multiSize)
Write-Host ("  每多内嵌 1 字节模型 -> exe 增 {0:N4} 字节" -f (($multiSize - $oneSize) / [double]($sumN - $sum1)))

Write-Host ""
Write-Host "=== 汇总 ===" -ForegroundColor Yellow
$results | Format-Table -AutoSize

$allOk = ($results | Where-Object { -not $_.BitIdentical } | Measure-Object).Count -eq 0
Write-Host ""
if ($allOk) {
    Write-Host "判定: PASS —— $($Models.Count) 个模型全部与读文件版逐位一致" -ForegroundColor Green
} else {
    Write-Host "判定: FAIL" -ForegroundColor Red
}

$out = [pscustomobject]@{
    BaseExe   = $baseSize
    OneExe    = $oneSize
    MultiExe  = $multiSize
    Count     = $Models.Count
    Sum1      = $sum1
    SumN      = $sumN
    Results   = $results
}
$out | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $Work "aot_multi_results.json") -Encoding UTF8
Write-Host "results -> $(Join-Path $Work 'aot_multi_results.json')"
exit $(if ($allOk) { 0 } else { 1 })
