# VMLM AOT 验收测试脚本（成员 B）
# 用法: pwsh -File <repo>/tools/aot_verify.ps1
# 对每个模型：
#   1) 生成 runtime/src/model_aot.rs
#   2) cargo build --release --features aot
#   3) 拍快照 exe，测体积
#   4) 在「无 .vmb 的临时目录」里跑，抓 stdout
#   5) 与「读文件版」的输出逐字节比对

$ErrorActionPreference = "Stop"

# cargo 往 stderr 写 warning，PS 5.1 会当成 NativeCommandError 中断脚本。
# 统一走 cmd.exe 包一层，把 stderr 并到 stdout，再判断 exit code。
#
# P8 变更（成员 G）：runtime/ 现在有 build.rs，而 `runtime/.cargo/config.toml`
# 的 `/NODEFAULTLIB` 会连 build script（std 宿主程序）一起干掉 → LNK1120。
# 只要 cargo 带 **显式 `--target x86_64-pc-windows-msvc`**，cargo 就区分宿主与
# 目标，config 里的 `target.*.rustflags` 只作用于目标产物，build script 回落到
# 正常宿主链接。所以这里统一带上 --target，产物路径也随之变成
# `target/x86_64-pc-windows-msvc/release\`（原先读 `target/release\`）。
$Target = "x86_64-pc-windows-msvc"

function Invoke-Cargo {
    param([string]$WorkDir, [string[]]$CargoArgs, [switch]$Quiet)
    $outFile = [System.IO.Path]::GetTempFileName()
    $q = [char]34
    # 只有在调用方没显式给 --target 时才补，避免重复
    $args2 = $CargoArgs
    if ($CargoArgs -notcontains "--target") {
        $args2 = $CargoArgs + @("--target", $Target)
    }
    $argline = ($args2 -join ' ')
    $cmd = "cd /d " + $q + $WorkDir + $q + " && cargo " + $argline + " > " + $q + $outFile + $q + " 2>&1"
    $null = & cmd.exe /c $cmd
    $code = $LASTEXITCODE
    $text = Get-Content $outFile -Raw -ErrorAction SilentlyContinue
    Remove-Item $outFile -Force -ErrorAction SilentlyContinue
    if (-not $Quiet) { Write-Host $text }
    if ($code -ne 0) { throw ("cargo " + $argline + " failed (" + $code + ")") }
    return $text
}
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Py   = "python"
$Work = Join-Path $Root "_b_aot_tmp\verify"
New-Item -ItemType Directory -Force -Path $Work | Out-Null

# 产物路径：带 --target 后在 target\<triple>/release\；
# 若那里没有（例如构建入口没走 --target），回落到 target/release\。
function Get-EngineExe {
    $scoped = Join-Path $Root "target\$Target\release\vmlm-runtime.exe"
    if (Test-Path $scoped) { return $scoped }
    return (Join-Path $Root "target/release/vmlm-runtime.exe")
}

$Models = @(
    @{ Name = "linear_relu"; Vmb = "examples/linear_relu.vmb" },
    @{ Name = "attn_block";  Vmb = "examples/attn/block.vmb"  },
    @{ Name = "sparse_ffn";  Vmb = "examples/sparse/ffn.vmb"  }
)

function Get-ExeSize($p) { (Get-Item $p).Length }

# ---------- 读文件基线 ----------
Write-Host "=== [1/3] 读文件基线 (default build) ===" -ForegroundColor Cyan
$null = Invoke-Cargo -WorkDir (Join-Path $Root "runtime") -CargoArgs @("build","--release") -Quiet
$baseExe = Get-EngineExe
$baseSize = Get-ExeSize $baseExe
$baseCopy = Join-Path $Work "baseline_default.exe"
Copy-Item $baseExe $baseCopy -Force
Write-Host "baseline exe = $baseSize B"

$results = @()

foreach ($m in $Models) {
    $name = $m.Name
    $vmb  = Join-Path $Root $m.Vmb
    $vmbSize = (Get-Item $vmb).Length

    Write-Host ""
    Write-Host "=== [$name] vmb=$vmbSize B ===" -ForegroundColor Cyan

    # 读文件版输出
    $rd = Join-Path $Work "read_$name"
    New-Item -ItemType Directory -Force -Path $rd | Out-Null
    Copy-Item $baseCopy (Join-Path $rd "run.exe") -Force
    Copy-Item $vmb (Join-Path $rd "model.vmb") -Force
    Push-Location $rd
    $readOut = (& (Join-Path $rd "run.exe") 2>&1 | Out-String)
    $readCode = $LASTEXITCODE
    Pop-Location

    # 生成内嵌常量
    Push-Location $Root
    $null = & $Py "tools/aot_build.py" $m.Vmb "runtime/src/model_aot.rs" 2>&1
    Pop-Location

    # AOT 构建
    $build = Invoke-Cargo -WorkDir (Join-Path $Root "runtime") -CargoArgs @("build","--release","--features","aot") -Quiet

    $aotExe  = Get-EngineExe
    $aotSize = Get-ExeSize $aotExe

    # 挪到「无 .vmb 的临时目录」
    $isoDir = Join-Path $Work "iso_$name"
    if (Test-Path $isoDir) { Remove-Item $isoDir -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $isoDir | Out-Null
    $isoExe = Join-Path $isoDir "$name.exe"
    Copy-Item $aotExe $isoExe -Force

    $leftover = Get-ChildItem $isoDir -File | Where-Object { $_.Extension -ne ".exe" }
    if ($leftover) { throw "isolation dir not clean for $name" }

    Push-Location $isoDir
    $aotOut = (& $isoExe 2>&1 | Out-String)
    $aotCode = $LASTEXITCODE
    Pop-Location

    $identical = ($readOut -ceq $aotOut)
    $delta = $aotSize - $baseSize
    $ratio = [math]::Round($aotSize / [double]$baseSize, 3)

    $results += [pscustomobject]@{
        Model        = $name
        VmbBytes     = $vmbSize
        ReadExe      = $baseSize
        AotExe       = $aotSize
        DeltaBytes   = $delta
        Ratio        = $ratio
        Overhead     = $delta - $vmbSize
        ReadExit     = $readCode
        AotExit      = $aotCode
        BitIdentical = $identical
        OutChars     = $aotOut.Length
    }

    Write-Host ("  read  exit={0} chars={1}" -f $readCode, $readOut.Length)
    Write-Host ("  aot   exit={0} chars={1}  exe={2} B" -f $aotCode, $aotOut.Length, $aotSize)
    Write-Host ("  bit-identical = {0}" -f $identical) -ForegroundColor $(if($identical){"Green"}else{"Red"})
    if (-not $identical) {
        Write-Host "  --- read ---"; Write-Host $readOut
        Write-Host "  --- aot  ---"; Write-Host $aotOut
    }
}

Write-Host ""
Write-Host "=== 汇总 ===" -ForegroundColor Yellow
$results | Format-Table -AutoSize

# 落盘结果
$results | ConvertTo-Json -Depth 4 | Set-Content (Join-Path $Work "aot_results.json") -Encoding UTF8
Write-Host "results -> $(Join-Path $Work 'aot_results.json')"
