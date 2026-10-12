# 打包单文件 EXE：对每个权重编一个 vmlm-chat 变体，并**在空文件夹里实测**。
# 关键防错：打包前**先载入该 .cpk 量一次 val**（--attn-calib 0，不改权重），
#           val 超过阈值就**拒绝打包**（-MaxVal，默认 5.0）——避免把凑数/占位模型当成品发出去。
#
# 用法:
#   powershell -File tools/build_chat_exes.ps1                 # 两个都打（不过阈值的会被拒绝）
#   powershell -File tools/build_chat_exes.ps1 -Only 1m
#   powershell -File tools/build_chat_exes.ps1 -SkipQuality    # 关掉质量闸门（不推荐）
param([string]$Only = "", [double]$MaxVal = 5.0, [switch]$SkipQuality)
$ErrorActionPreference = "Continue"
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$py = "python"
Set-Location $root
$dist = Join-Path $root "dist"
New-Item -ItemType Directory -Force -Path $dist | Out-Null

$targets = @(
    @{ tag = "10m"; model = "artifacts/b1_C3_10k.cpk";           exe = "vmlm-chat-10m.exe" },
    @{ tag = "1m";  model = "artifacts/b1_char_10k_seq64.cpk";   exe = "vmlm-chat-1m.exe" }
)

foreach ($t in $targets) {
    if ($Only -ne "" -and $Only -ne $t.tag) { continue }
    $mp = Join-Path $root $t.model
    Write-Host ("=== 目标 " + $t.tag + " : " + $t.model)
    if (-not (Test-Path $mp)) { Write-Host "    [skip] 文件不存在"; continue }
    $fi = Get-Item $mp
    Write-Host ("    源文件: {0:N2} MB, mtime {1}" -f ($fi.Length / 1MB), $fi.LastWriteTime)

    # ---- 质量闸门：先量 val ----
    if (-not $SkipQuality) {
        $flags = & $py (Join-Path $root "tools/cpk_info.py") --model $mp --flags
        if ($LASTEXITCODE -ne 0) { Write-Host "    [skip] 头部读取失败"; continue }
        $qdir = Join-Path $env:TEMP ("vmlm_q_" + $t.tag)
        New-Item -ItemType Directory -Force -Path $qdir | Out-Null
        $qlog = "$qdir\eval.log"
        & cmd.exe /c ".\target\release\vmlm-b1-full.exe $flags --data-dir artifacts/b1_char --seed 1 --out-dir $qdir --load-model $mp > $qlog 2>&1" | Out-Null
        $line = (Select-String -Path $qlog -Pattern "非UNK目标=" | Select-Object -First 1).Line
        $val = $null
        if ($line -match "非UNK目标=([0-9.]+)") { $val = [double]$Matches[1] }
        $hv = (Select-String -Path $qlog -Pattern "^final_val=" | Select-Object -First 1).Line
        Write-Host ("    载入实测: " + $hv)
        Write-Host ("    非UNK口径 val = " + $val)
        if ($val -ne $null -and $val -gt $MaxVal) {
            Write-Host ("    [拒绝打包] val {0} > 阈值 {1} —— 这个 .cpk 不像是训练好的模型。" -f $val, $MaxVal)
            Write-Host ("    证据: " + $qlog + " ; 确认无误可用 -SkipQuality 强制打包")
            continue
        }
    }

    # ---- 编译并内嵌 ----
    $env:VMLM_MODEL = $mp
    cargo build --release -p vmlm-train --bin vmlm-chat *> build_chat.log
    if ($LASTEXITCODE -ne 0) { Write-Host "    [fail] cargo build（见 build_chat.log）"; continue }
    Select-String -Path build_chat.log -Pattern "embed model|^error" | ForEach-Object { "    " + $_.Line }
    $out = Join-Path $dist $t.exe
    Copy-Item "target/release/vmlm-chat.exe" $out -Force
    Write-Host ("    -> {0}  {1:N2} MB" -f $out, ((Get-Item $out).Length / 1MB))

    # ---- 空文件夹实测 ----
    $test = Join-Path $env:TEMP ("vmlm_sf_" + $t.tag)
    Remove-Item -Recurse -Force $test -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $test | Out-Null
    Copy-Item $out (Join-Path $test $t.exe)
    [IO.File]::WriteAllText((Join-Path $test "in.txt"), "今天`nquit`n", (New-Object Text.UTF8Encoding($false)))
    Write-Host ("    空文件夹内容: " + ((Get-ChildItem $test | ForEach-Object { $_.Name }) -join ", "))
    Push-Location $test
    $sw = [Diagnostics.Stopwatch]::StartNew()
    & cmd.exe /c "$($t.exe) < in.txt > out.txt 2>&1" | Out-Null
    $code = $LASTEXITCODE
    $sw.Stop()
    Pop-Location
    Write-Host ("    实测: exit=$code 总耗时=" + [math]::Round($sw.Elapsed.TotalSeconds, 1) + "s")
    $txt = [IO.File]::ReadAllText((Join-Path $test "out.txt"), [Text.Encoding]::UTF8)
    New-Item -ItemType Directory -Force -Path (Join-Path $root "artifacts/singlefile") | Out-Null
    [IO.File]::WriteAllText((Join-Path $root ("artifacts/singlefile/test_" + $t.tag + ".txt")), $txt, (New-Object Text.UTF8Encoding($false)))
    ($txt -split "`n") | Select-Object -First 6 | ForEach-Object { "    | " + $_ }
    (($txt -split "`n") | Select-String -Pattern "字符/秒") | ForEach-Object { "    | " + $_.Line.Trim() }
    Write-Host ""
}
Remove-Item Env:\VMLM_MODEL -ErrorAction SilentlyContinue
Write-Host "完成。分发物在 $dist"
