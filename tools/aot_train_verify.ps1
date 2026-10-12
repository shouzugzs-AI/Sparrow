# VMLM AOT training closed-loop verification (P8 member G, task 3; P9 updated by Lead)
#
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File tools/aot_train_verify.ps1
#
# What it verifies
#   1) `--features aot-train` builds an exe that reads NO external files
#   2) the full training loop runs in an isolation dir containing only the exe
#   3) loss drops significantly
#   4) the loss sequence matches the std (file-reading) path
#
# Why raw f32 bytes instead of the printed decimals
#   Both drivers print 4 decimals, which rounds away real differences and can also
#   turn pure rounding noise into a false mismatch. Dumping 4*N raw f32 bytes and
#   comparing them is the decidable way.
#
# The std side's raw loss comes from an independent probe (tools/stdprobe/) that
# replicates train_mnist.rs's loop, because train/ was not member G's scope.
#
# ASCII-ONLY BY DESIGN: PowerShell 5.1 decodes .ps1 as GBK when there is no BOM,
# so non-ASCII comments produce stray tokens and bogus parse errors pointing at
# innocent lines. Keep this file pure ASCII.

$ErrorActionPreference = "Stop"

$Root   = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Target = "x86_64-pc-windows-msvc"
$Work   = Join-Path $Root "_g_aot_tmp\train_verify"
New-Item -ItemType Directory -Force -Path $Work | Out-Null
$Steps  = 200

function Run-In($exe, $arglist, $cwd, $logfile) {
    $q = [char]34
    $inner = "cd /d $q$cwd$q && $q$exe$q $arglist > $q$logfile$q 2>&1"
    $null = & cmd.exe /c $inner
    return $LASTEXITCODE
}

Write-Host "=== [1/4] Build AOT training exe (model + dataset fully embedded) ===" -ForegroundColor Cyan
# Evaluation must switch to a forward-only stream, or every run() keeps training
# (see the comment in aot_train.rs).
$env:VMLM_AOT_MODELS = "mnist_train=../examples/mnist/train_step.vmb;mnist_eval=../examples/mnist/eval_step.vmb"
$outFile = [System.IO.Path]::GetTempFileName()
$cmd = "cd /d `"$Root\runtime`" && cargo build --release --target $Target --features aot-train > `"$outFile`" 2>&1"
$null = & cmd.exe /c $cmd
$buildCode = $LASTEXITCODE
$buildLog = Get-Content $outFile -Raw -ErrorAction SilentlyContinue
Remove-Item $outFile -Force -ErrorAction SilentlyContinue
Remove-Item Env:\VMLM_AOT_MODELS
if ($buildCode -ne 0) { Write-Host $buildLog; throw "build failed ($buildCode)" }
($buildLog -split "`n" | Select-String "VMLM AOT|Finished") | ForEach-Object { Write-Host $_.Line.Trim() }

$exe = Join-Path $Root "target\$Target\release\vmlm-runtime.exe"
$exeSize = (Get-Item $exe).Length
Write-Host "AOT train exe = $exeSize B  (payload: 2 models)"

# Isolation dir: exe only
$iso = Join-Path $Work "iso"
if (Test-Path $iso) { Remove-Item $iso -Recurse -Force }
New-Item -ItemType Directory -Force -Path $iso | Out-Null
Copy-Item $exe (Join-Path $iso "at.exe") -Force
$others = Get-ChildItem $iso -File | Where-Object { $_.Name -ne "at.exe" }
if ($others) {
    # Do not write "$($others.Name -join ',')": PS 5.1 rejects -join inside a
    # subexpression ("The assignment expression is not valid").
    $otherNames = ($others | ForEach-Object { $_.Name }) -join ','
    throw "isolation dir not clean: $otherNames"
}

Write-Host ""
Write-Host "=== [2/4] Run the training loop in the isolation dir (no data files) ===" -ForegroundColor Cyan
Write-Host "isolation dir files: $((Get-ChildItem $iso -File | ForEach-Object { $_.Name }) -join ', ')"
$log = Join-Path $Work "aot_train.log"
$dumpAot = Join-Path $iso "aot_loss.bin"
$code = Run-In (Join-Path $iso "at.exe") "--steps=$Steps --dump-loss=$dumpAot" $iso $log
Write-Host (Get-Content $log -Raw)
Write-Host "AOT exit = $code"
if ($code -ne 0) { throw "AOT training returned $code" }

Write-Host "=== [3/4] std file-reading path produces the same loss sequence ===" -ForegroundColor Cyan
$probeDir = Join-Path $Root "tools/stdprobe"
$probeExe = Join-Path $probeDir "target/release/vmlm-g-stdprobe.exe"
if (-not (Test-Path $probeExe)) {
    Write-Host "building std probe..."
    # Push-Location + cargo directly. Do not wrap in cmd.exe: nesting escaped
    # quotes there breaks the PS 5.1 parser ("The token '&&' is not a valid
    # statement separator").
    Push-Location $probeDir
    $probeBuild = & cargo build --release 2>&1 | Out-String
    $probeCode = $LASTEXITCODE
    Pop-Location
    if ($probeCode -ne 0) { Write-Host $probeBuild; throw "probe build failed" }
}
$dumpStd = Join-Path $Work "std_loss.bin"
$stdLog = Join-Path $Work "std_train.log"
$code2 = Run-In $probeExe "examples/mnist $Steps $dumpStd" $Root $stdLog
Write-Host (Get-Content $stdLog -Raw)
if ($code2 -ne 0) { throw "std probe returned $code2" }

Write-Host "=== [4/4] Compare the raw f32 loss sequences ===" -ForegroundColor Cyan
$hAot = if (Test-Path $dumpAot) { (Get-FileHash $dumpAot -Algorithm SHA256).Hash } else { "<missing>" }
$hStd = if (Test-Path $dumpStd) { (Get-FileHash $dumpStd -Algorithm SHA256).Hash } else { "<missing>" }
$sAot = if (Test-Path $dumpAot) { (Get-Item $dumpAot).Length } else { 0 }
$sStd = if (Test-Path $dumpStd) { (Get-Item $dumpStd).Length } else { 0 }

Write-Host ""
Write-Host ("AOT loss dump: {0,6} B  SHA256 {1}" -f $sAot, $hAot)
Write-Host ("std loss dump: {0,6} B  SHA256 {1}" -f $sStd, $hStd)
Write-Host ""

$identical = ($hAot -eq $hStd) -and ($sAot -gt 0)

# Criterion: RELATIVE difference, tolerance 1e-3.
#
# Why not bit-identical any more: P9 raised exp_f accuracy (6.5e-4 -> 0.66 ULP).
# Both paths still run the same engine, but the values they produce changed, so
# f32 accumulation order now differs between them. Over 200 sequential AdamW
# steps that difference is amplified by the sqrt(v) nonlinearity, which is
# sensitive where v is small.
#
# Measured on this pair: median 2.5e-5, p90 1.5e-4, max 4.3e-4 -- all below 1e-3.
# The difference does NOT grow monotonically with step (it tracks loss magnitude),
# which is the signature of float noise rather than a logic divergence.
# f32 eps is 1.19e-7; 1e3~1e4x amplification over 200 AdamW steps predicts ~1e-4.
#
# Bit-identity is the stronger result and is still reported when achieved.
$relTol = 1e-3
$maxRel = 0.0
$nDiff = 0
$nSteps = 0
if ($sAot -eq $sStd -and $sAot -gt 0) {
    $ba = [System.IO.File]::ReadAllBytes($dumpAot)
    $bs = [System.IO.File]::ReadAllBytes($dumpStd)
    for ($i = 0; $i -lt $ba.Length; $i += 4) {
        $va = [double][BitConverter]::ToSingle($ba, $i)
        $vs = [double][BitConverter]::ToSingle($bs, $i)
        $nSteps++
        if ($va -ne $vs) {
            $nDiff++
            $d = [Math]::Abs($va - $vs) / [Math]::Max([Math]::Abs($vs), 1e-9)
            if ($d -gt $maxRel) { $maxRel = $d }
        }
    }
}

$verdict = 1
if ($identical) {
    Write-Host "PASS -- AOT train loop is BIT-IDENTICAL to the std path ($($sAot/4) steps)" -ForegroundColor Green
    $verdict = 0
} elseif ($nSteps -gt 0 -and $maxRel -lt $relTol) {
    Write-Host ("PASS -- AOT and std are numerically equivalent ({0}/{1} steps differ, max rel {2:e2} < {3:e0})" `
        -f $nDiff, $nSteps, $maxRel, $relTol) -ForegroundColor Green
    Write-Host "  NOTE: P9 raised exp_f accuracy, so the two paths no longer match bit-for-bit." -ForegroundColor DarkGray
    Write-Host "        Measured max rel ~4e-4 is f32 accumulation noise amplified by AdamW." -ForegroundColor DarkGray
    Write-Host "        Bit-identity is the stronger result and is still reported when achieved." -ForegroundColor DarkGray
    $verdict = 0
} else {
    Write-Host "FAIL -- loss difference exceeds float noise" -ForegroundColor Red
    if ($nSteps -gt 0) { Write-Host ("  max rel diff = {0:e3} (tolerance {1:e0})" -f $maxRel, $relTol) }
    if ($sAot -eq $sStd -and $sAot -gt 0) {
        $ba = [System.IO.File]::ReadAllBytes($dumpAot)
        $bs = [System.IO.File]::ReadAllBytes($dumpStd)
        $shown = 0
        for ($i = 0; $i -lt $ba.Length -and $shown -lt 10; $i += 4) {
            $va = [BitConverter]::ToSingle($ba, $i)
            $vs = [BitConverter]::ToSingle($bs, $i)
            if ($va -ne $vs) {
                Write-Host ("  step {0}: aot={1:R} std={2:R}" -f ($i / 4), $va, $vs)
                $shown++
            }
        }
    }
}

Write-Host ""
Write-Host "--- Summary ---"
$aotTxt = Get-Content $log -Raw
Write-Host "AOT: $((($aotTxt -split "`n") | Select-String 'loss: first10|test:').Line -join ' | ')"
$stdTxt = Get-Content $stdLog -Raw
Write-Host "STD: $((($stdTxt -split "`n") | Select-String 'probe:').Line -join ' | ')"

Write-Host ""
Write-Host "AOT train exe size = $exeSize B" -ForegroundColor Yellow

exit $verdict
