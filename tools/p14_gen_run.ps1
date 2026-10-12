# p14/O6 runner: assemble a p14 artifact and run it with stdout to a REAL FILE.
#
# WHY A .ps1 RATHER THAN PURE PYTHON: the canonical build/assemble path in this
# repo is PowerShell (tools/build_all.ps1, tools/asm_all.ps1). This stays
# consistent with that, and PURE ASCII is mandatory for .ps1 here.
#
# WHY STDOUT GOES TO A FILE, NEVER A PIPE: measured in P13, draining the
# engine's stdout through a PowerShell pipe made a 1.14 s run look like ~20
# minutes. Timing must always use a real file.
#
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File tools/p14_gen_run.ps1 -Dir examples/p14_smoke

param(
    [Parameter(Mandatory=$true)][string]$Dir,
    [string]$Sampler = "argmax",
    [string]$FeedTokens = "",
    [switch]$KeepGenFile
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$rel  = Join-Path $root "target/release"
$asm  = Join-Path $rel "vmlm-asm.exe"
$eng  = Join-Path $rel "vmlm-runtime.exe"
$adir = Join-Path $root $Dir
# Resolve to an ABSOLUTE, canonical path immediately. The weight paths inside
# weights.args are relative to the artifact dir; joining them onto a relative
# $adir yields e.g. "examples/p14_smoke/weights/embed.bf16", which the assembler
# cannot open, and the failure surfaces only as a generic "cannot read weight
# file" panic far from the real cause.
$adir = (Resolve-Path $adir).Path

if (-not (Test-Path $adir)) { Write-Host "no such dir: $adir" -ForegroundColor Red; exit 2 }

$vml = Join-Path $adir "full.vml"
$vmb = Join-Path $adir "full.vmb"
$argf = Join-Path $adir "weights.args"

# ---- assemble ----
# The assembler takes weights as `name:file:rows:cols[:dtype]` arguments.
$weights = @()
foreach ($line in (Get-Content $argf)) {
    if ($line.Trim() -ne "") { $weights += $line.Trim() }
}
Write-Host "assembling $vml  ($($weights.Count) constants)" -ForegroundColor Cyan

# Build the full argv as an array so no quoting is lost. Paths inside the
# artifact are relative to the artifact dir, so we run with that as CWD.
$json = Get-Content (Join-Path $adir "gen_rows.json") -Raw | ConvertFrom-Json
$wargs = @()
foreach ($line in $weights) {
    $f = $line.Split(":")
    # RELATIVE paths on purpose, and this is NOT a style choice.
    #
    # The assembler parses each weight argument with `w.split(':')`
    # (asm/src/main.rs:1186) and takes field [1] as the filename. An ABSOLUTE
    # Windows path contains a drive colon, so "name:D:\dir\w.bf16:3:4" splits
    # into ["name","D","\dir\w.bf16","3","4"] -- the assembler then tries to
    # open "D" and panics with a generic, misleading
    # "cannot read weight file" (asm/src/main.rs:1190).
    #
    # So: emit `weights/name.ext` exactly as weights.args already stores it, and
    # run the assembler with the artifact dir as CWD. That is what
    # tools/p12_build_minicpm.py + tools/asm_all.ps1 do, and it is the only
    # spelling the assembler accepts.
    $abs = Join-Path $adir ($f[1] -replace "/", "\")
    if (-not (Test-Path $abs)) {
        Write-Host "MISSING WEIGHT FILE for '$($f[0])': $abs" -ForegroundColor Red
        exit 3
    }
    if ($f.Count -ge 5) {
        $wargs += ("{0}:{1}:{2}:{3}:{4}" -f $f[0], $f[1], $f[2], $f[3], $f[4])
    } else {
        $wargs += ("{0}:{1}:{2}:{3}" -f $f[0], $f[1], $f[2], $f[3])
    }
}

# NOTE: the assembler writes its slot table to STDERR (it is diagnostic, not
# an error). With $ErrorActionPreference = "Stop" PowerShell treats a native
# command's stderr output as a terminating error, which would abort the run and
# report a false failure. So relax it for exactly this call and decide success
# from $LASTEXITCODE, which is the only sound signal.
Push-Location $adir
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$asmOut = & $asm "full.vml" "full.vmb" @wargs 2>&1 | Out-String
$asmCode = $LASTEXITCODE
$ErrorActionPreference = $prevEAP
Pop-Location
$asmOut | Out-File -Encoding ascii (Join-Path $adir "_asm.log")
if ($asmCode -ne 0) {
    Write-Host "ASSEMBLE FAILED:" -ForegroundColor Red
    Write-Host $asmOut
    exit 3
}
Write-Host "  $($asmOut.Split("`n")[0].Trim())"

# ---- POST-ASSEMBLY GUARD: no slot may be 0x0 ------------------------------
# The assembler SILENTLY MATERIALISES an unknown name as an activation slot of
# shape 0x0 (it does not refuse). A 0x0 slot means the op using it writes ZERO
# elements, and the rest of the program then computes on STALE activation memory
# -- which shows up much later as an unrelated-looking error. That is the P13
# -- P13 section 2.1 failure class, and this is the authoritative check for it
# because only the assembler knows the final UNIFIED shapes.
#
# Measured instance of exactly this: the builder emitted `GEMM logits, xn,
# lm_head_out` while declaring the constant as `lm_head`. The dump contained
# `slot83 lm_head_out [act] 0x0`, and the run later died in an EMB with
# "row index out of range / table_rows=4".
$zeroSlots = @()
foreach ($line in ($asmOut -split "`r?`n")) {
    if ($line -match "0x0") {
        # only slot-table lines look like: "   slotN name [kind] 0x0"
        if ($line -match "slot\d+\s+\S+\s+\[") { $zeroSlots += $line.Trim() }
    }
}
if ($zeroSlots.Count -gt 0) {
    Write-Host "ASSEMBLE GUARD FAILED: the assembled program has 0x0 slot(s)." -ForegroundColor Red
    Write-Host "A 0x0 slot means an op using it writes NOTHING and the rest of the" -ForegroundColor Red
    Write-Host "program runs on stale memory. Usual cause: a referenced name that was" -ForegroundColor Red
    Write-Host "never declared (typo, or a missing SHAPE)." -ForegroundColor Red
    foreach ($s in $zeroSlots) { Write-Host "   $s" -ForegroundColor Red }
    exit 6
}
Write-Host "  slot-table guard: no 0x0 slots" -ForegroundColor Green

# ---- stage the runtime inputs ----
# THE DEFAULT RUNTIME TAKES NO PATH ARGUMENT: it hardcodes model.vmb and reads
# tokens.bin + gen_tokens.bin for the st_tok / st_gen slots. All three must
# exist in the CWD or the run dies with exit(2) "cannot open file".
Copy-Item $vmb (Join-Path $rel "model.vmb") -Force
Copy-Item (Join-Path $adir "prompt_tok.bin") (Join-Path $rel "tokens.bin") -Force

$seed = Join-Path $adir "seed_gen.bin"
$feed = Join-Path $adir "feed_gen.bin"
$declaresGen = (Select-String -Path $vml -Pattern "^SHAPE st_gen," -Quiet)
if (-not $declaresGen) {
    # The program declares no st_gen (e.g. the --oneshot reference build), so
    # load_ids no-ops for it and no gen_tokens.bin is required. Writing one
    # anyway would be harmless but misleading; NOT writing it is the honest
    # statement that this program does not consume generated tokens.
    Write-Host "  program declares no st_gen: gen_tokens.bin not required"
    if (Test-Path (Join-Path $rel "gen_tokens.bin")) {
        Remove-Item (Join-Path $rel "gen_tokens.bin") -Force -ErrorAction SilentlyContinue
    }
} elseif ($Sampler -eq "feed") {
    if (-not (Test-Path $feed)) { Write-Host "no feed_gen.bin for --sampler feed" -ForegroundColor Red; exit 4 }
    Copy-Item $feed (Join-Path $rel "gen_tokens.bin") -Force
    Write-Host "  gen_tokens.bin <- feed_gen.bin (NEGATIVE CONTROL chain)" -ForegroundColor Yellow
} else {
    if (-not (Test-Path $seed)) { Write-Host "no seed_gen.bin" -ForegroundColor Red; exit 4 }
    # Closed loop: only the SEED is file-sourced. st_gen[1..G] are written
    # in-engine by each step's argmax.
    Copy-Item $seed (Join-Path $rel "gen_tokens.bin") -Force
    Write-Host "  gen_tokens.bin <- seed_gen.bin (seed only: $((Get-Item $seed).Length) bytes)"
}

# ---- run, stdout to a REAL FILE ----
#
# ENCODING TRAP (this cost a real debugging cycle): PowerShell's `>` and `>`
# redirection operators write text as UTF-16LE WITH A BOM (`\xff\xfe`) when the
# target is a file it creates. The engine emits ASCII f32 text, so the result is
# a perfectly valid-looking file that a byte-oriented parser reads as ZERO
# numbers -- measured: 50,169,716 bytes parsed as 0 values, and the checker then
# reported a confusing "stdout too short for block 'logits': need 2611200,
# have 0".
#
# The fix is to redirect the NATIVE command's stdout at the FILE DESCRIPTOR
# level with cmd-style redirection, bypassing PowerShell's text pipeline. Using
# [System.IO.File]::Create + a raw stream would work too, but Start-Process with
# -RedirectStandardOutput is the cleanest and keeps the byte stream untouched.
$out = Join-Path $adir "_engine_out.txt"
$errf = Join-Path $adir "_engine_err.txt"
$t0 = Get-Date
$proc = Start-Process -FilePath $eng -WorkingDirectory $rel -NoNewWindow -Wait `
    -RedirectStandardOutput $out -RedirectStandardError $errf -PassThru
$code = $proc.ExitCode
$dt = (Get-Date) - $t0

if ($code -ne 0) {
    Write-Host "ENGINE FAILED exit=$code" -ForegroundColor Red
    Get-Content $errf -ErrorAction SilentlyContinue | Select-Object -First 20
    exit 5
}

# Verify the capture is byte-oriented, not UTF-16 (see the ENCODING TRAP note
# above). A BOM here means the run's numbers are unreadable and every downstream
# check would silently see zero values, so fail loudly rather than let the
# checker mis-report.
$fs = [System.IO.File]::OpenRead($out)
$b0 = New-Object byte[] 2
$null = $fs.Read($b0, 0, 2)
$fs.Close()
if ($b0[0] -eq 0xFF -and $b0[1] -eq 0xFE) {
    Write-Host "STDOUT CAPTURE IS UTF-16LE (BOM detected): every downstream parser" -ForegroundColor Red
    Write-Host "would read ZERO values. Redirect as a raw byte stream, not via '>'." -ForegroundColor Red
    exit 7
}

$sz = (Get-Item $out).Length
Write-Host ("  ran in {0:N2} s, stdout {1:N0} bytes -> {2}" -f $dt.TotalSeconds, $sz, $out) -ForegroundColor Green

if (-not $KeepGenFile) { Remove-Item (Join-Path $rel "gen_tokens.bin") -Force -ErrorAction SilentlyContinue }
exit 0
