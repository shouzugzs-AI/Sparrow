# P15 S5-W3 runner: end-to-end proof that the cache path uses the neuron layout.
#
# WHY A .ps1: the canonical build/run path in this repo is PowerShell, and the
# engine's own build needs the runtime/ directory .cargo/config.toml (which the
# root config does NOT inherit -- see runtime/.cargo/config.toml). Keeping the
# build step here makes the exact command that produced the measured binary
# reproducible in one place.
#
# PURE ASCII is mandatory for .ps1 in this repo.
#
# WHAT THIS DOES
#   1. Builds the DEFAULT engine  (must stay 68096 B -- no S5 bytes).
#   2. Builds the neuron-kv engine (the one that exposes --s5-load) and records
#      BOTH sizes so the delta is a MEASURED number, not an estimate.
#   3. Runs tools/p15_s5_e2e.py, which measures addresses, bit-identity, the
#      on-disk section, and drives a real engine run over a real 24-layer
#      neuron_kv artifact.
#
# DOES NOT run tools/verify_all.ps1 or tools/build_all.ps1: the Lead owns those
# as the serialized acceptance gate.
#
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File tools/p15_s5_e2e.ps1

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$rt   = Join-Path $root "runtime"
$out  = Join-Path $root "examples/p15_s5_e2e"
$tgt  = Join-Path $root "target/x86_64-pc-windows-msvc/release/vmlm-runtime.exe"

New-Item -ItemType Directory -Force -Path $out | Out-Null

# ---- cargo helper ----
# TRAP, measured here and already documented in tools/p14_gen_run.ps1 lines 78-82:
# with $ErrorActionPreference = "Stop", PowerShell treats a native command's
# STDERR output as a TERMINATING error. cargo writes ordinary warnings to stderr
# ("profiles for the non root package will be ignored"), so the build aborts
# before it runs. Success must be decided from $LASTEXITCODE, which is the only
# sound signal. Relax the preference for exactly the native call.
function Invoke-Cargo([string[]]$cargoArgs) {
    Push-Location $rt
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $text = & cargo @cargoArgs 2>&1 | Out-String
    $code = $LASTEXITCODE
    $ErrorActionPreference = $prev
    Pop-Location
    $text | Write-Host
    return $code
}

# ---- 1. DEFAULT build: S5 must cost the engine 0 bytes ----
Write-Host "=== building DEFAULT engine (S5 must cost 0 B) ===" -ForegroundColor Cyan
$code = Invoke-Cargo @("build", "--release", "--target", "x86_64-pc-windows-msvc")
if ($code -ne 0) { Write-Host "default build FAILED (exit $code)" -ForegroundColor Red; exit 3 }
if (-not (Test-Path $tgt)) { Write-Host "no engine exe at $tgt" -ForegroundColor Red; exit 4 }
$defaultSize = (Get-Item $tgt).Length
Write-Host ("  DEFAULT engine size = {0} bytes" -f $defaultSize)
if ($defaultSize -ne 68096) {
    Write-Host ("  NOTE: default size is {0}, not the 68096 B budget figure." -f $defaultSize) -ForegroundColor Yellow
    Write-Host "  Either the default feature set changed or another agent is mid-build." -ForegroundColor Yellow
    Write-Host "  This script reports what it MEASURED; it does not adjust the number." -ForegroundColor Yellow
}

# ---- 2. neuron-kv build: measure the real delta ----
# NOTE: the exe path is the SAME for both builds, so the default size is read
# and recorded BEFORE the second build overwrites the file. That ordering is the
# whole reason this script exists rather than two copy-pasted commands.
Write-Host "=== building neuron-kv engine (the one with --s5-load) ===" -ForegroundColor Cyan
$code = Invoke-Cargo @("build", "--release", "--features", "neuron-kv", "--target", "x86_64-pc-windows-msvc")
if ($code -ne 0) { Write-Host "neuron-kv build FAILED (exit $code)" -ForegroundColor Red; exit 5 }
$s5Size = (Get-Item $tgt).Length
$delta = $s5Size - $defaultSize
Write-Host ("  neuron-kv engine size = {0} bytes   DELTA = {1} bytes" -f $s5Size, $delta) -ForegroundColor Green

# Positive control for the measurement apparatus itself: if BOTH builds had
# produced the same file size, the "delta" would be an empty measurement
# (docs/P13 section 3.2). The S5 entry really is linked, so the delta MUST be
# nonzero; assert that here rather than trusting it.
if ($delta -eq 0) {
    Write-Host "MEASUREMENT APPARATUS SUSPECT: delta is 0, so the two builds" -ForegroundColor Red
    Write-Host "produced the same bytes and the size probe cannot see a difference." -ForegroundColor Red
    exit 6
}

# Stage the measured binary for reproduction convenience ONLY.
#
# ⚠️ TASK-6: this copy is an OUTPUT of the script, NOT an input to the probe.
# My earlier version had engine_exe() PREFER this staged file, which silently
# pinned the harness to a SUPERSEDED engine (it accepted the vacuous artifact
# and had no pos_varying_neurons field at all). engine_exe() no longer looks
# here; keeping the copy is purely so a human can re-run the exact binary.
Copy-Item $tgt (Join-Path $out "vmlm-runtime-s5.exe") -Force

# ---- 3. the real acceptance ----
Write-Host "=== tools/p15_s5_e2e.py ===" -ForegroundColor Cyan
Push-Location $root
& python tools/p15_s5_e2e.py
$rc = $LASTEXITCODE
Pop-Location

Write-Host ""
Write-Host ("engine sizes: DEFAULT {0} B, neuron-kv {1} B, delta {2} B" -f $defaultSize, $s5Size, $delta)
# PROVISIONAL: stamp every size with the revision it came from. During task-6
# the tree was measured MOVING (s5_load.rs modified several times while three
# agents built), giving three different neuron-kv sizes from three revisions:
#   78848 B  @ _lead_s5on      (Lead, ~20:01)
#   79360 B  @ x86_64 release  (this harness)
#   79872 B  @ _lead_gate2     (Lead, after the ~20:07 edit)
# All three are PROVISIONAL and NONE is authoritative. The sha256 below pins
# ours to an exact revision so the number cannot drift silently underneath it.
$hS5 = (Get-FileHash $tgt -Algorithm SHA256).Hash
Write-Host ("  neuron-kv sha256 = {0}" -f $hS5)
Write-Host "  ALL SIZES ARE PROVISIONAL until the tree freezes; the Lead takes the"
Write-Host "  authoritative number on the frozen revision."
Write-Host "NOTE: the DEFAULT size is the one under the 68096 B budget."
Write-Host "NOTE: the engine trigger is a BARE flag --s5-load reading a fixed"
Write-Host "      filename s5_kv.vmb from the working directory (task-6 note 6)."
if ($rc -ne 0) { Write-Host "p15_s5_e2e.py reported FAIL" -ForegroundColor Red; exit 7 }
exit 0
