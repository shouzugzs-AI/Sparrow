#!/usr/bin/env pwsh
# P15-S5 verifier suite runner (task-4, independent adversarial verifier).
#
# Runs every verifier instrument in order and writes each one's raw output under
# examples/p15_verify/. It deliberately does NOT run tools/verify_all.ps1 or
# tools/build_all.ps1 -- those are the Lead's serialized acceptance gate and are
# out of this role's scope.
#
# WHY A RUNNER EXISTS
#   The numbers in docs/P15_S5_wiring_verification.md must be re-derivable by
#   anyone at any time. A single command that reproduces every one of them is
#   the difference between a claim and a measurement. It also records the FROZEN
#   revision (sizes + sha256) at the top of every artifact, because the tree was
#   moving during this task and a number without a revision is not a datum.
#
# Usage (from <repo>):
#     pwsh tools/p15_verify.ps1
#     pwsh tools/p15_verify.ps1 -SkipBudget     # skip the ~5 minute build sweep

param(
    [switch]$SkipBudget
)

$ErrorActionPreference = 'Continue'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $root
$out = Join-Path $root 'examples/p15_verify'
New-Item -ItemType Directory -Force -Path $out | Out-Null

$revFile = Join-Path $out 'FROZEN_REVISION.txt'
$revLines = @()
$revLines += "P15-S5 verifier suite -- frozen revision recorded at run time"
$revLines += "generated_utc = " + (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
$revLines += ""
$revLines += ("{0,-44} {1,10} {2} {3}" -f 'path', 'bytes', 'sha256[:16]', 'mtime')
$revLines += ("-" * 100)
$files = @(
    'core/src/neuron_kv.rs', 'core/src/format.rs', 'runtime/src/s5_load.rs',
    'runtime/src/main.rs', 'runtime/Cargo.toml', 'core/Cargo.toml',
    'tools/p15_s5_section.py', 'tools/p15_s5_load_probe.py', 'tools/p15_s5_e2e.py',
    'examples/p15_s5/neuron_kv_offsets.bin', 'target/release/vmlm-runtime.exe'
)
foreach ($f in $files) {
    $p = Join-Path $root $f
    if (Test-Path $p) {
        $h = (Get-FileHash $p -Algorithm SHA256).Hash.Substring(0, 16)
        $l = (Get-Item $p).Length
        $t = (Get-Item $p).LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss')
        $revLines += ("{0,-44} {1,10} {2} {3}" -f $f, $l, $h, $t)
    }
    else { $revLines += ("{0,-44} MISSING" -f $f) }
}
$revLines | Set-Content -Path $revFile -Encoding UTF8
Write-Host "=== FROZEN REVISION ===" -ForegroundColor Cyan
$revLines | ForEach-Object { Write-Host $_ }

$steps = @(
    @{ n = '1_instrument_selftest'; s = 'tools/p15_verify_instruments_selftest.py'; skip = $false },
    @{ n = '2_guards';              s = 'tools/p15_verify_guards.py';              skip = $false },
    @{ n = '3_layout';              s = 'tools/p15_verify_layout.py';              skip = $false },
    @{ n = '4_degradation';         s = 'tools/p15_verify_degradation.py';         skip = $false },
    @{ n = '5_honesty';             s = 'tools/p15_verify_honesty.py';             skip = $false },
    @{ n = '6_budget';              s = 'tools/p15_verify_budget.py';              skip = $SkipBudget }
)

$summary = @()
foreach ($st in $steps) {
    if ($st.skip) {
        Write-Host "`n=== SKIPPED (per flag): $($st.s) ===" -ForegroundColor Yellow
        $summary += [pscustomobject]@{ step = $st.n; script = $st.s; exit = 'SKIPPED' }
        continue
    }
    $log = Join-Path $out ("{0}.txt" -f $st.n)
    Write-Host "`n=== RUNNING $($st.s) -> $log ===" -ForegroundColor Cyan
    & python $st.s 2>&1 | Tee-Object -FilePath $log
    $code = $LASTEXITCODE
    Write-Host "--- exit=$code ---" -ForegroundColor $(if ($code -eq 0) { 'Green' } else { 'Yellow' })
    $summary += [pscustomobject]@{ step = $st.n; script = $st.s; exit = $code }
}

Write-Host "`n=== SUITE SUMMARY ===" -ForegroundColor Cyan
$summary | Format-Table -AutoSize
$summary | ConvertTo-Json | Set-Content -Path (Join-Path $out 'SUITE_SUMMARY.json') -Encoding UTF8

# NOTE: a nonzero exit here does NOT mean the code is broken. The verifier's
# MetaCheck deliberately returns nonzero when a criterion could not be shown
# able to FAIL (UNPROVEN) -- that is the instrument refusing to rubber-stamp,
# and it is reported in the document rather than smoothed away.
$fails = ($summary | Where-Object { $_.exit -ne 0 -and $_.exit -ne 'SKIPPED' }).Count
Write-Host "steps with nonzero exit: $fails (see the document's UNPROVEN rows)" -ForegroundColor Yellow
exit 0
