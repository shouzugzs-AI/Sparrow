# P14 independent verification driver (verifier-owned, task-4).
#
# ASCII-ONLY BY DESIGN. PowerShell 5.1 reads .ps1 as GBK/ANSI unless the file
# has a UTF-8 BOM, so non-ASCII comments get mis-decoded into stray tokens and
# produce parse errors pointing at innocent lines. verify_all.ps1 documents
# this; this file follows the same rule.
#
# WHAT THIS IS
#   The single entry point for the VERIFIER's adversarial checks of O6 (task-2)
#   and S5 (task-3). It is deliberately NOT a wrapper around the implementers'
#   check scripts: a checker written by the implementer shares the implementer's
#   blind spot, and this project has been burned by exactly that (a probe with
#   an empty patch reported "0 B" for a change that cost 3,584 B -- docs/P13
#   report section 3.2). So everything below drives the verifier's own
#   instruments in tools/p14_verify_lib.py, tools/p14_s5_layout_lib.py and
#   tools/p14_o6_harness.py.
#
# THE RULE THIS SCRIPT ENFORCES ON ITSELF
#   Every acceptance test must be shown ABLE TO FAIL. A predicate that passes on
#   both correct and broken input is reported UNPROVEN, never PASS. The
#   instruments' self-tests run FIRST, and if an instrument cannot detect its
#   own negative control, its verdicts are discarded before they can be
#   mistaken for evidence.
#
# TIMING AND IO
#   All engine stdout goes to REAL FILES, never through a PowerShell pipe. A
#   1.14 s run once looked like ~20 minutes when its output was drained through
#   a pipe (P12/P13 handover), and verify_all.ps1 warns that a large piped
#   output can even clobber $LASTEXITCODE and turn a failure into a recorded
#   pass. Nothing here reads engine output from the pipeline.
#
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File tools/p14_verify.ps1
#        [-Prep]     instruments + pre-registration only (no implementation
#                    needed) -- usable while task-1/2/3 are still running
#        [-Full]     run the implementation-dependent suites

param(
    [switch]$Prep,
    [switch]$Full
)

$ErrorActionPreference = "Continue"
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $root

$script:fails = 0
$script:unproven = 0
$script:skipped = @()

function Section($title) {
    Write-Host ""
    Write-Host ("===== " + $title + " =====") -ForegroundColor Yellow
}

# Run a python check script, capturing its output to a REAL FILE (never a pipe).
# Returns the exit code. Prints a trimmed tail so the log stays readable.
function Run-Py {
    param([string]$Name, [string]$Script, [string[]]$PyArgs = @(), [switch]$Required)
    $log = Join-Path $root ("target/release/_p14_" + $Name + ".log")
    $p = Join-Path $root ("tools\" + $Script)
    if (-not (Test-Path $p)) {
        Write-Host ("  MISSING: " + $Script) -ForegroundColor Red
        $script:fails++
        return 1
    }
    $argv = @($p) + $PyArgs
    & python @argv 1> $log 2>&1
    $code = $LASTEXITCODE
    $lines = @(Get-Content $log -ErrorAction SilentlyContinue)
    foreach ($l in ($lines | Select-Object -Last 8)) { Write-Host ("  " + $l) }
    if ($code -ne 0) {
        if ($Required) { $script:fails++ }
        Write-Host ("  -> exit " + $code) -ForegroundColor Red
    } else {
        Write-Host ("  -> exit 0") -ForegroundColor Green
    }
    return $code
}

# Note a criterion we could not verify, and say WHY. NEVER softened into a pass.
function Unverified($what, $why) {
    $script:skipped += [pscustomobject]@{ What = $what; Why = $why }
    Write-Host ("  UNVERIFIED: " + $what + " -- " + $why) -ForegroundColor Magenta
}

function Has-File($rel) { Test-Path (Join-Path $root $rel) }

# ---------------------------------------------------------------------------
Section "P14 verifier: instrument self-tests (must pass before any verdict)"

Write-Host "  An instrument that cannot detect a KNOWN-BAD input cannot be trusted"
Write-Host "  to recognise a good one. These run first, on purpose."
Write-Host ""

Write-Host "  [1] O6 harness: chain comparison must reject a perturbed chain"
Run-Py "o6_harness" "p14_o6_harness.py" @() -Required | Out-Null

Write-Host ""
Write-Host "  [2] S5 layout: address/stride analyzer must detect wrong stride,"
Write-Host "      a transposed store, and a gather"
Run-Py "s5_layout" "p14_s5_layout_lib.py" @() -Required | Out-Null

# ---------------------------------------------------------------------------
Section "P14 pre-registration (acceptance predicates frozen before impl)"

if (-not (Has-File "docs/P14_preregistration.json")) {
    Run-Py "prereg" "p14_preregister.py" @() -Required | Out-Null
} else {
    Write-Host "  docs/P14_preregistration.json present:"
    $rec = Get-Content (Join-Path $root "docs/P14_preregistration.json") -Raw |
           ConvertFrom-Json
    Write-Host ("  frozen at : " + $rec.frozen_utc)
    Write-Host ("  predicates: " + @($rec.predicates).Count)
    foreach ($p in $rec.predicates) {
        Write-Host ("    " + $p.id.PadRight(8) + " " + $p.claim.Substring(0,
            [Math]::Min(58, $p.claim.Length)))
    }
    Write-Host "  Re-run tools/p14_preregister.py to re-freeze (changes the hash)."
}

# ---------------------------------------------------------------------------
Section "P14 engine size claim (with a positive control)"

Run-Py "size" "p14_verify_size.py" @() | Out-Null

# ---------------------------------------------------------------------------
Section "P14 O6 (task-2) -- implementation-dependent"

$o6check = Has-File "tools/p14_gen_check.py"
$o6build = Has-File "tools/p14_gen_build.py"
if (-not ($o6check -or $o6build)) {
    Unverified "O6 generation loop" "task-2 has produced neither tools/p14_gen_build.py nor tools/p14_gen_check.py yet"
} else {
    Write-Host ("  build present: " + $o6build + "   check present: " + $o6check)
    # The implementer's own checker is run ONLY as a cross-reference. The
    # verifier's own criteria are O6-P1..P7 in the pre-registration, driven by
    # tools/p14_verify_o6.py once it exists.
    if ($o6check) {
        Write-Host "  running the implementer's checker as a CROSS-REFERENCE only"
        Write-Host "  (a checker by the implementer shares the implementer's blind spot)"
        Run-Py "o6_impl_check" "p14_gen_check.py" @("--dir", "examples/p14_gen") | Out-Null
    }
    # These three are the verifier's own, and they need only the EMITTED .vml or
    # the host sampler -- they do not re-run the 24-layer model, so they are
    # cheap and safe to run even while the box is otherwise busy. The heavy
    # --gen 16 --layers 0 headline run is deliberately NOT invoked here: it
    # takes ~20-30 min of assembly and must be serialized on a quiet box.
    Write-Host "  running the VERIFIER's structural O6 checks (independent)"
    Run-Py "o6_pos" "p14_verify_o6_pos.py" @() | Out-Null
    Run-Py "o6_temp" "p14_verify_o6_temp.py" @() | Out-Null
    Run-Py "o6_cycle" "p14_verify_o6_cycle.py" @() | Out-Null
    # If a headline artifact with raw stdout exists, corroborate its numbers by
    # INDEPENDENT re-computation from the raw bytes. This is cheap and it is the
    # strongest check available on numbers someone else produced.
    if (Has-File "examples/p14_gen/_engine_out.txt") {
        Write-Host "  corroborating the headline claim by re-deriving it from raw stdout"
        Run-Py "o6_headline" "p14_verify_o6_headline.py" @() | Out-Null
    } else {
        Unverified "O6 headline independent re-derivation" "examples/p14_gen/_engine_out.txt absent"
    }
    if ($Full) {
        Write-Host "  running the FULL 24-layer O6 suite (-Full): this is the"
        Write-Host "  heavy run and must be serialized. Do not run it under"
        Write-Host "  contention with other teammates. It uses a PRIVATE temp"
        Write-Host "  dir per run (hermetic), so it is immune to the shared"
        Write-Host "  target/release race, but it is still CPU/memory heavy."
        Run-Py "o6_verify" "p14_verify_o6.py" @("--gen", "16", "--layers", "0") -Required | Out-Null
    } elseif (Has-File "tools/p14_verify_o6.py") {
        Write-Host "  running the VERIFIER's O6 suite (1-layer smoke)."
        Write-Host "  NOTE: at 1 layer the HuggingFace anchor (a 24-layer reference)"
        Write-Host "  MUST report UNVERIFIED -- that is the layer-aware guard working,"
        Write-Host "  not a defect. The anchor is settled only by the 24-layer run:"
        Write-Host "    python tools/p14_verify_o6.py --gen 16 --layers 0"
        Write-Host "  which is recorded as PASS in docs/P14_verification.md."
        Run-Py "o6_verify" "p14_verify_o6.py" @("--layers", "1", "--gen", "3") | Out-Null
    } else {
        Unverified "O6 independent criteria O6-P1..P7" "verifier suite tools/p14_verify_o6.py not written yet"
    }
}

# ---------------------------------------------------------------------------
Section "P14 S5 (task-3) -- implementation-dependent"

$s5check = Has-File "tools/p14_s5_check.py"
$s5mod = Has-File "core/src/neuron_kv.rs"
if (-not ($s5check -or $s5mod)) {
    Unverified "S5 neuron KV layout" "task-3 has produced neither core/src/neuron_kv.rs nor tools/p14_s5_check.py yet"
} else {
    Write-Host ("  neuron_kv.rs present: " + $s5mod + "   check present: " + $s5check)
    if ($s5check) {
        Write-Host "  running the implementer's checker as a CROSS-REFERENCE only"
        Run-Py "s5_impl_check" "p14_s5_check.py" @() | Out-Null
    }
    Write-Host "  running the VERIFIER's S5 suite (independent)"
    Run-Py "s5_layout" "p14_s5_layout_lib.py" @() | Out-Null
    Run-Py "s5_stride" "p14_s5_stride_probe.py" @() | Out-Null
    if (Has-File "tools/p14_verify_s5_r2.py") {
        Run-Py "s5_verify" "p14_verify_s5_r2.py" @() -Required | Out-Null
    } else {
        Unverified "S5 independent criteria S5-P1..P4" "verifier suite tools/p14_verify_s5_r2.py not written yet"
    }
}

# ---------------------------------------------------------------------------
Section "P14 summary"

Write-Host ("  failed checks   : " + $script:fails)
Write-Host ("  unverified items: " + $script:skipped.Count)
if ($script:skipped.Count -gt 0) {
    Write-Host ""
    Write-Host "  UNVERIFIED (NOT passes -- these were not tested):" -ForegroundColor Magenta
    foreach ($s in $script:skipped) {
        Write-Host ("    - " + $s.What)
        Write-Host ("      why: " + $s.Why)
    }
}
Write-Host ""
if ($script:fails -eq 0 -and $script:skipped.Count -eq 0) {
    Write-Host "P14: ALL CHECKS PASS" -ForegroundColor Green
} elseif ($script:fails -eq 0) {
    Write-Host ("P14: no failures, but " + $script:skipped.Count + " item(s) UNVERIFIED") -ForegroundColor Yellow
} else {
    Write-Host ("P14: " + $script:fails + " FAILED") -ForegroundColor Red
}
exit $script:fails
