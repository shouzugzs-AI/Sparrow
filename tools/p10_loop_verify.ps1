# P10-A verification: runtime loop control.
#
# ASCII-ONLY (see verify_all.ps1 header for why).
#
# WHAT THIS PROVES
#   1. LOOP_CNT runs exactly N iterations at RUNTIME (counted from output).
#   2. Instruction count is INDEPENDENT of iteration count -- the whole point
#      of P10-A. The old model expanded per layer at COMPILE time
#      (ESERNN measured: 161 + 154*(layers-1)), which is why 24 layers x 16
#      heads could not be expressed.
#   3. The JMP label fix is real: a backward JMP to a NON-ZERO target lands
#      on that target (the pre-P10 engine always jumped to instruction 0).
#   4. Guards fire: counter double-decrement is rejected at assembly.
#
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File tools/p10_loop_verify.ps1

$ErrorActionPreference = "Continue"
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$asm = Join-Path $root "target/release/vmlm-asm.exe"
$eng = Join-Path $root "target/release/vmlm-runtime.exe"
$rel = Join-Path $root "target/release"
$tmp = Join-Path $root "examples/p10"
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

$script:fails = 0
function Check($name, $cond, $detail) {
    if ($cond) { Write-Host ("  " + $name + ": PASS  " + $detail) }
    else { Write-Host ("  " + $name + ": FAIL  " + $detail); $script:fails++ }
}

# The engine's default build hardcodes model.vmb (no argv -- argv lives in the
# bench feature, which is excluded from the size budget). So we copy in place
# AND run with CWD set to that directory: the engine opens the RELATIVE path
# "model.vmb", so inheriting the caller's CWD yields exit=2 "cannot open file".
function Run-Model($vmb, $tag) {
    Copy-Item $vmb (Join-Path $rel "model.vmb") -Force
    Push-Location $rel
    $out = & $eng 2>&1 | Out-String
    $code = $LASTEXITCODE
    Pop-Location
    return @{ code = $code; text = $out }
}

Write-Host "===== P10-A runtime loop control =====" -ForegroundColor Yellow

# ---- 1. N iterations for several N, from ONE program shape ----
# ctr is a CONSTANT here (loop count is data, not code) -- that is the point:
# changing the loop trip count does not change the instruction stream.
foreach ($n in @(3, 7, 24)) {
    $py = @"
import numpy as np, sys
np.array([$n], dtype=np.float32).tofile(r'$tmp\ctr.bin')
"@
    $py | Set-Content -Path (Join-Path $tmp "_mk.py") -Encoding ASCII
    python (Join-Path $tmp "_mk.py") | Out-Null

    $vml = @"
SHAPE   ctr, 1, 1
loop:
PRINT   ctr
LOOP_CNT ctr, @loop
HALT
"@
    $vml | Set-Content -Path (Join-Path $tmp "loop_$n.vml") -Encoding ASCII

    Push-Location $tmp
    $a = & $asm "loop_$n.vml" "loop_$n.vmb" "ctr:ctr.bin:1:1" 2>&1 | Out-String
    Pop-Location
    $asmOk = $a -match 'OK:'
    if (-not $asmOk) { Write-Host "  N=$n assemble failed"; $script:fails++; continue }

    $instr = 0
    if ($a -match 'OK:\s*(\d+)\s') { $instr = [int]$Matches[1] }

    $r = Run-Model (Join-Path $tmp "loop_$n.vmb") "n$n"
    $vals = @()
    foreach ($line in ($r.text -split "`n")) {
        foreach ($t in ($line -split '\s+')) {
            if ($t -match '^-?\d+\.\d+$') { $vals += [double]$t }
        }
    }
    $cnt = $vals.Count
    # LOOP_CNT decrements then tests >0, so ctr=N yields exactly N iterations
    # printing N, N-1, ... 1.
    $ok = ($r.code -eq 0) -and ($cnt -eq $n) -and ($vals[0] -eq $n) -and ($vals[$cnt-1] -eq 1)
    Check "iterations N=$n" $ok ("iterations=$cnt expected=$n first=$($vals[0]) last=$(if($cnt){$vals[$cnt-1]}) exit=$($r.code)")
    Check "flat instr (N=$n)" ($instr -eq 4) ("instructions=$instr (must be 4 regardless of N)")
}

# ---- 2. JMP label landing on a NON-ZERO target ----
# Pre-P10 the interpreter read the target from `a` (always 0) while the
# assembler wrote `dst`, so JMP ALWAYS went to instruction 0. A backward jump
# to a non-zero index is therefore the precise regression test.
$vml2 = @"
SHAPE   ctr, 1, 1
PRINT   ctr
loop:
PRINT   ctr
LOOP_CNT ctr, @loop
HALT
"@
$vml2 | Set-Content -Path (Join-Path $tmp "jmp_target.vml") -Encoding ASCII
python -c "import numpy as np; np.array([2],dtype=np.float32).tofile(r'$tmp\ctr2.bin')" | Out-Null
Push-Location $tmp
$b = & $asm "jmp_target.vml" "jmp_target.vmb" "ctr:ctr2.bin:1:1" 2>&1 | Out-String
Pop-Location
if ($b -match 'OK:') {
    $r2 = Run-Model (Join-Path $tmp "jmp_target.vmb") "jmp"
    $v2 = @()
    foreach ($line in ($r2.text -split "`n")) {
        foreach ($t in ($line -split '\s+')) { if ($t -match '^-?\d+\.\d+$') { $v2 += [double]$t } }
    }
    # ctr==2: pre-loop PRINT(2) once, then loop: PRINT(2), LOOP->1 PRINT(1) LOOP->0 stop
    # => exactly 3 values: 2,2,1
    $ok2 = ($v2.Count -eq 3) -and ($v2[0] -eq 2) -and ($v2[1] -eq 2) -and ($v2[2] -eq 1)
    Check "backward JMP to nonzero target" $ok2 ("values=$($v2 -join ',') expected=2,2,1")
} else {
    Check "backward JMP to nonzero target" $false "assemble failed"
}

# ---- 3. Guard: counter double-decrement must be REJECTED at assembly ----
$vml3 = @"
SHAPE   ctr, 1, 1
SHAPE   one, 1, 1
loop:
EWSUB   ctr, ctr, one
LOOP_CNT ctr, @loop
HALT
"@
$vml3 | Set-Content -Path (Join-Path $tmp "bad_dbl.vml") -Encoding ASCII
python -c "import numpy as np; np.array([1],dtype=np.float32).tofile(r'$tmp\one.bin')" | Out-Null
Push-Location $tmp
$c = & $asm "bad_dbl.vml" "bad_dbl.vmb" "ctr:ctr2.bin:1:1" "one:one.bin:1:1" 2>&1 | Out-String
Pop-Location
Check "guard: double decrement rejected" ($c -match 'LOOP_CNT' -and $c -match 'panicked') "assembler refused the program"

# ---- 4. Guard: forward-only loop rejected ----
$vml4 = @"
SHAPE   ctr, 1, 1
LOOP_CNT ctr, @fwd
fwd:
HALT
"@
$vml4 | Set-Content -Path (Join-Path $tmp "bad_fwd.vml") -Encoding ASCII
Push-Location $tmp
$d = & $asm "bad_fwd.vml" "bad_fwd.vmb" "ctr:ctr2.bin:1:1" 2>&1 | Out-String
Pop-Location
Check "guard: forward loop rejected" ($d -match 'panicked') "assembler refused forward target"

# ---- 5. Guard: unimplemented opcode rejected at ASSEMBLY ----
# MEAN is defined in format.rs and usto reach the engine, where it hit
# exit(5) "unimpl op" with no indication of WHICH instruction.
$vml5 = @"
SHAPE a, 1, 4
SHAPE b, 1, 1
MEAN  b, a
HALT
"@
$vml5 | Set-Content -Path (Join-Path $tmp "bad_unimpl.vml") -Encoding ASCII
Push-Location $tmp
$e = & $asm "bad_unimpl.vml" "bad_unimpl.vmb" "a:one.bin:1:1" 2>&1 | Out-String
Pop-Location
Check "guard: unimplemented op rejected" ($e -match 'panicked') "assembler refused MEAN by name"

Write-Host ""
if ($script:fails -eq 0) {
    Write-Host "P10-A loop control: ALL PASS" -ForegroundColor Green
} else {
    Write-Host ("P10-A loop control: " + $script:fails + " FAILED") -ForegroundColor Red
}
exit $script:fails
