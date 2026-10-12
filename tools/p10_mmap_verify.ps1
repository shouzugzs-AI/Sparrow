# P10-C verification: instruction-stream embed + weight mmap.
#
# ASCII-ONLY (see verify_all.ps1 header for why).
#
# WHAT THIS PROVES
#   1. Splitting a .vmb yields a byte-exact weights file (== the CONST region)
#      and a code file whose TEXT/META are unchanged.
#   2. The embedded blob really shrinks (this is the point of the split).
#   3. The mmap engine produces output BIT-IDENTICAL to the full-inline
#      engine -- i.e. mmap changes memory behaviour, not numerics.
#   4. The read-only guard fires: a training op against mmapped weights
#      gives exit(9) with a readable message, NOT 0xC0000005 (which in CI
#      shows only as exit=-1073741819).
#
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File tools/p10_mmap_verify.ps1

$ErrorActionPreference = "Continue"
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$asm  = Join-Path $root "target/release/vmlm-asm.exe"
$rt   = Join-Path $root "runtime"
$mmd  = Join-Path $root "_c_mmap_tmp"
$p10  = Join-Path $root "examples/p10"

$script:fails = 0
function Check($name, $cond, $detail) {
    if ($cond) { Write-Host ("  " + $name + ": PASS  " + $detail) }
    else { Write-Host ("  " + $name + ": FAIL  " + $detail); $script:fails++ }
}

Write-Host "===== P10-C AOT embed + weight mmap =====" -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $mmd | Out-Null

# ---- 1. build the small equivalence model ----
python -c @"
import numpy as np, os
d = r'$p10'
os.makedirs(d, exist_ok=True)
rng = np.random.RandomState(7)
np.array(rng.randn(4,32).astype(np.float32)).tofile(os.path.join(d,'x.bin'))
np.array((rng.randn(64,32)*0.1).astype(np.float32)).tofile(os.path.join(d,'W1.bin'))
np.array((rng.randn(64)*0.05).astype(np.float32)).tofile(os.path.join(d,'b1.bin'))
np.array((rng.randn(16,64)*0.1).astype(np.float32)).tofile(os.path.join(d,'W2.bin'))
"@ | Out-Null

$vml = @"
SHAPE x, 4, 32
SHAPE W1, 64, 32
SHAPE b1, 1, 64
SHAPE W2, 16, 64
SHAPE h1, 4, 64
SHAPE a1, 4, 64
SHAPE y, 4, 16
GEMM h1, x, W1
BIAS h1, b1
ACT a1, h1, silu
GEMM y, a1, W2
PRINT y
HALT
"@
$vml | Set-Content -Path (Join-Path $p10 "mmap_eq.vml") -Encoding ASCII

Push-Location $p10
$out = & $asm "mmap_eq.vml" "mmap_eq.vmb" "x:x.bin:4:32" "W1:W1.bin:64:32" "b1:b1.bin:1:64" "W2:W2.bin:16:64" 2>&1 | Out-String
Pop-Location
Check "assemble model" ($out -match 'OK:') "mmap_eq.vmb built"

$full = Join-Path $p10 "mmap_eq.vmb"
$origSize = (Get-Item $full).Length

# ---- 2. split ----
$code = Join-Path $p10 "mmap_eq_code.vmb"
$wts  = Join-Path $p10 "mmap_eq_w.vmbw"
$sp = & python (Join-Path $root "tools/p10_split_vmb.py") $full $code $wts 2>&1 | Out-String
$codeSize = (Get-Item $code).Length
Write-Host ("  original .vmb = " + $origSize + " B ; embedded code = " + $codeSize + " B")
Check "embedded blob shrinks" ($codeSize -lt $origSize) ("$codeSize < $origSize")

# byte-exactness of the split (this is the property that makes mmap safe)
$py = @"
import struct, sys
o = open(r'$full','rb').read()
w = open(r'$wts','rb').read()
c = open(r'$code','rb').read()
co, cl = struct.unpack_from('<II', o, 16)
orig_co, orig_cl = struct.unpack_from('<II', o, 16)
o_co, o_cl = struct.unpack_from('<II', o, 16)
ok_w = (o[co:co+cl] == w)
mo_orig = struct.unpack_from('<I', o, 24)[0]
mo_new  = struct.unpack_from('<I', c, 24)[0]
ok_text = (c[32:32+struct.unpack_from('<I',o,12)[0]] ==
           o[32:32+struct.unpack_from('<I',o,12)[0]])
ml = struct.unpack_from('<I', o, 28)[0]
ok_meta = (c[mo_new:mo_new+ml] == o[mo_orig:mo_orig+ml])
print('WEIGHTS_EXACT=%d TEXT_EXACT=%d META_EXACT=%d' % (ok_w, ok_text, ok_meta))
"@
$py | Set-Content -Path (Join-Path $p10 "_chk.py") -Encoding ASCII
$chk = & python (Join-Path $p10 "_chk.py") 2>&1 | Out-String
Check "weights file == CONST region" ($chk -match 'WEIGHTS_EXACT=1') "byte-exact"
Check "code TEXT preserved"          ($chk -match 'TEXT_EXACT=1') "byte-exact"
Check "code META preserved"          ($chk -match 'META_EXACT=1') "byte-exact (meta_off repatched)"

# ---- 3. build BOTH engines and compare output ----
& python (Join-Path $root "tools/aot_build.py") $code (Join-Path $rt "src\model_aot.rs") 2>&1 | Out-Null
Push-Location $rt
& cargo build --release --features mmap-aot --target x86_64-pc-windows-msvc 2>&1 | Out-Null
$bcode = $LASTEXITCODE
Pop-Location
if ($bcode -ne 0) { Check "build mmap engine" $false "cargo failed"; Write-Host "FAILED"; exit 1 }

Copy-Item (Join-Path $root "target/x86_64-pc-windows-msvc/release/vmlm-runtime.exe") (Join-Path $mmd "mmap_eq.exe") -Force
Copy-Item $wts (Join-Path $mmd "weights.vmbw") -Force

$rel = Join-Path $root "target/release"
Push-Location $mmd
$p1 = Start-Process -FilePath (Resolve-Path ".\mmap_eq.exe") -WorkingDirectory $mmd -RedirectStandardOutput (Join-Path $mmd "out_mmap.txt") -PassThru -NoNewWindow -Wait
Pop-Location
Copy-Item $full (Join-Path $rel "model.vmb") -Force
Push-Location $rel
$p2 = Start-Process -FilePath (Join-Path $rel "vmlm-runtime.exe") -WorkingDirectory $rel -RedirectStandardOutput (Join-Path $mmd "out_full.txt") -PassThru -NoNewWindow -Wait
Pop-Location

$o1 = (Get-Content (Join-Path $mmd "out_mmap.txt") -Raw)
$o2 = (Get-Content (Join-Path $mmd "out_full.txt") -Raw)
Check "mmap engine exit=0" ($p1.ExitCode -eq 0) ("exit=" + $p1.ExitCode)
Check "inline engine exit=0" ($p2.ExitCode -eq 0) ("exit=" + $p2.ExitCode)
Check "mmap output == inline output (bitwise)" ($o1 -eq $o2 -and $o1.Trim().Length -gt 0) "identical numeric output"

# ---- 4. read-only guard: training op against mmapped weights ----
$vml2 = @"
SHAPE ctr, 1, 1
SHAPE W, 64, 32
SHAPE gW, 64, 32
SHAPE lr, 1, 5
ADAMW W, gW, lr
HALT
"@
$vml2 | Set-Content -Path (Join-Path $p10 "mmap_badtrain.vml") -Encoding ASCII
python -c @"
import numpy as np, os
d = r'$p10'
np.array([1.0],dtype=np.float32).tofile(os.path.join(d,'ctr.bin'))
np.zeros(64*32,dtype=np.float32).tofile(os.path.join(d,'gW.bin'))
np.array([0.01,0.9,0.999,1e-8,0.0],dtype=np.float32).tofile(os.path.join(d,'lr.bin'))
"@ | Out-Null
Push-Location $p10
& $asm "mmap_badtrain.vml" "mmap_badtrain.vmb" "ctr:ctr.bin:1:1" "W:W1.bin:64:32" "gW:gW.bin:64:32" "lr:lr.bin:1:5" 2>&1 | Out-Null
Pop-Location
$btc = Join-Path $p10 "bt_code.vmb"
$btw = Join-Path $p10 "bt_w.vmbw"
& python (Join-Path $root "tools/p10_split_vmb.py") (Join-Path $p10 "mmap_badtrain.vmb") $btc $btw 2>&1 | Out-Null
& python (Join-Path $root "tools/aot_build.py") $btc (Join-Path $rt "src\model_aot.rs") 2>&1 | Out-Null
Push-Location $rt
& cargo build --release --features "mmap-aot train" --target x86_64-pc-windows-msvc 2>&1 | Out-Null
$bcode2 = $LASTEXITCODE
Pop-Location
if ($bcode2 -eq 0) {
    Copy-Item (Join-Path $root "target/x86_64-pc-windows-msvc/release/vmlm-runtime.exe") (Join-Path $mmd "badtrain.exe") -Force
    Copy-Item $btw (Join-Path $mmd "weights.vmbw") -Force
    Push-Location $mmd
    $p3 = Start-Process -FilePath (Resolve-Path ".\badtrain.exe") -WorkingDirectory $mmd -RedirectStandardOutput (Join-Path $mmd "out_bt.txt") -PassThru -NoNewWindow -Wait
    Pop-Location
    $msg = (Get-Content (Join-Path $mmd "out_bt.txt") -Raw)
    # 9 = our explicit guard. -1073741819 / 0xC0000005 = raw segfault (bad).
    Check "read-only guard fires" ($p3.ExitCode -eq 9) ("exit=" + $p3.ExitCode + " msg=" + $msg.Trim())
    Check "guard message is readable" ($msg -match 'mmap RO') $msg.Trim()
} else {
    Check "build train+mmap engine" $false "cargo failed"
}

# restore a plain engine so later suites see the normal binary
Push-Location $rt
& cargo build --release --target x86_64-pc-windows-msvc 2>&1 | Out-Null
Pop-Location
Copy-Item (Join-Path $root "target/x86_64-pc-windows-msvc/release/vmlm-runtime.exe") (Join-Path $root "target/release/vmlm-runtime.exe") -Force

Write-Host ""
if ($script:fails -eq 0) {
    Write-Host "P10-C mmap: ALL PASS" -ForegroundColor Green
} else {
    Write-Host ("P10-C mmap: " + $script:fails + " FAILED") -ForegroundColor Red
}
exit $script:fails
