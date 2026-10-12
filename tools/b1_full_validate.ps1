# B1 full-training driver -- self validation. ASCII-only (see tools/verify_all.ps1 header).
#
# The task requires the driver to be PROVEN usable before the user runs the real thing:
#   V1 build
#   V2 numerical gradient check (central differences vs analytic) at a small config
#   V3 same gradient check through the 24-layer path (deep backward indexing)
#   V4 smoke run: artifacts written, every metric finite, loss actually decreases
#   V5 determinism: same seed twice => identical loss_hash AND byte-identical metrics
#   V6 can-fail: an absurd lr MUST trip the divergence guard (diverged_at recorded, early stop)
#   V7 can-fail: lr=0 MUST trip the plateau guard (stopped_by=plateau) -- else it is a rubber stamp
#
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File tools/b1_full_validate.ps1

# NOTE: "Continue", not "Stop": in PS 5.1 a native command that writes to stderr
# (cargo prints a harmless workspace-profile warning) becomes a TERMINATING error
# under ErrorActionPreference=Stop. Every step below checks $LASTEXITCODE instead.
$ErrorActionPreference = "Continue"
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $root
$exe = Join-Path $root "target/release/vmlm-b1-full.exe"
$data = Join-Path $root "artifacts/b1_scaling"
$TAG = [guid]::NewGuid().ToString("N").Substring(0,8)
$tmp = Join-Path $env:TEMP ("b1v_" + $TAG)
if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
Write-Host ("    temp dir: " + $tmp)

$script:pass = 0
$script:fail = 0
function Chk([string]$name, [bool]$ok, [string]$detail) {
    if ($ok) { $script:pass++; Write-Host ("  [PASS] {0}  {1}" -f $name, $detail) }
    else { $script:fail++; Write-Host ("  [FAIL] {0}  {1}" -f $name, $detail) }
}

function Run-Exe-Child([string[]]$a) {
    # 在**全新的子 PowerShell 进程**里跑被测程序并取回退出码/输出。
    # 为什么需要它：本脚本里 native 调用的流捕获在被测程序 `std::process::exit()`
    # 之后会失效（实测同一条命令：独立脚本里 exit=0 且有 2508 字符输出；
    # 在大脚本里 exit 是残留值且输出为空）。子进程把两件事都隔离开。
    $q = ($a | ForEach-Object { "'" + ($_ -replace "'", "''") + "'" }) -join ','
    $cmd = "& '" + $exe + "' @(" + $q + ") | Write-Host ; exit `$LASTEXITCODE"
    $o = & powershell -NoProfile -Command $cmd 2>&1
    return @{ code = $LASTEXITCODE; out = ($o -join "`n") }
}

function Run-Exe-Cmd([string[]]$a, [string]$log) {
    # 走 cmd.exe 起一个**全新进程**并把输出重定向到文件：
    # 大量 native 调用之后 PowerShell 的流捕获会失效（实测：同一命令在独立脚本里
    # 能抓到 2508 字符、退出码 0；在大脚本里却抓不到任何输出且退出码是上一次的残留）。
    # 这一步只影响"怎么测"，不影响被测代码；用 cmd 拿退出码是最稳的。
    $q = ($a | ForEach-Object { '"' + $_ + '"' }) -join ' '
    $line = '"' + $exe + '" ' + $q + ' > "' + $log + '" 2>&1'
    & cmd.exe /c $line | Out-Null
    $code = $LASTEXITCODE
    $out = ""
    if (Test-Path $log) { $out = (Get-Content $log -Raw) }
    return @{ code = $code; out = $out }
}

function Run-Exe-File([string[]]$a, [string]$log) {
    & $exe @a *> $log
    $code = $LASTEXITCODE
    $out = ""
    if (Test-Path $log) { $out = (Get-Content $log -Raw) }
    return @{ code = $code; out = $out }
}

function Run-Exe([string[]]$a) {
    $o = & $exe @a 2>&1
    return @{ code = $LASTEXITCODE; out = ($o -join "`n") }
}

Write-Host "===== B1 full training: driver validation ====="

# ---------------- V1 build ----------------
Write-Host "--- V1 build ---"
$b = & cargo build --release -p vmlm-train --bin vmlm-b1-full 2>&1
Chk "V1 build" ($LASTEXITCODE -eq 0) ("cargo build --release -p vmlm-train --bin vmlm-b1-full -> exit " + $LASTEXITCODE)
if ($LASTEXITCODE -ne 0) { Write-Host ($b -join "`n"); exit 1 }

# ---------------- V2 gradcheck, shallow ----------------
Write-Host "--- V2 numerical gradient check (dim=64 k0=4 layers=4) ---"
$r = Run-Exe @("--dim","64","--k0","4","--layers","4","--steps","2","--seq","16","--batch","2",
               "--lr","3e-4","--warmup","1","--seed","1","--data-dir",$data,
               "--out-dir",(Join-Path $tmp "gc_shallow"),"--gradcheck","1","--gc-probes","3")
$line = ($r.out -split "`n" | Where-Object { $_ -like "*GRADCHECK PASS*" -or $_ -like "*GRADCHECK FAIL*" } | Select-Object -First 1)
Chk "V2 gradcheck shallow" ($r.code -eq 0 -and $line -like "*PASS*") ($line)

# ---------------- V3 gradcheck, 24-layer path ----------------
Write-Host "--- V3 numerical gradient check through 24 layers (dim=32 k0=4) ---"
$r3 = Run-Exe @("--dim","32","--k0","4","--layers","24","--steps","2","--seq","16","--batch","2",
                "--lr","3e-4","--warmup","1","--seed","1","--data-dir",$data,
                "--out-dir",(Join-Path $tmp "gc_deep"),"--gradcheck","1","--gc-probes","2")
$line3 = ($r3.out -split "`n" | Where-Object { $_ -like "*GRADCHECK PASS*" -or $_ -like "*GRADCHECK FAIL*" } | Select-Object -First 1)
Chk "V3 gradcheck deep(24 layers)" ($r3.code -eq 0 -and $line3 -like "*PASS*") ($line3)

# ---------------- V4 smoke: artifacts + finite + loss down ----------------
Write-Host "--- V4 smoke run (dim=64 k0=4 layers=4, 40 steps) ---"
$smoke = Join-Path $tmp "smoke"
$r4 = Run-Exe @("--dim","64","--k0","4","--layers","4","--steps","40","--seq","16","--batch","2",
                "--lr","3e-3","--warmup","5","--min-lr","3e-4","--seed","1","--data-dir",$data,
                "--out-dir",$smoke,"--eval-every","10","--log-every","10","--plateau-patience","0",
                "--doc",(Join-Path $tmp "smoke.md"))
Chk "V4a smoke exit 0" ($r4.code -eq 0) ("exit " + $r4.code)
foreach ($f in @("metrics.jsonl","loss_curve.csv","summary.json")) {
    Chk ("V4b artifact " + $f) (Test-Path (Join-Path $smoke $f)) (Join-Path $smoke $f)
}
Chk "V4c doc written" (Test-Path (Join-Path $tmp "smoke.md")) (Join-Path $tmp "smoke.md")
$mj = Get-Content (Join-Path $smoke "metrics.jsonl") | Where-Object { $_.Trim() -ne "" } | ForEach-Object { $_ | ConvertFrom-Json }
$first = $mj[0]; $last = $mj[-1]
$allFinite = $true
foreach ($m in $mj) {
    foreach ($k in @("lr","train_loss","val_loss","val_ppl","wall_s")) {
        $v = [double]$m.$k
        if ([double]::IsNaN($v) -or [double]::IsInfinity($v)) { $allFinite = $false }
    }
}
Chk "V4d all metrics finite" $allFinite ("evals=" + $mj.Count)
Chk "V4e loss decreased" ($last.val_loss -lt $first.val_loss) ("val " + $first.val_loss + " -> " + $last.val_loss)
Chk "V4f lr schedule warmup+cosine" (([double]$mj[0].lr -lt [double]$mj[-1].lr) -eq $false -and [double]$mj[0].lr -gt 0) `
    ("lr first=" + $mj[0].lr + " last=" + $mj[-1].lr)
$sm = Get-Content (Join-Path $smoke "summary.json") -Raw | ConvertFrom-Json
Chk "V4g summary.json parses" ($null -ne $sm.dim -and $sm.loss_hash -like "0x*") ("loss_hash=" + $sm.loss_hash)
Chk "V4h not diverged" ($sm.diverged -eq $false) ("diverged=" + $sm.diverged + " stopped_by=" + $sm.stopped_by)

# ---------------- V5 determinism ----------------
Write-Host "--- V5 determinism: same seed twice ---"
$d1 = Join-Path $tmp "det1"; $d2 = Join-Path $tmp "det2"
$common = @("--dim","64","--k0","4","--layers","4","--steps","30","--seq","16","--batch","2",
            "--lr","3e-3","--warmup","5","--min-lr","3e-4","--seed","7","--data-dir",$data,
            "--eval-every","10","--log-every","10","--plateau-patience","0")
$null = Run-Exe ($common + @("--out-dir",$d1))
$null = Run-Exe ($common + @("--out-dir",$d2))
$s1 = Get-Content (Join-Path $d1 "summary.json") -Raw | ConvertFrom-Json
$s2 = Get-Content (Join-Path $d2 "summary.json") -Raw | ConvertFrom-Json
# NOTE: metrics.jsonl contains wall_s, which is timing and therefore legitimately differs.
# The determinism claim is about the NUMBERS, so compare everything except wall_s.
function Norm-Metrics([string]$p) {
    $rows = @()
    foreach ($l in (Get-Content $p | Where-Object { $_.Trim() -ne "" })) {
        $o = $l | ConvertFrom-Json
        $rows += ("{0}|{1}|{2}|{3}|{4}" -f $o.step, $o.lr, $o.train_loss, $o.val_loss, $o.val_ppl)
    }
    return ($rows -join "`n")
}
$n1 = Norm-Metrics (Join-Path $d1 "metrics.jsonl")
$n2 = Norm-Metrics (Join-Path $d2 "metrics.jsonl")
Chk "V5a summary parsed" ($null -ne $s1.dim -and $null -ne $s2.dim) ("hash1=" + $s1.loss_hash + " hash2=" + $s2.loss_hash)
Chk "V5b loss_hash identical (non-empty)" ($s1.loss_hash -like "0x*" -and $s1.loss_hash -eq $s2.loss_hash) `
    ($s1.loss_hash + " vs " + $s2.loss_hash)
Chk "V5c metrics identical (excluding wall_s)" ($n1.Length -gt 20 -and $n1 -eq $n2) ("rows=" + ($n1 -split "`n").Count)
Chk "V5d same final val" ([double]$s1.final_val -eq [double]$s2.final_val) ("val " + $s1.final_val + " vs " + $s2.final_val)

# ---------------- V6 can-fail: divergence guard ----------------
Write-Host "--- V6 can-fail: absurd lr must trip the divergence guard ---"
$dv = Join-Path $tmp "div"
$r6 = Run-Exe @("--dim","64","--k0","4","--layers","4","--steps","20","--seq","16","--batch","2",
                "--lr","50","--warmup","1","--seed","1","--data-dir",$data,
                "--out-dir",$dv,"--eval-every","5","--plateau-patience","0")
$s6 = Get-Content (Join-Path $dv "summary.json") -Raw | ConvertFrom-Json
Chk "V6a0 summary parses" ($null -ne $s6.dim -and $null -ne $s6.diverged_at) ("diverged=" + $s6.diverged + " at=" + $s6.diverged_at)
Chk "V6a divergence recorded" ($s6.diverged -eq $true -and [int]$s6.diverged_at -ge 1) `
    ("diverged=" + $s6.diverged + " diverged_at=" + $s6.diverged_at)
Chk "V6b stopped early" ([int]$s6.steps_done -ge 1 -and [int]$s6.steps_done -lt 20 -and $s6.stopped_by -eq "diverged") ("steps_done=" + $s6.steps_done + " of 20, stopped_by=" + $s6.stopped_by)

# ---------------- V7 can-fail: plateau guard ----------------
Write-Host "--- V7 can-fail: lr=0 must trip the plateau guard ---"
$pl = Join-Path $tmp "plat"
$r7 = Run-Exe @("--dim","64","--k0","4","--layers","4","--steps","200","--seq","16","--batch","2",
                "--lr","0","--warmup","1","--seed","1","--data-dir",$data,
                "--out-dir",$pl,"--eval-every","10","--plateau-patience","2")
$s7 = Get-Content (Join-Path $pl "summary.json") -Raw | ConvertFrom-Json
Chk "V7a0 summary parses" ($null -ne $s7.dim -and $null -ne $s7.stopped_by) ("stopped_by=" + $s7.stopped_by)
Chk "V7 plateau stop fires" ($s7.stopped_by -eq "plateau" -and [int]$s7.steps_done -lt 200) `
    ("stopped_by=" + $s7.stopped_by + " steps_done=" + $s7.steps_done + " of 200")

Write-Host ""
Write-Host "===== S) sparse-IO integration (lm_head + embedding on the B1 kernels) ====="

# ---------------- S1..S5: three INDEPENDENT k0, no fallback ----------------
Write-Host "--- S1..S5 required k0 flags (no default / no fallback) ---"
$base = @("--sparse-io","1","--dim","128","--steps","2","--seq","16","--batch","2",
          "--data-dir",$data,"--out-dir",(Join-Path $tmp "s_flag"))
$r = Run-Exe ($base + @("--k0","2","--k0-embed","4"))
Chk "S1 missing --k0-head errors" ($r.code -eq 2 -and $r.out -like "*--k0-head is REQUIRED*") ("exit " + $r.code)
$r = Run-Exe ($base + @("--k0","2","--k0-head","4"))
Chk "S2 missing --k0-embed errors" ($r.code -eq 2 -and $r.out -like "*--k0-embed is REQUIRED*") ("exit " + $r.code)
$r = Run-Exe ($base + @("--k0-embed","4","--k0-head","4"))
Chk "S3 missing --k0 errors" ($r.code -eq 2 -and $r.out -like "*--k0 is REQUIRED*") ("exit " + $r.code)
$r = Run-Exe @("--dim","128","--k0","2","--k0-embed","4","--steps","2","--data-dir",$data,"--out-dir",(Join-Path $tmp "s_flag"))
Chk "S4 --k0-embed without --sparse-io is refused" ($r.code -eq 2) ("exit " + $r.code)
$r = Run-Exe (@("--sparse-io","1","--dim","128","--k0","2","--k0-embed","4","--k0-head","8","--steps","2",
                "--data-dir",$data,"--out-dir",(Join-Path $tmp "s_flag")))
Chk "S5 k0-head != k0-embed refused (shared subspace)" ($r.code -eq 2) ("exit " + $r.code)

# ---------------- S6: dense-equivalent forward agreement ----------------
Write-Host "--- S6 sparse forward vs equivalent dense formulation ---"
$r = Run-Exe @("--sparse-io","1","--dim","128","--k0","2","--k0-embed","4","--k0-head","4","--layers","4",
               "--steps","2","--seq","16","--batch","2","--seed","1","--data-dir",$data,
               "--out-dir",(Join-Path $tmp "s_x"),"--io-xcheck","1")
$xl = ($r.out -split "`n" | Where-Object { $_ -like "*IO-XCHECK*" } | Select-Object -Last 1)
Chk "S6 io-xcheck bit-exact vs dense" ($r.code -eq 0 -and $xl -like "*PASS*") ($xl)

# ---------------- S7: FD gradient check in sparse-IO mode ----------------
Write-Host "--- S7 finite-difference gradient check (emb_tab / head / layers) ---"
$r = Run-Exe @("--sparse-io","1","--dim","128","--k0","2","--k0-embed","4","--k0-head","4","--layers","4",
               "--steps","2","--seq","16","--batch","2","--seed","1","--data-dir",$data,
               "--out-dir",(Join-Path $tmp "s_g"),"--gradcheck","1","--gc-probes","3","--gc-eps","0.5")
$gl = ($r.out -split "`n" | Where-Object { $_ -like "*GRADCHECK PASS*" -or $_ -like "*GRADCHECK FAIL*" } | Select-Object -First 1)
Chk "S7 sparse gradcheck" ($r.code -eq 0 -and $gl -like "*PASS*") ($gl)

# ---------------- S8: dim=128 k0=2/4/4 smoke, 8 steps ----------------
Write-Host "--- S8 smoke dim=128 k0=2/4/4, 8 steps ---"
$s8 = Join-Path $tmp "s_smoke"
$r = Run-Exe @("--sparse-io","1","--dim","128","--k0","2","--k0-embed","4","--k0-head","4","--layers","4",
               "--steps","8","--seq","16","--batch","2","--lr","3e-3","--warmup","2","--min-lr","3e-4",
               "--seed","1","--data-dir",$data,"--out-dir",$s8,"--doc",(Join-Path $tmp "s_smoke.md"),
               "--eval-every","4","--log-every","2","--plateau-patience","0")
Chk "S8a sparse smoke exit 0" ($r.code -eq 0) ("exit " + $r.code)
$j8 = Get-Content (Join-Path $s8 "summary.json") -Raw | ConvertFrom-Json
Chk "S8b summary parses (io_sparse=true)" ($null -ne $j8.io_sparse -and $j8.io_sparse -eq $true) ("io_sparse=" + $j8.io_sparse)
$m8 = Get-Content (Join-Path $s8 "metrics.jsonl") | Where-Object { $_.Trim() -ne "" } | ForEach-Object { $_ | ConvertFrom-Json }
$fin = $true
foreach ($m in $m8) { foreach ($k in @("lr","train_loss","val_loss","val_ppl")) { $v = [double]$m.$k; if ([double]::IsNaN($v) -or [double]::IsInfinity($v)) { $fin = $false } } }
Chk "S8c metrics finite" $fin ("evals=" + $m8.Count)

# ---------------- S9: parameter / compute accounting ----------------
Write-Host "--- S9 param + MACs accounting (closed form) ---"
$exp_main = 4 * 128 * 2
$exp_emb = 10921 * 4
$exp_head = 10921 * 4
Chk "S9a params match closed form" ([int]$j8.params_main -eq $exp_main -and [int]$j8.params_embed -eq $exp_emb -and [int]$j8.params_head -eq $exp_head -and [int]$j8.params_total -eq ($exp_main + $exp_emb + $exp_head)) `
    ("main=" + $j8.params_main + " embed=" + $j8.params_embed + " head=" + $j8.params_head + " total=" + $j8.params_total)
Chk "S9b head MACs = V*k0_head" ([int]$j8.macs_head -eq 10921 * 4) ("macs_head=" + $j8.macs_head)
Chk "S9c n_blocks = dim/k0_embed" ([int]$j8.n_blocks -eq 32) ("n_blocks=" + $j8.n_blocks)

# ---------------- S10: determinism in sparse-IO mode ----------------
Write-Host "--- S10 determinism (sparse-IO, same seed twice) ---"
$sa = Join-Path $tmp "s_det1"; $sb = Join-Path $tmp "s_det2"
$commonS = @("--sparse-io","1","--dim","128","--k0","2","--k0-embed","4","--k0-head","4","--layers","4",
             "--steps","30","--seq","16","--batch","2","--lr","3e-3","--warmup","5","--min-lr","3e-4",
             "--seed","7","--data-dir",$data,"--eval-every","10","--log-every","10","--plateau-patience","0")
$null = Run-Exe ($commonS + @("--out-dir",$sa))
$null = Run-Exe ($commonS + @("--out-dir",$sb))
$ja = Get-Content (Join-Path $sa "summary.json") -Raw | ConvertFrom-Json
$jb = Get-Content (Join-Path $sb "summary.json") -Raw | ConvertFrom-Json
Chk "S10a sparse loss_hash identical" ($ja.loss_hash -like "0x*" -and $ja.loss_hash -eq $jb.loss_hash) ($ja.loss_hash + " vs " + $jb.loss_hash)
Chk "S10b sparse metrics identical (excl wall_s)" ((Norm-Metrics (Join-Path $sa "metrics.jsonl")) -eq (Norm-Metrics (Join-Path $sb "metrics.jsonl"))) "compared"

Write-Host "--- S11 fused backward is BIT-EXACT vs the two scalar calls ---"
$f1 = Join-Path $tmp "s_fus1"; $f2 = Join-Path $tmp "s_fus2"
$cF = @("--sparse-io","1","--dim","128","--k0","2","--k0-embed","4","--k0-head","4","--layers","4",
        "--steps","40","--seq","16","--batch","2","--lr","1e-3","--warmup","5","--min-lr","1e-4",
        "--seed","11","--data-dir",$data,"--eval-every","20","--log-every","10","--plateau-patience","0")
$null = Run-Exe ($cF + @("--out-dir",$f1))
$null = Run-Exe ($cF + @("--out-dir",$f2,"--no-fusion","1"))
$jf1 = Get-Content (Join-Path $f1 "summary.json") -Raw | ConvertFrom-Json
$jf2 = Get-Content (Join-Path $f2 "summary.json") -Raw | ConvertFrom-Json
Chk "S11a fused vs scalar loss_hash identical" ($jf1.loss_hash -like "0x*" -and $jf1.loss_hash -eq $jf2.loss_hash) ($jf1.loss_hash + " vs " + $jf2.loss_hash)
# NOTE: compare as strings -- a diverged run writes final_val as JSON null, and
# [double]$null throws InvalidArgument (that is exactly what broke this check once).
Chk "S11b fused vs scalar final_val identical" ("$($jf1.final_val)" -eq "$($jf2.final_val)" -and "$($jf1.final_val)" -ne "") ("$($jf1.final_val) vs $($jf2.final_val)")

Write-Host "--- S12 dynamic subspace (方案一): swaps fire and usage stays balanced ---"
$d1 = Join-Path $tmp "s_dyn"
$r = Run-Exe @("--sparse-io","1","--dim","128","--k0","2","--k0-embed","4","--k0-head","4","--layers","4",
               "--steps","20","--seq","16","--batch","2","--lr","3e-3","--warmup","2","--seed","1",
               "--data-dir",$data,"--out-dir",$d1,"--eval-every","20","--plateau-patience","0",
               "--dyn-subspace","1","--dyn-swap-every","5")
$dl = ($r.out -split "`n" | Where-Object { $_ -like "*dyn-subspace:*" } | Select-Object -Last 1)
# ASCII-only parse: take the LAST "a..b" pair on the line (the per-dim usage range).
# NOTE: do NOT put non-ASCII in a .ps1 regex here -- PowerShell 5.1 reads a BOM-less
# UTF-8 script as the system codepage, so Chinese literals silently fail to match.
$ok = $false; $spread = -1
$mm = [regex]::Matches("$dl", "(\d+)\.\.(\d+)")
if ($mm.Count -gt 0) {
    $last = $mm[$mm.Count - 1]
    $spread = [int]$last.Groups[2].Value - [int]$last.Groups[1].Value
    $ok = ($spread -lt 64)
}
Chk "S12a dyn-subspace fires + balanced" ($r.code -eq 0 -and $ok) ($dl + " | spread=" + $spread)

Write-Host "--- S13 方案二 hot-pool head runs (hit rate is measured by --stats-out) ---"
$r = Run-Exe @("--sparse-io","1","--dim","128","--k0","2","--k0-embed","4","--k0-head","4","--layers","4",
               "--steps","8","--seq","16","--batch","2","--lr","3e-3","--warmup","2","--seed","1",
               "--data-dir",$data,"--out-dir",(Join-Path $tmp "s_hot"),"--eval-every","8",
               "--plateau-patience","0","--head-hot-pool","8")
Chk "S13 hot-pool head runs" ($r.code -eq 0) ("exit " + $r.code)

Write-Host "--- S14 k0_head > k0_embed refused (head reads the shared subspace) ---"
$r = Run-Exe @("--sparse-io","1","--dim","128","--k0","2","--k0-embed","4","--k0-head","8","--steps","2",
               "--data-dir",$data,"--out-dir",(Join-Path $tmp "s_flag"))
Chk "S14 k0-head > k0-embed refused" ($r.code -eq 2) ("exit " + $r.code)

Write-Host "--- F1..F3 attribution runs: freezing must be PROVABLE (param hashes) ---"
$cA = @("--sparse-io","1","--dim","128","--k0","2","--k0-embed","4","--k0-head","4","--layers","4",
        "--steps","30","--seq","16","--batch","2","--lr","3e-3","--warmup","5","--min-lr","3e-4",
        "--seed","3","--data-dir",$data,"--eval-every","30","--log-every","10","--plateau-patience","0")
# A: freeze sparse layers
$a1 = Join-Path $tmp "att_A"
$null = Run-Exe ($cA + @("--out-dir",$a1,"--freeze-layers","1"))
$jA = Get-Content (Join-Path $a1 "summary.json") -Raw | ConvertFrom-Json
Chk "F1 A: frozen layers unchanged" ($jA.hash_layers_before -eq $jA.hash_layers_after -and $jA.freeze_layers -eq $true) `
    ("layers " + $jA.hash_layers_before + " -> " + $jA.hash_layers_after)
Chk "F1b A: embed/head DID change" ($jA.hash_embed_before -ne $jA.hash_embed_after -and $jA.hash_head_before -ne $jA.hash_head_after) `
    ("embed " + ($jA.hash_embed_before -ne $jA.hash_embed_after) + " head " + ($jA.hash_head_before -ne $jA.hash_head_after))
# B: freeze embed + head
$b1d = Join-Path $tmp "att_B"
$null = Run-Exe ($cA + @("--out-dir",$b1d,"--freeze-io","1"))
$jB = Get-Content (Join-Path $b1d "summary.json") -Raw | ConvertFrom-Json
Chk "F2 B: frozen io unchanged" ($jB.hash_embed_before -eq $jB.hash_embed_after -and $jB.hash_head_before -eq $jB.hash_head_after) `
    ("embed " + $jB.hash_embed_before + " -> " + $jB.hash_embed_after + " ; head " + $jB.hash_head_before + " -> " + $jB.hash_head_after)
Chk "F2b B: layers DID change" ($jB.hash_layers_before -ne $jB.hash_layers_after) `
    ("layers " + $jB.hash_layers_before + " -> " + $jB.hash_layers_after)
# C: identity layers
$c1d = Join-Path $tmp "att_C"
$r = Run-Exe ($cA + @("--out-dir",$c1d,"--identity-layers","1","--dump-acts",(Join-Path $tmp "acts_C")))
$jC = Get-Content (Join-Path $c1d "summary.json") -Raw | ConvertFrom-Json
Chk "F3 C: identity layers untouched + finite" ($r.code -eq 0 -and $jC.identity_layers -eq $true -and $jC.hash_layers_before -eq $jC.hash_layers_after) `
    ("exit " + $r.code + " identity=" + $jC.identity_layers)

Write-Host "--- F4 activation dump + effective-rank script (experiment D tooling) ---"
$dd = Join-Path $tmp "acts_C"
Chk "F4a dump files written" ((Test-Path (Join-Path $dd "input.f32")) -and (Test-Path (Join-Path $dd "layer_00.f32")) -and (Test-Path (Join-Path $dd "layer_03.f32")) -and (Test-Path (Join-Path $dd "meta.json"))) $dd
$py = "python"
$efOut = Join-Path $tmp "effrank.md"
$eo = & $py (Join-Path $root "tools/b1_effrank.py") --dir $dd --out $efOut 2>&1
Chk "F4b effrank script runs" ($LASTEXITCODE -eq 0 -and (Test-Path $efOut)) (($eo | Select-Object -Last 1))

Write-Host "--- L1..L4 non-linearity step: linearity probe + ReLU ---"
$Ldir = Join-Path $root "artifacts/b1_full_validate"
$cL = @("--sparse-io","1","--dim","128","--k0","2","--k0-embed","4","--k0-head","4","--layers","4",
        "--steps","2","--seq","16","--batch","2","--seed","1","--data-dir",$data,"--out-dir",(Join-Path $Ldir "lin"))
# L1: 不加 ReLU => 必须报逐位线性（齐次性 0）
$r = Run-Exe ($cL + @("--linearity-probe","1")) (Join-Path $Ldir ("L1_" + $TAG + ".log"))
$l1 = ($r.out -split "`n" | Where-Object { $_ -like "*LINEARITY*" } | Select-Object -Last 1)
$r1code = $r.code
Write-Host ("    (L1 probe exit=" + $r1code + " ; last line: " + $l1 + ")")
Chk "L1 linear (no ReLU) bit-exact homogeneity" ($r1code -eq 0) ("probe exit=" + $r1code)
# L2: 加 ReLU => 探针必须报非线性（证明 ReLU 真的接上了）
$r = Run-Exe ($cL + @("--linearity-probe","1","--relu","1")) (Join-Path $Ldir ("L2_" + $TAG + ".log"))
$l2 = ($r.out -split "`n" | Where-Object { $_ -like "*LINEARITY*" } | Select-Object -Last 1)
Write-Host ("    (L2 probe exit=" + $r.code + "; text=" + $l2 + ")")
Chk "L2 non-linear with ReLU (probe must exit nonzero)" ($r.code -ne 0) ("probe exit=" + $r.code)
# L3: ReLU 反向导数数值对账（eps=0.05：太小被 f32 噪声淹、太大跨 ReLU 折点）。
# 判据用**退出码**：驱动 PASS 时正常返回(0)、FAIL 时 exit(1)。文本只打印、不参与判定
#（长脚本里 native 输出捕获不可靠；而且驱动 PASS 路径现在会正常 flush）。
$r = Run-Exe ($cL + @("--gradcheck","1","--gc-probes","3","--gc-eps","0.05","--relu","1"))
Write-Host ("    (L3 gradcheck exit=" + $r.code + ")")
Chk "L3 ReLU backward passes FD (eps=0.05)" ($r.code -eq 0) ("gradcheck exit=" + $r.code)
# L4: ReLU 训练冒烟：有限、确定性（同 seed 两次 hash 相同）
$Ldir = Join-Path $root "artifacts/b1_full_validate"
$r1 = Join-Path $Ldir "relu1"; $r2 = Join-Path $Ldir "relu2"
$cR = @("--sparse-io","1","--dim","128","--k0","2","--k0-embed","4","--k0-head","4","--layers","4",
        "--steps","20","--seq","16","--batch","2","--lr","1e-3","--warmup","3","--min-lr","1e-4",
        "--seed","5","--data-dir",$data,"--eval-every","20","--log-every","10","--plateau-patience","0","--relu","1")
$null = Run-Exe ($cR + @("--out-dir",$r1))
$null = Run-Exe ($cR + @("--out-dir",$r2))
$j1 = Get-Content (Join-Path $r1 "summary.json") -Raw | ConvertFrom-Json
$j2 = Get-Content (Join-Path $r2 "summary.json") -Raw | ConvertFrom-Json
Chk "L4 ReLU deterministic + finite" ($j1.loss_hash -like "0x*" -and $j1.loss_hash -eq $j2.loss_hash -and "$($j1.final_train)" -ne "null") `
    ($j1.loss_hash + " vs " + $j2.loss_hash)

Write-Host "--- K1..K3 cross-layer skip (embedding -> head) ---"
$Kdir = Join-Path $root "artifacts/b1_full_validate"
$cK = @("--sparse-io","1","--dim","128","--k0","2","--k0-embed","4","--k0-head","4","--layers","4",
        "--steps","20","--seq","16","--batch","2","--lr","1e-3","--warmup","3","--min-lr","1e-4",
        "--seed","9","--data-dir",$data,"--eval-every","20","--log-every","10","--plateau-patience","0","--head-skip","1")
$k1 = Join-Path $Kdir "skip1"; $k2 = Join-Path $Kdir "skip2"
$null = Run-Exe ($cK + @("--out-dir",$k1))
$null = Run-Exe ($cK + @("--out-dir",$k2))
$jk1 = Get-Content (Join-Path $k1 "summary.json") -Raw | ConvertFrom-Json
$jk2 = Get-Content (Join-Path $k2 "summary.json") -Raw | ConvertFrom-Json
$expParams = (4*128*2) + (10921*4) + 2*(10921*4)
Chk "K1 skip param accounting" ([int]$jk1.params_total -eq $expParams -and [int]$jk1.macs_head -eq 2*(10921*4)) `
    ("params=" + $jk1.params_total + " (expect " + $expParams + ") macs_head=" + $jk1.macs_head)
Chk "K2 skip deterministic + finite" ($jk1.loss_hash -like "0x*" -and $jk1.loss_hash -eq $jk2.loss_hash -and "$($jk1.final_train)" -ne "null") `
    ($jk1.loss_hash + " vs " + $jk2.loss_hash)
$r = Run-Exe (@("--sparse-io","1","--dim","128","--k0","2","--k0-embed","4","--k0-head","4","--layers","4",
        "--steps","2","--seq","16","--batch","2","--seed","1","--data-dir",$data,"--out-dir",(Join-Path $Kdir "skipg"),
        "--gradcheck","1","--gc-probes","3","--gc-eps","0.05","--head-skip","1"))
$kg = ($r.out -split "`n" | Where-Object { $_ -like "*GRADCHECK PASS*" -or $_ -like "*GRADCHECK FAIL*" } | Select-Object -First 1)
Chk "K3 skip backward passes FD (incl. emb via direct path)" ($r.code -eq 0 -and $kg -like "*PASS*") ($kg)

Write-Host "--- A1..A3 global attention (cross-token mixing) ---"
$Adir = Join-Path $env:TEMP ("b1a_" + $TAG)
New-Item -ItemType Directory -Force -Path $Adir | Out-Null
# A1: attention 冒烟 + 参数账目
$cA2 = @("--sparse-io","1","--dim","128","--k0","2","--k0-embed","4","--k0-head","4","--layers","4",
         "--steps","20","--seq","16","--batch","2","--lr","1e-3","--warmup","3","--min-lr","1e-4",
         "--seed","13","--data-dir",$data,"--eval-every","20","--log-every","10","--plateau-patience","0",
         "--attn-every","1")
$null = Run-Exe ($cA2 + @("--out-dir",(Join-Path $Adir "a1")))
$ja = Get-Content (Join-Path $Adir "a1\summary.json") -Raw | ConvertFrom-Json
Chk "A1 attention smoke + params" ([int]$ja.attn_blocks -eq 4 -and "$($ja.final_train)" -ne "null") `
    ("attn_blocks=" + $ja.attn_blocks + " final_train=" + $ja.final_train)
# A2: attention 反向 FD 对账（attn_wo 探针）
$r = Run-Exe (@("--sparse-io","1","--dim","64","--k0","2","--k0-embed","4","--k0-head","4","--layers","2",
        "--steps","2","--seq","16","--batch","2","--seed","1","--data-dir",$data,"--out-dir",(Join-Path $Adir "a2"),
        "--gradcheck","1","--gc-probes","2","--gc-eps","0.05","--attn-every","1"))
Chk "A2 attention backward passes FD" ($r.code -eq 0) ("gradcheck exit=" + $r.code)
# A3: 默认（不开 attention）必须与历史哈希逐位相同 => 新代码没有动默认路径
$null = Run-Exe (@("--sparse-io","1","--dim","128","--k0","2","--k0-embed","4","--k0-head","4","--layers","4",
        "--steps","30","--seq","16","--batch","2","--lr","3e-3","--warmup","5","--min-lr","3e-4","--seed","7",
        "--data-dir",$data,"--out-dir",(Join-Path $Adir "a3"),"--eval-every","10","--log-every","10","--plateau-patience","0"))
$j3 = Get-Content (Join-Path $Adir "a3\summary.json") -Raw | ConvertFrom-Json
Chk "A3 default path bit-identical to history" ($j3.loss_hash -eq "0xc9c987c15d75d1d6") ("hash=" + $j3.loss_hash)

Write-Host "--- A4..A6 sparse attention: src sharing + q.k health ---"
$Sdir = Join-Path $env:TEMP ("b1s_" + $TAG)
New-Item -ItemType Directory -Force -Path $Sdir | Out-Null
$cS = @("--sparse-io","1","--dim","128","--k0","2","--k0-embed","4","--k0-head","4","--layers","8",
        "--steps","8","--seq","16","--batch","2","--lr","1e-3","--warmup","3","--min-lr","1e-4",
        "--seed","1","--data-dir",$data,"--out-dir",(Join-Path $Sdir "s1"),
        "--eval-every","8","--log-every","4","--plateau-patience","0",
        "--attn-every","4","--attn-sparse","1","--k0-attn","2")
# A4: 方案 A（Q/K 共享 src_qk, V/O 共享 src_vo）+ 尺度标定 => q.k 体检通过
$r = Run-Exe $cS
$s1 = Get-Content (Join-Path $Sdir "s1\summary.json") -Raw | ConvertFrom-Json
Chk "A4 sparse attn (shared src) passes q.k health" ($r.code -eq 0 -and "$($s1.final_train)" -ne "null") `
    ("exit=" + $r.code + " final_train=" + $s1.final_train)
# A5: 关掉尺度标定 => max|q.k| 远小于 0.01 => 体检必须**报错**（这条是 can-fail 判据）
$r = Run-Exe ($cS + @("--attn-calib","0")) 
Chk "A5 uncalibrated scores are DETECTED (must exit 2)" ($r.code -eq 2) ("exit=" + $r.code)
# A6: FD 覆盖 Q/K/V/O 四组（稀疏路径、满宽投影，让 Q/K 信号可测）
$r = Run-Exe (@("--sparse-io","1","--dim","64","--k0","2","--k0-embed","8","--k0-head","8","--layers","2",
        "--steps","2","--seq","16","--batch","4","--seed","1","--data-dir",$data,"--out-dir",(Join-Path $Sdir "s3"),
        "--gradcheck","1","--gc-probes","3","--gc-eps","0.05","--attn-every","1","--attn-sparse","1","--k0-attn","64"))
$s6 = ($r.out -split "`n" | Where-Object { $_ -like "*GRADCHECK PASS*" -or $_ -like "*GRADCHECK FAIL*" } | Select-Object -Last 1)
Chk "A6 sparse attn FD covers Q/K/V/O" ($r.code -eq 0 -and $s6 -like "*PASS*") ($s6)

Write-Host "--- A7 checkpoint round-trip: calibration must persist across paths ---"
$Cdir = Join-Path $env:TEMP ("b1c_" + $TAG)
New-Item -ItemType Directory -Force -Path $Cdir | Out-Null
$ck = Join-Path $Cdir "m.cpk"
$cC = @("--sparse-io","1","--dim","128","--k0","2","--k0-embed","4","--k0-head","4","--layers","8",
        "--steps","3","--seq","16","--batch","2","--lr","1e-3","--warmup","1","--min-lr","1e-4",
        "--seed","1","--data-dir",$data,"--eval-every","3","--log-every","3","--plateau-patience","0",
        "--attn-every","4","--attn-sparse","1","--k0-attn","2")
$null = Run-Exe ($cC + @("--out-dir",(Join-Path $Cdir "c1"),"--save-model",$ck))
Chk "A7a checkpoint written" (Test-Path $ck) ("size=" + (Get-Item $ck).Length)
$r2 = Run-Exe ($cC + @("--out-dir",(Join-Path $Cdir "c2"),"--steps","1","--load-model",$ck))
$t2 = $r2.out
Chk "A7b load works + calibration persists (g approx 1)" ($r2.code -eq 0 -and $t2 -like "*载入权重*" -and $t2 -notlike "*FATAL*") `
    ("exit=" + $r2.code)
$r3 = Run-Exe ($cC + @("--out-dir",(Join-Path $Cdir "c3"),"--steps","1","--attn-calib","0"))
Chk "A7c --attn-calib 0 is still detected as collapse (exit 2)" ($r3.code -eq 2) ("exit=" + $r3.code)

Write-Host ""
Write-Host ("===== driver validation: checks={0} fail={1} =====" -f ($script:pass + $script:fail), $script:fail)
if ($script:fail -ne 0) { exit 1 }
exit 0
