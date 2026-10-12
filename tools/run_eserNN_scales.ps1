# ==========================================================================
# P8-C2: ESERNN scale-up tiers -- generate + assemble + verify (member C2)
#
#   powershell -File tools/run_eserNN_scales.ps1                 # default tiers
#   powershell -File tools/run_eserNN_scales.ps1 -Only h512_l6
#   powershell -File tools/run_eserNN_scales.ps1 -Matrix         # hidden x layers grid
#
# ASCII only on purpose: Windows PowerShell 5.1 reads this file as GBK, so
# non-ASCII would break parsing (the generator's own output is UTF-8 and is
# captured as text, never parsed as source).
#
# Output per tier: examples/eserNN/gen_<tag>\{step.vml,step.vmb,weights.args,ref_*}
# Summary: examples/eserNN/scale_report.tsv  (or scale_matrix.tsv)
# ==========================================================================
param(
    [string]$Only = "",
    [switch]$Matrix
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$asm  = "$root\target\release\vmlm-asm.exe"
$drv  = "$root\target\release\vmlm-eserNN.exe"
$ex   = "$root\examples\eserNN"

if ($Matrix) {
    $scales = @()
    foreach ($h in 64, 128, 256, 512) {
        foreach ($nl in 1, 2, 3, 6) {
            $scales += @{ tag = "m_h${h}_l${nl}"; args = @("--hidden", "$h", "--layers", "$nl") }
        }
    }
    $tsv = "$ex\scale_matrix.tsv"
} else {
    $scales = @(
        @{ tag = "mini";    args = @("--cfg", "mini") },
        @{ tag = "h128_l2"; args = @("--hidden", "128", "--layers", "2") },
        @{ tag = "h256_l3"; args = @("--hidden", "256", "--layers", "3") },
        @{ tag = "h512_l6"; args = @("--hidden", "512", "--layers", "6") }
    )
    $tsv = "$ex\scale_report.tsv"
}

$rows = @()
foreach ($s in $scales) {
    if ($Only -ne "" -and $s.tag -ne $Only) { continue }
    $dir = "$ex\gen_$($s.tag)"
    Write-Host "`n========== $($s.tag) =========="

    $t0 = Get-Date
    & python "$root\tools\gen_eserNN.py" @($s.args) @("--out-dir", $dir) | Out-Null
    if ($LASTEXITCODE -ne 0) { Write-Host "GEN FAILED"; continue }
    $gen_ms = ((Get-Date) - $t0).TotalMilliseconds

    Push-Location $dir
    $wa = @("step.vml", "step.vmb") + (Get-Content weights.args | Where-Object { $_.Trim() -ne "" })
    $t1 = Get-Date
    $asmOut = & $asm @wa 2>&1
    $asm_ms = ((Get-Date) - $t1).TotalMilliseconds
    Pop-Location
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ASM FAILED: $asmOut"
        $rows += [pscustomobject]@{ tag = $s.tag; n_ins = "-"; vmb_MB = "-";
            asm_ms = [int]$asm_ms; load_ms = "-"; fwd_ms_token = "-";
            err_real = "-"; verdict = "ASM_FAIL" }
        continue
    }
    $asmLine = ($asmOut | Select-Object -First 1)
    Write-Host "  $asmLine"
    $nIns = [int]([regex]::Match($asmLine, "OK:\s+(\d+)\s+").Groups[1].Value)
    $sz = (Get-Item "$dir\step.vmb").Length / 1MB

    $o = & $drv $dir 2>&1
    $code = $LASTEXITCODE
    $flat = $o -join "`n"
    $comb = ($o | Select-String -Pattern "= ([\d.]+e[+-]\d+)\s*$" | Select-Object -Last 2)
    $errReal = [double]::NaN
    # The driver prints:  logits X / st_S Y / x_prev Z  =>  combined W
    # (the separator is a non-ASCII arrow, so match on the label instead)
    if ($flat -match "logits\s+([\d.eE+-]+)\s*/\s*st_S\s+([\d.eE+-]+)\s*/\s*x_prev\s+([\d.eE+-]+)\s*\S*\s*([\d.eE+-]+)") {
        $errReal = [double]$Matches[4]
    }
    $loadMs = [double]::NaN
    if ($flat -match "Vm::load = ([\d.]+)") { $loadMs = [double]$Matches[1] }
    $tokMs = [double]::NaN
    if ($flat -match "([\d.]+) ms/token") { $tokMs = [double]$Matches[1] }

    $rows += [pscustomobject]@{
        tag = $s.tag; n_ins = $nIns; vmb_MB = [math]::Round($sz, 2)
        asm_ms = [int]$asm_ms; load_ms = $loadMs; fwd_ms_token = $tokMs
        err_real = $errReal; verdict = $(if ($code -eq 0) { "PASS" } else { "FAIL" })
    }
    Write-Host ("  ins={0} vmb={1:N2}MB asm={2}ms load={3}ms tok={4}ms err={5}" -f `
        $nIns, $sz, [int]$asm_ms, $loadMs, $tokMs, $errReal)
}

if ($rows.Count -gt 0) {
    $rows | Export-Csv -Path $tsv -Delimiter "`t" -NoTypeInformation -Encoding UTF8
    Write-Host "`n===== summary: $tsv ====="
    $rows | Format-Table -AutoSize
}
