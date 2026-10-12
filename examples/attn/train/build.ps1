# P8 member F: assemble examples/attn/train/train_step.vml -> train_step.vmb
#
# Usage (from repo root):
#   pwsh -File examples/attn/train/build.ps1
#
# The constant list comes from asm.args (written by make_attn_train.py) so we
# never hand-copy 30 name:file:rows:cols triples.
# Note: do NOT write the .vml with Out-File -Encoding utf8 (a BOM makes the
# assembler report "unknown instruction").

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $root

$dir = "examples/attn/train"
$asm = "target\release\vmlm-asm.exe"

$argv = @("$dir/train_step.vml", "$dir/train_step.vmb")
Get-Content "$dir/asm.args" | ForEach-Object {
    $line = $_.Trim()
    if ($line -eq "" -or $line.StartsWith("#")) { return }
    $p = $line.Split(":")
    $argv += "$($p[0]):$dir/$($p[1]):$($p[2]):$($p[3])"
}

& $asm @argv
