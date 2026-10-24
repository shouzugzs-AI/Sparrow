# p45_keep_layers.ps1 -- 检验「最低可用 keep ≈ u*/L」：固定覆盖率与总读取量，只改层数
#   L=12, keep=5%  -> 覆盖 1-0.95^12=46%, 总读取 0.60d（已有 e_h05 = 6.9033 @244 步 lr3e-4）
#   L=6 , keep=10% -> 覆盖 1-0.90^6 =47%, 总读取 0.60d
#   L=24, keep=2.5%-> 覆盖 1-0.975^24=46%, 总读取 0.60d
# 另加 L=24 d=120（等 MAC 口径：约 1.85M ≈ C5 的 1.87M）
$ErrorActionPreference = "Stop"
$py = (Get-Command python).Source
$s = 'D:\czsai\vmlm\tools\p35_dyn_v2.py'
$o = 'D:\czsai\vmlm\artifacts\p35'

$arms = @(
  @{ t = 'Lk_L6_k10'; a = @('--layers', '6', '--d', '256', '--keep', '0.10') },
  @{ t = 'Lk_L24_k25'; a = @('--layers', '24', '--d', '256', '--keep', '0.025') },
  @{ t = 'Lk_L24_d120_k25'; a = @('--layers', '24', '--d', '120', '--heads', '2', '--hidden', '480', '--keep', '0.025') }
)
foreach ($x in $arms) {
  "=== start $($x.t) $(Get-Date -Format HH:mm:ss)"
  $al = @($s, '--mask-src', 'h') + $x.a + @('--lr', '3e-4', '--min-lr', '3e-5',
          '--warmup', '30', '--steps', '244', '--eval-every', '50', '--threads', '1',
          '--ckpt-every', '50', '--resume', '1', '--tag', $x.t)
  & $py @al *> "$o\$($x.t).log"
  "=== done $($x.t) exit=$LASTEXITCODE $(Get-Date -Format HH:mm:ss)"
}
"=== 完成 ==="
