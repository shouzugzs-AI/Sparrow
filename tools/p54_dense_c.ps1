# p54_dense_c.ps1 -- C 线（p35）：稠密 lr 括号 + C 系列 keep=10% 补点
$ErrorActionPreference = "Stop"
$py = (Get-Command python).Source
$s = 'D:\czsai\vmlm\tools\p35_dyn_v2.py'
$o = 'D:\czsai\vmlm\artifacts\p35'

$arms = @(
  @{ t = 'd256_lr2e3_1000'; a = @('--mask-src', 'none', '--d', '256', '--lr', '2e-3', '--min-lr', '2e-4') },
  @{ t = 'c10_lr3e3_1000'; a = @('--mask-src', 'h', '--keep', '0.10', '--lr', '3e-3', '--min-lr', '3e-4') },
  @{ t = 'd192_lr1e3_2000'; a = @('--mask-src', 'none', '--d', '192', '--lr', '1e-3', '--min-lr', '1e-4') }
)
foreach ($x in $arms) {
  "=== start $($x.t) $(Get-Date -Format HH:mm:ss)"
  $steps = if ($x.t -like '*2000') { '2000' } else { '1000' }
  $al = @($s) + $x.a + @('--warmup', '30', '--steps', $steps, '--eval-every', '200',
          '--threads', '1', '--ckpt-every', '100', '--resume', '1', '--tag', $x.t)
  & $py @al *> "$o\$($x.t).log"
  "=== done $($x.t) exit=$LASTEXITCODE $(Get-Date -Format HH:mm:ss)"
}
"=== 完成 ==="
