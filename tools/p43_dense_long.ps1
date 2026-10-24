# p43_dense_long.ps1 -- 环境崩溃后重跑稠密长程（带 checkpoint，崩溃可续）
$ErrorActionPreference = "Stop"
$py = (Get-Command python).Source
$s = 'D:\czsai\vmlm\tools\p35_dyn_v2.py'
$o = 'D:\czsai\vmlm\artifacts\p35'

$arms = @(
  @{ t = 'd192_lr1e3_t4000'; a = @('--mask-src', 'none', '--d', '192', '--lr', '1e-3', '--min-lr', '1e-4') },
  @{ t = 'd256_lr1e3_4000'; a = @('--mask-src', 'none', '--d', '256', '--lr', '1e-3', '--min-lr', '1e-4') }
)
foreach ($x in $arms) {
  "=== start $($x.t) $(Get-Date -Format HH:mm:ss)"
  $al = @($s) + $x.a + @('--steps', '4000', '--eval-every', '400', '--threads', '2',
                         '--ckpt-every', '100', '--resume', '1', '--tag', $x.t)
  & $py @al *> "$o\$($x.t).log"
  "=== done $($x.t) exit=$LASTEXITCODE $(Get-Date -Format HH:mm:ss)"
}
"=== 完成 ==="
