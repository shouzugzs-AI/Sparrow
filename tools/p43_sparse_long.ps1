# p43_sparse_long.ps1 -- 环境崩溃后重跑 C15 lr 复验（带 checkpoint，崩溃可续）
$ErrorActionPreference = "Stop"
$py = (Get-Command python).Source
$s = 'D:\czsai\vmlm\tools\p35_dyn_v2.py'
$o = 'D:\czsai\vmlm\artifacts\p35'

$arms = @(
  @{ t = 'c15_lr3e3_4000'; a = @('--mask-src', 'h', '--keep', '0.15', '--lr', '3e-3', '--min-lr', '3e-4') },
  @{ t = 'c15_lr5e3_4000'; a = @('--mask-src', 'h', '--keep', '0.15', '--lr', '5e-3', '--min-lr', '5e-4') }
)
foreach ($x in $arms) {
  "=== start $($x.t) $(Get-Date -Format HH:mm:ss)"
  $al = @($s) + $x.a + @('--warmup', '60', '--steps', '4000', '--eval-every', '400', '--threads', '2',
                         '--ckpt-every', '100', '--resume', '1', '--tag', $x.t)
  & $py @al *> "$o\$($x.t).log"
  "=== done $($x.t) exit=$LASTEXITCODE $(Get-Date -Format HH:mm:ss)"
}
"=== 完成 ==="
