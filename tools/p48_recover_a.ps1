# p48_recover_a.ps1 -- 崩溃恢复 A 线：稠密长程（d=192 续跑 → d=256）
$ErrorActionPreference = "Stop"
$py = (Get-Command python).Source
$s = 'D:\czsai\vmlm\tools\p35_dyn_v2.py'
$o = 'D:\czsai\vmlm\artifacts\p35'

$arms = @(
  @{ t = 'd192_lr1e3_t4000'; a = @('--d', '192') },
  @{ t = 'd256_lr1e3_4000'; a = @('--d', '256') }
)
foreach ($x in $arms) {
  "=== start $($x.t) $(Get-Date -Format HH:mm:ss)"
  $al = @($s, '--mask-src', 'none') + $x.a + @('--lr', '1e-3', '--min-lr', '1e-4',
          '--steps', '4000', '--eval-every', '400', '--threads', '2',
          '--ckpt-every', '100', '--resume', '1', '--tag', $x.t)
  & $py @al *> "$o\$($x.t).log"
  "=== done $($x.t) exit=$LASTEXITCODE $(Get-Date -Format HH:mm:ss)"
}
"=== 完成 ==="
