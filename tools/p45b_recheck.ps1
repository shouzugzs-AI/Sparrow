# p45b_recheck.ps1 -- 用新的「逐 token 跨层覆盖」指标重跑 keep=1%/2%（244 步，同 e_h01/h_k02 协议）
$ErrorActionPreference = "Stop"
$py = (Get-Command python).Source
$s = 'D:\czsai\vmlm\tools\p35_dyn_v2.py'
$o = 'D:\czsai\vmlm\artifacts\p35'

$arms = @(
  @{ t = 'm_h01'; k = '0.01' },
  @{ t = 'm_h02'; k = '0.02' }
)
foreach ($x in $arms) {
  "=== start $($x.t) $(Get-Date -Format HH:mm:ss)"
  $al = @($s, '--mask-src', 'h', '--keep', $x.k, '--lr', '3e-4', '--min-lr', '3e-5',
          '--warmup', '30', '--steps', '244', '--eval-every', '50', '--threads', '1',
          '--ckpt-every', '50', '--resume', '1', '--tag', $x.t)
  & $py @al *> "$o\$($x.t).log"
  "=== done $($x.t) exit=$LASTEXITCODE $(Get-Date -Format HH:mm:ss)"
}
"=== 完成 ==="
