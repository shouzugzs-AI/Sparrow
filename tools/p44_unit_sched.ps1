# p44_unit_sched.ps1 -- 逐通道 warmup/cosine（--sched unit）与全局调度的 A/B
# 对照组：c5_lr5e3_600 = 5.4708，c1_lr5e3_600 = 6.0014（同协议同 lr，仅调度不同）
$ErrorActionPreference = "Stop"
$py = (Get-Command python).Source
$s = 'D:\czsai\vmlm\tools\p35_dyn_v2.py'
$o = 'D:\czsai\vmlm\artifacts\p35'

$arms = @(
  @{ t = 'c5_unit_lr5e3_600'; k = '0.05' },
  @{ t = 'c1_unit_lr5e3_600'; k = '0.01' }
)
foreach ($x in $arms) {
  "=== start $($x.t) $(Get-Date -Format HH:mm:ss)"
  $al = @($s, '--mask-src', 'h', '--keep', $x.k, '--lr', '5e-3', '--min-lr', '5e-4',
          '--warmup', '30', '--steps', '600', '--eval-every', '100', '--threads', '2',
          '--sched', 'unit', '--ckpt-every', '100', '--resume', '1', '--tag', $x.t)
  & $py @al *> "$o\$($x.t).log"
  "=== done $($x.t) exit=$LASTEXITCODE $(Get-Date -Format HH:mm:ss)"
}
"=== 完成 ==="
