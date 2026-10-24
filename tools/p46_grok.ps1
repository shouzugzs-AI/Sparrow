# p46_grok.ps1 -- 细粒度复现 C15@3e-3 的早期轨迹（eval 每 25 步），看 400→800 的那次结构性收缩是台阶还是斜坡
# lr 曲线按 --steps 4000 走，只跑到 --stop-at 950；同一 seed 应逐点复现 c15_lr3e3_4000 在 400/800 的 val
$ErrorActionPreference = "Stop"
$py = (Get-Command python).Source
$s = 'D:\czsai\vmlm\tools\p35_dyn_v2.py'
$o = 'D:\czsai\vmlm\artifacts\p35'

$arms = @(
  @{ t = 'hi_c15_lr3e3'; lr = '3e-3' },
  @{ t = 'hi_c15_lr2e3'; lr = '2e-3' }
)
foreach ($x in $arms) {
  "=== start $($x.t) $(Get-Date -Format HH:mm:ss)"
  $al = @($s, '--mask-src', 'h', '--keep', '0.15', '--lr', $x.lr, '--min-lr', '3e-4',
          '--warmup', '60', '--steps', '4000', '--stop-at', '950', '--eval-every', '25',
          '--threads', '2', '--ckpt-every', '100', '--resume', '1', '--tag', $x.t)
  & $py @al *> "$o\$($x.t).log"
  "=== done $($x.t) exit=$LASTEXITCODE $(Get-Date -Format HH:mm:ss)"
}
"=== 完成 ==="
