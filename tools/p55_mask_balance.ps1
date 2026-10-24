# p55_mask_balance.ps1 -- aux-loss-free 负载均衡（每层每通道偏置推向均匀选中率）
# 目的：修跨层/跨通道塌缩（keep=1% 主掩码 0.85、L=24 keep=2.5% 实测覆盖只有 0.057）
$ErrorActionPreference = "Stop"
$py = (Get-Command python).Source
$s = 'D:\czsai\vmlm\tools\p35_dyn_v2.py'
$o = 'D:\czsai\vmlm\artifacts\p35'
$arms = @(
  @{ t='mb_h01_244';    a=@('--mask-src','h','--keep','0.01','--lr','3e-4','--min-lr','3e-5','--warmup','30','--steps','244','--eval-every','50','--layers','12','--mask-balance','0.2') },
  @{ t='mb_h05_600';    a=@('--mask-src','h','--keep','0.05','--lr','5e-3','--min-lr','5e-4','--warmup','30','--steps','600','--eval-every','100','--layers','12','--mask-balance','0.2') },
  @{ t='mb_L24_k25_244';a=@('--mask-src','h','--keep','0.025','--lr','3e-4','--min-lr','3e-5','--warmup','30','--steps','244','--eval-every','50','--layers','24','--mask-balance','0.2') }
)
foreach ($x in $arms) {
  "=== start $($x.t) $(Get-Date -Format HH:mm:ss)"
  $al = @($s) + $x.a + @('--threads','1','--ckpt-every','100','--resume','1','--tag',$x.t)
  & $py @al *> "$o\$($x.t).log"
  "=== done $($x.t) exit=$LASTEXITCODE $(Get-Date -Format HH:mm:ss)"
}
"=== 完成 ==="
