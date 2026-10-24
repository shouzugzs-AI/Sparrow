# p56_cov_penalty.ps1 -- 覆盖感知掩码：同一 token 前几层已读过的通道降权
# 目的：把实测覆盖从「2.5×keep」推向理论上限 L·keep（keep=1%/12 层时 0.025 → 0.12）
$ErrorActionPreference = "Stop"
$py = (Get-Command python).Source
$s = 'D:\czsai\vmlm\tools\p35_dyn_v2.py'
$o = 'D:\czsai\vmlm\artifacts\p35'
$arms = @(
  @{ t='cov_h01_244';     a=@('--mask-src','h','--keep','0.01','--lr','3e-4','--min-lr','3e-5','--warmup','30','--steps','244','--eval-every','50','--layers','12') },
  @{ t='cov_h05_600';     a=@('--mask-src','h','--keep','0.05','--lr','5e-3','--min-lr','5e-4','--warmup','30','--steps','600','--eval-every','100','--layers','12') },
  @{ t='cov_L24_k25_244'; a=@('--mask-src','h','--keep','0.025','--lr','3e-4','--min-lr','3e-5','--warmup','30','--steps','244','--eval-every','50','--layers','24') }
)
foreach ($x in $arms) {
  "=== start $($x.t) $(Get-Date -Format HH:mm:ss)"
  $al = @($s) + $x.a + @('--cov-penalty','0.3','--threads','1','--ckpt-every','100','--resume','1','--tag',$x.t)
  & $py @al *> "$o\$($x.t).log"
  "=== done $($x.t) exit=$LASTEXITCODE $(Get-Date -Format HH:mm:ss)"
}
"=== 完成 ==="
