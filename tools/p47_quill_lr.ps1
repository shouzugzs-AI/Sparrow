# p47_quill_lr.ps1 -- Quill 的 lr 阶梯 + 注意力密度扫描（全部 1000 步，与稠密 d=256@lr1e-3 的 5.0966 同协议）
# 背景：此前所有 Quill 臂都用 lr=3e-4，而同一 lr 下稠密臂比自己的最优（lr1e-3, 5.0966）差 0.43，
#       所以「Quill 与稠密打平」这个结论是没调 lr 的产物，必须重测。
$ErrorActionPreference = "Stop"
$py = (Get-Command python).Source
$s = 'D:\czsai\vmlm\tools\p36_attnonce.py'
$o = 'D:\czsai\vmlm\artifacts\p36'

$arms = @(
  @{ t = 'q1_lr1e3_1000';  a = @('--attn-every', '12') },
  @{ t = 'qk3_lr1e3_1000'; a = @('--attn-every', '3') },
  @{ t = 'qk2_lr1e3_1000'; a = @('--attn-every', '2') }
)
foreach ($x in $arms) {
  "=== start $($x.t) $(Get-Date -Format HH:mm:ss)"
  $al = @($s, '--share-attn', '1', '--seq', '128', '--batch', '32') + $x.a + @(
          '--lr', '1e-3', '--min-lr', '1e-4', '--warmup', '30', '--steps', '1000',
          '--eval-every', '200', '--threads', '1', '--tag', $x.t)
  & $py @al *> "$o\$($x.t).log"
  "=== done $($x.t) exit=$LASTEXITCODE $(Get-Date -Format HH:mm:ss)"
}
"=== 完成 ==="
