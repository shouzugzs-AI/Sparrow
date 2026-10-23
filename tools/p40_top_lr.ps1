# 探顶：C1/C5 在 1e-2 下是否还稳（600 步）。C1 在 5e-3 仍未停涨
$py = 'python'
$s = 'D:\czsai\vmlm\tools\p35_dyn_v2.py'
$o = 'D:\czsai\vmlm\artifacts\p35'
$runs = @(
  @('c1_lr1e2_600', @('--mask-src','h','--keep','0.01'), '80'),
  @('c5_lr1e2_600', @('--mask-src','h','--keep','0.05'), '80')
)
foreach ($r in $runs) {
  $tag = $r[0]
  "=== start $tag lr=1e-2"
  & $py $s @($r[1]) --lr 0.01 --min-lr 1e-3 --warmup 200 --steps 600 --eval-every 100 --threads 2 --tag $tag *> (Join-Path $o "$tag.log")
  "=== done $tag exit=$LASTEXITCODE"
}
"=== 探顶结束 ==="
