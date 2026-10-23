# 高学习率检验：小 keep 是否因"只更新选中通道"而能承受更高 lr
# 对照同时跑稠密，避免把"高 lr 普遍更好"误判成"小 keep 特有的鲁棒性"
$py = 'python'
$s = 'D:\czsai\vmlm\tools\p35_dyn_v2.py'
$o = 'D:\czsai\vmlm\artifacts\p35'
$runs = @(
  @('c1_lr1e3',  @('--mask-src','h','--keep','0.01'), '0.001'),
  @('c5_lr1e3',  @('--mask-src','h','--keep','0.05'), '0.001'),
  @('c15_lr1e3', @('--mask-src','h','--keep','0.15'), '0.001'),
  @('d256_lr1e3',@('--mask-src','none'), '0.001')
)
foreach ($r in $runs) {
  $tag = $r[0]
  "=== start $tag"
  & $py $s @($r[1]) --lr $r[2] --min-lr 1e-4 --steps 1000 --eval-every 100 --threads 2 --tag $tag *> (Join-Path $o "$tag.log")
  "=== done $tag exit=$LASTEXITCODE"
}
"=== 高学习率组结束 ==="
