# 高学习率扫描（600 步快速定位）：C1/C5 的理想 lr 可能很高，先粗扫到 5e-3
# 600 步 = 2.46M token，约 22 分钟/臂；lr 越大 warmup 越长，防止开局炸
$py = 'python'
$s = 'D:\czsai\vmlm\tools\p35_dyn_v2.py'
$o = 'D:\czsai\vmlm\artifacts\p35'
"=== 高 lr 粗扫（600 步）开始 ==="
$runs = @(
  @('c1_lr2e3_600', @('--mask-src','h','--keep','0.01'), '0.002',  '60'),
  @('c1_lr3e3_600', @('--mask-src','h','--keep','0.01'), '0.003',  '80'),
  @('c1_lr5e3_600', @('--mask-src','h','--keep','0.01'), '0.005',  '120'),
  @('c5_lr2e3_600', @('--mask-src','h','--keep','0.05'), '0.002',  '60'),
  @('c5_lr3e3_600', @('--mask-src','h','--keep','0.05'), '0.003',  '80'),
  @('c5_lr5e3_600', @('--mask-src','h','--keep','0.05'), '0.005',  '120'),
  @('c15_lr3e3_600',@('--mask-src','h','--keep','0.15'), '0.003',  '80')
)
foreach ($r in $runs) {
  $tag = $r[0]; $lr = [double]$r[2]; $wu = [int]$r[3]
  "=== start $tag lr=$lr warmup=$wu"
  & $py $s @($r[1]) --lr $lr --min-lr ($lr/10) --warmup $wu --steps 600 --eval-every 100 --threads 2 --tag $tag *> (Join-Path $o "$tag.log")
  "=== done $tag exit=$LASTEXITCODE"
}
"=== 高 lr 粗扫结束 ==="
