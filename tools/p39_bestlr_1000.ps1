# 用粗扫找到的最优 lr 重跑 1000 步，确认 600 步的排序在长一点的训练里仍成立
$py = 'python'
$s = 'D:\czsai\vmlm\tools\p35_dyn_v2.py'
$o = 'D:\czsai\vmlm\artifacts\p35'
while ($true) {
  $p = Join-Path $o 'c5_lr5e3_600.jsonl'
  if ((Test-Path $p) -and (Select-String -Path $p -Pattern '"kind": "final"' -Quiet)) { break }
  Start-Sleep -Seconds 180
}
"=== 最优 lr 复核（1000 步）开始 ==="
$runs = @(
  @('c1_lr3e3_1000',  @('--mask-src','h','--keep','0.01'), '0.003', '80'),
  @('c5_lr3e3_1000',  @('--mask-src','h','--keep','0.05'), '0.003', '80'),
  @('c15_lr2e3_1000', @('--mask-src','h','--keep','0.15'), '0.002', '60')
)
foreach ($r in $runs) {
  $tag = $r[0]; $lr = [double]$r[2]; $wu = [int]$r[3]
  "=== start $tag lr=$lr"
  & $py $s @($r[1]) --lr $lr --min-lr ($lr/10) --warmup $wu --steps 1000 --eval-every 100 --threads 2 --tag $tag *> (Join-Path $o "$tag.log")
  "=== done $tag exit=$LASTEXITCODE"
}
"=== 复核结束 ==="
