# LR 扫描（第一次粗扫）：在 C15（当前最好）与 C5、C1 上换学习率，找峰值
# 基线均为 lr=3e-4、1000 步；已有 lr=1e-3 的三臂。min-lr 取 lr/10。
$py = 'python'
$s = 'D:\czsai\vmlm\tools\p35_dyn_v2.py'
$o = 'D:\czsai\vmlm\artifacts\p35'
while ($true) {
  $p = Join-Path $o 'd256_lr1e3.jsonl'
  if ((Test-Path $p) -and (Select-String -Path $p -Pattern '"kind": "final"' -Quiet)) { break }
  Start-Sleep -Seconds 120
}
"=== LR 扫描开始 ==="
$runs = @(
  @('c15_lr5e4', @('--mask-src','h','--keep','0.15'), '0.0005'),
  @('c15_lr2e3', @('--mask-src','h','--keep','0.15'), '0.002'),
  @('c15_lr3e3', @('--mask-src','h','--keep','0.15'), '0.003'),
  @('c5_lr2e3',  @('--mask-src','h','--keep','0.05'), '0.002'),
  @('c1_lr2e3',  @('--mask-src','h','--keep','0.01'), '0.002'),
  @('c1_lr3e3',  @('--mask-src','h','--keep','0.01'), '0.003')
)
foreach ($r in $runs) {
  $tag = $r[0]; $lr = [double]$r[2]
  $minlr = $lr / 10
  "=== start $tag lr=$lr"
  & $py $s @($r[1]) --lr $lr --min-lr $minlr --steps 1000 --eval-every 100 --threads 2 --tag $tag *> (Join-Path $o "$tag.log")
  "=== done $tag exit=$LASTEXITCODE"
}
"=== LR 扫描结束 ==="
