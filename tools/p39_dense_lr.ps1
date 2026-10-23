# 稠密模型也要扫 lr：只有各自用最优 lr，"稀疏 vs 稠密"才是公平比较
# 600 步，与 C1/C5 粗扫同口径
$py = 'python'
$s = 'D:\czsai\vmlm\tools\p35_dyn_v2.py'
$o = 'D:\czsai\vmlm\artifacts\p35'
$runs = @(
  @('d64_lr2e3_600',  @('--mask-src','none','--d','64'),  '0.002', '60'),
  @('d64_lr5e3_600',  @('--mask-src','none','--d','64'),  '0.005', '120'),
  @('d112_lr1e3_600', @('--mask-src','none','--d','112'), '0.001', '60'),
  @('d112_lr5e3_600', @('--mask-src','none','--d','112'), '0.005', '120'),
  @('d256_lr5e3_600', @('--mask-src','none','--d','256'), '0.005', '120')
)
foreach ($r in $runs) {
  $tag = $r[0]; $lr = [double]$r[2]; $wu = [int]$r[3]
  "=== start $tag lr=$lr"
  & $py $s @($r[1]) --lr $lr --min-lr ($lr/10) --warmup $wu --steps 600 --eval-every 200 --threads 2 --tag $tag *> (Join-Path $o "$tag.log")
  "=== done $tag exit=$LASTEXITCODE"
}
"=== 稠密 lr 扫描结束 ==="
