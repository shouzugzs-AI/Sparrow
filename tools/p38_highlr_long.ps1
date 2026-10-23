# 高学习率 + 长训练：C1/C5 在 lr=1e-3 下各 4000 步（16.4M token）
$py = 'python'
$s = 'D:\czsai\vmlm\tools\p35_dyn_v2.py'
$o = 'D:\czsai\vmlm\artifacts\p35'
while ($true) {
  $p = Join-Path $o 'd64_t4000.jsonl'
  if ((Test-Path $p) -and (Select-String -Path $p -Pattern '"kind": "final"' -Quiet)) { break }
  Start-Sleep -Seconds 120
}
"=== 高 lr 长训开始 ==="
"=== start c1_lr1e3_t4000"
& $py $s --mask-src h --keep 0.01 --lr 1e-3 --min-lr 1e-4 --steps 4000 --eval-every 400 --threads 2 --tag c1_lr1e3_t4000 *> "$o\c1_lr1e3_t4000.log"
"=== done c1_lr1e3_t4000 exit=$LASTEXITCODE"
"=== start c5_lr1e3_t4000"
& $py $s --mask-src h --keep 0.05 --lr 1e-3 --min-lr 1e-4 --steps 4000 --eval-every 400 --threads 2 --tag c5_lr1e3_t4000 *> "$o\c5_lr1e3_t4000.log"
"=== done c5_lr1e3_t4000 exit=$LASTEXITCODE"
"=== 高 lr 长训结束 ==="
