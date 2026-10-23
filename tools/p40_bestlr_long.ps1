# 用最优 lr 做长训：C1/C5 在 lr=3e-3 下各 4000 步（16.4M token）
# 等 C1 的旧长训（lr1e-3）收尾后开始，避免两个 4000 步臂同时抢 CPU 太久
$py = 'python'
$s = 'D:\czsai\vmlm\tools\p35_dyn_v2.py'
$o = 'D:\czsai\vmlm\artifacts\p35'
while ($true) {
  $p = Join-Path $o 'c1_lr1e3_t4000.jsonl'
  if ((Test-Path $p) -and (Select-String -Path $p -Pattern '"kind": "final"' -Quiet)) { break }
  Start-Sleep -Seconds 180
}
"=== 最优 lr 长训开始 ==="
"=== start c1_lr3e3_4000"
& $py $s --mask-src h --keep 0.01 --lr 0.003 --min-lr 3e-4 --warmup 80 --steps 4000 --eval-every 400 --threads 2 --tag c1_lr3e3_4000 *> "$o\c1_lr3e3_4000.log"
"=== done c1_lr3e3_4000 exit=$LASTEXITCODE"
"=== start c5_lr3e3_4000"
& $py $s --mask-src h --keep 0.05 --lr 0.003 --min-lr 3e-4 --warmup 80 --steps 4000 --eval-every 400 --threads 2 --tag c5_lr3e3_4000 *> "$o\c5_lr3e3_4000.log"
"=== done c5_lr3e3_4000 exit=$LASTEXITCODE"
"=== 最优 lr 长训结束 ==="
