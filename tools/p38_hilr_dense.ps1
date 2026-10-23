# 高 lr 下的等 FLOP 稠密对照：d=80 对应 C5、d=112 对应 C15，都用 lr=1e-3
$py = 'python'
$s = 'D:\czsai\vmlm\tools\p35_dyn_v2.py'
$o = 'D:\czsai\vmlm\artifacts\p35'
while ($true) {
  $p = Join-Path $o 'c1_lr3e3.jsonl'
  if ((Test-Path $p) -and (Select-String -Path $p -Pattern '"kind": "final"' -Quiet)) { break }
  Start-Sleep -Seconds 180
}
"=== 高 lr 等 FLOP 稠密对照开始 ==="
foreach ($dd in @(80, 112)) {
  "=== start d$dd-lr1e3"
  & $py $s --mask-src none --d $dd --lr 1e-3 --min-lr 1e-4 --steps 1000 --eval-every 200 --threads 2 --tag "d$dd-lr1e3" *> "$o\d$dd-lr1e3.log"
  "=== done d$dd exit=$LASTEXITCODE"
}
"=== 对照结束 ==="
