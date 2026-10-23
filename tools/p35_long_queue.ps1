# 长程验证（修复窗口计数器 bug 后重跑）：1% / 5% / 15%，1000 步 = 4.1M token
$py = 'python'
$s = 'D:\czsai\vmlm\tools\p35_dyn_v2.py'
$o = 'D:\czsai\vmlm\artifacts\p35'
while ($true) {
  $p = Join-Path $o 't_v2b.jsonl'
  if ((Test-Path $p) -and (Select-String -Path $p -Pattern '"kind": "final"' -Quiet)) { break }
  Start-Sleep -Seconds 30
}
"=== 冒烟通过，开始长程三臂 ==="
foreach ($k in @('0.01', '0.05', '0.15')) {
  $tag = 'L_' + ($k -replace '0\.', 'h')
  "=== start $tag"
  & $py $s --mask-src h --keep $k --steps 1000 --eval-every 100 --threads 2 --tag $tag *> (Join-Path $o "$tag.log")
  "=== done $tag exit=$LASTEXITCODE"
}
"=== 长程三臂结束 ==="
