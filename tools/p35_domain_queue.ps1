# 跨领域通道分配检验：wiki(领域A) + fox_sft对话(领域B) 混合训练，
# 然后在两个领域上分别测掩码分配，看"死通道"是不是被另一个领域用掉了。
$py = 'python'
$s = 'D:\czsai\vmlm\tools\p35_dyn_v3.py'
$o = 'D:\czsai\vmlm\artifacts\p35'
$d2 = 'D:\czsai\vmlm\artifacts\fox_sft'
while ($true) {
  $p = Join-Path $o 'f_h40m05.jsonl'
  if ((Test-Path $p) -and (Select-String -Path $p -Pattern '"kind": "final"' -Quiet)) { break }
  Start-Sleep -Seconds 60
}
"=== 开始跨领域实验 ==="
foreach ($k in @('0.05', '0.01', '0.15')) {
  $tag = 'x_mix' + ($k -replace '0\.', '')
  "=== start $tag"
  & $py $s --mask-src h --keep $k --data2 $d2 --steps 700 --eval-every 100 --threads 2 --tag $tag *> (Join-Path $o "$tag.log")
  "=== done $tag exit=$LASTEXITCODE"
}
"=== 跨领域实验结束 ==="
