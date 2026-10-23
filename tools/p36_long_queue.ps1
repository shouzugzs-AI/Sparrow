# 长程复核"一个 attn + MLP 栈"相对 Transformer 的优势（1000 步 = 4.1M token）
$py = 'python'
$s = 'D:\czsai\vmlm\tools\p36_attnonce.py'
$o = 'D:\czsai\vmlm\artifacts\p36'
while ($true) {
  $p = 'D:\czsai\vmlm\artifacts\p35\L_h15.jsonl'
  if ((Test-Path $p) -and (Select-String -Path $p -Pattern '"kind": "final"' -Quiet)) { break }
  Start-Sleep -Seconds 60
}
"=== 开始 p36 长程复核 ==="
$runs = @(
  @('y_a12_1000', @('--attn-every','12','--share-attn','1','--seq','128','--batch','32')),
  @('y_tra_1000', @('--attn-every','1','--share-attn','0','--seq','128','--batch','32'))
)
foreach ($r in $runs) {
  $tag = $r[0]
  "=== start $tag"
  & $py $s @($r[1]) --steps 1000 --eval-every 100 --threads 2 --tag $tag *> (Join-Path $o "$tag.log")
  "=== done $tag exit=$LASTEXITCODE"
}
"=== p36 长程复核结束 ==="
