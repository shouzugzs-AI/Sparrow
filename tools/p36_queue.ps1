# P36 queue: wait until all phase-1 (f_*) arms are finished, then run the
# attention-once / attn-every-K arms with the same protocol (244 steps, 4096 tok/step).
$py = 'python'
$s = 'D:\czsai\vmlm\tools\p36_attnonce.py'
$out = 'D:\czsai\vmlm\artifacts\p36'
$need = @('f_h20', 'f_h30', 'f_h40', 'f_h15m05', 'f_d128', 'f_d152', 'f_d176')

New-Item -ItemType Directory -Force -Path $out | Out-Null
while ($true) {
  $done = 0
  foreach ($t in $need) {
    $p = Join-Path 'D:\czsai\vmlm\artifacts\p35' "$t.jsonl"
    if ((Test-Path $p) -and (Select-String -Path $p -Pattern '"kind": "final"' -Quiet)) { $done++ }
  }
  if ($done -eq $need.Count) { break }
  Start-Sleep -Seconds 45
}
"=== phase-1 complete, starting p36 arms ==="

$runs = @(
  @('b_tra_s128',  @('--attn-every', '1',  '--share-attn', '0', '--seq', '128', '--batch', '32')),
  @('b_a12_s128',  @('--attn-every', '12', '--share-attn', '1', '--seq', '128', '--batch', '32')),
  @('b_a4_s128',   @('--attn-every', '4',  '--share-attn', '1', '--seq', '128', '--batch', '32')),
  @('b_tra_s512',  @('--attn-every', '1',  '--share-attn', '0', '--seq', '512', '--batch', '8')),
  @('b_a12_s512',  @('--attn-every', '12', '--share-attn', '1', '--seq', '512', '--batch', '8'))
)
foreach ($r in $runs) {
  $tag = $r[0]
  $rest = $r[1]
  "=== start $tag"
  & $py $s @rest --steps 244 --threads 2 --tag $tag *> (Join-Path $out "$tag.log")
  "=== done $tag exit=$LASTEXITCODE"
}
"=== p36 queue finished ==="
