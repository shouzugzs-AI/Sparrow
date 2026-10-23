# P36-L24 queue: the user's architecture -- one N*N attention at the input, then
# 24 fully-connected (per-token) layers, each of which can RE-READ the shared
# N*N routing matrix.  Waits for the L12 reference stream, then runs 4 arms.
$py = 'python'
$s = 'D:\czsai\vmlm\tools\p36_attnonce.py'
$out = 'D:\czsai\vmlm\artifacts\p36'
$need = @('b_a12_s128', 'b_tra_s128', 'b_a12_s512', 'b_tra_s512')

while ($true) {
  $done = 0
  foreach ($t in $need) {
    $p = Join-Path $out "$t.jsonl"
    if ((Test-Path $p) -and (Select-String -Path $p -Pattern '"kind": "final"' -Quiet)) { $done++ }
  }
  if ($done -eq $need.Count) { break }
  Start-Sleep -Seconds 45
}
"=== L12 reference done, starting L24 arms ==="

$runs = @(
  @('c_L24_once',  @('--layers','24','--attn-every','24','--share-attn','1','--route','none','--seq','128','--batch','32')),
  @('c_L24_route', @('--layers','24','--attn-every','24','--share-attn','1','--route','full','--seq','128','--batch','32')),
  @('c_L24_tra',   @('--layers','24','--attn-every','1','--share-attn','0','--route','none','--seq','128','--batch','32')),
  @('c_L24_route_s512', @('--layers','24','--attn-every','24','--share-attn','1','--route','full','--seq','512','--batch','8'))
)
foreach ($r in $runs) {
  $tag = $r[0]
  $rest = $r[1]
  "=== start $tag"
  & $py $s @rest --steps 244 --threads 2 --tag $tag *> (Join-Path $out "$tag.log")
  "=== done $tag exit=$LASTEXITCODE"
}
"=== L24 queue finished ==="
