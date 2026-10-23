# P35 phase-2: find the keep optimum (40% already wins) and the mid-5% combo.
# Starts after the L12 reference stream of p36 finishes.
$py = 'python'
$s = 'D:\czsai\vmlm\tools\p35_snap_p1.py'
$out = 'D:\czsai\vmlm\artifacts\p35'
$need = @('b_a12_s128', 'b_tra_s128', 'b_a12_s512', 'b_tra_s512')
while ($true) {
  $done = 0
  foreach ($t in $need) {
    $p = Join-Path 'D:\czsai\vmlm\artifacts\p36' "$t.jsonl"
    if ((Test-Path $p) -and (Select-String -Path $p -Pattern '"kind": "final"' -Quiet)) { $done++ }
  }
  if ($done -eq $need.Count) { break }
  Start-Sleep -Seconds 45
}
"=== p35 phase-2 start ==="
$runs = @(
  @('f_h50',    @('--mask-src','h','--keep','0.50')),
  @('f_h60',    @('--mask-src','h','--keep','0.60')),
  @('f_h70',    @('--mask-src','h','--keep','0.70')),
  @('f_h40m05', @('--mask-src','h','--keep','0.40','--keep-mid','0.05'))
)
foreach ($r in $runs) {
  $tag = $r[0]
  "=== start $tag"
  & $py $s @($r[1]) --steps 244 --threads 2 --tag $tag *> (Join-Path $out "$tag.log")
  "=== done $tag exit=$LASTEXITCODE"
}
"=== p35 phase-2 finished ==="
