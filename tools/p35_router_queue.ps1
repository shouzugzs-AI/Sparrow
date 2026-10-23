# Phase-2 queue: wait for all phase-1 arms to finish, then run the learnable-router
# and overlap-metric control arms with the same protocol (244 steps, 2 threads).
$py = 'python'
$s = 'D:\czsai\vmlm\tools\p35_dyn.py'
$out = 'D:\czsai\vmlm\artifacts\p35'
$need = @('f_h20', 'f_h30', 'f_h40', 'f_h15m05', 'f_d128', 'f_d152', 'f_d176')

while ($true) {
  $done = 0
  foreach ($t in $need) {
    $p = Join-Path $out "$t.jsonl"
    if ((Test-Path $p) -and (Select-String -Path $p -Pattern '"kind": "final"' -Quiet)) { $done++ }
  }
  if ($done -eq $need.Count) { break }
  Start-Sleep -Seconds 60
}
"=== phase-1 complete, starting router arms ==="

$runs = @(
  @('g_r15',    '--mask-src', 'r', '--keep', '0.15', '--aux-coef', '0.01'),
  @('g_r15_lb0', '--mask-src', 'r', '--keep', '0.15', '--aux-coef', '0'),
  @('g_h15oc',  '--mask-src', 'h', '--keep', '0.15', '--aux-coef', '0.01')
)
foreach ($r in $runs) {
  $tag = $r[0]
  $rest = $r[1..($r.Count - 1)]
  "=== start $tag"
  & $py $s @rest --steps 244 --threads 2 --tag $tag *> (Join-Path $out "$tag.log")
  "=== done $tag exit=$LASTEXITCODE"
}
"=== router queue finished ==="
