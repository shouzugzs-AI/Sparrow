# p47b_quill_eqmac.ps1 -- Quill-1 的等 MAC 稠密对照：d=224（11.81M/token ≈ Quill-1 的 11.83M）
# 这也顺便把「Quill 赢是因为更省参数」和「赢是因为结构」分开：等 MAC 下若稠密仍不如，才算结构优势。
$ErrorActionPreference = "Stop"
$py = (Get-Command python).Source
$s = 'D:\czsai\vmlm\tools\p35_dyn_v2.py'
$o = 'D:\czsai\vmlm\artifacts\p35'

$arms = @(
  @{ t = 'd224_lr1e3_1000'; lr = '1e-3' },
  @{ t = 'd224_lr2e3_1000'; lr = '2e-3' }
)
foreach ($x in $arms) {
  "=== start $($x.t) $(Get-Date -Format HH:mm:ss)"
  $al = @($s, '--mask-src', 'none', '--d', '224', '--lr', $x.lr, '--min-lr', '1e-4',
          '--warmup', '30', '--steps', '1000', '--eval-every', '200', '--threads', '1',
          '--ckpt-every', '100', '--resume', '1', '--tag', $x.t)
  & $py @al *> "$o\$($x.t).log"
  "=== done $($x.t) exit=$LASTEXITCODE $(Get-Date -Format HH:mm:ss)"
}
"=== 完成 ==="
