# p50_ctx_readback.ps1 -- 上下文接口 + 主干回读（read-back）：让 ctx 看得到主干演化中的状态
# 对照：ctx6_c128（无回读）= 正在跑；Quill-1 = 6.3310；稠密 d=256 = 6.4897（均 244 步 / lr 3e-4）
$ErrorActionPreference = "Stop"
$py = (Get-Command python).Source
$s = 'D:\czsai\vmlm\tools\p36_ctx.py'
$o = 'D:\czsai\vmlm\artifacts\p36'

$arms = @(
  @{ t = 'ctx6_rb3'; a = @('--readback-every', '3') },
  @{ t = 'ctx6_rb1'; a = @('--readback-every', '1') }
)
foreach ($x in $arms) {
  "=== start $($x.t) $(Get-Date -Format HH:mm:ss)"
  $al = @($s, '--ctx-layers', '6', '--ctx-dim', '128', '--ctx-heads', '4',
          '--ctx-ffn', '1', '--ctx-out', 'perlayer', '--no-backbone-attn', '1') + $x.a + @(
          '--seq', '128', '--batch', '32', '--lr', '3e-4', '--min-lr', '3e-5',
          '--warmup', '30', '--steps', '244', '--eval-every', '50',
          '--threads', '1', '--tag', $x.t)
  & $py @al *> "$o\$($x.t).log"
  "=== done $($x.t) exit=$LASTEXITCODE $(Get-Date -Format HH:mm:ss)"
}
"=== 完成 ==="
