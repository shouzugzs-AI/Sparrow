# p49_ctx_interface.ps1 -- 上下文接口（前置 N 层窄注意力 + 主干纯 FFN + 每层 c->d 读出）
# 对照：Quill-1 @244 步 = 6.3310（1 次全宽 attn），稠密 d=256 @244 步 = 6.4897（12 层 attn）
# 全部 244 步 / lr 3e-4 / 同 p35 协议，可直接与 e_*/b_* 表比
$ErrorActionPreference = "Stop"
$py = (Get-Command python).Source
$s = 'D:\czsai\vmlm\tools\p36_ctx.py'
$o = 'D:\czsai\vmlm\artifacts\p36'

$arms = @(
  @{ t = 'ctx6_c128';        a = @('--ctx-layers', '6', '--ctx-dim', '128', '--ctx-ffn', '1', '--ctx-out', 'perlayer') },
  @{ t = 'ctx6_c128_shared'; a = @('--ctx-layers', '6', '--ctx-dim', '128', '--ctx-ffn', '1', '--ctx-out', 'shared') },
  @{ t = 'ctx6_c128_noffn';  a = @('--ctx-layers', '6', '--ctx-dim', '128', '--ctx-ffn', '0', '--ctx-out', 'perlayer') }
)
foreach ($x in $arms) {
  "=== start $($x.t) $(Get-Date -Format HH:mm:ss)"
  $al = @($s) + $x.a + @('--no-backbone-attn', '1', '--seq', '128', '--batch', '32',
          '--lr', '3e-4', '--min-lr', '3e-5', '--warmup', '30', '--steps', '244',
          '--eval-every', '50', '--threads', '1', '--tag', $x.t)
  & $py @al *> "$o\$($x.t).log"
  "=== done $($x.t) exit=$LASTEXITCODE $(Get-Date -Format HH:mm:ss)"
}
"=== 完成 ==="
