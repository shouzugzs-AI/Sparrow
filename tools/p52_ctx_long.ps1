# p52_ctx_long.ps1 -- A 线：ctx 接口的长程（4000 步）决定性对照
# 对照：稠密 d=256 @4000 步 lr1e-3 = 4.1548（15.075M MAC）
#       C15 @4000 步 lr2e-3 = 4.3196（3.264M MAC）
$ErrorActionPreference = "Stop"
$py = (Get-Command python).Source
$s = 'D:\czsai\vmlm\tools\p36_ctx.py'
$o = 'D:\czsai\vmlm\artifacts\p36'

"=== start ctx6_state_adapt_lr1e3_4000 $(Get-Date -Format HH:mm:ss)"
$al = @($s, '--ctx-layers', '6', '--ctx-dim', '128', '--ctx-heads', '4', '--ctx-ffn', '1',
        '--ctx-out', 'perlayer', '--ctx-out-kind', 'mlp', '--ctx-mix', 'state',
        '--no-backbone-attn', '1', '--seq', '128', '--batch', '32',
        '--lr', '1e-3', '--min-lr', '1e-4', '--warmup', '60', '--steps', '4000',
        '--eval-every', '400', '--threads', '2', '--ckpt-every', '100', '--resume', '1',
        '--tag', 'ctx6_state_adapt_lr1e3_4000')
& $py @al *> "$o\ctx6_state_adapt_lr1e3_4000.log"
"=== done exit=$LASTEXITCODE $(Get-Date -Format HH:mm:ss)"
