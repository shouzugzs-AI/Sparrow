# p51_ctx_variants.ps1 -- 上下文接口的三种变体（用户 2026-10-23 提案 1/2/3）
#   1) 每层一份非线性适配器对接（one shared attention does the token mixing）
#   2) 用状态向量代替外部上下文的注意力（StateMix，成本与 T 无关、推理常数内存）
#   3) 适配器权重全局共享
# 对照：ctx6_c128（线性读出，无回读）/ Quill-1 6.3310 / 稠密 d=256 6.4897（均 244 步 lr3e-4）
$ErrorActionPreference = "Stop"
$py = (Get-Command python).Source
$s = 'D:\czsai\vmlm\tools\p36_ctx.py'
$o = 'D:\czsai\vmlm\artifacts\p36'

$arms = @(
  @{ t = 'ctx6_adapt';        a = @('--ctx-out-kind', 'mlp') },
  @{ t = 'ctx6_adapt_shared'; a = @('--ctx-out', 'shared', '--ctx-out-kind', 'mlp') },
  @{ t = 'ctx6_state';        a = @('--ctx-mix', 'state') },
  @{ t = 'ctx6_state_adapt';  a = @('--ctx-mix', 'state', '--ctx-out-kind', 'mlp') }
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
