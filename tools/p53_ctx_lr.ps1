# p53_ctx_lr.ps1 -- B 线：ctx 接口的 lr 阶梯 + 注意力密度补齐 + seed 复现（全部 1000 步/244 步，threads 1）
$ErrorActionPreference = "Stop"
$py = (Get-Command python).Source
$s = 'D:\czsai\vmlm\tools\p36_ctx.py'
$s36 = 'D:\czsai\vmlm\tools\p36_attnonce.py'
$o = 'D:\czsai\vmlm\artifacts\p36'

$base = @('--ctx-layers', '6', '--ctx-dim', '128', '--ctx-heads', '4', '--ctx-ffn', '1',
          '--ctx-out', 'perlayer', '--no-backbone-attn', '1', '--seq', '128', '--batch', '32')

# 1) 最优变体在 lr1e-3 下的 1000 步（与 dense d256@1e-3 = 5.0966、d224@1e-3 = 5.2047、
#    Quill-1@1e-3 = 5.2631 对齐）
foreach ($x in @(
  @{ t = 'ctx6_sa_lr1e3_1000'; a = @('--ctx-out-kind', 'mlp', '--ctx-mix', 'state') },
  @{ t = 'ctx6_lr1e3_1000';    a = @() }
)) {
  "=== start $($x.t) $(Get-Date -Format HH:mm:ss)"
  $al = @($s) + $base + $x.a + @('--lr', '1e-3', '--min-lr', '1e-4', '--warmup', '30',
          '--steps', '1000', '--eval-every', '200', '--threads', '1',
          '--ckpt-every', '100', '--resume', '1', '--tag', $x.t)
  & $py @al *> "$o\$($x.t).log"
  "=== done $($x.t) exit=$LASTEXITCODE $(Get-Date -Format HH:mm:ss)"
}

# 2) lr 上探 2e-3
"=== start ctx6_sa_lr2e3_1000"
$al = @($s) + $base + @('--ctx-out-kind', 'mlp', '--ctx-mix', 'state',
        '--lr', '2e-3', '--min-lr', '2e-4', '--warmup', '30', '--steps', '1000',
        '--eval-every', '200', '--threads', '1', '--ckpt-every', '100', '--resume', '1',
        '--tag', 'ctx6_sa_lr2e3_1000')
& $py @al *> "$o\ctx6_sa_lr2e3_1000.log"
"=== done exit=$LASTEXITCODE"

# 3) seed 2 复现（244 步，检查噪声地板）
"=== start ctx6_sa_s2_244"
$al = @($s) + $base + @('--ctx-out-kind', 'mlp', '--ctx-mix', 'state', '--seed', '2',
        '--lr', '3e-4', '--min-lr', '3e-5', '--warmup', '30', '--steps', '244',
        '--eval-every', '50', '--threads', '1', '--ckpt-every', '100', '--resume', '1',
        '--tag', 'ctx6_sa_s2_244')
& $py @al *> "$o\ctx6_sa_s2_244.log"
"=== done exit=$LASTEXITCODE"

# 4) 注意力密度补齐：Quill K=2
"=== start qk2_lr1e3_1000"
$al = @($s36, '--share-attn', '1', '--seq', '128', '--batch', '32', '--attn-every', '2',
        '--lr', '1e-3', '--min-lr', '1e-4', '--warmup', '30', '--steps', '1000',
        '--eval-every', '200', '--threads', '1', '--tag', 'qk2_lr1e3_1000')
& $py @al *> "$o\qk2_lr1e3_1000.log"
"=== done exit=$LASTEXITCODE"
"=== 完成 ==="
