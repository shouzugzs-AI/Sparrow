# p48_recover_c.ps1 -- 崩溃恢复 C 线（threads 1，串行）：
#   逐通道调度 A/B -> keep×层数 -> 覆盖度重测 -> Quill lr/注意力密度
$ErrorActionPreference = "Stop"
$py = (Get-Command python).Source
$p35 = 'D:\czsai\vmlm\tools\p35_dyn_v2.py'
$p36 = 'D:\czsai\vmlm\tools\p36_attnonce.py'
$o35 = 'D:\czsai\vmlm\artifacts\p35'
$o36 = 'D:\czsai\vmlm\artifacts\p36'

function RunP35($tag, $extra) {
  "=== start $tag $(Get-Date -Format HH:mm:ss)"
  $al = @($p35) + $extra + @('--threads', '1', '--ckpt-every', '100', '--resume', '1', '--tag', $tag)
  & $py @al *> "$o35\$tag.log"
  "=== done $tag exit=$LASTEXITCODE $(Get-Date -Format HH:mm:ss)"
}

# 1) 逐通道调度（续跑），对照 c5_lr5e3_600 = 5.4708 / c1_lr5e3_600 = 6.0014
RunP35 'c5_unit_lr5e3_600' @('--mask-src', 'h', '--keep', '0.05', '--lr', '5e-3', '--min-lr', '5e-4',
                             '--warmup', '30', '--steps', '600', '--eval-every', '100', '--sched', 'unit')
RunP35 'c1_unit_lr5e3_600' @('--mask-src', 'h', '--keep', '0.01', '--lr', '5e-3', '--min-lr', '5e-4',
                             '--warmup', '30', '--steps', '600', '--eval-every', '100', '--sched', 'unit')

# 2) keep × 层数（覆盖度对齐）
RunP35 'Lk_L6_k10' @('--mask-src', 'h', '--layers', '6', '--d', '256', '--keep', '0.10',
                     '--lr', '3e-4', '--min-lr', '3e-5', '--warmup', '30', '--steps', '244', '--eval-every', '50')
RunP35 'Lk_L24_k25' @('--mask-src', 'h', '--layers', '24', '--d', '256', '--keep', '0.025',
                      '--lr', '3e-4', '--min-lr', '3e-5', '--warmup', '30', '--steps', '244', '--eval-every', '50')
RunP35 'Lk_L24_d120_k25' @('--mask-src', 'h', '--layers', '24', '--d', '120', '--heads', '2', '--hidden', '480',
                           '--keep', '0.025', '--lr', '3e-4', '--min-lr', '3e-5', '--warmup', '30',
                           '--steps', '244', '--eval-every', '50')

# 3) 覆盖度重测（1% / 2%）
RunP35 'm_h01' @('--mask-src', 'h', '--keep', '0.01', '--lr', '3e-4', '--min-lr', '3e-5',
                 '--warmup', '30', '--steps', '244', '--eval-every', '50')
RunP35 'm_h02' @('--mask-src', 'h', '--keep', '0.02', '--lr', '3e-4', '--min-lr', '3e-5',
                 '--warmup', '30', '--steps', '244', '--eval-every', '50')

# 4) Quill-1 的等 MAC 稠密对照 d=224
RunP35 'd224_lr1e3_1000' @('--mask-src', 'none', '--d', '224', '--lr', '1e-3', '--min-lr', '1e-4',
                           '--warmup', '30', '--steps', '1000', '--eval-every', '200')
RunP35 'd224_lr2e3_1000' @('--mask-src', 'none', '--d', '224', '--lr', '2e-3', '--min-lr', '2e-4',
                           '--warmup', '30', '--steps', '1000', '--eval-every', '200')

# 5) Quill（p36 没有 checkpoint，放最后）
foreach ($x in @(
  @{ t = 'q1_lr1e3_1000'; a = @('--attn-every', '12') },
  @{ t = 'qk3_lr1e3_1000'; a = @('--attn-every', '3') },
  @{ t = 'qk2_lr1e3_1000'; a = @('--attn-every', '2') }
)) {
  "=== start $($x.t) $(Get-Date -Format HH:mm:ss)"
  $al = @($p36, '--share-attn', '1', '--seq', '128', '--batch', '32') + $x.a + @(
          '--lr', '1e-3', '--min-lr', '1e-4', '--warmup', '30', '--steps', '1000',
          '--eval-every', '200', '--threads', '1', '--tag', $x.t)
  & $py @al *> "$o36\$($x.t).log"
  "=== done $($x.t) exit=$LASTEXITCODE $(Get-Date -Format HH:mm:ss)"
}

# 6) 细粒度复现（看 400→800 的收缩是台阶还是斜坡）
RunP35 'hi_c15_lr3e3' @('--mask-src', 'h', '--keep', '0.15', '--lr', '3e-3', '--min-lr', '3e-4',
                        '--warmup', '60', '--steps', '4000', '--stop-at', '950', '--eval-every', '25')
"=== 完成 ==="
