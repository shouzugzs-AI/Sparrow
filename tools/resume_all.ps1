# resume_all.ps1 -- DSH 更新/环境重启后的统一恢复入口
# 逻辑：逐臂检查 artifacts 里是否已有 "kind": "final" 记录；已完成就跳过，
#       未完成就以 --resume 1 从最近 checkpoint 续跑（p35/p36_ctx 都支持 ckpt）。
# 用法：  pwsh -NoProfile -File D:\czsai\vmlm\tools\resume_all.ps1
$ErrorActionPreference = "Stop"
$py = (Get-Command python).Source
$p35 = 'D:\czsai\vmlm\tools\p35_dyn_v2.py'
$p36 = 'D:\czsai\vmlm\tools\p36_ctx.py'
$p36a = 'D:\czsai\vmlm\tools\p36_attnonce.py'
$o35 = 'D:\czsai\vmlm\artifacts\p35'
$o36 = 'D:\czsai\vmlm\artifacts\p36'

function Is-Done([string]$jsonl) {
  if (-not (Test-Path $jsonl)) { return $false }
  return (Select-String -Path $jsonl -Pattern '"kind": "final"' -Quiet)
}

function Run35([string]$tag, [string[]]$extra, [string]$threads = '1') {
  if (Is-Done (Join-Path $o35 "$tag.jsonl")) { "== $tag 已完成，跳过"; return }
  "=== start $tag $(Get-Date -Format HH:mm:ss)"
  $al = @($p35) + $extra + @('--threads', $threads, '--ckpt-every', '100', '--resume', '1', '--tag', $tag)
  & $py @al *> (Join-Path $o35 "$tag.log")
  "=== done $tag exit=$LASTEXITCODE $(Get-Date -Format HH:mm:ss)"
}

function Run36([string]$tag, [string[]]$extra, [string]$threads = '1') {
  if (Is-Done (Join-Path $o36 "$tag.jsonl")) { "== $tag 已完成，跳过"; return }
  "=== start $tag $(Get-Date -Format HH:mm:ss)"
  $al = @($p36) + $extra + @('--threads', $threads, '--ckpt-every', '100', '--resume', '1', '--tag', $tag)
  & $py @al *> (Join-Path $o36 "$tag.log")
  "=== done $tag exit=$LASTEXITCODE $(Get-Date -Format HH:mm:ss)"
}

# ---------------- A 线：p36 上下文接口（threads 2，长任务优先） ----------------
Start-Process -FilePath pwsh -ArgumentList '-NoProfile', '-Command', @"
& {
  function Is-Done([string]`$j) { if (-not (Test-Path `$j)) { return `$false }; return (Select-String -Path `$j -Pattern '"kind": "final"' -Quiet) }
  `$py = '$py'; `$p36 = '$p36'; `$o36 = '$o36'
  `$arms = @(
    @{ t='ctx6_state_adapt_lr2e3_4000'; a=@('--ctx-layers','6','--ctx-dim','128','--ctx-heads','4','--ctx-ffn','1','--ctx-out','perlayer','--ctx-out-kind','mlp','--ctx-mix','state','--no-backbone-attn','1','--seq','128','--batch','32','--lr','2e-3','--min-lr','2e-4','--warmup','60','--steps','4000','--eval-every','400') }
  )
  foreach (`$x in `$arms) {
    if (Is-Done (Join-Path `$o36 (`$x.t + '.jsonl'))) { '== ' + `$x.t + ' 已跳过'; continue }
    '=== start ' + `$x.t + ' ' + (Get-Date -Format HH:mm:ss)
    `$al = @(`$p36) + `$x.a + @('--threads','2','--ckpt-every','100','--resume','1','--tag',`$x.t)
    & `$py @al *> (Join-Path `$o36 (`$x.t + '.log'))
  }
}
"@ -WindowStyle Hidden

# ---------------- B 线：p36 其余（threads 1） ----------------
Start-Process -FilePath pwsh -ArgumentList '-NoProfile', '-Command', @"
& {
  function Is-Done([string]`$j) { if (-not (Test-Path `$j)) { return `$false }; return (Select-String -Path `$j -Pattern '"kind": "final"' -Quiet) }
  `$py = '$py'; `$p36 = '$p36'; `$p36a = '$p36a'; `$o36 = '$o36'
  `$base = @('--ctx-layers','6','--ctx-dim','128','--ctx-heads','4','--ctx-ffn','1','--ctx-out','perlayer','--no-backbone-attn','1','--seq','128','--batch','32')
  `$arms = @(
    @{ t='ctx6_sa_lr2e3_1000'; a=(`$base + @('--ctx-out-kind','mlp','--ctx-mix','state','--lr','2e-3','--min-lr','2e-4','--warmup','30','--steps','1000','--eval-every','200')) },
    @{ t='ctx6_sa_s2_244';     a=(`$base + @('--ctx-out-kind','mlp','--ctx-mix','state','--seed','2','--lr','3e-4','--min-lr','3e-5','--warmup','30','--steps','244','--eval-every','50')) }
  )
  foreach (`$x in `$arms) {
    if (Is-Done (Join-Path `$o36 (`$x.t + '.jsonl'))) { '== ' + `$x.t + ' 已跳过'; continue }
    '=== start ' + `$x.t + ' ' + (Get-Date -Format HH:mm:ss)
    `$al = @(`$p36) + `$x.a + @('--threads','1','--ckpt-every','100','--resume','1','--tag',`$x.t)
    & `$py @al *> (Join-Path `$o36 (`$x.t + '.log'))
  }
  # qk2（p36_attnonce 没有 ckpt，只能从头）
  if (-not (Is-Done (Join-Path `$o36 'qk2_lr1e3_1000.jsonl'))) {
    '=== start qk2_lr1e3_1000 ' + (Get-Date -Format HH:mm:ss)
    & `$py $p36a --share-attn 1 --seq 128 --batch 32 --attn-every 2 --lr 1e-3 --min-lr 1e-4 --warmup 30 --steps 1000 --eval-every 200 --threads 1 --tag qk2_lr1e3_1000 *> (Join-Path `$o36 'qk2_lr1e3_1000.log')
  }
}
"@ -WindowStyle Hidden

# ---------------- C 线：p35（threads 1） ----------------
Start-Process -FilePath pwsh -ArgumentList '-NoProfile', '-Command', @"
& {
  function Is-Done([string]`$j) { if (-not (Test-Path `$j)) { return `$false }; return (Select-String -Path `$j -Pattern '"kind": "final"' -Quiet) }
  `$py = '$py'; `$p35 = '$p35'; `$o35 = '$o35'
  `$arms = @(
    @{ t='d192_lr1e3_2000';   a=@('--mask-src','none','--d','192','--lr','1e-3','--min-lr','1e-4','--warmup','30','--steps','2000','--eval-every','200') },
    @{ t='mb_h01_244';        a=@('--mask-src','h','--keep','0.01','--lr','3e-4','--min-lr','3e-5','--warmup','30','--steps','244','--eval-every','50','--layers','12','--mask-balance','0.2') },
    @{ t='mb_h05_600';        a=@('--mask-src','h','--keep','0.05','--lr','5e-3','--min-lr','5e-4','--warmup','30','--steps','600','--eval-every','100','--layers','12','--mask-balance','0.2') },
    @{ t='mb_L24_k25_244';    a=@('--mask-src','h','--keep','0.025','--lr','3e-4','--min-lr','3e-5','--warmup','30','--steps','244','--eval-every','50','--layers','24','--mask-balance','0.2') }
  )
  foreach (`$x in `$arms) {
    if (Is-Done (Join-Path `$o35 (`$x.t + '.jsonl'))) { '== ' + `$x.t + ' 已跳过'; continue }
    '=== start ' + `$x.t + ' ' + (Get-Date -Format HH:mm:ss)
    `$al = @(`$p35) + `$x.a + @('--threads','1','--ckpt-every','100','--resume','1','--tag',`$x.t)
    & `$py @al *> (Join-Path `$o35 (`$x.t + '.log'))
  }
}
"@ -WindowStyle Hidden

Start-Sleep -Seconds 8
"三条恢复流已启动。监控：http://127.0.0.1:8791/ （若未起，运行 python D:\czsai\vmlm\tools\watch_server.py）"
