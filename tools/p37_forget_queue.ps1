# 灾难性遗忘实验：wiki 预训练 -> fox 对话微调 -> 测 wiki 保留（遗忘量）
# 对照组：dense / 稀疏 C1(keep=5%) / 稀疏 C2(keep=15%)
$py = 'python'
$s = 'D:\czsai\vmlm\tools\p35_fgt.py'
$o = 'D:\czsai\vmlm\artifacts\p37'
$w = 'D:\czsai\vmlm\artifacts\zh_wiki'
$fox = 'D:\czsai\vmlm\artifacts\fox_sft'
New-Item -ItemType Directory -Force -Path $o | Out-Null

# 等前面三个队列收尾，避免 6 线程抢占
$need = @(
  @('D:\czsai\vmlm\artifacts\p35\L_h15.jsonl', ''),
  @('D:\czsai\vmlm\artifacts\p35\x_mix15.jsonl', ''),
  @('D:\czsai\vmlm\artifacts\p36\y_tra_1000.jsonl', '')
)
while ($true) {
  $done = 0
  foreach ($n in $need) {
    if ((Test-Path $n[0]) -and (Select-String -Path $n[0] -Pattern '"kind": "final"' -Quiet)) { $done++ }
  }
  if ($done -eq $need.Count) { break }
  Start-Sleep -Seconds 90
}
"=== 遗忘实验开始 ==="

$arms = @(
  @('dense', @('--mask-src','none')),
  @('c05',   @('--mask-src','h','--keep','0.05')),
  @('c15',   @('--mask-src','h','--keep','0.15'))
)
foreach ($a in $arms) {
  $name = $a[0]; $cfg = $a[1]
  # 1) 预训练：wiki 400 步，存 ckpt
  "=== 预训练 $name"
  & $py $s @cfg --data-dir $w --steps 400 --eval-every 200 --threads 2 `
      --tag "pre_$name" --save "$o\${name}_pre.pt" *> "$o\pre_$name.log"
  "=== 预训练 $name exit=$LASTEXITCODE"
  # 2) 微调：fox 150 步，同时评测 wiki 保留
  "=== 微调 $name"
  & $py $s @cfg --data-dir $fox --eval-dir $w --init "$o\${name}_pre.pt" `
      --lr 1e-4 --min-lr 1e-5 --steps 150 --eval-every 50 --threads 2 `
      --tag "ft_$name" --save "$o\${name}_ft.pt" *> "$o\ft_$name.log"
  "=== 微调 $name exit=$LASTEXITCODE"
}
"=== 遗忘实验结束 ==="
