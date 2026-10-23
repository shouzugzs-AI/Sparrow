# 补测"微调前"的语言/对话起点损失：用 1 步、lr≈0 的 run 只做评测，
# 这样可以画出 (Δ对话, Δwiki) 的权衡曲线，而不是只看"遗忘量"这个单边指标。
$py = 'python'
$s = 'D:\czsai\vmlm\tools\p35_fgt.py'
$o = 'D:\czsai\vmlm\artifacts\p37'
$w = 'D:\czsai\vmlm\artifacts\zh_wiki'
$fox = 'D:\czsai\vmlm\artifacts\fox_sft'
while ($true) {
  $p = 'D:\czsai\vmlm\artifacts\p35\ft_c15b.jsonl'
  if ((Test-Path $p) -and (Select-String -Path $p -Pattern '"kind": "final"' -Quiet)) { break }
  Start-Sleep -Seconds 60
}
"=== 补测微调前起点 ==="
& $py $s --mask-src none --data-dir $fox --eval-dir $w --init "$o\dense_pre.pt" `
    --lr 1e-9 --min-lr 1e-9 --steps 1 --eval-every 1 --threads 4 --tag "t0_dense" *> "$o\t0_dense.log"
"=== t0 dense exit=$LASTEXITCODE"
& $py $s --mask-src h --keep 0.05 --data-dir $fox --eval-dir $w --init "$o\c05_pre.pt" `
    --lr 1e-9 --min-lr 1e-9 --steps 1 --eval-every 1 --threads 4 --tag "t0_c05" *> "$o\t0_c05.log"
"=== t0 c05 exit=$LASTEXITCODE"
& $py $s --mask-src h --keep 0.15 --data-dir $fox --eval-dir $w --init "$o\c15_pre.pt" `
    --lr 1e-9 --min-lr 1e-9 --steps 1 --eval-every 1 --threads 4 --tag "t0_c15" *> "$o\t0_c15.log"
"=== t0 c15 exit=$LASTEXITCODE"
"=== t0 补测结束 ==="
