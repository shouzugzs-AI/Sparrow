# 遗忘实验的修正版：把预训练质量对齐后再微调（否则稀疏臂"忘得少"可能只是"学得少"）
# 做法：C15 从已有 400 步 ckpt 继续在 wiki 上训 300 步，使预训练 wiki 损失接近 dense 的 6.14，
# 然后再用同样设置微调对话 150 步，比较遗忘量。
$py = 'python'
$s = 'D:\czsai\vmlm\tools\p35_fgt.py'
$o = 'D:\czsai\vmlm\artifacts\p37'
$w = 'D:\czsai\vmlm\artifacts\zh_wiki'
$fox = 'D:\czsai\vmlm\artifacts\fox_sft'
"=== C15 预训练续训 300 步"
& $py $s --mask-src h --keep 0.15 --data-dir $w --init "$o\c15_pre.pt" --steps 300 --eval-every 150 --threads 4 `
    --tag "pre_c15b" --save "$o\c15b_pre.pt" *> "$o\pre_c15b.log"
"=== 预训练续训 exit=$LASTEXITCODE"
"=== C15 微调 150 步"
& $py $s --mask-src h --keep 0.15 --data-dir $fox --eval-dir $w --init "$o\c15b_pre.pt" `
    --lr 1e-4 --min-lr 1e-5 --steps 150 --eval-every 50 --threads 4 `
    --tag "ft_c15b" --save "$o\c15b_ft.pt" *> "$o\ft_c15b.log"
"=== 微调 exit=$LASTEXITCODE"
"=== 修正版遗忘实验结束 ==="
