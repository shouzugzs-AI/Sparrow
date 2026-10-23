# 超长训练：1% 稀疏能否在 4000 步（16M token）恢复；稠密 d=64 同预算对照
$py = 'python'
$s = 'D:\czsai\vmlm\tools\p35_dyn_v2.py'
$o = 'D:\czsai\vmlm\artifacts\p35'
"=== start c1_t4000"
& $py $s --mask-src h --keep 0.01 --steps 4000 --eval-every 400 --threads 2 --tag c1_t4000 *> "$o\c1_t4000.log"
"=== done c1_t4000 exit=$LASTEXITCODE"
"=== start d64_t4000"
& $py $s --mask-src none --d 64 --steps 4000 --eval-every 400 --threads 2 --tag d64_t4000 *> "$o\d64_t4000.log"
"=== done d64_t4000 exit=$LASTEXITCODE"
"=== 超长组结束 ==="
