# 长训(4000 步)的大稠密参照：C5/C15 的最优 lr 长训已经超过 d=112，
# 需要 d=192 / d=256 才能找到"等损失对应的稠密规模"。
$py = 'python'
$s = 'D:\czsai\vmlm\tools\p35_dyn_v2.py'
$o = 'D:\czsai\vmlm\artifacts\p35'
"=== start d192_lr1e3_t4000"
& $py $s --mask-src none --d 192 --lr 1e-3 --min-lr 1e-4 --steps 4000 --eval-every 400 --threads 2 --tag d192_lr1e3_t4000 *> "$o\d192_lr1e3_t4000.log"
"=== done d192 exit=$LASTEXITCODE"
"=== start d256_lr1e3_t4000"
& $py $s --mask-src none --d 256 --lr 1e-3 --min-lr 1e-4 --steps 4000 --eval-every 400 --threads 2 --tag d256_lr1e3_t4000 *> "$o\d256_lr1e3_t4000.log"
"=== done d256 exit=$LASTEXITCODE"
"=== 完成 ==="
