# 补两个关键对照：
#  ① d=80 在 lr1e-3 下 4000 步——d=64 的最优 lr 是 1e-3，所以 d=80 用 5e-3 可能被低配了
#  ② C15 在最优 lr(2e-3) 下 4000 步——和 ①、d112 一起给出 15% 档的最终判决
$py = 'python'
$s = 'D:\czsai\vmlm\tools\p35_dyn_v2.py'
$o = 'D:\czsai\vmlm\artifacts\p35'
"=== start d80_lr1e3_t4000"
& $py $s --mask-src none --d 80 --lr 1e-3 --min-lr 1e-4 --steps 4000 --eval-every 400 --threads 2 --tag d80_lr1e3_t4000 *> "$o\d80_lr1e3_t4000.log"
"=== done d80 exit=$LASTEXITCODE"
"=== start c15_lr2e3_4000"
& $py $s --mask-src h --keep 0.15 --lr 2e-3 --min-lr 2e-4 --warmup 60 --steps 4000 --eval-every 400 --threads 2 --tag c15_lr2e3_4000 *> "$o\c15_lr2e3_4000.log"
"=== done c15 exit=$LASTEXITCODE"
"=== 完成 ==="
