# 等 FLOP 稠密对照的"长训 + 各自最优 lr"版本：
#   d=80 对应 C5(1.874M)，d=112 对应 C15(3.264M)，都用 lr=5e-3 跑 4000 步
$py = 'python'
$s = 'D:\czsai\vmlm\tools\p35_dyn_v2.py'
$o = 'D:\czsai\vmlm\artifacts\p35'
"=== start d80_lr5e3_t4000"
& $py $s --mask-src none --d 80 --lr 0.005 --min-lr 5e-4 --warmup 120 --steps 4000 --eval-every 400 --threads 2 --tag d80_lr5e3_t4000 *> "$o\d80_lr5e3_t4000.log"
"=== done d80 exit=$LASTEXITCODE"
"=== start d112_lr5e3_t4000"
& $py $s --mask-src none --d 112 --lr 0.005 --min-lr 5e-4 --warmup 120 --steps 4000 --eval-every 400 --threads 2 --tag d112_lr5e3_t4000 *> "$o\d112_lr5e3_t4000.log"
"=== done d112 exit=$LASTEXITCODE"
"=== 完成 ==="
