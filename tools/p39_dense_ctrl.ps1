# 稠密对照：① d=64 在 lr1e-3 下 4000 步（对标 C1 高 lr 长训）
#           ② d=80 在 lr5e-3 下 600 步（看稠密能不能同样吃高 lr）
$py = 'python'
$s = 'D:\czsai\vmlm\tools\p35_dyn_v2.py'
$o = 'D:\czsai\vmlm\artifacts\p35'
"=== start d80_lr5e3_600"
& $py $s --mask-src none --d 80 --lr 0.005 --min-lr 5e-4 --warmup 120 --steps 600 --eval-every 100 --threads 2 --tag d80_lr5e3_600 *> "$o\d80_lr5e3_600.log"
"=== done d80 exit=$LASTEXITCODE"
"=== start d64_lr1e3_t4000"
& $py $s --mask-src none --d 64 --lr 1e-3 --min-lr 1e-4 --steps 4000 --eval-every 400 --threads 2 --tag d64_lr1e3_t4000 *> "$o\d64_lr1e3_t4000.log"
"=== done d64 exit=$LASTEXITCODE"
"=== 稠密对照结束 ==="
