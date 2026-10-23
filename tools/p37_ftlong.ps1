# 加长微调到 400 步：检验"稀疏学得慢"是不是只因微调太短（150 步）
# 三个臂都用同样 400 步、lr 1e-4，同时评测新领域(fox)与旧领域(wiki)。
$py = 'python'
$s = 'D:\czsai\vmlm\tools\p35_fgt.py'
$o = 'D:\czsai\vmlm\artifacts\p37'
$w = 'D:\czsai\vmlm\artifacts\zh_wiki'
$fox = 'D:\czsai\vmlm\artifacts\fox_sft'
"=== 加长微调 dense"
& $py $s --mask-src none --data-dir $fox --eval-dir $w --init "$o\dense_pre.pt" `
    --lr 1e-4 --min-lr 1e-5 --steps 400 --eval-every 100 --threads 4 --tag "ftl_dense" *> "$o\ftl_dense.log"
"=== dense exit=$LASTEXITCODE"
"=== 加长微调 c05"
& $py $s --mask-src h --keep 0.05 --data-dir $fox --eval-dir $w --init "$o\c05_pre.pt" `
    --lr 1e-4 --min-lr 1e-5 --steps 400 --eval-every 100 --threads 4 --tag "ftl_c05" *> "$o\ftl_c05.log"
"=== c05 exit=$LASTEXITCODE"
"=== 加长微调 c15b"
& $py $s --mask-src h --keep 0.15 --data-dir $fox --eval-dir $w --init "$o\c15b_pre.pt" `
    --lr 1e-4 --min-lr 1e-5 --steps 400 --eval-every 100 --threads 4 --tag "ftl_c15b" *> "$o\ftl_c15b.log"
"=== c15b exit=$LASTEXITCODE"
"=== 加长微调结束 ==="
