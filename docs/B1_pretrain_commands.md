# 大规模预训练命令（dim4096 / k0 8 / L32 / k0e=h16 / k0attn 4 / attn-every 8 / seq 64 / 1 epoch）

## 0. 先看实测（不是估算）

按你这组配置在 `artifacts/zh_wiki`（37.5M 字符）上实测 **5 步**：

```
parallel-mode: inter (nslot=8, 每 slot 4 样本 / 256 token, 体内并行度=1)
params: main=1048576 embed=131168 head=262336 attn=262144 total=1704224
threads: 8 (每线程私有 dw 分段 131168 f32)
[step 5/5] wall=20.3s  →  4.06 s/步, 504 tokens/s
val bits/char = 13.0014（未训练 ⇒ 正好是均匀地板 ln(8198)/ln2 = 13.00 ✓）
```

**1 epoch 步数** = `训练字符数 / (batch × seq)`：

| 数据 | 字符数 | batch 32 步数 | 预计耗时 |
|---|---|---|---|
| 现有包 `artifacts/zh_wiki`（前 5 万行）| 37,470,455 | **18,296** | **≈ 20.6 小时** |
| 全量维基（需先建包，见 §4）| 190,549,956 | 93,043 | ≈ 105 小时（4.4 天）|

（8 线程已用满；这台机器 4 物理核的并行上限实测 ~2.9×，所以时间基本是算力决定的，batch 加倍只是把步数减半、单步翻倍，总时长不变。）

## 1. 命令 A（推荐先跑）：1 epoch / 18,296 步 / ≈20.6 小时

**⚠️ cmd / PowerShell 不要混用**：`Tee-Object`、反引号续行是 **PowerShell** 的；`^` 续行、`%VAR%` 是 **cmd** 的。
最省事的做法是**完全不要重定向** —— 进度直接打在控制台，同时驱动器自己会把记录写进 `--out-dir`：
`loss_curve.csv`（step,lr,train,val,ppl）、`metrics.jsonl`（每次评估一行）、`summary.json`（结束时含 `loss_hash`/`diverged_at`/`stopped_by`）✓

```cmd
cd <repo>
.\target\release\vmlm-b1-full.exe --sparse-io 1 --dim 4096 --k0 8 --k0-embed 16 --k0-head 16 --k0-attn 4 --layers 32 --no-residual 1 --attn-every 8 --attn-sparse 1 --attn-src-shared 1 --attn-calib 1 --attn-calib-target qk --head-skip 1 --seq 64 --batch 32 --seed 1 --steps 18296 --lr 3e-4 --warmup 500 --min-lr 3e-5 --clip 1 --attn-window 1 --parallel-mode inter --persistent-threads 1 --threads 8 --pin 1 --vocab-mode full --chars-per-token 1.0 --data-dir artifacts/zh_wiki --out-dir artifacts/zh_run1 --doc docs/B1_zh_run1.md --eval-every 1000 --log-every 50 --plateau-patience 0 --save-model artifacts/zh_run1.cpk --save-every 500
```
（写成一行就不会有 `^` 续行的毛病；要后台跑就 `start /b` 或另开窗口。）

**想在 cmd 里既看实时又留日志**：两个窗口 ——
```cmd
:: 窗口1：跑（输出进文件）
...上面那条命令... > artifacts/zh_run1.log 2>&1
:: 窗口2：跟日志（cmd 里显式调 powershell，语法安全）
powershell -NoProfile -Command "Get-Content '<repo>/artifacts/zh_run1.log' -Tail 20 -Wait -Encoding UTF8"
:: 或者直接从曲线看（不用日志）
powershell -NoProfile -Command "Import-Csv '<repo>/artifacts/zh_run1/loss_curve.csv' | Select-Object -Last 5 | Format-Table"
```
**想用 `Tee-Object`**：那就整条命令在 **PowerShell** 里跑（单行即可）：
```powershell
cd <repo>; .\target\release\vmlm-b1-full.exe --sparse-io 1 --dim 4096 ... --save-every 500 2>&1 | Tee-Object -FilePath artifacts/zh_run1.log
```

**⚠️ 已修的坑**：`--log-every` 原来**只把 loss 记进曲线（影响 `loss_hash`），不打印任何东西**；
你看到的那行 `[step …]` 其实挂在 **eval 块**里 ⇒ `--eval-every 1000` 时前 1000 步（≈67 分钟）屏幕全空。
现在 `--log-every N` 会真的打印进度行（已实测）：

```
[log     50/18296] lr=2.98e-4 train=8.9123 | 203s, 504 tok/s, ETA 20.42h
[log    100/18296] lr=2.96e-4 train=8.8451 | 406s, 504 tok/s, ETA 20.40h
[ckpt] step 500: 已周期保存 artifacts/zh_run1.cpk.step500（67 张量, 1704224 f32, 6.8 MB）
[step   1000/18296] lr=2.87e-4 train=8.6012 val=8.5320 ppl=  376.10 wall=4067.5s      ← eval 行（val/ppl）
```
* `--log-every 50` ⇒ 每 ≈3.4 分钟一行（含 **tok/s 与 ETA**），`--eval-every 1000` ⇒ 每 ≈67 分钟一行带 val/ppl
* `--log-every` **参与 `loss_hash`**（哈希的是被记录的步）⇒ 想横向比较就固定它

要点：
* **`--attn-window 1` 必带**（逐窗口语义才正确；batch32 实测比整块快 8.5×）
* **`--plateau-patience 0`** 关掉早停 ⇒ 保证真的跑满 1 epoch
* **`--save-every 500`**（新加的功能）≈ 每 34 分钟落一次盘到 `zh_run1.cpk.step500/1000/...`
  ⇒ 20 小时的长跑崩了也不会全丢（原来只在结束时保存 ✗）。单个检查点只有 **7 MB**，随便留
* **`--seed 1` 请记下来**：`.cpk` 头部**不存 seed**，`summary.json` 存了 ⇒ 两者放一起
* 若你选的是 12,658 字词表，把 `--data-dir` 换成用该词表重建的包（见 §4），其余不变

## 2. 跑起来后怎么盯

```cmd
:: 看进度（每 50 步一行带 ETA + 每 1000 步一次 val）
powershell -Command "Get-Content <repo>/artifacts/zh_run1.log -Tail 20 -Encoding UTF8"
:: 看落盘
dir <repo>/artifacts/zh_run1.cpk*
```
正常的样子：`[log   850/18296] lr=… train=… | 3433s, 504 tok/s, ETA 20.4h`（每 50 步）与
`[step  1000/18296] lr=… train=… val=… ppl=…`（每 1000 步）；val bits/char 应从 **13.00**（均匀地板）稳步下降。
**收尾时**看 `out-dir\summary.json` 的 `steps_done / tokens_seen / best_val / diverged / loss_hash`，以及日志末尾的 **RESULT** 段。

## 3. 跑完的验收（照这几轮定的口径）

```cmd
:: ① val bits/char（summary.json 里就有；字符级均匀地板 13.00 是这次的新词表，旧 4054 词表是 11.99）
:: ② 生成质量自评：6-8 = 可读，11+ = 跑出分布
.\target\release\vmlm-b1-full.exe --sparse-io 1 --dim 4096 --k0 8 --k0-embed 16 --k0-head 16 --k0-attn 4 --layers 32 ^
  --no-residual 1 --attn-every 8 --attn-sparse 1 --attn-src-shared 1 --attn-calib 0 --head-skip 1 ^
  --seq 64 --batch 4 --threads 4 --vocab-mode full --chars-per-token 1.0 --data-dir artifacts/zh_wiki ^
  --steps 1 --eval-every 999999 --log-every 999999 --plateau-patience 0 --load-model artifacts/zh_run1.cpk ^
  --generate 1 --prompt-file artifacts/gen/p1.txt --gen-len 100 --attn-window 1 --n-candidates 4 ^
  --rep-penalty 1.2 --no-repeat-ngram 4 --gen-out artifacts/gen/zh_run1_sample.txt
:: ③ 给生成文本打分（同一命令加 --score-file artifacts/gen/zh_run1_sample.txt 并去掉 --generate 一串）
```
**建议**：再各跑 3~4 个**不同采样 seed**，看自评的**分布**（而不是只看运气最好的那个）——这才是"学会没有"的客观口径。

## 4. 可选：全量维基（190M 字符 / 4.4 天）+ 换 12.6k 词表

```cmd
:: ① 用推荐词表（12,658 字，OOV 0.0066%）重建数据包（全量 254,547 行 ⇒ ids 约 760 MB）
python tools/b1_text_to_ids.py --vocab artifacts/char_general16k/char_vocab_hex.txt ^
  --jsonl <数据集目录>/wiki_zh.jsonl --text-key completion --out-dir artifacts/zh_wiki_full --val-frac 0.002
:: ② 命令 A 里把 --data-dir 换成 artifacts/zh_wiki_full、--steps 换成 93043
```
（`zh_wiki_full` 的 `train_ids.bin` 约 760 MB，训练时整份读进内存 —— 15.8 GB 内存够，但启动会慢一些。）

## 5. 两个必须知道的限制

1. **没有优化器状态续训**：`.cpk` 只存权重（不含 Adam 的 m/v）⇒ 从检查点"继续跑"要用 `--load-model`，
   优化器会重置（会有短暂抖动）。所以**别指望精确续训**，长跑请靠 `--save-every` 兜底。
2. **换词表 = 从头训**（`n_vocab` 烧在检查点里）⇒ §4 换词表时不要复用旧 `.cpk`。

## 6. 本次为长跑新增的东西

* `--save-every N`：周期落盘 `<save-model>.step<N>.cpk`（已验证：step2/step4 都写出 ✓，终点保存仍走同一套张量清单）
* 回归：`intra` 逐位不变 `0xc9c987c15d75d1d6` ✓；套件 **65/65** ✓；引擎 core size 74752 B 未变 ✓
