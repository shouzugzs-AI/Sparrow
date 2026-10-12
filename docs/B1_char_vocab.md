# 第二步（修正版）：直接从数据集切**字符表**（不是切 BPE 类型）

> 你的判断对，我上一版测错了对象：我数的是 **BPE token 类型**（10,920），你指的是**字符**。
> 实测字符表 **4,054**（1 UNK + 4,053 字符），**train OOV = 0.00000%**、**val OOV = 0.173%**，
> 嵌入+head+head-skip 参数 **1,048,416 → 389,184（省 62.9%）**，总参数 **1,638,240 → 979,008（省 40.2%）**。
> 代价我也量了：字符级的**均匀地板换算成 bits/char 反而更差**（11.98 vs BPE 10.37），所以优劣要**训练后**比。

## 1. 字符表实测

```
token -> char : train 455996 tok -> 589976 char (x1.2938) | val 54082 -> 69909 (x1.2926)
distinct chars: train-only 4053 | train+val 4163
char categories: CJK 3887 | CJK标点/全角 30 | So(emoji/符号) 40 | Ll(拉丁) 25 | Po 16 | Sm 11 | Lu 8 | Nd 7 | ...
n_vocab (char) : 4054  (= 1 UNK + 4053 字符)
OOV train      : 0/589976 = 0.00000%     ← 构造上为 0
OOV val        : 121/69909 = 0.17308%    ← 只有 110 个"只在 val 出现的字形"（稀有汉字+个别 emoji）
```

**特殊标识符**：数据里实际出现的是 `</s>`（token id=1，train 流里只有这一个 added token）和 3 处字面量
`<end>`（1 次）、`</s>`（1 次）、`</br>`（1 次）。字表把它们折叠成**原子 id**（内部哨兵 U+E000..U+E002），
所以一个标识符 = 一个 id，不会被拆成 `<`/`e`/`n`/`d`/`>`。另外保留 `<unk>`=0。

**与上一版 BPE 裁剪的对比（同一条纪律：val OOV ≤ 5%）**：

| 方案 | n_vocab | val OOV | embed+head+skip 参数 |
|---|---|---|---|
| observed BPE | 10,921 | 0.608% | 1,048,416 |
| BPE CJK 过滤 | 10,501 | 3.829% | 1,008,096 |
| BPE 频次裁剪 5,000 | 5,001 | 3.513% | 480,096 |
| **字符表（本方案）** | **4,054** | **0.173%** | **389,184** |

⇒ 字符表**同时**做到词表更小、OOV 低 20 倍。

## 2. 参数与速度（dim=1536, k0=8, k0e=32, k0h=32, layers=24, attn-every 4 稀疏 attention, head-skip）

| | BPE observed | **字符表** | 变化 |
|---|---|---|---|
| main | 294,912 | 294,912 | — |
| embed | 349,472 | 129,728 | −62.9% |
| head（含 head-skip 第二条） | 698,944 | 259,456 | −62.9% |
| attention（6 块稀疏 k0_attn=8） | 294,912 | 294,912 | — |
| **总计** | **1,638,240** | **979,008** | **−40.2%** |
| 实测步时（seq 40/32, batch 4） | 0.40 s | **0.485 s** | +21%（每步 160 vs 128 个位置） |
| 5000 步预估 | ≈33 min | **≈40 min** | — |

**序列长度换算**：char/token = 1.2938 ⇒ 字符模式下 `--seq 40` ≈ BPE 模式 `--seq 32` 的文本跨度
（40 ÷ 1.2938 = 30.9 token ≈ 32 ✓）。所以对照实验用 **char+seq40 vs BPE+seq32**，两边每步覆盖的文本一样。

## 3. 必须先说明的"地板"问题（否则 val 会被误读）

不同分词下 CE/position **不可比**，正确的可比量是 **bits/char = CE ÷ (chars/position) ÷ ln2**：

| | 均匀地板（bits/char） | 20 步实测（bits/char） |
|---|---|---|
| BPE observed | ln(10921)/1.2938/ln2 = **10.37** | 10.367 |
| **字符表** | ln(4054)/ln2 = **11.99** | 11.976 |

⇒ **字符表的均匀地板比 BPE 差 1.6 bits/char**（因为每次只预测 1 个字符、候选只有 4054 个，
而 BPE 每次预测一个"词块"、候选 10921 个）。所以：
* 20 步的 val（8.30 vs 9.29 nats）**不能**用来宣布字符表更好 —— 两者都还贴在自己的地板上；
* 真正要比的是**训练后**的 `bits/char`，而且字符表必须**先跨过 10.37** 才算赢过 BPE 的起点。
* 驱动已内建这个度量：`--chars-per-token 1.2938`（BPE 跑）/ 默认 1.0（字符跑），日志里直接打
  `bits/char: all=… 非UNK=…`。

## 4. 实现

* **`tools/b1_char_vocab.py`**：把已切分的 token 流用官方 MiniCPM tokenizer 解回文本（`meta.json` 里
  `roundtrip_bad_docs=0`，这一步无损）→ 统计字符 → 建字表（0=UNK，1.. 按频次降序、并列按码点）→
  **重新编码**成字符 id 流，写出新数据目录 `artifacts/b1_char\{train_ids.bin,val_ids.bin,meta.json,char_vocab.json}`。
* **驱动零改动即可消费**：新目录的 id 已是 1:1，用 `--vocab-mode full --data-dir artifacts/b1_char` 即可
  （`full` 模式从 `meta.json` 读 `vocab_size`）。冒烟已验证：`n_vocab=4054`、`val OOV 0 tok`。
* 新增 `--chars-per-token R`（默认 1.0）⇒ 日志/S 打印 bits/char。
* 哨兵映射（`<end>`/`</s>`/`</br>` → U+E000..E002）记在 `char_vocab.json` 的 `sentinel_markers` 里。

## 5. CMD（你手动跑，5000 步）

### 5.1 字符表（本方案；与历史最优配置相同，只换数据+词表，seq 40 对齐文本跨度）

```cmd
cd <repo> && cargo build --release -p vmlm-train --bin vmlm-b1-full && .\target\release\vmlm-b1-full.exe --sparse-io 1 --dim 1536 --k0 8 --k0-embed 32 --k0-head 32 --layers 24 --no-residual 1 --steps 5000 --seq 40 --batch 4 --lr 3e-4 --warmup 100 --min-lr 3e-5 --seed 1 --data-dir artifacts/b1_char --vocab-mode full --chars-per-token 1.0 --out-dir artifacts/b1_char_k32 --doc docs/B1_char_k32.md --eval-every 100 --plateau-patience 1 --attn-every 4 --attn-sparse 1 --k0-attn 8 --attn-src-shared 1 --attn-calib 1 --attn-calib-target qk --head-skip 1
```

### 5.2 BPE 对照（同 seed/步数/文本跨度；请**现跑**，别用历史数）

```cmd
cd <repo> && .\target\release\vmlm-b1-full.exe --sparse-io 1 --dim 1536 --k0 8 --k0-embed 32 --k0-head 32 --layers 24 --no-residual 1 --steps 5000 --seq 32 --batch 4 --lr 3e-4 --warmup 100 --min-lr 3e-5 --seed 1 --data-dir artifacts/b1_scaling --vocab-mode observed --chars-per-token 1.2938 --out-dir artifacts/b1_bpe_k32 --doc docs/B1_bpe_k32.md --eval-every 100 --plateau-patience 1 --attn-every 4 --attn-sparse 1 --k0-attn 8 --attn-src-shared 1 --attn-calib 1 --attn-calib-target qk --head-skip 1
```

重新生成字表（想改 `--from train|all` 时）：

```cmd
python tools/b1_char_vocab.py --data-dir artifacts/b1_scaling --tokenizer "<外部模型目录>/MiniCPM5-1B/tokenizer.json" --out artifacts/b1_char --from train
```

**判读**：只看 `bits/char`（日志已打印），且必须跨过 **10.37**（BPE 均匀地板）才说明字符级真的赢；
`非UNK目标` 那列用于排除 UNK 假象（字符表 val OOV 只有 0.173%，影响很小）。

## 6. 诚实边界

1. **字表大小是 4,054，不是"3k 不到"**：3,887 个 CJK + 30 个 CJK 标点/全角 + 40 个 emoji/符号 + 25 个拉丁字母 + 少量其他。
   要压到 3k 需要丢掉约 1,100 个稀有字形（那些字在 val 里出现 121 次 / 0.17%）——可以再切一版，代价是这 0.17% 变 UNK。
2. **20 步的 val 不能说明谁更好**（见 §3 地板分析），必须等 5000 步的 bits/char。
3. **对照必须现跑**：这轮驱动改过（bits/char、诚实口径、词表路由），历史 6.79 那次没有这些口径。
4. **`--vocab-mode full` 绕过了词表映射逻辑**，所以它不会做 val OOV 检查之外的任何事 —— 切表是否正确由
   `tools/b1_char_vocab.py` 的 `meta.json`（`vocab_size`/`train_oov`/`val_oov`）负责，我已把这几项写进去。
5. 我没有跑 5000 步（按纪律）。第三步（扩规模）等这步结论后再算预算 —— 注意：**字符表已把总参数从 1.64M 降到 0.98M**，
   所以"10×"的目标基数要重新对齐（见第三步报告）。
