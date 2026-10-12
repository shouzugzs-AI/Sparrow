# 第二步：CJK 切字表（实测报告 + 对照 CMD）

> **一句话结论：你的前提不成立 —— 观测词表本来就有 94.5% 是中文**（10,920 个类型里 10,317 个是 CJK）。
> 纯 CJK 过滤只能把词表 10,921 → **10,501（−3.8%）**，达不到 3000-6000。
> 而**按训练频次裁剪**才能到那个区间；代价我已实测：**留 5,000 类 ⇒ val OOV 3.51%（<5% 达标）**，
> 且它**在两个轴上都优于 CJK 过滤**（词表更小、OOV 更低）。所以两个模式我都实现了，CMD 见 §5。

## 1. 统计结果（用官方 MiniCPM tokenizer 逐 id 解码后分类）

```
tokenizer         : <外部模型目录>/MiniCPM5-1B/tokenizer.json  (vocab_size=130560)
distinct train ids: 10920     train 455996 tok / val 54082 tok
kinds: CJK=10317   punct/digit/space=182   special=1   other=420
```

| 方案 | n_vocab | 旧→新 | val OOV | train OOV |
|---|---|---|---|---|
| observed（现用） | 10,921 | — | 0.608% | 0% |
| **CJK 过滤（你要的）** | **10,501** | **96.2%** | **3.829%** | 3.431% |
| 频次裁剪 8,000 | 8,001 | 73.3% | 1.337% | 0.693% |
| 频次裁剪 6,000 | 6,001 | 55.0% | 2.487% | 1.818% |
| **频次裁剪 5,000（推荐）** | **5,001** | **45.8%** | **3.513%** | 2.857% |
| 频次裁剪 4,000 | 4,001 | 36.6% | **5.163% ✗ >5%** | 4.482% |
| 频次裁剪 3,000 | 3,001 | 27.5% | **7.890% ✗ >5%** | 7.242% |
| 频次裁剪 2,000 | 2,001 | 18.3% | 12.940% ✗ | 12.299% |
| 频次裁剪 1,500 | 1,501 | 13.7% | 17.309% ✗ | 16.640% |

（`keep_N` = 保留训练集里频次最高的 N 个类型；OOV 按 **token 级**统计。）

**判据对照**：你要的 3000-6000 —— 3,000 档 OOV **7.89% > 5%**（按你的纪律必须回退）；4,000 档 5.16% 也超标；
**5,000 档 3.51% 达标 ✓**。所以"3000-6000"里可用的是 **5,000~6,000**。

## 2. 被砍掉的是什么（前 20 高频，按训练频次）

```
old_id    train_freq   text
   115          5064   '~'        ← 一个 token 占语料 1.1%
 33582           911   '~\n'
   263           220   U+FFFD     ← 字节级 BPE 的半个字符片段
   139           214   U+FFFD
   ...（大量 U+FFFD 片段）
```
⇒ "外文碎片"确实存在（420 个类型），但**频次最高的是 `~` 和 U+FFFD 片段**，而它们总共只占 3.4% 的 token。
**把它们换成 UNK 换来的是 3.8% 的参数节省 —— 不值得。**

## 3. 参数量影响（当前最优配置：k0e=32, k0h=32, head-skip ⇒ embed+head 共 3·V·32）

| 词表 | embed+head+skip | main | attention(稀疏) | **总计** | 相对现用 |
|---|---|---|---|---|---|
| observed 10,921 | 1,048,416 | 294,912 | 294,912 | **1,638,240** | — |
| **CJK 10,501** | 1,008,096 | 294,912 | 294,912 | **1,597,920** | **−2.5%** |
| 频次 6,001 | 576,096 | 294,912 | 294,912 | **1,165,920** | −28.8% |
| **频次 5,001** | 480,096 | 294,912 | 294,912 | **1,069,920** | **−34.7%** |

⇒ **CJK 过滤只省全模型的 2.5%**（因为 main+attention 共 590K 与词表无关）。
⇒ 频次裁剪 5,000 省 **34.7%**，这才是"嵌入和 head 参数按比例下降"的有效手段。

## 4. 实现（`--vocab-mode cjk` + `--vocab-map`）

* 新脚本 `tools/b1_vocab_cjk.py`（Python，因为 tokenizer 在 Python 侧）：
  - 逐 id `tokenizer.decode([id])` 解码 → 分类（CJK / 标点数字空白 / 特殊 / 其他）
  - CJK 范围：4E00-9FFF、3400-4DBF、F900-FAFF、20000-2A6DF、2A700-2EBEF、2E80-2EFF、3000-303F、31C0-31EF、FF00-FFEF
  - 产出映射：`vocab_*.txt`（**每行一个旧 raw id**，行号+1 = 新 id；0 = UNK）+ `vocab_*.json`（统计）
  - `--mode cjk`（默认）/ `--mode freq --keep-n N`（频次裁剪）
* 驱动新增：`--vocab-mode cjk|freq|... --vocab-map FILE`
  - **任何非 full/observed 的词表模式都必须显式给 `--vocab-map`，否则 `exit 2`**（不许静默回退到 observed）
  - 未列出的旧 id → 0 (UNK)
* **新增跨词表比较的诚实口径**（必需，否则 val 会"假性下降"）：
  切表后那些不可预测的稀有 token 全变成同一个**可预测**的 id 0 ⇒ 整体 val CE 会下降，但这不是模型变好。
  现在每次 eval 都打印两个数：
  ```
  [val 口径] all=9.2408  非UNK目标=9.2404  (UNK 目标 2/120 = 1.667%)
  ```
  **跨词表只能比"非UNK目标"那一列。**（8 步冒烟里 observed=9.3081 / cjk=9.2404 / freq5000=8.5282 —— 差异主要来自 ln V 的地板，还没有训练信号。）

## 5. CMD（你手动跑，5000 步）

### 5.1 你要的：CJK 词表（照抄当前最优配置，只改词表）

```cmd
cd <repo> && cargo build --release -p vmlm-train --bin vmlm-b1-full && .\target\release\vmlm-b1-full.exe --sparse-io 1 --dim 1536 --k0 8 --k0-embed 32 --k0-head 32 --layers 24 --no-residual 1 --steps 5000 --seq 32 --batch 4 --lr 3e-4 --warmup 100 --min-lr 3e-5 --seed 1 --data-dir artifacts/b1_scaling --vocab-mode cjk --vocab-map artifacts/b1_scaling/vocab_cjk.txt --out-dir artifacts/b1_cjk_k32 --doc docs/B1_cjk_k32.md --eval-every 100 --plateau-patience 1 --attn-every 4 --attn-sparse 1 --k0-attn 8 --attn-src-shared 1 --attn-calib 1 --attn-calib-target qk --head-skip 1
```
（`--attn-calib-target qk` = 你要的 raw 目标；`--head-skip 1` = 消融二配置；attention 残差默认开。）

### 5.2 我推荐的：频次裁剪 5,000（省 34.7% 参数，OOV 3.51% 仍达标）

```cmd
cd <repo> && .\target\release\vmlm-b1-full.exe --sparse-io 1 --dim 1536 --k0 8 --k0-embed 32 --k0-head 32 --layers 24 --no-residual 1 --steps 5000 --seq 32 --batch 4 --lr 3e-4 --warmup 100 --min-lr 3e-5 --seed 1 --data-dir artifacts/b1_scaling --vocab-mode freq --vocab-map artifacts/b1_scaling/vocab_freq5000.txt --out-dir artifacts/b1_freq5k_k32 --doc docs/B1_freq5k_k32.md --eval-every 100 --plateau-patience 1 --attn-every 4 --attn-sparse 1 --k0-attn 8 --attn-src-shared 1 --attn-calib 1 --attn-calib-target qk --head-skip 1
```

### 5.3 对照（同 seed/步数，同一份代码，只差词表）—— 做对照请**现跑一次** observed，别用历史数

```cmd
cd <repo> && .\target\release\vmlm-b1-full.exe --sparse-io 1 --dim 1536 --k0 8 --k0-embed 32 --k0-head 32 --layers 24 --no-residual 1 --steps 5000 --seq 32 --batch 4 --lr 3e-4 --warmup 100 --min-lr 3e-5 --seed 1 --data-dir artifacts/b1_scaling --vocab-mode observed --out-dir artifacts/b1_obs_k32 --doc docs/B1_obs_k32.md --eval-every 100 --plateau-patience 1 --attn-every 4 --attn-sparse 1 --k0-attn 8 --attn-src-shared 1 --attn-calib 1 --attn-calib-target qk --head-skip 1
```

**判读**：比较 `非UNK目标` 那一列的 val 曲线；同时记录 `params:` 行与 `tok/s`（切表后 logits/softmax 的 V 变小，
每步会更快 —— 速度差异请在 `summary.json` 的 `tok_per_s` 上读）。

重新生成映射（想换 keep_n 时）：

```cmd
python tools/b1_vocab_cjk.py --data-dir artifacts/b1_scaling --tokenizer "<外部模型目录>/MiniCPM5-1B/tokenizer.json" --mode cjk --out artifacts/b1_scaling/vocab_cjk
python tools/b1_vocab_cjk.py --data-dir artifacts/b1_scaling --tokenizer "<外部模型目录>/MiniCPM5-1B/tokenizer.json" --mode freq --keep-n 6000 --out artifacts/b1_scaling/vocab_freq6000
```

## 6. 诚实边界

1. **你的前提（"10921 含大量外文碎片" ⇒ 能砍到 3000-6000）被数据否掉了**：94.5% 已是 CJK，纯过滤只省 3.8% 类型 / **2.5% 总参数**。
2. **val 不能跨词表直接比**：UNK 目标变成可预测的单 id ⇒ 整体 CE 会假性下降。必须看 `非UNK目标` 列；我已把它做成每次 eval 都打印。
3. **OOV 是 token 级**统计（类型级另算）；1,500-4,000 档全部超过你的 5% 上限，**按纪律这些档不可用**。
4. 冒烟只跑了 8 步（dim=128）：它验证的是"映射能载入、参数量对、两条口径都在打"，**不是** val 优劣。
5. 我没有跑 5000 步（按纪律）。`~` 这一个 token 占语料 1.1%，把它变成 UNK 的影响要等真实训练才能量化。
6. 第三步（扩规模）**未开始**，等你确认第二步后再算预算。
