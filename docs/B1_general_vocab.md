# 通用字符词表（ModelScope 维基语料）—— 已建好，以后直接用

## 1. 数据集（ModelScope，< 1 GB）

| 项 | 值 |
|---|---|
| 数据集 | `AI-ModelScope/wikipedia-cn-20230720-filtered`（中文维基百科，过滤版）|
| 文件 | `wikipedia-cn-20230720-filtered.jsonl` **496.6 MB** |
| 下载后路径 | `<数据集目录>/wiki_zh.jsonl`（仓库外，不占 git）|
| 规模 | **254,547 篇 / 190,549,956 字符 / 18,642 个不同字符** |
| 格式 | 每行一个 JSON：`{"completion": "<维基正文>", "source": "wikipedia.zh2307"}` |

可复现的下载命令（实测 46 秒 / 10.6 MB/s）：
```cmd
curl.exe -L --retry 3 -o <数据集目录>/wiki_zh.jsonl "https://modelscope.cn/api/v1/datasets/AI-ModelScope/wikipedia-cn-20230720-filtered/repo?Revision=master&FilePath=wikipedia-cn-20230720-filtered.jsonl"
```
（`modelscope.cn` 可达；**huggingface.co 在这台机器上超时**，所以走魔搭 ✓）

## 2. 词表产物（两版，都是驱动的标准格式）

| 目录 | 字符数 | `vocab_size` | 覆盖率 | **留出 OOV** | 定位 |
|---|---|---|---|---|---|
| `artifacts/char_general\` | 8,192 | **8,198** | 99.9665% | **0.0369%** | 紧凑版（embed/head 最小）|
| `artifacts/char_general16k\` | 12,652 | **12,658** | **99.9950%** | **0.0066%** | **推荐：通用版**（OOV 低 5.6×）|

每版三个文件：
* `char_vocab_hex.txt` —— `id<TAB>kind<TAB>value`，**id 0 = `<unk>`，1..5 是 sentinel（`</s> <end> </br> <s> <pad>`），6 起按频率降序**（与 `b1_char` 完全同构，驱动/chat 直接吃）
* `char_freq.tsv` —— rank / 字符 / 码点 / 次数 / 累计覆盖率（以后想重截词表，直接从这里取，不用重扫语料）
* `meta.json` —— 来源、语料规模、覆盖率、**留出段 OOV**

前 11 高频字（合理性检查）：`， 的 。 \n (空格) 1 0 年 2 、 *` ✓（数字/星号多是维基表格残留，正常）

**代价**（稀疏 IO 架构下很便宜）：每个字 ≈ `k0_embed + k0_head(+k0_head)` 个参数 ⇒ `k0e=k0h=16` 时每字约 48 参数；
换到 12,658 字只比 4,054 字多约 **0.4M 参数**。

## 3. 顺手产出的"可直接训练"数据包

`artifacts/zh_wiki\`：`train_ids.bin` **142.9 MB（37,470,455 字符）**、`val_ids.bin`、`meta.json`、`char_vocab_hex.txt`
（取语料**前 50,000 行**做数据包，词表仍用全量 190M 字符统计 ✓）

**冒烟测试（已跑，exit 0）**：
```
data: 37470455 train tok / 78426 val tok | vocab_mode=full n_vocab=8198 (ln V=9.0116) val OOV 0 tok (0.000%)
final_train=9.012116 final_val=9.018031   bits/char=13.01（未训练 ⇒ 正好是均匀地板 ln8198/ln2=13.0 ✓）
```

## 4. 两个工具（以后换语料直接用）

```cmd
:: ① 从任意原始语料（jsonl 或纯文本）建通用词表（+可选数据包）；190M 字符约 20 秒
python tools/b1_general_vocab.py --jsonl <数据集目录>/wiki_zh.jsonl --text-key completion ^
    --out-vocab artifacts/char_general16k --out-data artifacts/zh_wiki ^
    --target-coverage 0.99995 --max-chars 16384 --val-frac 0.002 --data-max-lines 50000

:: ② 用**已有**词表编码任意文本/语料（不重新统计字频）
python tools/b1_text_to_ids.py --vocab artifacts/char_general16k/char_vocab_hex.txt ^
    --jsonl <数据集目录>/another.jsonl --text-key text --out-dir artifacts/another_data --val-frac 0.002
python tools/b1_text_to_ids.py --vocab ... --in some.txt --stats-only     :: 只看 OOV
```

## 5. 怎么用它训练（3 个注意点）

1. **换词表 = 从头训**：`n_vocab` 已经烧进检查点（`CkGeo` 里没有 seed/开关，但有 `n_vocab`），所以旧 `01.cpk` 不能复用。
2. `--vocab-mode full` 会从 `<data-dir>/meta.json` 读 `vocab_size` ✓；`dim` 仍必须是 `k0_embed` 的正整数倍 ✓。
3. 建议命令（把前面几轮的结论都带上）：
```cmd
.\target\release\vmlm-b1-full.exe --sparse-io 1 --dim 4096 --k0 8 --k0-embed 16 --k0-head 16 --k0-attn 8 ^
  --layers 32 --no-residual 1 --attn-every 4 --attn-sparse 1 --attn-src-shared 1 --attn-calib 1 --attn-calib-target qk ^
  --head-skip 1 --seq 64 --batch 32 --seed <记下来> --steps <更多> --lr 3e-4 --warmup 200 --min-lr 3e-5 --clip 1 ^
  --attn-window 1 --parallel-mode inter --persistent-threads 1 --threads 8 --pin 1 ^
  --vocab-mode full --chars-per-token 1.0 --data-dir artifacts/zh_wiki ^
  --out-dir artifacts/zh_wiki_run --eval-every 200 --log-every 50 --plateau-patience 1 --save-model artifacts/zh_wiki.cpk
```
* **`--attn-window 1` 必带**（逐窗口语义正确 + batch32 实测快 8.5×）
* 验收用 `--score-file`（生成文本自评 6-8 = 可读，11+ = 跑出分布）+ **≥4 个 seed 看自评分布**

## 6. 备注

* 语料是**维基百科**（百科体）。要更"通用"，后面可以用同一个工具把新闻/小说/对话 jsonl **追加**进数据包（词表不用重建，用工具 ② 编码即可）。
* 词表里同时有字面 `\n`（rank 4）和 sentinel `</br>`：`b1_text_to_ids.py` 默认把换行映射到 `</br>`（若词表里有），与现有 `b1_char` 数据一致 ✓；两条路径不要混用同一次训练。
* 本轮只建词表/数据，**没有跑长训练**（冒烟 2 步）。
