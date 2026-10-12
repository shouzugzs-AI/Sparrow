# 重复抑制已加 + 「是不是实现 bug」的判定性证据 + 预训练数据集接入清单

## 一、已完成：repetition penalty / no-repeat-ngram（默认关闭）

| 开关 | 作用 |
|---|---|
| `--rep-penalty P`（默认 1.0=关）| 历史里出现过的 token：正 logit 除以 P、负 logit 乘 P（只看最近 64 个）|
| `--no-repeat-ngram N`（默认 0=关）| 禁止形成重复的 N-gram（N≥2）|
| chat / EXE | 提示符里 `rp=1.2 ng=4` 可改 |

两者都在 **top-k 之前**生效（被压下去的词不会占 top-k 名额）。

**实测（`01.cpk`，prompt=今天，100 字，`--n-candidates 1`）**

| 运行 | 自评 bits/char | 不同字 | 最频繁4-gram覆盖 |
|---|---|---|---|
| seed1（关）| 2.90 | 49 | 0.08 |
| **seed1（开）** | 3.57 | **76** | 0.04 |
| seed2（关）| 10.00 | 94 | 0.04 |
| seed2（开）| 9.77 | 92 | 0.04 |
| seed5（关）| 10.10 | 85 | **0.19（在复读）** |
| **seed5（开）** | 10.10 | 96 | **0.04（不复读了）** |
| seed7/9（开）| 10.14 / 10.17 | 90 / 96 | 0.04 |

⇒ **能治复读**（seed1 更丰富、seed5 的环消失），**但治不了乱码**（自评仍 ~10，val 是 6.5）。
正文见 `artifacts\gen\REP_REPORT.md`。

## 二、你的判断"实现有问题"：我把可判定的都判了，结论是**推理实现是对的**

四个独立证据（都指向"模型/数据"，不是"推理代码"）：

1. **val=6.53 bits/char 是用同一条前向算出来的**（同一个 `step_fn`、同一套权重/词表/attention）⇒ 模型在**真实上下文**上确实会预测 ✓
2. **`--score-file` 用完全相同的训练前向给生成文本打分**：好样本 2.9、乱码 10.0-13.2 ⇒ **训练前向自己就认为那些乱码不合理** ⇒ 不是生成路径的假象 ✓
3. **全支撑采样（`--top-k 4054 --top-p 1.0 --temperature 1.0`）**：seed1 自评 **6.68**（= 模型熵，理论期望正是如此）⇒ **采样器确实按模型分布抽** ✓（采样器有 bug 的话这里会立刻暴露）
4. **重复抑制去掉环之后，乱码的自评没变好**（10.10 → 10.10）⇒ 乱码是**模型条件分布本身**的性质，不是采样/解码代码造成的 ✓
   （补：如果生成时的架构开关与训练不一致——比如 `--no-residual`/`--relu`——那么**连 seed1 的好文本和 val 打分都会崩**，而它们都好 ✓ ⇒ 架构也对得上）

**所以现象是**：teacher-forcing（val 6.5）与 free-running（自评 10±）之间有巨大落差 —— 这是 **exposure bias**，
在小模型（2.29M）+ 字符级 + **小数据集** 上必然被放大：**一旦上下文离开"真实文本流形"，模型的条件分布就退化到比均匀还差**。
**你的两个猜测（"训练存在问题" / "数据集太小"）就是对的**；而"只有 seed1 能用"= 只有它落回了好流形 ✓（不同随机流=不同轨迹）。

## 三、预训练数据集接入清单（按现有工具）

1. **词表必须重建**（`n_vocab` 已经烧进检查点 ⇒ 换词表 = 从头训）：
   `tools\b1_char_vocab.py --mode freq --keep-n 8000 --from all`
   ⇒ 产出 `train_ids.bin / val_ids.bin / meta.json / char_vocab_hex.txt`；**看 meta 里的 OOV**（现在这套是 4054 字、val OOV 0.17%）。
2. **语料编码**：现有脚本是把 MiniCPM tokenizer 流**解码**成字；如果你的预训练语料是**原始文本**，需要一个"原始文本 → 按现词表 char id"的编码器（约 30 行，我可以直接写）。
3. **训练命令建议（把这几轮的实测结论都带上）**：
   ```cmd
   .\target\release\vmlm-b1-full.exe --sparse-io 1 --dim 4096 --k0 8 --k0-embed 16 --k0-head 16 --k0-attn 8 ^
     --layers 32 --no-residual 1 --attn-every 4 --attn-sparse 1 --attn-src-shared 1 --attn-calib 1 --attn-calib-target qk ^
     --head-skip 1 --seq 64 --batch 32 --seed <记下来!> --steps <更多> --lr 3e-4 --warmup 200 --min-lr 3e-5 --clip 1 ^
     --attn-window 1 --parallel-mode inter --persistent-threads 1 --threads 8 --pin 1 ^
     --vocab-mode full --chars-per-token 1.0 --data-dir artifacts\<新数据> ^
     --out-dir artifacts\<新run> --eval-every 200 --log-every 50 --plateau-patience 1 --save-model artifacts\<新>.cpk
   ```
   * **`--attn-window 1` 一定要带**：逐窗口语义才正确，而且 batch 32 下快 **8.5×**（实测）
   * `--parallel-mode inter --persistent-threads 1`：与 spawn 逐位相同、稳
   * **seed 必须记下来**（`.cpk` 头部不存 seed，`summary.json` 存了 ⇒ 两者放一起备份）
4. **验收指标（用我们已有的工具，别只看"像不像话"）**：
   * val bits/char（字符级均匀地板 11.99；现在 6.5）
   * **`--score-file` 给生成文本打分**：6-8 = 可读；11+ = 已跑出分布
   * **跑 ≥4 个 seed，看自评的分布**（而不是只看一个"运气好的"）—— 这才是"模型到底学会没有"的客观口径
   * 复读检查：最频繁 4-gram 覆盖率（≈1 就是复读）

## 四、纪律

* 只改了采样器（新增两个默认关闭的开关）+ chat 参数；**没碰内核、没碰训练、没碰 intra**
* `intra` 逐位不变 `0xc9c987c15d75d1d6` ✓；套件 **65/65** ✓；`build_all` 引擎 74752 B 未变 ✓
* 未跑长训练（最长 1 步）

**下一步（等你定）**：
(a) 写"原始文本 → char id"编码器（换预训练语料必需）；
(b) 换词表后我可以把"vocab 重建 → 训练 → 4 seed 自评分布"整条验收脚本给你；
(c) 需要在 `.cpk` 旁边落一份 `<name>.meta.json`（seed + 全部开关）就说一声。

---

## 补记（2026 本轮）：chat 侧接线已完成

内核 `sample_next` 的 rep-penalty / no-repeat-ngram **早已实现**（`b1_full_train.rs` L497-508），
trainer 也有 `--rep-penalty`（默认 1.0=关）与 `--no-repeat-ngram`（默认 0=关）；缺的是 **`vmlm-chat` 没接线**。
本轮补上（约 20 行，只动 `train/src/bin/vmlm_chat.rs`）：

* 默认值取本文档 §1 的推荐：**`rp=1.2`、`ng=4`**（原默认 = 关闭）。
* 提示符里可改：`rp=1.0 ng=0 今天`（`rp=1.0` 关重复惩罚、`ng=0` 关 N-gram 禁止）。
* 启动横幅打印当前值：`采样默认: … | 重复抑制 rp=1.2 ng=4`。

**验证**
| 判据 | 结果 |
|---|---|
| 确定性 | ✅ 同 prompt 连跑两次输出完全一致 |
| **可回退性（关键）** | ✅ `rp=1.0 ng=0 今天` 的生成正文 sha = **`01170E142D5EA85B`**，与接线前的基线**逐字节相同** ⇒ 旋钮确实生效、且能一键回到旧行为 |
| 影响面 | ✅ 引擎 / tests / b1-full 均未改动（只在 chat 的 argv 组装与参数解析里加 rp/ng） |

新默认（rp=1.2 ng=4，prompt=`今天`）实际输出：
```
> 见你这就等着晨光，咱们再不然开始看？嗷呜~
（眼睛亮的手心地转圈圈）小尖上啦！
啊呜！那只好，我先去把尾巴勾住蜜蜂蜜香香气不过来，尾巴轻一半天要带着你可是有松鼠的话了，我们再来！那边看见过这么说到你这
```

