# 跨层残差（嵌入 → head）验证报告

> 本轮目标：验证"**嵌入 → head 跨层残差**"能否修复秩坍缩。**完整训练我没有跑**，CMD 在 §5。

## 1. 问题的定位（与你的判断一致的部分）

上一轮实测的三件事拼在一起：

| 事实 | 来源 |
|---|---|
| 24 层稀疏堆叠是**纯线性**（齐次性逐位 0） | 线性性探针（§1 of `B1_nonlinearity.md`）|
| 末层表示坍缩到**近秩 1**（keff95=2、stable rank 1.17）| 逐层 SVD |
| head 只读末层、每行 16 条稀疏连接 | 设计 |

⇒ head 的输入几乎是**一个方向**的常数向量，token 特异性只能靠"嵌入 → 24 层 → 末层"这条**很窄**的路传过来。
**这不是层内残差问题**（层内残差已被实验证明更差），而是**跨层**：嵌入与 head 之间缺一条直接通路。

## 2. 实现（最小改动，只接入既有内核）

新增 `--head-skip 1`：

```
前向：  logits[v] = Σ_j W_mid[v][j]·h_last[dim_of[v][j]]      (原来的 head)
                  + Σ_j W_skip[v][j]·h0[dim_of[v][j]]        (新增：直连嵌入展开)
反向：  两个 head 各自一套梯度；直连项的 dx 在**层循环之后**加回 dh（它不穿过 24 层），
        再与层传播的梯度一起写进嵌入表
```

* 复用的内核：`b1_forward` / `b1_backward_fused`（**零新内核**）
* 参数：`+V×k0_head`；k0_head=16 时 **+174,736**（总量 644,384 → **819,120**，仍在 1M 内）
* 计算：`+V×k0_head` MACs/token（与 head 同量级，远小于稠密 head）
* 中间层**一个字没动**（保持无残差）

**必须写明的结构限制**：head 行 v 读的是 `block(v)` 的维度，而 `h0` 的非零只在 `block(t)` 的维度
⇒ 直连项**只在 `block(v) == block(t)` 时非零**，即"块内 bigram"：每个输入 token 直连区分它自己
块里的 ~114 个候选 token。这**不是**全 V 路直连（那需要稠密 head 或低秩双线性，见 §6）。

## 3. 验证：直连是否真的接上了

| 检查 | 结果 |
|---|---|
| 参数账目（K1） | `params=132,076`（dim=128 时）与闭式一致；`macs_head` 翻倍 ✓ |
| **反向数值对账（K3）** | `GRADCHECK PASS: 18 探针，越界 0，worst rel=1.581e-3`——**包含 `emb_tab` 探针**，即直连项回传到嵌入的梯度也被对账覆盖 ✓ |
| 确定性（K2） | 同 seed 两次 `loss_hash` 相同（`0x7b5a6b2258ac371a`）✓ |
| 冒烟 | dim=128 8 步、无 NaN、产物齐全 ✓ |

## 4. 结果：**信息瓶颈被修好了，但 val 没有改善**

### 4.1 head 的输入信息（瓶颈本身）

新增度量：把**末层输出**与**原始嵌入展开**各自归一化后拼接，再算有效秩
（归一化是必须的：head 对两条通路各有独立权重矩阵，尺度差不该被算成"方向少"——
第一版直接拼接时 keff95 恒等于末层的 2，是个假象）。

| 配置 | 末层 keff95 | **head 可用输入 keff95** |
|---|---|---|
| 无直连 | 2 | **2**（只有末层） |
| **有直连** | 2 | **98** |

⇒ 直连把 head 的可用输入方向从 **2 提到 98（49×）**。**它不改变末层**（设计如此：残差加在嵌入→head，
不在层间），所以"末层秩坍缩"依旧存在，但**head 不再受那个瓶颈限制**。

### 4.2 val（dim=1536, 24 层, 1000 步, 同 seed/lr）

| 配置 | 无直连 | **有直连** |
|---|---|---|
| k0_embed=16, k0_head=16 | 8.2759 | **8.2783**（+0.002）|
| k0_embed=32, k0_head=32 | 7.6513 | **7.6507**（−0.001）|

**两个配置下都没有改善（差异在噪声内）。**

### 4.3 为什么"修好瓶颈"却不涨分？—— 参照点给了答案

`tools/b1_unigram_baseline.py`（在与训练器**完全相同**的评估位置、只数词频、无参数）：

| 参照 | val CE |
|---|---|
| 均匀分布 `ln V` | 9.2984 |
| **unigram（纯词频）** | **7.3973** |

* 上面所有配置（8.28 / 7.65）**都还在 unigram 之上** ⇒ 它们连"词频表"都还没打赢，
  处在**欠训练**区间。在欠训练区间比较两个架构，比的是"谁更欠训练"，不是架构优劣。
* 你已有的 3000 步数据：稀疏 IO **7.34 ≈ unigram 7.397**（刚好打平），稠密 IO **5.87**（真正低于）。

⇒ **当前的第一瓶颈是"训练步数/优化"，不是 head 的信息量。** 直连是一个**正确的结构修复**，
但它要等到模型跨过 unigram 线之后才可能显出差别。

## 5. CMD（给你的，我没跑完整训练）

时间基准：无直连 **0.060 s/步**、带直连约 **0.070 s/步** ⇒ 3000 步 ≈ **3.1 / 3.6 分钟**。

### 5.1 直连 + ReLU + k0e=32（把"结构修复"和"非线性"一起上）

```cmd
cd <repo> && cargo build --release -p vmlm-train --bin vmlm-b1-full && .\target\release\vmlm-b1-full.exe --sparse-io 1 --dim 1536 --k0 8 --k0-embed 32 --k0-head 32 --layers 24 --no-residual 1 --steps 3000 --seq 32 --batch 4 --lr 3e-4 --warmup 100 --min-lr 3e-5 --seed 1 --data-dir artifacts/b1_scaling --vocab-mode observed --out-dir artifacts/b1_skip_relu_k32 --doc docs/B1_skip_relu_k32.md --eval-every 50 --plateau-patience 1 --head-skip 1 --relu 1 --dump-acts artifacts/effrank/skip_relu_k32
```

### 5.2 严格对照：同 seed/同步数/同 lr，只差 `--head-skip`

```cmd
cd <repo> && .\target\release\vmlm-b1-full.exe --sparse-io 1 --dim 1536 --k0 8 --k0-embed 32 --k0-head 32 --layers 24 --no-residual 1 --steps 3000 --seq 32 --batch 4 --lr 3e-4 --warmup 100 --min-lr 3e-5 --seed 1 --data-dir artifacts/b1_scaling --vocab-mode observed --out-dir artifacts/b1_noskip_relu_k32 --doc docs/B1_noskip_relu_k32.md --eval-every 50 --plateau-patience 1 --relu 1
```

跑完后（不需要再训练）：

```cmd
python tools/b1_effrank.py --dir artifacts/effrank/skip_relu_k32 --out docs/B1_effrank_skip.md
python tools/b1_unigram_baseline.py --data-dir artifacts/b1_scaling
```

**判读标准（先写下来，避免事后解释）**：
* 两条曲线都**高于 7.40** ⇒ 仍然欠训练，结论是"步数不够"，与结构无关；
* 有直连的曲线**明显低于**无直连、且跨过 7.40 ⇒ 直连有效；
* 两条都低于 7.40 但几乎重合 ⇒ 直连的收益被 k0e=32 已经提供的信息覆盖了。

## 6. 诚实边界

1. **直连不是"全 V 直连"**：受共享子空间约束，它只在同块内非零（每输入 token 约 114 个候选）。
   要做到"head 对全部 V 直接读嵌入"，需要稠密 head（16.8M 参数，破坏 1M 预算）或**低秩双线性**
   `logits2 = U·(M·e_t)`（rank=16 时 +199K 参数 / +199K MACs，用既有 `gemm_nt` 实现）。
   要哪一种请指定，我可以按你选的做。
2. **1000 步的结果**：两个配置的差异 ≤0.002，**在 seed 噪声内**，不能作为"直连无效"的定论——
   只能作为"在这个预算下看不出来"。
3. **末层秩不会因直连改变**（这是设计而非缺陷）；直连改的是 head 的**可用输入**。
4. `head-in` 秩是我定义的一个度量（归一化拼接后的 keff95），**不是**标准的互信息/有效秩定义；
   它的用途只是"head 能看到多少方向"的可比数字，且两个 dump 都会给出 98（因为 dump 里总有
   `input`）—— 所以它度量的是**可用**信息，不是**实际被使用**的信息。
5. **【已解决】上一轮报告的 L3 失败，本轮找到并修掉了根因**：	ools/b1_full_validate.ps1 是无 BOM 的 UTF-8，PowerShell 5.1 按系统代码页(GBK)解码它，中文注释的多字节序列会吃掉后面的引号/括号（表现为偶发解析错误、抓不到输出、行被合并）。写回 UTF-8 with BOM 后套件 59/59 全过，L3 也拿到输出并 PASS。原文保留如下。原描述：：同一命令独立运行 `GRADCHECK PASS: 18 探针，越界 0`
   （exit 0，已复现两次），但从长验证脚本内部调用时 exit=1 且无输出。
   已排除：参数向量一致、数据文件指纹一致、并发写日志（已把临时目录改到私有 `%TEMP%`）。
   **尚未定位**，属测试脚手架问题；ReLU 反向另有两条独立证据：独立运行的 FD 对账、以及
   K3（skip 通路、同一段反向代码）在套件内 PASS。`--head-skip` 相关检查（K1/K2/K3）全 PASS。
