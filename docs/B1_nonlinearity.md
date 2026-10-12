# 线性/非线性检查报告 + ReLU 接入 + k0 拐点扫描命令

> 第一步（检查）**已做**：代码 + 数值双证据。第三/四步按检查结果执行。**完整训练我没有跑**，命令在 §5。

## 1. 第一步：当前 B1 有没有非线性？—— **没有，且已数值证明**

### 1.1 代码层面

| 位置 | 事实 |
|---|---|
| `core/src/b1_pure.rs:151` `b1_forward` | `y[i] = Σ_{j<k0} w[i][j]·x[src[i][j]]` —— 纯线性组合（gather + FMA） |
| 同上，文档注释（第 142 行） | **"不读也不写 `sc.act`。掩码恒为全 1"** —— 没有门控、没有阈值 |
| 全文件 grep `relu/gelu/tanh/sigmoid/clamp/max(` | **0 个逐元素非线性**（命中的 6 处都是注释/名字） |
| `train/src/b1_full_train.rs` 层循环 | `h = B1_l(h)`，之后**直接**进下一层；唯一的 `.max()` 都是数值守卫 |
| 反向 `b1_backward_fused` | 也只做 `dw += dy·x`、`dx += w·dy` |

唯一的非线性在**损失**里（softmax-CE），不在隐藏层。另外**没有任何跨 token 交互**（每个位置独立前向）。

### 1.2 数值层面（新增 `--linearity-probe 1`）

同一批 128 个 token、同一组随机权重，把**输入**分别 (a) 缩放 α=2、(b) 拆成互不相交的 A/B 两半：

| 检验 | 实测（dim=1536, 24 层） | 含义 |
|---|---|---|
| 齐次性 `max\|y(2x) − 2·y(x)\|` | **0.000000e0（逐位 0）** | α=2 在 f32 下是精确缩放；逐位 0 ⇒ 无非线性、无偏置 |
| 可加性 `max\|y(A+B) − y(A) − y(B)\|` | **5.96e-8**（相对 1.3e-7，出现在第 21 层） | 只有 f32 末位舍入（8+8 项 vs 16 项的加法次序不同），**不是非线性项** |

⇒ **24 层稀疏堆叠 = 一个线性映射**（数学上 `W₂₄·…·W₁`），再加上没有跨 token 交互，
结构上只能表达 **bigram**（`logits[v] = Σ_d M[v][d]·E[t][d]`）。

**这与"末层 stable rank 1.17"完全一致**：线性 + 无跨位置交互 ⇒ 表示按训练把能量集中到 1~2 个方向。

## 2. 第三步：加 ReLU（`--relu 1`）

改动（**最小、不碰主层内核、不引入 top-k、保持无残差**）：

1. 每层 `b1_forward` 之后：保存**预激活**到 `pres[l]`，再逐元素 `y = max(y, 0)`；
2. 反向：把 `dy` 按 `pres[l] > 0` 掩码后再调用**同一个** `b1_backward_fused`；
3. 线性性探针与激活 dump 都走同一条前向（保证探针测的是真模型）。

### 2.1 冒烟（dim=128, k0=2/4/4, 8 步）与"ReLU 真的生效"的证明

| 检验 | 结果 |
|---|---|
| 线性性探针 **不加** ReLU | `LINEARITY PASS`（齐次性逐位 0） |
| 线性性探针 **加** ReLU | **`LINEARITY FAIL`**，可加性偏差 2.218e-1（相对 2.27e-1）|
| dim=128 8 步冒烟 | exit 0、指标全有限、参数账目不变（88392）|
| ReLU 反向数值对账 | eps=0.5 **FAIL**（4/18 越界，跨 ReLU 折点）；**eps=0.05 PASS**（0/18，worst 1.78e-2）；eps=0.005 PASS（18/18 受噪声限制）|
| 确定性 | 同 seed 两次 `loss_hash` 相同（`0x42151e89e3334c62`）|

> 注：齐次性在 ReLU 下也是 0（ReLU 是正齐次的），**可加性**才是抓 ReLU 的那条 —— 探针因此能区分"没有非线性"与"有 ReLU 这种非线性"。
> 折点边界：FD 探针在大 eps 下会跨折点，所以 ReLU 的对账用 eps=0.05；线性模型在 eps=0.5 下也 PASS（worst 1e-3）。

### 2.2 完整配置实测（dim=1536, k0=8/16/16, 24 层, **1000 步**）

| 配置 | val loss | PPL | 末层 keff95 | 末层 stable rank |
|---|---|---|---|---|
| 无 ReLU | 8.2759 | ~3900 | 2 | 1.17 |
| **加 ReLU** | **7.7677**（−0.51） | 2363 | **1** | **1.03** |

**⇒ ReLU 让 val 明显改善（−0.51 nats），但末层秩仍然坍缩（甚至更彻底：1.17 → 1.03）。**
所以秩坍缩**不是**线性造成的，**加非线性也不能修复它**。

## 3. 一个必须先摆上桌的参照点：**unigram 基线 = 7.3973**

新增 `tools/b1_unigram_baseline.py`（在**与训练器完全相同**的评估位置上算，无参数）：

| 参照 | val CE | PPL |
|---|---|---|
| 均匀分布 `ln V` | 9.2984 | 10920 |
| **只数词频（unigram，10920 个计数 ≈ 11K 参数）** | **7.3973** | 1631 |

对照各配置：

| 模型 | val | 与 unigram(7.397) 比 |
|---|---|---|
| 稀疏 IO 无 ReLU（1000 步） | 8.2759 | **差 0.88（连词频都没学到）** |
| 稀疏 IO 加 ReLU（1000 步） | 7.7677 | 差 0.37 |
| 稀疏 IO k0e=32（1000 步） | 7.6513 | 差 0.25 |
| 稀疏 IO（**你的 3000 步**） | 7.34 | **基本等于 unigram（+0.06）** |
| 稠密 IO（**你的 3000 步**） | 5.87 | 比 unigram 好 1.53 ⇒ 真学到了结构 |

⇒ 稀疏 IO 这条路**目前连词频表都没打赢**；而"知识在嵌入+head"的怀疑在这条数据链上得到支持的形式是：
**稀疏层把末层压成近秩 1，head 读到的东西几乎没有 token 特异性 ⇒ 预测退化成与上下文无关的词频。**

评估口径体检（顺手加的）：评估批**目标 UNK = 1.411%**（7/496），与全量 val 的 0.608% 同量级 ⇒
**val loss 不是被"猜不到的 OOV 词"顶住的**，评估协议没问题。

## 4. 第四步：秩仍坍缩 ⇒ 转 k0 拐点扫描（命令在 §5.3）

判据与目标（写清楚，免得事后解释）：
* **k0_head**（决定 98% 算力）：找 `val(k0_head)` 的边际收益拐点；
* **k0_embed**：它同时是"输入秩上界"，预期收益最大；
* **k0（主层）**：加宽每层感受野。
* 每条扫描：**画 X=k0, Y=val**，找一阶差分明显变小（边际收益递减）的 k0。
* ⚠️ 结构约束：本实现要求 `k0_head ≤ k0_embed`（head 读共享子空间），
  所以 k0_head=64 那一档必须配 `k0_embed=64`（命令里已按此写）。

## 5. CMD（**给你的，我没跑**）

时间基准（实测）：无 ReLU **0.060 s/步**、加 ReLU **0.074 s/步** ⇒ 3000 步 ≈ **3.1 / 3.7 分钟**。

### 5.1 ReLU 完整训练（与同 seed/同步数/同 lr 的无 ReLU 对照）

```cmd
cd <repo> && cargo build --release -p vmlm-train --bin vmlm-b1-full && .\target\release\vmlm-b1-full.exe --sparse-io 1 --dim 1536 --k0 8 --k0-embed 16 --k0-head 16 --layers 24 --no-residual 1 --steps 3000 --seq 32 --batch 4 --lr 3e-4 --warmup 100 --min-lr 3e-5 --seed 1 --data-dir artifacts/b1_scaling --vocab-mode observed --out-dir artifacts/b1_relu_k16 --doc docs/B1_relu_k16.md --eval-every 50 --plateau-patience 1 --relu 1 --dump-acts artifacts/effrank/relu3000
```

对照（去掉 `--relu 1`，改输出目录为 `artifacts/b1_norelu_k16` / `docs/B1_norelu_k16.md`）即可同 seed 对比。

跑完后看秩（不需要再训练）：

```cmd
python tools/b1_effrank.py --dir artifacts/effrank/relu3000 --out docs/B1_effrank_relu3000.md
python tools/b1_unigram_baseline.py --data-dir artifacts/b1_scaling
```

### 5.2 ReLU + k0_embed=32（当前最好的组合）

```cmd
cd <repo> && .\target\release\vmlm-b1-full.exe --sparse-io 1 --dim 1536 --k0 8 --k0-embed 32 --k0-head 32 --layers 24 --no-residual 1 --steps 3000 --seq 32 --batch 4 --lr 3e-4 --warmup 100 --min-lr 3e-5 --seed 1 --data-dir artifacts/b1_scaling --vocab-mode observed --out-dir artifacts/b1_relu_k32 --doc docs/B1_relu_k32.md --eval-every 50 --plateau-patience 1 --relu 1
```

### 5.3 k0 拐点扫描（三条，共 15 个配置 ≈ 55 分钟）

**k0_head 扫描**（固定 dim=1536, k0=8, k0_embed=64 以允许 k0_head=64）：

```cmd
cd <repo> && for %K in (4 8 16 32 64) do .\target\release\vmlm-b1-full.exe --sparse-io 1 --dim 1536 --k0 8 --k0-embed 64 --k0-head %K --layers 24 --no-residual 1 --steps 3000 --seq 32 --batch 4 --lr 3e-4 --warmup 100 --min-lr 3e-5 --seed 1 --data-dir artifacts/b1_scaling --vocab-mode observed --out-dir artifacts/sweep_k0head_%K --doc docs/sweep_k0head_%K.md --eval-every 50 --plateau-patience 1
```

**k0_embed 扫描**（固定 dim=1536, k0=8, k0_head=16）：

```cmd
cd <repo> && for %K in (8 16 32 64 128) do .\target\release\vmlm-b1-full.exe --sparse-io 1 --dim 1536 --k0 8 --k0-embed %K --k0-head 16 --layers 24 --no-residual 1 --steps 3000 --seq 32 --batch 4 --lr 3e-4 --warmup 100 --min-lr 3e-5 --seed 1 --data-dir artifacts/b1_scaling --vocab-mode observed --out-dir artifacts/sweep_k0embed_%K --doc docs/sweep_k0embed_%K.md --eval-every 50 --plateau-patience 1
```

**k0（主层）扫描**（固定 dim=1536, k0_embed=32, k0_head=16）：

```cmd
cd <repo> && for %K in (2 4 8 16 32) do .\target\release\vmlm-b1-full.exe --sparse-io 1 --dim 1536 --k0 %K --k0-embed 32 --k0-head 16 --layers 24 --no-residual 1 --steps 3000 --seq 32 --batch 4 --lr 3e-4 --warmup 100 --min-lr 3e-5 --seed 1 --data-dir artifacts/b1_scaling --vocab-mode observed --out-dir artifacts/sweep_k0_%K --doc docs/sweep_k0_%K.md --eval-every 50 --plateau-patience 1
```

（`for %K in (...)` 是 **cmd.exe** 语法；PowerShell 里用 `foreach ($K in 4,8,16,32,64) { ... }`。）

## 6. 诚实边界

1. **ReLU 的 FD 对账必须用小 eps**（0.05）：eps=0.5 会跨折点，出现 4/18 越界 —— 那是折点的性质，不是反向写错（同 eps 下线性模型 PASS 且最坏 1e-3）。
2. §2.2 的秩对比是 **1000 步**、单 seed；"ReLU 让 val 降 0.51" 在 1000 步上成立，3000 步要按 §5.1 复核。
3. **秩坍缩的机制我只测到"现象+与线性一致"，没有证明它必然发生**：线性 + 无跨位置交互 ⇒ 最多 bigram 是**结构推断**；"训练把能量集中到 1 个方向"是观测。
4. unigram 基线**只数 train 的词频**（UNK 用 1/(2N) 兜底，另给 1e-12 兜底版 7.5936）；它是"零上下文"参照，不是"人类水平"参照。
5. 本轮的 `k0_head ≤ k0_embed` 约束让"k0_head 扫描"必须把 k0_embed 抬到 64；若你要真正的 k0_head>k0_embed（head 读非共享维度），需要先解除共享子空间设计。
6. 三份 rank 报告与频率统计已归档在 `artifacts/b1_io_stats/`；`--linearity-probe`/`--relu`/`--dump-acts` 都已进 `tools/b1_full_validate.ps1`（**53/53 通过**）。
