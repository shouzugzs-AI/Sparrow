# 稀疏 attention 的 src 共享修正 + q·k 体检报告

> 结论一句话：**你指出的"子空间不相交导致 q·k=0"这个机制经实测不成立（非零占比 1.0000）；
> 但你预测的"attention 退化为均匀分布、没有选择能力"是真的 —— 原因是分数尺度太小（q·k/√d≈5e-4），
> 已用一次性初始化标定修掉（mean|q·k| 3.5e-4 → 1.0，非均匀倍数 1.0 → 2.47）。**

## 1. 方案 A 已实现（Q/K 共享 src_qk，V/O 共享 src_vo）

`--attn-sparse 1` 时的 src 生成：

| 模式 | 开关 | src 份数 | 规则 |
|---|---|---|---|
| **方案 A（新默认）** | `--attn-src-shared 1` | **2** | Q/K 用 `off=0`；V/O 用 `off=4099`：`sid=(i·K+j·7919+off) mod dim` |
| 旧实现 | `--attn-src-shared 0` | 4 | 四个投影各自 `off=13·tag` |

保留旧模式**不是为了兼容，而是为了能测出两种模式的实际差别**（下面就是这么测的）。

## 2. q·k 体检：先测，再下结论

新增 `--attn-qk-check 1`（默认开）：训练前用固定 val 批跑一次前向，逐 attention 块打印
`max|q·k|`、`mean|q·k|`、`非零占比`、`max_prob`、`非均匀倍数`（= 该行 max_prob × (i+1)，1.0 即完全均匀）；
**worst max|q·k| ≤ 0.01 时直接 exit 2**。

### 2.1 你的机制：**不成立**（实测）

| 模式 | dim=128, k0_attn=2, 标定前 |
|---|---|
| 旧实现（4 份独立 src） | max\|q·k\|=5.61e-3，**非零占比=1.0000** |
| 方案 A（2 份共享 src） | max\|q·k\|=5.20e-3，**非零占比=1.0000** |

**两种模式下 q·k 从来不为零。** 原因：在 B1 的写法里 `q[t][i] = Σ_j w_q[i][j]·h[t][src(i,j)]`，
q 和 k 是 **投影输出维 i 上的稠密向量**；`q·k = Σ_i q[t][i]·k[t'][i]` 是**对输出维求和**，
每个 i 用的是它自己那条连接上的 h 分量。所以"输入维子空间不相交"**不会**让内积为零
（那只在 q/k 被当作输入维上的稀疏向量、直接点乘输入维时才成立）。

### 2.2 你预测的效果：**是真的**（但原因不同）

真正的问题是**分数尺度**：`max|q·k| ≈ 5e-3` ⇒ 除以 √d 后 `≈4.6e-4` ⇒ 分数几乎全为 0
⇒ **softmax ≈ 均匀分布 ⇒ attention 没有选择能力**。这与你预测的现象完全一致。

（顺带修掉我自己一个度量 bug：第一版集中度用 `1/T` 当均匀基线，i=0 行只有 1 个合法项、
prob 恒为 1.0，于是被算成"高度集中"，给出 `max_prob=1.0 / 集中度=4.06` 的**假象**。
现在第 i 行按 `1/(i+1)` 归一，且只统计 i≥3 的行。）

## 3. 修法：一次性分数尺度标定（`--attn-calib 1`，默认开）

训练前跑一次前向，测出 `mean|q·k|`，把 **Wq 与 Wk 同乘** `g = 1/√scale`，使初始 `q·k/√d` 量级为 1：

```
--- ATTENTION 尺度标定：worst 分数尺度(mean|q.k|)=3.4774e-4 ⇒ g=53.63（乘到 Wq/Wk）
    标定后 块0: max|q.k|=1.4957e1 mean|q.k|=9.9971e-1 非零占比=1.0000 max_prob=1.000000 非均匀倍数=2.473
```

* **只改初始化，不改前向数学**：没有 LayerNorm、没有逐步归一化、没有 top-k、稀疏层未动
* 标定后：`mean|q·k|` **3.48e-4 → 1.00**、`max|q·k|` **5.2e-3 → 15.0**、非均匀倍数 **1.0（均匀）→ 2.47**（attention 真的开始选择了）
* 方案 A 与旧模式在标定后都通过体检（非均匀倍数 2.47 / 1.40），8 步冒烟的 val 也几乎相同
  （9.30015 vs 9.30019 ⇒ 纯子空间共享这件事在 8 步尺度上看不出差别，**这本身是个数据结论**）

## 4. 判据逐条对账

| 判据 | 结果 |
|---|---|
| 1. `max\|q·k\| > 0.01` | ✅ 标定后 **15.0**（标定前 5.2e-3 ⇒ 会被判坍缩） |
| 2. softmax 非均匀（max prob > 1/seq） | ✅ 非均匀倍数 **2.47**（1.0 = 均匀）；`max_prob` 单看无意义（见 §2.2 度量 bug）|
| 3. FD 覆盖 Q/K/V/O 全 PASS | **V/O：强信号 PASS**（SNR 13→51）；**Q/K：PASS 但信号在噪声边缘**（SNR≈1，析取与数值同向、0 越界）；见 §5 |
| 4. 500 步不发散、val 合理 | **未跑**（按纪律"不自己跑"）；CMD 见 §6 |

## 5. FD 对账细节（四组都覆盖到了）

`--gradcheck 1 --attn-sparse 1 --k0-attn <dim>`（满宽投影，走**同一段** `attn_backward_sparse`）：

```
probe attn_sq  idx=3799 ana=-5.5545e-5 num=-5.7220e-5 rel=2.93e-2 SNR= 1 ok
probe attn_sk  idx=1204 ana=+6.5807e-5 num=+3.8147e-5 rel=7.25e-1 SNR= 1 ok
probe attn_sv  idx=2191 ana=-1.9710e-3 num=-1.9741e-3 rel=1.56e-3 SNR=30 ok
probe attn_so  idx=3106 ana=-3.3840e-3 num=-3.3665e-3 rel=5.21e-3 SNR=51 ok
GRADCHECK PASS: 24 探针，越界 0
```

* **V/O 是强判定**（相对误差 1.6e-3 ~ 9.6e-3，SNR 30/51）
* **Q/K 是弱判定**：解析梯度 ~5e-5，数值同向同量级，但 SNR≈1（attention 对 Q/K 的敏感度天然低于 V/O，
  因为 softmax 在初始化时仍偏平）。**我不把这条写成"已验证"，只写"无越界、无矛盾"**。
* 补充证据：softmax/分数雅可比这段代码（`attn_core_bwd`）在**稠密**路径上是被强判定的
  （`attn_wq` 探针 rel 4.0e-3/7.3e-4，SNR 69~72）；稀疏路径与它只差"投影反向由 `b1_backward_fused` 完成"，
  而该内核与本项目的标量参考实现已有逐位对账（S10/S11）。
* 上一轮那个"梯度写进权重"的 bug 已修，本轮**没有引入新 bug**（A3：默认路径 `loss_hash` 与历史逐位相同）。

## 6. CMD（你手动跑；两条都是 500 步）

### 6.1 消融二：稀疏 attention（方案 A + 标定）

```cmd
cd <repo> && cargo build --release -p vmlm-train --bin vmlm-b1-full && .\target\release\vmlm-b1-full.exe --sparse-io 1 --dim 1536 --k0 8 --k0-embed 32 --k0-head 32 --layers 24 --no-residual 1 --steps 500 --seq 32 --batch 4 --lr 3e-4 --warmup 100 --min-lr 3e-5 --seed 1 --data-dir artifacts/b1_scaling --vocab-mode observed --out-dir artifacts/abl2_sparse_attn --doc docs/abl2_sparse_attn.md --eval-every 25 --plateau-patience 0 --attn-every 4 --head-skip 1 --attn-sparse 1 --k0-attn 8 --attn-src-shared 1 --attn-calib 1 --attn-qk-check 1
```

### 6.2 消融四：组合（稀疏 attention + 无 attention 残差 + 无 head-skip）

```cmd
cd <repo> && .\target\release\vmlm-b1-full.exe --sparse-io 1 --dim 1536 --k0 8 --k0-embed 32 --k0-head 32 --layers 24 --no-residual 1 --steps 500 --seq 32 --batch 4 --lr 3e-4 --warmup 100 --min-lr 3e-5 --seed 1 --data-dir artifacts/b1_scaling --vocab-mode observed --out-dir artifacts/abl4_combo --doc docs/abl4_combo.md --eval-every 25 --plateau-patience 0 --attn-every 4 --attn-sparse 1 --k0-attn 8 --attn-src-shared 1 --attn-calib 1 --attn-qk-check 1 --attn-no-residual 1
```

（消融二/四的**上一轮 CMD 已作废**：那时的实现分数尺度是 5e-4、attention 等于没选择能力，
跑出来的 500 步是假信号 —— 正如你所料，只是原因不是子空间不相交。）

**开跑先看第一屏**：`ATTENTION 尺度标定` 与 `ATTENTION 体检` 两段必须出现且 `max|q.k| > 0.01`；
若打印 `FATAL: attention 坍缩` 并 exit 2，说明标定没生效（或被 `--attn-calib 0` 关掉了）。

## 7. 诚实边界

1. **我没跑 500 步**（按纪律）。§4 判据 4 未验证。
2. **Q/K 的 FD 只有弱判定**（SNR≈1），不写成"已验证"。
3. **标定会改变发散行为**：attention 从"没选择能力"变成"有选择能力"，它对 loss 的影响变大了，
   所以"消融二是否还发散"**不能**从上一轮的任何结果推断——必须重跑（这正是 §6.1 的意义）。
4. **方案 A vs 旧模式在 8 步尺度上分不出差别**（val 9.30015 vs 9.30019）。真正的差别要等 attention
   学会尖峰之后（500 步）才可能显出来；如果那时两者仍无差别，说明"共享子空间"对这个模型不重要——
   那也是数据结论，我会照实写。
5. 方案 B（Q/K/V/O 全共享一个 src）与方案 C（复用主层 src）**未实现**。要哪个我再做。
6. 归一化纪律：本轮**没有**引入 LayerNorm/逐步归一化；标定只是**初始化**的一步（不改前向数学）。
