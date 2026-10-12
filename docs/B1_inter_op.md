# inter-op 数据并行（按样本切分）—— 已实现并验证

> `--parallel-mode inter|intra`，**默认 inter**（按你的要求）；intra 保留为 fallback。
> 关键等价性已实测：**inter 4 线程 × 每线程 8 样本 ≡ 单线程 batch 32**（loss 差 **4.2e-5**，浮点归约顺序所致）。
> 实现过程中 FD 对账 + 确定性测试抓出 **4 个真 bug**（都很隐蔽，见 §5），全部修好后才通过。

## 1. 实现（最小改动：779 行 step 体**逐字不动**）

核心手法：把 step 体参数化成 `(off, tt, slot, toks, train_mode)`，在**体首按 region/slot 平移后重绑定同名变量**，
于是体内 700+ 行原样复用（没有第二份前向/反向）：

| 机制 | 做法 |
|---|---|
| 样本切分 | `nslot = min(threads, batch)`，每线程 `sb = batch/nslot` 个**完整样本**，`off = slot*sb*seq` |
| token 维缓冲（h_ins/pres/y/logits/dlogits/dh/dy/dx/h/d_h0…） | **不复制**：每线程只碰自己那一段行 ⇒ 按 `off` 平移 |
| attention 工作缓冲（aq/ak/av/ao/aout/aprob/adprob/adq/adk/adv/adobuf/arowsum/asp/adx） | **按 slot 分段**（并发写，必须私有）；步长用 `t_tokens`（分配尺寸），不是 `tt` |
| 每线程私有 dw | 稀疏层 `dws[l]`（`dim*k0`/slot）、`head_dw`/`head2_dw`/`etab_g`/`asq_g..aso_g`；`dwp`（反向暂存）也按 slot |
| 末端归约 | **按 slot 升序**把 slot1..nslot-1 加进 slot0（每层 dw、head、etab、attention 梯度…）|
| Adam | 归约后**只更新一次**（不是每线程各更新）|
| 体内并行度 | inter 下 `nt_in = 1`（并行性由外层 region 提供）；intra 下 `nt_in = nth`（旧行为）|

窗口数从 `batch` 改为 `sb`：inter 下每个 region 的 attention 窗口就是它自己的样本 ✓（intra 下 `sb == batch` 故逐位不变）。

## 2. 验证结果（全部实测）

| 验证 | 结果 |
|---|---|
| **intra 路径逐位不变** | `--parallel-mode intra --threads 1/2/4` ⇒ `loss_hash=0xc9c987c15d75d1d6`（与改动前一致）|
| **inter 退化（threads=1 ⇒ 1 个 region）** | 与 intra 逐位相同：`0x8718b4c63489fd74` |
| **确定性**（batch 32, 4 线程, 跑两次） | `loss_hash=0xd1837f6b798a36b6` **两次相同** ✓ |
| **等价性**：1 线程 batch 32 vs 4 线程 ×8 | `final_train` 8.304582 vs 8.304540（**Δ=4.2e-5**）；val 8.306237 vs 8.303803 |
| **等价性**：1 线程 batch 8 vs 4 线程 ×2 | 8.301596 vs 8.301398（**Δ=2.0e-4**）|
| **等价性**：1 线程 batch 4 vs 4 线程 ×1 | 8.305658 vs 8.305617（**Δ=4.1e-5**）|
| **FD 对账**（数值梯度） | `--parallel-mode inter --threads 2/4`：**PASS**（16 探针越界 0）|
| **intra 仍是 4 线程一致的** | 4 线程 intra 与 1 线程**逐位相同**（token 切分 + 固定归约顺序）|

**速度（1M 配置，`--attn-window 1`）**

| 配置 | 1 线程 | 4 线程 inter | 4 线程 intra | 加速 |
|---|---|---|---|---|
| batch 4, 10 步 | 3.7 s | **1.9 s** | — | **1.95×** |
| batch 8, 20 步 | 14.1 s | **7.4 s** | 7.9 s | **1.9×** |
| batch 32, 10 步 | 30.4 s | **16.0 s** | 16.9 s | **1.9×** |

**结论（据实说）**：
1. 4 线程拿到 **~1.9×**（不是 4×）。这台机器/执行环境的天花板本来就低（4 个独立进程实测 2.88×），
   而且 batch 32 时每线程只有 512 个 token，摊薄不了线程创建/同步的开销。
2. **inter 与 intra 的墙钟几乎一样**（16.0 s vs 16.9 s，inter 快 ~5%）：因为层内同步本来就不是主要成本
   （每层一次 dw 归约）。inter 的价值在**架构**（无层内同步、梯度天然私有、可直接扩到多机），
   而不是在当前这台 4 核机器上再挤出几倍。
3. 梯度质量 = batch 32 ✓（loss 差 4.2e-5）。

## 3. 对照实验 CMD

```cmd
:: ① batch 32：inter 4 线程（每线程 8 样本）
cd <repo> && .\target\release\vmlm-b1-full.exe --sparse-io 1 --dim 1536 --k0 8 --k0-embed 32 --k0-head 32 --k0-attn 8 --layers 24 --no-residual 1 --attn-every 4 --attn-sparse 1 --attn-src-shared 1 --attn-calib 1 --attn-calib-target qk --head-skip 1 --seq 64 --batch 32 --seed 1 --steps 10 --lr 3e-4 --warmup 3 --min-lr 3e-5 --attn-window 1 --threads 4 --parallel-mode inter --data-dir artifacts/b1_char --vocab-mode full --chars-per-token 1.0 --out-dir artifacts/iop_b32_t4 --eval-every 10 --log-every 1 --plateau-patience 0

:: ② 同配置单线程（1 个 region = 整个 batch 32）—— 与 ① 应只差浮点末位
::    把 --threads 4 换成 1、--out-dir 换 iop_b32_t1

:: ③ intra fallback（对照）
::    把 --parallel-mode inter 换成 intra、--out-dir 换 iop_b32_intra
```
**⚠️ 比较 `loss_hash` 时必须固定 `--log-every`**：`loss_hash` 是对**被记录的步**求哈希，
`--log-every 30` 与 `10` 会给出不同哈希（我为此白排查了一轮）。要么固定它，要么比 `loss_curve.csv`。

## 4. 纪律核对

| 纪律 | 结果 |
|---|---|
| 不改稀疏层内核 | ✅ `b1_forward/b1_backward*`、attention 内核未动（引擎 core size 仍 74752 B）|
| 不改 intra-op 路径 | ✅ intra 在 1/2/4 线程下逐位不变（`0xc9c987c15d75d1d6`）|
| 每线程私有 dw | ✅ 稀疏层/head/etab/attention 梯度都按 slot 分段 |
| 归约顺序固定 | ✅ slot 升序；同配置两次运行 `loss_hash` 相同 |
| 采样 top-k 与 B2 top-k 分离 | ✅ 未动 |
| 每步 bit-exact 或报告差异 | ✅ intra 逐位；inter 与单线程差 2e-4~4e-5（报告了量级）|
| 不自己跑完整训练 | ✅ 最长 20 步 |

## 5. 实现中抓到的 4 个 bug（都已修，记录备查）

1. **slot 步长用错变量**：重绑定块里的 `t_tokens` 被批量替换成了 `tt`（= `t_tokens/nslot`），
   而分配尺寸是 `nslot * t_tokens * …` ⇒ **各 region 的缓冲互相重叠** ⇒ NaN + 两次运行结果不同。
   （教训：*分配尺寸*与*区域 token 数*是两个量，必须分开。）
2. **共享 scratch 没私有化**：`aprob`（前向 attention 概率）、`d_h0`（head-skip 回传到 h0 的梯度）、
   `dwp`（`par_b1_backward` 的每线程 dw 暂存 —— inter 下内层 nt=1，所有 region 都用 tid=0 的同一段）⇒ 竞争。
3. **`h` 没平移**：最末层 `dest = h`，而 `h` 是 token 维共享缓冲 ⇒ 各 region 覆盖彼此的最末层输出。
4. **字段级捕获**：`par_for(nslot, nslot, |..| (sw.0)(..))` 会直接捕获 `SharedFn.0`（含裸指针）⇒ 不满足 `Sync`；
   必须整体借用（`let swr = &sw; move |..| (swr.0)(..)`）。

## 6. 诚实边界

1. **建议 inter 一律配 `--attn-window 1`**：inter 下每个 region 的 attention 上下文 = `sb*seq`，
   若用整块 attention，inter 与 intra 的**语义**就不同（驱动会打印警告）。本文所有对照都带了 `--attn-window 1`。
2. **默认已是 inter**（按你的要求）。`--threads 1` 时 nslot=1 ⇒ 与旧行为逐位相同，
   所以套件/历史脚本不受影响（已复验 65/65）。
3. **加速比不如预期**：~1.9×（不是 4×，也不是 2.9×）。原因是环境天花板 + 每线程工作量小；
   若要更接近上限，需要把 `par_for` 的**线程池常驻**（现在每步/每 region 都 spawn）——这是下一步可做的优化。
4. 本轮只测到 32 batch / 4 线程 / 20 步；没有做长时间训练或 8 线程 × 大 batch 的扫描。
