# 知识在嵌入+head 还是稀疏层？—— 归因实验设计 + 实验 D 结果

> 本文件由我写的脚本/驱动产出，**实验 D 已跑**（不需要训练）；**A/B/C 是给你的命令，我没有跑**。

## 0. 先说方法学：为什么"参数占比"不能回答这个问题

- 参数占比：稀疏层 0.295M(0.9%) / 嵌入 16.77M(49.5%) / head 16.77M(49.5%)。
- 但**参数量 ≠ 贡献**：嵌入天然是 `V × d` 查表矩阵，head 天然是 `d × V` 读出矩阵；它们大是因为词表大，不是因为它们在"思考"。
- 反过来，稀疏层小也可能关键（它就是唯一的非线性混叠装置）。
- 唯一能区分的方法是**冻结/替换对照**（A/B/C）+ **表示几何度量**（D）。

另一个必须先纠正的事实（上一轮实测，`artifacts/b1_io_stats/stats_128steps_dim1536.md`）：

| 常见直觉 | 实测 |
|---|---|
| "head 的 16 条连接里只有 0.08 条命中活跃神经元、99.5% 读零" | **假**：命中率 **1.000000**（22,366,208/22,366,208），每行平均命中 **16.0/16** |
| "最后一层只有 8 个神经元活跃" | **假**：1536/1536 个神经元非零率**全为 1.0000**（每 token 非零 1536.0 维）|

所以 A/B/C 要问的**不是**"head 有没有读到东西"，而是"读到的东西有没有token 特异性"。

## 1. 实验 D（已跑）：逐层激活的有效秩

工具：`--dump-acts <dir>`（驱动在训练结束后把每层激活写盘，**用 val 流的固定 token 批**）+ `tools/b1_effrank.py`（独立可运行，只依赖 numpy）。

定义：矩阵 `A = [tokens × dim]`（每行一个 token 的激活），**先按维中心化**再 SVD；
`keff95` = 奇异值平方累积能量达 95% 所需方向数；`stable` = ‖A‖_F²/σ₁²。

### 1.1 随机初始化（1 步，等价于"未训练"）

| 矩阵 | 每行非零 | 精确秩 | keff95 | stable |
|---|---|---|---|---|
| input | 16.0 | 117 | **103** | 37.2 |
| layer_00 | 122.0 | 117 | 100 | 34.9 |
| layer_01 | 750.0 | 117 | 98 | 33.4 |
| layer_02 | 1528.0 | 117 | 96 | 30.8 |
| layer_03 … layer_23 | 1536.0 | 117 | 94 → **61** | 30.6 → 15.6 |

- **支撑集确实逐层展开**：16 → 122 → 750 → 1536（第 3 层起全稠密）。这与"随机稀疏混叠会指数扩散"的预测一致。
- **有效秩不增反降**：103 → 61（中心化精确秩恒为 117 = 该 token 批的内在张成维数，受 tokens=128 限制）。
- ⇒ 未训练时，稀疏层的作用是**混叠（把稀疏支撑变稠密）但收缩能量方向**。

### 1.2 训练 1000 步后（k0_embed=16，val=8.2759）

| 矩阵 | keff95 | stable |
|---|---|---|
| input | 103 | 31.1 |
| layer_08 | 80 | 10.7 |
| layer_16 | 28 | 1.71 |
| layer_20 | 3 | 1.27 |
| layer_22 | 2 | 1.20 |
| **layer_23（末层）** | **2** | **1.17** |

### 1.3 训练 1000 步后（k0_embed=32，val=7.6514）

末层 **keff95 = 1，stable = 1.05**，同样坍缩（甚至更彻底）。

### 1.4 D 的结论（**这是本次最硬的一条数据**）

> 训练之后，24 层无残差稀疏堆叠把表示**压缩到近秩 1~2**：
> 128 个不同 token 的末层激活几乎共线（stable rank 1.17 ⇒ 单个方向占 ~85% 能量）。
> 也就是说末层状态**几乎不携带 token 特异性** —— 这与"稀疏层在传递信息"的直觉相反，
> 而**支持**"预测几乎由 head/嵌入这一侧决定"的怀疑，但机制不是"head 读到零"，而是
> **"head 读到的向量在所有 token 上几乎一样"**。

判别式（对应任务书判据）：
- "24 层后有效秩仍 ≈ k0_embed ⇒ 稀疏层没增加信息"：**方向对但形式不同** —— 秩没有掉到 16，而是掉到 **~2**，比 k0_embed 还低。
- "有效秩扩大 ⇒ 稀疏层在做变换"：**未观察到**（任何配置下都是单调下降）。
- 边界：秩是**表示几何**度量，**不能**单独证明"知识在哪"；它只说明末层状态对 head 几乎是常数。归因结论仍要靠 A/B/C。

## 2. 实验 A/B/C：冻结/替换对照（**命令给你，我不跑**）

三个实验与"全量基线"必须同 seed / 同 lr / 同 3000 步；每组的参数哈希都写进 `summary.json`，
`hash_*_before` 与 `hash_*_after` **相同即证明该组一个比特都没动**（驱动已实现并自测，见 §3）。

共同参数（dim=1536, k0=8/16/16, 24 层, 无残差, 3000 步, **融合反向已默认开启**）：

```cmd
--sparse-io 1 --dim 1536 --k0 8 --k0-embed 16 --k0-head 16 --layers 24 --no-residual 1 --steps 3000 --seq 32 --batch 4 --lr 3e-4 --warmup 100 --min-lr 3e-5 --seed 1 --data-dir artifacts/b1_scaling --vocab-mode observed --eval-every 50 --plateau-patience 1
```

### 实验 A（冻结稀疏层：只训嵌入 + head）

```cmd
cd <repo> && cargo build --release -p vmlm-train --bin vmlm-b1-full && .\target\release\vmlm-b1-full.exe --sparse-io 1 --dim 1536 --k0 8 --k0-embed 16 --k0-head 16 --layers 24 --no-residual 1 --steps 3000 --seq 32 --batch 4 --lr 3e-4 --warmup 100 --min-lr 3e-5 --seed 1 --data-dir artifacts/b1_scaling --vocab-mode observed --out-dir artifacts/b1_attr_A_freeze_layers --doc docs/B1_attr_A.md --eval-every 50 --plateau-patience 1 --freeze-layers 1
```

**读法**：若 val ≈ 全量基线（7.34 附近）⇒ 稀疏层贡献小；若明显更差 ⇒ 稀疏层在承载信息。
**可证**：`summary.json` 里 `hash_layers_before == hash_layers_after`。

### 实验 B（冻结嵌入 + head：只训稀疏层）

```cmd
cd <repo> && cargo build --release -p vmlm-train --bin vmlm-b1-full && .\target\release\vmlm-b1-full.exe --sparse-io 1 --dim 1536 --k0 8 --k0-embed 16 --k0-head 16 --layers 24 --no-residual 1 --steps 3000 --seq 32 --batch 4 --lr 3e-4 --warmup 100 --min-lr 3e-5 --seed 1 --data-dir artifacts/b1_scaling --vocab-mode observed --out-dir artifacts/b1_attr_B_freeze_io --doc docs/B1_attr_B.md --eval-every 50 --plateau-patience 1 --freeze-io 1
```

**读法**：若 val ≈ 基线 ⇒ 嵌入+head 不是关键；若 val 基本不动（≈ ln V = 9.30）⇒ 单独训稀疏层没用。
**可证**：`hash_embed_*` 与 `hash_head_*` 前后相同。

### 实验 C（稀疏层恒等：y = x，只训嵌入 + head）

```cmd
cd <repo> && cargo build --release -p vmlm-train --bin vmlm-b1-full && .\target\release\vmlm-b1-full.exe --sparse-io 1 --dim 1536 --k0 8 --k0-embed 16 --k0-head 16 --layers 24 --no-residual 1 --steps 3000 --seq 32 --batch 4 --lr 3e-4 --warmup 100 --min-lr 3e-5 --seed 1 --data-dir artifacts/b1_scaling --vocab-mode observed --out-dir artifacts/b1_attr_C_identity --doc docs/B1_attr_C.md --eval-every 50 --plateau-patience 1 --identity-layers 1
```

**读法**：恒等映射下模型退化为"嵌入与 head 在同一子空间上的双线性打分"，参数量不变（0.645M）。
若 val 与全量差不多 ⇒ 24 层稀疏层在这条数据上**没有提供超出"词对打分"的信息**（即模型近似 n-gram）。
**可证**：`identity_layers=true` 且 `hash_layers_*` 前后相同。

### 全量基线（对照，必须同参数同 seed 跑一遍）

```cmd
cd <repo> && cargo build --release -p vmlm-train --bin vmlm-b1-full && .\target\release\vmlm-b1-full.exe --sparse-io 1 --dim 1536 --k0 8 --k0-embed 16 --k0-head 16 --layers 24 --no-residual 1 --steps 3000 --seq 32 --batch 4 --lr 3e-4 --warmup 100 --min-lr 3e-5 --seed 1 --data-dir artifacts/b1_scaling --vocab-mode observed --out-dir artifacts/b1_attr_base --doc docs/B1_attr_base.md --eval-every 50 --plateau-patience 1
```

### 有效秩分析（任意一次跑完都能做，不额外训练）

```cmd
python tools/b1_effrank.py --dir artifacts/effrank/base --out docs/B1_effrank.md
```

（先把上面任一条命令加上 `--dump-acts artifacts/effrank/base`。）

## 3. 这些开关已被验证（不是"写了就算"）

`tools/b1_full_validate.ps1` 现有 **49 项检查**，其中与本任务直接相关：

| 检查 | 结果 |
|---|---|
| F1 A：冻结稀疏层后 `hash_layers` 前后**完全相同**，且 embed/head 确实变了 | PASS（`0xefc9fb9b3c1a9df7 -> 0xefc9fb9b3c1a9df7`）|
| F2 B：冻结 io 后 `hash_embed`/`hash_head` 前后相同，且 layers 确实变了 | PASS |
| F3 C：恒等模式下 layers 未被触碰且 loss 有限 | PASS |
| F4 dump + `b1_effrank.py` 可独立跑通并出报告 | PASS |
| S11 融合反向 vs 两条标量调用 `loss_hash`/`final_val` **逐位相同** | PASS |
| S12 动态子空间换维均衡（每维使用 spread=40 < 1536）| PASS |

## 4. 诚实边界

1. **实验 D 是 128 个 token 的一批 val 数据**（受 `min(tokens,dim)=117` 上限），中心化 SVD；换批/换 token 数会改变绝对数值，但**逐层单调下降**的形态在三种配置（未训练/k0e=16/k0e=32）都成立。
2. **秩坍缩 ≠ "知识在 head"**：它只说明末层状态对 head 近乎常数。要下归因结论必须看 A/B/C 的 val 曲线。
3. A/B/C 里冻结组的**最优 lr 可能不同**（例如只训 head 时 3e-4 未必最优），所以"冻结更差"不能直接换算成"该组贡献为零"；**证据强度：A/B/C 同 lr 的对照 > 单次冻结结果**。
4. **未做**：逐层冻结（只冻前 k 层）、按梯度范数量化的贡献、以及"打乱某一组参数"的随机对照（那是比冻结更强的因果证据）。要的话我加。
5. 本轮 `k0_head ≤ k0_embed`（共享子空间），`k0_head > k0_embed` 会报错退出（S14）。
