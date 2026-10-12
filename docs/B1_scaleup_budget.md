# 第三步：扩规模（参数预算 + 实测耗时/内存 + CMD）

> 纪律：软 attention（`--attn-calib-target qk`）、消融二配置（head-skip + attention 残差 + 稀疏 attention）、
> 无残差中间层、B1 主层、无 top-k、字符表数据（`artifacts/b1_char`，V=4054）。
> 所有耗时/内存都是**实测**（本机 i3-12100F，单线程，AVX2）；我没跑完整训练。

## 1. 记账公式与**结构性约束**（先算清，否则配置根本跑不起来）

```
n_attn = layers / attn_every
main   = layers * dim * k0
embed  = V * k0_embed
head   = V * k0_head            （--head-skip 1 时再来一份 ⇒ 2 份）
attn   = n_attn * 4 * dim * k0_attn     （稀疏；稠密则是 n_attn * 4 * dim^2）
total  = main + embed + head(+skip) + attn
V = 4054（字符表）
```
**约束**（都是硬报错，不会静默）：
* **`dim` 必须是 `k0_embed` 的整数倍**（实测：2048/48 → `ERROR: dim (2048) must be a positive multiple of --k0-embed (48)`）；
* `k0_head ≤ k0_embed`；
* `--attn-sparse 1` 必须显式给 `--k0-attn`。

## 2. 候选配置（参数，精确值）

| 配置 | dim | k0 | k0e | k0h | k0a | layers | every | main | embed | head(+skip) | attn | **总计** | ×基准 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **CHAR 基准（现在最优）** | 1536 | 8 | 32 | 32 | 8 | 24 | 4 | 294,912 | 129,728 | 259,456 | 294,912 | **979,008** | 1.00× |
| C1 稳一点 | 2048 | 12 | 32 | 32 | 12 | 24 | 4 | 589,824 | 129,728 | 259,456 | 589,824 | **1,568,832** | 1.60× |
| C2 中档 | 3072 | 20 | 64 | 64 | 24 | 24 | 4 | 1,474,560 | 259,456 | 518,912 | 1,769,472 | **4,022,400** | 4.11× |
| C3e8 省算力 | 4096 | 32 | 128 | 128 | 32 | 32 | **8** | 4,194,304 | 518,912 | 1,037,824 | 2,097,152 | **7,848,192** | 8.02× |
| **C3 ≈10×（推荐）** | 4096 | 32 | 128 | 128 | 32 | 32 | 4 | 4,194,304 | 518,912 | 1,037,824 | 4,194,304 | **9,945,344** | **10.16×** |
| C4 ≈你原目标 | 4096 | 40 | 128 | 128 | 40 | 32 | 4 | 5,242,880 | 518,912 | 1,037,824 | 5,242,880 | **12,042,496** | 12.30× |

（`k0e=48/80/160` 这些"按比例"的值大多**不整除** dim：2048/48✗、3072/80✗、4096/160✗ ⇒ 我用的都是能整除的组合。
若你坚持某个 k0e，我可以把 dim 调成它的倍数。）

## 3. 实测：每步耗时 + 峰值内存（batch=4，字符表）

| 配置 | seq | T=4·seq | **s/step（训练循环实测）** | peakWS | 5000 步（训练循环） |
|---|---|---|---|---|---|
| C3 (9.95M) | 32 | 128 | **≈1.7**（外推） | ~250 MB | ≈2.4 h |
| **C3 (9.95M)** | **64** | 256 | **4.95** | **466 MB** | **≈6.9 h** |
| C3 (9.95M) | 128 | 512 | **15.48** | **681 MB** | **≈21.5 h** |
| C3 (9.95M) | 256 | 1024 | ≈55–60（外推，T² 主导） | ~1.3 GB | ≈76–83 h ✗ |
| C3e8 (7.85M) | 128 | 512 | ≈8.5（attention 块减半） | ~600 MB | ≈12 h |
| C2 (4.02M) | 64 | 256 | ≈3.5 | ~300 MB | ≈4.9 h |
| CHAR 基准 (0.98M) | 40 | 160 | **0.485**（上一轮实测） | ~200 MB | ≈0.7 h |

**seq 缩放的实测形状**：64→128 时 4.95 → 15.48（×3.1）⇒ 大 seq 下 T² 项开始主导。
所以**把 seq 从 128 降到 64，省 3.1 倍**，这是最有效的算力杠杆。

## 4. 顺手修掉一个**会 OOM 的内存 bug**（必须报）

第一次测 C3@seq128 时 **峰值内存 8,872 MB**。根因：`--attn-sparse 1` 下驱动**仍然分配**了整套稠密 attention 权重
（`n_attn × 4 × dim²` 的 w/grad/Adam-m/Adam-v，dim=4096、8 块 = **8.6 GB**），与"稀疏 attention 省参数"自相矛盾。
已改为只在稠密模式下分配 ⇒ 峰值 **8,872 MB → 681 MB（−92%）**，参数量不变。
（副作用：稀疏模式下那些缓冲变成未初始化 malloc，被 gradcheck 探针读到会报 `rel=4e27`；验证套件 A6 抓到了，
已一并清零/跳过。套件回到 **65/65**。）

## 5. 推荐方案

**首选：C3（9,945,344 = 10.16×）+ `--seq 64 --attn-every 4`** ⇒ 4.95 s/步、466 MB、5000 步 ≈ **6.9 h**。
短程验证 500 步 ≈ **41 min**。
若你坚持 seq 128（"完整 turns"）：同样参数要 **21.5 h**；想压到 ~12 h 就把 `--attn-every` 改成 8（参数降到 7.85M = 8.0×）。
若你坚持 ~12.9M：用 C4（12,042,496 = 12.30×），seq 64 ≈ 6.2 s/步 ⇒ 8.6 h；seq 128 ≈ 19 s/步 ⇒ 27 h。

**为什么不是 seq 256**：外推 ≈55–60 s/步 ⇒ 5000 步 76–83 h，且峰值内存 ~1.3 GB 只是权重侧，激活侧还会涨。
除非把 `--attn-every` 抬到 16 或缩短步数，否则不建议。

## 6. CMD（你手动跑）

### 6.1 短程验证（500 步，先确认不发散）— 首选配置 C3 seq 64

```cmd
cd <repo> && cargo build --release -p vmlm-train --bin vmlm-b1-full && .\target\release\vmlm-b1-full.exe --sparse-io 1 --dim 4096 --k0 32 --k0-embed 128 --k0-head 128 --k0-attn 32 --layers 32 --no-residual 1 --attn-every 4 --attn-sparse 1 --attn-src-shared 1 --attn-calib 1 --attn-calib-target qk --head-skip 1 --steps 500 --seq 64 --batch 4 --lr 3e-4 --warmup 100 --min-lr 3e-5 --seed 1 --data-dir artifacts/b1_char --vocab-mode full --chars-per-token 1.0 --out-dir artifacts/b1_C3_500 --doc docs/B1_C3_500.md --eval-every 25 --plateau-patience 0 --save-model artifacts/b1_C3_500.cpk
```
（≈41 min。判据：`diverged=false`、loss_hash 有值、`非UNK目标` 那列的 val 单调下降。）

### 6.2 完整训练（5000 步，同配置）

```cmd
cd <repo> && .\target\release\vmlm-b1-full.exe --sparse-io 1 --dim 4096 --k0 32 --k0-embed 128 --k0-head 128 --k0-attn 32 --layers 32 --no-residual 1 --attn-every 4 --attn-sparse 1 --attn-src-shared 1 --attn-calib 1 --attn-calib-target qk --head-skip 1 --steps 5000 --seq 64 --batch 4 --lr 3e-4 --warmup 100 --min-lr 3e-5 --seed 1 --data-dir artifacts/b1_char --vocab-mode full --chars-per-token 1.0 --out-dir artifacts/b1_C3_5000 --doc docs/B1_C3_5000.md --eval-every 100 --plateau-patience 1 --save-model artifacts/b1_C3_5000.cpk
```
（≈6.9 h。`--save-model` 便于后续用 `--load-model` 做验证路径一致性，见 `docs/B1_verify_collapse_bug.md`。）

### 6.3 备选（要 seq 128 / 要 12.9M）

* **seq 128 省算力版**：把 6.2 的 `--seq 64` → `128`、`--attn-every 4` → `8`（参数 7,848,192，≈12 h）。
* **12.9M 版（C4）**：`--k0 32`→`40`、`--k0-attn 32`→`40`，其余同 6.2（参数 12,042,496，≈8.6 h @seq64）。
* **先对照规模效应**：把 6.2 的 `--k0-embed/--k0-head/--k0-attn` 各减半并发 `--dim 2048`（C1，1.57M）跑 500 步，
  与 char 基准（0.98M）和 C3（9.95M）三点连成"参数量-效果"曲线。

## 7. 诚实边界

1. **seq 256 没有实测**（只有 32/64/128 实测 + 外推）；外推基于 T² 项主导的形态，误差可能 ±20%。
2. **5000 步的 6.9 h 只是训练循环**；加上启动标定（dim=4096 下约 30–50 s）和每 100 步一次的 eval，
   实际墙钟再加 ~10%。
3. **模型效果完全未验证**：本报告只回答"参数、时间、内存"三件事。10× 参数是否真的更好，要 6.1 的 500 步。
4. 我修的内存 bug 只影响**稀疏 attention** 路径的分配（不再白占 8.6 GB）；`--attn-sparse 0` 的行为未变，
   套件 A1/A2（稠密 attention）仍全 PASS。
5. 训练仍是**单线程**；`--batch 4` 固定。若你想用 batch 8/16 换更稳的梯度，耗时会近似线性上涨（未测）。
