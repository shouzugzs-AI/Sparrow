# 四个消融：分离"哪部分导致发散"

> 本轮**只做冒烟**（按你的要求："先做冒烟测试，确认四项功能正常后给 CMD"、"不要自己跑"）。
> 500 步的发散数据请用 §4 的 CMD 产出。

## 1. 代码改动（最小化，三个新开关）

| 消融 | 开关 | 改动 |
|---|---|---|
| 一：去掉 attention 残差 | `--attn-no-residual 1` | 前向 `h = attn·Wo`（**替换**）；反向把 attention 输入的梯度**替换**掉层输出梯度（而不是并入恒等通道）|
| 二：attention 稀疏化 | `--attn-sparse 1 --k0-attn K` | Q/K/V/O 四个投影改用 `b1_forward`/`b1_backward_fused`（每 token 只连 K 个源）|
| 三：去掉 head→嵌入直连 | **不写** `--head-skip` | 无需改代码（默认就是只读末层）|
| 四：组合 | 上面三个一起 | — |

实现要点：
* 把 attention 的**标量核心**抽成 `attn_core`/`attn_core_bwd`，稠密与稀疏投影**共用同一段核心**（保证两条路只差投影）。
* 稀疏投影的 `src` 用确定性规则 `(i·K + j·7919 + tag·13) mod dim` 生成，`assert_src_u32_consistent` 校验。
* `--k0-attn` **必须显式给**（缺了直接 exit 2），与本项目"三条 k0 都要显式"的纪律一致。
* `--head-skip 0` **不会**覆盖已有的 `--head-skip 1`（`arg_u32` 取第一个匹配）⇒ 消融三的正确写法是**完全不写该 flag**。

**参数账目（dim=1536, layers=24, attn-every 4 ⇒ 6 个 attention 块）**：

| 配置 | attention 参数 | 总参数 |
|---|---|---|
| 基线（稠密 attention + head-skip） | 6×4×1536² = **56,623,104** | **57,966,432** |
| 消融一（同上） | 56,623,104 | 57,966,432 |
| **消融二（稀疏, k0_attn=8）** | 6×4×1536×8 = **294,912** | **1,638,240** |
| 消融三（无 head-skip） | 56,623,104 | 57,616,960 |
| **消融四（稀疏+无残差+无 head-skip）** | **294,912** | **1,288,768** |

（你估的"0.3M/块"实际是 **0.295M/全部 6 块** —— 比我原以为的更省。）

## 2. 冒烟测试结果（dim=128, attn-every 4, 8 步, seed 1）

| 用例 | exit | 参数总量 | 结果 |
|---|---|---|---|
| BASE（稠密 attention + 残差 + head-skip） | 0 | 264,172 | 未发散，指标有限 |
| ABL1 无 attention 残差 | 0 | 264,172 | 未发散 |
| ABL2 稀疏 attention（k0_attn=2） | 0 | 135,148 | 未发散 |
| ABL3 无 head-skip | 0 | 220,488 | 未发散 |
| ABL4 组合 | 0 | 135,148 | 未发散 |

**梯度对账（顺手，几乎免费）**：

| 通路 | 结果 |
|---|---|
| 稀疏 attention（消融二） | `GRADCHECK PASS: 16 探针，越界 0`；`attn_so` 探针 rel=5.0e-3 / SNR=12 ✓ |
| 无残差 attention（消融一） | `GRADCHECK PASS: 16 探针，越界 0` |
| 组合（消融四） | `GRADCHECK PASS: 12 探针，越界 0` |
| 默认路径逐位不变 | `loss_hash = 0xc9c987c15d75d1d6`（与历史相同）|

> ⚠️ **对账抓到了一个真 bug 并已修**：稀疏 attention 反向里我把第四个参数传成了 `wq.w`（**权重本身**）而不是梯度缓冲，
> 于是"梯度被写进权重、梯度缓冲恒为 0"。FD 探针立刻报出 `attn_sq` 全 0 → 定位 → 修 → 复验通过。
> 如果没有这一步，四个 500 步实验会全部跑在一个**梯度错位**的实现上。

## 3. 每项要看的三个数

| 指标 | 在哪看 |
|---|---|
| 是否发散 / 第几步 | `summary.json` 的 `diverged_at`（0 = 未发散）、stdout 的 `steps_done=... diverged=true/false` |
| 发散前的 val | `loss_curve.csv` / `metrics.jsonl` 最后几行；`summary.json` 的 `final_val`、`best_val` |
| 参数总量 | stdout 的 `params: main=... embed=... head=... attn=... total=...` |

## 4. CMD（你手动跑；每条 500 步）

共同部分（dim=1536, k0=8, k0e=32, k0h=32, 24 层, 无残差, lr=3e-4, warmup=100, **attn-every 4**）：
`--sparse-io 1 --dim 1536 --k0 8 --k0-embed 32 --k0-head 32 --layers 24 --no-residual 1 --steps 500 --seq 32 --batch 4 --lr 3e-4 --warmup 100 --min-lr 3e-5 --seed 1 --data-dir artifacts/b1_scaling --vocab-mode observed --eval-every 25 --plateau-patience 0 --attn-every 4`

**基线（会发散，对照）**
```cmd
cd <repo> && cargo build --release -p vmlm-train --bin vmlm-b1-full && .\target\release\vmlm-b1-full.exe --sparse-io 1 --dim 1536 --k0 8 --k0-embed 32 --k0-head 32 --layers 24 --no-residual 1 --steps 500 --seq 32 --batch 4 --lr 3e-4 --warmup 100 --min-lr 3e-5 --seed 1 --data-dir artifacts/b1_scaling --vocab-mode observed --out-dir artifacts/abl_base --doc docs/abl_base.md --eval-every 25 --plateau-patience 0 --attn-every 4 --head-skip 1
```
**消融一：去掉 attention 残差**
```cmd
cd <repo> && .\target\release\vmlm-b1-full.exe --sparse-io 1 --dim 1536 --k0 8 --k0-embed 32 --k0-head 32 --layers 24 --no-residual 1 --steps 500 --seq 32 --batch 4 --lr 3e-4 --warmup 100 --min-lr 3e-5 --seed 1 --data-dir artifacts/b1_scaling --vocab-mode observed --out-dir artifacts/abl1_no_attn_res --doc docs/abl1_no_attn_res.md --eval-every 25 --plateau-patience 0 --attn-every 4 --head-skip 1 --attn-no-residual 1
```
**消融二：attention 稀疏化（k0_attn=8）**
```cmd
cd <repo> && .\target\release\vmlm-b1-full.exe --sparse-io 1 --dim 1536 --k0 8 --k0-embed 32 --k0-head 32 --layers 24 --no-residual 1 --steps 500 --seq 32 --batch 4 --lr 3e-4 --warmup 100 --min-lr 3e-5 --seed 1 --data-dir artifacts/b1_scaling --vocab-mode observed --out-dir artifacts/abl2_sparse_attn --doc docs/abl2_sparse_attn.md --eval-every 25 --plateau-patience 0 --attn-every 4 --head-skip 1 --attn-sparse 1 --k0-attn 8
```
**消融三：去掉嵌入→head 跨层残差（注意：不写 `--head-skip`）**
```cmd
cd <repo> && .\target\release\vmlm-b1-full.exe --sparse-io 1 --dim 1536 --k0 8 --k0-embed 32 --k0-head 32 --layers 24 --no-residual 1 --steps 500 --seq 32 --batch 4 --lr 3e-4 --warmup 100 --min-lr 3e-5 --seed 1 --data-dir artifacts/b1_scaling --vocab-mode observed --out-dir artifacts/abl3_no_headskip --doc docs/abl3_no_headskip.md --eval-every 25 --plateau-patience 0 --attn-every 4
```
**消融四：组合（稀疏 attention + 无 attention 残差 + 无 head-skip）**
```cmd
cd <repo> && .\target\release\vmlm-b1-full.exe --sparse-io 1 --dim 1536 --k0 8 --k0-embed 32 --k0-head 32 --layers 24 --no-residual 1 --steps 500 --seq 32 --batch 4 --lr 3e-4 --warmup 100 --min-lr 3e-5 --seed 1 --data-dir artifacts/b1_scaling --vocab-mode observed --out-dir artifacts/abl4_combo --doc docs/abl4_combo.md --eval-every 25 --plateau-patience 0 --attn-every 4 --attn-sparse 1 --k0-attn 8 --attn-no-residual 1
```

完整版（5000 步）把 `--steps 500` 换成 `--steps 5000`、`--eval-every 25` 换成 `--eval-every 100` 即可。
**预算提示**：稠密 attention 在 dim=1536 下每块要 16 个 [d×d] 缓冲（权重+梯度+Adam×2）≈151 MB，
6 块 ≈ **906 MB** 常驻内存；稀疏版只有 6×4×1536×8×(同样 4 份) ≈ **4.7 MB**。

## 5. 诚实边界

1. **我没跑 500 步**（按你的"不要自己跑"）。发散是否真的发生在 64–109 步，本轮**未由我复核**——
   它是你给的对照基线数；CMD 里用 `--eval-every 25` 可以把发散点夹到 25 步精度。
2. 冒烟只有 **8 步、dim=128**：它证明的是"四个开关都能接上、都不 NaN、参数账目对"，
   **不能**证明它们在 dim=1536/长程下的稳定性。
3. 稀疏 attention 的 `src` 是**确定性均匀铺开**（不是学习的），所以消融二同时改变了
   "幅度受 k0 控制"和"连接模式"两件事——若要只怪幅度，需要另做一个"稀疏但 src=恒等"的对照。
4. 消融三的"末层秩可能重新坍缩"这个风险，本轮**没有测**（需要 dump + `b1_effrank.py`）；
   要的话在消融三的 CMD 上加 `--dump-acts artifacts/effrank/abl3` 再跑 `tools/b1_effrank.py`。
5. 本轮**没有引入 LayerNorm**（按你的纪律），所以"无归一化"这条假设本身没被验证为充分条件——
   四个消融只能说明"某一部分是否必要"。
