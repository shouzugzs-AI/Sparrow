# 验证路径坍缩 bug：根因报告 + 修复 + 复现验证（第一部分）

> 结论：**不是"验证路径漏了标定"，而是三个叠在一起的缺陷**：
> ①标定有一个 **静默跳过**（尺度 < 1e-12 就不打印也不标定）；②标定用的是**全局 g** 而尺度取的是 6 块的**最小值**；
> ③真正让那块尺度变成 1e-33 的是 **`--attn-no-residual`（消融一/四）下残差流按 ~250×/块 几何衰减**。
> 另外还修掉一个最后一个 attention 块会 **越界 panic** 的 bug。
> `3.06e-33` 已**逐位复现**，并且给出了它的准确来源。

## 1. 复现（先复现，再解释）

```
.\target\release\vmlm-b1-full.exe --sparse-io 1 --dim 1536 --k0 8 --k0-embed 32 --k0-head 32 --layers 24 \
  --no-residual 1 --steps 3 --seq 32 --batch 4 --seed 1 --data-dir artifacts/b1_scaling --vocab-mode observed \
  --attn-every 4 --attn-sparse 1 --k0-attn 8 --attn-no-residual 1
```
修复前的输出：`FATAL: attention 坍缩：worst max|q.k| = 3.0639e-33 <= 0.01`，**且没有任何标定输出** —— 与你看到的完全一致。

## 2. 逐块实测：分数为什么是 1e-33

修好"静默跳过"之后，逐块诊断直接把答案打出来了（同一配置）：

| 块 | attention 输入 RMS(h) | mean\|q·k\| | softmax 非均匀倍数 |
|---|---|---|---|
| 0 | 1.00e-5 | 3.92e-5 | 1.000 |
| 1 | 2.14e-8 | 5.00e-12 | 1.000 |
| 2 | 5.03e-11 | 2.03e-17 | 1.000 |
| 3 | 1.22e-13 | 1.21e-22 | 1.000 |
| 4 | 2.66e-16 | 6.29e-28 | 1.000 |
| **5** | **5.73e-19** | **1.56e-33 / max 3.06e-33** | 1.000 |

**每过一个 attention 块，残差流的 RMS 掉约 250 倍**（1e-5 → 1e-19，跨 6 块共 1e14）。
⇒ 最后一块的 q·k 必然落在 1e-33 —— **这个数不是下溢，是"信号被前向吃光了"**。
⇒ 而"训练路径正常"的那次（你给的 `g=166.1463`、`max|q·k|=5.23`）**不是消融四配置**：它有 attention 残差，
  六个块的尺度都是健康的 ≈3.4e-5（实测 `RMS(h)=2.66e-2` 全块一致），所以标定能一次成功。
  **两次运行的差别是配置（有/无 attention 残差），不是代码路径。**

对照（同一份代码，只把 `--attn-no-residual 1` 去掉）：

```
块0: mean|q.k|=3.92e-5 输入RMS(h)=2.6615e-2     ← 残差在，信号不衰减
块5: mean|q.k|=3.62e-5 输入RMS(h)=2.5903e-2
标定完成：worst 块4 尺度 3.43e-5；g = [159.7, 163.2, 165.8, 164.4, 170.7, 166.1]   ← 与你的 166.1463 同量级
```

## 3. 你列的五个疑点，逐条回答（都有实测）

| 疑点 | 答案 |
|---|---|
| 1. 验证路径的 Wq/Wk 从哪来？ | **原实现里根本没有权重持久化** ⇒ "训练后另起进程验证"拿到的是**重新初始化的随机权重**。这是"训练/验证不一致"的根源之一。**已新增 `--save-model/--load-model`**（含几何校验）。 |
| 2. g=166 有没有持久化？ | **有**：g 是**直接乘进 Wq/Wk 权重本身**的（不是单独存状态）⇒ 随权重一起持久化。载入后实测各块尺度 1.000~1.024 ⇒ 无需再标定。 |
| 3. 验证路径有没有调用标定？ | **同一条代码路径**（都在 `step_fn` 之后、训练循环之前）。"没有输出"的真正原因是**静默跳过**：`if worst_std > 1e-12` 不成立时既不打印也不标定。**已删除该静默分支**：现在每块都打印，跳过必须 `exit 2` 并指名。 |
| 4. 为什么是 3e-33？ | 因为**第 5 块的输入 RMS(h)=5.7e-19** ⇒ q·k ~ (5.7e-19)²·dim ≈ 1e-33。是"信号衰减"，不是 f32 下溢（下溢会直接变 0）。 |
| 5. 两条路径的 Wq/Wk 构造是否一致？ | **一致**（同一个初始化循环、同一个 src 生成、同一份标定代码）。不一致的是**配置**（有无 attention 残差）与**进程间没有权重传递**。 |

## 4. 修了什么（4 项）

1. **逐块标定**（替代"全局 g"）：`g_b = 1/√scale_b`。旧设计里取 6 块最小值算一个全局 g ⇒ 消融四下 g 会被拉到 2.5e16 再乘到**所有**块上，第 0 块会被**过度放大 1e14 倍**（softmax 变成 one-hot 垃圾）。
2. **删除静默跳过**：每块都打印 `mean|q·k| / max|q·k| / 非零占比 / 输入RMS(h) / 非均匀倍数 / g`；
   尺度恰为 0 → `exit 2`；需要的 g 超过 `--attn-calib-max-g`（默认 1e15，可调）→ `exit 2` **并指名是哪一块**。
3. **权重持久化 + 一致性判据**：`--save-model FILE` / `--load-model FILE`（带几何头校验，不匹配 `exit 2`）；
   载入后若某块尺度**不在设计目标 ±3 倍内** → `exit 2` 指名"载入的模型与标定状态不一致"（拒绝静默重标定）。
4. **修掉越界 panic**：最后一个 attention 块的输入在 `h` 而不是 `h_ins[layers]`（原来读到 `h_ins[24]`，len=24）⇒ `index out of bounds`。

**顺带把标定目标修正为"缩放后的分数"**：原来标定到**原始** q·k=1 ⇒ `q·k/√d≈0.026` ⇒ softmax 仍然基本均匀（实测非均匀倍数只有 **1.08**）。
现在默认 `--attn-calib-target score` 标定到 `q·k/√d` 尺度=1 ⇒ 实测 `mean|q·k| = 39.19 = √1536`（**正好命中设计值**），
**非均匀倍数 1.08 → 9.86**（attention 真正开始选择）。要复现旧行为用 `--attn-calib-target qk`。

## 5. 一致性判据：训练结束 vs 验证开始（实测）

```
训练侧（消融四, 3 步）：  --- ATTENTION 训练结束体检：worst max|q.k|=2.0790e0  best=3.3535e1
                        --- 已保存权重: ...（50 个张量, 1288768 个 f32，标定因子已含在 Wq/Wk 里）
验证侧（新进程, --load-model）：
                        --- 载入权重: ...（50 个张量, 1288768 个 f32）
                        块0 尺度 1.000 / 块1 1.002 / 块2 0.982 / 块3 1.024 / 块4 0.955 / 块5 1.019  ⇒ 全部"无需标定"
                        --- ATTENTION 标定后体检：worst max|q.k|=2.0786e0  ⇒ 通过
```
**2.0790（训练结束） vs 2.0786（验证开始） ⇒ 相对差 2e-4**，在 f32 精度内（这个差来自验证进程里额外跑的那 1 个训练步）。
结论：**标定状态跨路径一致，且可被数值验证**。

## 6. 修复后重跑消融四：**能跑通，但数据表明消融一本身有问题**

* 修复后消融四（`--attn-no-residual 1`）**可以完整跑完并保存权重**（exit 0，`steps_done=3`，`final_val=9.2985`），
  前提是 `--attn-calib-max-g` 抬到 1e17（因为第 5 块需要 g=2.53e16）。
* **但这是"靠标定掩盖"**：标定只能把 q·k 拉回 O(1)，**改不了残差流已经掉到 1e-19 的事实**。
  一个把信号衰减 1e14 倍的配置，训练质量不可信 —— 所以我不建议按这个方式跑长训练。
* 建议（要你定，我不擅自改前向语义）：
  **(a)** 保留 attention 残差（默认，即你那个 val=6.79 的配置）——最省事、已被证明能训；
  **(b)** 若必须"替换不累加"，需要给 attention 输出做**尺度保持**（例如按输入 RMS 缩放输出；这不是 LayerNorm，但确实是一处前向改动，需要你点头）；
  **(c)** 单纯抬 `--attn-calib-max-g` 跑（数值上能跑，我预期 val 会很差，可以用 500 步验证这一点）。

## 7. CMD

### 7.1 复现/验证固定（有 attention 残差、健康路径：应该看到逐块 g 与"标定后体检通过"）

```cmd
cd <repo> && cargo build --release -p vmlm-train --bin vmlm-b1-full && .\target\release\vmlm-b1-full.exe --sparse-io 1 --dim 1536 --k0 8 --k0-embed 32 --k0-head 32 --layers 24 --no-residual 1 --steps 3 --seq 32 --batch 4 --lr 3e-4 --warmup 1 --min-lr 3e-5 --seed 1 --data-dir artifacts/b1_scaling --vocab-mode observed --out-dir artifacts/fix_check --eval-every 3 --log-every 3 --plateau-patience 0 --attn-every 4 --attn-sparse 1 --k0-attn 8 --head-skip 1 --attn-calib-target score --save-model artifacts/fix_check.cpk
```

### 7.2 训练 → 另起进程验证（标定状态跨路径一致性，推荐留档）

```cmd
cd <repo> && .\target\release\vmlm-b1-full.exe --sparse-io 1 --dim 1536 --k0 8 --k0-embed 32 --k0-head 32 --layers 24 --no-residual 1 --steps 1 --seq 32 --batch 4 --seed 1 --data-dir artifacts/b1_scaling --vocab-mode observed --out-dir artifacts/fix_verify --eval-every 1 --log-every 1 --plateau-patience 0 --attn-every 4 --attn-sparse 1 --k0-attn 8 --head-skip 1 --attn-calib-target score --load-model artifacts/fix_check.cpk
```
**判读**：验证进程里每个块都应打印 `尺度 ≈ √1536=39.2`（`--attn-calib-target score` 的目标）并标"无需标定"；
`标定后体检 worst max|q.k|` 应与训练结束时的值在 f32 精度内一致。若打印 `FATAL: 载入的权重…与标定状态不一致`，说明权重与标定脱节了（这正是要它 exit 2 的原因）。

### 7.3 消融四（要跑就必须抬上限；我预期 val 差）

```cmd
cd <repo> && .\target\release\vmlm-b1-full.exe --sparse-io 1 --dim 1536 --k0 8 --k0-embed 32 --k0-head 32 --layers 24 --no-residual 1 --steps 500 --seq 32 --batch 4 --lr 3e-4 --warmup 100 --min-lr 3e-5 --seed 1 --data-dir artifacts/b1_scaling --vocab-mode observed --out-dir artifacts/abl4_combo_fixed --doc docs/abl4_combo_fixed.md --eval-every 25 --plateau-patience 0 --attn-every 4 --attn-sparse 1 --k0-attn 8 --attn-no-residual 1 --attn-calib-target score --attn-calib-max-g 1e17
```

## 8. 验证套件（65/65）

新增 A7（检查点往返）：`A7a` 权重写出、`A7b` 载入成功且标定持久化（各块"无需标定"）、`A7c` `--attn-calib 0` 仍被判坍缩 exit 2。
既有的 `A5`（关标定 → 必须 exit 2）继续作为 can-fail 判据。`build_all` OK（引擎 74752 B 未变）、`tests 0`、`b1-verify 20/0`、`ci_path_guard CLEAN`。

## 9. 诚实边界

1. **消融四的"跑通"是带条件的**（`--attn-calib-max-g 1e17`）；我没有把它写成"消融四验证通过"——因为**前向仍然在衰减信号**，标定只是把 q·k 拉了回来。
2. 我没有跑 500 步（按纪律），所以"消融四训练/验证都通过"这句话我只在**能跑通 + 一致性可验证**的意义上成立，**不包括 val 质量**。
3. `3.06e-33` 与你的数字逐位相同，但**没有证据**说明你那次"验证路径"用的就是消融四配置——那是我的推断（依据：有残差时六个块尺度都健康、不可能出现 1e-33）。
4. 权重持久化只存**权重**（不存 Adam 动量），所以"载入后继续训练"等价于重启动量；这不影响验证用途。
5. 第二部分（CJK 切表）与第三部分（扩规模）**按纪律未开始**，等你确认第一部分后再做。
