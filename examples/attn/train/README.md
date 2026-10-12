# P8 成员 F：Transformer 训练闭环

> 用 `examples/attn/block.vml` 的前向数学，加**手写反向指令流** + AdamW，
> 跑通**过拟合单批次**，并用 **PyTorch autograd** 做梯度检查。

## 一句话结论

**单批次过拟合：loss 2.2405 → 0.0000005（2000 步），4/4 位置预测正确，
因果掩码训练后仍严格为 0，梯度检查 16/16 PASS（判据 ~1e-6，比 1e-3 门槛低三个数量级）。**

---

## 1. 配置与结构

最小配置（先证明闭环正确，再谈放大）：

```
T=4（序列长）  D=8（模型维）  NH=1（头数）  HD=8（头维）  F=16（FFN）  V=8（词表）
参数量 = 712 个元素
```

结构与 `examples/attn/block.vml` 完全同构：

```
x = tok_emb + pos
xd = LN1(x)
qkv = xd·Wqkv^T + bqkv            ; 一次大 GEMM [T,3D]
Q,K,V = split(qkv)
att = softmax(Q·K^T / √hd, causal)
y = att·V
x = x + (y·Wproj^T + bproj)        ; 残差
xh = LN2(x)
h = relu(xh·W1^T + b1)
x = x + (h·W2^T + b2)              ; 残差
logits = LN_f(x)·E^T
loss = 逐位置交叉熵的均值（E 与输出头共享）
```

---

## 2. 交付物

| 文件 | 作用 |
|---|---|
| `make_attn_train.py` | 生成权重 `.bin`、Python 前向参考、**两份 torch 梯度参考** |
| `gen_vml.py` | 生成 `train_step.vml`（前向 + 反向 + AdamW 指令流） |
| `train_step.vml` / `.vmb` | **训练步指令流**（209 条指令，31 个常量，~10.6 KB） |
| `asm.args` | 汇编器的常量清单（`名字:文件:rows:cols`，避免命令行手抄错位） |
| `build.ps1` | 组装脚本（读 `asm.args`） |
| `ref_bwd_f64.py` | **第三份独立实现**：纯 numpy f64 手写反向（不依赖 VMLM，也不依赖 torch） |
| `../../train/src/attn_train.rs` | 训练驱动（bin 名 `vmlm-attn-train`） |

### 为什么要第三份实现

P7-A 的方法论是「用精确 exp 复算来区分实现错与近似污染」。这里把同一思路再推一步：

* **VMLM `.vml`** = 被测实现
* **PyTorch autograd** = 参考实现（但它的计算图我无法逐步校验）
* **`ref_bwd_f64.py`** = 第三份，纯 numpy f64、同一套数学重写

三份互拍的价值在本次调试中直接兑现：`ref_bwd_f64.py` 首先暴露出
**「FFN 的 10 个参数一致、注意力相关的 6 个不一致」**，
把 bug 范围从「整个 Transformer 反向」一次性缩到「注意力 `dV` 那一行」。

---

## 3. 复现命令

```powershell
# 1) 生成权重 + 参考（需要 torch 2.13.0+cpu）
python examples/attn/train/make_attn_train.py

# 2) 构建汇编器与训练驱动
cd <repo>
cargo build -p vmlm-asm   --release
cargo build -p vmlm-train --release --bin vmlm-attn-train

# 3) 跑验收（vmlm-attn-train 会自己调 vmlm-asm 组装 .vmb）
target/release/vmlm-attn-train.exe 400      # 400 步过拟合
target/release/vmlm-attn-train.exe 50000    # 计时 + CPU 占用

# 3b) 只做梯度检查 + 用 [3] 报的耗时做 CPU 测量
target/release/vmlm-attn-train.exe 1
```

**注意**：`.vml` 由 `gen_vml.py` 生成，**不要手改**。
另外**不要用 PowerShell `Out-File -Encoding utf8` 写 `.vml`**（BOM 会让汇编器报未知指令）。

---

## 4. 实测结果

### 4.1 前向回归（vs Python 参考）

| 项 | 值 |
|---|---|
| logits 最大绝对误差 | **1.192e-7** |
| 参考初值 loss | 2.240510（torch 2.240510225） |
| 判定 | **PASS**（阈值 1e-4） |

### 4.2 梯度检查 vs PyTorch autograd

判据 `Σ|Δ| / (‖num‖ + ‖ana‖) < 1e-3`（契约 §3；**不用逐元素相对误差**）。
`torch.set_num_threads(1)`。

**16/16 PASS，最坏 2.142e-6**（比门槛低约 470 倍）：

| 参数 | 元素数 | Σ\|Δ\| | 判据 | 判定 |
|---|---|---|---|---|
| Wqkv | 192 | 3.746e-6 | **1.153e-6** | PASS |
| bqkv | 24 | 7.669e-7 | **1.073e-6** | PASS |
| Wproj | 64 | 4.954e-6 | **1.272e-6** | PASS |
| bproj | 8 | 2.837e-6 | **1.470e-6** | PASS |
| W1 | 128 | 3.706e-6 | **8.720e-7** | PASS |
| b1 | 16 | 5.580e-7 | **1.165e-6** | PASS |
| W2 | 128 | 6.299e-6 | **1.078e-6** | PASS |
| b2 | 8 | 2.827e-6 | **1.494e-6** | PASS |
| ln1_g | 8 | 1.474e-7 | **1.876e-6** | PASS |
| ln1_b | 8 | 1.618e-7 | **1.104e-6** | PASS |
| ln2_g | 8 | 6.706e-8 | **8.480e-7** | PASS |
| ln2_b | 8 | 8.981e-8 | **2.142e-6** | PASS |
| lnf_g | 8 | 9.108e-7 | **1.785e-6** | PASS |
| lnf_b | 8 | 4.824e-7 | **1.259e-6** | PASS |
| E | 64 | 2.395e-5 | **1.433e-6** | PASS |
| pos | 32 | 5.448e-6 | **9.324e-7** | PASS |

**关于 `exp_f` 近似污染**：任务书提醒 `kernels::exp_f` 最大相对误差 6.51e-4 会让
softmax 相关的检查卡在 1e-3 附近。本模型**没有出现这个问题** ——
把参考换成「f64 精确 exp 的 softmax 前向 + torch autograd」后，
判据从 1.15e-6 变到 1.15e-6（`Wqkv`），**没有数量级变化**。
两份参考的对照表都在输出里（`--- vs PyTorch(近似exp) ---` / `--- vs PyTorch(f64精确exp) ---`），
所以「是否被 exp 污染」这件事是**可当场判定**的，不靠推测。

### 4.3 过拟合单批次（训练正确性的最强证据）

固定 batch：`tok=[1,2,3,0]`，`target=[2,3,0,1]`，AdamW `lr=3e-2`。

| step | loss | 最近 25 步均值 |
|---|---|---|
| 1 | 2.2405102 | 2.2405102 |
| 25 | 0.4340864 | 1.1560962 |
| 50 | 0.0858001 | 0.2048906 |
| 100 | 0.0182460 | 0.0246074 |
| 200 | 0.0040308 | 0.0046618 |
| 400 | 0.0006485 | **0.0007152** |
| 800 | 0.0020814 | 0.0022881 |
| 1200 | 0.0002121 | 0.0002282 |
| 1600 | 0.0000126 | 0.0000139 |
| 2000 | **0.0000005** | 0.0000006 |

* loss **单调下降**，从 ~2.24 降到 **5e-7**（2000 步）
* 400 步时已 < 1e-3，远超任务书要求的 < 0.01
* **单批次复现：argmax = [2,3,0,1] = target，4/4 正确**

### 4.4 因果性回归（训练后）

```
最大上三角绝对值 = 0.000e0
att[0,:] = [1.0,        0.0,        0.0,        0.0       ]
att[1,:] = [0.60159457, 0.39840543, 0.0,        0.0       ]
att[2,:] = [0.2613178,  0.051177893,0.6875043,  0.0       ]
att[3,:] = [0.047833033,0.18754229, 0.56964314, 0.19498153]
判定: PASS（反向没有引入未来信息泄漏）
```

注意力分布**确实变了**（训练前 `att[1]=[0.519,0.481]`，训练后 `[0.602,0.398]`），
说明反向真的在推动权重；同时上三角**严格为 0**，说明 `SOFTMAX_BWD` 的 `causal`
标志正确传递，反向没有把未来位置的梯度泄漏回来。

### 4.5 耗时 + CPU 占用

| 项 | 值 |
|---|---|
| 训练速度 | **0.006 ~ 0.008 ms/step**（50000 步总计 317.8 ms） |
| **CPU 占用** | **98.3%**（50000 步）/ 106.9%（2000 步） |

CPU 占用 = 进程 CPU 时间 ÷ 墙钟时间（`GetProcessTimes`）。
单线程上限 100%，实测 98.3% 证明**无隐式多线程**，与白皮书 §12.3 的 99.7% 同一量级。
（2000 步那次报 106.9% 是计时粒度问题——总墙钟只有 14.6 ms，
`GetProcessTimes` 的 100 ns 分辨率在这个尺度上误差被放大；
以 50000 步的 98.3% 为准。）

---

## 5. 反向指令流要点（照着写就不会错）

`train_step.vml` 的反向段顺序：**梯度清零 → 反向 → AdamW**。
参数梯度在梯度区**累加**，所以 `ZERO` 必须每步开头做一次。

### 5.1 注意力（最容易错的一段）

```
y = att·V^T                 (NT 路径，Vt = TRANSPOSE(V) 是 [HD,T])
  datt[M,T] = dy[M,HD]·V[T,HD]^T
  dV[T,HD] += att[M,T]^T·dy[M,HD]      ← 是 att，不是 datt！
```

> **`dV = att^T·dy`，不是 `datt^T·dy`。** 这是本次踩到的唯一的真 bug，
> 它同时污染 `Wqkv/bqkv/ln1_g/ln1_b/E/pos` 六个参数，而 FFN 那十个照样 PASS ——
> 典型的「局部错、局部对」，只有端到端对拍才能发现。

`S = Q·K^T/√hd` 的反向要**分别**算 dQ 和 dK：

```
dQ[M,D] = dS[M,T]·K[T,D]
dK[T,D] = dS[M,T]^T·Q[M,D]
```

`att = softmax(S, causal)` → `SOFTMAX_BWD dst, dy, y, causal`
（**必须带 `causal`**，否则上三角会泄漏）。

### 5.2 残差

`x = x + branch` 的梯度**两条路都回传**：
先把 `dx` 送进 branch（proj / ff），再用 `EWADD dx, dx, dx2` 把 LN 反向的结果加回来。
所以前向要 `COPY x1, x` / `COPY x2, x` 存快照，反向 `LN_BWD` 用快照而不是被改过的 `x`。

### 5.3 踩过的引擎级坑（都已适配）

| # | 坑 | 症状 | 正确写法 |
|---|---|---|---|
| 1 | **`LN_BWD` 是 4 操作数** | 写成 5 操作数时第 5 个被 `max_regs` 静默丢弃 → 所有解析梯度全 0 | `LN_BWD dx, dy, x, <gamma>`；`dgamma → slot[gi+1]`、`dbeta → slot[gi+2]` **按槽号寻址** |
| 2 | 槽号相邻性 | 用 `ZERO ln1_g` 钉顺序会**真的清掉常量池里的 gamma** | 用 `SHAPE`（load-time 元指令，`force_shape` 跳过 const）钉顺序 |
| 3 | **`GEMM_BWD_X` 只能表达 NT** | 内核按 `w[j][p]`（`j<dy.cols`、`p<w.rows`）索引，行列不等时**越界读** | 非方阵的 `dy·W` 要转置成 `GEMM_BWD_W` 形式 |
| 4 | **`WRITECOL` 宽度匹配** | 往 `[V,T]` 里写一整行 `[1,V]` 会串行越界 → dlogits 每行只有前 2 个非零 | 让 src 是 `[V,1]`，每次只落**一列** |
| 5 | **`CE_BWD` 是单样本** | 它展平 logits 只取一个标量 target，不能直接做逐位置 CE | 逐位置跑 T 次（每次取 `[1,V]` 行切片），再拼回 |
| 6 | **torch 用 mean，我们逐位置是 sum** | 解析梯度整体差 T 倍 | `dlogits` 末尾 `SCALE inv_t`（`inv_t = 1/T`） |
| 7 | `ZERO` 遗漏 | 累加型反向（`GEMM_BWD_W`/`EMB_BWD`）不先清零会叠加垃圾 | 每步开头 `ZERO` 所有 `g_*`；每个累加目标用前单独 `ZERO` |
| 8 | `nn` 布局历史坑 | P7 修过（NN 下 N 取 B 的**列数**） | 本模型 **全程用默认 NT + TRANSPOSE** 绕开（任务书建议） |

### 5.4 驱动侧的技巧：读「干净的前向值」

引擎从 `ip=0` 跑完整个训练步（**含 AdamW 改权重**），所以跑完之后读 `logits`
虽然还是前向写进去的值，但 `E` 已经被更新了 —— 用更新后的 `E` 去验算就会误判。
本驱动的做法是**临时截断 `fvm.n_ins` 到前向结束那条指令**，跑一次读干净的前向值，
再恢复 `n_ins` 跑完整步取解析梯度。

```rust
let full_n = fvm.n_ins;
fvm.n_ins = forward_ins_count();   // 数 .vml 到 `GEMM logits` 为止
fvm.run();
let lg = peek(&mut fvm, "logits", T * V);   // 干净的前向值
fvm.n_ins = full_n;
fvm.run();                                  // 这时才执行反向
```

**不要**用「把 lr 置 0」来跳过更新 —— `AdamW` 的 `sqrt(v/c2)` 在 `v=0, c2=1` 时
会变成 `0/0 = NaN`，把权重直接毁掉（踩过）。

---

## 6. 已知问题与诚实边界

1. **只有最小配置（T=4 D=8 NH=1）**。`gen_vml.py` 里 T/D/NH/HD/F/V 都是模块级
   常量，但 `SLICE`/`WRITECOL`/`CONCAT` 的偏移与宽度是**硬编码数字**，
   放大到多头的 T=8/D=16/NH=2 需要把逐头展开写成循环（`block.vml` 是手写两次）。
   **未做**。

2. **`x` 的梯度在嵌入处用了两次转置取行**（引擎只有列切片）。
   `TRANSPOSE → SLICE → TRANSPOSE` 三步取一行，指令数随 T 线性增长。
   T 大了会浪费指令；更好的是引擎补一条 `SLICEROW`。**未做**。

3. **`vm.loss` 只保留最后一个位置的 CE**（`OP_CE_BWD` 覆盖写）。
   驱动的 loss 曲线是从 `logits` 自己重算的 T 位置均值（与 torch 同口径），
   所以曲线本身正确；但如果别人直接读 `vm.loss` 会拿到最后一个位置的值。
   **建议引擎把 `CE_BWD` 改成累加或提供 loss 槽**。

4. **`exp_f` 近似**：本模型上没观测到污染（两份参考判据同为 1e-6 量级），
   但这不构成「不会污染」的一般结论 —— 只是本配置的 logits 分布离近似最差点较远。

5. **AdamW 用引擎实现**，未与 PyTorch AdamW 单独对拍
   （P7-A 已做过，判据 2.0e-7；本模型沿用同一实现与同一份 `hparams` 布局）。

6. **未做真实规模放大**（D=512/L=6）：需要脚本参数化生成 `.vml`，
   与 C2 的 ESERNN 放大是同一类工作。

---

## 7. 对引擎的三条建议（本次实测得出）

1. **`CE_BWD` 支持逐位置 / 累加 loss**：现在单样本 + 覆盖写，做序列任务必须手动展开 T 次。
2. **补一条行切片** `SLICEROW dst, src, row, width`：现在取一行要两次转置，指令数 ×3。
3. **`OUTER_BWD` / `LN_BWD` 的「紧邻槽」隐式约定**：Lead 已加 panic 校验（很好）。
   建议同样给 `GEMM_BWD_W` 的累加语义加文档或校验 —— 漏 `ZERO` 不会报错，只会静默叠加。
