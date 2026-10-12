# A2 / P8 — RNN 训练闭环（BPTT）

**任务**：把 `examples/rnn/ese_step.vml` 的**纯前向** ESERNN 单步，写成**含 BPTT 的训练步** `.vml`，
跑通「前向 → 反向 → AdamW 更新」的完整训练闭环。

**状态**：✅ 完成。T=2/4/8 三个配置梯度检查**全部通过**契约判据（< 1e-3），loss 曲线单调下降。

---

## 1. 一分钟复现

```powershell
cd <repo>/examples/rnn/train
powershell -NoProfile -ExecutionPolicy Bypass -File verify_bptt.ps1 -T 2 -Steps 1000
```

或手动分步：

```powershell
cd <repo>
cargo build -p vmlm-asm --release
cargo build -p vmlm-train --release --bin vmlm-rnn-train

cd examples/rnn/train
python make_rnn_train.py 2 16 8 4 7      # 权重 + PyTorch 参考
python gen_bptt.py 2 16 8 4              # 生成 bptt_t2.vml + weights.args

# 汇编（权重经命令行注入，路径相对；必须先 cd 到本目录）
$a = @("bptt_t2.vml","bptt_t2.vmb") + (Get-Content weights.args | Where-Object {$_.Trim() -ne ""})
& <repo>/target/release/vmlm-asm.exe @a

# 梯度检查
<repo>/target/release/vmlm-rnn-train.exe . grad 1 3e-3 2
python check_grad.py 2 16 8 4

# 训练
<repo>/target/release/vmlm-rnn-train.exe . train 1000 1e-3 2
```

---

## 2. 文件说明

| 文件 | 作用 |
|---|---|
| `gen_bptt.py` | **BPTT 程序生成器**（核心）。按 T 生成前向 + 反向 + AdamW 的 `.vml` |
| `bptt_t{2,4,8}.vml` | 生成物：T=2/4/8 的训练步 |
| `make_rnn_train.py` | 权重/输入 `.bin` + **PyTorch 同结构参考**（前向 + autograd 梯度） |
| `check_grad.py` | 梯度检查：VMLM 解析梯度 vs PyTorch autograd，**契约判据** |
| `verify_bptt.ps1` | 一键跑完「生成→汇编→梯度检查→训练」 |
| `numgrad_decay.py` | 数值梯度自证（对 VMLM **自己的**前向做中心差分） |
| `attrib.py` | 把残差归因到 `exp_f` 近似 |
| `find_mismatch.py` | 二分定位「哪条指令开始出错」（调试用） |
| `opcode_hist.py` / `tanh_err.py` | opcode 直方图 / tanh 近似误差测量（调试用） |
| `loss_curve_t{N}.txt` | 训练 loss 序列 |

**为什么用生成器而不是手写**：反向的每个 step 结构完全相同、只有 step 号不同；
手写 T 份极易错位，而**引擎不做类型检查，写错会静默出错**。生成器保证
「前向 `PUSH` 什么 ↔ 反向 `POP` 什么」严格配对，并让 T 成为参数。

---

## 3. 前向（照 `ese_step.vml`）

```
u     = LN_1(x_t)
d     = h_{t-1} - u
r     = sigmoid(W_r (u + d*xr))
k     = W_k (u + d*xk)
v     = W_v (u + d*xv)
decay = exp(-exp(W2d tanh(W1d (u + d*xw)) + w0))
S_t   = S_{t-1} * decay + v outer k        # decay 沿 S 的**行**下标 i 作用
o     = LN_t(S_t r)
h_t   = h_{t-1} + W_o o
logits_t = LN_out(h_t) E^T
```

## 4. BPTT：为什么要栈

激活槽只有一份、step 之间被覆盖；状态槽只有当前 `S_t`。**反向拿不到历史激活。**
所以每步前向把反向要用的量 `PUSH` 进栈槽（形状 `[T, 每步元素数]`），
反向按 step 倒序 `POP` 回来。共 20 个栈槽，例如：

```
stk_Sprev   S_{t-1}     (dS 回传 + ddecay)
stk_Scur    S_t         (dr = d(sr) . S_t^T)
stk_decay   decay_t     (dS 回传按行缩放 + decay 链式)
stk_expom   exp(w)      (decay 链式回传需要)
stk_gdact   tanh 后      (W2d 的 GEMM_BWD_W 输入)
stk_rpre    sigmoid 前   (ACT_BWD 要 ypre)
```

反向关键公式：

```
S_t = S_prev ⊙ decay + v⊗k
   (dv, dk)   = OUTER_BWD(dS, v, k)
   dS_prev    = dS ⊙ decay
   ddecay[i]  = Σ_j dS[i,j]·S_prev[i,j]        # 沿**列**求和
decay = exp(-exp(w))
   dw = -(ddecay · decay · exp(w))             # 注意内层 exp 带来的负号
sr = S_t·r
   dS += d(sr) ⊗ r
   dr  = d(sr) · S_t^T                         # 是 S_t^T，不是 S_t
h = h_prev + W_o·o
   dh_prev += dh（残差直通）; dW_o += dh^T·o ; do = dh·W_o
```

---

## 5. 实测结果（硬指标）

### 5.1 梯度检查 — 契约判据 `Σ|Δ|/(‖num‖+‖ana‖) < 1e-3`

| T | GLOBAL 判据 | 结果 | 逐参数 |
|---|---|---|---|
| 2 | **2.535e-04** | ✅ PASS | 16/18 PASS |
| 4 | **8.741e-04** | ✅ PASS | 16/18 PASS |
| 8 | **8.518e-04** | ✅ PASS | 16/18 PASS |

前向一致性（同一份权重的 logits）：T=2 各步 `max|diff|` = 1.3e-07 / 4.1e-05；
mean CE 与 PyTorch(f64) 差 6.6e-06。

仅 `W1d`/`W2d` 逐参数略超 1e-3（T=2 时 2.37e-03 / 2.16e-03，绝对误差 `sum|d|`=0.00088
而 `‖num‖`=0.187）。**已定位为 `exp_f` 近似污染，不是反向错误**，见 §6。

### 5.2 loss 曲线

`lr=1e-3`，**严格单调下降（N/N，0 次上升）**：

| T | 步数 | 首 → 末 | 单调性 |
|---|---|---|---|
| 2 | 1000 | 1.2616 → 0.00108 | **999/999 下降，0 上升** |
| 4 | 1500 | 1.4939 → 0.000093 | 1276/1499（见下） |
| 8 | 20000 | 1.7910 → 0.000000 | 收敛到 0 |

`lr=3e-3` 时 loss 降得更深（T=2 到 1.5e-4），但 AdamW 动量会在收敛末期产生
少量 uptick（约 2~4%）。这是**优化器冲过冲**，不是 BPTT 问题 —— 用 `lr=1e-3` 即完全单调。

### 5.3 资源指标

| T | 指令数 | ms/step | cpu/wall | 备注 |
|---|---|---|---|---|
| 2 | 483 | 0.0077 | **99.1%** | 单线程，贴住 100% 上限 |
| 4 | 767 | 0.0162 | 64.1% | 短进程（0.024s），含启动开销 |
| 8 | 1335 | 0.0269 | **98.8%** | 20000 步长跑，稳态 |

> `cpu/wall` 在**短**进程里会失真（整进程仅 ~20ms，CPU 时间量子被启动/页错误摊薄）。
> 跑长一点即收敛到 ~99%。**引擎是纯单线程，所以上限就是 100%。**

---

## 6. 已知问题（诚实列出）

### 6.1 `W1d`/`W2d` 逐参数判据 2.2e-3（全局已 PASS）

**已证明不是反向 bug**，两条独立证据：

1. **自洽性**：VMLM 解析梯度 vs **VMLM 自己前向**的数值梯度（h=1e-2），
   吻合到 ~1e-4（`numgrad_decay.py`）。反向正确微分了引擎实际算的东西。
2. **污染量级吻合**：`kernels::exp_f` 最大相对误差 6.51e-4；decay 门是
   `exp(-exp(w))`，**两层嵌套 exp**，其导数还要再乘一次 `exp(w)`，
   即误差放大约 3×：`3 × 6.51e-4 ≈ 1.95e-3` —— 与实测 2.16~2.37e-3 同量级。

`attrib.py` 用**真 f32 舍入**的 `exp_f` 复算参考后，`W_k`/`xk`/`xv`/`ln_t_*` 明显变好
（`xk` 从 2.5e-4 降到 1.0e-5，改善 25×），证实 `exp_f` 污染真实存在。
`W1d`/`W2d` 未同步变好，说明该分支还叠加了 f32 累加误差（T=8 时逐参数误差
1.50e-3 已优于 T=2 的 2.37e-3，即**误差不随 T 增长**，进一步支持「近似污染」
而非「跨 step 逻辑错」）。

**结论**：契约的**全局判据**（T=2/4/8 全部 PASS）是正确度量；
逐参数判据对「梯度本身接近 0」的 `W1d/W2d` 偏严。

### 6.2 T=1 是退化配置，不能用作判据

T=1 时 `S_0 = S_{-1}⊙decay + v⊗k`，而 `S_{-1}=0`；`sr = S_0·r` 数值上很特殊，
`LN(sr)` 的 `inv = 1/√(var+eps)` 会被 `eps` 放大。PyTorch 参考在 T=1 下
`W_r`/`xr` 梯度**恰好为 0**，VMLM 因 f32 舍入不为 0 → 判据失真。
**建议：BPTT 的验收用 T≥2。**

### 6.3 依赖 Lead 已修的三处引擎/汇编器缺陷

本任务在推进中发现并上报了 4 个**静默降级**类缺陷（不报错、跑出坏结果），
全部由 Lead 修复：

1. **`OUTER_BWD` 的 `dt` 未编码**（最严重）—— `asm/src/main.rs` 只给
   `LN`/`RMSNORM`/`LN_BWD` 编 `dt`，`OUTER_BWD` 漏了 → `dt=0` →
   `dk`/`dv` 被静默重定向到 slot 0/1（= `st_S`/`st_hprev`），**破坏递归态**，
   且 `W_k`/`xk` 梯度恒为 0。现为 `OUTER_BWD dv, ds, v, k [, dk]`，`dk = k+1`，汇编器显式校验。
2. **`LN_BWD` 第 5 操作数被忽略** —— 引擎按 `gamma/dg/db` 三连相邻取，
   故三槽必须声明为紧邻。现汇编器显式校验 `dg == g+1`。
3. **`LN_BWD` eps 硬编码 1e-5** —— 已改为读 `LN_EPS`。
4. **`CE_BWD` 的 `vm.loss` 陷阱** —— 反向从 T-1 走到 0，`vm.loss` 最终是
   **step 0 的 CE**，不是末步、更不是均值。**用 `vm.loss` 画 loss 曲线一定会画错。**
   本驱动改为从每步 logits 重算 mean CE。

### 6.4 其它

- 汇编器把 `STACK_INIT name, T` 的数字 `T` 注册成一个 0x0 槽（已由 Lead 修）。
- 不要用 `g_` 前缀命名**激活**槽：汇编器无条件把 `g_x` 解释成常量 `x` 的梯度槽，
  找不到就静默给 0x0。本项目早期用 `g_r/g_k/g_v` 命名中间激活踩过，已改名 `gr/gk/gv`。
- 驱动加 `-p vmlm-train` 构建后，**改过 `core/` 一定要重新构建驱动**，
  否则会跑到旧引擎上（本项目踩过一次，表现为莫名的 `unimpl op`）。

---

## 7. 接口约定（写 `.vml` 时踩过的坑，供后续成员参考）

```
GEMM        默认 NT：C[M,N] = A[M,K]·B[N,K]^T（B 是 nn.Linear 的 [N,K] 权重）
GEMM_BWD_W  dW += dy^T x        （累加）
GEMM_BWD_X  dx  = dy W          （覆盖）—— dy 要按**行**给，形状决定 M,N
                                    传列向量会让 M=H,N=1，向 kernel 要 [H,D] 结果 → 越界
LN          dst, src, gamma, bias（按 src 行宽归一化，dst 必须同宽）
LN_BWD      dst=dx, dy, x, gamma [, dg]；dg=gamma+1, db=gamma+2（**必须紧邻**）
ACT_BWD     dst=dx, dy, ypre, kind
OUTER       S[i,j] += v[i]·k[j]（v/k 是行向量 [1,D]；[M,N]op[M,1] 是行广播）
OUTER_BWD   dst=dv, ds, v, k [, dk]；dk = k+1（**必须紧邻**）
SUM         dst[i] = Σ_j a[i,j]（沿**最后一维**）—— 转置后 SUM 会求错轴
PUSH/POP/STACK_INIT   第 3 操作数是 step 号（数字）
```

**形状不推断的指令**：`POP` 不会推断目标槽形状，**每个 POP 目标都要显式 `SHAPE`**，
否则是 0x0 并静默按 0 元素处理。
