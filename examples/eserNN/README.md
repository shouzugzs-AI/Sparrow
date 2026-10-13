# P8-C2：ESERNN 从 mini 放大到真实配置

> 承接 P7-C（成员 C）的 mini 版单步前向迁移。
> P7-C 交付：`hidden=64 / 1 层` vs 真实 ESERNN **2.384e-7**（PASS）。
> 本任务：**参数化生成器 + 放大到真实配置 `hidden=512 / 6 层`**。

> **来源与许可：** ESERNN（Existential State Engine / 存在式状态引擎）是**本项目作者
> 金施哲**的独立项目；本目录是它的 `.vml` 移植与对照产物，著作权与 VMLM 同属一人，
> 与本仓库一并按 **MPL-2.0** 发布，不涉及任何第三方许可。
> 文中 `<ESERNN 仓库路径>` 指作者本地的 ESERNN 源码目录（为脱敏起见不写入真实绝对路径）。

---

## 1. 结论速览

| 项 | 结果 |
|---|---|
| **mini 回归**（生成器 vs C 手写 `step.vml`） | numpy 参考 4 项**最大绝对差 = 0.000e+00**（逐位相同）；过引擎 **2.384e-7 PASS** |
| **放大矩阵** | **16/16 PASS**（hidden ∈ {64,128,256,512} × layers ∈ {1,2,3,6}） |
| **真实配置 `hidden=512 / 6 层`** | **1.049e-05 PASS**（判据 <1e-3，余量 95×） |
| 与真实 ESERNN 源码的交叉验证（`hidden=512/6层`） | logits **9.54e-07** / S **4.47e-08** / x_prev **5.96e-07** |
| 卡点 | **无**（原本的硬阻塞 `Ins.dt: u8` 已由 Lead 修复，见 §6） |

---

## 2. 交付物

| 文件 | 作用 |
|---|---|
| `tools/gen_eserNN.py` | **参数化 `.vml` 生成器**（本任务核心） |
| `tools/run_eserNN_scales.ps1` | 各档一键「生成→汇编→验证」，可 `-Matrix` 跑全矩阵 |
| `tools/eserNN_matrix.py` | 汇总误差矩阵 / 指令数 / 体积 / 耗时 |
| `train/src/eserNN_run.rs` | 参数化验证驱动（从 `cfg.json` 读配置，兼容 P7-C 旧目录） |
| `examples/eserNN/gen_*/` | 各档生成物（`.vml` `.vmb` `weights.args` `weights/*.bin` `ref_*`） |
| `examples/eserNN/scale_matrix.tsv` | 16 档实测汇总 |

> `gen_h512_l6/` 就是**真实配置**那一档。

---

## 3. 复现命令

```powershell
cd <repo>

# 0) 构建（注意：不能用 cargo build --release，runtime 是 no_std 要单独构建）
cargo build -p vmlm-asm   --release
cargo build -p vmlm-train --release --bin vmlm-eserNN

# 1) 一键跑全部档位（含真实配置），结果写 examples/eserNN/scale_report.tsv
powershell -ExecutionPolicy Bypass -File tools/run_eserNN_scales.ps1

# 2) 跑 hidden × layers 全矩阵（16 档）
powershell -ExecutionPolicy Bypass -File tools/run_eserNN_scales.ps1 -Matrix

# 3) 只跑真实配置
powershell -ExecutionPolicy Bypass -File tools/run_eserNN_scales.ps1 -Only h512_l6

# 4) 打印误差矩阵
python tools/eserNN_matrix.py

# 5) 手工跑单档（演示：真实配置）
python tools/gen_eserNN.py --hidden 512 --layers 6 --out-dir examples/eserNN/gen_h512_l6
cd examples/eserNN/gen_h512_l6
$a = @("step.vml","step.vmb") + (Get-Content weights.args | Where-Object {$_.Trim() -ne ""})
& <repo>/target/release/vmlm-asm.exe @a
cd <repo>
.\target\release\vmlm-eserNN.exe examples/eserNN/gen_h512_l6

# 6) 与真实 ESERNN 类做源码级交叉验证
python tools/gen_eserNN.py --cfg real --vocab 32 --T 3 `
    --out-dir examples/eserNN/_cc --crosscheck

# 7) 用真实 checkpoint 的权重生成（自动识别架构、导出 nn.Parameter 转置副本）
python tools/gen_eserNN.py --ckpt <ESERNN 仓库路径>/checkpoints/checkpoint.active.pt `
    --out-dir examples/eserNN/gen_ckpt --crosscheck
```

> ⚠ 汇编**必须先 `cd` 进输出目录**：汇编器用 `:` 切分权重参数，
> Windows 绝对路径自带冒号会被切碎（P7 踩过）。

---

## 4. 误差矩阵（vs 真实 ESERNN，判据 <1e-3）

真实比例：`n_heads=8`，`head_dim=hidden/8`，`inter=3×hidden`，`decay_hidden=32`，`vocab=32`，`T=6`。

| hidden \ layers | 1 | 2 | 3 | 6 |
|---|---|---|---|---|
| **64**  | 4.768e-07 | 2.682e-07 | 2.682e-07 | 4.768e-07 |
| **128** | 9.537e-07 | 7.153e-07 | 1.431e-06 | 7.153e-07 |
| **256** | 1.907e-06 | 1.431e-06 | 3.338e-06 | 1.431e-06 |
| **512** | 5.722e-06 | 3.815e-06 | 6.676e-06 | **1.049e-05** ← 真实配置 |

**16/16 PASS**，最大误差 **1.049e-05**，是判据的 1/95。
误差随规模温和增长（float32 累积），全部在 float32 精度量级。

### 4.1 真实配置 `hidden=512 / 6 层` 明细

```
逐 step logits 误差（vs 真实 ESERNN, eps=1e-6）：1.6e-6 ~ 1.05e-5
递归态 st_S 每层：1.3e-7 ~ 1.8e-7（6 层全部）
st_xprev 每层：3.6e-7 ~ 9.5e-7（6 层全部）
综合（主判据）：1.049e-05  ⇒ PASS
对照（不发 LN_EPS，引擎默认 eps=1e-5）：3.650e-02
```

6 层的递归态全部在 **1e-7** 量级 —— 说明**逐层展开与状态推进语义正确**，
误差只来自 float32 末端的 logits 运算累积。

---

## 5. 指令数 / 体积 / 耗时

| hidden | layers | 指令数 | 槽位 | `.vmb` | 汇编 | 装载 | 前向 |
|---|---|---|---|---|---|---|---|
| 64  | 1 | 161 | 132 | 0.21 MB | 0.4 s | 0.2 ms | 0.03 ms/token |
| 64  | 6 | 931 | 742 | 1.19 MB | 0.6 s | 0.8 ms | 0.11 ms/token |
| 256 | 3 | 469 | 376 | 7.98 MB | 1.9 s | 6.3 ms | 1.15 ms/token |
| **512** | **6** | **931** | **742** | **61.78 MB** | **23.5 s** | **45.6 ms** | **8.68 ms/token** |

**指令数只随层数线性增长**（`161 + 154×(layers-1)`），**与 hidden 无关** ——
这正是参数化生成的意义：逐 head 循环是**编译期展开**，不占用指令数。

### 5.1 体积为什么这么大（诚实说明）

`.vmb` 61.78 MB 里 **99.5% 是权重常量池**（`hidden=512/6层` 真实参数量约 15M → 60 MB f32）。
指令文本只有 931×16 = **14.9 KB**。

**其中约一半是可省的转置副本**：`tm_w1p/w1pT/w2p/w2pT` 各存一份
（`w1p/w2p` 是 `nn.Parameter` 的权威布局，`w1pT/w2pT` 是给默认 NT GEMM 的转置副本）。
`hidden=512` 时每层 `w1` 是 `[512,32]`、`w2` 是 `[32,512]`，6 层合计约 0.4 MB，
**占比很小** —— 真正的大头是 `tm_Wr/Wk/Wv/Wo`（4×512²）与 FFN 的
`cm_Wkey[1536,512]` / `cm_Wval[512,1536]`，这些**没有副本、省不掉**。

⇒ 结论：体积**合理**，没有值得做的优化。（生成器仍导出 `w1p/w2p` 是因为
`--crosscheck` 要把权威布局灌回真实 ESERNN 类做验证；若只为跑引擎可以去掉，
但省下的量可忽略，且会让 checkpoint 导出路径无法自证。）

### 5.2 汇编耗时

`hidden=512/6层` 汇编 **23.5 s**（61.78 MB 常量池的 I/O + 排布）。
比 mini（0.07 s）慢 300×，但**是一次性离线成本**，且**装载只要 45.6 ms**。
若要反复迭代建议只重编改动的那一档。

---

## 6. 放大过程中的真正阻塞（已解决，记录在案）

### 6.1 症状

mini（1 层）与 2 层全对，**3 层及以上必错**（`hidden=256` 时 step0 就错 **6.5e-2**）。

### 6.2 定位过程（可复现）

| 实验 | 结果 | 推论 |
|---|---|---|
| 1 / 2 / 3 层的 numpy 参考 vs 真实 ESERNN | 全 **1e-7** | **参考正确**，问题在 `.vml` 或引擎 |
| 3 层 T=1（零初始状态）step0 | 仍错 6.5e-2 | 与跨 step 状态累积**无关** |
| 3 层 hidden=32/48/64（197 槽） | **全 PASS** | 与 hidden/层数**无关** |
| 3 层 hidden=128（257 槽） | **PASS** | 与「槽位总数 > 256」也**不**直接相关 |
| 在 3 层 `.vml` 里插一条 `COPY` | 误差 6.5e-2 → 5.1e-3 | **对具体槽位排布敏感** |

最后一条是决定性指纹：插一条指令只会改变**槽位号分配**。

### 6.3 根因：`Ins.dt: u8` 静默截断

`LN dst, src, gamma, bias` 是 4 操作数，放不进 `dst/a/b` 三个 u32，
于是汇编器把第 4 个（**bias 槽号**）编进 16 字节指令的 `dt` 字段：

```rust
// asm/src/main.rs（修复前）
dt = self.slot(regs[3]) as u8;     // Rust 的 `as` 静默回绕，无边界检查
```

3 层 `hidden=256` 有 **376 个槽**，实测被截断的正是这三条：

| 指令 | bias | 槽号 | `dt(u8)` | 实际指向 |
|---|---|---|---|---|
| 行 867 | `l2_tmln_b` | 358 | 102 | `L0_oh1` ✗ |
| 行 872 | `l2_ln2_b` | 364 | 108 | `L0_oh7` ✗ |
| 行 888 | `lnout_b` | 375 | 119 | `L0_tmo` ✗ |

2 层（254 槽）**0 条越界** → PASS。**层数无关，只跟「有没有 LN 的 bias 槽号 ≥ 256」有关。**

### 6.4 修复（Lead 在 `core/` + `asm/` 完成）

```rust
// 现在：仍 16 字节，无填充（实测 size_of::<Ins>() == 16）
pub struct Ins { op: u8, fl: u8, dt: u16, dst: u32, a: u32, b: u32 }
```

`fl` 只用 6 bit，从 `u16` 收到 `u8` 正好腾出空间；槽位上限 255 → **65535**；
三处 `as u8` 改为带检查的 u32→u16 转换，**超界直接 panic**（不再静默）。

> 落地时踩到字段顺序坑：第一版 `op:u8 | dt:u16 | fl:u8 | ...` 因对齐插入
> 1 字节 padding，结构变 20 字节，破坏定长指令（成员 F 的模型立刻坏）。
> 正确顺序是 `op | fl | dt:u16 | ...`。

**修复后本生成器无需任何槽位顺序规避**，用自然的槽位分配即可。

### 6.5 P7-C 手写版里一个**潜伏**的语义错误（已在本生成器修正）

P7-C 的 mini `step.vml` 开头有 `COPY st_xprev, x`，即把 `x_prev` 初始化成**本步输入 x**。
源码 `forward_one` 里 `prev_x = state[i][0]` 是**上一 step 该层自己的块输出**，
初值为 0 —— **不是**当前 `x`。

**为什么 mini 下没暴露**：mini 的 `tm_xr/xk/xv/xw` 与 `cm_xk` 全是 zeros，于是

```
r/k/v 的输入 = x1 + xx*0      = x1     ← xx 无影响
decay 的输入 = x1 + xx*xw     = x1     ← xx 无影响
FFN 的输入   = x3 + (xp-x3)*0 = x3     ← xx 无影响
```

`x_prev` 在 mini 下**完全不影响输出**，那条 `COPY` 是**死代码**。
实测：mini 下两种语义的 logits 差 **0.000e+00**（逐位相同），所以 P7-C 不可能发现。

**放大后权重非零，它立刻变成真 bug。** 本生成器的处理：
- **不做**开头的 `x_prev` 锁存
- 每层末尾 `COPY st_xprev{l}, L{l}_out` 把该层块输出写回自己的状态槽

修正后 mini 仍与 P7-C 逐位一致（因为该路径在 mini 下确实是死代码）。

---

## 7. 生成器设计要点

### 7.1 参数化

```
--hidden / --heads / --head-dim / --inter / --decay-hidden / --layers / --vocab / --eps / --T
--cfg mini|real          # 预设
--ckpt <path.pt>         # 用真实 checkpoint 权重（自动识别架构）
--crosscheck             # 与真实 ESERNN 类做源码级交叉验证
```

**`SLICE` 的偏移/宽度全部按 `head_dim` 参数化**（P7-C README §9.4 指出的核心难点）：

```python
for h in range(NC):
    E(f"SLICE   {p}S{h}, {ss}, {h*D}, {D}")      # 偏移 h*D，宽度 D
    E(f"WRITECOL {ss}, {p}S{h}, {h*D}")
```

逐头、逐层全部**编译期展开**，所以**指令数只随层数增长**。

### 7.2 单层命名对齐（让回归有意义）

单层时槽位/权重**不带层前缀**（`st_S` / `ln1_g`），多层才加 `L{l}_` / `l{l}_`。
这样 `--cfg mini` 的产物与 P7-C 手写 `step.vml` **逐条同构**，
指令级 diff 与数值 diff 都能直接做。

### 7.3 输入序列与层数解耦

`TOKENS` 用**独立 rng**（`seed + 99991`），不消耗权重序列。
否则不同层数会拿到不同输入，跨档对比失去意义
（我一度因此把「输入不同」误判成「层间 bug」，浪费了不少时间）。

### 7.4 权重导出（含 `nn.Parameter` 转置副本）

`TimeMixing.w1/w2` 是 **`nn.Parameter`**，布局与 `nn.Linear` 的 `[out,in]` **相反**：
`w1=[H,DH]`、`w2=[DH,H]`。VMLM 的默认 NT GEMM 需要 `[N,K]`，所以额外导出
`w1pT=[DH,H]`、`w2pT=[H,DH]` 转置副本：

```
numpy: tanh(g4 @ w1p)          # w1p 是 [H,DH]
VMLM : GEMM(gd, g4, w1pT)      # w1pT 是 [DH,H] ⇒ g4·w1pT^T = g4·w1p ✓
```

`--ckpt` 路径会自动给 `w0` 补 `[H,1]` 列向量布局、给缺失的
`channel_mixing.ln` 补恒等 gamma（旧 checkpoint 没有该参数）。

### 7.5 沿用 P7-C 的实测约定（不重新发明）

- **`GEMM` 的 `nn` 标志一律不用**（README §6 实测不可用）；需要 `A·B` 时先 `TRANSPOSE`
- **`LN_EPS` 必须发第一条**（引擎默认 1e-5，ESERNN 是 1e-6）
- **输入槽 `x` / `logits` 必须显式 `SHAPE`**
- **`M=1` 的 GEMM 链条显式声明形状**

---

## 8. 诚实边界

- ✅ 已证明：**`hidden=512 / 6 层` 与真实 ESERNN 一致，综合 1.049e-05**（判据 <1e-3）
- ✅ 已证明：16 档 `hidden × layers` 矩阵**全部 PASS**
- ✅ 已证明：生成器的 mini 产物与 P7-C 手写版**逐位相同**（参考差 0.000e+00）
- ✅ 已证明：`hidden=512/6层` 下 numpy 参考 == 真实 ESERNN 类（logits 9.54e-07）
- ⚠ `--ckpt` 路径已用**真实 checkpoint 的架构与权重**验证过（`hidden=256/3层`，
  与真实 ESERNN 类差 4.8e-7），但该 checkpoint 里**没有 `embed_tokens.weight`**
  （真实加载路径由 `ese_convert` 覆盖），所以用真实 checkpoint 时需外部提供嵌入矩阵
- ⚠ `vocab` 用的是 `3379`（P7-C README §9 的口径）；真实源码默认 `16384`。
  vocab **只影响 `E` 的形状与 `logits` 宽度**，不改变本次验证的结论
- ⚠ §5.2 的汇编 23.5 s 是**单次离线成本**；若要长期迭代，建议按档缓存 `.vmb`
- ⚠ 本迁移**只覆盖前向**，不含反向/训练（P8 训练链路是 A2/F 的范围）
- ⚠ 误差随规模增长（1e-7 → 1e-5）是 float32 累积的预期行为；
  **未**在 `hidden=1024` 等更大配置上验证过

---

## 9. 改动文件清单

| 文件 | 状态 |
|---|---|
| `tools/gen_eserNN.py` | 新建 |
| `tools/run_eserNN_scales.ps1` | 新建 |
| `tools/eserNN_matrix.py` | 新建 |
| `train/src/eserNN_run.rs` | 重写（参数化 + 兼容 P7-C 旧目录） |
| `examples/eserNN/README.md` | 本文件（新建；P7-C 的说明见 git 历史/上一版） |
| `examples/eserNN/gen_*/` | 自动生成 |

**未改动**：`core/` 下任何文件、`asm/src/main.rs`、`train/Cargo.toml`、
其他成员目录（`rnn/`、`attn/`、`sparse/`、`mnist/`）。
