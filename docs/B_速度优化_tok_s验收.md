# PATH B（稀疏 B）速度优化 + tok/s 验收

> 本轮（2026-10-02）Lead 实测。**每个数字都来自本会话真跑过的命令**，并标注口径。
> 本文档同时记录三次**判据自身的缺陷**（其中一个导致 0xC0000005），因为它们比
> 结论更值得留档。

---

## 0. 一句话结论

| 问题 | 答案 |
|---|---|
| 稀疏 B 的验收现在包含 tok/s 吗？ | **包含**。新驱动 `vmlm-sparseb-tok`（判据 C1–C9 + 真循环 tok/s），并挂进 `tools/verify_all.ps1`。 |
| B1 快了没有？ | **快了 4.10x**（内核口径）/ **3.23x**（含转置的端到端口径）。1433 → **5839 / 4591 tok/s**。 |
| B2 SHARED 快了没有？ | **快了 2.49x**（152.6 → **380.6 tok/s**）。 |
| 既有稀疏 B 的判据还过吗？ | **全过**。B1：20/20；B1 can-fail：7/7；B2：40/40，唯一 FAIL 仍是**先存**的 A3 PER-ROW；转译：bit-exact 0 分叉；路径守卫 CLEAN；引擎体积 **74752 B 未变**。 |
| 代价是什么？ | 新路径要求激活是 **source-major `[d][t8]`** 布局（多一次转置，已计入 tok/s），且**只支持 8 的倍数分块**（尾部走标量）。旧 API 一行未改。 |

---

## 1. tok/s 的口径（先说清楚，避免口径漂移）

```
1 token  := 一个位置在模型**全部 24 层、每层 5 个稀疏矩阵**（wq,wk,wv,wo,ffn）上的**前向**
            = 120 次 matvec（每次 n=d=1536, k0=20）
T        := 每次测量处理的 token 数 = 128（= batch 4 × seq 32，P18 验收配置）
tok/s    := T / 秒          （QPC 中位数；单线程；**仅前向**）
ms/step  := 处理 T 个 token 的全部 120 个稀疏矩阵的墙钟时间
```

**与历史口径的区别**：本项目此前的 `ms/step(5×24×128)` 是
`单算子中位数 × 15360` 的**外推**（`b2_verify.rs:1394`）。本驱动的表里数字是
**一次真正计时的嵌套循环**，并保留一条 C8 判据把两个口径互相对账
（实测比值 0.939 ~ 1.006，同量级）。

**边界（不许淡化）**：
1. **仅前向**。训练口径（fwd+bwd）本驱动不报，也不与它混算。
2. 权重来自确定性 `init_sparse_conn_scaled`，**不是训练后的权重**；tok/s 与
   权重值无关（连接拓扑与形状固定），但**训练收敛性不在本验收范围内**。
3. 5 个矩阵槽位各自有一份激活（`xbuf[m]`），因此**转置成本按每矩阵一次计**
   （保守：真实图里 q/k/v 共享一个输入，可省 2/3）。
4. `b1_pure` / `b2_topk` **并未接入任何真实训练器**（仓库里只有验收驱动调用它们）。
   本验收证明的是**内核级**吞吐，不是端到端训练器吞吐。

---

## 2. 优化 1：B1 令牌分块（transposed activation）

### 2.1 先证明瓶颈在哪（不是猜的）

发布版 `b1_forward` = `kernels::b1_dot_row_u32_avx2`（gather + mul + 8 路归约），
实测 `vmlm-b1-prod-eps`：**0.005800 ms/op = 0.189 ns/conn**。

换算：@3.3 GHz = **0.623 cycles/连接**。而该内核每条连接恰好消耗
**1/8 次 `_mm256_i32gather_ps`**（k0=20 → 2 次整 gather + 4 条标量尾）：

```
0.623 cycles/conn x 8 conn/gather = 5.0 cycles / ymm-gather
```

Alder Lake P-core 的 `vgatherdps ymm` 吞吐正好是 ~5 cycles。
⇒ **发布版 B1 是 gather 吞吐受限**（三条独立证据：实测 ns/conn、每连接 gather
次数、与已知 gather 吞吐吻合）。**唯一出路是减少 gather 次数。**

### 2.2 手法

把「**1 条连接 × 8 个 token**」放进一个 ymm 通道：激活布局从 token-major
`[T][d]` 换成 **source-major `[d][t8]`**（`xt[s*t8 + t] = x[t*d + s]`，`t8` = 行距）。
于是取 x 从「每连接一次 gather」变成「每连接一次**连续 32B load**」：

```
发布版：  1 次 gather(8) 服务 1 个有用值/通道
令牌分块：1 次 loadu_ps(8) 服务 8 个有用值/通道
```

新增内核（`core/src/kernels.rs`，PATH: B1 段）：
* `transpose_t8_avx2` —— 8x8 分块 SIMD 转置（含 t/d 尾部与填充列清零）；
* `b1_forward_t8_avx2` —— 1 块（8 token）版本；
* `b1_forward_t8x4_avx2` —— **生产版本**：4 块（32 token）一次处理，
  摊薄每连接的「取 src / 算地址 / broadcast w」，并给出 **4 条独立累加链**
  （j 升序累加是一条串行 vaddps 依赖链，20 条串起来 ~50 cycles）；
* `b1_forward_t8_scalar` —— 逐位契约参考（对拍基准）。

### 2.3 精度：比发布版**更强**（不只是"不差"）

逐通道（= 逐 token）的运算序列与标量契约参考**逐位相同**：

* j **升序**累加；
* **首项直接赋值**（先把 j=0 的乘积放进累加器，而不是从 `0.0` 起加）——
  `0.0 + (-0.0) = +0.0` 会洗掉 `-0.0` 的位型；
* 单累加器/通道（**没有横向归约**，所以 8 路分组从不发生）；
* **不用 FMA**（`fmadd` 把乘加融成一次舍入，位型不同）。

实测（`vmlm-sparseb-tok` C2）：4 组形状（含 `n=1,k0=20,T=3`、`k0=1`、奇数 T）
**0 分叉 / 逐位比较**；`-0.0` 定向构造保住 `0x80000000`（C5）。
**发布版 SIMD 只有 45.1% 位相同**（8 路归约），所以新路径在精度上是**升级**。

### 2.4 速度（真循环，QPC 中位数，reps=20）

| 口径 | ms/step | ms/token | **tok/s** | vs 发布版 |
|---|---|---|---|---|
| W2 B1 发布版（gather + 8 路归约） | 89.94 | 0.7026 | **1423.3** | 1.00x |
| W3 分块 1 块（8 token/次） | 36.20 | 0.2828 | **3536.4** | 2.48x |
| **W4 分块 4 块（32 token/次）** | **21.92** | 0.1713 | **5838.9** | **4.10x** |
| **W5 分块 4 块 + 每矩阵一次转置** | **27.88** | 0.2178 | **4591.4** | **3.23x** |

单矩阵口径（1 个矩阵 × 128 token，同一个 x/sc，cache 全热）：

```
发布版 128 次调用 = 0.7040 ms  (0.1790 ns/conn)
令牌分块          = 0.1561 ms  (0.0397 ns/conn)   => 4.51x
```

**第二次独立运行（同一命令，reps=20）**，用于展示漂移量：

```
W2 发布版   94.55 ms/step  1353.7 tok/s
W4 分块 x4  22.73 ms/step  5632.0 tok/s   => 4.16x
W5 分块+转置 27.90 ms/step  4587.0 tok/s   => 3.39x
```

⇒ 绝对值有 ~5% 机器漂移（本机历史记录 9.2%），**比值稳定在 4.1~4.2x / 3.2~3.4x**。

转置成本：120 次/step = **5.55 ms/step**，占 W5 的 **19.9%**。
（`transpose_t8_avx2` 与按定义的标量转置逐位相同，C1，5 组形状 0 分叉。）

### 2.5 为什么不是 5~10x（如实报告）

原始估计（uop 吞吐 + 去掉 gather）给出 5~10x，实测 4.1x（内核口径）。剩余差距
来自三处，均已计入上表而不是被隐藏：j 升序累加的**串行依赖**（4 条链已摊开）、
**每连接的 src/地址/w-broadcast 固定开销**（4 块摊薄后仍占 ~1/3 uop）、
以及令牌分块后 **`xt` 的随机行访问**（每条连接换一行，L1 命中率下降）。
这三条都可以继续做（更大的块、预取、按源分组的重排），
但**按源重排会改变每个输出通道的累加顺序 ⇒ 破坏逐位契约**，属于产品决策而不是
自由优化 —— 与 `docs/B1_B2_交付报告.md` §6.6 的结论一致。

---

## 3. 优化 2：B2（稀疏激活）

### 3.1 强度行向量化（**逐位免费**）

`core/src/b2_topk.rs` 原先是**全标量**（无 AVX2、无 `src_u32`）。它的强度行
`out[j] = |w·x[src[j]]|` 恰好是既有内核 `sparse_strength_row_avx2` 的语义
（该内核用 `_mm256_andnot_ps` 清符号位，与 `f32::abs` 逐位一致）。

> 注意方向：B1 的前向乘积**不能**用这个内核（它带绝对值，实测造成 15314/30720
> 纯符号翻转，即 `b1_pure.rs` 文件头记录的 S1 回退事故）。B2 的**强度**按定义
> 带绝对值，所以在这里它是**对的**。两者不可互换，这是本仓库最贵的一次教训。

逐位论证：每个元素独立一次 `mul` + 清符号位，**无归约、无 FMA、无重排**，
k0%8 的尾部由内核用与标量版逐字相同的表达式处理 ⇒ 与标量循环逐位相同。
实测（C6）：PER-ROW 与 SHARED 两种 mode 的输出**0/1536 位分叉**；
can-fail（把强度换成保符号乘积）会让 **6/8 个槽位**改变选择。

同时新增 **融合**内核 `sparse_strength_agg_avx2`（强度 + 列聚合一次算完）：
省掉 1536 次逐行调用（`#[target_feature]` 函数不能被内联回普通调用者）
与一趟强度缓冲往返。逐槽位的累加仍是 **i 升序、每元素一次 add** ⇒ agg 逐位不变。

### 3.2 令牌分块 SHARED（新增 API `b2_forward_t8`）

B2-SHARED 的主成本是「遍历全部 d·k0 条连接算强度并聚合」，逐 token 版本对每个
token 都要为每条连接做一次 gather ⇒ 与发布版 B1 **同一个瓶颈**。
新内核 `sparse_strength_agg_t8_avx2` 用同一个手法：通道 = token，取 x 是连续
32B load。**循环顺序刻意取「行 → 槽位 → 令牌块」**：

* 反过来（令牌块外层）时，每换一个令牌块就要把整个 `xt`（786 KB @ d=1536/T=128）
  重新随机访问一遍 ⇒ 每个 (i,j) 一次 L1 miss。实测同一内核换个顺序
  **387 → 334 ms/step**。

逐位与逐 token 版本相同（C9，4 组 T 含奇数与 T=1）：输出 **0 分叉**，且
**四个计数器（macs/compares/selections/strength_ops）逐项相等**。

### 3.3 速度

| 口径 | ms/step | **tok/s** | vs 基线 | 说明 |
|---|---|---|---|---|
| 改动前 B2 SHARED（外推口径，标量强度） | 838.7 | **152.6** | 1.00x | 历史基线 |
| 改动后 B2 SHARED 逐 token（`b2_verify` 外推） | 403.6 | **317.2** | 2.08x | 强度 SIMD + 融合聚合 |
| 改动后 B2 SHARED 逐 token（真循环） | 437.1 | **292.8** | 1.92x | W6 |
| **改动后 B2 SHARED 令牌分块（真循环）** | **336.3** | **380.6** | **2.49x** | W8 |
| B2 SHARED 令牌分块 + 转置 | 346.0 | 369.9 | 2.42x | W9 |
| B2 PER-ROW（未优化选择阶段） | 5682.6 | 22.5 | 0.95x | W7，**慢，如实报** |

**第二次独立运行**：W6 288.2 / W8 375.7 / W9 363.2 tok/s（比值 1.89x / 2.46x / 2.38x），
与上表一致（绝对值 ~5% 漂移，比值稳定）。

`vmlm-b2-verify` 的口径变化（40 条判据逐条 diff，**只有测量值变、判定不变**）：

```
B1           0.0058 -> 0.0056 ms/op        A3 SHARED 选择占比 0.383% -> 0.718%（仍 <5%）
B2 SHARED    0.0546 -> 0.0263 ms/op        A5 SHARED vs PER-ROW 7.28x -> 13.25x
B2 PER-ROW   0.3975 -> 0.3485 ms/op        唯一 FAIL 仍是 A3 [PER-ROW]（结构性，先存）
```

### 3.4 B2 的边界（不许淡化）

1. **PER-ROW 的选择阶段没动**：它的比较计数被 A4 判据钉死（`st_sh.compares < 上界`
   要求"数据相关的提前退出"真的发生），改算法就会改计数器。它是**已知不成立的
   变体**（选择比它选中的算术贵 6.3x），本轮不为它改判据。
2. **令牌分块的累加阶段仍是逐 token 标量**：每个 token 的命中槽位不同，
   无法跨 token 共享 SIMD 通道；换成 gather 会破坏「升序单累加器」契约。
   实测它现在约占 B2 分块耗时的 **一半**（k=1 对照实验：155 vs 334 ms/step）。
3. **聚合阶段受 L2 延迟限制**：每个源行（512 B）被约 20 个 (i,j) 引用，
   但引用散布在 1536 行上 ⇒ 无时间局部性（实测 ≈15 cycles / L1 miss，MLP≈1）。
   按源分组重排能拿到 20x 的复用，但**会改变每个输出通道的累加顺序 ⇒ 破坏逐位**。
4. `b2_forward_t8` **只支持 `B2Mode::Shared`**（PER-ROW 传进来会 `exit(9)`，
   而不是悄悄退化成另一个语义）。

---

## 4. 验收：判据清单与结果

### 4.1 新驱动 `vmlm-sparseb-tok`（12 条判据，全 PASS）

```
.\target\release\vmlm-sparseb-tok.exe --d 1536 --k0 20 --k 8 --layers 24 --mats 5 --tokens 128 --reps 20
-> checks=12 fail=0   RESULT: ACCEPTANCE PASS（含 tok/s）
```

| 判据 | 内容 | 实测 |
|---|---|---|
| C0 | 120 个矩阵几何自洽 + `src`/`src_u32` 镜像一致 | ok（硬失败守卫） |
| C1 | 转置 `[T][d]→[d][t8]` 逐位 == 按定义的标量转置 | 5 组形状（含 d=1/t=1、非 8 倍数）**0 分叉** |
| C2 | 令牌分块内核逐位 == 标量契约参考 | 4 组形状（含 n=1/k0=1/T=3）**0 分叉**；macs == n·k0 |
| C3 | 确定性：8 次重复调用逐位相同 | 196608 槽位 0 分叉 |
| C4 | can-fail：漏项 / 降序 / FMA | 分别抓到 196160 / 150023 / 119930 |
| C5 | 「首项赋值」契约可观测 | `-0.0` → `0x80000000`；从 0 起加 → `0x00000000` |
| C6 | B2 SIMD 强度行不改变输出（两种 mode）+ 丢 abs 会被抓 | 0/1536 分叉；can-fail 改 6/8 槽位 |
| C7 | 假零守卫（计时下限 + consume） | 最小 21.9 ms ≫ 下限 1 µs |
| C8 | 真循环与「单算子×15360」外推同量级 | 比值 0.939 |
| C9 | B2 分块 == 逐 token（逐位 + 四个计数器逐项） | 4 组 T **0 分叉**、计数完全相等 |

> C4 只有 3 项要求"被抓到"：`wrong-3 从 0.0 起加` 在**随机数据上抓不到是已知
> 边界**（`0.0 + t == t` 对绝大多数 t 成立），它由 C5 的**定向构造**覆盖。
> 这条分工与 `b1_canfail.rs` 的 wrong-4 完全一致，不是为过判据而放宽。

### 4.2 既有稀疏 B 判据（全绿）

| 驱动 | 结果 |
|---|---|
| `vmlm-b1-prod-eps --d 1536 --k0 20 --layers 24` | exit 0，`fail count = 0`，ε 比值 1.229e-7（比上限低 8.1e6 倍），24 层残差 PASS |
| `vmlm-b1-verify --n 1536 --k0 20 --k 8 --reps 200` | **checks=20 fail=0** |
| 边界 sweep `512/64/16`、`32/1/1`、`1/20/8` | 三条均 **checks=20 fail=0** |
| `vmlm-b1-canfail --n 1536 --k0 20` | checks=7 fail=0 |
| `vmlm-b2-verify --n 1536 --k0 20 --k 8 --reps 200` | checks=40 fail=1，唯一 FAIL = **先存**的 A3 [PER-ROW]（结构性） |
| `vmlm-trans-verify` | exit 0，bit-exact 0 分叉 |
| `vmlm-b1-eps --reps 2000` | exit 0，无 `[FATAL]` |
| `vmlm-tests` | exit 0 |
| `tools/ci_path_guard.ps1` | 8 checks / 0 violations / **CLEAN** |
| `tools/build_all.ps1` | All builds OK；**Engine core size 74752 B 未变**（新内核全在 `train`+`conn-sparse` 门后，引擎 0 B 代价） |

`vmlm-sparseb-tok` 已挂进 `tools/verify_all.ps1`（`--reps 6`，约 15 s），
所以"稀疏 B 的要求 + tok/s"现在是**同一条验收命令**里的一部分。

---

## 5. 本轮发现并修掉的**既有缺陷**（都不是新优化引入的）

### 5.1 🔴 `b1_verify` 的 `k=1` 对照用了未初始化的 `src_u32` → 0xC0000005

回归时 `vmlm-b1-verify` 直接**访问违例**（exit `0xC0000005`），停在 A3 can-fail 之后。
根因不是新代码：`b1_verify.rs:307-310` 手工写 `sc_1.src`，但**没有同步
`src_u32` 镜像**，而 `b1_forward` 的 SIMD 路径正是**按 `src_u32` 做 gather** 的 ⇒
用 malloc 的垃圾 u32 去 gather 一个只有 4 个元素的 `xs`。
**它此前没崩，靠的是那页内存恰好是 0**（即"旧代码活下来了"不是正确性证明，
与 `conn_train.rs:193-210` 记录的同类事故一模一样）。修法：补 `sc_1.sync_src_u32()`。

### 5.2 C4 判据自身有两个缺陷（第一版）

* **读了未初始化的转置缓冲**：C4 在 `xtbuf` 转置**之前**运行，于是"契约参考"和
  4 个错误实现读的是同一块 malloc 垃圾 ⇒ 4 个 can-fail 全部报 `caught=0`
  （**判据自己坏了，看起来却像"内核全对"**）。修法：转置提前到判据之前。
* **填充列断言测的是 malloc 垃圾**：`ya` 未清零就断言 padding 为 0。修法：先清零。

### 5.3 一处 128x 单位错误（又被互检抓出来）

第一版把「发布版**单 token** 时间」和「令牌分块 **T 个 token** 时间」都除以
`d·k0·T` 再比，报出 `0.02x`（"分块慢 50 倍"）。正确口径是同为 T 个 token：
`0.7040 ms vs 0.1561 ms = 4.51x`。**这与 `docs/B1_B2_交付报告.md` §2.7 记录的两
次单位错误是同一类**：同一个量用两种口径表达时对不上，靠互检而不是重读代码发现。

---

## 6. 复现命令

```powershell
cd <repo>
cargo build --release -p vmlm-train --bins

# tok/s 速度验收（含 C1-C9 正确性判据）
.\target\release\vmlm-sparseb-tok.exe --d 1536 --k0 20 --k 8 --layers 24 --mats 5 --tokens 128 --reps 20

# 既有稀疏 B 判据
.\target\release\vmlm-b1-prod-eps.exe --d 1536 --k0 20 --layers 24
.\target\release\vmlm-b1-verify.exe   --n 1536 --k0 20 --k 8 --reps 200
.\target\release\vmlm-b1-canfail.exe  --n 1536 --k0 20
.\target\release\vmlm-b2-verify.exe   --n 1536 --k0 20 --k 8 --reps 200   # exit=1：唯一 FAIL 为先存 A3[PER-ROW]

# 工程门
powershell -NoProfile -ExecutionPolicy Bypass -File tools/ci_path_guard.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File tools/build_all.ps1
```

---

## 7. 改了哪些文件（写者唯一）

| 文件 | 改动 |
|---|---|
| `core/src/kernels.rs` | 新增 PATH: B1 段：`transpose_t8_avx2` / `transpose_t8_scalar` / `b1_forward_t8_avx2` / `b1_forward_t8x4_avx2` / `b1_forward_t8_scalar`；新增 B2 段：`sparse_strength_agg_avx2` / `sparse_strength_agg_u32_avx2` / `sparse_strength_rows_avx2` / `sparse_strength_agg_t8_avx2`。**既有内核一行未改。** |
| `core/src/b1_pure.rs` | 新增 `b1_forward_t8` / `b1_forward_t8_x1` / `b1_forward_t8_scalar_reference` / `b1_transpose_activations`。既有 `b1_forward` / `b1_backward` 一行未改。 |
| `core/src/b2_topk.rs` | 新增 `strength_row` + `b2_forward_t8`；`b2_forward`/`b2_backward` 的强度阶段改走 SIMD/融合内核（逐位等价），PER-ROW 累加里 `hit_local` 提到循环外。选择算法、计数器语义、输出契约**未改**。 |
| `train/src/sparseb_tok.rs` | **新增**验收驱动（C1–C9 + tok/s）。 |
| `train/Cargo.toml` | 追加 `[[bin]] vmlm-sparseb-tok`（仅追加）。 |
| `train/src/b1_verify.rs` | **修一处既有缺陷**：`sc_1` 补 `sync_src_u32()`（否则 SIMD gather 越界）。 |
| `tools/verify_all.ps1` | 追加一条 `Run-Check "B sparse-B speed + tok/s"`。 |

---

## 8. 未做到（不许淡化）

1. **B2-SHARED 的绝对水平仍然低**（380 tok/s vs B1 的 5839）。它的成本结构是
   「全 k0 连接算强度 + 选择 + 只累加 k 条」= 比 B1 多一遍全量扫描，加上逐 token
   标量累加。要再上一个数量级，需要动**逐位契约**（按源分组重排 / 跨 token 归约），
   那是产品决策。
2. **B2-PER-ROW 基本没动**（22.5 tok/s，比基线还略低 5%）。它的选择阶段被 A4 的
   比较计数判据锁死，本轮不为它改判据；它本来就是被证否的变体。
3. **未接入真实训练器/推理引擎**：`b1_pure`/`b2_topk` 至今只被验收驱动调用
   （真实前向走 `conn_train::sparse_forward`）。因此本 tok/s **不是端到端模型吞吐**，
   也不是训练吞吐。
4. **令牌分块路径要求 source-major 激活布局**。真实集成需要在使用点付转置
   （本报告按每矩阵一次计入），或在非稀疏算子处一并改布局 —— 未做。
5. **`n_tokens` 不是 8 的倍数时尾部走标量**（已对拍，但吞吐随尾部增大而下降）；
   本报告只测了 T=128 与 T=13/1/8 的正确性，没做 T 的吞吐扫描。
6. 机器漂移：本机 d=1536 的 ms 级数字有 ~5-9% 跨度（历史记录 9.2%），
   本报告全部用 QPC 中位数（reps=20，W7 用 2 reps 并标注）。
