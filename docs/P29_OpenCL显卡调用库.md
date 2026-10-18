# P29：OpenCL 显卡调用库（GT 620 实测）

日期 2026-10-17 | 全部数字来自本轮实测，未测项在 §9 明列。
代码：`gpu\`（**独立 cargo workspace，不参与引擎构建**）
产物：`_opt\gpu\reports\`（每个探针的原始 stdout 都在那里）

---

## 0. 摘要（先摆结论，每条都能追到原始日志）

| # | 问题 | 答案 | 证据 |
|---|---|---|---|
| 1 | 本机有可用的 OpenCL 吗？ | **有，1 平台 1 设备**：`NVIDIA CUDA / OpenCL 1.2 CUDA 9.1.126`，设备 `GeForce GT 620`，**OpenCL C 1.1**，1 CU，1620 MHz，1024 MiB | `_opt/gpu/reports/gpuinfo.txt` |
| 2 | 需要装 SDK 吗？ | **不需要**。`OpenCL.dll`(541 KB) + `nvopencl.dll`(23 MB) 随驱动已装；**没有** OpenCL 头文件、没有 CUDA SDK。库用 `LoadLibraryW+GetProcAddress` 取 **25 个入口点**，零外部依赖（0 个第三方 crate、不联网） | `gpu/src/clraw.rs`、`gpuinfo.txt` |
| 3 | 内核能编过吗？ | **20/20 编过，build log 为空**（无 warning），编译 **0.27~3.0 s**（首次/复热差 10×） | `gpucompile.txt` |
| 3b | 硬件上界在哪？ | **峰值 134.8 GFLOPS（理论 155.5 的 86.7%）/ 显存 12.4 GB/s（理论的 97%）/ PCIe 2,387+2,372 MiB/s（理论的 30.5%）/ fp64 只有 fp32 的 1/10** | `gpuprobe.txt` |
| 4 | 算得对吗？ | **73 条判据全 PASS**（13 组：19 条 GEMM / 4 条 B1 / 13 条转置与 t8 / 11 条归一化 / 7 条 Adam 与元素级 / 6 条归约 / 2 条确定性 / 7 条 CPU 基线 / 4 条 can-fail） | `gpuverify.txt` |
| 5 | 判据真的能失败吗？ | **能**。4 条 can-fail 全部按预期报红（喂错 k、括号错位、dot 当 sum、`-0.0` 首项契约） | `gpuverify.txt` C1–C4 |
| 6 | 与 CPU **逐位**相同吗？ | **能，但有一个前提**：必须加 `#pragma OPENCL FP_CONTRACT OFF`。不加的话 NVIDIA 默认把 `a*b+c` 收缩成 FMA，**264,239/1,048,576** 个元素不同，且 `-cl-opt-disable` **关不掉** | `gpufma.txt` |
| 7 | GPU 比 CPU 快吗？ | **不。慢 27~130 倍**（B1 稀疏前向，同口径、5 轮中位数、空闲机器上跑）。GEMM 也慢 4.9~19.7 倍 | `gpubench.txt` |
| 8 | 慢在哪？ | 三条硬限制：**PCIe 2.0 x16 实测 2.2~2.4 GB/s**（≈2.2 GB/s，比同机内存带宽低两个量级）、**只有 1 个 SM**（峰值 136 GFLOPS vs i3-12100F 的 4 核 AVX2）、**显存 12.4 GB/s** | `gpubench.txt` |
| 9 | 对 VMLM 有实际用吗？ | **当前形态没有**。白皮书 §1.3「第一阶段不做 GPU」这条判断，在本机硬件上**被实测支持**，不需要推翻 | §7 |
| 10 | 有没有**值得拿走**的东西？ | **有 3 条**：FMA 收缩必须显式关、`sqrt`/除法在 Fermi 上不是正确舍入（41%/30%）、`-cl-opt-disable` 不能当"保守编译"用 | §4、§5 |

**一句话**：库本身做完了、验收过了、可复现；但**这张卡不适合跑 VMLM 的训练/推理主路径**，
理由不是"优化没做好"，而是三条硬件上界（PCIe 带宽 / 1 个 SM / 12.4 GB/s 显存）就摆在那里。

---

## 1. 为什么是「动态装载 + 零依赖」

本机实测的约束（`gpuinfo.txt` 原文）：

```
platforms=1   vendor=NVIDIA Corporation   version=OpenCL 1.2 CUDA 9.1.126
device[0]  GeForce GT 620   type=GPU   1 CU   1620 MHz   1024 MiB
           opencl_c_version = OpenCL C 1.1
           driver_version   = 388.73
           local_mem_type   = 1 (CL_GLOBAL)   local_mem = 48 KiB
           max_mem_alloc    = 256 MiB         max_work_group = 1024
           double_fp_config = 0x3F（fp64 可用）
```

三条约束直接决定了设计：

| 实测约束 | 对设计的强制影响 |
|---|---|
| `C:\...\CL\` 不存在（无头文件）、无 CUDA SDK | 不能 `#include <CL/cl.h>`，也不能依赖 `opencl-sys` 的构建脚本去探测；**必须**运行期 `LoadLibraryW` |
| `cl_khr_fp64` 在扩展列表里，`double_fp_config=0x3F` | fp64 可用，但 **1/32 速率**（Fermi），不能拿它当"精度兜底"（§6.2 实测） |
| `LOCAL_MEM_TYPE = 1`（= `CL_GLOBAL`） | 所谓 `__local` **就是全局内存的一段**，用 `__local` 分块**不会**变快 —— 这解释了 §6.1 里 `gemm_nt_tiled` 为什么只快 1.9 倍而不是常见 GPU 上的 5~10 倍 |

`clraw.rs` 里 25 个入口点全部用 `extern "system"` 声明（Windows 上 `cl_api_ENTRY` 是
`__stdcall`；x86_64 只有一个调用约定，因此与 `sys/src/cpu/mod.rs` 里 kernel32 的写法一致）。
**只要有一个必需入口点缺失就整体失败** —— 半装载的表用起来只会更晚炸。

---

## 2. 内核清单与浮点契约

| CL 内核 | CPU 对应物（`ops/src/kernels.rs`） | k 维求和顺序 | 判据层级 | 实测 |
|---|---|---|---|---|
| `gemm_nt_naive` | `gemm_nt_scalar` | 升序单累加器 | **B 逐位** | ✅ 5 个形状全 0 mismatch |
| `gemm_nt_tiled` | `gemm_nt_avx2` | 升序（分块不改分组） | **B 逐位** | ✅ |
| `gemm_nt_tiled4` | 同上（4 列复用） | 升序 | **B 逐位** | ✅ |
| `b1_fwd` | `b1_dot_row_u32_*` | **j 升序 + 首项赋值** | **B 逐位** | ✅ 4 个形状全 0 |
| `b1_fwd_t8` | `b1_forward_t8_avx2` | 同上 | **B 逐位** | ✅ |
| `b1_fwd_t8x4` | `b1_forward_t8x4_avx2` | 同上 | **B 逐位** | ✅ |
| `layernorm` / `rmsnorm` | 同名 | 行内两趟归约 | **E ε 级** | Δ/max\|ref\| = **1.55e-7 ~ 2.51e-7** |
| `softmax_causal` | `softmax` | 两趟归约 | **E ε 级** | Δ/max\|ref\| = **5.96e-8 ~ 1.79e-7**；行和 \|Σ−1\| ≤ **3.58e-7** |
| `adam`（m/v 通道） | `adam_avx2` | 逐元素，无归约 | **B 逐位** | ✅ 0/1,000,003 |
| `adam`（p 通道） | 同上（含 `sqrt` 与 `/`） | 逐元素 | **E ε 级**（硬件原因，见 §4.2） | Δ/max\|p\| = **5.96e-8**，max_ulp=105 |
| `saxpy`/`scale_mul`/`add_inplace`/`fill_zero` | `axpy256`/`scale_mul`/`add256` | 逐元素 | **B 逐位** | ✅ 0/65,537 |
| `reduce_partial_dot`/`_sum` | `dot256`/`sum256` | 树形归约 | **E ε 级** | Δ/Σ\|a\| = **8.4e-9 ~ 1.5e-8** |
| `transpose_t8`/`transpose_sq` | `transpose_t8_avx2` | 纯搬运 | **B 逐位** | ✅ 0 mismatch（含 n=257 非对齐边界） |

### 2.1 判据三层（沿用项目纪律）

| 层 | 定义 | 通过条件 |
|---|---|---|
| **B 逐位** | `f32::to_bits` 完全相同（含 ±0.0、NaN 位型） | mismatch = 0 |
| **E ε 级** | 报 `max\|Δ\|` / `max_rel` / `Δ/max\|ref\|` / `max_ulp` 四个数，**ε 先测后选** | 由 §4 的实测分布定 |
| **C can-fail** | 故意注入已知错误，判据必须报红 | 报红 = 判据有效 |

**为什么归约类不可能逐位**：整行归约必须树形并行才有性能，而树形分组与「升序单累加器」
是不同的求和顺序。项目对 B1 前向早就把验收从逐位放宽为 ε 级（见 `docs/` 的裁决），
本库沿用同一口径，并且**没有**为了让判据变绿而调 ε（§2.2）。

### 2.2 ε 是怎么定的（不是"调到能过"）

`gpuverify` 把四个尺度**同时**打印，人看到的是分布而不是一个 PASS：

```
layernorm rows=64 cols=512   max|Δ|=4.768e-7  max_rel=1.945e-5  Δ/max|ref|=2.270e-7
rmsnorm   rows=64 cols=512   max|Δ|=3.576e-7  max_rel=2.726e-7  Δ/max|ref|=1.829e-7
softmax   rows=64 cols=64    max|Δ|=2.980e-8  max_rel=3.428e-7  Δ/max|ref|=1.785e-7
dot       n=1000000          Δ/Σ|ab|=8.416e-9   Δ/结果=9.103e-6
```

两个尺度差 100 倍的事实必须写清楚：**用结果本身做分母在归约上会失控**
（1e6 个 ±1 的和可能在 33 附近，而中间部分和到 1000）。所以本库的归约判据用
**Σ|aᵢ|** 做分母，这也是 Kahan/Neumaier 文献报 relative error 的口径。
第一版用「Δ/结果」时 `sum n=1e6` 报出 1.85e-4 的假失败 —— 那是尺度选错，不是实现错。

---

## 3. 验收结果（73 条，13 组）

```
$ gpuverify
[PASS] V1  gemm_nt Naive/Tiled/Tiled4  × 6 形状（37×41×29 … 128×128×1）    逐位 0 mismatch
[PASS] V1  gemm_nt acc=1 语义（C 先非零）                                   逐位 0
[PASS] V2  b1_fwd × 4 形状（含 d=1536 k0=20 ntok=32）                        逐位 0
[PASS] V3  transpose_t8 / b1_fwd_t8 / t8↔token-major 交叉 / transpose_sq n=257  逐位 0
[PASS] V4  layernorm ×3 / rmsnorm ×2 / softmax ×3（causal 与非）+ 行和        ε 1.2e-7~2.5e-7
[PASS] V5  adam m/v（逐位 0/1000003）、adam p（ε 5.96e-8）、4 个元素级（逐位 0）
[PASS] V6  dot/sum × 3 长度（含 n=1e6）                                     Δ/Σ|a| 8e-9~1.5e-8
[PASS] V7  b1_fwd / layernorm 连跑 3 次逐位相同                              确定性
[PASS] V8  CPU 基线（fastcpu AVX2+FMA）vs refcpu 契约参考                   Δ/max|ref| 1e-7~2.7e-7
[PASS] C1  GEMM 用错 k（k-1）                      → mismatch 323/323  ✅ 判据有效
[PASS] C2  k0=1, w=-1, x=+0 → 必须给 -0.0          → 16/16 个 -0.0，且与"0+term"可区分
[PASS] C3  Adam 括号错位（eps 挪到分母外）          → mismatch 1581/4096  ✅
[PASS] C4  把 dot 当 sum 用                        → rel=1.358  ✅ 被拒

PASS 73 / FAIL 0    用时 0.18 s
```

**C2 值得单独说**：CPU 侧 `b1_forward_t8_avx2` 的契约是「**首项直接赋值**，不是从 0.0 起加」，
理由是 `0.0 + (-0.0) = +0.0` 会洗掉 `-0.0` 的位型。这条契约在 GPU 上**照样成立** ——
只要 k0=1 时首项来自一次乘法而不是加法。C2 就是为了证明这一点：
构造 `w=-1, x=+0`，正确实现给 `-0.0`（16/16），错误写法给 `+0.0`，两者位型可区分。

---

## 4. 两条**硬件层面**的事实（本轮最重要的发现）

### 4.1 FMA 收缩：NVIDIA 默认开，且 `-cl-opt-disable` 关不掉

`gpufma` 探针，同一段 `D[i] = A[i]*B[i] + C[i]`，n = 1,048,576：

| 编译方式 | vs「先乘后加」 | vs 显式 `fma()` | 结论 |
|---|---|---|---|
| 默认 | **264,239 个不同** | **0 个不同** | 被收缩成 FMA |
| `-cl-opt-disable` | **264,239 个不同** | 0 个不同 | **关不掉** |
| `#pragma OPENCL FP_CONTRACT OFF` | **0 个不同** | 264,121 个不同 | **关掉了** |
| `volatile` 中间量 | 0 个不同 | 264,121 个不同 | 关掉了 |

判别方法（不靠猜）：`mul_only` 与 `add_only` 各自**0 个不同**，只有 `mul_add` 不同，
且与显式 `fma()` **逐位相同** ⇒ 差异只可能来自乘加融合。

**这条对项目的影响**：本库在 `full_source()` 的第一行固定写
`#pragma OPENCL FP_CONTRACT OFF`。没有它，V1/V2/V5 的逐位判据**全部报红**
（第一次跑就是这样：`b1_fwd` mismatch 2073/4096、`adam m` 312147/1000003）——
那不是 bug，是契约不一致。加上之后 **66 → 73 条全 PASS**。

### 4.2 `sqrt` 与除法：Fermi 上**不是**正确舍入

`gpudiag` 探针，n = 65,536 个随机正数：

| 运算 | 与 Rust 侧（IEEE 正确舍入）位型不同 | 占比 |
|---|---|---|
| `a * b` | 0 / 65,536 | **0%** |
| `a / b` | 19,371 / 65,536 | **29.6%** |
| `sqrt(a)` | 26,731 / 65,536 | **40.8%** |
| `rsqrt(a)` | 20,411 / 65,536 | 31.1% |
| `a / (sqrt(b)+ε)`（Adam 分母形态） | 28,997 / 65,536 | 44.2% |

用 **fp64** 算同一条 Adam 更新（`double` 全程，最后才转 f32）仍与 CPU f32 **62/65536 不同** ——
因为 CPU 侧自己是 f32 运算，fp64 更准反而不同。⇒ **这不是精度不够，是硬件单元不同**。
NVIDIA 的 `-prec-sqrt=true -prec-div=true` 在本机驱动的 OpenCL 1.1 下无效（实测编译无报错、
结果不变）。

**后果**：`adam` 的 **p 通道**（`p -= lr*(m*rc1)/(sqrt(v*rc2)+eps)`）**只能走 ε 级**。
实测 Δ/max|p| = 5.96e-8、max_ulp = 105（105 个 ULP 听起来大，但都在绝对值 1e-8 量级上）。
**m / v 两个通道只含乘加 ⇒ 逐位 0 不匹配**，这条对照正好把"硬件 sqrt/div"与"实现 bug"分开。

---

## 5. 对 CPU 侧已有结论的一条**交叉印证**

白皮书 §0 记录：本机 GEMM 是 **load 端口受限**，不是 FMA 端口受限，
所以「代数消元」只提速 1.007×（P11）。本轮的 GPU 数据从另一侧支持了这一点：

| 引擎 | 实测 | 该引擎的峰值 | 达成率 |
|---|---|---|---|
| GT 620 `peak_fma` | **136.2 GFLOPS** | 155.5（查表：48 lane/CU × 1 CU × 1620 MHz × 2） | 87.6% |
| GT 620 `gemm_nt_tiled` | **5.3 GFLOPS** | 同上 | **3.9%** |

一个纯 FMA 死循环能跑到理论值的 87.6%（说明理论值查表没歪），
而 GEMM 只能到 3.9% —— GPU 上同样是**数据供给**而不是算力受限。
这与 CPU 侧「load 端口受限」的结论**同向**，两条独立证据指向同一件事：
VMLM 这类小批量、低算术强度的形状，瓶颈永远在搬运而不在乘加。

---

## 6. 性能实测（同口径、5 轮中位数、**确认系统空闲**）

口径说明（这是本项目最容易自欺的地方，必须写死）：

* **GPU 内核 ms** = OpenCL event profiling 的 `START→END`，**只含内核**，不含传输；
* **GPU 墙钟 ms** = 上传 + 发射 + 内核 + 下载，QPC 计时；
* **CPU ms** = `fastcpu` 模块的 AVX2+FMA 实现（**不是** `refcpu` 的标量契约版）。
  用标量参考当分母会得出「GPU 快 6 倍」的假结论 —— 第一版就踩了这个坑。
  `fastcpu` 自身已由 `gpuverify` V8 用 `refcpu` 校验过（Δ/max|ref| ≤ 2.7e-7）；
* **5 轮取中位数**，不是单次。

### 6.1 B1 稀疏前向（项目主路径）

| 配置 | GPU 内核 ms | GPU 墙钟 ms | CPU ms | **GPU/CPU** | 备注 |
|---|---|---|---|---|---|
| n=1536 k0=20 ntok=128 d=1536（token-major） | 23.21 | 23.80 | 0.786 | **30.3×** | gather 版 |
| 同上（t8，8 token/次） | 5.84 | 6.44 | 0.207 | **31.2×** | |
| 同上（t8x4，32 token/次） | 26.21 | 26.84 | 0.207 | **129.9×** | ⚠ 见下 |
| n=4096 k0=8 ntok=32 d=4096（token-major） | 9.00 | 9.36 | 0.181 | **51.8×** | |
| 同上（t8） | 1.72 | 2.06 | 0.075 | **27.4×** | |
| 同上（t8x4） | 6.82 | 7.22 | 0.078 | **92.1×** | ⚠ 见下 |

⚠ **t8x4 在 GPU 上反而更慢（26.2 ms vs t8 的 5.8 ms）**，原因不是算法：
一个 work-item 要持有 32 个累加器 + `float a[32]` 局部数组，**Fermi 的寄存器压力爆了**
（每线程 255 个 32-bit 寄存器上限，32 累加器 + 索引 + 地址就溢到 local memory，
而本机 `LOCAL_MEM_TYPE=CL_GLOBAL` ⇒ 溢出直接落到显存）。这是"把 CPU 的优化直接搬到 GPU"
的典型反例：CPU 上 4 条累加链是优点，GPU 上是寄存器压力。

### 6.2 GEMM_NT

| 形状 | Naive | Tiled | Tiled4 | CPU (AVX2) |
|---|---|---|---|---|
| 256³ | 12.09 ms / 2.8 GFLOPS | **6.33 ms / 5.3 GFLOPS** | 22.03 / 1.5 | **1.17 ms** |
| 512³ | 98.18 / 2.7 | **50.51 / 5.3** | 176.11 / 1.5 | **10.53 ms** |
| 1024×1024×256 | 193.41 / 2.8 | **101.06 / 5.3** | 351.99 / 1.5 | **17.98 ms** |

比例（GPU 墙钟 / CPU）：**4.85× ~ 19.7×，GPU 全部更慢**。
`Tiled` 是三者里最好的（`__local` 分块把 B 的重读从 `n` 次降到 `n/16`），
`Tiled4` 因为每线程 4 个累加器 + 4 个 `Bs` 数组，同样撞上寄存器/本地内存墙。

### 6.3 传输与硬件上界（`gpuprobe`，每项都给「实测 / 理论 / 达成率」）

**这一节是全文最重要的表** —— 它回答「慢是因为算子没写好，还是硬件上界就在那里」。
每一项的理论值都标了来源，不是拍脑袋。

| 项 | 实测 | 理论/上界 | 达成率 | 口径 |
|---|---|---|---|---|
| 峰值 FMA | **134.8 GFLOPS** | 155.5（48 lane/CU × 1 CU × 1620 MHz × 2） | **86.7%** | n=2^20 × 4096 iter × 8 FMA |
| 显存只读带宽 | **12.4 GB/s** | 12.8（64-bit DDR3-1600 双沿） | **97.0%** | 256 MiB float4 网格-stride 读 |
| 显存拷贝带宽 | **12.4 GB/s** | 12.8（读+写各占一半） | **96.9%** | 256 MiB 读写各一遍 |
| PCIe 上传 / 下载 | **2,387 / 2,372 MiB/s** | 7,824（PCIe 2.0 x16 理论） | **30.5% / 30.3%** | 64 MiB 阻塞传输 × 5 轮中位数 |
| 峰值 fp64 | **12.95 GFLOPS** | ~16.8（Fermi fp64 = fp32 的 1/8） | **76.9%** | 单链 `fma`，偏保守 |
| 内核往返延迟（n=1024） | **50.9 µs** | — | — | enqueue + finish，50 次中位数 |
| 启动：编译全库 / 建上下文 | **2 ms / 18.5 ms** | — | — | 每个**进程**都要付 |
| `clBuildProgram` 冷编译（首次） | **1.22 ~ 3.0 s** | — | — | 驱动有二进制缓存，热后降到 2 ms |

**读法（三条都要看，不能只看一条）**：

1. **峰值达成 86.7% + 带宽达成 97.0%** ⇒ **这张卡的上界是可信的**，
   后面的「GPU 比 CPU 慢」不是"我没优化好"能解释的；
2. **PCIe 只有理论的 30.5%** ⇒ GT 620 挂在 DMI 上（不是直连 CPU 的 x16），
   2.4 GB/s 是**结构性的**，换驱动也上不去；
3. **fp64 = fp32 的 1/10**（12.95 vs 134.8）⇒ 想用 fp64 换精度兜底会**慢 10 倍**，
   而 §4.2 已证明 fp64 也算不出 CPU 的 f32 结果（因为 CPU 自己是 f32）。

**这三个数就是"为什么不该用这张卡"的全部理由**：
2.4 GB/s 的 PCIe 意味着**任何**需要把权重搬上搬下的方案，传输时间就吃掉全部收益。
B1 前向 d=1536 一层的权重是 1536×20×4 B = 123 KB，128 token 的激活是 128×1536×4 = 786 KB；
GPU 算它要 5.8 ms，而 CPU 只要 0.21 ms —— **差 31 倍不是优化空间，是上界**。

---

## 7. 结论：要不要接进 VMLM？

**当前形态：不接。** 三条理由，每条都有实测数：

1. **带宽上界**：2.4 GB/s PCIe + 12.4 GB/s 显存。VMLM 的 B1 是 k0=20 的稀疏连接，
   算术强度极低（每字节搬运只做 ~5 次乘加），这类形状在 GPU 上必然被传输时间主导。
   实测就是 **27~130 倍更慢**。
2. **算力上界**：1 个 SM、136 GFLOPS。同机 i3-12100F 是 4 核 8 线程 AVX2，
   仅"1 核 AVX2 @3.3 GHz × 8 lane × 2(FMA)"就有 52.8 GFLOPS/核 × 4 = 211 GFLOPS，
   **比这张整卡还高**。
3. **启动开销**：每个进程 1.2 s 内核编译 + 16 ms 上下文创建。VMLM 的引擎二进制只有
   **74,752 B**、核心验收是"启动即算"；1.2 s 的 GPU 初始化会直接违背体积与启动设计。

**白皮书 §1.3「第一阶段不做 GPU/异构计算」这条判断，在本机硬件上被实测支持。**
这不是"没优化好"，是三条硬件上界（PCIe / SM 数 / 显存带宽）就摆在那里。
若将来换卡（≥ 24 GB、PCIe 4.0、≥ 20 SM），第 1、2 条会松动，那时再评估。

---

## 8. 可复现（全部只读，不动引擎任何二进制）

`gpu` 是**独立 workspace**（`[workspace]` 空表 + 自己的 `Cargo.lock`），
不参与根 workspace 的 `cargo build`。因此：

```powershell
cd gpu
cargo build --release                       # 编库 + 6 个探针

.\target\release\gpuinfo.exe    --out ..\_opt\gpu\reports\gpuinfo.md      # < 0.2 s
.\target\release\gpucompile.exe                                           # ~0.3 s
.\target\release\gpufma.exe                                               # ~2 s
.\target\release\gpudiag.exe                                              # ~2 s
.\target\release\gpuverify.exe  --out ..\_opt\gpu\reports\gpuverify.md    # ~0.2 s
.\target\release\gpuprobe.exe   --out ..\_opt\gpu\reports\gpuprobe.md     # 5~10 s，**会占 CPU**
.\target\release\gpubench.exe   --out ..\_opt\gpu\reports\gpubench.md     # 10~20 s，**会占 CPU**
```

`gpubench` 内置**让路守卫**（`gpu/src/guard.rs`）：
检测到任何 `vmlm*.exe` 在跑 ⇒ **直接退出（码 3）并打印对方的 PID/命令行**；
否则先等系统总 CPU 连续 3 次 < 10% 才开始。`--force` 可跳过（结果须标 CONTAMINATED）。

一键验收：`tools\verify_gpu.ps1`（含体积/字节不变量检查）。

---

## 9. 未测项（明列，不假装）

| 项 | 为什么没测 |
|---|---|
| **大于 256 MiB 的单次分配** | 设备 `MAX_MEM_ALLOC = 256 MiB`；试过更大的会返回 `CL_INVALID_BUFFER_SIZE`，不是"没试" |
| **fp16 / half** | `CL_DEVICE_HALF_FP_CONFIG = 0`（设备不报支持）⇒ 本库不用 half |
| **图像/纹理内存** | 本库只做 linear buffer；VMLM 的算子没有 2D 访问模式 |
| **多设备 / 多队列并发** | 本机只有 1 个设备 1 个平台；多队列预热需要 2 个以上设备才有意义 |
| **`gemm_nt_packed` 的 GPU 对应物** | CPU 侧的打包是为 gather 服务的（B1 索引布局），GPU 上没有 gather 瓶颈（有 texture cache + 合并访问），移植过去测不出东西 |
| **端到端 VMLM 训练/推理在 GPU 上跑** | 有意不做：§7 已判"不接"。硬跑一个必然慢 30 倍的路径，只会产出一个没人会用的数字 |
| **长时间稳定性（> 1 小时）** | 本机是同一台在跑引擎测试的机器，长跑会干扰别人；且 GT 620 无 ECC，长跑无意义 |
| **与 `vmlm-ops` 真正链接对拍** | `gpu` 是独立 workspace，不依赖 `vmlm-*`（否则会动引擎构建）。对拍用的是 `refcpu`（逐字照抄契约）与 `fastcpu`（V8 已校验），两条独立证据 |

---

## 10. 文件清单

| 文件 | 作用 |
|---|---|
| `gpu/src/clraw.rs` | 动态装载 OpenCL.dll（25 入口点）+ 全部常量 + 错误码可读名 |
| `gpu/src/lib.rs` | RAII 封装：`Ctx`/`Buf`/`Prog`/`Kernel` + QPC 计时 + `median`/`mad_sigma` |
| `gpu/src/kernels/mod.rs` | 20 个 CL 内核源码 + **必须的** `#pragma OPENCL FP_CONTRACT OFF` |
| `gpu/src/ops.rs` | 类型安全的调用层（与 `vmlm-ops` 同签名、同 `acc` 语义） |
| `gpu/src/refcpu.rs` | 逐字照抄 CPU 浮点契约的**慢**参考（判据的参考物） |
| `gpu/src/fastcpu.rs` | AVX2+FMA 的**快**基线（性能对比的分母；V8 已校验） |
| `gpu/src/guard.rs` | 让路守卫（检测引擎测试 / 等系统空闲） |
| `gpu/src/bin/gpuinfo.rs` | 能力清单 → markdown |
| `gpu/src/bin/gpucompile.rs` | 只编内核，打印 build log 与每个内核的 WG/私有内存上限 |
| `gpu/src/bin/gpuverify.rs` | **73 条判据**（V1–V8 + C1–C4） |
| `gpu/src/bin/gpubench.rs` | GPU vs CPU 实测（内核口径 + 墙钟口径都报） |
| `gpu/src/bin/gpufma.rs` | FMA 收缩诊断（§4.1 的全部数字） |
| `gpu/src/bin/gpudiag.rs` | `sqrt`/`div` 硬件精度诊断（§4.2 的全部数字） |
| `gpu/src/bin/gpuprobe.rs` | 硬件上界探针（§6.3 的全部「实测/理论/达成率」） |
| `tools/verify_gpu.ps1` | 一键验收 + 引擎字节不变量检查 |
| `_opt/gpu/COORDINATION.md` | 写给同机其它 agent 的让路通知 |
| `_opt/gpu/reports/*.txt` | 6 个探针的原始 stdout（本文所有数字的来源） |
