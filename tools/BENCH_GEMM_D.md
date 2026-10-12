# 成员 D（性能 / GEMM 内核）实测数据与失败尝试记录

> 生成方式：所有数字都来自本目录下的脚本，可复现。
> 机器：i3-12100F（4 P-core，无 E-core，AVX2+FMA，无 AVX-512），Windows。

## 0. 统计口径（先说清楚，否则数字没意义）

这台机器的**单次采样完全不可信**：同一个内核同一个 shape，
`max/min` 能差到 3~4 倍（后台调度 + 睿频）。所以：

* 每个 shape 至少 **5 个独立进程轮**，每轮 200~400 次迭代。
* 报告 **min**（噪声只会让耗时变长，min 最接近真实速度）与 **median** 两个值。
* PyTorch 基线同样多轮采样。**实测 PyTorch 的 min 非常稳**
  （5 轮 2085.2 / 2085.4 / 2085.7 / 2085.8 / 2085.9 µs，离散 0.0%），
  所以「相对倍率」这个指标是可信的。
  注意：早先单次测到的 111.95 GFLOPS 是**欠采样离群值**，不是真实基线。

## 1. 结论先行

| 指标 | 优化前 | 优化后 | 变化 |
|---|---|---|---|
| 内核级 GEMM 512³（min，5 轮） | ~2770 µs / **96.5~98.0 GFLOPS** | 2598.8~2620.9 µs / **102.4~103.3 GFLOPS** | **+3.5~6%** |
| 端到端 `vmlm-runtime bench_gemm.vmb` | ~2800~2950 µs | **2878 µs**（5 轮 best 2700） | 基本持平 |
| PyTorch 单核 512³（min，5 轮） | — | **2085~2146 µs / 125~129 GFLOPS** | 基线 |
| 相对 PyTorch（**内核级 min** 口径） | ~0.75x | **~0.82~0.83x** | 未达 0.90x |
| 相对 PyTorch（**端到端** 口径） | ~0.70x | **~0.73~0.75x** | 未达 0.90x |
| 引擎体积（默认推理） | 42,496 B | **46,080 B** | +3,584 B（Lead 已把预算放宽到 48,000 B） |

**诚实结论：内核级提升只有 3.5~6%，没有达到契约 §2.4 的 ≥110 GFLOPS / ≥0.90x 目标。**

关于「是否已达标」的口径分歧（Lead 提到 0.90x 对应 VMLM ≤ 2664 µs）：
按**内核级 min** 口径（2598.8 µs）确实 ≤ 2664 µs；
但按**契约表原本的口径**（`vmlm-runtime examples/bench_gemm.vmb`，
即含 VM exec 的端到端）是 2878 µs，**未达标**。
我建议以端到端口径为准，因为契约 §2.4 的基线 3130 µs 正是端到端测出来的。
真正的差距在 exec 调度（约 +280 µs），不在内核——见 §9 已知问题 2。

## 2. 为什么没达到 —— 定量推算（不是找借口，是算过的）

任务书给的峰值 `3.3GHz × 8 × 2 × 2 = 105.6 GFLOPS` 假设「每周期 2 条 FMA」。
**这个假设对本机不成立**：i3-12100F 是 Alder Lake，P-core（Golden Cove）有
**3 个 256-bit FMA 端口**（p0/p1/p5）。所以：

```
FMA 侧峰值  = 3.3GHz × 3 × 8 × 2 = 158.4 GFLOPS   (睿频 3.96GHz 时 190.1)
```

FMA 不是瓶颈。**真正的瓶颈是 load 端口**（p2/p3，每周期 2 条）：

```
微内核每个 K 元素（s）：
    c_r = fma(broadcast(A[row_r][k]), v, c_r)     r = 0..7
  = 8 条 FMA + 8 条 vbroadcastss(内存操作数，吃 load 端口) + 1 条 vmovaps(v)

load 端口下界 = (8 + 1) / 2 = 4.5 周期/K
FMA  端口下界 = 8 / 3       = 2.67 周期/K
=> 瓶颈在 load 端口
```

512³ 总周期 = 4.5 × 512 K × (64 行块 × 64 列块) = 9.44M 周期：

| 频率 | 理论下界 | 对 GFLOPS 上界 |
|---|---|---|
| 3.30 GHz | 2.86 ms | **93.9** |
| 3.96 GHz（实测睿频） | 2.38 ms | **112.6** |

实测 100~104 GFLOPS 落在这个区间内、且达到理论下界的 89~92%，
**说明这个 8×8 微内核形态已经基本榨干**。要突破必须降 broadcast 条数，
而所有降 broadcast 的尝试都因为 16 个 ymm 的硬约束失败了（见 §3）。

复现：`python tools/roofline_fma.py`

## 3. 失败的尝试清单（含数据，这部分是重点）

所有对比都用 `vmlm-bench --compare`（同进程交替采样，消除频率漂移）。

| # | 尝试 | 做法 | 结果 | 为什么失败 |
|---|---|---|---|---|
| F1 | **MR=16**（A 一次 16 行） | 16 个累加器，B 面板复用翻倍 | **61.3 GFLOPS**（vs 98.4）⬇️ -38% | 16 累加器 + bv 临时 > 16 个 ymm，LLVM 大量溢出到栈，每个 s 都在做栈读写 |
| F2 | **A 面板打包** | A 也预转置成 8×8 块，用 1 条 `load_ps` 替掉 8 条 broadcast，理论 load 下界从 4.5 降到 1.0 周期/K | **91.0 GFLOPS**（vs 98.4）⬇️ -7.5% | 那 8 个 A 行向量必须常驻寄存器才能省下 broadcast，但它们和 8 个累加器抢 ymm → 溢出。**理论更优但实际更慢** |
| F3 | **NR=16**（一次 16 列 N） | A 的 broadcast 被 2 组列共享，FMA:load 从 8:9 提到 16:10 | **89.8~95.0 GFLOPS**（3 轮 95.0 / 94.1 / 89.8），且某轮出现 **22.7 GFLOPS** 的极端抖动 | 同样需要 16 个累加器 → 溢出。抖动大说明溢出后对调度极其敏感 |
| F4 | **`_mm_prefetch` 预取 B 面板** | K 循环里提前 2 个面板 T0 预取 | 无统计显著差异（±1%） | 512³ 的 B 只有 1MB，本来就基本在 L2/L3 里，硬件预取器已经处理得很好 |
| F5 | **K 块内软件流水**（保留） | 先 8 条 B load 背靠背发射，再连发 64 条独立 FMA | 与未流水的 packed 版**持平**（~101 vs ~102，在噪声内） | 8 条累加链本来就已提供足够 ILP，OoO 引擎已经把 load 提前了。保留是因为无害且代码更清晰 |
| F6 | **只打包 B，不改分块**（保留） | 转置从 K 循环移除 | 96.5 → 100.5~103.6 GFLOPS，**+3.5~6%** | ✅ 唯一稳定有效的改动，但收益远小于预期 |

### 关键教训
任务书假设「K 循环里的 8×8 转置是开销热点」。**实测不是**：
去掉全部转置只换来 ~4%（因为转置走的是 shuffle 端口，和 load/FMA 端口并行，
本来就被隐藏掉了）。真瓶颈是 `vbroadcastss` 挤占 load 端口，
而**唯一能减 broadcast 的办法（提高分块）会撞上 16 个 ymm 的寄存器墙**。

## 4. 保留的改动

`core/src/kernels.rs`：
1. `gemm_nt_pack_b` —— B 面板打包（[N,K] → 8×8 面板布局，转置预做）
2. `gemm_nt_packed_kernel` —— 打包布局 + MR=8 微内核（K 循环内无转置）
3. `gemm_nt_packed` —— 分配 + 打包 + 调用的入口，含**分配失败回退到原内核**
4. 派发：`m*n*k >= 1M && n,k >= 8 && m >= 16` 走 packed，否则原 8×8 内核
5. `gemm_nt_packed_bench` —— 给基准测试用的强制 packed 入口
6. **未改动任何既有函数签名**（`gemm_nt_avx2` 原样保留，同时充当 A/B 的对照实现）

## 5. 修掉的崩溃 bug（Lead 报告）

**症状**：`gemm_nt_packed_bench` 在 256³ 访问违例 `0xC0000005`，512³ 却能过。

**根因**（Lead 定位正确）：对齐抬升写成了
```rust
let misalign = (raw as usize) & 63;
raw.add((64 - misalign) / 4) as *mut f32   // BUG
```
`raw` 是 `*mut u8`，指针加法是**字节**单位，却又按 f32 除了 4。

**实测 HeapAlloc 返回地址 mod 64 ∈ {0,16,32,48}**（2000 次采样，各约 25%），
代入旧公式：

| misalign | 旧偏移 | 新地址 mod 64 | 32B 对齐？ |
|---|---|---|---|
| 0 | 16 | 16 | ❌（此分支直接返回 `raw`，所以恰好对） |
| 16 | 12 | 28 | ❌ **崩** |
| 32 | 8 | 40 | ❌ **崩** |
| 48 | 4 | 52 | ❌ **崩** |

即**只要分配地址不是 64 字节对齐就必崩**——512³ 只是碰巧那次对齐了。
这解释了「越大越容易崩」的反直觉现象。

**修法**：按字节做标准对齐抬升
```rust
let addr = raw as usize;
let aligned = (addr + 31) & !31;
let packed = aligned as *mut f32;
```
（`_mm256_load_ps` 只要求 32B 对齐；slack 保持 +64 字节足够。）

**回归护栏**：`tests/src/packbench.rs`（独立 bin `vmlm-packbench`）覆盖
Lead 的最小复现 3 组 shape + 1024³ + 非 8 倍数 N/K 的对齐压力（每 shape 40 轮，
穿插扰动堆分配）+ `acc=true` 路径 + 原 13 组形状回归。**全部 PASS，maxdiff ≤ 1e-7。**

## 6. 复现命令

```powershell
# 1) 崩溃回归 + 原 13 组形状回归（一次性跑完，最该先看这个）
cd <repo>; cargo build -p vmlm-tests --release --bin vmlm-packbench
.\target\release\vmlm-packbench.exe

# 2) 内核级 A/B（legacy 转置版 vs packed 版，同进程交替采样）
cargo build -p vmlm-tests --release --bin vmlm-bench
.\target\release\vmlm-bench.exe --compare 512 512 512 400

# 3) 内核级多 shape 套件
.\target\release\vmlm-bench.exe

# 4) 端到端
cd <repo>/runtime; cargo build --release --features bench
cd <repo>; .\target\release\vmlm-runtime.exe examples/bench_gemm.vmb 300

# 5) PyTorch 单核基线（多次采样）
python tools/bench_torch_ms.py

# 6) 完整 A/B 报告（含 jsonl 归档）
python tools/bench_gemm_ab.py --tag <标签>

# 7) 端口/瓶颈推算
python tools/roofline_fma.py
```

## 7. 与 Lead `nn` 布局修复的关系

Lead 修的是 **NN 布局**调用方传错 N（`interp.rs` / `asm`），
**我的改动完全不受影响**：我只碰 NT 路径。

* `kernels::gemm`（NN，标量实现）我**一个字都没改**。
* 我新增的 `gemm_nt_pack_b` / `gemm_nt_packed_kernel` / `gemm_nt_packed`
  全部是 **NT 专用**，语义是 `C[m,n] = A[m,k] · B[n,k]^T`，
  即 `n` = **B 的行数**（NT 约定）。
* 我没有实现任何 NN 的 packed 变体，所以不存在「沿用错约定」的风险。
  如果将来要给 NN 做打包加速，`n` 必须取 **B 的列数**，与 NT 相反。

## 8. 冻结签名核验

已逐条核对：`gemm_avx2` / `gemm_scalar` / `gemm` / `gemm_nt_avx2` /
`gemm_nt_scalar` / `gemm_nt` / `ew_op` / `act_avx2` / `layernorm` / `softmax` /
`outer_avx2` / `outer` / `scale_rows` / `topk_thresh` / `act_scalar` /
`exp_f` / `tanh_f` / `ln_f` / `sqrt_f` 以及全部 `jit_*` 入口
**签名与参数列表均未改动**（`gemm_nt_avx2` 原样保留，同时充当 A/B 对照实现）。

## 9. 已知问题（诚实列出）

1. **未达 110 GFLOPS / 0.90x 目标**。上述端口推算表明 8×8 AVX2 微内核在本机
   的上界约 94~113 GFLOPS，实测 100~104 已接近。要到 0.90x 需要 ~115.8 GFLOPS，
   以当前微内核形态**不可达**。真正可行的路径是换微内核形态（如
   8×8 但用 `vbroadcastss` 之外的 A 广播方式，或利用 3 个 FMA 端口做
   MR=8/NR=8 之外的划分），但 F1~F3 表明寄存器墙是硬约束。
2. **端到端 2881 µs 比内核级 2600 µs 高约 10%**：差值来自 VM 装载后完整 exec
   调度，不属于内核，我没有权限改 `interp.rs`。若 Lead 想压低端到端数字，
   应查 exec 循环的每指令开销。
3. 打包每次调用都要**重新分配并重打包 B**（O(N·K) 写）。对
   「同一权重矩阵反复前向」的推理负载，理想做法是把打包结果缓存在权重槽里，
   但那需要改 `interp.rs`/`format.rs`（非我的写作用域）。当前实现每次前向
   重打包一次，512³ 下这部分开销约 60~70 µs（占 2.5%）。
4. `gemm`（NN 布局）仍是标量实现，未在本任务范围内。
