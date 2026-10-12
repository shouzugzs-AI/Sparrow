// PATH: B1
//
// B1 = 纯稀疏存储 + 全连接计算：每神经元 k0 条连接，**遍历全部 k0 条**。
//
// 无 top-k、无掩码、无筛选、无 gather 筛选 —— 连接列表里本来就只有 k0=20 条，
// 所以「遍历 20 条」就是全部的工作量，没有第二步选择。
//
// ===========================================================================
// ⚠️ S1 向量化已回退 —— 实测为符号级错误，记录在此以免重犯
// ===========================================================================
//
// bench-builder 的 `vmlm-b1-vec` 提议 S1：向量化「取乘积」、累加仍走标量顺序，
// 声称 2.08x 且 bit-exact（他的探针用自建的复刻内核，bit-different=0/1536）。
// Lead 把它落地到本文件后，**A1 判据立刻报 1536/1536 分叉**（`vmlm-b1-verify`）。
//
// 根因（`train/src/lead_b1_s1diag.rs` 直接实测，不是推理）：
//   `sparse_strength_row_avx2` **返回的是 |w*x|**，不是 w*x。
//   它是**强度行**内核，绝对值是它的**语义**（`kernels.rs:1698` 的
//   `_mm256_andnot_ps(sign, prod)` 清符号位）。把它复用去算**前向求和的乘积**
//   会摧毁每一个负项的符号。
//
//   实测：30720 个值里 15314 个不匹配，**其中 15314 个是纯符号翻转（100%）**：
//       kernel=+0.005257437  scalar=-0.005257437   |equal|=true
//   端到端：b1_forward vs 算子参考 = **1536/1536 分叉**。
//
// 结论：**这个复用是错的，S1 已完整回退。** 若将来要做这条优化，必须
// **新写一个保符号的乘积内核**（不能复用强度行内核），并重跑 A1-A5。
//
// 附带的方法论收获（两个都值得留档）：
//   1. bench-builder 的 S1 复刻内核是他自己写的，**不带绝对值** ——
//      所以他的探针给出了 bit-different=0 的「可落地」结论，而真实内核带绝对值。
//      探针的复刻体与真实实现漂移，会让「已证明可落地」变成假的。
//   2. Lead 的 `lead_b1_measure.rs` 初版**自己定义了一份 b1_forward**，
//      于是落地后它仍在测那份副本、报 PASS。**已修**：现在调用
//      `vmlm_core::b1_pure::b1_forward`，并新增 A0 判据
//      （oracle vs production 逐位对拍）来防止同类事故。
//
// ===========================================================================
// 为什么累加顺序是契约而不是风格（这是本模块唯一的正确性风险）
// ===========================================================================
//
// B1 的验收判据是「与算子版**逐位相同**」。浮点加法**不满足结合律**，
// 所以同一个数学表达式换个求和顺序就是另一个比特串。于是：
//
//   * 行内按 j 升序（0..k0）—— 与 sparse_accum 同序；
//   * 单个标量累加器 acc —— 不做 8 路 SIMD 归约（归约会改变分组，进而改变位型）；
//   * **首个命中直接赋值** `acc = term`，而不是从 `0.0f32` 起加。
//
// 最后一条最容易被当成「无意义的微优化」而删掉，但它真的会改变结果：
//   * `0.0 + (-0.0)` 在 IEEE-754 下等于 `+0.0`，而 `-0.0` 的位型是 0x8000_0000。
//     从 0.0 起加会把 x 行里的 `-0.0` 权重产物「洗」成 `+0.0`，位型分叉。
//   * 同理，若某一行的第一项是 `-0.0` 而后续项把它抵成 `-0.0`，两条路径分叉。
// 所以 `acc = term` 是 bit-exact 的**前提**，不是可选的优化。判据 A1 因此
// 用 `f32::to_bits()` 比较而不是 `==`：`==` 对 NaN 恒假、对 ±0 恒真，
// 恰好对上面这类分叉**全盲**。
//
// ===========================================================================
// 「禁止 8 路归约」现在有实测支撑，不再只是警告（bench-builder, vmlm-b1-vec）
// ===========================================================================
//
// 本条原先只是文档里的警告。现已有对照实测（d=1536 k0=20，5 matvec x 24 层
// x 128 位置外推）：
//
//     S0 标量现状（= 本文件当前实现）            1.204 ns/conn   568.3 ms/step
//     S1 向量 gather + 标量顺序累加              0.579 ns/conn   273.4   (2.08x)
//     S2 向量 gather + SIMD 8 路归约             0.352 ns/conn   165.9   (3.43x)
//     S3 无 gather 天花板                        0.247 ns/conn   116.7   (4.87x)
//
//     S1 vs S0 bit-different =   0 / 1536
//     S2 vs S0 bit-different = 841 / 1536   <= 归约改变了分组
//
// ⚠️ **这三行的用途是「证明禁令有代价」，不是「本模块已采用 S1」。**
//   本文件当前是 **S0 纯标量**：S1 曾按 bench-builder 的探针落地，但落地后
//   A1 立即报 1536/1536 分叉（根因见文件头 S1 回退一节 —— 探针的复刻内核
//   不带 abs，真实内核带），**已完整回退**。
//
//   所以现在读这段的正确含义是：
//     * S2 的 841/1536 证明「禁止 8 路归约」不是风格问题 ⇒ 归约禁令是硬的；
//     * S1 的 2.08x 是**尚未兑现的上限**，要拿到它必须**新写一个保符号的
//       乘积内核**（不能复用 `sparse_strength_row_avx2`），并重跑 A1-A5。
//   在有人真的写出那个内核并通过全套判据之前，**本模块保持 S0**。
//
// ===========================================================================
// 存储：直接复用 conn_train::SparseConnW，不另起结构
// ===========================================================================
//
// 任务书要求 B1 与既有存储同构，这样两条路径算的是**同一批数**，
// 差异只能来自算法（有没有选择），不能来自布局。
// 本模块因此**只**依赖 `SparseConnW` 的 (w, src, k0, n_neurons) 四个字段：
//   * 完全不读 `act` —— 见 b1_forward 的说明与判据 A2；
//   * 不读 `layer` / `e`（`e` 只用于断言布局自洽）。
//
// 这样 B1 可以挂在任何已有的 K0_DEFAULT=20 层上直接跑，不需要新建存储格式。
//
// ===========================================================================
// 约定：本模块是「无掩码」路径，所以它**不维护 act**
// ===========================================================================
//
// 既有算子路径的契约是「前向先写掩码，反向读掩码」（单一真相源，见
// conn_train::compute_act_mask 的长注释）。B1 没有掩码这个概念，
// 因为 k0 条全部参与 —— 掩码恒等于全 1，把它物化出来只会
// (a) 花 O(E) 的写带宽，(b) 引入一个可以不同步的状态。
// ⇒ B1 前向**不写 act**；反向也不读 act（全部 20 条都算梯度）。
// 这不是「漏了」，是本路径的定义。判据 A2 把这条**当成主张去测**：
// 前向调用前后对 act[] 做整体位快照比对，必须 0 处变化。
//
// 代价必须说清楚：若上层仍按「反向读掩码」的写法调用
// `sparse_gemm_bwd_w`，而 act 是被 B1 前向留下的旧掩码（或从未初始化），
// 反向就会按一张**与本次前向无关**的掩码算梯度。所以 B1 的反向必须
// 用**本模块的** `b1_backward`，且调用方不得把 B1 前向与
// `sparse_gemm_bwd_w` 混用。这一条在验收驱动里由 A5 覆盖。

use crate::conn::SparseConnW;

/// 行内乘积缓冲的栈上界。与 `conn_train::ROW_TOPK_STACK_K` 同一个理由：
/// 固定的小尺寸 + 超界**硬失败**，而不是开一个大栈帧然后静默崩溃。
///
/// 为什么需要它（这条不是装饰）：本缓冲开在 `b1_forward` 里，而该函数
/// 逐神经元调用 —— 24 层 × 1536 神经元 × 128 位置 = 4.7e6 次/step。
/// 栈帧一大就跳过 guard page，表现为 **0xC0000005 且没有任何诊断输出**
/// （本项目 build_all.ps1 记录过这个类，conn_train 也栽过一次）。
/// ⇒ k0 超过本上界时 `exit(9)` 并打印是哪一条，把静默崩溃换成可读报错。
///
/// ⚠️ 2026-10 上界 256 → **1024**（B1 scaling law 扫描需要 k0=512）：
///   这个常量**只界住标量回退路径**里的 `prod` 栈数组（`1024 × 4 B = 4 KB`），
///   而 AVX2 生产路径（`b1_dot_row_u32_avx2`）**不用任何栈数组** —— 所以提高上界
///   不改变生产路径的栈占用，也不改变任何数值。实测：k0=512 在修前 exit(9)，
///   修后正常跑完；`vmlm-b1-verify`（20 判据）与 `vmlm-b1-canfail`（7 判据）复跑
///   仍然全 PASS（k0=20 的路径一个字节未变）。
pub const B1_STACK_K: usize = 1024;

/// B1 前向：y = Σ_{j<k0} w[i][j] · x[src[i][j]]，**全部 k0 条都乘都加**。
///
/// 返回实际执行的乘加次数 = n_neurons * k0。报计数（而不是返回 `()`）
/// 是沿用本项目的一贯做法：让「算了多少」这条主张**可失败** —— 计数器
/// 一旦漂移，判据就能报出来，而不是靠读代码相信。
///
/// **纯标量实现。** 见文件头的 S1 回退记录：唯一现成的向量内核
/// `sparse_strength_row_avx2` 返回 |w*x|，复用它做前向求和是符号级错误，
/// 已实测证伪并回退。要做这条优化必须先写一个**保符号**的乘积内核。
///
/// **不读也不写 `sc.act`。** 掩码恒为全 1，物化它纯属浪费带宽，
/// 且会引入一个能与前向不同步的状态。见模块头注释。
///
/// # Safety
/// * `sc.w` / `sc.src` 各至少 `sc.e = n_neurons * k0` 个 f32（由
///   `SparseConnW::from_base` 保证）。
/// * `x` 至少 `n_neurons` 个 f32；每个 `src` 值都必须落在 `[0, n_neurons)`
///   （由 `neuron_conn_src` 的契约保证；越界是调用方的契约违反）。
/// * `y` 至少 `n_neurons` 个 f32 且可写。
pub unsafe fn b1_forward(sc: &SparseConnW, x: *const f32, y: *mut f32) -> usize {
    let n = sc.n_neurons as usize;
    let k0 = sc.k0;

    // 几何自洽守卫：先于任何向量读。
    //
    // 为什么必须在**向量读之前**：SIMD 路径按 k0 以 8 宽读 `src_u32`，
    // 若 k0 与缓冲区不符（实测事故：某探针把 k0 改成 k0+1），最后一行会
    // 读越过 `src_u32` 末尾 —— 而它是分配区的最后一块 —— 直接
    // **0xC0000005 且无诊断**。旧标量版一次读一个 f32，越界一格仍落在
    // 分配的余量里，所以"活下来了"，但那是运气不是正确性。
    // 这里把静默崩溃换成可读报错。
    sc.assert_geometry();

    // 栈上界硬失败（与 conn_train 同一口径）：不静默截断。
    if k0 > B1_STACK_K {
        vmlm_sys::io::print(b"B1: k0 exceeds the safe stack bound\n");
        vmlm_sys::io::print(b" k0=");
        vmlm_sys::io::print_u32(k0 as u32);
        vmlm_sys::io::print(b" bound=");
        vmlm_sys::io::print_u32(B1_STACK_K as u32);
        vmlm_sys::io::print(b"\n");
        vmlm_sys::io::exit(9);
    }

    // ---- SIMD 归约路径（验收：ε 级差异 + 确定性）----
    //
    // 这是**用户裁决后**的路径：验收标准从逐位相同放宽为「舍入级差异」。
    // 实测（`vmlm-b1-eps`）：
    //     单层   max|diff| = 2.980e-08
    //     24 层  max|diff| = 1.526e-05（相对 1.371e-07）
    //   位型与标量相同者 44.7%，即**多数输出会变**，这是有意的取舍。
    //
    // 同时使用 u32 镜像索引（`sc.src_u32`），省掉每条连接的 `cvttss2dq`：
    // 实测发布形态 1.43x（`vmlm-b1-shipped-u32`）。
    //
    // 融合的意义：不融合就要先写 k0 个乘积进缓冲再读回来做归约，
    // 那趟 store/reload 实测值 0.127 ns/conn（`vmlm-b1-reconcile`）。
    #[cfg(target_arch = "x86_64")]
    {
        if vmlm_sys::cpu::HAS_AVX2 {
            // [P19-5] 先走「小 k0 打包」内核（k0 ∈ {1,2,4,8}，G=8/k0 行/ymm）；
            // 它返回已处理的行数，其余行仍走原逐行内核 ⇒ 不支持的 k0 行为完全不变。
            let done = crate::kernels::b1_forward_packed_avx2(sc.w, sc.src_u32, x, k0, n, y);
            for i in done..n {
                let base = i * k0;
                *y.add(i) = crate::kernels::b1_dot_row_u32_avx2(
                    sc.w.add(base),
                    sc.src_u32.add(base),
                    x,
                    k0,
                );
            }
            return n * k0;
        }
    }

    // ---- 标量回退路径：无 AVX2 时使用，也是精度对账的参考实现 ----
    //
    // **保留理由**：验收标准放宽后，**必须有一个参考物去算那个差异**。
    // 用它做基准，而不是拿 SIMD 版自己跟自己比。
    //
    // 累加契约（本路径仍严格遵守）：升序 j、单标量累加器、首项直接赋值。
    let mut macs = 0usize;
    let mut prod = [0.0f32; B1_STACK_K];
    let prod_ptr = prod.as_mut_ptr();
    for i in 0..n {
        let base = i * k0;
        for j in 0..k0 {
            let sid = *sc.src.add(base + j) as usize;
            *prod_ptr.add(j) = *sc.w.add(base + j) * *x.add(sid);
        }
        *y.add(i) = crate::kernels::b1_reduce_scalar(prod_ptr, k0);
        macs += k0;
    }
    macs
}

// ===========================================================================
// PATH: B1 —— 令牌分块（transposed activation）前向
// ===========================================================================
//
// 为什么加这条路径（**实测归因，不是猜想**）：发布版 `b1_forward` 走
// `kernels::b1_dot_row_u32_avx2`（8 路 gather + 8 路归约），实测
// 0.0058 ms/矩阵 = 0.189 ns/conn。换算成周期：@3.3GHz 为 **0.623 cycles/连接**，
// 而该内核每条连接恰好消耗 1/8 次 `_mm256_i32gather_ps`（k0=20 → 2 次整
// gather + 4 条标量尾）⇒ 0.623 × 8 = **5.0 cycles / ymm-gather**，与
// Alder Lake P-core 的 gather 吞吐完全吻合。
// ⇒ **发布版 B1 是 gather 吞吐受限。** 想更快只能**减少 gather 次数**。
//
// 手段：把「1 条连接 × 8 个 token」放进一个 ymm 通道。激活布局从
// token-major `[T][d]` 换成 **source-major** `[d][t8]`，于是取 x 变成
// **连续 32B load**，gather 次数从「每连接一次」降到 **0**。
//
// 浮点契约：本路径**逐位等于标量契约参考**（`b1_forward_scalar_reference`
// 的同一算法：j 升序、单累加器/通道、首项直接赋值、不用 FMA、无横向归约）。
// 也就是说它比发布版 SIMD 路径（8 路归约、与标量参考仅 45.8% 位相同）
// **精度更强**：验收用 `f32::to_bits`，期望 0 分叉。
//
// 代价：调用方要付一次转置 `[T][d] -> [d][t8]`。`b1_transpose_activations`
// 提供 8x8 分块 SIMD 转置；tok/s 验收把这次转置**计入**计时区间。

/// 令牌分块 B1 前向：`yt[i*t8 + t] = Σ_j w[i][j]·xt[src[i][j]*t8 + t]`。
///
/// 输出是**转置**的（source-major），因为一条 ymm 通道 = 一个 token，
/// 而 AVX2 没有 scatter：转置输出让 8 个 token 一次 `storeu` 写完，
/// token-major 输出则需要 8 次标量写。
///
/// 逐位契约与 `b1_forward_scalar_reference` 相同（见上）。
///
/// # 前置契约
/// * `sc.src_u32` 已与 `sc.src` 同步（生产初始化路径 `init_sparse_conn` 会同步；
///   判据用 `assert_src_u32_consistent()` 一次性校验，不在热路径里做 O(E) 检查）。
/// * `t8 >= n_tokens`，且 `t8` 是 `xt` / `yt` 的行距。
/// * `xt` 至少 `n_neurons * t8`，`yt` 至少 `n_neurons * t8`。
///
/// 返回实际乘加次数 `n_neurons * k0`（与发布版同口径）。
///
/// # Safety
/// 见上；越界/几何不符会硬失败而不是静默读坏内存。
pub unsafe fn b1_forward_t8(
    sc: &SparseConnW,
    xt: *const f32,
    t8: usize,
    n_tokens: usize,
    yt: *mut f32,
) -> usize {
    sc.assert_geometry();
    if t8 < n_tokens {
        vmlm_sys::io::print(b"B1_T8: row stride t8 < n_tokens\n");
        vmlm_sys::io::exit(9);
    }
    if sc.k0 > B1_STACK_K {
        vmlm_sys::io::print(b"B1_T8: k0 exceeds the safe bound\n");
        vmlm_sys::io::exit(9);
    }
    let n = sc.n_neurons as usize;
    let k0 = sc.k0;
    #[cfg(target_arch = "x86_64")]
    {
        if vmlm_sys::cpu::HAS_AVX2 {
            return crate::kernels::b1_forward_t8x4_avx2(
                sc.w, sc.src_u32, n, k0, xt, t8, n_tokens, yt,
            );
        }
    }
    crate::kernels::b1_forward_t8_scalar(sc.w, sc.src_u32, n, k0, xt, t8, n_tokens, yt)
}

/// 令牌分块前向的**带镜像校验**入口：先 `assert_src_u32_consistent()` 再前向。
///
/// 为什么需要它：令牌分块路径与发布版一样**按 `src_u32` 取数**，而 `src_u32`
/// 是独立缓冲 —— 一个只写了 `sc.src` 的调用方会读到垃圾下标（本轮实测：
/// `b1_verify` 的 k=1 对照就是这样拿到 **0xC0000005**，而不是一个错误的数）。
/// 校验是 O(E)，而本调用一次处理 `n_tokens` 个 token（≥8），所以摊薄后 ≤12.5%；
/// **热路径用 `b1_forward_t8`，首次接入/边界用例用本入口**。
///
/// 不一致时硬失败并打印首个位置（见 `SparseConnW::assert_src_u32_consistent`）。
///
/// # Safety
/// 同 `b1_forward_t8`。
pub unsafe fn b1_forward_t8_checked(
    sc: &SparseConnW,
    xt: *const f32,
    t8: usize,
    n_tokens: usize,
    yt: *mut f32,
) -> usize {
    sc.assert_src_u32_consistent();
    b1_forward_t8(sc, xt, t8, n_tokens, yt)
}

/// 令牌分块前向的 **1 块（8 token）版本** —— 保留用于测量与对照。
///
/// 生产入口 `b1_forward_t8` 走 4 块版本（摊薄每连接固定开销 + 4 条独立累加链）。
/// 两个版本必须给出**逐位相同**的结果（判据 C2 对 `b1_forward_t8` 生效；
/// 本入口用于把「摊薄」这件事的收益单独量出来）。
///
/// # Safety
/// 同 `b1_forward_t8`。
pub unsafe fn b1_forward_t8_x1(
    sc: &SparseConnW,
    xt: *const f32,
    t8: usize,
    n_tokens: usize,
    yt: *mut f32,
) -> usize {
    let n = sc.n_neurons as usize;
    #[cfg(target_arch = "x86_64")]
    {
        if vmlm_sys::cpu::HAS_AVX2 {
            return crate::kernels::b1_forward_t8_avx2(
                sc.w, sc.src_u32, n, sc.k0, xt, t8, n_tokens, yt,
            );
        }
    }
    crate::kernels::b1_forward_t8_scalar(sc.w, sc.src_u32, n, sc.k0, xt, t8, n_tokens, yt)
}

/// 令牌分块前向的**标量契约参考**（对拍基准；无 ISA 分支）。
///
/// # Safety
/// 同 `b1_forward_t8`。
pub unsafe fn b1_forward_t8_scalar_reference(
    sc: &SparseConnW,
    xt: *const f32,
    t8: usize,
    n_tokens: usize,
    yt: *mut f32,
) -> usize {
    let n = sc.n_neurons as usize;
    crate::kernels::b1_forward_t8_scalar(sc.w, sc.src_u32, n, sc.k0, xt, t8, n_tokens, yt)
}

/// 激活转置 `[T][d] -> [d][t8]`：`xt[s*t8 + t] = x[t*d + s]`，填充列写 0。
///
/// 这是令牌分块路径唯一的新增数据搬运成本，必须计入 tok/s。
///
/// # Safety
/// `x` 至少 `t*d`；`xt` 至少 `d*t8` 且可写；`t <= t8`。
pub unsafe fn b1_transpose_activations(x: *const f32, t: usize, d: usize, xt: *mut f32, t8: usize) {
    if t8 < t {
        vmlm_sys::io::print(b"B1_T8: transpose row stride t8 < t\n");
        vmlm_sys::io::exit(9);
    }
    #[cfg(target_arch = "x86_64")]
    {
        if vmlm_sys::cpu::HAS_AVX2 {
            crate::kernels::transpose_t8_avx2(x, t, d, xt, t8);
            return;
        }
    }
    crate::kernels::transpose_t8_scalar(x, t, d, xt, t8);
}

// ===========================================================================
// PATH: B1 —— 反向的输入侧梯度（dx），以及 P1 的「dw+dx 一次遍历」融合
// ===========================================================================
//
// 为什么 B1 必须有 dx：只有 `b1_backward`（dw）时，梯度**传不到上一层**，
// 于是 24 层里只有最后一层在学。一个只有 dw 的"训练"会跑通、loss 也会降，
// 但那不是深度训练。P0 接入训练器时这条是**前置**。
//
// 累加顺序契约（与 dw 同一纪律）：
//   dx[sid] += dy[i] * w[i][j]，**i 升序、j 升序**，逐个元素一次 add。
//   这正是既有训练器里手写掩码循环的顺序（`sparse_train24.rs` 的 ffn/Wo/qkv 段），
//   所以 B1 的 dx 与掩码版在 k=k0 时**逐位相同**。
//
// ⚠️ dx 是**散射累加**（多个 (i,j) 写同一个 sid），AVX2 没有 scatter。
//   融合内核因此让 dw 走向量、dx 走标量；顺序不变 ⇒ 仍然逐位。

/// dx 反向：`dx[src[i][j]] += dy[i] · w[i][j]`，**全部 k0 条都写**（B1 无掩码）。
///
/// `dx` 由调用方清零（本函数只做 `+=`，与引擎侧反向指令的累加语义一致）。
/// 返回累加次数 = `n_neurons * k0`。
///
/// # Safety
/// * `sc.w` / `sc.src_u32` 各至少 `sc.e`；
/// * `dy` 至少 `n_neurons`；`dx` 至少源维度个 f32 且**已清零**。
pub unsafe fn b1_backward_dx(sc: &SparseConnW, dy: *const f32, dx: *mut f32) -> usize {
    let n = sc.n_neurons as usize;
    let k0 = sc.k0;
    sc.assert_geometry();
    let mut accs = 0usize;
    for i in 0..n {
        let g = *dy.add(i);
        let base = i * k0;
        for j in 0..k0 {
            let sid = *sc.src_u32.add(base + j) as usize;
            *dx.add(sid) += g * *sc.w.add(base + j);
            accs += 1;
        }
    }
    accs
}

/// dx 反向的**标量契约参考**（显式 f32 src，用于对拍与无镜像场景）。
///
/// # Safety
/// 同 `b1_backward_dx`，但读 `sc.src`（f32）而不是镜像。
pub unsafe fn b1_backward_dx_scalar_reference(
    sc: &SparseConnW,
    dy: *const f32,
    dx: *mut f32,
) -> usize {
    let n = sc.n_neurons as usize;
    let k0 = sc.k0;
    for i in 0..n {
        let g = *dy.add(i);
        let base = i * k0;
        for j in 0..k0 {
            let sid = *sc.src.add(base + j) as usize;
            *dx.add(sid) += g * *sc.w.add(base + j);
        }
    }
    n * k0
}

/// **P1 融合**：一次遍历同时得到 dw 与 dx（等价于 `b1_backward` + `b1_backward_dx`）。
///
/// 为什么这是可以真正落地的融合：`dw` 与 `dx` 都在 `dy` 已知之后执行，二者共享
/// `src` 与 `dy[i]`，且 `x`/`w` 各只需各读一次（分开写是两遍遍历）。
///
/// 融合后**逐位不变**（不是 ε）：见 `kernels::b1_backward_fused_dwdx_avx2` 的论证。
///
/// ⚠️ 诚实边界：**「前向 + dw + dx 一次遍历」在标准反向里做不到** ——
///   前向需要 `dy`，而 `dy` 是整网前向 + loss 之后才有的量。三合一只有在
///   **重算激活**（activation recomputation）的调度下才有意义；本项目没有这种调度，
///   所以本轮交付的是可落地且可验证的 **dw+dx 二合一**。详见
///   `docs/B_P0接入训练器_P1融合.md`。
///
/// 返回遍历的连接数 = `n_neurons * k0`。
///
/// # Safety
/// 同 `b1_backward` 与 `b1_backward_dx`：`dw` 与 `dx` 都**必须由调用方先清零**。
pub unsafe fn b1_backward_fused(
    sc: &SparseConnW,
    dy: *const f32,
    x: *const f32,
    dw: *mut f32,
    dx: *mut f32,
) -> usize {
    let n = sc.n_neurons as usize;
    let k0 = sc.k0;
    sc.assert_geometry();
    #[cfg(target_arch = "x86_64")]
    {
        if vmlm_sys::cpu::HAS_AVX2 {
            crate::kernels::b1_backward_fused_dwdx_avx2(
                sc.w, sc.src_u32, n, k0, dy, x, dw, dx,
            );
            return n * k0;
        }
    }
    crate::kernels::b1_backward_fused_dwdx_scalar(sc.w, sc.src_u32, n, k0, dy, x, dw, dx);
    n * k0
}

/// 融合反向的**标量契约参考**（对拍基准，与生产融合内核逐位相同）。
///
/// # Safety
/// 同 `b1_backward_fused`。
pub unsafe fn b1_backward_fused_scalar_reference(
    sc: &SparseConnW,
    dy: *const f32,
    x: *const f32,
    dw: *mut f32,
    dx: *mut f32,
) -> usize {
    crate::kernels::b1_backward_fused_dwdx_scalar(
        sc.w,
        sc.src_u32,
        sc.n_neurons as usize,
        sc.k0,
        dy,
        x,
        dw,
        dx,
    );
    sc.n_neurons as usize * sc.k0
}

// =====================================================================
// P20：**drop-in** 的「一层 = 转置 + source-major 内核 + 转置回」
// =====================================================================
// 为什么做成 drop-in（而不是把整个 12 层栈都换成 source-major 布局）：
// 布局改造要动 step_fn 的 region/slot 缓冲与注意力边界，风险大；而实测显示
// **把两次转置也算进去**，t8 前向仍快 1.83x、反向快 2.01x（kbench --mode t8 / t8bwd）。
// 于是先拿到收益，不动任何调用方的布局。

/// 一层前向（drop-in t8）：`y[t][n] = W·x[t][d]`，内部 `x->xt[d][t8]`、`yt[n][t8]->y`。
///
/// `xt` 至少 `d*t8`，`yt` 至少 `n*t8`，且 `t8 >= t`、`t8 % 8 == 0`。
/// 调用方负责 t8 缓冲（可复用）。
///
/// # Safety 见上；`sc.src_u32` 已与 `src` 同步。
pub unsafe fn b1_layer_forward_t8(
    sc: &SparseConnW,
    x: *const f32,
    xs: usize,
    y: *mut f32,
    ys: usize,
    t: usize,
    xt: *mut f32,
    yt: *mut f32,
    t8: usize,
) -> usize {
    let d = sc.n_neurons as usize;
    crate::kernels::transpose_t8_avx2(x, t, d, xt, t8);
    let macs = b1_forward_t8(sc, xt, t8, t, yt);
    crate::kernels::transpose_t8_back_avx2(yt, d, t, y, ys, t8);
    let _ = xs;
    macs
}

/// 一层反向（drop-in t8）：产出 `dw`（累加）与 `dx[t][d]`（累加）。
///
/// * `dy` 是 `[t][ys]`；`x` 是 `[t][xs]`；`dx` 是 `[t][xs]`（**调用方先清零**）。
/// * `xt`/`dyt`/`dxt` 各至少 `max(d,n)*t8`（d == n 时一份即可）。
///
/// # Safety 见上。
#[allow(clippy::too_many_arguments)]
pub unsafe fn b1_layer_backward_t8(
    sc: &SparseConnW,
    dy: *const f32,
    ys: usize,
    x: *const f32,
    xs: usize,
    t: usize,
    dw: *mut f32,
    dx: *mut f32,
    xt: *mut f32,
    dyt: *mut f32,
    dxt: *mut f32,
    t8: usize,
) -> usize {
    let n = sc.n_neurons as usize;
    let d = n;
    crate::kernels::transpose_t8_avx2(x, t, d, xt, t8);
    crate::kernels::transpose_t8_avx2(dy, t, n, dyt, t8);
    for i in 0..d * t8 {
        *dxt.add(i) = 0.0;
    }
    crate::kernels::b1_backward_t8_avx2(sc.w, sc.src_u32, xt, dyt, t8, n, sc.k0, dxt, dw);
    crate::kernels::transpose_t8_back_avx2(dxt, d, t, dx, xs, t8);
    let _ = ys;
    n * sc.k0
}
/// **参考实现**：逐位契约版（升序、单累加器、首项直接赋值），纯标量。
///
/// 用途只有一个：`vmlm-b1-eps` / 验收判据拿它算 ε。
/// 生产前向走 `b1_forward` 的 SIMD 路径；本函数是**对照物**。
///
/// 为什么必须单独留一个：验收标准从 bit-exact 放宽为 ε 之后，
/// 如果再拿 SIMD 版自己跟自己比，就无法回答「差异到底多少」。
///
/// # Safety
/// 同 `b1_forward`。
pub unsafe fn b1_forward_scalar_reference(
    sc: &SparseConnW,
    x: *const f32,
    y: *mut f32,
) -> usize {
    let n = sc.n_neurons as usize;
    let k0 = sc.k0;
    let mut acc_guard = 0.0f32;
    for i in 0..n {
        let base = i * k0;
        let mut acc = 0.0f32;
        let mut first = true;
        for j in 0..k0 {
            let sid = *sc.src.add(base + j) as usize;
            let term = *sc.w.add(base + j) * *x.add(sid);
            acc = if first { term } else { acc + term };
            first = false;
        }
        *y.add(i) = acc;
        acc_guard += acc;
    }
    // 防止整个循环被优化掉（本项目已四次踩到假零）
    if acc_guard.is_nan() {
        vmlm_sys::io::print(b"B1: nan in scalar reference\n");
    }
    n * k0
}

/// B1 反向：dW[i][j] += dy[i] · x[src[i][j]]，**全部 k0 条都写**。
///
/// 与 `sparse_gemm_bwd_w` 的关系：那一个唯一的算法差异是「非激活连接不写」。
/// B1 没有非激活连接（全部 20 条都激活），所以那条掩码判断的**恒真分支**
/// 被删掉，剩下的循环体与它逐字同构：
///   * 同样是 `+=`（**累加**语义，不是赋值）—— 引擎侧反向指令全是累加，
///     调用方先 OP_ZERO 清零；改成赋值会让一个 batch 里多帧的梯度互相覆盖，
///     而 loss 仍会下降（只是慢），于是这个 bug 极难被发现；
///   * 同样是行 i 外循环、列 j 内循环、`i*k0+j` 线性下标；
///   * 乘法的两个操作数顺序也相同（`g * x`），保证逐位一致。
///
/// 反向**不做向量化**：它的写是 `dw[t] += g * x[sid]`，是**读-改-写**，
/// 向量化要么需要 gather+scatter（AVX2 无 scatter），要么改变累加分组。
/// 前者不可用、后者违反位精确契约，所以保持标量。
/// （bench-builder 的探针也未测反向，这条边界如实标注，不假装已优化。）
///
/// 返回写入的连接数 = n_neurons * k0（B1 下恒等于 E）。
/// 报计数是为了让「全部都算」这条主张**可失败**。
///
/// **不读 `sc.act`。** 反向只需要 (w 不在内) dy/x/src：B1 的梯度不依赖
/// 任何选择结果。注意 `dw` 是梯度，**权重 `sc.w` 不被本函数修改**。
///
/// # Safety
/// 同 `b1_forward`，外加 `dy` 至少 `n_neurons` 个 f32、
/// `dw` 至少 `sc.e` 个 f32 且可写。
pub unsafe fn b1_backward(sc: &SparseConnW, dy: *const f32, x: *const f32, dw: *mut f32) -> usize {
    let n = sc.n_neurons as usize;
    let k0 = sc.k0;
    let mut written = 0usize;
    for i in 0..n {
        let g = *dy.add(i);
        let base = i * k0;
        for j in 0..k0 {
            let sid = *sc.src.add(base + j) as usize;
            *dw.add(base + j) += g * *x.add(sid);
            written += 1;
        }
    }
    written
}
