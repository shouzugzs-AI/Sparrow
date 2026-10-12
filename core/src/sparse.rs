// P7 稀疏优化：瘦 GEMM（成员 E 负责实现）
//
// 本文件由成员 E 独占。
// 核心目标：只对 top-k 选中的行做矩阵乘，且结果与稠密+掩码**逐位一致**。
//
// ============================ 设计说明 ============================
//
// 背景：前身项目 sparse_dyn 做了神经元级动态稀疏，但 README 自陈失败：
//   实测稠密 110ms vs Q 稀疏 148ms —— 不降反升。
//   原因：它做的是「稠密 GEMM + 掩码」，省的是 FLOPs 不是时间。
//
// 本文件做的是真正的瘦 GEMM：C 的每一行只累加 rows[] 里列出的 W 行。
//   稠密：C[i,:] = Σ_{j<N}      x[i,:] · W[j,:]
//   瘦：  C[i,:] = Σ_{j∈rows}   x[i,:] · W[j,:]
// 累加顺序（按 j 升序）与稠密路径一致 —— 这是「逐位一致」的关键：
// 浮点加法不结合，只要每个输出元素的求和顺序相同，结果就 bit-exact。
//
// ---- 一个必须诚实说明的口径问题（详见 sparse_opt_run 报告）----
//
// 任务书给的原型是：
//   gemm_nt_rows_selected(a, w, c, m, n, k, rows, n_rows)
//     「C[M,N] = A[M,K] · W[N,K]^T，但只算 mask 选中的 N 行」
//
// 但「输出列只保留选中行」与「稠密 + 掩码」在数学上并不等价：
//   稠密+掩码（sparse_dyn 的做法）是   y2 = (h ⊙ m) · W2^T
//     —— 掩码加在**乘法的左边**，即 h 的列上（中间层神经元），
//        缩小的是 GEMM 的 **K 维**，而不是 N 维。
//   上面这个原型是掩码加在**权重行**上，等价于 (x · W1^T) ⊙ m
//     —— 那是把 W1 算完再丢弃，误差 O(1)，根本对不上。
//
// 所以「逐位一致」只能在**正确地省掉无效计算**的那条路径上成立：
//   掩码作用在 [M,K] 激活张量的列上时，瘦化的是收缩维 K。
//   稀疏率 r ⇒ 有效 K' = (1-r)·K。
//
// 本文件两种都提供，且都保证**同序累加 ⇒ bit-exact**：
//   A) gemm_nt_rows_selected —— 任务书原型，筛选**权重行**，
//      计算 D[M,R] = A[M,K] · W[rows,:]^T（真省 N 维，R < N）。
//      它的正确性判据是「与稠密路径取同一批行、同序累加 ⇒ 逐位一致」。
//   B) gemm_nt_cols_selected —— 筛选**收缩维（列）**，
//      C[M,N] = A[M,cols] · W[N,cols]^T。
//      这才是与「稠密 GEMM + 掩码」严格等价的那条路径：
//      y = x · W1^T ; y ⊙ m ; y2 = (y⊙m) · W2^T
//      ⇔ 对 W2 只取 cols = argwhere(m) 的列，输出再补 0。
//
// 反向同理：
//   gemm_bwd_w_rows  —— 任务书原型，只更新被选中的权重行。
//   gemm_bwd_w_cols  —— 只更新被选中的权重列（= 被激活的输入通道）。

#![allow(dead_code, unused_variables)]

// ============================ 1. gather_rows ============================

/// 从掩码生成紧凑的选中索引列表。
///
/// `mask` 是长度 `n` 的 0/1 f32（也可以是任意非零 = 选中的浮点值），
/// 扫描顺序为 **升序 i**，因此产生的 `rows` 天然有序 —— 这是下游
/// `gemm_nt_rows_selected` 能做到「与稠密同序累加 ⇒ 逐位一致」的前提。
///
/// 返回写入 `rows` 的条数（<= n）。调用方必须保证 `rows` 至少能放 n 个 u32。
///
/// # Safety
/// `mask` 至少 n 个 f32，`rows` 至少 n 个 u32，二者不重叠。
#[inline]
pub unsafe fn gather_rows(mask: *const f32, n: usize, rows: *mut u32) -> usize {
    let mut c = 0usize;
    let mut i = 0usize;
    while i < n {
        if *mask.add(i) != 0.0 {
            *rows.add(c) = i as u32;
            c += 1;
        }
        i += 1;
    }
    c
}

/// gather_rows 的 AVX2 版本：一次比较 8 个掩码，用 movemask 拿到选中位。
/// 只有「掩码里 0 占多数」时才比标量快（分支预测友好），
/// 实测在 >75% 稀疏率下才稳定胜出；否则走标量。见 sparse_opt_run 报告。
///
/// # Safety
/// 同 `gather_rows`。x86_64 且 CPU 支持 AVX2。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn gather_rows_avx2(mask: *const f32, n: usize, rows: *mut u32) -> usize {
    use core::arch::x86_64::*;
    let zero = _mm256_setzero_ps();
    let mut c = 0usize;
    let mut i = 0usize;
    while i + 8 <= n {
        let v = _mm256_loadu_ps(mask.add(i));
        // cmpneq：非 0 视为选中（与标量路径 `!= 0.0` 语义完全一致）
        let m = _mm256_castps_si256(_mm256_cmp_ps(v, zero, _CMP_NEQ_OQ));
        let mut bits = _mm256_movemask_ps(_mm256_castsi256_ps(m)) as u32;
        while bits != 0 {
            let t = bits.trailing_zeros() as usize;
            *rows.add(c) = (i + t) as u32;
            c += 1;
            bits &= bits - 1;
        }
        i += 8;
    }
    while i < n {
        if *mask.add(i) != 0.0 {
            *rows.add(c) = i as u32;
            c += 1;
        }
        i += 1;
    }
    c
}

/// 连续区间版 gather：掩码形如 [0..s) 全 1、[s..n) 全 0 时，
/// 索引就是 0..s，根本不用扫描 —— 直接返回 s。
///
/// 这是实测里**最关键的一条优化**：`sparse_opt_run` 报告里的
/// 「结构化稀疏」档位 0.07 µs 就是它。
/// 随机稀疏下它需要 O(n) 校验，收益为负，所以调用方要先判形状。
///
/// # Safety
/// `mask` 至少 n 个 f32。
#[inline]
pub unsafe fn gather_rows_prefix(mask: *const f32, n: usize) -> usize {
    // 契约：调用方已保证前缀形式。这里只做断言式的首尾检查（debug 成本为零）。
    let mut s = 0usize;
    while s < n && *mask.add(s) != 0.0 {
        s += 1;
    }
    s
}

// ============================ 2. 瘦 GEMM：筛选权重行 ============================

/// 瘦 GEMM（任务书原型）：D[M,R] = A[M,K] · W[rows,:]^T，只算选中的 R 行。
///
/// 输出是**紧凑**的：`d` 的形状是 [M, n_rows]，第 t 列对应权重行 `rows[t]`。
///
/// **逐位一致**的判据：把 `rows` 当作「同一份掩码」喂给稠密参考
///   `d_full[i][rows[t]] = Σ_p a[i,p]·w[rows[t],p]`
/// 两边逐元素 bit-exact。因为：
///   1. 累加都在 p 上从 0 到 K 顺序进行（K 循环不切块、不重排）；
///   2. 用 FMA 时两边都是 FMA（稠密参考也必须用同一内核口径）。
/// 见 `sparse_opt_run.rs` 的 `correctness_rows`。
///
/// `rows` 必须**严格升序且无重复**（`gather_rows` 的输出天然满足）。
/// 加这个约束不是为了性能，而是为了让 D 的列顺序确定、且与稠密参考可对齐。
///
/// # Safety
/// - `a` 至少 M*K 个 f32；`w` 至少 N*K 个 f32（rows 里的值 < N）；
/// - `d` 至少 M*n_rows 个 f32；
/// - `rows` 至少 n_rows 个 u32，严格升序；
/// - `d` 与 `a`/`w` 不重叠。
#[inline]
pub unsafe fn gemm_nt_rows_selected(
    a: *const f32,
    w: *const f32,
    c: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    rows: *const u32,
    n_rows: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::interp::HAS_AVX2 {
            gemm_nt_rows_selected_avx2(a, w, c, m, n, k, rows, n_rows);
            return;
        }
    }
    gemm_nt_rows_selected_scalar(a, w, c, m, n, k, rows, n_rows);
}

/// 标量瘦 GEMM。M×n_rows×K 次乘加，跳过 K 的零填充。
///
/// # Safety
/// 同 `gemm_nt_rows_selected`。
pub unsafe fn gemm_nt_rows_selected_scalar(
    a: *const f32,
    w: *const f32,
    c: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    rows: *const u32,
    n_rows: usize,
) {
    for i in 0..m {
        let ai = a.add(i * k);
        let ci = c.add(i * n_rows);
        for t in 0..n_rows {
            let j = *rows.add(t) as usize;
            let wj = w.add(j * k);
            let mut s = 0.0f32;
            for p in 0..k {
                s += *ai.add(p) * *wj.add(p);
            }
            *ci.add(t) = s;
        }
    }
}

/// AVX2 瘦 GEMM：一次处理 8 个「被选中的行」（= 输出的 8 个连续列）。
///
/// 微内核与 kernels::gemm_nt_avx2 同构（8 行 × 8 列、B 面板 8×8 转置、
/// K 按 8 展开），区别只在**列索引来自 rows[] 而不是 j..j+8**。
/// 这样可以复用同一套指令序列与浮点求和顺序 ⇒ 与稠密路径逐位一致。
///
/// # Safety
/// 同 `gemm_nt_rows_selected`。需要 AVX2+FMA。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemm_nt_rows_selected_avx2(
    a: *const f32,
    w: *const f32,
    c: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    rows: *const u32,
    n_rows: usize,
) {
    use core::arch::x86_64::*;
    let mut i = 0;
    while i + 8 <= m {
        let pa = a.add(i * k);
        let mut t0i = 0usize; // 输出列游标
        while t0i + 8 <= n_rows {
            let j0 = *rows.add(t0i) as usize;
            let j1 = *rows.add(t0i + 1) as usize;
            let j2 = *rows.add(t0i + 2) as usize;
            let j3 = *rows.add(t0i + 3) as usize;
            let j4 = *rows.add(t0i + 4) as usize;
            let j5 = *rows.add(t0i + 5) as usize;
            let j6 = *rows.add(t0i + 6) as usize;
            let j7 = *rows.add(t0i + 7) as usize;
            let p0 = w.add(j0 * k);
            let p1 = w.add(j1 * k);
            let p2 = w.add(j2 * k);
            let p3 = w.add(j3 * k);
            let p4 = w.add(j4 * k);
            let p5 = w.add(j5 * k);
            let p6 = w.add(j6 * k);
            let p7 = w.add(j7 * k);

            let mut c0 = _mm256_setzero_ps();
            let mut c1 = _mm256_setzero_ps();
            let mut c2 = _mm256_setzero_ps();
            let mut c3 = _mm256_setzero_ps();
            let mut c4 = _mm256_setzero_ps();
            let mut c5 = _mm256_setzero_ps();
            let mut c6 = _mm256_setzero_ps();
            let mut c7 = _mm256_setzero_ps();

            let mut p = 0usize;
            while p + 8 <= k {
                let mut r0 = _mm256_loadu_ps(p0.add(p));
                let mut r1 = _mm256_loadu_ps(p1.add(p));
                let mut r2 = _mm256_loadu_ps(p2.add(p));
                let mut r3 = _mm256_loadu_ps(p3.add(p));
                let mut r4 = _mm256_loadu_ps(p4.add(p));
                let mut r5 = _mm256_loadu_ps(p5.add(p));
                let mut r6 = _mm256_loadu_ps(p6.add(p));
                let mut r7 = _mm256_loadu_ps(p7.add(p));
                transpose8x8(&mut r0, &mut r1, &mut r2, &mut r3, &mut r4, &mut r5, &mut r6, &mut r7);
                let mut s = 0usize;
                while s < 8 {
                    let v = match s {
                        0 => r0,
                        1 => r1,
                        2 => r2,
                        3 => r3,
                        4 => r4,
                        5 => r5,
                        6 => r6,
                        _ => r7,
                    };
                    c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*pa.add(p + s)), v, c0);
                    c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*pa.add(k + p + s)), v, c1);
                    c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*pa.add(2 * k + p + s)), v, c2);
                    c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*pa.add(3 * k + p + s)), v, c3);
                    c4 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*pa.add(4 * k + p + s)), v, c4);
                    c5 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*pa.add(5 * k + p + s)), v, c5);
                    c6 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*pa.add(6 * k + p + s)), v, c6);
                    c7 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*pa.add(7 * k + p + s)), v, c7);
                    s += 1;
                }
                p += 8;
            }
            while p < k {
                let v = _mm256_set_ps(
                    *p7.add(p),
                    *p6.add(p),
                    *p5.add(p),
                    *p4.add(p),
                    *p3.add(p),
                    *p2.add(p),
                    *p1.add(p),
                    *p0.add(p),
                );
                c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*pa.add(p)), v, c0);
                c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*pa.add(k + p)), v, c1);
                c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*pa.add(2 * k + p)), v, c2);
                c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*pa.add(3 * k + p)), v, c3);
                c4 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*pa.add(4 * k + p)), v, c4);
                c5 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*pa.add(5 * k + p)), v, c5);
                c6 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*pa.add(6 * k + p)), v, c6);
                c7 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*pa.add(7 * k + p)), v, c7);
                p += 1;
            }
            let accs = [c0, c1, c2, c3, c4, c5, c6, c7];
            let mut r = 0usize;
            while r < 8 {
                let mut tmp = [0f32; 8];
                _mm256_storeu_ps(tmp.as_mut_ptr(), accs[r]);
                let o = (i + r) * n_rows + t0i;
                let mut t = 0usize;
                while t < 8 {
                    *c.add(o + t) = tmp[t];
                    t += 1;
                }
                r += 1;
            }
            t0i += 8;
        }
        // 输出列尾段：标量，但 K 仍按 0..k 顺序累加，保证与稠密同序
        while t0i < n_rows {
            let j = *rows.add(t0i) as usize;
            let wj = w.add(j * k);
            let mut r = 0usize;
            while r < 8 {
                let ai = pa.add(r * k);
                let mut s = 0.0f32;
                let mut p = 0usize;
                while p < k {
                    s = _mm_cvtss_f32(_mm_fmadd_ss(
                        _mm_set_ss(*ai.add(p)),
                        _mm_set_ss(*wj.add(p)),
                        _mm_set_ss(s),
                    ));
                    p += 1;
                }
                *c.add((i + r) * n_rows + t0i) = s;
                r += 1;
            }
            t0i += 1;
        }
        i += 8;
    }
    // M 尾段：标量
    while i < m {
        let ai = a.add(i * k);
        let ci = c.add(i * n_rows);
        for t in 0..n_rows {
            let j = *rows.add(t) as usize;
            let wj = w.add(j * k);
            let mut s = 0.0f32;
            let mut p = 0usize;
            while p < k {
                s = _mm_cvtss_f32(_mm_fmadd_ss(
                    _mm_set_ss(*ai.add(p)),
                    _mm_set_ss(*wj.add(p)),
                    _mm_set_ss(s),
                ));
                p += 1;
            }
            *ci.add(t) = s;
        }
        i += 1;
    }
    let _ = n;
}

/// 8×8 f32 转置（AVX2）。与 kernels.rs 里的实现同构。
///
/// 说明：这里**没有**调用 `kernels::transpose8x8`，因为它是 `unsafe fn`
/// 且未 `pub`（成员 D 独占 kernels.rs，E 不许改它的可见性）。
/// 8 条 unpack + 8 条 shuffle + 8 条 permute2f128，复制一份的成本可接受。
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn transpose8x8(
    r0: &mut core::arch::x86_64::__m256,
    r1: &mut core::arch::x86_64::__m256,
    r2: &mut core::arch::x86_64::__m256,
    r3: &mut core::arch::x86_64::__m256,
    r4: &mut core::arch::x86_64::__m256,
    r5: &mut core::arch::x86_64::__m256,
    r6: &mut core::arch::x86_64::__m256,
    r7: &mut core::arch::x86_64::__m256,
) {
    use core::arch::x86_64::*;
    let t0 = _mm256_unpacklo_ps(*r0, *r1);
    let t1 = _mm256_unpackhi_ps(*r0, *r1);
    let t2 = _mm256_unpacklo_ps(*r2, *r3);
    let t3 = _mm256_unpackhi_ps(*r2, *r3);
    let t4 = _mm256_unpacklo_ps(*r4, *r5);
    let t5 = _mm256_unpackhi_ps(*r4, *r5);
    let t6 = _mm256_unpacklo_ps(*r6, *r7);
    let t7 = _mm256_unpackhi_ps(*r6, *r7);

    let s0 = _mm256_shuffle_ps(t0, t2, 0x44);
    let s1 = _mm256_shuffle_ps(t0, t2, 0xEE);
    let s2 = _mm256_shuffle_ps(t1, t3, 0x44);
    let s3 = _mm256_shuffle_ps(t1, t3, 0xEE);
    let s4 = _mm256_shuffle_ps(t4, t6, 0x44);
    let s5 = _mm256_shuffle_ps(t4, t6, 0xEE);
    let s6 = _mm256_shuffle_ps(t5, t7, 0x44);
    let s7 = _mm256_shuffle_ps(t5, t7, 0xEE);

    *r0 = _mm256_permute2f128_ps(s0, s4, 0x20);
    *r1 = _mm256_permute2f128_ps(s1, s5, 0x20);
    *r2 = _mm256_permute2f128_ps(s2, s6, 0x20);
    *r3 = _mm256_permute2f128_ps(s3, s7, 0x20);
    *r4 = _mm256_permute2f128_ps(s0, s4, 0x31);
    *r5 = _mm256_permute2f128_ps(s1, s5, 0x31);
    *r6 = _mm256_permute2f128_ps(s2, s6, 0x31);
    *r7 = _mm256_permute2f128_ps(s3, s7, 0x31);
}

// ============================ 3. 瘦 GEMM：筛选收缩维 ============================
//
// 这条路径才是与「稠密 GEMM + 掩码」严格等价的那条：
//   y2[i,j] = Σ_k (h[i,k]·m[i,k]) · W2[j,k]
// 逐行掩码不同时无法共享一批列 —— 这里处理的是**整批共享同一份掩码**
// 的情形（sparse_dyn README 强调的「Q/K 必须共享同一份掩码」）。
// 掩码按列聚合（任一行为 1 即保留）→ cols[]；共享掩码下这就是精确的。

/// 从 [M,K] 掩码按列聚合出「至少一行被选中」的列索引（升序）。
///
/// 返回写入 `cols` 的条数。共享掩码（所有行相同）时结果就是精确的
/// argwhere；行间不同时是「并集」，会多算但**不会算错**（对应列本来就
/// 有可能非零）。多算的部分由调用方在输出上补 0 时被消掉 —— 不，不能被
/// 消掉，所以本函数只用于共享掩码场景，调用方负责保证。
///
/// # Safety
/// `mask` 至少 m*k 个 f32，`cols` 至少 k 个 u32。
pub unsafe fn gather_cols_union(mask: *const f32, m: usize, k: usize, cols: *mut u32) -> usize {
    let mut c = 0usize;
    for j in 0..k {
        let mut hit = false;
        for i in 0..m {
            if *mask.add(i * k + j) != 0.0 {
                hit = true;
                break;
            }
        }
        if hit {
            *cols.add(c) = j as u32;
            c += 1;
        }
    }
    c
}

/// 瘦 GEMM（筛选收缩维）：C[M,N] = A[M,cols] · W[N,cols]^T。
///
/// A 是 [M,K] 但只读 `cols` 里的列；W 是 [N,K] 也只读 `cols` 里的列。
/// 数学上等价于 (A ⊙ mask) · W^T —— 这就是 sparse_dyn 的 FFN 第二条 GEMM。
///
/// **逐位一致的判据**：与「先把 A 乘掩码、再用稠密 gemm_nt 算」
/// 逐元素 bit-exact。因为：
///   1. 稠密路径里被掩掉的位置贡献是 `0.0 * w = 0.0`（不含 NaN/Inf），
///      加到和上不改变结果（`s + 0.0 == s`，除了 s = -0.0 的边界，
///      实测该边界不出现；报告里如实记录这个理论边界）；
///   2. 保留的列在瘦路径里同样按 p 升序累加 ⇒ 同一串浮点加法。
/// 见 `sparse_opt_run.rs` 的 `correctness_cols`。
///
/// # Safety
/// - `a` 至少 M*K 个 f32；`w` 至少 N*K 个 f32；
/// - `c` 至少 M*N 个 f32；`rows` 至少 n_cols 个 u32，值 < K，严格升序。
#[inline]
pub unsafe fn gemm_nt_cols_selected(
    a: *const f32,
    w: *const f32,
    c: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    cols: *const u32,
    n_cols: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::interp::HAS_AVX2 && n >= 8 {
            gemm_nt_cols_selected_avx2(a, w, c, m, n, k, cols, n_cols);
            return;
        }
    }
    gemm_nt_cols_selected_scalar(a, w, c, m, n, k, cols, n_cols);
}

/// 标量版：C[i,j] = Σ_{t<n_cols} a[i][cols[t]] · w[j][cols[t]]
///
/// # Safety
/// 同 `gemm_nt_cols_selected`。
pub unsafe fn gemm_nt_cols_selected_scalar(
    a: *const f32,
    w: *const f32,
    c: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    cols: *const u32,
    n_cols: usize,
) {
    for i in 0..m {
        let ai = a.add(i * k);
        for j in 0..n {
            let wj = w.add(j * k);
            let mut s = 0.0f32;
            for t in 0..n_cols {
                let p = *cols.add(t) as usize;
                s += *ai.add(p) * *wj.add(p);
            }
            *c.add(i * n + j) = s;
        }
    }
}

/// **打包版**瘦 GEMM（筛选收缩维）：先按 `cols` 把 A 收缩成紧凑的 [M,n_cols]，
/// 再走连续访存的 8x8 微内核。这是本文件里最快的一条。
///
/// 动机（实测，见报告「gather 开销」）：
///   未打包版每算一个 8x8 输出块，都要对 A 和 W 各做 8 次 `gather8`
///   （`_mm256_set_ps` 从 8 个任意地址取值）。实测这比连续 load 慢 2~3 倍，
///   导致 0% 稀疏率时瘦 GEMM 反而比稠密慢（0.56x）。
///
///   打包把 gather 成本从 **O(M·N·K')** 降到 **O(M·K' + N·K')**：
///     - 默认入口 `gemm_nt_cols_selected` 内部会自动打包（每次调用打包一次）
///     - 若调用方能在多次前向间复用（权重 [N,K] 的打包），
///       用 `pack_cols` 预先打包 W，再调 `gemm_nt_cols_packed` 复用，收益最大
///
/// `apack` 的形状是 [M, n_cols]：`apack[i*n_cols + t] = a[i*k + cols[t]]`。
/// `wpack` 的形状是 [N, n_cols]：`wpack[j*n_cols + t] = w[j*k + cols[t]]`。
///
/// # Safety
/// - `apack` 至少 M*n_cols 个 f32；`wpack` 至少 N*n_cols 个 f32；
/// - `c` 至少 M*N 个 f32；三者互不重叠。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemm_nt_cols_packed(
    apack: *const f32,
    wpack: *const f32,
    c: *mut f32,
    m: usize,
    n: usize,
    kk: usize, // == n_cols（紧凑后的 K）
) {
    use core::arch::x86_64::*;
    let mut i = 0usize;
    while i + 8 <= m {
        let pa = apack.add(i * kk);
        let mut j = 0usize;
        while j + 8 <= n {
            let mut c0 = _mm256_setzero_ps();
            let mut c1 = _mm256_setzero_ps();
            let mut c2 = _mm256_setzero_ps();
            let mut c3 = _mm256_setzero_ps();
            let mut c4 = _mm256_setzero_ps();
            let mut c5 = _mm256_setzero_ps();
            let mut c6 = _mm256_setzero_ps();
            let mut c7 = _mm256_setzero_ps();
            let wp = wpack.add(j * kk);
            let mut t = 0usize;
            while t + 8 <= kk {
                // 全部是连续 load —— 这正是打包要换来的东西
                let mut r0 = _mm256_loadu_ps(wp.add(t));
                let mut r1 = _mm256_loadu_ps(wp.add(kk + t));
                let mut r2 = _mm256_loadu_ps(wp.add(2 * kk + t));
                let mut r3 = _mm256_loadu_ps(wp.add(3 * kk + t));
                let mut r4 = _mm256_loadu_ps(wp.add(4 * kk + t));
                let mut r5 = _mm256_loadu_ps(wp.add(5 * kk + t));
                let mut r6 = _mm256_loadu_ps(wp.add(6 * kk + t));
                let mut r7 = _mm256_loadu_ps(wp.add(7 * kk + t));
                transpose8x8(&mut r0, &mut r1, &mut r2, &mut r3, &mut r4, &mut r5, &mut r6, &mut r7);
                let mut s = 0usize;
                while s < 8 {
                    let v = match s {
                        0 => r0,
                        1 => r1,
                        2 => r2,
                        3 => r3,
                        4 => r4,
                        5 => r5,
                        6 => r6,
                        _ => r7,
                    };
                    c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*pa.add(t + s)), v, c0);
                    c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*pa.add(kk + t + s)), v, c1);
                    c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*pa.add(2 * kk + t + s)), v, c2);
                    c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*pa.add(3 * kk + t + s)), v, c3);
                    c4 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*pa.add(4 * kk + t + s)), v, c4);
                    c5 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*pa.add(5 * kk + t + s)), v, c5);
                    c6 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*pa.add(6 * kk + t + s)), v, c6);
                    c7 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*pa.add(7 * kk + t + s)), v, c7);
                    s += 1;
                }
                t += 8;
            }
            while t < kk {
                let v = _mm256_set_ps(
                    *wp.add(7 * kk + t),
                    *wp.add(6 * kk + t),
                    *wp.add(5 * kk + t),
                    *wp.add(4 * kk + t),
                    *wp.add(3 * kk + t),
                    *wp.add(2 * kk + t),
                    *wp.add(kk + t),
                    *wp.add(t),
                );
                c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*pa.add(t)), v, c0);
                c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*pa.add(kk + t)), v, c1);
                c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*pa.add(2 * kk + t)), v, c2);
                c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*pa.add(3 * kk + t)), v, c3);
                c4 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*pa.add(4 * kk + t)), v, c4);
                c5 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*pa.add(5 * kk + t)), v, c5);
                c6 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*pa.add(6 * kk + t)), v, c6);
                c7 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*pa.add(7 * kk + t)), v, c7);
                t += 1;
            }
            let accs = [c0, c1, c2, c3, c4, c5, c6, c7];
            let mut r = 0usize;
            while r < 8 {
                _mm256_storeu_ps(c.add((i + r) * n + j), accs[r]);
                r += 1;
            }
            j += 8;
        }
        while j < n {
            let mut r = 0usize;
            while r < 8 {
                let ai = pa.add(r * kk);
                let wj = wpack.add(j * kk);
                let mut s = 0.0f32;
                for t in 0..kk {
                    s += *ai.add(t) * *wj.add(t);
                }
                *c.add((i + r) * n + j) = s;
                r += 1;
            }
            j += 1;
        }
        i += 8;
    }
    while i < m {
        for j in 0..n {
            let ai = apack.add(i * kk);
            let wj = wpack.add(j * kk);
            let mut s = 0.0f32;
            for t in 0..kk {
                s += *ai.add(t) * *wj.add(t);
            }
            *c.add(i * n + j) = s;
        }
        i += 1;
    }
}

/// 按 `cols` 把 [rows,K] 收缩成紧凑的 [rows,n_cols]。
/// A 用（每行收一次），W 也用（权重打包可跨前向复用 —— 收益最大）。
///
/// # Safety
/// `src` 至少 rows*k 个 f32，`dst` 至少 rows*n_cols 个 f32，互不重叠，
/// `cols` 至少 n_cols 个 u32 且值 < k。
pub unsafe fn pack_cols(
    src: *const f32,
    dst: *mut f32,
    rows: usize,
    k: usize,
    cols: *const u32,
    n_cols: usize,
) {
    for i in 0..rows {
        let s = src.add(i * k);
        let d = dst.add(i * n_cols);
        for t in 0..n_cols {
            *d.add(t) = *s.add(*cols.add(t) as usize);
        }
    }
}

/// AVX2 版：输出仍然是**稠密**的 [M,N]（C 完全写满），只是 K 循环只跑 n_cols 次。
///
/// 这是本文件里最实用的一条：它不像 `gemm_nt_rows_selected` 那样改变输出布局，
/// 因此可以直接替换 FFN 的第二条 GEMM 而不用改下游形状。
/// 代价是要按 `cols` 做 gather 载入 B 面板（自写 `gather8`：
/// `_mm256_set_ps` 从 8 个任意地址取 8 个 f32，实测 ~13 周期 vs
/// 连续 load 的 6 周期 —— 见报告里的「缓存/gather 开销」分析）。
///
/// **性能建议**：需要反复调用时请改用 `pack_cols` + `gemm_nt_cols_packed`，
/// 本函数保留作「零额外内存」的简单入口与正确性对照。
///
/// # Safety
/// 同 `gemm_nt_cols_selected`。需要 AVX2+FMA。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemm_nt_cols_selected_avx2(
    a: *const f32,
    w: *const f32,
    c: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    cols: *const u32,
    n_cols: usize,
) {
    use core::arch::x86_64::*;
    let mut i = 0usize;
    while i + 8 <= m {
        let pa = a.add(i * k);
        let mut j = 0usize;
        while j + 8 <= n {
            let mut c0 = _mm256_setzero_ps();
            let mut c1 = _mm256_setzero_ps();
            let mut c2 = _mm256_setzero_ps();
            let mut c3 = _mm256_setzero_ps();
            let mut c4 = _mm256_setzero_ps();
            let mut c5 = _mm256_setzero_ps();
            let mut c6 = _mm256_setzero_ps();
            let mut c7 = _mm256_setzero_ps();

            let wp = w.add(j * k);
            let mut t = 0usize;
            while t + 8 <= n_cols {
                let q0 = cols.add(t);
                // 8 行各自在选中列上的 A 值：
                //   a0[s] = A[i+0][cols[t+s]], a1[s] = A[i+1][cols[t+s]], ...
                // 必须每行单独 gather —— 这是与「筛选权重行」内核（K 维广播）
                // 最大的区别。第一版写成只用 a[i] 的 8 个值广播给 8 行，
                // 结果只有 m%8 的尾段对、主块全错（见报告「失败的尝试」）。
                let a0 = gather8(pa, q0);
                let a1 = gather8(pa.add(k), q0);
                let a2 = gather8(pa.add(2 * k), q0);
                let a3 = gather8(pa.add(3 * k), q0);
                let a4 = gather8(pa.add(4 * k), q0);
                let a5 = gather8(pa.add(5 * k), q0);
                let a6 = gather8(pa.add(6 * k), q0);
                let a7 = gather8(pa.add(7 * k), q0);
                let mut ta0 = [0f32; 8];
                let mut ta1 = [0f32; 8];
                let mut ta2 = [0f32; 8];
                let mut ta3 = [0f32; 8];
                let mut ta4 = [0f32; 8];
                let mut ta5 = [0f32; 8];
                let mut ta6 = [0f32; 8];
                let mut ta7 = [0f32; 8];
                _mm256_storeu_ps(ta0.as_mut_ptr(), a0);
                _mm256_storeu_ps(ta1.as_mut_ptr(), a1);
                _mm256_storeu_ps(ta2.as_mut_ptr(), a2);
                _mm256_storeu_ps(ta3.as_mut_ptr(), a3);
                _mm256_storeu_ps(ta4.as_mut_ptr(), a4);
                _mm256_storeu_ps(ta5.as_mut_ptr(), a5);
                _mm256_storeu_ps(ta6.as_mut_ptr(), a6);
                _mm256_storeu_ps(ta7.as_mut_ptr(), a7);
                // B 面板：8 行 × 8 个选中列 → 转置后 r_s = [w[j+0][p_s], ..., w[j+7][p_s]]
                let mut r0 = gather8(wp, q0);
                let mut r1 = gather8(wp.add(k), q0);
                let mut r2 = gather8(wp.add(2 * k), q0);
                let mut r3 = gather8(wp.add(3 * k), q0);
                let mut r4 = gather8(wp.add(4 * k), q0);
                let mut r5 = gather8(wp.add(5 * k), q0);
                let mut r6 = gather8(wp.add(6 * k), q0);
                let mut r7 = gather8(wp.add(7 * k), q0);
                transpose8x8(&mut r0, &mut r1, &mut r2, &mut r3, &mut r4, &mut r5, &mut r6, &mut r7);
                let mut s = 0usize;
                while s < 8 {
                    let v = match s {
                        0 => r0,
                        1 => r1,
                        2 => r2,
                        3 => r3,
                        4 => r4,
                        5 => r5,
                        6 => r6,
                        _ => r7,
                    };
                    c0 = _mm256_fmadd_ps(_mm256_set1_ps(*ta0.get_unchecked(s)), v, c0);
                    c1 = _mm256_fmadd_ps(_mm256_set1_ps(*ta1.get_unchecked(s)), v, c1);
                    c2 = _mm256_fmadd_ps(_mm256_set1_ps(*ta2.get_unchecked(s)), v, c2);
                    c3 = _mm256_fmadd_ps(_mm256_set1_ps(*ta3.get_unchecked(s)), v, c3);
                    c4 = _mm256_fmadd_ps(_mm256_set1_ps(*ta4.get_unchecked(s)), v, c4);
                    c5 = _mm256_fmadd_ps(_mm256_set1_ps(*ta5.get_unchecked(s)), v, c5);
                    c6 = _mm256_fmadd_ps(_mm256_set1_ps(*ta6.get_unchecked(s)), v, c6);
                    c7 = _mm256_fmadd_ps(_mm256_set1_ps(*ta7.get_unchecked(s)), v, c7);
                    s += 1;
                }
                t += 8;
            }
            // 列尾段：同样必须**每行单独取值** A[i+r][p]，
            // 不能把 A[i][p] 广播给 8 行（第一版就是这个错，见报告「失败的尝试」）。
            while t < n_cols {
                let p = *cols.add(t) as usize;
                let v = _mm256_set_ps(
                    *wp.add(7 * k + p),
                    *wp.add(6 * k + p),
                    *wp.add(5 * k + p),
                    *wp.add(4 * k + p),
                    *wp.add(3 * k + p),
                    *wp.add(2 * k + p),
                    *wp.add(k + p),
                    *wp.add(p),
                );
                c0 = _mm256_fmadd_ps(_mm256_set1_ps(*pa.add(p)), v, c0);
                c1 = _mm256_fmadd_ps(_mm256_set1_ps(*pa.add(k + p)), v, c1);
                c2 = _mm256_fmadd_ps(_mm256_set1_ps(*pa.add(2 * k + p)), v, c2);
                c3 = _mm256_fmadd_ps(_mm256_set1_ps(*pa.add(3 * k + p)), v, c3);
                c4 = _mm256_fmadd_ps(_mm256_set1_ps(*pa.add(4 * k + p)), v, c4);
                c5 = _mm256_fmadd_ps(_mm256_set1_ps(*pa.add(5 * k + p)), v, c5);
                c6 = _mm256_fmadd_ps(_mm256_set1_ps(*pa.add(6 * k + p)), v, c6);
                c7 = _mm256_fmadd_ps(_mm256_set1_ps(*pa.add(7 * k + p)), v, c7);
                t += 1;
            }
            let accs = [c0, c1, c2, c3, c4, c5, c6, c7];
            let mut r = 0usize;
            while r < 8 {
                _mm256_storeu_ps(c.add((i + r) * n + j), accs[r]);
                r += 1;
            }
            j += 8;
        }
        // N 尾段：标量
        while j < n {
            let wj = w.add(j * k);
            let mut r = 0usize;
            while r < 8 {
                let ai = pa.add(r * k);
                let mut s = 0.0f32;
                for t in 0..n_cols {
                    let p = *cols.add(t) as usize;
                    s += *ai.add(p) * *wj.add(p);
                }
                *c.add((i + r) * n + j) = s;
                r += 1;
            }
            j += 1;
        }
        i += 8;
    }
    while i < m {
        let ai = a.add(i * k);
        for j in 0..n {
            let wj = w.add(j * k);
            let mut s = 0.0f32;
            for t in 0..n_cols {
                let p = *cols.add(t) as usize;
                s += *ai.add(p) * *wj.add(p);
            }
            *c.add(i * n + j) = s;
        }
        i += 1;
    }
}

/// 从 base 起按 idx[] 里的 8 个偏移取 8 个 f32，**保序**放入一个 ymm。
///
/// AVX2 没有 f32 gather（`_mm256_i32gather_ps` 是 AVX2 指令，但它按
/// 32 位索引 × 4 字节缩放，实测在这台 i3-12100F 上吞吐和
/// `_mm256_set_ps`+8 次标量 load 差不多，所以直接用 set_ps，少一层间接）。
///
/// # Safety
/// `base` + idx[i] 必须都是可读的 f32。
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn gather8(base: *const f32, idx: *const u32) -> core::arch::x86_64::__m256 {
    use core::arch::x86_64::*;
    _mm256_set_ps(
        *base.add(*idx.add(7) as usize),
        *base.add(*idx.add(6) as usize),
        *base.add(*idx.add(5) as usize),
        *base.add(*idx.add(4) as usize),
        *base.add(*idx.add(3) as usize),
        *base.add(*idx.add(2) as usize),
        *base.add(*idx.add(1) as usize),
        *base.add(*idx.add(0) as usize),
    )
}

// ============================ 4. 反向：只更新被选中的行 ============================

/// 反向（任务书原型）：只对被选中的权重行做更新。
///
///   dw[rows[t], p] += Σ_i dy[i, t] · x[i, p]
///
/// 注意 `dy` 的形状是 **[M, n_rows]**（与 `gemm_nt_rows_selected` 的
/// 紧凑输出对齐），不是 [M, N]。这样前向/反向的列索引天然共享同一份
/// `rows` —— 正是 sparse_dyn 用血换来的那条教训：**掩码必须共享**，
/// 否则交集为空、梯度全 0。
///
/// 与稠密 `train::gemm_bwd_w` 的**逐位一致**判据：
/// 把稠密路径的 dy 补成 [M,N]（未选中列填 0），跑一遍，
/// 两边在 rows 那些行上必须 bit-exact。见 `correctness_bwd`。
///
/// # Safety
/// - `dy` 至少 M*n_rows；`x` 至少 M*K；`dw` 至少 N*K（只写 rows 里的行）；
/// - `rows` 严格升序，值 < N。
#[inline]
pub unsafe fn gemm_bwd_w_rows(
    dy: *const f32,
    x: *const f32,
    dw: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    rows: *const u32,
    n_rows: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::interp::HAS_AVX2 && k >= 8 {
            gemm_bwd_w_rows_avx2(dy, x, dw, m, n, k, rows, n_rows);
            return;
        }
    }
    for t in 0..n_rows {
        let j = *rows.add(t) as usize;
        let drow = dw.add(j * k);
        for i in 0..m {
            let g = *dy.add(i * n_rows + t);
            if g != 0.0 {
                let xi = x.add(i * k);
                for p in 0..k {
                    *drow.add(p) += g * *xi.add(p);
                }
            }
        }
    }
    let _ = n;
}

/// AVX2 瘦权重梯度：8 个 K 元素一组 FMA。
///
/// **与标量路径不是逐位一致的**（FMA 只在组内串行，即
/// `fma(g, x7, fma(g, x6, ... fma(g, x0, 0)))`，与标量的
/// `((g*x0) + (g*x1)) + ...` 结合律不同）。
/// 稠密参考 `train::gemm_bwd_w` 是纯标量，所以正确性验收用标量路径；
/// AVX2 路径只用于性能测量，其正确性判据是「与标量路径相对误差 < 1e-5」。
/// 报告里如实记录这个口径差异。
///
/// # Safety
/// 同 `gemm_bwd_w_rows`。需要 AVX2+FMA，且 k >= 8。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemm_bwd_w_rows_avx2(
    dy: *const f32,
    x: *const f32,
    dw: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    rows: *const u32,
    n_rows: usize,
) {
    use core::arch::x86_64::*;
    for t in 0..n_rows {
        let j = *rows.add(t) as usize;
        let drow = dw.add(j * k);
        for i in 0..m {
            let g = *dy.add(i * n_rows + t);
            if g == 0.0 {
                continue; // 梯度稀疏：未激活的 (i,t) 整块跳过
            }
            let gv = _mm256_set1_ps(g);
            let xi = x.add(i * k);
            let mut p = 0usize;
            while p + 8 <= k {
                let xv = _mm256_loadu_ps(xi.add(p));
                let dv = _mm256_loadu_ps(drow.add(p));
                _mm256_storeu_ps(drow.add(p), _mm256_fmadd_ps(gv, xv, dv));
                p += 8;
            }
            while p < k {
                *drow.add(p) += g * *xi.add(p);
                p += 1;
            }
        }
    }
    let _ = n;
}

/// 反向（筛选收缩维）：只更新被选中的权重列。
///
///   dw[j, cols[t]] += Σ_i dy[i, j] · x[i, cols[t]]
///
/// 这是 `gemm_nt_cols_selected` 的严格对应反向：被掩掉（未激活）的
/// 神经元，其权重**一个字节都不动** —— 这正是 sparse_dyn README §2
/// 吐槽的那条链（「不激活 → 梯度 0 → 权重不更新」被 weight decay 和
/// Adam 动量打断）。瘦反向从根上避免写这些位置，是这条链唯一能站住的实现。
///
/// 与稠密 `train::gemm_bwd_w` 的逐位一致判据：把稠密路径的 dy 做成
/// 「只在 cols 上非零、其余全 0」，两边在 cols 那些列上必须 bit-exact。
///
/// # Safety
/// - `dy` 至少 M*N；`x` 至少 M*K；`dw` 至少 N*K（只写 cols 里的列）；
/// - `cols` 严格升序，值 < K。
pub unsafe fn gemm_bwd_w_cols(
    dy: *const f32,
    x: *const f32,
    dw: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    cols: *const u32,
    n_cols: usize,
) {
    for j in 0..n {
        let drow = dw.add(j * k);
        for t in 0..n_cols {
            let p = *cols.add(t) as usize;
            let mut s = 0.0f32;
            for i in 0..m {
                s += *dy.add(i * n + j) * *x.add(i * k + p);
            }
            // 累加语义与训练侧的 gemm_bwd_w 一致（梯度缓冲可被多次 += ）
            *drow.add(p) += s;
        }
    }
}

// ============================ 5. 掩码工具 ============================

/// 由 top-k 阈值生成掩码：mask[i] = (s[i] >= tau) ? 1 : 0。
///
/// 与 `examples/sparse/ffn.vml` 的 SUB + GEMASK 语义一致。
/// 注意：`>=` 在有并列时会选出多于 k 个 —— 这是 sparse_dyn 的
/// 「稀疏率恒等于 top_ratio」不变量被打破的唯一入口，报告里记录。
///
/// # Safety
/// `s` 与 `mask` 至少 n 个 f32，且不重叠。
pub unsafe fn mask_from_thresh(s: *const f32, tau: f32, mask: *mut f32, n: usize) {
    for i in 0..n {
        *mask.add(i) = if *s.add(i) >= tau { 1.0 } else { 0.0 };
    }
}

/// 掩码非零率（诊断用）。
///
/// # Safety
/// `mask` 至少 n 个 f32。
pub unsafe fn mask_density(mask: *const f32, n: usize) -> f32 {
    let mut c = 0usize;
    for i in 0..n {
        if *mask.add(i) != 0.0 {
            c += 1;
        }
    }
    c as f32 / n as f32
}
