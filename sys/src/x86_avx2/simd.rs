// ===========================================================================
// vmlm-sys / L0 底层调用层 · x86 AVX2 后端
// ---------------------------------------------------------------------------
// 本文件内容**逐字节搬运**自 core/src/kernels.rs 的 AVX2 原语块
// （`hsum256` + P19 原语 dot/axpy/axpy2/add/max/exp_neg/exp_neg_sum/scale_mul/sum256）。
// 只动了两处：文件头新增 `use crate::math::exp_f;` 与 `use core::arch::x86_64::*;`，`hsum256` 由私有改为 pub
// （跨 crate 必需）。数值口径、累加顺序、循环结构、cfg 全部未动。
// ===========================================================================
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]

use crate::math::exp_f;
use core::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn hsum256(v: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_hadd_ps(s, s);
    let s = _mm_hadd_ps(s, s);
    _mm_cvtss_f32(s)
}

/// `Σ a[i]·b[i]`，4 条独立 FMA 累加链 + 横向归约。
///
/// # Safety
/// `a` / `b` 各至少 `n` 个 f32。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot256_f32(a: *const f32, b: *const f32, n: usize) -> f32 {
    let mut a0 = _mm256_setzero_ps();
    let mut a1 = _mm256_setzero_ps();
    let mut a2 = _mm256_setzero_ps();
    let mut a3 = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 32 <= n {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(i)), _mm256_loadu_ps(b.add(i)), a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(i + 8)), _mm256_loadu_ps(b.add(i + 8)), a1);
        a2 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(i + 16)), _mm256_loadu_ps(b.add(i + 16)), a2);
        a3 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(i + 24)), _mm256_loadu_ps(b.add(i + 24)), a3);
        i += 32;
    }
    while i + 8 <= n {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(i)), _mm256_loadu_ps(b.add(i)), a0);
        i += 8;
    }
    let s = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
    let mut acc = hsum256(s);
    while i < n {
        acc += *a.add(i) * *b.add(i);
        i += 1;
    }
    acc
}

/// 非 x86_64 回退：单累加器标量点积。
///
/// # Safety 同 [`dot256_f32`]。
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn dot256_f32(a: *const f32, b: *const f32, n: usize) -> f32 {
    let mut s = 0.0f32;
    for i in 0..n {
        s += *a.add(i) * *b.add(i);
    }
    s
}

/// `y[i] += a·x[i]`（标量 `a` 广播），8 宽。
///
/// # Safety
/// `y` / `x` 各至少 `n` 个 f32，`y` 可写。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn axpy256_f32(y: *mut f32, x: *const f32, a: f32, n: usize) {
    let va = _mm256_set1_ps(a);
    let mut i = 0usize;
    while i + 32 <= n {
        let y0 = _mm256_loadu_ps(y.add(i));
        let y1 = _mm256_loadu_ps(y.add(i + 8));
        let y2 = _mm256_loadu_ps(y.add(i + 16));
        let y3 = _mm256_loadu_ps(y.add(i + 24));
        _mm256_storeu_ps(y.add(i), _mm256_fmadd_ps(va, _mm256_loadu_ps(x.add(i)), y0));
        _mm256_storeu_ps(y.add(i + 8), _mm256_fmadd_ps(va, _mm256_loadu_ps(x.add(i + 8)), y1));
        _mm256_storeu_ps(y.add(i + 16), _mm256_fmadd_ps(va, _mm256_loadu_ps(x.add(i + 16)), y2));
        _mm256_storeu_ps(y.add(i + 24), _mm256_fmadd_ps(va, _mm256_loadu_ps(x.add(i + 24)), y3));
        i += 32;
    }
    while i + 8 <= n {
        let y0 = _mm256_loadu_ps(y.add(i));
        _mm256_storeu_ps(y.add(i), _mm256_fmadd_ps(va, _mm256_loadu_ps(x.add(i)), y0));
        i += 8;
    }
    while i < n {
        *y.add(i) += a * *x.add(i);
        i += 1;
    }
}

/// 非 x86_64 回退。 # Safety 同 [`axpy256_f32`]。
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn axpy256_f32(y: *mut f32, x: *const f32, a: f32, n: usize) {
    for i in 0..n {
        *y.add(i) += a * *x.add(i);
    }
}

/// 双路 `y0 += a0·x0` 与 `y1 += a1·x1` 一次遍历完成（attention 反向 passB 用）。
/// 两条链无依赖 ⇒ 单次遍历拿到 2 倍 ILP，且省一半循环开销。
///
/// # Safety
/// 四个指针各至少 `n` 个 f32，`y0` / `y1` 可写。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn axpy2_256_f32(
    y0: *mut f32,
    x0: *const f32,
    a0: f32,
    y1: *mut f32,
    x1: *const f32,
    a1: f32,
    n: usize,
) {
    let va0 = _mm256_set1_ps(a0);
    let va1 = _mm256_set1_ps(a1);
    let mut i = 0usize;
    while i + 16 <= n {
        let p0 = _mm256_fmadd_ps(va0, _mm256_loadu_ps(x0.add(i)), _mm256_loadu_ps(y0.add(i)));
        let p1 = _mm256_fmadd_ps(va1, _mm256_loadu_ps(x1.add(i)), _mm256_loadu_ps(y1.add(i)));
        let q0 = _mm256_fmadd_ps(va0, _mm256_loadu_ps(x0.add(i + 8)), _mm256_loadu_ps(y0.add(i + 8)));
        let q1 = _mm256_fmadd_ps(va1, _mm256_loadu_ps(x1.add(i + 8)), _mm256_loadu_ps(y1.add(i + 8)));
        _mm256_storeu_ps(y0.add(i), p0);
        _mm256_storeu_ps(y1.add(i), p1);
        _mm256_storeu_ps(y0.add(i + 8), q0);
        _mm256_storeu_ps(y1.add(i + 8), q1);
        i += 16;
    }
    while i + 8 <= n {
        let p0 = _mm256_fmadd_ps(va0, _mm256_loadu_ps(x0.add(i)), _mm256_loadu_ps(y0.add(i)));
        let p1 = _mm256_fmadd_ps(va1, _mm256_loadu_ps(x1.add(i)), _mm256_loadu_ps(y1.add(i)));
        _mm256_storeu_ps(y0.add(i), p0);
        _mm256_storeu_ps(y1.add(i), p1);
        i += 8;
    }
    while i < n {
        *y0.add(i) += a0 * *x0.add(i);
        *y1.add(i) += a1 * *x1.add(i);
        i += 1;
    }
}

/// 非 x86_64 回退。 # Safety 同 [`axpy2_256_f32`]。
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn axpy2_256_f32(
    y0: *mut f32,
    x0: *const f32,
    a0: f32,
    y1: *mut f32,
    x1: *const f32,
    a1: f32,
    n: usize,
) {
    for i in 0..n {
        *y0.add(i) += a0 * *x0.add(i);
        *y1.add(i) += a1 * *x1.add(i);
    }
}

/// `y[i] += x[i]`，8 宽。
///
/// # Safety `y` / `x` 各至少 `n` 个 f32，`y` 可写。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn add256_f32(y: *mut f32, x: *const f32, n: usize) {
    let mut i = 0usize;
    while i + 32 <= n {
        _mm256_storeu_ps(y.add(i), _mm256_add_ps(_mm256_loadu_ps(y.add(i)), _mm256_loadu_ps(x.add(i))));
        _mm256_storeu_ps(y.add(i + 8), _mm256_add_ps(_mm256_loadu_ps(y.add(i + 8)), _mm256_loadu_ps(x.add(i + 8))));
        _mm256_storeu_ps(y.add(i + 16), _mm256_add_ps(_mm256_loadu_ps(y.add(i + 16)), _mm256_loadu_ps(x.add(i + 16))));
        _mm256_storeu_ps(y.add(i + 24), _mm256_add_ps(_mm256_loadu_ps(y.add(i + 24)), _mm256_loadu_ps(x.add(i + 24))));
        i += 32;
    }
    while i + 8 <= n {
        _mm256_storeu_ps(y.add(i), _mm256_add_ps(_mm256_loadu_ps(y.add(i)), _mm256_loadu_ps(x.add(i))));
        i += 8;
    }
    while i < n {
        *y.add(i) += *x.add(i);
        i += 1;
    }
}

/// 非 x86_64 回退。 # Safety 同 [`add256_f32`]。
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn add256_f32(y: *mut f32, x: *const f32, n: usize) {
    for i in 0..n {
        *y.add(i) += *x.add(i);
    }
}

/// `max(x[0..n])`，8 宽。空切片返回 `-inf`。
///
/// # Safety `x` 至少 `n` 个 f32。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn max256_f32(x: *const f32, n: usize) -> f32 {
    let mut m = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut i = 0usize;
    while i + 32 <= n {
        m = _mm256_max_ps(m, _mm256_loadu_ps(x.add(i)));
        m = _mm256_max_ps(m, _mm256_loadu_ps(x.add(i + 8)));
        m = _mm256_max_ps(m, _mm256_loadu_ps(x.add(i + 16)));
        m = _mm256_max_ps(m, _mm256_loadu_ps(x.add(i + 24)));
        i += 32;
    }
    while i + 8 <= n {
        m = _mm256_max_ps(m, _mm256_loadu_ps(x.add(i)));
        i += 8;
    }
    let lo = _mm256_castps256_ps128(m);
    let hi = _mm256_extractf128_ps(m, 1);
    let s = _mm_max_ps(lo, hi);
    let s = _mm_max_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_max_ps(s, _mm_shuffle_ps(s, s, 0x55));
    let mut r = _mm_cvtss_f32(s);
    while i < n {
        if *x.add(i) > r {
            r = *x.add(i);
        }
        i += 1;
    }
    r
}

/// 非 x86_64 回退。 # Safety 同 [`max256_f32`]。
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn max256_f32(x: *const f32, n: usize) -> f32 {
    let mut m = f32::NEG_INFINITY;
    for i in 0..n {
        if *x.add(i) > m {
            m = *x.add(i);
        }
    }
    m
}

/// 8 路 `exp(x - mx)`（向量化 `exp_f` 的**半开区间口径**：只保证 `x - mx <= 0`）。
///
/// 与标量 `exp_f` 的关系：
/// * 同一个 Cody-Waite 两段归约 + 同一组 Cephes minimax 系数，但 Horner 用 FMA
///   ⇒ ε 级差异（本文件其余 AVX2 内核同一口径）；
/// * `n <= -127` 段（真值在 f32 次正规区，`x - mx < -87.3365`）**直接给 0**。
///   训练器已开 FTZ/DAZ ⇒ 标量版在那一区的结果同样被冲成 0，两者一致；
///   这也是这里不实现次正规重建的理由（调用点是 softmax，那一区的项无语义贡献）。
///
/// # Safety `x` 至少 `n` 个 f32。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn exp_neg_ps_avx2(x: __m256) -> __m256 {
    const LOG2E: f32 = 1.442_695_0;
    const LN2_HI: f32 = 0.693_145_75;
    const LN2_LO: f32 = 1.428_606_8e-6;
    // (a) 下界夹紧：与 exp_f 的 -88 夹紧同义（更小的一律 0，见上）
    let lo = _mm256_set1_ps(-88.0);
    let hi = _mm256_set1_ps(0.0);
    let x = _mm256_min_ps(_mm256_max_ps(x, lo), hi);
    // (b) n = round(x·log2e)，round-to-nearest-even（与 round_f32 同舍入）
    let nf = _mm256_round_ps(
        _mm256_mul_ps(x, _mm256_set1_ps(LOG2E)),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
    );
    // (c) 两段式残差
    let r = _mm256_sub_ps(
        _mm256_sub_ps(x, _mm256_mul_ps(nf, _mm256_set1_ps(LN2_HI))),
        _mm256_mul_ps(nf, _mm256_set1_ps(LN2_LO)),
    );
    // (d) minimax（FMA Horner，与 exp_f 同系数）
    let mut p = _mm256_set1_ps(1.987_569_1e-4);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.398_199_9e-3));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.333_452e-3));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.166_579_7e-2));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.666_666_5e-1));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.5));
    let r2 = _mm256_mul_ps(r, r);
    p = _mm256_fmadd_ps(p, r2, _mm256_add_ps(r, _mm256_set1_ps(1.0)));
    // (e) 2^n（n ∈ [-126, 0]，正规数域内）
    let ni = _mm256_cvtps_epi32(nf);
    let bits = _mm256_slli_epi32(_mm256_add_epi32(ni, _mm256_set1_epi32(127)), 23);
    let scale = _mm256_castsi256_ps(bits);
    let mut res = _mm256_mul_ps(p, scale);
    // (f) x - mx < -87.3365 ⇒ 真值在次正规区 ⇒ 0（FTZ 口径）
    let tiny = _mm256_cmp_ps(x, _mm256_set1_ps(-87.336_55), _CMP_GT_OS);
    res = _mm256_and_ps(res, tiny);
    res
}

/// `x[i] = exp(x[i] - mx)`（就地）并返回 Σ，8 宽。
///
/// 就地写回是**契约**：调用方（CE 的 softmax 反向）随后把同一块缓冲当作
/// softmax 概率使用，省掉一次整块搬运。
///
/// # Safety `x` 至少 `n` 个 f32 且可写。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn exp_neg_sum_avx2(x: *mut f32, n: usize, mx: f32) -> f32 {
    let vmx = _mm256_set1_ps(mx);
    let mut s0 = _mm256_setzero_ps();
    let mut s1 = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 16 <= n {
        let e0 = exp_neg_ps_avx2(_mm256_sub_ps(_mm256_loadu_ps(x.add(i)), vmx));
        let e1 = exp_neg_ps_avx2(_mm256_sub_ps(_mm256_loadu_ps(x.add(i + 8)), vmx));
        _mm256_storeu_ps(x.add(i), e0);
        _mm256_storeu_ps(x.add(i + 8), e1);
        s0 = _mm256_add_ps(s0, e0);
        s1 = _mm256_add_ps(s1, e1);
        i += 16;
    }
    while i + 8 <= n {
        let e0 = exp_neg_ps_avx2(_mm256_sub_ps(_mm256_loadu_ps(x.add(i)), vmx));
        _mm256_storeu_ps(x.add(i), e0);
        s0 = _mm256_add_ps(s0, e0);
        i += 8;
    }
    let mut sum = hsum256(_mm256_add_ps(s0, s1));
    while i < n {
        let e = exp_f(*x.add(i) - mx);
        *x.add(i) = e;
        sum += e;
        i += 1;
    }
    sum
}

/// 非 x86_64 回退。 # Safety 同 [`exp_neg_sum_avx2`]。
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn exp_neg_sum_avx2(x: *mut f32, n: usize, mx: f32) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..n {
        let e = exp_f(*x.add(i) - mx);
        *x.add(i) = e;
        sum += e;
    }
    sum
}

/// `dst[i] = a·src[i]`，8 宽。
///
/// # Safety `dst` / `src` 各至少 `n` 个 f32，`dst` 可写。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn scale_mul_avx2(dst: *mut f32, src: *const f32, a: f32, n: usize) {
    let va = _mm256_set1_ps(a);
    let mut i = 0usize;
    while i + 32 <= n {
        _mm256_storeu_ps(dst.add(i), _mm256_mul_ps(va, _mm256_loadu_ps(src.add(i))));
        _mm256_storeu_ps(dst.add(i + 8), _mm256_mul_ps(va, _mm256_loadu_ps(src.add(i + 8))));
        _mm256_storeu_ps(dst.add(i + 16), _mm256_mul_ps(va, _mm256_loadu_ps(src.add(i + 16))));
        _mm256_storeu_ps(dst.add(i + 24), _mm256_mul_ps(va, _mm256_loadu_ps(src.add(i + 24))));
        i += 32;
    }
    while i + 8 <= n {
        _mm256_storeu_ps(dst.add(i), _mm256_mul_ps(va, _mm256_loadu_ps(src.add(i))));
        i += 8;
    }
    while i < n {
        *dst.add(i) = a * *src.add(i);
        i += 1;
    }
}

/// 非 x86_64 回退。 # Safety 同 [`scale_mul_avx2`]。
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn scale_mul_avx2(dst: *mut f32, src: *const f32, a: f32, n: usize) {
    for i in 0..n {
        *dst.add(i) = a * *src.add(i);
    }
}

/// `Σ x[i]`，8 宽（4 条累加链）。
///
/// # Safety `x` 至少 `n` 个 f32。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn sum256_f32(x: *const f32, n: usize) -> f32 {
    let mut s0 = _mm256_setzero_ps();
    let mut s1 = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 32 <= n {
        s0 = _mm256_add_ps(s0, _mm256_loadu_ps(x.add(i)));
        s1 = _mm256_add_ps(s1, _mm256_loadu_ps(x.add(i + 8)));
        s0 = _mm256_add_ps(s0, _mm256_loadu_ps(x.add(i + 16)));
        s1 = _mm256_add_ps(s1, _mm256_loadu_ps(x.add(i + 24)));
        i += 32;
    }
    while i + 8 <= n {
        s0 = _mm256_add_ps(s0, _mm256_loadu_ps(x.add(i)));
        i += 8;
    }
    let mut s = hsum256(_mm256_add_ps(s0, s1));
    while i < n {
        s += *x.add(i);
        i += 1;
    }
    s
}

/// 非 x86_64 回退。 # Safety 同 [`sum256_f32`]。
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn sum256_f32(x: *const f32, n: usize) -> f32 {
    let mut s = 0.0f32;
    for i in 0..n {
        s += *x.add(i);
    }
    s
}
