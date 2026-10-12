//! L2a 算子层：GEMM / matvec / attention / norm / activation / loss 等内核。
//! 从 `core/src/kernels.rs` **逐字节搬运**；跨 crate 适配只有两处：
//! * 文件头三行 `use`（原文件在 crate 级已有 simd/math/dtype 的可见性）
//! * 通过 crate 根的兼容 shim 保持 `use crate::format;` / `crate::interp::*` / `crate::alloc` 等**逐字不变**
#![allow(clippy::too_many_arguments)]
#![allow(clippy::missing_safety_doc)]

use vmlm_sys::math::*;
use vmlm_sys::x86_avx2::simd::*;
use vmlm_tensor::dtype::*;

// AVX2 内核：所有数学都在这里。指令流只做调度。
// 纯 CPU、单线程、无防御性检查。AVX-512 走 feature 开关（见白皮书 §0）。
use crate::format;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

// ---------------------------------------------------------------------------
// [抽库·第一轮] L0 已独立成 crate `vmlm-sys`：AVX2 原语 + hsum256 搬去
// `vmlm-sys/src/x86_avx2/simd.rs`（逐字节搬运）。这里用 pub use 重导出，
// 因此 `vmlm_core::kernels::<原语>` 的既有路径与所有调用点**完全不变**。
pub use vmlm_sys::x86_avx2::simd::*;
pub use vmlm_tensor::dtype::*;
pub use vmlm_sys::math::*;


// ---------- 标量激活 ----------
#[inline(always)]
pub fn act_scalar(x: f32, kind: u16) -> f32 {
    match kind {
        format::ACT_RELU => {
            if x > 0.0 {
                x
            } else {
                0.0
            }
        }
        format::ACT_GELU => {
            // tanh 近似，免 exp
            0.5 * x * (1.0 + tanh_f(0.797_884_6 * (x + 0.044_715 * x * x * x)))
        }
        format::ACT_SILU => x / (1.0 + exp_f(-x)),
        format::ACT_SIGMOID => 1.0 / (1.0 + exp_f(-x)),
        format::ACT_TANH => tanh_f(x),
        format::ACT_RELU2 => {
            let r = if x > 0.0 { x } else { 0.0 };
            r * r
        }
        _ => x,
    }
}





#[inline(always)]
pub fn tanh_f(x: f32) -> f32 {
    if x > 15.0 {
        return 1.0;
    }
    if x < -15.0 {
        return -1.0;
    }
    let e = exp_f(2.0 * x);
    (e - 1.0) / (e + 1.0)
}

// ---------- 三角函数（P10：RoPE 需要）----------
//
// 为什么不能用 libm：引擎是 no_std + /NODEFAULTLIB，没有 libm。
// 为什么不用「调和公式从 sqrt 推」：不稳定且更慢。
// 做法与 exp_f 完全同构（Cody-Waite 两段式 + minimax 多项式），
// 这样精度量级与 exp_f（0.66 ULP）保持一致，不会成为新的精度债。
//
// 归约：x = n*(pi/2) + r，n = round(x * 2/pi)，|r| <= pi/4。
// 然后按 n mod 4 选 cos/sin 与符号 —— 这样多项式只需覆盖 [-pi/4, pi/4]，
// 4~5 阶就够 f32 精度。
//
// pi/2 的三段展开 —— **符号必须由计算得出，不能凭印象写**。
//
// Cody-Waite 的要点：hi 是 float32(π/2)，而 `π/2 - hi` 的**真实残差是负的**
// （hi 向上舍入）。用精确十进制算：
//     π/2            = 1.570796326794896619231321691639751442...
//     float32(π/2)   = 1.5707963705062866...
//     π/2 - hi       = -4.371138828673793e-08
//     float32(残差)  = -4.3711388287e-08
//     再下一段       = -1.7151245100e-15
//
// ⚠️ 第一版我把 MID 写成 **+4.371139e-8**（符号反了）。后果是每单位 n 有
// 8.7e-08 的系统偏差。修正后大角度误差从 1.62e-02 降到 9.75e-03。
//
// ⚠️⚠️ 但 **9.75e-03 不是这条式子的 bug，而是 f32 的物理下限**。
// 实测（tools/_reduce_limit.py）：
//
//     x         n        f32(x)        f32(n*π/2)      差距
//     100       64       100.000000    100.530968      7.63e-06
//     10,000    6,366    10,000.000    9,999.689       9.77e-04
//     100,000   63,662   100,000.000   100,000.039     7.81e-03   <-- 与实测吻合
//     1,000,000 636,620  1,000,000.0   1,000,000.375   6.25e-02
//
// 原因：x ~ 1e5 时 ulp(x) ~ 7.8e-3，而归约后的余数 r 是 O(1) 量级。
// **`x - n*(π/2)` 是两个 ~1e5 的数相减（灾难性抵消）**，
// 所以 r 最多只能准到 ~ulp(x)，**与 n*(π/2) 乘积算得多准无关**。
// 我试过 10 种 (HI 位宽, n 拆分) 组合，全部收敛到 7.5e-03 —— 这就是证据。
//
// ⇒ **任何「先算出大角度再归约」的方案在 f32 下都到此为止。**
//   正确的做法是**不要让角度变大**：RoPE 里按分量的周期对**位置**取模
//   （pos 是整数，取模可以精确），于是进入三角函数的参数永远在 [0, 2π)。
//   见 `rope_apply` 的实现与注释。
//
// 这条结论直接影响 RoPE 的写法，所以留在内核里而不是只写进文档。
const PI2_HI: f32 = 1.570_796_37; //  float32(π/2)
const PI2_MID: f32 = -4.371_138_8e-8; //  float32(π/2 - HI)     负号！
const PI2_LO: f32 = -1.715_124_5e-15; // float32(π/2 - HI - MID)

#[inline(always)]
fn sin_cos_reduce(x: f32) -> (f32, f32, i32) {
    // x 有限即可；f32 的 |x| 上界约 3.4e38，n 会超 i32 —— 但 RoPE 里
    // 的相位是 pos * theta^(-2i/hd)，theta^(-2i/hd) <= 1 且 pos <= 131072，
    // 所以实际 |x| < 2^17。超过 1e7 的相位本身在 f32 下已无有效位，
    // 这里夹紧而不是让它溢出成 NaN（与 exp_f 的夹紧语义一致）。
    let x = if x > 1.0e7 {
        1.0e7
    } else if x < -1.0e7 {
        -1.0e7
    } else {
        x
    };
    const TWO_OVER_PI: f32 = 0.636_619_8; // 2/pi
    let n = round_f32(x * TWO_OVER_PI);
    let r = ((x - n * PI2_HI) - n * PI2_MID) - n * PI2_LO;
    (r, r * r, n as i32)
}

/// sin(x)，f32 精度（Cephes 风格 minimax + Cody-Waite 两段归约）
///
/// 象限选择（n = round(x/(pi/2))，r = x - n*pi/2，|r| <= pi/4）：
///   n mod 4 = 0 -> sin(r)
///             1 -> +cos(r)      (sin(r+pi/2) = cos r)
///             2 -> -sin(r)      (sin(r+pi)   = -sin r)
///             3 -> -cos(r)      (sin(r+3pi/2)= -cos r)
#[inline(always)]
pub fn sin_f(x: f32) -> f32 {
    let (r, r2, n) = sin_cos_reduce(x);
    match n & 3 {
        0 => sin_poly(r, r2),
        1 => cos_poly(r, r2),
        2 => -sin_poly(r, r2),
        _ => -cos_poly(r, r2),
    }
}

/// cos(x)：cos(r + n*pi/2) 的象限选择
///   n mod 4 = 0 -> +cos(r)
///             1 -> -sin(r)      (cos(r+pi/2) = -sin r)
///             2 -> -cos(r)      (cos(r+pi)   = -cos r)
///             3 -> +sin(r)      (cos(r+3pi/2)= +sin r)
#[inline(always)]
pub fn cos_f(x: f32) -> f32 {
    let (r, r2, n) = sin_cos_reduce(x);
    match n & 3 {
        0 => cos_poly(r, r2),
        1 => -sin_poly(r, r2),
        2 => -cos_poly(r, r2),
        _ => sin_poly(r, r2),
    }
}

/// sin(r) 的 minimax 多项式，|r| <= pi/4。
///
/// 写成 `r * (1 + r2*(c3 + r2*(c5 + r2*(c7 + r2*c9))))` —— **先提出 r**，
/// 这是奇函数的正确形状。
///
/// 第一版我把它写成 `s*r2*r + r` 的累加形式，虽然代数上等价，
/// 但 cos_poly 那边把符号弄错了（见下），实测整体误差 1.6e-2。
/// sin(r) 的 minimax 多项式，|r| <= pi/4。
///
/// `sin(r) = r * (1 + r2*(c3 + r2*(c5 + r2*(c7 + r2*c9))))`
///
/// ⚠️ 系数是**实测拟合**出来的，不是抄的。第一版我用了记忆中的
/// 「Cephes 系数」(-1.9515e-8, 2.7183e-6, -1.9841e-4, 8.3333e-3)，
/// 那组系数对应的归约区间/归一化与本文件不同，实测在 r=pi/4 处
/// 误差 **8.2e-2** —— 比目标差 6 个数量级，而且**在小角度也错**，
/// 所以一眼就能排除「归约坏了」这个方向。
///
/// 修法：直接在 [-pi/4, pi/4] 上最小二乘拟合，得到上面那组
/// （数值上就是 Taylor 的 -1/6, 1/120, -1/5040, 1/362880）。
/// 实测 f32 下 max|err| = 3.1e-08（见 tools/p10_sincos_accuracy.py）。
///
/// **教训**：抄来的系数必须在本文件的归一化下**实测**，
/// 「代数上看起来对」不等于「数值上对」。
#[inline(always)]
fn sin_poly(r: f32, r2: f32) -> f32 {
    let mut q = 2.726_982_9e-6; // c9
    q = q * r2 - 1.984_024_6e-4; // c7
    q = q * r2 + 8.333_332_2e-3; // c5
    q = q * r2 - 1.666_666_7e-1; // c3
    r * (1.0 + r2 * q)
}

/// cos(r) 的 minimax 多项式，|r| <= pi/4。
///
/// `cos(r) = 1 + r2*(d2 + r2*(d4 + r2*(d6 + r2*d8)))`
///
/// 同样是实测拟合（见 sin_poly 的说明）。第一版有两个独立错误：
///   1. 系数照抄，量级不对；
///   2. Horner 链末尾写成 `c*r2*(-1.0) + 1.0`，把 `d2 = -0.5`
///      又取了一次负 ⇒ 等于 `1 - (0.5+...)*r2`，符号反了。
/// 两个错误叠加后整体误差 1.6e-2 ~ 8.7e-2。
/// floor(x)（f32）。用于 RoPE 里「按周期对位置取模」。
///
/// 为什么不用 `as i32`：`as` 是**截断**（向 0 取整），对负数是错的；
/// 而且位置可能超过 i32 范围。这里用「截断 + 负则减 1」的经典写法。
#[inline(always)]
pub fn floor_f(x: f32) -> f32 {
    let t = x as i64 as f32; // 截断（f32 -> i64 -> f32）
    if t > x {
        t - 1.0
    } else {
        t
    }
}

#[inline(always)]
fn cos_poly(r: f32, r2: f32) -> f32 {
    let mut p = 2.441_057_3e-5; // d8
    p = p * r2 - 1.388_697_7e-3; // d6
    p = p * r2 + 4.166_663_0e-2; // d4
    p = p * r2 - 5.0e-1; // d2   <- 负号在系数里，不在表达式里
    r2 * p + 1.0
}

/// f32 round-to-nearest —— 复用文件上方已有的 `round_f32`（line 92），
/// 不重复定义（重复定义会 E0428，而且两套舍入行为迟早会漂移）。
/// RoPE 的相位归约与 exp_f 用**同一个**舍入，精度量级才一致。

// 自然对数：位分解 + 多项式（不依赖 libm）
#[inline(always)]
pub fn ln_f(x: f32) -> f32 {
    if x <= 0.0 {
        return -88.0; // 下界保护，避免 NaN 传播
    }
    let bits = x.to_bits();
    let mut e = ((bits >> 23) & 0xff) as i32 - 127;
    let mut m = f32::from_bits((bits & 0x007f_ffff) | 0x3f80_0000); // m ∈ [1,2)
    if m > 1.414_213_6 {
        m *= 0.5;
        e += 1;
    }
    let t = (m - 1.0) / (m + 1.0);
    let t2 = t * t;
    let p = 2.0 * t * (1.0 + t2 * (0.333_333_34 + t2 * (0.2 + t2 * 0.142_857_15)));
    p + (e as f32) * 0.693_147_18
}

// ---------- GEMM：C[M,N] = A[M,K] * B[K,N]（B 行主序，K 连续）----------
// 8x8 微内核 + FMA；K 循环展开；B 面板预取
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemm_avx2(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    acc: bool,
) {
    let mut i = 0;
    while i + 8 <= m {
        let mut j = 0;
        while j + 8 <= n {
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut acc2 = _mm256_setzero_ps();
            let mut acc3 = _mm256_setzero_ps();
            let mut acc4 = _mm256_setzero_ps();
            let mut acc5 = _mm256_setzero_ps();
            let mut acc6 = _mm256_setzero_ps();
            let mut acc7 = _mm256_setzero_ps();
            let mut p = 0;
            while p < k {
                let bv = _mm256_loadu_ps(b.add(p * n + j));
                let a0 = _mm256_set1_ps(*a.add(i * k + p));
                acc0 = _mm256_fmadd_ps(a0, bv, acc0);
                let a1 = _mm256_set1_ps(*a.add((i + 1) * k + p));
                acc1 = _mm256_fmadd_ps(a1, bv, acc1);
                let a2 = _mm256_set1_ps(*a.add((i + 2) * k + p));
                acc2 = _mm256_fmadd_ps(a2, bv, acc2);
                let a3 = _mm256_set1_ps(*a.add((i + 3) * k + p));
                acc3 = _mm256_fmadd_ps(a3, bv, acc3);
                let a4 = _mm256_set1_ps(*a.add((i + 4) * k + p));
                acc4 = _mm256_fmadd_ps(a4, bv, acc4);
                let a5 = _mm256_set1_ps(*a.add((i + 5) * k + p));
                acc5 = _mm256_fmadd_ps(a5, bv, acc5);
                let a6 = _mm256_set1_ps(*a.add((i + 6) * k + p));
                acc6 = _mm256_fmadd_ps(a6, bv, acc6);
                let a7 = _mm256_set1_ps(*a.add((i + 7) * k + p));
                acc7 = _mm256_fmadd_ps(a7, bv, acc7);
                p += 1;
            }
            if acc {
                acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(c.add(i * n + j)));
                acc1 = _mm256_add_ps(acc1, _mm256_loadu_ps(c.add((i + 1) * n + j)));
                acc2 = _mm256_add_ps(acc2, _mm256_loadu_ps(c.add((i + 2) * n + j)));
                acc3 = _mm256_add_ps(acc3, _mm256_loadu_ps(c.add((i + 3) * n + j)));
                acc4 = _mm256_add_ps(acc4, _mm256_loadu_ps(c.add((i + 4) * n + j)));
                acc5 = _mm256_add_ps(acc5, _mm256_loadu_ps(c.add((i + 5) * n + j)));
                acc6 = _mm256_add_ps(acc6, _mm256_loadu_ps(c.add((i + 6) * n + j)));
                acc7 = _mm256_add_ps(acc7, _mm256_loadu_ps(c.add((i + 7) * n + j)));
            }
            _mm256_storeu_ps(c.add(i * n + j), acc0);
            _mm256_storeu_ps(c.add((i + 1) * n + j), acc1);
            _mm256_storeu_ps(c.add((i + 2) * n + j), acc2);
            _mm256_storeu_ps(c.add((i + 3) * n + j), acc3);
            _mm256_storeu_ps(c.add((i + 4) * n + j), acc4);
            _mm256_storeu_ps(c.add((i + 5) * n + j), acc5);
            _mm256_storeu_ps(c.add((i + 6) * n + j), acc6);
            _mm256_storeu_ps(c.add((i + 7) * n + j), acc7);
            j += 8;
        }
        // N 尾部
        while j < n {
            let mut s0 = 0.0f32;
            let mut s1 = 0.0f32;
            let mut s2 = 0.0f32;
            let mut s3 = 0.0f32;
            let mut s4 = 0.0f32;
            let mut s5 = 0.0f32;
            let mut s6 = 0.0f32;
            let mut s7 = 0.0f32;
            for p in 0..k {
                let bv = *b.add(p * n + j);
                s0 += *a.add(i * k + p) * bv;
                s1 += *a.add((i + 1) * k + p) * bv;
                s2 += *a.add((i + 2) * k + p) * bv;
                s3 += *a.add((i + 3) * k + p) * bv;
                s4 += *a.add((i + 4) * k + p) * bv;
                s5 += *a.add((i + 5) * k + p) * bv;
                s6 += *a.add((i + 6) * k + p) * bv;
                s7 += *a.add((i + 7) * k + p) * bv;
            }
            if acc {
                s0 += *c.add(i * n + j);
                s1 += *c.add((i + 1) * n + j);
                s2 += *c.add((i + 2) * n + j);
                s3 += *c.add((i + 3) * n + j);
                s4 += *c.add((i + 4) * n + j);
                s5 += *c.add((i + 5) * n + j);
                s6 += *c.add((i + 6) * n + j);
                s7 += *c.add((i + 7) * n + j);
            }
            *c.add(i * n + j) = s0;
            *c.add((i + 1) * n + j) = s1;
            *c.add((i + 2) * n + j) = s2;
            *c.add((i + 3) * n + j) = s3;
            *c.add((i + 4) * n + j) = s4;
            *c.add((i + 5) * n + j) = s5;
            *c.add((i + 6) * n + j) = s6;
            *c.add((i + 7) * n + j) = s7;
            j += 1;
        }
        i += 8;
    }
    // M 尾部：标量
    while i < m {
        for j in 0..n {
            let mut s = 0.0f32;
            for p in 0..k {
                s += *a.add(i * k + p) * *b.add(p * n + j);
            }
            if acc {
                s += *c.add(i * n + j);
            }
            *c.add(i * n + j) = s;
        }
        i += 1;
    }
}

// 标量回退（非 x86_64 或未启用 avx2 时）
pub unsafe fn gemm_scalar(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    acc: bool,
) {
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for p in 0..k {
                s += *a.add(i * k + p) * *b.add(p * n + j);
            }
            if acc {
                s += *c.add(i * n + j);
            }
            *c.add(i * n + j) = s;
        }
    }
}

#[inline]
pub unsafe fn gemm(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    acc: bool,
) {
    // NN 布局（B 为 [K,N]）：注意力 QK^T 之后会用到，先用标量实现占位。
    // NT 布局（B 为 [N,K]，nn.Linear 权重）走 gemm_nt 的 AVX2 8x8 微内核。
    gemm_scalar(a, b, c, m, n, k, acc);
}

// ---------- GEMM_NT：C[M,N] = A[M,K] × B[N,K]^T ----------
// PyTorch Linear 的权重是 [N,K]，内层沿 K 连续。
//
// 8×8 寄存器分块微内核：
//   8 个 ymm 累加器 c0..c7，c_r = C 第 r 行、列 j..j+8
//   K 循环内：broadcast(a[i+r][p]) × bv(p) → FMA，其中
//     bv(p) = [b[j+0][p], ..., b[j+7][p]]（8 个 j 列的同一 K 偏移）
//   bv 由 B 的 8 行 × 8 个 K 值经 8x8 转置得到
//   归约：K 循环结束后每个 ymm 直接就是 C 的一行 8 列 → 零水平归约
//
// 复用：内层 j 循环复用已转置的 B 面板（见 packed 缓冲）
//
// [D/性能] 优化记录 —— 见下方 gemm_nt_pack_b / gemm_nt_n1_avx2 的注释头。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemm_nt_avx2(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    acc: bool,
) {
    let mut i = 0;
    while i + 8 <= m {
        let pa = a.add(i * k);
        let mut j = 0;
        while j + 8 <= n {
            let mut c0 = _mm256_setzero_ps();
            let mut c1 = _mm256_setzero_ps();
            let mut c2 = _mm256_setzero_ps();
            let mut c3 = _mm256_setzero_ps();
            let mut c4 = _mm256_setzero_ps();
            let mut c5 = _mm256_setzero_ps();
            let mut c6 = _mm256_setzero_ps();
            let mut c7 = _mm256_setzero_ps();

            let mut p = 0;
            while p + 8 <= k {
                // 载入 B 的 8 行 × 8 个 K 值并转置
                let mut t0 = _mm256_loadu_ps(b.add(j * k + p));
                let mut t1 = _mm256_loadu_ps(b.add((j + 1) * k + p));
                let mut t2 = _mm256_loadu_ps(b.add((j + 2) * k + p));
                let mut t3 = _mm256_loadu_ps(b.add((j + 3) * k + p));
                let mut t4 = _mm256_loadu_ps(b.add((j + 4) * k + p));
                let mut t5 = _mm256_loadu_ps(b.add((j + 5) * k + p));
                let mut t6 = _mm256_loadu_ps(b.add((j + 6) * k + p));
                let mut t7 = _mm256_loadu_ps(b.add((j + 7) * k + p));
                transpose8x8(
                    &mut t0, &mut t1, &mut t2, &mut t3, &mut t4, &mut t5, &mut t6, &mut t7,
                );
                // t_s[m] = b[j+m][p+s]
                let mut s = 0;
                while s < 8 {
                    let v = match s {
                        0 => t0,
                        1 => t1,
                        2 => t2,
                        3 => t3,
                        4 => t4,
                        5 => t5,
                        6 => t6,
                        _ => t7,
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
            // K 尾段
            while p < k {
                let bv = _mm256_set_ps(
                    *b.add((j + 7) * k + p),
                    *b.add((j + 6) * k + p),
                    *b.add((j + 5) * k + p),
                    *b.add((j + 4) * k + p),
                    *b.add((j + 3) * k + p),
                    *b.add((j + 2) * k + p),
                    *b.add((j + 1) * k + p),
                    *b.add(j * k + p),
                );
                c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*pa.add(p)), bv, c0);
                c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*pa.add(k + p)), bv, c1);
                c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*pa.add(2 * k + p)), bv, c2);
                c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*pa.add(3 * k + p)), bv, c3);
                c4 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*pa.add(4 * k + p)), bv, c4);
                c5 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*pa.add(5 * k + p)), bv, c5);
                c6 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*pa.add(6 * k + p)), bv, c6);
                c7 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*pa.add(7 * k + p)), bv, c7);
                p += 1;
            }

            // 每个 ymm 就是 C 的一行 8 个连续列（块内 n>=8 是入口前提）
            let rows = [c0, c1, c2, c3, c4, c5, c6, c7];
            let mut r = 0;
            while r < 8 {
                let o = (i + r) * n + j;
                let mut tmp = [0f32; 8];
                _mm256_storeu_ps(tmp.as_mut_ptr(), rows[r]);
                let lim = if j + 8 <= n { 8 } else { n - j };
                let mut t = 0;
                while t < lim {
                    if acc {
                        *c.add(o + t) += tmp[t];
                    } else {
                        *c.add(o + t) = tmp[t];
                    }
                    t += 1;
                }
                r += 1;
            }
            j += 8;
        }
        i += 8;
    }
    // 边界补齐
    gemm_nt_scalar_tail(a, b, c, m, n, k, acc, i / 8 * 8, n / 8 * 8);
}

// =====================================================================
// [D/性能] 优化 1：B 面板打包（packing）
// ---------------------------------------------------------------------
// 问题：8x8 微内核每处理一个 K-block 都要把 B 的 8 行 × 8 个 K 值
//       做一次 8x8 转置。512³ 下这是 (512/8)^3 = 262144 次转置 × 24 条
//       shuffle/permute = 约 630 万条非 FMA 指令，与 3.1M 条 FMA 同量级，
//       而且每次转置有 8 个 load 的延迟链，直接卡住 FMA 端口。
//
// 做法：把 B 从 [N,K] 行主序一次性重排成微内核友好的「面板」布局：
//         pack[(jb/8)][(p/8)][lane][s]  = B[jb + lane][p + s]
//       即每个 (8 列 × 8 K) 面板连续存放，panel 内 lane 步长 8 个 f32。
//       面板尺寸固定 8*8=64 f32 = 256 B（正好 4 条 cache line，且 64B 对齐）。
//
//       于是 K 循环里 "bv(p+s) = [b[j+0][p+s..], ..., b[j+7][p+s]]" 变成
//       「第 s 个 ymm 直接 load」——转置从 K 循环里彻底消失。
//
// 布局论证（为什么不是转成 [k][n]）：微内核要的向量是「同一 K 偏移上
// 8 个不同 j 列」，那是 B^T 的行。但 B^T 行不连续（跨度 k），仍然要
// gather；而 [k][n] 布局下 8 个连续 f32 是 8 个不同 K、同一 j——微内核
// 用不上。所以必须按「j 在内、k 在外」排列，就是本函数的 ndx。
//
// 代价：一次 O(N*K) 的打包写。只做 1 次，被 M/8 = 64 次行块复用，
//       摊薄后每次行块只付 O(N*K/64)。
//
// 内存：N*K 个 f32，用 crate::alloc（HeapAlloc）。打包失败不影响正确性。
// =====================================================================

/// 把 B[N,K] 行主序打包成 8x8 面板布局。
/// 输出长度 = ((n+7)/8) * ((k+7)/8) * 64 个 f32；尾部补零。
///
/// 面板内偏移 = s * 8 + lane，其中 lane = j%8（B 的行），s = p%8（K 偏移）。
/// 这样 `_mm256_load_ps(bp + s*8)` 一次读出的 8 个连续 f32 正好是
///   [B[j+0][kk+s], B[j+1][kk+s], ..., B[j+7][kk+s]]
/// 即微内核要的那个 bv 向量——这正是「转置预做」的含义。
/// （写反成 lane*8 + s 会让 load 读到的元素角色互换：实测 512³ maxerr=36.5。）
#[cfg(target_arch = "x86_64")]
unsafe fn gemm_nt_pack_b(b: *const f32, n: usize, k: usize, out: *mut f32) {
    let kp = (k + 7) / 8;
    let panel_stride = 64usize;
    let mut jb = 0usize;
    while jb < n {
        let jrows = if jb + 8 <= n { 8 } else { n - jb };
        let jbase = (jb / 8) * kp * panel_stride;
        let mut pb = 0usize;
        while pb < k {
            let kcols = if pb + 8 <= k { 8 } else { k - pb };
            let pbase = jbase + (pb / 8) * panel_stride;
            let mut lane = 0usize;
            while lane < 8 {
                let src = if lane < jrows {
                    b.add((jb + lane) * k + pb)
                } else {
                    core::ptr::null()
                };
                let mut s = 0usize;
                while s < 8 {
                    let v = if s < kcols && !src.is_null() {
                        *src.add(s)
                    } else {
                        0.0
                    };
                    *out.add(pbase + s * 8 + lane) = v;
                    s += 1;
                }
                lane += 1;
            }
            pb += 8;
        }
        jb += 8;
    }
}

// =====================================================================
// [D/性能] 优化 2：打包布局 + 16 行 A 分块的 NT 微内核
// ---------------------------------------------------------------------
// 相对 gemm_nt_avx2 的三处改动：
//   1) B 已打包 → K 循环内只有 load + broadcast + FMA，没有 shuffle
//   2) A 一次吃 16 行（2 组 × 8 个 ymm 累加器 = 16 个），B 面板复用翻倍，
//      每个 B 面板的 load 数不变但服务的 FMA 数翻倍 → load 压力减半
//   3) 每个 B 面板做 _mm_prefetch 提前两轮拉进来
//
// 寄存器预算：16 个累加器 + 2 个 bv 临时 + 少量地址 = 20 个 ymm，
// 在 16 个 architectural ymm 之外的会被 LLVM 溢到栈上——这是本实现的
// 已知风险点，实测数据见 tools/bench_ab_log.jsonl。
// =====================================================================
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_nt_packed_kernel(
    a: *const f32,
    packed: *const f32,
    c: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    acc: bool,
) {
    let kp = (k + 7) / 8;
    let panel_stride = 64usize;
    let nj = (n + 7) / 8;

    // M 方向按 8 行分块（MR=8）：
    //   MR=8  -> 8 个累加器 + 1 个 bv，寄存器压力 9/16，不溢出。
    //   MR=16 -> 16 个累加器 + bv = 17+，超过 16 个 ymm，LLVM 会溢到栈，
    //            实测 512³ 掉到 61 GFLOPS（vs MR=8 的 98），见 bench_ab_log。
    let mut i = 0usize;
    while i + 8 <= m {
        let pa = a.add(i * k);
        let mut jb = 0usize;
        while jb < nj {
            let j = jb * 8;
            let mut c0 = _mm256_setzero_ps();
            let mut c1 = _mm256_setzero_ps();
            let mut c2 = _mm256_setzero_ps();
            let mut c3 = _mm256_setzero_ps();
            let mut c4 = _mm256_setzero_ps();
            let mut c5 = _mm256_setzero_ps();
            let mut c6 = _mm256_setzero_ps();
            let mut c7 = _mm256_setzero_ps();

            let jbase = jb * kp * panel_stride;
            let mut pb = 0usize;
            while pb < kp {
                let bp = packed.add(jbase + pb * panel_stride);
                if pb + 2 < kp {
                    _mm_prefetch(bp.add(2 * panel_stride) as *const i8, _MM_HINT_T0);
                }
                let kk = pb * 8;
                let mut s = 0usize;
                // [D/性能] 优化 5：K 块内显式软件流水。
                // 先把本块 8 个 B 向量一次性 load 进寄存器，避免
                // 「load 完立刻喂 FMA」的流水停顿；随后连发 64 条互相
                // 独立的 FMA，把 8 条累加链的延迟充分叠起来。
                let v0 = _mm256_load_ps(bp.add(0));
                let v1 = _mm256_load_ps(bp.add(8));
                let v2 = _mm256_load_ps(bp.add(16));
                let v3 = _mm256_load_ps(bp.add(24));
                let v4 = _mm256_load_ps(bp.add(32));
                let v5 = _mm256_load_ps(bp.add(40));
                let v6 = _mm256_load_ps(bp.add(48));
                let v7 = _mm256_load_ps(bp.add(56));

                let q0 = pa.add(0 * k + kk);
                let q1 = pa.add(1 * k + kk);
                let q2 = pa.add(2 * k + kk);
                let q3 = pa.add(3 * k + kk);
                let q4 = pa.add(4 * k + kk);
                let q5 = pa.add(5 * k + kk);
                let q6 = pa.add(6 * k + kk);
                let q7 = pa.add(7 * k + kk);


                c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q0.add(0)), v0, c0);
                c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q1.add(0)), v0, c1);
                c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q2.add(0)), v0, c2);
                c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q3.add(0)), v0, c3);
                c4 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q4.add(0)), v0, c4);
                c5 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q5.add(0)), v0, c5);
                c6 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q6.add(0)), v0, c6);
                c7 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q7.add(0)), v0, c7);

                c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q0.add(1)), v1, c0);
                c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q1.add(1)), v1, c1);
                c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q2.add(1)), v1, c2);
                c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q3.add(1)), v1, c3);
                c4 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q4.add(1)), v1, c4);
                c5 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q5.add(1)), v1, c5);
                c6 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q6.add(1)), v1, c6);
                c7 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q7.add(1)), v1, c7);

                c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q0.add(2)), v2, c0);
                c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q1.add(2)), v2, c1);
                c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q2.add(2)), v2, c2);
                c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q3.add(2)), v2, c3);
                c4 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q4.add(2)), v2, c4);
                c5 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q5.add(2)), v2, c5);
                c6 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q6.add(2)), v2, c6);
                c7 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q7.add(2)), v2, c7);

                c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q0.add(3)), v3, c0);
                c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q1.add(3)), v3, c1);
                c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q2.add(3)), v3, c2);
                c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q3.add(3)), v3, c3);
                c4 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q4.add(3)), v3, c4);
                c5 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q5.add(3)), v3, c5);
                c6 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q6.add(3)), v3, c6);
                c7 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q7.add(3)), v3, c7);

                c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q0.add(4)), v4, c0);
                c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q1.add(4)), v4, c1);
                c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q2.add(4)), v4, c2);
                c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q3.add(4)), v4, c3);
                c4 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q4.add(4)), v4, c4);
                c5 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q5.add(4)), v4, c5);
                c6 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q6.add(4)), v4, c6);
                c7 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q7.add(4)), v4, c7);

                c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q0.add(5)), v5, c0);
                c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q1.add(5)), v5, c1);
                c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q2.add(5)), v5, c2);
                c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q3.add(5)), v5, c3);
                c4 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q4.add(5)), v5, c4);
                c5 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q5.add(5)), v5, c5);
                c6 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q6.add(5)), v5, c6);
                c7 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q7.add(5)), v5, c7);

                c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q0.add(6)), v6, c0);
                c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q1.add(6)), v6, c1);
                c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q2.add(6)), v6, c2);
                c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q3.add(6)), v6, c3);
                c4 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q4.add(6)), v6, c4);
                c5 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q5.add(6)), v6, c5);
                c6 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q6.add(6)), v6, c6);
                c7 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q7.add(6)), v6, c7);

                c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q0.add(7)), v7, c0);
                c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q1.add(7)), v7, c1);
                c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q2.add(7)), v7, c2);
                c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q3.add(7)), v7, c3);
                c4 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q4.add(7)), v7, c4);
                c5 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q5.add(7)), v7, c5);
                c6 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q6.add(7)), v7, c6);
                c7 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q7.add(7)), v7, c7);
                pb += 1;
            }

            let rows = [c0, c1, c2, c3, c4, c5, c6, c7];
            if j + 8 <= n {
                let mut r = 0usize;
                while r < 8 {
                    let o = c.add((i + r) * n + j);
                    if acc {
                        _mm256_storeu_ps(o, _mm256_add_ps(rows[r], _mm256_loadu_ps(o)));
                    } else {
                        _mm256_storeu_ps(o, rows[r]);
                    }
                    r += 1;
                }
            } else {
                let lim = n - j;
                let mut r = 0usize;
                while r < 8 {
                    let o = c.add((i + r) * n + j);
                    let mut tmp = [0f32; 8];
                    _mm256_storeu_ps(tmp.as_mut_ptr(), rows[r]);
                    let mut t = 0usize;
                    while t < lim {
                        if acc {
                            *o.add(t) += tmp[t];
                        } else {
                            *o.add(t) = tmp[t];
                        }
                        t += 1;
                    }
                    r += 1;
                }
            }
            jb += 1;
        }
        i += 8;
    }

    // ---- M 尾巴：标量 ----
    while i < m {
        for j in 0..n {
            let mut s = 0.0f32;
            for p in 0..k {
                s += *a.add(i * k + p) * at_packed(packed, kp, j, p);
            }
            if acc {
                s += *c.add(i * n + j);
            }
            *c.add(i * n + j) = s;
        }
        i += 1;
    }
}

/// 从打包布局读回 B[j][p]（只给 M 尾部的标量路径用；主内核不调它）。
/// 面板布局: panel(jb,pb) 在 (jb*kp + pb)*64，元素在 + (p%8)*8 + (j%8)。
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn at_packed(packed: *const f32, kp: usize, j: usize, p: usize) -> f32 {
    let panel_stride = 64usize;
    let (jb, lane) = (j / 8, j % 8);
    let (pb, s) = (p / 8, p % 8);
    *packed.add((jb * kp + pb) * panel_stride + s * 8 + lane)
}

/// 打包 + 主内核的总入口。分配失败时退回原 8x8 内核（正确性优先）。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_nt_packed(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    acc: bool,
) {
    let kp = (k + 7) / 8;
    let nj = (n + 7) / 8;
    let nb = nj * kp * 64;
    // 末尾留 64 字节 slack：对齐抬升最多用掉 63 字节。
    let bytes = nb * 4 + 64;
    let raw = crate::alloc(bytes);
    if raw.is_null() {
        gemm_nt_avx2(a, b, c, m, n, k, acc);
        return;
    }
    // 32B 对齐（AVX2 的 _mm256_load_ps 要求）。
    //
    // [FIX/崩溃] 这里曾经写成 `raw.add((64 - misalign) / 4)`：`raw` 是 *mut u8，
    // 指针加法是**字节**单位，却又按 f32 除了 4。当 misalign % 4 != 0 时
    // 偏移被截断变小 → 基址仍未对齐 → _mm256_load_ps 触发访问违例
    // (0xC0000005)。256³ 崩而 512³ 过，正是因为两者的分配地址对齐情况不同。
    // 正确写法：按字节加，且用 32 的倍数。
    let addr = raw as usize;
    let aligned = (addr + 31) & !31;
    let packed = aligned as *mut f32;
    debug_assert!(aligned + nb * 4 <= addr + bytes);
    gemm_nt_pack_b(b, n, k, packed);
    gemm_nt_packed_kernel(a, packed, c, m, n, k, acc);
    crate::dealloc(raw, bytes);
}


// 8x8 f32 转置（AVX2）
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn transpose8x8(
    r0: &mut __m256,
    r1: &mut __m256,
    r2: &mut __m256,
    r3: &mut __m256,
    r4: &mut __m256,
    r5: &mut __m256,
    r6: &mut __m256,
    r7: &mut __m256,
) {
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

// 只算 [i0..m) × [j0..n) 的边界（主块已由向量内核算完）
unsafe fn gemm_nt_scalar_tail(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    acc: bool,
    i0: usize,
    j0: usize,
) {
    for i in 0..m {
        for j in 0..n {
            if i < i0 && j < j0 {
                continue; // 已被向量内核覆盖
            }
            let mut s = 0.0f32;
            for p in 0..k {
                s += *a.add(i * k + p) * *b.add(j * k + p);
            }
            if acc {
                s += *c.add(i * n + j);
            }
            *c.add(i * n + j) = s;
        }
    }
}



pub unsafe fn gemm_nt_scalar(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    acc: bool,
) {
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for p in 0..k {
                s += *a.add(i * k + p) * *b.add(j * k + p);
            }
            if acc {
                s += *c.add(i * n + j);
            }
            *c.add(i * n + j) = s;
        }
    }
}

/// [D/性能] A/B 基准专用入口。
/// [D/性能] A/B 基准专用：强制走 packed 路径（不做形状判断）。
/// 只被 tests/src/bench.rs 调用；功能路径（gemm_nt）不受影响。
#[cfg(target_arch = "x86_64")]
pub unsafe fn gemm_nt_packed_bench(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    acc: bool,
) {
    gemm_nt_packed(a, b, c, m, n, k, acc);
}

#[inline]
pub unsafe fn gemm_nt(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    acc: bool,
) {
    #[cfg(target_arch = "x86_64")]
    {
        // 8x8 向量内核要求 M 和 N 都至少 8（它一次处理 8 行 × 8 列）。
        // 任一维 < 8 时（如列向量 S·r 的 N=1）走标量路径，
        // 避免向量化的「写 8 个连续列」越过行边界写坏相邻数据。
        //
        // [D/性能] 大形状（打包摊得开）走 packed 路径：
        //   m*n*k >= 1M 且 n,k >= 8 —— 打包的 O(N*K) 开销被 M/16 次行块复用摊薄。
        //   小形状继续用原 8x8 内核（省掉一次 heap 分配 + 打包写）。
        if crate::interp::HAS_AVX2 && n >= 8 && m >= 8 {
            if n >= 8 && k >= 8 && m >= 16 && (m as u64) * (n as u64) * (k as u64) >= 1_000_000 {
                gemm_nt_packed(a, b, c, m, n, k, acc);
            } else {
                gemm_nt_avx2(a, b, c, m, n, k, acc);
            }
            return;
        }
    }
    gemm_nt_scalar(a, b, c, m, n, k, acc);
}

// ============ JIT 内核：JIT 生成的胶水只 call 这些函数 ============
// 参数按 Win64 寄存器传（rcx,rdx,r8,r9），形状用 u64 打包以避免超过 4 个参数。

// GEMM_NT: dims = m | n<<16 | k<<32 | acc<<48
#[no_mangle]
pub unsafe extern "system" fn jit_gemm_nt(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    dims: u64,
) {
    let m = (dims & 0xFFFF) as usize;
    let n = ((dims >> 16) & 0xFFFF) as usize;
    let k = ((dims >> 32) & 0xFFFF) as usize;
    let acc = (dims >> 48) & 1 != 0;
    gemm_nt(a, b, c, m, n, k, acc);
}

#[no_mangle]
pub unsafe extern "system" fn jit_act(p: *mut f32, n: usize, kind: u64) {
    act_avx2(p, n, kind as u16);
}

#[no_mangle]
pub unsafe extern "system" fn jit_ewadd(d: *mut f32, x: *const f32, y: *const f32, n: usize) {
    ew_op(d, x, y, n, format::OP_EWADD);
}
#[no_mangle]
pub unsafe extern "system" fn jit_ewmul(d: *mut f32, x: *const f32, y: *const f32, n: usize) {
    ew_op(d, x, y, n, format::OP_EWMUL);
}
#[no_mangle]
pub unsafe extern "system" fn jit_ewsub(d: *mut f32, x: *const f32, y: *const f32, n: usize) {
    ew_op(d, x, y, n, format::OP_EWSUB);
}
#[no_mangle]
pub unsafe extern "system" fn jit_ewdiv(d: *mut f32, x: *const f32, y: *const f32, n: usize) {
    ew_op(d, x, y, n, format::OP_EWDIV);
}

#[no_mangle]
pub unsafe extern "system" fn jit_bias(d: *mut f32, b: *const f32, rows: usize, cols: usize) {
    for i in 0..rows {
        for j in 0..cols {
            *d.add(i * cols + j) += *b.add(j);
        }
    }
}

// LN: rcx=x, rdx=gamma, r8=bias, r9=dims(m|c<<32), 栈=pad
#[no_mangle]
pub unsafe extern "system" fn jit_ln(
    x: *mut f32,
    g: *const f32,
    b: *const f32,
    dims: u64,
    _pad: u64,
) {
    let m = (dims & 0xFFFF_FFFF) as usize;
    let c = (dims >> 32) as usize;
    layernorm(x, g, b, m, c, 1e-5);
}

#[no_mangle]
pub unsafe extern "system" fn jit_softmax(p: *mut f32, dims: u64, causal: u64) {
    let m = (dims & 0xFFFF_FFFF) as usize;
    let c = (dims >> 32) as usize;
    softmax(p, m, c, causal != 0);
}

#[no_mangle]
pub unsafe extern "system" fn jit_outer(s: *mut f32, v: *const f32, k: *const f32, dims: u64) {
    let m = (dims & 0xFFFF_FFFF) as usize;
    let n = (dims >> 32) as usize;
    outer(s, v, k, m, n, 1.0);
}

#[no_mangle]
pub unsafe extern "system" fn jit_print(p: *const f32, n: usize) {
    crate::interp::print_f32(p, n);
}

#[no_mangle]
pub unsafe extern "system" fn jit_unimpl(op: u64) {
    // 防御纵深：`jit.rs` 现在会在**编译期**拒绝未实现的 opcode，
    // 正常路径下这个函数不该被调用到。若真被调到，说明有路径绕过了编译期检查
    // —— 那也必须硬失败，而不是打印一行继续跑（否则表现为
    // 「梯度恒为 0、参数永不更新，但程序正常退出」的静默降级，见 P9 成员 J 的报告）。
    let _ = op;
    crate::print(b"JIT: unimplemented opcode at runtime -- aborting\n");
    crate::interp::exit_i(6);
}

#[no_mangle]
pub unsafe extern "system" fn jit_memcpy(dst: *mut u8, src: *const u8, n: usize) {
    core::ptr::copy_nonoverlapping(src, dst, n);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn ew_op(dst: *mut f32, x: *const f32, y: *const f32, n: usize, op: u8) {
    let mut i = 0;
    while i + 8 <= n {
        let a = _mm256_loadu_ps(x.add(i));
        let b = _mm256_loadu_ps(y.add(i));
        let r = match op {
            format::OP_EWADD => _mm256_add_ps(a, b),
            format::OP_EWMUL => _mm256_mul_ps(a, b),
            format::OP_EWSUB => _mm256_sub_ps(a, b),
            _ => _mm256_div_ps(a, b),
        };
        _mm256_storeu_ps(dst.add(i), r);
        i += 8;
    }
    while i < n {
        let a = *x.add(i);
        let b = *y.add(i);
        *dst.add(i) = match op {
            format::OP_EWADD => a + b,
            format::OP_EWMUL => a * b,
            format::OP_EWSUB => a - b,
            _ => a / b,
        };
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn act_avx2(dst: *mut f32, n: usize, kind: u16) {
    // relu/relu2 可纯 SIMD；其余走标量（gelu/silu 等含超越函数）
    if kind == format::ACT_RELU {
        let zero = _mm256_setzero_ps();
        let mut i = 0;
        while i + 8 <= n {
            let v = _mm256_loadu_ps(dst.add(i));
            _mm256_storeu_ps(dst.add(i), _mm256_max_ps(v, zero));
            i += 8;
        }
        while i < n {
            if *dst.add(i) < 0.0 {
                *dst.add(i) = 0.0
            }
            i += 1;
        }
        return;
    }
    for i in 0..n {
        *dst.add(i) = act_scalar(*dst.add(i), kind);
    }
}

// ---------- P12: bf16 权重 GEMM ----------
//
// 为什么需要（这是「实际加载模型」的关键）：
//   MiniCPM / Llama 的权重是 **bf16**。此前引擎把常量池一律当 f32 读，
//   于是 bf16 权重会被解释成垃圾数（且不报错）。
//   而若在装载时把 bf16 全部转成 f32，2.16 GB 权重会变成 **4.32 GB 私有提交** ——
//   正好把 mmap 的意义抹掉。
//
// ⇒ 正确做法：让 bf16 **原地**留在映射里，在 GEMM 内层做加宽。
//
// 为什么这样**更快**（实测，不是推断）：
//   P11 的 `vmlm-why` 已证明本机 GEMM 是 **load 端口受限**
//   （load 下界 4.5 周期/K vs FMA 2.67）。bf16 把每个权重的加载字节**减半**，
//   正好压的是瓶颈那一侧。实测 4.4~5.0x（8 个/次）与 7.5~7.9x（16 个/次）。
//
// 转换是**精确的**：bf16 就是 f32 的高 16 位，所以
//   `f32::from_bits((h as u32) << 16)`
// 无损、无舍入、不需要任何浮点转换指令。实测与「先转 f32 再算」的
// max_abs_err 只到 f32 求和次序的量级（~1e-6 相对）。
//
// 布局（**实测确定**，见 tests/src/shufprobe.rs，不要手推 lane 语义）：
//   一次 ymm load = 16 个 bf16 = 两个连续的 128 位半区
//   低半区 = b0..b7，高半区 = b8..b15
//   各自 `cvtepu16_epi32` 零扩展到 32 位，再左移 16 位即得 f32。
//   **不需要 unpack、不需要 permute。**





/// `C[M,N] = A[M,K] · B[N,K]^T`，B 是 bf16。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemm_nt_bf16(
    a: *const f32, b: *const u16, c: *mut f32, m: usize, n: usize, k: usize,
) {
    use core::arch::x86_64::*;
    for i in 0..m {
        let ar = a.add(i * k);
        for j in 0..n {
            let br = b.add(j * k);
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut kk = 0usize;
            while kk + 16 <= k {
                let (w0, w1) = bf16_widen_16(br.add(kk));
                acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(ar.add(kk)), w0, acc0);
                acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(ar.add(kk + 8)), w1, acc1);
                kk += 16;
            }
            let mut s = hsum_ps(acc0) + hsum_ps(acc1);
            while kk < k {
                s += *ar.add(kk) * f32::from_bits((*br.add(kk) as u32) << 16);
                kk += 1;
            }
            *c.add(i * n + j) = s;
        }
    }
}

// ---------- P13 task D: int8 权重 GEMM（反量化在核内）----------
//
// 为什么需要：
//   bf16 已把权重压到 1.76 GB，int8 再减半到 ~0.9 GB。P11 在内核级实测
//   int8 **3.4x**（vs f32），因为本机 GEMM 受**权重字节流量**限制
//   （P11 的 vmlm-why 已定量证明）—— 压缩的正是瓶颈那一侧。
//
// 布局：C[M,N] = (A[M,K] · W_i8[N,K]) · scale[N]
//   每个**输出行**一个 scale（LLM.int8 的标准布局）。
//   scale 放在 `dt` 指名的槽里（Ins 只有 op/fl/dt/dst/a/b，没有第 5 个操作数；
//   已对照 repr(C) 定义核实，不要设想有 _pad 字段）。
//
// why 内层是 f32(f32 激活 × int8 权重)：
//   `_mm256_cvtepi8_epi32` 把 8 个 int8 一次性扩成 8×i32，再 `cvtepi32_ps`
//   转 f32 做 FMA。**scale 的乘法提到内积之外**（整行共用），
//   所以内层不必为每个元素乘 scale。
//
// P11 的 int8 内核已用数值验证过（相对误差 7.2e-04 均匀分布 /
// 4.0e-02 合成离群分布）；这里是**同一算法的引擎侧移植**，
// 差别只在于数据来自常量池而不是 Vec。
//
// ⚠️ P13 实测（真实 MiniCPM 权重，24 层，见 tools/p13_quant_plan.py）：
//   int8 让 argmax 保持 4/4，但 rel = 2.89e-02，**未达 <1e-2 判据**。
//   而且 MiniCPM 权重**没有离群通道**（离群列 0/1536），
//   所以 P11 的 LLM.int8 混合精度方案在这里**不起作用**（rel 只降到 2.74e-02）。
//   误差来源是 24 层累积的普通量化噪声，不是离群值。
//   => 这个内核可用且正确，但"int8 满足 <1e-2"这个结论**不成立**，如实记录。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemm_nt_int8(
    a: *const f32, q: *const i8, scale: *const f32, c: *mut f32,
    m: usize, n: usize, k: usize,
) {
    use core::arch::x86_64::*;
    for i in 0..m {
        let ar = a.add(i * k);
        for j in 0..n {
            let qr = q.add(j * k);
            let mut acc = _mm256_setzero_ps();
            let mut kk = 0usize;
            // 8 个 int8 一次 -> 8×f32
            while kk + 8 <= k {
                let eight = _mm_loadl_epi64(qr.add(kk) as *const __m128i);
                let i32v = _mm256_cvtepi8_epi32(eight);
                let lo = _mm256_cvtepi32_ps(i32v);
                let av = _mm256_loadu_ps(ar.add(kk));
                acc = _mm256_fmadd_ps(av, lo, acc);
                kk += 8;
            }
            let mut s = hsum_ps(acc);
            while kk < k {
                s += *ar.add(kk) * (*qr.add(kk) as f32);
                kk += 1;
            }
            // 整行共用一个 scale，所以乘在内积之外。
            *c.add(i * n + j) = s * *scale.add(j);
        }
    }
}

/// 非 x86_64 的标量回退，语义与上面**逐位一致**（同样的求和次序：
/// 先向量部分、再尾部标量），避免不同平台给出不同结果。
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn gemm_nt_int8(
    a: *const f32, q: *const i8, scale: *const f32, c: *mut f32,
    m: usize, n: usize, k: usize,
) {
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for kk in 0..k {
                s += *a.add(i * k + kk) * (*q.add(j * k + kk) as f32);
            }
            *c.add(i * n + j) = s * *scale.add(j);
        }
    }
}

// ---------- LayerNorm ----------
pub unsafe fn layernorm(
    x: *mut f32,
    g: *const f32,
    b: *const f32,
    rows: usize,
    cols: usize,
    eps: f32,
) {
    for r in 0..rows {
        let p = x.add(r * cols);
        let mut mean = 0.0f32;
        for i in 0..cols {
            mean += *p.add(i);
        }
        mean /= cols as f32;
        let mut var = 0.0f32;
        for i in 0..cols {
            let d = *p.add(i) - mean;
            var += d * d;
        }
        var /= cols as f32;
        let inv = 1.0 / sqrt_f(var + eps);
        for i in 0..cols {
            *p.add(i) = (*p.add(i) - mean) * inv * *g.add(i) + *b.add(i);
        }
    }
}

// ---------- RMSNorm（P10：Llama / MiniCPM 用）----------
//
// 与 LayerNorm 的三点差别（不能复用的原因）：
//   1. **不减均值** —— 只按均方归一化，所以没有 mean 那一趟归约。
//   2. **没有 bias** —— Llama 的 RMSNorm 只有 gamma。
//   3. **eps 加在均方上**，不是加在方差上。
//
//   y_i = x_i / sqrt(mean(x^2) + eps) * g_i
//
// 注意 `g` 是逐列（per-channel）权重，行内不动。
pub unsafe fn rmsnorm(
    dst: *mut f32,
    src: *const f32,
    g: *const f32,
    rows: usize,
    cols: usize,
    eps: f32,
) {
    for r in 0..rows {
        let p = src.add(r * cols);
        let q = dst.add(r * cols);
        let mut ss = 0.0f32;
        for i in 0..cols {
            let v = *p.add(i);
            ss += v * v;
        }
        ss /= cols as f32;
        let inv = 1.0 / sqrt_f(ss + eps);
        for i in 0..cols {
            *q.add(i) = *p.add(i) * inv * *g.add(i);
        }
    }
}

#[inline(always)]
pub fn sqrt_f(x: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        return _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(x)));
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let mut g = x * 0.5;
        let mut i = 0;
        while i < 20 {
            g = 0.5 * (g + x / g);
            i += 1;
        }
        g
    }
}

// ---------- softmax（行内，可选因果掩码）----------
pub unsafe fn softmax(p: *mut f32, rows: usize, cols: usize, causal: bool) {
    for r in 0..rows {
        let row = p.add(r * cols);
        let lim = if causal { r + 1 } else { cols };
        let mut mx = f32::NEG_INFINITY;
        for i in 0..lim {
            let v = *row.add(i);
            if v > mx {
                mx = v;
            }
        }
        let mut sum = 0.0f32;
        for i in 0..lim {
            let e = exp_f(*row.add(i) - mx);
            *row.add(i) = e;
            sum += e;
        }
        let inv = 1.0 / sum;
        for i in 0..lim {
            *row.add(i) *= inv;
        }
        for i in lim..cols {
            *row.add(i) = 0.0;
        }
    }
}

// ---------- OUTER：S[i,j] += scale * v[i] * k[j]（ESERNN 递归态写入）----------
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn outer_avx2(s: *mut f32, v: *const f32, k: *const f32, m: usize, n: usize, sc: f32) {
    for i in 0..m {
        let vi = _mm256_set1_ps(*v.add(i) * sc);
        let mut j = 0;
        while j + 8 <= n {
            let kv = _mm256_loadu_ps(k.add(j));
            let sv = _mm256_loadu_ps(s.add(i * n + j));
            _mm256_storeu_ps(s.add(i * n + j), _mm256_fmadd_ps(vi, kv, sv));
            j += 8;
        }
        while j < n {
            *s.add(i * n + j) += *v.add(i) * sc * *k.add(j);
            j += 1;
        }
    }
}

pub unsafe fn outer(s: *mut f32, v: *const f32, k: *const f32, m: usize, n: usize, sc: f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::interp::HAS_AVX2 {
            outer_avx2(s, v, k, m, n, sc);
            return;
        }
    }
    for i in 0..m {
        for j in 0..n {
            *s.add(i * n + j) += *v.add(i) * sc * *k.add(j);
        }
    }
}

// ---------- 指数衰减：S *= decay（逐行广播）----------
pub unsafe fn scale_rows(s: *mut f32, d: *const f32, rows: usize, cols: usize) {
    for i in 0..rows {
        let m = *d.add(i);
        for j in 0..cols {
            *s.add(i * cols + j) *= m;
        }
    }
}

// ---------- TOPK_THRESH：取第 k 大的值作阈值（不排序，sparse_dyn 核心）----------
pub unsafe fn topk_thresh(x: *const f32, n: usize, k: usize) -> f32 {
    // 用计数法：对小 n 直接放栈数组
    let mut cnt = 0usize;
    let mut best = f32::NEG_INFINITY;
    // 简单选择：重复找第 k 大（k 通常小）
    let mut used = [false; 8192];
    let mut found = 0;
    while found < k && cnt < n {
        let mut bi = usize::MAX;
        let mut bv = f32::NEG_INFINITY;
        for i in 0..n {
            if !*used.get_unchecked(i) {
                let v = *x.add(i);
                if v > bv {
                    bv = v;
                    bi = i;
                }
            }
        }
        if bi == usize::MAX {
            break;
        }
        *used.get_unchecked_mut(bi) = true;
        best = bv;
        found += 1;
        cnt += 1;
    }
    best
}

// =====================================================================
// P18：稀疏连接（连接稀疏 + 稀疏激活）的向量化内核
//
// 为什么新增这一节，而不是去改 interp.rs 里已有的标量循环 —— 见
// docs/P18_步骤0_剖析实测.md。要点：
//
//   实测（QPC，d=1536 k0=20 k=8 T=128 L=24，batch=4 seq=32）：
//       qkv 稠密投影   3738 ms/step   31.3%
//       Wo  稠密投影   3409 ms/step   28.5%
//       topk 选连接    3347 ms/step   28.0%
//       lm_head         799 ms/step    6.7%
//       attention       577 ms/step    4.8%
//       sparse matvec    72 ms/step    0.6%   <-- 任务书点名要优化的那个
//
//   ⇒ 仅按任务书点名的「先优化 OP_SPARSE_MATVEC」做，最多省 72 ms/step，
//     不可能达到 <5 s/step。所以本节的四个内核按**实测占比**排序投入：
//     sparse_topk_avx2（28%）→ sparse_forward_avx2（0.6%，但任务书点名）
//     → sparse_gemm_bwd_w_avx2（反向同构，前向的 2 倍量级）。
//
//   稠密 GEMM 那 59.8% 不在这里修 —— 它走 core/src/kernels.rs 里**已有**的
//   gemm_nt AVX2 8x8 微内核，训练侧此前从未调用它。接线在 train 侧做。
//
// ---- 位精确（bit-exact）是硬要求，不是可喜的附加项 ----
//
// 本项目对拍一律用「逐位相同」（conn_sparse.rs 的 bitwise_diff_u32、
// bwd_mismatch 里的 to_bits() 比较都出于同一理由）：`==` 对 NaN 恒假、
// 对 +0.0/-0.0 恒真，会把「符号错了」当成通过。
//
// 这些内核因此**刻意不改求和顺序**：累加仍按 k0 内升序、用单个标量累加器
// （不是 8 路归约），所以与标量版逐位一致。代价是放弃了部分并行度，收益是
// 「提速之后算的还是同一个数」可以被机器验证，而不是靠人相信。
// =====================================================================

/// 强度 s = |w_j * x[src_j]|：一次算出整行的 k0 个强度。
///
/// 这是 `row_topk` 里最贵的一步（k0 次 gather + 乘 + 绝对值），也是
/// `OP_TOPK_BY_STRENGTH` 的等价物。gather 用 `_mm256_i32gather_ps`。
///
/// **浮点语义警告**：`_mm256_andnot_ps` 清符号位实现的 `abs` 与标量
/// `f32::abs` 逐位一致（都只清 bit31），包括 NaN 的位型；所以这里可以放心
/// 用位运算而不是比较。若改成 `max(v,-v)`，NaN 与 ±0 就会与标量版分叉。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn sparse_strength_row_avx2(
    wrow: *const f32,
    srcrow: *const f32,
    x: *const f32,
    k0: usize,
    out: *mut f32,
) {
    let sign = _mm256_set1_ps(-0.0);
    let mut j = 0usize;
    while j + 8 <= k0 {
        let wv = _mm256_loadu_ps(wrow.add(j));
        // src ids are stored as f32 (the whole engine stores indices as f32);
        // convert to i32 for the gather index vector.
        let sid_f = _mm256_loadu_ps(srcrow.add(j));
        let sid = _mm256_cvttps_epi32(sid_f);
        let xv = _mm256_i32gather_ps(x, sid, 4);
        let prod = _mm256_mul_ps(wv, xv);
        let a = _mm256_andnot_ps(sign, prod);
        _mm256_storeu_ps(out.add(j), a);
        j += 8;
    }
    while j < k0 {
        let sid = *srcrow.add(j) as usize;
        *out.add(j) = (*wrow.add(j) * *x.add(sid)).abs();
        j += 1;
    }
}

/// **保符号**的乘积行：out[j] = w[j] * x[src[j]]，整行 k0 个一次算完。
///
/// ===========================================================================
/// ⚠️ 与 `sparse_strength_row_avx2` 的区别是本函数存在的**全部理由**
/// ===========================================================================
///
/// 上面那个函数返回的是 **|w*x|**（它用 `_mm256_andnot_ps(sign, prod)` 清符号位，
/// 因为它是给 top-k **选强度**用的 —— 强度按定义非负）。
/// 本函数是给**前向求和**用的，**必须保留符号**。
///
/// 这两者曾经被混用，后果是**符号级错误**（详见 core/src/b1_pure.rs 文件头
/// 的 S1 回退记录）：把强度内核拿去算乘积，会把每一个负项翻成正项。
/// 实测 30720 个值里 15314 个不匹配，**其中 15314 个是纯符号翻转（100%）**。
/// ⇒ 两者**不可互换**。本函数刻意与它并列放在一起，就是为了让「选错」这件事
///   在代码结构上可见，而不是靠读者记得。
///
/// ===========================================================================
/// 为什么这样向量化仍然 bit-exact（这是硬约束，不是可选优化）
/// ===========================================================================
///
/// `b1_pure.rs` 把累加顺序写成契约：**行内升序 j、单个标量累加器、首项直接赋值**。
/// 浮点加法不满足结合律，所以**归约的向量化会改变位型**（实测 8 路归约让
/// 841/1536 个输出分叉）。
///
/// ⇒ 所以本函数**只向量化「取乘积」，不碰归约**：
///   * 每个乘积 `w[j]*x[src[j]]` 是**逐元素独立**的，向量化它不改变任何单个
///     乘积的位型（`_mm256_mul_ps` 与标量 `*` 在 IEEE-754 下逐位相同）；
///   * 调用方拿回这 k0 个乘积后，**仍按 j 升序、单标量累加器、首项直接赋值**
///     求和，一个字节都没变。
/// 即：向量化的是「取值」，不是「累加」。实测 S1 bit-different = 0/1536。
///
/// **不用 FMA**：`_mm256_fmadd_ps` 会把乘加融合成单次运算（舍入一次而不是两次），
/// 与标量的 `w*x` 然后 `+` **位型不同**。本函数只做 mul，不做 add。
///
/// `k0` 不要求是 8 的倍数：尾部走标量，与向量部分**逐位一致**（同一批乘积）。
///
/// # Safety
/// `wrow` / `srcrow` 各至少 `k0` 个元素；`srcrow` 里每个值都必须落在
/// `[0, n_neurons)`；`x` 至少 `n_neurons` 个 f32；`out` 至少 `k0` 个 f32 且可写。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn sparse_product_row_avx2(
    wrow: *const f32,
    srcrow: *const f32,
    x: *const f32,
    k0: usize,
    out: *mut f32,
) {
    let mut j = 0usize;
    while j + 8 <= k0 {
        let wv = _mm256_loadu_ps(wrow.add(j));
        // src 以 f32 携带索引（引擎惯例），转 i32 供 gather 使用。
        let sid_f = _mm256_loadu_ps(srcrow.add(j));
        let sid = _mm256_cvttps_epi32(sid_f);
        let xv = _mm256_i32gather_ps(x, sid, 4);
        // 只做 mul —— **不取 abs、不 FMA**。这两条都是 bit-exact 的前提。
        let prod = _mm256_mul_ps(wv, xv);
        _mm256_storeu_ps(out.add(j), prod);
        j += 8;
    }
    // 标量尾块：与向量部分同一批乘积，逐位一致。
    while j < k0 {
        let sid = *srcrow.add(j) as usize;
        *out.add(j) = *wrow.add(j) * *x.add(sid);
        j += 1;
    }
}

/// 标量参考版：`sparse_product_row_avx2` 的逐位等价实现（无 AVX2 时使用，
/// 也是 AVX2 版的对拍基准）。
///
/// # Safety
/// 同 `sparse_product_row_avx2`。
#[inline]
pub unsafe fn sparse_product_row_scalar(
    wrow: *const f32,
    srcrow: *const f32,
    x: *const f32,
    k0: usize,
    out: *mut f32,
) {
    let mut j = 0usize;
    while j < k0 {
        let sid = *srcrow.add(j) as usize;
        *out.add(j) = *wrow.add(j) * *x.add(sid);
        j += 1;
    }
}

// =====================================================================
// B1 SIMD 归约（PATH: B1）—— 验收标准已从 bit-exact 放宽为舍入级差异
// =====================================================================
//
// ===========================================================================
// ⚠️ 丢掉的是什么（必须写清楚，因为它是一条被主动放弃的契约）
// ===========================================================================
//
// 在此之前，B1 的累加顺序是**契约**：行内升序 j、单标量累加器、首项直接赋值。
// 判据 A1 用 `f32::to_bits()` 逐位比较。下面这些内核**违反**该契约：它们把
// k0 个乘积分成 8 路（或 4 路）并行部分和，再做横向归约。
//
// 浮点加法不满足结合律 ⇒ **位型必然改变**。实测（`vmlm-b1-eps`，d=1536 k0=20）：
//
//     单层   max|diff| = 2.980e-08   与标量位型相同者 687/1536 (44.7%)
//     24 层  max|diff| = 1.526e-05   残差流 max|value| = 1.113e+02
//            => 相对误差 1.371e-07
//
// ⇒ 所以这不是"免费的加速"：它是**用逐位契约换取速度**的显式决策，
//   由用户裁决（原文：「不需要 bit-exact，计算本质一样就不必这么追究了」）。
//   标量路径**保留**（`b1_reduce_scalar`），作为精度对账的参考实现。
//
// ===========================================================================
// 为什么横向归约的顺序被写死（这是验收标准的一部分）
// ===========================================================================
//
// 验收第一条是**确定性**：同输入必须同输出。若把横向归约交给编译器
// （或写成 `_mm256_hadd_ps` 的嵌套、或依赖 `sum()` 的展开），不同优化级别
// 可能给出不同位型。所以这里的 8→4→2→1 折叠顺序是**显式写死**的：
// 实测 5 次独立遍历位型完全相同，且清零重建后仍相同（无未初始化读）。

/// B1 的标量归约：**当前契约的参考实现**（升序 j、单累加器、首项直接赋值）。
///
/// 保留理由：验收标准放宽为「差异 ≤ ε」之后，**必须有一个参考物去算那个差异**。
/// 用它做对账基准，而不是拿 SIMD 版自己跟自己比。
///
/// # Safety
/// `prod` 至少 `k0` 个 f32。
#[inline]
pub unsafe fn b1_reduce_scalar(prod: *const f32, k0: usize) -> f32 {
    let mut acc = 0.0f32;
    let mut first = true;
    let mut j = 0usize;
    while j < k0 {
        let t = *prod.add(j);
        acc = if first { t } else { acc + t };
        first = false;
        j += 1;
    }
    acc
}

/// SIMD 8 路横向归约。
///
/// 折叠顺序 **写死**（8→4→2→1），不交给编译器 —— 见上「为什么横向归约的
/// 顺序被写死」。尾部（k0 % 8）按升序标量累加。
///
/// 注意：这**不是** `_mm256_hadd_ps`。`hadd` 在 Intel 上有跨 lane 的
/// 特殊语义（lane 内相邻相加），用它会让结果依赖微架构实现细节；
/// 而 `extractf128 + add` 的语义在手册里是确定的。
///
/// # Safety
/// `prod` 至少 `k0` 个 f32。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn b1_reduce_simd8(prod: *const f32, k0: usize) -> f32 {
    let mut acc = _mm256_setzero_ps();
    let mut j = 0usize;
    while j + 8 <= k0 {
        acc = _mm256_add_ps(acc, _mm256_loadu_ps(prod.add(j)));
        j += 8;
    }
    // 写死的横向折叠：8 -> 4 -> 2 -> 1
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let mut v = _mm_add_ps(lo, hi);
    let sh = _mm_movehl_ps(v, v);
    v = _mm_add_ps(v, sh);
    let sh2 = _mm_shuffle_ps(v, v, 0x55);
    v = _mm_add_ss(v, sh2);
    let mut s = _mm_cvtss_f32(v);
    // 尾部：升序标量
    while j < k0 {
        s += *prod.add(j);
        j += 1;
    }
    s
}

/// 无 AVX2 时的回退：与 `b1_reduce_simd8` **不同**（它就是标量契约版）。
/// 判据不需要按 ISA 分支，因为验收是 ε 级而非逐位级。
///
/// # Safety
/// 同 `b1_reduce_scalar`。
#[inline]
pub unsafe fn b1_reduce_simd8_fallback(prod: *const f32, k0: usize) -> f32 {
    b1_reduce_scalar(prod, k0)
}

/// **融合内核**：一步完成「gather 乘积 + 8 路归约」，`src` 为 **u32**。
///
/// 为什么把 src 改成 u32：`vmlm-b1-shipped-u32` 在**发布形态**上实测
/// **1.43x**（0.534 -> 0.374 ns/conn），且两种布局输出 **bit-exact 0/1536**
/// （id < 2^24 时 f32 往返无损）。磁盘格式 `CS_SRC_ENTRY=4` 本来就是 u32，
/// 训练侧存 f32 是纯往返转换。
///
/// 为什么融合：不融合就要先把 k0 个乘积写进缓冲、再读回来做归约
/// （`vmlm-b1-reconcile` 实测那趟 store/reload 值 0.127 ns/conn）。
/// 融合后乘积留在寄存器里直接进累加器。
///
/// ⚠️ **不取 abs、不用 FMA**：
///   * 取 abs 是强度内核的语义，会翻掉每个负项（见 `sparse_product_row_avx2`
///     的警告与 `b1_pure.rs` 的 S1 回退记录）；
///   * FMA 把乘加融成一次舍入，与「先乘后加」位型不同 —— 本函数的验收是
///     ε 级，但仍**不引入额外误差源**：只用 mul + add，与参考实现同构。
///
/// # Safety
/// `wrow` 至少 `k0` 个 f32；`srcrow` 至少 `k0` 个 u32 且每个值 < `n_neurons`；
/// `x` 至少 `n_neurons` 个 f32。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn b1_dot_row_u32_avx2(
    wrow: *const f32,
    srcrow: *const u32,
    x: *const f32,
    k0: usize,
) -> f32 {
    let mut acc = _mm256_setzero_ps();
    let mut j = 0usize;
    while j + 8 <= k0 {
        let wv = _mm256_loadu_ps(wrow.add(j));
        // u32 索引直接载入 —— **无 cvttps2dq**，这就是 1.43x 的来源。
        let sid = _mm256_loadu_si256(srcrow.add(j) as *const __m256i);
        let xv = _mm256_i32gather_ps(x, sid, 4);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(wv, xv));
        j += 8;
    }
    // 写死的横向折叠，与 b1_reduce_simd8 完全一致
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let mut v = _mm_add_ps(lo, hi);
    let sh = _mm_movehl_ps(v, v);
    v = _mm_add_ps(v, sh);
    let sh2 = _mm_shuffle_ps(v, v, 0x55);
    v = _mm_add_ss(v, sh2);
    let mut s = _mm_cvtss_f32(v);
    // 尾部：升序标量
    while j < k0 {
        s += *wrow.add(j) * *x.add(*srcrow.add(j) as usize);
        j += 1;
    }
    s
}

/// 标量参考版（无 AVX2 时用，也是精度对账基准）。`src` 为 u32。
///
/// # Safety
/// 同 `b1_dot_row_u32_avx2`。
#[inline]
pub unsafe fn b1_dot_row_u32_scalar(
    wrow: *const f32,
    srcrow: *const u32,
    x: *const f32,
    k0: usize,
) -> f32 {
    let mut acc = 0.0f32;
    let mut first = true;
    let mut j = 0usize;
    while j < k0 {
        let t = *wrow.add(j) * *x.add(*srcrow.add(j) as usize);
        acc = if first { t } else { acc + t };
        first = false;
        j += 1;
    }
    acc
}

// =====================================================================
// PATH: B1 —— 令牌分块（transposed-activation）稀疏前向
// ---------------------------------------------------------------------
// 为什么需要这一族内核（实测归因，不是猜想）：
//
//   发布版 `b1_forward`（gather-mul + 8 路归约，见 `b1_dot_row_u32_avx2`）
//   实测 0.189 ns/conn = 0.623 cycles/conn @3.3GHz。而一条连接在发布版里
//   恰好消耗 **1/8 次 `_mm256_i32gather_ps`**（k0=20 → 2 次整 gather +
//   4 条标量尾），若该指令吞吐 = 5 cycles/ymm，则正好是 0.625 cycles/conn。
//   ⇒ **发布版 B1 是 gather 吞吐受限，不是乘法受限。**
//   ⇒ 唯一能突破它的办法是**减少 gather 次数**，不是把 gather 换得更花哨。
//
// 手段：把「一条连接 × 8 个 token」放进一个 ymm 通道，即对**激活**做
// 【令牌维向量化】，把 x 的布局从 token-major [T][d] 换成 **source-major**
// [d][t8]（t8 = 令牌数向上取整到 8，同时是行距）。于是：
//
//   发布版： 每连接 1 次 gather(8)（8 个源、1 个 token）→ 1 个有用值/通道
//   本内核： 每连接 1 次 **连续** loadu_ps(8)（1 个源、8 个 token）→ 8 个有用值
//
// 即：**gather 从「每连接一次」变成「每连接零次」**。
//
// ===========================================================================
// 浮点契约：本内核不是「ε 级」，而是**逐位等于标量契约参考**
// ===========================================================================
// 每个通道只累加**一个 token**，且：
//   * j **升序**（1..k0），与 `b1_dot_row_u32_scalar` 同序；
//   * **首项直接赋值**（先把 j=0 的乘积放进累加器，再从 j=1 起 `+`）——
//     `0.0 + (-0.0) = +0.0`，所以「从 0 起加」会丢掉 -0.0 的位型；
//   * 通道之间**没有横向归约**，所以 8 路分组从不发生；
//   * **不用 FMA**：`_mm256_fmadd_ps` 把乘加融成一次舍入，与「先乘后加」位型不同。
// ⇒ 本内核的输出与 `b1_forward_scalar_reference` **逐位相同**（可用
//   `f32::to_bits` 断言，判据见 `vmlm-sparseb-tok` 的 C1）。
//   这比发布版 B1（8 路归约、与标量参考 45.8% 位相同）**更强**，不是更弱。
//
// 代价（必须一起报）：调用方需要一次 [T][d] -> [d][t8] 的转置。
// `transpose_t8_avx2` 用 8x8 分块 SIMD 转置，成本已计入 tok/s 验收。
//
// 边界：`xt` 行距必须是 `t8`；`t8 >= n_tokens`（尾部列若为填充必须已清零，
// 本内核只写 t < n_tokens 的列）。`k0 == 0` 时输出全 0（与发布版一致）。
//
// # Safety
// `w` / `src_u32` 各至少 `n*k0`；`src_u32` 的每个值 < 源维度；
// `xt` 至少 `src_dim * t8`；`yt` 至少 `n * t8` 且可写。

/// 转置 [T][d] -> [d][t8]（source-major），8x8 分块 AVX2。
///
/// 输出 `xt[s*t8 + t] = x[t*d + s]`；`t <= t8`，`t..t8` 的填充列写 0。
///
/// # Safety
/// `x` 至少 `t*d` 个 f32；`xt` 至少 `d*t8` 个 f32 且可写。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn transpose_t8_avx2(x: *const f32, t: usize, d: usize, xt: *mut f32, t8: usize) {
    let mut s0 = 0usize;
    while s0 + 8 <= d {
        let mut t0 = 0usize;
        while t0 + 8 <= t {
            let r0 = _mm256_loadu_ps(x.add((t0 + 0) * d + s0));
            let r1 = _mm256_loadu_ps(x.add((t0 + 1) * d + s0));
            let r2 = _mm256_loadu_ps(x.add((t0 + 2) * d + s0));
            let r3 = _mm256_loadu_ps(x.add((t0 + 3) * d + s0));
            let r4 = _mm256_loadu_ps(x.add((t0 + 4) * d + s0));
            let r5 = _mm256_loadu_ps(x.add((t0 + 5) * d + s0));
            let r6 = _mm256_loadu_ps(x.add((t0 + 6) * d + s0));
            let r7 = _mm256_loadu_ps(x.add((t0 + 7) * d + s0));
            let a0 = _mm256_unpacklo_ps(r0, r1);
            let a1 = _mm256_unpackhi_ps(r0, r1);
            let a2 = _mm256_unpacklo_ps(r2, r3);
            let a3 = _mm256_unpackhi_ps(r2, r3);
            let a4 = _mm256_unpacklo_ps(r4, r5);
            let a5 = _mm256_unpackhi_ps(r4, r5);
            let a6 = _mm256_unpacklo_ps(r6, r7);
            let a7 = _mm256_unpackhi_ps(r6, r7);
            let b0 = _mm256_shuffle_ps(a0, a2, 0x44);
            let b1 = _mm256_shuffle_ps(a0, a2, 0xEE);
            let b2 = _mm256_shuffle_ps(a1, a3, 0x44);
            let b3 = _mm256_shuffle_ps(a1, a3, 0xEE);
            let b4 = _mm256_shuffle_ps(a4, a6, 0x44);
            let b5 = _mm256_shuffle_ps(a4, a6, 0xEE);
            let b6 = _mm256_shuffle_ps(a5, a7, 0x44);
            let b7 = _mm256_shuffle_ps(a5, a7, 0xEE);
            // c_v = 第 v 个源在 8 个 token 上的值（= 转置后的第 v 行）
            _mm256_storeu_ps(xt.add((s0 + 0) * t8 + t0), _mm256_permute2f128_ps(b0, b4, 0x20));
            _mm256_storeu_ps(xt.add((s0 + 1) * t8 + t0), _mm256_permute2f128_ps(b1, b5, 0x20));
            _mm256_storeu_ps(xt.add((s0 + 2) * t8 + t0), _mm256_permute2f128_ps(b2, b6, 0x20));
            _mm256_storeu_ps(xt.add((s0 + 3) * t8 + t0), _mm256_permute2f128_ps(b3, b7, 0x20));
            _mm256_storeu_ps(xt.add((s0 + 4) * t8 + t0), _mm256_permute2f128_ps(b0, b4, 0x31));
            _mm256_storeu_ps(xt.add((s0 + 5) * t8 + t0), _mm256_permute2f128_ps(b1, b5, 0x31));
            _mm256_storeu_ps(xt.add((s0 + 6) * t8 + t0), _mm256_permute2f128_ps(b2, b6, 0x31));
            _mm256_storeu_ps(xt.add((s0 + 7) * t8 + t0), _mm256_permute2f128_ps(b3, b7, 0x31));
            t0 += 8;
        }
        // t 方向尾部：标量
        while t0 < t {
            let mut v = 0usize;
            while v < 8 {
                *xt.add((s0 + v) * t8 + t0) = *x.add(t0 * d + s0 + v);
                v += 1;
            }
            t0 += 1;
        }
        s0 += 8;
    }
    // d 方向尾部：标量
    while s0 < d {
        let mut t0 = 0usize;
        while t0 < t {
            *xt.add(s0 * t8 + t0) = *x.add(t0 * d + s0);
            t0 += 1;
        }
        s0 += 1;
    }
    // 填充列清零（t..t8）
    let mut s = 0usize;
    while s < d {
        let mut t0 = t;
        while t0 < t8 {
            *xt.add(s * t8 + t0) = 0.0;
            t0 += 1;
        }
        s += 1;
    }
}

/// `transpose_t8_avx2` 的标量参考（也是无 AVX2 时的回退）。
///
/// # Safety
/// 同 `transpose_t8_avx2`。
#[inline]
pub unsafe fn transpose_t8_scalar(x: *const f32, t: usize, d: usize, xt: *mut f32, t8: usize) {
    let mut s = 0usize;
    while s < d {
        let mut t0 = 0usize;
        while t0 < t {
            *xt.add(s * t8 + t0) = *x.add(t0 * d + s);
            t0 += 1;
        }
        while t0 < t8 {
            *xt.add(s * t8 + t0) = 0.0;
            t0 += 1;
        }
        s += 1;
    }
}

/// 令牌分块 B1 前向：每个 ymm 通道 = **一个 token**，内层对 j 升序标量累加。
///
/// 输出 `yt[i*t8 + t]`（转置输出：**一行 y 连续写 8 个 token**，无 scatter）。
/// 返回乘加次数 = `n * k0`（与发布版同口径，便于对拍）。
///
/// # Safety
/// 见本节说明；`xt` 行距 = `t8 >= n_tokens`。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn b1_forward_t8_avx2(
    w: *const f32,
    src_u32: *const u32,
    n: usize,
    k0: usize,
    xt: *const f32,
    t8: usize,
    n_tokens: usize,
    yt: *mut f32,
) -> usize {
    let nb = n_tokens / 8;
    let mut i = 0usize;
    while i < n {
        let base = i * k0;
        let yrow = yt.add(i * t8);
        if k0 == 0 {
            let mut t0 = 0usize;
            while t0 < n_tokens {
                *yrow.add(t0) = 0.0;
                t0 += 1;
            }
        } else {
            let mut b = 0usize;
            while b < nb {
                let t0 = b * 8;
                // j = 0：**首项直接赋值**（不是从 0.0 起加）——契约要求。
                let sid = *src_u32.add(base) as usize;
                let mut acc = _mm256_mul_ps(
                    _mm256_set1_ps(*w.add(base)),
                    _mm256_loadu_ps(xt.add(sid * t8 + t0)),
                );
                let mut j = 1usize;
                while j < k0 {
                    let sid = *src_u32.add(base + j) as usize;
                    let wv = _mm256_set1_ps(*w.add(base + j));
                    let xv = _mm256_loadu_ps(xt.add(sid * t8 + t0));
                    acc = _mm256_add_ps(acc, _mm256_mul_ps(wv, xv));
                    j += 1;
                }
                _mm256_storeu_ps(yrow.add(t0), acc);
                b += 1;
            }
            // 令牌尾部（n_tokens % 8）：标量，同一契约顺序
            let mut t0 = nb * 8;
            while t0 < n_tokens {
                let mut acc = 0.0f32;
                let mut first = true;
                let mut j = 0usize;
                while j < k0 {
                    let sid = *src_u32.add(base + j) as usize;
                    let term = *w.add(base + j) * *xt.add(sid * t8 + t0);
                    acc = if first { term } else { acc + term };
                    first = false;
                    j += 1;
                }
                *yrow.add(t0) = acc;
                t0 += 1;
            }
        }
        i += 1;
    }
    n * k0
}

/// 令牌分块 B1 前向 —— **4 个 token 块（32 token）一次处理**的版本。
///
/// 为什么在 1 块版本之上还要这个（实测驱动的两个动机）：
///   1. **摊薄每连接的固定开销**：1 块版本每条连接要做「load src / 算地址 /
///      broadcast w」3 次，却只服务 8 个乘加；4 块版本同样 3 次服务 32 个。
///      每 8 个乘加的 uop 数从 6 降到 3.75。
///   2. **4 条独立的累加器链**：j 升序的 `acc = acc + w*x` 是一条串行依赖
///      （vaddps 延迟 ~2.5 cycles），20 条串起来 50 cycles。4 条链并行后
///      吞吐受限而不是延迟受限。
///
/// 浮点契约与 `b1_forward_t8_avx2` **完全相同**（每通道=j 升序单累加器、
/// 首项赋值、无 FMA、无横向归约）⇒ 同样逐位等于标量契约参考。
///
/// # Safety
/// 同 `b1_forward_t8_avx2`。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn b1_forward_t8x4_avx2(
    w: *const f32,
    src_u32: *const u32,
    n: usize,
    k0: usize,
    xt: *const f32,
    t8: usize,
    n_tokens: usize,
    yt: *mut f32,
) -> usize {
    let nblk = n_tokens / 8; // 完整 8-token 块数
    let ngrp = nblk / 4; // 4 块一组
    let mut i = 0usize;
    while i < n {
        let base = i * k0;
        let yrow = yt.add(i * t8);
        if k0 == 0 {
            let mut t0 = 0usize;
            while t0 < n_tokens {
                *yrow.add(t0) = 0.0;
                t0 += 1;
            }
        } else {
            // ---- 主体：4 块一组 ----
            let mut g = 0usize;
            while g < ngrp {
                let t0 = g * 32;
                let sid0 = *src_u32.add(base) as usize;
                let w0 = _mm256_set1_ps(*w.add(base));
                let p = xt.add(sid0 * t8 + t0);
                let mut a0 = _mm256_mul_ps(w0, _mm256_loadu_ps(p));
                let mut a1 = _mm256_mul_ps(w0, _mm256_loadu_ps(p.add(8)));
                let mut a2 = _mm256_mul_ps(w0, _mm256_loadu_ps(p.add(16)));
                let mut a3 = _mm256_mul_ps(w0, _mm256_loadu_ps(p.add(24)));
                let mut j = 1usize;
                while j < k0 {
                    let sid = *src_u32.add(base + j) as usize;
                    let wv = _mm256_set1_ps(*w.add(base + j));
                    let q = xt.add(sid * t8 + t0);
                    a0 = _mm256_add_ps(a0, _mm256_mul_ps(wv, _mm256_loadu_ps(q)));
                    a1 = _mm256_add_ps(a1, _mm256_mul_ps(wv, _mm256_loadu_ps(q.add(8))));
                    a2 = _mm256_add_ps(a2, _mm256_mul_ps(wv, _mm256_loadu_ps(q.add(16))));
                    a3 = _mm256_add_ps(a3, _mm256_mul_ps(wv, _mm256_loadu_ps(q.add(24))));
                    j += 1;
                }
                _mm256_storeu_ps(yrow.add(t0), a0);
                _mm256_storeu_ps(yrow.add(t0 + 8), a1);
                _mm256_storeu_ps(yrow.add(t0 + 16), a2);
                _mm256_storeu_ps(yrow.add(t0 + 24), a3);
                g += 1;
            }
            // ---- 剩余完整块：1 块一组 ----
            let mut b = ngrp * 4;
            while b < nblk {
                let t0 = b * 8;
                let sid0 = *src_u32.add(base) as usize;
                let mut acc = _mm256_mul_ps(
                    _mm256_set1_ps(*w.add(base)),
                    _mm256_loadu_ps(xt.add(sid0 * t8 + t0)),
                );
                let mut j = 1usize;
                while j < k0 {
                    let sid = *src_u32.add(base + j) as usize;
                    let wv = _mm256_set1_ps(*w.add(base + j));
                    let xv = _mm256_loadu_ps(xt.add(sid * t8 + t0));
                    acc = _mm256_add_ps(acc, _mm256_mul_ps(wv, xv));
                    j += 1;
                }
                _mm256_storeu_ps(yrow.add(t0), acc);
                b += 1;
            }
            // ---- 令牌尾部（n_tokens % 8）：标量 ----
            let mut t0 = nblk * 8;
            while t0 < n_tokens {
                let mut acc = 0.0f32;
                let mut first = true;
                let mut j = 0usize;
                while j < k0 {
                    let sid = *src_u32.add(base + j) as usize;
                    let term = *w.add(base + j) * *xt.add(sid * t8 + t0);
                    acc = if first { term } else { acc + term };
                    first = false;
                    j += 1;
                }
                *yrow.add(t0) = acc;
                t0 += 1;
            }
        }
        i += 1;
    }
    n * k0
}

/// 令牌分块 B1 前向的**逐位契约参考**（纯标量，无 AVX2 依赖）。
///
/// 唯一用途：给上面的向量内核做 `f32::to_bits` 对拍基准。
///
/// # Safety
/// 同 `b1_forward_t8_avx2`。
pub unsafe fn b1_forward_t8_scalar(
    w: *const f32,
    src_u32: *const u32,
    n: usize,
    k0: usize,
    xt: *const f32,
    t8: usize,
    n_tokens: usize,
    yt: *mut f32,
) -> usize {
    for i in 0..n {
        let base = i * k0;
        for t0 in 0..n_tokens {
            let mut acc = 0.0f32;
            let mut first = true;
            for j in 0..k0 {
                let sid = *src_u32.add(base + j) as usize;
                let term = *w.add(base + j) * *xt.add(sid * t8 + t0);
                acc = if first { term } else { acc + term };
                first = false;
            }
            *yt.add(i * t8 + t0) = acc;
        }
    }
    n * k0
}

/// **融合**强度行 + 列聚合：`agg[j] += Σ_i |w[i][j]·x[src[i][j]]|`，一次调用算完整个矩阵。
///
/// 为什么需要融合（实测动机）：
///   * 逐行调用 `sparse_strength_row_avx2` 时，`#[target_feature]` 函数
///     **不能被内联**回普通调用者，于是 1536 行 = 1536 次调用开销；
///   * 强度行先 store 进缓冲、再 load 回来做 `agg +=`，多一趟往返。
///
/// 逐位一致性：每个 j 的累加顺序仍是 **i 升序、每元素一次 add**，
/// 与「先算整行强度、再按 i 升序加进 agg」逐位相同；abs 用清符号位，
/// 与 `f32::abs` 逐位相同。所以 B2 的 A1 逐位判据不受影响。
///
/// # Safety
/// `w` / `src` 各至少 `n*k0`；`src` 里每个值 < 源维度；`x` 至少源维度个 f32；
/// `agg` 至少 `k0` 个 f32 且已被调用方清零（本函数只做累加）。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn sparse_strength_agg_avx2(
    w: *const f32,
    src: *const f32,
    x: *const f32,
    n: usize,
    k0: usize,
    agg: *mut f32,
) {
    let sign = _mm256_set1_ps(-0.0);
    let mut i = 0usize;
    while i < n {
        let base = i * k0;
        let mut j = 0usize;
        while j + 8 <= k0 {
            let wv = _mm256_loadu_ps(w.add(base + j));
            let sid_f = _mm256_loadu_ps(src.add(base + j));
            let sid = _mm256_cvttps_epi32(sid_f);
            let xv = _mm256_i32gather_ps(x, sid, 4);
            let prod = _mm256_mul_ps(wv, xv);
            let a = _mm256_andnot_ps(sign, prod);
            let acc = _mm256_loadu_ps(agg.add(j));
            _mm256_storeu_ps(agg.add(j), _mm256_add_ps(acc, a));
            j += 8;
        }
        while j < k0 {
            let sid = *src.add(base + j) as usize;
            *agg.add(j) += (*w.add(base + j) * *x.add(sid)).abs();
            j += 1;
        }
        i += 1;
    }
}

/// **融合**强度行 + 列聚合（`src` 为 **u32** 索引）：省掉每 8 条的 `cvttps2dq`。
///
/// 与本文件 `sparse_strength_agg_avx2` 逐位相同（f32→u32 往返在 < 2^24 时无损，
/// 且磁盘格式本来就是 u32）。逐位论证同前：i 升序、每元素一次 add、abs 清符号位。
///
/// # Safety
/// 同 `sparse_strength_agg_avx2`，但 `src` 是 u32 且每个值 < 源维度。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn sparse_strength_agg_u32_avx2(
    w: *const f32,
    src: *const u32,
    x: *const f32,
    n: usize,
    k0: usize,
    agg: *mut f32,
) {
    let sign = _mm256_set1_ps(-0.0);
    let mut i = 0usize;
    while i < n {
        let base = i * k0;
        let mut j = 0usize;
        while j + 8 <= k0 {
            let wv = _mm256_loadu_ps(w.add(base + j));
            let sid = _mm256_loadu_si256(src.add(base + j) as *const __m256i);
            let xv = _mm256_i32gather_ps(x, sid, 4);
            let prod = _mm256_mul_ps(wv, xv);
            let a = _mm256_andnot_ps(sign, prod);
            let acc = _mm256_loadu_ps(agg.add(j));
            _mm256_storeu_ps(agg.add(j), _mm256_add_ps(acc, a));
            j += 8;
        }
        while j < k0 {
            let sid = *src.add(base + j) as usize;
            *agg.add(j) += (*w.add(base + j) * *x.add(sid)).abs();
            j += 1;
        }
        i += 1;
    }
}

/// 整矩阵强度行（`src` 为 f32，与引擎既有惯例一致）：
/// `out[i*k0 + j] = |w[i][j]·x[src[i][j]]|`。
///
/// 存在理由与 `sparse_strength_agg_avx2` 相同：把 1536 次逐行调用收敛成 1 次
/// （`#[target_feature]` 函数不能被内联回普通调用者）。产出布局与
/// `b2_strength_rows` 的 `out` 完全相同，逐位同标量。
///
/// # Safety
/// `w` / `src` 各至少 `n*k0`；`src` 里每个值 < 源维度；`x` 至少源维度个 f32；
/// `out` 至少 `n*k0` 个 f32 且可写。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn sparse_strength_rows_avx2(
    w: *const f32,
    src: *const f32,
    x: *const f32,
    n: usize,
    k0: usize,
    out: *mut f32,
) {
    let sign = _mm256_set1_ps(-0.0);
    let mut i = 0usize;
    while i < n {
        let base = i * k0;
        let mut j = 0usize;
        while j + 8 <= k0 {
            let wv = _mm256_loadu_ps(w.add(base + j));
            let sid_f = _mm256_loadu_ps(src.add(base + j));
            let sid = _mm256_cvttps_epi32(sid_f);
            let xv = _mm256_i32gather_ps(x, sid, 4);
            let a = _mm256_andnot_ps(sign, _mm256_mul_ps(wv, xv));
            _mm256_storeu_ps(out.add(base + j), a);
            j += 8;
        }
        while j < k0 {
            let sid = *src.add(base + j) as usize;
            *out.add(base + j) = (*w.add(base + j) * *x.add(sid)).abs();
            j += 1;
        }
        i += 1;
    }
}

/// **令牌分块**的强度聚合（**行优先、令牌块内层**）：整矩阵 + 全部完整 8-token 块。
///
/// `agg_t[j*t8 + t] += Σ_i |w[i][j]·xt[src[i][j]*t8 + t]|`（只写 t < nblk*8 的列）。
///
/// 为什么循环顺序是「行 → 槽位 → 令牌块」而不是「令牌块 → 行 → 槽位」：
/// 后者每换一个令牌块就要把整个 `xt`（786 KB @ d=1536/T=128）重新随机访问一遍
/// ⇒ 每个 (i,j) 都是一次 L1 miss。行优先时，(i,j) 的 xt 行只有 512 B，
/// 16 个块的访问全部落在同一条 64 B 行序列上 ⇒ 首次 miss 之后全命中。
/// `agg_t` 每行的 512 B 同样在 L1 里被反复读写。
///
/// 逐位一致性：每个 (j,t) 通道的累加仍是 **i 升序、每元素一次 add**，
/// abs 清符号位（与 `f32::abs` 逐位相同），无跨通道归约。
///
/// # Safety
/// `w`/`src` 各至少 `n*k0`；`src` 每个值 < 源维度；`xt` 至少 `源维度*t8`；
/// `agg_t` 至少 `k0*t8` 且已清零；`nblk*8 <= t8`。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn sparse_strength_agg_t8_avx2(
    w: *const f32,
    src: *const f32,
    xt: *const f32,
    t8: usize,
    n: usize,
    k0: usize,
    nblk: usize,
    agg_t: *mut f32,
) {
    let sign = _mm256_set1_ps(-0.0);
    let mut i = 0usize;
    while i < n {
        let base = i * k0;
        let mut j = 0usize;
        while j < k0 {
            let sid = *src.add(base + j) as usize;
            let wv = _mm256_set1_ps(*w.add(base + j));
            let q = xt.add(sid * t8);
            let mut p = agg_t.add(j * t8);
            let mut b = 0usize;
            while b < nblk {
                let off = b * 8;
                let a = _mm256_andnot_ps(sign, _mm256_mul_ps(wv, _mm256_loadu_ps(q.add(off))));
                _mm256_storeu_ps(p.add(off), _mm256_add_ps(_mm256_loadu_ps(p.add(off)), a));
                b += 1;
            }
            j += 1;
        }
        i += 1;
    }
}

/// **B1 反向融合内核**：一次遍历同时得到 `dw` 与 `dx`。
///
/// ```
/// dw[i*k0+j] += dy[i] * x[src[i*k0+j]]      （只读 x，不需要 w）
/// dx[src[i*k0+j]] += dy[i] * w[i*k0+j]      （散射，AVX2 没有 scatter）
/// ```
///
/// 为什么融合（P1）：两件事**共享** `src` 与 `dy[i]`，且 `x`/`w` 各只需读一次。
/// 分开写就是两遍遍历（src 读 2 次、dy 广播 2 次、循环开销 2 份）。
///
/// 融合方法：**dw 走 8 宽向量（gather + 广播），dx 走逐通道标量**。
/// 这是 AVX2 的物理限制（无 scatter），不是偷懒 —— 但**逐位契约仍然成立**：
///
///   * dw：`_mm256_mul_ps` + `_mm256_add_ps`，**不用 FMA**，每元素一次 mul 一次 add，
///     与标量 `dw += dy*x` 逐位相同；累加顺序仍是 i 升序（每个元素每 i 更新一次）；
///   * dx：逐通道标量 `dx[sid] += dy*w`，顺序是 **i 升序、j 升序**，与手写参考
///     （`for i { for j { dx[sid] += g*w } }`）逐位相同。
///
/// ⇒ 本内核与「`b1_backward`(dw) + `b1_backward_dx`(dx) 分别跑」**逐位相同**，
///   因此 P1 的验收不是 ε 而是**逐位**（比要求的 1e-6 更强）。
///
/// # Safety
/// `w`/`src` 各至少 `n*k0`；`src` 每个值 < 目标维度；`dy` 至少 `n`；
/// `x` 至少源维度；`dw` 至少 `n*k0` 且可写（调用方先清零）；
/// `dx` 至少源维度且**调用方已清零**（本函数只做 `+=`）。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn b1_backward_fused_dwdx_avx2(
    w: *const f32,
    src_u32: *const u32,
    n: usize,
    k0: usize,
    dy: *const f32,
    x: *const f32,
    dw: *mut f32,
    dx: *mut f32,
) {
    let mut sids = [0u32; 8];
    let mut i = 0usize;
    while i < n {
        let base = i * k0;
        let g = *dy.add(i);
        let gb = _mm256_set1_ps(g);
        let mut j = 0usize;
        while j + 8 <= k0 {
            let sid = _mm256_loadu_si256(src_u32.add(base + j) as *const __m256i);
            let xv = _mm256_i32gather_ps(x, sid, 4);
            let wv = _mm256_loadu_ps(w.add(base + j));
            // (1) dw：向量，无 FMA ⇒ 与标量逐位相同
            // [P20-B] 已实测：mul+add→FMA 在本内核**无可测收益**（kbench 0.602→0.630 ns/conn，
            // 在噪声内）却破坏本内核**文档化的逐位契约**（不用 FMA ⇒ 与标量参考逐位相同）
            // ⇒ 按「不快就回退」纪律**已回退**，此行保留为原始 mul+add。
            let acc = _mm256_add_ps(_mm256_loadu_ps(dw.add(base + j)), _mm256_mul_ps(gb, xv));
            _mm256_storeu_ps(dw.add(base + j), acc);
            // (2) dx：逐通道标量（AVX2 无 scatter），j 升序 ⇒ 与手写参考同序
            _mm256_storeu_si256(sids.as_mut_ptr() as *mut __m256i, sid);
            let mut lane = 0usize;
            while lane < 8 {
                let s = sids[lane] as usize;
                // 标量重读 w：值与 wv 的对应通道相同（L1 命中），避免把 ymm 取值化
                *dx.add(s) += g * *w.add(base + j + lane);
                lane += 1;
            }
            j += 8;
        }
        while j < k0 {
            let s = *src_u32.add(base + j) as usize;
            let gx = g * *x.add(s);
            *dw.add(base + j) += gx;
            *dx.add(s) += g * *w.add(base + j);
            j += 1;
        }
        i += 1;
    }
}

/// `b1_backward_fused_dwdx_avx2` 的**标量契约参考**（也是无 AVX2 时的回退）。
///
/// 两条语句的顺序与手写参考完全一致：`dw` 先、`dx` 后，i 升序、j 升序。
/// 注意 `dw` 的乘法是 `dy * x`（与 `b1_backward` 的操作数顺序相同）。
///
/// # Safety
/// 同 `b1_backward_fused_dwdx_avx2`。
#[inline]
pub unsafe fn b1_backward_fused_dwdx_scalar(
    w: *const f32,
    src_u32: *const u32,
    n: usize,
    k0: usize,
    dy: *const f32,
    x: *const f32,
    dw: *mut f32,
    dx: *mut f32,
) {
    for i in 0..n {
        let base = i * k0;
        let g = *dy.add(i);
        for j in 0..k0 {
            let s = *src_u32.add(base + j) as usize;
            *dw.add(base + j) += g * *x.add(s);
            *dx.add(s) += g * *w.add(base + j);
        }
    }
}

/// 单遍 top-k：与本文件 `conn_train::row_topk` 的**选择规则逐条相同**。
///
/// 平局规则（契约，不是实现细节）：强度相等时保留**下标更小**的那个 ——
/// 由「升序扫描 + 只在严格 `>` 时替换/插入」保证。任何一处 `>` 改成 `>=`
/// 都会改变谁是幸存者，`tools/p18_topk_canfail.py` 的对拍会报 FAIL。
///
/// `strength` 是已经算好的强度行（由 `sparse_strength_row_avx2` 产出），
/// 这样选连接与算强度分成两步，各自可以被独立对拍。
///
/// 返回命中数（<= k），`hit` 里是**升序**下标（与引擎 93 的输出序一致）。
pub unsafe fn sparse_topk_from_strength(
    strength: *const f32,
    k0: usize,
    k: usize,
    hit: *mut usize,
    bestv: *mut f32,
) -> usize {
    let kk = if k > k0 { k0 } else { k };
    if kk == 0 {
        return 0;
    }
    let mut nk = 0usize;
    for j in 0..k0 {
        let v = *strength.add(j);
        // 缓冲满了且赢不了最弱的幸存者 -> 跳过（用的是同一个严格 `>`）
        if nk == kk && !(v > *bestv.add(kk - 1)) {
            continue;
        }
        let mut q = if nk < kk { nk } else { kk - 1 };
        while q > 0 && v > *bestv.add(q - 1) {
            *hit.add(q) = *hit.add(q - 1);
            *bestv.add(q) = *bestv.add(q - 1);
            q -= 1;
        }
        *hit.add(q) = j;
        *bestv.add(q) = v;
        if nk < kk {
            nk += 1;
        }
    }
    // 输出升序（引擎 93 的契约）。插入排序，元素至多 kk 个。
    for i in 1..nk {
        let v = *hit.add(i);
        let mut q = i;
        while q > 0 && *hit.add(q - 1) > v {
            *hit.add(q) = *hit.add(q - 1);
            q -= 1;
        }
        *hit.add(q) = v;
    }
    nk
}

/// 稀疏前向：y[i] = Σ_{激活连接} w * x（**只算激活连接，零连接不乘不加**）。
///
/// **求和顺序与标量版完全相同**（k0 内升序、单标量累加器），所以与
/// `conn_train::sparse_forward` 逐位一致。这是刻意的：见本节顶部的说明。
///
/// 返回实际执行的乘加次数 —— 与引擎 `OP_SPARSE_MATVEC` 的 FL_NNZ 同一个量，
/// 是「跳过」这条主张**可失败**的证据（计数器算错就报出来）。
pub unsafe fn sparse_forward_masked(
    w: *const f32,
    src: *const f32,
    act: *const f32,
    x: *const f32,
    y: *mut f32,
    n: usize,
    k0: usize,
) -> usize {
    let mut macs = 0usize;
    for i in 0..n {
        let base = i * k0;
        let mut acc = 0.0f32;
        let mut first = true;
        for j in 0..k0 {
            if *act.add(base + j) == 0.0 {
                continue; // 零连接跳过 —— 不乘、不加、不计数
            }
            let sid = *src.add(base + j) as usize;
            let term = *w.add(base + j) * *x.add(sid);
            acc = if first { term } else { acc + term };
            first = false;
            macs += 1;
        }
        *y.add(i) = acc;
    }
    macs
}

/// 稀疏反向（只对激活连接写梯度）：dW[i][j] += dy[i] * x[src[i][j]]。
///
/// 语义与 `conn_train::sparse_gemm_bwd_w` 相同（累加、非激活严格不写）。
/// 返回写入的连接数。
pub unsafe fn sparse_gemm_bwd_w_masked(
    w_act: *const f32,
    src: *const f32,
    dy: *const f32,
    x: *const f32,
    dw: *mut f32,
    n: usize,
    k0: usize,
) -> usize {
    let mut written = 0usize;
    for i in 0..n {
        let g = *dy.add(i);
        let base = i * k0;
        if g == 0.0 {
            continue;
        }
        for j in 0..k0 {
            if *w_act.add(base + j) == 0.0 {
                continue;
            }
            let sid = *src.add(base + j) as usize;
            *dw.add(base + j) += g * *x.add(sid);
            written += 1;
        }
    }
    written
}

// =====================================================================
// P19：attention / CE 的 AVX2 融合原语
// =====================================================================
// 为什么加这一族（实测归因，不是猜测）：`--profile 1` 的分阶段耗时显示，
// primary 配置（dim 2048 / L12 / attn-every 4 / 逐窗口 attention）下，
// 「嵌入+逐层前向」与「逐层反向」两个相位的大头不是 B1 稀疏内核，而是
// `b1_full_train.rs` 里**手写标量**的 attention core：
//   · q·k / dobuf·v 的点积是**单累加器串行链**（每元素 4 周期延迟、零 ILP）；
//   · 概率加权和 `y[x] += a·v[x]` 的指针无 noalias 信息 ⇒ LLVM 不向量化。
// 本文件里已有的 AVX2 内核全部服务 B1/gemm/激活，attention 路径一条都没用上。
//
// 数值口径（用户已裁决）：B1 前向验收从逐位放宽为 ε 级，本族沿用同一口径。
// FMA 把「乘+加」融成一次舍入、8 路归约改变求和分组 ⇒ **不逐位**，
// 差异量级见 `docs/` 的 ε 实测（单层 3e-8，24 层 1.5e-5）。
//
// 所有入口在非 x86_64 上都有标量回退（保证 crate 仍可移植编译）。



// =====================================================================
// P19-9：Adam 与逐元素缩放的 AVX2 版本
// =====================================================================
// 训练器每步对 ~15 组参数跑 Adam（层 dim*k0 × L + head/embed n_vocab*k0_head），
// 原来每个元素一次标量 `sqrt_f` + 一次标量除法。这里 8 宽向量化：
//   m ← 0.9m + 0.1g ; v ← 0.999v + 0.001g² ; p ← p − lr·(m/c1)/(√(v/c2)+eps)
// 数值口径：`m/c1` 改成 `m·(1/c1)`、除法仍是 `vdivps` ⇒ ε 级差异。

/// Adam 一步（8 宽）。`rc1 = 1/(1−β1^t)`、`rc2 = 1/(1−β2^t)` 由调用方算好。
///
/// # Safety `p`/`g`/`m`/`v` 各至少 `n` 个 f32；`p`/`m`/`v` 可写。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn adam_avx2(
    p: *mut f32,
    g: *const f32,
    m: *mut f32,
    v: *mut f32,
    n: usize,
    lr: f32,
    rc1: f32,
    rc2: f32,
    eps: f32,
) {
    let vb1 = _mm256_set1_ps(0.9);
    let vo1 = _mm256_set1_ps(0.1);
    let vb2 = _mm256_set1_ps(0.999);
    let vo2 = _mm256_set1_ps(0.001);
    let vlr = _mm256_set1_ps(lr);
    let vc1 = _mm256_set1_ps(rc1);
    let vc2 = _mm256_set1_ps(rc2);
    let veps = _mm256_set1_ps(eps);
    let mut i = 0usize;
    while i + 8 <= n {
        let gi = _mm256_loadu_ps(g.add(i));
        let mi = _mm256_fmadd_ps(vb1, _mm256_loadu_ps(m.add(i)), _mm256_mul_ps(vo1, gi));
        let vi = _mm256_fmadd_ps(vb2, _mm256_loadu_ps(v.add(i)), _mm256_mul_ps(vo2, _mm256_mul_ps(gi, gi)));
        _mm256_storeu_ps(m.add(i), mi);
        _mm256_storeu_ps(v.add(i), vi);
        let den = _mm256_add_ps(_mm256_sqrt_ps(_mm256_mul_ps(vi, vc2)), veps);
        let upd = _mm256_div_ps(_mm256_mul_ps(mi, vc1), den);
        _mm256_storeu_ps(p.add(i), _mm256_sub_ps(_mm256_loadu_ps(p.add(i)), _mm256_mul_ps(vlr, upd)));
        i += 8;
    }
    while i < n {
        let gi = *g.add(i);
        let mi = 0.9 * *m.add(i) + 0.1 * gi;
        let vi = 0.999 * *v.add(i) + 0.001 * gi * gi;
        *m.add(i) = mi;
        *v.add(i) = vi;
        *p.add(i) -= lr * (mi * rc1) / (sqrt_f(vi * rc2) + eps);
        i += 1;
    }
}

/// 非 x86_64 回退。 # Safety 同 [`adam_avx2`]。
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn adam_avx2(
    p: *mut f32,
    g: *const f32,
    m: *mut f32,
    v: *mut f32,
    n: usize,
    lr: f32,
    rc1: f32,
    rc2: f32,
    eps: f32,
) {
    for i in 0..n {
        let gi = *g.add(i);
        let mi = 0.9 * *m.add(i) + 0.1 * gi;
        let vi = 0.999 * *v.add(i) + 0.001 * gi * gi;
        *m.add(i) = mi;
        *v.add(i) = vi;
        *p.add(i) -= lr * (mi * rc1) / (sqrt_f(vi * rc2) + eps);
    }
}

/// B1 前向的**小 k0 打包**路径：把 `G = 8/k0` 个**连续行**塞进一个 ymm
/// （每行占 k0 个通道），于是「一次 gather + 一次连续 `w` 载入 + 一次连续
/// `src` 载入」覆盖 G 行 ⇒ 每行的固定开销（函数调用/循环/横向归约）被摊薄 G 倍，
/// **gather 次数也降到每行 1/G 次**。
///
/// 为什么加它（实测，`:kbench --mode smallk`）：发布版 `b1_forward` 对 k0=4
/// 是 **0.4754 ns/conn**，而 k0=8 只要 **0.1896** —— 因为 `while j+8 <= k0`
/// 对 k0<8 一次都不成立，整行落到标量尾部循环，每行 ~20 uops 全是行开销。
/// 本内核（2 行/ymm）实测 **0.2150 ns/conn（2.21x）**；k0=8 的版本顺手把
/// 横向归约从 4 步折成「先 128+128 再 2 步」，实测 **0.1596 ns/conn（1.19x）**，
/// 且与发布版**逐位相同**（kbench 报 max|diff| = 0）。
///
/// 覆盖 k0 ∈ {1,2,4,8}（8/k0 为整数）；其他 k0 返回 0，调用方走原路径。
///
/// 浮点口径：每行只做「k0 个乘积求和」，但归约分组与原路径不同 ⇒ ε 级
/// （k0=8 实测逐位相同）。**不做首项直接赋值**，因此 `-0.0` 的那条位型契约不适用；
/// 调用方是训练侧（验收已放宽为 ε）。
///
/// # Safety
/// `w` / `src_u32` 至少 `n*k0`；`src_u32` 每个值 < 源维度；`x` 至少源维度；`y` 至少 `n`。
/// 返回**已处理的行数**（调用方从该行起用原路径收尾）。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn b1_forward_packed_avx2(
    w: *const f32,
    src_u32: *const u32,
    x: *const f32,
    k0: usize,
    n: usize,
    y: *mut f32,
) -> usize {
    // 128 位内的横向求和
    #[inline(always)]
    unsafe fn hsum128(v: __m128) -> f32 {
        let s = _mm_add_ps(v, _mm_movehl_ps(v, v));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
        _mm_cvtss_f32(s)
    }
    match k0 {
        1 => {
            // 8 行/ymm：8 个乘积就是 8 行的输出，**不需要任何归约**
            let mut i = 0usize;
            while i + 8 <= n {
                let base = i;
                let sid = _mm256_loadu_si256(src_u32.add(base) as *const __m256i);
                let xv = _mm256_i32gather_ps(x, sid, 4);
                let wv = _mm256_loadu_ps(w.add(base));
                _mm256_storeu_ps(y.add(i), _mm256_mul_ps(wv, xv));
                i += 8;
            }
            i
        }
        2 => {
            // 4 行/ymm：每 2 个通道一行
            let mut i = 0usize;
            while i + 4 <= n {
                let base = i * 2;
                let sid = _mm256_loadu_si256(src_u32.add(base) as *const __m256i);
                let xv = _mm256_i32gather_ps(x, sid, 4);
                let wv = _mm256_loadu_ps(w.add(base));
                let p = _mm256_mul_ps(wv, xv);
                let lo = _mm256_castps256_ps128(p);
                let hi = _mm256_extractf128_ps(p, 1);
                *y.add(i) = hsum128(lo);
                *y.add(i + 1) = hsum128(_mm_shuffle_ps(lo, lo, 0x4E)); // [l2,l3,l0,l1]
                *y.add(i + 2) = hsum128(hi);
                *y.add(i + 3) = hsum128(_mm_shuffle_ps(hi, hi, 0x4E));
                i += 4;
            }
            i
        }
        4 => {
            // 2 行/ymm：每 4 个通道一行
            let mut i = 0usize;
            while i + 2 <= n {
                let base = i * 4;
                let sid = _mm256_loadu_si256(src_u32.add(base) as *const __m256i);
                let xv = _mm256_i32gather_ps(x, sid, 4);
                let wv = _mm256_loadu_ps(w.add(base));
                let p = _mm256_mul_ps(wv, xv);
                *y.add(i) = hsum128(_mm256_castps256_ps128(p));
                *y.add(i + 1) = hsum128(_mm256_extractf128_ps(p, 1));
                i += 2;
            }
            i
        }
        8 => {
            // 1 行/ymm（与原内核同形，少一次归约步）
            let mut i = 0usize;
            while i < n {
                let base = i * 8;
                let sid = _mm256_loadu_si256(src_u32.add(base) as *const __m256i);
                let xv = _mm256_i32gather_ps(x, sid, 4);
                let wv = _mm256_loadu_ps(w.add(base));
                let p = _mm256_mul_ps(wv, xv);
                *y.add(i) = hsum128(_mm_add_ps(_mm256_castps256_ps128(p), _mm256_extractf128_ps(p, 1)));
                i += 1;
            }
            i
        }
        _ => 0,
    }
}

/// 非 x86_64 回退：不做任何行，全部交给调用方的标量/原路径。
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn b1_forward_packed_avx2(
    _w: *const f32,
    _src_u32: *const u32,
    _x: *const f32,
    _k0: usize,
    _n: usize,
    _y: *mut f32,
) -> usize {
    0
}

/// [P21-5] attention 反向 passB+passC 的**寄存器分块**版（串行路径）。
///
/// 现状（`attn_core_bwd` 的 passB/passC）：每对 (i,j) 都要 RMW 两条 d 长行
/// （dv[j]/dk[j] 或 dq[i]/k[j]），dobuf[i]/q[i]/k[j] 也各读一遍
/// ⇒ 实测约 66 MB/window 的 L1/L2 流量（3 个 attention 块 × 2 窗口/步）。
///
/// 本内核把「4 行累加器」放进 ymm 寄存器，按 d 的 8 宽分块扫：
///   passB: dv[j] = Σ_{i≥j} P[i][j]·dobuf[i] ,  dk[j] = Σ_{i≥j} D[i][j]·q[i]
///   passC: dq[i] = Σ_{j≤i} D[i][j]·k[j]
/// 每个 (j-block, d-chunk) 内 dobuf[i]/q[i] 只读一次、累加器不进内存
/// ⇒ 流量降到 ~1/4，且累加顺序仍是 i 升序（与标量参考同序，只用 FMA 与向量宽度改变分组）。
///
/// `dq` 必须由调用方清零（本内核用赋值写回）；`dv`/`dk` 也由调用方清零。
///
/// # Safety 各缓冲尺寸同 `attn_core_bwd`；`rowsum` 至少 `t`。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn attn_bwd_bc_blocked_avx2(
    prob: *const f32,
    dprob: *const f32,
    rowsum: *const f32,
    dobuf: *const f32,
    q: *const f32,
    kk: *const f32,
    dv: *mut f32,
    dk: *mut f32,
    dq: *mut f32,
    t: usize,
    d: usize,
    scale: f32,
) {
    const JB: usize = 4;
    // ---- passB ----
    let mut jb = 0usize;
    while jb < t {
        let jn = (t - jb).min(JB);
        let mut x = 0usize;
        while x + 8 <= d {
            let mut adv = [_mm256_setzero_ps(); JB];
            let mut adk = [_mm256_setzero_ps(); JB];
            for i in 0..t {
                let dr = _mm256_loadu_ps(dobuf.add(i * d + x));
                let qr = _mm256_loadu_ps(q.add(i * d + x));
                let rs = *rowsum.add(i);
                let mut l = 0usize;
                while l < jn {
                    let j = jb + l;
                    if j > i {
                        break;
                    }
                    let p = *prob.add(i * t + j);
                    let ds = p * (*dprob.add(i * t + j) - rs) * scale;
                    adv[l] = _mm256_fmadd_ps(_mm256_set1_ps(p), dr, adv[l]);
                    adk[l] = _mm256_fmadd_ps(_mm256_set1_ps(ds), qr, adk[l]);
                    l += 1;
                }
            }
            for l in 0..jn {
                let j = jb + l;
                let vp = dv.add(j * d + x);
                _mm256_storeu_ps(vp, _mm256_add_ps(_mm256_loadu_ps(vp), adv[l]));
                let kp = dk.add(j * d + x);
                _mm256_storeu_ps(kp, _mm256_add_ps(_mm256_loadu_ps(kp), adk[l]));
            }
            x += 8;
        }
        while x < d {
            for i in 0..t {
                let rs = *rowsum.add(i);
                let mut l = 0usize;
                while l < jn {
                    let j = jb + l;
                    if j > i {
                        break;
                    }
                    let p = *prob.add(i * t + j);
                    let ds = p * (*dprob.add(i * t + j) - rs) * scale;
                    *dv.add(j * d + x) += p * *dobuf.add(i * d + x);
                    *dk.add(j * d + x) += ds * *q.add(i * d + x);
                    l += 1;
                }
            }
            x += 1;
        }
        jb += JB;
    }
    // ---- passC ----
    let mut ib = 0usize;
    while ib < t {
        let inb = (t - ib).min(JB);
        let mut x = 0usize;
        while x + 8 <= d {
            let mut adq = [_mm256_setzero_ps(); JB];
            for j in 0..t {
                let kr = _mm256_loadu_ps(kk.add(j * d + x));
                let mut l = 0usize;
                while l < inb {
                    let i = ib + l;
                    if j > i {
                        break;
                    }
                    let p = *prob.add(i * t + j);
                    let ds = p * (*dprob.add(i * t + j) - *rowsum.add(i)) * scale;
                    adq[l] = _mm256_fmadd_ps(_mm256_set1_ps(ds), kr, adq[l]);
                    l += 1;
                }
            }
            for l in 0..inb {
                let i = ib + l;
                _mm256_storeu_ps(dq.add(i * d + x), adq[l]);
            }
            x += 8;
        }
        while x < d {
            for j in 0..t {
                let mut l = 0usize;
                while l < inb {
                    let i = ib + l;
                    if j > i {
                        break;
                    }
                    let p = *prob.add(i * t + j);
                    let ds = p * (*dprob.add(i * t + j) - *rowsum.add(i)) * scale;
                    *dq.add(i * d + x) += ds * *kk.add(j * d + x);
                    l += 1;
                }
            }
            x += 1;
        }
        ib += JB;
    }
}

/// 非 x86_64 回退。 # Safety 同 [`attn_bwd_bc_blocked_avx2`]。
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn attn_bwd_bc_blocked_avx2(
    prob: *const f32,
    dprob: *const f32,
    rowsum: *const f32,
    dobuf: *const f32,
    q: *const f32,
    kk: *const f32,
    dv: *mut f32,
    dk: *mut f32,
    dq: *mut f32,
    t: usize,
    d: usize,
    scale: f32,
) {
    for j in 0..t {
        for i in j..t {
            let p = *prob.add(i * t + j);
            let ds = p * (*dprob.add(i * t + j) - *rowsum.add(i)) * scale;
            for x in 0..d {
                *dv.add(j * d + x) += p * *dobuf.add(i * d + x);
                *dk.add(j * d + x) += ds * *q.add(i * d + x);
            }
        }
    }
    for i in 0..t {
        for j in 0..=i {
            let p = *prob.add(i * t + j);
            let ds = p * (*dprob.add(i * t + j) - *rowsum.add(i)) * scale;
            for x in 0..d {
                *dq.add(i * d + x) += ds * *kk.add(j * d + x);
            }
        }
    }
}

/// **4 行共享一条 B 行**的点积：`r[l] = Σ a_l[x]·b[x]`（l = 0..4）。
///
/// 动机（P22，KV 流量）：attention 前向的 q·k 与反向 passA 的 dobuf·v 都是
/// 「多行对同一条 K/V 行」做点积。逐行调用时同一条 B 行被读 4 次；
/// 这里 4 行一起算，B 行每个 8 宽块只读一次 ⇒ B 侧载入流量降到 1/4
/// （A 侧仍是 4 次，总载入从 8 降到 5 / 32 MAC）。
///
/// `valid[l]`：因果遮罩，0 表示该行不参与（不累加、返回 0）。
///
/// # Safety 四个 A 行与 B 行各至少 `n` 个 f32。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot4_shared_avx2(
    a0: *const f32,
    a1: *const f32,
    a2: *const f32,
    a3: *const f32,
    b: *const f32,
    n: usize,
    valid: [bool; 4],
) -> [f32; 4] {
    let ap = [a0, a1, a2, a3];
    let mut acc = [_mm256_setzero_ps(); 4];
    let mut x = 0usize;
    while x + 8 <= n {
        let bv = _mm256_loadu_ps(b.add(x));
        let mut l = 0usize;
        while l < 4 {
            if valid[l] {
                acc[l] = _mm256_fmadd_ps(_mm256_loadu_ps(ap[l].add(x)), bv, acc[l]);
            }
            l += 1;
        }
        x += 8;
    }
    let mut out = [0.0f32; 4];
    let mut l = 0usize;
    while l < 4 {
        if valid[l] {
            let mut s = hsum256(acc[l]);
            let mut xx = x;
            while xx < n {
                s += *ap[l].add(xx) * *b.add(xx);
                xx += 1;
            }
            out[l] = s;
        }
        l += 1;
    }
    out
}

/// 非 x86_64 回退。 # Safety 同 [`dot4_shared_avx2`]。
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn dot4_shared_avx2(
    a0: *const f32,
    a1: *const f32,
    a2: *const f32,
    a3: *const f32,
    b: *const f32,
    n: usize,
    valid: [bool; 4],
) -> [f32; 4] {
    let ap = [a0, a1, a2, a3];
    let mut out = [0.0f32; 4];
    for l in 0..4 {
        if valid[l] {
            let mut s = 0.0f32;
            for x in 0..n {
                s += *ap[l].add(x) * *b.add(x);
            }
            out[l] = s;
        }
    }
    out
}

/// **bf16 权重（u16，高 16 位为 bf16）的内联扩展点积**（P23 A/B 2）。
///
/// 每个 8 宽块：`vpmovzxwd`(u16→u32) + `vpslld $16`（→ f32 位型）后与激活做 FMA。
/// 每 8 个权重多 2 条微指令，其中 `vpmovzxwd` 落在**端口 5**（与广播/洗牌同端口）。
/// 对照见 [`dot256_f32`]（f32 权重，零扩展开销）——A/B 3「预扩展」等价于直接用 f32 内核。
///
/// # Safety `a`/`b` 各至少 `n` 个元素；`n` 为 8 的倍数时最快。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot256_bf16_avx2(a: *const f32, b: *const u16, n: usize) -> f32 {
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut x = 0usize;
    while x + 16 <= n {
        let av0 = _mm256_loadu_ps(a.add(x));
        let av1 = _mm256_loadu_ps(a.add(x + 8));
        let h0 = _mm_loadu_si128(b.add(x) as *const __m128i);
        let h1 = _mm_loadu_si128(b.add(x + 8) as *const __m128i);
        let f0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h0), 16));
        let f1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h1), 16));
        acc0 = _mm256_fmadd_ps(av0, f0, acc0);
        acc1 = _mm256_fmadd_ps(av1, f1, acc1);
        x += 16;
    }
    while x + 8 <= n {
        let av0 = _mm256_loadu_ps(a.add(x));
        let h0 = _mm_loadu_si128(b.add(x) as *const __m128i);
        let f0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h0), 16));
        acc0 = _mm256_fmadd_ps(av0, f0, acc0);
        x += 8;
    }
    let mut s = hsum256(_mm256_add_ps(acc0, acc1));
    while x < n {
        s += *a.add(x) * f32::from_bits((*b.add(x) as u32) << 16);
        x += 1;
    }
    s
}

/// 非 x86_64 回退。 # Safety 同 [`dot256_bf16_avx2`]。
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn dot256_bf16_avx2(a: *const f32, b: *const u16, n: usize) -> f32 {
    let mut s = 0.0f32;
    for x in 0..n {
        s += *a.add(x) * f32::from_bits((*b.add(x) as u32) << 16);
    }
    s
}





/// [P21-4] 块对齐 head 的**按 token 批量**反向（替代逐 token 的 head_block_bwd_avx2）。
///
/// 逐 token 版每 token 都要把 `dw[nv*k0h]`（生产配置 0.5 MB）整块 RMW 一遍
/// ⇒ tt=256 时单步每个 head 就要读写 256×1 MB；批量版把「对 token 的求和」放进寄存器：
///   dW[v][j]  = Σ_t dY[v][t]·Ht[(v%nb)*k0e + j][t]
///   dHt[s][t] = Σ_{v: v%nb==b} dY[v][t]·w[v][j]   （s = b*k0e + j，t 维向量化、连续 RMW）
/// 其中 dY 是 dlogits 的转置、Ht 是 head 输入的转置（各一次转置，见调用方）。
///
/// # Safety `dy`/`dht` 各至少 `nv*tt` / `nb*k0e*tt`；`ht` 至少 `nb*k0e*tt`；
/// `w`/`dw` 各至少 `nv*k0h`；`tt % 8 == 0`；`nb` 是 2 的幂。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn head_bwd_batched_avx2(
    w: *const f32,
    ht: *const f32,
    dy: *const f32,
    tt: usize,
    nv: usize,
    nb: usize,
    k0e: usize,
    k0h: usize,
    dw: *mut f32,
    dht: *mut f32,
) {
    let mask = nb - 1;
    // (1) dW[v][j]：一行一个 tt 长点积（同一个块的 64 行共用 16 条 Ht 行 ⇒ L1/L2 命中）
    for v in 0..nv {
        let b = v & mask;
        let dyv = dy.add(v * tt);
        let dwrow = dw.add(v * k0h);
        for j in 0..k0h {
            let hr = ht.add((b * k0e + j) * tt);
            let mut acc = _mm256_setzero_ps();
            let mut p = 0usize;
            while p + 16 <= tt {
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(dyv.add(p)), _mm256_loadu_ps(hr.add(p)), acc);
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(dyv.add(p + 8)), _mm256_loadu_ps(hr.add(p + 8)), acc);
                p += 16;
            }
            while p + 8 <= tt {
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(dyv.add(p)), _mm256_loadu_ps(hr.add(p)), acc);
                p += 8;
            }
            *dwrow.add(j) += hsum256(acc);
        }
    }
    // (2) dHt：s = b*k0e+j 的整行累加（连续 32B RMW，无 scatter）
    for v in 0..nv {
        let b = v & mask;
        let dyv = dy.add(v * tt);
        let wrow = w.add(v * k0h);
        for j in 0..k0h {
            let wj = _mm256_set1_ps(*wrow.add(j));
            let dr = dht.add((b * k0e + j) * tt);
            let mut p = 0usize;
            while p + 16 <= tt {
                _mm256_storeu_ps(dr.add(p), _mm256_fmadd_ps(_mm256_loadu_ps(dyv.add(p)), wj, _mm256_loadu_ps(dr.add(p))));
                _mm256_storeu_ps(dr.add(p + 8), _mm256_fmadd_ps(_mm256_loadu_ps(dyv.add(p + 8)), wj, _mm256_loadu_ps(dr.add(p + 8))));
                p += 16;
            }
            while p + 8 <= tt {
                _mm256_storeu_ps(dr.add(p), _mm256_fmadd_ps(_mm256_loadu_ps(dyv.add(p)), wj, _mm256_loadu_ps(dr.add(p))));
                p += 8;
            }
        }
    }
}

/// 非 x86_64 回退。 # Safety 同 [`head_bwd_batched_avx2`]。
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn head_bwd_batched_avx2(
    w: *const f32,
    ht: *const f32,
    dy: *const f32,
    tt: usize,
    nv: usize,
    nb: usize,
    k0e: usize,
    k0h: usize,
    dw: *mut f32,
    dht: *mut f32,
) {
    let mask = nb - 1;
    for v in 0..nv {
        let b = v & mask;
        for j in 0..k0h {
            let mut acc = 0.0f32;
            for p in 0..tt {
                acc += *dy.add(v * tt + p) * *ht.add((b * k0e + j) * tt + p);
                *dht.add((b * k0e + j) * tt + p) += *dy.add(v * tt + p) * *w.add(v * k0h + j);
            }
            *dw.add(v * k0h + j) += acc;
        }
    }
}

// =====================================================================
// P20：令牌分块（source-major）**反向** + 反置转置
// =====================================================================
// 指令级审计的结论（见 docs/P20_指令级审计.md）：发布版 b1_backward_fused_dwdx_avx2
// 每条连接要付「1/8 次 gather + ~4 条 dw 向量 uop + 8 条**标量** dx 散写 uop」；
// 而 source-major 布局把散写换成**连续 32B RMW**、把 dw 的乘加换成按 token 维的向量 FMA：
//   dw[i][j] += Σ_t dyt[i][t]·xt[s][t]   （t 维点积，8 宽）
//   dxt[s][:] += dyt[i][:]·w[i][j]       （连续 axpy，8 宽）
// 实测（kbench --mode t8bwd，n=d=2048, k0=8, t=t8=128，**含 3 次转置**）：
//   发布版 1.348 ms/round → t8 0.671 ms/round = 2.01x
// 前向同理（kbench --mode t8，含 2 次转置）：0.370 → 0.203 ms/round = 1.83x。

/// t8 反向内核（dw + dx 一次遍历，xt 只读一遍）。
///
/// `xt[d][t8]` / `dyt[n][t8]` / `dxt[d][t8]`（dxt 与 dw 由调用方清零并累加）。
///
/// # Safety 各缓冲按上述尺寸；`src_u32` 每个值 < d；`t8` 是 8 的倍数。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn b1_backward_t8_avx2(
    w: *const f32,
    src_u32: *const u32,
    xt: *const f32,
    dyt: *const f32,
    t8: usize,
    n: usize,
    k0: usize,
    dxt: *mut f32,
    dw: *mut f32,
) {
    for i in 0..n {
        let dy_row = dyt.add(i * t8);
        let b = i * k0;
        for j in 0..k0 {
            let s = *src_u32.add(b + j) as usize;
            let wij = *w.add(b + j);
            let xr = xt.add(s * t8);
            let dr = dxt.add(s * t8);
            let vw = _mm256_set1_ps(wij);
            let mut acc = _mm256_setzero_ps();
            let mut p = 0usize;
            while p + 16 <= t8 {
                let y0 = _mm256_loadu_ps(dy_row.add(p));
                let x0 = _mm256_loadu_ps(xr.add(p));
                acc = _mm256_fmadd_ps(y0, x0, acc);
                // dx: 必须用 **dy**（dx[s] += dy[i]·w[i][j]）；第一版误用 x0 ⇒ 梯度整体错，
                // gradcheck 立刻报 ana 比 FD 大 ~300 倍（dw 却是对的，所以只错在 dx 这一路）。
                _mm256_storeu_ps(dr.add(p), _mm256_fmadd_ps(y0, vw, _mm256_loadu_ps(dr.add(p))));
                let y1 = _mm256_loadu_ps(dy_row.add(p + 8));
                let x1 = _mm256_loadu_ps(xr.add(p + 8));
                acc = _mm256_fmadd_ps(y1, x1, acc);
                _mm256_storeu_ps(dr.add(p + 8), _mm256_fmadd_ps(y1, vw, _mm256_loadu_ps(dr.add(p + 8))));
                p += 16;
            }
            while p + 8 <= t8 {
                let y0 = _mm256_loadu_ps(dy_row.add(p));
                let x0 = _mm256_loadu_ps(xr.add(p));
                acc = _mm256_fmadd_ps(y0, x0, acc);
                _mm256_storeu_ps(dr.add(p), _mm256_fmadd_ps(y0, vw, _mm256_loadu_ps(dr.add(p))));
                p += 8;
            }
            *dw.add(b + j) += hsum256(acc);
        }
    }
}

/// 非 x86_64 回退。 # Safety 同 [`b1_backward_t8_avx2`]。
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn b1_backward_t8_avx2(
    w: *const f32,
    src_u32: *const u32,
    xt: *const f32,
    dyt: *const f32,
    t8: usize,
    n: usize,
    k0: usize,
    dxt: *mut f32,
    dw: *mut f32,
) {
    for i in 0..n {
        for j in 0..k0 {
            let s = *src_u32.add(i * k0 + j) as usize;
            let mut acc = 0.0f32;
            for p in 0..t8 {
                acc += *dyt.add(i * t8 + p) * *xt.add(s * t8 + p);
                *dxt.add(s * t8 + p) += *dyt.add(i * t8 + p) * *w.add(i * k0 + j);
            }
            *dw.add(i * k0 + j) += acc;
        }
    }
}

/// 8x8 分块反置：`out[i*os + s] = a[s*in_stride + i]`（dxt[d][t8] -> dx[t][d]）。
///
/// # Safety `a` 至少 `rows*in_stride`；`out` 至少 `cols*os`。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn transpose_t8_back_avx2(
    a: *const f32,
    rows: usize,
    cols: usize,
    out: *mut f32,
    os: usize,
    in_stride: usize,
) {
    let mut s0 = 0usize;
    while s0 < rows {
        let mut i0 = 0usize;
        while i0 < cols {
            let n_s = (rows - s0).min(8);
            let n_i = (cols - i0).min(8);
            if n_s == 8 && n_i == 8 {
                let r0 = _mm256_loadu_ps(a.add((s0 + 0) * in_stride + i0));
                let r1 = _mm256_loadu_ps(a.add((s0 + 1) * in_stride + i0));
                let r2 = _mm256_loadu_ps(a.add((s0 + 2) * in_stride + i0));
                let r3 = _mm256_loadu_ps(a.add((s0 + 3) * in_stride + i0));
                let r4 = _mm256_loadu_ps(a.add((s0 + 4) * in_stride + i0));
                let r5 = _mm256_loadu_ps(a.add((s0 + 5) * in_stride + i0));
                let r6 = _mm256_loadu_ps(a.add((s0 + 6) * in_stride + i0));
                let r7 = _mm256_loadu_ps(a.add((s0 + 7) * in_stride + i0));
                let a0 = _mm256_unpacklo_ps(r0, r1);
                let a1 = _mm256_unpackhi_ps(r0, r1);
                let a2 = _mm256_unpacklo_ps(r2, r3);
                let a3 = _mm256_unpackhi_ps(r2, r3);
                let a4 = _mm256_unpacklo_ps(r4, r5);
                let a5 = _mm256_unpackhi_ps(r4, r5);
                let a6 = _mm256_unpacklo_ps(r6, r7);
                let a7 = _mm256_unpackhi_ps(r6, r7);
                let b0 = _mm256_shuffle_ps(a0, a2, 0x44);
                let b1 = _mm256_shuffle_ps(a0, a2, 0xEE);
                let b2 = _mm256_shuffle_ps(a1, a3, 0x44);
                let b3 = _mm256_shuffle_ps(a1, a3, 0xEE);
                let b4 = _mm256_shuffle_ps(a4, a6, 0x44);
                let b5 = _mm256_shuffle_ps(a4, a6, 0xEE);
                let b6 = _mm256_shuffle_ps(a5, a7, 0x44);
                let b7 = _mm256_shuffle_ps(a5, a7, 0xEE);
                let cs = [
                    _mm256_permute2f128_ps(b0, b4, 0x20),
                    _mm256_permute2f128_ps(b1, b5, 0x20),
                    _mm256_permute2f128_ps(b2, b6, 0x20),
                    _mm256_permute2f128_ps(b3, b7, 0x20),
                    _mm256_permute2f128_ps(b0, b4, 0x31),
                    _mm256_permute2f128_ps(b1, b5, 0x31),
                    _mm256_permute2f128_ps(b2, b6, 0x31),
                    _mm256_permute2f128_ps(b3, b7, 0x31),
                ];
                for (c, v) in cs.iter().enumerate() {
                    _mm256_storeu_ps(out.add((i0 + c) * os + s0), *v);
                }
            } else {
                for r in 0..n_s {
                    for c in 0..n_i {
                        *out.add((i0 + c) * os + s0 + r) = *a.add((s0 + r) * in_stride + i0 + c);
                    }
                }
            }
            i0 += 8;
        }
        s0 += 8;
    }
}

/// 非 x86_64 回退。 # Safety 同 [`transpose_t8_back_avx2`]。
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn transpose_t8_back_avx2(
    a: *const f32,
    rows: usize,
    cols: usize,
    out: *mut f32,
    os: usize,
    in_stride: usize,
) {
    for s in 0..rows {
        for i in 0..cols {
            *out.add(i * os + s) = *a.add(s * in_stride + i);
        }
    }
}
// =====================================================================
// P19-7：**块对齐稀疏 head** 的前向 / 反向（无 gather、无标量散写、无转置）
// =====================================================================
// 事实（trainer 的 io_sparse 初始化路径）：块分配是「贪心取使用次数最少的块、
// 并列取最小下标」，在 n_vocab > n_blocks 时结果就是 `block_of[v] = v % n_blocks`，
// 于是 head 第 v 行的 k0_head 个源是 **h[(v%nb)*k0_embed + 0 .. k0_head]** —— 一段连续内存。
// 实测（`:kbench --mode headallrm`，vocab=8198, dim=2048, k0e=k0h=16, t=128）：
//   前向：通用 gather 内核 2.57 ms/round → 块对齐 1.40 ms/round（1.84x）
//   反向：通用 fused 内核 8.39 ms/round → 块对齐 4.37 ms/round（1.92x）
// 本函数族**不做任何数据重排**：`dw` 仍是 [nv][k0h] 行主序（= head_dw 的布局）、
// `dh` 仍是 [dim] 块主序（= h 的自然布局），所以接进 trainer 不需要新缓冲、
// 不需要转置、也不改 checkpoint 布局。
//
// 前提（由调用方断言）：`nb` 是 2 的幂、`src[v][j] == (v%nb)*k0e + j`。

/// 块对齐 head 前向：`y[v] = Σ_{j<k0h} w[v][j]·h[(v%nb)*k0e + j]`。
///
/// # Safety
/// `w` 至少 `nv*k0h`；`h` 至少 `nb*k0e`；`y` 至少 `nv`；`nb` 为 2 的幂且 `k0h <= k0e`。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn head_block_fwd_avx2(
    w: *const f32,
    h: *const f32,
    y: *mut f32,
    nv: usize,
    nb: usize,
    k0e: usize,
    k0h: usize,
) {
    let mut b = 0usize;
    while b < nb {
        let hrow = h.add(b * k0e);
        let mut v = b;
        while v < nv {
            let wrow = w.add(v * k0h);
            let mut acc = _mm256_setzero_ps();
            let mut j = 0usize;
            while j + 8 <= k0h {
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(wrow.add(j)), _mm256_loadu_ps(hrow.add(j)), acc);
                j += 8;
            }
            let mut s = hsum256(acc);
            while j < k0h {
                s += *wrow.add(j) * *hrow.add(j);
                j += 1;
            }
            *y.add(v) = s;
            v += nb;
        }
        b += 1;
    }
}

/// 块对齐 head 反向：`dw[v][j] += dy[v]·h[b* k0e + j]`、`dh[b*k0e + j] += dy[v]·w[v][j]`。
///
/// # Safety
/// `w`/`dw` 各至少 `nv*k0h`；`h`/`dh` 各至少 `nb*k0e`；`dy` 至少 `nv`；
/// `nb` 为 2 的幂且 `k0h <= k0e`。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn head_block_bwd_avx2(
    w: *const f32,
    h: *const f32,
    dy: *const f32,
    dw: *mut f32,
    dh: *mut f32,
    nv: usize,
    nb: usize,
    k0e: usize,
    k0h: usize,
) {
    let mask = nb - 1;
    for v in 0..nv {
        let b = v & mask;
        let d = _mm256_set1_ps(*dy.add(v));
        let wrow = w.add(v * k0h);
        let hrow = h.add(b * k0e);
        let drow = dw.add(v * k0h);
        let dhrow = dh.add(b * k0e);
        let mut j = 0usize;
        while j + 8 <= k0h {
            let p = _mm256_fmadd_ps(d, _mm256_loadu_ps(hrow.add(j)), _mm256_loadu_ps(drow.add(j)));
            _mm256_storeu_ps(drow.add(j), p);
            let q = _mm256_fmadd_ps(d, _mm256_loadu_ps(wrow.add(j)), _mm256_loadu_ps(dhrow.add(j)));
            _mm256_storeu_ps(dhrow.add(j), q);
            j += 8;
        }
        while j < k0h {
            *drow.add(j) += *dy.add(v) * *hrow.add(j);
            *dhrow.add(j) += *dy.add(v) * *wrow.add(j);
            j += 1;
        }
    }
}

// ---------------------------------------------------------------------
// P19-8：块对齐 head 的**列主序（WT）前向** —— 8 行/ymm，无横向归约
// ---------------------------------------------------------------------
// 实测（`:kbench --mode headall`，vocab=8198, dim=2048, k0e=k0h=16, t=128）：
//   通用 gather 前向      2.555 ms/round   (0.152 ns/conn)
//   块对齐行主序前向      2.874 ms/round   (0.89x —— **更慢**：W 只能按行取，
//                                          每行都要一次横向归约，且 W 的访问是 8KB 跨步)
//   块对齐行主序(v 序)    2.699 ms/round   (0.95x)
//   块对齐 **WT 列主序**  1.174 ms/round   (2.18x)  ← 采用这一条
// 原因：把 W 预先转置成 `WT[j*nv + v]` 后，**8 个连续行（= 8 个相邻块）** 可以放进
// 一个 ymm 通道，按 j 累加即可，完全不需要横向归约；h 侧也只需一次 16×nb 的小转置
// （`head_pack_ht`，2 KB 级、L1 常驻）。
//
// 代价：`WT` 必须在权重更新后**重新打包**（每步一次，`nv*k0h` 次搬运，实测 <0.1% 步时）。

/// `WT[j*nv + v] = w[v*k0h + j]`（8×8 分块，尾部标量）。权重更新后调用。
///
/// # Safety `w` 至少 `nv*k0h`；`wt` 至少 `k0h*nv` 且可写。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn head_pack_wt_avx2(w: *const f32, wt: *mut f32, nv: usize, k0h: usize) {
    let mut v0 = 0usize;
    while v0 + 8 <= nv {
        let mut j0 = 0usize;
        while j0 + 8 <= k0h {
            let mut tmp = [0.0f32; 64];
            for r in 0..8 {
                for c in 0..8 {
                    tmp[r * 8 + c] = *w.add((v0 + r) * k0h + j0 + c);
                }
            }
            for c in 0..8 {
                for r in 0..8 {
                    *wt.add((j0 + c) * nv + v0 + r) = tmp[r * 8 + c];
                }
            }
            j0 += 8;
        }
        while j0 < k0h {
            for r in 0..8 {
                *wt.add(j0 * nv + v0 + r) = *w.add((v0 + r) * k0h + j0);
            }
            j0 += 1;
        }
        v0 += 8;
    }
    while v0 < nv {
        for j in 0..k0h {
            *wt.add(j * nv + v0) = *w.add(v0 * k0h + j);
        }
        v0 += 1;
    }
}

/// `ht[j*nb + b] = h[b*k0e + j]`（j < k0h）—— 单个 token 的 h 转置，2 KB 级。
///
/// # Safety `h` 至少 `nb*k0e`；`ht` 至少 `k0h*nb` 且可写。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn head_pack_ht_avx2(h: *const f32, ht: *mut f32, nb: usize, k0e: usize, k0h: usize) {
    for j in 0..k0h {
        for b in 0..nb {
            *ht.add(j * nb + b) = *h.add(b * k0e + j);
        }
    }
}

/// 块对齐 head 前向（WT 版）：`y[v0..v0+8] = Σ_j WT[j][v0..v0+8] · ht[j*nb + (v0%nb) ..+8]`。
///
/// # Safety `wt` 至少 `k0h*nv`；`ht` 至少 `k0h*nb`；`y` 至少 `nv`；`nb` 为 2 的幂。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn head_block_fwd_wt_avx2(
    wt: *const f32,
    ht: *const f32,
    y: *mut f32,
    nv: usize,
    nb: usize,
    k0h: usize,
) {
    // 行按 g*nb + s 分段：同一 s 段的 8 行共享 s..s+8 号块
    let mut g = 0usize;
    while g * nb + nb <= nv {
        let mut s = 0usize;
        while s + 8 <= nb {
            let v0 = g * nb + s;
            let mut acc = _mm256_setzero_ps();
            let mut j = 0usize;
            while j < k0h {
                let hv = _mm256_loadu_ps(ht.add(j * nb + s));
                acc = _mm256_fmadd_ps(hv, _mm256_loadu_ps(wt.add(j * nv + v0)), acc);
                j += 1;
            }
            _mm256_storeu_ps(y.add(v0), acc);
            s += 8;
        }
        while s < nb {
            let v = g * nb + s;
            let mut a = 0.0f32;
            for j in 0..k0h {
                a += *wt.add(j * nv + v) * *ht.add(j * nb + s);
            }
            *y.add(v) = a;
            s += 1;
        }
        g += 1;
    }
    let mut v = g * nb;
    while v < nv {
        let b = v % nb;
        let mut a = 0.0f32;
        for j in 0..k0h {
            a += *wt.add(j * nv + v) * *ht.add(j * nb + b);
        }
        *y.add(v) = a;
        v += 1;
    }
}
