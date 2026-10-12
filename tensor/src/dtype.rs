//! L1 dtype：bf16 ⇄ f32 转换与 bf16 加宽（从 core/src/kernels.rs 逐字节搬运）。
//! 唯一新增：文件头 `use core::arch::x86_64::*;`（原文件在 crate 级已引入）。

use core::arch::x86_64::*;

/// bf16 权重 × f32 激活，NT 布局（B 为 [N,K]）。
/// 每次内层迭代处理 16 个 K 元素（一次 ymm load）。
///
/// P13：改为 `pub` —— `OP_EMB` 的 bf16 表加宽也用这一份实现。
/// 复用同一处 lane 布局结论，避免第二处手推 shuffle 语义
/// （P12 在 `unpacklo`/`permute4x64` 上错过两次，见 tests/src/shufprobe.rs）。
#[inline(always)]
pub unsafe fn bf16_widen_16(p: *const u16) -> (core::arch::x86_64::__m256, core::arch::x86_64::__m256) {
    use core::arch::x86_64::*;
    let packed = _mm256_loadu_si256(p as *const __m256i);
    let lo128 = _mm256_castsi256_si128(packed);
    let hi128 = _mm256_extracti128_si256(packed, 1);
    let l0 = _mm256_cvtepu16_epi32(lo128);
    let l1 = _mm256_cvtepu16_epi32(hi128);
    (
        _mm256_castsi256_ps(_mm256_slli_epi32(l0, 16)),
        _mm256_castsi256_ps(_mm256_slli_epi32(l1, 16)),
    )
}

#[inline(always)]
pub unsafe fn hsum_ps(v: core::arch::x86_64::__m256) -> f32 {
    use core::arch::x86_64::*;
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let s4 = _mm_add_ps(hi, lo);
    let sh = _mm_movehl_ps(s4, s4);
    let s2 = _mm_add_ps(s4, sh);
    let s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
    _mm_cvtss_f32(s1)
}

/// f32 → bf16（就近舍入到偶数，取高 16 位）。
#[inline]
pub fn f32_to_bf16(v: f32) -> u16 {
    let u = v.to_bits();
    let lsb = (u >> 16) & 1;
    let round = 0x7fffu32 + lsb;
    ((u.wrapping_add(round)) >> 16) as u16
}

/// bf16 → f32（左移 16 位）。
#[inline]
pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// 就地 bf16 舍入（预扩展版：把 f32 权重按 bf16 精度圆整，之后所有 f32 内核直接用）。
pub fn round_weights_bf16(w: &mut [f32]) {
    for v in w.iter_mut() {
        *v = bf16_to_f32(f32_to_bf16(*v));
    }
}
