//! x86 AVX2 后端（L0）。多后端并列：将来 `x86_avx512.rs` / `arm_neon.rs` / `cuda.rs` 同层。
pub mod simd;
