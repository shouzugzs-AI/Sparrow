//! `vmlm-sys` —— L0 底层调用层。
//!
//! 职责边界（用户契约）：
//! * **只**放 SIMD intrinsics、内存、线程原语、计时、CPU 探测；
//! * **不做** region 切分/调度（调度属于 L2a `vmlm-ops`）；
//! * 多后端并列：`x86_avx2` / `x86_avx512` / `arm_neon` / `cuda` 同目录，运行时 dispatch；
//! * 跨层接口用 **C ABI 或函数指针表**，不用 Rust trait。
// 引擎路径（未启用 thread-std）保持 no_std；trainer 启用 thread-std ⇒ 链接 std，
// 被搬来的线程池可继续用 `std::` 路径与 `thread_local!`/`format!` 宏（函数体逐字节不变）。
#![cfg_attr(not(feature = "thread-std"), no_std)]
#![allow(clippy::missing_safety_doc)]

pub mod io;
pub mod cpu;
pub mod math;
pub mod thread;
pub mod mem;
pub mod x86_avx2;

pub use io::*;
pub use cpu::HAS_AVX2;
pub use mem::{alloc, alloc_rwx, dealloc, kernel_addr, make_exec};

/// 多后端函数指针表（C ABI 兼容的 `extern "C"` 签名）。
///
/// 这是给「将来换语言 / 加 GPU 后端」留的**唯一**跨层接口形态：
/// 上层只认这张表的签名，不认实现语言。目前只填 x86_avx2 后端；
/// 表本身**不被热路径引用**，因此不影响任何既有二进制的体积。
#[derive(Clone, Copy)]
#[repr(C)]
pub struct Backend {
    pub name: *const u8,
    pub dot256: unsafe extern "C" fn(*const f32, *const f32, usize) -> f32,
    pub axpy256: unsafe extern "C" fn(*mut f32, *const f32, f32, usize),
    pub add256: unsafe extern "C" fn(*mut f32, *const f32, usize),
    pub max256: unsafe extern "C" fn(*const f32, usize) -> f32,
    pub sum256: unsafe extern "C" fn(*const f32, usize) -> f32,
    pub scale_mul: unsafe extern "C" fn(*mut f32, *const f32, f32, usize),
}

unsafe extern "C" fn dot256_c(a: *const f32, b: *const f32, n: usize) -> f32 {
    x86_avx2::simd::dot256_f32(a, b, n)
}
unsafe extern "C" fn axpy256_c(y: *mut f32, x: *const f32, a: f32, n: usize) {
    x86_avx2::simd::axpy256_f32(y, x, a, n)
}
unsafe extern "C" fn add256_c(y: *mut f32, x: *const f32, n: usize) {
    x86_avx2::simd::add256_f32(y, x, n)
}
unsafe extern "C" fn max256_c(x: *const f32, n: usize) -> f32 {
    x86_avx2::simd::max256_f32(x, n)
}
unsafe extern "C" fn sum256_c(x: *const f32, n: usize) -> f32 {
    x86_avx2::simd::sum256_f32(x, n)
}
unsafe extern "C" fn scale_mul_c(dst: *mut f32, src: *const f32, a: f32, n: usize) {
    x86_avx2::simd::scale_mul_avx2(dst, src, a, n)
}

/// 当前 CPU 的默认后端（目前恒为 AVX2 表）。
#[inline]
pub fn default_backend() -> Backend {
    Backend {
        name: b"x86_avx2\0".as_ptr(),
        dot256: dot256_c,
        axpy256: axpy256_c,
        add256: add256_c,
        max256: max256_c,
        sum256: sum256_c,
        scale_mul: scale_mul_c,
    }
}
