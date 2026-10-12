//! [抽库 L3-1] 通用原语（从 `train/src/b1_full_train.rs` 逐字节搬运）。
//!
//! 搬运范围（原文件行号）：`zero`(82) / `tr`(203) / `SendPtr`(1042)。
//! 逐字外改动：`unsafe fn` → `pub unsafe fn`（2 处）、
//! `struct SendPtr<T>(*mut T)` → `pub struct SendPtr<T>(pub *mut T)`
//! （跨 crate/跨模块必需，与 L2a 的 `SharedFn` 同一种可见性变化）。
#![allow(clippy::missing_safety_doc)]

pub unsafe fn zero(p: *mut f32, n: usize) {
    std::ptr::write_bytes(p as *mut u8, 0, n * 4);
}/// `dst[j*m + i] = src[i*n + j]`
pub unsafe fn tr(src: *const f32, dst: *mut f32, m: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            *dst.add(j * m + i) = *src.add(i * n + j);
        }
    }
}pub struct SendPtr<T>(pub *mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}
impl<T> Clone for SendPtr<T> {
    fn clone(&self) -> Self {
        SendPtr(self.0)
    }
}
impl<T> Copy for SendPtr<T> {}
