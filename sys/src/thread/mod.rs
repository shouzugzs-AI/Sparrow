//! L0 线程原语：pin 原语 + 常驻线程池（`thread-std`）。
//!
//! 边界（用户契约）：L0 **只**提供 `pin_current_thread(cpu)` 这类 OS 原语；
//! "绑到哪几个核"的策略（`t * npin / nt`、`(t + 1) % npin`）与 region 切分/归约留在 L2a。

/// 亲和性 extern（从 trainer 逐字节搬运）。
#[link(name = "kernel32")]
extern "system" {
    fn GetCurrentThread() -> *mut u8;
    fn GetCurrentProcess() -> *mut u8;
    fn SetThreadAffinityMask(hThread: *mut u8, dwThreadAffinityMask: usize) -> usize;
    fn GetProcessAffinityMask(hProcess: *mut u8, lpProcessAffinityMask: *mut usize, lpSystemAffinityMask: *mut usize) -> i32;
}

pub fn pin_current_thread(cpu: usize) {
    unsafe {
        let mask: usize = 1usize << (cpu % usize::BITS as usize);
        SetThreadAffinityMask(GetCurrentThread(), mask);
    }
}

#[cfg(feature = "thread-std")]
pub mod pool;
#[cfg(feature = "thread-std")]
pub use pool::*;
