//! L0 CPU 探测：逻辑核数（GetProcessAffinityMask）与 AVX2 探测（CPUID）。
//! `logical_cpu_count` 来自 trainer、`cpu_has_avx2` 来自 `core/src/interp.rs`，均逐字节搬运。

#[link(name = "kernel32")]
extern "system" {
    fn GetCurrentProcess() -> *mut u8;
    fn GetProcessAffinityMask(hProcess: *mut u8, lpProcessAffinityMask: *mut usize, lpSystemAffinityMask: *mut usize) -> i32;
}

pub fn logical_cpu_count() -> usize {
    unsafe {
        let mut pm: usize = 0;
        let mut sm: usize = 0;
        if GetProcessAffinityMask(GetCurrentProcess(), &mut pm, &mut sm) != 0 && pm != 0 {
            return pm.count_ones() as usize;
        }
    }
    1
}

pub fn cpu_has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        return core::arch::x86_64::__cpuid(7).ebx & (1 << 5) != 0;
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// AVX2 可用位（运行时探测一次；从 core/src/interp.rs 逐字节搬运）。
pub static mut HAS_AVX2: bool = false;
