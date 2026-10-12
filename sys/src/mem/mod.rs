//! L0 内存：堆分配/释放（`conn_sparse` 的 CSR 存储、各内核 scratch 都用它）。
//!
//! 从 `core/src/lib.rs` **逐字节搬运**（`alloc` / `dealloc`）；唯一新增的是文件头
//! `use crate::io::exit;` —— 原文件里 `exit` 在同模块内可见，跨 crate 后需显式引入。

use crate::io::exit;

extern "system" {
    fn GetProcessHeap() -> *mut u8;
    fn HeapAlloc(h: *mut u8, flags: u32, n: usize) -> *mut u8;
    fn HeapFree(h: *mut u8, flags: u32, p: *mut u8) -> i32;
}

// 堆分配：引擎只在装载时分配几块
pub unsafe fn alloc(n: usize) -> *mut u8 {
    let p = HeapAlloc(GetProcessHeap(), 0, n);
    if p.is_null() {
        exit(1)
    }
    p
}

/// 释放 `alloc` 返回的指针。
///
/// ⚠️ 这里曾经是**空实现** `dealloc(_p, _n) {}`，注释写着"引擎只在装载时分配几块"——
/// 那个前提在 `gemm_nt_packed` 出现之后就不成立了：它在**每次调用**里
/// `alloc(nb*4+64)` 打包 B 矩阵，而 pack 缓冲的大小是
/// `ceil(n/8)*ceil(k/8)*64` 个 f32 —— 对 lm_head 的形状 (n=词表, k=dim) 来说
/// 就是**和权重同量级**的一块（dim=1536、词表=10921 时 **67 MB/次**）。
/// 空实现 ⇒ 每次 gemm 泄漏 67 MB：
///   · 实测 dim=1536 训练：工作集 3.8 GB(11s) → 10.5 GB(31s)，约 190 MB/步；
///   · 随后 `HeapAlloc` 返回 null，`alloc` 里的 `exit(1)` 让进程**无任何输出地退出**，
///     表现为"训练到第 150~200 步静默死掉、退出码 1"；
///   · 在此之前它还表现为**越来越慢**（堆压到换页）：0.4 s/步 → 2.6 s/步。
/// 全部症状同一个根因。修法是真正调用 `HeapFree`。
pub unsafe fn dealloc(p: *mut u8, _n: usize) {
    if !p.is_null() {
        HeapFree(GetProcessHeap(), 0, p);
    }
}


extern "system" {
    fn VirtualAlloc(a: *mut u8, n: usize, t: u32, p: u32) -> *mut u8;
    fn VirtualProtect(a: *mut u8, n: usize, new: u32, old: *mut u32) -> i32;
}

pub unsafe fn alloc_rwx(n: usize) -> *mut u8 {
    // 先申请可读写（PAGE_READWRITE = 0x04），码写完后由 make_exec 切成可执行。
    // 注意：不能在这里就 VirtualProtect(0x20=EXECUTE_READ)，那样会丢掉写权限，
    // 后续写码立即触发访问违例。
    let p = VirtualAlloc(core::ptr::null_mut(), n, 0x3000, 0x04);
    if p.is_null() {
        crate::print(b"VirtualAlloc failed\n");
        crate::exit(9)
    }
    p
}

// 写码完成后切到 PAGE_EXECUTE_READ
pub unsafe fn make_exec(p: *mut u8, n: usize) {
    let mut old = 0u32;
    VirtualProtect(p, n, 0x20, &mut old);
}

// JIT 生成代码时用到的内核地址（由 Rust 侧提供，JIT 只 call）
pub fn kernel_addr(f: usize) -> u64 {
    f as u64
}
