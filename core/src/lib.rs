// VMLM 引擎核心：常量池 + 槽位 + 解释器
// no_std，无泛型抽象，无防御性检查（允许 panic），单线程
#![no_std]
#![allow(clippy::missing_safety_doc)]

pub use vmlm_sys::mem::{alloc, dealloc};

pub use vmlm_sys::io::exit;

pub mod format;
pub mod kernels;
pub mod interp;
pub mod jit;
#[cfg(feature = "train")]
pub mod train;
pub mod autograd;
pub mod sparse;
// S5 neuron-organized KV-Cache. DELIBERATELY behind a feature:
// it is load-time/verification-only code (the reorder runs once; the addressing
// helper is never called in a steady-state forward), so linking it into the
// inference engine unconditionally would make "engine core size" measure test
// scaffolding instead of the engine. Measured: contributes 0 B when OFF.
// Same pattern as the existing `train` feature. DO NOT un-gate this line --
// an earlier teammate size-probe left it unconditional and silently added
// bytes to the engine budget.
#[cfg(feature = "neuron-kv")]
pub mod neuron_kv;
// P16-1 connection-sparse STORAGE (core/src/conn_sparse.rs): CSR adjacency
// list (conn_offsets/conn_src/conn_weights) + reader/writer + hard-fail guards.
// Same gating rule as neuron-kv: load/build-time code, so the DEFAULT
// inference engine must pay 0 B for it. Measured with isolated target dirs.
#[cfg(feature = "conn-sparse")]
pub mod conn_sparse;
// P17 TRAINING SIDE of the sparse-connection architecture.
//
// Gating rule: this is TRAINING code (it mutates weights in place), and the
// engine's size budget measures the INFERENCE build. So it sits behind `train`,
// which is already the established boundary for the backward path -- NOT behind
// `conn-sparse`, because inference must not link it even when the conn-sparse
// STORAGE format is enabled.
//
// It needs conn_sparse for the shared deterministic PRNG (the task explicitly
// forbids a second random source), so the two features compose rather than
// duplicate. `#[cfg(all(...))]` states that dependency instead of hiding it.
#[cfg(all(feature = "train", feature = "conn-sparse"))]
pub mod conn_train;
// B1 = pure-sparse storage + FULL-k0 compute (no top-k, no mask, no selection).
//
// SAME GATE as conn_train, and for the same reason: it consumes
// `conn_train::SparseConnW` as its storage, is TRAINING-side code, and must
// therefore cost the default inference build 0 B. Gating it differently from
// conn_train would be a bug in either direction -- looser and inference links
// training code; tighter and the module becomes unreachable from the driver.
//
// The brief asked for this implementation under `sparse_b/b1_pure/`. Rust needs
// a module the crate can actually compile, so the implementation lives in
// `core/src/b1_pure.rs` (this line) and `sparse_b/b1_pure/README.md` records the
// directory<->module mapping. Compilability beats directory literal.
#[cfg(all(feature = "train", feature = "conn-sparse"))]
pub mod b1_pure;
// B2 = sparse storage + SPARSE ACTIVATION: walk the k0 connections, compute a
// strength per connection, take top-k, accumulate only those k.
//
// SAME GATE again, and for a third reason on top of the two above: B2 consumes
// `conn_train::SparseConnW` as its storage AND its top-k must agree bit-for-bit
// with `b1_pure` at k = k0 (that agreement is the A1 criterion). Gating it
// differently from b1_pure would let the two drift apart in exactly the build
// configuration where nothing checks them.
//
// It carries BOTH readings of "take top-k" (per-neuron vs one-per-layer) in one
// module, behind `B2Mode`, because the measured difference between them is a
// factor of d and the user's decision needs both numbers measured on the same
// data by the same code. See the long header in b2_topk.rs.
#[cfg(all(feature = "train", feature = "conn-sparse"))]
pub mod b2_topk;
// P17-P3 BPE tokenizer (core/src/bpe.rs): external mmap'd vocabulary + a
// hand-written pure-Rust encoder/decoder + hard-fail guards.
//
// Gating rule: same boundary as neuron-kv / conn-sparse -- the vocabulary is
// LOADED, EXPANDED and VALIDATED once, and the steady-state path is an
// indexed byte copy. Linking it into the default inference engine
// unconditionally would make "engine core size" measure vocabulary
// scaffolding instead of the engine, so it must cost the default build 0 B
// (measured with isolated target dirs; see tools/p17_bpe_size.py).
//
// It is ORTHOGONAL to neuron-kv and conn-sparse: bpe.rs depends on nothing
// but lib.rs itself (print/exit/mmap_file_ro), so `--features bpe` alone must
// compile -- verified by building it in an isolated crate with
// default-features = false.
#[cfg(feature = "bpe")]
pub mod bpe;

pub use format::*;
pub use interp::{Ins, Vm};

// 直接绑 kernel32 —— 完全绕开 CRT（白皮书 §7.4）。
// 这是 30KB 目标的关键：静态 CRT 一家就 100KB+。
#[link(name = "kernel32")]
extern "system" {
    fn GetProcessHeap() -> *mut u8;
    fn HeapAlloc(h: *mut u8, flags: u32, n: usize) -> *mut u8;
    fn HeapFree(h: *mut u8, flags: u32, p: *mut u8) -> i32;
    fn GetStdHandle(which: u32) -> isize;
    fn GetCommandLineA() -> *const u8;
    fn WriteFile(
        h: isize,
        buf: *const u8,
        n: u32,
        written: *mut u32,
        ov: *mut u8,
    ) -> i32;
    fn CreateFileA(
        name: *const u8,
        access: u32,
        share: u32,
        sec: *mut u8,
        disp: u32,
        flags: u32,
        templ: *mut u8,
    ) -> isize;
    fn ReadFile(h: isize, buf: *mut u8, n: u32, read: *mut u32, ov: *mut u8) -> i32;
    fn GetFileSizeEx(h: isize, out: *mut i64) -> i32;
    fn CloseHandle(h: isize) -> i32;
    fn ExitProcess(code: u32) -> !;
    fn VirtualAlloc(a: *mut u8, n: usize, t: u32, p: u32) -> *mut u8;
    fn VirtualProtect(a: *mut u8, n: usize, new: u32, old: *mut u32) -> i32;
    // ---- P10: 内存映射（把权重文件映射进地址空间，而不是读进私有内存）----
    // 为什么手写这两个绑定而不是用 std：
    //   引擎是 no_std + /NODEFAULTLIB，没有任何 std::fs / memmap 可用。
    //   与既有 kernel32 绑定（CreateFileA / ReadFile）保持同一风格。
    fn CreateFileMappingA(
        file: isize,
        sec: *mut u8,
        protect: u32,
        max_hi: u32,
        max_lo: u32,
        name: *const u8,
    ) -> isize;
    fn MapViewOfFile(
        mapping: isize,
        access: u32,
        off_hi: u32,
        off_lo: u32,
        n: usize,
    ) -> *mut u8;
    fn UnmapViewOfFile(addr: *const u8) -> i32;
}



// ---- 编译器内建符号：自己提供，避免拉入整个 CRT ----
// __chkstk：栈探测。我们的栈用量远小于一页，直接返回即可。
#[no_mangle]
pub unsafe extern "C" fn __chkstk() {}

#[no_mangle]
pub unsafe extern "C" fn memcpy(dst: *mut u8, src: *const u8, n: usize) -> *mut u8 {
    let mut i = 0;
    while i + 8 <= n {
        core::ptr::write_unaligned(dst.add(i) as *mut u64, core::ptr::read_unaligned(src.add(i) as *const u64));
        i += 8;
    }
    while i < n {
        *dst.add(i) = *src.add(i);
        i += 1;
    }
    dst
}

#[no_mangle]
pub unsafe extern "C" fn memset(dst: *mut u8, v: i32, n: usize) -> *mut u8 {
    let b = v as u8;
    let mut i = 0;
    while i < n {
        *dst.add(i) = b;
        i += 1;
    }
    dst
}

#[no_mangle]
pub unsafe extern "C" fn memmove(dst: *mut u8, src: *const u8, n: usize) -> *mut u8 {
    if (dst as usize) < (src as usize) {
        return memcpy(dst, src, n);
    }
    let mut i = n;
    while i > 0 {
        i -= 1;
        *dst.add(i) = *src.add(i);
    }
    dst
}





// 写文件（基准校验用）
pub unsafe fn dump_f32(path: *const u8, p: *const f32, n: usize) {
    let h = CreateFileA(
        path,
        0x4000_0000, // GENERIC_WRITE
        0,
        core::ptr::null_mut(),
        2, // CREATE_ALWAYS
        0x80,
        core::ptr::null_mut(),
    );
    if h == -1 {
        return;
    }
    let mut w = 0u32;
    WriteFile(
        h,
        p as *const u8,
        (n * 4) as u32,
        &mut w,
        core::ptr::null_mut(),
    );
    CloseHandle(h);
}

// 取命令行第 n 个参数（n=0 是程序名），写入 dst（NUL 结尾）。
// 无 CRT 时没有 argv，用 GetCommandLineA 自己切。
pub unsafe fn arg(n: usize, dst: *mut u8) {
    let cl = GetCommandLineA();
    let mut i = 0usize;
    let mut idx = 0usize;
    loop {
        while *cl.add(i) == b' ' || *cl.add(i) == b'\t' {
            i += 1;
        }
        if *cl.add(i) == 0 {
            *dst = 0;
            return;
        }
        let start = i;
        if *cl.add(i) == b'"' {
            i += 1;
            let s = i;
            while *cl.add(i) != b'"' && *cl.add(i) != 0 {
                i += 1;
            }
            if idx == n {
                let mut k = 0;
                while s + k < i {
                    *dst.add(k) = *cl.add(s + k);
                    k += 1;
                }
                *dst.add(k) = 0;
                return;
            }
            if *cl.add(i) == b'"' {
                i += 1;
            }
        } else {
            while *cl.add(i) != b' ' && *cl.add(i) != b'\t' && *cl.add(i) != 0 {
                i += 1;
            }
            if idx == n {
                let mut k = 0;
                while start + k < i {
                    *dst.add(k) = *cl.add(start + k);
                    k += 1;
                }
                *dst.add(k) = 0;
                return;
            }
        }
        idx += 1;
    }
}

pub unsafe fn print(s: &[u8]) {
    let h = GetStdHandle(0xFFFF_FFF5); // STD_OUTPUT_HANDLE = (DWORD)-11
    let mut w = 0u32;
    WriteFile(h, s.as_ptr(), s.len() as u32, &mut w, core::ptr::null_mut());
}

// 读文件：CreateFileA + GetFileSizeEx + ReadFile
pub unsafe fn read_file(path: *const u8) -> (*mut u8, usize) {
    let h = CreateFileA(
        path,
        0x8000_0000, // GENERIC_READ
        1,           // FILE_SHARE_READ
        core::ptr::null_mut(),
        3,   // OPEN_EXISTING
        0x80, // FILE_ATTRIBUTE_NORMAL
        core::ptr::null_mut(),
    );
    if h == -1 {
        print(b"cannot open file\n");
        exit(2)
    }
    let mut sz: i64 = 0;
    GetFileSizeEx(h, &mut sz);
    let n = sz as usize;
    let buf = alloc(n + CONST_ALIGN);
    let mut got = 0u32;
    ReadFile(h, buf, n as u32, &mut got, core::ptr::null_mut());
    CloseHandle(h);
    (buf, got as usize)
}

// ---- P10: 只读内存映射 ----
//
// 为什么需要它（任务 C 的核心）：
//   `read_file` 把整个文件**读进私有堆内存**。对 MiniCPM 的 2.16 GB 权重，
//   那意味着进程私有提交 2.16 GB（外加 Vm::load 再分配的激活/梯度区）。
//   mmap 把文件**映射进地址空间**：页由系统按需从文件回填，
//   可共享、可丢弃、不占页面文件 —— 进程「私有」占用接近 0。
//
// 为什么只做只读（PAGE_READONLY / FILE_MAP_READ）：
//   这是关键的安全性质，不是省事。映射区是**只读**的，所以
//   `OP_ADAM` / `OP_SGD` 之类「就地更新参数」的指令一旦作用到 mmap 区域
//   会立刻**访问违例崩溃**，而不是静默写坏一个共享的映射页
//   （多进程场景下那是真正的静默数据损坏）。
//   ⇒ 推理路径用 mmap；**训练路径必须继续用 read_file**。
//     这条边界由调用方（runtime/src/mmap_weights.rs）显式选择。
//
// 对齐：MapViewOfFile 的返回地址总是 64 KB 对齐（分配粒度），
// 远优于常量池要求的 CONST_ALIGN = 64，所以不需要再拷一次。
//
// 返回 (base, len)。文件不存在 / 映射失败一律**硬失败**（不是返回 null
// 让调用方自己去猜）—— 这是本项目「输入不合法必须报错」的一贯做法。
pub unsafe fn mmap_file_ro(path: *const u8) -> (*const u8, usize) {
    let h = CreateFileA(
        path,
        0x8000_0000, // GENERIC_READ
        1,           // FILE_SHARE_READ
        core::ptr::null_mut(),
        3,   // OPEN_EXISTING
        0x80, // FILE_ATTRIBUTE_NORMAL
        core::ptr::null_mut(),
    );
    if h == -1 {
        print(b"mmap: cannot open\n");
        exit(2)
    }
    let mut sz: i64 = 0;
    if GetFileSizeEx(h, &mut sz) == 0 || sz <= 0 {
        print(b"mmap: size failed\n");
        exit(2)
    }
    let n = sz as usize;

    // PAGE_READONLY = 0x02；max size 传 0 = 用文件当前大小
    let m = CreateFileMappingA(h, core::ptr::null_mut(), 0x02, 0, 0, core::ptr::null());
    if m == 0 {
        print(b"mmap: CreateFileMapping failed\n");
        exit(2)
    }
    // FILE_MAP_READ = 0x0004
    let base = MapViewOfFile(m, 0x0004, 0, 0, 0);
    // 映射句柄可以立刻关：view 持有自己的引用，文件句柄也已不需要
    CloseHandle(m);
    CloseHandle(h);
    if base.is_null() {
        print(b"mmap: MapViewOfFile failed\n");
        exit(2)
    }
    (base as *const u8, n)
}

/// 解除映射（仅用于「同进程内换模型」这类场景；进程退出时系统自动回收）。
pub unsafe fn munmap(base: *const u8) {
    UnmapViewOfFile(base);
}

/// P10 任务 C：把只读的内嵌模型字节镜像到 **64B 对齐的可写堆块**。
///
/// 为什么 AOT 路径必须拷贝（这在 P7 就踩过，见 runtime/src/aot.rs 的长注释）：
///   1. `.rdata` 是只读页，而 `Vm::load` 把常量池当可写区（训练就地更新）。
///   2. `include_bytes!` 只保证 **1 字节**对齐，而 `Ins` 要求 4 字节、
///      常量池要求 64 字节 —— 直接喂进去就是对齐 UB。
///
/// 与 aot.rs 里那份 `copy_to_aligned_writable` 的区别：这里放在 core，
/// 这样 mmap 入口（runtime）不必再依赖 aot feature 的模块。
/// 尾部多留 CONST_ALIGN 字节零余量，让越界读落在确定值上（便于逐位复现）。
pub unsafe fn aot_mirror(src: *const u8, len: usize) -> *mut u8 {
    let a = CONST_ALIGN;
    let raw = alloc(len + 2 * a);
    let base = ((raw as usize + (a - 1)) & !(a - 1)) as *mut u8;
    core::ptr::copy_nonoverlapping(src, base, len);
    core::ptr::write_bytes(base.add(len), 0, a);
    base
}

/// 读一个已装载到内存的 .vmb 头里的 **CONST 段长度**。
///
/// 用途：校验外置权重文件确实够长。若权重与代码不是同一次汇编的产物，
/// 偏移会整体错位，读出来是「看似合理的垃圾数」——
/// 这类静默降级只有尺寸校验能在装载时拦住。
/// 头布局见 format 文档：magic(0) ver(4) flags(6) text_off(8) text_len(12)
///                   const_off(16) const_len(20) meta_off(24) meta_len(28)
pub unsafe fn vm_const_len(buf: *const u8) -> usize {
    u32_at(buf, 20) as usize
}

/// 读 .vmb 头里的 CONST 段偏移（与权重文件内的偏移一致）。
pub unsafe fn vm_const_off(buf: *const u8) -> usize {
    u32_at(buf, 16) as usize
}

pub fn u32_at(p: *const u8, off: usize) -> u32 {
    unsafe { core::ptr::read_unaligned(p.add(off) as *const u32) }
}
pub fn u16_at(p: *const u8, off: usize) -> u16 {
    unsafe { core::ptr::read_unaligned(p.add(off) as *const u16) }
}
