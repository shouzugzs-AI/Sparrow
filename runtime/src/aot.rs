// `--features aot` 的运行时支持：把烧进 .exe 的模型字节变成可用的 Vm。
//
// 背景（为什么不能直接把 static 交给 Vm::load）：
//   1. `static` 落在 .rdata —— 只读页。`Vm::load` 把 `const_off` 处当
//      **可写**常量池（train/Adam 就地更新参数），写只读页 = 访问违例。
//   2. `Vm::load` 从 `buf.add(text_off)` 取 `*const Ins`，`Ins` 是
//      `#[repr(C)] { u8, u8, u16, u32, u32, u32 }`，要求 **4 字节**对齐；
//      常量池按 `CONST_ALIGN = 64` 对齐。`include_bytes!` 只保证 1 字节
//      对齐，直接喂进去就是对齐 UB。
//
// 解法：启动时把模型字节整块拷进 `vmlm_core::alloc`（HeapAlloc）拿到的
// 可写内存，并把基址向上取整到 64 字节。
//
// 布局：一次分配 `len + 2*ALIGN`，在里头切出
//   [ raw .. raw+len )                      ← HeapAlloc 原始返回
//   [ base = align_up(raw, 64) .. +len )    ← 交出去的可写镜像
//   [ base+len .. raw+len+ALIGN )           ← 尾部余量，保证越界读安全
//
// 为什么留 ALIGN 而不是 ALIGN-1 的余量：
//   `read_file` 走的是 `alloc(n + CONST_ALIGN)` 的路径，运行时基线同样有余量。
//   让 AOT 路径的堆占用与读文件路径**逐字节一致**，才能保证「内嵌 vs 读文件」
//   是纯粹的可执行文件差异，而不是内存布局差异。多 64 字节堆开销可忽略。
//
// 为什么不写 build.rs：
//   `runtime/.cargo/config.toml` 给整个 crate 挂了 `/ENTRY:mainCRTStartup`
//   `/NODEFAULTLIB`，这些 link-arg 同样会作用到 build script（宿主 std 程序），
//   直接 LNK2001（找不到 mainCRTStartup / memcpy / _fltused）。
//   所以模型常量由 `tools/aot_build.py` 离线生成到 `src/model_aot.rs`，
//   这里用 `include!` 内联 —— 零构建期依赖。
//
// 生成命令（换模型只需重跑再 cargo build）：
//   python tools/aot_build.py examples/linear_relu.vmb runtime/src/model_aot.rs
//   cd runtime; cargo build --release --features aot
//
// 无 CRT 约束：这里只用 `vmlm_core::alloc`（内部 HeapAlloc），不碰 libc。

use vmlm_core::interp::Vm;

pub const AOT_ALIGN: usize = vmlm_core::CONST_ALIGN;

// tools/aot_build.py 生成的 align(64) 常量数组
include!("model_aot.rs");

/// 用内嵌模型字节构造 Vm。
///
/// # Safety
/// 只应在启动时调用一次；返回的 `Vm` 持有堆内存指针，调用方负责其生命周期。
pub unsafe fn load_embedded() -> Vm {
    let src = AOT_MODEL.bytes.as_ptr();
    let len = AOT_MODEL.bytes.len();
    Vm::load(copy_to_aligned_writable(src, len))
}

/// 把只读的模型字节拷进 64 字节对齐的可写堆块，返回镜像基址。
///
/// 返回的指针满足：
///   * 可写（HeapAlloc 出来的堆内存）
///   * `% AOT_ALIGN == 0`（64B，满足常量池对齐）
///   * 后随至少 `AOT_ALIGN` 字节余量（满足 `Vm::load` 的越界读假设）
///
/// # Safety
/// `src` 必须指向至少 `len` 个可读字节。
pub unsafe fn copy_to_aligned_writable(src: *const u8, len: usize) -> *mut u8 {
    // 多要 2*ALIGN：一份给对齐取整，一份给尾部余量。
    let raw = vmlm_core::alloc(len + 2 * AOT_ALIGN);
    let base = align_up(raw as usize, AOT_ALIGN) as *mut u8;

    // 只读段 -> 可写堆
    core::ptr::copy_nonoverlapping(src, base, len);

    // 尾部余量清零，让任何越界读都落在确定值上（便于逐位复现）
    core::ptr::write_bytes(base.add(len), 0, AOT_ALIGN);

    base
}

#[inline]
const fn align_up(v: usize, a: usize) -> usize {
    (v + (a - 1)) & !(a - 1)
}
