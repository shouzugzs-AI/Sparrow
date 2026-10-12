// P10 任务 C：**指令流内嵌 + 权重 mmap** 的 AOT 入口。
//
// 设计（为什么是这两个文件）
// ------------------------
//   <exe 内嵌>  代码文件：Header + TEXT + META + **零填充的 CONST 占位**
//   <外部文件>  权重文件：与代码文件 **同一汇编产物** 的 CONST 区字节
//
// 启动时：mmap(只读) 权重文件 -> `Vm::load_weights_mapped(code, weights)`。
// 因为 cpool 取的是 `mmap + const_off`，而 META 里的常量偏移都相对 CONST
// 起点，所以两边**天然对齐**，不需要第二份布局定义。
//
// 为什么这比「整个 .vmb 读进内存」好
// --------------------------------
//   read_file 路径：2.16 GB 私有提交（外加常量池被就地训练时的可写副本）。
//   mmap 路径：页由系统按需从文件回填，**文件共享、可丢弃、不占页面文件**，
//              进程私有占用接近 0。这是内存行为的改变。
//
//   ⚠️ 它**不减小文件体积**。真正能压体积的是 bf16 -> int8/int4，
//      那需要引擎有反量化 GEMM 内核（目前没有）。见 tools/p10_quant_eval.py。
//
// 为什么只读映射是安全的（而不是「暂时省事」）
// ------------------------------------------
//   mmap 用 PAGE_READONLY。训练指令（ADAM/ADAMW/SGD）会就地写常量池；
//   引擎在 `cpool_readonly` 时对这三条指令**显式 exit(9)**，
//   而不是让它变成 0xC0000005（CI 里只显示 exit=3221225477，看不出原因）。
//
// 无 CRT：只用 kernel32 绑定（core/src/lib.rs 里的 mmap_file_ro）。

use vmlm_core::interp::Vm;
use vmlm_core::{mmap_file_ro, read_file};

// tools/aot_build.py 生成的「代码文件」常量数组（含零填充 CONST 占位）
include!("model_aot.rs");

// 权重文件名（相对 CWD）。放在 .rdata 里的 NUL 结尾串，避免 core::fmt。
static WEIGHTS_PATH: &[u8] = b"weights.vmbw\0";

pub unsafe fn run_mapped() -> u32 {
    let code = AOT_MODEL.bytes.as_ptr();
    let len = AOT_MODEL.bytes.len();

    // 代码文件可能只有 1 字节对齐（include_bytes! 的保证），而 `Ins` 要求
    // 4 字节、常量池基址要求 64 字节。所以仍然拷进对齐的可写堆 —— 但它很
    // 小（KB 级），与「权重 2 GB」完全不是一个量级。
    let aligned = vmlm_core::aot_mirror(code, len);

    let (w, wlen) = mmap_file_ro(WEIGHTS_PATH.as_ptr());
    // 尺寸一致性**必须**硬校验：权重文件与代码文件若不是同一次汇编的产物，
    // 偏移会错位，而错位读出来的是「看似合理的垃圾数」——
    // 正是最难查的一类静默降级。
    let want = vmlm_core::vm_const_len(aligned);
    if wlen < want {
        vmlm_core::print(b"weights file too small\n");
        vmlm_core::exit(9);
    }

    let mut vm = Vm::load_weights_mapped(aligned, w);
    vm.run()
}

// 备用：没有权重文件时，退回到「全部内嵌」（旧行为），便于对照。
pub unsafe fn run_embedded() -> u32 {
    let code = AOT_MODEL.bytes.as_ptr();
    let len = AOT_MODEL.bytes.len();
    let buf = read_file(b"model.vmb\0".as_ptr());
    let _ = (code, len);
    let mut vm = Vm::load(buf.0);
    vm.run()
}
