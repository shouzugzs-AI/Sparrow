// vmlm 运行时入口：no_std，无 CRT（白皮书 §0 / §7.4）
//
// 体积目标：引擎核心 ≤30KB（决策 5：只约束引擎本体）
//
// 三种构建：
//   --features bench  : 带 argv 解析 + 计时 + dump（基准用，体积不设限）
//   --features aot    : 把 .vmb 烧进 .exe，不读外部文件（体积 = 引擎 + 模型）
//   默认              : 极简入口，只跑 model.vmb（受 30KB 约束）
#![no_std]
#![no_main]

#[cfg(not(feature = "aot"))]
use vmlm_core::interp::Vm;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    // 不格式化输出 —— 省掉 core::fmt（数十 KB）
    unsafe { vmlm_core::exit(70) }
}

#[no_mangle]
pub static _fltused: i32 = 0;

#[cfg(feature = "bench")]
mod bench;

// 共享 argv 解析（no_std 无 CRT，没有 argv）。
// bench 与 aot 共用，避免两份实现漂移；默认推理构建不编译它。
#[cfg(any(feature = "bench", feature = "aot"))]
mod args;

// AOT：把 .vmb 烧进 .exe（见 aot.rs / tools/aot_build.py）
#[cfg(feature = "aot")]
mod aot;

// AOT 多模型：一个 exe 内嵌 N 个 .vmb，--model=<name> 选择
// （见 aot_multi.rs / tools/aot_multi_build.py）
#[cfg(feature = "aot-multi")]
mod aot_multi;

// AOT 训练闭环：数据集也烧进 exe，不读任何外部文件
// （见 aot_train.rs / tools/aot_embed.py）
#[cfg(feature = "aot-train")]
mod aot_train;

// P10 任务 C：指令流内嵌 + 权重从文件 mmap
// （见 mmap_aot.rs / tools/p10_split_vmb.py）
#[cfg(feature = "mmap-aot")]
mod mmap_aot;

// S5-W2：neuron-organized KV-Cache 的**装载期入口**。
//
// 为什么是一个独立 argv 开关而不是默认路径的一部分：
//   * 默认推理路径受 68096 B 预算约束，接进来就必须读 neuron_kv 文件，
//     而绝大多数既有 artifact 根本没有这个 section —— 它们会从「正常跑完」
//     变成「装载失败」。
//   * 单独走 argv 之后，`--features neuron-kv` 这个构建**本身**就是
//     「把 S5 链接进引擎要花多少字节」的可测量对象（见 tools/p15_s5_load_probe.py）。
//
// 只在 `--s5-load` 出现时才进入；不带参数时默认路径逐字节不变。
#[cfg(feature = "neuron-kv")]
mod s5_load;

#[no_mangle]
pub extern "system" fn mainCRTStartup() -> u32 {
    unsafe {
        vmlm_core::interp::init_cpu();
        // S5-W2 装载期入口：只在 `--features neuron-kv` 且 argv[1] 恰为
        // `--s5-load` 时进入。放在最前面（在其它 feature 分支之前），
        // 这样它是一条**显式选择**的独立路径，不会改变任何既有构建的行为。
        //
        // 为什么是一处**字面量比较**而不是一个 argv 扫描器：
        //   扫描器（循环 + 前缀匹配 + `--flag=value` 语法 + 路径缓冲区）
        //   是几千字节的引擎代码，而本入口只需要「证明 S5 接上了」。
        //   传入的路径固定为同目录的 `s5_kv.vmb`，所以不需要解析值。
        //   这**没有**牺牲任何守卫 —— 所有校验都在 core/src/neuron_kv.rs。
        #[cfg(feature = "neuron-kv")]
        {
            if s5_load::wants_load() {
                return s5_load::run_file(s5_load::DEFAULT_PATH.as_ptr());
            }
        }
        #[cfg(feature = "bench")]
        {
            return bench::run();
        }
        #[cfg(feature = "mmap-aot")]
        {
            // P10 任务 C：指令流内嵌 + 权重 mmap。
            // 优先级最高：这是一条显式选择的独立路径。
            return mmap_aot::run_mapped();
        }
        #[cfg(all(feature = "aot-train", not(feature = "bench")))]
        {
            // AOT 训练闭环：模型 + 数据集全部内嵌，不读任何外部文件。
            // 内部会按名字从 aot-multi 的表里装两条指令流
            // （train 含反向 / eval 只前向），并处理只读->可写镜像。
            return aot_train::run_training();
        }
        #[cfg(all(feature = "aot-multi", not(feature = "aot-train"), not(feature = "bench")))]
        {
            // 多模型 AOT：exe 内嵌 N 个模型，用 --model=<name> 选。
            // 无参数时用表首项（与单模型版语义一致）。
            // `--list` 只列名字并以 0 退出。
            match aot_multi::load_selected() {
                Ok(mut vm) => vm.run(),
                Err(code) => code,
            }
        }
        #[cfg(all(
            feature = "aot",
            not(feature = "aot-multi"),
            not(feature = "aot-train"),
            not(feature = "bench")
        ))]
        {
            // AOT 构建：不读任何文件，直接用烧进 .rdata 的模型字节。
            // aot::load_embedded 负责「只读源 -> 可写 64B 对齐镜像」的拷贝：
            //   * .rdata 是只读页，而 Vm::load 把常量池当可写区
            //     （OP_ADAM / OP_SGD 就地更新参数）—— 直接喂进去 = 访问违例
            //   * 常量池要求 CONST_ALIGN = 64 对齐，text 的 Ins 要求 4 对齐
            //   * 尾部留 CONST_ALIGN 余量，与 read_file 路径的堆占用逐字节一致
            let mut vm = aot::load_embedded();
            vm.run()
        }
        #[cfg(all(
            not(feature = "aot"),
            not(feature = "aot-multi"),
            not(feature = "aot-train"),
            not(feature = "bench")
        ))]
        {
            // 默认构建：固定读 model.vmb。
            // 需要命令行参数请用 --features bench（决策 5：30KB 只约束引擎核心，
            // 而 argv/stdout 计时/dump 属于基准设施，不算引擎本体）。
            let p = b"model.vmb\0";
            let (buf, _len) = vmlm_core::read_file(p.as_ptr());
            let mut vm = Vm::load(buf);
            // P13 task A: feed the prompt from a real file, not from a
            // compiled-in constant.
            //
            // WHY THIS MATTERS: before P13 the prompt rows were extracted
            // host-side and shipped as a `prompt_emb` constant with
            // `COPY h, prompt_emb`, so the token ids were baked into the
            // 1.76 GB .vmb -- every new prompt meant a ~106 s reassembly.
            // Embedding is a lookup, not a compilation.
            //
            // The mechanism needs NO new engine code: `st_` slots are already
            // SLOT_STATE (engine-owned, persistent -- ESERNN's recurrence uses
            // them), and `find_slot`/`slot_mut` are already public. So the
            // runtime writes the ids into the state region before run().
            load_tokens(&mut vm);
            vm.run()
        }
    }
}

/// True if the program declares a slot named `name`.
///
/// Implemented HERE (runtime side) rather than in the engine on purpose: the
/// engine has a hard size budget (53312 B, currently 53248 B) and this is pure
/// benchmark/plumbing code. `Vm::name_tab` and `format::hash32` are already
/// public, so the engine needs no new bytes at all. The alternative --
/// `Vm::find_slot` -- exits(7) on a miss, which is exactly wrong for a probe.
#[cfg(all(
    not(feature = "aot"),
    not(feature = "aot-multi"),
    not(feature = "aot-train"),
    not(feature = "bench")
))]
unsafe fn slot_index(vm: &Vm, name: &[u8]) -> Option<usize> {
    let mut h = 2166136261u32; // FNV-1a 32, matching format::hash32
    let mut k = 0;
    while k < name.len() {
        h ^= name[k] as u32;
        h = h.wrapping_mul(16777619);
        k += 1;
    }
    let mut i = 0usize;
    while i < vm.n_slots {
        let e = vm.name_tab.add(i * 4);
        let v = (*e as u32) | ((*e.add(1) as u32) << 8) | ((*e.add(2) as u32) << 16)
            | ((*e.add(3) as u32) << 24);
        if v == h {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// Read token ids from `path` (little-endian u32) into the named state slot.
///
/// SILENT when the program does not declare the slot: most example programs
/// have no prompt and must keep working unchanged. But when the program DOES
/// declare the slot and the file is present, an over-long or empty read is a
/// hard failure -- a partially-filled prompt would otherwise produce logits
/// that look plausible and are wrong, which is exactly the silent-degradation
/// class this project converts to hard failures.
///
/// P13 task C reuses this for `st_gen` (the token ids for the decode steps), so
/// the same rules and the same guard apply to both.
#[cfg(all(
    not(feature = "aot"),
    not(feature = "aot-multi"),
    not(feature = "aot-train"),
    not(feature = "bench")
))]
unsafe fn load_ids(vm: &mut Vm, slot_name: &[u8], path: *const u8, what: &[u8]) {
    let slot = match slot_index(vm, slot_name) {
        Some(s) => s,
        None => return, // program does not use this input
    };
    // No existence probe: a program that DECLARES the slot but ships no file is
    // malformed input, and this project's rule is that illegal input must fail
    // loudly rather than silently proceed. `read_file` already exits(2) with
    // "cannot open file" in that case, which is exactly right -- and it saves a
    // kernel32 import plus its call site.
    let (buf, len) = vmlm_core::read_file(path);
    let n = len / 4;
    if n == 0 {
        vmlm_core::print(what);
        vmlm_core::print(b" is empty\n");
        vmlm_core::exit(9);
    }
    let (rows, cols) = vm.slot_shape(slot);
    if n > rows * cols {
        vmlm_core::print(what);
        vmlm_core::print(b" is longer than its slot\n");
        vmlm_core::exit(9);
    }
    let dst = vm.slot_mut(slot);
    let mut i = 0usize;
    while i < n {
        // read_unaligned: read_file's buffer is CONST_ALIGN(64)-aligned, but
        // writing the u32 read this way keeps the loop small and is correct
        // regardless of alignment. Measured: the four-byte-shift version of
        // this loop cost 512 B of the engine budget (tools/p13_attrib.py),
        // which is the 448 B overrun plus slack.
        let v = core::ptr::read_unaligned(buf.add(i * 4) as *const u32);
        *dst.add(i) = v as f32;
        i += 1;
    }
}

/// Load the prompt (st_tok) and, in decode builds, the tokens to generate
/// (st_gen). Split out so the default entry point stays a single call.
#[cfg(all(
    not(feature = "aot"),
    not(feature = "aot-multi"),
    not(feature = "aot-train"),
    not(feature = "bench")
))]
unsafe fn load_tokens(vm: &mut Vm) {
    load_ids(vm, b"st_tok", b"tokens.bin\0".as_ptr(), b"tokens.bin");
    // P13 task C: the decode steps read their token ids from st_gen. Absent
    // slot (no --decode in the program) => this is a no-op, so non-decode
    // artifacts behave exactly as before.
    load_ids(vm, b"st_gen", b"gen_tokens.bin\0".as_ptr(), b"gen_tokens.bin");
}
