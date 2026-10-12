// 多模型 AOT：一个 .exe 内嵌 N 个 .vmb，运行时用 `--model=<name>` 选择。
//
// 解决的遗留问题（P7-B）：
//   原来 `src/model_aot.rs` 是单一生成文件，一次只能内嵌一个模型，
//   换模型必须重跑 `aot_build.py` 再 `cargo build`。现在由
//   `tools/aot_multi_build.py`（离线）或 `build.rs`（环境变量驱动）
//   生成 `src/models_aot.rs`（N 个 align(64) 常量数组 + 一张名字表），
//   本模块在启动时按名字选一个。
//
// 与 P7-B 的关系：`aot.rs`（单模型路径）原样保留、继续可用；
// 本模块是叠加的多模型路径。两者共用 `aot::copy_to_aligned_writable`
// 的「只读 .rdata -> 64B 对齐可写堆」拷贝语义 —— 那个坑（写只读常量池
// 崩 0xC0000005）对多模型同样存在，且**每个被选中的模型都要各自拷一份**。
// 复用同一份实现是为了避免两处的对齐/尾余量语义漂移。
//
// 无参数时的行为：用表首项（与 P7-B 单模型版「跑内嵌的那一个」语义一致），
// 保证既有调用方式不破。
//
// 命令行（no_std 无 CRT，没有 argv，见 args.rs）：
//   vmlm-runtime.exe                      -> 表首项（默认模型）
//   vmlm-runtime.exe --model=attn         -> 选 attn
//   vmlm-runtime.exe --list               -> 列出所有内嵌模型名，退出码 0

use crate::args;
use vmlm_core::interp::Vm;

// 模型表来源二选一（P8 任务 2）：
//   * 默认走 build.rs 生成的 OUT_DIR/models_aot.rs —— 换模型只改环境变量
//     VMLM_AOT_MODELS，不用手工重跑生成器。
//     **前置条件：cargo 必须带 `--target x86_64-pc-windows-msvc`**，否则
//     config.toml 的 /NODEFAULTLIB 会连 build script 一起干掉（LNK1120）。
//   * `--features aot-offline` 时走 tools/aot_multi_build.py 生成的
//     src/models_aot.rs —— 离线路径，零构建期依赖。
//     **注意**：src/models_aot.rs 是生成物、不入库；用这条路径前必须先跑
//         python tools/aot_multi_build.py runtime/src/models_aot.rs <name>=<vmb>...
//     否则会 "couldn't read runtime\src\models_aot.rs"。
//
// 两条路径的源格式完全相同（同一套 AotEntry），所以下面的代码不用改。
#[cfg(not(feature = "aot-offline"))]
include!(concat!(env!("OUT_DIR"), "/models_aot.rs"));

#[cfg(feature = "aot-offline")]
include!("models_aot.rs");

/// 按名字取模型字节（供其它 AOT 模块复用，如 aot_train 的评估阶段）。
///
/// # Safety
/// 只读静态表。
pub unsafe fn model_bytes(name: &[u8]) -> Option<&'static [u8]> {
    find(name)
}

/// 在 AOT_MODELS 里按名字找模型，返回字节切片。
///
/// 名字用 NUL 结尾串存放（生成器给每个名字带了 `\0`），
/// 比较时把表项剥掉结尾 NUL 再与查询串比。
fn find(name: &[u8]) -> Option<&'static [u8]> {
    let mut i = 0;
    while i < AOT_MODELS.len() {
        let e = &AOT_MODELS[i];
        let n = e.name.len() - 1;
        if n == name.len() && args::eq(&e.name[..n], name) {
            return Some(e.bytes);
        }
        i += 1;
    }
    None
}

/// 列出所有内嵌模型名（用于 `--list` 与「名字打错」时的提示）。
pub unsafe fn print_model_list() {
    vmlm_core::print(b"embedded models:\n");
    let mut i = 0;
    while i < AOT_MODELS.len() {
        let e = &AOT_MODELS[i];
        vmlm_core::print(b"  ");
        vmlm_core::print(&e.name[..e.name.len() - 1]);
        if i == 0 {
            vmlm_core::print(b"  (default)");
        }
        vmlm_core::print(b"\n");
        i += 1;
    }
    vmlm_core::print(b"usage: --model=<name>\n");
}

/// 启动时按 `--model=<name>` 选模型并构造 Vm。
///
/// 返回 `Err(code)` 表示「不装载、直接以该退出码结束」：
///   * `--list`         -> Err(0)
///   * 模型名不存在      -> Err(1)（打印可选列表）
///
/// # Safety
/// 只应在启动时调用一次。
pub unsafe fn load_selected() -> Result<Vm, u32> {
    // `--list`：只列名字，不装载
    if args::has_flag(b"list") {
        print_model_list();
        return Err(0);
    }

    // 选模型：--model=<name>，缺省表首项
    let mut mbuf = [0u8; 512];
    let chosen: &[u8] = match args::find_flag(b"model", &mut mbuf) {
        Some(v) => v,
        None => {
            let e = &AOT_MODELS[0];
            &e.name[..e.name.len() - 1]
        }
    };

    let bytes = match find(chosen) {
        Some(b) => b,
        None => {
            vmlm_core::print(b"unknown model: ");
            vmlm_core::print(chosen);
            vmlm_core::print(b"\n");
            print_model_list();
            return Err(1);
        }
    };

    // 关键：只读 .rdata -> 64B 对齐可写堆。
    // 与单模型路径共用同一份实现，避免两处对齐/余量语义漂移。
    let src = bytes.as_ptr();
    let len = bytes.len();
    Ok(Vm::load(crate::aot::copy_to_aligned_writable(src, len)))
}
