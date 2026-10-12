// build.rs —— AOT 模型的**构建期**生成（VMLM P8 成员 G）
//
// 为什么这条路曾经被认为不可行（P7-B 的结论）
// ------------------------------------------------
// `runtime/.cargo/config.toml` 里是
//     [target.x86_64-pc-windows-msvc]
//     rustflags = [ "-C", "link-arg=/ENTRY:mainCRTStartup",
//                   "-C", "link-arg=/NODEFAULTLIB", ... ]
// cargo 会把 `target.*.rustflags` 应用到**该 crate 的每一次 rustc 调用**，
// 其中就包括 build script（一个 std 宿主程序）。`/NODEFAULTLIB` 把它的
// CRT 关掉，于是 build script 自己链不过：
//     LNK2001 mainCRTStartup / __CxxFrameHandler3 / memcpy / memmove /
//     memset / memcmp / strlen / _fltused / _tls_index / _tls_used /
//     _CxxThrowException / type_info::vftable   → LNK1120 (13 个)
// **这个错误是真实的，已逐条复现**。
//
// 为什么其实可行（本文件能用）
// ------------------------------------------------
// 只要 cargo 显式带 `--target x86_64-pc-windows-msvc` 调用，cargo 就区分
// 「宿主」与「目标」，`target.*.rustflags` 只作用于**目标**产物，build script
// 回落到正常的宿主链接 —— 于是能跑。实测：
//     不带 --target → LNK1120（13 个符号）
//     带   --target → 正常，产物尺寸与不带 build.rs 时一致
//
// 所以：**本文件要求 `cargo build --release --target x86_64-pc-windows-msvc`**。
// 不带 --target 时会在链接 build script 这一步就失败 —— 这是预期的、
// 也是唯一正确的失败方式（而不是产出一个坏 exe）。
//
// 产物语义
// ------------------------------------------------
// build.rs 把模型表写进 OUT_DIR，由 aot_multi.rs `include!`（构建期驱动）；
// `tools/aot_multi_build.py` 则离线写到 `src/models_aot.rs`。
// 两条路径由 aot_multi.rs 用 cfg 二选一，见该文件的 `include!` 段。
//
// 环境变量
// ------------------------------------------------
//   VMLM_AOT_MODELS   分号分隔的 `name=path.vmb` 列表（相对 crate 根或绝对）
//   VMLM_AOT_MANIFEST 从一个文本文件读模型列表（每行 name=path，# 注释）
//   (都不设时)         读 `aot_models.txt`（若存在）；再不行则用内置默认表
//
// 例：
//   $env:VMLM_AOT_MODELS="linear_relu=../examples/linear_relu.vmb;attn=../examples/attn/block.vmb"
//   cd runtime; cargo build --release --target x86_64-pc-windows-msvc --features aot-multi

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const PER_LINE: usize = 16;

fn main() {
    println!("cargo:rerun-if-env-changed=VMLM_AOT_MODELS");
    println!("cargo:rerun-if-env-changed=VMLM_AOT_MANIFEST");
    println!("cargo:rerun-if-changed=aot_models.txt");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    println!("cargo:rerun-if-env-changed=VMLM_AOT_DATA");

    // ---- aot-train：数据集也要内嵌（模型表之外的第二件事）----
    // 数据集约 19 MB，生成的 .rs 约 128 MB —— 不适合当源码提交，
    // 所以构建期生成到 OUT_DIR，而不是放 src/data_aot.rs。
    if env::var("CARGO_FEATURE_AOT_TRAIN").is_ok() {
        let src = render_data(&manifest_dir);
        fs::write(out_dir.join("data_aot.rs"), src).unwrap();
    } else {
        let _ = fs::write(out_dir.join("data_aot.rs"), "// aot-train not enabled\n");
    }

    // 只有 aot-multi 才需要模型表。其它 feature 组合下写个占位，
    // 免得 OUT_DIR 里留着上一次的旧文件让人误判。
    if env::var("CARGO_FEATURE_AOT_MULTI").is_err() {
        let _ = fs::write(out_dir.join("models_aot.rs"), "// aot-multi not enabled\n");
        return;
    }

    let specs = collect_specs(&manifest_dir);

    let mut models: Vec<(String, Vec<u8>, String)> = Vec::new();
    let mut seen: Vec<String> = Vec::new();
    for (name, path) in specs {
        if seen.contains(&name) {
            panic!("VMLM AOT: duplicate model name {:?}", name);
        }
        seen.push(name.clone());

        let rel = if Path::new(&path).is_absolute() {
            PathBuf::from(&path)
        } else {
            manifest_dir.join(&path)
        };
        match fs::read(&rel) {
            Ok(data) => {
                if data.len() < 32
                    || u32::from_le_bytes([data[0], data[1], data[2], data[3]]) != 0x4D4C_4D56
                {
                    panic!(
                        "VMLM AOT: {} is not a valid .vmb (bad magic or too small)",
                        rel.display()
                    );
                }
                println!("cargo:rerun-if-changed={}", rel.display());
                models.push((name, data, rel.display().to_string()));
            }
            Err(e) => panic!("VMLM AOT: cannot read {}: {}", rel.display(), e),
        }
    }

    if models.is_empty() {
        panic!("VMLM AOT: no models given; set VMLM_AOT_MODELS or create aot_models.txt");
    }

    let src = render(&models);
    fs::write(out_dir.join("models_aot.rs"), src).unwrap();

    // cargo:warning 是构建期唯一能向外输出的通道
    let names: Vec<&str> = models.iter().map(|m| m.0.as_str()).collect();
    let total: usize = models.iter().map(|m| m.1.len()).sum();
    println!(
        "cargo:warning=VMLM AOT: embedded {} model(s) [{}], {} B payload",
        models.len(),
        names.join(", "),
        total
    );
}

/// 按优先级收集 (name, path) 列表。
fn collect_specs(manifest_dir: &Path) -> Vec<(String, String)> {
    // 1) VMLM_AOT_MODELS="a=1.vmb;b=2.vmb"
    if let Ok(v) = env::var("VMLM_AOT_MODELS") {
        if !v.trim().is_empty() {
            return parse_lines(v.split(';'));
        }
    }
    // 2) VMLM_AOT_MANIFEST=<file>
    if let Ok(mf) = env::var("VMLM_AOT_MANIFEST") {
        if let Ok(text) = fs::read_to_string(&mf) {
            return parse_lines(text.lines());
        }
    }
    // 3) aot_models.txt（crate 根）
    let local = manifest_dir.join("aot_models.txt");
    if local.exists() {
        if let Ok(text) = fs::read_to_string(&local) {
            return parse_lines(text.lines());
        }
    }
    // 4) 内置默认（与验收脚本一致的一组小模型）
    parse_lines(
        [
            "linear_relu=../examples/linear_relu.vmb",
            "attn=../examples/attn/block.vmb",
            "sparse=../examples/sparse/ffn.vmb",
            "p8stack=../examples/p8_stack_test.vmb",
            "nnreg=../examples/p7/nn_regress.vmb",
        ]
        .iter()
        .copied(),
    )
}

fn parse_lines<'a, I: Iterator<Item = &'a str>>(it: I) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for raw in it {
        let ln = raw.trim().trim_matches('"');
        if ln.is_empty() || ln.starts_with('#') {
            continue;
        }
        match ln.split_once('=') {
            Some((n, p)) => out.push((n.trim().to_string(), p.trim().to_string())),
            None => {
                let stem = Path::new(ln)
                    .file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| ln.to_string());
                out.push((stem, ln.to_string()));
            }
        }
    }
    out
}

/// 渲染成与 tools/aot_multi_build.py 相同的 Rust 源。
fn render(models: &[(String, Vec<u8>, String)]) -> String {
    let mut s = String::new();
    s.push_str("// 本文件由 build.rs 自动生成 —— 请勿手改。\n");
    s.push_str(&format!(
        "// 内嵌模型数: {}，总载荷 {} 字节\n",
        models.len(),
        models.iter().map(|m| m.1.len()).sum::<usize>()
    ));
    for (n, d, p) in models {
        s.push_str(&format!("//   {:<16} {:>8} B   <- {}\n", n, d.len(), p));
    }
    s.push_str("//\n");
    s.push_str("// 用途: runtime --features aot-multi 时被 aot_multi.rs include! 进 main.rs。\n");
    s.push_str("// 运行时用 --model=<name> 选择；无参数则用表首项。\n\n");
    s.push_str("pub struct AotEntry {\n");
    s.push_str("    /// NUL 结尾的名字，no_std 下便于按字节比较\n");
    s.push_str("    pub name: &'static [u8],\n");
    s.push_str("    pub bytes: &'static [u8],\n");
    s.push_str("}\n\n");

    let mut syms = Vec::new();
    for (idx, (name, data, _)) in models.iter().enumerate() {
        let sym = format!("AOT_BYTES_{}", ident(name));
        s.push_str(&format!("// [{}] {}  ({} 字节)\n", idx, name, data.len()));
        s.push_str("#[repr(C, align(64))]\n");
        s.push_str(&format!("pub struct AotBlob{} {{\n", idx));
        s.push_str(&format!("    pub bytes: [u8; {}],\n", data.len()));
        s.push_str("}\n\n");
        s.push_str(&format!(
            "static {}_BLOB: AotBlob{} = AotBlob{} {{\n",
            sym, idx, idx
        ));
        s.push_str("    bytes: [\n");
        for chunk in data.chunks(PER_LINE) {
            let hex: Vec<String> = chunk.iter().map(|b| format!("0x{:02x},", b)).collect();
            s.push_str("        ");
            s.push_str(&hex.join(" "));
            s.push('\n');
        }
        s.push_str("    ],\n};\n\n");
        s.push_str(&format!("static {}: &[u8] = &{}_BLOB.bytes;\n\n", sym, sym));
        syms.push((name.clone(), sym));
    }

    s.push_str("/// 名字表 + 载荷表；顺序即 `--model` 的匹配顺序。\n");
    s.push_str("pub static AOT_MODELS: &[AotEntry] = &[\n");
    for (name, sym) in &syms {
        s.push_str(&format!(
            "    AotEntry {{ name: b\"{}\\0\", bytes: {} }},\n",
            name, sym
        ));
    }
    s.push_str("];\n\n");
    s.push_str(&format!(
        "pub const AOT_MODEL_COUNT: usize = {};\n",
        models.len()
    ));
    s.push_str(&format!(
        "pub const AOT_TOTAL_BYTES: usize = {};\n",
        models
            .iter()
            .map(|m| m.1.len().to_string())
            .collect::<Vec<_>>()
            .join(" + ")
    ));
    s
}

/// 与 python 生成器同样的标识符规则：大写，非 ASCII 字母数字 -> _
fn ident(name: &str) -> String {
    let mut out: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect();
    if out.is_empty() || out.chars().next().unwrap().is_ascii_digit() {
        out.insert_str(0, "M_");
    }
    out
}

/// aot-train 用的数据集表。
///
/// 形如 `NAME=path.bin`，分号分隔，经 `VMLM_AOT_DATA` 传入；
/// 未设置时用 examples/mnist 的默认五件套。
///
/// 注意：`examples/mnist/Xte.bin`（MNIST 测试集切片，约 1.9 MB）不入库，
/// 需先跑 `python examples/mnist/make_mnist.py` 生成，见该目录的 README。
fn render_data(manifest_dir: &Path) -> String {
    let default = "META=../examples/mnist/meta_f32.bin;\
                   X=../examples/mnist/X.bin;\
                   Y=../examples/mnist/Y.bin;\
                   XTE=../examples/mnist/Xte.bin;\
                   YTE=../examples/mnist/Yte.bin";
    let spec = env::var("VMLM_AOT_DATA").unwrap_or_else(|_| default.to_string());

    let mut items: Vec<(String, Vec<u8>, String)> = Vec::new();
    for part in spec.split(';') {
        let ln = part.trim();
        if ln.is_empty() {
            continue;
        }
        let (name, path) = match ln.split_once('=') {
            Some((n, p)) => (n.trim().to_string(), p.trim().to_string()),
            None => continue,
        };
        let full = if Path::new(&path).is_absolute() {
            PathBuf::from(&path)
        } else {
            manifest_dir.join(&path)
        };
        let data = fs::read(&full)
            .unwrap_or_else(|e| panic!("VMLM AOT data: cannot read {}: {}", full.display(), e));
        if data.len() % 4 != 0 {
            panic!(
                "VMLM AOT data: {} size {} not a multiple of 4 (f32 data)",
                full.display(),
                data.len()
            );
        }
        println!("cargo:rerun-if-changed={}", full.display());
        items.push((name, data, full.display().to_string()));
    }
    if items.is_empty() {
        panic!("VMLM AOT data: no data blocks; set VMLM_AOT_DATA");
    }

    let mut s = String::new();
    s.push_str("// 本文件由 build.rs 自动生成 —— 请勿手改。\n");
    s.push_str(&format!(
        "// 内嵌数据块: {}，总载荷 {} 字节\n",
        items.len(),
        items.iter().map(|i| i.1.len()).sum::<usize>()
    ));
    for (n, d, p) in &items {
        s.push_str(&format!(
            "//   {:<8} {:>10} B ({} f32)  <- {}\n",
            n,
            d.len(),
            d.len() / 4,
            p
        ));
    }
    s.push_str("//\n// 用途: runtime --features aot-train 时被 aot_train.rs include! 进 main.rs，\n");
    s.push_str("// 让训练闭环**不读任何外部文件**。\n\n");
    s.push_str("pub struct AotData {\n");
    s.push_str("    /// NUL 结尾的名字\n");
    s.push_str("    pub name: &'static [u8],\n");
    s.push_str("    /// 小端 f32 原始字节\n");
    s.push_str("    pub bytes: &'static [u8],\n");
    s.push_str("}\n\n");

    let mut syms = Vec::new();
    for (idx, (name, data, _)) in items.iter().enumerate() {
        let sym = format!("AOT_DATA_{}", ident(name));
        s.push_str(&format!("// [{}] {}  ({} 字节)\n", idx, name, data.len()));
        s.push_str("#[repr(C, align(64))]\n");
        s.push_str(&format!("pub struct AotDataBlob{} {{\n", idx));
        s.push_str(&format!("    pub bytes: [u8; {}],\n", data.len()));
        s.push_str("}\n\n");
        s.push_str(&format!(
            "static {}_BLOB: AotDataBlob{} = AotDataBlob{} {{\n",
            sym, idx, idx
        ));
        s.push_str("    bytes: [\n");
        for chunk in data.chunks(PER_LINE) {
            let hex: Vec<String> = chunk.iter().map(|b| format!("0x{:02x},", b)).collect();
            s.push_str("        ");
            s.push_str(&hex.join(" "));
            s.push('\n');
        }
        s.push_str("    ],\n};\n\n");
        s.push_str(&format!("static {}: &[u8] = &{}_BLOB.bytes;\n\n", sym, sym));
        syms.push((name.clone(), sym));
    }

    s.push_str("/// 数据表；按名字查找。\n");
    s.push_str("pub static AOT_DATA: &[AotData] = &[\n");
    for (name, sym) in &syms {
        s.push_str(&format!(
            "    AotData {{ name: b\"{}\\0\", bytes: {} }},\n",
            name, sym
        ));
    }
    s.push_str("];\n");

    println!(
        "cargo:warning=VMLM AOT: embedded {} data block(s), {} B payload",
        items.len(),
        items.iter().map(|i| i.1.len()).sum::<usize>()
    );
    s
}
