//! build.rs：把"要嵌入 EXE 的权重/词表"路径在**编译期**解析出来，交给 include_bytes! 用。
//!
//! 为什么需要它：`include_bytes!` 只接受字面量或 `env!()` 展开的路径。
//! 优先级：
//!   1. 环境变量 VMLM_MODEL / VMLM_VOCAB（显式指定）
//!   2. artifacts/b1_C3_10k.cpk（用户预期的 40MB 权重）
//!   3. artifacts 下最新的 *.cpk（兜底，便于本机验证）
//! 词表同理：artifacts/b1_char/char_vocab_hex.txt
//!
//! ⚠️ 注意：这个 build.rs 会影响**整个 vmlm-train 包**（包括 vmlm-b1-full），
//!    所以它只设置环境变量、不生成代码、不碰任何源文件。

use std::path::{Path, PathBuf};

fn newest_cpk(dir: &Path) -> Option<PathBuf> {
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    for e in std::fs::read_dir(dir).ok()? {
        let e = e.ok()?;
        let p = e.path();
        if p.extension().map(|x| x == "cpk").unwrap_or(false) {
            let t = e
                .metadata()
                .and_then(|m| m.modified())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
            if best.as_ref().map(|(bt, _)| t > *bt).unwrap_or(true) {
                best = Some((t, p));
            }
        }
    }
    best.map(|(_, p)| p)
}

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    let art = root.join("artifacts");

    let model = std::env::var("VMLM_MODEL")
        .map(PathBuf::from)
        .ok()
        .filter(|p| p.is_file())
        .or_else(|| {
            let p = art.join("b1_C3_10k.cpk");
            if p.is_file() {
                Some(p)
            } else {
                None
            }
        })
        .or_else(|| newest_cpk(&art));
    // 开源默认：仓库不带权重。没有 .cpk 时生成占位文件（刻意 <60 字节，让 vmlm-chat 的格式校验先拦下）。
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap_or_else(|_| ".".into()));
    let model = match model {
        Some(p) => {
            println!(
                "cargo:warning=embed model: {} ({:.1} MB)",
                p.display(),
                std::fs::metadata(&p).map(|m| m.len() as f64 / 1e6).unwrap_or(0.0)
            );
            println!("cargo:rerun-if-changed={}", p.display());
            p
        }
        None => {
            println!(
                "cargo:warning=未找到 artifacts/*.cpk —— 生成占位权重（开源默认：仓库不带权重）。\
                 想构建可用的 vmlm-chat，请训练出 .cpk 放进 artifacts/，或用 VMLM_MODEL=<path> 指定。"
            );
            let p = out_dir.join("stub_model.cpk");
            let _ = std::fs::create_dir_all(&out_dir);
            let _ = std::fs::write(&p, [0xE1u8, 0x0D, 0xC0, 0xB1, 0, 0, 0, 0]);
            p
        }
    };

    let vocab = std::env::var("VMLM_VOCAB")
        .map(PathBuf::from)
        .ok()
        .filter(|p| p.is_file())
        .unwrap_or_else(|| art.join("b1_char").join("char_vocab_hex.txt"));
    let vocab = if vocab.is_file() {
        println!("cargo:rerun-if-changed={}", vocab.display());
        vocab
    } else {
        println!("cargo:warning=未找到词表 —— 生成占位词表（开源默认：仓库不带词表）。");
        let p = out_dir.join("stub_vocab_hex.txt");
        let _ = std::fs::create_dir_all(&out_dir);
        let _ = std::fs::write(&p, b"0000\n");
        p
    };

    println!("cargo:warning=embed model: {} ({:.1} MB)", model.display(), std::fs::metadata(&model).map(|m| m.len() as f64 / 1e6).unwrap_or(0.0));
    println!("cargo:warning=embed vocab: {}", vocab.display());
    println!("cargo:rustc-env=VMLM_MODEL_RESOLVED={}", model.display());
    println!("cargo:rustc-env=VMLM_VOCAB_RESOLVED={}", vocab.display());
    println!("cargo:rerun-if-env-changed=VMLM_MODEL");
    println!("cargo:rerun-if-env-changed=VMLM_VOCAB");
    println!("cargo:rerun-if-changed={}", model.display());
    println!("cargo:rerun-if-changed={}", vocab.display());
}
