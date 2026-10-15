//! Sparrow V0.1 —— 命令行前端。
//!
//! ```text
//! sparrow <file.sparrow>                   # 输出架构级 .spar IR 到 stdout
//! sparrow <file.sparrow> -o out.spar       # 写文件
//! sparrow <file.sparrow> --backend vml     # 输出 opcode 级 .vml（V0.1.1 后端）
//! sparrow <file.sparrow> --emit-ast-json   # 输出 AST JSON（验收 12）
//! sparrow <file.sparrow> --emit-tokens     # 输出 token 流（调试宽容语法用）
//! sparrow <file.sparrow> --sha256          # 顺带打印产物的 sha256
//! ```
//!
//! 四层后缀（V0.1.1 第一部分）：`.sparrow` 源码 → `.spar` 架构 IR → `.vml` opcode → `.vmb` 字节码。

use std::io::Read;
use std::process::ExitCode;

use sparrow::diag::SourceFile;
use sparrow::sha256::sha256_hex;

#[derive(PartialEq, Eq, Clone, Copy)]
enum Mode {
    /// 架构级 `.spar`（V0.1 的 `.vml` 改名）
    Spar,
    /// opcode 级 `.vml`（V0.1.1 新后端）
    VmlOpcode,
    /// Rust AOT 源（V0.1.4 路径 2 后端）
    Lib,
    AstJson,
    Tokens,
    Help,
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut mode = Mode::Spar;
    let mut out_path: Option<String> = None;
    let mut args_path: Option<String> = None;
    let mut asm_path: Option<String> = None;
    let mut consts_dir: Option<String> = None;
    let mut mode_explicit = false;
    let mut input: Option<String> = None;
    let mut print_hash = false;
    let mut loop_repeat = false;
    let mut slot_reuse = false;
    let mut weight_dtype = sparrow::vmlgen::DataType::F32;
    let mut vmlm_root: Option<String> = None;
    let mut bench_iters: u32 = 0;

    let mut i = 0;
    while i < argv.len() {
        let a = argv[i].as_str();
        match a {
            "--emit-ast-json" | "--emit-ast" => {
                mode = Mode::AstJson;
                mode_explicit = true;
            }
            "--emit-tokens" => {
                mode = Mode::Tokens;
                mode_explicit = true;
            }
            "--emit-spar" => {
                mode = Mode::Spar;
                mode_explicit = true;
            }
            "--asm" => {
                i += 1;
                if i >= argv.len() {
                    return fail_usage("`--asm` 后面要跟汇编器路径（如 ./vmlm-asm.exe）");
                }
                asm_path = Some(argv[i].clone());
            }
            "--consts-dir" => {
                i += 1;
                if i >= argv.len() {
                    return fail_usage("`--consts-dir` 后面要跟目录（权重 .bin 所在处）");
                }
                consts_dir = Some(argv[i].clone());
            }
            "--backend" => {
                i += 1;
                if i >= argv.len() {
                    return fail_usage("`--backend` 后面要跟 `vml`");
                }
                match argv[i].as_str() {
                    "vml" => mode = Mode::VmlOpcode,
                    "spar" => mode = Mode::Spar,
                    // 任务书第四部分的 `--backend lib --lang rust` 排在 V0.1.4。
                    // 这里显式报"未实现"而不是"未知后端"，避免把路线图当成拼写错误。
                    // 路径 2 后端：`.spar → Rust AOT 源`（V0.1.4）
                    "lib" => mode = Mode::Lib,
                    other => {
                        return fail_usage(&format!(
                            "未知后端 `{other}`（`vml` = opcode 中间层，`lib` = Rust AOT 源）"
                        ))
                    }
                }
            }
            "--lang" => {
                i += 1;
                if i >= argv.len() {
                    return fail_usage("`--lang` 后面要跟 `rust`");
                }
                if argv[i] != "rust" {
                    return fail_usage(&format!("`--lang {}` 还不支持（只有 `rust`）", argv[i]));
                }
            }
            "--vmlm-root" => {
                i += 1;
                if i >= argv.len() {
                    return fail_usage("`--vmlm-root` 后面要跟 vmlm 仓库根目录（其下有 `ops/`）");
                }
                vmlm_root = Some(argv[i].clone());
            }
            "--sha256" => print_hash = true,
            "--loop-repeat" => loop_repeat = true,
            "--slot-reuse" => slot_reuse = true,
            "--weight-dtype" => {
                i += 1;
                if i >= argv.len() {
                    return fail_usage("`--weight-dtype` 后面要跟 `f32` 或 `bf16`");
                }
                weight_dtype = match argv[i].as_str() {
                    "f32" => sparrow::vmlgen::DataType::F32,
                    "bf16" => sparrow::vmlgen::DataType::Bf16,
                    "int8" => sparrow::vmlgen::DataType::Int8,
                    other => {
                        return fail_usage(&format!(
                            "未知权重精度 `{other}`（f32 / bf16 / int8；int8 只配 `--backend lib`）"
                        ))
                    }
                };
            }
            "--no-loop-repeat" => loop_repeat = false,
            "--bench-iters" => {
                i += 1;
                if i >= argv.len() {
                    return fail_usage("`--bench-iters` 后面要跟正整数（生成的 AOT 程序计时几遍）");
                }
                bench_iters = match argv[i].parse::<u32>() {
                    Ok(n) if n > 0 => n,
                    _ => {
                        return fail_usage(&format!(
                            "`--bench-iters {}` 不是正整数",
                            argv[i]
                        ))
                    }
                };
            }
            "-h" | "--help" => mode = Mode::Help,
            "-o" | "--output" => {
                i += 1;
                if i >= argv.len() {
                    return fail_usage("`-o` 后面要跟输出路径");
                }
                out_path = Some(argv[i].clone());
            }
            "--args" => {
                i += 1;
                if i >= argv.len() {
                    return fail_usage("`--args` 后面要跟权重清单路径");
                }
                args_path = Some(argv[i].clone());
            }
            _ if a.starts_with("--") => {
                return fail_usage(&format!("未知选项 `{a}`"));
            }
            _ => {
                if input.is_some() {
                    return fail_usage("一次只能编译一个文件");
                }
                input = Some(a.to_string());
            }
        }
        i += 1;
    }

    if mode == Mode::Help {
        print_help();
        return ExitCode::SUCCESS;
    }

    let Some(path) = input else {
        print_help();
        return ExitCode::from(2);
    };

    let text = if path == "-" {
        let mut s = String::new();
        if std::io::stdin().read_to_string(&mut s).is_err() {
            eprintln!("错误[E000]：读取 stdin 失败");
            return ExitCode::from(1);
        }
        s
    } else {
        match std::fs::read_to_string(&path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("错误[E000]：读不到文件 `{path}`：{e}");
                return ExitCode::from(1);
            }
        }
    };

    let src = SourceFile::new(path.clone(), text);

    // V0.1.2 P0-1：`--asm` 一条命令跑完全链（.sparrow → .spar → .vml → fork 汇编器 → .vmb）
    if let Some(asm) = asm_path {
        if mode_explicit && mode != Mode::VmlOpcode {
            return fail_usage("`--asm` 与 `--emit-ast-json`/`--emit-tokens`/`--emit-spar` 不能同时用");
        }
        if weight_dtype == sparrow::vmlgen::DataType::Int8 {
            eprintln!("错误[E109]：`--weight-dtype int8` 不能走 `.vmb`/汇编器这条路");
            eprintln!("  理由：`.vmb` 指令集里**没有**反量化 GEMM（int8 权重会被当成 f32 读，");
            eprintln!("  直接算出垃圾）。int8 目前只在 `--backend lib` 里可用 —— 它调");
            eprintln!("  `vmlm_ops::kernels::gemm_nt_int8`（q:i8 + 每输出行一个 scale）。");
            return ExitCode::from(2);
        }
        return asm_pipeline(&src, &path, asm, out_path, args_path, consts_dir, loop_repeat, slot_reuse, weight_dtype);
    }

    let opts = sparrow::vmlgen::GenOptions { loop_repeat, slot_reuse, weight_dtype };

    if mode == Mode::Lib {
        if mode_explicit {
            eprintln!("错误[E109]：`--backend lib` 不能与 `--emit-*` 同时用");
            return ExitCode::from(2);
        }
        return lib_backend(&src, &path, out_path, consts_dir, vmlm_root, opts, bench_iters);
    }

    // int8 只服务 lib 后端（上面 `--asm` 分支已挡一次，这里挡住 `--backend vml` 的其余入口）
    if weight_dtype == sparrow::vmlgen::DataType::Int8 {
        eprintln!("错误[E109]：`--weight-dtype int8` 需要 `--backend lib`");
        eprintln!("  `.vmb`/汇编器那条路没有反量化 GEMM 指令；用 `--backend lib --lang rust`");
        eprintln!("  生成 Rust AOT 源（它调 `gemm_nt_int8`），或改用 `--weight-dtype bf16`。");
        return ExitCode::from(2);
    }

    let output = match mode {
        Mode::VmlOpcode => match sparrow::compile_spar(&src) {
            Ok((_, lowered)) => match sparrow::vmlgen::generate_with(&lowered, &path, opts) {
                Ok(gen) => {
                    if let Some(ap) = &args_path {
                        if let Err(e) = std::fs::write(ap, gen.args_text.as_bytes()) {
                            eprintln!("错误[E000]：写不到 `{ap}`：{e}");
                            return ExitCode::from(1);
                        }
                    }
                    // 循环化的计数器/步长、注意力缩放是前端产出的极小常量 —— 必须落在
                    // `.args` 里那些相对路径的基准目录（= 清单所在目录）。
                    //
                    // 兜底顺序（V0.1.5 补了第 3 条）：`--args` 的目录 → `-o` 的目录 →
                    // **输入源码所在目录** → `.`。
                    // 为什么要有第 3 条：原来没给 `-o`/`--args` 时直接落到**进程 CWD**，
                    // 于是"只想把 .vml 打到 stdout"的调用（比如验收驱动的负例）
                    // 会在调用者的当前目录里丢下几个 .bin —— 实测把仓库根弄脏了 4 个文件
                    //（ATTN_scale.bin / t__1_{iters,lstride,nstride}.bin）。
                    // 源码所在目录是确定的、且与这次编译强相关。
                    let dir = args_path
                        .as_deref()
                        .or(out_path.as_deref())
                        .and_then(|p| std::path::Path::new(p).parent())
                        .or_else(|| std::path::Path::new(&path).parent())
                        .filter(|p| !p.as_os_str().is_empty())
                        .map(|p| p.to_path_buf())
                        .unwrap_or_else(|| std::path::PathBuf::from("."));
                    if let Err(e) = write_extra_bins(&gen, &dir) {
                        eprintln!("错误[E000]：写不到循环控制常量 `{}`：{e}", dir.display());
                        return ExitCode::from(1);
                    }
                    gen.vml
                }
                Err(d) => return report(&d.render(&src)),
            },
            Err(d) => return report(&d.render(&src)),
        },
        Mode::Spar => match sparrow::compile_vml(&src) {
            Ok((vml, lowered)) => {
                if lowered.all_models.len() > 1 {
                    eprintln!(
                        "警告[E015]：文件里有 {} 个模型（{}），只输出第一个",
                        lowered.all_models.len(),
                        lowered.all_models.join(", ")
                    );
                }
                vml
            }
            Err(d) => return report(&d.render(&src)),
        },
        Mode::AstJson => match sparrow::parse(&src) {
            Ok(ast) => sparrow::ast_to_json(&ast),
            Err(d) => return report(&d.render(&src)),
        },
        Mode::Tokens => match sparrow::lex(&src) {
            Ok(toks) => {
                let mut s = String::new();
                for t in &toks {
                    let l = src.loc(t.span.start);
                    s.push_str(&format!("{:>4}:{:<4} {}\n", l.line, l.col, t.describe()));
                }
                s
            }
            Err(d) => return report(&d.render(&src)),
        },
        Mode::Help | Mode::Lib => unreachable!(),
    };

    match &out_path {
        Some(p) => {
            if let Err(e) = std::fs::write(p, output.as_bytes()) {
                eprintln!("错误[E000]：写不到 `{p}`：{e}");
                return ExitCode::from(1);
            }
            if print_hash {
                println!("{}  {}", sha256_hex(output.as_bytes()), p);
            }
        }
        None => {
            print!("{output}");
            if print_hash {
                eprintln!("sha256: {}", sha256_hex(output.as_bytes()));
            }
        }
    }
    ExitCode::SUCCESS
}

fn report(rendered: &str) -> ExitCode {
    eprintln!("{rendered}");
    ExitCode::from(1)
}

/// 把 `<path>` 的扩展名换成 `ext`（`a/b/c.vmb` → `a/b/c.vml`）。
fn replace_ext(path: &str, ext: &str) -> String {
    let p = std::path::Path::new(path);
    let mut out = p.to_path_buf();
    out.set_extension(ext);
    out.to_string_lossy().to_string()
}

/// 转绝对路径（**不**做 canonicalize：不要求文件已存在，也不解析链接）。
///
/// 为什么必须转：汇编器子进程的 CWD 被设成常量目录（清单里的 `.bin` 是相对路径），
/// 于是 `.vml`/`.vmb` 若也传相对路径，会被解析成 `<常量目录>/<相对路径>` ——
/// 实测汇编器报 `读不到 .vml`，而那句话只说不存在，不说路径基准变了。
fn abs_path(p: &str) -> std::path::PathBuf {
    let path = std::path::Path::new(p);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map(|c| c.join(path))
            .unwrap_or_else(|_| path.to_path_buf())
    }
}

/// V0.1.2 P0-1：`.sparrow → .spar → .vml →(fork 既有汇编器)→ .vmb`
///
/// 三条硬约束（任务书）：
/// 1. 不内联汇编器逻辑 —— 只 fork；
/// 2. `.vml` 是必经中间层 —— **先落盘**，再把它交给汇编器（用户可留档/复现）；
/// 3. 汇编器报错时原样转发 stderr 与退出码 —— 包括 `0xC0000409` 这种负数码。
fn asm_pipeline(
    src: &SourceFile,
    in_path: &str,
    asm: String,
    out: Option<String>,
    args: Option<String>,
    consts_dir: Option<String>,
    loop_repeat: bool,
    slot_reuse: bool,
    weight_dtype: sparrow::vmlgen::DataType,
) -> ExitCode {
    // 1) 前端两步（架构 IR → opcode .vml + 权重清单）
    let (spar_text, lowered) = match sparrow::compile_spar(src) {
        Ok(v) => v,
        Err(d) => return report(&d.render(src)),
    };
    let gen = match sparrow::vmlgen::generate_with(
        &lowered,
        in_path,
        sparrow::vmlgen::GenOptions { loop_repeat, slot_reuse, weight_dtype },
    ) {
        Ok(g) => g,
        Err(d) => return report(&d.render(src)),
    };

    // 2) 落盘：.vml 必经中间层
    let vmb_path = out.unwrap_or_else(|| replace_ext(in_path, "vmb"));
    let spar_path = replace_ext(&vmb_path, "spar");
    let vml_path = replace_ext(&vmb_path, "vml");
    let args_path = args.unwrap_or_else(|| replace_ext(&vmb_path, "args"));
    // 四层后缀各留一份：.spar / .vml / .args 都是用户可留档的中间产物
    for (p, body) in [
        (&spar_path, spar_text.clone()),
        (&vml_path, gen.vml.clone()),
        (&args_path, gen.args_text.clone()),
    ] {
        if let Err(e) = std::fs::write(p, body.as_bytes()) {
            eprintln!("错误[E000]：写不到 `{p}`：{e}");
            return ExitCode::from(1);
        }
    }

    // 3) 常量目录：清单里的 .bin 是相对路径，默认取清单所在目录
    let cwd = consts_dir.or_else(|| {
        std::path::Path::new(&args_path)
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map(|p| p.to_string_lossy().to_string())
    });
    let cwd_abs = match &cwd {
        Some(d) => abs_path(d),
        None => std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from(".")),
    };
    // 循环化的计数器/步长是前端产出的极小常量：汇编器按 CWD 找 `.bin`，
    // 所以必须写进 CWD（与常量同处）。
    if let Err(e) = write_extra_bins(&gen, &cwd_abs) {
        eprintln!("错误[E000]：写不到循环控制常量（{}）：{e}", cwd_abs.display());
        return ExitCode::from(1);
    }

    let inv = sparrow::asmdriver::AsmInvocation {
        exe: asm.into(),
        // 绝对路径：子进程 CWD 是常量目录，相对路径会以它为基准
        vml: abs_path(&vml_path),
        vmb: abs_path(&vmb_path),
        consts: gen.consts.iter().map(|c| c.arg()).collect(),
        cwd: Some(cwd_abs),
    };

    // 4) fork + 原样转发
    let outcome = match sparrow::asmdriver::run(&inv) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("错误[E200]：fork 汇编器失败：{e}");
            eprintln!("  命令：{}", inv.command_line());
            return ExitCode::from(1);
        }
    };
    sparrow::asmdriver::forward(&outcome);

    let size_of = |p: &str| {
        std::fs::metadata(p)
            .map(|m| format!("{} B", m.len()))
            .unwrap_or_else(|_| "缺失".to_string())
    };

    if !outcome.ok() {
        eprintln!();
        eprintln!(
            "错误[E201]：汇编器失败（退出码 {}），退出码原样转发给调用方",
            outcome.code
        );
        eprintln!("  命令：{}", inv.command_line());
        eprintln!("  .vml 中间产物已保留：{vml_path}");
        eprintln!("  权重清单已保留：{args_path}");
        // 关键：用 i32 退出，`ExitCode::from(u8)` 会把 -1073740791 截成 153
        let _ = std::io::Write::flush(&mut std::io::stdout());
        std::process::exit(outcome.code);
    }
    if !sparrow::asmdriver::vmb_written(&inv) {
        eprintln!(
            "错误[E202]：汇编器报成功但没产出 .vmb（{}）—— 拒绝把矛盾态当成功",
            vmb_path
        );
        return ExitCode::from(1);
    }

    // 5) P0-2 对账：循环化的步长必须与 .vmb 里的真实布局一致。
    //
    // 为什么放在**汇编之后**：步长是汇编器的布局函数。前端算的只是一份
    // 预测（`--backend vml` 没有汇编器可读），只有 `.vmb` 是权威。
    // 对不上就删掉产物并报错 —— 绝不留下一个步长未知的 `.vmb`：
    // 步长错了不会崩，只会从第 2 层起静默读错权重。
    if !gen.loop_groups.is_empty() {
        match verify_after_assembly(&gen, &vmb_path) {
            Ok(n) => {
                let s: Vec<String> = gen
                    .loop_groups
                    .iter()
                    .map(|g| format!("{}×{} 步长 {} B", g.cprefix, g.iters, g.stride))
                    .collect();
                println!("[loop] {} 组已按 .vmb 常量表对账：{}", n, s.join("；"));
            }
            Err(why) => {
                let _ = std::fs::remove_file(&vmb_path);
                eprintln!("错误[E210]：循环步长对账失败：{why}");
                eprintln!("  已删除不可信的产物：{vmb_path}");
                eprintln!("  命令：{}", inv.command_line());
                return ExitCode::from(1);
            }
        }
    }

    println!("[spar] {spar_path} ({})", size_of(&spar_path));
    println!("[vml ] {vml_path} ({})", size_of(&vml_path));
    println!("[args] {args_path} ({} 条常量)", gen.consts.len());
    println!("[vmb ] {vmb_path} ({})", size_of(&vmb_path));
    print_artifact_size(&vmb_path);
    ExitCode::SUCCESS
}

/// 打印 `.vmb` 的规模（槽位复用要拿它证明"激活区真的小了"）。
///
/// 为什么从 `.vmb` 读而不是从生成器自己的账本里算：**产物是唯一真相**。
/// 生成器以为省了多少不算数，汇编器排出来的激活区才算。
fn print_artifact_size(vmb: &str) {
    let Ok(bytes) = std::fs::read(vmb) else { return };
    match sparrow::vmbmeta::read_meta(&bytes) {
        Ok(m) => println!(
            "[slots] {} 个槽位；激活区 {} B；常量 {} 条；文件 {} B",
            m.n_slot,
            m.act_bytes,
            m.n_const,
            bytes.len()
        ),
        Err(e) => eprintln!("警告[E203]：读不出 `{vmb}` 的 META：{e}"),
    }
}

/// 写出前端自己产出的极小常量（循环计数器/步长）。目录必须 = 汇编器的 CWD。
fn write_extra_bins(gen: &sparrow::vmlgen::Generated, dir: &std::path::Path) -> std::io::Result<()> {
    if gen.extra_bins.is_empty() {
        return Ok(());
    }
    std::fs::create_dir_all(dir)?;
    for (name, bytes) in &gen.extra_bins {
        std::fs::write(dir.join(name), bytes)?;
    }
    Ok(())
}

/// 读 `.vmb` 的常量表，核对循环步长。
fn verify_after_assembly(
    gen: &sparrow::vmlgen::Generated,
    vmb: &str,
) -> std::result::Result<u64, String> {
    let bytes = std::fs::read(vmb).map_err(|e| format!("读不到 `{vmb}`：{e}"))?;
    let ents = sparrow::vmbmeta::read_const_table(&bytes)?;
    sparrow::vmlgen::verify_loop_groups(gen, &ents)
}

/// V0.1.4 路径 2：`.spar → Rust AOT 源`（一个可以 `cargo run` 的 crate）。
///
/// 为什么必须给 `--vmlm-root`：生成的代码 `use vmlm_ops::kernels::*`，
/// 用**同一套内核**才能与 `.vmb` 后端逐位一致（朴素循环做不到，见 libgen 的模块注释）。
fn lib_backend(
    src: &SourceFile,
    in_path: &str,
    out: Option<String>,
    consts_dir: Option<String>,
    vmlm_root: Option<String>,
    opts: sparrow::vmlgen::GenOptions,
    bench_iters: u32,
) -> ExitCode {
    let (_, lowered) = match sparrow::compile_spar(src) {
        Ok(v) => v,
        Err(d) => return report(&d.render(src)),
    };
    let gen = match sparrow::vmlgen::generate_with(&lowered, in_path, opts) {
        Ok(g) => g,
        Err(d) => return report(&d.render(src)),
    };
    let Some(root) = vmlm_root else {
        eprintln!("错误[E109]：`--backend lib` 需要 `--vmlm-root <vmlm 仓库根>`");
        eprintln!("  生成的 Rust 会 `use vmlm_ops::kernels::*` —— 只有用**同一套内核**");
        eprintln!("  才能与 `.vmb` 后端逐位一致（朴素循环差 1 ULP，过不了'逐字符'判据）。");
        return ExitCode::from(2);
    };
    if !std::path::Path::new(&root).join("ops").join("Cargo.toml").is_file() {
        eprintln!("错误[E109]：`{root}/ops/Cargo.toml` 不存在 —— `--vmlm-root` 要给 vmlm 仓库根");
        return ExitCode::from(2);
    }
    let Some(dir) = consts_dir else {
        eprintln!("错误[E109]：`--backend lib` 需要 `--consts-dir <权重目录>`");
        eprintln!("  （生成的代码用 include_bytes! 内嵌它们，所以要绝对路径）");
        return ExitCode::from(2);
    };
    let dir_abs = abs_path(&dir).to_string_lossy().to_string();

    let crate_dir = out.unwrap_or_else(|| {
        let stem = std::path::Path::new(in_path)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "model".into());
        format!("{stem}_lib")
    });
    let crate_name = std::path::Path::new(&crate_dir)
        .file_name()
        .map(|s| s.to_string_lossy().to_string().replace(['-', '.'], "_"))
        .unwrap_or_else(|| "sparrow_model".into());

    let body = match sparrow::libgen::generate_rust_with(
        &gen,
        &dir_abs,
        sparrow::libgen::LibOptions { bench_iters },
    ) {
        Ok(b) => b,
        Err(d) => return report(&d.render(src)),
    };
    let src_dir = std::path::Path::new(&crate_dir).join("src");
    if let Err(e) = std::fs::create_dir_all(&src_dir) {
        eprintln!("错误[E000]：建不了 `{}`：{e}", src_dir.display());
        return ExitCode::from(1);
    }
    let files = [
        (std::path::Path::new(&crate_dir).join("Cargo.toml"), sparrow::libgen::cargo_toml(&abs_path(&root).to_string_lossy().replace('\\', "/"), &crate_name)),
        (src_dir.join("main.rs"), body),
    ];
    for (p, text) in &files {
        if let Err(e) = std::fs::write(p, text.as_bytes()) {
            eprintln!("错误[E000]：写不到 `{}`：{e}", p.display());
            return ExitCode::from(1);
        }
        println!("[lib] {} ({} B)", p.display(), text.len());
    }
    println!();
    println!("跑它：");
    println!("  cargo run --release --manifest-path {crate_dir}/Cargo.toml");
    ExitCode::SUCCESS
}

fn fail_usage(msg: &str) -> ExitCode {
    eprintln!("用法错误：{msg}");
    print_help();
    ExitCode::from(2)
}

fn print_help() {
    println!("sparrow 0.1.0 —— 神经网络架构描述语言（文本前端）");
    println!();
    println!("用法：sparrow <file.sparrow> [选项]");
    println!();
    println!("四层后缀：.sparrow 源码 → .spar 架构 IR → .vml opcode → .vmb 字节码");
    println!();
    println!("选项：");
    println!("  -o, --output <PATH>   把结果写到文件（默认 stdout）");
    println!("      --emit-spar       输出架构级 .spar（默认）");
    println!("      --backend vml     输出 opcode 级 .vml（V0.1.1 后端）");
    println!("      --args <PATH>     随 --backend vml 写出权重清单 name:file:rows:cols");
    println!("      --loop-repeat     `重复 N` 走 LOOP_CNT + LAYERB（O(1) 指令流）");
    println!("      --no-loop-repeat  关掉循环化（默认；= V0.1.1 的展开语义）");
    println!("      --slot-reuse      激活槽按生命周期复用（激活区更小）");
    println!("      --weight-dtype D  权重精度 f32（默认）/ bf16（线性权重与嵌入表减半）");
    println!("      --backend lib     路径 2 后端：生成 Rust AOT 源码（配 --lang rust --vmlm-root）");
    println!("      --lang rust       --backend lib 的目标语言（只有 rust）");
    println!("      --vmlm-root <D>   vmlm 仓库根（生成代码依赖其 `ops/` crate —— 同源算术）");
    println!("      --bench-iters N   --backend lib 时给生成的程序加计时循环（预热 1 遍 + 计时 N 遍，");
    println!("                        打印与引擎 bench harness 同格式的 `avg us/iter = X`；默认 0 = 不加）");
    println!("      --asm <PATH>      一条命令跑全链：.sparrow → .spar → .vml → fork 汇编器 → .vmb");
    println!("      --consts-dir <D>  --asm 时权重 .bin 所在目录（默认 = 清单所在目录）");
    println!("      --emit-ast-json   输出 AST JSON");
    println!("      --emit-tokens     输出 token 流");
    println!("      --sha256          同时打印输出的 sha256");
    println!("  -h, --help            显示帮助");
    println!();
    println!("输入 `-` 表示从 stdin 读取。");
}
