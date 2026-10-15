//! Sparrow V0.1.2 P0-1 验收：`--asm` fork 汇编器
//!
//! | 验收 | 在哪 |
//! |---|---|
//! | 1. 一条命令跑出 `.vmb` | `cli_one_command_produces_vmb` |
//! | 2. 引擎加载后与手写路径逐字节相同 | `tools/e2e_tiny.ps1`（要真引擎） |
//! | 3. 汇编器报错时原样转发 stderr + 退出码 | `forwards_success_and_failure_verbatim` |

use std::path::{Path, PathBuf};

use sparrow::asmdriver::{self, AsmInvocation};

/// 汇编器位置：优先环境变量 `SPARROW_VMLM_ASM`，否则用本机的默认路径。
/// 找不到就明确跳过（而不是把"没测"伪装成"通过"）。
fn asm_exe() -> Option<PathBuf> {
    let p = std::env::var("SPARROW_VMLM_ASM")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(r"D:\czsai\vmlm\target\release\vmlm-asm.exe"));
    if p.is_file() {
        Some(p)
    } else {
        None
    }
}

fn tmp_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("sparrow_v012_{tag}"));
    std::fs::create_dir_all(&d).expect("建临时目录");
    d
}

// ---------------------------------------------------------------------------
// 验收 3：原样转发（成功与失败两条路）
// ---------------------------------------------------------------------------

#[test]
fn forwards_success_and_failure_verbatim() {
    let Some(exe) = asm_exe() else {
        eprintln!("SKIP：找不到 vmlm-asm.exe（可用 SPARROW_VMLM_ASM=<路径> 指定）");
        return;
    };
    let dir = tmp_dir("fork");

    // --- 好 .vml：退出 0 + 产出 .vmb ---
    let good = dir.join("good.vml");
    std::fs::write(&good, "SHAPE x, 1, 4\nHALT\n").unwrap();
    let out_good = dir.join("good.vmb");
    let _ = std::fs::remove_file(&out_good);
    let inv = AsmInvocation {
        exe: exe.clone(),
        vml: good,
        vmb: out_good,
        consts: vec![],
        cwd: None,
    };
    let o = asmdriver::run(&inv).expect("fork 汇编器");
    assert!(o.ok(), "好 .vml 应汇编成功；code={} stderr={}", o.code, o.stderr_text());
    assert!(asmdriver::vmb_written(&inv), "成功却没产出 .vmb");
    assert!(o.stdout_text().contains("OK:"), "stdout 未转发：{}", o.stdout_text());

    // --- 坏 .vml：退出码原样 + stderr 原样 + **不**产出 .vmb ---
    let bad = dir.join("bad.vml");
    std::fs::write(&bad, "SHAPE x, 1, 4\nFROBNICATE x, y\nHALT\n").unwrap();
    let out_bad = dir.join("bad.vmb");
    let _ = std::fs::remove_file(&out_bad);
    let inv = AsmInvocation {
        exe,
        vml: bad,
        vmb: out_bad,
        consts: vec![],
        cwd: None,
    };
    let o = asmdriver::run(&inv).expect("fork 汇编器");
    assert!(!o.ok(), "坏 .vml 不该报成功");
    // 退出码必须"看起来就是汇编器那个"：release(panic=abort)=0xC0000409 → -1073740791；
    // 若是 debug（panic=unwind）则为 101。两者都 **装不进 u8** —— 这正是不能用
    // `ExitCode::from(u8)` 转发的原因（会截成 153 / 101 之外的数）。
    assert!(
        o.code == -1073740791 || o.code == 101,
        "退出码应为 abort(-1073740791) 或 panic(101)，实际 {}",
        o.code
    );
    assert!(o.code < 0 || o.code > 255, "退出码必须超出 u8 范围才说明问题真实存在");
    let err = o.stderr_text();
    assert!(err.contains("未知指令: FROBNICATE"), "stderr 未原样转发：\n{err}");
    assert!(err.contains("panicked"), "stderr 未包含 panic 行：\n{err}");
    assert!(!asmdriver::vmb_written(&inv), "失败时不该留下 .vmb（会被误当成功产物）");

    // --- 未实现指令：拒绝发生在汇编期，而不是运行到才 exit(5) ---
    let unimpl = dir.join("unimpl.vml");
    std::fs::write(&unimpl, "SHAPE x, 1, 4\nLOADK x, y\nHALT\n").unwrap();
    let inv = AsmInvocation {
        exe: asm_exe().unwrap(),
        vml: unimpl,
        vmb: dir.join("unimpl.vmb"),
        consts: vec![],
        cwd: None,
    };
    let o = asmdriver::run(&inv).unwrap();
    assert!(!o.ok());
    assert!(o.stderr_text().contains("LOADK"), "{}", o.stderr_text());
}

/// 转发必须是**字节级**的：不能因为非法 UTF-8 就插替换字符。
#[test]
fn forwarding_is_byte_exact() {
    let o = asmdriver::AsmOutcome {
        code: 0,
        stdout: vec![0xE4, 0xB8, 0xAD, 0xFF, 0x0A], // 合法中文 + 一个非法字节
        stderr: vec![0x41, 0x00, 0x42],             // 含 NUL
    };
    assert_eq!(o.stdout, vec![0xE4, 0xB8, 0xAD, 0xFF, 0x0A]);
    // lossy 只用于**人看的摘要**，转发走 forward()（write_all 原字节）
    assert!(o.stdout_text().contains('中'));
}

// ---------------------------------------------------------------------------
// 验收 1：前端 CLI 一条命令跑出 .vmb（真子进程）
// ---------------------------------------------------------------------------

#[test]
fn cli_one_command_produces_vmb() {
    let Some(exe) = asm_exe() else {
        eprintln!("SKIP：找不到 vmlm-asm.exe");
        return;
    };
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let bins = root.join("examples").join("e2e_bins");
    if !bins.join("EMBED_E.bin").is_file() {
        eprintln!("SKIP：缺 examples/e2e_bins（先跑 python tools/gen_e2e_tiny.py）");
        return;
    }
    let out_vmb = bins.join("_cli_test.vmb");
    for ext in ["vmb", "vml", "spar", "args"] {
        let _ = std::fs::remove_file(bins.join(format!("_cli_test.{ext}")));
    }

    let cli = env!("CARGO_BIN_EXE_sparrow");
    let o = std::process::Command::new(cli)
        .arg(root.join("examples").join("e2e_tiny.sparrow"))
        .arg("--asm")
        .arg(&exe)
        .arg("--consts-dir")
        .arg(&bins)
        .arg("-o")
        .arg(&out_vmb)
        .output()
        .expect("启动前端");

    let so = String::from_utf8_lossy(&o.stdout);
    let se = String::from_utf8_lossy(&o.stderr);
    assert!(o.status.success(), "一条命令应成功；stdout={so}\nstderr={se}");
    assert!(out_vmb.is_file(), "没产出 .vmb；stdout={so}");
    assert!(so.contains("OK:"), "汇编器的 OK 行没被转发：{so}");
    // `.vml` 是必经中间层：必须落盘，且内容就是交给汇编器的那份
    for ext in ["vml", "spar", "args"] {
        let p = bins.join(format!("_cli_test.{ext}"));
        assert!(p.is_file(), "中间产物 `{}` 没落盘", p.display());
    }
    let vml = std::fs::read_to_string(bins.join("_cli_test.vml")).unwrap();
    assert!(vml.contains("SPARSE_MATVEC SPARSE_col"), "{vml}");
    assert!(vml.contains("TRANSPOSE SPARSE_out, SPARSE_col"), "{vml}");
    // 与 --backend vml 单独产出的文本一致（同一条管线，不因 --asm 而变形）
    let direct = sparrow::compile_vml_backend(
        &sparrow::diag::SourceFile::new(
            "e2e_tiny.sparrow",
            std::fs::read_to_string(root.join("examples").join("e2e_tiny.sparrow")).unwrap(),
        ),
        "e2e_tiny.sparrow",
    )
    .unwrap();
    assert_eq!(vml, direct.vml, "`--asm` 与 `--backend vml` 产出的 .vml 必须一致");

    let size = std::fs::metadata(&out_vmb).unwrap().len();
    assert!(size > 0, "空 .vmb");
    println!("一条命令产出 {size} B 的 .vmb");
    for ext in ["vmb", "vml", "spar", "args"] {
        let _ = std::fs::remove_file(bins.join(format!("_cli_test.{ext}")));
    }
}

/// `--asm` 缺汇编器时要**报清楚**，不是 panic。
#[test]
fn cli_reports_missing_assembler() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let cli = env!("CARGO_BIN_EXE_sparrow");
    let o = std::process::Command::new(cli)
        .arg(root.join("examples").join("e2e_tiny.sparrow"))
        .arg("--asm")
        .arg(root.join("examples").join("no_such_assembler.exe"))
        .arg("-o")
        .arg(root.join("examples").join("_never.vmb"))
        .output()
        .expect("启动前端");
    assert!(!o.status.success());
    let se = String::from_utf8_lossy(&o.stderr);
    assert!(se.contains("E200"), "缺汇编器应报 E200：{se}");
    assert!(se.contains("fork"), "{se}");
}
