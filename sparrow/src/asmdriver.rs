//! Sparrow V0.1.2 P0-1 —— 前端 fork 汇编器（`--asm <path>`）
//!
//! 边界（任务书写死，逐条落地）：
//!
//! 1. **不内联汇编器逻辑** —— 这里只 `Command::new(exe)`，汇编器源码一行不动。
//! 2. **`.vml` 仍是必经中间层** —— 调用方先把 `.vml` 落盘，再把**路径**交给汇编器；
//!    本模块不生成 `.vml`，也不接受"跳过 `.vml` 直接出 `.vmb`"的调用方式。
//! 3. **报错原样转发** —— stdout/stderr 按**字节**透传（不做 UTF-8 转换、不加前缀），
//!    退出码按 `i32` 原样返回给进程。
//!
//! 实测的汇编器契约（`vmlm-asm.exe`，release、`panic=abort`）：
//!
//! | 情况 | 退出码 | 输出 | `.vmb` |
//! |---|---|---|---|
//! | 成功 | `0` | stdout: `OK: N 条指令…`；stderr: `⚠️ 警告…` | 产出 |
//! | 未知指令 / 未实现指令 / 读不到权重 | `-1073740791`（`0xC0000409`，abort） | stderr: `thread 'main' panicked at …` | **不产出** |
//!
//! 注意 `-1073740791` 装不进 `u8`：转发必须走 `std::process::exit(i32)`，
//! 不能用 `ExitCode::from(u8)`（会把码截断成 153，脚本就再也分辨不出是哪种失败）。

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

/// 一次汇编调用的完整描述（也是报错时要打给人看的命令行）。
#[derive(Clone, Debug)]
pub struct AsmInvocation {
    pub exe: PathBuf,
    /// 必经中间层：opcode `.vml`（必须已经落盘）
    pub vml: PathBuf,
    pub vmb: PathBuf,
    /// `name:file.bin:rows:cols` 形式的权重清单
    pub consts: Vec<String>,
    /// 汇编器的 CWD —— 清单里的 `.bin` 是相对路径，必须在这里能找到
    pub cwd: Option<PathBuf>,
}

impl AsmInvocation {
    pub fn command_line(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("{} {} {}", self.exe.display(), self.vml.display(), self.vmb.display()));
        for c in &self.consts {
            s.push(' ');
            s.push_str(c);
        }
        if let Some(d) = &self.cwd {
            s.push_str(&format!("   [cwd={}]", d.display()));
        }
        s
    }
}

#[derive(Clone, Debug)]
pub struct AsmOutcome {
    /// 汇编器的退出码，已按 i32 解释（abort 是负数）
    pub code: i32,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}

impl AsmOutcome {
    pub fn ok(&self) -> bool {
        self.code == 0
    }

    pub fn stdout_text(&self) -> String {
        String::from_utf8_lossy(&self.stdout).to_string()
    }

    pub fn stderr_text(&self) -> String {
        String::from_utf8_lossy(&self.stderr).to_string()
    }
}

/// 退出码还原成 i32。
///
/// Windows 上 `ExitStatus::code()` 直接给出 `dwExitCode`，abort 的
/// `0xC0000409` 于是以 `-1073740791`（i32）出现 —— 与 PowerShell 的
/// `$LASTEXITCODE` 观察值一致（实测见 `tests/asm_fork.rs`）。
/// `None` 只在进程仍在运行时出现，这里不该发生；真发生了就退化为 1，
/// **不假装成功**。
fn status_code(st: &std::process::ExitStatus) -> i32 {
    st.code().unwrap_or(1)
}

/// fork 汇编器并收集输出。**不打印任何东西**（转发交给 [`forward`]，好让调用方决定顺序）。
pub fn run(inv: &AsmInvocation) -> std::io::Result<AsmOutcome> {
    let mut cmd = Command::new(&inv.exe);
    cmd.arg(&inv.vml).arg(&inv.vmb);
    for c in &inv.consts {
        cmd.arg(c);
    }
    if let Some(d) = &inv.cwd {
        cmd.current_dir(d);
    }
    let out = cmd.output()?;
    Ok(AsmOutcome {
        code: status_code(&out.status),
        stdout: out.stdout,
        stderr: out.stderr,
    })
}

/// 原样转发：stdout → 本进程 stdout，stderr → 本进程 stderr。
///
/// 按**字节**写，不做 UTF-8 校验/替换 —— 汇编器的中文提示是 UTF-8，
/// 但它同时会打印槽位名（用户可控），`from_utf8_lossy` 会在非法序列处插 `�`，
/// 那就不是"原样"了。
pub fn forward(o: &AsmOutcome) {
    let so = std::io::stdout();
    let mut so = so.lock();
    let _ = so.write_all(&o.stdout);
    let _ = so.flush();
    let se = std::io::stderr();
    let mut se = se.lock();
    let _ = se.write_all(&o.stderr);
    let _ = se.flush();
}

/// `.vmb` 是否真的产出了 —— 退出码 0 但没文件，是必须拦下来的矛盾态。
pub fn vmb_written(inv: &AsmInvocation) -> bool {
    Path::new(&inv.vmb).is_file()
}
