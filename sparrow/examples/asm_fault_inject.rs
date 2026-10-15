//! 故障注入用的「假汇编器」—— 只给 **can-fail** 测试用（Sparrow V0.1.2 P0-2）。
//!
//! 为什么需要它：P0-2 的 `E210`（循环步长对账）只有在**汇编器产出的常量池布局
//! 与前端预测不一致**时才会触发，而真实的 `vmlm-asm.exe` 永远产出一致的布局 ——
//! 任务书又禁止改汇编器。于是用一个"假装成功、实则篡改"的假汇编器来制造故障：
//! 前端 fork 的是它，写出的 `.vmb` 是坏的，**对账必须发现**。
//!
//! 用法（前端调用约定完全一致）：
//!
//! ```text
//! asm_fault_inject <in.vml> <out.vmb> [name:file:rows:cols ...]
//! ```
//!
//! 环境变量：
//!
//! * `SPARROW_FAULT_SRC=<good.vmb>` —— 以这份**真汇编器产出的** `.vmb` 为底改，
//!   这样故障之外的一切都是真的（否则测的就不是对账而是别的东西）。
//! * `SPARROW_FAULT`：
//!   * `shift_const:<i>:<delta>` —— 把第 i 条常量的池偏移量 +delta（破坏步长恒定）；
//!   * `drop_const:<n>` —— 谎报常量条数为 n（破坏"池序 = 清单序"）。
//!
//! 退出码 0（**假装成功**）：故障必须由前端的对账抓出来，不是靠汇编器报错。

use std::io::Write;

fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

fn put_u32(b: &mut [u8], o: usize, v: u32) {
    b[o..o + 4].copy_from_slice(&v.to_le_bytes());
}

fn die(msg: &str) -> ! {
    // 打印到 stderr 并**非零**退出 —— 调用方会原样转发
    eprintln!("asm_fault_inject: {msg}");
    std::process::exit(2);
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 {
        die("用法: asm_fault_inject <in.vml> <out.vmb> [consts...]");
    }
    let out_path = &a[2];
    let src = std::env::var("SPARROW_FAULT_SRC").unwrap_or_else(|_| die("缺 SPARROW_FAULT_SRC"));
    let fault = std::env::var("SPARROW_FAULT").unwrap_or_else(|_| die("缺 SPARROW_FAULT"));
    let mut b = std::fs::read(&src).unwrap_or_else(|e| die(&format!("读不到 {src}: {e}")));
    if b.len() < 32 {
        die("底本 .vmb 太短");
    }
    let meta_off = u32_at(&b, 24) as usize;
    let meta_len = u32_at(&b, 28) as usize;
    if meta_off + meta_len > b.len() || meta_len < 28 {
        die("底本 .vmb 的 META 区间越界");
    }
    let parts: Vec<&str> = fault.split(':').collect();
    match parts.first().copied() {
        Some("shift_const") => {
            let i: usize = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or_else(|| die("缺下标"));
            let d: i64 = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or_else(|| die("缺增量"));
            let n_const = u32_at(&b, meta_off) as usize;
            if i >= n_const {
                die(&format!("下标 {i} 越界（n_const={n_const}）"));
            }
            let ent = meta_off + 28 + i * 24;
            let off = u32_at(&b, ent) as i64;
            put_u32(&mut b, ent, (off + d) as u32);
            let _ = writeln!(std::io::stdout(), "FAKE: 第 {i} 条常量偏移 {off} → {}", off + d);
        }
        Some("drop_const") => {
            let n: u32 = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or_else(|| die("缺条数"));
            put_u32(&mut b, meta_off, n);
            let _ = writeln!(std::io::stdout(), "FAKE: n_const 谎报为 {n}");
        }
        other => die(&format!("未知故障 {other:?}")),
    }
    std::fs::write(out_path, &b).unwrap_or_else(|e| die(&format!("写不到 {out_path}: {e}")));
    // 假装汇编成功：产物存在，退出码 0 —— 就等前端对账把它拆穿
    let _ = writeln!(std::io::stdout(), "OK: (fake) {} 字节 → {out_path}", b.len());
}
