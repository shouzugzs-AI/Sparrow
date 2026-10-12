// P8 成员 F：Transformer 训练闭环（前向 + 手写反向 + AdamW）
//
// 职责（std 侧，体积不受限）：
//   1. 组装 train_step.vml（调 vmlm-asm），装载 train_step.vmb
//   2. **过拟合测试**：固定单批次，训练 N 步，打印 loss 曲线
//   3. **梯度检查**：VMLM 解析梯度 vs PyTorch autograd
//      - 判据 Σ|Δ|/(‖num‖+‖ana‖) < 1e-3（契约 §3 / P7-A 方法论）
//      - softmax 相关会被 kernels::exp_f（最大相对误差 6.51e-4）污染，
//        故同时给一份「f64 精确 exp 前向」的 torch 参考做对照，
//        用以区分「实现错」与「exp 近似污染」
//   4. **因果性回归**：训练后 att 上三角必须仍为 0
//   5. 耗时 + CPU 占用（进程 CPU 时间 / 墙钟时间，单线程应 ≈1.0）
//
// 训练逻辑全在 .vml 指令流里 —— 引擎不做自动微分（白皮书 §6.2）。
use std::time::Instant;
use vmlm_core::interp::Vm;

extern "C" {
    fn malloc(n: usize) -> *mut u8;
}

const DIR: &str = "examples/attn/train";

// 维度（与 make_attn_train.py 必须一致）
const T: usize = 4;
const D: usize = 8;
const F: usize = 16;
const V: usize = 8;

// 可训练参数顺序（= 常量注册顺序 = torch 梯度拼接顺序）
const NAMES: [&str; 16] = [
    "Wqkv", "bqkv", "Wproj", "bproj", "W1", "b1", "W2", "b2", "ln1_g", "ln1_b", "ln2_g", "ln2_b",
    "lnf_g", "lnf_b", "E", "pos",
];
// 各参数形状 (rows, cols)，顺序同上
const SHAPES: [(usize, usize); 16] = [
    (3 * D, D), (1, 3 * D), (D, D), (1, D), // Wqkv bqkv Wproj bproj
    (F, D), (1, F), (D, F), (1, D),          // W1 b1 W2 b2
    (1, D), (1, D), (1, D), (1, D), (1, D), (1, D), // ln1_g..lnf_b
    (V, D), (T, D),                          // E pos
];

const THRESH: f64 = 1e-3;

unsafe fn load_vmb(path: &str) -> *mut u8 {
    let data = std::fs::read(path).unwrap_or_else(|_| panic!("读不到 {}", path));
    let n = data.len();
    let p = malloc(n + 64);
    std::ptr::copy_nonoverlapping(data.as_ptr(), p, n);
    for i in n..n + 64 {
        *p.add(i) = 0;
    }
    p
}

unsafe fn slot_ptr(vm: &mut Vm, name: &str) -> *mut f32 {
    let si = vm.find_slot(name);
    vm.slot_mut(si)
}

unsafe fn peek(vm: &mut Vm, name: &str, n: usize) -> Vec<f32> {
    let p = slot_ptr(vm, name);
    (0..n).map(|i| *p.add(i)).collect()
}

unsafe fn poke(vm: &mut Vm, name: &str, vals: &[f32]) {
    let p = slot_ptr(vm, name);
    for (i, v) in vals.iter().enumerate() {
        *p.add(i) = *v;
    }
}

// 用 vmlm-asm 组装 .vml → .vmb（常量清单来自 asm.args）
fn assemble() -> bool {
    let args_file = format!("{}/asm.args", DIR);
    let list = match std::fs::read_to_string(&args_file) {
        Ok(s) => s,
        Err(_) => {
            println!("  找不到 {}（先跑 make_attn_train.py）", args_file);
            return false;
        }
    };
    let asm = "target/release/vmlm-asm.exe";
    if !std::path::Path::new(asm).exists() {
        println!("  找不到 {}（先 cargo build -p vmlm-asm --release）", asm);
        return false;
    }
    let mut cmd = std::process::Command::new(asm);
    cmd.arg(format!("{}/train_step.vml", DIR))
        .arg(format!("{}/train_step.vmb", DIR));
    for line in list.lines() {
        let l = line.trim();
        if l.is_empty() || l.starts_with('#') {
            continue;
        }
        let p: Vec<&str> = l.split(':').collect();
        cmd.arg(format!("{}:{}/{}:{}:{}", p[0], DIR, p[1], p[2], p[3]));
    }
    match cmd.output() {
        Ok(o) => {
            if !o.status.success() {
                println!("  汇编失败:\n{}", String::from_utf8_lossy(&o.stderr));
                return false;
            }
            let head = String::from_utf8_lossy(&o.stdout);
            println!("  汇编: {}", head.lines().next().unwrap_or(""));
            true
        }
        Err(e) => {
            println!("  启动汇编器失败: {}", e);
            false
        }
    }
}

// ---------------------------------------------------------------------------
// 进程 CPU 占用（Windows）：GetProcessTimes
// ---------------------------------------------------------------------------
#[cfg(windows)]
mod cpu {
    #[repr(C)]
    #[derive(Default, Clone, Copy)]
    pub struct FileTime {
        pub low: u32,
        pub high: u32,
    }
    impl FileTime {
        pub fn secs(&self) -> f64 {
            (((self.high as u64) << 32) | self.low as u64) as f64 / 1e7
        }
    }
    extern "system" {
        fn GetCurrentProcess() -> isize;
        fn GetProcessTimes(
            h: isize,
            creation: *mut FileTime,
            exit: *mut FileTime,
            kernel: *mut FileTime,
            user: *mut FileTime,
        ) -> i32;
    }
    pub fn cpu_seconds() -> f64 {
        unsafe {
            let (mut c, mut e, mut k, mut u) = (
                FileTime::default(),
                FileTime::default(),
                FileTime::default(),
                FileTime::default(),
            );
            if GetProcessTimes(GetCurrentProcess(), &mut c, &mut e, &mut k, &mut u) == 0 {
                return 0.0;
            }
            k.secs() + u.secs()
        }
    }
}

#[cfg(windows)]
use cpu::cpu_seconds;

// 非 Windows 退化：返回 0（本机是 Windows，此分支不会被走到）
#[cfg(not(windows))]
fn cpu_seconds() -> f64 {
    0.0
}

// ---------------------------------------------------------------------------
// 读 torch 参考梯度（二进制 + meta）
// ---------------------------------------------------------------------------
fn read_meta(path: &str) -> Option<(f64, Vec<(String, usize)>)> {
    let txt = std::fs::read_to_string(path).ok()?;
    let mut lines = txt.lines();
    let loss: f64 = lines.next()?.trim().parse().ok()?;
    let mut v = Vec::new();
    for l in lines {
        let mut it = l.split_whitespace();
        let n = it.next()?.to_string();
        let c: usize = it.next()?.parse().ok()?;
        v.push((n, c));
    }
    Some((loss, v))
}

fn read_f32(path: &str) -> Vec<f32> {
    let b = std::fs::read(path).unwrap_or_else(|_| panic!("读不到 {}", path));
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

// ---------------------------------------------------------------------------
// 判据：Σ|Δ| / (‖num‖ + ‖ana‖)      （契约 §3 / §5：不用逐元素相对误差）
// ---------------------------------------------------------------------------
fn criterion(num: &[f32], ana: &[f32]) -> (f64, f64, f64) {
    let mut sum_d = 0.0f64;
    let mut sum_abs = 0.0f64;
    let mut worst = 0.0f64;
    for i in 0..num.len() {
        let a = num[i] as f64;
        let b = ana[i] as f64;
        let d = (a - b).abs();
        sum_d += d;
        sum_abs += a.abs() + b.abs();
        let den = a.abs() + b.abs();
        if den > 1e-30 {
            let r = d / den;
            if r > worst {
                worst = r;
            }
        }
    }
    let g = if sum_abs == 0.0 { 0.0 } else { sum_d / sum_abs };
    (g, sum_d, worst)
}

// 逐参数切分 pad 到参数元素数（跳过多余的 hparams 等槽）
fn param_sizes() -> Vec<usize> {
    SHAPES.iter().map(|&(r, c)| r * c).collect()
}

// 逐位置交叉熵的均值（torch cross_entropy 同口径）
fn mean_ce(logits: &[f32], tgt: &[f32]) -> f32 {
    let t = tgt.len();
    let mut s = 0.0f64;
    for i in 0..t {
        let row = &logits[i * V..(i + 1) * V];
        let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut se = 0.0f64;
        for &x in row {
            se += ((x - mx) as f64).exp();
        }
        let lse = mx as f64 + se.ln();
        s += lse - row[tgt[i] as usize] as f64;
    }
    (s / t as f64) as f32
}

// 前向指令条数：train_step.vml 里 `GEMM logits, xf, E` 之后紧跟 backward 段。
// 由 gen_vml.py 的顺序保证 —— 前向段以 logits 那条 GEMM 结束。
// 这里把它找出来：读 .vml 数到 `GEMM        logits` 那一行为止。
fn forward_ins_count() -> usize {
    let src = std::fs::read_to_string(format!("{}/train_step.vml", DIR)).unwrap_or_default();
    let mut n = 0usize;
    for raw in src.lines() {
        let line = match raw.find(';') {
            Some(i) => &raw[..i],
            None => raw,
        };
        let line = line.trim();
        if line.is_empty() || line.starts_with('.') || line.ends_with(':') {
            continue;
        }
        n += 1;
        let up = line.to_ascii_uppercase();
        if up.starts_with("GEMM") && up.contains("LOGITS") {
            return n; // 含这条
        }
    }
    n
}

const N_PARAM: usize = 24 * 8 + 24 + 8 * 8 + 8 + 16 * 8 + 16 + 8 * 16 + 8
    + 8 * 6
    + 8 * 8
    + 4 * 8; // = Wqkv+bqkv+Wproj+bproj + W1+b1+W2+b2 + 6×LN + E + pos

// ---------------------------------------------------------------------------
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let steps: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(400);
    let skip_asm = args.iter().any(|a| a == "--no-asm");

    println!("=== P8-F：Transformer 训练闭环（前向 + 手写反向 + AdamW）===");
    println!("配置: T={} D={} NH=1 HD={} F={} V={}  参数元素={}", T, D, D, F, V, N_PARAM);
    unsafe { vmlm_core::interp::init_cpu() };

    if !skip_asm && !assemble() {
        std::process::exit(1);
    }

    let mut all_ok = true;
    unsafe {
        let buf = load_vmb(&format!("{}/train_step.vmb", DIR));
        let mut vm = Vm::load(buf);

        // ---------------- 0. 前向回归（vs Python 参考） ----------------
        // 用一个**独立**的 Vm 实例跑（权重同一份 .vmb 初值），这样既不污染
        // 训练实例的权重，也不需要把 lr 置 0（置 0 会让 AdamW 的 sqrt(v/c2)
        // 变成 0/0 = NaN，把权重毁掉 —— 这是踩过的坑）。
        // 这个实例同时用来取「初始权重处的解析梯度」做梯度检查。
        let fbuf = load_vmb(&format!("{}/train_step.vmb", DIR));
        let mut fvm = Vm::load(fbuf);
        // 前向回归必须在 **AdamW 更新权重之前** 读 logits。
        // 引擎从 ip=0 跑完整步（含 ADAMW E），所以这里用 n_ins 截断：
        // 只跑「到 logits 那条指令为止」的前缀，读到的才是干净的前向值。
        // （这纯粹是驱动侧的调试手段，不改引擎、不改 .vml。）
        let full_n = fvm.n_ins;
        let fwd_n = forward_ins_count();
        fvm.n_ins = fwd_n;
        fvm.run();
        let lg = peek(&mut fvm, "logits", T * V);
        let att = peek(&mut fvm, "att", T * T);
        // 恢复完整步长并**再跑一次**：这时才真正执行反向，把解析梯度写进梯度区。
        // （梯度是在 AdamW 之前算的，所以即使权重随后被更新一步，
        //   梯度区里的值仍是「初始权重处」的解析梯度。）
        fvm.n_ins = full_n;
        fvm.run();

        let ref_txt = std::fs::read_to_string(format!("{}/ref_logits.txt", DIR)).unwrap();
        let ref_lg: Vec<f32> = ref_txt
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        let mut maxerr = 0.0f32;
        for i in 0..T * V {
            let d = (lg[i] - ref_lg[i]).abs();
            if d > maxerr {
                maxerr = d;
            }
        }
        println!("\n[0] 前向回归 vs Python 参考：最大绝对误差 = {:.3e}", maxerr);
        println!(
            "    logits[0] = [{}]",
            lg[..V]
                .iter()
                .map(|x| format!("{:>9.5}", x))
                .collect::<Vec<_>>()
                .join(",")
        );
        println!(
            "    参考[0]  = [{}]",
            ref_lg[..V]
                .iter()
                .map(|x| format!("{:>9.5}", x))
                .collect::<Vec<_>>()
                .join(",")
        );
        println!("    att 行 0 = {:?}", &att[..T]);
        all_ok &= maxerr < 1e-4;

        // 诊断：把关键中间槽 dump 出来（--dump）
        if args.iter().any(|a| a == "--dump") {
            let dump = |v: &mut Vm, name: &str, r: usize, c: usize| {
                let p = slot_ptr(v, name);
                let vals: Vec<f32> = (0..r * c).map(|i| *p.add(i)).collect();
                let mut s = format!("    {:<8} [{}x{}] ", name, r, c);
                for i in 0..r * c {
                    s.push_str(&format!("{:>8.4} ", vals[i]));
                    if (i + 1) % c == 0 && i + 1 < r * c {
                        s.push_str("| ");
                    }
                }
                println!("{}", s);
            };
            println!("\n    --- dump (fvm) ---");
            for r in 0..T {
                dump(&mut fvm, &format!("e{}", r), 1, D);
                dump(&mut fvm, &format!("e{}t", r), D, 1);
            }
            dump(&mut fvm, "embT", D, T);
            dump(&mut fvm, "emb", T, D);
            dump(&mut fvm, "x0", T, D);
            dump(&mut fvm, "xd", T, D);
            dump(&mut fvm, "xf", T, D);
            dump(&mut fvm, "logits", T, V);
            dump(&mut fvm, "dlogits", T, V);
            dump(&mut fvm, "dxf", T, D);
            dump(&mut fvm, "dx", T, D);
            dump(&mut fvm, "dy", T, D);
            dump(&mut fvm, "datt", T, T);
            dump(&mut fvm, "dS", T, T);
            dump(&mut fvm, "dvc", T, D);
            dump(&mut fvm, "dq", T, D);
            dump(&mut fvm, "dk", T, D);
            dump(&mut fvm, "dqkv", T, 3 * D);
            dump(&mut fvm, "dxd", T, D);
            dump(&mut fvm, "dx1", T, D);
            dump(&mut fvm, "dxt", D, T);
            for r in 0..T {
                dump(&mut fvm, &format!("dxtc{}", r), D, 1);
                dump(&mut fvm, &format!("dxr{}", r), 1, D);
            }
            dump(&mut fvm, "demb", V, D);
            dump(&mut fvm, "dpos", T, D);
            // 用 8 位小数核对 logits == xf @ E^T
            let ep = slot_ptr(&mut fvm, "E");
            let xfp = slot_ptr(&mut fvm, "xf");
            let lp = slot_ptr(&mut fvm, "logits");
            println!("    --- E as VM sees it ---");
            for i in 0..V {
                let mut s = String::from("      ");
                for j in 0..D {
                    s.push_str(&format!("{:>10.6} ", *ep.add(i * D + j)));
                }
                println!("{}", s);
            }
            let disk = read_f32(&format!("{}/E.bin", DIR));
            println!("    --- E on disk ---");
            for i in 0..V {
                let mut s = String::from("      ");
                for j in 0..D {
                    s.push_str(&format!("{:>10.6} ", disk[i * D + j]));
                }
                println!("{}", s);
            }
            println!("    --- logits vs xf·E^T (8dp) ---");
            for i in 0..T {
                for j in 0..V {
                    let mut s = 0.0f32;
                    for p in 0..D {
                        s += *xfp.add(i * D + p) * *ep.add(j * D + p);
                    }
                    println!(
                        "      [{},{}] logits={:>12.8}  xf·E^T={:>12.8}  diff={:>10.3e}",
                        i,
                        j,
                        *lp.add(i * V + j),
                        s,
                        (*lp.add(i * V + j) - s).abs()
                    );
                }
            }
        }

        // ---------------- 1. 梯度检查 vs PyTorch autograd ----------------
        // fvm 跑完一个完整训练步后，梯度槽里就是「初始权重处的解析梯度」
        //（该实例只跑这一次，权重未被更新过 —— lr 在 .vml 里由 hparams 给出，
        //  所以严格说权重被更新了一步；但梯度是在更新**之前**算出来的，
        //  这正是我们要的解析梯度）。
        println!("\n[1] 梯度检查：VMLM 解析梯度 vs PyTorch autograd");
        println!("    判据 Σ|Δ|/(‖num‖+‖ana‖) < {:e}（不用逐元素相对误差）", THRESH);

        let sizes = param_sizes();
        let ana_all: Vec<f32> = {
            let mut v = Vec::new();
            for n in NAMES.iter() {
                let g = format!("g_{}", n);
                let sz = {
                    let si = fvm.find_slot(&g);
                    let (r, c) = fvm.slot_shape(si);
                    r * c
                };
                let p = slot_ptr(&mut fvm, &g);
                for i in 0..sz {
                    v.push(*p.add(i));
                }
            }
            v
        };

        // (a) 与 kernels::exp_f 同族的近似 exp 前向
        check_grads(&ana_all, &sizes, "torch_grads", "PyTorch(近似exp)", &mut all_ok);
        // (b) f64 精确 exp 的 softmax 前向 —— 用来剥离 exp_f 6.51e-4 的污染
        check_grads(
            &ana_all,
            &sizes,
            "torch_exact_grads",
            "PyTorch(f64精确exp)",
            &mut all_ok,
        );

        // 诊断：直接打印 ana_all 里各段的值，核对读出来的布局
        if std::env::args().any(|a| a == "--raw-ana") {
            let mut off = 0usize;
            for (i, n) in NAMES.iter().enumerate() {
                let sz = sizes[i];
                println!("      ana {:<8} = {:?}", n, &ana_all[off..(off + sz).min(off + 8)]);
                off += sz;
            }
            let num_all = read_f32(&format!("{}/torch_grads.bin", DIR));
            let mut off = 0usize;
            for (i, n) in NAMES.iter().enumerate() {
                let sz = sizes[i];
                println!("      num {:<8} = {:?}", n, &num_all[off..(off + sz).min(off + 8)]);
                off += sz;
            }
        }

        // 诊断：把解析梯度按 1/T 缩放后再比（验证「torch 取 mean、我们取 sum」假设）
        if std::env::args().any(|a| a == "--scale-t") {
            let scaled: Vec<f32> = ana_all.iter().map(|x| x / T as f32).collect();
            check_grads(&scaled, &sizes, "torch_grads", "PyTorch(近似exp, ana/T)", &mut all_ok);
        }

        // 诊断：打印 VMLM 解析梯度的量级
        {
            let mut off = 0usize;
            let mut s = String::new();
            for (i, n) in NAMES.iter().enumerate() {
                let sz = sizes[i];
                let seg = &ana_all[off..off + sz];
                let l1: f32 = seg.iter().map(|x| x.abs()).sum();
                let mx = seg.iter().fold(0.0f32, |a, b| a.max(b.abs()));
                s.push_str(&format!("    g_{:<8} ‖·‖₁={:>10.4} max|·|={:>10.4}\n", n, l1, mx));
                off += sz;
            }
            println!("\n    --- VMLM 解析梯度量级（诊断）---\n{}", s);
        }

        // ---------------- 2. 过拟合测试 ----------------
        let hp = slot_ptr(&mut vm, "hparams");
        let lr = *hp;

        println!("\n[2] 过拟合测试：固定单批次训练 {} 步（lr={:.3}）", steps, lr);
        println!("    tok={:?}  target={:?}", [1, 2, 3, 0], [2, 3, 0, 1]);
        println!("    loss = 逐位置交叉熵的**均值**（与 torch cross_entropy 同口径）");
        println!("    {:>6} {:>13} {:>15} {:>10}", "step", "loss", "最近25步均值", "ms");

        let mut losses = Vec::with_capacity(steps);
        let cpu0 = cpu_seconds();
        let t0 = Instant::now();
        for step in 0..steps {
            vm.run();
            // vm.loss 是最后一个位置的 CE；这里从 logits 重算 T 位置均值，
            // 保证曲线口径与反向传播的 loss 一致（torch 的 mean）。
            let lgv = peek(&mut vm, "logits", T * V);
            losses.push(mean_ce(&lgv, &[2.0, 3.0, 0.0, 1.0]));
            if step == 0 || (step + 1) % 25 == 0 {
                let w = 25.min(step + 1);
                let avg: f32 = losses[step + 1 - w..].iter().sum::<f32>() / w as f32;
                println!(
                    "    {:>6} {:>13.7} {:>15.7} {:>10.0}",
                    step + 1,
                    losses[step],
                    avg,
                    t0.elapsed().as_secs_f64() * 1000.0
                );
            }
        }
        let wall = t0.elapsed().as_secs_f64();
        let cpu = cpu_seconds() - cpu0;

        let first = losses[0];
        let last_w = 25.min(losses.len());
        let last: f32 = losses[losses.len() - last_w..].iter().sum::<f32>() / last_w as f32;
        let best = losses.iter().cloned().fold(f32::INFINITY, f32::min);
        println!(
            "\n    loss: 首步 {:.6} → 末{}步均值 {:.6}（最小 {:.6}）",
            first, last_w, last, best
        );
        println!(
            "    耗时: {:.1} ms / {} 步 = {:.3} ms/step    CPU 占用 = {:.1}%",
            wall * 1000.0,
            steps,
            wall * 1000.0 / steps as f64,
            cpu / wall * 100.0
        );
        all_ok &= last < 0.01;

        // ---------------- 3. 因果性回归（训练后） ----------------
        let att2 = peek(&mut vm, "att", T * T);
        let mut leak = 0.0f32;
        let mut causal_ok = true;
        for i in 0..T {
            for j in (i + 1)..T {
                let v = att2[i * T + j].abs();
                if v > leak {
                    leak = v;
                }
                if v > 1e-6 {
                    causal_ok = false;
                }
            }
        }
        println!("\n[3] 因果性回归（训练后 att 上三角必须为 0）");
        println!("    最大上三角绝对值 = {:.3e}", leak);
        for i in 0..T {
            println!("    att[{},:] = {:?}", i, &att2[i * T..(i + 1) * T]);
        }
        println!(
            "    判定: {}",
            if causal_ok {
                "PASS（反向没有引入未来信息泄漏）"
            } else {
                "FAIL —— 反向泄漏了未来信息"
            }
        );
        all_ok &= causal_ok;

        // ---------------- 4. 完美复现 batch 检查 ----------------
        let lg2 = peek(&mut vm, "logits", T * V);
        let tgt = [2usize, 3, 0, 1];
        let mut correct = 0;
        let mut argmax = Vec::new();
        for i in 0..T {
            let row = &lg2[i * V..(i + 1) * V];
            let mut b = 0;
            for k in 1..V {
                if row[k] > row[b] {
                    b = k;
                }
            }
            argmax.push(b);
            if b == tgt[i] {
                correct += 1;
            }
        }
        println!("\n[4] 单批次复现：argmax = {:?}  目标 = {:?}  {}/{} 正确", argmax, tgt, correct, T);
        all_ok &= correct == T;

        println!(
            "\n判定: {}",
            if all_ok {
                "PASS（前向回归 + 梯度检查 + 过拟合 + 因果性 + 单批次复现）"
            } else {
                "FAIL"
            }
        );
        std::process::exit(if all_ok { 0 } else { 1 });
    }
}

// 与一份 torch 参考做梯度对比并打印
fn check_grads(
    ana_all: &[f32],
    sizes: &[usize],
    stem: &str,
    label: &str,
    all_ok: &mut bool,
) {
    let bin = format!("{}/{}.bin", DIR, stem);
    let meta = format!("{}/{}.meta", DIR, stem);
    if !std::path::Path::new(&bin).exists() {
        println!("    {} 参考缺失（{}），跳过", label, bin);
        return;
    }
    let num_all = read_f32(&bin);
    let (tloss, list) = read_meta(&meta).expect("meta 读不了");

    println!("\n    --- vs {} （torch loss = {:.6}）---", label, tloss);
    println!(
        "    {:<10}{:>8}{:>14}{:>14}{:>12}  {}",
        "参数", "元素数", "Σ|Δ|", "判据", "逐元素最坏", "判定"
    );

    // meta 里的顺序可能与 NAMES 一致；按名字对齐切片
    let mut off_num = 0usize;
    let mut off_ana = 0usize;
    let mut worst_g = 0.0f64;
    let mut worst_name = String::new();
    for (i, (name, cnt)) in list.iter().enumerate() {
        let _ = i;
        let num = &num_all[off_num..off_num + cnt];
        let ana = &ana_all[off_ana..off_ana + cnt];
        let (g, sum_d, worst) = criterion(num, ana);
        let ok = g < THRESH;
        *all_ok &= ok;
        if g > worst_g {
            worst_g = g;
            worst_name = name.clone();
        }
        println!(
            "    {:<10}{:>8}{:>14.3e}{:>14.3e}{:>12.3e}  {}",
            name,
            cnt,
            sum_d,
            g,
            worst,
            if ok { "PASS" } else { "FAIL" }
        );
        off_num += cnt;
        off_ana += cnt;
    }
    println!(
        "    → 最坏参数 {} 判据 {:.3e}   总计 {} 个参数元素",
        worst_name,
        worst_g,
        off_num
    );

    // 逐元素对照（定位符号错 / 转置错 / 少一条路径）
    if std::env::args().any(|a| a == "--trace-grad") {
        let mut off_n = 0usize;
        let mut off_a = 0usize;
        for (name, cnt) in list.iter() {
            let num = &num_all[off_n..off_n + cnt];
            let ana = &ana_all[off_a..off_a + cnt];
            println!("      --- {} ---", name);
            // 找出第一处显著偏差
            let mut bad = None;
            for i in 0..*cnt {
                let d = (num[i] - ana[i]).abs();
                let s = num[i].abs() + ana[i].abs();
                if s > 1e-6 && d / s > 1e-3 {
                    bad = Some(i);
                    break;
                }
            }
            println!("        first bad idx = {:?}", bad);
            for i in 0..(*cnt).min(12) {
                println!(
                    "        [{:>3}] num={:>12.6}  ana={:>12.6}  diff={:>11.3e}  ratio={:>8.4}",
                    i,
                    num[i],
                    ana[i],
                    (num[i] - ana[i]).abs(),
                    if ana[i].abs() > 1e-12 {
                        num[i] / ana[i]
                    } else {
                        f32::NAN
                    }
                );
            }
            off_n += cnt;
            off_a += cnt;
        }
    }
    let _ = sizes;
}
