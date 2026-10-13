// P8-C2 验证驱动：参数化 ESERNN 单步前向的 VMLM 实现 vs Python 参考
//
// 与 P7-C 的 `eserNN_run.rs` 的区别：
//   * 配置（hidden / n_heads / head_dim / vocab / n_layers / T）从 `cfg.json` 读，
//     不再硬编码 mini；同一份驱动可验证 mini 与放大后的所有档位。
//   * `st_S` / `st_xprev` 是**每层一份**（st_S0..st_S{L-1}），逐层比对。
//   * logits 只在最后一层之后产生，与源码一致。
//
// 验证点：
//   1. 逐步 logits 对比（每一步都给最大绝对误差）
//   2. 每层递归态 st_S{l} 末步值对比
//   3. 每层 st_xprev{l} 末步值对比（跨 step 状态推进语义）
//   4. 判据：最大绝对误差 < 1e-3
//
// 两份参考：
//   ref_*       真实 ESERNN 语义，eps = cfg.rms_eps = 1e-6  ← 主判据
//   ref_eps5_*  引擎 OP_LN 默认语义，eps = 1e-5            ← 对照
use vmlm_core::interp::Vm;

extern "C" {
    fn malloc(n: usize) -> *mut u8;
}

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

unsafe fn set_slot(vm: &mut Vm, name: &str, vals: &[f32]) {
    let si = vm.find_slot(name);
    let p = vm.slot_mut(si);
    for (i, v) in vals.iter().enumerate() {
        *p.add(i) = *v;
    }
}

unsafe fn get_slot(vm: &mut Vm, name: &str, n: usize) -> Vec<f32> {
    let si = vm.find_slot(name);
    let p = vm.slot_mut(si);
    (0..n).map(|i| *p.add(i)).collect()
}

/// 非 panic 的槽位探测：把名字哈希与名字表逐项比一遍。
/// `Vm::find_slot` 找不到会直接 exit，诊断路径需要「没有就跳过」的语义。
///
/// ⚠ 只用于**可选诊断**路径，且诊断是否启用一律以「对应参考文件是否存在」
///   为准（更确定）。不要用它来做主流程的正确性判断 —— 它依赖
///   `Vm::name_tab` 的内部排布，字段一变就会给出假阴性。
#[allow(dead_code)]
unsafe fn slot_exists(vm: &Vm, name: &str) -> bool {
    let h = vmlm_core::format::hash32(name);
    for i in 0..vm.n_slots {
        // name_tab 每项 4 字节（u32 哈希），见 asm/src/main.rs 的 META 布局
        let p = vm.name_tab.add(i * 4);
        let v = u32::from_le_bytes([*p, *p.add(1), *p.add(2), *p.add(3)]);
        if v == h {
            return true;
        }
    }
    false
}

fn maxabs(a: &[f32], b: &[f32]) -> f32 {
    let mut e = 0.0f32;
    for i in 0..a.len().min(b.len()) {
        let d = (a[i] - b[i]).abs();
        // NaN 必须显式判失败：NaN 与任何数比较都是 false，会静默通过 < 1e-3
        if !d.is_finite() {
            return f32::INFINITY;
        }
        if d > e {
            e = d;
        }
    }
    e
}

/// 把 VMLM 的 st_S[D, NHEAD*D]（head 水平拼接）重排成 [NHEAD, D, D]，
/// 以便与 Python 参考（真实 ESERNN 的 state 布局 [n_heads, D, D]）逐元素比较。
fn reorder_s(flat: &[f32], nhead: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; nhead * d * d];
    for head in 0..nhead {
        for i in 0..d {
            for j in 0..d {
                out[head * d * d + i * d + j] = flat[i * (nhead * d) + head * d + j];
            }
        }
    }
    out
}

/// 极简 JSON 数字读取（避免引 serde；cfg.json 是我们自己生成的，格式可控）。
fn json_usize(txt: &str, key: &str) -> usize {
    let pat = format!("\"{}\"", key);
    let i = txt.find(&pat).unwrap_or_else(|| panic!("cfg.json 缺字段 {}", key));
    let rest = &txt[i + pat.len()..];
    let c = rest.find(':').expect("JSON 格式错");
    let s = rest[c + 1..].trim_start();
    let end = s
        .find(|ch: char| !ch.is_ascii_digit())
        .unwrap_or(s.len());
    s[..end].parse().unwrap_or_else(|_| panic!("字段 {} 不是整数", key))
}

/// 可选字段：按顺序试多个候选键名，全都没有就返回 `default`。
///
/// 【为什么需要】P7-C 手写的 `make_eserNN.py` 生成的 `cfg.json` **没有
/// `layers` 键**（mini 是单层，层数被硬编码在 `.vml` 里）。P8 的生成器
/// 会写 `layers`。为了让同一份驱动既能验收 P7-C 的旧 mini 目录、又能验收
/// 放大后的各档，这里做向后兼容的键回退。
fn json_usize_opt(txt: &str, keys: &[&str], default: usize) -> usize {
    for k in keys {
        if txt.contains(&format!("\"{}\"", k)) {
            return json_usize(txt, k);
        }
    }
    default
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = args.get(1).cloned().unwrap_or_else(|| "examples/eserNN".into());
    unsafe { vmlm_core::interp::init_cpu() };

    let cfg_txt = std::fs::read_to_string(format!("{}/cfg.json", dir))
        .unwrap_or_else(|_| panic!("读不到 {}/cfg.json", dir));
    let h = json_usize(&cfg_txt, "hidden");
    let nhead = json_usize(&cfg_txt, "heads");
    let d = json_usize(&cfg_txt, "head_dim");
    let v = json_usize(&cfg_txt, "vocab");
    // ⚠ `heads` 是注意力头数，`layers` 是层数 —— 两者**不是**一回事。
    // P7-C 的 mini cfg.json 没有 `layers`，按单层处理（它的 .vml 确实是 1 层）。
    let nl = json_usize_opt(&cfg_txt, &["layers", "n_layers"], 1);
    let t_steps = json_usize(&cfg_txt, "T");

    let rd = |n: &str| -> Vec<f32> {
        let b = std::fs::read(format!("{}/{}.bin", dir, n))
            .unwrap_or_else(|_| panic!("读不到 {}.bin", n));
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    let rd_txt = |n: &str| -> Vec<Vec<f32>> {
        std::fs::read_to_string(format!("{}/{}", dir, n))
            .unwrap_or_else(|_| panic!("读不到 {}", n))
            .lines()
            .map(|l| l.split_whitespace().map(|s| s.parse().unwrap()).collect())
            .collect()
    };

    let xseq = rd("x_all");
    let ref_logits = rd_txt("ref_logits.txt");
    let ref_s = rd("ref_S");
    let ref_xprev = rd("ref_xprev");
    let _ref_x_steps = rd("ref_x_steps");
    let ref5_logits = rd_txt("ref_eps5_logits.txt");
    let ref5_s = rd("ref_eps5_S");
    let ref5_xprev = rd("ref_eps5_xprev");

    println!("=== P8-C2: 参数化 ESERNN 单步前向迁移验证 ===");
    println!(
        "配置: vocab={} hidden={} heads={} head_dim={} layers={} steps={}",
        v, h, nhead, d, nl, t_steps
    );
    println!("源: <ESERNN 仓库路径>/core/ese_rnn.py (forward_one / ESERNNBlock / TimeMixing / ChannelMixing)");
    println!("指令流由 tools/gen_eserNN.py 生成\n");

    unsafe {
        let t0 = std::time::Instant::now();
        let buf = load_vmb(&format!("{}/step.vmb", dir));
        let mut vm = Vm::load(buf);
        let load_us = t0.elapsed().as_micros();
        println!("[装载] Vm::load = {:.3} ms", load_us as f64 / 1000.0);

        let mut worst_real = 0.0f32;
        let mut worst_eng = 0.0f32;

        let t1 = std::time::Instant::now();
        println!("\n--- 逐步 logits 对比（VMLM vs Python 参考）---");
        println!(
            "{:>5}  {:>12}  {:>12}   {}",
            "step", "vs 真实(1e-6)", "vs 引擎(1e-5)", "VMLM logits[0..3]"
        );
        for t in 0..t_steps {
            let xs = &xseq[t * h..(t + 1) * h];
            set_slot(&mut vm, "x", xs);
            vm.run();
            let lg = get_slot(&mut vm, "logits", v);
            let e_real = maxabs(&lg, &ref_logits[t]);
            let e_eng = maxabs(&lg, &ref5_logits[t]);
            worst_real = worst_real.max(e_real);
            worst_eng = worst_eng.max(e_eng);
            println!(
                "{:>5}  {:>12.3e}  {:>12.3e}   [{:.6}, {:.6}, {:.6}]",
                t, e_real, e_eng, lg[0], lg[1], lg[2]
            );
        }
        let fwd_ms = t1.elapsed().as_micros() as f64 / 1000.0;
        println!(
            "\n[性能] {} 步前向合计 {:.2} ms（{:.2} ms/token，单步含全部 {} 层）",
            t_steps, fwd_ms, fwd_ms / t_steps as f64, nl
        );

        // ---- 每层递归态 st_S{l} 对比（末步值）----
        println!("\n--- 递归态 st_S 对比（末步，每层 {} 个 head × [{}×{}]）---", nhead, d, d);
        let sname = |l: usize| if nl > 1 { format!("st_S{}", l) } else { "st_S".to_string() };
        let xpname = |l: usize| if nl > 1 { format!("st_xprev{}", l) } else { "st_xprev".to_string() };
        let per = nhead * d * d;
        let mut s_err_real = 0.0f32;
        let mut s_err_eng = 0.0f32;
        let mut shown = false;
        for l in 0..nl {
            let flat = get_slot(&mut vm, &sname(l), d * nhead * d);
            let got = reorder_s(&flat, nhead, d);
            let e_r = maxabs(&got, &ref_s[l * per..(l + 1) * per]);
            let e_e = maxabs(&got, &ref5_s[l * per..(l + 1) * per]);
            s_err_real = s_err_real.max(e_r);
            s_err_eng = s_err_eng.max(e_e);
            println!("  第 {} 层: vs 真实 = {:.3e}   vs 引擎 = {:.3e}", l, e_r, e_e);
            if !shown && nl > 1 {
                // 抽样打印一层，便于人工核对量级（放大后 head 数很多，不宜全打）
                println!(
                    "         VMLM  S[0][0][0..3] = [{:.6}, {:.6}, {:.6}]",
                    got[0], got[1], got[2]
                );
                println!(
                    "         参考  S[0][0][0..3] = [{:.6}, {:.6}, {:.6}]",
                    ref_s[0], ref_s[1], ref_s[2]
                );
                shown = true;
            }
        }

        // ---- 每层 st_xprev{l} 对比 ----
        println!("\n--- st_xprev 对比（末步，跨 step 状态推进）---");
        let mut xp_err_real = 0.0f32;
        let mut xp_err_eng = 0.0f32;
        for l in 0..nl {
            let xp = get_slot(&mut vm, &xpname(l), h);
            let e_r = maxabs(&xp, &ref_xprev[l * h..(l + 1) * h]);
            let e_e = maxabs(&xp, &ref5_xprev[l * h..(l + 1) * h]);
            xp_err_real = xp_err_real.max(e_r);
            xp_err_eng = xp_err_eng.max(e_e);
            println!("  第 {} 层: vs 真实 = {:.3e}   vs 引擎 = {:.3e}", l, e_r, e_e);
        }

        // ---- 可选：逐层块输出诊断 ----
        // 若目录里有 `ref_per_layer.bin`（[T, NL, H]）且 .vml 里把每层块输出
        // 拷进了 `probe_l{l}` 持久槽，就逐步逐层比对 —— 用来定位
        // 「logits 错但 st_S 对」这类层间传递 bug 到底起自哪一层。
        // 这两个文件都是**可选**的，正常验收路径不产生它们。
        let perpath = format!("{}/ref_per_layer.bin", dir);
        let mut diag_started = false;
        if std::path::Path::new(&perpath).exists() {
            let refpl = rd("ref_per_layer");
            if refpl.len() >= t_steps * nl * h {
                // ⚠ probe_l* 是**瞬态激活槽**，每次 run() 结束（HALT 前）都会被
                // 重写成「当前 step」的值。所以只能在跑完整个序列后用**末步**的值
                // 与参考的末步比对 —— 而且**绝不能再 run()**，否则状态会推进到
                // 第 t_steps 步之后，比出来的就是「第 11 步 vs 第 5 步」的假误差
                // （这个坑我踩过：层2 因此虚报 3.4e-2）。
                println!("\n--- 逐层块输出对比（诊断，末步）---");
                let t = t_steps - 1;
                for l in 0..nl {
                    let got = get_slot(&mut vm, &format!("probe_l{}", l), h);
                    let r = &refpl[(t * nl + l) * h..(t * nl + l + 1) * h];
                    println!("  末步 第{}层: max|Δ| = {:.3e}", l, maxabs(&got, r));
                }
            }
        }

        // ---- 可选：层内中间量诊断 ----
        // 若目录里有 `refp_<name>.bin` 且 .vml 把该中间量拷进了 `probe_<name>`
        // 持久槽，就逐项比对 —— 用于把「哪一步开始错」钉死到具体算子。
        for nm in ["x1", "dec", "hdec", "r", "kk", "vv", "ocat", "x2"] {
            let rp = format!("{}/refp_{}.bin", dir, nm);
            let pn = format!("probe_{}", nm);
            if !std::path::Path::new(&rp).exists() {
                continue;
            }
            if !diag_started {
                println!("\n--- 层内中间量对比（诊断，末步）---");
                diag_started = true;
            }
            let r = rd(&format!("refp_{}", nm));
            let got = get_slot(&mut vm, &pn, r.len());
            println!("  {:<6} max|Δ| = {:.3e}", nm, maxabs(&got, &r));
        }

        // ---- 可选：自洽性检查 ----
        // .vml 里的 `COPY st_xprev{l}, L{l}_out` 之后若再有人改写 L{l}_out，
        // 那么「末步把 L{l}_out 拷进 pl{l}」读到的就不是写回时的值。
        // 这个检查直接把两者都存下来比 —— 相等说明写回是自洽的。
        if std::path::Path::new(&format!("{}/px0.bin", dir)).exists() {
            println!("\n--- .vml 自洽性：L{{l}}_out vs st_xprev{{l}}（末步）---");
            for l in 0..nl {
                let a = get_slot(&mut vm, &format!("pl{}", l), h);
                let b = get_slot(&mut vm, &format!("px{}", l), h);
                println!("  第{}层: max|L_out - st_xprev| = {:.3e}", l, maxabs(&a, &b));
            }
        }

        // ---- 可选：快照 dump（诊断用）----
        // 若存在 `dump_slots.txt`（每行一个槽位名）：
        //   * 打印末步前 4 个值到屏幕
        //   * 把**完整**向量写到 <dir>/vmldump_<name>.bin（供 python 精确复算）
        let dl = format!("{}/dump_slots.txt", dir);
        if std::path::Path::new(&dl).exists() {
            println!("\n--- 快照 dump（末步，前 4 个值）---");
            for nm in std::fs::read_to_string(&dl).unwrap().lines() {
                let nm = nm.trim();
                if nm.is_empty() {
                    continue;
                }
                let v = get_slot(&mut vm, nm, 4);
                println!("  {:<12} [{:.6}, {:.6}, {:.6}, {:.6}]",
                         nm, v[0], v[1], v[2], v[3]);
                // 完整 dump：形状不可知，按 4096 个元素上探（够本项目所有向量用）。
                // ⚠ 用已经查好的 si，不要重复 find_slot（找不到会 exit 7）。
                let si = vm.find_slot(nm);
                let p = vm.slot_mut(si);
                let full: Vec<f32> = (0..4096).map(|i| *p.add(i)).collect();
                let mut bytes = Vec::with_capacity(full.len() * 4);
                for x in &full {
                    bytes.extend_from_slice(&x.to_le_bytes());
                }
                let _ = std::fs::write(format!("{}/vmldump_{}.bin", dir, nm), bytes);
            }
        }

        // ---- 汇总 ----
        let worst_eng_all = worst_eng.max(s_err_eng).max(xp_err_eng);
        let worst_real_all = worst_real.max(s_err_real).max(xp_err_real);

        println!("\n=== 汇总 ===");
        println!("  【验收判据 <1e-3】VMLM vs 真实 ESERNN(eps=1e-6) —— 主判据");
        println!(
            "    logits {:.3e} / st_S {:.3e} / x_prev {:.3e}  ⇒ 综合 {:.3e}",
            worst_real, s_err_real, xp_err_real, worst_real_all
        );
        println!();
        println!("  【对照】若**不**发 LN_EPS（引擎默认 eps=1e-5）会差多少");
        println!(
            "    logits {:.3e} / st_S {:.3e} / x_prev {:.3e}  ⇒ 综合 {:.3e}",
            worst_eng, s_err_eng, xp_err_eng, worst_eng_all
        );

        let ok = worst_real_all.is_finite() && worst_real_all < 1e-3;
        println!("\n判定:");
        println!(
            "  [验收判据 <1e-3] {} —— 与真实 ESERNN 的差距为 {:.3e}",
            if ok { "PASS" } else { "FAIL" },
            worst_real_all
        );
        if !ok {
            println!("  ⚠ 误差超判据。请检查：NaN/Inf（数值发散）、eps 是否生效、层间 x_prev 语义。");
        }
        std::process::exit(if ok { 0 } else { 1 });
    }
}
