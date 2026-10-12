// P9-I: ESERNN training-loop driver (forward check / gradient dump / training).
//
// Three modes, matching the acceptance items of the P9 contract:
//
//   fwd    run the forward-only program, compare the per-step logits against the
//          PyTorch f64 reference, and report the max absolute difference.
//          This exists so the forward can be cleared BEFORE the backward is
//          trusted: a mislabelled forward makes every gradient look wrong.
//
//   grad   run the full train step once with the learning rate forced to zero,
//          recompute the sequence loss from the per-step logits, and dump the
//          analytic gradients to <dir>/grads/ana_<param>.bin for check_grad.py.
//
//   train  run N steps, record the loss curve, and report wall/CPU time.
//
// Why the loss is recomputed rather than read from `vm.loss`: CE_BWD OVERWRITES
// vm.loss, and the backward pass runs from step T-1 down to 0, so after a run
// vm.loss holds STEP 0's cross-entropy -- not the mean, and not the last step.
// The generator surfaces each step's logits as `loss_out{t}`-style slots (here
// `logits{t}` plus the targets `y{t}`), so the driver recomputes:
//
//     mean_CE = (1/T) * sum_t  -log softmax(logits_t)[y_t]
//
// and the generator also exposes `loss_sum`, the *sum* of the per-step CEs,
// which is the quantity the backward actually differentiates (the *_BWD ops
// accumulate) and the one comparable to PyTorch's reduction='sum'.
use vmlm_core::interp::Vm;

extern "C" {
    fn malloc(n: usize) -> *mut u8;
}

// ---------------------------------------------------------------------------
// Configuration, read from cfg.json (our own format, so a tiny scanner is fine;
// pulling in serde for this would be silly).
// ---------------------------------------------------------------------------
struct Cfg {
    hidden: usize,
    heads: usize,
    head_dim: usize,
    inter: usize,
    decay_hidden: usize,
    layers: usize,
    vocab: usize,
    t: usize,
    lr: f32,
}

fn json_num(txt: &str, key: &str) -> f64 {
    let pat = format!("\"{}\"", key);
    let i = txt
        .find(&pat)
        .unwrap_or_else(|| panic!("cfg.json is missing the key {}", key));
    let rest = &txt[i + pat.len()..];
    let c = rest.find(':').expect("malformed cfg.json");
    let rest = rest[c + 1..].trim_start();
    let end = rest
        .find(|ch: char| !(ch.is_ascii_digit() || ch == '.' || ch == '-' || ch == 'e' || ch == 'E' || ch == '+'))
        .unwrap_or(rest.len());
    rest[..end].parse().expect("malformed number in cfg.json")
}

fn read_cfg(dir: &str) -> Cfg {
    let txt = std::fs::read_to_string(format!("{}/cfg.json", dir))
        .unwrap_or_else(|_| panic!("cannot read {}/cfg.json -- run make_eserNN_train.py first", dir));
    Cfg {
        hidden: json_num(&txt, "hidden") as usize,
        heads: json_num(&txt, "heads") as usize,
        head_dim: json_num(&txt, "head_dim") as usize,
        inter: json_num(&txt, "inter") as usize,
        decay_hidden: json_num(&txt, "decay_hidden") as usize,
        layers: json_num(&txt, "layers") as usize,
        vocab: json_num(&txt, "vocab") as usize,
        t: json_num(&txt, "T") as usize,
        lr: json_num(&txt, "lr") as f32,
    }
}

/// The trainable parameters, in the same order the reference exports them.
fn param_names(c: &Cfg) -> Vec<String> {
    let lp = |l: usize| -> String {
        if c.layers > 1 {
            format!("L{}_", l)
        } else {
            String::new()
        }
    };
    let wp = |l: usize| -> String {
        if c.layers > 1 {
            format!("l{}_", l)
        } else {
            String::new()
        }
    };
    let mut v: Vec<String> = ["E", "ln0_g", "ln0_b", "lnout_g", "lnout_b"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    for l in 0..c.layers {
        let w = wp(l);
        for n in ["ln1_g", "ln1_b", "ln2_g", "ln2_b"] {
            v.push(format!("{}{}", w, n));
        }
        for n in ["xr", "xk", "xv", "xw"] {
            v.push(format!("{}tm_{}", w, n));
        }
        for n in ["tm_w0row", "tm_w1p", "tm_w2p"] {
            v.push(format!("{}{}", w, n));
        }
        for n in ["Wr", "Wk", "Wv", "Wo"] {
            v.push(format!("{}tm_{}", w, n));
        }
        for n in ["tmln_g", "tmln_b", "cm_xk", "cm_Wkey", "cm_Wval", "cmln_g"] {
            v.push(format!("{}{}", w, n));
        }
    }
    v
}

unsafe fn load_vmb(path: &str) -> *mut u8 {
    let data = std::fs::read(path).unwrap_or_else(|_| panic!("cannot read {}", path));
    let n = data.len();
    let p = malloc(n + 64);
    std::ptr::copy_nonoverlapping(data.as_ptr(), p, n);
    for i in n..n + 64 {
        *p.add(i) = 0;
    }
    p
}

unsafe fn read_slot(vm: &mut Vm, name: &str) -> Vec<f32> {
    let si = vm.find_slot(name);
    let (r, c) = vm.slot_shape(si);
    let p = vm.slot_mut(si);
    (0..r * c).map(|i| *p.add(i)).collect()
}

unsafe fn write_slot(vm: &mut Vm, name: &str, vals: &[f32]) {
    let si = vm.find_slot(name);
    let p = vm.slot_mut(si);
    for (i, v) in vals.iter().enumerate() {
        *p.add(i) = *v;
    }
}

unsafe fn write_scalar(vm: &mut Vm, name: &str, v: f32) {
    let si = vm.find_slot(name);
    *vm.slot_mut(si) = v;
}

/// Does this program mention the slot at all?
///
/// Only the slots a program actually names end up in the slot table, and the
/// forward-only probe deliberately omits the optimizer plumbing. Asking for a
/// slot that was never mentioned aborts the process (`find_slot` exits 7), so
/// every optional write has to be guarded -- otherwise a probe run dies for a
/// reason that has nothing to do with what it is probing.
unsafe fn has_slot(vm: &mut Vm, name: &str) -> bool {
    let h = vmlm_core::format::hash32(name);
    for i in 0..vm.n_slots {
        if vmlm_core::u32_at(vm.name_tab, i * 4) == h {
            return true;
        }
    }
    false
}

unsafe fn read_bin(path: &str) -> Vec<f32> {
    let b = std::fs::read(path).unwrap_or_else(|_| panic!("cannot read {}", path));
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

unsafe fn write_bin(path: &str, v: &[f32]) {
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    std::fs::write(path, &b).unwrap();
}

/// Per-step cross-entropy, recomputed from the logits (see the file header).
unsafe fn step_cross_entropies(vm: &mut Vm, c: &Cfg) -> Vec<f32> {
    let mut out = Vec::with_capacity(c.t);
    for t in 0..c.t {
        let lg = read_slot(vm, &format!("logits{}", t));
        let yv = read_slot(vm, &format!("y{}", t));
        let tgt = yv[0] as usize;
        let mx = lg.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in &lg {
            sum += (v - mx).exp();
        }
        out.push(-((lg[tgt] - mx).exp() / sum).ln());
    }
    out
}

unsafe fn mean_ce(vm: &mut Vm, c: &Cfg) -> f32 {
    let v = step_cross_entropies(vm, c);
    v.iter().sum::<f32>() / c.t as f32
}

/// Process CPU time (user+kernel) in seconds, straight from the OS.
#[cfg(windows)]
fn cpu_seconds() -> f64 {
    #[repr(C)]
    #[derive(Default, Clone, Copy)]
    struct FileTime {
        lo: u32,
        hi: u32,
    }
    // GetProcessTimes lives in kernel32 and uses the Win64 calling convention;
    // `extern "system"` is what makes the FFI contract match.
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> *mut core::ffi::c_void;
        fn GetProcessTimes(
            h: *mut core::ffi::c_void,
            creation: *mut FileTime,
            exit: *mut FileTime,
            kernel: *mut FileTime,
            user: *mut FileTime,
        ) -> i32;
    }
    fn to_u64(t: &FileTime) -> u64 {
        ((t.hi as u64) << 32) | t.lo as u64
    }
    unsafe {
        let mut a = FileTime::default();
        let mut b = FileTime::default();
        let mut k = FileTime::default();
        let mut u = FileTime::default();
        if GetProcessTimes(GetCurrentProcess(), &mut a, &mut b, &mut k, &mut u) == 0 {
            return f64::NAN;
        }
        // FILETIME counts 100-nanosecond units.
        (to_u64(&k) + to_u64(&u)) as f64 * 1e-7
    }
}

#[cfg(not(windows))]
fn cpu_seconds() -> f64 {
    0.0
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "examples/eserNN/train".into());
    let mode = args.get(2).cloned().unwrap_or_else(|| "train".into());
    let steps: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(300);
    let lr: f32 = args
        .get(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);

    unsafe { vmlm_core::interp::init_cpu() };

    let c = read_cfg(&dir);
    println!("=== P9-I : ESERNN training loop (BPTT) ===");
    println!(
        "hidden={} heads={} head_dim={} inter={} decay_hidden={} layers={} vocab={} T={}",
        c.hidden, c.heads, c.head_dim, c.inter, c.decay_hidden, c.layers, c.vocab, c.t
    );
    println!("dir = {}   mode = {}   steps = {}   lr = {}", dir, mode, steps, if lr > 0.0 { lr } else { c.lr });

    let params = param_names(&c);
    let vmb = match mode.as_str() {
        "fwd" => format!("{}/fwd_t{}.vmb", dir, c.t),
        // `run` / `dump` are the debug entry points: execute whatever was
        // assembled, without needing a reference file. argv[5] names the .vmb so
        // the forward bisector can point them at a truncated program.
        "run" | "dump" => args
            .get(5)
            .cloned()
            .unwrap_or_else(|| format!("{}/fwd_t{}.vmb", dir, c.t)),
        _ => format!("{}/bptt_t{}.vmb", dir, c.t),
    };

    unsafe {
        let buf = load_vmb(&vmb);
        let mut vm = Vm::load(buf);
        println!("instructions = {}", vm.n_ins);

        // hparams = [lr, b1, b2, eps, wd]; clip_norm stays huge so the gradients
        // are never rescaled (the gradient check compares raw gradients).
        // The forward-only probe has no optimizer plumbing, so these are guarded.
        let use_lr = if lr > 0.0 { lr } else { c.lr };
        if has_slot(&mut vm, "hparams") {
            write_slot(&mut vm, "hparams", &[use_lr, 0.9, 0.999, 1e-8, 1e-4]);
        }
        if has_slot(&mut vm, "clip_norm") {
            write_scalar(&mut vm, "clip_norm", 1e9);
        }

        match mode.as_str() {
            // ================= raw run (debug / bisector entry) =================
            "run" => {
                vm.run();
                for t in 0..c.t {
                    if has_slot(&mut vm, &format!("logits{}", t)) {
                        let lg = read_slot(&mut vm, &format!("logits{}", t));
                        println!(
                            "  logits{} = {:?}",
                            t,
                            lg.iter().map(|x| format!("{:.6}", x)).collect::<Vec<_>>()
                        );
                    }
                }
                println!("ok: run completed ({} instructions)", vm.n_ins);
            }

            // ================= dump arbitrary slots (debug) =================
            // usage: vmlm-eserNN-train <dir> dump <steps> <lr> <T> slot1 slot2 ...
            "dump" => {
                vm.run();
                for name in args.iter().skip(6) {
                    if !has_slot(&mut vm, name) {
                        println!("{} MISSING", name);
                        continue;
                    }
                    let si = vm.find_slot(name);
                    let (r, c) = vm.slot_shape(si);
                    let p = vm.slot_mut(si);
                    let v: Vec<f32> = (0..r * c).map(|i| *p.add(i)).collect();
                    println!(
                        "{} [{}x{}] = {:?}",
                        name,
                        r,
                        c,
                        v.iter().map(|x| format!("{:.8}", x)).collect::<Vec<_>>()
                    );
                }
            }

            // ================= forward check =================
            "fwd" => {
                vm.run();
                let ref_lg = read_bin(&format!("{}/ref_logits.bin", dir));                let mut worst = 0.0f32;
                let mut worst_at = (0usize, 0usize);
                let mut got_all: Vec<f32> = Vec::new();
                for t in 0..c.t {
                    let got = read_slot(&mut vm, &format!("logits{}", t));
                    for j in 0..c.vocab {
                        let d = (got[j] - ref_lg[t * c.vocab + j]).abs();
                        if d > worst {
                            worst = d;
                            worst_at = (t, j);
                        }
                    }
                    println!(
                        "  step {}  vmlm = {}",
                        t,
                        got.iter().map(|x| format!("{:+.6}", x)).collect::<Vec<_>>().join(" ")
                    );
                    println!(
                        "           ref  = {}",
                        (0..c.vocab)
                            .map(|j| format!("{:+.6}", ref_lg[t * c.vocab + j]))
                            .collect::<Vec<_>>()
                            .join(" ")
                    );
                    got_all.extend_from_slice(&got);
                }
                println!();
                println!("max |vmlm - torch(f64)| = {:.3e} at step {} elem {}", worst, worst_at.0, worst_at.1);
                let ok = worst < 1e-4;
                println!("forward verdict: {}", if ok { "PASS" } else { "FAIL" });
                // Persist the logits so a later check can reuse them.
                std::fs::create_dir_all(format!("{}/grads", dir)).ok();
                write_bin(&format!("{}/grads/vmlm_logits_all.bin", dir), &got_all);
                std::process::exit(if ok { 0 } else { 1 });
            }

            // ================= gradient dump =================
            "grad" => {
                // lr = 0 freezes the parameters, so the gradient slots hold the
                // gradients of the ORIGINAL parameters (AdamW with lr=0 drops the
                // decay term and adds 0/sqrt(v)+eps -- i.e. nothing).
                if has_slot(&mut vm, "hparams") {
                    let hp = vm.find_slot("hparams");
                    *vm.slot_mut(hp) = 0.0;
                }
                vm.run();
                let per = step_cross_entropies(&mut vm, &c);
                let mean: f32 = per.iter().sum::<f32>() / c.t as f32;
                let sum_: f32 = per.iter().sum();
                println!("per-step CE   = {:?}", per.iter().map(|x| format!("{:.8}", x)).collect::<Vec<_>>());
                println!("mean CE over {} steps = {:.8}", c.t, mean);
                println!("sum  CE over {} steps = {:.8}   <- the SUM is what the backward differentiates", c.t, sum_);
                println!("(engine vm.loss = {:.8} -- that is step 0's CE, not the mean)", vm.loss);
                for t in 0..c.t {
                    let lg = read_slot(&mut vm, &format!("logits{}", t));
                    println!(
                        "  step {} logits = {:?}",
                        t,
                        lg.iter().map(|x| format!("{:.6}", x)).collect::<Vec<_>>()
                    );
                }
                let out_dir = format!("{}/grads", dir);
                std::fs::create_dir_all(&out_dir).ok();
                for p in &params {
                    let g = read_slot(&mut vm, &format!("g_{}", p));
                    write_bin(&format!("{}/ana_{}.bin", out_dir, p), &g);
                }
                write_bin(&format!("{}/vmlm_loss.bin", out_dir), &[sum_]);
                write_bin(&format!("{}/vmlm_mean_ce.bin", out_dir), &[mean]);
                // Record T and the targets actually used, so the checker can
                // never silently compare against a stale dump from another T.
                std::fs::write(format!("{}/vmlm_T.txt", out_dir), format!("{}", c.t)).unwrap();
                let mut tb: Vec<u8> = Vec::new();
                for t in 0..c.t {
                    let yv = read_slot(&mut vm, &format!("y{}", t));
                    tb.extend_from_slice(&(yv[0] as i32).to_le_bytes());
                }
                std::fs::write(format!("{}/vmlm_y.bin", out_dir), &tb).unwrap();
                println!("analytic gradients written to {}/ana_*.bin ({} params)", out_dir, params.len());
            }

            // ================= training =================
            _ => {
                // Warm-up run: the first run() may fault in pages and prime
                // caches; excluding it keeps ms/step representative.
                vm.run();
                let mut losses: Vec<f32> = Vec::with_capacity(steps);

                let wall0 = std::time::Instant::now();
                let cpu0 = cpu_seconds();
                for i in 0..steps {
                    vm.run();
                    let l = mean_ce(&mut vm, &c);
                    losses.push(l);
                    if i < 5 || (i + 1) % (steps / 10).max(1) == 0 {
                        println!("  step {:>5}  loss = {:.8}", i + 1, l);
                    }
                }
                let wall = wall0.elapsed().as_secs_f64();
                let cpu = cpu_seconds() - cpu0;

                println!();
                println!("--- loss curve ({} steps) ---", steps);
                println!(
                    "first = {:.8}   last = {:.8}   min = {:.8}",
                    losses[0],
                    losses[steps - 1],
                    losses.iter().cloned().fold(f32::INFINITY, f32::min)
                );
                let mut dec = 0usize;
                let mut inc = 0usize;
                for w in losses.windows(2) {
                    if w[1] < w[0] {
                        dec += 1;
                    } else if w[1] > w[0] {
                        inc += 1;
                    }
                }
                println!(
                    "decreasing transitions = {} / {}  (increasing = {})",
                    dec,
                    steps - 1,
                    inc
                );
                println!(
                    "loss: {:?}",
                    losses.iter().map(|x| (x * 1e6).round() / 1e6).collect::<Vec<_>>()
                );

                println!();
                println!("--- resource metrics ---");
                println!(
                    "wall time total = {:.4} s  ({:.6} ms/step)",
                    wall,
                    wall * 1e3 / steps as f64
                );
                if cpu.is_nan() {
                    println!("cpu time: GetProcessTimes unavailable on this host");
                } else {
                    println!(
                        "cpu  time total = {:.4} s  ({:.6} ms/step)",
                        cpu,
                        cpu * 1e3 / steps as f64
                    );
                    println!(
                        "cpu/wall = {:.1}%  (the engine is single-threaded; the ceiling is 100%)",
                        100.0 * cpu / wall.max(1e-9)
                    );
                }
                println!("instr/step = {}", vm.n_ins);

                let mut txt = String::new();
                for (i, l) in losses.iter().enumerate() {
                    txt.push_str(&format!("{} {:.8}\n", i + 1, l));
                }
                std::fs::write(format!("{}/loss_curve_t{}.txt", dir, c.t), txt).unwrap();
                println!("loss curve written to {}/loss_curve_t{}.txt", dir, c.t);
            }
        }
    }
}
