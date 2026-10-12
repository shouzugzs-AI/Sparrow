// A2 / P8: RNN training loop (BPTT) driver.
//
// Responsibilities:
//   1. Train  : run the generated `bptt_tT.vmb` N times; record a loss curve.
//   2. GradChk: dump the analytic gradients written by the backward pass so they
//               can be compared against PyTorch autograd (see check_grad.py).
//   3. Reset  : restore initial weights so gradcheck starts from the exact
//               parameters PyTorch referenced.
//
// The engine JITs the instruction stream once; each training step re-runs the
// same code and only mutates the constant pool (whitepaper requirement 2).
//
// Resource metrics: wall time per step and process CPU time (single thread,
// so CPU% <= 100%).
use vmlm_core::interp::Vm;

extern "C" {
    fn malloc(n: usize) -> *mut u8;
}

const PARAMS: [&str; 18] = [
    "W_r", "W_k", "W_v", "W_o", "W1d", "W2d", "w0", "xr", "xk", "xv", "xw",
    "ln1_g", "ln1_b", "ln_t_g", "ln_t_b", "ln_out_g", "ln_out_b", "E",
];
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

unsafe fn slot_n(vm: &mut Vm, name: &str) -> usize {
    let si = vm.find_slot(name);
    let (r, c) = vm.slot_shape(si);
    r * c
}

unsafe fn read_slot(vm: &mut Vm, name: &str) -> Vec<f32> {
    let si = vm.find_slot(name);
    let n = slot_n(vm, name);
    let p = vm.slot_mut(si);
    (0..n).map(|i| *p.add(i)).collect()
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

/// Mean cross-entropy over the T steps, recomputed from the per-step logits.
///
/// Why not use `vm.loss`: CE_BWD stores the loss in the engine's `vm.loss`
/// field, and the backward pass runs from step T-1 down to 0 -- so `vm.loss`
/// ends up holding **step 0's** CE, not the last step's, and certainly not the
/// mean. The generator surfaces each step's logits as `loss_out{t}`, so we
/// recompute the loss here to get a quantity that is actually comparable to
/// PyTorch's CrossEntropyLoss (which averages over the sequence).
unsafe fn mean_ce(vm: &mut Vm, t_steps: usize) -> f32 {
    let mut total = 0.0f32;
    for t in 0..t_steps {
        let lg = read_slot(vm, &format!("loss_out{}", t));
        let yv = read_slot(vm, &format!("y{}", t));
        let tgt = yv[0] as usize;
        let mx = lg.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in &lg {
            sum += (v - mx).exp();
        }
        let p = (lg[tgt] - mx).exp() / sum;
        total += -p.ln();
    }
    total / t_steps as f32
}

/// Process CPU time (user+kernel) in seconds.
///
/// We read the value the OS already exposes rather than measuring inside the
/// timed region, so the number reflects real CPU consumption (and can exceed
/// wall time if the process were multithreaded; here it should track wall time
/// at ~100% since the engine is single-threaded).
#[cfg(windows)]
fn cpu_seconds() -> f64 {
    #[repr(C)]
    #[derive(Default, Clone, Copy)]
    struct FileTime {
        lo: u32,
        hi: u32,
    }
    // GetProcessTimes lives in kernel32 and uses the Win64 calling convention.
    // Declaring it as `extern "system"` is what makes the FFI match; a plain
    // `extern "C"` would still link on x86_64 but is the wrong contract to state.
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
        let mut c = FileTime::default();
        let mut e = FileTime::default();
        let mut k = FileTime::default();
        let mut u = FileTime::default();
        if GetProcessTimes(GetCurrentProcess(), &mut c, &mut e, &mut k, &mut u) == 0 {
            return f64::NAN;
        }
        // FILETIME is in 100-nanosecond units.
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
        .unwrap_or_else(|| "examples/rnn/train".into());
    let mode = args.get(2).cloned().unwrap_or_else(|| "train".into());
    let steps: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(300);
    let lr: f32 = args
        .get(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3e-3);
    // T is inferred from which bptt_t*.vmb exists / is given
    let t: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(2);

    unsafe { vmlm_core::interp::init_cpu() };

    let vmb = format!("{}/bptt_t{}.vmb", dir, t);
    println!("=== P8 / A2 : RNN training loop (BPTT) ===");
    println!("vmb      = {}", vmb);
    println!("mode     = {}  steps = {}  lr = {}  T = {}", mode, steps, lr, t);

    unsafe {
        let buf = load_vmb(&vmb);
        let mut vm = Vm::load(buf);

        // -------- snapshot the initial parameters (for Reset / gradcheck) --------
        let mut snap: Vec<(&str, Vec<f32>)> = Vec::new();
        if mode != "run" {
            for p in PARAMS {
                snap.push((p, read_slot(&mut vm, p)));
            }
        }

        // -------- hyper-parameters: hparams = [lr, b1, b2, eps, wd] --------
        // (raw "run" debug mode skips this so a truncated program can still run)
        if mode != "run" {
            write_slot(&mut vm, "hparams", &[lr, 0.9, 0.999, 1e-8, 1e-4]);
            // keep clipping off so gradients are not rescaled (gradcheck needs raw grads)
            write_scalar(&mut vm, "clip_norm", 1e9);
        }

        match mode.as_str() {
            // ================= raw run (debug: no gradient slot access) =================
            "run" => {
                vm.run();
                println!("ok: mean CE = {:.8}", mean_ce(&mut vm, t));
            }

            // ================= dump arbitrary slots (debug) =================
            // usage: vmlm-rnn-train <dir> dump <steps> <lr> <T> slot1 slot2 ...
            "dump" => {
                vm.run();
                println!("mean CE = {:.8}", mean_ce(&mut vm, t));
                // slot names follow the positional args: argv[6..]
                for name in args.iter().skip(6) {
                    let si = vm.find_slot(name);
                    let (r, c) = vm.slot_shape(si);
                    let p = vm.slot_mut(si);
                    let v: Vec<f32> = (0..r * c).map(|i| *p.add(i)).collect();
                    println!(
                        "{} [{}x{}] = {:?}",
                        name, r, c,
                        v.iter().map(|x| format!("{:.8}", x)).collect::<Vec<_>>()
                    );
                }
            }

            // ================= gradient dump =================
            "grad" => {
                vm.run();
                let l = mean_ce(&mut vm, t);
                println!("mean CE (recomputed over {} steps) = {:.8}", t, l);
                println!("(engine vm.loss = {:.8} -- this is step 0's CE, not the mean)", vm.loss);
                for i in 0..t {
                    let lg = read_slot(&mut vm, &format!("loss_out{}", i));
                    println!(
                        "  step {} logits = {:?}",
                        i,
                        lg.iter().map(|x| format!("{:.6}", x)).collect::<Vec<_>>()
                    );
                }
                // dump analytic gradients in the same flat order PyTorch uses
                let out_dir = format!("{}/grads", dir);
                std::fs::create_dir_all(&out_dir).ok();
                for p in PARAMS {
                    let g = read_slot(&mut vm, &format!("g_{}", p));
                    let mut bytes = Vec::with_capacity(g.len() * 4);
                    for v in &g {
                        bytes.extend_from_slice(&v.to_le_bytes());
                    }
                    std::fs::write(format!("{}/ana_{}.bin", out_dir, p), &bytes).unwrap();
                }
                println!("analytic gradients written to {}/ana_*.bin", out_dir);
                // surface every step's logits + the recomputed loss
                for i in 0..t {
                    let lg = read_slot(&mut vm, &format!("loss_out{}", i));
                    let mut lb = Vec::new();
                    for v in &lg {
                        lb.extend_from_slice(&v.to_le_bytes());
                    }
                    std::fs::write(format!("{}/vmlm_logits{}.bin", out_dir, i), &lb).unwrap();
                }
                let mut lossb = Vec::new();
                lossb.extend_from_slice(&l.to_le_bytes());
                std::fs::write(format!("{}/vmlm_loss.bin", out_dir), &lossb).unwrap();
                // record T and the targets actually used, so the checker can
                // never silently compare against a stale/other-T dump
                std::fs::write(format!("{}/vmlm_T.txt", out_dir), format!("{}", t)).unwrap();
                let mut tb = Vec::new();
                for i in 0..t {
                    let yv = read_slot(&mut vm, &format!("y{}", i));
                    tb.extend_from_slice(&(yv[0] as i32).to_le_bytes());
                }
                std::fs::write(format!("{}/vmlm_y.bin", out_dir), &tb).unwrap();
            }

            // ================= training =================
            _ => {
                let mut losses = Vec::with_capacity(steps);
                // Warm up: the first `vm.run()` may fault in pages / prime caches.
                // Excluding it keeps ms/step representative of steady state.
                vm.run();
                let _ = mean_ce(&mut vm, t);
                losses.clear();

                let wall0 = std::time::Instant::now();
                let cpu0 = cpu_seconds();
                for i in 0..steps {
                    vm.run();
                    let l = mean_ce(&mut vm, t);
                    losses.push(l);
                    if i < 5 || (i + 1) % (steps / 10).max(1) == 0 {
                        println!("  step {:>5}  loss = {:.6}", i + 1, l);
                    }
                }
                let wall = wall0.elapsed().as_secs_f64();
                let cpu = cpu_seconds() - cpu0;

                println!();
                println!("--- loss curve ({} steps) ---", steps);
                println!(
                    "first = {:.6}   last = {:.6}   min = {:.6}",
                    losses[0],
                    losses[steps - 1],
                    losses.iter().cloned().fold(f32::INFINITY, f32::min)
                );
                // monotonicity: count strict decreases
                let mut dec = 0usize;
                let mut inc = 0usize;
                for w in losses.windows(2) {
                    if w[1] < w[0] {
                        dec += 1;
                    } else if w[1] > w[0] {
                        inc += 1;
                    }
                }
                println!("decreasing transitions = {} / {}  (increasing = {})",
                         dec, steps - 1, inc);
                println!("loss: {:?}",
                         losses.iter().map(|x| (x * 1e4).round() / 1e4).collect::<Vec<_>>());

                println!();
                println!("--- resource metrics ---");
                println!("wall time total = {:.4} s  ({:.4} ms/step)",
                         wall, wall * 1e3 / steps as f64);
                if cpu.is_nan() {
                    println!("cpu time: GetProcessTimes unavailable on this host");
                } else {
                    println!("cpu  time total = {:.4} s  ({:.4} ms/step)",
                             cpu, cpu * 1e3 / steps as f64);
                    println!("cpu/wall = {:.1}%  (single-threaded engine; ceiling is 100%)",
                             100.0 * cpu / wall.max(1e-9));
                }
                // Also report the pure engine throughput, excluding the driver's
                // own Python-free loss bookkeeping (mean_ce reads T slots).
                println!("instr/step      = {} (from the assembled program)",
                         vm.n_ins);

                // write the curve for plotting / reporting
                let mut txt = String::new();
                for (i, l) in losses.iter().enumerate() {
                    txt.push_str(&format!("{} {:.6}\n", i + 1, l));
                }
                std::fs::write(format!("{}/loss_curve_t{}.txt", dir, t), txt).unwrap();
                println!("loss curve written to {}/loss_curve_t{}.txt", dir, t);
            }
        }
        let _ = snap;
    }
}
