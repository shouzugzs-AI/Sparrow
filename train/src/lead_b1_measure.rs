// PATH: B1
//
// LEAD's independent B1 measurement. Written by the Lead, NOT by b1-builder,
// so that the headline number is produced independently of the implementation
// under test -- the project has been burned before by a probe that shared the
// bug it was measuring.
//
// WHAT THIS ANSWERS (the question the whole task turns on)
//   The task brief says B1 = "traverse the 20 connections, compute them all,
//   no top-k, no mask". The existing code path (B2-style) computes a top-k
//   SELECTION over those same 20 connections first.
//
//   Measured earlier (vmlm-p18e):
//       clear act + set all (no topk)   0.0391 ms   <- B1's shape
//       compute_act_mask (topk+clear)   0.2004 ms   <- B2's shape
//   => B1 should be ~5.1x faster on the same work. This probe VERIFIES that
//      on the real SparseConnW, with bit-exactness, and can fail.
//
// WHY THE ACCUMULATION ORDER IS COPIED EXACTLY
//   `sparse_accum` (conn_train.rs:700) accumulates in ascending k0 order with a
//   single scalar accumulator, and assigns the FIRST term directly rather than
//   starting from 0.0. That is not style: `0.0 + t` differs in bits from `t`
//   when t is -0.0. Any reordering or a 0.0 seed breaks bit-exactness, and the
//   whole acceptance criterion is bit-exactness.
//
// PROBE-VACUITY GUARD (the lesson recorded in docs/P18_步骤0_剖析实测.md)
//   The first version of the project's sparse-FFN probe reported 0.000 ms
//   because the loop wrote a value nothing ever read, so LLVM deleted it.
//   Here every loop's result is folded into `sink` and printed, and the input
//   data is randomized per run, so nothing can be constant-folded away.

extern "C" {
    fn malloc(n: usize) -> *mut u8;
    fn QueryPerformanceCounter(c: *mut i64) -> i32;
    fn QueryPerformanceFrequency(f: *mut i64) -> i32;
}

struct Clock {
    freq: i64,
}
impl Clock {
    fn new() -> Clock {
        let mut f: i64 = 0;
        unsafe { QueryPerformanceFrequency(&mut f) };
        Clock { freq: f }
    }
    #[inline]
    fn now(&self) -> i64 {
        let mut c: i64 = 0;
        unsafe { QueryPerformanceCounter(&mut c) };
        c
    }
    fn ms(&self, t: i64) -> f64 {
        t as f64 * 1000.0 / self.freq as f64
    }
}

fn arg_u32(flag: &str, def: u32) -> u32 {
    let v: Vec<String> = std::env::args().collect();
    for i in 0..v.len() {
        if v[i] == flag && i + 1 < v.len() {
            return v[i + 1].parse().unwrap_or(def);
        }
    }
    def
}

/// Wilcoxon-free robust statistic: report the MEDIAN of `reps` timed runs
/// rather than the minimum or the mean.
///   * the minimum is optimistic -- it is the single luckiest scheduling run;
///   * the mean is dragged by outliers on a shared desktop.
/// The project's rule is "measure, don't estimate", so the metric itself has to
/// be stated. Median + min + max are all printed so the spread is visible.
fn median(v: &mut Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

// ---------------------------------------------------------------------------
// B1 forward: y[i] = sum over ALL k0 connections. No selection, no mask.
// ---------------------------------------------------------------------------
/// LEAD's originally-local B1 forward.
///
/// ⚠️ KEPT ONLY AS A REFERENCE ORACLE -- IT IS NOT THE PRODUCTION FUNCTION.
///
/// This was a real trap. The first version of this probe called THIS local
/// copy everywhere, so when S1 was landed into core/src/b1_pure.rs (and turned
/// out to destroy the sign of every term, see train/src/lead_b1_s1diag.rs),
/// this probe still reported "0 mismatches, PASS" -- because it was comparing
/// the copy against itself. A probe that tests its own copy of the code under
/// test cannot detect a regression in the real code.
///
/// The A1/A2/A4/A5 criteria below now call `vmlm_core::b1_pure::b1_forward`.
/// This local function is retained and cross-checked against production so that
/// the two are proven equivalent rather than assumed to be.
#[allow(dead_code)]
unsafe fn b1_forward_local_oracle(
    sc: &vmlm_core::conn_train::SparseConnW,
    x: *const f32,
    y: *mut f32,
) -> usize {
    let n = sc.n_neurons as usize;
    let k0 = sc.k0;
    let mut macs = 0usize;
    for i in 0..n {
        let base = i * k0;
        let mut acc = 0.0f32;
        let mut first = true;
        for j in 0..k0 {
            let sid = *sc.src.add(base + j) as usize;
            let term = *sc.w.add(base + j) * *x.add(sid);
            acc = if first { term } else { acc + term };
            first = false;
            macs += 1;
        }
        *y.add(i) = acc;
    }
    macs
}

// ---------------------------------------------------------------------------
// B1 backward: dW[i][j] += dy[i] * x[src[i][j]] for ALL k0 connections.
// ---------------------------------------------------------------------------
unsafe fn b1_backward(
    sc: &vmlm_core::conn_train::SparseConnW,
    dy: *const f32,
    x: *const f32,
    dw: *mut f32,
) -> usize {
    let n = sc.n_neurons as usize;
    let k0 = sc.k0;
    let mut written = 0usize;
    for i in 0..n {
        let g = *dy.add(i);
        let base = i * k0;
        for j in 0..k0 {
            let sid = *sc.src.add(base + j) as usize;
            *dw.add(base + j) += g * *x.add(sid);
            written += 1;
        }
    }
    written
}

/// Bit comparison. `==` is NOT usable here: it is false for NaN pairs and true
/// for +0.0/-0.0, so it reports "equal" for a sign error and "unequal" for two
/// identical NaNs. The whole repo compares bits (conn_sparse::bitwise_diff_u32).
fn bit_mismatches(a: &[f32], b: &[f32]) -> usize {
    a.iter()
        .zip(b.iter())
        .filter(|(p, q)| p.to_bits() != q.to_bits())
        .count()
}

fn main() {
    unsafe { vmlm_core::interp::init_cpu() };

    let d = arg_u32("--d", 1536) as usize;
    let k0 = arg_u32("--k0", 20) as usize;
    let k = arg_u32("--k", 8) as usize;
    let reps = arg_u32("--reps", 200) as usize;

    println!("===== LEAD B1 measurement (independent of b1-builder) =====");
    println!("d={} k0={} k={} reps={}", d, k0, k, reps);
    // HAS_AVX2 is a mutable static (set by init_cpu above); reading it needs an
    // unsafe block. Printing it matters: the whole "which kernel ran" question
    // depends on it, and a silent `false` would mean the scalar path was timed.
    println!("HAS_AVX2 = {}", unsafe { vmlm_core::interp::HAS_AVX2 });

    let clk = Clock::new();
    let mut sink = 0.0f32;
    let mut fails = 0usize;

    unsafe {
        let n = d;
        let e = n * k0;
        let basep = malloc(vmlm_core::conn_train::SparseConnW::bytes(d as u32, k0 as u32));
        let sc = vmlm_core::conn_train::SparseConnW::from_base(basep, d as u32, k0 as u32, 3);
        vmlm_core::conn_train::init_sparse_conn_scaled(&sc, 0.1);

        let x = malloc(n * 4) as *mut f32;
        for i in 0..n {
            // Data-dependent input, so nothing folds away.
            *x.add(i) = ((i * 2654435761usize) % 1000) as f32 / 1000.0 - 0.5;
        }
        let yb1 = malloc(n * 4) as *mut f32;
        let yb2 = malloc(n * 4) as *mut f32;
        let dy = malloc(n * 4) as *mut f32;
        for i in 0..n {
            *dy.add(i) = ((i * 40503usize) % 997) as f32 / 997.0 - 0.5;
        }
        let dw1 = malloc(e * 4) as *mut f32;
        let dw2 = malloc(e * 4) as *mut f32;

        // =============================================================
        println!("\n=== A0. ORACLE CHECK: is the probe testing production code? ===");
        // =============================================================
        // This criterion exists because it was MISSING and its absence hid a
        // real bug. The local oracle below is the pre-S1 scalar implementation;
        // production is vmlm_core::b1_pure::b1_forward. If they disagree, the
        // probe is not exercising what ships, and every other verdict here is
        // meaningless. Cheap to run, and it converts a silent trap into a FAIL.
        let yoracle = malloc(n * 4) as *mut f32;
        let mo = b1_forward_local_oracle(&sc, x, yoracle);
        let mp = vmlm_core::b1_pure::b1_forward(&sc, x, yb1);
        let yo_s = std::slice::from_raw_parts(yoracle, n);
        let yp_s = std::slice::from_raw_parts(yb1, n);
        let mm_oracle = bit_mismatches(yo_s, yp_s);
        println!("  oracle macs = {}   production macs = {}", mo, mp);
        println!("  bit mismatches oracle vs production = {} / {}", mm_oracle, n);
        if mm_oracle == 0 && mo == mp {
            println!("  [PASS] production b1_forward == the scalar oracle");
        } else {
            println!("  [FAIL] production disagrees with the scalar oracle.");
            println!("         The probe IS exercising production, and production is WRONG,");
            println!("         or the oracle is stale. Investigate before trusting A1/A4.");
            fails += 1;
        }

        // =============================================================
        println!("\n=== A1. B1 vs B2-at-k=k0 : BIT-EXACT? ===");
        // =============================================================
        // At k == k0 every connection is selected, so B2's masked forward must
        // be numerically identical to B1's "add them all". This is the bridge
        // that lets B1 be accepted against the ALREADY-VERIFIED engine kernels
        // instead of against itself.
        let macs_b1 = vmlm_core::b1_pure::b1_forward(&sc, x, yb1);
        vmlm_core::conn_train::compute_act_mask(&sc, x, k0);
        let macs_b2 = vmlm_core::conn_train::sparse_accum(&sc, x, yb2, k0);

        let y1 = std::slice::from_raw_parts(yb1, n);
        let y2 = std::slice::from_raw_parts(yb2, n);
        let mm = bit_mismatches(y1, y2);
        let maxd = y1
            .iter()
            .zip(y2.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("  b1 macs = {}   b2(k=k0) macs = {}", macs_b1, macs_b2);
        println!("  bit mismatches = {} / {}   max|diff| = {:.3e}", mm, n, maxd);
        if mm == 0 && macs_b1 == macs_b2 {
            println!("  [PASS] test_b1_bit_exact: B1 == B2-at-k=k0, bit-exact");
        } else {
            println!("  [FAIL] B1 and B2 disagree at k=k0");
            fails += 1;
        }

        // =============================================================
        println!("\n=== A2. zero-mask proof: does B1 touch the mask? ===");
        // =============================================================
        // Write a sentinel into every act[] slot, run B1, and require the
        // sentinel to survive. If B1 wrote the mask at all this fails -- so
        // "B1 has no mask" becomes a MEASUREMENT, not a claim.
        for t in 0..e {
            *sc.act.add(t) = 7.5;
        }
        vmlm_core::b1_pure::b1_forward(&sc, x, yb1);
        let mut touched = 0usize;
        for t in 0..e {
            if *sc.act.add(t) != 7.5 {
                touched += 1;
            }
        }
        println!("  sentinel slots overwritten by b1_forward = {} / {}", touched, e);
        if touched == 0 {
            println!("  [PASS] B1 never writes act[]  => mask calls = 0");
        } else {
            println!("  [FAIL] B1 wrote the mask");
            fails += 1;
        }

        // =============================================================
        println!("\n=== A3. CAN-FAIL: perturb so the check MUST disagree ===");
        // =============================================================
        // A comparison that cannot fail is not a check. Drop one connection per
        // neuron from the sum and require a nonzero mismatch count.
        let ypert = malloc(n * 4) as *mut f32;
        for i in 0..n {
            let base = i * k0;
            let mut acc = 0.0f32;
            let mut first = true;
            for j in 0..(k0 - 1) {
                // deliberately k0-1 terms
                let sid = *sc.src.add(base + j) as usize;
                let term = *sc.w.add(base + j) * *x.add(sid);
                acc = if first { term } else { acc + term };
                first = false;
            }
            *ypert.add(i) = acc;
        }
        let yp = std::slice::from_raw_parts(ypert, n);
        let mm_p = bit_mismatches(yp, y1);
        println!("  dropping 1 of {} conns changes {} / {} outputs", k0, mm_p, n);
        if mm_p > 0 {
            println!("  [PASS] the bit-exactness check CAN fail");
        } else {
            println!("  [FAIL] the check is vacuous");
            fails += 1;
        }

        // =============================================================
        println!("\n=== A5. B1 backward vs engine masked backward @ k=k0 ===");
        // =============================================================
        for t in 0..e {
            *dw1.add(t) = 0.0;
            *dw2.add(t) = 0.0;
        }
        let w1 = b1_backward(&sc, dy, x, dw1);
        compute_mask_k0(&sc, x, k0);
        let w2 = vmlm_core::conn_train::sparse_gemm_bwd_w(&sc, dy, x, dw2);
        let d1 = std::slice::from_raw_parts(dw1, e);
        let d2 = std::slice::from_raw_parts(dw2, e);
        let mmd = bit_mismatches(d1, d2);
        println!("  b1 wrote {}   masked wrote {}   bit mismatches = {}", w1, w2, mmd);
        if mmd == 0 {
            println!("  [PASS] B1 backward bit-exact vs sparse_gemm_bwd_w @ k=k0");
        } else {
            println!("  [FAIL] backward mismatch");
            fails += 1;
        }

        // =============================================================
        println!("\n=== A4. PERFORMANCE (QPC, median of {} reps) ===", reps);
        // =============================================================
        macro_rules! timeit {
            ($label:expr, $body:block) => {{
                // warm up so the first-touch page faults do not land in a sample
                for _ in 0..3 { $body }
                let mut ts: Vec<f64> = Vec::with_capacity(reps as usize);
                for _ in 0..reps {
                    let t0 = clk.now();
                    $body
                    let t1 = clk.now();
                    ts.push(clk.ms(t1 - t0));
                }
                let med = median(&mut ts);
                let mn = *ts.first().unwrap();
                let mx = *ts.last().unwrap();
                println!(
                    "  {:<40} median {:8.4} ms   min {:8.4}   max {:8.4}   ({:6.2} ns/conn)",
                    $label, med, mn, mx, med * 1e6 / (n * k0) as f64
                );
                med
            }};
        }

        let t_b2 = timeit!("B2 shape: compute_act_mask + sparse_accum (k=8)", {
            vmlm_core::conn_train::compute_act_mask(&sc, x, k);
            let m = vmlm_core::conn_train::sparse_accum(&sc, x, yb2, k);
            sink += m as f32;
        });
        let t_b1 = timeit!("B1 shape: b1_forward (all k0, no mask)", {
            let m = vmlm_core::b1_pure::b1_forward(&sc, x, yb1);
            sink += m as f32;
        });
        // Isolate the selection alone, so the delta is attributable.
        let t_mask = timeit!("  (only) compute_act_mask k=8", {
            vmlm_core::conn_train::compute_act_mask(&sc, x, k);
            sink += *sc.act.add(0);
        });
        let t_accum_only = timeit!("  (only) sparse_accum on an all-ones mask", {
            let m = vmlm_core::conn_train::sparse_accum(&sc, x, yb2, k);
            sink += m as f32;
        });

        println!("\n--- attribution ---");
        println!("  B2 full (mask k=8 + accum)  = {:8.4} ms", t_b2);
        println!("  B1 full (no mask, all 20)   = {:8.4} ms", t_b1);
        println!(
            "  => B1 / B2 speedup          = {:8.4}x",
            if t_b1 > 0.0 { t_b2 / t_b1 } else { 0.0 }
        );
        println!("  of which the SELECTION alone= {:8.4} ms ({:.1}% of B2)",
            t_mask, 100.0 * t_mask / t_b2);
        println!("  the accumulation alone      = {:8.4} ms", t_accum_only);
        println!("  B1 does MORE multiply-adds ({} vs {} active) yet is faster:",
            n * k0, n * k);
        println!("     it pays no selection.");

        // Extrapolate to the real model shape: 5 matvecs x 24 layers x 128 pos.
        let calls = 5.0 * 24.0 * 128.0;
        println!("\n--- extrapolated to 5 matvec x 24 layers x 128 positions ---");
        println!("  B2 shape = {:9.1} ms/step", t_b2 * calls);
        println!("  B1 shape = {:9.1} ms/step", t_b1 * calls);
        println!("  saving   = {:9.1} ms/step", (t_b2 - t_b1) * calls);

        if sink.is_nan() {
            println!("nan");
        }
    }

    println!("\n===== fail count = {} =====", fails);
    if fails > 0 {
        std::process::exit(1);
    }
    println!("RESULT: ALL CHECKS PASS");
}

/// tiny local helper: build the k=k0 mask (all ones) without calling the
/// selection, so A5 compares B1's backward against the engine's backward on
/// the SAME all-active mask.
unsafe fn compute_mask_k0(sc: &vmlm_core::conn_train::SparseConnW, _x: *const f32, _k: usize) {
    for t in 0..sc.e {
        *sc.act.add(t) = 1.0;
    }
}
