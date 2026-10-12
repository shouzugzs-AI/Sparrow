// PATH: B1
//
// PRODUCTION epsilon acceptance: the SHIPPED b1_forward (SIMD reduction + u32
// mirror) vs the SHIPPED scalar reference (b1_forward_scalar_reference).
//
// WHY THIS REPLACES THE BIT-EXACT CRITERION
//   The user ruled: "不需要 bit-exact ... 计算本质一样就不必这么追究了".
//   So the acceptance bar is now:
//     (1) DETERMINISM   same input -> same output, bit-identical, repeatably
//     (2) ACCURACY      |simd - scalar| <= epsilon, where epsilon is chosen
//                       FROM the measured data (not picked in advance)
//   The old bit-exact criteria are DEMOTED to "reference", not deleted: this
//   probe reports the bit-difference count so a regression back toward large
//   divergence is still visible.
//
// WHAT IS MEASURED
//   - one layer, per-neuron |diff| against the scalar reference
//   - the same through 24 layers of a residual stream (compounding)
//   - determinism over repeated independent calls
//   - the u32 mirror consistency (src vs src_u32), which is a NEW invariant
//     introduced by the mirror and must be asserted, not assumed
//
// USAGE: vmlm-b1-prod-eps.exe [--d 1536] [--k0 20] [--layers 24]

extern "C" {
    fn malloc(n: usize) -> *mut u8;
    fn QueryPerformanceCounter(c: *mut i64) -> i32;
    fn QueryPerformanceFrequency(f: *mut i64) -> i32;
}

use vmlm_core::conn_train::SparseConnW;

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

fn main() {
    unsafe { vmlm_core::interp::init_cpu() };
    let d = arg_u32("--d", 1536) as usize;
    let k0 = arg_u32("--k0", 20) as usize;
    let layers = arg_u32("--layers", 24) as usize;

    println!("===== PRODUCTION epsilon acceptance (shipped b1_forward) =====");
    println!("d={} k0={} layers={}", d, k0, layers);
    println!("HAS_AVX2 = {}", unsafe { vmlm_core::interp::HAS_AVX2 });

    let n = d;
    let e = n * k0;
    let mut fails = 0usize;

    unsafe {
        let base = malloc(SparseConnW::bytes(d as u32, k0 as u32));
        let sc = SparseConnW::from_base(base, d as u32, k0 as u32, 7);
        vmlm_core::conn_train::init_sparse_conn_scaled(&sc, 0.144);
        // init_sparse_conn writes src AND the mirror; assert they agree.
        sc.assert_src_u32_consistent();
        println!("\n  [PASS] src/src_u32 mirror consistent after init ({} entries)", e);

        let x = malloc(n * 4) as *mut f32;
        for i in 0..n {
            *x.add(i) = (((i * 22695477usize) % 1000) as f32 / 1000.0) - 0.5;
        }
        let y_simd = malloc(n * 4) as *mut f32;
        let y_ref = malloc(n * 4) as *mut f32;

        // ---- (1) ONE LAYER: shipped vs scalar reference ----
        let m_simd = vmlm_core::b1_pure::b1_forward(&sc, x, y_simd);
        let m_ref = vmlm_core::b1_pure::b1_forward_scalar_reference(&sc, x, y_ref);

        let mut max_d = 0.0f32;
        let mut sum_d = 0.0f64;
        let mut bits_same = 0usize;
        let mut max_rel = 0.0f32;
        for i in 0..n {
            let a = *y_simd.add(i);
            let b = *y_ref.add(i);
            let dd = (a - b).abs();
            if a.to_bits() == b.to_bits() {
                bits_same += 1;
            }
            if dd > max_d {
                max_d = dd;
            }
            sum_d += dd as f64;
            if b != 0.0 {
                let r = dd / b.abs();
                if r > max_rel {
                    max_rel = r;
                }
            }
        }
        println!("\n--- (1) ONE LAYER: shipped SIMD vs scalar reference ---");
        println!("  macs: simd={} ref={} (must both be n*k0={})", m_simd, m_ref, e);
        println!("  max|diff|      = {:.6e}", max_d);
        println!("  mean|diff|     = {:.6e}", sum_d / n as f64);
        println!("  max relative   = {:.6e}", max_rel);
        println!("  bit-identical  = {} / {} ({:.1}%)", bits_same, n, 100.0 * bits_same as f64 / n as f64);
        if m_simd != e || m_ref != e {
            println!("  [FAIL] macs count wrong");
            fails += 1;
        }
        // Accuracy criterion: rounding level, measured against the RIGHT scale.
        //
        // ⚠️ THIS BAR WAS WRONG TWICE, and the second failure is instructive.
        //   Attempt 1: per-element relative (|a-b|/|b|). Failed at 8.4e-5.
        //     Wrong because y is a sum of SIGNED terms: when they cancel,
        //     |y| -> 0 while |a-b| stays at the rounding scale of the TERMS.
        //   Attempt 2: ulp of the SUM's magnitude. Failed at 706 ulp.
        //     Same root cause, one level deeper -- the worst element had
        //     |y_ref| = 1.77e-4, i.e. the sum nearly cancelled to zero, so
        //     dividing by its own ulp inflates the ratio without bound.
        //
        //   Both bars punish CANCELLATION, which is a property of the data, not
        //   of the reduction. The correct scale for a summation is the sum of
        //   the TERM MAGNITUDES, S = sum|w_j * x_j|:
        //
        //     a reordered f32 sum differs by at most ~k0 * eps * S
        //     (here 20 * 1.19e-7 * S = 2.4e-6 * S)
        //
        //   whereas a real defect (sign error, wrong index, bad tail) produces
        //   an error of order S itself, i.e. a ratio near 1 -- six orders of
        //   magnitude apart. So |a-b| / S is a SHARP discriminator, and it does
        //   not collapse when the sum cancels.
        let mut worst_ratio = 0.0f32;
        let mut worst_i = 0usize;
        let mut worst_abs = 0.0f32;
        for i in 0..n {
            let a = *y_simd.add(i);
            let b = *y_ref.add(i);
            // S = sum of term magnitudes for this neuron
            let mut s = 0.0f32;
            let base = i * k0;
            for j in 0..k0 {
                let sid = *sc.src.add(base + j) as usize;
                s += (*sc.w.add(base + j) * *x.add(sid)).abs();
            }
            if s <= 0.0 {
                continue;
            }
            let ratio = (a - b).abs() / s;
            if ratio > worst_ratio {
                worst_ratio = ratio;
                worst_i = i;
                worst_abs = (a - b).abs();
            }
        }
        // Bound: k0 * eps, with a modest safety factor. For k0=20:
        //   20 * 1.19e-7 = 2.4e-6. Bar at 1e-5 = ~4x that, still 5 orders of
        //   magnitude below a real bug's signature (ratio ~1).
        let ratio_bar = 1.0e-5f32;
        // Also report the ulp-of-sum figure so the earlier metric stays visible
        // (it is what a reader of the previous report would have seen).
        let mut worst_ulp = 0.0f32;
        for i in 0..n {
            let a = *y_simd.add(i);
            let b = *y_ref.add(i);
            let scale = b.abs().max(1.0e-30);
            let u = (a - b).abs() / (f32::EPSILON * scale);
            if u > worst_ulp {
                worst_ulp = u;
            }
        }
        println!("  max|diff|      = {:.6e}", max_d);
        println!("  mean|diff|     = {:.6e}", sum_d / n as f64);
        println!("  bit-identical  = {} / {} ({:.1}%)", bits_same, n, 100.0 * bits_same as f64 / n as f64);
        println!();
        println!("  --- the two metrics that looked alarming ---");
        println!("  per-element max relative = {:.3e}  (dominated by cancellation)", max_rel);
        println!("  worst ulp of the SUM     = {:.1} ulp  (also inflated by cancellation)", worst_ulp);
        println!();
        println!("  --- the discriminating metric: |diff| vs the TERM-MAGNITUDE scale ---");
        println!("  worst |a-b| / sum|terms| = {:.3e}  (element {}, |diff|={:.3e})",
            worst_ratio, worst_i, worst_abs);
        println!("  theoretical reorder bound = k0 * f32_eps = {} * {:.3e} = {:.3e}",
            k0, f32::EPSILON, k0 as f32 * f32::EPSILON);
        println!("  a real defect (sign/index bug) would give ratio ~1.0 (order of the value)");
        if m_simd != e || m_ref != e {
            println!("  [FAIL] macs count wrong");
            fails += 1;
        }
        if worst_ratio <= ratio_bar {
            println!(
                "  [PASS] accuracy: {:.3e} <= {:.0e}  => rounding level, {}x below a real bug",
                worst_ratio, ratio_bar, 1.0 / worst_ratio
            );
        } else {
            println!("  [FAIL] accuracy: {:.3e} > {:.0e}", worst_ratio, ratio_bar);
            fails += 1;
        }

        // ---- (2) DETERMINISM ----
        println!("\n--- (2) DETERMINISM (acceptance bar) ---");
        let y_again = malloc(n * 4) as *mut f32;
        let mut det = true;
        for _r in 0..8 {
            vmlm_core::b1_pure::b1_forward(&sc, x, y_again);
            for i in 0..n {
                if (*y_again.add(i)).to_bits() != (*y_simd.add(i)).to_bits() {
                    det = false;
                }
            }
        }
        if det {
            println!("  [PASS] 8 repeated calls give bit-identical output");
        } else {
            println!("  [FAIL] NOT deterministic");
            fails += 1;
        }
        // also: a fresh SparseConnW built identically must match (no state leak)
        let base2 = malloc(SparseConnW::bytes(d as u32, k0 as u32));
        let sc2 = SparseConnW::from_base(base2, d as u32, k0 as u32, 7);
        vmlm_core::conn_train::init_sparse_conn_scaled(&sc2, 0.144);
        let y2 = malloc(n * 4) as *mut f32;
        vmlm_core::b1_pure::b1_forward(&sc2, x, y2);
        let mut cross = 0usize;
        for i in 0..n {
            if (*y2.add(i)).to_bits() != (*y_simd.add(i)).to_bits() {
                cross += 1;
            }
        }
        if cross == 0 {
            println!("  [PASS] an independently constructed identical struct matches bit-for-bit");
        } else {
            println!("  [FAIL] cross-construction mismatch = {} / {}", cross, n);
            fails += 1;
        }

        // ---- (3) 24 LAYERS: compounding through a residual stream ----
        println!("\n--- (3) {} LAYERS residual stream: does the gap stay bounded? ---", layers);
        let h_s = malloc(n * 4) as *mut f32;
        let h_r = malloc(n * 4) as *mut f32;
        for i in 0..n {
            *h_s.add(i) = *x.add(i);
            *h_r.add(i) = *x.add(i);
        }
        let mut rel_layers = 0.0f32;
        let mut abs_layers = 0.0f32;
        let mut worst_ratio_layers = 0.0f32;
        let mut worst_abs_layers = 0.0f32;
        for l in 0..layers {
            vmlm_core::b1_pure::b1_forward(&sc, h_s, y_simd);
            vmlm_core::b1_pure::b1_forward_scalar_reference(&sc, h_r, y_ref);
            // scale of this layer's output (term magnitudes), captured before
            // the residual update overwrites h
            let mut scale = 0.0f32;
            for i in 0..n {
                for j in 0..k0 {
                    let base = i * k0 + j;
                    let sid = *sc.src.add(base) as usize;
                    scale += (*sc.w.add(base) * *h_s.add(sid)).abs();
                }
            }
            for i in 0..n {
                *h_s.add(i) = *h_s.add(i) + *y_simd.add(i);
                *h_r.add(i) = *h_r.add(i) + *y_ref.add(i);
            }
            let mut mabs = 0.0f32;
            let mut wu = 0.0f32;
            for i in 0..n {
                let a = *h_s.add(i);
                let b = *h_r.add(i);
                let dd = (a - b).abs();
                if dd > mabs {
                    mabs = dd;
                }
                if b.abs() > 1e-30 {
                    let u = dd / (f32::EPSILON * b.abs());
                    if u > wu {
                        wu = u;
                    }
                }
            }
            // ratio against the LAYER's term scale (cancellation-robust)
            let ratio = if scale > 0.0 { mabs / scale } else { 0.0 };
            if ratio > worst_ratio_layers {
                worst_ratio_layers = ratio;
            }
            if mabs > worst_abs_layers {
                worst_abs_layers = mabs;
            }
            if l == 0 || l == layers - 1 || (l + 1) % 8 == 0 {
                println!(
                    "    after layer {:2}: max|abs diff| = {:.3e}   |diff|/scale = {:.3e}   (ulp-of-sum {:.0})",
                    l + 1, mabs, ratio, wu
                );
            }
        }
        // Same ratio-of-term-scale bar as criterion (1). Compounding through a
        // residual stream adds a rounding-level deviation per layer, so the
        // bound is loosened by sqrt(layers) rather than being removed: a real
        // defect would still show up 5+ orders of magnitude above it.
        let ratio_bar_layers = ratio_bar * (layers as f32).sqrt();
        println!(
            "  worst |diff|/scale over {} layers = {:.3e}   (bar {:.3e} = {:.0e} * sqrt({}))",
            layers, worst_ratio_layers, ratio_bar_layers, ratio_bar, layers
        );
        if worst_ratio_layers <= ratio_bar_layers {
            println!("  [PASS] after {} layers the gap stays at rounding level", layers);
        } else {
            println!("  [FAIL] after {} layers gap {:.3e} exceeds {:.3e}",
                layers, worst_ratio_layers, ratio_bar_layers);
            fails += 1;
        }

        // ---- (4) SPEED: shipped vs scalar reference ----
        println!("\n--- (4) SPEED: shipped vs scalar reference (QPC median) ---");
        let clk = Clock::new();
        let mut sink = 0.0f32;
        macro_rules! timeit {
            ($label:expr, $body:block) => {{
                for _ in 0..3 { $body }
                let mut ts: Vec<f64> = Vec::new();
                for _ in 0..300 {
                    let t0 = clk.now();
                    $body
                    let t1 = clk.now();
                    ts.push(clk.ms(t1 - t0));
                }
                let med = median(&mut ts);
                println!("  {:<44} {:9.6} ms  {:6.3} ns/conn", $label, med, med * 1e6 / e as f64);
                med
            }};
        }
        let t_simd = timeit!("b1_forward (SIMD reduce + u32 mirror)", {
            let m = vmlm_core::b1_pure::b1_forward(&sc, x, y_simd);
            sink += m as f32;
        });
        let t_ref = timeit!("b1_forward_scalar_reference (bit-exact)", {
            let m = vmlm_core::b1_pure::b1_forward_scalar_reference(&sc, x, y_ref);
            sink += m as f32;
        });
        println!("\n  speedup = {:.3}x", t_ref / t_simd);
        let calls = 5.0 * 24.0 * 128.0;
        println!("  projected: ref {:.1} ms/step  ->  shipped {:.1} ms/step", t_ref * calls, t_simd * calls);

        if sink.is_nan() {
            println!("nan");
        }
    }

    println!("\n===== fail count = {} =====", fails);
    if fails > 0 {
        std::process::exit(1);
    }
    println!("RESULT: ALL ACCEPTANCE CRITERIA PASS");
}
