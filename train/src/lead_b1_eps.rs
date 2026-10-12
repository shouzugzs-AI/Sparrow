// PATH: B1
//
// MEASURE epsilon BEFORE writing the production kernel.
//
// WHY MEASURE FIRST
//   The acceptance bar changed from bit-exact to "rounding-level difference".
//   That is only a legitimate relaxation if we KNOW what the difference actually
//   is. Picking epsilon after seeing the number is how a tolerance becomes a
//   rubber stamp.
//
//   So this probe establishes the reference distribution:
//     - SIMD 8-way reduction vs the scalar ascending reduction, ONE layer
//     - the same, accumulated over 24 layers (error compounds -- this is the
//       case that actually matters for the model, and the brief's "24 layers"
//       is the real usage)
//   and reports max|diff|, mean|diff|, and the relative error, WITHOUT saying
//   what epsilon "should" be. The choice is made from this data afterwards.
//
// WHAT VARIANTS ARE COMPARED
//   R_scalar   acc = sum in ascending j, single accumulator, first-term assign
//              (the current contract, = what b1_forward does today)
//   R_simd8    eight partial sums accumulated in 8 lanes, then a horizontal
//              reduction. This is the standard, fastest shape.
//   R_simd4    4 lanes -- fewer lanes means less reordering, so LESS error.
//              Included so the epsilon/accuracy tradeoff is visible.
//
// USAGE: vmlm-b1-eps.exe [--d 1536] [--k0 20] [--layers 24] [--reps 2000]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

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

static mut GUARD: f64 = 0.0;
#[inline(never)]
unsafe fn consume(p: *const f32, n: usize) -> f32 {
    let mut s = 0.0f32;
    let mut i = 0usize;
    while i < n {
        s += *p.add(i);
        i += 64;
    }
    GUARD += s as f64;
    s
}

/// Reference: the CURRENT contract -- ascending, single accumulator.
unsafe fn reduce_scalar(prod: *const f32, k0: usize) -> f32 {
    let mut acc = 0.0f32;
    let mut first = true;
    for j in 0..k0 {
        let t = *prod.add(j);
        acc = if first { t } else { acc + t };
        first = false;
    }
    acc
}

/// 8-lane SIMD reduction: 8 partial sums, then a horizontal fold.
/// The horizontal fold order below is FIXED (not left to the compiler) so the
/// result is deterministic across runs -- that is an explicit acceptance bar.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn reduce_simd8(prod: *const f32, k0: usize) -> f32 {
    let mut acc = _mm256_setzero_ps();
    let mut j = 0usize;
    while j + 8 <= k0 {
        acc = _mm256_add_ps(acc, _mm256_loadu_ps(prod.add(j)));
        j += 8;
    }
    // fixed horizontal fold: 8 -> 4 -> 2 -> 1
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let mut v = _mm_add_ps(lo, hi);
    let sh = _mm_movehl_ps(v, v);
    v = _mm_add_ps(v, sh);
    let sh2 = _mm_shuffle_ps(v, v, 0x55);
    v = _mm_add_ss(v, sh2);
    let mut s = _mm_cvtss_f32(v);
    // scalar tail, ascending -- appended AFTER the vector part
    while j < k0 {
        s += *prod.add(j);
        j += 1;
    }
    s
}

/// 4-lane variant: fewer reordered terms => smaller deviation.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn reduce_simd4(prod: *const f32, k0: usize) -> f32 {
    let mut acc = _mm_setzero_ps();
    let mut j = 0usize;
    while j + 4 <= k0 {
        acc = _mm_add_ps(acc, _mm_loadu_ps(prod.add(j)));
        j += 4;
    }
    let sh = _mm_movehl_ps(acc, acc);
    let mut v = _mm_add_ps(acc, sh);
    let sh2 = _mm_shuffle_ps(v, v, 0x55);
    v = _mm_add_ss(v, sh2);
    let mut s = _mm_cvtss_f32(v);
    while j < k0 {
        s += *prod.add(j);
        j += 1;
    }
    s
}

fn main() {
    unsafe { vmlm_core::interp::init_cpu() };
    let d = arg_u32("--d", 1536) as usize;
    let k0 = arg_u32("--k0", 20) as usize;
    let layers = arg_u32("--layers", 24) as usize;
    let reps = arg_u32("--reps", 2000) as usize;

    println!("===== MEASURE epsilon (before choosing it) =====");
    println!("d={} k0={} layers={} reps={}", d, k0, layers, reps);
    println!("HAS_AVX2 = {}", unsafe { vmlm_core::interp::HAS_AVX2 });

    let clk = Clock::new();
    let mut sink = 0.0f32;
    let n = d;
    let e = n * k0;

    unsafe {
        let w = malloc(e * 4) as *mut f32;
        let src = malloc(e * 4) as *mut f32;
        let x = malloc(n * 4) as *mut f32;
        let prod = malloc(256 * 4) as *mut f32;
        let y_scalar = malloc(n * 4) as *mut f32;
        let y_s8 = malloc(n * 4) as *mut f32;
        let y_s4 = malloc(n * 4) as *mut f32;

        for i in 0..e {
            // Realistic magnitudes: the artifact uses small Glorot-ish weights.
            *w.add(i) = (((i * 2654435761usize) % 2000) as f32 / 1000.0 - 1.0) * 0.1;
            *src.add(i) = (((i * 40503usize) % n) as u32) as f32;
        }
        for i in 0..n {
            *x.add(i) = (((i * 22695477usize) % 1000) as f32 / 1000.0) - 0.5;
        }

        // ---- products (identical for all three reducers) ----
        let build_products = |i: usize| {
            let base = i * k0;
            for j in 0..k0 {
                *prod.add(j) = *w.add(base + j) * *x.add(*src.add(base + j) as usize);
            }
        };

        // ---- ONE LAYER: compare the three reducers ----
        let mut max_s8 = 0.0f32;
        let mut max_s4 = 0.0f32;
        let mut sum_s8 = 0.0f64;
        let mut sum_s4 = 0.0f64;
        let mut rel_s8 = 0.0f32;
        let mut nz = 0usize;
        let mut exact_s8 = 0usize;
        let mut exact_s4 = 0usize;
        for i in 0..n {
            build_products(i);
            let r = reduce_scalar(prod, k0);
            let a8 = reduce_simd8(prod, k0);
            let a4 = reduce_simd4(prod, k0);
            *y_scalar.add(i) = r;
            *y_s8.add(i) = a8;
            *y_s4.add(i) = a4;
            let d8 = (a8 - r).abs();
            let d4 = (a4 - r).abs();
            if d8 == 0.0 { exact_s8 += 1; }
            if d4 == 0.0 { exact_s4 += 1; }
            if d8 > max_s8 { max_s8 = d8; }
            if d4 > max_s4 { max_s4 = d4; }
            sum_s8 += d8 as f64;
            sum_s4 += d4 as f64;
            if r != 0.0 {
                nz += 1;
                let rel = d8 / r.abs();
                if rel > rel_s8 { rel_s8 = rel; }
            }
        }

        println!("\n--- ONE LAYER: SIMD reduction vs scalar reference ---");
        println!("  R_simd8  max|diff| = {:.3e}   mean|diff| = {:.3e}   relative max = {:.3e}",
            max_s8, sum_s8 / n as f64, rel_s8);
        println!("  bits identical      = {} / {} ({:.1}%)", exact_s8, n, 100.0 * exact_s8 as f64 / n as f64);
        println!("  R_simd4  max|diff| = {:.3e}   mean|diff| = {:.3e}", max_s4, sum_s4 / n as f64);
        println!("  bits identical      = {} / {} ({:.1}%)", exact_s4, n, 100.0 * exact_s4 as f64 / n as f64);

        // ---- 24 LAYERS: error compounds. This is the case that matters. ----
        // Emulate a residual stream: h feeds the next layer, so a difference in
        // layer L propagates and is amplified/summed by layer L+1.
        let h_s = malloc(n * 4) as *mut f32;
        let h_8 = malloc(n * 4) as *mut f32;
        for i in 0..n {
            *h_s.add(i) = *x.add(i);
            *h_8.add(i) = *x.add(i);
        }
        let mut max_layer = 0.0f32;
        println!("\n--- {} LAYERS: error compounding through a residual stream ---", layers);
        for l in 0..layers {
            let mut max_this = 0.0f32;
            for i in 0..n {
                let base = i * k0;
                // scalar path
                for j in 0..k0 {
                    let sid = *src.add(base + j) as usize;
                    *prod.add(j) = *w.add(base + j) * *h_s.add(sid);
                }
                let rs = reduce_scalar(prod, k0);
                // simd8 path reads ITS OWN h
                for j in 0..k0 {
                    let sid = *src.add(base + j) as usize;
                    *prod.add(j) = *w.add(base + j) * *h_8.add(sid);
                }
                let r8 = reduce_simd8(prod, k0);
                // residual: h = h + y  (so a divergence persists)
                *h_s.add(i) = *h_s.add(i) + rs;
                *h_8.add(i) = *h_8.add(i) + r8;
                let dd = (*h_8.add(i) - *h_s.add(i)).abs();
                if dd > max_this { max_this = dd; }
            }
            if l == 0 || l == layers - 1 || l % 6 == 0 {
                println!("  after layer {:2}: max|diff| in residual stream = {:.3e}", l + 1, max_this);
            }
            max_layer = max_this;
        }
        println!("  FINAL (layer {}): max|diff| = {:.3e}", layers, max_layer);

        // relative to the stream magnitude
        let mut mx = 0.0f32;
        for i in 0..n {
            if (*h_s.add(i)).abs() > mx { mx = (*h_s.add(i)).abs(); }
        }
        println!("  residual stream max|value| = {:.3e}", mx);
        println!("  FINAL relative = {:.3e}", if mx > 0.0 { max_layer / mx } else { 0.0 });

        // ---- determinism: same input, same output, repeated ----
        println!("\n--- DETERMINISM of the SIMD reducer (acceptance bar) ---");
        let mut det_ok = true;
        let mut first_bits: Vec<u32> = Vec::new();
        for r in 0..5 {
            for i in 0..n {
                build_products(i);
                let a = reduce_simd8(prod, k0);
                if r == 0 {
                    first_bits.push(a.to_bits());
                } else if first_bits[i] != a.to_bits() {
                    det_ok = false;
                }
            }
        }
        println!("  5 independent passes, bit-identical results = {}", det_ok);
        // and across a fresh product buffer (to catch uninitialised reads)
        for i in 0..k0 { *prod.add(i) = 0.0; }
        build_products(0);
        let a = reduce_simd8(prod, k0);
        for i in 0..k0 { *prod.add(i) = 0.0; }
        build_products(0);
        let b = reduce_simd8(prod, k0);
        println!("  after zeroing + rebuild, bit-identical = {}", a.to_bits() == b.to_bits());

        // ---- SIMD reduction is also FASTER; measure it ----
        macro_rules! timeit {
            ($label:expr, $body:block) => {{
                for _ in 0..3 { $body }
                let mut ts: Vec<f64> = Vec::with_capacity(reps as usize);
                for _ in 0..reps {
                    let t0 = clk.now();
                    $body
                    let t1 = clk.now();
                    ts.push(clk.ms(t1 - t0));
                }
                let med = median(&mut ts);
                println!("  {:<40} {:9.6} ms  {:6.3} ns/conn", $label, med, med * 1e6 / e as f64);
                med
            }};
        }
        println!("\n--- reduction cost alone ({} reductions/op) ---", n);
        // ⚠️ The first version of this timing called build_products() INSIDE the
        // timed body, so "reduction cost" actually measured products+reduction and
        // reported a meaningless 1.03x. Products must be built ONCE, outside the
        // timed region, and the product buffer must be LARGE enough to hold one
        // row per neuron (otherwise the timing degrades to re-reading one row).
        //
        // Using a per-neuron product buffer also makes the comparison fair:
        // both reducers read the same pre-computed data, so the ONLY difference
        // is the reduction itself.
        let prod_big = malloc(e * 4) as *mut f32;
        for i in 0..n {
            let base = i * k0;
            for j in 0..k0 {
                *prod_big.add(base + j) =
                    *w.add(base + j) * *x.add(*src.add(base + j) as usize);
            }
        }
        sink += consume(prod_big, e);

        let t_scalar = timeit!("reduce_scalar, products pre-computed", {
            let mut s = 0.0f32;
            for i in 0..n {
                s += reduce_scalar(prod_big.add(i * k0), k0);
            }
            sink += s;
        });
        let t_s8 = timeit!("reduce_simd8, products pre-computed", {
            let mut s = 0.0f32;
            for i in 0..n {
                s += reduce_simd8(prod_big.add(i * k0), k0);
            }
            sink += s;
        });
        let t_s4 = timeit!("reduce_simd4, products pre-computed", {
            let mut s = 0.0f32;
            for i in 0..n {
                s += reduce_simd4(prod_big.add(i * k0), k0);
            }
            sink += s;
        });

        let floor_ms = 1.0e-5;
        if t_scalar < floor_ms || t_s8 < floor_ms {
            println!("  [FATAL] below floor -> deleted loop");
            std::process::exit(1);
        }
        println!("\n  reduction speedup = {:.3}x", t_scalar / t_s8);

        println!("\n===== DATA FOR CHOOSING epsilon =====");
        println!("  one layer  max|abs diff| (simd8) = {:.3e}", max_s8);
        println!("  one layer  max relative          = {:.3e}", rel_s8);
        println!("  {} layers  max|abs diff|         = {:.3e}", layers, max_layer);
        println!("  bits identical (simd8, 1 layer)  = {:.1}%", 100.0 * exact_s8 as f64 / n as f64);
        println!("  NOTE: this probe deliberately does NOT pick epsilon.");

        if sink.is_nan() {
            println!("nan");
        }
    }
}
