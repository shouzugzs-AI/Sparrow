// PATH: B1
//
// THE decisive measurement for this audit: what does u32 src buy on the ACTUAL
// SHIPPED B1 path (vectorised product row + scalar reduce)?
//
// WHY THIS IS NEEDED
//   Two regimes disagree:
//     scalar loop            f32 1.185 -> u32 0.309 ns/conn   (3.8x)
//     vectorised product row f32 0.167 -> u32 0.159 ns/conn   (1.05x)
//   (the second from bench-builder's vmlm-vec-gather, products only)
//
//   The shipped path is a HYBRID: vectorised product computation, then a SCALAR
//   reduction. So neither number necessarily applies, and I have already been
//   burned once by extrapolating a scalar figure to a vector path (AUDIT 5.2).
//   The only safe answer is to measure the shipped shape directly.
//
// WHAT IS MEASURED
//   P_f32  = sparse_product_row_avx2 with f32 src, + scalar reduce  (= SHIPPED)
//   P_u32  = the same kernel shape with u32 src,        + scalar reduce
//   Both use the real kernel's dataflow; the u32 variant is written inline here
//   because the shipped kernel takes f32 src.
//
//   The u32 variant must be BIT-EXACT vs the f32 variant: for ids < 2^24 the
//   float round trip is exact, so the identity of every connection is unchanged.
//   That is asserted, not assumed.
//
// USAGE: vmlm-b1-shipped-u32.exe [--d 1536] [--k0 20] [--reps 2000]

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

/// The u32-src twin of `sparse_product_row_avx2`. Same dataflow, same
/// no-abs, no-FMA discipline; only the index load differs (no convert).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn product_row_u32_avx2(
    wrow: *const f32,
    srcrow: *const u32,
    x: *const f32,
    k0: usize,
    out: *mut f32,
) {
    let mut j = 0usize;
    while j + 8 <= k0 {
        let wv = _mm256_loadu_ps(wrow.add(j));
        let sid = _mm256_loadu_si256(srcrow.add(j) as *const __m256i);
        let xv = _mm256_i32gather_ps(x, sid, 4);
        _mm256_storeu_ps(out.add(j), _mm256_mul_ps(wv, xv));
        j += 8;
    }
    while j < k0 {
        let sid = *srcrow.add(j) as usize;
        *out.add(j) = *wrow.add(j) * *x.add(sid);
        j += 1;
    }
}

fn main() {
    unsafe { vmlm_core::interp::init_cpu() };
    let d = arg_u32("--d", 1536) as usize;
    let k0 = arg_u32("--k0", 20) as usize;
    let reps = arg_u32("--reps", 2000) as usize;

    println!("===== SHIPPED-shape: u32 src vs f32 src (vector row + scalar reduce) =====");
    println!("d={} k0={} reps={}", d, k0, reps);
    println!("HAS_AVX2 = {}", unsafe { vmlm_core::interp::HAS_AVX2 });

    let clk = Clock::new();
    let mut sink = 0.0f32;
    let n = d;
    let e = n * k0;

    unsafe {
        let w = malloc(e * 4) as *mut f32;
        let src_f = malloc(e * 4) as *mut f32;
        let src_u = malloc(e * 4) as *mut u32;
        let x = malloc(n * 4) as *mut f32;
        let yf = malloc(n * 4) as *mut f32;
        let yu = malloc(n * 4) as *mut f32;
        let prod = malloc(256 * 4) as *mut f32;

        for i in 0..e {
            *w.add(i) = (((i * 2654435761usize) % 2000) as f32 / 1000.0) - 1.0;
            let s = ((i * 40503usize) % n) as u32;
            *src_f.add(i) = s as f32;
            *src_u.add(i) = s;
        }
        for i in 0..n {
            *x.add(i) = (((i * 22695477usize) % 1000) as f32 / 1000.0) - 0.5;
        }

        // ---- exactness: the two layouts must hold identical ids ----
        let mut rt = 0usize;
        for i in 0..e {
            if (*src_f.add(i) as usize) != (*src_u.add(i) as usize) {
                rt += 1;
            }
        }
        println!("\n  f32/u32 identity mismatches = {} / {} (must be 0)", rt, e);

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
                println!("  {:<44} {:9.6} ms  {:6.3} ns/conn", $label, med, med * 1e6 / e as f64);
                med
            }};
        }

        println!("\n--- shipped shape ---");
        let t_f = timeit!("P_f32 SHIPPED: vec row(f32) + scalar reduce", {
            for i in 0..n {
                let base = i * k0;
                if vmlm_core::interp::HAS_AVX2 {
                    vmlm_core::kernels::sparse_product_row_avx2(
                        w.add(base), src_f.add(base), x, k0, prod,
                    );
                } else {
                    vmlm_core::kernels::sparse_product_row_scalar(
                        w.add(base), src_f.add(base), x, k0, prod,
                    );
                }
                let mut acc = 0.0f32;
                let mut first = true;
                for j in 0..k0 {
                    let term = *prod.add(j);
                    acc = if first { term } else { acc + term };
                    first = false;
                }
                *yf.add(i) = acc;
            }
            sink += consume(yf, n);
        });

        let t_u = timeit!("P_u32 vec row(u32) + scalar reduce", {
            for i in 0..n {
                let base = i * k0;
                if vmlm_core::interp::HAS_AVX2 {
                    product_row_u32_avx2(w.add(base), src_u.add(base), x, k0, prod);
                } else {
                    for j in 0..k0 {
                        *prod.add(j) = *w.add(base + j) * *x.add(*src_u.add(base + j) as usize);
                    }
                }
                let mut acc = 0.0f32;
                let mut first = true;
                for j in 0..k0 {
                    let term = *prod.add(j);
                    acc = if first { term } else { acc + term };
                    first = false;
                }
                *yu.add(i) = acc;
            }
            sink += consume(yu, n);
        });

        // ---- bit-exactness of the two outputs ----
        let mut bad = 0usize;
        for i in 0..n {
            if (*yf.add(i)).to_bits() != (*yu.add(i)).to_bits() {
                bad += 1;
            }
        }
        println!("\n  OUTPUT bit mismatches f32-src vs u32-src = {} / {} (must be 0)", bad, n);

        let floor_ms = 1.0e-5;
        if t_f < floor_ms || t_u < floor_ms {
            println!("  [FATAL] below floor -> deleted loop");
            std::process::exit(1);
        }

        let p = |t: f64| t * 1e6 / e as f64;
        println!("\n===== RESULT (shipped shape) =====");
        println!("  f32 src (today)  {:6.3} ns/conn   {:8.4} ms/op", p(t_f), t_f);
        println!("  u32 src          {:6.3} ns/conn   {:8.4} ms/op", p(t_u), t_u);
        println!("  speedup          {:.3}x", t_f / t_u);
        println!("  saving           {:6.3} ns/conn", p(t_f) - p(t_u));
        println!();
        println!("  Compare with the two disagreeing regimes:");
        println!("    scalar loop estimate  3.8x");
        println!("    vector-products-only  1.05x");
        println!("    SHIPPED (measured)    {:.2}x   <- the one that counts", t_f / t_u);

        if sink.is_nan() {
            println!("nan");
        }
    }
}
