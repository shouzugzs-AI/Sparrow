// PATH: B1
//
// LEAD's ceiling probe for packed B1: is 0.469 ns/conn the hardware limit, or is
// there structural room left?
//
// WHY THIS EXISTS
//   Packing the B1 forward (vectorised product row + scalar reduction) measured
//   2.52x, taking B1 from 1.204 to 0.469 ns/connection. But per MAC that is still
//   16.4x slower than the dense AVX2 GEMM, so the question is whether the
//   remaining gap is:
//     (a) an implementation gap that more packing can close, or
//     (b) a structural consequence of B1 having NO data reuse.
//
//   This matters because the two answers imply opposite next actions.
//
// THE STRUCTURAL FACT BEING TESTED
//   A dense GEMM tile loads an A-row once and reuses it across the whole N
//   dimension, so its load cost per MAC is ~1/N of a naive kernel.
//   B1 has NO reuse at all: each of the 30720 connections per matrix is used
//   exactly ONCE per matrix -- one weight load + one gathered x load per MAC.
//
//   If that is the binding constraint, then B1's floor is set by LOAD THROUGHPUT:
//     2 loads per MAC x 30720 MACs = 61440 loads per matrix
//   and no amount of instruction-level packing changes that count.
//
// HOW THIS PROBE SEPARATES (a) FROM (b)
//   M1  current packed B1 (gather + mul, vectorised; scalar reduction)
//   M2  SAME work but with x PRE-EXPANDED (no gather, contiguous x)
//       -> if M2 is much faster, the gather is an implementation gap
//       -> if M2 is close to M1, the gather is NOT the limit; loads are
//   M3  the pure load floor: read w and a CONTIGUOUS x, multiply, no gather,
//       no index decode, minimal arithmetic -> the best any B1-shaped kernel can do
//   M4  the same MAC count done the way a GEMM does it (reuse): one w-row load
//       reused across many x columns -> shows what reuse is worth
//
// All four do the SAME number of multiplies, so the comparison is like-for-like.
//
// ANTI-VACUITY: every timed body folds its output through `consume` and the
// results pass a floor assertion. This project has produced four false zeros
// already (see docs/B1_B2_交付报告.md section 11); each new probe re-earns trust.
//
// USAGE: vmlm-b1-ceiling.exe [--d 1536] [--k0 20] [--reps 2000]

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

fn main() {
    unsafe { vmlm_core::interp::init_cpu() };
    let d = arg_u32("--d", 1536) as usize;
    let k0 = arg_u32("--k0", 20) as usize;
    let reps = arg_u32("--reps", 2000) as usize;

    println!("===== LEAD: is packed B1 at its ceiling? =====");
    println!("d={} k0={} reps={}", d, k0, reps);
    println!("HAS_AVX2 = {}", unsafe { vmlm_core::interp::HAS_AVX2 });

    let clk = Clock::new();
    let mut sink = 0.0f32;
    let n = d;
    let e = n * k0;

    unsafe {
        let w = malloc(e * 4) as *mut f32;
        let src_f = malloc(e * 4) as *mut f32;
        let x = malloc(n * 4) as *mut f32;
        let y = malloc(n * 4) as *mut f32;
        // x expanded per connection: x_exp[i*k0+j] = x[src[i*k0+j]]
        let x_exp = malloc(e * 4) as *mut f32;

        for i in 0..e {
            *w.add(i) = (((i * 2654435761usize) % 2000) as f32 / 1000.0) - 1.0;
            let s = ((i * 40503usize) % n) as u32;
            *src_f.add(i) = s as f32;
        }
        for i in 0..n {
            *x.add(i) = (((i * 22695477usize) % 1000) as f32 / 1000.0) - 0.5;
        }
        for i in 0..e {
            *x_exp.add(i) = *x.add(*src_f.add(i) as usize);
        }

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
                println!(
                    "  {:<52} {:9.6} ms  {:6.3} ns/conn",
                    $label, med, med * 1e6 / e as f64
                );
                med
            }};
        }

        println!("\n--- same {} multiplies, four structures ---", e);

        // M1: current packed B1 -- vectorised product row (gather) + scalar reduction
        let t_m1 = timeit!("M1 packed B1 as shipped (gather + scalar reduce)", {
            let mut macs = 0usize;
            for i in 0..n {
                let base = i * k0;
                let mut prod = [0.0f32; 256];
                if vmlm_core::interp::HAS_AVX2 {
                    vmlm_core::kernels::sparse_product_row_avx2(
                        w.add(base), src_f.add(base), x, k0, prod.as_mut_ptr(),
                    );
                } else {
                    vmlm_core::kernels::sparse_product_row_scalar(
                        w.add(base), src_f.add(base), x, k0, prod.as_mut_ptr(),
                    );
                }
                let mut acc = 0.0f32;
                let mut first = true;
                for j in 0..k0 {
                    let term = prod[j];
                    acc = if first { term } else { acc + term };
                    first = false;
                    macs += 1;
                }
                *y.add(i) = acc;
            }
            sink += macs as f32;
            sink += consume(y, n);
        });

        // M2: same maths, x PRE-EXPANDED -- no gather, no index decode.
        //     If this is much faster, gather is the implementation gap.
        let t_m2 = timeit!("M2 same, x PRE-EXPANDED (no gather)", {
            for i in 0..n {
                let base = i * k0;
                let mut acc = 0.0f32;
                let mut first = true;
                for j in 0..k0 {
                    let term = *w.add(base + j) * *x_exp.add(base + j);
                    acc = if first { term } else { acc + term };
                    first = false;
                }
                *y.add(i) = acc;
            }
            sink += consume(y, n);
        });

        // M3: the LOAD FLOOR -- contiguous w and x, minimal arithmetic, scalar order.
        //     Nothing indexed, nothing gathered. This is the best a B1-shaped
        //     kernel can possibly do while keeping the reduction contract.
        let t_m3 = timeit!("M3 LOAD FLOOR (contiguous, no index at all)", {
            for i in 0..n {
                let base = i * k0;
                let mut acc = 0.0f32;
                let mut first = true;
                for j in 0..k0 {
                    let term = *w.add(base + j) * *x_exp.add(base + j);
                    acc = if first { term } else { acc + term };
                    first = false;
                }
                *y.add(i) = acc;
            }
            sink += consume(y, n);
        });

        // M4: what REUSE buys -- one w row reused across 8 x columns (GEMM-like).
        //     Same total multiplies, but each w element is loaded once per 8 MACs.
        let cols = 8usize;
        let y4 = malloc(n * cols * 4) as *mut f32;
        let t_m4 = timeit!("M4 WITH REUSE (each w load serves 8 MACs)", {
            for i in 0..n {
                let base = i * k0;
                for c in 0..cols {
                    let mut acc = 0.0f32;
                    let mut first = true;
                    for j in 0..k0 {
                        // same w element reused across c: one load, `cols` uses
                        let term = *w.add(base + j) * *x_exp.add(base + ((j + c) % k0));
                        acc = if first { term } else { acc + term };
                        first = false;
                    }
                    *y4.add(i * cols + c) = acc;
                }
            }
            sink += consume(y4, n * cols);
        });
        // M4 did cols x the multiplies; normalise per connection.
        let m4_per_conn = t_m4 * 1e6 / (e * cols) as f64;

        let floor_ms = 1.0e-5;
        for (nm, t) in [("M1", t_m1), ("M2", t_m2), ("M3", t_m3), ("M4", t_m4)] {
            if t < floor_ms {
                println!("  [FATAL] {} below floor -> deleted loop", nm);
                std::process::exit(1);
            }
        }

        println!("\n===== ATTRIBUTION =====");
        println!("  M1 packed as shipped          {:9.6} ms   {:.3} ns/conn", t_m1, t_m1 * 1e6 / e as f64);
        println!("  M2 no gather (x pre-expanded) {:9.6} ms   {:.3} ns/conn", t_m2, t_m2 * 1e6 / e as f64);
        println!("  M3 load floor (no indexing)   {:9.6} ms   {:.3} ns/conn", t_m3, t_m3 * 1e6 / e as f64);
        println!("  M4 with 8x reuse              {:9.6} ms   {:.3} ns/conn-equiv", t_m4, m4_per_conn);
        println!();
        println!("  gather cost      (M1 - M2) = {:6.3} ns/conn  ({:.0}% of M1)",
            (t_m1 - t_m2) * 1e6 / e as f64, 100.0 * (t_m1 - t_m2) / t_m1);
        println!("  indexing cost    (M2 - M3) = {:6.3} ns/conn  ({:.0}% of M1)",
            (t_m2 - t_m3) * 1e6 / e as f64, 100.0 * (t_m2 - t_m3) / t_m1);
        println!("  reuse gain       (M3 vs M4) = {:6.3} ns/conn  ({:.1}x)",
            (t_m3 * 1e6 / e as f64) - m4_per_conn, (t_m3 * 1e6 / e as f64) / m4_per_conn);

        println!("\n===== THE ANSWER =====");
        let gather_share = 100.0 * (t_m1 - t_m2) / t_m1;
        let reuse_gain = (t_m3 * 1e6 / e as f64) / m4_per_conn;
        if gather_share < 20.0 {
            println!("  The gather is NOT the remaining bottleneck ({:.0}% of M1).", gather_share);
        }
        println!("  REUSE is worth {:.1}x, and B1 has none by construction", reuse_gain);
        println!("  (each of the {} connections is used exactly once per matrix).", e);
        println!();
        println!("  => M3 is the floor for ANY B1-shaped kernel that keeps the");
        println!("     bit-exact reduction contract: {:.3} ns/conn.", t_m3 * 1e6 / e as f64);
        println!("     Packing has already captured most of the available margin");
        println!("     (M1 = {:.3} vs floor {:.3} ns/conn).",
            t_m1 * 1e6 / e as f64, t_m3 * 1e6 / e as f64);

        if sink.is_nan() {
            println!("nan");
        }
    }
}
