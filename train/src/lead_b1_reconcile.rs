// PATH: B1
//
// Reconcile two measurements that disagree by 4.5x for the "same" work.
//
// THE CONFLICT
//   bench-builder's vmlm-vec-gather V1 (vectorised product row, no reduction):
//       0.167 ns/conn
//   Lead's vmlm-b1-ceiling M1 (same product row + scalar reduction + per-row buffer):
//       0.749 ns/conn
//
//   Both claim to measure "the packed B1 product step". A 4.5x gap means at
//   least one is not measuring what its label says. This project has already
//   produced four false measurement readings, so the gap is treated as a bug
//   until explained -- not averaged away.
//
// THREE CANDIDATE CAUSES, EACH MEASURED SEPARATELY
//   C1  the scalar reduction over k0=20 (present in M1, absent in V1)
//   C2  the per-row 256-float buffer in M1 (a 1 KB zeroing per row!)
//   C3  loop-level effects: V1's inner loop may be hoisted because its inputs
//       do not change across reps, while M1's writes to y defeat hoisting
//
// WHY C2 IS THE PRIME SUSPECT
//   M1 declares `let mut prod = [0.0f32; 256];` INSIDE the row loop and passes
//   it to the kernel. That is a 1024-byte stack array per row, per call. Even if
//   LLVM does not literally zero it each iteration, its existence constrains
//   register allocation and can force stack traffic. This is exactly the kind
//   of "packaging overhead" the task is about, and it is FIXABLE.
//
// USAGE: vmlm-b1-reconcile.exe [--d 1536] [--k0 20] [--reps 2000]

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

    println!("===== RECONCILE: why do V1 (0.167) and M1 (0.749) differ 4.5x? =====");
    println!("d={} k0={} reps={}", d, k0, reps);

    let clk = Clock::new();
    let mut sink = 0.0f32;
    let n = d;
    let e = n * k0;

    unsafe {
        let w = malloc(e * 4) as *mut f32;
        let src_f = malloc(e * 4) as *mut f32;
        let x = malloc(n * 4) as *mut f32;
        let y = malloc(n * 4) as *mut f32;
        // ONE persistent buffer, shared by all rows (as a good kernel would)
        let prod_shared = malloc(256 * 4) as *mut f32;

        for i in 0..e {
            *w.add(i) = (((i * 2654435761usize) % 2000) as f32 / 1000.0) - 1.0;
            *src_f.add(i) = (((i * 40503usize) % n) as u32) as f32;
        }
        for i in 0..n {
            *x.add(i) = (((i * 22695477usize) % 1000) as f32 / 1000.0) - 0.5;
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
                println!("  {:<50} {:9.6} ms  {:6.3} ns/conn", $label, med, med * 1e6 / e as f64);
                med
            }};
        }

        println!("\n--- C1/C2: isolate the reduction and the buffer ---");

        // A: products only, row-resident stack buffer (M1's shape)
        let t_a = timeit!("A products only, per-row stack buf [256]", {
            let mut acc_guard = 0.0f32;
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
                acc_guard += prod[0]; // keep the kernel call alive
            }
            sink += acc_guard;
        });

        // B: products only, ONE shared heap buffer (no per-row 1KB array)
        let t_b = timeit!("B products only, SHARED heap buf", {
            let mut acc_guard = 0.0f32;
            for i in 0..n {
                let base = i * k0;
                if vmlm_core::interp::HAS_AVX2 {
                    vmlm_core::kernels::sparse_product_row_avx2(
                        w.add(base), src_f.add(base), x, k0, prod_shared,
                    );
                } else {
                    vmlm_core::kernels::sparse_product_row_scalar(
                        w.add(base), src_f.add(base), x, k0, prod_shared,
                    );
                }
                acc_guard += *prod_shared;
            }
            sink += acc_guard;
        });

        // C: products AND reduction, shared heap buffer (the shippable shape)
        let t_c = timeit!("C products + reduction, SHARED heap buf", {
            for i in 0..n {
                let base = i * k0;
                if vmlm_core::interp::HAS_AVX2 {
                    vmlm_core::kernels::sparse_product_row_avx2(
                        w.add(base), src_f.add(base), x, k0, prod_shared,
                    );
                } else {
                    vmlm_core::kernels::sparse_product_row_scalar(
                        w.add(base), src_f.add(base), x, k0, prod_shared,
                    );
                }
                let mut acc = 0.0f32;
                let mut first = true;
                for j in 0..k0 {
                    let term = *prod_shared.add(j);
                    acc = if first { term } else { acc + term };
                    first = false;
                }
                *y.add(i) = acc;
            }
            sink += consume(y, n);
        });

        // D: M1 verbatim (per-row stack buffer + reduction)
        let t_d = timeit!("D M1 verbatim (stack buf + reduction)", {
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
                }
                *y.add(i) = acc;
            }
            sink += consume(y, n);
        });

        let floor_ms = 1.0e-5;
        for (nm, t) in [("A", t_a), ("B", t_b), ("C", t_c), ("D", t_d)] {
            if t < floor_ms {
                println!("  [FATAL] {} below floor -> deleted loop", nm);
                std::process::exit(1);
            }
        }

        let p = |t: f64| t * 1e6 / e as f64;
        println!("\n===== RECONCILIATION =====");
        println!("  A products only, stack buf  {:6.3} ns/conn", p(t_a));
        println!("  B products only, heap buf   {:6.3} ns/conn", p(t_b));
        println!("  C products+reduce, heap buf {:6.3} ns/conn", p(t_c));
        println!("  D products+reduce, stack buf{:6.3} ns/conn   (= M1)", p(t_d));
        println!();
        println!("  redistribution cost (D - B) = {:6.3} ns/conn  ({:.0}% of D)",
            p(t_d) - p(t_b), 100.0 * (t_d - t_b) / t_d);
        println!("  reduction cost      (C - B) = {:6.3} ns/conn  ({:.0}% of C)",
            p(t_c) - p(t_b), 100.0 * (t_c - t_b) / t_c);
        println!("  buffer residency    (D - C) = {:6.3} ns/conn  ({:.0}% of D)",
            p(t_d) - p(t_c), 100.0 * (t_d - t_c) / t_d);
        println!();
        println!("  bench-builder V1 (products only, no reduce) = 0.167 ns/conn");
        println!("  My A (products only, stack buf)             = {:.3} ns/conn", p(t_a));
        println!("  My B (products only, heap buf)              = {:.3} ns/conn", p(t_b));
        println!();
        if p(t_b) < p(t_a) * 0.7 {
            println!("  => THE PER-ROW STACK BUFFER IS REAL OVERHEAD: {:.3} -> {:.3} ns/conn",
                p(t_a), p(t_b));
            println!("     A shared buffer is the fix. This is a genuine, cheap win.");
        } else {
            println!("  => The buffer is not the explanation; look elsewhere.");
        }

        if sink.is_nan() {
            println!("nan");
        }
    }
}
