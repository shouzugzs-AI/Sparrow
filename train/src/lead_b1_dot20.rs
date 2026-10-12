// PATH: B1
//
// Q4: B1 (20 connections) vs a PURE d=20 dense dot product.
//     The difference = the cost of the index-mapping table.
// READ-ONLY audit: nothing in core/ is modified by this probe.
//
// WHY THIS COMPARISON NEEDS CARE
//   The framing is "B1 = a d=20 dense dot product + an index map". If that is
//   literally true, the two should cost about the same, and any difference IS
//   the index overhead. Three things can break that equivalence, and each is
//   measured rather than assumed:
//
//   (a) WIDTH. d=20 is not a SIMD width. In the dense case a compiler will
//       vectorise 20 = 16+4 (or 8+8+4). In B1 the sources are gathered, so the
//       vector width that helps dense does NOT help B1's x side.
//   (b) THE INDEX LOAD. Dense reads x[j] with j implicit in the loop counter.
//       B1 must LOAD src[j] and convert it (src is stored as f32) before it can
//       even address x. That is one extra load + one convert per MAC.
//   (c) WHICH BUFFER. Dense's x is 20 contiguous floats = 80 bytes = 2 cache
//       lines. B1's x is 1536 floats (6 KB) accessed at 20 scattered offsets.
//
// THE FOUR VARIANTS (all do exactly 20 multiplies per neuron, 1536 neurons)
//   V_dense20   pure d=20 dense dot products, contiguous x (the reference)
//   V_src_cont  identical maths but the source ids come from src[] AND
//               those ids happen to be 0..19 (so no real scatter)
//   V_src_rand  same as above but ids are the REAL random ids from the artifact
//   V_b1        the shipped B1 path (vectorised product row + scalar reduce)
//
//   V_dense20 vs V_src_cont  isolates the INDEX LOAD (b)
//   V_src_cont vs V_src_rand isolates the SCATTER   (c)
//
// USAGE: vmlm-b1-dot20.exe [--d 1536] [--k0 20] [--reps 3000]

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
    let reps = arg_u32("--reps", 3000) as usize;

    println!("===== Q4 AUDIT: B1 vs a PURE d=20 dense dot product =====");
    println!("d={} k0={} reps={}", d, k0, reps);
    println!("HAS_AVX2 = {}", unsafe { vmlm_core::interp::HAS_AVX2 });

    let clk = Clock::new();
    let mut sink = 0.0f32;
    let n = d;
    let e = n * k0;

    unsafe {
        // Real weights + real (random) sources, copied from the shipped layout.
        let w = malloc(e * 4) as *mut f32;
        let src = malloc(e * 4) as *mut f32;
        let x = malloc(n * 4) as *mut f32;
        let y = malloc(n * 4) as *mut f32;
        for i in 0..e {
            *w.add(i) = (((i * 2654435761usize) % 2000) as f32 / 1000.0) - 1.0;
        }
        // Deterministic but well-scattered ids (same character as the artifact:
        // 0/1536 rows ordered, union = full space).
        for i in 0..e {
            *src.add(i) = (((i * 40503usize) % n) as u32) as f32;
        }
        for i in 0..n {
            *x.add(i) = (((i * 22695477usize) % 1000) as f32 / 1000.0) - 0.5;
        }
        // A dedicated 20-element dense vector: x20[j] = x[j], contiguous.
        let x20 = malloc(k0 * 4) as *mut f32;
        for j in 0..k0 {
            *x20.add(j) = *x.add(j);
        }
        // Weights for the dense case: same values, one 20-block per neuron.
        let w20 = malloc(e * 4) as *mut f32;
        for i in 0..e {
            *w20.add(i) = *w.add(i);
        }
        // Identity source map: src[j] = j  (index loaded, but no scatter)
        let src_ident = malloc(e * 4) as *mut f32;
        for i in 0..e {
            *src_ident.add(i) = (i % k0) as f32;
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
                println!("  {:<46} {:9.6} ms  {:6.3} ns/conn", $label, med, med * 1e6 / e as f64);
                med
            }};
        }

        println!("\n--- all variants: {} multiplies + {} adds ---", e, e);

        // V_dense20: PURE dense dot product. No src array at all.
        // Note: x20 is only k0 floats, so it is L1-resident and the same
        // 20 values are reused by every neuron -- that is what "dense d=20"
        // means structurally, and it is the fairest reading of the framing.
        let t_dense = timeit!("V_dense20 pure dense d=20 dot (no src)", {
            for i in 0..n {
                let base = i * k0;
                let mut acc = 0.0f32;
                let mut first = true;
                for j in 0..k0 {
                    let term = *w20.add(base + j) * *x20.add(j);
                    acc = if first { term } else { acc + term };
                    first = false;
                }
                *y.add(i) = acc;
            }
            sink += consume(y, n);
        });

        // V_src_cont: identical maths, but ids are LOADED from src[] and are 0..19
        // so the gather target is the same tiny L1 vector. Isolates the index load.
        let t_cont = timeit!("V_src_cont ids loaded, contiguous targets", {
            for i in 0..n {
                let base = i * k0;
                let mut acc = 0.0f32;
                let mut first = true;
                for j in 0..k0 {
                    let sid = *src_ident.add(base + j) as usize;
                    let term = *w.add(base + j) * *x20.add(sid);
                    acc = if first { term } else { acc + term };
                    first = false;
                }
                *y.add(i) = acc;
            }
            sink += consume(y, n);
        });

        // V_src_rand: the REAL pattern -- ids loaded AND scattered over 1536.
        let t_rand = timeit!("V_src_rand ids loaded, REAL scatter (1536)", {
            for i in 0..n {
                let base = i * k0;
                let mut acc = 0.0f32;
                let mut first = true;
                for j in 0..k0 {
                    let sid = *src.add(base + j) as usize;
                    let term = *w.add(base + j) * *x.add(sid);
                    acc = if first { term } else { acc + term };
                    first = false;
                }
                *y.add(i) = acc;
            }
            sink += consume(y, n);
        });

        // V_b1: the SHIPPED path (vectorised product row + separate scalar reduce)
        let t_b1 = timeit!("V_b1 SHIPPED (vec product row + reduce)", {
            let mut macs = 0usize;
            for i in 0..n {
                let base = i * k0;
                let mut prod = [0.0f32; 256];
                if vmlm_core::interp::HAS_AVX2 {
                    vmlm_core::kernels::sparse_product_row_avx2(
                        w.add(base), src.add(base), x, k0, prod.as_mut_ptr(),
                    );
                } else {
                    vmlm_core::kernels::sparse_product_row_scalar(
                        w.add(base), src.add(base), x, k0, prod.as_mut_ptr(),
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

        let floor_ms = 1.0e-5;
        for (nm, t) in [("dense20", t_dense), ("src_cont", t_cont), ("src_rand", t_rand), ("b1", t_b1)] {
            if t < floor_ms {
                println!("  [FATAL] {} below floor -> deleted loop", nm);
                std::process::exit(1);
            }
        }

        let p = |t: f64| t * 1e6 / e as f64;
        println!("\n===== Q4 RESULT =====");
        println!("  V_dense20  pure dense d=20 dot   {:6.3} ns/conn   (1.000x reference)", p(t_dense));
        println!("  V_src_cont + index load, no scatter {:6.3} ns/conn   ({:.3}x)", p(t_cont), t_cont / t_dense);
        println!("  V_src_rand + index load + scatter   {:6.3} ns/conn   ({:.3}x)", p(t_rand), t_rand / t_dense);
        println!("  V_b1       shipped path          {:6.3} ns/conn   ({:.3}x)", p(t_b1), t_b1 / t_dense);
        println!();
        println!("  ---- decomposition of the B1-vs-dense gap ----");
        println!("  index load (table lookup)     {:6.3} ns/conn  {:5.1}% of V_b1",
            p(t_cont) - p(t_dense), 100.0 * (t_cont - t_dense) / t_b1);
        println!("  scatter / cache misses        {:6.3} ns/conn  {:5.1}% of V_b1",
            p(t_rand) - p(t_cont), 100.0 * (t_rand - t_cont) / t_b1);
        println!("  vectorise+reduce overhead     {:6.3} ns/conn  {:5.1}% of V_b1",
            p(t_b1) - p(t_rand), 100.0 * (t_b1 - t_rand) / t_b1);
        println!();
        println!("  total B1 / dense20 ratio = {:.2}x", t_b1 / t_dense);
        println!();
        println!("  Reads: if B1 really were 'a d=20 dense dot + an index table',");
        println!("  the ratio would be ~1x. The measured ratio is {:.2}x, so the index", t_b1 / t_dense);
        println!("  table is NOT free -- but note WHICH part costs: see above.");

        if sink.is_nan() {
            println!("nan");
        }
    }
}
