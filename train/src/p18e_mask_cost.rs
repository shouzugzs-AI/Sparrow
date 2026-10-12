// P18-E: WHY is the mask 13 ns per connection? Isolate the cost.
//
// Finding so far: compute_act_mask on d=1536, k0=20 costs ~0.4 ms, i.e. ~13 ns
// per connection. A scalar `w*x` with an L1-resident gather should be ~1-2 ns.
// So ~8x is unexplained. This probe decomposes the inner loop into its parts and
// times them SEPARATELY, so the next optimisation is aimed at the real cost
// instead of at a guess.
//
// Candidate costs, in the order they appear in the loop:
//   (1) the gather  x[src[j]]  -- RANDOM access into a 1536-float (6 KB) array.
//       6 KB fits in L1, but the access ORDER is data-dependent.
//   (2) `src[j] as usize` -- src is stored as f32, so this is a cvttss2si
//       (float->int) per connection, on the critical path BEFORE the load.
//   (3) the multiply and abs.
//   (4) the top-k insertion with its branches.
//
// USAGE: vmlm-p18e.exe [--d 1536] [--k0 20] [--k 8] [--reps 200]

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
    fn now(&self) -> i64 {
        let mut c: i64 = 0;
        unsafe { QueryPerformanceCounter(&mut c) };
        c
    }
    fn ms(&self, t: i64) -> f64 {
        t as f64 * 1000.0 / self.freq as f64
    }
}

unsafe fn alloc_f32(n: usize) -> *mut f32 {
    let p = malloc(n * 4) as *mut f32;
    if p.is_null() {
        println!("OOM");
        std::process::exit(9);
    }
    p
}
unsafe fn alloc_u32(n: usize) -> *mut u32 {
    let p = malloc(n * 4) as *mut u32;
    if p.is_null() {
        println!("OOM");
        std::process::exit(9);
    }
    p
}

struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Rng {
        Rng(s | 1)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (z ^ (z >> 31)) as u32
    }
    fn uni(&mut self) -> f32 {
        (self.next_u32() as f64 / (u32::MAX as f64)) as f32
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

fn main() {
    unsafe {
        vmlm_core::interp::init_cpu();
    }
    let d = arg_u32("--d", 1536) as usize;
    let k0 = arg_u32("--k0", 20) as usize;
    let k = arg_u32("--k", 8) as usize;
    let reps = arg_u32("--reps", 200) as usize;
    let clk = Clock::new();

    println!("===== P18-E: decompose the mask cost =====");
    println!("d={} k0={} k={} reps={}", d, k0, k, reps);

    unsafe {
        let e = d * k0;
        let mut rng = Rng::new(99);
        let w = alloc_f32(e);
        let src_f = alloc_f32(e); // src as f32 (what the struct stores)
        let src_u = alloc_u32(e); // src as u32 (the alternative)
        let x = alloc_f32(d);
        for i in 0..e {
            *w.add(i) = rng.uni() * 0.2 - 0.1;
            let s = (rng.next_u32() as usize) % d;
            *src_f.add(i) = s as f32;
            *src_u.add(i) = s as u32;
        }
        for i in 0..d {
            *x.add(i) = rng.uni() * 0.2 - 0.1;
        }

        let n = d;
        let mut sink = 0.0f32;

        macro_rules! timeit {
            ($label:expr, $body:block) => {{
                // warm up
                $body
                let t0 = clk.now();
                for _ in 0..reps {
                    $body
                }
                let t1 = clk.now();
                let per = clk.ms(t1 - t0) / reps as f64;
                let ns = per * 1e6 / (n * k0) as f64;
                println!("  {:<44} {:8.4} ms  {:6.2} ns/conn", $label, per, ns);
                per
            }};
        }

        println!("\n--- (1) gather ONLY, src as f32 (float->int each time) ---");
        let t1 = timeit!("sum += x[src_f[j] as usize]", {
            let mut s = 0.0f32;
            for i in 0..n {
                let base = i * k0;
                for j in 0..k0 {
                    let sid = *src_f.add(base + j) as usize;
                    s += *x.add(sid);
                }
            }
            sink += s;
        });

        println!("\n--- (2) gather ONLY, src as u32 (no conversion) ---");
        let t2 = timeit!("sum += x[src_u[j] as usize]", {
            let mut s = 0.0f32;
            for i in 0..n {
                let base = i * k0;
                for j in 0..k0 {
                    let sid = *src_u.add(base + j) as usize;
                    s += *x.add(sid);
                }
            }
            sink += s;
        });

        println!("\n--- (3) full strength row: |w*x| with f32 src ---");
        let sbuf = alloc_f32(k0);
        let t3 = timeit!("strbuf[j] = (w*x[src]).abs()", {
            for i in 0..n {
                let base = i * k0;
                for j in 0..k0 {
                    let sid = *src_f.add(base + j) as usize;
                    *sbuf.add(j) = (*w.add(base + j) * *x.add(sid)).abs();
                }
            }
            sink += *sbuf.add(0);
        });

        println!("\n--- (4) strength + single-pass top-k (the real mask) ---");
        let t4 = timeit!("strength + row_topk_from_strength", {
            for i in 0..n {
                let base = i * k0;
                for j in 0..k0 {
                    let sid = *src_f.add(base + j) as usize;
                    *sbuf.add(j) = (*w.add(base + j) * *x.add(sid)).abs();
                }
                let mut hit = [0usize; 256];
                let nk = vmlm_core::conn_train::row_topk_from_strength(
                    sbuf,
                    k0,
                    k,
                    &mut hit,
                );
                sink += nk as f32;
            }
        });

        println!("\n--- (5) the ACTUAL compute_act_mask on a real SparseConnW ---");
        let basep = malloc(vmlm_core::conn_train::SparseConnW::bytes(d as u32, k0 as u32));
        let sc = vmlm_core::conn_train::SparseConnW::from_base(basep, d as u32, k0 as u32, 3);
        // copy our w/src in
        for i in 0..e {
            *sc.w.add(i) = *w.add(i);
            *sc.src.add(i) = *src_f.add(i);
        }
        vmlm_core::conn_train::init_sparse_conn_scaled(&sc, 0.1);
        for i in 0..e {
            *sc.w.add(i) = *w.add(i);
            *sc.src.add(i) = *src_f.add(i);
        }
        let t5 = timeit!("compute_act_mask (full, incl. act clear)", {
            vmlm_core::conn_train::compute_act_mask(&sc, x, k);
            sink += *sc.act.add(0);
        });

        println!("\n--- (6) gather into act directly (skip top-k, k=k0 upper bound) ---");
        let t6 = timeit!("clear act + set all (no topk)", {
            for i in 0..e {
                *sc.act.add(i) = 0.0;
            }
            for i in 0..n {
                let base = i * k0;
                for j in 0..k0 {
                    let sid = *sc.src.add(base + j) as usize;
                    if (*sc.w.add(base + j) * *x.add(sid)).abs() > 1e-9 {
                        *sc.act.add(base + j) = 1.0;
                    }
                }
            }
            sink += *sc.act.add(0);
        });

        println!("\n--- breakdown summary ---");
        println!("  gather only (f32 src)   {:8.4} ms", t1);
        println!("  gather only (u32 src)   {:8.4} ms   => f32->int conversion costs {:.1}%",
            t2, 100.0 * (t1 - t2) / t1);
        println!("  strength row            {:8.4} ms", t3);
        println!("  strength + topk         {:8.4} ms   => top-k adds {:.1}%", t4, 100.0 * (t4 - t3) / t4);
        println!("  compute_act_mask real   {:8.4} ms", t5);
        println!("  clear+set (no topk)     {:8.4} ms", t6);

        if sink.is_nan() {
            println!("nan");
        }
    }
}
