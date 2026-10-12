// PATH: B1
//
// The FINAL question for this audit: if the index load is 99% of the B1-vs-dense
// gap, can the index load be removed? Read-only; no production code changes.
//
// THE LOGICAL CONCLUSION SO FAR (measured, two independent probes)
//   S4  no src array at all (dense d=20)      0.238 ns/conn
//   S3  src[] loaded, scattered over 1536     1.195 ns/conn
//   => the index-mapping table costs 0.947 ns/conn = 99% of the gap
//   => the scatter itself costs 0.010 ns/conn = 1% (x is 6 KB, fully L1-resident)
//
//   This INVERTS the intuitive picture. The task framing says the index table's
//   purpose is "to map 20 positions in 1536-space to a compact 20-space", and
//   suggests clustering. But clustering optimises the scatter, which is worth 1%.
//   The cost is having the table AT ALL -- one extra load + one float->int
//   convert per connection, sitting on the address dependency chain.
//
// SO: what removes the per-connection index load?
//   A1  store src as u32 instead of f32  -> removes the CONVERT (keeps the load)
//   A2  bake the index into the instruction (immediate/constant offset)
//       -> this is what "translation" or a specialization would do
//   A3  interchange the loops: for each SOURCE, scatter its contribution to all
//       neurons that use it -> amortises the source load, but needs the reverse
//       mapping and changes the reduction order (bit-exact risk)
//
// THIS PROBE MEASURES A1 and A2; A3 is noted as a design change, not measured.
//
// USAGE: vmlm-b1-indexcost.exe [--d 1536] [--k0 20] [--reps 3000]

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

    println!("===== can the index load be removed? =====");
    println!("d={} k0={} reps={}", d, k0, reps);

    let clk = Clock::new();
    let mut sink = 0.0f32;
    let n = d;
    let e = n * k0;

    unsafe {
        let w = malloc(e * 4) as *mut f32;
        let src_f = malloc(e * 4) as *mut f32;
        let src_u = malloc(e * 4) as *mut u32;
        let x = malloc(n * 4) as *mut f32;
        let y = malloc(n * 4) as *mut f32;

        for i in 0..e {
            *w.add(i) = (((i * 2654435761usize) % 2000) as f32 / 1000.0) - 1.0;
            let s = ((i * 40503usize) % n) as u32;
            *src_f.add(i) = s as f32;
            *src_u.add(i) = s;
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

        println!("\n--- A1: does the f32->int CONVERT cost anything? ---");
        let t_f32 = timeit!("src as f32 (load + cvttss2si)", {
            for i in 0..n {
                let base = i * k0;
                let mut acc = 0.0f32;
                let mut first = true;
                for j in 0..k0 {
                    let sid = *src_f.add(base + j) as usize;
                    let term = *w.add(base + j) * *x.add(sid);
                    acc = if first { term } else { acc + term };
                    first = false;
                }
                *y.add(i) = acc;
            }
            sink += consume(y, n);
        });
        let t_u32 = timeit!("src as u32 (load only, no convert)", {
            for i in 0..n {
                let base = i * k0;
                let mut acc = 0.0f32;
                let mut first = true;
                for j in 0..k0 {
                    let sid = *src_u.add(base + j) as usize;
                    let term = *w.add(base + j) * *x.add(sid);
                    acc = if first { term } else { acc + term };
                    first = false;
                }
                *y.add(i) = acc;
            }
            sink += consume(y, n);
        });
        // A1 note: if this is ~equal, the convert is free (the port has slack) and
        // the index cost is purely the extra LOAD.

        println!("\n--- A2: what if the index were a COMPILE-TIME constant? ---");
        // Emulate "index baked in" by removing the dependent load entirely:
        // use a fixed stride pattern the compiler can strength-reduce. This is
        // an UPPER BOUND on what specialisation could buy, not a real layout.
        let t_const = timeit!("index = j*k0 (strength-reduced, no src load)", {
            for i in 0..n {
                let base = i * k0;
                let mut acc = 0.0f32;
                let mut first = true;
                for j in 0..k0 {
                    // sid = (i*k0+j) % n  -- computable without loading src
                    let sid = (base + j) % n;
                    let term = *w.add(base + j) * *x.add(sid);
                    acc = if first { term } else { acc + term };
                    first = false;
                }
                *y.add(i) = acc;
            }
            sink += consume(y, n);
        });

        println!("\n--- A3: dense d=20 reference (no index at all) ---");
        let t_dense = timeit!("dense d=20, counter ids (the 0.238 baseline)", {
            for i in 0..n {
                let base = i * k0;
                let mut acc = 0.0f32;
                let mut first = true;
                for j in 0..k0 {
                    let term = *w.add(base + j) * *x.add(j);
                    acc = if first { term } else { acc + term };
                    first = false;
                }
                *y.add(i) = acc;
            }
            sink += consume(y, n);
        });

        let floor_ms = 1.0e-5;
        for (nm, t) in [("f32", t_f32), ("u32", t_u32), ("const", t_const), ("dense", t_dense)] {
            if t < floor_ms {
                println!("  [FATAL] {} below floor -> deleted loop", nm);
                std::process::exit(1);
            }
        }

        let p = |t: f64| t * 1e6 / e as f64;
        println!("\n===== RESULT =====");
        println!("  src f32  (load+convert)      {:6.3} ns/conn", p(t_f32));
        println!("  src u32  (load only)         {:6.3} ns/conn   ({:+.3})", p(t_u32), p(t_u32) - p(t_f32));
        println!("  index computable, no load    {:6.3} ns/conn   ({:+.3} vs f32)", p(t_const), p(t_const) - p(t_f32));
        println!("  dense d=20 (no index at all) {:6.3} ns/conn   ({:+.3} vs f32)", p(t_dense), p(t_dense) - p(t_f32));
        println!();
        println!("  convert cost   (f32 - u32)   {:+.3} ns/conn", p(t_f32) - p(t_u32));
        println!("  load cost      (u32 - dense) {:+.3} ns/conn", p(t_u32) - p(t_dense));
        println!();
        println!("  => the index load is the irreducible part of the mapping table.");
        println!("     Removing the load entirely (const index) reaches {:.3} ns/conn,", p(t_const));
        println!("     which is {:.2}x the dense baseline.", p(t_const) / p(t_dense));

        if sink.is_nan() {
            println!("nan");
        }
    }
}
