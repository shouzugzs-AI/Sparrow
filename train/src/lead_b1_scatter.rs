// PATH: B1
//
// ADVERSARIAL CHECK of the Q4 result.
//
// THE SURPRISING CLAIM
//   Q4 reported: index load = 0.951 ns/conn, scatter = 0.007 ns/conn.
//   i.e. loading src[j] and converting it costs ~1 ns, while scattering the
//   x access over 1536 floats costs nothing at all.
//
//   That is counter-intuitive: random access into a 6 KB array should cost
//   something. When a measurement says "this known-expensive thing is free",
//   the project's rule is to suspect the probe, not to celebrate.
//
// THE PRIME SUSPECT
//   In bin/b1_dot20.rs the "contiguous" variant read `x20[sid]` where x20 has
//   only k0=20 floats and sid is 0..19. But x20 is built as x20[j] = x[j] --
//   a 20-element array. A 20-float array is 80 bytes: it fits in ONE 64-byte
//   line plus a bit, and the compiler may even keep it in registers.
//   Meanwhile the scattered variant reads a 1536-float array.
//   So the two variants differ in ARRAY SIZE as well as in scatter -- which
//   means the "scatter costs 0.007 ns" reading may be an artifact of the
//   contiguous case being unfairly cheap, not of scatter being free.
//
// HOW THIS PROBE SEPARATES IT
//   S1  gather from a SMALL array (20 floats), sequential ids
//   S2  gather from a BIG array (1536 floats) but with SEQUENTIAL ids
//       -> same array size as the real case, no scatter
//   S3  gather from the BIG array with the REAL scattered ids
//       -> the shipped case
//   S1 vs S2 isolates ARRAY SIZE.  S2 vs S3 isolates SCATTER. Cleanly.
//
// USAGE: vmlm-b1-scatter.exe [--d 1536] [--k0 20] [--reps 3000]

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

    println!("===== ADVERSARIAL: is scatter really free? =====");
    println!("d={} k0={} reps={}", d, k0, reps);

    let clk = Clock::new();
    let mut sink = 0.0f32;
    let n = d;
    let e = n * k0;

    unsafe {
        let w = malloc(e * 4) as *mut f32;
        let src_rand = malloc(e * 4) as *mut f32;
        let src_seq = malloc(e * 4) as *mut f32;
        let x_big = malloc(n * 4) as *mut f32;
        let x_small = malloc(k0 * 4) as *mut f32;
        let y = malloc(n * 4) as *mut f32;

        for i in 0..e {
            *w.add(i) = (((i * 2654435761usize) % 2000) as f32 / 1000.0) - 1.0;
            *src_rand.add(i) = (((i * 40503usize) % n) as u32) as f32;
            // sequential within the row, but spread across the whole array:
            // row i uses ids i*k0 .. i*k0+k0 (mod n) -- ascending, no jumps
            *src_seq.add(i) = ((i % (n - k0)) as u32) as f32;
        }
        for i in 0..n {
            *x_big.add(i) = (((i * 22695477usize) % 1000) as f32 / 1000.0) - 0.5;
        }
        for j in 0..k0 {
            *x_small.add(j) = *x_big.add(j);
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
                println!("  {:<52} {:9.6} ms  {:6.3} ns/conn", $label, med, med * 1e6 / e as f64);
                med
            }};
        }

        println!("\n--- separating ARRAY SIZE from SCATTER ---");

        // S1: read from a SMALL (20-float) array with sequential ids 0..19.
        //
        // ⚠️ The first version wrote `let sid = *src_seq.add(base+j) as usize % k0;`
        // and measured 1.702 ns/conn -- SLOWER than the big-array variants, which
        // is impossible if "small array" is the cheap case. The cause is the `%`
        // itself: an integer modulo is a division, executed once per connection,
        // and it dominated the measurement. That is exactly the "probe costs more
        // than the thing probed" failure this project keeps hitting.
        // Fix: precompute the mod-k0 ids into their own array, OUTSIDE the timing.
        let src_small = malloc(e * 4) as *mut f32;
        for i in 0..e {
            *src_small.add(i) = (i % k0) as f32;
        }

        let t_s1 = timeit!("S1 SMALL array (20f), sequential ids", {
            for i in 0..n {
                let base = i * k0;
                let mut acc = 0.0f32;
                let mut first = true;
                for j in 0..k0 {
                    let sid = *src_small.add(base + j) as usize;
                    let term = *w.add(base + j) * *x_small.add(sid);
                    acc = if first { term } else { acc + term };
                    first = false;
                }
                *y.add(i) = acc;
            }
            sink += consume(y, n);
        });

        let t_s2 = timeit!("S2 BIG array (1536f), sequential ids", {
            for i in 0..n {
                let base = i * k0;
                let mut acc = 0.0f32;
                let mut first = true;
                for j in 0..k0 {
                    let sid = *src_seq.add(base + j) as usize;
                    let term = *w.add(base + j) * *x_big.add(sid);
                    acc = if first { term } else { acc + term };
                    first = false;
                }
                *y.add(i) = acc;
            }
            sink += consume(y, n);
        });

        let t_s3 = timeit!("S3 BIG array, REAL scattered ids", {
            for i in 0..n {
                let base = i * k0;
                let mut acc = 0.0f32;
                let mut first = true;
                for j in 0..k0 {
                    let sid = *src_rand.add(base + j) as usize;
                    let term = *w.add(base + j) * *x_big.add(sid);
                    acc = if first { term } else { acc + term };
                    first = false;
                }
                *y.add(i) = acc;
            }
            sink += consume(y, n);
        });

        // S4: no src array at all -- ids are the loop counter. The true
        // "d=20 dense dot" clean-room baseline.
        let t_s4 = timeit!("S4 NO src array (dense d=20, counter ids)", {
            for i in 0..n {
                let base = i * k0;
                let mut acc = 0.0f32;
                let mut first = true;
                for j in 0..k0 {
                    let term = *w.add(base + j) * *x_small.add(j);
                    acc = if first { term } else { acc + term };
                    first = false;
                }
                *y.add(i) = acc;
            }
            sink += consume(y, n);
        });

        let floor_ms = 1.0e-5;
        for (nm, t) in [("S1", t_s1), ("S2", t_s2), ("S3", t_s3), ("S4", t_s4)] {
            if t < floor_ms {
                println!("  [FATAL] {} below floor -> deleted loop", nm);
                std::process::exit(1);
            }
        }

        let p = |t: f64| t * 1e6 / e as f64;
        println!("\n===== CLEAN DECOMPOSITION =====");
        println!("  S4 no src array, dense d=20     {:6.3} ns/conn  (baseline)", p(t_s4));
        println!("  S1 src loaded -> SMALL array    {:6.3} ns/conn  ({:+.3} vs S4)", p(t_s1), p(t_s1) - p(t_s4));
        println!("  S2 src loaded -> BIG, seq ids   {:6.3} ns/conn  ({:+.3} vs S1 = array size)", p(t_s2), p(t_s2) - p(t_s1));
        println!("  S3 src loaded -> BIG, scattered {:6.3} ns/conn  ({:+.3} vs S2 = SCATTER)", p(t_s3), p(t_s3) - p(t_s2));
        println!();
        println!("  cost of HAVING a src array at all (S4 -> S1): {:+.3} ns/conn", p(t_s1) - p(t_s4));
        println!("  cost of scrubbing a BIG x instead of tiny  : {:+.3} ns/conn", p(t_s2) - p(t_s1));
        println!("  cost of SCATTER within the big array       : {:+.3} ns/conn", p(t_s3) - p(t_s2));
        println!();
        let scatter = p(t_s3) - p(t_s2);
        let total = p(t_s3) - p(t_s4);
        if total > 0.0 {
            println!("  => scatter accounts for {:.1}% of the total B1-vs-dense20 gap.", 100.0 * scatter / total);
        }
        println!("  => the index-mapping table (presence of src[]) is {:.1}%.",
            100.0 * (p(t_s1) - p(t_s4)) / total);

        if sink.is_nan() {
            println!("nan");
        }
    }
}
