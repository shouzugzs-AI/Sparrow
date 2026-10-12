// PATH: B1
//
// LEAD's investigation of the REMAINING GAP.
//
// WHERE WE ARE (all measured, see the sibling probes)
//   same-structure dense qkv+wo     = 7582.1 ms/step   (vmlm-profile, AVX2)
//   sparse theoretical @ 76.8x      =   98.7 ms/step
//   B1 actual (all 20, no mask)     =  560.6 ms/step   <-- 5.7x SLOWER than theory
//   B2 shared                       =  850.9 ms/step
//   B2 per-row (current code)       = 3664.1 ms/step
//
// So B1 already removes the selection and lands 6.5x better than the current
// per-row path -- but it is STILL 5.7x off the 76.8x figure. This probe asks
// why, and the answer decides whether "translate to machine code" is the fix.
//
// THE HYPOTHESIS TO TEST
//   B1 does e = d*k0 = 30,720 multiply-adds per matvec, i.e. 3.78e8 MACs across
//   the whole step (5 matvec x 24 layers x 128 positions). Even at a modest
//   4 GFLOP/s that is ~10 ms/step. Measured: 560 ms. So B1 is bound by
//   SOMETHING OTHER THAN arithmetic.
//
//   Two candidates, both measurable:
//     H1  THE GATHER. Each term needs x[src[i][j]] where src is effectively
//         random over 1536 floats. That is a random load per MAC -- no
//         ILP-friendly stream. Measured earlier at 0.37 ns/conn for the
//         strength row, and 1.19 ns/conn for the full B1.
//     H2  THE f32->int CONVERSION of src. conn_train stores src as f32 (the
//         engine carries indices as f32), so every gather pays cvttss2si on the
//         critical path. Measured earlier (vmlm-p18e): 1.20 ns/conn with f32
//         src vs 0.49 ns/conn with u32 src => the conversion is 59% of a bare
//         gather. That is a LAYOUT decision, not a codegen decision.
//
// WHY THIS MATTERS FOR THE TRANSLATION DECISION
//   If B1 is gather-bound (H1/H2), then inlining it into machine code changes
//   almost nothing -- the loads stay random and the conversion stays on the
//   path. If B1 is CALL/DISPATCH-bound, then inlining is the fix.
//   This probe distinguishes the two by measuring the same arithmetic under
//   different data layouts, which is a change NO compiler or JIT can make.
//
// USAGE: vmlm-lead-b1-gap.exe [--d 1536] [--k0 20] [--reps 300]

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

/// A tiny deterministic PRNG, so the sparse indices are reproducible and not
/// collapsible. Same constants as the repo's other probes.
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

fn main() {
    unsafe { vmlm_core::interp::init_cpu() };
    let d = arg_u32("--d", 1536) as usize;
    let k0 = arg_u32("--k0", 20) as usize;
    let reps = arg_u32("--reps", 300) as usize;

    println!("===== LEAD: where does B1's remaining 5.7x go? =====");
    println!("d={} k0={} reps={}", d, k0, reps);
    println!("HAS_AVX2 = {}", unsafe { vmlm_core::interp::HAS_AVX2 });

    let clk = Clock::new();
    let mut sink = 0.0f32;
    let n = d;
    let e = n * k0;

    unsafe {
        let mut rng = Rng::new(12345);

        // ---- Layout 1: what the code ACTUALLY has (src as f32, random order)
        let w = malloc(e * 4) as *mut f32;
        let src_f = malloc(e * 4) as *mut f32;
        let src_u = malloc(e * 4) as *mut u32;
        let x = malloc(n * 4) as *mut f32;
        let y = malloc(n * 4) as *mut f32;
        for i in 0..e {
            *w.add(i) = rng.uni() * 0.2 - 0.1;
            let s = (rng.next_u32() as usize) % n;
            *src_f.add(i) = s as f32;
            *src_u.add(i) = s as u32;
        }
        for i in 0..n {
            *x.add(i) = rng.uni() * 0.2 - 0.1;
        }

        // ---- Layout 2: sorted src (sequential-ish access, same arithmetic)
        //      Same multiset of (w, src) pairs, but the src ids ascend. This is
        //      a PURE LAYOUT change: identical MAC count, identical values.
        let mut pairs: Vec<(u32, f32)> = (0..e).map(|i| (*src_u.add(i), *w.add(i))).collect();
        pairs.sort_by_key(|p| p.0);
        let src_sorted = malloc(e * 4) as *mut u32;
        let w_sorted = malloc(e * 4) as *mut f32;
        for (i, (s, wv)) in pairs.iter().enumerate() {
            *src_sorted.add(i) = *s;
            *w_sorted.add(i) = *wv;
        }

        // ---- Layout 3: DENSE, contiguous, no gather at all.
        //      This is the ceiling: same MAC count as layout 1, but every load
        //      is a stream. It answers "how fast could this be if the gather
        //      were free?", which is exactly what a JIT/translator CANNOT fix.
        let w_dense = malloc(e * 4) as *mut f32;
        let x_padded = malloc(e * 4) as *mut f32;
        for i in 0..e {
            *w_dense.add(i) = *w.add(i);
            *x_padded.add(i) = *x.add(*src_u.add(i) as usize); // pre-gathered
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
                println!("  {:<52} {:8.4} ms  {:6.2} ns/conn", $label, med, med * 1e6 / e as f64);
                med
            }};
        }

        println!("\n--- the SAME 30,720 MACs, four data layouts ---");

        // L1: what the code has -- src as f32, random gather
        let t_f32 = timeit!("L1 src=random f32  (what the code does NOW)", {
            let mut macs = 0usize;
            for i in 0..n {
                let base = i * k0;
                let mut acc = 0.0f32;
                let mut first = true;
                for j in 0..k0 {
                    let sid = *src_f.add(base + j) as usize;
                    let term = *w.add(base + j) * *x.add(sid);
                    acc = if first { term } else { acc + term };
                    first = false;
                    macs += 1;
                }
                *y.add(i) = acc;
            }
            sink += macs as f32;
            sink += consume(y, n);
        });

        // L2: src as u32 -- removes the cvttss2si on the critical path
        let t_u32 = timeit!("L2 src=random u32  (no f32->int on the path)", {
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

        // L3: sorted src -- same gathers, but cache-friendly order
        let t_sorted = timeit!("L3 src=SORTED u32 (gather, but ascending)", {
            for i in 0..n {
                let base = i * k0;
                let mut acc = 0.0f32;
                let mut first = true;
                for j in 0..k0 {
                    let sid = *src_sorted.add(base + j) as usize;
                    let term = *w_sorted.add(base + j) * *x.add(sid);
                    acc = if first { term } else { acc + term };
                    first = false;
                }
                *y.add(i) = acc;
            }
            sink += consume(y, n);
        });

        // L4: dense, pre-gathered -- NO index load, NO random access
        let t_dense = timeit!("L4 DENSE pre-gathered (no gather at all)", {
            for i in 0..n {
                let base = i * k0;
                let mut acc = 0.0f32;
                let mut first = true;
                for j in 0..k0 {
                    let term = *w_dense.add(base + j) * *x_padded.add(base + j);
                    acc = if first { term } else { acc + term };
                    first = false;
                }
                *y.add(i) = acc;
            }
            sink += consume(y, n);
        });

        // L5: the pure arithmetic ceiling -- no memory at all.
        //
        // ANTI-VACUITY: the serial dependency chain a=b*a+c is what stops the
        // optimiser collapsing this, but the first version still reported
        // 0.0000 ms because `sink` was accumulated and printed but LLVM hoisted
        // the chain out of the timing region. Reading through `consume` and
        // requiring a floor makes the loop observable.
        let t_ceiling = timeit!("L5 pure FMA on registers (no loads)", {
            let mut a = 1.0f32;
            let mut b = 1.000001f32;
            for _ in 0..(n * k0) {
                a = a * b + 0.5;
                b = b * 0.999999 + 0.25;
            }
            // publish through a real buffer read so the chain cannot move out
            *y.add(0) = a;
            *y.add(1) = b;
            sink += consume(y, n);
        });

        let floor_ms = 1.0e-4;
        for (nm, t) in [("L1", t_f32), ("L2", t_u32), ("L3", t_sorted), ("L4", t_dense)] {
            if t < floor_ms {
                println!("  [FATAL] {} timed below the floor -> deleted loop", nm);
                std::process::exit(1);
            }
        }
        // L5 is a dependency chain over n*k0 iterations; it cannot be faster
        // than ~1 cycle per iteration (~0.01 ms at 3 GHz for 30720 iters).
        let l5_floor = 30_720.0 / 3.0e9 * 1000.0 * 0.2; // 20% of 1 cycle/iter
        if t_ceiling < l5_floor {
            println!(
                "  [WARN] L5 timed {:.6} ms, below the dependency-chain floor {:.6} ms.",
                t_ceiling, l5_floor
            );
            println!("         Reporting L5 as UNRELIABLE (chain partially hoisted).");
            println!("         It is EXCLUDED from the attribution below rather than");
            println!("         silently reported as 0.0000 -- a false zero is worse than");
            println!("         a missing number (see docs/P18_步骤0_剖析实测.md).");
        }

        println!("\n===== ATTRIBUTION =====");
        println!("  L1 actual layout            {:8.4} ms  100.0%", t_f32);
        println!("  L2 u32 src saves            {:8.4} ms  {:5.1}%  (f32->int conversion)",
            t_f32 - t_u32, 100.0 * (t_f32 - t_u32) / t_f32);
        println!("  L3 sorted src saves         {:8.4} ms  {:5.1}%  (cache-friendly gather)",
            t_u32 - t_sorted, 100.0 * (t_u32 - t_sorted) / t_u32);
        println!("  L4 no gather saves          {:8.4} ms  {:5.1}%  (RANDOM ACCESS = the real cost)",
            t_sorted - t_dense, 100.0 * (t_sorted - t_dense) / t_sorted);
        if t_ceiling >= l5_floor {
            println!("  L5 pure FMA on registers    {:8.4} ms  (arithmetic is {:.1}x of L1)",
                t_ceiling, t_f32 / t_ceiling);
        } else {
            println!("  L5 pure FMA on registers    UNRELIABLE (see warning above) -- omitted");
        }

        let calls = 5.0 * 24.0 * 128.0;
        println!("\n--- extrapolated to 5 matvec x 24 layers x 128 positions ---");
        println!("  L1 as-is      = {:9.1} ms/step", t_f32 * calls);
        println!("  L2 u32 src    = {:9.1} ms/step", t_u32 * calls);
        println!("  L3 sorted     = {:9.1} ms/step", t_sorted * calls);
        println!("  L4 no gather  = {:9.1} ms/step   <-- the layout ceiling", t_dense * calls);
        println!("  theoretical   =     98.7 ms/step   (dense/76.8)");

        println!("\n===== WHAT THIS MEANS FOR THE TRANSLATION DECISION =====");
        // Decompose against the LAYOUT CEILING (L4), not against a vacuous L5.
        // L4 is the same 30720 MACs with every load a stream, so whatever L1
        // costs above L4 is attributable to the ACCESS PATTERN and the index
        // format -- things a code generator cannot change.
        let layout_share = 100.0 * (t_f32 - t_dense) / t_f32;
        println!("  L1 vs the layout ceiling L4: {:.1}% of B1's time is", layout_share);
        println!("  the ACCESS PATTERN (f32 index decode + random gather).");
        println!();
        println!("  Breakdown of what is fixable and by WHOM:");
        println!("    {:.1}%  f32->int index decode  -> fixable by STORAGE (src as u32)",
            100.0 * (t_f32 - t_u32) / t_f32);
        println!("    {:.1}%  random-access gather   -> fixable by LAYOUT only",
            100.0 * (t_u32 - t_dense) / t_u32);
        println!("    {:.1}%  the remaining L4 cost  -> arithmetic/streaming floor",
            100.0 * t_dense / t_f32);
        println!();
        println!("  A translator CANNOT change the access pattern: the 20 sources are");
        println!("  what they are. Inlining them into machine code keeps the same 20");
        println!("  random loads. Only a LAYOUT change (L2/L3/L4) moves this number.");
        println!();
        println!("  => if the goal is the 76.8x figure, the lever is the STORAGE");
        println!("     LAYOUT (u32 src, and a source ordering that makes the rows");
        println!("     contiguous), NOT the code generator.");
    }
}
