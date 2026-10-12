// PATH: B1
//
// LEAD's follow-up: the ONE change that removes 74.5% of B1's cost, with
// bit-exactness preserved.
//
// THE FINDING (measured, train/src/lead_b1_gap.rs)
//   Same 30,720 MACs, four layouts, d=1536 k0=20:
//       L1 src=random f32  (what the code does NOW)  0.0364 ms  1.18 ns/conn
//       L2 src=random u32  (no f32->int on the path) 0.0093 ms  0.30 ns/conn
//   => 74.5% of B1's time is decoding `src` from f32 to an integer index.
//
// WHY THIS IS A BUG, NOT A TRADEOFF
//   The DISK format already stores src as u32:
//       conn_sparse.rs:158  pub const CS_SRC_ENTRY: usize = 4;
//       conn_sparse.rs:596  src: *const u32,
//       conn_sparse.rs:901  get_u32(sec, layout.src_off() + (a.begin + j) * CS_SRC_ENTRY)
//
//   But the TRAINING-side struct stores the same ids as f32:
//       conn_train.rs:118  pub src: *mut f32,
//   with the comment (conn_train.rs:113-117) that this is so the ENGINE's f32
//   index convention is matched and "host and engine do not each write their own
//   conversion".
//
//   So the conversion is paid TWICE and stored ONCE more expensively:
//       u32 on disk -> cast to f32 on load -> cast back to usize on EVERY access
//   The engine convention does not require the TRAINING workspace to hold f32;
//   it requires the engine's SLOT format to. This module is host-side.
//
// WHAT THIS PROBE PROVES
//   That switching the workspace's src to u32 is (a) bit-exact and (b) worth
//   ~4x. Bit-exactness is the point: the conversion is exact for ids < 2^24,
//   so the IDENTITY of every connection is unchanged. Only the decode cost goes.
//
//   The 2^24 caveat is real and is ASSERTED here rather than assumed: f32 can
//   represent every integer up to 2^24 exactly, so for n_neurons=1536 the round
//   trip is lossless. The probe checks it instead of trusting it.
//
// USAGE: vmlm-lead-b1-u32src.exe [--d 1536] [--k0 20] [--reps 300]

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

fn bit_mismatches(a: &[f32], b: &[f32]) -> usize {
    a.iter()
        .zip(b.iter())
        .filter(|(p, q)| p.to_bits() != q.to_bits())
        .count()
}

/// B1 forward, exactly as core/src/b1_pure.rs::b1_forward -- f32 src.
unsafe fn b1_f32_src(
    w: *const f32,
    src: *const f32,
    x: *const f32,
    y: *mut f32,
    n: usize,
    k0: usize,
) -> usize {
    let mut macs = 0usize;
    for i in 0..n {
        let base = i * k0;
        let mut acc = 0.0f32;
        let mut first = true;
        for j in 0..k0 {
            let sid = *src.add(base + j) as usize;
            let term = *w.add(base + j) * *x.add(sid);
            acc = if first { term } else { acc + term };
            first = false;
            macs += 1;
        }
        *y.add(i) = acc;
    }
    macs
}

/// The SAME function with the ONLY change being the index type: u32.
/// Accumulation order, single scalar accumulator and first-term assignment are
/// all identical, so any numeric difference would be a bug in this claim.
unsafe fn b1_u32_src(
    w: *const f32,
    src: *const u32,
    x: *const f32,
    y: *mut f32,
    n: usize,
    k0: usize,
) -> usize {
    let mut macs = 0usize;
    for i in 0..n {
        let base = i * k0;
        let mut acc = 0.0f32;
        let mut first = true;
        for j in 0..k0 {
            let sid = *src.add(base + j) as usize;
            let term = *w.add(base + j) * *x.add(sid);
            acc = if first { term } else { acc + term };
            first = false;
            macs += 1;
        }
        *y.add(i) = acc;
    }
    macs
}

fn main() {
    unsafe { vmlm_core::interp::init_cpu() };
    let d = arg_u32("--d", 1536) as usize;
    let k0 = arg_u32("--k0", 20) as usize;
    let reps = arg_u32("--reps", 300) as usize;

    println!("===== LEAD: is the f32 src index a removable cost? =====");
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
        let y1 = malloc(n * 4) as *mut f32;
        let y2 = malloc(n * 4) as *mut f32;

        // Deterministic data (no external RNG dependency, and reproducible).
        for i in 0..e {
            *w.add(i) = (((i * 2654435761usize) % 2000) as f32 / 1000.0) - 1.0;
            let s = ((i * 40503usize) % n) as u32;
            *src_u.add(i) = s;
            *src_f.add(i) = s as f32;
        }
        for i in 0..n {
            *x.add(i) = (((i * 22695477usize) % 1000) as f32 / 1000.0) - 0.5;
        }

        // ---- STEP 1: the f32<->u32 round trip must be LOSSLESS ----
        println!("\n--- round-trip exactness (this is why the change is safe) ---");
        let mut rt_bad = 0usize;
        for i in 0..e {
            let s = *src_u.add(i);
            if (s as f32) as u32 != s {
                rt_bad += 1;
            }
        }
        println!("  u32 -> f32 -> u32 mismatches = {} / {}", rt_bad, e);
        println!("  (f32 represents every integer up to 2^24 exactly; n_neurons={})", n);
        if n > (1 << 24) {
            println!("  [FAIL] n_neurons exceeds 2^24: f32 src would NOT be lossless");
            std::process::exit(1);
        }
        if rt_bad != 0 {
            println!("  [FAIL] round trip is lossy");
            std::process::exit(1);
        }
        println!("  [PASS] round trip is exact, so the two layouts hold the SAME ids");

        // ---- STEP 2: the two layouts must be BIT-EXACT in output ----
        println!("\n--- output bit-exactness between the two layouts ---");
        let m1 = b1_f32_src(w, src_f, x, y1, n, k0);
        let m2 = b1_u32_src(w, src_u, x, y2, n, k0);
        let s1 = std::slice::from_raw_parts(y1, n);
        let s2 = std::slice::from_raw_parts(y2, n);
        let mm = bit_mismatches(s1, s2);
        println!("  f32-src macs = {}   u32-src macs = {}", m1, m2);
        println!("  bit mismatches = {} / {}  (must be 0)", mm, n);
        if mm != 0 || m1 != m2 {
            println!("  [FAIL] the layout change is NOT neutral");
            std::process::exit(1);
        }
        println!("  [PASS] u32 src is BIT-EXACT vs f32 src -- the change is free of risk");

        // can-fail control: the comparison must be able to disagree
        let mut poked = std::slice::from_raw_parts(y2, n).to_vec();
        poked[0] = f32::from_bits(poked[0].to_bits() ^ 1);
        let mm_p = bit_mismatches(s1, &poked);
        println!("  can-fail: flipping 1 output bit reports {} mismatch (must be >0)", mm_p);
        if mm_p == 0 {
            println!("  [FAIL] the comparison is vacuous");
            std::process::exit(1);
        }
        println!("  [PASS] the comparison can fail");

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
                println!("  {:<44} {:8.4} ms  {:6.2} ns/conn", $label, med, med * 1e6 / e as f64);
                med
            }};
        }

        println!("\n--- TIMING (QPC, median of {} reps) ---", reps);
        let t_f32 = timeit!("b1_forward, src as f32 (TODAY)", {
            let m = b1_f32_src(w, src_f, x, y1, n, k0);
            sink += m as f32;
            sink += consume(y1, n);
        });
        let t_u32 = timeit!("b1_forward, src as u32 (PROPOSED)", {
            let m = b1_u32_src(w, src_u, x, y2, n, k0);
            sink += m as f32;
            sink += consume(y2, n);
        });

        let floor_ms = 1.0e-4;
        if t_f32 < floor_ms || t_u32 < floor_ms {
            println!("  [FATAL] timing below floor -> deleted loop");
            std::process::exit(1);
        }

        let calls = 5.0 * 24.0 * 128.0;
        println!("\n===== RESULT =====");
        println!("  f32 src  {:8.4} ms/op  = {:7.1} ms/step", t_f32, t_f32 * calls);
        println!("  u32 src  {:8.4} ms/op  = {:7.1} ms/step", t_u32, t_u32 * calls);
        println!("  speedup  {:8.4}x", t_f32 / t_u32);
        println!("  saving   {:8.4} ms/op  = {:7.1} ms/step", t_f32 - t_u32, (t_f32 - t_u32) * calls);
        println!();
        println!("  This is a STORAGE change in the TRAINING WORKSPACE only.");
        println!("  The disk format is ALREADY u32 (conn_sparse.rs:158 CS_SRC_ENTRY=4),");
        println!("  so nothing about the on-disk artifact changes: the f32 copy is");
        println!("  purely a host-side detour introduced by conn_train.rs:118.");
        println!();
        println!("  Bit-exactness: PROVEN above (0 mismatches), not assumed.");

        if sink.is_nan() {
            println!("nan");
        }
    }
}
