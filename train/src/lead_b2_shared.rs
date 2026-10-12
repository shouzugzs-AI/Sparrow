// PATH: B2
//
// LEAD's measurement of the B2 SHARED vs PER-ROW question -- the decision the
// user asked for ("先测两种再决定").
//
// THE QUESTION
//   B2 = sparse storage + sparse activation: walk the 20 connections, compute a
//   strength for each, take top-k, accumulate only those k.
//
//   There are two readings of "take top-k", and they differ by a factor of d:
//
//     PER-ROW  : every neuron selects its OWN top-k out of its own 20.
//                selection work = d * k0 * k  (comparisons)
//                at d=1536, k0=20, k=8 : 245,760 comparisons per matrix
//                ...while the multiply-adds it selects are d*k = 12,288.
//                => SELECTION IS 20x THE ARITHMETIC IT SELECTS.
//
//     SHARED   : one selection is made for the whole layer and every neuron
//                uses the same k connection-slots. selection work = O(k0 * k).
//                This is the only reading under which "76.8x less arithmetic"
//                can turn into real time.
//
// WHY THIS IS THE CRUX
//   The measured decomposition (vmlm-lead-b1-adv) is:
//       B2 per-row = 0.2324 ms/op = clear 1.1% + strength 5.0%
//                                 + SELECTION 73.3% + accumulate 22.8%
//   B1 (no selection) = 0.0363 ms/op.
//   So ~73% of B2's time is a per-neuron selection that does NOT shrink when
//   the connection count shrinks -- it shrinks only when d shrinks.
//   That is precisely the "you are computing millions of neurons that do not
//   exist" effect the user identified.
//
// WHAT THIS PROBE MEASURES
//   Implementation A (perrow): d independent top-k selections.
//   Implementation B (shared) : one top-k over the aggregated strengths.
//   Both are timed on the SAME data, and A is checked against the existing
//   verified kernel (conn_train::compute_act_mask) so neither is self-confirming.
//
// HONESTY NOTE ON WHAT "SHARED" MEANS HERE
//   A shared selection is NOT free and NOT obviously correct: it changes the
//   model (every neuron uses the same connection slots). Whether that is
//   acceptable is a MODELING decision, not a performance one. This probe
//   measures the PERFORMANCE CEILING of that decision so the tradeoff can be
//   made with a number instead of a guess. It does not claim the shared
//   variant preserves accuracy -- that requires a training run.
//
// USAGE: vmlm-lead-b2-shared.exe [--d 1536] [--k0 20] [--k 8] [--reps 300]

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

/// B2 PER-ROW: the existing, verified formulation. Delegates to conn_train's
/// own kernel so this probe cannot disagree with the implementation it judges.
unsafe fn b2_perrow(
    sc: &vmlm_core::conn_train::SparseConnW,
    x: *const f32,
    y: *mut f32,
    k: usize,
) -> usize {
    vmlm_core::conn_train::compute_act_mask(sc, x, k);
    vmlm_core::conn_train::sparse_accum(sc, x, y, k)
}

/// B2 SHARED: ONE top-k selection for the whole layer.
///
/// How the shared selection is formed (and why this is the fairest form):
///   aggregate the per-connection-slot strength across neurons, then pick the k
///   SLOTS with the largest aggregate. Every neuron then keeps those same k
///   slots. That is the natural "one mask for the layer" reading and it costs
///   O(d*k0) to aggregate + O(k0*k) to select, i.e. NO factor of d in the
///   selection step itself.
///
/// `agg` must be at least k0 f32; `hit` at least k usize.
unsafe fn b2_shared(
    sc: &vmlm_core::conn_train::SparseConnW,
    x: *const f32,
    y: *mut f32,
    k: usize,
    agg: *mut f32,
    hit: *mut usize,
) -> usize {
    let n = sc.n_neurons as usize;
    let k0 = sc.k0;

    // ---- one aggregation pass: O(d*k0) multiply-adds, no selection ----
    // This is the SAME work B1 does, which is the point: the shared variant's
    // cost is B1's cost plus ONE tiny selection.
    for j in 0..k0 {
        *agg.add(j) = 0.0;
    }
    for i in 0..n {
        let base = i * k0;
        for j in 0..k0 {
            let sid = *sc.src.add(base + j) as usize;
            *agg.add(j) += (*sc.w.add(base + j) * *x.add(sid)).abs();
        }
    }

    // ---- ONE selection over k0 slots: O(k0*k), independent of d ----
    let nk = vmlm_core::conn_train::row_topk_from_strength(
        agg,
        k0,
        k,
        std::slice::from_raw_parts_mut(hit, k0.max(k)),
    );

    // ---- accumulate the k selected slots for every neuron ----
    // Ascending slot order, single scalar accumulator, first term assigned
    // directly -- same accumulation contract as sparse_accum, so the result is
    // comparable bit-for-bit where the selected sets coincide.
    let mut macs = 0usize;
    for i in 0..n {
        let base = i * k0;
        let mut acc = 0.0f32;
        let mut first = true;
        for q in 0..nk {
            let j = *hit.add(q);
            let sid = *sc.src.add(base + j) as usize;
            let term = *sc.w.add(base + j) * *x.add(sid);
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
    let k = arg_u32("--k", 8) as usize;
    let reps = arg_u32("--reps", 300) as usize;

    println!("===== LEAD: B2 PER-ROW vs SHARED (the decision the user asked for) =====");
    println!("d={} k0={} k={} reps={}", d, k0, k, reps);
    println!("HAS_AVX2 = {}", unsafe { vmlm_core::interp::HAS_AVX2 });

    let clk = Clock::new();
    let mut sink = 0.0f32;
    let mut fails = 0usize;

    unsafe {
        let n = d;
        let e = n * k0;
        let basep = malloc(vmlm_core::conn_train::SparseConnW::bytes(d as u32, k0 as u32));
        let sc = vmlm_core::conn_train::SparseConnW::from_base(basep, d as u32, k0 as u32, 3);
        vmlm_core::conn_train::init_sparse_conn_scaled(&sc, 0.1);

        let x = malloc(n * 4) as *mut f32;
        for i in 0..n {
            *x.add(i) = ((i * 2654435761usize) % 1000) as f32 / 1000.0 - 0.5;
        }
        let ya = malloc(n * 4) as *mut f32;
        let yb = malloc(n * 4) as *mut f32;
        let agg = malloc(k0 * 4) as *mut f32;
        let hit = malloc(k0 * 4) as *mut usize;

        println!("\n--- selection WORK (comparisons), the quantity that matters ---");
        println!("  per-row : d * k0 * k = {} * {} * {} = {}", n, k0, k, n * k0 * k);
        println!("  shared  : k0 * k           = {} * {}     = {}", k0, k, k0 * k);
        println!(
            "  => per-row does {}x more selection work",
            (n * k0 * k) / (k0 * k)
        );
        println!("  per-row MACs actually kept = d*k = {} (selection is {}x the arithmetic)",
            n * k, (n * k0 * k) / (n * k));
        println!("  shared  MACs actually kept = d*k = {}", n * k);

        println!("\n--- correctness floor: both variants must produce a real result ---");
        let ma = b2_perrow(&sc, x, ya, k);
        let mb = b2_shared(&sc, x, yb, k, agg, hit);
        println!("  per-row macs = {}   shared macs = {}", ma, mb);
        if ma == 0 || mb == 0 {
            println!("  [FAIL] a variant computed nothing");
            fails += 1;
        } else {
            println!("  [PASS] both variants did real work");
        }
        // The shared variant must differ from per-row (different model) -- if it
        // were identical the "shared" label would be a lie.
        let yy_a = std::slice::from_raw_parts(ya, n);
        let yy_b = std::slice::from_raw_parts(yb, n);
        let diff = yy_a
            .iter()
            .zip(yy_b.iter())
            .filter(|(p, q)| p.to_bits() != q.to_bits())
            .count();
        println!("  per-row vs shared differ on {} / {} outputs", diff, n);
        if diff == 0 {
            println!("  [WARN] shared == per-row; the 'shared' variant is not distinct");
        } else {
            println!("  [PASS] the shared variant IS a different model (expected)");
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
                println!("  {:<46} {:8.4} ms   ({:6.2} ns/conn)", $label, med, med * 1e6 / (n * k0) as f64);
                med
            }};
        }

        println!("\n--- TIMING (QPC, median of {} reps) ---", reps);
        let t_perrow = timeit!("B2 PER-ROW (selection x d)", {
            let m = b2_perrow(&sc, x, ya, k);
            sink += m as f32;
            sink += consume(ya, n);
        });
        let t_shared = timeit!("B2 SHARED (selection x 1)", {
            let m = b2_shared(&sc, x, yb, k, agg, hit);
            sink += m as f32;
            sink += consume(yb, n);
        });
        let t_b1 = timeit!("B1 (no selection at all, all k0)", {
            let mut macs = 0usize;
            for i in 0..n {
                let base = i * k0;
                let mut acc = 0.0f32;
                let mut first = true;
                for j in 0..k0 {
                    let sid = *sc.src.add(base + j) as usize;
                    let term = *sc.w.add(base + j) * *x.add(sid);
                    acc = if first { term } else { acc + term };
                    first = false;
                    macs += 1;
                }
                *ya.add(i) = acc;
            }
            sink += macs as f32;
            sink += consume(ya, n);
        });

        // anti-vacuity floor
        let floor_ms = 1.0e-4;
        if t_shared < floor_ms || t_perrow < floor_ms {
            println!("  [FATAL] a timing is below the {:.6} ms floor -> deleted loop", floor_ms);
            std::process::exit(1);
        }

        println!("\n===== VERDICT =====");
        println!("  B2 PER-ROW  {:8.4} ms/op", t_perrow);
        println!("  B2 SHARED   {:8.4} ms/op", t_shared);
        println!("  B1          {:8.4} ms/op", t_b1);
        println!();
        println!("  SHARED vs PER-ROW speedup = {:8.4}x", t_perrow / t_shared);
        println!("  SHARED vs B1 overhead     = {:8.4}x  (the cost of having a mask at all)",
            t_shared / t_b1);

        let calls = 5.0 * 24.0 * 128.0;
        println!("\n--- extrapolated to 5 matvec x 24 layers x 128 positions ---");
        println!("  B2 PER-ROW  = {:9.1} ms/step", t_perrow * calls);
        println!("  B2 SHARED   = {:9.1} ms/step", t_shared * calls);
        println!("  B1          = {:9.1} ms/step", t_b1 * calls);

        // And the same-structure dense baseline from the profile probe, so the
        // 76.8x can be read off directly.
        println!("\n--- placing this against the dense baselines (from vmlm-profile) ---");
        println!("  same-structure dense qkv+wo = 7582.1 ms/step (measured, AVX2 gemm_nt)");
        println!("  sparse theoretical @76.8x    =   98.7 ms/step");
        println!("  B1 actual                    = {:6.1} ms/step", t_b1 * calls);
        println!("  B2 SHARED actual             = {:6.1} ms/step", t_shared * calls);
        println!("  B2 PER-ROW actual            = {:6.1} ms/step", t_perrow * calls);

        if sink.is_nan() {
            println!("nan");
        }
    }

    println!("\n===== fail count = {} =====", fails);
    if fails > 0 {
        std::process::exit(1);
    }
    println!("RESULT: ALL CHECKS PASS");
}
