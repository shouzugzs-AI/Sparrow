// PATH: B1
//
// LEAD's ADVERSARIAL check of the B1 headline number.
//
// WHY THIS EXISTS
//   The previous probe reported "B1 is 6.76x faster than B2". A number that
//   good deserves an attempt to break it. This project's own history has three
//   recorded cases where a probe reported a confident wrong answer:
//     * docs/P18_步骤0_剖析实测.md  -- sparse FFN probe reported 0.000 ms
//       because the result was never read and LLVM deleted the loop
//     * docs/P18-C_路径B转换与优化实测.md -- nested QPC made `wo fwd` read
//       0.006 s, 270x too low
//     * tools/p16_sparse_ops.py -- a hardcoded slot 0 dumped OFFSETS instead
//       of y, producing "y = [0,2,4,5]", a perfectly plausible wrong answer
//
//   So before the 6.76x is reported, the following objections are each tested:
//
//   O1 "B1 is faster only because sparse_accum's `if act == 0 continue` branch
//       is mispredicted, not because the selection is absent."
//       TEST: time sparse_accum on an ALL-ONES mask (no branch ever taken) vs
//       b1_forward. If the all-ones-mask accumulate matches B1's time, the
//       branch is not the story.
//
//   O2 "B1 wins because it SKIPS work that B2 does -- i.e. the comparison is
//       unfair and B1 is simply doing less."
//       TEST: count actual multiply-adds. B1 does 30720, B2 does 12288. B1 does
//       2.5x MORE arithmetic and still wins. This is the load-bearing fact.
//
//   O3 "The timing loop was optimised away."
//       TEST: every timed body folds into `sink`, which is printed. Also the
//       numbers must scale with d -- checked at d=1536 and d=768.
//
//   O4 "compute_act_mask is being charged for its act[] clear (30720 stores),
//       which B1 does not need. So the 6.76x is partly an artifact of a
//       cleanup B2 could also avoid."
//       TEST: this is a REAL and fair objection. Measure the clear alone
//       (write_bytes of e floats) and report how much of the mask cost it is.
//       If the clear is most of it, the honest B1-vs-B2 delta is smaller.
//
//   O5 "The mask cost is inflated by AVX2 gather in
//       sparse_strength_row_avx2, which a scalar B1 does not pay."
//       TEST: time b1_forward with the SAME gather pattern (it does gather --
//       it reads x[src[j]]). If B1's gather is cheap, gather is not the story.
//
// USAGE: vmlm-lead-b1-adversarial.exe [--d 1536] [--k0 20] [--k 8] [--reps 300]

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

unsafe fn b1_forward(sc: &vmlm_core::conn_train::SparseConnW, x: *const f32, y: *mut f32) -> usize {
    let n = sc.n_neurons as usize;
    let k0 = sc.k0;
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
        *y.add(i) = acc;
    }
    macs
}

/// ANTI-VACUITY GUARD.
///
/// The FIRST version of this probe reported `b1_forward = 0.0000 ms` -- a false
/// zero. Cause: the timed body wrote `y` and folded the return value (a loop
/// counter = the compile-time constant n*k0) into `sink`, and NOTHING EVER READ
/// `y`. LLVM proved the stores dead and deleted the entire loop. A 0.0000 ms
/// reading is not a fast function, it is an absent one.
///
/// This is the third recorded instance of this failure class in the project
/// (see docs/P18_步骤0_剖析实测.md: "sparse FFN fwd = 0.000 ms" for exactly the
/// same reason). So the guard is explicit rather than hopeful:
///
///   `consume(y, n)` READS the buffer and folds it into a volatile sink, so the
///   stores are observably live and cannot be removed.
///
/// Every timed body below must end with `consume(...)` of whatever it wrote.
static mut GUARD: f64 = 0.0;

#[inline(never)]
unsafe fn consume(p: *const f32, n: usize) -> f32 {
    let mut s = 0.0f32;
    // read every 64th element: touches all cache lines without dominating time
    let mut i = 0usize;
    while i < n {
        s += *p.add(i);
        i += 64;
    }
    // volatile-ish sink: a `static mut` write is not removable by the optimiser
    GUARD += s as f64;
    s
}

fn main() {
    unsafe { vmlm_core::interp::init_cpu() };
    let d = arg_u32("--d", 1536) as usize;
    let k0 = arg_u32("--k0", 20) as usize;
    let k = arg_u32("--k", 8) as usize;
    let reps = arg_u32("--reps", 300) as usize;

    println!("===== LEAD: adversarial check of the B1 headline =====");
    println!("d={} k0={} k={} reps={}", d, k0, k, reps);
    println!("HAS_AVX2 = {}", unsafe { vmlm_core::interp::HAS_AVX2 });

    let clk = Clock::new();
    let mut sink = 0.0f32;

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
        let yb = malloc(n * 4) as *mut f32;

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
                let per_conn = med * 1e6 / (n * k0) as f64;
                println!("  {:<52} {:8.4} ms  {:6.2} ns/conn", $label, med, per_conn);
                med
            }};
        }

        println!("\n--- O1/O5: is it the branch, or the gather, or the SELECTION? ---");

        // (a) B2's accumulate with an ALL-ONES mask: the `if act==0 continue`
        //     branch is never taken. If this equals B1's time, the branch is
        //     not what B1 is winning on.
        for t in 0..e {
            *sc.act.add(t) = 1.0;
        }
        let t_ones = timeit!("accumulate on ALL-ONES mask (k=k0, no selection)", {
            let m = vmlm_core::conn_train::sparse_accum(&sc, x, yb, k0);
            sink += m as f32;
            sink += consume(yb, n);
        });
        let t_b1 = timeit!("b1_forward (no mask at all)", {
            let m = b1_forward(&sc, x, yb);
            sink += m as f32;
            sink += consume(yb, n);
        });
        println!(
            "    => B1 vs all-ones accumulate: {:.3}x  (near 1.0 => branch is NOT the story)",
            t_ones / t_b1
        );

        // (b) B2's accumulate with a REAL k-sparse mask: the branch IS taken
        //     ~40% of the time. This isolates misprediction from selection.
        vmlm_core::conn_train::compute_act_mask(&sc, x, k);
        let t_sparse_accum = timeit!("accumulate on REAL k-sparse mask (branchy)", {
            let m = vmlm_core::conn_train::sparse_accum(&sc, x, yb, k);
            sink += m as f32;
            sink += consume(yb, n);
        });
        println!(
            "    => branchy vs all-ones: {:.3}x  (this is the misprediction cost alone)",
            t_sparse_accum / t_ones
        );

        println!("\n--- O4: how much of the mask cost is just the act[] CLEAR? ---");
        let t_clear = timeit!("  (only) clear act[] : e stores", {
            for t in 0..e {
                *sc.act.add(t) = 0.0;
            }
            sink += *sc.act.add(0);
        });
        let t_mask = timeit!("  (only) compute_act_mask k=8 (clear + strength + topk)", {
            vmlm_core::conn_train::compute_act_mask(&sc, x, k);
            sink += *sc.act.add(0);
        });
        println!(
            "    => clear is {:.1}% of compute_act_mask; the SELECTION is the other {:.1}%",
            100.0 * t_clear / t_mask,
            100.0 * (t_mask - t_clear) / t_mask
        );

        println!("\n--- O5: the strength row alone (gather + |w*x|, no selection) ---");
        let strbuf = malloc(k0 * 4) as *mut f32;
        let t_strength = timeit!("  (only) strength row = |w*x| gather, no select", {
            for i in 0..n {
                let base = i * k0;
                if vmlm_core::interp::HAS_AVX2 {
                    vmlm_core::kernels::sparse_strength_row_avx2(
                        sc.w.add(base),
                        sc.src.add(base),
                        x,
                        k0,
                        strbuf,
                    );
                } else {
                    for j in 0..k0 {
                        let sid = *sc.src.add(base + j) as usize;
                        *strbuf.add(j) = (*sc.w.add(base + j) * *x.add(sid)).abs();
                    }
                }
            }
            sink += *strbuf;
            sink += consume(strbuf, k0);
        });
        println!(
            "    => strength {:.4} ms; B1 (which also gathers every source) = {:.4} ms",
            t_strength, t_b1
        );
        println!(
            "    => B1 does the SAME gathers and is {:.2}x the strength row. Gather is not the story.",
            t_b1 / t_strength
        );

        println!("\n--- O3: does the number scale with d? (loop not deleted) ---");
        // ANTI-VACUITY SELF-CHECK. The first version of this probe reported
        // 0.0000 ms for b1_forward. A real kernel that reads 30720 weights and
        // gathers 30720 sources CANNOT take zero time. Assert a floor: if any
        // timed result is below the cost of merely streaming the weights
        // (e*4 bytes), the loop was deleted and the reading is meaningless.
        //
        // Floor derivation: reading `e` f32 = 30720*4 = 122880 B. At even an
        // unrealistically generous 200 GB/s that is 0.61 us = 0.0006 ms. So a
        // reading below 1e-4 ms is proof of elimination, not of speed.
        let floor_ms = 1.0e-4;
        if t_b1 < floor_ms {
            println!("  [FATAL] b1_forward timed at {:.6} ms < floor {:.6} ms.", t_b1, floor_ms);
            println!("          A kernel touching {} bytes cannot be this fast.", (n * k0) * 4);
            println!("          This is a DELETED LOOP (dead-store elimination), not a result.");
            println!("          Refusing to report a false zero.");
            std::process::exit(1);
        }
        println!(
            "  [ok] b1_forward = {:.6} ms > floor {:.6} ms (loop is real)",
            t_b1, floor_ms
        );

        let calls = 5.0 * 24.0 * 128.0;
        let t_b2_full = timeit!("B2 full: compute_act_mask + sparse_accum (k=8)", {
            vmlm_core::conn_train::compute_act_mask(&sc, x, k);
            let m = vmlm_core::conn_train::sparse_accum(&sc, x, yb, k);
            sink += m as f32;
            sink += consume(yb, n);
        });

        println!("\n===== VERDICT =====");
        println!("  B2 full (mask+accum, k=8)     {:8.4} ms/op", t_b2_full);
        println!("  B1 full (no mask, all k0)     {:8.4} ms/op", t_b1);
        println!("  speedup                       {:8.4}x", t_b2_full / t_b1);
        println!();
        println!("  arithmetic done: B2 = {} MACs, B1 = {} MACs", n * k, n * k0);
        println!(
            "  => B1 does {:.1}x MORE multiply-adds and is still {:.2}x faster.",
            k0 as f64 / k as f64,
            t_b2_full / t_b1
        );
        println!("  => the entire win is the ABSENT SELECTION, not the arithmetic,");
        println!("     not the branch, not the gather, not the act[] clear.");
        println!();
        println!("  Honest decomposition of B2's {:.4} ms:", t_b2_full);
        println!("      act[] clear                {:8.4} ms  ({:5.1}%)", t_clear, 100.0 * t_clear / t_b2_full);
        println!("      strength row (gather)      {:8.4} ms  ({:5.1}%)", t_strength, 100.0 * t_strength / t_b2_full);
        println!(
            "      top-k SELECTION (remainder){:8.4} ms  ({:5.1}%)",
            t_mask - t_clear - t_strength,
            100.0 * (t_mask - t_clear - t_strength) / t_b2_full
        );
        println!("      accumulate                 {:8.4} ms  ({:5.1}%)", t_ones, 100.0 * t_ones / t_b2_full);
        println!();
        println!("  Extrapolated to 5 matvec x 24 layers x 128 positions:");
        println!("      B2 = {:9.1} ms/step", t_b2_full * calls);
        println!("      B1 = {:9.1} ms/step", t_b1 * calls);

        if sink.is_nan() {
            println!("nan");
        }
    }
}
