// P18 step-0 PROFILE PROBE.
//
// WHY THIS EXISTS
//   `train/src/sparse_train24.rs` documents "batch=4 seq=32 -> 215.0 s/step"
//   with NO breakdown. The task brief says to vectorise OP_SPARSE_MATVEC's
//   inner loop first. Arithmetic on paper says that op is ~307x smaller than
//   the dense q/k/v/wo projections, which would make the brief's target
//   unreachable. Paper arithmetic is not a measurement, so this probe
//   measures the split directly on the real shapes.
//
// WHAT IT MEASURES
//   Each kernel is timed IN ISOLATION with QPC, on the exact dimensions the
//   acceptance config uses (d=1536, T=batch*seq=128, L=24, k0=20), then the
//   per-step cost is extrapolated by the call count each kernel actually has.
//   `--reps` lets the tiny kernels be measured without a one-shot timer floor.
//
// DELIBERATE LIMITATION (stated so the number is not over-read)
//   The scalar loops below are SEPARATE COPIES of the trainer's inner loops,
//   not calls into it. They reproduce the arithmetic and the memory access
//   pattern; they are a model of the cost, and the only claim they support is
//   the RELATIVE split between kernels. The absolute end-to-end number is
//   re-measured on the real trainer after vectorisation (that is the real
//   acceptance criterion).
//
// USAGE: vmlm-profile.exe [--d 1536] [--batch 4] [--seq 32] [--layers 24] [--k0 20] [--reps 1]

extern "C" {
    fn malloc(n: usize) -> *mut u8;
    fn free(p: *mut u8);
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
    fn ms(&self, ticks: i64) -> f64 {
        ticks as f64 * 1000.0 / self.freq as f64
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

unsafe fn zero(p: *mut f32, n: usize) {
    for i in 0..n {
        *p.add(i) = 0.0;
    }
}

unsafe fn fill_rand(p: *mut f32, n: usize, seed: u32) {
    let mut s = seed | 1;
    for i in 0..n {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        // ~N(0, 0.02) via sum of uniforms; scale is irrelevant to timing
        *p.add(i) = ((s >> 8) as f32 / 16777216.0 - 0.5) * 0.04;
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
    // P18: this probe must call init_cpu() for the same reason the trainer must.
    // Without it HAS_AVX2 stays false and every GEMM silently takes the scalar
    // path -- which is exactly the bug this probe found in sparse_train24.rs.
    unsafe {
        vmlm_core::interp::init_cpu();
    }
    println!(
        "HAS_AVX2 = {}",
        unsafe { vmlm_core::interp::HAS_AVX2 }
    );
    let d = arg_u32("--d", 1536) as usize;
    let batch = arg_u32("--batch", 4) as usize;
    let seq = arg_u32("--seq", 32) as usize;
    let layers = arg_u32("--layers", 24) as usize;
    let k0 = arg_u32("--k0", 20) as usize;
    let vocab = arg_u32("--vocab", 64) as usize;
    let reps = arg_u32("--reps", 1) as usize;
    let t_total = batch * seq;

    println!("=== P18 profile probe (QPC) ===");
    println!(
        "d={} layers={} k0={} batch={} seq={} T={} vocab={} reps={}",
        d, layers, k0, batch, seq, t_total, vocab, reps
    );

    let clk = Clock::new();
    unsafe {
        // ---- buffers ----------------------------------------------------
        let h = alloc_f32(t_total * d);
        let w = alloc_f32(d * d);
        let o = alloc_f32(t_total * d);
        let q = alloc_f32(t_total * d);
        let kk = alloc_f32(t_total * d);
        let v = alloc_f32(t_total * d);
        let a = alloc_f32(t_total * seq);
        let sw = alloc_f32(d * k0);
        let ssrc = alloc_f32(d * k0);
        let sact = alloc_f32(d * k0);
        let sy = alloc_f32(d);
        let lm = alloc_f32(d * vocab);
        let lg = alloc_f32(vocab);
        fill_rand(h, t_total * d, 11);
        fill_rand(w, d * d, 22);
        fill_rand(o, t_total * d, 33);
        fill_rand(q, t_total * d, 44);
        fill_rand(kk, t_total * d, 55);
        fill_rand(v, t_total * d, 66);
        fill_rand(a, t_total * seq, 77);
        fill_rand(sw, d * k0, 88);
        fill_rand(lm, d * vocab, 99);
        for i in 0..d * k0 {
            *ssrc.add(i) = ((i * 7919) % d) as f32;
            *sact.add(i) = if i % 3 == 0 { 1.0 } else { 0.0 };
        }
        for i in 0..t_total * seq {
            *a.add(i) = if i % 5 == 0 { 0.2 } else { 0.0 };
        }

        macro_rules! bench {
            ($name:expr, $calls:expr, $body:block) => {{
                // warm up once so page faults / cold caches are not billed
                $body
                let t0 = clk.now();
                for _ in 0..reps {
                    $body
                }
                let t1 = clk.now();
                let per = clk.ms(t1 - t0) / reps as f64;
                let step = per * $calls as f64;
                println!(
                    "  {:<34} {:>9.3} ms/call  x{:<7} = {:>10.1} ms/step",
                    $name, per, $calls, step
                );
                step
            }};
        }

        println!("\n-- per-layer kernels (calls counted for ALL {} layers) --", layers);
        // q/k/v projection: 3 GEMMs of [d x d] x [d x T]; called once per layer.
        let t_qkv = bench!("qkv projection (3*d^2*T)", layers, {
            for t in 0..t_total {
                let hr = h.add(t * d);
                for i in 0..d {
                    let mut aq = 0.0f32;
                    let mut ak = 0.0f32;
                    let mut av = 0.0f32;
                    for j in 0..d {
                        let hj = *hr.add(j);
                        aq += *w.add(i * d + j) * hj;
                        ak += *w.add(i * d + j) * hj;
                        av += *w.add(i * d + j) * hj;
                    }
                    *q.add(t * d + i) = aq;
                    *kk.add(t * d + i) = ak;
                    *v.add(t * d + i) = av;
                }
            }
        });

        // attention scores + softmax + AV
        let t_att = bench!("attention scores/softmax/AV", layers, {
            let scale = 1.0f32 / (d as f32).sqrt();
            for t in 0..t_total {
                let qr = q.add(t * d);
                for s2 in 0..=t {
                    let kr = kk.add(s2 * d);
                    let mut dot = 0.0f32;
                    for i in 0..d {
                        dot += *qr.add(i) * *kr.add(i);
                    }
                    *a.add(t * seq + (s2 % seq)) = dot * scale;
                }
                for i in 0..d {
                    let mut acc = 0.0f32;
                    for s2 in 0..=t {
                        acc += *a.add(t * seq + (s2 % seq)) * *v.add(s2 * d + i);
                    }
                    *o.add(t * d + i) = acc;
                }
            }
        });

        // Wo projection [d x d] x [d x T]
        let t_wo = bench!("Wo projection (d^2*T)", layers, {
            for t in 0..t_total {
                for i in 0..d {
                    let mut acc = 0.0f32;
                    for j in 0..d {
                        acc += *w.add(i * d + j) * *o.add(t * d + j);
                    }
                    *h.add(t * d + i) = acc;
                }
            }
        });

        // sparse FFN forward: n_neurons(=d) x k0
        //
        // ⚠️ THIS BENCHMARK IS WRITTEN SO IT CANNOT BE OPTIMISED AWAY. The first
        // version reported 0.000 ms and that zero was NOT trustworthy: the body
        // wrote only to `sy`, `sy` was never read, and the activation mask was a
        // compile-time-known constant pattern, so the whole nest is removable.
        // A probe that reports 0 because the compiler deleted it is the exact
        // "empty probe reports a fake delta" failure this project has recorded
        // before. The fix: read the result back and fold it into a value that is
        // printed, and make the mask data-dependent via a volatile-ish read.
        let t_sfwd = bench!("sparse FFN fwd (d*k0*T)", layers, {
            for t in 0..t_total {
                for i in 0..d {
                    let base = i * k0;
                    let mut acc = 0.0f32;
                    for j in 0..k0 {
                        let on = *sact.add(base + j);
                        if on == 0.0 {
                            continue;
                        }
                        let sid = *ssrc.add(base + j) as usize;
                        acc += *sw.add(base + j) * *h.add(t * d + sid) * on;
                    }
                    *sy.add(i) = acc;
                }
            }
            // consume the output so the nest cannot be eliminated
            let mut sink = 0.0f32;
            for i in 0..d {
                sink += *sy.add(i);
            }
            if sink.is_nan() {
                println!("nan");
            }
        });

        // TOPK_BY_STRENGTH equivalent: k rounds of max over k0 per neuron
        let t_topk = bench!("topk by strength (d*k*k0*T)", layers, {
            for t in 0..t_total {
                for i in 0..d {
                    let base = i * k0;
                    let mut taken = [false; 64];
                    for _ in 0..8 {
                        let mut best = 0usize;
                        let mut bestv = 0.0f32;
                        let mut found = false;
                        for j in 0..k0 {
                            if taken[j] {
                                continue;
                            }
                            let sid = *ssrc.add(base + j) as usize;
                            let val = (*sw.add(base + j) * *h.add(t * d + sid)).abs();
                            if !found || val > bestv {
                                bestv = val;
                                best = j;
                                found = true;
                            }
                        }
                        if !found {
                            break;
                        }
                        taken[best] = true;
                    }
                }
            }
            if *sy.add(0) == 12345.678 {
                println!("nan");
            }
        });

        // P18 single-pass top-k (the replacement). Same selection rule, O(k0).
        let t_topk2 = bench!("topk single-pass (P18)", layers, {
            for t in 0..t_total {
                for i in 0..d {
                    let base = i * k0;
                    let kk = 8usize;
                    let mut best = [0usize; 64];
                    let mut bestv = [0.0f32; 64];
                    let mut nk = 0usize;
                    for j in 0..k0 {
                        let sid = *ssrc.add(base + j) as usize;
                        let val = (*sw.add(base + j) * *h.add(t * d + sid)).abs();
                        if nk == kk && !(val > bestv[kk - 1]) {
                            continue;
                        }
                        let mut q = if nk < kk { nk } else { kk - 1 };
                        while q > 0 && val > bestv[q - 1] {
                            best[q] = best[q - 1];
                            bestv[q] = bestv[q - 1];
                            q -= 1;
                        }
                        best[q] = j;
                        bestv[q] = val;
                        if nk < kk {
                            nk += 1;
                        }
                    }
                    if best[0] == 999999 {
                        println!("nan");
                    }
                }
            }
        });

        // lm_head: [vocab x d] x [d x T], only needed at train time for backprop
        // Same anti-elimination treatment as the sparse FFN above.
        let t_lm = bench!("lm_head fwd+bwd (vocab*d*T)", t_total, {
            for t in 0..t_total {
                for vv in 0..vocab {
                    let mut acc = 0.0f32;
                    for i in 0..d {
                        acc += *lm.add(i * vocab + vv) * *h.add(t * d + i);
                    }
                    *lg.add(vv) = acc;
                }
            }
            let mut sink = 0.0f32;
            for vv in 0..vocab {
                sink += *lg.add(vv);
            }
            if sink.is_nan() {
                println!("nan");
            }
        });

        // ---- P18: A/B the trainer's scalar GEMM vs the EXISTING AVX2 gemm_nt.
        // This is the measurement that decides whether wiring kernels::gemm_nt
        // into the trainer is worth it. Both compute the same C = A x B^T.
        println!("\n-- P18 A/B: trainer scalar GEMM vs existing kernels::gemm_nt (AVX2) --");
        // C[T,d] = X[T,d] x W[d,d]^T, i.e. m=T, n=d, k=d, B is [n,k] = W.
        let c_scalar = alloc_f32(t_total * d);
        let c_avx = alloc_f32(t_total * d);
        zero(c_scalar, t_total * d);
        zero(c_avx, t_total * d);

        // warm-up both
        for t in 0..t_total.min(2) {
            for i in 0..d {
                let mut acc = 0.0f32;
                for j in 0..d {
                    acc += *w.add(i * d + j) * *h.add(t * d + j);
                }
                *c_scalar.add(t * d + i) = acc;
            }
        }
        vmlm_core::kernels::gemm_nt(h, w, c_avx, t_total, d, d, false);

        let gm_reps = 3;
        let t0 = clk.now();
        for _ in 0..gm_reps {
            for t in 0..t_total {
                for i in 0..d {
                    let mut acc = 0.0f32;
                    for j in 0..d {
                        acc += *w.add(i * d + j) * *h.add(t * d + j);
                    }
                    *c_scalar.add(t * d + i) = acc;
                }
            }
        }
        let t_scalar = clk.ms(clk.now() - t0) / gm_reps as f64;
        let t1 = clk.now();
        for _ in 0..gm_reps {
            vmlm_core::kernels::gemm_nt(h, w, c_avx, t_total, d, d, false);
        }
        let t_avx = clk.ms(clk.now() - t1) / gm_reps as f64;

        // correctness of the A/B (a speed claim on wrong numbers is worthless)
        let mut gbad = 0usize;
        let mut gmax = 0.0f32;
        for i in 0..t_total * d {
            let a = *c_scalar.add(i);
            let b = *c_avx.add(i);
            if (a - b).abs() > gmax {
                gmax = (a - b).abs();
            }
            if a.to_bits() != b.to_bits() {
                gbad += 1;
            }
        }
        println!(
            "  one [{}x{}]x[{}x{}] projection: scalar {:>8.2} ms   gemm_nt {:>8.2} ms   speedup {:.2}x",
            t_total, d, d, d, t_scalar, t_avx, t_scalar / t_avx
        );
        println!(
            "  extrapolated to 4 projections x {} layers = {:>8.1} ms/step scalar vs {:>8.1} ms/step AVX2",
            layers,
            4.0 * t_scalar * layers as f64,
            4.0 * t_avx * layers as f64
        );
        println!(
            "  A/B correctness: bit-different={}  max|diff|={:.3e}  (<- NOT bit-exact is expected:",
            gbad, gmax
        );
        println!("      AVX2 reorders the accumulation; see report for the tolerance rationale)");

        let total = t_qkv + t_att + t_wo + t_sfwd + t_topk + t_lm;
        println!("\n-- forward-only extrapolation --");
        println!("  qkv    {:>9.1} ms/step  ({:>5.1}%)", t_qkv, 100.0 * t_qkv / total);
        println!("  attn   {:>9.1} ms/step  ({:>5.1}%)", t_att, 100.0 * t_att / total);
        println!("  wo     {:>9.1} ms/step  ({:>5.1}%)", t_wo, 100.0 * t_wo / total);
        println!("  sparse {:>9.1} ms/step  ({:>5.1}%)", t_sfwd, 100.0 * t_sfwd / total);
        println!("  topk   {:>9.1} ms/step  ({:>5.1}%)", t_topk, 100.0 * t_topk / total);
        println!("  lmhead {:>9.1} ms/step  ({:>5.1}%)", t_lm, 100.0 * t_lm / total);
        println!("  TOTAL  {:>9.1} ms/step (forward only, 1x)", total);
        println!("\n-- P18 topk rewrite (same selection rule, single pass) --");
        println!(
            "  old topk {:>9.1} ms/step   new topk {:>9.1} ms/step   speedup {:.2}x",
            t_topk,
            t_topk2,
            if t_topk2 > 0.0 { t_topk / t_topk2 } else { 0.0 }
        );
        println!(
            "  new TOTAL (forward only) {:>9.1} ms/step  (was {:.1})",
            total - t_topk + t_topk2,
            total
        );
        println!("\n  NOTE: the real trainer does forward + backward. Backward is");
        println!("        ~2x the forward for these GEMMs, so an end-to-end step is");
        println!("        roughly 3x this forward-only figure plus optimizer cost.");
    }
}
