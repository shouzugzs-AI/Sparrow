// P18-D: after the path-B conversion, WHERE DOES THE TIME GO?
//
// Why this probe exists: converting QKV/Wo to path B (native sparse, k0=20) was
// expected to make the projections ~77x cheaper. It DID remove the dense GEMMs,
// but the measured wall-clock went UP (215 -> ~225 s/step) instead of down.
// That is a contradiction until you count the CALLS rather than the MACs:
//
//   Before (path A): 1 sparse matvec per layer  (FFN only)
//   After  (path B): 5 sparse matvecs per layer (q, k, v, wo, ffn)
//
// and EVERY sparse_forward call runs compute_act_mask(), i.e. a full k-round
// top-k over k0 for all d neurons. So the top-k cost got multiplied by 5, and
// it is now the dominant term. This probe measures each piece separately so the
// next step is aimed by data rather than by the MAC table.
//
// USAGE: vmlm-p18d.exe [--d 1536] [--t 128] [--layers 24] [--k0 20] [--k 8] [--reps 5]
use vmlm_core::conn_train::*;

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
    let tt = arg_u32("--t", 128) as usize;
    let layers = arg_u32("--layers", 24) as usize;
    let k0 = arg_u32("--k0", 20) as usize;
    let k = arg_u32("--k", 8) as usize;
    let vocab = arg_u32("--vocab", 64) as usize;
    let reps = arg_u32("--reps", 5) as usize;

    println!("===== P18-D: post-path-B cost breakdown =====");
    println!("d={} T={} layers={} k0={} k={} vocab={} reps={}", d, tt, layers, k0, k, vocab, reps);

    let clk = Clock::new();
    unsafe {
        let e = d * k0;
        let mut rng = Rng::new(4242);

        // one SparseConnW, as the trainer builds them
        let base = malloc(SparseConnW::bytes(d as u32, k0 as u32));
        let sc = SparseConnW::from_base(base, d as u32, k0 as u32, 7);
        init_sparse_conn_scaled(&sc, 0.144);
        // give src ids a realistic spread
        for i in 0..e {
            *sc.src.add(i) = (rng.next_u32() % (d as u32)) as f32;
        }

        let x = alloc_f32(d);
        let y = alloc_f32(d);
        let dy = alloc_f32(d);
        let dw = alloc_f32(e);
        for i in 0..d {
            *x.add(i) = rng.uni() * 0.2 - 0.1;
            *dy.add(i) = rng.uni() * 0.2 - 0.1;
        }
        for i in 0..e {
            *dw.add(i) = 0.0;
        }

        // ---- 1) compute_act_mask alone (the top-k) ----
        compute_act_mask(&sc, x, k);
        let t0 = clk.now();
        for _ in 0..reps {
            compute_act_mask(&sc, x, k);
        }
        let t_mask = clk.ms(clk.now() - t0) / reps as f64;

        // ---- 2) sparse_forward (mask + accumulate) ----
        sparse_forward(&sc, x, y, k);
        let t1 = clk.now();
        for _ in 0..reps {
            sparse_forward(&sc, x, y, k);
        }
        let t_fwd = clk.ms(clk.now() - t1) / reps as f64;

        // ---- 3) row_topk alone (what compute_act_mask spends its time in) ----
        let mut hit = vec![0usize; k0];
        let t2 = clk.now();
        for _ in 0..reps {
            let mut acc = 0usize;
            for i in 0..d {
                acc += row_topk(
                    sc.w.add(i * k0),
                    sc.src.add(i * k0),
                    x,
                    k0,
                    k,
                    &mut hit,
                );
            }
            if acc == usize::MAX {
                println!("x");
            }
        }
        let t_topk = clk.ms(clk.now() - t2) / reps as f64;

        // ---- 4) sparse_gemm_bwd_w alone ----
        sparse_gemm_bwd_w(&sc, dy, x, dw);
        let t3 = clk.now();
        for _ in 0..reps {
            sparse_gemm_bwd_w(&sc, dy, x, dw);
        }
        let t_bwd = clk.ms(clk.now() - t3) / reps as f64;

        println!("\n--- per ONE sparse matvec on d={} k0={} k={} (one position) ---", d, k0, k);
        println!("  row_topk over all neurons      {:8.4} ms", t_topk);
        println!("  compute_act_mask (topk+clear)  {:8.4} ms", t_mask);
        println!("  sparse_forward (mask+accum)    {:8.4} ms", t_fwd);
        println!("  sparse_gemm_bwd_w              {:8.4} ms", t_bwd);

        // ---- extrapolate to a real step ----
        // per layer per position: 5 forward sparse matvecs (q,k,v,wo,ffn)
        // backward per layer per position: 5 dW accumulations + 4 input-grad scans
        let fwd_calls = 5 * layers * tt;
        let bwd_calls = 5 * layers * tt;
        let mask_calls = 5 * layers * tt; // each forward call re-runs the mask
        let step_fwd = t_fwd * fwd_calls as f64;
        let step_bwd = t_bwd * bwd_calls as f64;
        println!("\n--- extrapolated to {} layers x {} positions ---", layers, tt);
        println!("  forward  sparse matvec x{}  = {:9.1} ms/step", fwd_calls, step_fwd);
        println!("  of which the top-k mask x{} = {:9.1} ms/step  ({:.1}% of fwd)",
            mask_calls, t_mask * mask_calls as f64, 100.0 * t_mask / t_fwd);
        println!("  backward dW accum  x{}      = {:9.1} ms/step", bwd_calls, step_bwd);
        println!("  SPARSE TOTAL                = {:9.1} ms/step", step_fwd + step_bwd);

        // attention: dense score + AV, unavoidable (whitepaper 13.6)
        let q = alloc_f32(tt * d);
        let kk2 = alloc_f32(tt * d);
        let vv = alloc_f32(tt * d);
        let oo = alloc_f32(tt * d);
        let aa = alloc_f32(tt * tt);
        for i in 0..tt * d {
            *q.add(i) = rng.uni() * 0.1;
            *kk2.add(i) = rng.uni() * 0.1;
            *vv.add(i) = rng.uni() * 0.1;
        }
        let t_att = {
            let scale = 1.0f32 / (d as f32).sqrt();
            let body = || {
                for t in 0..tt {
                    let qr = q.add(t * d);
                    for s2 in 0..=t {
                        let kr = kk2.add(s2 * d);
                        let mut dot = 0.0f32;
                        for i in 0..d {
                            dot += *qr.add(i) * *kr.add(i);
                        }
                        *aa.add(t * tt + s2) = dot * scale;
                    }
                    for i in 0..d {
                        let mut acc = 0.0f32;
                        for s2 in 0..=t {
                            acc += *aa.add(t * tt + s2) * *vv.add(s2 * d + i);
                        }
                        *oo.add(t * d + i) = acc;
                    }
                }
            };
            body();
            let t4 = clk.now();
            for _ in 0..reps {
                body();
            }
            clk.ms(clk.now() - t4) / reps as f64
        };
        println!("\n--- attention (dense, NOT sparsifiable per whitepaper 13.6) ---");
        println!("  per layer                  {:8.3} ms", t_att);
        println!("  x{} layers                 {:9.1} ms/step", layers, t_att * layers as f64);

        let sparse_total = step_fwd + step_bwd;
        let att_total = t_att * layers as f64;
        println!("\n--- SUMMARY ---");
        println!("  sparse (5x per layer) : {:9.1} ms/step", sparse_total);
        println!("  attention (dense)     : {:9.1} ms/step", att_total);
        println!("  sum (fwd+bwd, no opt) : {:9.1} ms/step", sparse_total + att_total + att_total * 2.0);
        let _ = e;
    }
}
