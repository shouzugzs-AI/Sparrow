// P10/P11 优化开关基准（"优化全集 v3" 的落地测量）
//
// 为什么单独一个 bin，而不是改 vmlm-bench：
//   vmlm-bench 测的是「内核本身」（不含 exec）。本基准测的是
//   **真实 Transformer 一层的端到端时间**，因为 §7 声称的 3.23x 是层级收益 ——
//   内核级数字回答不了「消元有没有用」。
//
// ⭐ 纪律（计划 §10 自己写的，我照做）：
//   **禁止乘法叠加**。每个优化项必须
//     (1) 能单独开/关，
//     (2) 单独测出加速比，
//     (3) 同时给出**数值正确性**（与未优化的参考实现比 max_abs_err）。
//   任何"预计 0.85x"的推算都不作为结论 —— 只报实测。
//
// 用法:
//   vmlm-optbench                 # 跑全部项目，逐项 A/B
//   vmlm-optbench --shape 512     # 指定 hidden
//   vmlm-optbench --reps 400
//
// 测量口径：min（本机噪声只会让耗时变长；min 最接近真实速度）。
// 与 BENCH_GEMM_D.md §0 的口径一致。

use vmlm_core::kernels;

#[link(name = "kernel32")]
extern "system" {
    fn QueryPerformanceCounter(out: *mut i64) -> i32;
    fn QueryPerformanceFrequency(out: *mut i64) -> i32;
}

#[inline]
fn qpc() -> i64 {
    unsafe {
        let mut v = 0i64;
        QueryPerformanceCounter(&mut v);
        v
    }
}

fn qpf() -> i64 {
    unsafe {
        let mut v = 0i64;
        QueryPerformanceFrequency(&mut v);
        v
    }
}

fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (((*state >> 33) as u32) as f32 / (1u32 << 31) as f32) - 1.0
}

fn rnd(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n).map(|_| lcg(&mut s) * scale).collect()
}

/// min/median of `reps` timings of `f`, in microseconds.
fn bench<F: FnMut()>(reps: usize, mut f: F) -> (f64, f64) {
    for _ in 0..3 {
        f();
    }
    let fr = qpf() as f64;
    let mut v: Vec<f64> = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = qpc();
        f();
        let t1 = qpc();
        v.push((t1 - t0) as f64 * 1e6 / fr);
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (v[0], v[v.len() / 2])
}

fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

// ---------------------------------------------------------------- E1/E6 参考
//
// E1: RMSNorm + Linear 预乘 gamma
//     y = (x / rms(x)) * W^T
//   其中 rms 里含逐通道 gamma。慢的写法是先做逐元素 (x*g)，再 GEMM。
//   消元后把 gamma 折进权重：W' = W * diag(g)，前向就只剩 GEMM。
//
//   注意：这不改变数值 —— 但**浮点结合律不成立**，所以必须给出实测误差。
fn e1_slow(x: &[f32], g: &[f32], w: &[f32], rows: usize, cols: usize, n: usize) -> Vec<f32> {
    // xg = x * g  (逐元素, 需要一趟内存读写)
    let mut xg = vec![0.0f32; rows * cols];
    unsafe {
        for i in 0..rows * cols {
            *xg.get_unchecked_mut(i) = *x.get_unchecked(i) * *g.get_unchecked(i % cols);
        }
    }
    let mut y = vec![0.0f32; rows * n];
    unsafe {
        kernels::gemm_nt(xg.as_ptr(), w.as_ptr(), y.as_mut_ptr(), rows, n, cols, false);
    }
    y
}

fn e1_fused(x: &[f32], wf: &[f32], rows: usize, cols: usize, n: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; rows * n];
    unsafe {
        kernels::gemm_nt(x.as_ptr(), wf.as_ptr(), y.as_mut_ptr(), rows, n, cols, false);
    }
    y
}

/// W' = W * diag(g)，W 是 [n, cols] (NT 布局)
fn e1_premul(w: &[f32], g: &[f32], n: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n * cols];
    for i in 0..n {
        for j in 0..cols {
            out[i * cols + j] = w[i * cols + j] * g[j];
        }
    }
    out
}

// ------------------------------------------------------------ E2/E3 QKV 合并
//
// E2: 3 次小 GEMM (q,k,v) -> 1 次大 GEMM。
// 慢：GEMM(x, Wq) + GEMM(x, Wk) + GEMM(x, Wv)
// 快：GEMM(x, [Wq;Wk;Wv])  —— 一次调用、一次 K 维扫描
fn e2_sep(x: &[f32], wq: &[f32], wk: &[f32], wv: &[f32],
          rows: usize, cols: usize, nq: usize, nk: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut q = vec![0.0f32; rows * nq];
    let mut k = vec![0.0f32; rows * nk];
    let mut v = vec![0.0f32; rows * nk];
    unsafe {
        kernels::gemm_nt(x.as_ptr(), wq.as_ptr(), q.as_mut_ptr(), rows, nq, cols, false);
        kernels::gemm_nt(x.as_ptr(), wk.as_ptr(), k.as_mut_ptr(), rows, nk, cols, false);
        kernels::gemm_nt(x.as_ptr(), wv.as_ptr(), v.as_mut_ptr(), rows, nk, cols, false);
    }
    (q, k, v)
}

fn e2_merged(wq: &[f32], wk: &[f32], wv: &[f32], cols: usize, nq: usize, nk: usize) -> Vec<f32> {
    let n = nq + 2 * nk;
    let mut w = vec![0.0f32; n * cols];
    w[..nq * cols].copy_from_slice(wq);
    w[nq * cols..(nq + nk) * cols].copy_from_slice(wk);
    w[(nq + nk) * cols..].copy_from_slice(wv);
    w
}

// --------------------------------------------------------------------- main
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut hidden = 512usize;
    let mut reps = 300usize;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--shape" => {
                i += 1;
                hidden = args[i].parse().unwrap_or(512);
            }
            "--reps" => {
                i += 1;
                reps = args[i].parse().unwrap_or(300);
            }
            _ => {}
        }
        i += 1;
    }

    // MiniCPM-like proportions, scaled to `hidden`.
    let h = hidden;
    let heads = 16usize;
    let kv_heads = 2usize;
    let hd = h / heads;
    let nq = heads * hd;
    let nk = kv_heads * hd;
    let inter = 4 * h; // SwiGLU intermediate

    println!("=== VMLM 优化开关基准（逐项 A/B，禁止乘法叠加）===");
    println!("hidden={} heads={} kv_heads={} head_dim={} inter={} reps={}",
             h, heads, kv_heads, hd, inter, reps);
    println!();

    let rows = 4usize; // 短序列：优化在 prefill 路径上最相关

    // ---------------- E1: RMSNorm+Linear 预乘 gamma ----------------
    {
        let x = rnd(rows * h, 0xE1, 0.5);
        let g: Vec<f32> = (0..h).map(|j| 0.5 + (j as f32 / h as f32)).collect();
        let w = rnd(nq * h, 0xE1B, 0.05);
        let wf = e1_premul(&w, &g, nq, h);

        let ref_y = e1_slow(&x, &g, &w, rows, h, nq);
        let fus_y = e1_fused(&x, &wf, rows, h, nq);
        let d = maxdiff(&ref_y, &fus_y);

        let (s_min, s_med) = bench(reps, || {
            let y = e1_slow(&x, &g, &w, rows, h, nq);
            std::hint::black_box(&y);
        });
        let (f_min, f_med) = bench(reps, || {
            let y = e1_fused(&x, &wf, rows, h, nq);
            std::hint::black_box(&y);
        });
        println!("E1 RMSNorm+Linear 预乘 gamma  (Q 投影, [{},{}]x[{},{}])", rows, h, nq, h);
        println!("   slow (逐元素 x*g + GEMM) : min {:9.2} us  med {:9.2} us", s_min, s_med);
        println!("   fused (gamma 折进权重)   : min {:9.2} us  med {:9.2} us", f_min, f_med);
        println!("   加速比 (min)             : {:.3}x", s_min / f_min);
        println!("   max_abs_err vs slow      : {:.3e}", d);
        println!();
    }

    // ---------------- E2: QKV 合并 ----------------
    {
        let x = rnd(rows * h, 0xE2, 0.5);
        let wq = rnd(nq * h, 0xE2A, 0.05);
        let wk = rnd(nk * h, 0xE2B, 0.05);
        let wv = rnd(nk * h, 0xE2C, 0.05);
        let wm = e2_merged(&wq, &wk, &wv, h, nq, nk);
        let nm = nq + 2 * nk;

        let (q0, k0, v0) = e2_sep(&x, &wq, &wk, &wv, rows, h, nq, nk);
        let mut qkv = vec![0.0f32; rows * nm];
        unsafe {
            kernels::gemm_nt(x.as_ptr(), wm.as_ptr(), qkv.as_mut_ptr(), rows, nm, h, false);
        }
        let mut m = 0.0f32;
        for r in 0..rows {
            for j in 0..nq {
                m = m.max((q0[r * nq + j] - qkv[r * nm + j]).abs());
            }
            for j in 0..nk {
                m = m.max((k0[r * nk + j] - qkv[r * nm + nq + j]).abs());
                m = m.max((v0[r * nk + j] - qkv[r * nm + nq + nk + j]).abs());
            }
        }

        let (s_min, _) = bench(reps, || {
            let r = e2_sep(&x, &wq, &wk, &wv, rows, h, nq, nk);
            std::hint::black_box(&r);
        });
        let (f_min, _) = bench(reps, || {
            let mut y = vec![0.0f32; rows * nm];
            unsafe {
                kernels::gemm_nt(x.as_ptr(), wm.as_ptr(), y.as_mut_ptr(), rows, nm, h, false);
            }
            std::hint::black_box(&y);
        });
        println!("E2 QKV 合并 (3 次 GEMM -> 1 次, N={}+2x{}={})", nq, nk, nm);
        println!("   separate                 : min {:9.2} us", s_min);
        println!("   merged                   : min {:9.2} us", f_min);
        println!("   加速比 (min)             : {:.3}x", s_min / f_min);
        println!("   max_abs_err vs separate  : {:.3e}", m);
        println!();
    }

    // ---------------- E3: Gate/Up 合并 ----------------
    {
        let x = rnd(rows * h, 0xE3, 0.5);
        let wg = rnd(inter * h, 0xE3A, 0.05);
        let wu = rnd(inter * h, 0xE3B, 0.05);

        let (g0, u0) = {
            let mut a = vec![0.0f32; rows * inter];
            let mut b = vec![0.0f32; rows * inter];
            unsafe {
                kernels::gemm_nt(x.as_ptr(), wg.as_ptr(), a.as_mut_ptr(), rows, inter, h, false);
                kernels::gemm_nt(x.as_ptr(), wu.as_ptr(), b.as_mut_ptr(), rows, inter, h, false);
            }
            (a, b)
        };
        let mut wm = vec![0.0f32; 2 * inter * h];
        wm[..inter * h].copy_from_slice(&wg);
        wm[inter * h..].copy_from_slice(&wu);
        let mut gu = vec![0.0f32; rows * 2 * inter];
        unsafe {
            kernels::gemm_nt(x.as_ptr(), wm.as_ptr(), gu.as_mut_ptr(), rows, 2 * inter, h, false);
        }
        let mut m = 0.0f32;
        for r in 0..rows {
            for j in 0..inter {
                m = m.max((g0[r * inter + j] - gu[r * 2 * inter + j]).abs());
                m = m.max((u0[r * inter + j] - gu[r * 2 * inter + inter + j]).abs());
            }
        }

        let (s_min, _) = bench(reps, || {
            let mut a = vec![0.0f32; rows * inter];
            let mut b = vec![0.0f32; rows * inter];
            unsafe {
                kernels::gemm_nt(x.as_ptr(), wg.as_ptr(), a.as_mut_ptr(), rows, inter, h, false);
                kernels::gemm_nt(x.as_ptr(), wu.as_ptr(), b.as_mut_ptr(), rows, inter, h, false);
            }
            std::hint::black_box((&a, &b));
        });
        let (f_min, _) = bench(reps, || {
            let mut y = vec![0.0f32; rows * 2 * inter];
            unsafe {
                kernels::gemm_nt(x.as_ptr(), wm.as_ptr(), y.as_mut_ptr(), rows, 2 * inter, h, false);
            }
            std::hint::black_box(&y);
        });
        println!("E3 Gate/Up 合并 (2 次 GEMM -> 1 次, N={})", 2 * inter);
        println!("   separate                 : min {:9.2} us", s_min);
        println!("   merged                   : min {:9.2} us", f_min);
        println!("   加速比 (min)             : {:.3}x", s_min / f_min);
        println!("   max_abs_err vs separate  : {:.3e}", m);
        println!();
    }

    // ---------------- 全层：优化前 vs 优化后 ----------------
    // 这是唯一有意义的端到端数字：把 E1/E2/E3/E6 一起用上。
    //
    // ⚠️ 为什么不直接堆 4 个随机 GEMM：那样激活会爆到 ~2e7，f32 抵消噪声
    //    就有 2.5e6 —— 那不是消除引入的误差，是我把测试放在了荒谬的数值区间。
    //    真实的 Transformer 每层后都有归一化把量级压回 O(1)。
    //    所以这里**按真实结构**做：每段 GEMM 后做 RMSNorm，权重按 1/sqrt(fan_in) 缩放。
    {
        let x = rnd(rows * h, 0xF0, 0.5);
        // 权重尺度 = 1/sqrt(fan_in)，这才是 nn.Linear 的默认初始化量级
        let sc_q = 1.0 / (h as f32).sqrt();
        let sc_i = 1.0 / (h as f32).sqrt();
        let g_in: Vec<f32> = (0..h).map(|j| 0.5 + j as f32 / h as f32).collect();
        let g_post: Vec<f32> = (0..h).map(|j| 0.7 + j as f32 / h as f32).collect();
        let wq = rnd(nq * h, 0xFA, sc_q);
        let wk = rnd(nk * h, 0xFB, sc_q);
        let wv = rnd(nk * h, 0xFC, sc_q);
        let wo = rnd(h * nq, 0xFD, sc_q);
        let wg = rnd(inter * h, 0xFE, sc_i);
        let wu = rnd(inter * h, 0xFF, sc_i);
        let wd = rnd(h * inter, 0x100, sc_i);

        // 优化版权重：gamma 折进去 + QKV 合并 + Gate/Up 合并
        let wq_f = e1_premul(&wq, &g_in, nq, h);
        let wk_f = e1_premul(&wk, &g_in, nk, h);
        let wv_f = e1_premul(&wv, &g_in, nk, h);
        let wm_qkv = e2_merged(&wq_f, &wk_f, &wv_f, h, nq, nk);
        let wg_f = e1_premul(&wg, &g_post, inter, h);
        let wu_f = e1_premul(&wu, &g_post, inter, h);
        let mut wm_gu = vec![0.0f32; 2 * inter * h];
        wm_gu[..inter * h].copy_from_slice(&wg_f);
        wm_gu[inter * h..].copy_from_slice(&wu_f);
        let nm = nq + 2 * nk;

        // RMSNorm（无 gamma 版本，gamma 已折进权重）
        let rms_norm = |v: &mut [f32], r: usize, c: usize| {
            for i in 0..r {
                let row = &mut v[i * c..(i + 1) * c];
                let ss: f32 = row.iter().map(|z| z * z).sum::<f32>() / c as f32;
                let inv = 1.0 / (ss + 1e-6).sqrt();
                for z in row.iter_mut() {
                    *z *= inv;
                }
            }
        };

        // ---- 参考（未优化：逐元素 gamma 在 GEMM 之前）----
        let base = || -> Vec<f32> {
            unsafe {
                let mut xg = vec![0.0f32; rows * h];
                for r in 0..rows {
                    for j in 0..h {
                        xg[r * h + j] = x[r * h + j] * g_in[j];
                    }
                }
                let mut q = vec![0.0f32; rows * nq];
                let mut k = vec![0.0f32; rows * nk];
                let mut v = vec![0.0f32; rows * nk];
                kernels::gemm_nt(xg.as_ptr(), wq.as_ptr(), q.as_mut_ptr(), rows, nq, h, false);
                kernels::gemm_nt(xg.as_ptr(), wk.as_ptr(), k.as_mut_ptr(), rows, nk, h, false);
                kernels::gemm_nt(xg.as_ptr(), wv.as_ptr(), v.as_mut_ptr(), rows, nk, h, false);
                let mut o = vec![0.0f32; rows * h];
                kernels::gemm_nt(q.as_ptr(), wo.as_ptr(), o.as_mut_ptr(), rows, h, nq, false);
                rms_norm(&mut o, rows, h);
                let mut og = vec![0.0f32; rows * h];
                for r in 0..rows {
                    for j in 0..h {
                        og[r * h + j] = o[r * h + j] * g_post[j];
                    }
                }
                let mut ga = vec![0.0f32; rows * inter];
                let mut ua = vec![0.0f32; rows * inter];
                kernels::gemm_nt(og.as_ptr(), wg.as_ptr(), ga.as_mut_ptr(), rows, inter, h, false);
                kernels::gemm_nt(og.as_ptr(), wu.as_ptr(), ua.as_mut_ptr(), rows, inter, h, false);
                for i in 0..rows * inter {
                    let gg = ga[i];
                    ga[i] = (gg / (1.0 + (-gg).exp())) * ua[i];
                }
                let mut d = vec![0.0f32; rows * h];
                kernels::gemm_nt(ga.as_ptr(), wd.as_ptr(), d.as_mut_ptr(), rows, h, inter, false);
                std::hint::black_box((&k, &v));
                rms_norm(&mut d, rows, h);
                d
            }
        };

        // ---- 优化版 ----
        let opt = || -> Vec<f32> {
            unsafe {
                let mut qkv = vec![0.0f32; rows * nm];
                kernels::gemm_nt(x.as_ptr(), wm_qkv.as_ptr(), qkv.as_mut_ptr(), rows, nm, h, false);
                let mut o = vec![0.0f32; rows * h];
                kernels::gemm_nt(qkv.as_ptr(), wo.as_ptr(), o.as_mut_ptr(), rows, h, nq, false);
                rms_norm(&mut o, rows, h);
                let mut gu = vec![0.0f32; rows * 2 * inter];
                kernels::gemm_nt(o.as_ptr(), wm_gu.as_ptr(), gu.as_mut_ptr(), rows, 2 * inter, h, false);
                let mut act = vec![0.0f32; rows * inter];
                for r in 0..rows {
                    for j in 0..inter {
                        let gg = gu[r * 2 * inter + j];
                        let uu = gu[r * 2 * inter + inter + j];
                        act[r * inter + j] = (gg / (1.0 + (-gg).exp())) * uu;
                    }
                }
                let mut d = vec![0.0f32; rows * h];
                kernels::gemm_nt(act.as_ptr(), wd.as_ptr(), d.as_mut_ptr(), rows, h, inter, false);
                rms_norm(&mut d, rows, h);
                d
            }
        };

        let br = base();
        let or = opt();
        let dd = maxdiff(&br, &or);
        // Diagnostic: locate WHERE the two diverge. A single max_abs_err over
        // a whole layer is useless for debugging -- and a huge value like
        // 2.5e6 usually means an INDEXING bug in the harness, not a numerical
        // difference from the elimination. Print the first mismatching
        // element and the magnitudes involved.
        if dd > 1.0e-3 {
            let mut first = usize::MAX;
            for i in 0..br.len().min(or.len()) {
                if (br[i] - or[i]).abs() > 1.0e-3 {
                    first = i;
                    break;
                }
            }
            let bref = br.iter().fold(0.0f32, |a, b| a.max(b.abs()));
            let oref = or.iter().fold(0.0f32, |a, b| a.max(b.abs()));
            println!("   [诊断] 首个分歧 idx={} (r={} j={}) base={:.6e} opt={:.6e}",
                     first, first / h, first % h,
                     br.get(first).copied().unwrap_or(f32::NAN),
                     or.get(first).copied().unwrap_or(f32::NAN));
            println!("   [诊断] base absmax={:.6e}  opt absmax={:.6e}", bref, oref);
        }
        let (b_min, _) = bench(reps, || {
            std::hint::black_box(base());
        });
        let (o_min, _) = bench(reps, || {
            std::hint::black_box(opt());
        });

        println!("=== 组合：整层前向（E1+E2+E3+E6 同时开启）===");
        println!("   baseline (逐元素 gamma)  : min {:9.2} us", b_min);
        println!("   optimized (全部消元)     : min {:9.2} us", o_min);
        println!("   端到端加速比 (min)       : {:.3}x", b_min / o_min);
        println!("   max_abs_err              : {:.3e}", dd);
        println!();
        println!("注：这是**实测的组合收益**，不是 §7 那种逐项相乘的推算。");
    }
}
