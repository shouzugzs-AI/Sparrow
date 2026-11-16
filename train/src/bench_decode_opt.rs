//! decode 生产路径优化基准（第 2 轮）：把「融合 + bf16 + 批量」三件事量化。
//!
//! 第 1 轮（bench_decode.rs）确认了：Rust 融合 GEMV 下 keep=15% 的 FFN 比稠密快 14.5x。
//! 这一轮回答「还能再快多少、代价是什么」：
//!   1. **融合选择**：真实路径要先算 |x| 的 top-k（分组 top-k），把选择成本并进每层耗时；
//!   2. **bf16 权重**：权重字节减半（累加仍 fp32），看带宽是否真的变成瓶颈；
//!   3. **批量**：decode 也可能批量（8/32 并发请求），行 gather 不摊薄但选/循环能摊；
//!   4. **Quill 侧**：KV cache 下 1 次 vs 12 次注意力的 per-token 成本。
//!
//! 口径：单线程 fp32 累加；多轮取 min 与 median（机器有外部负载，min 最接近上限）。
//! 运行：cargo run --release --bin vmlm-bench-decode-opt

use std::time::Instant;

use vmlm_ops::kernels::gemm_nt;

const D: usize = 256;
const I: usize = 1024;
const L: usize = 12;
const HEADS: usize = 4;
const HD: usize = D / HEADS;

struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Rng {
        Rng(s.wrapping_mul(6364136223846793005).wrapping_add(1))
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn uni(&mut self) -> f32 {
        (self.next_u32() as f32) / (u32::MAX as f32)
    }
    fn fill(&mut self, v: &mut [f32]) {
        for x in v.iter_mut() {
            *x = (self.uni() * 2.0 - 1.0) * 0.05;
        }
    }
}

fn stats<F: FnMut()>(mut f: F, n: usize) -> (f64, f64) {
    f();
    let mut ts: Vec<f64> = Vec::with_capacity(n);
    for _ in 0..n {
        let t0 = Instant::now();
        f();
        ts.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    let min = ts.iter().cloned().fold(f64::INFINITY, f64::min);
    let mut s = ts.clone();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (min, s[s.len() / 2])
}

#[inline]
unsafe fn gemv_dense(x: *const f32, w: *const f32, y: *mut f32, nin: usize, out: usize) {
    for o in (0..out).step_by(8) {
        let mut a = [0.0f32; 8];
        for p in 0..nin {
            let xv = *x.add(p);
            let w = w.add(p * out + o);
            let mut j = 0;
            while j < 8 {
                a[j] += xv * *w.add(j);
                j += 1;
            }
        }
        let mut j = 0;
        while j < 8 {
            *y.add(o + j) = a[j];
            j += 1;
        }
    }
}

#[inline]
unsafe fn gemv_rows_f32(x: *const f32, w: *const f32, y: *mut f32, idx: *const u32, m: usize, out: usize) {
    for o in (0..out).step_by(8) {
        let mut a = [0.0f32; 8];
        for t in 0..m {
            let p = *idx.add(t) as usize;
            let xv = *x.add(p);
            let w = w.add(p * out + o);
            let mut j = 0;
            while j < 8 {
                a[j] += xv * *w.add(j);
                j += 1;
            }
        }
        let mut j = 0;
        while j < 8 {
            *y.add(o + j) = a[j];
            j += 1;
        }
    }
}

/// 批量稠密：Y[T,out] = X[T,nin] @ W[nin,out]，外层走权重、内层走 token（权重只读一遍）
#[inline]
unsafe fn gemm_dense_t(x: *const f32, w: *const f32, y: *mut f32, t: usize, nin: usize, out: usize) {
    for o in (0..out).step_by(8) {
        for i in 0..t {
            let mut a = [0.0f32; 8];
            for p in 0..nin {
                let xv = *x.add(i * nin + p);
                let wp = w.add(p * out + o);
                let mut j = 0;
                while j < 8 { a[j] += xv * *wp.add(j); j += 1; }
            }
            let mut j = 0;
            while j < 8 { *y.add(i * out + o + j) = a[j]; j += 1; }
        }
    }
}

/// 批量稀疏：每个 token 用自己的行号（行号表 t*m）
#[inline]
unsafe fn gemm_rows_t(x: *const f32, w: *const f32, y: *mut f32, idx: *const u32,
                      t: usize, nin: usize, out: usize, m: usize) {
    for i in 0..t {
        gemv_rows_f32(x.add(i * nin), w, y.add(i * out), idx.add(i * m), m, out);
    }
}

/// bf16 权重（高位 16bit），累计 fp32
#[inline]
unsafe fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

#[inline]
unsafe fn gemv_rows_bf16(x: *const f32, w: *const u16, y: *mut f32, idx: *const u32, m: usize, out: usize) {
    for o in (0..out).step_by(8) {
        let mut a = [0.0f32; 8];
        for t in 0..m {
            let p = *idx.add(t) as usize;
            let xv = *x.add(p);
            let w = w.add(p * out + o);
            let mut j = 0;
            while j < 8 {
                a[j] += xv * bf16_to_f32(*w.add(j));
                j += 1;
            }
        }
        let mut j = 0;
        while j < 8 {
            *y.add(o + j) = a[j];
            j += 1;
        }
    }
}

/// 分组 top-k：返回升序行号。全程零分配（scratch/rows 由调用方复用）。
fn group_topk(x: &[f32], keep: f64, group: usize, rows: &mut Vec<u32>, scratch: &mut Vec<(f32, u32)>) {
    let n = x.len();
    let ng = (n / group).max(1);
    let mg = ((keep * ng as f64).round() as usize).max(1).min(ng);
    scratch.clear();
    for g in 0..ng {
        let mut m = -1.0f32;
        let base = g * group;
        for j in 0..group {
            let v = x[base + j].abs();
            if v > m { m = v; }
        }
        scratch.push((m, g as u32));
    }
    if mg < ng {
        scratch.select_nth_unstable_by(mg - 1, |a, b| b.0.partial_cmp(&a.0).unwrap());
    }
    // 前缀里就是选中的 mg 个组；按组号升序，让行号连续（访存友好）
    scratch[..mg].sort_unstable_by_key(|p| p.1);
    rows.clear();
    for t in 0..mg {
        let g = scratch[t].1 as usize;
        for j in 0..group {
            rows.push((g * group + j) as u32);
        }
    }
}

// ---------- AVX2 decode GEMV 快路径（[in,out] 行主序，out 需 8 的倍数）----------
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemv_dense_avx2(x: *const f32, w: *const f32, y: *mut f32, nin: usize, out: usize) {
    use core::arch::x86_64::*;
    let mut o = 0usize;
    while o + 16 <= out {
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        for p in 0..nin {
            let xv = _mm256_broadcast_ss(&*x.add(p));
            let wp = w.add(p * out + o);
            a0 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(wp), a0);
            a1 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(wp.add(8)), a1);
        }
        _mm256_storeu_ps(y.add(o), a0);
        _mm256_storeu_ps(y.add(o + 8), a1);
        o += 16;
    }
    while o + 8 <= out {
        let mut a0 = _mm256_setzero_ps();
        for p in 0..nin {
            let xv = _mm256_broadcast_ss(&*x.add(p));
            a0 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w.add(p * out + o)), a0);
        }
        _mm256_storeu_ps(y.add(o), a0);
        o += 8;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemv_rows_avx2(x: *const f32, w: *const f32, y: *mut f32, idx: *const u32,
                         m: usize, out: usize) {
    use core::arch::x86_64::*;
    let mut o = 0usize;
    while o + 16 <= out {
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        for t in 0..m {
            let p = *idx.add(t) as usize;
            let xv = _mm256_broadcast_ss(&*x.add(p));
            let wp = w.add(p * out + o);
            a0 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(wp), a0);
            a1 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(wp.add(8)), a1);
        }
        _mm256_storeu_ps(y.add(o), a0);
        _mm256_storeu_ps(y.add(o + 8), a1);
        o += 16;
    }
    while o + 8 <= out {
        let mut a0 = _mm256_setzero_ps();
        for t in 0..m {
            let p = *idx.add(t) as usize;
            let xv = _mm256_broadcast_ss(&*x.add(p));
            a0 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w.add(p * out + o)), a0);
        }
        _mm256_storeu_ps(y.add(o), a0);
        o += 8;
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let keep: f64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(0.15);
    let group: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);
    let reps: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(50);

    let mut rng = Rng::new(4242);
    let mut wg = vec![0f32; D * I];
    let mut wu = vec![0f32; D * I];
    let mut wd = vec![0f32; I * D];
    rng.fill(&mut wg);
    rng.fill(&mut wu);
    rng.fill(&mut wd);
    // bf16 版本
    let wg_h: Vec<u16> = wg.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
    let wu_h: Vec<u16> = wu.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
    let wd_h: Vec<u16> = wd.iter().map(|v| (v.to_bits() >> 16) as u16).collect();

    let mut x = vec![0f32; D];
    rng.fill(&mut x);
    let mut h = vec![0f32; I];
    rng.fill(&mut h);
    let mut y = vec![0f32; I.max(D)];

    let mut rows_in: Vec<u32> = Vec::new();
    let mut rows_mid: Vec<u32> = Vec::new();
    let mut scr_in: Vec<(f32, u32)> = Vec::new();
    let mut scr_mid: Vec<(f32, u32)> = Vec::new();
    let mut rows_in2 = Vec::new();
    let mut rows_mid2 = Vec::new();

    // 稠密基线（fp32）
    let dense_bytes = (2 * D * I + I * D) * 4;
    let (dmin, dmed) = stats(
        || unsafe {
            for _ in 0..L {
                gemv_dense(x.as_ptr(), wg.as_ptr(), y.as_mut_ptr(), D, I);
                gemv_dense(x.as_ptr(), wu.as_ptr(), y.as_mut_ptr(), D, I);
                gemv_dense(h.as_ptr(), wd.as_ptr(), y.as_mut_ptr(), I, D);
            }
        },
        reps,
    );

    // 稀疏 fp32：预算好行号（不含选择成本）
    group_topk(&x, keep, group, &mut rows_in, &mut scr_in);
    group_topk(&h, keep, group, &mut rows_mid, &mut scr_mid);
    let sp_bytes = (2 * rows_in.len() * I + rows_mid.len() * D) * 4;
    let (smin, smed) = stats(
        || unsafe {
            for _ in 0..L {
                gemv_rows_f32(x.as_ptr(), wg.as_ptr(), y.as_mut_ptr(), rows_in.as_ptr(), rows_in.len(), I);
                gemv_rows_f32(x.as_ptr(), wu.as_ptr(), y.as_mut_ptr(), rows_in.as_ptr(), rows_in.len(), I);
                gemv_rows_f32(h.as_ptr(), wd.as_ptr(), y.as_mut_ptr(), rows_mid.as_ptr(), rows_mid.len(), D);
            }
        },
        reps,
    );

    // 稀疏 bf16
    let sp_bytes_h = sp_bytes / 2;
    let (bmin, bmed) = stats(
        || unsafe {
            for _ in 0..L {
                gemv_rows_bf16(x.as_ptr(), wg_h.as_ptr(), y.as_mut_ptr(), rows_in.as_ptr(), rows_in.len(), I);
                gemv_rows_bf16(x.as_ptr(), wu_h.as_ptr(), y.as_mut_ptr(), rows_in.as_ptr(), rows_in.len(), I);
                gemv_rows_bf16(h.as_ptr(), wd_h.as_ptr(), y.as_mut_ptr(), rows_mid.as_ptr(), rows_mid.len(), D);
            }
        },
        reps,
    );

    // 融合：每层都重新做一次分组 top-k（真实路径）
    let (fmin, fmed) = stats(
        || unsafe {
            for _ in 0..L {
                group_topk(&x, keep, group, &mut rows_in2, &mut scr_in);
                gemv_rows_f32(x.as_ptr(), wg.as_ptr(), y.as_mut_ptr(), rows_in2.as_ptr(), rows_in2.len(), I);
                gemv_rows_f32(x.as_ptr(), wu.as_ptr(), y.as_mut_ptr(), rows_in2.as_ptr(), rows_in2.len(), I);
                group_topk(&h, keep, group, &mut rows_mid2, &mut scr_mid);
                gemv_rows_f32(h.as_ptr(), wd.as_ptr(), y.as_mut_ptr(), rows_mid2.as_ptr(), rows_mid2.len(), D);
            }
        },
        reps,
    );

    println!("# decode FFN 优化（d={} I={} L={}，keep={:.2} group={}，单线程）", D, I, L, keep, group);
    println!("# dense 每层权重 {} KB；sparse {} KB（{:.2}x）；bf16 sparse {} KB（{:.2}x）",
        dense_bytes / 1024, sp_bytes / 1024,
        dense_bytes as f64 / sp_bytes as f64, sp_bytes_h / 1024,
        dense_bytes as f64 / sp_bytes_h as f64);
    println!("  dense fp32          min {:7.1} us/layer  median {:7.1} us/layer", dmin * 1000.0 / L as f64, dmed * 1000.0 / L as f64);
    println!("  sparse fp32         min {:7.1} us/layer  median {:7.1} us/layer  ({:.1}x)", smin * 1000.0 / L as f64, smed * 1000.0 / L as f64, dmin / smin);
    println!("  sparse bf16          min {:7.1} us/layer  median {:7.1} us/layer  ({:.1}x)", bmin * 1000.0 / L as f64, bmed * 1000.0 / L as f64, dmin / bmin);
    println!("  sparse fp32 + 选择   min {:7.1} us/layer  median {:7.1} us/layer  ({:.1}x)", fmin * 1000.0 / L as f64, fmed * 1000.0 / L as f64, dmin / fmin);
    println!("  → 选择成本 = {:.1} us/layer（占融合路径 {:.0}%）；bf16 再省 {:.0}%",
        (fmin - smin) * 1000.0 / L as f64,
        100.0 * (fmin - smin) / fmin,
        100.0 * (smin - bmin) / smin);

    // ---- AVX2 GEMV 快路径（decode 专用，绕开 gemm_nt 的小 M 兜底）----
    let (amin, amed) = stats(
        || unsafe {
            for _ in 0..L {
                gemv_dense_avx2(x.as_ptr(), wg.as_ptr(), y.as_mut_ptr(), D, I);
                gemv_dense_avx2(x.as_ptr(), wu.as_ptr(), y.as_mut_ptr(), D, I);
                gemv_dense_avx2(h.as_ptr(), wd.as_ptr(), y.as_mut_ptr(), I, D);
            }
        },
        reps,
    );
    let (asmin, asmed) = stats(
        || unsafe {
            for _ in 0..L {
                gemv_rows_avx2(x.as_ptr(), wg.as_ptr(), y.as_mut_ptr(), rows_in.as_ptr(), rows_in.len(), I);
                gemv_rows_avx2(x.as_ptr(), wu.as_ptr(), y.as_mut_ptr(), rows_in.as_ptr(), rows_in.len(), I);
                gemv_rows_avx2(h.as_ptr(), wd.as_ptr(), y.as_mut_ptr(), rows_mid.as_ptr(), rows_mid.len(), D);
            }
        },
        reps,
    );
    println!("
# AVX2 decode GEMV 快路径");
    println!("  dense AVX2          min {:7.1} us/layer  median {:7.1} us/layer  带宽 {:5.1} GB/s",
        amin * 1000.0 / L as f64, amed * 1000.0 / L as f64,
        dense_bytes as f64 * L as f64 / (amin / 1e3) / 1e9);
    println!("  sparse AVX2         min {:7.1} us/layer  median {:7.1} us/layer  带宽 {:5.1} GB/s  ({:.2}x)",
        asmin * 1000.0 / L as f64, asmed * 1000.0 / L as f64,
        sp_bytes as f64 * L as f64 / (asmin / 1e3) / 1e9, amin / asmin);

    // ---- 冷缓存（生产真实性）：12 层各用一份独立权重，总 36MB > L3 ----
    {
        let mut wg12 = vec![0f32; L * D * I];
        let mut wu12 = vec![0f32; L * D * I];
        let mut wd12 = vec![0f32; L * D * I];
        rng.fill(&mut wg12);
        rng.fill(&mut wu12);
        rng.fill(&mut wd12);
        let wg12h: Vec<u16> = wg12.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
        let wu12h: Vec<u16> = wu12.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
        let wd12h: Vec<u16> = wd12.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
        let (c_d, _) = stats(|| unsafe {
            for li in 0..L {
                gemv_dense_avx2(x.as_ptr(), wg12.as_ptr().add(li * D * I), y.as_mut_ptr(), D, I);
                gemv_dense_avx2(x.as_ptr(), wu12.as_ptr().add(li * D * I), y.as_mut_ptr(), D, I);
                gemv_dense_avx2(h.as_ptr(), wd12.as_ptr().add(li * D * I), y.as_mut_ptr(), I, D);
            }
        }, reps.min(40));
        let (c_s, _) = stats(|| unsafe {
            for li in 0..L {
                gemv_rows_avx2(x.as_ptr(), wg12.as_ptr().add(li * D * I), y.as_mut_ptr(), rows_in.as_ptr(), rows_in.len(), I);
                gemv_rows_avx2(x.as_ptr(), wu12.as_ptr().add(li * D * I), y.as_mut_ptr(), rows_in.as_ptr(), rows_in.len(), I);
                gemv_rows_avx2(h.as_ptr(), wd12.as_ptr().add(li * D * I), y.as_mut_ptr(), rows_mid.as_ptr(), rows_mid.len(), D);
            }
        }, reps.min(40));
        let (c_sb, _) = stats(|| unsafe {
            for li in 0..L {
                gemv_rows_bf16(x.as_ptr(), wg12h.as_ptr().add(li * D * I), y.as_mut_ptr(), rows_in.as_ptr(), rows_in.len(), I);
                gemv_rows_bf16(x.as_ptr(), wu12h.as_ptr().add(li * D * I), y.as_mut_ptr(), rows_in.as_ptr(), rows_in.len(), I);
                gemv_rows_bf16(h.as_ptr(), wd12h.as_ptr().add(li * D * I), y.as_mut_ptr(), rows_mid.as_ptr(), rows_mid.len(), D);
            }
        }, reps.min(40));
        println!("
# 冷缓存（12 层各一份权重，总 36MB > L3）—— 生产真实性口径");
        println!("  dense  AVX2 fp32   min {:7.1} us/layer  带宽 {:5.1} GB/s", c_d * 1000.0 / L as f64,
            dense_bytes as f64 * L as f64 / (c_d / 1e3) / 1e9);
        println!("  sparse AVX2 fp32   min {:7.1} us/layer  带宽 {:5.1} GB/s  ({:.2}x)", c_s * 1000.0 / L as f64,
            sp_bytes as f64 * L as f64 / (c_s / 1e3) / 1e9, c_d / c_s);
        println!("  sparse bf16        min {:7.1} us/layer  带宽 {:5.1} GB/s  ({:.2}x)", c_sb * 1000.0 / L as f64,
            (sp_bytes / 2) as f64 * L as f64 / (c_sb / 1e3) / 1e9, c_d / c_sb);
    }

    // ---- 批量扫描：稀疏只在「小批量」赢，找出拐点 ----
    println!("
# 批量扫描（每 token 每层耗时，含 3 个 GEMV，fp32）");
    for &t in &[1usize, 2, 4, 8, 16, 32, 64] {
        let mut xb = vec![0f32; t * D];
        rng.fill(&mut xb);
        let mut hb = vec![0f32; t * I];
        rng.fill(&mut hb);
        let mut yb = vec![0f32; t * I.max(D)];
        // 每个 token 自己的行号
        let mut ri = vec![0u32; t * rows_in.len()];
        let mut rm = vec![0u32; t * rows_mid.len()];
        for i in 0..t {
            let xs = &xb[i * D..(i + 1) * D];
            group_topk(xs, keep, group, &mut rows_in2, &mut scr_in);
            ri[i * rows_in2.len()..(i + 1) * rows_in2.len()].copy_from_slice(&rows_in2);
            let hs = &hb[i * I..(i + 1) * I];
            group_topk(hs, keep, group, &mut rows_mid2, &mut scr_mid);
            rm[i * rows_mid2.len()..(i + 1) * rows_mid2.len()].copy_from_slice(&rows_mid2);
        }
        let mi = rows_in2.len();
        let mm = rows_mid2.len();
        let (dmn, _) = stats(|| unsafe {
            for _ in 0..L {
                gemm_dense_t(xb.as_ptr(), wg.as_ptr(), yb.as_mut_ptr(), t, D, I);
                gemm_dense_t(xb.as_ptr(), wu.as_ptr(), yb.as_mut_ptr(), t, D, I);
                gemm_dense_t(hb.as_ptr(), wd.as_ptr(), yb.as_mut_ptr(), t, I, D);
            }
        }, reps.min(30));
        let (smn, _) = stats(|| unsafe {
            for _ in 0..L {
                gemm_rows_t(xb.as_ptr(), wg.as_ptr(), yb.as_mut_ptr(), ri.as_ptr(), t, D, I, mi);
                gemm_rows_t(xb.as_ptr(), wu.as_ptr(), yb.as_mut_ptr(), ri.as_ptr(), t, D, I, mi);
                gemm_rows_t(hb.as_ptr(), wd.as_ptr(), yb.as_mut_ptr(), rm.as_ptr(), t, I, D, mm);
            }
        }, reps.min(30));
        // 同一批量的「仓库自带 AVX2 gemm_nt」稠密路径（生产会用这个，而不是我的标量循环）
        let mut wg_nt = vec![0f32; I * D];
        let mut wu_nt = vec![0f32; I * D];
        let mut wd_nt = vec![0f32; D * I];
        for i in 0..I { for j in 0..D { wg_nt[i * D + j] = wg[j * I + i]; wu_nt[i * D + j] = wu[j * I + i]; } }
        for i in 0..D { for j in 0..I { wd_nt[i * I + j] = wd[j * D + i]; } }
        let mut yk = vec![0f32; t * I.max(D)];
        let (kmn, _) = stats(|| unsafe {
            for _ in 0..L {
                gemm_nt(xb.as_ptr(), wg_nt.as_ptr(), yk.as_mut_ptr(), t, I, D, false);
                gemm_nt(xb.as_ptr(), wu_nt.as_ptr(), yk.as_mut_ptr(), t, I, D, false);
                gemm_nt(hb.as_ptr(), wd_nt.as_ptr(), yk.as_mut_ptr(), t, D, I, false);
            }
        }, reps.min(30));
        println!("  T={:3}: 标量dense {:7.1}  AVX2dense {:7.1}  sparse {:7.1} us/token  |  sparse/AVX2 {:5.2}x",
            t, dmn * 1000.0 / L as f64 / t as f64, kmn * 1000.0 / L as f64 / t as f64,
            smn * 1000.0 / L as f64 / t as f64, kmn / smn);
    }

    // ---- Quill：KV cache 下的注意力（1 次 vs 12 次）----
    println!("\n# Quill decode 注意力（KV cache，heads={} hd={}）", HEADS, HD);
    for lc in [512usize, 2048, 8192] {
        let mut kk = vec![0f32; lc * D];
        let mut vv = vec![0f32; lc * D];
        rng.fill(&mut kk);
        rng.fill(&mut vv);
        let q = vec![0.01f32; D];
        let mut out = vec![0f32; D];
        // KV cache 量化副本（bf16 / int8 + 对称 scale）
        let kk_bf: Vec<u16> = kk.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
        let vv_bf: Vec<u16> = vv.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
        let (kmax, vmax) = (kk.iter().fold(0f32, |a, b| a.max(b.abs())),
                            vv.iter().fold(0f32, |a, b| a.max(b.abs())));
        let kscale = kmax / 127.0;
        let vscale = vmax / 127.0;
        let kk_i8: Vec<i8> = kk.iter().map(|v| (v / kscale).round().clamp(-127.0, 127.0) as i8).collect();
        let vv_i8: Vec<i8> = vv.iter().map(|v| (v / vscale).round().clamp(-127.0, 127.0) as i8).collect();
        let (bmin, _) = stats(|| unsafe {
            attn_decode_bf16(q.as_ptr(), kk_bf.as_ptr(), vv_bf.as_ptr(), lc, out.as_mut_ptr())
        }, reps.min(20));
        let (imin, _) = stats(|| unsafe {
            attn_decode_i8(q.as_ptr(), kk_i8.as_ptr(), vv_i8.as_ptr(), kscale, vscale, lc, out.as_mut_ptr())
        }, reps.min(20));
        let (tmin, _) = stats(|| attn_decode(&q, &kk, &vv, lc, &mut out), reps.min(20));
        let (fmin, _) = stats(|| unsafe { attn_decode_fast(q.as_ptr(), kk.as_ptr(), vv.as_ptr(), lc, out.as_mut_ptr()) }, reps.min(20));
        println!("  ctx={:5}: fp32 {:8.1} us | bf16 {:8.1} us ({:.2}x) | int8 {:8.1} us ({:.2}x) | 朴素 {:8.1} us",
            lc, fmin * 1000.0, bmin * 1000.0, fmin / bmin, imin * 1000.0, fmin / imin, tmin * 1000.0);
    }

    // ---- 状态向量（选择性 SSM）decode：把 KV cache 换成常数状态 ----
    // 训练侧见 tools/p36_attnonce.py --mix state --state-kind ssm：
    //   dt = softplus(W_d x)         A = softplus(logA)+1e-4
    //   a  = exp(-dt*A)              h = a⊙h + (1-a)⊙(W x)
    //   y  = W_o h
    // 推理只需要 h[d]（1KB），与 ctx 无关；每 token 三次 d×d GEMV + 一次 d 维逐元素。
    println!("\n# 状态向量 decode（选择性 SSM，无 KV cache）");
    let mut w_w = vec![0f32; D * D];
    let mut w_wd = vec![0f32; D * D];
    let mut w_o = vec![0f32; D * D];
    rng.fill(&mut w_w);
    rng.fill(&mut w_wd);
    rng.fill(&mut w_o);
    // logA 按对数间隔初始化（与训练侧 p_lo=-4, p_hi=2.5 一致），A = softplus(logA)+1e-4
    let a_arr: Vec<f32> = (0..D)
        .map(|j| {
            let lg = -4.0 + (2.5 - -4.0) * (j as f32) / ((D - 1) as f32);
            softplus(lg) + 1e-4
        })
        .collect();
    let x0 = vec![0.05f32; D];
    let mut h = vec![0f32; D];
    let mut out_s = vec![0f32; D];

    // 正确性：AVX2 内核 vs 标量参考
    {
        let mut href = vec![0f32; D];
        let mut oref = vec![0f32; D];
        unsafe { ssm_step(x0.as_ptr(), h.as_mut_ptr(), w_w.as_ptr(), w_wd.as_ptr(),
                          w_o.as_ptr(), a_arr.as_ptr(), out_s.as_mut_ptr()) };
        ssm_step_ref(&x0, &mut href, &w_w, &w_wd, &w_o, &a_arr, &mut oref);
        let mx = out_s.iter().zip(oref.iter()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let hmx = h.iter().zip(href.iter()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        println!("  正确性：h 最大差 {:.2e}，输出最大差 {:.2e}", hmx, mx);
        h.iter_mut().for_each(|v| *v = 0.0);
    }
    let (smin, _) = stats(
        || unsafe { ssm_step(x0.as_ptr(), h.as_mut_ptr(), w_w.as_ptr(), w_wd.as_ptr(),
                             w_o.as_ptr(), a_arr.as_ptr(), out_s.as_mut_ptr()) },
        reps,
    );
    println!("  状态混合器单次 = {:.1} us（常数，与 ctx 无关；常驻 {:.1} KB）",
        smin * 1000.0, (D * 4) as f64 / 1024.0);

    // 12 层稠密 MLP 的端到端基线（gated: 3 个 d×I 矩阵 + silu）
    let mut xn = vec![0.05f32; D];
    let mut mg = vec![0f32; I];
    let mut mu = vec![0f32; I];
    let mut yn = vec![0f32; D];
    let (mlp_min, _) = stats(|| unsafe {
        for _ in 0..L {
            gemv_dense_avx2(xn.as_ptr(), wg.as_ptr(), mg.as_mut_ptr(), D, I);
            gemv_dense_avx2(xn.as_ptr(), wu.as_ptr(), mu.as_mut_ptr(), D, I);
            silu_mul(mg.as_mut_ptr(), mu.as_ptr(), I);
            gemv_dense_avx2(mg.as_ptr(), wd.as_ptr(), yn.as_mut_ptr(), I, D);
            for j in 0..D { xn[j] += yn[j]; }
        }
    }, reps.min(20));
    println!("  12 层 MLP（稠密，端到端必经） = {:.3} ms/token", mlp_min);

    println!("\n  {:>6} | {:>11} | {:>11} | {:>10} | {:>9} | {:>9}", "ctx", "注意力fp32", "注意力int8", "状态", "端到端fp32", "端到端int8");
    for lc in [512usize, 2048, 8192] {
        let mut kk = vec![0f32; lc * D];
        let mut vv = vec![0f32; lc * D];
        rng.fill(&mut kk);
        rng.fill(&mut vv);
        let q = vec![0.01f32; D];
        let mut outa = vec![0f32; D];
        let (kmax, vmax) = (
            kk.iter().fold(0f32, |a, b| a.max(b.abs())),
            vv.iter().fold(0f32, |a, b| a.max(b.abs())),
        );
        let kscale = kmax / 127.0;
        let vscale = vmax / 127.0;
        let kk_i8: Vec<i8> = kk.iter().map(|v| (v / kscale).round().clamp(-127.0, 127.0) as i8).collect();
        let vv_i8: Vec<i8> = vv.iter().map(|v| (v / vscale).round().clamp(-127.0, 127.0) as i8).collect();
        let (fmin, _) = stats(|| unsafe {
            attn_decode_fast(q.as_ptr(), kk.as_ptr(), vv.as_ptr(), lc, outa.as_mut_ptr())
        }, reps.min(20));
        let (imin, _) = stats(|| unsafe {
            attn_decode_i8(q.as_ptr(), kk_i8.as_ptr(), vv_i8.as_ptr(), kscale, vscale, lc, outa.as_mut_ptr())
        }, reps.min(20));
        println!("  {:>6} | {:>9.1} us | {:>9.1} us | {:>8.1} us | {:>7.3} ms | {:>7.3} ms",
            lc, fmin * 1000.0, imin * 1000.0, smin * 1000.0,
            mlp_min + fmin, mlp_min + imin);
        println!("         KV 常驻：fp32 {:.1} MB / int8 {:.1} MB   状态：{:.1} KB",
            (2 * lc * D * 4) as f64 / 1048576.0, (2 * lc * D) as f64 / 1048576.0,
            (D * 4) as f64 / 1024.0);
    }
    println!("  端到端（状态）= {:.3} ms/token，与 ctx 无关", mlp_min + smin);
}

/// AVX2 向量化 exp（2^f 多项式 + 指数位拼接），相对误差 ~1e-6，够 softmax 用。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vexp_ps(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use core::arch::x86_64::*;
    let y = _mm256_mul_ps(x, _mm256_set1_ps(1.4426950408889634));
    let magic = _mm256_set1_ps(12582912.0);
    let t = _mm256_add_ps(y, magic);
    let n = _mm256_sub_ps(t, magic);
    let f = _mm256_sub_ps(y, n);
    let mut p = _mm256_fmadd_ps(_mm256_set1_ps(0.009618129107628477), f, _mm256_set1_ps(0.05550410866482158));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(0.2402265069591007));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(0.6931471805599453));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.0));
    let ni = _mm256_cvtps_epi32(n);
    let bits = _mm256_slli_epi32(ni, 23);
    let scale = _mm256_castsi256_ps(_mm256_add_epi32(bits, _mm256_set1_epi32(0x3f800000)));
    _mm256_mul_ps(p, scale)
}

/// AVX2 单查询注意力（flash-decode 风格）：分块 + 在线 softmax，不落地完整 scores。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn attn_decode_fast(q: *const f32, kk: *const f32, vv: *const f32, lc: usize, out: *mut f32) {
    use core::arch::x86_64::*;
    const BLK: usize = 128;
    let mut sc = [0f32; BLK];
    for hh in 0..HEADS {
        let qo = q.add(hh * HD);
        let mut qs_buf = [0f32; HD];
        {
            let inv = 1.0f32 / (HD as f32).sqrt();
            for j in 0..HD { qs_buf[j] = *qo.add(j) * inv; }
        }
        let qs = qs_buf.as_ptr();
        let mut o = [0f32; HD];
        let mut m = f32::NEG_INFINITY;
        let mut l = 0.0f32;
        let mut i = 0usize;
        while i < lc {
            let n = (lc - i).min(BLK);
                // 1) 分数：q 预先缩放 1/sqrt(HD)，每 key 4 个 FMA + 水平求和
            for t in 0..n {
                let kb = kk.add((i + t) * D + hh * HD);
                let mut a0 = _mm256_mul_ps(_mm256_loadu_ps(qs), _mm256_loadu_ps(kb));
                let mut a1 = _mm256_mul_ps(_mm256_loadu_ps(qs.add(8)), _mm256_loadu_ps(kb.add(8)));
                let mut a2 = _mm256_mul_ps(_mm256_loadu_ps(qs.add(16)), _mm256_loadu_ps(kb.add(16)));
                let mut a3 = _mm256_mul_ps(_mm256_loadu_ps(qs.add(24)), _mm256_loadu_ps(kb.add(24)));
                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(qs.add(32)), _mm256_loadu_ps(kb.add(32)), a0);
                a1 = _mm256_fmadd_ps(_mm256_loadu_ps(qs.add(40)), _mm256_loadu_ps(kb.add(40)), a1);
                a2 = _mm256_fmadd_ps(_mm256_loadu_ps(qs.add(48)), _mm256_loadu_ps(kb.add(48)), a2);
                a3 = _mm256_fmadd_ps(_mm256_loadu_ps(qs.add(56)), _mm256_loadu_ps(kb.add(56)), a3);
                a0 = _mm256_add_ps(a0, a1);
                a2 = _mm256_add_ps(a2, a3);
                a0 = _mm256_add_ps(a0, a2);
                let lo = _mm256_castps256_ps128(a0);
                let hi = _mm256_extractf128_ps(a0, 1);
                let s4 = _mm_add_ps(lo, hi);
                let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
                let s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
                sc[t] = _mm_cvtss_f32(s1);
            }
            // 2) 在线 softmax：块内最大值 + 重缩放
            let mut bm = f32::NEG_INFINITY;
            for t in 0..n { if sc[t] > bm { bm = sc[t]; } }
            let nm = if bm > m { bm } else { m };
            if nm > m {
                let alpha = (m - nm).exp();
                let av = _mm256_set1_ps(alpha);
                for j in (0..HD).step_by(8) {
                    let p = o.as_mut_ptr().add(j);
                    _mm256_storeu_ps(p, _mm256_mul_ps(_mm256_loadu_ps(p), av));
                }
                l *= alpha;
                m = nm;
            }
            // 3) 概率（向量化 exp）+ 加权累加 V
            {
                let mv = _mm256_set1_ps(m);
                let mut t8 = 0usize;
                while t8 + 8 <= n {
                    let p = sc.as_mut_ptr().add(t8);
                    let v = _mm256_loadu_ps(p);
                    _mm256_storeu_ps(p, vexp_ps(_mm256_sub_ps(v, mv)));
                    t8 += 8;
                }
                while t8 < n {
                    sc[t8] = (sc[t8] - m).exp();
                    t8 += 1;
                }
            }
            for t in 0..n {
                let e = sc[t];
                l += e;
                let vb = vv.add((i + t) * D + hh * HD);
                let ev = _mm256_set1_ps(e);
                for j in (0..HD).step_by(8) {
                    let op = o.as_mut_ptr().add(j);
                    let acc = _mm256_fmadd_ps(ev, _mm256_loadu_ps(vb.add(j)), _mm256_loadu_ps(op));
                    _mm256_storeu_ps(op, acc);
                }
            }
            i += n;
        }
        let inv = 1.0 / l;
        let iv = _mm256_set1_ps(inv);
        for j in (0..HD).step_by(8) {
            let p = o.as_ptr().add(j);
            _mm256_storeu_ps(out.add(hh * HD + j), _mm256_mul_ps(_mm256_loadu_ps(p), iv));
        }
    }
}


// ---------- KV cache 量化：bf16 / int8 的解量化读取 ----------
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn load_bf16x8(p: *const u16) -> std::arch::x86_64::__m256 {
    use core::arch::x86_64::*;
    let v = _mm_loadu_si128(p as *const __m128i);        // 8 x u16
    let i32v = _mm256_cvtepu16_epi32(v);
    let bits = _mm256_slli_epi32(i32v, 16);
    _mm256_castsi256_ps(bits)
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn load_i8x8(p: *const i8, scale: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use core::arch::x86_64::*;
    let v = _mm_loadl_epi64(p as *const __m128i);        // 8 x i8
    let i32v = _mm256_cvtepi8_epi32(v);
    _mm256_mul_ps(_mm256_cvtepi32_ps(i32v), scale)
}

/// bf16 KV 的注意力（K/V 各占一半字节）
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn attn_decode_bf16(q: *const f32, kk: *const u16, vv: *const u16, lc: usize, out: *mut f32) {
    use core::arch::x86_64::*;
    const BLK: usize = 128;
    let mut sc = [0f32; BLK];
    for hh in 0..HEADS {
        let qo = q.add(hh * HD);
        let mut qs_buf = [0f32; HD];
        {
            let inv = 1.0f32 / (HD as f32).sqrt();
            for j in 0..HD { qs_buf[j] = *qo.add(j) * inv; }
        }
        let qs = qs_buf.as_ptr();
        let mut o = [0f32; HD];
        let mut m = f32::NEG_INFINITY;
        let mut l = 0.0f32;
        let mut i = 0usize;
        while i < lc {
            let n = (lc - i).min(BLK);
            for t in 0..n {
                let kb = kk.add(((i + t) * D + hh * HD) as usize);
                let mut a0 = _mm256_mul_ps(_mm256_loadu_ps(qs), load_bf16x8(kb));
                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(qs.add(8)), load_bf16x8(kb.add(8)), a0);
                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(qs.add(16)), load_bf16x8(kb.add(16)), a0);
                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(qs.add(24)), load_bf16x8(kb.add(24)), a0);
                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(qs.add(32)), load_bf16x8(kb.add(32)), a0);
                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(qs.add(40)), load_bf16x8(kb.add(40)), a0);
                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(qs.add(48)), load_bf16x8(kb.add(48)), a0);
                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(qs.add(56)), load_bf16x8(kb.add(56)), a0);
                let lo = _mm256_castps256_ps128(a0);
                let hi = _mm256_extractf128_ps(a0, 1);
                let s4 = _mm_add_ps(lo, hi);
                let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
                let s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
                sc[t] = _mm_cvtss_f32(s1);
            }
            let mut bm = f32::NEG_INFINITY;
            for t in 0..n { if sc[t] > bm { bm = sc[t]; } }
            if bm > m {
                let alpha = (m - bm).exp();
                let av = _mm256_set1_ps(alpha);
                for j in (0..HD).step_by(8) {
                    let p = o.as_mut_ptr().add(j);
                    _mm256_storeu_ps(p, _mm256_mul_ps(_mm256_loadu_ps(p), av));
                }
                l *= alpha;
                m = bm;
            }
            {
                let mv = _mm256_set1_ps(m);
                let mut t8 = 0usize;
                while t8 + 8 <= n {
                    let p = sc.as_mut_ptr().add(t8);
                    _mm256_storeu_ps(p, vexp_ps(_mm256_sub_ps(_mm256_loadu_ps(p), mv)));
                    t8 += 8;
                }
                while t8 < n { sc[t8] = (sc[t8] - m).exp(); t8 += 1; }
            }
            for t in 0..n {
                let e = sc[t];
                l += e;
                let vb = vv.add(((i + t) * D + hh * HD) as usize);
                let ev = _mm256_set1_ps(e);
                for j in (0..HD).step_by(8) {
                    let op = o.as_mut_ptr().add(j);
                    let acc = _mm256_fmadd_ps(ev, load_bf16x8(vb.add(j)), _mm256_loadu_ps(op));
                    _mm256_storeu_ps(op, acc);
                }
            }
            i += n;
        }
        let inv = 1.0 / l;
        let iv = _mm256_set1_ps(inv);
        for j in (0..HD).step_by(8) {
            let p = o.as_ptr().add(j);
            _mm256_storeu_ps(out.add(hh * HD + j), _mm256_mul_ps(_mm256_loadu_ps(p), iv));
        }
    }
}

/// int8 KV 的注意力（K/V 各占 1/4 字节，带对称量化 scale）
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn attn_decode_i8(q: *const f32, kk: *const i8, vv: *const i8, ks: f32, vs_: f32, lc: usize, out: *mut f32) {
    use core::arch::x86_64::*;
    const BLK: usize = 128;
    let mut sc = [0f32; BLK];
    let kscale = _mm256_set1_ps(ks);
    let vscale = _mm256_set1_ps(vs_);
    for hh in 0..HEADS {
        let qo = q.add(hh * HD);
        let mut qs_buf = [0f32; HD];
        {
            let inv = 1.0f32 / (HD as f32).sqrt();
            for j in 0..HD { qs_buf[j] = *qo.add(j) * inv; }
        }
        let qs = qs_buf.as_ptr();
        let mut o = [0f32; HD];
        let mut m = f32::NEG_INFINITY;
        let mut l = 0.0f32;
        let mut i = 0usize;
        while i < lc {
            let n = (lc - i).min(BLK);
            for t in 0..n {
                let kb = kk.add(((i + t) * D + hh * HD) as usize);
                let mut a0 = _mm256_mul_ps(_mm256_loadu_ps(qs), load_i8x8(kb, kscale));
                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(qs.add(8)), load_i8x8(kb.add(8), kscale), a0);
                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(qs.add(16)), load_i8x8(kb.add(16), kscale), a0);
                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(qs.add(24)), load_i8x8(kb.add(24), kscale), a0);
                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(qs.add(32)), load_i8x8(kb.add(32), kscale), a0);
                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(qs.add(40)), load_i8x8(kb.add(40), kscale), a0);
                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(qs.add(48)), load_i8x8(kb.add(48), kscale), a0);
                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(qs.add(56)), load_i8x8(kb.add(56), kscale), a0);
                let lo = _mm256_castps256_ps128(a0);
                let hi = _mm256_extractf128_ps(a0, 1);
                let s4 = _mm_add_ps(lo, hi);
                let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
                let s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
                sc[t] = _mm_cvtss_f32(s1);
            }
            let mut bm = f32::NEG_INFINITY;
            for t in 0..n { if sc[t] > bm { bm = sc[t]; } }
            if bm > m {
                let alpha = (m - bm).exp();
                let av = _mm256_set1_ps(alpha);
                for j in (0..HD).step_by(8) {
                    let p = o.as_mut_ptr().add(j);
                    _mm256_storeu_ps(p, _mm256_mul_ps(_mm256_loadu_ps(p), av));
                }
                l *= alpha;
                m = bm;
            }
            {
                let mv = _mm256_set1_ps(m);
                let mut t8 = 0usize;
                while t8 + 8 <= n {
                    let p = sc.as_mut_ptr().add(t8);
                    _mm256_storeu_ps(p, vexp_ps(_mm256_sub_ps(_mm256_loadu_ps(p), mv)));
                    t8 += 8;
                }
                while t8 < n { sc[t8] = (sc[t8] - m).exp(); t8 += 1; }
            }
            for t in 0..n {
                let e = sc[t];
                l += e;
                let vb = vv.add(((i + t) * D + hh * HD) as usize);
                let ev = _mm256_set1_ps(e);
                for j in (0..HD).step_by(8) {
                    let op = o.as_mut_ptr().add(j);
                    let acc = _mm256_fmadd_ps(ev, load_i8x8(vb.add(j), vscale), _mm256_loadu_ps(op));
                    _mm256_storeu_ps(op, acc);
                }
            }
            i += n;
        }
        let inv = 1.0 / l;
        let iv = _mm256_set1_ps(inv);
        for j in (0..HD).step_by(8) {
            let p = o.as_ptr().add(j);
            _mm256_storeu_ps(out.add(hh * HD + j), _mm256_mul_ps(_mm256_loadu_ps(p), iv));
        }
    }
}

/// 单 token 的单层注意力：scores = q·K^T / sqrt(hd)，softmax，再乘 V（含 KV cache）
fn attn_decode(q: &[f32], kk: &[f32], vv: &[f32], lc: usize, out: &mut [f32]) {
    let mut scores = vec![0f32; lc * HEADS];
    for hh in 0..HEADS {
        let qo = hh * HD;
        for i in 0..lc {
            let mut s = 0.0f32;
            let kb = i * D + qo;
            for p in 0..HD {
                s += q[qo + p] * kk[kb + p];
            }
            scores[hh * lc + i] = s / (HD as f32).sqrt();
        }
        // softmax（这里不追求数值稳定性，实现在生产里会减 max）
        let mut mx = f32::NEG_INFINITY;
        for i in 0..lc {
            mx = mx.max(scores[hh * lc + i]);
        }
        let mut sum = 0.0f32;
        for i in 0..lc {
            let e = (scores[hh * lc + i] - mx).exp();
            scores[hh * lc + i] = e;
            sum += e;
        }
        let inv = 1.0 / sum;
        for p in 0..HD {
            let mut acc = 0.0f32;
            for i in 0..lc {
                acc += scores[hh * lc + i] * inv * vv[i * D + qo + p];
            }
            out[qo + p] = acc;
        }
    }
}

/// softplus(x) = max(x,0) + ln(1+exp(-|x|))，标量参考用
#[inline]
fn softplus(x: f32) -> f32 {
    let ax = x.abs();
    if ax > 30.0 { x.max(0.0) } else { x.max(0.0) + (-ax).exp().ln_1p() }
}

/// AVX2 ln（atanh 级数，m∈[1,2) 时绝对误差 ~2e-6，够 softplus 用）
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vlog_ps(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use core::arch::x86_64::*;
    let xi = _mm256_castps_si256(x);
    let e = _mm256_sub_epi32(_mm256_srli_epi32(xi, 23), _mm256_set1_epi32(127));
    let ef = _mm256_cvtepi32_ps(e);
    let m = _mm256_castsi256_ps(_mm256_or_si256(
        _mm256_and_si256(xi, _mm256_set1_epi32(0x007f_ffff)),
        _mm256_set1_epi32(0x3f80_0000),
    ));
    let y = _mm256_div_ps(
        _mm256_sub_ps(m, _mm256_set1_ps(1.0)),
        _mm256_add_ps(m, _mm256_set1_ps(1.0)),
    );
    let y2 = _mm256_mul_ps(y, y);
    let mut p = _mm256_set1_ps(1.0 / 7.0);
    p = _mm256_fmadd_ps(p, y2, _mm256_set1_ps(1.0 / 5.0));
    p = _mm256_fmadd_ps(p, y2, _mm256_set1_ps(1.0 / 3.0));
    p = _mm256_fmadd_ps(p, y2, _mm256_set1_ps(1.0));
    let lnm = _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(2.0), y), p);
    _mm256_fmadd_ps(ef, _mm256_set1_ps(0.693_147_18), lnm)
}

/// 状态向量 decode 一步（选择性 SSM）：常数内存、常数计算，与 ctx 无关。
/// 权重按 [in, out] 行主序（GEMV 友好），与训练侧 nn.Linear 是转置关系。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn ssm_step(x0: *const f32, h: *mut f32, w_w: *const f32, w_wd: *const f32,
                   w_o: *const f32, a_arr: *const f32, out: *mut f32) {
    use core::arch::x86_64::*;
    let mut dt = [0f32; D];
    let mut b = [0f32; D];
    gemv_dense_avx2(x0, w_wd, dt.as_mut_ptr(), D, D);
    gemv_dense_avx2(x0, w_w, b.as_mut_ptr(), D, D);
    let mut j = 0usize;
    while j + 8 <= D {
        let dv = _mm256_loadu_ps(dt.as_ptr().add(j));
        let av = _mm256_andnot_ps(_mm256_set1_ps(-0.0), dv);           // |dt|
        let ex = vexp_ps(_mm256_sub_ps(_mm256_setzero_ps(), av));      // e^{-|dt|}
        let ln = vlog_ps(_mm256_add_ps(_mm256_set1_ps(1.0), ex));      // ln(1+e^{-|dt|})
        let sp = _mm256_add_ps(_mm256_max_ps(dv, _mm256_setzero_ps()), ln);
        let aa = _mm256_loadu_ps(a_arr.add(j));
        let a = vexp_ps(_mm256_sub_ps(_mm256_setzero_ps(), _mm256_mul_ps(sp, aa)));
        let bb = _mm256_loadu_ps(b.as_ptr().add(j));
        let write = _mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(1.0), a), bb);
        let hv = _mm256_loadu_ps(h.add(j));
        _mm256_storeu_ps(h.add(j), _mm256_fmadd_ps(a, hv, write));
        j += 8;
    }
    gemv_dense_avx2(h, w_o, out, D, D);
}

/// 同上，纯标量参考实现（用于对拍）
fn ssm_step_ref(x0: &[f32], h: &mut [f32], w_w: &[f32], w_wd: &[f32], w_o: &[f32],
                a_arr: &[f32], out: &mut [f32]) {
    let mut dt = vec![0f32; D];
    let mut b = vec![0f32; D];
    for o in 0..D {
        let mut acc = 0f32;
        for p in 0..D { acc += x0[p] * w_wd[p * D + o]; }
        dt[o] = acc;
        let mut acc2 = 0f32;
        for p in 0..D { acc2 += x0[p] * w_w[p * D + o]; }
        b[o] = acc2;
    }
    for o in 0..D {
        let a = (-softplus(dt[o]) * a_arr[o]).exp();
        h[o] = a * h[o] + (1.0 - a) * b[o];
    }
    for o in 0..D {
        let mut acc = 0f32;
        for p in 0..D { acc += h[p] * w_o[p * D + o]; }
        out[o] = acc;
    }
}

/// silu(gate) * up，向量化（gate 原地覆盖）#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn silu_mul(gate: *mut f32, up: *const f32, n: usize) {
    use core::arch::x86_64::*;
    let one = _mm256_set1_ps(1.0);
    let mut j = 0usize;
    while j + 8 <= n {
        let g = _mm256_loadu_ps(gate.add(j));
        let e = vexp_ps(_mm256_sub_ps(_mm256_setzero_ps(), g));
        let sig = _mm256_div_ps(one, _mm256_add_ps(one, e));
        let u = _mm256_loadu_ps(up.add(j));
        _mm256_storeu_ps(gate.add(j), _mm256_mul_ps(_mm256_mul_ps(g, sig), u));
        j += 8;
    }
}
