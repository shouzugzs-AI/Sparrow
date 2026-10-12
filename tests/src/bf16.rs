// P12f16  GEMM f16 f32 //
// 
//   P11 
//   //     Q1: bf16 f32 //     Q2: //
// 
//   P11 vmlm-why GEMM **load  p2/p3 **//    FMA oad  4.5 /K vs FMA 2.67//   bf16 ****//   1 vpmovzxwd + 1 vpslld 8 p5//   p5 //
// bf16 
//   bf16 f32 *16 *bf16 -> f32 //   `(u32)h << 16` f32  ****//   f32 -> bf16 
//
// : vmlm-bf16 --shape 512

use std::arch::x86_64::*;
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
fn lcg(s: &mut u64) -> f32 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (((*s >> 33) as u32) as f32 / (1u32 << 31) as f32) - 1.0
}
fn rnd(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n).map(|_| lcg(&mut s) * scale).collect()
}
fn bench<F: FnMut()>(reps: usize, mut f: F) -> f64 {
    for _ in 0..3 {
        f();
    }
    let fr = qpf() as f64;
    let mut best = f64::MAX;
    for _ in 0..reps {
        let t0 = qpc();
        f();
        let t1 = qpc();
        let dt = (t1 - t0) as f64 * 1e6 / fr;
        if dt < best {
            best = dt;
        }
    }
    best
}
fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

/// f32 -> bf16 (round-to-nearest-even). This is the STORAGE direction.
fn f32_to_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    // add 0x7FFF + (bit16 of b) implements round-to-nearest-even
    let round = 0x7FFF + ((b >> 16) & 1);
    ((b.wrapping_add(round)) >> 16) as u16
}

/// bf16 weights + f32 activations GEMM; the widening happens in registers.
///
/// Inner loop: load 8 bf16 (16 bytes), zero-extend to 32-bit with
/// `_mm256_cvtepu16_epi32`, shift left 16 to get the f32 bit pattern, then
/// reinterpret as f32. **No floating-point conversion instruction is needed**
/// because bf16 IS the high half of f32.
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_nt_bf16(
    a: *const f32, w: *const u16, c: *mut f32, m: usize, n: usize, k: usize,
) {
    for i in 0..m {
        for j in 0..n {
            let ar = a.add(i * k);
            let wr = w.add(j * k);
            let mut acc = _mm256_setzero_ps();
            let mut kk = 0usize;
            while kk + 8 <= k {
                // 8 bf16 = 16 
                let packed = _mm_loadu_si128(wr.add(kk) as *const __m128i);
                //  8 u32
                let as32 = _mm256_cvtepu16_epi32(packed);
                //  16 =  f32 f16 f32  16 
                let asf32 = _mm256_slli_epi32(as32, 16);
                let wv = _mm256_castsi256_ps(asf32);
                let av = _mm256_loadu_ps(ar.add(kk));
                acc = _mm256_fmadd_ps(av, wv, acc);
                kk += 8;
            }
            let mut s = {
                let hi = _mm256_extractf128_ps(acc, 1);
                let lo = _mm256_castps256_ps128(acc);
                let sum4 = _mm_add_ps(hi, lo);
                let sh = _mm_movehl_ps(sum4, sum4);
                let s2 = _mm_add_ps(sum4, sh);
                let s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
                _mm_cvtss_f32(s1)
            };
            while kk < k {
                s += *ar.add(kk) * f32::from_bits((*wr.add(kk) as u32) << 16);
                kk += 1;
            }
            *c.add(i * n + j) = s;
        }
    }
}

/// bf16 **** 16 bf16/// load ///
///   bug7%
/// 
///     lo16 = packed & 0xFFFF        -> [bf16_0, bf16_2, bf16_4, ...]  (
///     hi16 = packed >> 16           -> [bf16_1, bf16_3, bf16_5, ...]  (
/// **lo16  kk..kk+7 ** 2 /// ///
///  `_mm256_unpacklo/hi_epi16` **16 */// 
///     unpacklo_epi16(lo16, hi16)  64 = bf16_0..bf16_3
///     unpackhi_epi16(lo16, hi16)  64 = bf16_4..bf16_7
/// 32 16  f32#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_nt_bf16_x16(
    a: *const f32, w: *const u16, c: *mut f32, m: usize, n: usize, k: usize,
) {
    for i in 0..m {
        for j in 0..n {
            let ar = a.add(i * k);
            let wr = w.add(j * k);
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut kk = 0usize;
            while kk + 16 <= k {
                // 16 bf16 = 32 bytes = one ymm load.
                let packed = _mm256_loadu_si256(wr.add(kk) as *const __m256i);
                // LAYOUT DETERMINED EMPIRICALLY (tests/src/shufprobe.rs),
                // not derived by hand:
                //   low  128 bits = b0..b7   (contiguous)
                //   high 128 bits = b8..b15  (contiguous)
                // so `cvtepu16_epi32` on each half is all that is needed --
                // NO unpack and NO permute.
                //
                // My first two versions tried to split halves with
                // `and 0xFFFF` + `srli 16`; that leaves the u16s split across
                // the two ends of each 32-bit word, giving a zero-holed layout
                // (`0 0 2 0 4 0 6 0 ...`) that no unpack can re-contiguate.
                // Errors 7.06e-2 / 5.07e-1 were both caught by the
                // correctness check -- which is exactly why it exists.
                let lo128 = _mm256_castsi256_si128(packed);
                let hi128 = _mm256_extracti128_si256(packed, 1);
                let l0 = _mm256_cvtepu16_epi32(lo128); // b0..b7
                let w0 = _mm256_castsi256_ps(_mm256_slli_epi32(l0, 16));
                let l1 = _mm256_cvtepu16_epi32(hi128); // b8..b15
                let w1 = _mm256_castsi256_ps(_mm256_slli_epi32(l1, 16));

                acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(ar.add(kk)), w0, acc0);
                acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(ar.add(kk + 8)), w1, acc1);
                kk += 16;
            }
            let hsum = |v: __m256| -> f32 {
                let hi = _mm256_extractf128_ps(v, 1);
                let lo = _mm256_castps256_ps128(v);
                let sum4 = _mm_add_ps(hi, lo);
                let sh = _mm_movehl_ps(sum4, sum4);
                let s2 = _mm_add_ps(sum4, sh);
                let s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
                _mm_cvtss_f32(s1)
            };
            let mut s = hsum(acc0) + hsum(acc1);
            while kk < k {
                s += *ar.add(kk) * f32::from_bits((*wr.add(kk) as u32) << 16);
                kk += 1;
            }
            *c.add(i * n + j) = s;
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut h = 512usize;
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--shape" {
            i += 1;
            h = args[i].parse().unwrap_or(512);
        }
        i += 1;
    }
    let reps = 100;
    let m = 4usize;
    let n = h;
    let k = h;

    if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
        println!("AVX2+FMA");
        return;
    }

    println!("=== bf16 weight GEMM (M={} N={} K={}) ===", m, n, k);
    println!("Proving bf16 is FASTER before touching the engine (P11 lesson:");
    println!("do not change code on expectation alone).");
    println!();

    let a = rnd(m * k, 0xAA, 0.5);
    let w = rnd(n * k, 0xBB, 1.0 / (k as f32).sqrt());

    // bf16 storage, mirroring what real safetensors files hold.
    let w_bf: Vec<u16> = w.iter().map(|&x| f32_to_bf16(x)).collect();
    // bf16  f32  f32  
    let w_ref: Vec<f32> = w_bf.iter().map(|&h| f32::from_bits((h as u32) << 16)).collect();

    let mut c_f32 = vec![0.0f32; m * n];
    let mut c_bf = vec![0.0f32; m * n];
    let mut c_bf16x = vec![0.0f32; m * n];

    unsafe {
        kernels::gemm_nt(a.as_ptr(), w_ref.as_ptr(), c_f32.as_mut_ptr(), m, n, k, false);
        gemm_nt_bf16(a.as_ptr(), w_bf.as_ptr(), c_bf.as_mut_ptr(), m, n, k);
        gemm_nt_bf16_x16(a.as_ptr(), w_bf.as_ptr(), c_bf16x.as_mut_ptr(), m, n, k);
    }

    let e_bf = maxdiff(&c_f32, &c_bf);
    let e_bfx = maxdiff(&c_f32, &c_bf16x);
    let refmax = c_f32.iter().fold(0.0f32, |x, y| x.max(y.abs()));

    let t_f32 = bench(reps, || unsafe {
        kernels::gemm_nt(a.as_ptr(), w_ref.as_ptr(), c_f32.as_mut_ptr(), m, n, k, false);
        std::hint::black_box(&c_f32);
    });
    let t_bf = bench(reps, || unsafe {
        gemm_nt_bf16(a.as_ptr(), w_bf.as_ptr(), c_bf.as_mut_ptr(), m, n, k);
        std::hint::black_box(&c_bf);
    });
    let t_bfx = bench(reps, || unsafe {
        gemm_nt_bf16_x16(a.as_ptr(), w_bf.as_ptr(), c_bf16x.as_mut_ptr(), m, n, k);
        std::hint::black_box(&c_bf16x);
    });

    let g = |t: f64| 2.0 * m as f64 * n as f64 * k as f64 / (t * 1e-6) / 1e9;
    println!("{:<30} {:>10} {:>12} {:>10}", "kernel", "min_us", "weight_bytes", "GFLOPS");
    println!("{:<30} {:>10.2} {:>12} {:>10.2}", "A. f32 (bf16 widened back)",
             t_f32, format!("{} B", n * k * 4), g(t_f32));
    println!("{:<30} {:>10.2} {:>12} {:>10.2}", "B. bf16 in-place (8 per load)",
             t_bf, format!("{} B", n * k * 2), g(t_bf));
    println!("{:<30} {:>10.2} {:>12} {:>10.2}", "C. bf16 in-place (16 per load)",
             t_bfx, format!("{} B", n * k * 2), g(t_bfx));
    println!();
    println!("speedup B/A : {:.3}x", t_f32 / t_bf);
    println!("speedup C/A : {:.3}x", t_f32 / t_bfx);
    println!();
    println!("correctness (bf16 vs bf16-widened-to-f32 -- must be identical):");
    println!("  B vs A max_abs_err : {:.3e}  rel {:.3e}", e_bf, e_bf / refmax);
    println!("  C vs A max_abs_err : {:.3e}  rel {:.3e}", e_bfx, e_bfx / refmax);
    println!();
    // The conversion bf16 -> f32 is EXACT (left-shift by 16, no rounding), so
    // any residual vs the f32 reference is purely the f32 summation order in
    // a different kernel structure -- it must be at float-rounding level
    // (~1e-6 relative here), not a conversion bug.
    let tol = 1e-5 * refmax;
    if e_bf <= tol && e_bfx <= tol {
        println!("=> conversion EXACT; residual is f32 summation order only.");
    } else {
        println!("=> WARNING: error above float-rounding level, check conversion.");
    }
    println!();
    println!("memory implication:");
    println!("  f32  weights = {} B private commit", n * k * 4);
    println!("  bf16 weights = {} B (or mmap file-backed, 0 private)", n * k * 2);
}
