// P12 诊断：bf16 GEMM 在**真实 MiniCPM shape** 上到底多慢，以及慢在哪。
//
// 起因：1 层 MiniCPM（S=4）在引擎里跑了 20 分钟没出结果，而按 FLOPs 估算
// 应该只要 ~10 ms。差 5 个数量级 —— 必须实测定位，不能靠推测。
//
// 这里对比三种内层循环次序（都用 bf16 权重原地加宽）：
//   N1 朴素：for i, for j, for k        （我第一版写的）
//   N2 i-k-j：for i, for k, for j       （B 行连续，A 广播复用）
//   N3 分块：  for jb, for kb, for i, for j（B 面板驻留寄存器）
//
// 用法: vmlm-bf16diag

use std::arch::x86_64::*;

#[link(name = "kernel32")]
extern "system" {
    fn QueryPerformanceCounter(out: *mut i64) -> i32;
    fn QueryPerformanceFrequency(out: *mut i64) -> i32;
}

#[inline]
fn qpc() -> i64 {
    unsafe { let mut v = 0i64; QueryPerformanceCounter(&mut v); v }
}
fn qpf() -> i64 {
    unsafe { let mut v = 0i64; QueryPerformanceFrequency(&mut v); v }
}
fn lcg(s: &mut u64) -> f32 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (((*s >> 33) as u32) as f32 / (1u32 << 31) as f32) - 1.0
}
fn rnd(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n).map(|_| lcg(&mut s) * scale).collect()
}
fn f32_to_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    let round = 0x7FFF + ((b >> 16) & 1);
    ((b.wrapping_add(round)) >> 16) as u16
}
fn bench<F: FnMut()>(reps: usize, mut f: F) -> f64 {
    for _ in 0..2 { f(); }
    let fr = qpf() as f64;
    let mut best = f64::MAX;
    for _ in 0..reps {
        let t0 = qpc(); f(); let t1 = qpc();
        let dt = (t1 - t0) as f64 * 1e6 / fr;
        if dt < best { best = dt; }
    }
    best
}

#[inline(always)]
unsafe fn widen16(p: *const u16) -> (__m256, __m256) {
    let packed = _mm256_loadu_si256(p as *const __m256i);
    let lo = _mm256_castsi256_si128(packed);
    let hi = _mm256_extracti128_si256(packed, 1);
    (
        _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(lo), 16)),
        _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(hi), 16)),
    )
}
#[inline(always)]
unsafe fn hsum(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let s4 = _mm_add_ps(hi, lo);
    let sh = _mm_movehl_ps(s4, s4);
    let s2 = _mm_add_ps(s4, sh);
    let s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
    _mm_cvtss_f32(s1)
}

/// N1 朴素：i -> j -> k  （每算一个 C[i][j] 都把 A 的第 i 行重读一遍）
#[target_feature(enable = "avx2,fma")]
unsafe fn n1(a: *const f32, b: *const u16, c: *mut f32, m: usize, n: usize, k: usize) {
    for i in 0..m {
        let ar = a.add(i * k);
        for j in 0..n {
            let br = b.add(j * k);
            let mut a0 = _mm256_setzero_ps();
            let mut a1 = _mm256_setzero_ps();
            let mut kk = 0;
            while kk + 16 <= k {
                let (w0, w1) = widen16(br.add(kk));
                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(ar.add(kk)), w0, a0);
                a1 = _mm256_fmadd_ps(_mm256_loadu_ps(ar.add(kk + 8)), w1, a1);
                kk += 16;
            }
            let mut s = hsum(a0) + hsum(a1);
            while kk < k { s += *ar.add(kk) * f32::from_bits((*br.add(kk) as u32) << 16); kk += 1; }
            *c.add(i * n + j) = s;
        }
    }
}

/// N3 分块：B 面板 4 行驻留，A 一次广播复用 —— 这是标准做法。
/// 块大小 4x16：4 个 B 行 x 16 个 K 元素。
#[target_feature(enable = "avx2,fma")]
unsafe fn n3(a: *const f32, b: *const u16, c: *mut f32, m: usize, n: usize, k: usize) {
    const NR: usize = 4;
    for i in 0..m {
        let ar = a.add(i * k);
        let cr = c.add(i * n);
        let mut j = 0usize;
        while j + NR <= n {
            // 4 个累加器对应 4 个 B 行
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut acc2 = _mm256_setzero_ps();
            let mut acc3 = _mm256_setzero_ps();
            let b0 = b.add(j * k);
            let b1 = b.add((j + 1) * k);
            let b2 = b.add((j + 2) * k);
            let b3 = b.add((j + 3) * k);
            let mut kk = 0usize;
            while kk + 16 <= k {
                // A 载入一次
                let av0 = _mm256_loadu_ps(ar.add(kk));
                let av1 = _mm256_loadu_ps(ar.add(kk + 8));
                // 4 个 B 行各自加宽
                let (x0, y0) = widen16(b0.add(kk));
                let (x1, y1) = widen16(b1.add(kk));
                let (x2, y2) = widen16(b2.add(kk));
                let (x3, y3) = widen16(b3.add(kk));
                acc0 = _mm256_fmadd_ps(av0, x0, acc0);
                acc0 = _mm256_fmadd_ps(av1, y0, acc0);
                acc1 = _mm256_fmadd_ps(av0, x1, acc1);
                acc1 = _mm256_fmadd_ps(av1, y1, acc1);
                acc2 = _mm256_fmadd_ps(av0, x2, acc2);
                acc2 = _mm256_fmadd_ps(av1, y2, acc2);
                acc3 = _mm256_fmadd_ps(av0, x3, acc3);
                acc3 = _mm256_fmadd_ps(av1, y3, acc3);
                kk += 16;
            }
            let mut s0 = hsum(acc0);
            let mut s1 = hsum(acc1);
            let mut s2 = hsum(acc2);
            let mut s3 = hsum(acc3);
            while kk < k {
                let av = *ar.add(kk);
                s0 += av * f32::from_bits((*b0.add(kk) as u32) << 16);
                s1 += av * f32::from_bits((*b1.add(kk) as u32) << 16);
                s2 += av * f32::from_bits((*b2.add(kk) as u32) << 16);
                s3 += av * f32::from_bits((*b3.add(kk) as u32) << 16);
                kk += 1;
            }
            *cr.add(j) = s0;
            *cr.add(j + 1) = s1;
            *cr.add(j + 2) = s2;
            *cr.add(j + 3) = s3;
            j += NR;
        }
        while j < n { *cr.add(j) = 0.0; j += 1; }
    }
}

fn main() {
    if !std::is_x86_feature_detected!("avx2") {
        println!("need avx2");
        return;
    }
    let reps = 20;

    // REAL MiniCPM layer shapes (S=4)
    let shapes: &[(&str, usize, usize, usize)] = &[
        ("q_proj   [4,2048]x[2048,1536]", 4, 2048, 1536),
        ("o_proj   [4,1536]x[1536,2048]", 4, 1536, 2048),
        ("gate     [4,4608]x[4608,1536]", 4, 4608, 1536),
        ("down     [4,1536]x[1536,4608]", 4, 1536, 4608),
        ("lm_head  [4,130560]x[130560,1536]", 4, 130560, 1536),
    ];

    println!("=== bf16 GEMM on REAL MiniCPM shapes ===");
    println!("{:<36} {:>12} {:>12} {:>10} {:>10}", "shape", "N1 naive ms", "N3 blk4 ms", "N1 GFLOPS", "N3 GFLOPS");
    for &(name, m, n, k) in shapes {
        let a = rnd(m * k, 1, 0.5);
        let w = rnd(n * k, 2, 1.0 / (k as f32).sqrt());
        let wb: Vec<u16> = w.iter().map(|&x| f32_to_bf16(x)).collect();
        let mut c1 = vec![0.0f32; m * n];
        let mut c3 = vec![0.0f32; m * n];
        unsafe {
            n1(a.as_ptr(), wb.as_ptr(), c1.as_mut_ptr(), m, n, k);
            n3(a.as_ptr(), wb.as_ptr(), c3.as_mut_ptr(), m, n, k);
        }
        let d = c1.iter().zip(c3.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        let t1 = bench(reps, || unsafe {
            n1(a.as_ptr(), wb.as_ptr(), c1.as_mut_ptr(), m, n, k);
            std::hint::black_box(&c1);
        });
        let t3 = bench(reps, || unsafe {
            n3(a.as_ptr(), wb.as_ptr(), c3.as_mut_ptr(), m, n, k);
            std::hint::black_box(&c3);
        });
        let fl = 2.0 * m as f64 * n as f64 * k as f64;
        println!("{:<36} {:>12.1} {:>12.1} {:>10.2} {:>10.2}   diff={:.1e}",
                 name, t1 / 1000.0, t3 / 1000.0, fl / (t1 * 1e-6) / 1e9, fl / (t3 * 1e-6) / 1e9, d);
    }
    println!();
    println!("If N1 is orders of magnitude slower, that explains the 20-minute run:");
    println!("the naive order re-reads the A row for every one of the n output columns.");
}
