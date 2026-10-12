// 崩溃回归测试（成员 D）：packed GEMM 的对齐/边界压力测试。
//
// 背景：gemm_nt_packed 曾经用 `raw.add((64 - misalign) / 4)` 做对齐抬升，
// 但 raw 是 *mut u8（字节单位）却又按 f32 除了 4，导致 misalign % 4 != 0 时
// 抬升不足 → _mm256_load_ps 未对齐 → 访问违例 0xC0000005。
// 256³ 崩、512³ 过，就是分配地址对齐情况不同造成的。
//
// 本测试用**多轮不同分配尺寸**反复触发堆分配，覆盖各种 misalign 取值，
// 并对每个 shape 比对 packed 与 legacy 的数值一致性。
use vmlm_core::kernels;

fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (((*state >> 33) as u32) as f32 / (1u32 << 31) as f32) - 1.0
}

fn one(m: usize, n: usize, k: usize, seed: u64) -> f32 {
    let mut s = seed;
    let a: Vec<f32> = (0..m * k).map(|_| lcg(&mut s) * 0.1).collect();
    let b: Vec<f32> = (0..n * k).map(|_| lcg(&mut s) * 0.1).collect();
    let mut ca: Vec<f32> = vec![0f32; m * n];
    let mut cb: Vec<f32> = vec![0f32; m * n];
    unsafe {
        kernels::gemm_nt_avx2(a.as_ptr(), b.as_ptr(), ca.as_mut_ptr(), m, n, k, false);
        kernels::gemm_nt_packed_bench(a.as_ptr(), b.as_ptr(), cb.as_mut_ptr(), m, n, k, false);
    }
    let mut mx = 0f32;
    for i in 0..m * n {
        let e = (ca[i] - cb[i]).abs();
        if e > mx {
            mx = e;
        }
    }
    mx
}

fn main() {
    unsafe {
        vmlm_core::interp::init_cpu();
    }
    let mut ok = true;

    println!("=== packed GEMM 崩溃回归：Lead 的最小复现 shape ===");
    for (m, n, k) in [(512usize, 512usize, 512usize), (256, 256, 256), (128, 512, 512)] {
        let mx = one(m, n, k, 7);
        let pass = mx < 1e-3;
        ok &= pass;
        println!(
            "  {:>5}x{:<5}x{:<5} maxdiff={:.7} {}",
            m,
            n,
            k,
            mx,
            if pass { "PASS" } else { "FAIL" }
        );
    }

    println!("\n=== 大 shape（含 1024³）===");
    for (m, n, k) in [(1024usize, 1024usize, 1024usize), (768, 512, 512), (512, 768, 512)] {
        let mx = one(m, n, k, 99);
        let pass = mx < 1e-3;
        ok &= pass;
        println!(
            "  {:>5}x{:<5}x{:<5} maxdiff={:.7} {}",
            m,
            n,
            k,
            mx,
            if pass { "PASS" } else { "FAIL" }
        );
    }

    // 对齐压力：这些尺寸的 nb*4 不是 64 的倍数，反复分配能撞出各种 misalign。
    // 每个 shape 跑 40 轮，中间穿插不同大小的分配以扰动堆地址。
    println!("\n=== 对齐压力（非 8 倍数 N/K，每个 shape 40 轮，穿插扰動分配）===");
    let shapes = [
        (16usize, 8usize, 8usize),
        (16, 9, 8),
        (16, 8, 9),
        (24, 17, 23),
        (33, 41, 27),
        (64, 64, 64),
        (100, 100, 100),
        (129, 127, 65),
        (256, 256, 256),
        (300, 137, 91),
        (511, 513, 129),
    ];
    for (m, n, k) in shapes {
        let mut worst = 0f32;
        for r in 0..40u64 {
            // 扰动堆：分配一个大小随轮次变化（且不是 64 倍数）的块
            let junk_n = 1 + ((r * 37) % 997) as usize;
            let junk: Vec<f32> = vec![0.5f32; junk_n];
            std::hint::black_box(&junk);
            let mx = one(m, n, k, r * 131 + 7);
            if mx > worst {
                worst = mx;
            }
        }
        let pass = worst < 1e-3;
        ok &= pass;
        println!(
            "  {:>5}x{:<5}x{:<5} worst_maxdiff={:.7} {}",
            m,
            n,
            k,
            worst,
            if pass { "PASS" } else { "FAIL" }
        );
    }

    // acc 累加模式也要覆盖（走不同的写回分支）
    println!("\n=== acc=true 路径 ===");
    for (m, n, k) in [(256usize, 256usize, 256usize), (300, 137, 91), (128, 512, 512)] {
        let mut s = 5u64;
        let a: Vec<f32> = (0..m * k).map(|_| lcg(&mut s) * 0.1).collect();
        let b: Vec<f32> = (0..n * k).map(|_| lcg(&mut s) * 0.1).collect();
        let base: Vec<f32> = (0..m * n).map(|_| lcg(&mut s)).collect();
        let mut ca = base.clone();
        let mut cb = base.clone();
        unsafe {
            kernels::gemm_nt_avx2(a.as_ptr(), b.as_ptr(), ca.as_mut_ptr(), m, n, k, true);
            kernels::gemm_nt_packed_bench(a.as_ptr(), b.as_ptr(), cb.as_mut_ptr(), m, n, k, true);
        }
        let mut mx = 0f32;
        for i in 0..m * n {
            let e = (ca[i] - cb[i]).abs();
            if e > mx {
                mx = e;
            }
        }
        // acc 模式下数值量级更大，用绝对误差 1e-2 作判据（K 累加 256~512 次）
        let pass = mx < 1e-2;
        ok &= pass;
        println!(
            "  {:>5}x{:<5}x{:<5} maxdiff={:.7} {}",
            m,
            n,
            k,
            mx,
            if pass { "PASS" } else { "FAIL" }
        );
    }


    // ---- 原 tests/src/main.rs 的 13 组形状回归（含 N=1/M=1 边界）----
    // main.rs 已被 Lead 临时替换成最小复现脚本，所以这里把原始回归套件
    // 搬进来，保证「内核回归仍全过」这件事有独立的、可复现的证据。
    println!("\n=== 原 13 组形状回归（NT 路径）===");
    fn naive(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut c = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0f32;
                for p in 0..k {
                    s += a[i * k + p] * b[j * k + p];
                }
                c[i * n + j] = s;
            }
        }
        c
    }
    for (m, n, k) in [
        (4usize, 8usize, 8usize),
        (1, 1, 64),
        (1, 32, 64),
        (1, 10, 32),
        (8, 8, 512),
        (512, 512, 512),
        (3, 5, 7),
        (5, 3, 9),
        (17, 23, 31),
        (64, 64, 64),
        (16, 1, 16),
        (1, 16, 16),
        (16, 16, 1),
    ] {
        let a: Vec<f32> = (0..m * k).map(|i| ((i * 37 % 100) as f32) / 50.0 - 1.0).collect();
        let b: Vec<f32> = (0..n * k).map(|i| ((i * 53 % 100) as f32) / 50.0 - 1.0).collect();
        let r = naive(&a, &b, m, n, k);
        let mut c = vec![0f32; m * n];
        unsafe {
            kernels::gemm_nt(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m, n, k, false);
        }
        let mut mx = 0f32;
        for i in 0..m * n {
            let e = (c[i] - r[i]).abs();
            if e > mx {
                mx = e;
            }
        }
        let pass = mx < 1e-3;
        ok &= pass;
        println!(
            "  GEMM {:>4}x{:<4}x{:<4} maxerr={:.7} {}",
            m,
            n,
            k,
            mx,
            if pass { "PASS" } else { "FAIL" }
        );
    }
    let v: Vec<f32> = vec![3.0, 1.0, 5.0, 2.0, 4.0, 0.5, 6.0, 1.5];
    let t = unsafe { kernels::topk_thresh(v.as_ptr(), v.len(), 3) };
    let tok = (t - 4.0f32).abs() < 1e-6;
    ok &= tok;
    println!(
        "  TOPK_THRESH k=3 = {} 期望 4.0  {}",
        t,
        if tok { "PASS" } else { "FAIL" }
    );

    println!("\n{}", if ok { "全部 PASS" } else { "存在 FAIL" });
    std::process::exit(if ok { 0 } else { 1 });
}
