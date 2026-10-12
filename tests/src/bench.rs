// VMLM 内核微基准（成员 D 专用；与 tests/src/main.rs 的回归测试互不影响）
//
// 目的：把「GEMM 内核本身的耗时」从「解释器 exec 循环的固定开销」里剥出来。
//   vmlm-runtime 跑 bench_gemm.vmb 测到的 us/iter 里，除了一次 512³ GEMM，
//   还含一次 VM 装载后的完整 exec 调度（约 500+ us，与 GEMM 无关）。
//   本 bench 直接调 vmlm_core::kernels::gemm_nt，只测内核。
//
// 用法:
//   vmlm-bench                 # 默认跑预设形状套件
//   vmlm-bench 512 512 512 200 # M N K reps
use vmlm_core::kernels;

#[inline]
fn qpc() -> i64 {
    // 直接读 QPC，避免 Instant 的额外抽象开销（只测内核，开销越小越好）。
    unsafe {
        let mut v = 0i64;
        QueryPerformanceCounter(&mut v);
        v
    }
}

#[link(name = "kernel32")]
extern "system" {
    fn QueryPerformanceCounter(out: *mut i64) -> i32;
    fn QueryPerformanceFrequency(out: *mut i64) -> i32;
}

fn qpf() -> i64 {
    unsafe {
        let mut v = 0i64;
        QueryPerformanceFrequency(&mut v);
        v
    }
}

fn lcg(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (((*state >> 33) as u32) as f32 / (1u32 << 31) as f32) - 1.0
}

/// 一次形状的测量：预热 + reps 次计时，返回 (中位数, min, max, p90) us。
/// min 是这台机器上最有信息量的统计量（噪声只会让耗时变长，不会变短）。
fn measure(m: usize, n: usize, k: usize, reps: usize) -> (f64, f64, f64, f64) {
    let mut s = 0x1234_5678_9abc_def0u64 ^ ((m as u64) << 32) ^ ((n as u64) << 16) ^ k as u64;
    let a: Vec<f32> = (0..m * k).map(|_| lcg(&mut s) * 0.1).collect();
    let b: Vec<f32> = (0..n * k).map(|_| lcg(&mut s) * 0.1).collect();
    let mut c: Vec<f32> = vec![0.0f32; m * n];

    unsafe {
        kernels::gemm_nt(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m, n, k, false);
        for _ in 0..5 {
            kernels::gemm_nt(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m, n, k, false);
        }
    }
    std::hint::black_box(&c);

    let f = qpf() as f64;
    let mut samples: Vec<f64> = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = qpc();
        unsafe {
            kernels::gemm_nt(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m, n, k, false);
        }
        let t1 = qpc();
        std::hint::black_box(&c);
        samples.push((t1 - t0) as f64 * 1e6 / f);
    }
    samples.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let med = samples[samples.len() / 2];
    let p90 = samples[samples.len() * 9 / 10];
    (med, samples[0], samples[reps - 1], p90)
}

/// 对比模式：同一份数据、同一个进程里交替测 A/B 两个内核。
/// 交替采样能抵消这台 i3-12100F 上的睿频/调度漂移（实测 max/min 能差 3 倍）。
///   A = kernels::gemm_nt_avx2          原 8x8 微内核（K 循环内做 8x8 转置）
///   B = kernels::gemm_nt_packed_bench  B 面板预打包 + MR=8（无转置）
///
/// 已试过但更慢的第三个变体（NR=16 宽 N，需 16 个累加器 → 溢出）保留在
/// tools/bench_ab_log.jsonl 的历史数据里，代码已移除。
fn compare(m: usize, n: usize, k: usize, reps: usize) {
    let mut s = 0x9e37_79b9_7f4a_7c15u64 ^ ((m as u64) << 32) ^ k as u64;
    let a: Vec<f32> = (0..m * k).map(|_| lcg(&mut s) * 0.1).collect();
    let b: Vec<f32> = (0..n * k).map(|_| lcg(&mut s) * 0.1).collect();
    let mut ca: Vec<f32> = vec![0.0f32; m * n];
    let mut cb: Vec<f32> = vec![0.0f32; m * n];

    unsafe {
        for _ in 0..5 {
            kernels::gemm_nt_avx2(a.as_ptr(), b.as_ptr(), ca.as_mut_ptr(), m, n, k, false);
            kernels::gemm_nt_packed_bench(a.as_ptr(), b.as_ptr(), cb.as_mut_ptr(), m, n, k, false);
        }
    }
    let mut mab = 0f32;
    for i in 0..m * n {
        let e = (ca[i] - cb[i]).abs();
        if e > mab {
            mab = e;
        }
    }
    println!("数值最大差: |A-B|={:.7}", mab);

    let f = qpf() as f64;
    let mut sa: Vec<f64> = Vec::with_capacity(reps);
    let mut sb: Vec<f64> = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = qpc();
        unsafe { kernels::gemm_nt_avx2(a.as_ptr(), b.as_ptr(), ca.as_mut_ptr(), m, n, k, false) };
        let t1 = qpc();
        unsafe { kernels::gemm_nt_packed_bench(a.as_ptr(), b.as_ptr(), cb.as_mut_ptr(), m, n, k, false) };
        let t2 = qpc();
        std::hint::black_box((&ca, &cb));
        sa.push((t1 - t0) as f64 * 1e6 / f);
        sb.push((t2 - t1) as f64 * 1e6 / f);
    }
    sa.sort_by(|x, y| x.partial_cmp(y).unwrap());
    sb.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let fl = 2.0 * (m * n * k) as f64;
    let (ma, mb) = (sa[0], sb[0]);
    println!(
        "A legacy 8x8  (转置)   min={:>8.1}us med={:>8.1}us  {:>6.2} GFLOPS",
        ma,
        sa[reps / 2],
        fl / (ma / 1e6) / 1e9
    );
    println!(
        "B packed MR=8 (无转置) min={:>8.1}us med={:>8.1}us  {:>6.2} GFLOPS   B/A={:.3}",
        mb,
        sb[reps / 2],
        fl / (mb / 1e6) / 1e9,
        ma / mb
    );
}

fn main() {
    unsafe {
        vmlm_core::interp::init_cpu();
    }
    let args: Vec<String> = std::env::args().collect();
    println!("AVX2 = {}", unsafe { vmlm_core::interp::HAS_AVX2 });

    if args.len() >= 2 && args[1] == "--compare" {
        let m: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(512);
        let n: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(512);
        let k: usize = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(512);
        let reps: usize = args.get(5).map(|s| s.parse().unwrap()).unwrap_or(300);
        compare(m, n, k, reps);
        return;
    }

    if args.len() >= 5 {
        let m: usize = args[1].parse().unwrap();
        let n: usize = args[2].parse().unwrap();
        let k: usize = args[3].parse().unwrap();
        let reps: usize = args[4].parse().unwrap();
        let (med, mn, mx, p90) = measure(m, n, k, reps);
        let gf = 2.0 * (m * n * k) as f64 / (mn / 1e6) / 1e9;
        println!(
            "GEMM_NT {:>5}x{:<5}x{:<5} reps={} median={:.1}us min={:.1}us p90={:.1}us max={:.1}us  {:.2} GFLOPS(min) {:.2} GFLOPS(median)",
            m, n, k, reps, med, mn, p90, mx, gf,
            2.0 * (m * n * k) as f64 / (med / 1e6) / 1e9
        );
        return;
    }

    println!("=== GEMM_NT 微基准（只测内核，不含 VM exec 开销）===");
    println!(
        "{:>6} {:>6} {:>6} {:>11} {:>11} {:>11} {:>11} {:>14}",
        "M", "N", "K", "median us", "min us", "p90 us", "max us", "GFLOPS(min)"
    );
    let suite: [(usize, usize, usize); 8] = [
        (512, 512, 512),
        (256, 256, 256),
        (1024, 1024, 1024),
        (128, 512, 512),
        (512, 128, 512),
        (512, 512, 128),
        (64, 512, 512),
        (512, 512, 64),
    ];
    for (m, n, k) in suite {
        let reps = if m * n * k > 300_000_000 { 40 } else { 200 };
        let (med, mn, mx, p90) = measure(m, n, k, reps);
        println!(
            "{:>6} {:>6} {:>6} {:>11.1} {:>11.1} {:>11.1} {:>11.1} {:>14.2}",
            m,
            n,
            k,
            med,
            mn,
            p90,
            mx,
            2.0 * (m * n * k) as f64 / (mn / 1e6) / 1e9
        );
    }
}
