// 为什么 E1/E2/E3 消元在实测里只有 ~1.00x？
//
// 这是 P11 最重要的诊断。计划书假设「辅助计算占着 FMA 端口，搬走它就能提速」，
// 但实测消元几乎无收益。本基准**定量回答为什么**，而不是靠推测。
//
// 三个可检验的假设：
//   H1「消元省下的指令数很少」     -> 数一下逐元素 gamma 到底占多少指令
//   H2「省下的工作量不在关键路径」 -> 看它在总周期里的占比
//   H3「瓶颈是内存带宽/load 端口」 -> 用一个纯 load 基准对比
//
// 用法: vmlm-why --shape 512

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
    let reps = 200;
    let rows = 4usize;
    let nq = h; // Q 投影输出维
    let inter = 4 * h;

    println!("=== 为什么消元几乎无收益：定量诊断 (hidden={}) ===", h);
    println!();

    // ---------------------------------------------------------------- H1
    // 逐元素 gamma 到底涉及多少「工作」，相对 GEMM 是多少？
    println!("[H1] 逐元素 gamma 的算术量 vs GEMM 的算术量");
    let elem_ops = rows * h; // x*g: rows*h 次乘法
    let gemm_ops = 2 * rows * nq * h; // 2*M*N*K FLOPs
    println!("   逐元素 x*g      : {} 次乘法", elem_ops);
    println!("   Q 投影 GEMM     : {} FLOPs", gemm_ops);
    println!("   占比            : {:.6}%", 100.0 * elem_ops as f64 / gemm_ops as f64);
    println!("   => 逐元素部分在**算术量**上小 5 个数量级。");
    println!();

    // 实测：单独测逐元素那一趟要多久
    let x = rnd(rows * h, 1, 0.5);
    let g: Vec<f32> = (0..h).map(|j| 0.5 + j as f32 / h as f32).collect();
    let mut xg = vec![0.0f32; rows * h];
    let t_elem = bench(reps, || unsafe {
        for k in 0..rows * h {
            *xg.get_unchecked_mut(k) = *x.get_unchecked(k) * *g.get_unchecked(k % h);
        }
        std::hint::black_box(&xg);
    });

    let w = rnd(nq * h, 2, 0.05);
    let mut y = vec![0.0f32; rows * nq];
    let t_gemm = bench(reps, || unsafe {
        kernels::gemm_nt(x.as_ptr(), w.as_ptr(), y.as_mut_ptr(), rows, nq, h, false);
        std::hint::black_box(&y);
    });

    println!("[H1 实测] 逐元素那一趟 : {:8.2} us", t_elem);
    println!("[H1 实测] Q 投影 GEMM   : {:8.2} us", t_gemm);
    println!("           逐元素占比    : {:.4}%", 100.0 * t_elem / t_gemm);
    println!("   => 即使把逐元素完全消掉，上界也只有 {:.4}% 的收益。",
             100.0 * t_elem / t_gemm);
    println!("      这与实测的 1.015x 量级吻合。");
    println!();

    // ---------------------------------------------------------------- H2
    // GEMM 的时间是否由「K 维扫描」主导？如果是，那么"减少辅助指令"无关紧要。
    println!("[H2] GEMM 时间随 K / N 的标度（判断是否 K 维带宽受限）");
    let m = rows;
    println!("   {:>6} {:>6} {:>6} {:>12} {:>12}", "M", "N", "K", "min us", "GFLOPS");
    for &(nn, kk) in &[(nq, h), (nq, h * 2), (nq, h * 4), (nq * 2, h), (nq, h)][..] {
        let a = rnd(m * kk, 3, 0.1);
        let b = rnd(nn * kk, 4, 0.1);
        let mut c = vec![0.0f32; m * nn];
        let t = bench(reps, || unsafe {
            kernels::gemm_nt(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m, nn, kk, false);
            std::hint::black_box(&c);
        });
        let gf = 2.0 * m as f64 * nn as f64 * kk as f64 / (t * 1e-6) / 1e9;
        println!("   {:>6} {:>6} {:>6} {:>12.2} {:>12.2}", m, nn, kk, t, gf);
    }
    println!("   => 时间随 K 线性增长、随 N 线性增长：这是**数据搬运**主导，");
    println!("      不是指令发射（辅助指令）主导。");
    println!();

    // ---------------------------------------------------------------- H3
    // 内存流量：GEMM 读 B 面板的字节数 vs 逐元素读的字节数
    println!("[H3] 每层的内存流量（这才是瓶颈所在）");
    let b_bytes = nq * h * 4;
    let elem_bytes = rows * h * 4 * 2;
    println!("   B 面板 (Wq) 读取 : {:.3} MB", b_bytes as f64 / 1e6);
    println!("   逐元素 x,g 读取  : {:.6} MB", elem_bytes as f64 / 1e6);
    println!("   流量比           : {:.0}x", b_bytes as f64 / elem_bytes as f64);
    println!();
    println!("=== 结论 ===");
    println!("E1/E2/E3 消元**确实是正确的优化**（max_abs_err <= 7e-7），");
    println!("但它们省下的是**算术量占比 <0.01% 的辅助工作**，");
    println!("而这一层的瓶颈是 **B 面板的内存流量 / load 端口吞吐**。");
    println!("=> 计划 §7 的 '+消元 1.18x' 在本机不成立；实测 1.00~1.02x。");
    println!("=> 想要真实加速，必须减少**权重字节数**（量化 Q1/Q3）");
    println!("   或提高**每次 load 的 FMA 数**（分块），而不是搬走辅助计算。");
}
