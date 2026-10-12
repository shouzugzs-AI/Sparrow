// PATH: DENSE
//
// B 基准：B1 `b1_forward` 向量化能到多少 ms/step？
//
// Lead 的问题：B1 全 k0=20、含 20% 标量尾块，向量化后实测能到多少。
//
// ⚠️ 本探针要回答的**不是**「能多快」，而是「能多快**且仍然 bit-exact**」。
//    `core/src/b1_pure.rs` 头部把累加顺序写成了**契约**：
//      * 行内按 j 升序；
//      * **单个标量累加器 acc**，明确「不做 8 路 SIMD 归约（归约会改变分组，
//        进而改变位型）」；
//      * 首个命中直接赋值 `acc = term`（因为 0.0 + (-0.0) = +0.0，位型不同）。
//    B1 的验收判据 A1 是 `f32::to_bits()` 逐位比较。
//    ⇒ 所以「向量化 B1」有一个硬约束：**不能改累加分组**。
//    本探针因此测三个版本，并把 bit-diff 一起报出来：
//
//      S0  标量（现状，b1_pure.rs:74 的逐字复刻）        <- 基线
//      S1  向量化 gather 算乘积，但**标量顺序累加**        <- 保住 bit-exact
//      S2  向量化 gather + SIMD 归约（8 路）              <- 最快，但会分叉
//      S3  纯算术上界（无 gather，同 MAC 数）             <- 布局天花板
//
//    S1 是**唯一**既能加速又保住 A1 的路线，所以它的数字才是「B1 能到多少」。
//    S2 用来量化「如果允许放弃 bit-exact 能多拿多少」—— 这是一个决策输入，
//    不是实现建议。
//
// USAGE: vmlm-b1-vec.exe [--d 1536] [--k0 20] [--reps 400]
//        （默认 d=128 是为了让标量 S0 不至于跑太久；用 --d 1536 复现真实形状）

// PATH: DENSE —— 只用 DENSE 参考内核，不引用 ref_sparse_a，不 use vmlm_core::sparse。

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

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

static mut GUARD: f64 = 0.0;

// ---------------------------------------------------------------
// S0: 现状 —— b1_pure.rs:74 的逐字复刻（标量、单累加器、首项直接赋值）
// ---------------------------------------------------------------
unsafe fn s0_scalar(w: *const f32, src: *const f32, x: *const f32, y: *mut f32, n: usize, k0: usize) {
    for i in 0..n {
        let base = i * k0;
        let mut acc = 0.0f32;
        let mut first = true;
        let mut j = 0usize;
        while j < k0 {
            let sid = *src.add(base + j) as usize;
            let term = *w.add(base + j) * *x.add(sid);
            acc = if first { term } else { acc + term };
            first = false;
            j += 1;
        }
        *y.add(i) = acc;
    }
}

// ---------------------------------------------------------------
// S1: 向量化 gather 取乘积，但**标量顺序累加** —— 保住 bit-exact
//     关键：gather 出来的 8 个值写进一个栈数组，再按 j 升序标量加。
//     分组不变 ⇒ 位型不变（A1 可过）。
// ---------------------------------------------------------------
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn s1_vec_gather_scalar_accum(
    w: *const f32,
    src: *const f32,
    x: *const f32,
    y: *mut f32,
    n: usize,
    k0: usize,
) {
    let mut buf = [0.0f32; 64];
    for i in 0..n {
        let base = i * k0;
        // 先算全部 k0 个乘积（向量化），存进 buf
        let mut j = 0usize;
        while j + 8 <= k0 {
            let wv = _mm256_loadu_ps(w.add(base + j));
            let sid_f = _mm256_loadu_ps(src.add(base + j));
            let sid = _mm256_cvttps_epi32(sid_f);
            let xv = _mm256_i32gather_ps(x, sid, 4);
            _mm256_storeu_ps(buf.as_mut_ptr().add(j), _mm256_mul_ps(wv, xv));
            j += 8;
        }
        while j < k0 {
            let sid = *src.add(base + j) as usize;
            buf[j] = *w.add(base + j) * *x.add(sid);
            j += 1;
        }
        // 再按 j 升序、单累加器、首项直接赋值 —— 与 S0 完全同序
        let mut acc = 0.0f32;
        let mut first = true;
        let mut q = 0usize;
        while q < k0 {
            let term = buf[q];
            acc = if first { term } else { acc + term };
            first = false;
            q += 1;
        }
        *y.add(i) = acc;
    }
}

// ---------------------------------------------------------------
// S2: 向量化 gather + SIMD 8 路归约 —— 最快，但**位型会分叉**
// ---------------------------------------------------------------
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn s2_vec_reduce(w: *const f32, src: *const f32, x: *const f32, y: *mut f32, n: usize, k0: usize) {
    for i in 0..n {
        let base = i * k0;
        let mut vsum = _mm256_setzero_ps();
        let mut j = 0usize;
        while j + 8 <= k0 {
            let wv = _mm256_loadu_ps(w.add(base + j));
            let sid_f = _mm256_loadu_ps(src.add(base + j));
            let sid = _mm256_cvttps_epi32(sid_f);
            let xv = _mm256_i32gather_ps(x, sid, 4);
            vsum = _mm256_add_ps(vsum, _mm256_mul_ps(wv, xv));
            j += 8;
        }
        // 水平归约
        let hi = _mm256_extractf128_ps(vsum, 1);
        let lo = _mm256_castps256_ps128(vsum);
        let s4 = _mm_add_ps(hi, lo);
        let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
        let s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
        let mut acc = _mm_cvtss_f32(s1);
        // 标量尾块
        while j < k0 {
            let sid = *src.add(base + j) as usize;
            acc += *w.add(base + j) * *x.add(sid);
            j += 1;
        }
        *y.add(i) = acc;
    }
}

// ---------------------------------------------------------------
// S3: 天花板 —— 无 gather，其余同 S0 的算术量
//
// ⚠️ 第一版这个函数报了 **0.00000 ms**，而且我的 consume 与 [FATAL] 守卫
//    都没抓到它 —— 守卫只**打印**警告，那个 0 仍然流进了外推表，打出
//    「infx vs S0」。根因不是 consume 缺失，而是：
//      s3 读 w/xg、写 y3，三者在 reps 循环里**全部不变量**，
//      而 y3 只在计时循环**之后**才被读 —— 于是 LLVM 把整个调用提到循环外，
//      循环体变成空。
//    对比：S0/S1/S2 之所以没被提走，是因为它们经 **gather** 读内存，
//    LLVM 无法证明可安全提升。**也就是说 S0 的正确性是运气，不是设计。**
//    修法：让每一次迭代的输出成为下一次迭代的输入（跨迭代真依赖），
//    这样循环体不可能被提升。同时 y3 在循环内被使用。
// ---------------------------------------------------------------
unsafe fn s3_no_gather(w: *const f32, xg: *const f32, y: *mut f32, n: usize, k0: usize) {
    for i in 0..n {
        let base = i * k0;
        // 依赖上一轮的 y[i]：制造跨迭代依赖，同时语义仍是「同 MAC 数的累加」
        let mut acc = *y.add(i);
        let mut first = true;
        let mut j = 0usize;
        while j < k0 {
            let term = *w.add(base + j) * *xg.add(base + j);
            acc = if first { term } else { acc + term };
            first = false;
            j += 1;
        }
        *y.add(i) = acc;
    }
}

fn main() {
    unsafe { vmlm_core::interp::init_cpu() };
    let d = arg_u32("--d", 128) as usize;
    let k0 = arg_u32("--k0", 20) as usize;
    let reps = arg_u32("--reps", 400) as usize;
    let has_avx2 = unsafe { vmlm_core::interp::HAS_AVX2 };

    println!("=== B1 b1_forward 向量化：能到多少 ms/step（含 bit-exact 约束）===");
    println!("d={} k0={} reps={} HAS_AVX2={}", d, k0, reps, has_avx2);
    println!("⚠️ S1 才是有意义的版本（向量 gather + 标量顺序累加，保住 A1 逐位判据）");
    println!("   S2 会分叉 A1，仅用于量化「放弃 bit-exact 能多拿多少」\n");

    let clk = Clock::new();
    let e = d * k0;
    unsafe {
        let w = malloc(e * 4) as *mut f32;
        let src = malloc(e * 4) as *mut f32;
        let x = malloc(d * 4) as *mut f32;
        let xg = malloc(e * 4) as *mut f32;
        let y0 = malloc(d * 4) as *mut f32;
        let y1 = malloc(d * 4) as *mut f32;
        let y2 = malloc(d * 4) as *mut f32;
        let y3 = malloc(d * 4) as *mut f32;
        let mut s: u32 = 12345;
        for i in 0..e {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            // 用含 -0.0 与符号变化的权重，让「累加顺序」的位型差异真的能出现
            *w.add(i) = if i % 97 == 0 { -0.0 } else { ((s >> 8) as f32 / 16777216.0 - 0.5) * 0.2 };
            let sid = (s as usize) % d;
            *src.add(i) = sid as f32;
        }
        for i in 0..d {
            *x.add(i) = ((i * 7919) % 1000) as f32 * 1e-3 - 0.05;
        }
        // 预 gather 版给 S3
        for i in 0..e {
            *xg.add(i) = *x.add(*src.add(i) as usize);
        }

        macro_rules! timeit {
            ($label:expr, $yo:expr, $body:block) => {{
                for _ in 0..3 { $body }
                let mut ts: Vec<f64> = Vec::with_capacity(reps as usize);
                for _ in 0..reps {
                    let t0 = clk.now();
                    $body
                    let t1 = clk.now();
                    ts.push(clk.ms(t1 - t0));
                }
                let mut med = median(&mut ts);
                // 回读，防优化消除（P18 §3 教训）。第一版 S3 报 0.000 ms 就是这个坑：
                // 它读 w/xg、写 y3，三者在 reps 循环里全是不变量，而 y3 只在计时
                // 循环之后才被读 —— LLVM 把整个调用提到了循环外，循环体变空。
                // 对比 S0/S1/S2 没被提走，是因为它们经 gather 读内存，LLVM 无法证明
                // 可安全提升 —— 也就是说 S0 的正确性是运气，不是设计。
                let mut sink = 0.0f64;
                let yp: *mut f32 = $yo;
                let mut i = 0usize;
                while i < d {
                    sink += *yp.add(i) as f64;
                    i += 16;
                }
                GUARD += sink;
                if med < 1.0e-5 {
                    println!("  [FATAL] {} 低于计时下限 -> 循环被删，数字不可用", $label);
                    // 不再只是打印：这个 0 必须被挡在外推表之外（NAN 会被 fmt/rel 拦截）。
                    // 上一次它一路流进表里，打出「infx vs S0」这种垃圾结论。
                    med = f64::NAN;
                }
                println!("  {:<46} {:9.5} ms  {:6.3} ns/conn", $label, med, med * 1e6 / e as f64);
                med
            }};
        }

        let t0 = timeit!("S0 标量现状 (b1_pure.rs:74 逐字复刻)", y0, {
            s0_scalar(w, src, x, y0, d, k0);
        });
        let t1 = timeit!("S1 向量gather + 标量顺序累加 [bit-exact]", y1, {
            s1_vec_gather_scalar_accum(w, src, x, y1, d, k0);
        });
        let t2 = timeit!("S2 向量gather + SIMD归约 [会分叉]", y2, {
            s2_vec_reduce(w, src, x, y2, d, k0);
        });
        let t3 = timeit!("S3 无gather 天花板", y3, {
            s3_no_gather(w, xg, y3, d, k0);
        });

        // ---- bit-exact 判据（这是本探针的重点）----
        s0_scalar(w, src, x, y0, d, k0);
        s1_vec_gather_scalar_accum(w, src, x, y1, d, k0);
        s2_vec_reduce(w, src, x, y2, d, k0);
        let mut d1 = 0usize;
        let mut d2 = 0usize;
        let mut max1 = 0.0f32;
        let mut max2 = 0.0f32;
        for i in 0..d {
            let a = y0.add(i).read();
            let b = y1.add(i).read();
            let c = y2.add(i).read();
            if a.to_bits() != b.to_bits() {
                d1 += 1;
                if (a - b).abs() > max1 {
                    max1 = (a - b).abs();
                }
            }
            if a.to_bits() != c.to_bits() {
                d2 += 1;
                if (a - c).abs() > max2 {
                    max2 = (a - c).abs();
                }
            }
        }
        println!("\n===== A1 逐位判据（vs S0 现状）=====");
        println!("  S1 bit-different = {:<6} / {}  max|diff|={:.3e}   {}", d1, d, max1,
            if d1 == 0 { "<= PASS (bit-exact)" } else { "<= FAIL" });
        println!("  S2 bit-different = {:<6} / {}  max|diff|={:.3e}   {}", d2, d, max2,
            if d2 == 0 { "<= PASS" } else { "<= FAIL (归约改变了分组)" });

        // ---- 外推到 5 matvec x 24 层 x 128 位置 ----
        let calls = 5.0f64 * 24.0 * 128.0;
        let fmt = |t: f64| -> String {
            if t.is_nan() {
                "UNRELIABLE (loop deleted)".to_string()
            } else {
                format!("{:9.1} ms/step", t * calls)
            }
        };
        let rel = |t: f64, base: f64| -> String {
            if t.is_nan() || base.is_nan() || t == 0.0 {
                "  n/a".to_string()
            } else {
                format!("({:.2}x vs S0)", base / t)
            }
        };
        println!("\n===== 外推到 5 matvec x 24 layers x 128 pos =====");
        println!("  S0 标量现状        = {}", fmt(t0));
        println!("  S1 向量+顺序累加   = {}   {}  [bit-exact]", fmt(t1), rel(t1, t0));
        println!("  S2 向量+SIMD归约   = {}   {}  [分叉]", fmt(t2), rel(t2, t0));
        println!("  S3 无gather 天花板 = {}   {}", fmt(t3), rel(t3, t0));
        println!();
        println!("  ⚠️ 注意：S1 把「取乘积」向量化，但**累加仍是标量串行**，");
        println!("     所以 S1 的上限由累加依赖链决定，不是由 gather 决定。");
        println!("     S1 与 S3 的差 = 剩余的 gather 成本；S3 与 0 的差 = 算术/累加下限。");
        println!();
        println!("  用 d={} 测的；真实形状是 d=1536（同 k0，ns/conn 应相近）。", d);
    }
}
