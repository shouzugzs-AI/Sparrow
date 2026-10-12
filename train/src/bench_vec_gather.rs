// PATH: DENSE
//
// B 基准：复核 Lead 在 `lead_b1_gap.rs` 的推论 —— 「74.5% 的 B1 时间是
// f32→int 索引解码」。Lead 的 L1/L2 是**标量循环**，本探针回答的是
// 「换成已有的**向量化** gather 路径后，收益是否同比例」。
//
// 为什么必须单独测这一点
//   `sparse_strength_row_avx2`（kernels.rs:1681-1707）是**唯一**向量化的
//   gather 路径，而它的形状与 Lead 的标量 L1/L2 不同：
//     - 它每次处理 8 个连接（256-bit），而标量循环每次 1 个；
//     - 关键的是 `_mm256_cvttps_epi32`（:1695）产生 **8 个** i32，
//       而标量版 `*src.add(j) as usize` 每次只转换 1 个。
//   向量化之后，转换指令的执行**槽位**被 8 路摊薄，但 `vcvttps2dq`
//   的吞吐（典型 1/cycle on port 1 或 5）未必与标量 `cvttss2si` 同倍。
//   所以「标量测出 4x」**不能**直接外推到向量路径 —— 这正是 Lead 问的问题。
//
// 本探针的做法
//   同一个 `sparse_strength_row_avx2` 的数据流，两种 src 布局：
//     V1: src 存成 f32（现状）—— 走 `_mm256_cvttps_epi32` + gather
//     V2: src 存成 u32 —— 直接 `_mm256_loadu_si256`，**去掉转换**
//   两版都真的调用 `_mm256_i32gather_ps`，都是 8 宽，唯一差别是转换。
//   另外报 V3 = 纯 gather（无 mul/abs）以分离 gather 本身的成本。
//
// USAGE: vmlm-vec-gather.exe [--d 1536] [--k0 20] [--reps 2000]

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
// V1: 现状 —— src 是 f32，每次 gather 前 cvttps2dq（与 kernels.rs:1695 同构）
// ---------------------------------------------------------------
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn v1_f32_src(
    wrow: *const f32,
    srcrow: *const f32,
    x: *const f32,
    k0: usize,
    out: *mut f32,
) {
    let sign = _mm256_set1_ps(-0.0);
    let mut j = 0usize;
    while j + 8 <= k0 {
        let wv = _mm256_loadu_ps(wrow.add(j));
        let sid_f = _mm256_loadu_ps(srcrow.add(j));
        let sid = _mm256_cvttps_epi32(sid_f); // <-- 被测量的转换
        let xv = _mm256_i32gather_ps(x, sid, 4);
        let a = _mm256_andnot_ps(sign, _mm256_mul_ps(wv, xv));
        _mm256_storeu_ps(out.add(j), a);
        j += 8;
    }
}

// ---------------------------------------------------------------
// V2: src 是 u32 —— 转换消失，其余逐条相同
// ---------------------------------------------------------------
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn v2_u32_src(
    wrow: *const f32,
    srcrow: *const u32,
    x: *const f32,
    k0: usize,
    out: *mut f32,
) {
    let sign = _mm256_set1_ps(-0.0);
    let mut j = 0usize;
    while j + 8 <= k0 {
        let wv = _mm256_loadu_ps(wrow.add(j));
        let sid = _mm256_loadu_si256(srcrow.add(j) as *const __m256i); // 无转换
        let xv = _mm256_i32gather_ps(x, sid, 4);
        let a = _mm256_andnot_ps(sign, _mm256_mul_ps(wv, xv));
        _mm256_storeu_ps(out.add(j), a);
        j += 8;
    }
}

// ---------------------------------------------------------------
// V3: 纯 gather（无 mul、无 abs）—— 分离 gather 本身
// ---------------------------------------------------------------
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn v3_gather_only(srcrow: *const u32, x: *const f32, k0: usize, out: *mut f32) {
    let mut j = 0usize;
    while j + 8 <= k0 {
        let sid = _mm256_loadu_si256(srcrow.add(j) as *const __m256i);
        let xv = _mm256_i32gather_ps(x, sid, 4);
        _mm256_storeu_ps(out.add(j), xv);
        j += 8;
    }
}

// ---------------------------------------------------------------
// V4: f32 src 但**只转换、不用**（把转换单独拎出来，看它值多少）
//     说明：这版会被优化器把 sid 消掉，所以加一个依赖它的写。
// ---------------------------------------------------------------
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn v4_convert_only(srcrow: *const f32, k0: usize, out: *mut f32) {
    let mut j = 0usize;
    while j + 8 <= k0 {
        let sid_f = _mm256_loadu_ps(srcrow.add(j));
        let sid = _mm256_cvttps_epi32(sid_f);
        // 用一个依赖 sid 的浮点运算把它钉住，但不做 gather
        let sf = _mm256_cvtepi32_ps(sid);
        _mm256_storeu_ps(out.add(j), sf);
        j += 8;
    }
}

fn main() {
    unsafe { vmlm_core::interp::init_cpu() };
    let d = arg_u32("--d", 1536) as usize;
    let k0 = arg_u32("--k0", 20) as usize;
    let reps = arg_u32("--reps", 2000) as usize;
    let has_avx2 = unsafe { vmlm_core::interp::HAS_AVX2 };

    println!("=== 向量化 gather: src=f32 (现状) vs src=u32 ===");
    println!("d={} k0={} reps={} HAS_AVX2={}", d, k0, reps, has_avx2);
    if !has_avx2 {
        println!("FATAL: 无 AVX2，本探针的向量路径不可用");
        std::process::exit(2);
    }
    // k0=20 不是 8 的倍数：向量主体只覆盖 j=0..16，余下 4 条走标量。
    // 这里只测**向量部分**（16 条/神经元），并在报告里写明覆盖率。
    let vec_k = (k0 / 8) * 8;
    let e_vec = d * vec_k;
    println!(
        "注意：k0={} -> 向量主体覆盖每神经元 {} 条（{:.0}%），余 {} 条走标量尾块。",
        k0,
        vec_k,
        100.0 * vec_k as f64 / k0 as f64,
        k0 - vec_k
    );

    let clk = Clock::new();
    unsafe {
        let w = malloc(e_vec * 4) as *mut f32;
        let src_f = malloc(e_vec * 4) as *mut f32;
        let src_u = malloc(e_vec * 4) as *mut u32;
        let x = malloc(d * 4) as *mut f32;
        let o1 = malloc(e_vec * 4) as *mut f32;
        let o2 = malloc(e_vec * 4) as *mut f32;
        let mut s: u32 = 12345;
        for i in 0..e_vec {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *w.add(i) = ((s >> 8) as f32 / 16777216.0 - 0.5) * 0.2;
            let sid = (s as usize) % d;
            *src_f.add(i) = sid as f32;
            *src_u.add(i) = sid as u32;
        }
        for i in 0..d {
            *x.add(i) = ((i * 7919) % 1000) as f32 * 1e-3;
        }

        macro_rules! timeit {
            ($label:expr, $body:block) => {{
                for _ in 0..5 { $body }
                let mut ts: Vec<f64> = Vec::with_capacity(reps as usize);
                for _ in 0..reps {
                    let t0 = clk.now();
                    $body
                    let t1 = clk.now();
                    ts.push(clk.ms(t1 - t0));
                }
                let med = median(&mut ts);
                // 回读，防优化消除（P18 §3 教训）
                let mut sink = 0.0f64;
                let mut i = 0usize;
                while i < e_vec {
                    sink += *o1.add(i) as f64;
                    i += 64;
                }
                GUARD += sink;
                println!(
                    "  {:<44} {:9.5} ms  {:6.3} ns/conn",
                    $label,
                    med,
                    med * 1e6 / e_vec as f64
                );
                med
            }};
        }

        let t1 = timeit!("V1 src=f32 (cvttps2dq + gather) [现状]", {
            let mut i = 0usize;
            while i < d {
                v1_f32_src(w.add(i * vec_k), src_f.add(i * vec_k), x, vec_k, o1.add(i * vec_k));
                i += 1;
            }
        });
        let t2 = timeit!("V2 src=u32 (无转换 + gather)", {
            let mut i = 0usize;
            while i < d {
                v2_u32_src(w.add(i * vec_k), src_u.add(i * vec_k), x, vec_k, o2.add(i * vec_k));
                i += 1;
            }
        });
        let t3 = timeit!("V3 纯 gather (无 mul/abs)", {
            let mut i = 0usize;
            while i < d {
                v3_gather_only(src_u.add(i * vec_k), x, vec_k, o1.add(i * vec_k));
                i += 1;
            }
        });
        let t4 = timeit!("V4 只做转换 (cvttps2dq, 无 gather)", {
            let mut i = 0usize;
            while i < d {
                v4_convert_only(src_f.add(i * vec_k), vec_k, o1.add(i * vec_k));
                i += 1;
            }
        });

        // 正确性：V1 与 V2 必须给出相同结果（同一个 src 集合）
        let mut bad = 0usize;
        v1_f32_src(w, src_f, x, e_vec, o1);
        v2_u32_src(w, src_u, x, e_vec, o2);
        for i in 0..e_vec {
            if o1.add(i).read().to_bits() != o2.add(i).read().to_bits() {
                bad += 1;
            }
        }
        println!("\n  正确性 V1 vs V2: bit-different = {} / {}", bad, e_vec);

        if t1 < 1e-4 || t2 < 1e-4 {
            println!("  [FATAL] 某版低于计时下限 -> 循环被删，数字不可用");
            std::process::exit(1);
        }

        println!("\n===== 结论（向量化路径）=====");
        println!("  V1 (f32 src) = {:9.5} ms", t1);
        println!("  V2 (u32 src) = {:9.5} ms", t2);
        println!("  => u32 src 省下 {:9.5} ms = {:.1}%", t1 - t2, 100.0 * (t1 - t2) / t1);
        println!("  => 向量化路径的加速比 = {:.2}x", t1 / t2);
        println!();
        println!("  分离项：");
        println!("  V3 纯 gather      = {:9.5} ms  ({:.3} ns/conn)", t3, t3 * 1e6 / e_vec as f64);
        println!("  V4 只做转换       = {:9.5} ms  ({:.3} ns/conn)", t4, t4 * 1e6 / e_vec as f64);
        println!(
            "  V1 - V3 (转换+乘+abs 的增量) = {:9.5} ms",
            t1 - t3
        );
        println!();
        println!("  ⚠️ 用 V4（纯转换）作为「转换的天花板」：即便转换完全消除，");
        println!(
            "     向量路径最多省 {:.1}%（V4/V1），而不是标量测出的 74.5%。",
            100.0 * t4 / t1
        );
    }
}
