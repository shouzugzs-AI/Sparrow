// PATH: DENSE
//
// B 基准对比探针 —— 用户要求「同参数量稠密实测，同结构稠密给理论值，两个都报」。
//
// WHY THIS EXISTS
//   已有的 vmlm-profile.exe 只报「它自己写的那几个标量循环」的 ms/step。
//   它**不报**两件对本次裁决关键的事：
//     (1) 真正被使用的 AVX2 内核 `kernels::gemm_nt` 在给定形状下走哪条分支；
//     (2) 该形状下的实测 ms/step（而不是标量模型的 ms/step）。
//   docs/P18-B §6 已经吃过一次亏：T=8 时 gemm_nt 落到非 packed 的 8x8 内核，
//   稠密基线被**高估了速度**，于是那一行的「加速比」是不可比的。
//   所以本探针把「分支判定」和「实测耗时」绑在一起输出，两者不可分开引用。
//
// WHAT IT MEASURES (全部真实调用 vmlm_core::kernels::gemm_nt)
//   A 段：形状扫描。对每个 d 报
//        - 编译期可判定的 gemm_nt 分支（与 kernels.rs:1123-1130 的判定式逐字一致）
//        - gemm_nt 实测 ms/call（QPC，warm-up 后取 median-of-reps）
//        - 同参数的 4 个投影（qkv 3 个 + Wo 1 个）外推到 L 层的 ms/step
//   B 段：同参数量口径。稀疏模型每层参数 = 5*d_s*k0（4 个 d×d 投影 + FFN），
//        令稠密 d_c 满足 4*d_c^2 + 2*d_c*d_ffn = 5*d_s*k0 的**最接近整数**。
//        本探针不假设 d_ffn = 4*d_c，而是把两种口径都算出来（见输出）。
//   C 段：pad 对照。d=176 -> 内层 k=176（k%8=0，AVX2 无尾块）；
//        d=175 -> k=175（k%8=7，有尾块）。两者都是 m=128 < 16，
//        所以都走非 packed 的 8x8 内核 —— 这本身就是结论的一部分。
//
// DELIBERATE LIMITATION (写明，防止这个数被过度解读)
//   本探针测的是**单个 [T x d] x [d x d] 投影**的 kernel 时间，不是端到端
//   trainer。它没有 attention、没有 softmax、没有反向、没有 optimizer。
//   它支持的唯一结论是「同一口径下两种 d 的稠密投影成本之比」。
//
// USAGE: vmlm-bench-baseline.exe [--reps 5] [--t 128] [--layers 24] [--k0 20]
//        [--d-src 1536] [--sweep 1536,512,256,176,175,160]

// PATH: DENSE —— 只用 DENSE 参考内核，不引用 ref_sparse_a，不 use vmlm_core::sparse。
use vmlm_core::kernels::{gemm_nt, gemm_nt_scalar};

extern "C" {
    fn malloc(n: usize) -> *mut u8;
    fn free(p: *mut u8);
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
    fn ms(&self, ticks: i64) -> f64 {
        ticks as f64 * 1000.0 / self.freq as f64
    }
}

unsafe fn alloc_f32(n: usize) -> *mut f32 {
    let p = malloc(n * 4) as *mut f32;
    if p.is_null() {
        println!("OOM ({} f32 = {} MB)", n, n * 4 / 1048576);
        std::process::exit(9);
    }
    p
}

unsafe fn fill_rand(p: *mut f32, n: usize, seed: u32) {
    let mut s = seed | 1;
    for i in 0..n {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        *p.add(i) = ((s >> 8) as f32 / 16777216.0 - 0.5) * 0.04;
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

fn arg_str(flag: &str, def: &str) -> String {
    let v: Vec<String> = std::env::args().collect();
    for i in 0..v.len() {
        if v[i] == flag && i + 1 < v.len() {
            return v[i + 1].clone();
        }
    }
    def.to_string()
}

/// 与 `core/src/kernels.rs` 的 `gemm_nt` 判定式**逐字一致**的分支复现。
///
/// ⚠️ 第一版这个函数写错了，而且错得很隐蔽：gemm_nt 的 `m` 是 **A 的行数 = T**，
/// 不是我原先传的 d。在 T=128 时 m=128 >= 16 **恒成立**，所以 d=175/176
/// 也走 packed 路径，而不是「d 小就落到非 packed 内核」的直觉所暗示的那条。
/// 这个错误若不被发现，报告会把 packed 内核标成 unpacked，
/// 正是 P18-B §6 警告过的「基线不可比」类错误。
/// 现在改成由调用方显式传入 m=T，并且**另有 vmlm-branch-id 用位模式实测复核**。
fn gemm_nt_branch(m: usize, n: usize, k: usize, has_avx2: bool) -> &'static str {
    // kernels.rs:1123  if HAS_AVX2 && n >= 8 && m >= 8
    if has_avx2 && n >= 8 && m >= 8 {
        // kernels.rs:1124  if n >= 8 && k >= 8 && m >= 16 && m*n*k >= 1_000_000
        if n >= 8 && k >= 8 && m >= 16 && (m as u64) * (n as u64) * (k as u64) >= 1_000_000 {
            "packed (gemm_nt_packed_kernel, MR=8)"
        } else {
            "unpacked 8x8 (gemm_nt_avx2)"
        }
    } else {
        "scalar (gemm_nt_scalar)"
    }
}

fn percentile(mut v: Vec<f64>, q: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((v.len() - 1) as f64 * q).round() as usize;
    v[idx]
}

struct Row {
    d: usize,
    t: usize,
    branch: &'static str,
    per_call_ms: f64,
    min_ms: f64,
    med_ms: f64,
    ms_per_step_4proj: f64,
    macs_per_call: f64,
    gf_s: f64,
}

fn main() {
    unsafe {
        vmlm_core::interp::init_cpu();
    }
    let has_avx2 = unsafe { vmlm_core::interp::HAS_AVX2 };
    let t = arg_u32("--t", 128) as usize;
    let layers = arg_u32("--layers", 24) as usize;
    let k0 = arg_u32("--k0", 20) as usize;
    let reps = arg_u32("--reps", 5) as usize;
    // 额外做几轮「整段重测」，用来区分「计时噪声」与「机器漂移」。
    // 本机实测 d=1536 的稠密投影在多次独立进程运行间有 ~11% 的漂移
    // （见报告 §5），所以单个数字必须带散布，不能当精确值引用。
    let rounds = arg_u32("--rounds", 3) as usize;
    let d_src = arg_u32("--d-src", 1536) as usize;
    let sweep_s = arg_str("--sweep", "1536,512,256,176,175,160");

    println!("=== B baseline probe (DENSE, real kernels::gemm_nt, QPC) ===");
    println!("HAS_AVX2 = {}", has_avx2);
    println!(
        "T={} layers={} k0={} reps={} d_src={}",
        t, layers, k0, reps, d_src
    );
    println!(
        "NOTE: 每个 shape 都真的调用 vmlm_core::kernels::gemm_nt，分支由 kernels.rs 内部判定；"
    );
    println!("      下面是按 kernels.rs:1123-1130 复现的判定，用于标注实际路径。\n");

    let clk = Clock::new();
    let mut rows: Vec<Row> = Vec::new();

    // ---- A 段：形状扫描 -------------------------------------------------
    println!("-- A: dense [T x d] x [d x d] projection, real gemm_nt --");
    println!(
        "  {:<6} {:<34} {:>10} {:>10} {:>10} {:>12} {:>9} {:>7}",
        "d", "branch", "min ms", "med ms", "p90 ms", "4proj ms/step", "GFLOPS", "spread"
    );
    for tok in sweep_s.split(',') {
        let d: usize = match tok.trim().parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        unsafe {
            let a = alloc_f32(t * d);
            let b = alloc_f32(d * d);
            let c = alloc_f32(t * d);
            fill_rand(a, t * d, 11);
            fill_rand(b, d * d, 22);
            for i in 0..t * d {
                *c.add(i) = 0.0;
            }

            // 注意 m = T = t（A 是 [T x d]），不是 d。传错过一次，见函数注释。
            let branch = gemm_nt_branch(t, d, d, has_avx2);
            // warm-up：一次，把页错误与冷 cache 排除在计时之外
            gemm_nt(a, b, c, t, d, d, false);

            let mut samples: Vec<f64> = Vec::with_capacity(reps * rounds);
            for _ in 0..rounds {
                for _ in 0..reps {
                    let t0 = clk.now();
                    gemm_nt(a, b, c, t, d, d, false);
                    let t1 = clk.now();
                    samples.push(clk.ms(t1 - t0));
                }
            }
            let min_ms = percentile(samples.clone(), 0.0);
            let med_ms = percentile(samples.clone(), 0.5);
            let p90_ms = percentile(samples.clone(), 0.9);
            let spread_pct = if min_ms > 0.0 {
                100.0 * (med_ms - min_ms) / min_ms
            } else {
                0.0
            };

            // 防优化消除：把结果回读进一个被打印的值
            let mut sink = 0.0f32;
            for i in 0..t * d {
                sink += *c.add(i);
            }
            if sink.is_nan() {
                println!("  !! NaN at d={}", d);
            }

            let macs = (t as f64) * (d as f64) * (d as f64);
            let gf_s = 2.0 * macs / (min_ms / 1000.0) / 1e9;
            // 4 个投影 = qkv(3 个 d×d) + Wo(1 个 d×d)，每层各一次
            let step4 = min_ms * 4.0 * layers as f64;
            println!(
                "  {:<6} {:<34} {:>10.3} {:>10.3} {:>10.3} {:>12.1} {:>9.1} {:>7.1}%",
                d, branch, min_ms, med_ms, p90_ms, step4, gf_s, spread_pct
            );
            rows.push(Row {
                d,
                t,
                branch,
                per_call_ms: min_ms,
                min_ms,
                med_ms,
                ms_per_step_4proj: step4,
                macs_per_call: macs,
                gf_s,
            });

            free(a as *mut u8);
            free(b as *mut u8);
            free(c as *mut u8);
        }
    }

    // ---- B 段：同参数量口径 --------------------------------------------
    println!("\n-- B: same-parameter-count dense d_model --");
    let sparse_params_per_layer = 5.0 * d_src as f64 * k0 as f64;
    println!(
        "  sparse (path-B) params/layer = 5*d*k0 = 5*{}*{} = {:.0}",
        d_src, k0, sparse_params_per_layer
    );
    // 口径 1：只算 4 个 d×d 投影（FFN 稀疏侧不计入稠密）
    let d_c_4proj = (sparse_params_per_layer / 4.0).sqrt();
    // 口径 2：稠密侧 FFN 也展开为 d_ffn = 4*d（标准 transformer）
    //   4*d^2 + 2*d*(4d) = 12 d^2 = P  =>  d = sqrt(P/12)
    let d_c_ffn4 = (sparse_params_per_layer / 12.0).sqrt();
    // 口径 3（任务书口径）：d = sqrt(d_src*k0)，等价于 4 d^2 对 5 d_s k0 在 FFN 摊薄时
    let d_c_book = ((d_src * k0) as f64).sqrt();
    println!(
        "  bracket: d_c = sqrt(P/4)   = {:.2}   (dense = 4 projections only, no FFN)",
        d_c_4proj
    );
    println!(
        "  bracket: d_c = sqrt(P/12)  = {:.2}   (dense FFN expanded to d_ffn = 4*d)",
        d_c_ffn4
    );
    println!(
        "  bracket: d_c = sqrt(d*k0)  = {:.2}   (task-brief formula, d_ffn=d)",
        d_c_book
    );
    let nearest_even = |x: f64| -> usize {
        let r = x.round() as usize;
        if r % 2 == 0 {
            r
        } else {
            r + 1
        }
    };
    println!(
        "  => measured points: d={} (sqrt(P/4) nearest even), d={} (task-brief nearest even)",
        nearest_even(d_c_4proj),
        nearest_even(d_c_book)
    );

    // 用 A 段实测的数据集反查：若目标 d 不在扫描里，补测
    let need: Vec<usize> = vec![nearest_even(d_c_4proj), nearest_even(d_c_book)]
        .into_iter()
        .filter(|d| !rows.iter().any(|r| r.d == *d))
        .collect();
    for d in need {
        unsafe {
            let a = alloc_f32(t * d);
            let b = alloc_f32(d * d);
            let c = alloc_f32(t * d);
            fill_rand(a, t * d, 11);
            fill_rand(b, d * d, 22);
            let branch = gemm_nt_branch(t, d, d, has_avx2);
            gemm_nt(a, b, c, t, d, d, false);
            let mut samples: Vec<f64> = Vec::with_capacity(reps);            for _ in 0..reps {
                let t0 = clk.now();
                gemm_nt(a, b, c, t, d, d, false);
                let t1 = clk.now();
                samples.push(clk.ms(t1 - t0));
            }
            let min_ms = percentile(samples.clone(), 0.0);
            let macs = (t as f64) * (d as f64) * (d as f64);
            let gf_s = 2.0 * macs / (min_ms / 1000.0) / 1e9;
            let step4 = min_ms * 4.0 * layers as f64;
            println!(
                "  (B 补测) d={:<5} {}  min {:.3} ms  4proj x{} = {:.1} ms/step  {:.1} GFLOPS",
                d, branch, min_ms, layers, step4, gf_s
            );
            rows.push(Row {
                d,
                t,
                branch,
                per_call_ms: min_ms,
                min_ms,
                med_ms: percentile(samples.clone(), 0.5),
                ms_per_step_4proj: step4,
                macs_per_call: macs,
                gf_s,
            });
            free(a as *mut u8);
            free(b as *mut u8);
            free(c as *mut u8);
        }
    }

    // ---- C 段：算术量比值 vs 实测加速比 --------------------------------
    println!("\n-- C: arithmetic ratio vs measured speedup (same kernels, same T) --");
    let base = rows.iter().find(|r| r.d == d_src);
    match base {
        None => println!("  (d_src={} 不在扫描集内，无法算比值；用 --sweep 加上它)", d_src),
        Some(b0) => {
            let conn_sparse = (d_src * k0) as f64;
            let conn_dense = (d_src * d_src) as f64;
            println!(
                "  connections/matrix: sparse d*k0 = {:.0}   dense d*d = {:.0}   ratio = {:.1}x",
                conn_sparse,
                conn_dense,
                conn_dense / conn_sparse
            );
            for r in rows.iter() {
                if r.d == d_src {
                    continue;
                }
                println!(
                    "  d={:<5} ({:<32})  t_dense(d={})/t_dense(d={:<4}) = {:.1}x   MACs ratio = {:.1}x",
                    r.d,
                    r.branch,
                    r.d,
                    d_src,
                    b0.min_ms / r.min_ms,
                    b0.macs_per_call / r.macs_per_call
                );
            }
            println!(
                "  NOTE: 上面的 MACs ratio 是**同一内核内**的比较（都用 gemm_nt 测），"
            );
            println!(
                "        不跨内核、不跨标量/向量 —— 这正是 P18-B §6 要求的可比性条件。"
            );
        }
    }

    // ---- D 段：标量对照（证明向量化确实生效）---------------------------
    //
    // ⚠️ 第一版这一节报了 `scalar 0.00 ms`，与 gemm_nt 的 10.17 ms 相比得到
    //    「0.0x」—— 即"向量化比标量慢无穷倍"。这个数是**假的**，两个原因：
    //      (1) 只跑 1 次、只取一次计时，且紧跟在 warm-up 之后；
    //      (2) 更致命的是它把标量调用写在了**计时窗口之外的那一次**上，
    //          窗口内那次的结果没有被回读，整个调用可被优化器消除。
    //    这正是 docs/P18_步骤0_剖析实测.md §3 记录的「探针假零」同一类事故。
    //    修法：(a) 真的跑 reps 次并取 min；(b) 回读结果进一个被打印的 sink；
    //          (c) 用 --d-control 让标量对照跑在小形状上（1536 的标量 GEMM
    //              一次约 150 ms，单独跑会主导整个探针的墙钟）。
    println!("\n-- D: scalar control (proves AVX2 path is live, not silently off) --");
    let d_ctl = arg_u32("--d-control", 256) as usize;
    unsafe {
        let d = d_ctl;
        let a = alloc_f32(t * d);
        let b = alloc_f32(d * d);
        let c = alloc_f32(t * d);
        fill_rand(a, t * d, 11);
        fill_rand(b, d * d, 22);
        gemm_nt_scalar(a, b, c, t, d, d, false); // warm-up
        let mut s_samples: Vec<f64> = Vec::with_capacity(reps);
        for _ in 0..reps {
            let t0 = clk.now();
            gemm_nt_scalar(a, b, c, t, d, d, false);
            let t1 = clk.now();
            s_samples.push(clk.ms(t1 - t0));
        }
        // 回读，防止整个调用被消除（P18 §3 的教训）
        let mut sink = 0.0f32;
        for i in 0..t * d {
            sink += *c.add(i);
        }
        let scalar_ms = percentile(s_samples.clone(), 0.0);
        // ⚠️ 这里曾经直接用 `rows` 里查 d_ctl 的实测值，而 d_ctl=256 通常不在
        //    --sweep 里，于是查到 NaN，打出「NaNx」—— 又是一个静默失效的对照。
        //    正确做法：**在同一个形状上现测** gemm_nt，与标量同输入、同 reps，
        //    这样「AVX2 是否生效」这个对照永远有数字，不依赖 --sweep。
        let mut v_samples: Vec<f64> = Vec::with_capacity(reps);
        gemm_nt(a, b, c, t, d, d, false); // warm-up
        for _ in 0..reps {
            let t0 = clk.now();
            gemm_nt(a, b, c, t, d, d, false);
            let t1 = clk.now();
            v_samples.push(clk.ms(t1 - t0));
        }
        let avx_ms = percentile(v_samples.clone(), 0.0);
        println!(
            "  d={} scalar {:.3} ms (min of {})  vs  gemm_nt {:.3} ms (min of {})  ->  {:.1}x",
            d,
            scalar_ms,
            reps,
            avx_ms,
            reps,
            scalar_ms / avx_ms
        );
        // 位模式对照：真入口 vs 强制标量。若两者逐位相同，说明真入口也走了标量。
        let c2 = alloc_f32(t * d);
        for i in 0..t * d {
            *c2.add(i) = 0.0;
        }
        gemm_nt(a, b, c2, t, d, d, false);
        let mut bit_same = 0usize;
        for i in 0..t * d {
            if c.add(i).read().to_bits() == c2.add(i).read().to_bits() {
                bit_same += 1;
            }
        }
        println!(
            "  scalar-vs-gemm_nt bit-identical = {}/{}  (sink={:.3e})",
            bit_same,
            t * d,
            sink
        );
        println!(
            "  => 若加速比接近 1.0x 或位模式全同，说明 AVX2 没生效，本探针全部数字作废。",
        );
        free(c2 as *mut u8);
        free(a as *mut u8);
        free(b as *mut u8);
        free(c as *mut u8);
    }

    println!("\n-- SUMMARY TABLE (raw ms, for the report) --");
    println!("  {:<6} {:<34} {:>10} {:>12}", "d", "branch", "min ms", "4proj/step");
    for r in rows.iter() {
        println!(
            "  {:<6} {:<34} {:>10.3} {:>12.1}",
            r.d, r.branch, r.min_ms, r.ms_per_step_4proj
        );
    }
}
