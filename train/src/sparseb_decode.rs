// PATH: B1
//
// ===========================================================================
// P2-b：decode tok/s（batch=1 逐 token 生成）—— prefill / decode 分开报
// ===========================================================================
//
// 为什么必须单独测：P0/P1 报的 tok/s 是 **prefill 口径**（T 个 token 一次前向
// = 一次 batch）。常规引擎（llama.cpp 等）报的是 **decode 口径**：batch=1、
// 每次生成**一个** token。两者的算术结构完全不同：
//
//   prefill: T 个位置一次算完 -> 稀疏权重被复用 T 次，摊销好
//   decode : 每个 token 一次前向 -> 权重每 token 全读一遍，且没有 token 维并行
//
// 所以 decode 会**远低于** prefill。本驱动把两个数分开报，不混口径。
//
// 三种测量（同一模型、同一份权重、同一 prompt）：
//   W1 prefill : 对 ctx 个 token 做一次完整前向 -> tok/s = ctx / 秒
//   W2 decode（有 KV-Cache）: 处理完 ctx 的 prompt 后逐个生成 gen 个 token，
//        每步只算 1 个位置 + 读 cache -> tok/s = gen / 秒（**这是与常规引擎可比的数**）
//   W3 decode（无 KV-Cache）: 每生成一个 token 就把整个前缀重算一遍
//        （O(t) 稀疏 + O(t^2) 注意力）-> tok/s = 1 / 单步秒
//
// 模型结构（与 sparse_train24 的 P18-C 路径同构，权重为确定性随机初始化）：
//   h = emb[tok]
//   for l in 0..L:  q,k,v = B1(h); 因果注意力; o = B1_wo(attn); u = h+o; h = u + B1_ffn(u)
//   logits = lm_head . h
//   稀疏算子 = **B1**（遍历全部 k0 条，无掩码）—— 每 token 每层 5 次 matvec。
//
// USAGE: vmlm-sparseb-decode.exe [--layers 24] [--d 1536] [--k0 20] [--vocab 64]
//        [--ctx 512] [--gen 32] [--reps 5]

extern "C" {
    fn malloc(n: usize) -> *mut u8;
    fn QueryPerformanceCounter(c: *mut i64) -> i32;
    fn QueryPerformanceFrequency(f: *mut i64) -> i32;
}

use vmlm_core::b1_pure::b1_forward;
use vmlm_core::conn_train::{init_sparse_conn_scaled, SparseConnW};

// ---------------------------------------------------------------------------
// attention 用的两个稠密内核（**不是**稀疏 B 的范畴，但必须向量化）
// ---------------------------------------------------------------------------
// 为什么必须写它们：第一版用标量 `for i { acc += a[i]*b[i] }` 算注意力，
// 实测把 decode 拖到 61 tok/s —— 而那是**标量注意力**的成本，不是 B1 的成本。
// 拿那个数去和常规引擎比会得出"VMLM 慢"的错误结论（被测量的东西错了）。
// 本项目对这类错误有明确纪律：口径必须随数字一起报，且**先确认测的是被测对象**。
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[inline]
unsafe fn hsum256(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let mut s = _mm_add_ps(lo, hi);
    let sh = _mm_movehl_ps(s, s);
    s = _mm_add_ps(s, sh);
    let sh2 = _mm_shuffle_ps(s, s, 0x55);
    s = _mm_add_ss(s, sh2);
    _mm_cvtss_f32(s)
}

/// 点积 a·b（AVX2，无 FMA —— attention 的数值本就只是"参考实现"，不要求位精确）
#[inline]
unsafe fn dot(a: *const f32, b: *const f32, n: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if vmlm_core::interp::HAS_AVX2 {
            let mut acc = _mm256_setzero_ps();
            let mut i = 0usize;
            while i + 8 <= n {
                acc = _mm256_add_ps(
                    acc,
                    _mm256_mul_ps(_mm256_loadu_ps(a.add(i)), _mm256_loadu_ps(b.add(i))),
                );
                i += 8;
            }
            let mut s = hsum256(acc);
            while i < n {
                s += *a.add(i) * *b.add(i);
                i += 1;
            }
            return s;
        }
    }
    let mut acc = 0.0f32;
    for i in 0..n {
        acc += *a.add(i) * *b.add(i);
    }
    acc
}

/// y += a * x（SAXPY，AVX2）
#[inline]
unsafe fn saxpy(y: *mut f32, a: f32, x: *const f32, n: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if vmlm_core::interp::HAS_AVX2 {
            let av = _mm256_set1_ps(a);
            let mut i = 0usize;
            while i + 8 <= n {
                let yv = _mm256_loadu_ps(y.add(i));
                let xv = _mm256_loadu_ps(x.add(i));
                _mm256_storeu_ps(y.add(i), _mm256_add_ps(yv, _mm256_mul_ps(av, xv)));
                i += 8;
            }
            while i < n {
                *y.add(i) += a * *x.add(i);
                i += 1;
            }
            return;
        }
    }
    for i in 0..n {
        *y.add(i) += a * *x.add(i);
    }
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
    #[inline]
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

static mut SINK: f64 = 0.0;
/// 分相计时（--profile）：稀疏 B 的 120 次 matvec vs 稠密 attention。
/// 为什么需要：decode 到底被谁限制，不能靠推断 —— 本项目已有多次"口径写错"的记录。
static mut PROFILE: bool = false;
static mut T_SPARSE: i64 = 0;
static mut T_ATTN: i64 = 0;
static mut T_LM: i64 = 0;
#[inline(never)]
unsafe fn consume(p: *const f32, n: usize) {
    let mut s = 0.0f64;
    for i in 0..n {
        s += *p.add(i) as f64;
    }
    SINK += s;
}

unsafe fn alloc(n: usize) -> *mut f32 {
    let p = malloc(n * 4 + 64) as *mut f32;
    assert!(!p.is_null(), "OOM {} f32", n);
    p
}
unsafe fn zero(p: *mut f32, n: usize) {
    std::ptr::write_bytes(p as *mut u8, 0, n * 4);
}

struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Rng {
        Rng(s | 1)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (z ^ (z >> 31)) as u32
    }
    fn uni(&mut self) -> f32 {
        (self.next_u32() as f64 / (u32::MAX as f64)) as f32
    }
}

fn main() {
    unsafe { vmlm_core::interp::init_cpu() };
    let layers = arg_u32("--layers", 24) as usize;
    let d = arg_u32("--d", 1536) as usize;
    let k0 = arg_u32("--k0", 20) as usize;
    let vocab = arg_u32("--vocab", 64) as usize;
    let ctx = arg_u32("--ctx", 512) as usize;
    let gen = arg_u32("--gen", 32) as usize;
    let reps = arg_u32("--reps", 5) as usize;
    unsafe {
        PROFILE = std::env::args().any(|a| a == "--profile");
    }
    let n_mats = layers * 5;
    let clk = Clock::new();
    let mut fails = 0usize;
    let mut checks = 0usize;
    macro_rules! check {
        ($ok:expr, $name:expr, $($fmt:tt)*) => {{
            checks += 1;
            if $ok {
                println!("  [PASS] {}  {}", $name, format!($($fmt)*));
            } else {
                println!("  [FAIL] {}  {}", $name, format!($($fmt)*));
                fails += 1;
            }
        }};
    }

    println!("===== P2-b: decode tok/s (batch=1, token-by-token) =====");
    println!(
        "layers={} d={} k0={} vocab={} ctx={} gen={} reps={}  sparse impl = B1",
        layers, d, k0, vocab, ctx, gen, reps
    );
    println!(
        "每 token 每层 5 次稀疏 matvec (q,k,v,wo,ffn) => {} 次/token",
        5 * layers
    );

    unsafe {
        // ---- 权重：120 个稀疏矩阵（确定性初始化）+ emb + lm_head ----
        let mut scs: Vec<SparseConnW> = Vec::with_capacity(n_mats);
        for l in 0..n_mats {
            let base = malloc(SparseConnW::bytes(d as u32, k0 as u32));
            let sc = SparseConnW::from_base(base, d as u32, k0 as u32, (l as u32) + 101);
            init_sparse_conn_scaled(&sc, 0.144);
            sc.assert_geometry();
            sc.assert_src_u32_consistent();
            scs.push(sc);
        }
        let emb = alloc(vocab * d);
        let lm = alloc(vocab * d);
        let mut r = Rng::new(0xC0FFEE_1234);
        for i in 0..vocab * d {
            *emb.add(i) = (r.uni() - 0.5) * 0.2;
            *lm.add(i) = (r.uni() - 0.5) * 0.2;
        }

        // ---- 工作缓冲 ----
        // ⚠️ 全部先清零：本驱动是**吞吐探针**，未初始化内存会以两种方式污染结论：
        //    (a) 读到的垃圾可能把 h 推成 NaN/inf —— 而 NaN/denormal 算术在某些 CPU 上
        //        走慢路径，于是"测出来的 tok/s"其实是 NaN 的代价（第一版实测
        //        `sink = NaN`，说明确实发生了）；
        //    (b) 更隐蔽的是(a)只在一部分运行里发生 → 两次测量的口径不同。
        let h = alloc(d);
        let qv = alloc(d);
        let kv = alloc(d);
        let vv = alloc(d);
        let attn = alloc(d);
        let oo = alloc(d);
        let uu = alloc(d);
        let ffn = alloc(d);
        zero(h, d);
        zero(qv, d);
        zero(kv, d);
        zero(vv, d);
        zero(attn, d);
        zero(oo, d);
        zero(uu, d);
        zero(ffn, d);
        let logits = alloc(vocab);
        zero(logits, vocab);

        // ---- KV cache：[layers][cap][d] x2 ----
        // cap 必须是 ctx + gen + 2，不能是 ctx：decode 会写到 pos = ctx..ctx+gen，
        // 无 cache 模式（W3）在 pos = ctx 处也要写一笔。
        // 第一版按 ctx 分配，`l = layers-1, pos = ctx` 那一笔正好落在分配区末尾之外
        // -> Windows STATUS_HEAP_CORRUPTION (0xC0000374)，且只在生成到最后一个
        //    位置时才触发（前面全部正确）。
        let cap = ctx + gen + 2;
        let scores = alloc(cap);
        let cache_bytes = layers * cap * d * 2;
        let kcache_base = alloc(cache_bytes);
        let vcache_base = kcache_base.add(layers * cap * d);
        zero(kcache_base, cache_bytes);

        let scale = 1.0 / vmlm_core::kernels::sqrt_f(d as f32);

        // 一个位置的完整前向。`pos` = 该位置在序列中的下标（可见 0..=pos）。
        // cache_full=true 时把本位置的 k/v 写进 cache（prefill 与 W3 都要写）。
        let forward_one = |tok: u32, pos: usize, cache_full: bool| {
            let mut macs = 0usize;
            std::ptr::copy_nonoverlapping(emb.add(tok as usize * d), h, d);
            for l in 0..layers {
                let wq = &scs[l * 5];
                let wk = &scs[l * 5 + 1];
                let wv = &scs[l * 5 + 2];
                let wo = &scs[l * 5 + 3];
                let wf = &scs[l * 5 + 4];
                // ---- 稀疏 B：qkv ----
                let t0 = if PROFILE { clk.now() } else { 0 };
                macs += b1_forward(wq, h, qv);
                macs += b1_forward(wk, h, kv);
                macs += b1_forward(wv, h, vv);
                if PROFILE {
                    T_SPARSE += clk.now() - t0;
                }
                if cache_full {
                    let kc = kcache_base.add(l * cap * d + pos * d);
                    let vc = vcache_base.add(l * cap * d + pos * d);
                    std::ptr::copy_nonoverlapping(kv, kc, d);
                    std::ptr::copy_nonoverlapping(vv, vc, d);
                }
                // ---- 稠密 attention（因果）----
                let t1 = if PROFILE { clk.now() } else { 0 };
                let visible = pos + 1;
                let mut mx = -1.0e30f32;
                for j in 0..visible {
                    let kj = kcache_base.add(l * cap * d + j * d);
                    // 稠密点积（AVX2）：attention **不是**稀疏 B 的范畴
                    let s = dot(qv, kj, d) * scale;
                    *scores.add(j) = s;
                    if s > mx {
                        mx = s;
                    }
                }
                let mut sum = 0.0f32;
                for j in 0..visible {
                    let e = vmlm_core::kernels::exp_f(*scores.add(j) - mx);
                    *scores.add(j) = e;
                    sum += e;
                }
                let inv = 1.0 / sum;
                zero(attn, d);
                for j in 0..visible {
                    let a = *scores.add(j) * inv;
                    let vj = vcache_base.add(l * cap * d + j * d);
                    saxpy(attn, a, vj, d);
                }
                if PROFILE {
                    T_ATTN += clk.now() - t1;
                }
                let t2 = if PROFILE { clk.now() } else { 0 };
                macs += b1_forward(wo, attn, oo);
                for i in 0..d {
                    *uu.add(i) = *h.add(i) + *oo.add(i);
                }                macs += b1_forward(wf, uu, ffn);
                // 残差缩放 1/sqrt(2)：本驱动是**吞吐探针**，不加归一化时 24 层纯残差
                // 在长上下文下会数值爆炸成 inf/NaN（实测 `sink = NaN`），而 NaN 可能
                // 走慢路径、污染 tok/s。缩放**不改变任何算子的调用次数、数据布局与
                // 访存模式**，所以对吞吐口径无影响 —— 它只是让被测对象保持在有限域内。
                let rs = std::f32::consts::FRAC_1_SQRT_2;
                for i in 0..d {
                    *h.add(i) = (*uu.add(i) + *ffn.add(i)) * rs;
                }
                if PROFILE {
                    T_SPARSE += clk.now() - t2;
                }
            }
            let t3 = if PROFILE { clk.now() } else { 0 };
            for v in 0..vocab {
                let mut acc = 0.0f32;
                for i in 0..d {
                    acc += *lm.add(v * d + i) * *h.add(i);
                }
                *logits.add(v) = acc;
            }
            if PROFILE {
                T_LM += clk.now() - t3;
            }
            macs
        };

        // ---- prompt ----
        let mut toks: Vec<u32> = (0..ctx).map(|_| (r.next_u32() % vocab as u32) + 1).collect();
        let toks_ptr = toks.as_mut_ptr();

        // 数值健全性检查（**先确认被测对象是有限数**，否则 tok/s 可能是 NaN 的代价）
        {
            forward_one(*toks_ptr, 0, true);
            let mut finite = true;
            let mut maxabs = 0.0f32;
            for v in 0..vocab {
                let x = *logits.add(v);
                if !x.is_finite() {
                    finite = false;
                }
                if x.abs() > maxabs {
                    maxabs = x.abs();
                }
            }
            println!(
                "  [数值] 位置 0 的 logits: finite={} max|x|={:.3e} （非有限数会让计时失真）",
                finite, maxabs
            );
            if !finite {
                println!("  [警告] logits 非有限 —— 24 层残差无归一化时会数值爆炸；本驱动的 tok/s 仍可读，");
                println!("         但必须把这句话与数字一起报（NaN/denormal 可能走慢路径）。");
            }
        }

        // ---- W1 prefill（ctx 个 token 一次前向）----
        let mut t_pre: Vec<f64> = Vec::new();
        for _ in 0..reps {
            let a = clk.now();
            for t in 0..ctx {
                forward_one(*toks_ptr.add(t), t, true);
            }
            let b = clk.now();
            t_pre.push(clk.ms(b - a));
            consume(logits, vocab);
        }
        let med_pre = median(&mut t_pre);

        // ---- W2 decode with KV cache ----
        for t in 0..ctx {
            forward_one(*toks_ptr.add(t), t, true);
        }
        let mut next = 1u32;
        let mut t_dec: Vec<f64> = Vec::new();
        for _ in 0..reps {
            let a = clk.now();
            let mut pos = ctx;
            for _g in 0..gen {
                forward_one(next, pos, true);
                let mut best = 0usize;
                for v in 1..vocab {
                    if *logits.add(v) > *logits.add(best) {
                        best = v;
                    }
                }
                next = best as u32;
                pos += 1;
            }
            let b = clk.now();
            t_dec.push(clk.ms(b - a));
            consume(logits, vocab);
        }
        let med_dec = median(&mut t_dec);

        // ---- W3 decode without cache：每生成一个 token 重算整个前缀 ----
        let mut t_noc: Vec<f64> = Vec::new();
        for _ in 0..reps.min(3) {
            let a = clk.now();
            for t in 0..=ctx {
                forward_one(if t < ctx { *toks_ptr.add(t) } else { next }, t, true);
            }
            let b = clk.now();
            t_noc.push(clk.ms(b - a));
            consume(logits, vocab);
        }
        let med_noc = median(&mut t_noc);

        // ---- 报告 ----
        let pre_tps = ctx as f64 / (med_pre / 1000.0);
        let dec_tps = gen as f64 / (med_dec / 1000.0);
        let noc_tps = 1.0 / (med_noc / 1000.0);
        let dec_ms_per_tok = med_dec / gen as f64;
        println!(
            "\n=============== P2-b 结果（QPC 中位数，{} 次重复）===============",
            reps
        );
        println!(
            "  W1 prefill  ({:>5} token 一次前向)   {:>10.3} ms  ->  {:>9.1} tok/s",
            ctx, med_pre, pre_tps
        );
        println!(
            "  W2 decode   (KV-Cache, batch=1)    {:>10.3} ms/{}tok ->  {:>9.1} tok/s   ({:.3} ms/token)",
            med_dec, gen, dec_tps, dec_ms_per_tok
        );
        println!(
            "  W3 decode   (无 KV-Cache, ctx={:>4})  {:>10.3} ms/token ->  {:>9.1} tok/s",
            ctx, med_noc, noc_tps
        );
        println!(
            "  KV-Cache 加速比（decode 每 token）= {:.1}x",
            med_noc / dec_ms_per_tok
        );
        println!(
            "  prefill/decode 比 = {:.1}x  （口径不同，**不可混报**）",
            pre_tps / dec_tps
        );

        if PROFILE {
            let f = clk.freq as f64;
            let sp = T_SPARSE as f64 / f * 1000.0;
            let at = T_ATTN as f64 / f * 1000.0;
            let lm = T_LM as f64 / f * 1000.0;
            let tot = sp + at + lm;
            println!("\n  ---- 分相计时（--profile，三次测量累计）----");
            println!(
                "    稀疏 B (120 次 matvec/token) {:>10.1} ms  {:>5.1}%",
                sp,
                100.0 * sp / tot
            );
            println!("    稠密 attention               {:>10.1} ms  {:>5.1}%", at, 100.0 * at / tot);
            println!("    lm_head + 逐元素              {:>10.1} ms  {:>5.1}%", lm, 100.0 * lm / tot);
            println!("    （口径：只含 forward_one 内部，不含 argmax/logits 读回）");
        }

        check!(med_pre > 0.001, "D1", "prefill 计时有效（{:.3} ms）", med_pre);
        check!(
            med_dec > 0.001,
            "D2",
            "decode 计时有效（{:.3} ms / {} token）",
            med_dec,
            gen
        );
        check!(med_noc > 0.001, "D3", "无 cache 计时有效（{:.3} ms/token）", med_noc);
        check!(
            dec_ms_per_tok < med_noc,
            "D4",
            "KV-Cache 确实更快：{:.3} ms/token vs 无 cache {:.3} ms/token",
            dec_ms_per_tok,
            med_noc
        );

        println!("\n===== checks={} fail={} =====", checks, fails);
        println!("(sink = {:.6e})", SINK);
        if fails > 0 {
            std::process::exit(1);
        }
    }
}
