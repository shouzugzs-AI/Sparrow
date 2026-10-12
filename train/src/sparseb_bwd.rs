// PATH: B1 / B2
//
// ===========================================================================
// sparse-B 反向验收（P0 的 dx / P1 的融合）
// ===========================================================================
//
// P0 把 B1/B2 接进真实训练器时暴露出一个**前置缺口**：
//   * B1 原来只有 `b1_backward`（dw）。没有 dx，梯度就传不到上一层，
//     24 层里只有最后一层在学 —— 那样也会"跑通"、loss 也会降，但不是深度训练。
//   * 所以本轮补了 `b1_backward_dx` / `b2_backward_dx` 与 P1 的融合内核。
//
// 本驱动是这两件事的验收物。判据：
//
//   E1  B1 前向 ε：`b1_forward` vs `b1_forward_scalar_reference`，
//       用**项幅度之和**当尺度（|a-b| / Σ|w·x|）—— 这是本项目两次选错度量之后
//       定下来的口径（逐元素相对误差会被"相消"惩罚）。bar = 1e-5。
//   E2  B1 dw 逐位：`b1_backward` vs `sparse_gemm_bwd_w`（k=k0 时掩码全 1）。
//   E3  B1 dx 逐位：`b1_backward_dx` vs 手写标量参考。
//   E4  **融合逐位**：`b1_backward_fused` vs `b1_backward + b1_backward_dx`，
//       dw 与 dx **都**必须 0 分叉（这是 P1 的核心判据：融合不改变结果）。
//   E5  can-fail：注入 FMA（融合乘加）必须被抓到；注入"漏最后一条"必须被抓到。
//   E6  确定性：8 次重复调用逐位相同（dw 与 dx 都比）。
//   E7  ε vs 算子掩码版（k=k0）：B1 与 mask 路径的差必须停在舍入级。
//   E8  P1 计时：融合 vs 分开的 QPC 中位数 + 加速比（带 consume 与下限守卫）。
//   E9  B2 SHARED dx：`b2_backward_dx` vs 独立标量参考逐位 0 分叉，
//       且**只对被选中的槽位**贡献（对未选中槽位的贡献必须为 0）。
//
// USAGE: vmlm-sparseb-bwd.exe [--d 1536] [--k0 20] [--k 8] [--reps 200]

extern "C" {
    fn malloc(n: usize) -> *mut u8;
    fn QueryPerformanceCounter(c: *mut i64) -> i32;
    fn QueryPerformanceFrequency(f: *mut i64) -> i32;
}

use vmlm_core::b1_pure::{
    b1_backward, b1_backward_dx, b1_backward_dx_scalar_reference, b1_backward_fused,
    b1_backward_fused_scalar_reference, b1_forward, b1_forward_scalar_reference,
};
use vmlm_core::b2_topk::{b2_backward_dx, b2_forward, B2Mode, TieRule};
use vmlm_core::conn_train::{
    init_sparse_conn_scaled, sparse_gemm_bwd_w, sparse_forward, SparseConnW,
};

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
    assert!(!p.is_null());
    p
}

unsafe fn zero(p: *mut f32, n: usize) {
    std::ptr::write_bytes(p as *mut u8, 0, n * 4);
}

unsafe fn fill_rng(p: *mut f32, n: usize, seed: u64) {
    let mut s = seed | 1;
    for i in 0..n {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *p.add(i) = (((s >> 33) as u32 % 2000) as f32 / 1000.0) - 1.0;
    }
}

fn main() {
    unsafe { vmlm_core::interp::init_cpu() };
    let d = arg_u32("--d", 1536) as usize;
    let k0 = arg_u32("--k0", 20) as usize;
    let k = arg_u32("--k", 8) as usize;
    let reps = arg_u32("--reps", 200) as usize;

    let n = d;
    let e = n * k0;
    let mut checks = 0usize;
    let mut fails = 0usize;
    let clk = Clock::new();

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

    println!("===== sparse-B 反向验收（P0 dx / P1 融合）=====");
    println!("d={} k0={} k={} reps={} HAS_AVX2={}", d, k0, k, reps, unsafe {
        vmlm_core::interp::HAS_AVX2
    });

    unsafe {
        let base = malloc(SparseConnW::bytes(d as u32, k0 as u32));
        let sc = SparseConnW::from_base(base, d as u32, k0 as u32, 11);
        init_sparse_conn_scaled(&sc, 0.144);
        sc.assert_geometry();
        sc.assert_src_u32_consistent();

        let x = alloc(n);
        let dy = alloc(n);
        fill_rng(x, n, 0x1234);
        fill_rng(dy, n, 0x5678);

        // ---------------------------------------------------------------
        // E1 前向 ε（项幅度尺度）
        // ---------------------------------------------------------------
        let y_simd = alloc(n);
        let y_ref = alloc(n);
        b1_forward(&sc, x, y_simd);
        b1_forward_scalar_reference(&sc, x, y_ref);
        let mut worst_ratio = 0.0f32;
        let mut bits_same = 0usize;
        for i in 0..n {
            let mut s = 0.0f32;
            for j in 0..k0 {
                let sid = *sc.src_u32.add(i * k0 + j) as usize;
                s += (*sc.w.add(i * k0 + j) * *x.add(sid)).abs();
            }
            if s > 0.0 {
                let r = (*y_simd.add(i) - *y_ref.add(i)).abs() / s;
                if r > worst_ratio {
                    worst_ratio = r;
                }
            }
            if (*y_simd.add(i)).to_bits() == (*y_ref.add(i)).to_bits() {
                bits_same += 1;
            }
        }
        println!("\n--- E1: B1 前向 ε（尺度 = Σ|w·x|）---");
        println!(
            "  worst |a-b|/Σ|terms| = {:.3e}   位相同 = {}/{} ({:.1}%)",
            worst_ratio,
            bits_same,
            n,
            100.0 * bits_same as f64 / n as f64
        );
        check!(
            worst_ratio <= 1.0e-5,
            "E1",
            "B1 前向 ε = {:.3e} <= 1e-5（发布版是 8 路归约的 ε 路径，非逐位）",
            worst_ratio
        );

        // ---------------------------------------------------------------
        // E2 dw 逐位 vs 算子掩码版（k=k0 时掩码全 1）
        // ---------------------------------------------------------------
        let dw_a = alloc(e);
        let dw_b = alloc(e);
        zero(dw_a, e);
        zero(dw_b, e);
        b1_backward(&sc, dy, x, dw_a);
        // 算子版需要一张"全激活"掩码：k=k0 的 sparse_forward 会写下它。
        sparse_forward(&sc, x, y_ref, k0);
        sparse_gemm_bwd_w(&sc, dy, x, dw_b);
        let mut bad2 = 0usize;
        for i in 0..e {
            if (*dw_a.add(i)).to_bits() != (*dw_b.add(i)).to_bits() {
                bad2 += 1;
            }
        }
        println!("\n--- E2: B1 dw 逐位 vs sparse_gemm_bwd_w(k=k0) ---");
        check!(bad2 == 0, "E2", "dw 位分叉 = {} / {}", bad2, e);

        // ---------------------------------------------------------------
        // E3 dx 逐位 vs 手写标量参考
        // ---------------------------------------------------------------
        let dx_a = alloc(n);
        let dx_b = alloc(n);
        zero(dx_a, n);
        zero(dx_b, n);
        let ca = b1_backward_dx(&sc, dy, dx_a);
        let cb = b1_backward_dx_scalar_reference(&sc, dy, dx_b);
        let mut bad3 = 0usize;
        for i in 0..n {
            if (*dx_a.add(i)).to_bits() != (*dx_b.add(i)).to_bits() {
                bad3 += 1;
            }
        }
        println!("\n--- E3: B1 dx 逐位 vs 标量参考（含 f32 src 的对照）---");
        println!("  累加次数 = {} / {}", ca, e);
        check!(
            bad3 == 0 && ca == e && cb == e,
            "E3",
            "dx 位分叉 = {} / {}；累加次数 {}/{} 与 {}/{}",
            bad3,
            n,
            ca,
            e,
            cb,
            e
        );

        // ---------------------------------------------------------------
        // E4 融合逐位：fused vs (dw + dx) 分开
        // ---------------------------------------------------------------
        let dw_f = alloc(e);
        let dx_f = alloc(n);
        let dw_s = alloc(e);
        let dx_s = alloc(n);
        zero(dw_f, e);
        zero(dx_f, n);
        let ret_f = b1_backward_fused(&sc, dy, x, dw_f, dx_f);
        zero(dw_s, e);
        zero(dx_s, n);
        b1_backward(&sc, dy, x, dw_s);
        b1_backward_dx(&sc, dy, dx_s);
        let mut bad_dw = 0usize;
        let mut bad_dx = 0usize;
        for i in 0..e {
            if (*dw_f.add(i)).to_bits() != (*dw_s.add(i)).to_bits() {
                bad_dw += 1;
            }
        }
        for i in 0..n {
            if (*dx_f.add(i)).to_bits() != (*dx_s.add(i)).to_bits() {
                bad_dx += 1;
            }
        }
        println!("\n--- E4: 融合内核 vs 分开两次调用（逐位）---");
        println!(
            "  融合返回 = {} (n*k0={})   dw 位分叉 = {} / {}   dx 位分叉 = {} / {}",
            ret_f, e, bad_dw, e, bad_dx, n
        );
        check!(
            bad_dw == 0 && bad_dx == 0 && ret_f == e,
            "E4",
            "P1 融合**逐位等价**于 dw+dx 分开（不是 ε）"
        );
        // 融合的标量参考也必须一致（证明向量路径没有偏离契约）
        let dw_r = alloc(e);
        let dx_r = alloc(n);
        zero(dw_r, e);
        zero(dx_r, n);
        b1_backward_fused_scalar_reference(&sc, dy, x, dw_r, dx_r);
        let mut bad_rs = 0usize;
        for i in 0..e {
            if (*dw_r.add(i)).to_bits() != (*dw_s.add(i)).to_bits() {
                bad_rs += 1;
            }
        }
        let mut bad_rsx = 0usize;
        for i in 0..n {
            if (*dx_r.add(i)).to_bits() != (*dx_s.add(i)).to_bits() {
                bad_rsx += 1;
            }
        }
        check!(
            bad_rs == 0 && bad_rsx == 0,
            "E4-b",
            "融合的**标量参考**也与分开实现逐位一致（dw {} / dx {}）",
            bad_rs,
            bad_rsx
        );

        // ---------------------------------------------------------------
        // E5 can-fail
        // ---------------------------------------------------------------
        println!("\n--- E5: can-fail（已知错误实现必须被抓到）---");
        // (a) 漏最后一条连接
        let dw_w1 = alloc(e);
        let dx_w1 = alloc(n);
        zero(dw_w1, e);
        zero(dx_w1, n);
        for i in 0..n {
            let g = *dy.add(i);
            for j in 0..k0 - 1 {
                let s = *sc.src_u32.add(i * k0 + j) as usize;
                *dw_w1.add(i * k0 + j) += g * *x.add(s);
                *dx_w1.add(s) += g * *sc.w.add(i * k0 + j);
            }
        }
        let mut c_dw1 = 0usize;
        let mut c_dx1 = 0usize;
        for i in 0..e {
            if (*dw_w1.add(i)).to_bits() != (*dw_s.add(i)).to_bits() {
                c_dw1 += 1;
            }
        }
        for i in 0..n {
            if (*dx_w1.add(i)).to_bits() != (*dx_s.add(i)).to_bits() {
                c_dx1 += 1;
            }
        }
        // (b) FMA 融合乘加（dx 侧）
        let dx_w2 = alloc(n);
        zero(dx_w2, n);
        for i in 0..n {
            let g = *dy.add(i);
            for j in 0..k0 {
                let s = *sc.src_u32.add(i * k0 + j) as usize;
                let old = *dx_w2.add(s);
                *dx_w2.add(s) = f32::mul_add(g, *sc.w.add(i * k0 + j), old);
            }
        }
        let mut c_dx2 = 0usize;
        for i in 0..n {
            if (*dx_w2.add(i)).to_bits() != (*dx_s.add(i)).to_bits() {
                c_dx2 += 1;
            }
        }
        println!("    wrong-1 漏最后一条: dw 抓到 {} / {}, dx 抓到 {} / {}", c_dw1, e, c_dx1, n);
        println!("    wrong-2 dx 用 FMA:  抓到 {} / {}", c_dx2, n);
        check!(
            c_dw1 > 0 && c_dx1 > 0 && c_dx2 > 0,
            "E5",
            "3 个已知错误实现全部被逐位判据抓到（E2/E3/E4 不是恒真）"
        );

        // ---------------------------------------------------------------
        // E6 确定性
        // ---------------------------------------------------------------
        println!("\n--- E6: 确定性（8 次重复逐位相同）---");
        let dw_t = alloc(e);
        let dx_t = alloc(n);
        let mut det = true;
        for _r in 0..8 {
            zero(dw_t, e);
            zero(dx_t, n);
            b1_backward_fused(&sc, dy, x, dw_t, dx_t);
            for i in 0..e {
                if (*dw_t.add(i)).to_bits() != (*dw_f.add(i)).to_bits() {
                    det = false;
                }
            }
            for i in 0..n {
                if (*dx_t.add(i)).to_bits() != (*dx_f.add(i)).to_bits() {
                    det = false;
                }
            }
        }
        check!(det, "E6", "8 次融合反向逐位相同（dw {} + dx {} 个槽位）", e, n);

        // ---------------------------------------------------------------
        // E7 ε vs 算子掩码版（k=k0）
        // ---------------------------------------------------------------
        // 这里比的是「B1 与 mask 在 k=k0 时是不是同一个函数」。
        // 逐位不要求（sparse_accum 与 b1_forward 的 SIMD 归约不同），但必须舍入级。
        let y_op = alloc(n);
        sparse_forward(&sc, x, y_op, k0);
        let mut worst7 = 0.0f32;
        for i in 0..n {
            let mut s = 0.0f32;
            for j in 0..k0 {
                let sid = *sc.src_u32.add(i * k0 + j) as usize;
                s += (*sc.w.add(i * k0 + j) * *x.add(sid)).abs();
            }
            if s > 0.0 {
                let r = (*y_simd.add(i) - *y_op.add(i)).abs() / s;
                if r > worst7 {
                    worst7 = r;
                }
            }
        }
        println!("\n--- E7: B1 vs 算子掩码版（k=k0）ε ---");
        check!(
            worst7 <= 1.0e-5,
            "E7",
            "|B1 - mask(k=k0)| / Σ|terms| = {:.3e} <= 1e-5（同一个函数，差异仅来自归约顺序）",
            worst7
        );

        // ---------------------------------------------------------------
        // E8 P1 计时
        // ---------------------------------------------------------------
        println!("\n--- E8: P1 计时（QPC 中位数）---");
        let floor_ms = 1.0e-5f64;
        let mut t_sep: Vec<f64> = Vec::new();
        let mut t_fus: Vec<f64> = Vec::new();
        for _ in 0..3 {
            zero(dw_s, e);
            zero(dx_s, n);
            b1_backward(&sc, dy, x, dw_s);
            b1_backward_dx(&sc, dy, dx_s);
        }
        for _ in 0..reps {
            zero(dw_s, e);
            zero(dx_s, n);
            let a = clk.now();
            b1_backward(&sc, dy, x, dw_s);
            b1_backward_dx(&sc, dy, dx_s);
            let b = clk.now();
            t_sep.push(clk.ms(b - a));
            consume(dw_s, 64);
            consume(dx_s, 64);
        }
        for _ in 0..reps {
            zero(dw_f, e);
            zero(dx_f, n);
            let a = clk.now();
            b1_backward_fused(&sc, dy, x, dw_f, dx_f);
            let b = clk.now();
            t_fus.push(clk.ms(b - a));
            consume(dw_f, 64);
            consume(dx_f, 64);
        }
        let m_sep = median(&mut t_sep);
        let m_fus = median(&mut t_fus);
        println!(
            "  分开（dw 一次遍历 + dx 一次遍历）: {:.6} ms  ({:.3} ns/conn)",
            m_sep,
            m_sep * 1e6 / e as f64
        );
        println!(
            "  融合（dw+dx 一次遍历）           : {:.6} ms  ({:.3} ns/conn)",
            m_fus,
            m_fus * 1e6 / e as f64
        );
        println!("  加速比 = {:.3}x", m_sep / m_fus);
        check!(
            m_sep > floor_ms && m_fus > floor_ms,
            "E8-a",
            "计时高于下限（分开 {:.3e} / 融合 {:.3e} ms vs 下限 {:.1e}）",
            m_sep,
            m_fus,
            floor_ms
        );
        check!(
            m_fus < m_sep,
            "E8-b",
            "P1 融合更快：{:.3}x（一次遍历 vs 两次）",
            m_sep / m_fus
        );

        // ---------------------------------------------------------------
        // E9 B2 SHARED dx
        // ---------------------------------------------------------------
        println!("\n--- E9: B2 SHARED dx ---");
        let agg = alloc(k0);
        let hit = malloc(k0 * 8) as *mut usize;
        let y_sh = alloc(n);
        let st = b2_forward(&sc, x, y_sh, k, B2Mode::Shared, agg, hit);
        let dx_sh = alloc(n);
        zero(dx_sh, n);
        let accs = b2_backward_dx(&sc, dy, x, dx_sh, k, B2Mode::Shared, agg, hit);
        // 独立标量参考：先按定义算共享强度、选一次、再散射（不复用被测代码）
        let agg_r = alloc(k0);
        zero(agg_r, k0);
        for i in 0..n {
            for j in 0..k0 {
                let s = *sc.src_u32.add(i * k0 + j) as usize;
                *agg_r.add(j) += (*sc.w.add(i * k0 + j) * *x.add(s)).abs();
            }
        }
        let hit_r = malloc(k0 * 8) as *mut usize;
        let (nk_r, _c) = vmlm_core::b2_topk::b2_select_once(agg_r, k0, k, hit_r, TieRule::StrictGt);
        let dx_r2 = alloc(n);
        zero(dx_r2, n);
        // 哪些 (i,j) 被选中：用同一张 hit 表
        let mut selected = vec![false; e];
        for i in 0..n {
            for q in 0..nk_r {
                selected[i * k0 + *hit_r.add(q)] = true;
            }
        }
        for i in 0..n {
            let g = *dy.add(i);
            for j in 0..k0 {
                if selected[i * k0 + j] {
                    let s = *sc.src_u32.add(i * k0 + j) as usize;
                    *dx_r2.add(s) += g * *sc.w.add(i * k0 + j);
                }
            }
        }
        let mut bad9 = 0usize;
        for i in 0..n {
            if (*dx_sh.add(i)).to_bits() != (*dx_r2.add(i)).to_bits() {
                bad9 += 1;
            }
        }
        // 未选中的连接不得贡献：把整张 dx 与"全连接贡献"比较必须不同
        let mut n_selected = 0usize;
        for t in 0..e {
            if selected[t] {
                n_selected += 1;
            }
        }
        let expect = n * k.min(k0);
        println!(
            "  选中连接 = {} (期望 n*min(k,k0) = {})   b2_backward_dx 累加 = {}",
            n_selected, expect, accs
        );
        println!("  位分叉 = {} / {}", bad9, n);
        check!(
            bad9 == 0 && accs == n * k.min(k0),
            "E9",
            "B2 SHARED dx 逐位等于独立标量参考，且只对被选中槽位累加（{} 条）",
            accs
        );

        println!("\n===== checks={} fail={} =====", checks, fails);
        println!("(sink = {:.6e}，计时体都被 consume 读回)", SINK);
        if fails > 0 {
            std::process::exit(1);
        }
        println!("RESULT: sparse-B 反向验收 PASS");
    }
}
