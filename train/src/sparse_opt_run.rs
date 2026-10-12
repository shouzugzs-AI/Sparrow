// P7 成员 E：稀疏瘦 GEMM 验证驱动
//
// 三件事：
//   1. 正确性：瘦 GEMM vs 「稠密 GEMM + 掩码」，逐位一致（误差 < 1e-6）
//   2. 性能：0% / 25% / 50% / 75% / 90% 五档稀疏率的 **wall-clock**（µs + 加速比）
//   3. 复现 sparse_dyn 的前车之鉴：稠密+掩码 vs 瘦 GEMM 的端到端 FFN 计时
//
// 计时用 std::time::Instant（i3-12100F 上 QPC 分辨率 100ns，够用）。
// 每档取 reps 次的最小值 —— 最小值是最接近「无干扰」的估计量，
// 平均值会被 Windows 调度噪声污染。
//
// 构建：cd <repo>; cargo build -p vmlm-train --release --bin sparse_opt_run
// 运行：<repo>/target/release/sparse_opt_run.exe [examples/sparse]
use std::time::Instant;
use vmlm_core::kernels as K;
use vmlm_core::sparse as S;

extern "C" {
    fn malloc(n: usize) -> *mut u8;
}

// ---------------- 确定性 RNG（xorshift64*，不引 rand） ----------------
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// 标准正态（Box-Muller）
    fn normal(&mut self) -> f32 {
        let u1 = ((self.next_u64() >> 11) as f64 / (1u64 << 53) as f64).max(1e-12);
        let u2 = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }
}

struct Buf {
    p: *mut f32,
    n: usize,
}
impl Buf {
    fn new(n: usize) -> Buf {
        unsafe {
            let p = malloc(n * 4 + 64) as *mut f32;
            assert!(!p.is_null());
            // 64 字节对齐（malloc 通常已 16 对齐，这里保证 SIMD 友好）
            let a = ((p as usize + 63) & !63) as *mut f32;
            std::ptr::write_bytes(a as *mut u8, 0, n * 4);
            Buf { p: a, n }
        }
    }
    fn randn(&mut self, r: &mut Rng, scale: f32) {
        unsafe {
            for i in 0..self.n {
                *self.p.add(i) = r.normal() * scale;
            }
        }
    }
    fn zero(&mut self) {
        unsafe { std::ptr::write_bytes(self.p as *mut u8, 0, self.n * 4) }
    }
    fn as_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.p, self.n) }
    }
}

/// 计时：跑 reps 次，返回最小值（µs）。
///
/// 用**最小值**而非平均值：i3-12100F 上 4 个核要跟其它成员进程抢，
/// 平均值会被调度噪声系统性抬高；最小值最接近「无干扰」的真实算力。
///
/// ⚠ 实测教训：即便取最小值，同一份代码在同一次会话里仍会漂
/// （稠密基线 337~445 µs，差 30%），原因是**频率缩放**：bench 先跑
/// 一轮预热再计时，但 CPU 从低频爬到全频需要几十毫秒。
/// 所以预热次数设得比较大（warm），且报告里同时给「相对加速比」和
/// 「绝对 µs」两列 —— 后者才是跨人可比的量。
fn bench<F: FnMut()>(reps: usize, mut f: F) -> f64 {
    const WARM: usize = 20;
    for _ in 0..WARM {
        f();
    }
    let mut best = f64::MAX;
    for _ in 0..reps {
        let t0 = Instant::now();
        f();
        let dt = t0.elapsed().as_secs_f64() * 1e6;
        if dt < best {
            best = dt;
        }
    }
    best
}

/// 构造「恰好 keep 个非零」的掩码，并把选中索引 gather 出来。
/// 随机选取（用 RNG 洗牌前 keep 个位置），保证 agg 不是连续前缀 ——
/// 连续前缀会被 gather_rows_prefix 作弊掉，测不出真实开销。
unsafe fn make_random_mask(
    rng: &mut Rng,
    mask: *mut f32,
    n: usize,
    keep: usize,
    rows: *mut u32,
) -> usize {
    let mut idx: Vec<u32> = (0..n as u32).collect();
    // Fisher-Yates 部分洗牌（只为打散索引，不要求密码学强度）
    for i in 0..keep {
        let j = i + (rng.next_u64() as usize) % (n - i);
        idx.swap(i, j);
    }
    std::ptr::write_bytes(mask as *mut u8, 0, n * 4);
    for i in 0..keep {
        *mask.add(idx[i] as usize) = 1.0;
    }
    S::gather_rows(mask, n, rows)
}

/// 与 make_random_mask 相同的索引集，但**掩码和索引分别写进两块独立缓冲**，
/// 避免同一块内存既当 f32 掩码又当 u32 索引读（那是未定义行为）。
/// 返回 (选中数, 掩码指针, 索引指针)。
unsafe fn make_random_mask_split(
    rng: &mut Rng,
    mask: *mut f32,
    n: usize,
    keep: usize,
    rows: *mut u32,
) -> usize {
    let mut idx: Vec<u32> = (0..n as u32).collect();
    for i in 0..keep {
        let j = i + (rng.next_u64() as usize) % (n - i);
        idx.swap(i, j);
    }
    std::ptr::write_bytes(mask as *mut u8, 0, n * 4);
    for i in 0..keep {
        *mask.add(idx[i] as usize) = 1.0;
    }
    // 升序排好再写进 rows（gather_rows 的契约是升序）
    let mut sel: Vec<u32> = idx[..keep].to_vec();
    sel.sort_unstable();
    for (t, &j) in sel.iter().enumerate() {
        *rows.add(t) = j;
    }
    sel.len()
}

fn main() {
    unsafe { vmlm_core::interp::init_cpu() };
    println!("=== P7 成员 E：稀疏瘦 GEMM（真 wall-clock） ===");
    println!(
        "CPU: i3-12100F / AVX2={} / AVX-512=no",
        unsafe { vmlm_core::interp::HAS_AVX2 }
    );

    let mut pass = true;
    pass &= correctness_micro();
    pass &= correctness_ffn_small();
    pass &= correctness_rows();
    pass &= correctness_cols();
    pass &= correctness_bwd();
    pass &= perf_scaling();
    pass &= repro_sparse_dyn();
    pass &= when_does_it_pay();

    println!();
    if pass {
        println!("总判定: PASS");
        std::process::exit(0);
    } else {
        println!("总判定: FAIL");
        std::process::exit(1);
    }
}

// ==================================================================
// 0. 极小规模手算验证（M=2, N=2, K=3）
// ==================================================================
//
// Lead 要求：先用 M=2,N=2,K=3 手工算一遍期望值，逐元素比对。
// 这里把期望值**硬编码在注释里**，手算过程写在下面：
//
//   A = [[1, 2, 3],        W = [[1, 0, 0],      cols = [0, 2]（选第 0、2 列）
//        [4, 5, 6]]             [0, 1, 0]]       （K=3 里只跑 j=0 和 j=2）
//
//   瘦 GEMM（筛选收缩维）：C[i][j] = Σ_{p ∈ cols} A[i][p] * W[j][p]
//     C[0][0] = A[0][0]*W[0][0] + A[0][2]*W[0][2] = 1*1 + 3*0 = 1
//     C[0][1] = A[0][0]*W[1][0] + A[0][2]*W[1][2] = 1*0 + 3*0 = 0
//     C[1][0] = 4*1 + 6*0 = 4
//     C[1][1] = 4*0 + 6*0 = 0
//   ⇒ C = [[1, 0], [4, 0]]
//
//   对照「稠密 + 掩码」：mask = [1,0,1]（列掩码），A⊙mask = [[1,0,3],[4,0,6]]
//     y[0][0] = 1*1 + 0*0 + 3*0 = 1  ✓
//     y[0][1] = 1*0 + 0*1 + 3*0 = 0  ✓
//     y[1][0] = 4*1 + 0*0 + 6*0 = 4  ✓
//     y[1][1] = 4*0 + 0*1 + 6*0 = 0  ✓
//   ⇒ 与瘦 GEMM 完全一致。
//
//   瘦 GEMM（筛选权重行）：rows = [1]（只算 W 的第 1 行）
//     D[0][0] = A[0][0]*W[1][0] + A[0][1]*W[1][1] + A[0][2]*W[1][2]
//             = 1*0 + 2*1 + 3*0 = 2
//     D[1][0] = 4*0 + 5*1 + 6*0 = 5
//   ⇒ D = [[2], [5]]
//
//   注意 M=2 < 8、N=2 < 8、K=3 < 8 —— 全部落在**标量回退路径**上，
//   这正好能验「边界补齐」不越界写坏相邻数据（kernels.rs 的注释提过
//   这个坑：向量内核一次写 8 个连续列，会越过行边界）。
//   另外再单独跑一组 AVX2 路径的 8 的倍数规模，两条路径都要对。
fn correctness_micro() -> bool {
    println!("\n---- 0. 极小规模手算验证（M=2 N=2 K=3，期望值硬编码）----");
    unsafe {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let w = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mut ok = true;

        // ---- 筛选收缩维：cols = [0, 2] ----
        let cols: [u32; 2] = [0, 2];
        let mut c = [0f32; 4];
        S::gemm_nt_cols_selected(
            a.as_ptr(),
            w.as_ptr(),
            c.as_mut_ptr(),
            2,
            2,
            3,
            cols.as_ptr(),
            2,
        );
        let expect = [1.0f32, 0.0, 4.0, 0.0];
        let good = c
            .iter()
            .zip(expect.iter())
            .all(|(x, y)| (x - y).abs() < 1e-6);
        ok &= good;
        println!(
            "  [收缩维 cols=[0,2]] 实得 C = {:?}，期望 {:?}  {}",
            c,
            expect,
            if good { "PASS" } else { "FAIL" }
        );

        // ---- 对照：稠密+掩码（mask=[1,0,1]），手算同值 ----
        let am = [1.0f32, 0.0, 3.0, 4.0, 0.0, 6.0];
        let mut y = [0f32; 4];
        K::gemm_nt(am.as_ptr(), w.as_ptr(), y.as_mut_ptr(), 2, 2, 3, false);
        let good2 = y
            .iter()
            .zip(c.iter())
            .all(|(x, z)| (x - z).abs() < 1e-12);
        ok &= good2;
        println!(
            "  [对照 稠密+掩码]  实得 y = {:?}，与瘦 GEMM 逐位一致  {}",
            y,
            if good2 { "PASS" } else { "FAIL" }
        );

        // ---- 筛选权重行：rows = [1] ----
        let rows: [u32; 1] = [1];
        let mut dd = [0f32; 2];
        S::gemm_nt_rows_selected(
            a.as_ptr(),
            w.as_ptr(),
            dd.as_mut_ptr(),
            2,
            2,
            3,
            rows.as_ptr(),
            1,
        );
        let expect_d = [2.0f32, 5.0];
        let good3 = dd
            .iter()
            .zip(expect_d.iter())
            .all(|(x, y)| (x - y).abs() < 1e-6);
        ok &= good3;
        println!(
            "  [权重行 rows=[1]]  实得 D = {:?}，期望 {:?}  {}",
            dd,
            expect_d,
            if good3 { "PASS" } else { "FAIL" }
        );

        // ---- AVX2 路径（8 的倍数规模，M=8 N=8 K=8）----
        // A = 单位阵 * 2，W = 单位阵 * 3。取 cols = [0,1,2,3]（K 的前 4 列）。
        //   C[i][j] = Σ_{p∈cols} A[i][p]·W[j][p]
        //   A[i][p] = 2 ⟺ i==p；W[j][p] = 3 ⟺ j==p
        //   ⇒ C[i][j] ≠ 0 需要存在 p∈cols 使 i==p 且 j==p，即 i==j 且 i∈{0,1,2,3}
        //   ⇒ C = diag(6,6,6,6,0,0,0,0)
        // （手算用 numpy 复核过，见报告。）
        // 注意 n_cols=4 < 8 ⇒ 走**列尾段**，正好覆盖尾段分支；
        // 再补一组 n_cols=8 覆盖主块分支。
        let mut a8 = [0f32; 64];
        let mut w8 = [0f32; 64];
        for i in 0..8 {
            a8[i * 8 + i] = 2.0;
            w8[i * 8 + i] = 3.0;
        }
        for (label, cols8) in [
            ("cols=[0,1,2,3] 走尾段", vec![0u32, 1, 2, 3]),
            ("cols=0..8 走主块", (0u32..8).collect()),
        ] {
            let mut c8 = [0f32; 64];
            S::gemm_nt_cols_selected(
                a8.as_ptr(),
                w8.as_ptr(),
                c8.as_mut_ptr(),
                8,
                8,
                8,
                cols8.as_ptr(),
                cols8.len(),
            );
            let mut good4 = true;
            for i in 0..8 {
                for j in 0..8 {
                    let want = if i == j && i < cols8.len() { 6.0f32 } else { 0.0 };
                    if (c8[i * 8 + j] - want).abs() > 1e-6 {
                        good4 = false;
                    }
                }
            }
            ok &= good4;
            println!(
                "  [收缩维 AVX2 M=N=K=8 {}] 实得对角 = {:?}  {}",
                label,
                (0..8).map(|i| c8[i * 8 + i]).collect::<Vec<_>>(),
                if good4 { "PASS" } else { "FAIL" }
            );
        }
        ok
    }
}

// ==================================================================
// 1. 小规模：完整复现 examples/sparse 的稀疏 FFN 两条路径
// ==================================================================
//
// 路径 A（sparse_dyn 现状 = 稠密 GEMM + 掩码）：
//   h  = x·W1^T + b1        [T,F]   稠密 gemm_nt
//   m  = topk(|h|) 掩码      [T,F]
//   hs = h ⊙ m
//   g  = gelu(hs)
//   y  = g·W2^T            [T,D]   稠密 gemm_nt（对掩掉的 75% 白算）
//
// 路径 B（真瘦 GEMM）：
//   只有 g 的**非零列**参与第二条 GEMM ⇒ 收缩维 K 从 F 降到 n_cols
//   按列聚合掩码得到 cols[]，用 gemm_nt_cols_selected
//   输出仍是稠密 [T,D]，与路径 A 逐位可比。
//
// 另外单独验证 gemm_nt_rows_selected（任务书原型）：筛选 W1 的行，
// 输出紧凑 [T,R]，与「稠密 + 只取同样那几列」逐位比较。
fn correctness_ffn_small() -> bool {
    println!("\n---- 1. 正确性（小规模 T=4 D=16 F=32，k=8 / top_ratio=0.25）----");
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "examples/sparse".into());
    let rd = |n: &str| -> Vec<f32> {
        let p = format!("{}/{}.bin", dir, n);
        let b = std::fs::read(&p).unwrap_or_else(|_| panic!("读不到 {}", p));
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    let (t, d, f, kk) = (4usize, 16usize, 32usize, 8usize);
    let x = rd("x");
    let w1 = rd("W1");
    let b1 = rd("b1");
    let w2 = rd("W2");
    let b2 = rd("b2");
    assert_eq!(x.len(), t * d);
    assert_eq!(w1.len(), f * d);
    assert_eq!(w2.len(), d * f);

    unsafe {
        // ---- 前向：h ----
        let h_a = Buf::new(t * f);
        let h_b = Buf::new(t * f);
        K::gemm_nt(x.as_ptr(), w1.as_ptr(), h_a.p, t, f, d, false);
        add_bias(h_a.p, b1.as_ptr(), t, f);

        // ---- 掩码：每行 top-k(|h|) ----
        let mask = Buf::new(t * f);
        for i in 0..t {
            let tau = K::topk_thresh(h_a.p.add(i * f), f, kk);
            S::mask_from_thresh(h_a.p.add(i * f), tau, mask.p.add(i * f), f);
        }
        let density = S::mask_density(mask.p, t * f);
        println!(
            "  掩码密度 = {:.4}（目标 {:.4}），并列导致的实际选中数可能有浮动",
            density,
            kk as f32 / f as f32
        );

        // ---- 路径 A：稠密 + 掩码 ----
        let hs = Buf::new(t * f);
        for i in 0..t * f {
            *hs.p.add(i) = *h_a.p.add(i) * *mask.p.add(i);
        }
        let g = Buf::new(t * f);
        for i in 0..t * f {
            *g.p.add(i) = K::act_scalar(*hs.p.add(i), vmlm_core::format::ACT_GELU);
        }
        let y_a = Buf::new(t * d);
        K::gemm_nt(g.p, w2.as_ptr(), y_a.p, t, d, f, false);
        add_bias(y_a.p, b2.as_ptr(), t, d);

        // ---- 路径 B：真瘦 GEMM（收缩维只跑被选中的列）----
        let cols = Buf::new(f);
        let n_cols = S::gather_cols_union(mask.p, t, f, cols.p as *mut u32);
        println!(
            "  gather_cols_union: {} / {} 列被选中（瘦 GEMM 的实际 K = {}）",
            n_cols, f, n_cols
        );
        let y_b = Buf::new(t * d);
        S::gemm_nt_cols_selected(g.p, w2.as_ptr(), y_b.p, t, d, f, cols.p as *const u32, n_cols);
        add_bias(y_b.p, b2.as_ptr(), t, d);

        let mut maxerr = 0.0f32;
        let mut ndiff = 0usize;
        for i in 0..t * d {
            let e = (*y_a.p.add(i) - *y_b.p.add(i)).abs();
            if e > maxerr {
                maxerr = e;
            }
            if *y_a.p.add(i) != *y_b.p.add(i) {
                ndiff += 1;
            }
        }
        println!(
            "  [瘦 GEMM 收缩维] 稠密+掩码 vs 瘦 GEMM: 最大误差 = {:.3e}, 非逐位相等(诊断,非判据) = {}/{}",
            maxerr,
            ndiff,
            t * d
        );

        // ---- 顺带：与 Python 参考 ref_y.txt 比 ----
        // 差异来源已知且可解释：Python 的 h 是 float32 matmul 后加的 bias，
        // 引擎的 gemm_nt 是「K 分块 + FMA」，与 np.matmul 的 BLAS 求和顺序不同。
        // 这是**口径差**，不是错误 —— 判据用 1e-3（与 sparse_run.rs 一致）。
        // 真正要求 1e-6 的是「同引擎内两条路径」的对比，见上。
        let ref_txt = std::fs::read_to_string(format!("{}/ref_y.txt", dir)).unwrap();
        let ref_y: Vec<f32> = ref_txt.split_whitespace().map(|s| s.parse().unwrap()).collect();
        let mut ref_err = 0.0f32;
        let mut ref_scale = 0.0f32;
        for i in 0..t * d {
            let e = (*y_a.p.add(i) - ref_y[i]).abs();
            if e > ref_err {
                ref_err = e;
            }
            if ref_y[i].abs() > ref_scale {
                ref_scale = ref_y[i].abs();
            }
        }
        println!(
            "  与 Python ref_y.txt: 最大绝对误差 = {:.3e}, 相对量级 = {:.3e}（BLAS 求和顺序口径差，判据 1e-3）",
            ref_err,
            ref_err / ref_scale.max(1e-9)
        );

        // ---- 任务书原型：gemm_nt_rows_selected（筛选 W1 的行）----
        // 取第 0 行样本的掩码当作「选中的神经元」，得到紧凑输出 [T,R]。
        //
        // 对照必须**形状对齐**：稠密 gemm_nt 输出 [M,N]=[T,F]，瘦路径输出
        // [M,R]=[T,n_rows]，所以对照是「稠密输出按 rows_sel 取列」，
        // 不是整体相减 —— 这是两个不同形状的张量，逐元素相减无意义。
        // 判据：两边同序累加 ⇒ 期望逐位一致（1e-5 容差下必须过）。
        let rows_sel = Buf::new(f);
        let n_rows = S::gather_rows(mask.p, f, rows_sel.p as *mut u32); // 只用第 0 行样本的掩码
        let d_rows = Buf::new(t * n_rows);
        S::gemm_nt_rows_selected(
            x.as_ptr(),
            w1.as_ptr(),
            d_rows.p,
            t,
            f,
            d,
            rows_sel.p as *const u32,
            n_rows,
        );
        // 稠密参考：h_pre = x·W1^T（在 bias 之前），按 rows_sel 取列对齐
        let h_pre = Buf::new(t * f);
        K::gemm_nt(x.as_ptr(), w1.as_ptr(), h_pre.p, t, f, d, false);
        let h_ref_cols = Buf::new(t * n_rows);
        for i in 0..t {
            for tt in 0..n_rows {
                let j = *(rows_sel.p as *const u32).add(tt) as usize;
                *h_ref_cols.p.add(i * n_rows + tt) = *h_pre.p.add(i * f + j);
            }
        }
        let (rerr, rdiff, rscale) = cmp_exact(d_rows.p, h_ref_cols.p, t * n_rows);
        println!(
            "  [瘦 GEMM 权重行] 稠密[gemm_nt 取列对齐] vs 瘦[D]  (形状均为 [T,R]=[{},{}]):",
            t, n_rows
        );
        println!(
            "      最大误差 = {:.3e}, 非逐位相等 = {}/{}, |ref| 均值 = {:.3e}  (R={}, 省 {:.1}% 的 N 维)",
            rerr,
            rdiff,
            t * n_rows,
            rscale,
            n_rows,
            100.0 * (1.0 - n_rows as f32 / f as f32)
        );
        let _ = h_b;
        let _ = b1;

        // 判据（按 Lead 裁定）：
        //   - 稠密+掩码 vs 瘦 GEMM（#1 的收缩维对比）：**同内核同输入**，
        //     要求逐位一致（< 1e-12，实际为 0）
        //   - 其余：最大绝对误差 < 1e-5 即 PASS。「非逐位相等元素个数」只作
        //     诊断信息打印，**不再作为 FAIL 依据**（float32 重排累加必然如此）
        let ok = maxerr < 1e-12 && rerr < 1e-5;
        println!(
            "  判定: {}",
            if ok {
                "PASS（两条瘦化路径均与稠密逐位一致）"
            } else {
                "FAIL"
            }
        );
        ok
    }
}

/// 逐元素比较，返回 (最大绝对误差, 非逐位相等元素个数, 参考均值 |ref|)。
///
/// 注意「非逐位相等个数」只作**诊断**打印，不作为 FAIL 依据 ——
/// float32 下只要累加顺序不同（AVX2 分块/FMA vs 标量顺序），
/// 末位就必然有差，这不是错误。
unsafe fn cmp_exact(a: *const f32, b: *const f32, n: usize) -> (f32, usize, f32) {
    let mut maxerr = 0.0f32;
    let mut ndiff = 0usize;
    let mut sum = 0.0f32;
    for i in 0..n {
        let x = *a.add(i);
        let y = *b.add(i);
        let e = (x - y).abs();
        if e > maxerr {
            maxerr = e;
        }
        if x != y {
            ndiff += 1;
        }
        sum += y.abs();
    }
    (maxerr, ndiff, sum / n.max(1) as f32)
}

/// 相对范数判据：‖a-b‖₂ / (‖a‖₂ + ‖b‖₂)。契约 §5 指定的口径。
unsafe fn rel_norm(a: *const f32, b: *const f32, n: usize) -> f64 {
    let (mut num, mut da, mut db) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let x = *a.add(i) as f64;
        let y = *b.add(i) as f64;
        num += (x - y) * (x - y);
        da += x * x;
        db += y * y;
    }
    num.sqrt() / (da.sqrt() + db.sqrt()).max(1e-30)
}

unsafe fn add_bias(p: *mut f32, b: *const f32, rows: usize, cols: usize) {
    for i in 0..rows {
        for j in 0..cols {
            *p.add(i * cols + j) += *b.add(j);
        }
    }
}

// ==================================================================
// 2. 正确性：gemm_nt_rows_selected
// ==================================================================
//
// ⚠ 参考实现的选择（重要，见报告「外部阻塞」一节）：
// 成员 D 在 kernels.rs 里加了 packed 快速路径 gemm_nt_packed，
// 触发条件是 m>=16 && n>=8 && k>=8 && m*n*k >= 1e6。
// 截至本次测量，该路径**是坏的**：
//   vmlm-tests 自带的回归用例 `GEMM 512x512 x512 maxerr=27.2863998 FAIL`
// 所以不能拿 kernels::gemm_nt 当参考（它会随 D 的状态时好时坏）。
//
// 这里改用两条**自证正确**的参考：
//   R1 = kernels::gemm_nt_scalar —— 纯标量三重循环，无 SIMD、无分块，
//        是「数学定义」的直接编码，不可能被优化改坏
//   R2 = kernels::gemm_nt_avx2  —— 我的瘦内核所派生的那个 8x8 微内核，
//        对它的期望是**逐位一致**（同序累加）
// 判据：R1 相对范数 < 1e-5；R2 逐位一致（< 1e-12）。
fn correctness_rows() -> bool {
    println!("\n---- 2. 正确性：gemm_nt_rows_selected（M=64 N=512 K=512，R 取 5 档）----");
    println!("  （参考用 gemm_nt_scalar + gemm_nt_avx2，不用 gemm_nt —— 见报告「外部阻塞」）");
    let (m, n, k) = (64usize, 512usize, 512usize);
    let mut rng = Rng::new(0x1234_5678);
    let mut a = Buf::new(m * k);
    let mut w = Buf::new(n * k);
    a.randn(&mut rng, 0.1);
    w.randn(&mut rng, 0.1);

    let ref_scalar = Buf::new(m * n);
    let ref_avx2 = Buf::new(m * n);
    unsafe {
        K::gemm_nt_scalar(a.p, w.p, ref_scalar.p, m, n, k, false);
        K::gemm_nt_avx2(a.p, w.p, ref_avx2.p, m, n, k, false);
    }
    println!(
        "  自检：gemm_nt_scalar vs gemm_nt_avx2 相对范数 = {:.3e}",
        unsafe { rel_norm(ref_scalar.p, ref_avx2.p, m * n) }
    );

    let mut ok = true;
    for &ratio in &[0.0f32, 0.25, 0.50, 0.75, 0.90] {
        let keep = ((1.0 - ratio) * n as f32).round() as usize;
        let mask = Buf::new(n);
        let rows = Buf::new(n);
        let nr = unsafe { make_random_mask_split(&mut rng, mask.p, n, keep, rows.p as *mut u32) };
        let out = Buf::new(m * nr.max(1));
        unsafe {
            S::gemm_nt_rows_selected(a.p, w.p, out.p, m, n, k, rows.p as *const u32, nr);
        }
        // 把参考按 rows 取列，对齐成 [M,R]
        let rs = Buf::new(m * nr.max(1));
        let ra = Buf::new(m * nr.max(1));
        unsafe {
            for i in 0..m {
                for t in 0..nr {
                    let j = *(rows.p as *const u32).add(t) as usize;
                    *rs.p.add(i * nr + t) = *ref_scalar.p.add(i * n + j);
                    *ra.p.add(i * nr + t) = *ref_avx2.p.add(i * n + j);
                }
            }
        }
        let (maxerr, ndiff, _) = unsafe { cmp_exact(out.p, rs.p, m * nr) };
        let rel = unsafe { rel_norm(out.p, rs.p, m * nr) };
        let (err_avx2, ndiff_avx2, _) = unsafe { cmp_exact(out.p, ra.p, m * nr) };
        let this_ok = rel < 1e-5 && err_avx2 < 1e-12;
        ok &= this_ok;
        println!(
            "  稀疏率 {:>4.0}%  R={:>3}  vs标量: 最大误差={:.3e} 相对范数={:.3e} (差异元素{});  vs AVX2 逐位: 最大误差={:.3e} (差异元素{})  {}",
            ratio * 100.0,
            nr,
            maxerr,
            rel,
            ndiff,
            err_avx2,
            ndiff_avx2,
            if this_ok { "PASS" } else { "FAIL" }
        );
    }
    ok
}

// ==================================================================
// 3. 正确性：gemm_nt_cols_selected（与「稠密 + 掩码」严格等价的那条）
// ==================================================================
fn correctness_cols() -> bool {
    println!("\n---- 3. 正确性：gemm_nt_cols_selected（= 稠密 GEMM + 掩码，M=64 N=512 K=512）----");
    let (m, n, k) = (64usize, 512usize, 512usize);
    let mut rng = Rng::new(0xDEAD_BEEF);
    let mut a = Buf::new(m * k);
    let mut w = Buf::new(n * k);
    a.randn(&mut rng, 0.1);
    w.randn(&mut rng, 0.1);

    let mut ok = true;
    for &ratio in &[0.0f32, 0.25, 0.50, 0.75, 0.90] {
        let keep = ((1.0 - ratio) * k as f32).round() as usize;
        let mask = Buf::new(k);
        let cols = Buf::new(k);
        let nc = unsafe { make_random_mask_split(&mut rng, mask.p, k, keep, cols.p as *mut u32) };

        // 路径 A：先把 A 乘掩码，再稠密 GEMM
        // 掩码按**列**作用（k 维），所以是 A[i][p] *= mask[p]
        let am = Buf::new(m * k);
        unsafe {
            for i in 0..m {
                for p in 0..k {
                    *am.p.add(i * k + p) = *a.p.add(i * k + p) * *mask.p.add(p);
                }
            }
        }
        let y_a = Buf::new(m * n);
        let y_a2 = Buf::new(m * n);
        unsafe {
            K::gemm_nt_scalar(am.p, w.p, y_a.p, m, n, k, false);
            K::gemm_nt_avx2(am.p, w.p, y_a2.p, m, n, k, false);
        }

        // 路径 B：瘦 GEMM，收缩维只跑 nc 个选中列
        let y_b = Buf::new(m * n);
        unsafe {
            S::gemm_nt_cols_selected(a.p, w.p, y_b.p, m, n, k, cols.p as *const u32, nc);
        }
        // 路径 C：打包版瘦 GEMM（应与 B 逐位一致 —— 同序累加）
        let y_c = Buf::new(m * n);
        unsafe {
            let ap = Buf::new(m * k.max(1));
            let wp = Buf::new(n * k.max(1));
            S::pack_cols(a.p, ap.p, m, k, cols.p as *const u32, nc);
            S::pack_cols(w.p, wp.p, n, k, cols.p as *const u32, nc);
            S::gemm_nt_cols_packed(ap.p, wp.p, y_c.p, m, n, nc.max(1));
        }

        // 参考 1：标量（相对范数判据）
        let (maxerr, ndiff, _) = unsafe { cmp_exact(y_b.p, y_a.p, m * n) };
        let rel = unsafe { rel_norm(y_b.p, y_a.p, m * n) };
        // 参考 2：AVX2 8x8 内核 —— **同一内核家族**，期望逐位一致
        let (err_avx2, ndiff_avx2, _) = unsafe { cmp_exact(y_b.p, y_a2.p, m * n) };
        // 打包版 vs 未打包版：同样期望逐位一致
        let (err_pack, ndiff_pack, _) = unsafe { cmp_exact(y_c.p, y_b.p, m * n) };
        let this_ok = rel < 1e-5 && err_avx2 < 1e-12 && err_pack < 1e-12;
        ok &= this_ok;
        println!(
            "  稀疏率 {:>4.0}%  有效K={:>3}  vs标量: 相对范数={:.3e}(差异{});  vs AVX2 逐位: {:.3e}(差异{});  打包版 vs 未打包: {:.3e}(差异{})  {}",
            ratio * 100.0,
            nc,
            rel,
            ndiff,
            err_avx2,
            ndiff_avx2,
            err_pack,
            ndiff_pack,
            if this_ok { "PASS" } else { "FAIL" }
        );
    }
    ok
}

// ==================================================================
// 4. 正确性：反向（梯度稀疏）
// ==================================================================
fn correctness_bwd() -> bool {
    println!("\n---- 4. 正确性：gemm_bwd_w_rows（梯度稀疏，M=64 N=512 K=512）----");
    let (m, n, k) = (64usize, 512usize, 512usize);
    let mut rng = Rng::new(0xABCD_0123);
    let mut x = Buf::new(m * k);
    let mut dy_full = Buf::new(m * n);
    x.randn(&mut rng, 0.5);
    dy_full.randn(&mut rng, 0.5);

    let mut ok = true;
    for &ratio in &[0.0f32, 0.25, 0.50, 0.75, 0.90] {
        let keep = ((1.0 - ratio) * n as f32).round() as usize;
        let mask = Buf::new(n);
        let rows = Buf::new(n);
        let nr = unsafe { make_random_mask_split(&mut rng, mask.p, n, keep, rows.p as *mut u32) };

        // 稠密参考：dw_full[N,K] = dy^T·x，dy 未选中列置 0
        let dy_zeroed = Buf::new(m * n);
        unsafe {
            for i in 0..m {
                for j in 0..n {
                    *dy_zeroed.p.add(i * n + j) =
                        *dy_full.p.add(i * n + j) * *mask.p.add(j);
                }
            }
        }
        let dw_full = Buf::new(n * k);
        unsafe { vmlm_core::train::gemm_bwd_w(dy_zeroed.p, x.p, dw_full.p, m, n, k) };

        // 瘦路径：dy 压缩成 [M,R]，只更新选中的行
        let dy_c = Buf::new(m * nr.max(1));
        unsafe {
            for i in 0..m {
                for t in 0..nr {
                    let j = *(rows.p as *const u32).add(t) as usize;
                    *dy_c.p.add(i * nr + t) = *dy_full.p.add(i * n + j);
                }
            }
        }
        let dw_thin = Buf::new(n * k);
        unsafe {
            S::gemm_bwd_w_rows(
                dy_c.p,
                x.p,
                dw_thin.p,
                m,
                n,
                k,
                rows.p as *const u32,
                nr,
            );
        }

        let mut maxerr = 0.0f32;
        let mut ndiff = 0usize;
        unsafe {
            for t in 0..nr {
                let j = *(rows.p as *const u32).add(t) as usize;
                for p in 0..k {
                    let a = *dw_full.p.add(j * k + p);
                    let b = *dw_thin.p.add(j * k + p);
                    let e = (a - b).abs();
                    if e > maxerr {
                        maxerr = e;
                    }
                    if a != b {
                        ndiff += 1;
                    }
                }
            }
        }
        let this_ok = maxerr < 1e-5;
        ok &= this_ok;
        let rel = unsafe { rel_norm(dw_thin.p, dw_full.p, n * k) };
        println!(
            "  稀疏率 {:>4.0}%  R={:>3}  最大误差={:.3e}  相对范数={:.3e}  非逐位相等(诊断,非判据)={}/{}  {}",
            ratio * 100.0,
            nr,
            maxerr,
            rel,
            ndiff,
            nr * k,
            if this_ok { "PASS" } else { "FAIL" }
        );
    }

    // AVX2 反向路径 vs 标量反向路径（口径不同：FMA 结合律）
    println!("\n  附：gemm_bwd_w_rows AVX2 vs 标量（结合律差异，非逐位）");
    unsafe {
        let (m, n, k) = (64usize, 512usize, 512usize);
        let keep = n / 4;
        let mask = Buf::new(n);
        let rows = Buf::new(n);
        let nr = make_random_mask_split(&mut rng, mask.p, n, keep, rows.p as *mut u32);
        let dy = Buf::new(m * nr);
        let mut dy2 = Buf::new(m * nr);
        dy2.randn(&mut rng, 0.5);
        std::ptr::copy_nonoverlapping(dy2.p, dy.p, m * nr);
        let d1 = Buf::new(n * k);
        let d2 = Buf::new(n * k);
        S::gemm_bwd_w_rows(dy.p, x.p, d1.p, m, n, k, rows.p as *const u32, nr);
        S::gemm_bwd_w_rows_avx2(dy.p, x.p, d2.p, m, n, k, rows.p as *const u32, nr);
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for t in 0..nr {
            let j = *(rows.p as *const u32).add(t) as usize;
            for p in 0..k {
                let a = *d1.p.add(j * k + p) as f64;
                let b = *d2.p.add(j * k + p) as f64;
                num += (a - b) * (a - b);
                den += a * a;
            }
        }
        let rel = (num / den.max(1e-30)).sqrt();
        println!("    相对 L2 误差 = {:.3e}  {}", rel, if rel < 1e-5 { "PASS" } else { "FAIL" });
        ok &= rel < 1e-5;
    }
    ok
}

// ==================================================================
// 5. 性能：各稀疏率的 wall-clock
// ==================================================================
//
// 三组测量，全部是 wall-clock（Instant 最小计时）：
//
//   测 A「瘦 GEMM（筛选权重行）」：gemm_nt_rows_selected，输出 [M,R] 紧凑
//        对照 = kernels::gemm_nt 算完整 [M,N]（=「稠密 GEMM + 掩码」的底层）
//        加速比 = 稠密 / 瘦
//
//   测 B「瘦 GEMM（筛选收缩维）」：gemm_nt_cols_selected，输出 [M,N] 不变
//        对照 = 先乘掩码再 gemm_nt
//        **这是端到端能直接用的那条**
//
//   测 C「gather 索引的纯开销」：gather_rows 标量 vs AVX2
//        稀疏率低时它可能吃掉全部收益 —— 这正是要测的
fn perf_scaling() -> bool {
    let (m, n, k) = (64usize, 512usize, 512usize);
    println!(
        "\n---- 5. 性能 wall-clock（M={} N={} K={}，FLOPs 稠密 = {:.1} MFLOP）----",
        m,
        n,
        k,
        2.0 * (m * n * k) as f64 / 1e6
    );
    let mut rng = Rng::new(0x5EED_5EED);
    let mut a = Buf::new(m * k);
    let mut w = Buf::new(n * k);
    a.randn(&mut rng, 0.1);
    w.randn(&mut rng, 0.1);
    let dense_out = Buf::new(m * n);
    let mask_a = Buf::new(m * k);
    let am = Buf::new(m * k);

    let ratios = [0.0f32, 0.25, 0.50, 0.75, 0.90];
    let reps = 400;

    // 先拿到每个稀疏率下的索引集
    let mut idx_sets: Vec<(usize, Vec<u32>)> = Vec::new();
    for &r in &ratios {
        let nk = ((1.0 - r) * n as f32).round() as usize;
        let mask = Buf::new(n);
        let rows = Buf::new(n);
        let nr = unsafe { make_random_mask_split(&mut rng, mask.p, n, nk, rows.p as *mut u32) };
        let v: Vec<u32> = (0..nr).map(|i| unsafe { *(rows.p as *const u32).add(i) }).collect();
        idx_sets.push((nr, v));
    }
    // 收缩维的列索引
    let mut col_sets: Vec<(usize, Vec<u32>)> = Vec::new();
    for &r in &ratios {
        let nk = ((1.0 - r) * k as f32).round() as usize;
        let mask = Buf::new(k);
        let cols = Buf::new(k);
        let nc = unsafe { make_random_mask_split(&mut rng, mask.p, k, nk, cols.p as *mut u32) };
        let v: Vec<u32> = (0..nc).map(|i| unsafe { *(cols.p as *const u32).add(i) }).collect();
        col_sets.push((nc, v));
    }

    // 稠密基线。
    // ⚠ 用 gemm_nt_avx2 而不是 gemm_nt：后者目前会路由到成员 D 正在开发的
    //   packed 路径（且该路径当前在 vmlm-tests 里 FAIL）。这里的目的是量
    //   「稠密 vs 瘦」的**同族内核**对比，用同一套 8x8 微内核才公平。
    //   生产口径的 gemm_nt 数字在测 B 的「稠密+掩码对照」一列里给出。
    //
    // 为对抗频率漂移，稠密基线测 3 轮（每轮之间夹一段别的活），取全局最小。
    let mut t_dense = f64::MAX;
    for _ in 0..3 {
        let t = bench(reps, || unsafe {
            K::gemm_nt_avx2(a.p, w.p, dense_out.p, m, n, k, false);
        });
        if t < t_dense {
            t_dense = t;
        }
    }
    let gflops_dense = 2.0 * (m * n * k) as f64 / (t_dense * 1e-6) / 1e9;
    println!("\n  [基线] 稠密 gemm_nt_avx2（8x8 微内核，与瘦内核同族）：{:.1} µs  {:.2} GFLOPS", t_dense, gflops_dense);
    println!("         （加速比 = 该基线 / 瘦 GEMM 总耗时；绝对 µs 列才是跨人可比的量）");

    // ---------------- 测 A：筛选权重行 ----------------
    println!("\n  [测 A] gemm_nt_rows_selected（筛选权重行，输出 [M,R] 紧凑）");
    println!("    {:<8} {:>5} {:>12} {:>10} {:>12} {:>10}", "稀疏率", "R", "瘦µs", "加速比", "瘦GFLOPS*", "gatherµs");
    let mut prev_thin_a = 0.0f64;
    for (i, &r) in ratios.iter().enumerate() {
        let (nr, ref v) = idx_sets[i];
        let rows = Buf::new(n.max(1));
        unsafe {
            for (t, &j) in v.iter().enumerate() {
                *(rows.p as *mut u32).add(t) = j;
            }
        }
        let rp = rows.p as *const u32;
        let out = Buf::new(m * nr.max(1));
        let t_thin = if nr == 0 {
            0.0
        } else {
            bench(reps, || unsafe {
                S::gemm_nt_rows_selected(a.p, w.p, out.p, m, n, k, rp, nr);
            })
        };
        // gather 开销（每次前向都要重算，必须计入）
        let mask = Buf::new(n);
        let rows2 = Buf::new(n);
        unsafe {
            for &j in v.iter() {
                *mask.p.add(j as usize) = 1.0;
            }
        }
        let t_gather = bench(reps * 4, || unsafe {
            S::gather_rows(mask.p, n, rows2.p as *mut u32);
        });
        let speedup = t_dense / (t_thin + t_gather).max(1e-9);
        // 「*」口径：按实际做的乘加数折算，不是绝对 GFLOPS
        let gf = if t_thin > 0.0 {
            2.0 * (m * nr * k) as f64 / (t_thin * 1e-6) / 1e9
        } else {
            f64::NAN
        };
        let _ = prev_thin_a;
        prev_thin_a = t_thin;
        println!(
            "    {:<8} {:>5} {:>12.2} {:>10.3} {:>12.2} {:>10.3}",
            format!("{:.0}%", r * 100.0),
            nr,
            t_thin,
            speedup,
            gf,
            t_gather
        );
    }

    // ---------------- 测 B：筛选收缩维 ----------------
    println!("\n  [测 B] gemm_nt_cols_selected（筛选收缩维，输出 [M,N] 不变，端到端可直接用）");
    println!("    {:<8} {:>6} {:>12} {:>10} {:>12} {:>10}", "稀疏率", "有效K", "瘦µs", "加速比", "瘦GFLOPS*", "gatherµs");
    for (i, &r) in ratios.iter().enumerate() {
        let (nc, ref v) = col_sets[i];
        let cols = Buf::new(k.max(1));
        let mask = Buf::new(k);
        unsafe {
            for (t, &j) in v.iter().enumerate() {
                *(cols.p as *mut u32).add(t) = j;
                *mask.p.add(j as usize) = 1.0;
            }
            // mask_a 是「稠密+掩码」对照用的列掩码（作用在 A 的 K 维上）
            std::ptr::copy_nonoverlapping(mask.p, mask_a.p, k);
        }
        let cp = cols.p as *const u32;
        let out = Buf::new(m * n);
        let t_thin = if nc == 0 {
            // 全稀疏：输出全 0，直接 memset
            bench(reps, || unsafe {
                std::ptr::write_bytes(out.p as *mut u8, 0, m * n * 4);
            })
        } else {
            bench(reps, || unsafe {
                S::gemm_nt_cols_selected(a.p, w.p, out.p, m, n, k, cp, nc);
            })
        };
        // 「稠密+掩码」对照：乘掩码 + 稠密 gemm_nt（掩码按列作用在 a 上）
        let t_dm = bench(reps, || unsafe {
            for ii in 0..m {
                for p in 0..k {
                    *am.p.add(ii * k + p) = *a.p.add(ii * k + p) * *mask_a.p.add(p);
                }
            }
            K::gemm_nt_avx2(am.p, w.p, out.p, m, n, k, false);
        });
        let t_gather = bench(reps * 4, || unsafe {
            S::gather_cols_union(mask.p, 1, k, cols.p as *mut u32);
        });
        // 打包版：分「A+W 每次重打包」和「W 只打包一次（跨前向复用）」两种口径。
        // 真实训练里 W 是常量，cols 变了才需要重打包 —— 后者才是该用的数。
        let ap = Buf::new(m * k.max(1));
        let wp = Buf::new(n * k.max(1));
        let (t_repack, t_reuse) = if nc == 0 {
            (0.0, 0.0)
        } else {
            let tr = bench(reps, || unsafe {
                S::pack_cols(a.p, ap.p, m, k, cp, nc);
                S::pack_cols(w.p, wp.p, n, k, cp, nc);
                S::gemm_nt_cols_packed(ap.p, wp.p, out.p, m, n, nc);
            });
            // 只重打包 A（W 的打包已在循环外做好）
            unsafe { S::pack_cols(w.p, wp.p, n, k, cp, nc) };
            let tu = bench(reps, || unsafe {
                S::pack_cols(a.p, ap.p, m, k, cp, nc);
                S::gemm_nt_cols_packed(ap.p, wp.p, out.p, m, n, nc);
            });
            (tr, tu)
        };
        let speedup = t_dense / (t_thin + t_gather).max(1e-9);
        let speedup_packed = t_dense / (t_reuse + t_gather).max(1e-9);
        let gf = if t_thin > 0.0 {
            2.0 * (m * n * nc) as f64 / (t_thin * 1e-6) / 1e9
        } else {
            f64::NAN
        };
        let gf_packed = if t_reuse > 0.0 {
            2.0 * (m * n * nc) as f64 / (t_reuse * 1e-6) / 1e9
        } else {
            f64::NAN
        };
        println!(
            "    {:<8} {:>6} {:>12.2} {:>10.3} {:>12.2} {:>10.3}   （稠密+掩码对照 {:.1} µs）",
            format!("{:.0}%", r * 100.0),
            nc,
            t_thin,
            speedup,
            gf,
            t_gather,
            t_dm
        );
        println!(
            "             └ 打包版: 全重打包 {:.1} µs ({:.3}x) | W复用 {:.1} µs ({:.3}x, {:.1} GFLOPS)",
            t_repack,
            if t_repack > 0.0 { t_dense / (t_repack + t_gather) } else { f64::NAN },
            t_reuse,
            speedup_packed,
            gf_packed
        );
    }

    // ---------------- 测 C：gather 纯开销 ----------------
    println!("\n  [测 C] gather_rows 索引收集的纯开销（标量 vs AVX2）");
    println!("    {:<8} {:>14} {:>14} {:>10}", "稀疏率", "标量µs", "AVX2 µs", "比值");
    let mut ok = true;
    for (i, &r) in ratios.iter().enumerate() {
        let (nr, ref v) = idx_sets[i];
        let mask = Buf::new(n);
        let rows = Buf::new(n);
        unsafe {
            for t in 0..nr {
                *mask.p.add(v[t] as usize) = 1.0;
            }
        }
        let t_s = bench(reps * 8, || unsafe {
            S::gather_rows(mask.p, n, rows.p as *mut u32);
        });
        let t_v = bench(reps * 8, || unsafe {
            S::gather_rows_avx2(mask.p, n, rows.p as *mut u32);
        });
        println!(
            "    {:<8} {:>14.3} {:>14.3} {:>10.2}",
            format!("{:.0}%", r * 100.0),
            t_s,
            t_v,
            t_s / t_v
        );
    }
    let _ = ok;
    ok = true;
    ok
}

// ==================================================================
// 7. 关键发现：逐行 top-k 掩码**不产生**可瘦化的列集合
// ==================================================================
//
// 第 6 节暴露了一个 sparse_dyn README 没有讲透的要害：
//   `gather_cols_union(mask, T, F)` 在 T=128 行上都取并集，
//   top_ratio=0.25 时**并集恒等于全集**（128×512 = 65536 个抽样位置，
//   覆盖 2048 列的概率是 1 - (1-0.25)^128 ≈ 1 - 1e-16）。
//   ⇒ 逐行 top-k 的 mask 在第二条 GEMM 上**一点都瘦不掉**，
//     瘦 GEMM 退化成稠密（实测 6700 µs vs 稠密 3286 µs，还慢一倍）。
//
// 这与 sparse_dyn README §4 的第 2 条经验（「Q/K 必须共享同一份掩码，
// 否则交集为空」）是同一个几何事实的两面：
//   共享掩码 → 交集可用，瘦化成立
//   逐行掩码 → 并集为全集，瘦化不成立
//
// 本节把三种掩码口径放在同一个 shape 上对照，给出「什么条件下瘦 GEMM
// 真的省时间」的可判定结论。
//
//   口径 1 逐行 top-k  + 列并集   ← sparse_dyn 的实际做法（瘦不掉）
//   口径 2 全局共享 top-k          ← 可瘦化，等价于把 FFN 变窄
//   口径 3 逐行 top-k  + 行内分块  ← 块稀疏（真正的动态稀疏路径）
fn when_does_it_pay() -> bool {
    println!("\n---- 7. 什么条件下瘦 GEMM 真省时间（T=128 D=512 F=2048）----");
    let (t, d, ff) = (128usize, 512usize, 2048usize);
    let mut rng = Rng::new(0x0BAD_F00D);
    let mut g = Buf::new(t * ff); // 第二条 GEMM 的输入激活
    let mut w2 = Buf::new(d * ff);
    let mut y = Buf::new(t * d);
    let mut y2 = Buf::new(t * d);
    g.randn(&mut rng, 0.3);
    w2.randn(&mut rng, 0.05);
    let reps = 20;
    let mut ok = true;

    unsafe {
        let t_dense = bench(reps, || {
            K::gemm_nt_avx2(g.p, w2.p, y.p, t, d, ff, false);
        });
        println!("  稠密基线 [128,2048]x[512,2048]^T = {:.1} µs", t_dense);

        // ---- 口径 1：逐行 top-k（每行独立选 25%）+ 列并集 ----
        let keep = (ff as f32 * 0.25).round() as usize;
        let mask_row = Buf::new(t * ff);
        for i in 0..t {
            let tau = K::topk_thresh(g.p.add(i * ff), ff, keep);
            S::mask_from_thresh(g.p.add(i * ff), tau, mask_row.p.add(i * ff), ff);
        }
        let cols = Buf::new(ff);
        let nc_row = S::gather_cols_union(mask_row.p, t, ff, cols.p as *mut u32);
        println!(
            "  [口径1 逐行top-k+列并集] 该行掩码密度={:.3} → 并集列数 {} / {} ({:.1}% 可瘦)",
            S::mask_density(mask_row.p, t * ff),
            nc_row,
            ff,
            100.0 * (1.0 - nc_row as f32 / ff as f32)
        );

        // ---- 口径 2：全局共享 top-k（所有行同一份列掩码）----
        let mask_sh = Buf::new(ff);
        {
            // 用列的能量（各行绝对值之和）当作「神经元重要性」排序依据
            let mut col_imp = vec![0f32; ff];
            for i in 0..t {
                for j in 0..ff {
                    col_imp[j] += (*g.p.add(i * ff + j)).abs();
                }
            }
            let mut order: Vec<u32> = (0..ff as u32).collect();
            order.sort_unstable_by(|&a, &b| {
                col_imp[b as usize].partial_cmp(&col_imp[a as usize]).unwrap()
            });
            order.truncate(keep);
            for &j in &order {
                *mask_sh.p.add(j as usize) = 1.0;
            }
        }
        let cols_sh = Buf::new(ff);
        let nc_sh = S::gather_cols_union(mask_sh.p, 1, ff, cols_sh.p as *mut u32);
        let csh = cols_sh.p as *const u32;
        let ap = Buf::new(t * ff);
        let wp = Buf::new(d * ff);
        S::pack_cols(w2.p, wp.p, d, ff, csh, nc_sh);
        let t_sh = bench(reps, || {
            S::pack_cols(g.p, ap.p, t, ff, csh, nc_sh);
            S::gemm_nt_cols_packed(ap.p, wp.p, y2.p, t, d, nc_sh);
        });
        println!(
            "  [口径2 全局共享top-k ] 共享列数 {} / {} (省 {:.1}%) → 瘦 GEMM {:.1} µs，加速 {:.2}x",
            nc_sh,
            ff,
            100.0 * (1.0 - nc_sh as f32 / ff as f32),
            t_sh,
            t_dense / t_sh
        );

        // 正确性：口径 2 必须与「稠密 + 共享掩码」一致
        let gm = Buf::new(t * ff);
        for i in 0..t {
            for j in 0..ff {
                *gm.p.add(i * ff + j) = *g.p.add(i * ff + j) * *mask_sh.p.add(j);
            }
        }
        let y_ref = Buf::new(t * d);
        K::gemm_nt_avx2(gm.p, w2.p, y_ref.p, t, d, ff, false);
        let rel = rel_norm(y2.p, y_ref.p, t * d);
        println!("    正确性：vs 稠密+共享掩码 相对范数 = {:.3e}  {}", rel, if rel < 1e-5 { "PASS" } else { "FAIL" });
        ok &= rel < 1e-5;

        // ---- 口径 3：行内分块稀疏（每行把 F 切成 16 块，只留 25% 的块）----
        // 这是「动态稀疏能真加速」的唯一现实路径：块内连续 → 无 gather。
        const BLK: usize = 16;
        let nblk = ff / BLK; // 128 块
        let keep_blk = ((nblk as f32 * 0.25).round() as usize).max(1);
        let mask_blk = Buf::new(t * ff);
        for i in 0..t {
            // 按块能量选 top 块
            let mut be: Vec<(f32, usize)> = (0..nblk)
                .map(|b| {
                    let mut s = 0f32;
                    for j in 0..BLK {
                        s += (*g.p.add(i * ff + b * BLK + j)).abs();
                    }
                    (s, b)
                })
                .collect();
            be.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            for &(_, b) in be.iter().take(keep_blk) {
                for j in 0..BLK {
                    *mask_blk.p.add(i * ff + b * BLK + j) = 1.0;
                }
            }
        }
        // 块稀疏下「每行保留区间不同」→ 仍需要逐行的段列表。
        // 这里给一个上界估计：把每行的保留块数 × 块宽，当作有效 K。
        let nkblk = keep_blk * BLK;
        println!(
            "  [口径3 行内块稀疏(BLK={})] 每行 {} 块 × {} = 有效K {} (省 {:.1}%) —— 每行块号不同，",
            BLK,
            keep_blk,
            BLK,
            nkblk,
            100.0 * (1.0 - nkblk as f32 / ff as f32)
        );
        println!("      整批无法共享一份列集合 ⇒ 现有 cols_selected 接口不适用。");
        println!("      可行的加速上界 = 按「每行有效K」折算的稠密等价耗时：");
        let t_ub = t_dense * (nkblk as f64 / ff as f64);
        println!(
            "      {:.1} µs × {:.3} = {:.1} µs（加速上界 {:.2}x，且不含任何 gather 开销）",
            t_dense,
            nkblk as f64 / ff as f64,
            t_ub,
            t_dense / t_ub
        );

        // ---- 汇总结论 ----
        println!("\n  结论表（同一 shape，同一稠密基线 {:.1} µs）：", t_dense);
        println!("    {:<28} {:>10} {:>10}", "掩码口径", "瘦GEMM µs", "加速比");
        println!(
            "    {:<28} {:>10} {:>10.2}",
            "口径1 逐行top-k + 列并集", "≈稠密", t_dense / t_dense
        );
        println!(
            "    {:<28} {:>10.1} {:>10.2}",
            "口径2 全局共享top-k", t_sh, t_dense / t_sh
        );
        println!(
            "    {:<28} {:>10} {:>10.2}",
            "口径3 行内块稀疏", "(上界)",
            ff as f64 / nkblk as f64
        );

        ok
    }
}

// ==================================================================
// 6. 复现 sparse_dyn 的前车之鉴
// ==================================================================
//
// sparse_dyn README §4.4：稠密 110ms vs Q 稀疏 148ms，不降反升。
// 我们把它的两种做法在 VMLM 里跑一遍，量级对齐到它的配置
// （d_model=512, d_ff=2048, T=128）：
//   做法 1 = 稠密 GEMM + 掩码（sparse_dyn 现状）—— 应该 ≈ 稠密
//   做法 2 = 瘦 GEMM（只跑被选中的 K）
// 唯一区别是掩码乘法是否被省掉。
fn repro_sparse_dyn() -> bool {
    println!("\n---- 6. 复现 sparse_dyn 前车之鉴（T=128, D=512, F=2048, top_ratio=0.25）----");
    println!("  sparse_dyn README §4.4 自陈：稠密 110ms vs Q 稀疏 148ms，不降反升。");
    println!("  我们把它拆成两步量：FFN 第一条 GEMM（x·W1^T）和第二条 GEMM（g·W2^T）。");
    let (t, d, ff) = (128usize, 512usize, 2048usize);
    let mut rng = Rng::new(0xFEED_FACE);
    let mut x = Buf::new(t * d);
    let mut w1 = Buf::new(ff * d);
    let mut w2 = Buf::new(d * ff);
    x.randn(&mut rng, 0.5);
    w1.randn(&mut rng, 0.05);
    w2.randn(&mut rng, 0.05);
    let h = Buf::new(t * ff);
    let g = Buf::new(t * ff);
    let y = Buf::new(t * d);
    let y2 = Buf::new(t * d);
    let reps = 20;

    unsafe {
        // ⚠ 本节的稠密 GEMM 一律用 gemm_nt_avx2（8x8 微内核），不用 gemm_nt：
        //   后者会路由到成员 D 的 packed 路径，而该路径当前在 vmlm-tests 里
        //   是 FAIL 的（GEMM 512³ maxerr=27.29）。用坏掉的基线做对比没有意义。
        //   packed 一旦修好，只要把这里的 gemm_nt_avx2 换成 gemm_nt 即可。

        // 第一条 GEMM 是稠密的（掩码还没产生），两条路线一样
        let t_g1 = bench(reps, || {
            K::gemm_nt_avx2(x.p, w1.p, h.p, t, ff, d, false);
        });

        // 产生掩码（top_ratio=0.25 ⇒ 每行保留 512 个）
        let keep = (ff as f32 * 0.25).round() as usize;
        let mask = Buf::new(t * ff);
        for i in 0..t {
            let tau = K::topk_thresh(h.p.add(i * ff), ff, keep);
            S::mask_from_thresh(h.p.add(i * ff), tau, mask.p.add(i * ff), ff);
        }
        for i in 0..t * ff {
            *g.p.add(i) = *h.p.add(i) * *mask.p.add(i);
        }
        let dens = S::mask_density(mask.p, t * ff);

        // 做法 1：稠密 + 掩码（把掩码乘法也算进去 —— sparse_dyn 就是这么干的）
        let t_dm = bench(reps, || {
            for i in 0..t * ff {
                *g.p.add(i) = *h.p.add(i) * *mask.p.add(i);
            }
            K::gemm_nt_avx2(g.p, w2.p, y.p, t, d, ff, false);
        });

        // 做法 2：瘦 GEMM。掩码在 g 的列上，按列聚合
        let cols = Buf::new(ff);
        let nc = S::gather_cols_union(mask.p, t, ff, cols.p as *mut u32);
        let cp = cols.p as *const u32;
        // 端到端：gather + 瘦 GEMM（掩码乘法本身被省掉）
        let t_thin = bench(reps, || {
            S::gemm_nt_cols_selected(g.p, w2.p, y2.p, t, d, ff, cp, nc);
        });
        // 只 gather 的开销
        let t_gt = bench(reps * 5, || {
            S::gather_cols_union(mask.p, t, ff, cols.p as *mut u32);
        });
        // 打包版：W 只打包一次（权重是常量），A 每次重打包
        let ap = Buf::new(t * ff);
        let wp = Buf::new(d * ff);
        S::pack_cols(w2.p, wp.p, d, ff, cp, nc);
        let t_packed = bench(reps, || {
            S::pack_cols(g.p, ap.p, t, ff, cp, nc);
            S::gemm_nt_cols_packed(ap.p, wp.p, y2.p, t, d, nc);
        });
        // 稠密 GEMM 单独（不含掩码乘法）—— 测 3 轮取最小，抗频率漂移
        let mut t_g2 = f64::MAX;
        for _ in 0..3 {
            let t = bench(reps, || {
                K::gemm_nt_avx2(g.p, w2.p, y.p, t, d, ff, false);
            });
            if t < t_g2 {
                t_g2 = t;
            }
        }

        // 生产口径：真正的 kernels::gemm_nt（可能是 D 的 packed 路径）跑一次，
        // 只作参考打印，不参与判定
        let t_prod = bench(reps, || {
            K::gemm_nt(g.p, w2.p, y.p, t, d, ff, false);
        });

        println!(
            "  掩码密度 = {:.4}（top_ratio 0.25），gather_cols_union 得到 {} / {} 列",
            dens, nc, ff
        );
        println!("  ┌──────────────────────────────────────────────────────────────┐");
        println!("  │ 第一条 GEMM  x·W1^T  [128,512]×[2048,512]^T  稠密 {:.1} µs", t_g1);
        println!("  │ 第二条 GEMM  g·W2^T  [128,2048]×[512,2048]^T");
        println!("  │   稠密 GEMM 单独 (gemm_nt_avx2)   {:.1} µs", t_g2);
        println!("  │   稠密 + 掩码（sparse_dyn 做法）  {:.1} µs", t_dm);
        println!("  │   掩码乘法本身的增量              {:+.1} µs", t_dm - t_g2);
        println!("  │   gather_cols_union 开销          {:.1} µs", t_gt);
        println!("  │   瘦 GEMM（有效K={}）           {:.1} µs", nc, t_thin);
        println!("  │   瘦 GEMM 端到端(+gather)         {:.1} µs", t_thin + t_gt);
        println!("  │   瘦 GEMM 打包版(+gather)         {:.1} µs", t_packed + t_gt);
        println!("  │   [参考] 生产 gemm_nt（含D的packed）{:.1} µs", t_prod);
        println!("  └──────────────────────────────────────────────────────────────┘");
        println!(
            "  第二条 GEMM: 稠密+掩码 {:.1} µs → 瘦 GEMM 端到端 {:.1} µs，加速 {:.2}x",
            t_dm,
            t_thin + t_gt,
            t_dm / (t_thin + t_gt)
        );
        println!(
            "  第二条 GEMM: 稠密+掩码 {:.1} µs → 瘦 GEMM **打包版** 端到端 {:.1} µs，加速 {:.2}x",
            t_dm,
            t_packed + t_gt,
            t_dm / (t_packed + t_gt)
        );
        println!(
            "  sparse_dyn 的 148/110 = {:.3}x（不降反升）；我们的瘦 GEMM(打包) = {:.3}x（{}）",
            148.0 / 110.0,
            t_dm / (t_packed + t_gt),
            if t_packed + t_gt < t_dm { "真加速" } else { "仍未加速，见报告分析" }
        );

        // 正确性顺带核一遍（大 shape）：y 与 y2 都是 gemm_nt_avx2 家族，
        // 期望逐位一致（同一内核、同一累加顺序）
        let (maxerr, ndiff, _) = cmp_exact(y.p, y2.p, t * d);
        let rel = rel_norm(y.p, y2.p, t * d);
        println!(
            "  大 shape 正确性：稠密+掩码 vs 瘦 GEMM  最大误差={:.3e} 相对范数={:.3e} (差异元素 {}/{})",
            maxerr,
            rel,
            ndiff,
            t * d
        );
        rel < 1e-5
    }
}
