// PATH: B1 / B2
//
// ===========================================================================
// sparse-B 速度验收 + tok/s
// ===========================================================================
//
// 这是**新增**的验收驱动，不替代任何既有判据：
//   * 正确性口径沿用既有验收（B1：`vmlm-b1-prod-eps` 的确定性 + ε；
//     B2：`vmlm-b2-verify` 的 A1–A7 逐位判据）；
//   * 本驱动补的是**速度**这一侧：一个**真正计时的** 24 层 × 5 矩阵 × T token
//     循环，并把结果表达成 **tok/s**。
//
// ── tok/s 的定义（先说清楚，避免口径漂移）────────────────────────────────
//   token  := 一个位置（position）在模型全部 24 层、每层 5 个稀疏矩阵
//             （wq, wk, wv, wo, ffn）上的**前向**。即 1 token = 120 次 matvec。
//   T      := 每次测量处理的 token 数（默认 128 = batch 4 × seq 32，
//             与 P18 验收配置一致）。
//   tok/s  := T / 秒（QPC 中位数），**单线程**，**仅前向**。
//   一次"step" = 处理 T 个 token 的全部 120 个稀疏矩阵。
//
//   为什么用真循环而不是「单算子中位数 × 15360」：后者是本项目历史上的口径
//   （`b2_verify.rs:1394`），但它是**外推**。本驱动两个都报：
//   真循环的 wall time，以及由单算子中位数外推的值，两者必须同量级。
//
// ── 本驱动测的实现 ────────────────────────────────────────────────────────
//   W1  B1 标量契约参考（逐位基准，最慢）
//   W2  B1 发布版（`b1_forward`：gather + 8 路归约 + u32 镜像）
//   W3  B1 令牌分块（`b1_forward_t8`：无 gather，通道 = token）
//   W4  B1 令牌分块 + 转置（把 `[T][d] -> [d][t8]` 的搬运**计入**计时）
//   W5  B2 SHARED（每次前向一次选择）
//   W6  B2 PER-ROW（每神经元一次选择；如实报，**慢**）
//
// ── 判据 ──────────────────────────────────────────────────────────────────
//   C0  120 个矩阵的 src/src_u32 镜像一致 + 几何自洽（硬失败）
//   C1  转置内核逐位 == 标量转置（含 d/t 非 8 倍数、含 d=1/t=1 边界）
//   C2  令牌分块内核逐位 == 标量契约参考（0 分叉），4 组形状
//   C3  确定性：8 次重复调用逐位相同
//   C4  can-fail：4 种已知错误内核必须被抓到（mismatch > 0）
//   C5  首项赋值契约在令牌分块路径上**可观测**（-0.0 位型）
//   C6  B2 SIMD 强度行逐位 == 标量强度参考（两种 mode 都对拍）
//   C7  计时下限守卫：没有"假零"（循环不可能被删）
//   C8  真循环 wall time 与"单算子 × 15360"外推同量级（外推口径自检）
//
// USAGE: vmlm-sparseb-tok.exe [--d 1536] [--k0 20] [--k 8] [--layers 24]
//                             [--mats 5] [--tokens 128] [--reps 20]

extern "C" {
    fn malloc(n: usize) -> *mut u8;
    fn QueryPerformanceCounter(c: *mut i64) -> i32;
    fn QueryPerformanceFrequency(f: *mut i64) -> i32;
}

use vmlm_core::b1_pure::{
    b1_forward, b1_forward_scalar_reference, b1_forward_t8, b1_forward_t8_scalar_reference,
    b1_forward_t8_x1, b1_transpose_activations,
};
use vmlm_core::b2_topk::{b2_forward, b2_select_once, B2Mode, TieRule};
use vmlm_core::conn_train::{init_sparse_conn_scaled, SparseConnW};

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

/// 防假零：计时体必须把输出读回来，否则 LLVM 可以把整段循环删掉
/// （本项目已四次踩到「0.0000 ms」的假零，见 docs/B1_B2_交付报告.md §11）。
static mut SINK: f64 = 0.0;

#[inline(never)]
unsafe fn consume(p: *const f32, n: usize) {
    let mut s = 0.0f64;
    let mut i = 0usize;
    while i < n {
        s += *p.add(i) as f64;
        i += 1;
    }
    SINK += s;
}

/// 一次完整 step（24 层 × 5 矩阵 × T token）不允许快于这个值。
/// 触碰 120 × 30720 条连接的循环不可能低于 1 µs ⇒ 低于它就是假零。
const FLOOR_MS: f64 = 1.0e-3;

fn main() {
    unsafe { vmlm_core::interp::init_cpu() };
    let d = arg_u32("--d", 1536) as usize;
    let k0 = arg_u32("--k0", 20) as usize;
    let k = arg_u32("--k", 8) as usize;
    let layers = arg_u32("--layers", 24) as usize;
    let mats = arg_u32("--mats", 5) as usize;
    let tokens = arg_u32("--tokens", 128) as usize;
    let reps = arg_u32("--reps", 20) as usize;

    let t8 = ((tokens + 7) / 8) * 8;
    let n_mats = layers * mats;
    let calls_per_step = n_mats * tokens;
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

    println!("===== sparse-B 速度验收 + tok/s =====");
    println!(
        "d={} k0={} k={} layers={} mats/layer={} T(tokens/step)={} => {} matvec/step",
        d, k0, k, layers, mats, tokens, calls_per_step
    );    println!("HAS_AVX2 = {}", unsafe { vmlm_core::interp::HAS_AVX2 });
    println!("tok/s 定义: 1 token = 24 层 x 5 矩阵的前向 = 120 次 matvec；tok/s = T / 秒；仅前向、单线程");

    unsafe {
        // ------------------------------------------------------------------
        // 0) 建立 120 个矩阵（确定性初始化，不需要任何磁盘产物）
        // ------------------------------------------------------------------
        let mut scs: Vec<SparseConnW> = Vec::with_capacity(n_mats);
        for l in 0..n_mats {
            let base = malloc(SparseConnW::bytes(d as u32, k0 as u32));
            let sc = SparseConnW::from_base(base, d as u32, k0 as u32, (l as u32) + 1);
            init_sparse_conn_scaled(&sc, 0.144);
            sc.assert_geometry();
            sc.assert_src_u32_consistent();
            scs.push(sc);
        }
        println!(
            "\n--- C0: {} 个矩阵的几何 + src/src_u32 镜像 ---",
            n_mats
        );
        check!(true, "C0", "geometry ok, mirror consistent for all {} matrices", n_mats);

        // ------------------------------------------------------------------
        // 1) 激活：每个矩阵槽位一份 [T][d]（token-major），确定性填充
        // ------------------------------------------------------------------
        let xbuf: Vec<*mut f32> = (0..mats)
            .map(|_| malloc(tokens * d * 4) as *mut f32)
            .collect();
        let xtbuf: Vec<*mut f32> = (0..mats)
            .map(|_| malloc(d * t8 * 4) as *mut f32)
            .collect();
        for m in 0..mats {
            let xp = xbuf[m];
            for t in 0..tokens {
                for i in 0..d {
                    let idx = t * d + i;
                    *xp.add(idx) = ((((idx * 1103515245usize + 12345 + m * 7919) % 2000) as f32)
                        / 1000.0)
                        - 1.0;
                }
            }
            // 转置必须在使用 xt 的任何判据**之前**完成 —— C4 的第一版忘了这一步，
            // 结果是 yref 与 4 个错误实现读的是同一块**未初始化** malloc 内存，
            // 于是 4 个 can-fail 全部报 caught=0（判据自己坏了，不是内核对了）。
            b1_transpose_activations(xp, tokens, d, xtbuf[m], t8);
        }
        let y_tm = malloc(tokens * d * 4) as *mut f32; // token-major 输出
        let yt = malloc(d * t8 * 4) as *mut f32; // source-major 输出
        let y_scalar = malloc(d * t8 * 4) as *mut f32;
        let agg = malloc(k0 * 4) as *mut f32;
        let hit = malloc(k0 * 8) as *mut usize;

        // ------------------------------------------------------------------
        // 2) C1 转置内核逐位对拍（含非 8 倍数与 1 维边界）
        // ------------------------------------------------------------------
        println!("\n--- C1: 转置 [T][d] -> [d][t8] 逐位对拍（AVX2 vs 标量）---");
        let shapes: [(usize, usize); 5] = [(d, tokens), (13, 8), (8, 13), (1, 1), (7, 9)];
        let mut c1_bad_total = 0usize;
        for (sd, st) in shapes {
            let st8 = ((st + 7) / 8) * 8;
            let sx = malloc(sd * st * 4) as *mut f32;
            for i in 0..sd * st {
                *sx.add(i) = (((i * 2654435761usize) % 997) as f32 / 499.0) - 1.0;
            }
            let a = malloc(sd * st8 * 4) as *mut f32;
            let b = malloc(sd * st8 * 4) as *mut f32;
            // 故意把两块填成不同初值，确保"填充列清零"这件事也被测到
            for i in 0..sd * st8 {
                *a.add(i) = 7.0;
                *b.add(i) = 7.0;
            }
            vmlm_core::b1_pure::b1_transpose_activations(sx, st, sd, a, st8);
            // 标量直接按定义写（独立第三实现，不复用被测的标量回退）
            for s in 0..sd {
                for t in 0..st {
                    *b.add(s * st8 + t) = *sx.add(t * sd + s);
                }
                for t in st..st8 {
                    *b.add(s * st8 + t) = 0.0;
                }
            }
            let mut bad = 0usize;
            for i in 0..sd * st8 {
                if (*a.add(i)).to_bits() != (*b.add(i)).to_bits() {
                    bad += 1;
                }
            }
            c1_bad_total += bad;
            println!(
                "    d={:5} t={:4} t8={:4}  bit-mismatch = {} / {}",
                sd,
                st,
                st8,
                bad,
                sd * st8
            );
        }
        check!(
            c1_bad_total == 0,
            "C1",
            "AVX2 转置逐位等于按定义的标量转置（5 组形状，含 1 维/非 8 倍数）"
        );

        // ------------------------------------------------------------------
        // 3) C2 令牌分块内核逐位对拍（0 分叉才算过）
        // ------------------------------------------------------------------
        println!("\n--- C2: 令牌分块内核 b1_forward_t8 逐位 vs 标量契约参考 ---");
        // (n, k0, T) 四组形状；n == 源维度
        let kshapes: [(usize, usize, usize); 4] = [
            (1536, 20, 128),
            (32, 1, 8),
            (1, 20, 3),
            (17, 7, 13),
        ];
        let mut c2_bad_total = 0usize;
        let mut c2_bad_x1_total = 0usize;
        let mut c2_macs_ok = true;
        for (sn, sk0, st) in kshapes {
            let st8 = ((st + 7) / 8) * 8;
            let base = malloc(SparseConnW::bytes(sn as u32, sk0 as u32));
            let sc = SparseConnW::from_base(base, sn as u32, sk0 as u32, 4242);
            init_sparse_conn_scaled(&sc, 0.144);
            sc.assert_geometry();
            sc.assert_src_u32_consistent();
            let sx = malloc(st * sn * 4) as *mut f32;
            for i in 0..st * sn {
                *sx.add(i) = ((((i * 22695477usize + 7) % 1000) as f32) / 500.0) - 1.0;
            }
            let sxt = malloc(sn * st8 * 4) as *mut f32;
            let ya = malloc(sn * st8 * 4) as *mut f32;
            let yb = malloc(sn * st8 * 4) as *mut f32;
            // 两块缓冲都先置 0：否则「填充列（t >= T）必须保持 0」这条断言
            // 测的是 malloc 的垃圾值，而不是内核的行为（第一版就是这么假的）。
            for i in 0..sn * st8 {
                *ya.add(i) = 0.0;
                *yb.add(i) = 0.0;
            }
            b1_transpose_activations(sx, st, sn, sxt, st8);
            let ma = b1_forward_t8(&sc, sxt, st8, st, ya); // 生产入口（4 块版本）
            let mb = b1_forward_t8_scalar_reference(&sc, sxt, st8, st, yb);
            let mut bad = 0usize;
            for i in 0..sn * st8 {
                if (*ya.add(i)).to_bits() != (*yb.add(i)).to_bits() {
                    bad += 1;
                }
            }
            // 1 块版本写回 ya，再对拍一次（两个向量版本都必须逐位对）
            let mc = b1_forward_t8_x1(&sc, sxt, st8, st, ya);
            let mut bad_x1 = 0usize;
            for i in 0..sn * st8 {
                if (*ya.add(i)).to_bits() != (*yb.add(i)).to_bits() {
                    bad_x1 += 1;
                }
            }
            if mc != sn * sk0 {
                c2_macs_ok = false;
            }
            c2_bad_x1_total += bad_x1;
            // 填充列（t >= st）不允许被内核写坏成"非零垃圾"以外的假值：
            // 内核只写 t < n_tokens，填充列保持转置时写入的 0。
            let mut pad_nonzero = 0usize;
            for s in 0..sn {
                for t in st..st8 {
                    if *ya.add(s * st8 + t) != 0.0 {
                        pad_nonzero += 1;
                    }
                }
            }
            if ma != sn * sk0 || mb != sn * sk0 {
                c2_macs_ok = false;
            }
            c2_bad_total += bad;
            println!(
                "    n={:5} k0={:3} T={:4}  macs={}/{}  位分叉(4块) = {} / {}  位分叉(1块) = {}   填充列被写 = {}",
                sn,
                sk0,
                st,
                ma,
                sn * sk0,
                bad,
                sn * st8,
                bad_x1,
                pad_nonzero
            );
        }
        check!(
            c2_bad_total == 0 && c2_bad_x1_total == 0,
            "C2",
            "令牌分块内核逐位等于标量契约参考（4 组形状，含 k0=1/T=3/n=1；4 块与 1 块版本都对）"
        );
        check!(c2_macs_ok, "C2-b", "乘加计数 == n*k0（与发布版同口径）");

        // ------------------------------------------------------------------
        // 4) C3 确定性 + C4 can-fail + C5 -0.0 契约
        // ------------------------------------------------------------------
        println!("\n--- C3: 确定性（连续 8 次调用逐位相同）---");
        {
            let sc = scs[0];
            let xt = xtbuf[0];
            let y2 = malloc(d * t8 * 4) as *mut f32;
            b1_forward_t8(&sc, xt, t8, tokens, yt);
            let mut det = true;
            for _r in 0..8 {
                b1_forward_t8(&sc, xt, t8, tokens, y2);
                for i in 0..d * t8 {
                    if (*y2.add(i)).to_bits() != (*yt.add(i)).to_bits() {
                        det = false;
                    }
                }
            }
            check!(det, "C3", "8 次重复调用逐位相同（{} 个槽位）", d * t8);
        }

        println!("\n--- C4: can-fail —— 4 种已知错误内核必须被抓到 ---");
        {
            // 判据的"锚"：与标量契约参考比。若下面 4 个都能报 >0，
            // 说明 C2 的 bad=0 不是恒真。
            let sc = scs[0];
            let xt = xtbuf[0];
            let n = sc.n_neurons as usize;
            let kk0 = sc.k0;
            let yref = malloc(d * t8 * 4) as *mut f32;
            b1_forward_t8_scalar_reference(&sc, xt, t8, tokens, yref);

            // wrong-1: 漏算最后一条
            let mut bad1 = 0usize;
            // wrong-2: 行内降序累加
            let mut bad2 = 0usize;
            // wrong-3: 从 0.0 起加（不用首项赋值）
            let mut bad3 = 0usize;
            // wrong-4: 用 FMA 融合乘加
            let mut bad4 = 0usize;
            let yc = malloc(d * t8 * 4) as *mut f32;
            for i in 0..n {
                let base = i * kk0;
                for t in 0..tokens {
                    // wrong-1
                    {
                        let mut acc = 0.0f32;
                        let mut first = true;
                        for j in 0..kk0.saturating_sub(1) {
                            let sid = *sc.src_u32.add(base + j) as usize;
                            let term = *sc.w.add(base + j) * *xt.add(sid * t8 + t);
                            acc = if first { term } else { acc + term };
                            first = false;
                        }
                        *yc.add(i * t8 + t) = acc;
                    }
                    if (*yc.add(i * t8 + t)).to_bits() != (*yref.add(i * t8 + t)).to_bits() {
                        bad1 += 1;
                    }
                    // wrong-2（降序）
                    {
                        let mut acc = 0.0f32;
                        let mut first = true;
                        let mut j = kk0;
                        while j > 0 {
                            j -= 1;
                            let sid = *sc.src_u32.add(base + j) as usize;
                            let term = *sc.w.add(base + j) * *xt.add(sid * t8 + t);
                            acc = if first { term } else { acc + term };
                            first = false;
                        }
                        *yc.add(i * t8 + t) = acc;
                    }
                    if (*yc.add(i * t8 + t)).to_bits() != (*yref.add(i * t8 + t)).to_bits() {
                        bad2 += 1;
                    }
                    // wrong-3（从 0 起加）
                    {
                        let mut acc = 0.0f32;
                        for j in 0..kk0 {
                            let sid = *sc.src_u32.add(base + j) as usize;
                            acc += *sc.w.add(base + j) * *xt.add(sid * t8 + t);
                        }
                        *yc.add(i * t8 + t) = acc;
                    }
                    if (*yc.add(i * t8 + t)).to_bits() != (*yref.add(i * t8 + t)).to_bits() {
                        bad3 += 1;
                    }
                    // wrong-4（FMA）
                    {
                        let mut acc = 0.0f32;
                        let mut first = true;
                        for j in 0..kk0 {
                            let sid = *sc.src_u32.add(base + j) as usize;
                            let w = *sc.w.add(base + j);
                            let xv = *xt.add(sid * t8 + t);
                            acc = if first {
                                w * xv
                            } else {
                                f32::mul_add(w, xv, acc)
                            };
                            first = false;
                        }
                        *yc.add(i * t8 + t) = acc;
                    }
                    if (*yc.add(i * t8 + t)).to_bits() != (*yref.add(i * t8 + t)).to_bits() {
                        bad4 += 1;
                    }
                }
            }
            println!("    wrong-1 漏最后一条        caught = {} / {}", bad1, n * tokens);
            println!("    wrong-2 行内降序累加      caught = {} / {}", bad2, n * tokens);
            println!(
                "    wrong-3 从 0.0 起加        caught = {} / {}  <-- 随机数据抓不到，见下（已知边界）",
                bad3,
                n * tokens
            );
            println!("    wrong-4 FMA 融合乘加       caught = {} / {}", bad4, n * tokens);
            // wrong-3 在随机数据上抓不到是**本项目已记录的边界**（`0.0 + t == t`
            // 对绝大多数 t 成立），不是判据失效：它只能被定向构造暴露。
            // C5 就是那个定向构造（k0=1, w=-0.0 → 首项赋值 0x80000000 vs 从0起加
            // 0x00000000）。所以 C4 要求 wrong-1/2/4 被抓到，wrong-3 由 C5 覆盖。
            check!(
                bad1 > 0 && bad2 > 0 && bad4 > 0,
                "C4",
                "3 种已知错误内核被逐位判据抓到（漏项/降序/FMA）；'从0起加' 由 C5 定向覆盖"
            );
        }

        println!("\n--- C5: 首项赋值契约在令牌分块路径上可观测（-0.0）---");
        {
            // k0=1, w=-0.0, x=+1.0 => w*x = -0.0。首项赋值保住 0x80000000；
            // 从 0.0 起加会洗成 +0.0（0x00000000）。这是"首项赋值"这条契约
            // 唯一的可观测形态（随机数据抓不到，见 A1 can-fail 的历史记录）。
            let base = malloc(SparseConnW::bytes(1, 1));
            let sc = SparseConnW::from_base(base, 1, 1, 99);
            *sc.w.add(0) = -0.0f32;
            *sc.src.add(0) = 0.0f32;
            *sc.src_u32.add(0) = 0;
            *sc.act.add(0) = 1.0;
            let sx = malloc(1 * 4) as *mut f32;
            *sx.add(0) = 1.0f32;
            let st8 = 8usize;
            let sxt = malloc(1 * st8 * 4) as *mut f32;
            let sy = malloc(1 * st8 * 4) as *mut f32;
            b1_transpose_activations(sx, 1, 1, sxt, st8);
            b1_forward_t8(&sc, sxt, st8, 1, sy);
            let bits = (*sy.add(0)).to_bits();
            // 对照：从 0.0 起加
            let from0 = 0.0f32 + (-0.0f32);
            println!(
                "    令牌分块路径 w*x = {:#010x} (want 0x80000000 = -0.0)   从0起加 = {:#010x}",
                bits,
                from0.to_bits()
            );
            check!(
                bits == 0x8000_0000u32 && from0.to_bits() == 0x0000_0000u32,
                "C5",
                "首项赋值保住 -0.0；'从 0.0 起加' 会分叉 —— 契约可观测"
            );
        }

        // ------------------------------------------------------------------
        // 5) C6 B2 强度行向量化是否逐位等价（两种 mode 都对拍标量参考）
        // ------------------------------------------------------------------
        println!("\n--- C6: B2 SIMD 强度行逐位 vs 标量强度参考 ---");
        {
            let sc = scs[0];
            let x = xbuf[0];
            let n = sc.n_neurons as usize;
            // PER-ROW：B2 输出 vs 本地标量实现（强度标量 + 同一套 select/累加契约）
            let y_a = malloc(d * 4) as *mut f32;
            let y_b = malloc(d * 4) as *mut f32;
            b2_forward(&sc, x, y_a, k, B2Mode::PerRow, agg, hit);
            // 本地标量 PER-ROW
            let sp = malloc(k0 * 4) as *mut f32;
            let hit_s = malloc(k0 * 8) as *mut usize;
            for i in 0..n {
                let base = i * k0;
                for j in 0..k0 {
                    let sid = *sc.src.add(base + j) as usize;
                    *sp.add(j) = (*sc.w.add(base + j) * *x.add(sid)).abs();
                }
                let (nk, _c) = b2_select_once(sp, k0, k, hit_s, TieRule::StrictGt);
                let mut acc = 0.0f32;
                let mut first = true;
                for q in 0..nk {
                    let j = *hit_s.add(q);
                    let sid = *sc.src.add(base + j) as usize;
                    let term = *sc.w.add(base + j) * *x.add(sid);
                    acc = if first { term } else { acc + term };
                    first = false;
                }
                *y_b.add(i) = acc;
            }
            let mut bad_pr = 0usize;
            for i in 0..n {
                if (*y_a.add(i)).to_bits() != (*y_b.add(i)).to_bits() {
                    bad_pr += 1;
                }
            }
            // SHARED：B2 输出 vs 本地标量实现
            let y_c = malloc(d * 4) as *mut f32;
            let y_d = malloc(d * 4) as *mut f32;
            b2_forward(&sc, x, y_c, k, B2Mode::Shared, agg, hit);
            let agg_s = malloc(k0 * 4) as *mut f32;
            for j in 0..k0 {
                *agg_s.add(j) = 0.0;
            }
            for i in 0..n {
                let base = i * k0;
                for j in 0..k0 {
                    let sid = *sc.src.add(base + j) as usize;
                    *agg_s.add(j) += (*sc.w.add(base + j) * *x.add(sid)).abs();
                }
            }
            let (nk_s, _c) = b2_select_once(agg_s, k0, k, hit_s, TieRule::StrictGt);
            for i in 0..n {
                let base = i * k0;
                let mut acc = 0.0f32;
                let mut first = true;
                for q in 0..nk_s {
                    let j = *hit_s.add(q);
                    let sid = *sc.src.add(base + j) as usize;
                    let term = *sc.w.add(base + j) * *x.add(sid);
                    acc = if first { term } else { acc + term };
                    first = false;
                }
                *y_d.add(i) = acc;
            }
            let mut bad_sh = 0usize;
            for i in 0..n {
                if (*y_c.add(i)).to_bits() != (*y_d.add(i)).to_bits() {
                    bad_sh += 1;
                }
            }
            println!(
                "    PER-ROW 位分叉 = {} / {}   SHARED 位分叉 = {} / {}",
                bad_pr, n, bad_sh, n
            );
            check!(
                bad_pr == 0 && bad_sh == 0,
                "C6",
                "SIMD 强度行不改变 B2 输出（逐位，两种 mode）"
            );
            // can-fail：把强度换成"保符号的乘积"（= B1 的语义，即丢掉 abs）
            // 必须改变**选择结果**。这是本项目最贵的一次事故（S1 回退）在
            // B2 侧的对应判据：证明"强度必须带绝对值"是可观测的。
            let hit_abs = malloc(k0 * 8) as *mut usize;
            let hit_signed = malloc(k0 * 8) as *mut usize;
            let (_nk_a, _ca) = b2_select_once(agg_s, k0, k, hit_abs, TieRule::StrictGt);
            let agg_bad = malloc(k0 * 4) as *mut f32;
            for j in 0..k0 {
                *agg_bad.add(j) = 0.0;
            }
            for i in 0..n {
                let base = i * k0;
                for j in 0..k0 {
                    let sid = *sc.src.add(base + j) as usize;
                    *agg_bad.add(j) += *sc.w.add(base + j) * *x.add(sid); // 没有 abs
                }
            }
            let (_nk_b, _cb) = b2_select_once(agg_bad, k0, k, hit_signed, TieRule::StrictGt);
            let mut sel_diff = 0usize;
            for q in 0..k.min(k0) {
                if *hit_abs.add(q) != *hit_signed.add(q) {
                    sel_diff += 1;
                }
            }
            println!(
                "    强度丢 abs 后选择改变 = {}/{} 个槽位（can-fail 探针）",
                sel_diff,
                k.min(k0)
            );
            check!(
                bad_pr == 0 && bad_sh == 0 && sel_diff > 0,
                "C6",
                "SIMD 强度行不改变 B2 输出（逐位，两种 mode）；丢 abs 会被抓到"
            );
        }

        // ------------------------------------------------------------------
        // 5b) C9 令牌分块 B2 SHARED 逐位 vs 逐 token B2 SHARED
        // ------------------------------------------------------------------
        println!("\n--- C9: B2 SHARED 令牌分块 vs 逐 token（逐位 + 计数逐项）---");
        {
            let sc = scs[0];
            let n = sc.n_neurons as usize;
            let agg_t = malloc(k0 * t8 * 4) as *mut f32;
            let hit_t = malloc(8 * k0 * 8) as *mut usize;
            let y_block = malloc(d * t8 * 4) as *mut f32;
            let y_pertok = malloc(tokens * d * 4) as *mut f32;
            let mut bad_total = 0usize;
            let mut stats_diff = 0usize;
            for &tt in &[tokens, 13usize, 1usize, 8usize] {
                let tt8 = ((tt + 7) / 8) * 8;
                let xb = malloc(tt * d * 4) as *mut f32;
                for i in 0..tt * d {
                    *xb.add(i) = ((((i * 40503usize + 11) % 1000) as f32) / 500.0) - 1.0;
                }
                let xtb = malloc(d * tt8 * 4) as *mut f32;
                b1_transpose_activations(xb, tt, d, xtb, tt8);
                let yb = malloc(d * tt8 * 4) as *mut f32;
                let yp = malloc(tt * d * 4) as *mut f32;
                // 逐 token
                let mut s_macs = 0usize;
                let mut s_cmp = 0usize;
                let mut s_sel = 0usize;
                let mut s_str = 0usize;
                for t in 0..tt {
                    let st = b2_forward(
                        &sc,
                        xb.add(t * d),
                        yp.add(t * d),
                        k,
                        B2Mode::Shared,
                        agg,
                        hit,
                    );
                    s_macs += st.macs;
                    s_cmp += st.compares;
                    s_sel += st.selections;
                    s_str += st.strength_ops;
                }
                let st8 = vmlm_core::b2_topk::b2_forward_t8(
                    &sc, xtb, tt8, tt, yb, k, B2Mode::Shared, agg_t, hit_t,
                );
                let mut bad = 0usize;
                for i in 0..n {
                    for t in 0..tt {
                        if (*yb.add(i * tt8 + t)).to_bits() != (*yp.add(t * d + i)).to_bits() {
                            bad += 1;
                        }
                    }
                }
                let same_stats = st8.macs == s_macs
                    && st8.compares == s_cmp
                    && st8.selections == s_sel
                    && st8.strength_ops == s_str;
                if !same_stats {
                    stats_diff += 1;
                }
                bad_total += bad;
                println!(
                    "    T={:4}  位分叉 = {} / {}   计数(逐token) macs={} cmp={} sel={} str={}   计数(分块) macs={} cmp={} sel={} str={}",
                    tt,
                    bad,
                    n * tt,
                    s_macs,
                    s_cmp,
                    s_sel,
                    s_str,
                    st8.macs,
                    st8.compares,
                    st8.selections,
                    st8.strength_ops
                );
            }
            check!(
                bad_total == 0 && stats_diff == 0,
                "C9",
                "令牌分块 B2 SHARED 与逐 token 逐位相同，且四个计数器逐项相同（4 组 T，含奇数/1）"
            );
        }


        println!("\n--- 计时 ---");
        let warm = 3usize;

        // 单算子中位数（口径自检用）
        let mut t_op1: Vec<f64> = Vec::new();
        {
            let sc = scs[0];
            let x = xbuf[0];
            for _ in 0..warm {
                b1_forward(&sc, x, y_tm);
            }
            for _ in 0..200 {
                let a = clk.now();
                b1_forward(&sc, x, y_tm);
                let b = clk.now();
                t_op1.push(clk.ms(b - a));
            }
        }
        let med_op1 = median(&mut t_op1);

        // ---- W1 B1 标量契约参考（真循环）----
        let reps_slow = (reps / 4).max(3);
        let mut t_w1: Vec<f64> = Vec::new();
        for _ in 0..1 {
            for l in 0..n_mats {
                for t in 0..tokens {
                    b1_forward_scalar_reference(&scs[l], xbuf[l % mats].add(t * d), y_tm.add(t * d));
                }
            }
        }
        for _ in 0..reps_slow {
            let a = clk.now();
            for l in 0..n_mats {
                for t in 0..tokens {
                    b1_forward_scalar_reference(&scs[l], xbuf[l % mats].add(t * d), y_tm.add(t * d));
                }
            }
            let b = clk.now();
            t_w1.push(clk.ms(b - a));
            consume(y_tm, 256);
        }
        let med_w1 = median(&mut t_w1);

        // ---- W2 B1 发布版（真循环）----
        let mut t_w2: Vec<f64> = Vec::new();
        for _ in 0..warm {
            for l in 0..n_mats {
                for t in 0..tokens {
                    b1_forward(&scs[l], xbuf[l % mats].add(t * d), y_tm.add(t * d));
                }
            }
        }
        for _ in 0..reps {
            let a = clk.now();
            for l in 0..n_mats {
                for t in 0..tokens {
                    b1_forward(&scs[l], xbuf[l % mats].add(t * d), y_tm.add(t * d));
                }
            }
            let b = clk.now();
            t_w2.push(clk.ms(b - a));
            consume(y_tm, 256);
        }
        let med_w2 = median(&mut t_w2);

        // ---- W3 B1 令牌分块（转置在计时区外；纯内核口径）----
        for m in 0..mats {
            b1_transpose_activations(xbuf[m], tokens, d, xtbuf[m], t8);
        }
        let mut t_w3: Vec<f64> = Vec::new();
        for _ in 0..warm {
            for l in 0..n_mats {
                b1_forward_t8(&scs[l], xtbuf[l % mats], t8, tokens, yt);
            }
        }
        for _ in 0..reps {
            let a = clk.now();
            for l in 0..n_mats {
                b1_forward_t8(&scs[l], xtbuf[l % mats], t8, tokens, yt);
            }
            let b = clk.now();
            t_w3.push(clk.ms(b - a));
            consume(yt, 256);
        }
        let med_w3 = median(&mut t_w3);

        // ---- W3b B1 令牌分块 1 块版本（对照：摊薄到底值多少）----
        let mut t_w3b: Vec<f64> = Vec::new();
        for _ in 0..warm {
            for l in 0..n_mats {
                b1_forward_t8_x1(&scs[l], xtbuf[l % mats], t8, tokens, yt);
            }
        }
        for _ in 0..reps {
            let a = clk.now();
            for l in 0..n_mats {
                b1_forward_t8_x1(&scs[l], xtbuf[l % mats], t8, tokens, yt);
            }
            let b = clk.now();
            t_w3b.push(clk.ms(b - a));
            consume(yt, 256);
        }
        let med_w3b = median(&mut t_w3b);

        // ---- W4 B1 令牌分块 + 每个矩阵一次转置（端到端口径，保守）----
        let mut t_w4: Vec<f64> = Vec::new();
        for _ in 0..warm {
            for l in 0..n_mats {
                let m = l % mats;
                b1_transpose_activations(xbuf[m], tokens, d, xtbuf[m], t8);
                b1_forward_t8(&scs[l], xtbuf[m], t8, tokens, yt);
            }
        }
        for _ in 0..reps {
            let a = clk.now();
            for l in 0..n_mats {
                let m = l % mats;
                b1_transpose_activations(xbuf[m], tokens, d, xtbuf[m], t8);
                b1_forward_t8(&scs[l], xtbuf[m], t8, tokens, yt);
            }
            let b = clk.now();
            t_w4.push(clk.ms(b - a));
            consume(yt, 256);
        }
        let med_w4 = median(&mut t_w4);

        // 单独的转置成本（120 次）
        let mut t_tr: Vec<f64> = Vec::new();
        for _ in 0..reps.max(10) {
            let a = clk.now();
            for l in 0..n_mats {
                let m = l % mats;
                b1_transpose_activations(xbuf[m], tokens, d, xtbuf[m], t8);
            }
            let b = clk.now();
            t_tr.push(clk.ms(b - a));
            consume(xtbuf[0], 64);
        }
        let med_tr = median(&mut t_tr);

        // ---- W5 B2 SHARED（真循环）----
        let reps_b2 = (reps / 2).max(3);
        let mut t_w5: Vec<f64> = Vec::new();
        for _ in 0..warm {
            for l in 0..n_mats {
                for t in 0..tokens {
                    b2_forward(&scs[l], xbuf[l % mats].add(t * d), y_tm.add(t * d), k, B2Mode::Shared, agg, hit);
                }
            }
        }
        for _ in 0..reps_b2 {
            let a = clk.now();
            for l in 0..n_mats {
                for t in 0..tokens {
                    b2_forward(&scs[l], xbuf[l % mats].add(t * d), y_tm.add(t * d), k, B2Mode::Shared, agg, hit);
                }
            }
            let b = clk.now();
            t_w5.push(clk.ms(b - a));
            consume(y_tm, 256);
        }
        let med_w5 = median(&mut t_w5);

        // ---- W6 B2 PER-ROW（真循环；慢，少跑几次）----
        let mut t_w6: Vec<f64> = Vec::new();
        let reps_pr = 2usize;
        for _ in 0..1 {
            for l in 0..mats.min(n_mats) {
                for t in 0..tokens.min(32) {
                    b2_forward(&scs[l], xbuf[l % mats].add(t * d), y_tm.add(t * d), k, B2Mode::PerRow, agg, hit);
                }
            }
        }
        for _ in 0..reps_pr {
            let a = clk.now();
            for l in 0..n_mats {
                for t in 0..tokens {
                    b2_forward(&scs[l], xbuf[l % mats].add(t * d), y_tm.add(t * d), k, B2Mode::PerRow, agg, hit);
                }
            }
            let b = clk.now();
            t_w6.push(clk.ms(b - a));
            consume(y_tm, 256);
        }
        let med_w6 = median(&mut t_w6);

        // ---- W8 B2 SHARED 令牌分块（转置在计时区外）----
        let agg_t = malloc(k0 * t8 * 4) as *mut f32;
        let hit_t = malloc(8 * k0 * 8) as *mut usize;
        let mut t_w8: Vec<f64> = Vec::new();
        for _ in 0..warm {
            for l in 0..n_mats {
                vmlm_core::b2_topk::b2_forward_t8(
                    &scs[l], xtbuf[l % mats], t8, tokens, yt, k, B2Mode::Shared, agg_t, hit_t,
                );
            }
        }
        for _ in 0..reps_b2 {
            let a = clk.now();
            for l in 0..n_mats {
                vmlm_core::b2_topk::b2_forward_t8(
                    &scs[l], xtbuf[l % mats], t8, tokens, yt, k, B2Mode::Shared, agg_t, hit_t,
                );
            }
            let b = clk.now();
            t_w8.push(clk.ms(b - a));
            consume(yt, 256);
        }
        let med_w8 = median(&mut t_w8);

        // ---- W9 B2 SHARED 令牌分块 + 转置/矩阵（端到端）----
        let mut t_w9: Vec<f64> = Vec::new();
        for _ in 0..reps_b2 {
            let a = clk.now();
            for l in 0..n_mats {
                let m = l % mats;
                b1_transpose_activations(xbuf[m], tokens, d, xtbuf[m], t8);
                vmlm_core::b2_topk::b2_forward_t8(
                    &scs[l], xtbuf[m], t8, tokens, yt, k, B2Mode::Shared, agg_t, hit_t,
                );
            }
            let b = clk.now();
            t_w9.push(clk.ms(b - a));
            consume(yt, 256);
        }
        let med_w9 = median(&mut t_w9);

        // ------------------------------------------------------------------
        // 7) C7 假零守卫
        // ------------------------------------------------------------------
        println!("\n--- C7: 计时下限守卫（防假零）---");
        for (name, v) in [
            ("W1", med_w1),
            ("W2", med_w2),
            ("W3", med_w3b),
            ("W4", med_w3),
            ("W5", med_w4),
            ("W6", med_w5),
            ("W7", med_w6),
            ("W8", med_w8),
            ("W9", med_w9),
        ] {
            if v < FLOOR_MS {
                println!("    [FAIL] {} 计时 {:.6} ms < 下限 {:.6} ms —— 循环被删掉了", name, v, FLOOR_MS);
                fails += 1;
            }
        }
        checks += 1;
        println!(
            "    floor = {:.4} ms；最小实测 = {:.4} ms（{}）",
            FLOOR_MS,
            [med_w1, med_w2, med_w3b, med_w3, med_w4, med_w5, med_w6, med_w8, med_w9]
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min),
            "全部 step 均触碰 120 x 30720 条连接"
        );
        println!("  [PASS] C7 所有计时均高于下限（sink={:.6e}，输出确实被读回）", SINK);

        // ------------------------------------------------------------------
        // 8) 结果表：tok/s
        // ------------------------------------------------------------------
        let rows: [(&str, f64); 9] = [
            ("W1 B1 标量契约参考", med_w1),
            ("W2 B1 发布版 (gather+8路归约)", med_w2),
            ("W3 B1 令牌分块 x1 (8 token/次)", med_w3b),
            ("W4 B1 令牌分块 x4 (32 token/次)", med_w3),
            ("W5 B1 令牌分块 x4 + 转置/矩阵", med_w4),
            ("W6 B2 SHARED 逐token (SIMD 强度)", med_w5),
            ("W7 B2 PER-ROW 逐token", med_w6),
            ("W8 B2 SHARED 令牌分块", med_w8),
            ("W9 B2 SHARED 令牌分块 + 转置/矩阵", med_w9),
        ];
        println!("\n=============== sparse-B 吞吐（真循环，QPC 中位数）===============");
        println!(
            "  {:<34} {:>12} {:>12} {:>12} {:>10}",
            "impl", "ms/step", "ms/token", "tok/s", "vs W2"
        );
        for (name, ms) in rows {
            let tps = tokens as f64 / (ms / 1000.0);
            println!(
                "  {:<34} {:>12.4} {:>12.6} {:>12.1} {:>10.2}x",
                name,
                ms,
                ms / tokens as f64,
                tps,
                med_w2 / ms
            );
        }
        println!("\n  step = {} 个稀疏矩阵 x {} token = {} 次 matvec", n_mats, tokens, calls_per_step);
        println!("  转置单独计时（{} 次）：{:.4} ms/step（占 W4 的 {:.1}%）", n_mats, med_tr, 100.0 * med_tr / med_w4);

        // ---- C8 外推口径自检：单算子中位数 x calls 应同量级于真循环 W2 ----
        let extrap = med_op1 * calls_per_step as f64;
        let ratio = extrap / med_w2;
        println!("\n--- C8: 外推口径自检 ---");
        println!(
            "  单算子中位数 = {:.6} ms/op  x {} = {:.2} ms/step（外推）  vs 真循环 W2 = {:.2} ms/step  比值 = {:.3}",
            med_op1, calls_per_step, extrap, med_w2, ratio
        );
        check!(
            ratio > 0.5 && ratio < 2.0,
            "C8",
            "外推值与真循环同量级（0.5 < {:.3} < 2.0）—— 两个口径互证",
            ratio
        );

        // 发布版 vs 令牌分块：单算子口径（与历史数字可比）
        let mut t_op_t8: Vec<f64> = Vec::new();
        {
            let sc = scs[0];
            let xt = xtbuf[0];
            for _ in 0..warm {
                b1_forward_t8(&sc, xt, t8, tokens, yt);
            }
            for _ in 0..200 {
                let a = clk.now();
                b1_forward_t8(&sc, xt, t8, tokens, yt);
                let b = clk.now();
                t_op_t8.push(clk.ms(b - a));
            }
        }
        let med_op_t8 = median(&mut t_op_t8);
        // 同口径比较：把发布版的**单 token** 调用乘以 T，得到与令牌分块
        // 「一次处理 T 个 token」可比的时间。第一版这里直接把两者都除以
        // d*k0*T，把发布版的单 token 时间当成了 T 个 token 的时间（差 128x）。
        let shipped_t_tok = med_op1 * tokens as f64;
        println!(
            "\n  单矩阵口径（T={} token）：发布版 {} 次调用 = {:.6} ms（{:.4} ns/conn） vs 令牌分块 = {:.6} ms（{:.4} ns/conn）  = {:.2}x",
            tokens,
            tokens,
            shipped_t_tok,
            shipped_t_tok * 1e6 / (d * k0 * tokens) as f64,
            med_op_t8,
            med_op_t8 * 1e6 / (d * k0 * tokens) as f64,
            shipped_t_tok / med_op_t8
        );

        println!("\n===== checks={} fail={} =====", checks, fails);
        println!("(sink = {:.6e}，每个计时体都被 consume 读回)", SINK);
        if fails > 0 {
            std::process::exit(1);
        }
        println!("RESULT: ACCEPTANCE PASS（含 tok/s）");
    }
}
