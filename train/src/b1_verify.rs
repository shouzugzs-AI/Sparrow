// PATH: B1
//
// B1 验收驱动（判据 A1-A5）。**这是 B1**，不是 B2，也不是已废弃的路径 A。
//
// 用法：vmlm-b1-verify.exe [--n 1536] [--k0 20] [--k 8] [--reps 200]
// 退出码：0 = 全部判据 PASS；1 = 有判据 FAIL。
//
// 设计原则（沿用本项目 P17/P18 的纪律）：
//   1. 每个判据打印**计数**而不是布尔值 —— 「不能失败的检查不是测量」。
//   2. 每个核心判据都配一个 **can-fail 对照**：故意喂一个已知错的形态，
//      必须看到判据报错。否则判据可能恒真，而恒真的判据零信息。
//   3. 位比较一律用 `f32::to_bits()`。`==` 对 NaN 恒假、对 ±0 恒真，
//      恰好对本路径最可能的缺陷（±0 符号、首项赋值 vs 从 0 起加）全盲。
use vmlm_core::b1_pure::{b1_backward, b1_forward};
use vmlm_core::conn_sparse::K0_DEFAULT;
use vmlm_core::conn_train::*;

extern "C" {
    fn malloc(n: usize) -> *mut u8;
    fn QueryPerformanceCounter(c: *mut i64) -> i32;
    fn QueryPerformanceFrequency(f: *mut i64) -> i32;
}

// ---- 判据记账 ----
static mut N_CHECK: u32 = 0;
static mut N_FAIL: u32 = 0;

fn check(name: &str, ok: bool, detail: &str) {
    unsafe {
        N_CHECK += 1;
        if !ok {
            N_FAIL += 1;
        }
    }
    let tag = if ok { "PASS" } else { "FAIL" };
    println!("[{}] {}  {}", tag, name, detail);
}

fn check_count(name: &str, bad: usize, detail: &str) {
    check(name, bad == 0, &format!("bad={}  {}", bad, detail));
}

fn args_u32(flag: &str, def: u32) -> u32 {
    let v: Vec<String> = std::env::args().collect();
    for i in 0..v.len() {
        if v[i] == flag && i + 1 < v.len() {
            return v[i + 1].parse().unwrap_or(def);
        }
    }
    def
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

unsafe fn alloc_f32(n: usize) -> *mut f32 {
    let p = malloc(n * 4 + 64) as *mut f32;
    assert!(!p.is_null(), "OOM");
    p
}

unsafe fn slice<'a>(p: *mut f32, n: usize) -> &'a mut [f32] {
    std::slice::from_raw_parts_mut(p, n)
}

fn det_fill(buf: &mut [f32], seed: u32, lo: f32, hi: f32) {
    let mut s = seed | 1;
    for x in buf.iter_mut() {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        let u = (s as f64) / (u32::MAX as f64);
        *x = (lo as f64 + (hi - lo) as f64 * u) as f32;
    }
}

/// 逐位比较（**不用 `==`**）：返回不同的元素个数。
unsafe fn bit_diff(a: *const f32, b: *const f32, n: usize) -> usize {
    let mut bad = 0usize;
    for i in 0..n {
        if (*a.add(i)).to_bits() != (*b.add(i)).to_bits() {
            bad += 1;
        }
    }
    bad
}

fn main() {
    let n_neurons = args_u32("--n", 1536);
    let k0 = args_u32("--k0", K0_DEFAULT);
    let k = args_u32("--k", 8);
    let reps = args_u32("--reps", 200) as usize;

    println!("===== B1 verification (pure-sparse full-k0: no top-k, no mask) =====");
    println!(
        "n_neurons={} k0={} k={} reps={}",
        n_neurons, k0, k, reps
    );
    println!("THIS IS B1.");

    let n = n_neurons as usize;
    let k0u = k0 as usize;
    let e = n * k0u;

    unsafe {
        vmlm_core::interp::init_cpu();

        // ---------------------------------------------------------------
        // 构型：一个 SparseConnW，和 trainer 造它们的方式一样
        // ---------------------------------------------------------------
        let base = malloc(SparseConnW::bytes(n_neurons, k0));
        assert!(!base.is_null());
        let sc = SparseConnW::from_base(base, n_neurons, k0, 7);
        init_sparse_conn_scaled(&sc, 0.144);

        let x = alloc_f32(n);
        det_fill(slice(x, n), 0x1234_5678, -1.0, 1.0);
        let y_b1 = alloc_f32(n);
        let y_op = alloc_f32(n);

        // ===============================================================
        // A1: b1_forward vs compute_act_mask(k=k0) + sparse_accum  —— 逐位相同
        // ===============================================================
        //
        // 为什么拿「k=k0 的算子版」当参考：k0=20 时 top-k 会选中**全部 20 条**
        // （选择空间就是 k0），于是算子版的 act 掩码恒为全 1，它的可观测行为
        // 与 B1 完全相同。这给了我们一条**独立实现**的参考路径来对账，
        // 而不是拿 B1 自己跟自己比。
        //
        // ⚠️ 前向调用顺序有讲究（踩过的坑）：
        //   算子版会**改写 act**（先清零再置 1）。A2 要证明的是「B1 前向不写
        //   act」，所以必须在 B1 调用**之前**取快照，且 B1 调用不得被夹在
        //   compute_act_mask 中间。
        //   这里刻意先跑 B1、再做 A2 快照比对，然后用算子版算参考。
        let macs_b1 = b1_forward(&sc, x, y_b1);
        check(
            "A1-a b1_forward macs == n*k0 (all connections computed)",
            macs_b1 == e,
            &format!("macs={} want={} (n={} k0={})", macs_b1, e, n, k0u),
        );

        // ===============================================================
        // A2: 零掩码证明 —— B1 前向期间 act[] 被写入 0 次
        // ===============================================================
        //
        // 做法：把 act 填成一个**可识别的哨兵位型**（不是 0.0 —— 0.0 也是
        // 合法的「B1 会写出」的值，用哨兵才能区分「没写」和「写了 0」），
        // 跑一次 b1_forward，再逐位比对整块 act。
        //
        // 为什么用哨兵而不是「统计调用次数」：调用次数要靠插桩或读代码，
        // 那是主张不是测量。位快照是**可观测事实**。
        let sentinel = f32::from_bits(0x7FC0_1234); // 一个确定的 NaN 位型
        for t in 0..e {
            *sc.act.add(t) = sentinel;
        }
        let act_before = alloc_f32(e);
        for t in 0..e {
            *act_before.add(t) = *sc.act.add(t);
        }
        let macs_a2 = b1_forward(&sc, x, y_b1);
        let act_touched = bit_diff(sc.act as *const f32, act_before as *const f32, e);
        check_count(
            "A2 b1_forward writes act[] 0 times (mask is not materialized)",
            act_touched,
            &format!("(act_bytes_compared={}; macs={})", e, macs_a2),
        );
        // 前提非平凡性：哨兵必须真的与「0.0」不同，否则上面的 0 可能是
        // 「写了 0 但哨兵本来就是 0」的巧合。
        check(
            "A2 premise: sentinel differs from every plausible written value",
            sentinel.to_bits() == 0x7FC0_1234 && sentinel.to_bits() != 0.0f32.to_bits(),
            &format!("sentinel_bits=0x{:08X}", sentinel.to_bits()),
        );
        // can-fail 对照：把 act 人为改一个位置，A2 的比对必须能看见。
        // （证明这套快照比对不是恒 0 的哑判据。）
        let probe = e / 2;
        *sc.act.add(probe) = 1.0;
        let act_control = bit_diff(sc.act as *const f32, act_before as *const f32, e);
        *sc.act.add(probe) = sentinel; // 还原
        check(
            "A2 can-fail control (poke 1 act slot => diff must be exactly 1)",
            act_control == 1,
            &format!("control_diff={} (want 1)", act_control),
        );

        // ---- 现在才跑算子版参考（它会改写 act，必须在 A2 之后）----
        let macs_op = sparse_forward(&sc, x, y_op, k0u);
        check(
            "A1-a' operator version macs == n*k0",
            macs_op == e,
            &format!("op_macs={} want={}", macs_op, e),
        );
        let a1_bad = bit_diff(y_b1 as *const f32, y_op as *const f32, n);
        // ---- A1 已按用户裁决**降级为参考**（不再作为阻塞判据）----
        //
        // 用户原文：「不需要 bit-exact，计算本质一样就不必这么追究了。」
        // 验收改为：确定性 + |simd-scalar| ≤ ε（见 `vmlm-b1-prod-eps`）。
        //
        // **没有删除**这条判据，而是降级为**参考**：位差计数仍然打印出来，
        // 这样"偏离变大"这件事依然可见 —— 只是不再要求它等于 0。
        // 一个真实缺陷（符号错、索引错、漏尾块）会让位差变成 1536/1536
        // 且误差是**数值量级**的，而不是舍入量级；那仍然会被 prod-eps 的
        // 比值判据抓住（实测比值 1.229e-7 << 1.0）。
        //
        // 同时给出**量级**判据：差异必须相对项幅度为舍入级。
        let mut worst_ratio = 0.0f32;
        for i in 0..n {
            let mut s = 0.0f32;
            let base = i * k0u;
            for j in 0..k0u {
                let sid = *sc.src.add(base + j) as usize;
                s += (*sc.w.add(base + j) * *x.add(sid)).abs();
            }
            if s > 0.0 {
                let d = (*y_b1.add(i) - *y_op.add(i)).abs();
                let r = d / s;
                if r > worst_ratio {
                    worst_ratio = r;
                }
            }
        }
        println!(
            "  [REF] A1 (demoted) b1_forward vs operator: bit-diff = {} / {} ({:.1}% identical)",
            a1_bad,
            n,
            100.0 * (n - a1_bad) as f64 / n as f64
        );
        println!(
            "  [REF]     worst |diff|/scale = {:.3e}  (rounding level if << 1; a real bug is ~1.0)",
            worst_ratio
        );
        check(
            "A1-eq (replaces bit-exact): |b1 - operator| is at ROUNDING level",
            worst_ratio <= 1.0e-5,
            &format!(
                "worst |diff|/scale = {:.3e} <= 1e-5  (bit-exact降级为参考：{}/{} 位差)",
                worst_ratio, a1_bad, n
            ),
        );

        // ---- A1 can-fail 1：错维度（漏算最后一条）必须报错 ----
        // 这是任务书的 A3，放在这里因为它是 A1 的对照。
        //
        // ⚠️ 第一版写成 `SparseConnW { k0: k0u-1, ..sc }` —— k0 与缓冲区不符。
        //    那**违反内核的安全前提**，且 SIMD 路径下会读到越界（实测崩溃）。
        //    现在生产侧有 `assert_geometry()` 守卫（给出可读报错而非崩溃），
        //    但这条判据的**本意**是"漏算一条连接"，不是"喂非法结构"。
        //    修法：分配一个**几何自洽**的 (k0-1) 宽结构，真的少摆一条。
        let y_wrong = alloc_f32(n);
        {
            let k0s = k0u - 1;
            let es = n * k0s;
            let bs = malloc(SparseConnW::bytes(n as u32, k0s as u32));
            let mut sc_short = SparseConnW::from_base(bs, n as u32, k0s as u32, 7);
            sc_short.e = es;
            for i in 0..n {
                for j in 0..k0s {
                    let sid = *sc.src.add(i * k0u + j);
                    *sc_short.src.add(i * k0s + j) = sid;
                    *sc_short.src_u32.add(i * k0s + j) = sid as u32;
                    *sc_short.w.add(i * k0s + j) = *sc.w.add(i * k0u + j);
                }
            }
            sc_short.assert_geometry();
            b1_forward(&sc_short, x, y_wrong);
        }
        let a3_bad = bit_diff(y_wrong as *const f32, y_op as *const f32, n);
        check(
            "A3 can-fail control (k0-1 => mismatch>0)",
            a3_bad > 0,
            &format!("mismatch={} (must be >0, else A1 is vacuous)", a3_bad),
        );

        // ---- A1 can-fail 2：把「首项直接赋值」改成「从 0.0 起加」----
        // 这是本模块最微妙的一条契约。构造一个能让它必现的输入：
        //   令某行第一项 = -0.0，且**没有后续项**（k=1 单列布局）。
        //   直接赋值   => -0.0                    （位型 0x80000000）
        //   从 0.0 起加 => 0.0 + (-0.0) = +0.0   （位型 0x00000000）
        // 若这条对照报 0，说明「首项赋值」这条契约在判据里**没被覆盖**，
        // A1 就是弱判据。
        let n_small = 4usize;
        let k1 = 1usize;
        let base_1 = malloc(SparseConnW::bytes(n_small as u32, k1 as u32));
        let sc_1 = SparseConnW::from_base(base_1, n_small as u32, k1 as u32, 3);
        let xs = alloc_f32(n_small);
        let y1 = alloc_f32(n_small);
        let y1_zero = alloc_f32(n_small);
        for i in 0..n_small {
            *xs.add(i) = 1.0;
        }
        for t in 0..n_small * k1 {
            *sc_1.src.add(t) = 0.0; // 全部指向 x[0]
            *sc_1.w.add(t) = -0.0; // term = -0.0 * 1.0 = -0.0
        }
        // ⚠️ 必须同步 u32 镜像：`b1_forward` 的 SIMD 路径**按 src_u32 做 gather**。
        // 本驱动此前漏了这一行，于是这里读的是 malloc 未初始化的 u32 下标，
        // 用它们去 gather `xs`（只有 4 个元素）—— 判据的成功/不崩溃全靠运气。
        // 实测：加过零拷贝代码后同一处直接 0xC0000005（访问违例）。
        // 这不是被优化"弄坏"的：是驱动违反了 conn_train 的镜像一致性契约。
        sc_1.sync_src_u32();
        b1_forward(&sc_1, xs, y1);
        // 反例：显式从 0.0 起加（这正是「首项赋值」被删掉后的形态）
        for i in 0..n_small {
            let mut acc = 0.0f32;
            for j in 0..k1 {
                let sid = *sc_1.src.add(i * k1 + j) as usize;
                acc += *sc_1.w.add(i * k1 + j) * *xs.add(sid);
            }
            *y1_zero.add(i) = acc;
        }
        let neg_zero_b1 = (0..n_small)
            .filter(|&i| (*y1.add(i)).to_bits() == 0x8000_0000)
            .count();
        let neg_zero_zerostart = (0..n_small)
            .filter(|&i| (*y1_zero.add(i)).to_bits() == 0x8000_0000)
            .count();
        // ---- A1 的 -0.0 契约：**已随 bit-exact 一起放弃**，改为参考 ----
        //
        // 原判据测的是「首项直接赋值」这个契约：`acc = term` 而非从 0.0 起加，
        // 于是 `-0.0` 能被保留（`0.0 + (-0.0) = +0.0` 会把它洗掉）。
        //
        // SIMD 归约路径**从零向量起步**，所以这条契约按定义不再成立 ——
        // 这是用户裁决「不需要 bit-exact」的直接后果，不是回归。
        // 实测 neg_zero(b1) 从 4 变成 0，正是这个原因。
        //
        // 保留为**参考**而不删除：它记录了一条**被主动放弃的契约**，
        // 后人若看到 `-0.0` 消失，这里能查到原因，而不是去"修"一个没坏的东西。
        println!(
            "  [REF] A1 (demoted) -0.0 preservation: neg_zero(b1)={} neg_zero(from0)={} of {}",
            neg_zero_b1, neg_zero_zerostart, n_small
        );
        println!(
            "  [REF]     'first-term-assign' was DROPPED with bit-exact: the SIMD reducer"
        );
        println!(
            "  [REF]     starts from a zero vector, so 0.0+(-0.0)=+0.0 now applies."
        );
        // 仍要证明这条判据**曾经**能失败（否则它当初就是哑判据）：
        // 用纯标量参考实现去测"首项赋值"的形态。
        {
            let mut y_assign = alloc_f32(n_small);
            for i in 0..n_small {
                let mut first = true;
                let mut acc = 0.0f32;
                for j in 0..k1 {
                    let sid = *sc_1.src.add(i * k1 + j) as usize;
                    let term = *sc_1.w.add(i * k1 + j) * *xs.add(sid);
                    acc = if first { term } else { acc + term };
                    first = false;
                }
                *y_assign.add(i) = acc;
            }
            let nz_assign = (0..n_small)
                .filter(|&i| (*y_assign.add(i)).to_bits() == 0x8000_0000)
                .count();
            check(
                "A1-ref: the pure-scalar 'first-term-assign' form DOES preserve -0.0 (contract was observable)",
                nz_assign == n_small,
                &format!("neg_zero(scalar assign)={} of {} (must be all => the old bar was real)", nz_assign, n_small),
            );
        }
        // 顺带证明 `==` 会骗人：两个 +0.0/-0.0 值 `==` 相等但位型不同。
        check(
            "A1 note: == is blind to +0.0/-0.0 (why to_bits was mandatory)",
            (0.0f32 == -0.0f32) && (0.0f32.to_bits() != (-0.0f32).to_bits()),
            "0.0 == -0.0 is true, but bits differ (0x00000000 vs 0x80000000)",
        );

        // ===============================================================
        // A5: 反向 bit-exact vs sparse_gemm_bwd_w（k=k0 全激活）
        // ===============================================================
        //
        // 参考侧必须先把掩码做成「全激活」：k=k0 的 compute_act_mask 会
        // 选中全部 k0 条。上面 `sparse_forward(&sc, x, y_op, k0u)` 已经留下了
        // 这张全激活掩码 —— 但**不假设**，而是显式数一遍激活数再对账。
        let dy = alloc_f32(n);
        det_fill(slice(dy, n), 0xDEAD_BEEF, -1.0, 1.0);
        let dw_b1 = alloc_f32(e);
        let dw_op = alloc_f32(e);
        std::ptr::write_bytes(dw_b1 as *mut u8, 0, e * 4);
        std::ptr::write_bytes(dw_op as *mut u8, 0, e * 4);

        let mut n_act = 0usize;
        for t in 0..e {
            if *sc.act.add(t) != 0.0 {
                n_act += 1;
            }
        }
        check(
            "A5 premise: act is all-active after compute_act_mask(k=k0)",
            n_act == e,
            &format!("active={} of E={}", n_act, e),
        );

        let wrote_b1 = b1_backward(&sc, dy, x, dw_b1);
        let wrote_op = sparse_gemm_bwd_w(&sc, dy, x, dw_op);
        check(
            "A5-a b1_backward writes all E connections",
            wrote_b1 == e,
            &format!("written={} want E={}", wrote_b1, e),
        );
        check(
            "A5-a' operator version writes all E connections",
            wrote_op == e,
            &format!("op_written={} want E={}", wrote_op, e),
        );
        let a5_bad = bit_diff(dw_b1 as *const f32, dw_op as *const f32, e);
        check_count(
            "A5 b1_backward == sparse_gemm_bwd_w (bitwise, k=k0 all active)",
            a5_bad,
            &format!("(E={} compared)", e),
        );

        // ---- A5 前提非平凡性：b1_backward 确实改了 dw ----
        // 若两边都全是 0，上面的 0 是平凡的。
        let mut nz = 0usize;
        for t in 0..e {
            if *dw_b1.add(t) != 0.0 {
                nz += 1;
            }
        }
        check(
            "A5 criterion is non-vacuous (gradients are non-zero)",
            nz > 0,
            &format!("nonzero_grads={} of E={}", nz, e),
        );

        // ---- A5 can-fail：用错 src 宽度必须报错 ----
        let dw_bad = alloc_f32(e);
        std::ptr::write_bytes(dw_bad as *mut u8, 0, e * 4);
        {
            let sc_short = SparseConnW {
                k0: k0u - 1,
                ..sc
            };
            b1_backward(&sc_short, dy, x, dw_bad);
        }
        let a5_cf = bit_diff(dw_bad as *const f32, dw_op as *const f32, e);
        check(
            "A5 can-fail control (k0-1 bwd => mismatch>0)",
            a5_cf > 0,
            &format!("mismatch={} (must be >0, else A5 is vacuous)", a5_cf),
        );

        // ===============================================================
        // A5-b: 累加语义（`+=` 而不是 `=`）必须被真的测到
        // ===============================================================
        //
        // ⚠️ 这一条补的是一个**真实的覆盖漏洞**，不是锦上添花。
        //
        // 上面的 A5 从**清零的** dw 出发，于是在那种输入下 `dw = g*x` 与
        // `dw += g*x` 给出**完全相同**的结果 —— 也就是说 A5 报 bad=0
        // **无法区分这两种语义**。而本模块的文件头把「必须是 +=」写成
        // 一条极重要的契约，并给了理由（改成赋值会让一个 batch 里多帧的
        // 梯度互相覆盖，而 loss 仍会下降，只是慢，于是极难发现）。
        // ⇒ 一条被文档写成「极难发现」的契约，却没有判据能发现它。
        //
        // 测法：预置一份**非零**的 dw，跑一次 b1_backward，然后要求结果
        // 等于「预置值 + 本次梯度」。若实现被改成赋值，残差立刻爆掉。
        // 这里不依赖算子版，而是用**独立手算**的参考（一个朴素标量循环），
        // 避免「拿实现和自己的另一种调用方式对拍」。
        {
            let pre = alloc_f32(e);
            // 预置值：确定性的非零、有正有负，且**没有一个是 ±0**。
            // ⚠️ 第一版写的是 `((t % 97) as f32 - 48.0) * 0.001`，它在
            // `t % 97 == 48` 时正好给出 0.0 —— 于是有 317 个位置退化成
            // 「清零起始」，而那些位置上的 += / = 又变得不可区分。
            // 判据自己把这个抓了出来（premise 检查报 30403/30720）。
            // 修法：用 `(t % 97) + 1` 保证偏移恒不为 0 项。
            for t in 0..e {
                *pre.add(t) = ((t % 97) as f32 + 1.0) * 0.001;
            }
            let dw_acc = alloc_f32(e);
            for t in 0..e {
                *dw_acc.add(t) = *pre.add(t);
            }
            let wrote_acc = b1_backward(&sc, dy, x, dw_acc);

            // 独立参考：朴素标量循环，累加在预置值上。
            // 乘数顺序与实现一致（g * x），保证是逐位比较而不是近似比较。
            let dw_ref = alloc_f32(e);
            for t in 0..e {
                *dw_ref.add(t) = *pre.add(t);
            }
            for i in 0..n {
                let g = *dy.add(i);
                let base = i * k0u;
                for j in 0..k0u {
                    let sid = *sc.src.add(base + j) as usize;
                    *dw_ref.add(base + j) += g * *x.add(sid);
                }
            }
            let a5b_bad = bit_diff(dw_acc as *const f32, dw_ref as *const f32, e);
            check_count(
                "A5-b b1_backward ACCUMULATES onto a non-zero dw (not assign)",
                a5b_bad,
                &format!(
                    "(E={} compared against an independent scalar reference, pre-seeded non-zero)",
                    e
                ),
            );
            check(
                "A5-b b1_backward still reports all E writes when accumulating",
                wrote_acc == e,
                &format!("written={} want E={}", wrote_acc, e),
            );

            // ---- can-fail：把「累加」换成「赋值」必须被抓出来 ----
            // 这是本判据存在的全部理由。若这一步报 0，说明 A5-b 是哑判据。
            let dw_assign = alloc_f32(e);
            for t in 0..e {
                *dw_assign.add(t) = *pre.add(t);
            }
            for i in 0..n {
                let g = *dy.add(i);
                let base = i * k0u;
                for j in 0..k0u {
                    let sid = *sc.src.add(base + j) as usize;
                    // 故意的错误语义：赋值而不是累加
                    *dw_assign.add(base + j) = g * *x.add(sid);
                }
            }
            let a5b_cf = bit_diff(dw_assign as *const f32, dw_ref as *const f32, e);
            check(
                "A5-b can-fail: 'assign' instead of '+=' IS detected",
                a5b_cf > 0,
                &format!(
                    "mismatch={} of E={} (must be >0, else A5-b could not catch the bug it exists for)",
                    a5b_cf, e
                ),
            );

            // ---- 非平凡性前提：预置值必须真的非零，否则又退化成 A5 ----
            let mut pre_nz = 0usize;
            for t in 0..e {
                if *pre.add(t) != 0.0 {
                    pre_nz += 1;
                }
            }
            check(
                "A5-b premise: the pre-seeded buffer is non-zero (else it degenerates to A5)",
                pre_nz == e,
                &format!("non-zero pre-seed = {} of E={}", pre_nz, e),
            );
        }

        // ===============================================================
        // A4: QPC 性能 —— b1_forward vs 算子版 sparse_forward(k=8)
        // ===============================================================
        //
        // 任务书预期 b1 快 ~5.1x（依据：实测 clear+set no-topk 0.0391 ms
        // vs compute_act_mask 0.2004 ms）。**报告实测值，不报告预期值。**
        //
        // 多次重复取中位数：单次 QPC 测量受调度抖动影响，中位数比均值稳健
        // （本项目不引入统计库，手写选择即可）。
        let clk = Clock::new();
        let mut t_b1: Vec<f64> = Vec::with_capacity(reps);
        let mut t_op: Vec<f64> = Vec::with_capacity(reps);
        // 预热（让分支预测器/缓存进入稳态，避免把冷启动算进中位数）
        for _ in 0..20 {
            b1_forward(&sc, x, y_b1);
            sparse_forward(&sc, x, y_op, k as usize);
        }
        for _ in 0..reps {
            let c0 = clk.now();
            b1_forward(&sc, x, y_b1);
            let c1 = clk.now();
            t_b1.push(clk.ms(c1 - c0));

            let c2 = clk.now();
            sparse_forward(&sc, x, y_op, k as usize);
            let c3 = clk.now();
            t_op.push(clk.ms(c3 - c2));
        }
        let med_b1 = median(&mut t_b1);
        let med_op = median(&mut t_op);
        let speedup = med_op / med_b1;
        println!(
            "  A4 QPC: b1_forward median = {:.4} ms  ({:.3} ns/conn)",
            med_b1,
            med_b1 * 1e6 / e as f64
        );
        println!(
            "  A4 QPC: sparse_forward(k={}) median = {:.4} ms  ({:.3} ns/conn)",
            k,
            med_op,
            med_op * 1e6 / e as f64
        );
        println!(
            "  A4 QPC: speedup = {:.3}x  (brief's prior expectation was ~5.1x)",
            speedup
        );
        println!(
            "  A4 QPC: raw b1 samples (first 8) = {:?}",
            &t_b1[..t_b1.len().min(8)]
        );
        println!(
            "  A4 QPC: raw op samples (first 8) = {:?}",
            &t_op[..t_op.len().min(8)]
        );
        // 判据：B1 必须**不慢于**算子版（免掩码是净收益）。这是可失败的：
        // 若 b1 反而更慢，说明测量或实现有问题，必须报出来。
        check(
            "A4 b1_forward is not slower than sparse_forward(k=8)",
            med_b1 <= med_op,
            &format!(
                "b1={:.4} ms op={:.4} ms speedup={:.3}x (reps={})",
                med_b1, med_op, speedup, reps
            ),
        );
        // 单独测一次 compute_act_mask 本身，好让「省掉的到底是什么」有数字。
        let mut t_mask: Vec<f64> = Vec::with_capacity(reps);
        for _ in 0..reps {
            let c0 = clk.now();
            compute_act_mask(&sc, x, k as usize);
            let c1 = clk.now();
            t_mask.push(clk.ms(c1 - c0));
        }
        let med_mask = median(&mut t_mask);
        println!(
            "  A4 QPC: compute_act_mask(k={}) median = {:.4} ms  (the step B1 removes)",
            k, med_mask
        );
        println!(
            "  A4 QPC: mask / op-forward = {:.1}%",
            100.0 * med_mask / med_op
        );

        println!();
        println!("===== checks={} fail={} =====", N_CHECK, N_FAIL);
        let rc = if N_FAIL == 0 { 0 } else { 1 };
        std::process::exit(rc);
    }
}

fn median(v: &mut Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}
