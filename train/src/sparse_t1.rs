// P17 T1/T2/T3 单层自检驱动（P0 第一步）
//
// 任务书 §5 建议顺序第 2 条原文：
//   「**T1+T2 单层**：先做**单层**稀疏连接初始化 + 只更新激活连接，
//     用手算小例子验证梯度正确（**与稠密参考对账**，误差给数字）。」
//
// 本驱动就是这个「单层、手算、与稠密参考对账」的可执行形式。
// 它**不**跑 24 层（那是下一步 vmlm-sparse-train 的事），
// 目的是让单个机制的错误能在**没有模型复杂度掩盖**的地方暴露。
//
// 使用：vmlm-sparse-t1.exe [--k0 20] [--n 64] [--k 8] [--steps 400]
//
// 退出码：0 = 全部判据 PASS；1 = 有判据 FAIL。
// 每个判据都打印**计数**（不是布尔值），并且关键判据都有一个
// **can-fail 对照**：故意用一个已知错的输入让它报 FAIL。
// 「不能失败的检查不是测量」（P17 §0.2）。
use vmlm_core::conn_sparse::{neuron_conn_src, ConnSparseLayout, K0_DEFAULT};
use vmlm_core::conn_train::*;

extern "C" {
    fn malloc(n: usize) -> *mut u8;
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

/// 计数型判据：`bad == 0` 才算 PASS，并把**差多少**报出来。
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

/// 确定性填充（与 conn_train 的 PRNG 无关的**测试侧**数据）：
/// 用 xorshift 保证「同一 seed 给同一输入」，且**不是**全同值。
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

unsafe fn alloc_f32(n: usize) -> *mut f32 {
    let p = malloc(n * 4 + 64) as *mut f32;
    assert!(!p.is_null());
    p
}

unsafe fn slice<'a>(p: *mut f32, n: usize) -> &'a mut [f32] {
    std::slice::from_raw_parts_mut(p, n)
}

fn libm_sqrt(x: f32) -> f32 {
    vmlm_core::kernels::sqrt_f(x)
}
fn libm_pow(b: f32, e: f32) -> f32 {
    vmlm_core::kernels::exp_f(e * vmlm_core::kernels::ln_f(b))
}

fn main() {
    let n_neurons = args_u32("--n", 64);
    let k0 = args_u32("--k0", K0_DEFAULT);
    let k = args_u32("--k", 8);
    let steps = args_u32("--steps", 400);
    let lr = 0.05f32;

    println!("===== P17 T1/T2/T3 single-layer self-check =====");
    println!(
        "n_neurons={} k0={} k={} steps={} lr={}",
        n_neurons, k0, k, steps, lr
    );

    let e = n_neurons as usize * k0 as usize;
    unsafe {
        // ---------------------------------------------------------------
        // T1: 初始化
        // ---------------------------------------------------------------
        let base = malloc(SparseConnW::bytes(n_neurons, k0));
        assert!(!base.is_null());
        let sc = SparseConnW::from_base(base, n_neurons, k0, 0);
        init_sparse_conn(&sc);

        // 判据 T1-a：拓扑恰好 k0 条/神经元，且 src 都在范围内。
        let mut bad_src = 0usize;
        let mut bad_cnt = 0usize;
        for i in 0..n_neurons as usize {
            for j in 0..k0 as usize {
                let t = sc.idx(i, j);
                let s = *sc.src.add(t) as i64;
                if s < 0 || s >= n_neurons as i64 {
                    bad_src += 1;
                }
                // 与 conn_sparse 的**独立**公式对拍（这正是「复用 PRNG」
                // 的验收形式：两条路径必须给出同一个 src）
                let want = neuron_conn_src(0, i as u32, j as u32, n_neurons, k0);
                if *sc.src.add(t) as u32 != want {
                    bad_cnt += 1;
                }
            }
        }
        check_count("T1-a src in range", bad_src, "src_id < n_neurons");
        check_count(
            "T1-a src == conn_sparse::neuron_conn_src",
            bad_cnt,
            "(PRNG reuse)",
        );

        // 判据 T1-b：权重非平凡（不是全 0、不是全同值、正负都有）。
        let mut wmin = f32::INFINITY;
        let mut wmax = f32::NEG_INFINITY;
        let mut npos = 0usize;
        let mut nneg = 0usize;
        for t in 0..e {
            let w = *sc.w.add(t);
            if w < wmin {
                wmin = w;
            }
            if w > wmax {
                wmax = w;
            }
            if w > 0.0 {
                npos += 1;
            }
            if w < 0.0 {
                nneg += 1;
            }
        }
        let spread = wmax - wmin;
        check(
            "T1-b weight not degenerate",
            spread > 1e-3 && npos > 0 && nneg > 0,
            &format!(
                "min={:.6} max={:.6} spread={:.6} npos={} nneg={}",
                wmin, wmax, spread, npos, nneg
            ),
        );

        // ---------------------------------------------------------------
        // T2: 前向 —— 只算激活连接
        // ---------------------------------------------------------------
        let x = alloc_f32(n_neurons as usize);
        det_fill(slice(x, n_neurons as usize), 0x1234_5678, -1.0, 1.0);
        let y = alloc_f32(n_neurons as usize);

        let macs_small_k = sparse_forward(&sc, x, y, k as usize);
        // 判据 T2-a：实做乘加数 == n_neurons * k（掩码恰好 k 个/神经元）
        let want_macs = n_neurons as usize * k as usize;
        check(
            "T2-a sparse_forward macs == n*k",
            macs_small_k == want_macs,
            &format!("macs={} want={}", macs_small_k, want_macs),
        );

        // 判据 T2-b：掩码的每行激活数恰好 k
        let mut bad_rows = 0usize;
        for i in 0..n_neurons as usize {
            let mut c = 0usize;
            for j in 0..k0 as usize {
                if *sc.act.add(sc.idx(i, j)) != 0.0 {
                    c += 1;
                }
            }
            if c != k as usize {
                bad_rows += 1;
            }
        }
        check_count("T2-b act mask has exactly k per row", bad_rows, "");

        // 判据 T2-c：全激活（k = k0）时，稀疏前向 == 稠密前向（逐位）
        //   这是「稀疏 = 稠密 加上一个掩码」的正面校准。
        //   用**独立的**稠密循环（遍历 k0 全部连接）算参考。
        let y_dense = alloc_f32(n_neurons as usize);
        sparse_forward(&sc, x, y_dense, k0 as usize);
        let mut dense_bad = 0usize;
        for i in 0..n_neurons as usize {
            let b = i * k0 as usize;
            let mut acc = 0.0f32;
            let mut first = true;
            for j in 0..k0 as usize {
                let sid = *sc.src.add(b + j) as usize;
                let term = *sc.w.add(b + j) * *x.add(sid);
                acc = if first { term } else { acc + term };
                first = false;
            }
            if (*y_dense.add(i)).to_bits() != acc.to_bits() {
                dense_bad += 1;
            }
        }
        check_count(
            "T2-c k=k0 sparse == dense (bitwise)",
            dense_bad,
            "(independent dense loop)",
        );

        // ---------------------------------------------------------------
        // T3: 反向 —— 只对激活连接算梯度
        // ---------------------------------------------------------------
        // ⚠️ ORDERING IS LOAD-BEARING (this was a real harness bug, caught by
        // the harness itself): the T2-c calibration above ran the forward with
        // k = k0, which makes EVERY connection active. The mask is a single
        // mutable buffer, so it was still all-active when T3 ran -- and T3
        // then reported `written=1280` (all connections) instead of 512, and
        // T3-c's "inactive gradients are zero" PASSED VACUOUSLY because there
        // were no inactive connections left to check. So: re-establish the
        // k < k0 mask before every claim that depends on k < k0.
        sparse_forward(&sc, x, y, k as usize);

        let dy = alloc_f32(n_neurons as usize);
        det_fill(slice(dy, n_neurons as usize), 0xDEAD_BEEF, -1.0, 1.0);
        let dw = alloc_f32(e);
        std::ptr::write_bytes(dw as *mut u8, 0, e * 4);
        let dw_dense = alloc_f32(n_neurons as usize * n_neurons as usize);
        std::ptr::write_bytes(
            dw_dense as *mut u8,
            0,
            n_neurons as usize * n_neurons as usize * 4,
        );

        let wrote = sparse_gemm_bwd_w(&sc, dy, x, dw);
        check(
            "T3-a bwd writes only active conns",
            wrote == want_macs,
            &format!("written={} want={}", wrote, want_macs),
        );

        // 判据 T3-b：与**结构不同**的稠密展开参考对账（计数不一致项）
        dense_ref_bwd_w(&sc, dy, x, dw_dense);
        let mism = bwd_mismatch(&sc, dw, dw_dense);
        check_count(
            "T3-b sparse bwd == dense-expanded ref",
            mism,
            "(different loop structure)",
        );

        // 判据 T3-c：**非激活连接的梯度严格为 0**
        //   这是本模块唯一的算法差异，所以它必须有一条自己的判据。
        let mut bad_zero = 0usize;
        for t in 0..e {
            if *sc.act.add(t) == 0.0 && *dw.add(t) != 0.0 {
                bad_zero += 1;
            }
        }
        check_count("T3-c inactive conns have exactly 0 grad", bad_zero, "");

        // ---- can-fail 对照 ----
        // 故意把**一个**非激活位置的梯度写成非零，看判据是否报出来
        // （bad_zero 必须恰好变成 1）。若判据是恒真的，这里会报 0。
        //
        // ⚠️ THE PREMISE MUST BE CHECKED, NOT ASSUMED (first run: this control
        // reported control_bad=0 and looked like a broken criterion, when in
        // fact the mask was all-active -- there was no inactive connection to
        // pollute). So assert the premise separately, as its own check.
        let mut n_inactive = 0usize;
        for t in 0..e {
            if *sc.act.add(t) == 0.0 {
                n_inactive += 1;
            }
        }
        check(
            "T3-c premise: inactive connections exist",
            n_inactive > 0 && n_inactive == e - want_macs,
            &format!("inactive={} expected={}", n_inactive, e - want_macs),
        );
        let mut probe_idx = usize::MAX;
        for t in 0..e {
            if *sc.act.add(t) == 0.0 {
                probe_idx = t;
                break;
            }
        }
        let mut control_bad = 0usize;
        if probe_idx != usize::MAX {
            let orig = *dw.add(probe_idx);
            *dw.add(probe_idx) = 1.0; // 人为污染
            for t in 0..e {
                if *sc.act.add(t) == 0.0 && *dw.add(t) != 0.0 {
                    control_bad += 1;
                }
            }
            *dw.add(probe_idx) = orig;
        }
        check(
            "T3-c can-fail control (pollute 1 inactive grad)",
            control_bad == 1,
            &format!("control_bad={} (want exactly 1)", control_bad),
        );

        // ---------------------------------------------------------------
        // T2 严格形式：未激活权重**逐位冻结**
        // ---------------------------------------------------------------
        //
        // ⚠️ THE HARD PART HERE IS THE *CRITERION*, NOT THE CODE (the first
        // version of this check was wrong and the harness caught it: it
        // reported moved_inactive=414 while the "can-fail" control reported
        // 420, i.e. the check could barely distinguish the correct
        // implementation from the buggy one).
        //
        // WHY IT WAS WRONG: the activation mask is recomputed from
        // s = |w*x| at every step, and top-k SELECTION DEPENDS ON w. So as
        // Adam moves the active weights, the identity of the top-k set
        // churns, and a connection that was inactive at step t can become
        // active at step t+1 (at which point it is *supposed* to move).
        // Comparing "final mask" against "weights at step 0" therefore
        // measures CHURN, not "frozen", and it can never be 0.
        //
        // THE CORRECT CRITERION: at each step, record which positions the
        // step actually touched (mask BEFORE the update), and require that
        // every position NEVER touched across all steps is bitwise unchanged.
        // That is the honest statement of "non-activated connections are not
        // updated", and it is what the implementation actually promises.
        let w_before = alloc_f32(e);
        for t in 0..e {
            *w_before.add(t) = *sc.w.add(t);
        }
        // ever_active[t] = 1 if t was activated in ANY step.
        let ever = vec![0u8; e];
        let ever_p = ever.as_ptr() as *mut u8;
        let m_adam = alloc_f32(e);
        let v_adam = alloc_f32(e);
        std::ptr::write_bytes(m_adam as *mut u8, 0, e * 4);
        std::ptr::write_bytes(v_adam as *mut u8, 0, e * 4);
        for st in 1..=20u32 {
            sparse_forward(&sc, x, y, k as usize);
            for t in 0..e {
                if *sc.act.add(t) != 0.0 {
                    *ever_p.add(t) = 1;
                }
            }
            std::ptr::write_bytes(dw as *mut u8, 0, e * 4);
            sparse_gemm_bwd_w(&sc, dy, x, dw);
            sparse_adam_step(
                sc.w, dw, m_adam, v_adam, sc.act, e, lr, 0.9, 0.999, 1e-8, st,
            );
        }
        let mut moved_never_active = 0usize;
        let mut moved_ever_active = 0usize;
        let mut n_never_active = 0usize;
        for t in 0..e {
            let changed = (*sc.w.add(t)).to_bits() != (*w_before.add(t)).to_bits();
            if *ever_p.add(t) == 0 {
                n_never_active += 1;
                if changed {
                    moved_never_active += 1;
                }
            } else if changed {
                moved_ever_active += 1;
            }
        }
        check_count(
            "T2-d never-activated weights bitwise frozen (20 Adam steps)",
            moved_never_active,
            &format!(
                "(never_active={} of E={}; must be 0)",
                n_never_active, e
            ),
        );
        check(
            "T2-d positive control: activated weights DID move",
            moved_ever_active > 0,
            &format!(
                "moved_ever_active={} (must be >0, else the test is vacuous)",
                moved_ever_active
            ),
        );
        // 判据必须是**非空**的：如果 never_active 是 0，上面那条 0 是平凡的。
        check(
            "T2-d criterion is non-vacuous (some conns never activated)",
            n_never_active > 0,
            &format!("never_active={} (must be >0)", n_never_active),
        );

        // ---- can-fail 对照：用**稠密** Adam（不按掩码跳过）----
        // 用**同一判据**（never-active 位置是否逐位不动）对照。
        // 预期：> 0 ⇒ 证明 T2-d 不是恒真，且证明「只在反向不写 dw」**不够**，
        // 必须优化器侧也按掩码跳过。
        let w_before2 = alloc_f32(e);
        for t in 0..e {
            *w_before2.add(t) = *sc.w.add(t);
        }
        let ever2 = vec![0u8; e];
        let ever2_p = ever2.as_ptr() as *mut u8;
        let m2 = alloc_f32(e);
        let v2 = alloc_f32(e);
        for t in 0..e {
            *m2.add(t) = *m_adam.add(t);
            *v2.add(t) = *v_adam.add(t);
        }
        for st in 21..=40u32 {
            sparse_forward(&sc, x, y, k as usize);
            for t in 0..e {
                if *sc.act.add(t) != 0.0 {
                    *ever2_p.add(t) = 1;
                }
            }
            std::ptr::write_bytes(dw as *mut u8, 0, e * 4);
            sparse_gemm_bwd_w(&sc, dy, x, dw);
            // 稠密 Adam：不做掩码判断（这正是「如果忘了按掩码跳过」的形态）
            let tf = st as f32;
            let c1 = 1.0 - libm_pow(0.9, tf);
            let c2 = 1.0 - libm_pow(0.999, tf);
            for t in 0..e {
                let g = *dw.add(t);
                let mi = 0.9 * *m2.add(t) + 0.1 * g;
                let vi = 0.999 * *v2.add(t) + 0.001 * g * g;
                *m2.add(t) = mi;
                *v2.add(t) = vi;
                *sc.w.add(t) -= lr * (mi / c1) / (libm_sqrt(vi / c2) + 1e-8);
            }
        }
        let mut dense_moved_never = 0usize;
        for t in 0..e {
            if *ever2_p.add(t) == 0 && (*sc.w.add(t)).to_bits() != (*w_before2.add(t)).to_bits() {
                dense_moved_never += 1;
            }
        }
        check(
            "T2-d can-fail control (dense Adam instead of masked)",
            dense_moved_never > 0,
            &format!(
                "dense_moved_never_active={} (must be >0 for T2-d to be meaningful)",
                dense_moved_never
            ),
        );

        // ---------------------------------------------------------------
        // k0 与 conn_sparse 落盘格式的往返（T1 的产物必须能被 P16 的格式承载）
        // ---------------------------------------------------------------
        let layout = ConnSparseLayout::tight(n_neurons, k0, 0);
        let sec_bytes = layout.section_bytes();
        let sec = malloc(sec_bytes + 64);
        // 用 conn_sparse 自己的 writer（它按 neuron_conn_weight_bits 写占位权重）
        // —— 这里只为验「训练侧的 src 拓扑与落盘格式的 offsets 一致」。
        let n_written = vmlm_core::conn_sparse::write_section(&layout, sec);
        check(
            "T1-c section size matches the formula",
            n_written == sec_bytes,
            &format!(
                "write_section={} section_bytes={} (n={} k0={})",
                n_written, sec_bytes, n_neurons, k0
            ),
        );
        // 落盘 offsets 与训练侧 i*k0 的**独立**推导一致
        let mut off_bad = 0usize;
        for i in 0..n_neurons as usize {
            let off_i = vmlm_core::conn_sparse::get_u32(sec, layout.offsets_off() + i * 4);
            if off_i as usize != i * k0 as usize {
                off_bad += 1;
            }
        }
        check_count(
            "T1-c CSR offsets == i*k0 (independent derivation)",
            off_bad,
            "",
        );

        // ---------------------------------------------------------------
        // T2/T3 训练闭环：loss 必须下降（单层小例子）
        // ---------------------------------------------------------------
        // 任务：把一个固定的随机目标 y* 拟合出来（MSE）。
        // 这不是「训练一个语言模型」，而是「证明稀疏反向的梯度方向正确、
        // 且优化器真的在降低损失」。用 MSE 而不是 CE，是因为单层无词表时
        // CE 的目标定义是任意的，而 MSE 的目标是显式的、可独立验算的。
        let y = alloc_f32(n_neurons as usize);
        let ytar = alloc_f32(n_neurons as usize);
        det_fill(slice(ytar, n_neurons as usize), 0x5EED_5EED, -0.5, 0.5);
        let dyd = alloc_f32(n_neurons as usize);
        let dw2 = alloc_f32(e);
        let m3 = alloc_f32(e);
        let v3 = alloc_f32(e);
        std::ptr::write_bytes(m3 as *mut u8, 0, e * 4);
        std::ptr::write_bytes(v3 as *mut u8, 0, e * 4);

        let mut first_loss = 0.0f32;
        let mut last_loss = 0.0f32;
        let mut hist: Vec<f32> = Vec::new();
        for st in 1..=steps {
            // 激活区每次运行前清零（P13 前提，P17 纪律）
            std::ptr::write_bytes(y as *mut u8, 0, n_neurons as usize * 4);
            sparse_forward(&sc, x, y, k as usize);
            // MSE 前向损失（用独立的循环算，不用引擎内核，便于对照）
            let mut l = 0.0f32;
            for i in 0..n_neurons as usize {
                let d = *y.add(i) - *ytar.add(i);
                l += d * d;
            }
            l /= n_neurons as f32;
            if st == 1 {
                first_loss = l;
            }
            last_loss = l;
            if st % 100 == 0 || st == 1 {
                hist.push(l);
            }
            // dL/dy = 2(y - t)/n
            let kn = 2.0 / n_neurons as f32;
            for i in 0..n_neurons as usize {
                *dyd.add(i) = kn * (*y.add(i) - *ytar.add(i));
            }
            std::ptr::write_bytes(dw2 as *mut u8, 0, e * 4);
            sparse_gemm_bwd_w(&sc, dyd, x, dw2);
            sparse_adam_step(sc.w, dw2, m3, v3, sc.act, e, lr, 0.9, 0.999, 1e-8, st);
        }
        let drop = first_loss - last_loss;
        let ratio = last_loss / first_loss;
        check(
            "T2-e loss decreases (single layer, sparse bwd)",
            last_loss < first_loss,
            &format!(
                "first={:.6e} last={:.6e} drop={:.6e} ratio={:.4}",
                first_loss, last_loss, drop, ratio
            ),
        );
        println!(
            "  loss curve (every 100 steps): {}",
            hist.iter()
                .map(|v| format!("{:.6}", v))
                .collect::<Vec<_>>()
                .join(" ")
        );

        // ---- can-fail 对照：把梯度整体取负（上升方向）----
        // 用同一份初始权重复现一遍，但把 dyd 取反 —— loss 必须**上升**。
        // 如果它照样下降，说明「loss 下降」不是由梯度驱动的。
        let base2 = malloc(SparseConnW::bytes(n_neurons, k0));
        let sc2 = SparseConnW::from_base(base2, n_neurons, k0, 0);
        init_sparse_conn(&sc2);
        let m4 = alloc_f32(e);
        let v4 = alloc_f32(e);
        std::ptr::write_bytes(m4 as *mut u8, 0, e * 4);
        std::ptr::write_bytes(v4 as *mut u8, 0, e * 4);
        let dw4 = alloc_f32(e);
        let y2 = alloc_f32(n_neurons as usize);
        for st in 1..=steps {
            sparse_forward(&sc2, x, y2, k as usize);
            let kn = 2.0 / n_neurons as f32;
            for i in 0..n_neurons as usize {
                *dyd.add(i) = -kn * (*y2.add(i) - *ytar.add(i)); // 取负
            }
            std::ptr::write_bytes(dw4 as *mut u8, 0, e * 4);
            sparse_gemm_bwd_w(&sc2, dyd, x, dw4);
            sparse_adam_step(sc2.w, dw4, m4, v4, sc2.act, e, lr, 0.9, 0.999, 1e-8, st);
        }
        let mut l2 = 0.0f32;
        for i in 0..n_neurons as usize {
            let d = *y2.add(i) - *ytar.add(i);
            l2 += d * d;
        }
        l2 /= n_neurons as f32;
        check(
            "T2-e can-fail control (negated grad => loss must rise)",
            l2 > first_loss,
            &format!("negated_grad_loss={:.6e} vs first={:.6e}", l2, first_loss),
        );

        // ---------------------------------------------------------------
        // 训练后统计（任务书要求）
        // ---------------------------------------------------------------
        let histbuf = malloc((k0 as usize + 1) * 4) as *mut u32;
        let st1 = conn_stats(&sc, 1e-12, histbuf);
        println!(
            "  stats: n={} k0={} effective={} zero_w={} dead={} ({:.4}%)",
            st1.n,
            st1.k0,
            st1.effective,
            st1.zero_w,
            st1.dead,
            100.0 * st1.dead as f32 / st1.n as f32
        );
        // 判据：有效连接数 + 零权重数 == 总数（划分完整性）
        check(
            "STATS effective + zero_w == E",
            st1.effective + st1.zero_w == e as u32,
            &format!("effective={} zero_w={} E={}", st1.effective, st1.zero_w, e),
        );
        // 判据：直方图求和 == n
        let mut hsum = 0u32;
        for c in 0..=k0 as usize {
            hsum += *histbuf.add(c);
        }
        check(
            "STATS histogram sums to n",
            hsum == n_neurons,
            &format!("sum={} n={}", hsum, n_neurons),
        );
        // can-fail：把 eps 抬到 1e9 ⇒ 所有连接都算无效 ⇒ effective=0, dead=n
        let st2 = conn_stats(&sc, 1e9, histbuf);
        check(
            "STATS can-fail control (eps=1e9 => all ineffective)",
            st2.effective == 0 && st2.dead == n_neurons,
            &format!("effective={} dead={}", st2.effective, st2.dead),
        );

        // ---------------------------------------------------------------
        // top-k 选择的对拍（与全排序参考）
        // ---------------------------------------------------------------
        let mut hit_a = vec![0usize; k0 as usize];
        let mut hit_b = vec![0usize; k0 as usize];
        let mut sel_bad = 0usize;
        for i in 0..n_neurons as usize {
            let b = i * k0 as usize;
            let na = row_topk(
                sc.w.add(b),
                sc.src.add(b),
                x,
                k0 as usize,
                k as usize,
                &mut hit_a,
            );
            let nb = ref_topk_sorted(
                sc.w.add(b),
                sc.src.add(b),
                x,
                k0 as usize,
                k as usize,
                &mut hit_b,
            );
            if na != nb {
                sel_bad += 1;
                continue;
            }
            for q in 0..na {
                if hit_a[q] != hit_b[q] {
                    sel_bad += 1;
                    break;
                }
            }
        }
        check_count(
            "TOPK row_topk == ref_topk_sorted (independent algorithm)",
            sel_bad,
            "",
        );
        // can-fail：用一个错的 k0 宽度去比，必须不一致
        let mut sel_bad2 = 0usize;
        for i in 0..n_neurons as usize {
            let b = i * k0 as usize;
            let na = row_topk(
                sc.w.add(b),
                sc.src.add(b),
                x,
                k0 as usize,
                k as usize,
                &mut hit_a,
            );
            let nb = ref_topk_sorted(
                sc.w.add(b),
                sc.src.add(b),
                x,
                (k0 - 1) as usize,
                k as usize,
                &mut hit_b,
            );
            if na != nb {
                sel_bad2 += 1;
                continue;
            }
            for q in 0..na {
                if hit_a[q] != hit_b[q] {
                    sel_bad2 += 1;
                    break;
                }
            }
        }
        check(
            "TOPK can-fail control (wrong k0 => mismatch)",
            sel_bad2 > 0,
            &format!("mismatch_rows={}", sel_bad2),
        );

        println!();
        println!("===== checks={} fail={} =====", N_CHECK, N_FAIL);
        let rc = if N_FAIL == 0 { 0 } else { 1 };
        std::process::exit(rc);
    }
}
