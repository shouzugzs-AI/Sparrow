//! [抽库 L3-2] Head 块（从 `train/src/b1_full_train.rs` 逐字节搬运）。
//!
//! 搬运范围（`_opt/l3b_pre_src/b1_full_train.rs` 行号）：`par_head_fwd`(727) /
//! `par_head_bwd`(767) / `head_fwd_wt_one`(824) / `par_head_fwd_wt`(841) /
//! `repack_head_wt`(885) / `HBUF`+`hbuf`(907-926) / `head_bwd_batched`(928)。
//!
//! 逐字外改动（全部记账）：
//!   * 可见性 `unsafe fn` → `pub unsafe fn`（6 处）；`hbuf`/`HBUF` 保持私有（只在块内用）
//!   * 头部新增 3 行 `use`（`crate::util::*` / `vmlm_ops::conn::SparseConnW` /
//!     `vmlm_ops::par::par_for`）
//!   * **零路径改写**：`vmlm_ops::kernels::head_*_avx2` 等路径逐字未动 ⇒ 与抽库前逐位一致
//!
//! `repack_head_wt` 的契约（原注释保留）：**每一个改过 `head.w` 的地方之后都必须调用**
//! （Adam、检查点载入、gradcheck 的 FD 扰动），否则前向读到旧权重。
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]

use vmlm_blocks::util::*;
use vmlm_ops::conn::SparseConnW;
use vmlm_ops::par::par_for;

/// [P19-7] 块对齐 head（`src[v][j] = (v%nb)*k0e + j`）的每 token 前向。
/// 与 `par_b1_forward` 同形（nt<=1 串行；否则按 token 切并行）。
#[allow(clippy::too_many_arguments)]
pub unsafe fn par_head_fwd(
    sc: &SparseConnW,
    x: *const f32,
    y: *mut f32,
    t: usize,
    xs: usize,
    ys: usize,
    nt: usize,
    nb: usize,
    k0e: usize,
) {
    let nv = sc.n_neurons as usize;
    if nt <= 1 {
        for i in 0..t {
            vmlm_ops::kernels::head_block_fwd_avx2(sc.w, x.add(i * xs), y.add(i * ys), nv, nb, k0e, sc.k0);
        }
        return;
    }
    let xp = SendPtr(x as *mut f32);
    let yp = SendPtr(y);
    let scp = SendPtr(sc as *const SparseConnW as *mut SparseConnW);
    par_for(t, nt, move |lo, hi| {
        let (xp, yp, scp) = (xp, yp, scp);
        for i in lo..hi {
            vmlm_ops::kernels::head_block_fwd_avx2(
                (*scp.0).w,
                xp.0.add(i * xs),
                yp.0.add(i * ys),
                nv,
                nb,
                k0e,
                (*scp.0).k0,
            );
        }
    });
}

/// [P19-7] 块对齐 head 的每 token 反向（dw/dx 一次遍历，无 gather/无散写）。
/// `dw` 为 [n_neurons][k0] 行主序（= head_dw 的布局），`dx` 为块主序（= h 的布局）；
/// 两者都必须由调用方预先清零。
#[allow(clippy::too_many_arguments)]
pub unsafe fn par_head_bwd(
    sc: &SparseConnW,
    dy: *const f32,
    x: *const f32,
    dw: *mut f32,
    dx: *mut f32,
    t: usize,
    xs: usize,
    ys: usize,
    nt: usize,
    nb: usize,
    k0e: usize,
) {
    let nv = sc.n_neurons as usize;
    if nt <= 1 {
        for i in 0..t {
            vmlm_ops::kernels::head_block_bwd_avx2(
                sc.w,
                x.add(i * xs),
                dy.add(i * ys),
                dw,
                dx.add(i * xs),
                nv,
                nb,
                k0e,
                sc.k0,
            );
        }
        return;
    }
    let scp = SendPtr(sc as *const SparseConnW as *mut SparseConnW);
    let dyp = SendPtr(dy as *mut f32);
    let xp = SendPtr(x as *mut f32);
    let dxp = SendPtr(dx);
    let dwp2 = SendPtr(dw);
    par_for(t, nt, move |lo, hi| {
        let (scp, dyp, xp, dxp, dwp2) = (scp, dyp, xp, dxp, dwp2);
        for i in lo..hi {
            vmlm_ops::kernels::head_block_bwd_avx2(
                (*scp.0).w,
                xp.0.add(i * xs),
                dyp.0.add(i * ys),
                dwp2.0,
                dxp.0.add(i * xs),
                nv,
                nb,
                k0e,
                (*scp.0).k0,
            );
        }
    });
}

/// [P19-8] 块对齐 head 的每 token 前向（WT 列主序版）：
/// 先把该 token 的 h 转置进小块 `ht`，再按 8 行/ymm 扫 logits。
#[allow(clippy::too_many_arguments)]
pub unsafe fn head_fwd_wt_one(
    wt: *const f32,
    h: *const f32,
    y: *mut f32,
    ht: *mut f32,
    nv: usize,
    nb: usize,
    k0e: usize,
    k0h: usize,
) {
    vmlm_ops::kernels::head_pack_ht_avx2(h, ht, nb, k0e, k0h);
    vmlm_ops::kernels::head_block_fwd_wt_avx2(wt, ht, y, nv, nb, k0h);
}

/// 并行版：每个 token 用**私有**的 ht（避免 8 个 worker 抢同一块 2KB 缓冲）。
#[allow(clippy::too_many_arguments)]
pub unsafe fn par_head_fwd_wt(
    wt: *const f32,
    x: *const f32,
    y: *mut f32,
    ht_all: *mut f32,
    t: usize,
    xs: usize,
    ys: usize,
    nt: usize,
    nv: usize,
    nb: usize,
    k0e: usize,
    k0h: usize,
) {
    let ht_row = k0h * nb;
    if nt <= 1 {
        for i in 0..t {
            head_fwd_wt_one(wt, x.add(i * xs), y.add(i * ys), ht_all.add(i * ht_row), nv, nb, k0e, k0h);
        }
        return;
    }
    let wtp = SendPtr(wt as *mut f32);
    let xp = SendPtr(x as *mut f32);
    let yp = SendPtr(y);
    let hp = SendPtr(ht_all);
    par_for(t, nt, move |lo, hi| {
        let (wtp, xp, yp, hp) = (wtp, xp, yp, hp);
        for i in lo..hi {
            head_fwd_wt_one(
                wtp.0,
                xp.0.add(i * xs),
                yp.0.add(i * ys),
                hp.0.add(i * ht_row),
                nv,
                nb,
                k0e,
                k0h,
            );
        }
    });
}

/// [P19-8] 重打包 head / head2 的 WT 列主序副本。
/// **必须在每一个改过 head.w 的地方之后调用**（Adam、检查点载入、gradcheck 的 FD 扰动），
/// 否则前向会读到旧权重 —— 实测 gradcheck 的 head 组 6 个探针会全部 OVER（FD 恒为 0）。
pub unsafe fn repack_head_wt(
    head_blocked: bool,
    head_w: *mut f32,
    head_wt: *mut f32,
    head2_w: *mut f32,
    head2_wt: *mut f32,
    head_skip: bool,
    n_vocab: usize,
    k0_head: usize,
) {
    if !head_blocked {
        return;
    }
    vmlm_ops::kernels::head_pack_wt_avx2(head_w, head_wt, n_vocab, k0_head);
    if head_skip {
        vmlm_ops::kernels::head_pack_wt_avx2(head2_w, head2_wt, n_vocab, k0_head);
    }
}

// [P21-4] head 反向**按 token 批量**：dlogits 与 head 输入各转置一次，
// 然后在 token 维上做 GEMM（dW[v][j] = Σ_t dY[v][t]·Ht[b*k0e+j][t]）。
// 逐 token 版每 token 都要把 dw[nv*k0h] 整块 RMW 一遍；批量版只写一遍。
// 实测（kbench --mode hbatch，nv=8198/dim=2048/tt=128，**含 3 次转置**）：
//   逐 token 3.884 ms/round → 批量 2.537 ms/round = 1.53x；dw 比值 1.00000、dh **逐位相同**。
thread_local! {
    static HBUF: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// 3 份 scratch：dyT[nv*t8]、htT[dim*t8]、dht[dim*t8]。
fn hbuf(nv_tt: usize, dim_tt: usize) -> (*mut f32, *mut f32, *mut f32) {
    HBUF.with(|c| {
        let mut v = c.borrow_mut();
        if v.len() < nv_tt + 2 * dim_tt {
            v.resize(nv_tt + 2 * dim_tt, 0.0);
        }
        let p = v.as_mut_ptr();
        unsafe { (p, p.add(nv_tt), p.add(nv_tt + dim_tt)) }
    })
}

/// 块对齐 head 的批量反向（`dw_out`/`dh_out` 由调用方清零；`dh_out` 逐位被赋值）。
#[allow(clippy::too_many_arguments)]
pub unsafe fn head_bwd_batched(
    w: *const f32,
    w_row_major: *const f32,
    dlogits: *const f32,
    h: *const f32,
    tt: usize,
    dim: usize,
    n_vocab: usize,
    nb: usize,
    k0_embed: usize,
    k0_head: usize,
    dw_out: *mut f32,
    dh_out: *mut f32,
) {
    // [P21-4b] **分块**：一次转置整个 [tt][nv] 会把 4 MB dlogits 推进 DRAM
    // （实测端到端反而 +3%）。改成每 32 个 token 一块：转置后的 dY 只有 nv*32*4 = 1 MB、
    // Ht 262 KB，**留在 L2**；dw 每块累加一次（128 token 从 128 遍 RMW 降到 4 遍）。
    let chunk = 32usize;
    let mut c0 = 0usize;
    while c0 < tt {
        let c = chunk.min(tt - c0);
        let c8 = (c + 7) & !7usize;
        let (dyt, htt, dht) = hbuf(n_vocab * c8, dim * c8);
        vmlm_ops::kernels::transpose_t8_avx2(dlogits.add(c0 * n_vocab), c, n_vocab, dyt, c8);
        vmlm_ops::kernels::transpose_t8_avx2(h.add(c0 * dim), c, dim, htt, c8);
        for i in 0..dim * c8 {
            *dht.add(i) = 0.0;
        }
        vmlm_ops::kernels::head_bwd_batched_avx2(
            w, htt, dyt, c8, n_vocab, nb, k0_embed, k0_head, dw_out, dht,
        );
        vmlm_ops::kernels::transpose_t8_back_avx2(dht, dim, c, dh_out.add(c0 * dim), dim, c8);
        c0 += c;
    }
    let _ = w_row_major;
}
