//! [抽库 L3-4] attention 稀疏投影包装（消融二）：Q/K/V/O 走 `b1_forward`，
//! 核心复用同一个 `attn_core`。从 `train/src/b1_full_train.rs` 逐字节搬运。
//!
//! 搬运范围（`_opt/l3c_src/b1_full_train.rs` 行号）：`attn_backward_sparse_par`(315) /
//! `attn_forward_sparse`(423) / `attn_backward_sparse`(449)。
//!
//! 逐字外改动：可见性 `unsafe fn` → `pub unsafe fn`（3 处）、头部新增 5 行 `use`、
//! **零路径改写**。这三个函数原先只差 `par_b1_forward`（L3-3 已搬入 `crate::b1`），
//! 所以放在最后一批。
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]

use vmlm_blocks::attn::{attn_core_bwd, attn_core_bwd_par, attn_core_par};
use crate::b1::par_b1_forward;
use vmlm_blocks::util::*;
use vmlm_ops::b1_pure::b1_backward_fused;
use vmlm_ops::conn::SparseConnW;
use vmlm_ops::par::par_for;

/// 并行版 attn_backward_sparse：O 投影与 Q/K/V 投影都改成"每线程私有 dw 分段 + 归约"。
/// `dwp` 至少 nt*dim*k0_attn 个 f32。
#[allow(clippy::too_many_arguments)]
// [抽库 L3-4] 强制内联：跨 crate 后不再内联 ⇒ 多出独立副本（实测 +512 B）；
// `#[inline(always)]` 恢复抽库前的内联决策 ⇒ −512 B。
#[inline(always)]
pub unsafe fn attn_backward_sparse_par(
    h: *const f32,
    wq: &SparseConnW,
    wk: &SparseConnW,
    wv: &SparseConnW,
    wo: &SparseConnW,
    dout: *const f32,
    t: usize,
    d: usize,
    q: *const f32,
    kk: *const f32,
    v: *const f32,
    o: *const f32,
    prob: *const f32,
    dq: *mut f32,
    dk: *mut f32,
    dv: *mut f32,
    adx: *mut f32,
    dobuf: *mut f32,
    dprob: *mut f32,
    rowsum: *mut f32,
    dwq: *mut f32,
    dwk: *mut f32,
    dwv: *mut f32,
    dwo: *mut f32,
    nt: usize,
    dwp: *mut f32,
) {
    if nt <= 1 {
        attn_backward_sparse(h, wq, wk, wv, wo, dout, t, d, q, kk, v, o, prob, dq, dk, dv, adx, dobuf,
                             dprob, dwq, dwk, dwv, dwo);
        return;
    }
    let m = wq.e as usize;
    // O 投影：dobuf 逐 token 私有；dwo 每线程分段
    zero(dobuf, t * d);
    zero(dwp, nt * m);
    {
        let (dop, oo, doutp, dwpp) = (
            SendPtr(dobuf),
            SendPtr(o as *mut f32),
            SendPtr(dout as *mut f32),
            SendPtr(dwp),
        );
        let wop = SendPtr(wo as *const SparseConnW as *mut SparseConnW);
        par_for(t, nt, move |lo, hi| {
            let (dop, oo, doutp, dwpp, wop) = (dop, oo, doutp, dwpp, wop);
            let tid = (lo * nt / t.max(1)).min(nt - 1);
            let slot = dwpp.0.add(tid * m);
            for i in lo..hi {
                b1_backward_fused(&*wop.0, doutp.0.add(i * d), oo.0.add(i * d), slot, dop.0.add(i * d));
            }
        });
    }
    for tid in 0..nt {
        let slot = dwp.add(tid * m);
        for j in 0..m {
            *dwo.add(j) += *slot.add(j);
        }
    }
    attn_core_bwd_par(dobuf, prob, q, kk, v, t, d, dq, dk, dv, dprob, rowsum, nt);
    // Q/K/V 投影：adx 逐 token 私有；dw* 每线程分段
    for which in 0..3usize {
        let sc = match which {
            0 => wq as *const SparseConnW,
            1 => wk as *const SparseConnW,
            _ => wv as *const SparseConnW,
        };
        let dxsrc = match which {
            0 => dq,
            1 => dk,
            _ => dv,
        };
        let dw = match which {
            0 => dwq,
            1 => dwk,
            _ => dwv,
        };
        zero(dwp, nt * m);
        {
            let (adxp, hh, dsrc, dwpp) = (
                SendPtr(adx),
                SendPtr(h as *mut f32),
                SendPtr(dxsrc),
                SendPtr(dwp),
            );
            let scp = SendPtr(sc as *mut SparseConnW);
            par_for(t, nt, move |lo, hi| {
                let (adxp, hh, dsrc, dwpp, scp) = (adxp, hh, dsrc, dwpp, scp);
                let tid = (lo * nt / t.max(1)).min(nt - 1);
                let slot = dwpp.0.add(tid * m);
                for i in lo..hi {
                    b1_backward_fused(&*scp.0, dsrc.0.add(i * d), hh.0.add(i * d), slot, adxp.0.add(i * d));
                }
            });
        }
        for tid in 0..nt {
            let slot = dwp.add(tid * m);
            for j in 0..m {
                *dw.add(j) += *slot.add(j);
            }
        }
    }
}

/// 消融二：**稀疏投影**的 attention 前向 —— Q/K/V/O 各用 `b1_forward`
/// （每 token 只连 k0_attn 个源）。核心仍是同一个 `attn_core`。
#[allow(clippy::too_many_arguments)]
// [抽库 L3-4] 强制内联：跨 crate 后不再内联 ⇒ 多出独立副本（实测 +512 B）；
// `#[inline(always)]` 恢复抽库前的内联决策 ⇒ −512 B。
#[inline(always)]
pub unsafe fn attn_forward_sparse(
    h: *const f32,
    wq: &SparseConnW,
    wk: &SparseConnW,
    wv: &SparseConnW,
    wo: &SparseConnW,
    t: usize,
    d: usize,
    q: *mut f32,
    kk: *mut f32,
    v: *mut f32,
    o: *mut f32,
    prob: *mut f32,
    out: *mut f32,
    nt: usize,
) {
    par_b1_forward(wq, h, q, t, d, d, nt);
    par_b1_forward(wk, h, kk, t, d, d, nt);
    par_b1_forward(wv, h, v, t, d, d, nt);
    attn_core_par(q, kk, v, t, d, o, prob, nt);
    par_b1_forward(wo, o, out, t, d, d, nt);
}

/// 消融二：**稀疏投影**的 attention 反向。
/// O 的投影反向用 `b1_backward_fused`（它同时累加 dWo 的稀疏权重梯度），
/// 核心反向复用 `attn_core_bwd`，Q/K/V 的投影反向同样用 `b1_backward_fused` 并累加进 `adx`。
#[allow(clippy::too_many_arguments)]
// [抽库 L3-4] 强制内联：跨 crate 后不再内联 ⇒ 多出独立副本（实测 +512 B）；
// `#[inline(always)]` 恢复抽库前的内联决策 ⇒ −512 B。
#[inline(always)]
pub unsafe fn attn_backward_sparse(
    h: *const f32,
    wq: &SparseConnW,
    wk: &SparseConnW,
    wv: &SparseConnW,
    wo: &SparseConnW,
    dout: *const f32,
    t: usize,
    d: usize,
    q: *const f32,
    kk: *const f32,
    v: *const f32,
    o: *const f32,
    prob: *const f32,
    dq: *mut f32,
    dk: *mut f32,
    dv: *mut f32,
    adx: *mut f32,
    dobuf: *mut f32,
    dprob: *mut f32,
    dwq: *mut f32,
    dwk: *mut f32,
    dwv: *mut f32,
    dwo: *mut f32,
) {
    // ⚠️ 这里第四个参数必须是**梯度缓冲**。第一版我传成了 `wq.w`（权重本身），
    //    结果是"把梯度写进权重、梯度缓冲恒为 0"——FD 对账立刻报出来（attn_sq 探针全 0）。
    zero(dobuf, t * d);
    for i in 0..t {
        b1_backward_fused(wo, dout.add(i * d), o.add(i * d), dwo, dobuf.add(i * d));
    }
    attn_core_bwd(dobuf, prob, q, kk, v, t, d, dq, dk, dv, dprob);
    for i in 0..t {
        b1_backward_fused(wq, dq.add(i * d), h.add(i * d), dwq, adx.add(i * d));
        b1_backward_fused(wk, dk.add(i * d), h.add(i * d), dwk, adx.add(i * d));
        b1_backward_fused(wv, dv.add(i * d), h.add(i * d), dwv, adx.add(i * d));
    }
}
