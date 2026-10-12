//! [抽库 L3-3] 稀疏层块：t8（source-major）整栈 scratch 集群 + B1 层前向/反向。
//!
//! 搬运范围（`_opt/l3b_src/b1_full_train.rs` 行号）：t8 scratch 集群(612-686) /
//! `par_b1_forward`(688) / `par_b1_backward`(729)（逐字节）。
//!
//! 逐字外改动（全部记账）：
//!   * `pub`：`par_b1_forward` / `par_b1_backward` / `t8tok` / `t8s4` +
//!     开关 static `T8_ON` / `T8_FWD` / `T8_BWD`（trainer 侧读写它们）—— 共 7 处
//!   * 保持私有：`T8STACK_ON`、`t8_scratch`、`rsbuf`、4 个 thread_local（只在块内引用；
//!     `T8STACK_ON`/`rsbuf` 在抽库前就是未被引用的，警告状态不变）
//!   * 头部新增 4 行 `use`；[L4-2] 原语下移：`vmlm_ops::b1_pure::*` → `vmlm_ops::b1_pure::*`
//!     （同一份实现，只换 crate 位置；本次不再有 core 依赖）
//!
//! 注意 `t8tok`/`t8s4` 也被 trainer 的整栈路径直接调用，所以是 `pub`；
//! `T8S4`/`T8TOKB`/`T8BUF`/`RSBUF` 四个 thread_local 只在块内用，保持私有。
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]

use vmlm_blocks::util::*;
use vmlm_ops::b1_pure::{b1_backward, b1_backward_dx, b1_backward_fused, b1_forward, b1_layer_backward_t8, b1_layer_forward_t8};
use vmlm_ops::conn::SparseConnW;
use vmlm_ops::par::par_for;

// [P21-7] 第四组：整栈 source-major（t8）—— 只在 n_attn==0 且无 relu/identity 时启用，
// 生产配置（3 个注意力块）走原路径 ⇒ 对生产零风险。每步 4 次转置（入口/出口/反向入口/反向出口），
// **层间零转置**。kbench --mode t8stack 实测：12 层 3.62x@2048 / 2.09x@4096。
static T8STACK_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
thread_local! {
    static T8S4: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}
/// 边界层专用的 token-major 输入缓冲（**不能复用 y**：y 同时是层输出暂存，
/// `par_b1_forward(xin=y, y)` 会在跨神经元的 gather 里读到已被覆写的输入 ⇒ 前向错）。
thread_local! {
    static T8TOKB: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}
// [抽库 L3-3] 强制内联：跨 crate 后这两个 thread_local 取缓冲函数不再被内联
// ⇒ TLS 机制被 outline（实测 +3072 B）；`#[inline(always)]` 恢复抽库前的内联 ⇒ −512 B。
#[inline(always)]
pub fn t8tok(need: usize) -> *mut f32 {
    T8TOKB.with(|c| {
        let mut v = c.borrow_mut();
        if v.len() < need {
            v.resize(need, 0.0);
        }
        v.as_mut_ptr()
    })
}

/// 4 份 `dim*t8` scratch：ta（入口）、tb（出口）、dyc（反向 dy）、dxn（反向 dx）
// [抽库 L3-3] 强制内联：跨 crate 后这两个 thread_local 取缓冲函数不再被内联
// ⇒ TLS 机制被 outline（实测 +3072 B）；`#[inline(always)]` 恢复抽库前的内联 ⇒ −512 B。
#[inline(always)]
pub fn t8s4(need: usize) -> (*mut f32, *mut f32, *mut f32, *mut f32) {
    T8S4.with(|c| {
        let mut v = c.borrow_mut();
        if v.len() < need * 4 {
            v.resize(need * 4, 0.0);
        }
        let p = v.as_mut_ptr();
        unsafe { (p, p.add(need), p.add(2 * need), p.add(3 * need)) }
    })
}
// =====================================================================
// P20：t8（source-major）drop-in 路径的线程私有 scratch + 开关
// =====================================================================
// 为什么用 thread_local：t8 路径需要 3 份 `n*t8` 的临时缓冲，而每一步的每次调用
// 都在**同一个 worker 线程内串行**发生 ⇒ 每线程一份即可，不必进 step_fn 的 slot 体系
// （那样要改 8 个调用点的签名）。池 worker 是常驻线程，所以这块只分配/增长一次。
//
// 内存：inter 模式下每 worker 只需 `3 * n * tt`（生产配置 3×4096×256×4B = 12 MB），
// 主线程的 eval/标定路径 tt = t_tokens ⇒ 3×4096×2048×4B = 100 MB。合计 ~200 MB。
pub static T8_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub static T8_FWD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub static T8_BWD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
thread_local! {
    static T8BUF: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// 取本线程的 3 份 t8 scratch（不足则增长）。返回 (a, b, c)。
fn t8_scratch(need: usize) -> (*mut f32, *mut f32, *mut f32) {
    T8BUF.with(|c| {
        let mut v = c.borrow_mut();
        if v.len() < need * 3 {
            v.resize(need * 3, 0.0);
        }
        let p = v.as_mut_ptr();
        (p, unsafe { p.add(need) }, unsafe { p.add(2 * need) })
    })
}

thread_local! {
    static RSBUF: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// 行均值暂存（attention 反向 passB/C 分块内核用）
fn rsbuf(n: usize) -> *mut f32 {
    RSBUF.with(|c| {
        let mut v = c.borrow_mut();
        if v.len() < n {
            v.resize(n, 0.0);
        }
        v.as_mut_ptr()
    })
}

/// 每 token 一行的 B1 前向（并行；`xs`=输入行跨度, `ys`=输出行跨度）
#[allow(clippy::too_many_arguments)]
pub unsafe fn par_b1_forward(
    sc: &SparseConnW,
    x: *const f32,
    y: *mut f32,
    t: usize,
    xs: usize,
    ys: usize,
    nt: usize,
) {
    // [P20] t8（source-major）drop-in：转置 + 分块内核 + 转置回。
    // 只在 nt<=1（inter 模式，每 region 一个线程）时走 —— intra 的按 token 分段会让
    // 每段各自转置，收益被摊薄且要动 slot 体系。实测（含 2 次转置）1.83x。
    if nt <= 1 && T8_FWD.load(std::sync::atomic::Ordering::Relaxed) {
        let n = sc.n_neurons as usize;
        if n == xs && ys >= n {
            let t8 = (t + 7) & !7usize;
            let need = n * t8.max(1);
            let (a, b, _c) = t8_scratch(need);
            b1_layer_forward_t8(sc, x, xs, y, ys, t, a, b, t8.max(1));
            return;
        }
    }
    if nt <= 1 {
        for i in 0..t {
            b1_forward(sc, x.add(i * xs), y.add(i * ys));
        }
        return;
    }
    let scp = SendPtr(sc as *const SparseConnW as *mut SparseConnW);
    let xp = SendPtr(x as *mut f32);
    let yp = SendPtr(y);
    par_for(t, nt, move |lo, hi| {
        let (scp, xp, yp) = (scp, xp, yp); // 强制整体捕获（按字段捕获会拿到裸指针 ⇒ 不满足 Sync）
        for i in lo..hi {
            b1_forward(&*scp.0, xp.0.add(i * xs), yp.0.add(i * ys));
        }
    });
}

/// 每 token 一行的 B1 反向（并行 + 每线程私有 dw 分段后归约）。/// `dwp` 必须至少有 `nt * sc.e` 个 f32；`dw` 由调用方清零。
#[allow(clippy::too_many_arguments)]
pub unsafe fn par_b1_backward(
    sc: &SparseConnW,
    dy: *const f32,
    x: *const f32,
    dw: *mut f32,
    dx: *mut f32,
    t: usize,
    xs: usize,
    ys: usize,
    nt: usize,
    dwp: *mut f32,
    fusion: bool,
) {
    let m = sc.e as usize;
    // [P20] t8（source-major）drop-in 反向：实测（含 3 次转置）2.01x。
    // 只走 fusion 路径（t8 内核天然同时算 dw 与 dx）；--no-fusion 保留原 A/B 语义。
    if nt <= 1 && fusion && T8_BWD.load(std::sync::atomic::Ordering::Relaxed) {
        let n = sc.n_neurons as usize;
        if n == xs && ys >= n {
            let t8 = (t + 7) & !7usize;
            let need = n * t8.max(1);
            let (a, b, c) = t8_scratch(need);
            b1_layer_backward_t8(sc, dy, ys, x, xs, t, dw, dx, a, b, c, t8.max(1));
            return;
        }
    }
    if nt <= 1 {
        for i in 0..t {
            if fusion {
                b1_backward_fused(sc, dy.add(i * ys), x.add(i * xs), dw, dx.add(i * xs));
            } else {
                b1_backward(sc, dy.add(i * ys), x.add(i * xs), dw);
                b1_backward_dx(sc, dy.add(i * ys), dx.add(i * xs));
            }
        }
        return;
    }
    zero(dwp, nt * m);
    let scp = SendPtr(sc as *const SparseConnW as *mut SparseConnW);
    let dyp = SendPtr(dy as *mut f32);
    let xp = SendPtr(x as *mut f32);
    let dxp = SendPtr(dx);
    let dwpp = SendPtr(dwp);
    par_for(t, nt, move |lo, hi| {
        let (scp, dyp, xp, dxp, dwpp) = (scp, dyp, xp, dxp, dwpp);
        let tid = lo * nt / t.max(1);
        let tid = if t == 0 { 0 } else { tid.min(nt - 1) };
        let slot = dwpp.0.add(tid * m);
        for i in lo..hi {
            if fusion {
                b1_backward_fused(
                    &*scp.0,
                    dyp.0.add(i * ys),
                    xp.0.add(i * xs),
                    slot,
                    dxp.0.add(i * xs),
                );
            } else {
                b1_backward(&*scp.0, dyp.0.add(i * ys), xp.0.add(i * xs), slot);
                b1_backward_dx(&*scp.0, dyp.0.add(i * ys), dxp.0.add(i * xs));
            }
        }
    });
    // 归约（固定线程号顺序 ⇒ 同一线程数下可复现）
    for tid in 0..nt {
        let slot = dwp.add(tid * m);
        for j in 0..m {
            *dw.add(j) += *slot.add(j);
        }
    }
}
