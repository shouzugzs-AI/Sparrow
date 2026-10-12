//! [抽库 L3-1] attention **稠密**块（从 `train/src/b1_full_train.rs` 逐字节搬运）。
//!
//! 搬运范围（原文件行号）：`attn_core_row`(321) / `attn_core_par`(378) / `attn_core_bwd`(411) /
//! `attn_core_bwd_par`(470) / `attn_forward`(718) / `attn_backward`(745) / `attn_qk_stats`(859)。
//!
//! 逐字外改动（全部记账）：
//!   * 可见性 `unsafe fn` → `pub unsafe fn`（7 处；跨 crate 必需）
//!   * [L4-1] `vmlm_core::kernels::` → `vmlm_ops::kernels::`（17 处 = 16 调用 + 1 处 use）与
//!     `vmlm_core::interp::HAS_AVX2` → `vmlm_sys::cpu::HAS_AVX2`（1 处）：`HAS_AVX2` 合并成一份后
//!     两者指向同一个 static（L3-1 v1 曾因读到那份恒 false 的副本差 5 步 loss；缺陷修复后已安全）；
//!     `blocks` 因此**不再依赖 `vmlm-core`**。
//!   * 头部新增 3 行 `use`（`crate::util::*` / `vmlm_core::kernels::gemm_nt` / `vmlm_ops::par::par_for`）
//!   * `attn_forward` / `attn_backward` 加 `#[inline(never)]`（2 处）：跨 crate 后 LTO 把这
//!     两个大函数内联进了 `run_main` 的闭包。链接器符号表实测（`_opt/l3_diag/*.map`）：
//!     不加提示时 `run_main::{{closure}}` 61936→62624 B、b1-full 610816→**611840 B（+1024）**；
//!     加提示后闭包 →**49264 B**（那段代码成为独立函数 `vmlm_blocks::attn::attn_backward`
//!     12096 B 等），b1-full →**610304 B（比抽库前小 512 B）**。纯代码生成提示。
//!
//! **本批刻意不做路径改写**：搬运 v1 曾把 `vmlm_core::interp::HAS_AVX2` 改写成
//! `vmlm_sys::cpu::HAS_AVX2`，结果 100 步 loss 有 5 步差 1–2 ULP。根因是 `HAS_AVX2`
//! 有**两份 static**（`core/src/interp.rs:8` 被 `init_cpu()` 赋值；
//! `sys/src/cpu/mod.rs:33` 全仓库无人赋值 ⇒ 恒 false）。所以本版让 blocks 依赖
//! `vmlm-core`，`vmlm_core::*` 路径**逐字不变** ⇒ 与抽库前逐位一致。
//! （`sys::cpu::HAS_AVX2` 那份死副本是本次发现的**遗留缺陷**，修它会让 ops 内核的
//!  AVX2 分支从 false 翻到 true，必须单独一轮验证，不在本轮范围。）
//!
//! **本批未搬**（留给下一批）：`attn_forward_sparse` / `attn_backward_sparse` /
//! `attn_backward_sparse_par` —— 它们要 `par_b1_forward`（稀疏层块，下一批）。
//!
//! **P24-1（本文件唯一一次"改行为"的优化，也是唯一一次动内核结构）**：核心从
//! "一条 query 行一趟"改成**按 `ATTN_TILE_BI` 行分块（tile）**，消灭 K/V 重流。
//! 数值契约**逐位不变**，依据全部写在 `attn_tile_fwd` / `attn_tile_bwd_a` /
//! `attn_tile_bwd_dqdk` 的注释里：每个 (i,j) 仍是**同一个** `dot256_f32` 独立点积、
//! `mx` 与归约顺序无关、每行仍调用**同一个** `exp_neg_sum_avx2(prob+i·t, i+1, mx)`、
//! AV/反向加权和对每个元素仍按 j（或 i）**升序**累加，`if a == 0.0 / ds == 0.0` 的跳过照旧。
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]

use crate::util::*;
use vmlm_ops::kernels::gemm_nt;
use vmlm_ops::par::par_for;

/// [P24-1] attention 核心的**块高**：一次放进同一遍 K/V 扫描的 query 行数。
///
/// **为什么要分块**（P24-1 的诊断）：原核心是"一条 query 行一趟"——每条行都把整条 K、
/// 整条 V 从内存里重流一遍。d=1536 时一行 K/V 就是 6144 B，单行的 q 加上这一行 K 已经
/// 顶到 L1（32 KB）边缘，于是三角循环把 K/V **反复从 DRAM 拉回来**：每层每 slot 重扫
/// K ≈49.5 MB、V ≈49.5 MB。实测 dim1536/k0=8/24 层/T=128/8 slot：attention 相位
/// 0.893 s/步，MACs 4.83 GMAC/步 ⇒ 5.4 GMAC/s（AVX2 峰值的 3~6%），而流量 ≈19 GB/步
/// ÷ 0.893 s ≈ **21 GB/s = DRAM 带宽** ⇒ 确诊**内存带宽瓶颈**，不是算力瓶颈。
/// 对照：同样的代码在 d=112（行 448 B，全在 L1）时每 FLOP 速率相同 ⇒ 只有大 dim 才暴露。
///
/// **为什么 BI=4**：块内相位①的工作集 = BI 行 q(BI·6144 B) + 当前 1 行 k(6144 B)。
/// BI=4 ⇒ 30 KB，压在 L1(32 KB) 边缘：**第 j 行 K 每块只从 DRAM 取一次**，块内其余
/// 几行的点积在 L1 命中；相位③同理（V 行）。BI=8 ⇒ 54 KB 溢出到 L2：K/V 的 DRAM 流量
/// 还能再减半，但 q/orow 的复用掉一层（两档实测对比见报告）。做成常量便于继续调。
pub const ATTN_TILE_BI: usize = 4;

/// [P24-1] 前向分块核心：处理 query 行区间 `[lo, hi)`（`hi - lo <= ATTN_TILE_BI`）。
///
/// 三个相位的**分工与累加口径一律不动**，这是"逐位不变"的全部依据：
///   ① `prob[i*t+j] = dot256_f32(q + i*d, kk + j*d, d) * scale`：每个 (i,j) 仍是**独立点积**
///      （同一个跨 crate 内核、同一份 `d`），与循环顺序无关；但这里**必须 j 在外、i 在内**，
///      第 j 行 K 才只从 DRAM 取一次、被块内各行共享——这就是本次优化的全部收益来源。
///   ② 每行的 exp+Σ 仍调用**同一个** `exp_neg_sum_avx2(prob + i*t, i+1, mx)`，参数一字不差
///      （它的 8 宽树形归约分组必须原样保留）；`mx` 是该行 `prob` 的最大值，max 与顺序无关。
///   ③ AV 加权和：对**每个输出元素**仍按 j 升序累加 `axpy256_f32(o + i*d, v + j*d, a, d)`。
///      j 在外、i 在内 ⇒ 第 j 行 V 也只取一次；而对固定的 i 来说 j 仍是升序，
///      所以浮点累加顺序与"逐行版"**逐位相同**；`if a == 0.0 { continue }` 也照旧保留。
///
/// 块之间**互不依赖**（行与行本来就独立）：只有本块的行被本块写 ⇒ 并行时不需要任何
/// 私有累加缓冲，也不需要归约。
#[inline(always)]
unsafe fn attn_tile_fwd(
    q: *const f32,
    kk: *const f32,
    v: *const f32,
    t: usize,
    d: usize,
    o: *mut f32,
    prob: *mut f32,
    lo: usize,
    hi: usize,
) {
    let scale = 1.0 / vmlm_ops::kernels::sqrt_f(d as f32);
    let bi = hi - lo;
    // ---- 相位①：扫 j 算 prob/mx（K 行每块只流一遍）----
    let mut mx = [-1.0e30f32; ATTN_TILE_BI];
    for j in 0..hi {
        let kr = kk.add(j * d);
        // 因果掩码 j <= i：块内只有 k >= j-lo 的行要这一项
        let k0 = if j > lo { j - lo } else { 0 };
        for k in k0..bi {
            let i = lo + k;
            // [P19-1] q·k 仍是 AVX2 4 链点积（FMA）——内核与调用参数都没变
            let acc = vmlm_ops::kernels::dot256_f32(q.add(i * d), kr, d);
            let s = acc * scale;
            *prob.add(i * t + j) = s;
            if s > mx[k] {
                mx[k] = s;
            }
        }
    }
    // ---- 相位②：逐行 exp+Σ、算 inv、归零 orow ----
    let mut inv = [0.0f32; ATTN_TILE_BI];
    for k in 0..bi {
        let i = lo + k;
        // [P21-2/D 类] softmax 的 exp+Σ：行首 (i+1) 项就地 exp(x-mx) 并返回 8 宽树归约和
        // （要求 x-mx<=0，这里恒成立）。入口与单行版完全相同 ⇒ 归约分组不变。
        let sum = if vmlm_sys::cpu::HAS_AVX2 {
            vmlm_ops::kernels::exp_neg_sum_avx2(prob.add(i * t), i + 1, mx[k])
        } else {
            let mut s = 0.0f32;
            for j in 0..=i {
                let e = vmlm_ops::kernels::exp_f(*prob.add(i * t + j) - mx[k]);
                *prob.add(i * t + j) = e;
                s += e;
            }
            s
        };
        inv[k] = 1.0 / (sum + 1.0e-30);
        // [P19-1] 归零改为整块 write_bytes（原来是标量循环，够聪明就别指望优化器）
        std::ptr::write_bytes(o.add(i * d) as *mut u8, 0, d * 4);
    }
    // ---- 相位③：按 j 升序做 AV 加权和（V 行每块只流一遍）----
    for j in 0..hi {
        let vr = v.add(j * d);
        let k0 = if j > lo { j - lo } else { 0 };
        for k in k0..bi {
            let i = lo + k;
            let a = *prob.add(i * t + j) * inv[k];
            *prob.add(i * t + j) = a;
            if a == 0.0 {
                continue;
            }
            // [P19-1] 概率加权和 y += a·v[x]：显式 AVX2（指针无 noalias，LLVM 不向量化）
            vmlm_ops::kernels::axpy256_f32(o.add(i * d), vr, a, d);
        }
    }
}

/// attention 的"标量核心"：给定 q/k/v，算 softmax(q kᵀ/√d)·v。
/// 稠密投影与**稀疏投影（消融二）共用这一段**，保证两条路只差投影。
/// attn_core 的单行（第 i 行：分数 → softmax → 加权和）。行与行之间**完全独立**，
/// 所以并行化就是"按 i 分段"，无需任何私有缓冲/归约。
/// （P24-1 起实现转调 `attn_tile_fwd` 的单行特例：`lo=i, hi=i+1` ⇒ 相位①的 j 上界恰好是 i、
///  内层 `k0` 恒为 0、相位③的 j 仍是 0..=i —— 与改写前的单行实现逐字等价。公开签名不变。）
pub unsafe fn attn_core_row(
    q: *const f32,
    kk: *const f32,
    v: *const f32,
    t: usize,
    d: usize,
    o: *mut f32,
    prob: *mut f32,
    i: usize,
) {
    attn_tile_fwd(q, kk, v, t, d, o, prob, i, i + 1);
}

/// 并行版 attn_core：**按 `ATTN_TILE_BI` 行分块**再按块分段（nt<=1 时逐块串行）。
/// 行与行之间完全独立 ⇒ 分块只决定"哪几行共享一遍 K/V 行的读取"，不改变任何一行的
/// 加法顺序；块划分与线程数无关 ⇒ 串行/并行/任意 nt 都**逐位相同**。
/// 每个块只写自己那几行的 prob/orow，线程之间没有共享可变累加缓冲。
pub unsafe fn attn_core_par(
    q: *const f32,
    kk: *const f32,
    v: *const f32,
    t: usize,
    d: usize,
    o: *mut f32,
    prob: *mut f32,
    nt: usize,
) {
    let ntile = (t + ATTN_TILE_BI - 1) / ATTN_TILE_BI;
    if nt <= 1 {
        for tl in 0..ntile {
            let lo = tl * ATTN_TILE_BI;
            attn_tile_fwd(q, kk, v, t, d, o, prob, lo, (lo + ATTN_TILE_BI).min(t));
        }
        return;
    }
    let (qp, kp, vp, op, pp) = (
        SendPtr(q as *mut f32),
        SendPtr(kk as *mut f32),
        SendPtr(v as *mut f32),
        SendPtr(o),
        SendPtr(prob),
    );
    par_for(ntile, nt, move |tlo, thi| {
        let (qp, kp, vp, op, pp) = (qp, kp, vp, op, pp);
        for tl in tlo..thi {
            let lo = tl * ATTN_TILE_BI;
            attn_tile_fwd(qp.0, kp.0, vp.0, t, d, op.0, pp.0, lo, (lo + ATTN_TILE_BI).min(t));
        }
    });
}/// [P24-1] 反向 passA 的分块核心（**串行与并行 passA 共用**）：对 query 行块 `[lo, hi)` 算
///   `dprob[i][j] = dot256_f32(dobuf[i], v[j], d)`
///   `mean[k] = Σ_j prob[i][j]·dprob[i][j]`（`k = i-lo`；就是原来第二相里的 `mean`、并行版的 `rowsum`）
/// 并按 `WITH_DV` 决定要不要顺手做 `dv[j] += prob[i][j]·dobuf[i]`：
///   * 串行版要——它的 dv 只在这里累加一次；
///   * 并行 passA **不能**要——那里的 dv 是"按 j 跨块归约"，会被不同线程同时写；
///     并行路线的 dv 由 passB 先 `write_bytes` 再从零重算，口径与这里完全相同，所以原来
///     passA 里那份 dv 累加其实是**被 passB 丢掉的白工**（还白付一条 d 长行的 RMW）。
///
/// **为什么把 i/j 两重循环对调（j 在外、i 在内）**：第 j 行 V 每块只从 DRAM 取一次，
/// 块内各行共享它。对固定的 i，j 仍是升序（外层 j 从 0 到 hi，内层 k 从 `max(0, j-lo)` 起），
/// 所以三类浮点累加的顺序与原实现逐位相同：
///   * `mean`：原实现是第二相里独立的 `mean += prob·dprob` 标量链（j 升序）——
///     这里按同一顺序累加同样的两个操作数 ⇒ 逐位相同（并行版早就这么做）。
///   * `dv[j]`：原实现按 i 升序累加；块按 lo 升序处理 ⇒ 全局仍是 i 升序。
/// `WITH_DV` 用 const 泛型而不是运行时判断：每个调用点常量折叠后与手写两份循环等价。
#[inline(always)]
unsafe fn attn_tile_bwd_a<const WITH_DV: bool>(
    dobuf: *const f32,
    prob: *const f32,
    v: *const f32,
    t: usize,
    d: usize,
    dv: *mut f32,
    dprob: *mut f32,
    mean: *mut f32,
    lo: usize,
    hi: usize,
) {
    let bi = hi - lo;
    for j in 0..hi {
        let vr = v.add(j * d);
        let k0 = if j > lo { j - lo } else { 0 };
        for k in k0..bi {
            let i = lo + k;
            let a = *prob.add(i * t + j);
            let dr = dobuf.add(i * d);
            // [P19-2] 点积向量化（同一个内核、同一份 d ⇒ 与串行/并行两条路逐位一致）
            let dp = vmlm_ops::kernels::dot256_f32(dr, vr, d);
            *dprob.add(i * t + j) = dp;
            if WITH_DV {
                // [P19-2] dv[j] += a·dobuf[i]（原来每对 (i,j) 都要 RMW 一条 d 长行）
                vmlm_ops::kernels::axpy256_f32(dv.add(j * d), dr, a, d);
            }
            // 与第二相 `mean += prob·dprob` 完全相同的加数序列（j 升序）
            *mean.add(k) += a * dp;
        }
    }
}

/// [P24-1] 反向 pass2 的分块核心（**串行版**）：`ds = a·(dprob - mean)·scale`，
/// 一次遍历同时做 `dq[i] += ds·k[j]` 与 `dk[j] += ds·q[i]`（`axpy2_256_f32`）。
/// j 在外、i 在内 ⇒ 第 j 行 K 每块只取一次；对固定的 i，j 仍是升序 ⇒ dq 的累加顺序不变；
/// dk[j] 由"块按 lo 升序 + 块内 i 升序"累加 ⇒ 全局仍是 i 升序（与原实现相同）。
#[inline(always)]
unsafe fn attn_tile_bwd_dqdk(
    prob: *const f32,
    q: *const f32,
    kk: *const f32,
    t: usize,
    d: usize,
    dq: *mut f32,
    dk: *mut f32,
    dprob: *const f32,
    mean: *const f32,
    scale: f32,
    lo: usize,
    hi: usize,
) {
    let bi = hi - lo;
    for j in 0..hi {
        let kr = kk.add(j * d);
        let k0 = if j > lo { j - lo } else { 0 };
        for k in k0..bi {
            let i = lo + k;
            let a = *prob.add(i * t + j);
            let ds = a * (*dprob.add(i * t + j) - *mean.add(k)) * scale;
            if ds == 0.0 {
                continue;
            }
            // dq[i] 与 dk[j] 两条互不依赖的累加链，一次遍历（原先是两个标量内循环）
            vmlm_ops::kernels::axpy2_256_f32(dq.add(i * d), kr, ds, dk.add(j * d), q.add(i * d), ds, d);
        }
    }
}

/// [P24-1] 串行反向的分块核心：整块 `[lo, hi)` 走完 passA + pass2。
/// 归零只做在本块那几行上（`dv/dq/dk` 的每一行都只被"拥有它的块"第一次写，块的 `lo..hi`
/// 恰好划分 `0..t`）⇒ 顺带省掉 3 次 t·d 的全局 memset，且归零时机不影响任何累加顺序
/// （见上面两个 helper 的注释：先归零、再按升序累加，结果不变）。
#[inline(always)]
unsafe fn attn_tile_bwd(
    dobuf: *const f32,
    prob: *const f32,
    q: *const f32,
    kk: *const f32,
    v: *const f32,
    t: usize,
    d: usize,
    dq: *mut f32,
    dk: *mut f32,
    dv: *mut f32,
    dprob: *mut f32,
    scale: f32,
    lo: usize,
    hi: usize,
) {
    std::ptr::write_bytes(dv.add(lo * d) as *mut u8, 0, (hi - lo) * d * 4);
    std::ptr::write_bytes(dq.add(lo * d) as *mut u8, 0, (hi - lo) * d * 4);
    std::ptr::write_bytes(dk.add(lo * d) as *mut u8, 0, (hi - lo) * d * 4);
    let mut mean = [0.0f32; ATTN_TILE_BI];
    attn_tile_bwd_a::<true>(dobuf, prob, v, t, d, dv, dprob, mean.as_mut_ptr(), lo, hi);
    attn_tile_bwd_dqdk(prob, q, kk, t, d, dq, dk, dprob, mean.as_ptr(), scale, lo, hi);
}

/// attention 核心的反向：给定 dobuf（对 ao 的梯度）与 q/k/v/prob，产出 dq/dk/dv。
/// P24-1 起**按 query 行分块**：K/V 行每块只从 DRAM 取一次（原来每条 i 都把整条 V、整条 K
/// 重流一遍）。两相的结构、每个 (i,j) 的操作数与各自的累加顺序全部不变 ⇒ 逐位不变。
#[allow(clippy::too_many_arguments)]
pub unsafe fn attn_core_bwd(
    dobuf: *const f32,
    prob: *const f32,
    q: *const f32,
    kk: *const f32,
    v: *const f32,
    t: usize,
    d: usize,
    dq: *mut f32,
    dk: *mut f32,
    dv: *mut f32,
    dprob: *mut f32,
) {
    let scale = 1.0 / vmlm_ops::kernels::sqrt_f(d as f32);
    // [P21-5] 已实测**回退**：寄存器分块版（4 行累加器 + d 的 8 宽扫）把
    // `set1(p)/set1(ds)` 的广播次数从「每对 (i,j) 一次」放大到「每 (i,j,d-chunk) 一次」
    // ⇒ 端口 5 打满，端到端 0.1451 → 0.1493（+2.9%），且 gradcheck 变得勉强（4/40 越界）。
    // 内核 attn_bwd_bc_blocked_avx2 保留在 core 里（结论见 docs/P21）。
    // （P24-1 的"按 query 行分块"与那次"按 d 分块"不是一回事：这里**没有**动 d 方向的扫描。）
    let ntile = (t + ATTN_TILE_BI - 1) / ATTN_TILE_BI;
    for tl in 0..ntile {
        let lo = tl * ATTN_TILE_BI;
        attn_tile_bwd(dobuf, prob, q, kk, v, t, d, dq, dk, dv, dprob, scale, lo, (lo + ATTN_TILE_BI).min(t));
    }
}

/// 并行版 attn_core 反向：把原来"按 i 两遍"的写法改成**三段互不冲突**的写法。
/// 原来 pass1 里 `dv[j] += Σ_i` 与 pass2 里 `dk[j] += Σ_i` 都是"按 j 归约"，
/// 按 i 并行会有写冲突；拆成
///   passA（并行块）：dprob[i][j] = dobuf[i]·v[j]；rowsum[i] = Σ_j prob·dprob
///   passB（并行 j 块）：dv[j]、dk[j] 各自只被自己的 j 写
///   passC（并行 i 块）：dq[i] 只被自己的 i 写
/// nt<=1 时仍走串行实现（逐位不变）。
/// P24-1 起三段都**按块**跑：原来每条 i 都把整条 V 重流一遍（passA/passC）、每条 j 都把
/// 整条 q/dobuf 重流一遍（passB），现在这些行在每个块里只从 DRAM 取一次。
#[allow(clippy::too_many_arguments)]
pub unsafe fn attn_core_bwd_par(
    dobuf: *const f32,
    prob: *const f32,
    q: *const f32,
    kk: *const f32,
    v: *const f32,
    t: usize,
    d: usize,
    dq: *mut f32,
    dk: *mut f32,
    dv: *mut f32,
    dprob: *mut f32,
    rowsum: *mut f32,
    nt: usize,
) {
    if nt <= 1 {
        attn_core_bwd(dobuf, prob, q, kk, v, t, d, dq, dk, dv, dprob);
        return;
    }
    let ntile = (t + ATTN_TILE_BI - 1) / ATTN_TILE_BI;
    // ---- passA（按块并行；每块只写自己那几行的 dprob/rowsum）----
    // 不再累加 dv：见 `attn_tile_bwd_a` 的注释（passB 会从零重算 dv，原来那份是白工）。
    {
        let (dp_, pr_, do_, pv_) = (
            SendPtr(dprob),
            SendPtr(prob as *mut f32),
            SendPtr(dobuf as *mut f32),
            SendPtr(v as *mut f32),
        );
        let rs_ = SendPtr(rowsum);
        par_for(ntile, nt, move |tlo, thi| {
            let (dp_, pr_, do_, pv_, rs_) = (dp_, pr_, do_, pv_, rs_);
            for tl in tlo..thi {
                let lo = tl * ATTN_TILE_BI;
                let hi = (lo + ATTN_TILE_BI).min(t);
                let mut mean = [0.0f32; ATTN_TILE_BI];
                attn_tile_bwd_a::<false>(
                    do_.0, pr_.0, pv_.0, t, d, std::ptr::null_mut(), dp_.0, mean.as_mut_ptr(), lo, hi,
                );
                for k in 0..hi - lo {
                    *rs_.0.add(lo + k) = mean[k];
                }
            }
        });
    }
    // ---- passB（按 j 分块并行）----
    // 原来 j 在外、i 在内 ⇒ 每条 j 都把整条 q/dobuf 重流一遍（≈t 次）。
    // 现在 j 分块、i 在外：q[i]/dobuf[i] 每个 j 块只取一次，块内 BI 行的 dv/dk 常驻缓存。
    // 对固定 j，i 仍是升序 ⇒ dv/dk 的浮点累加顺序逐位不变。
    {
        let (dp_, pr_, do_, qq_, dk_, dv_, rs_) = (
            SendPtr(dprob),
            SendPtr(prob as *mut f32),
            SendPtr(dobuf as *mut f32),
            SendPtr(q as *mut f32),
            SendPtr(dk),
            SendPtr(dv),
            SendPtr(rowsum),
        );
        let scale = 1.0 / vmlm_ops::kernels::sqrt_f(d as f32);
        par_for(ntile, nt, move |jlo_t, jhi_t| {
            let (dp_, pr_, do_, qq_, dk_, dv_, rs_) = (dp_, pr_, do_, qq_, dk_, dv_, rs_);
            for jt in jlo_t..jhi_t {
                let jlo = jt * ATTN_TILE_BI;
                let jhi = (jlo + ATTN_TILE_BI).min(t);
                for j in jlo..jhi {
                    // [P19-2] 归零改整块 write_bytes（原为标量循环）
                    std::ptr::write_bytes(dk_.0.add(j * d) as *mut u8, 0, d * 4);
                    std::ptr::write_bytes(dv_.0.add(j * d) as *mut u8, 0, d * 4);
                }
                for i in 0..t {
                    let j1 = jhi.min(i + 1);
                    if jlo >= j1 {
                        continue; // i < jlo：本块这一行不参与（原实现 i 从 j 起，等价）
                    }
                    let dr = do_.0.add(i * d);
                    let qr = qq_.0.add(i * d);
                    let rs = *rs_.0.add(i);
                    for j in jlo..j1 {
                        let a = *pr_.0.add(i * t + j);
                        let ds = a * (*dp_.0.add(i * t + j) - rs) * scale;
                        // [P19-2] dv[j] += a·dobuf[i] 与 dk[j] += ds·q[i] 两条独立链，一次遍历向量化
                        vmlm_ops::kernels::axpy2_256_f32(
                            dv_.0.add(j * d), dr, a, dk_.0.add(j * d), qr, ds, d,
                        );
                    }
                }
            }
        });
    }
    // ---- passC（按 query 行分块并行）----
    // 原来 i 在外、j 在内 ⇒ 每条 i 都把整条 K 重流一遍。现在 j 在外、i 在内 ⇒
    // 第 j 行 K 每块只取一次；对固定 i，j 仍是升序 ⇒ dq 的累加顺序逐位不变。
    {
        let (dp_, pr_, kk_, dq_, rs_) = (
            SendPtr(dprob),
            SendPtr(prob as *mut f32),
            SendPtr(kk as *mut f32),
            SendPtr(dq),
            SendPtr(rowsum),
        );
        let scale = 1.0 / vmlm_ops::kernels::sqrt_f(d as f32);
        par_for(ntile, nt, move |tlo, thi| {
            let (dp_, pr_, kk_, dq_, rs_) = (dp_, pr_, kk_, dq_, rs_);
            for tl in tlo..thi {
                let lo = tl * ATTN_TILE_BI;
                let hi = (lo + ATTN_TILE_BI).min(t);
                let bi = hi - lo;
                std::ptr::write_bytes(dq_.0.add(lo * d) as *mut u8, 0, bi * d * 4);
                for j in 0..hi {
                    let kr = kk_.0.add(j * d);
                    let k0 = if j > lo { j - lo } else { 0 };
                    for k in k0..bi {
                        let i = lo + k;
                        let a = *pr_.0.add(i * t + j);
                        let ds = a * (*dp_.0.add(i * t + j) - *rs_.0.add(i)) * scale;
                        if ds == 0.0 {
                            continue;
                        }
                        // [P19-2] dq[i] += ds·k[j] 向量化
                        vmlm_ops::kernels::axpy256_f32(dq_.0.add(i * d), kr, ds, d);
                    }
                }
            }
        });
    }
}/// 全局因果 attention（**复用项目既有的 MiniCPM 路径**：单头全宽）
///   q = h·Wq, k = h·Wk, v = h·Wv, a = softmax(q kᵀ/√d) v, out = a·Wo
/// 形状与 `train/src/sparse_train24.rs` 的注意力一致（n_head=1、head_dim=d，
/// 原文：「single-head full-width attention: no separate projection dims」）。
/// 因果掩码 j ≤ i（预测下一个 token 时不许看未来）。
#[allow(clippy::too_many_arguments)]
// [抽库 L3-1] 显式禁止内联这两个大函数：跨 crate 后 LTO 把它们内联进了
// run_main 的闭包（闭包 61936→62624 B，b1-full +1024 B）。
// 见 _opt/l3_diag/*.map（链接器符号表）与 docs/抽库进展.md。
#[inline(never)]
pub unsafe fn attn_forward(
    h: *const f32,
    wq: *const f32,
    wk: *const f32,
    wv: *const f32,
    wo: *const f32,
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
    let scale = 1.0 / vmlm_ops::kernels::sqrt_f(d as f32);
    let _ = scale;
    gemm_nt(h, wq, q, t, d, d, false);
    gemm_nt(h, wk, kk, t, d, d, false);
    gemm_nt(h, wv, v, t, d, d, false);
    attn_core_par(q, kk, v, t, d, o, prob, nt);
    gemm_nt(o, wo, out, t, d, d, false);
}/// attention 反向：累加 dWq/dWk/dWv/dWo 与 dh（对 attention **输入**的梯度）。
#[allow(clippy::too_many_arguments)]
// [抽库 L3-1] 显式禁止内联这两个大函数：跨 crate 后 LTO 把它们内联进了
// run_main 的闭包（闭包 61936→62624 B，b1-full +1024 B）。
// 见 _opt/l3_diag/*.map（链接器符号表）与 docs/抽库进展.md。
#[inline(never)]
pub unsafe fn attn_backward(
    h: *const f32,
    wq: *const f32,
    wk: *const f32,
    wv: *const f32,
    wo: *const f32,
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
    dwq: *mut f32,
    dwk: *mut f32,
    dwv: *mut f32,
    dwo: *mut f32,
    adx: *mut f32, // 对 attention **输入** 的梯度（由调用方决定是否并入残差）
    at1: *mut f32,
    at2: *mut f32,
    wt: *mut f32,
    dobuf: *mut f32,
    dprob: *mut f32,
) {
    let scale = 1.0 / vmlm_ops::kernels::sqrt_f(d as f32);
    let _ = scale;
    // dWo += dout ⊗ o
    tr(dout, at1, t, d);
    tr(o, at2, t, d);
    gemm_nt(at1, at2, dwo, d, d, t, true);
    // do = dout · Woᵀ  —— 写进 dobuf（对 ao 的梯度）
    tr(wo, wt, d, d);
    gemm_nt(dout, wt, dobuf, t, d, d, false);
    // 核心反向：dq/dk/dv
    attn_core_bwd(dobuf, prob, q, kk, v, t, d, dq, dk, dv, dprob);
    // dWq/dWk/dWv += Σ_t d*[t][i]·h[t][x]
    tr(h, at1, t, d);
    tr(dq, at2, t, d);
    gemm_nt(at2, at1, dwq, d, d, t, true);
    tr(dk, at2, t, d);
    gemm_nt(at2, at1, dwk, d, d, t, true);
    tr(dv, at2, t, d);
    gemm_nt(at2, at1, dwv, d, d, t, true);
    // adx += dq·Wqᵀ + dk·Wkᵀ + dv·Wvᵀ
    tr(wq, wt, d, d);
    gemm_nt(dq, wt, at2, t, d, d, false);
    for x in 0..t * d {
        *adx.add(x) += *at2.add(x);
    }
    tr(wk, wt, d, d);
    gemm_nt(dk, wt, at2, t, d, d, false);
    for x in 0..t * d {
        *adx.add(x) += *at2.add(x);
    }
    tr(wv, wt, d, d);
    gemm_nt(dv, wt, at2, t, d, d, false);
    for x in 0..t * d {
        *adx.add(x) += *at2.add(x);
    }
}/// attention 体检：注意力**有没有选择能力**。
/// 直接测两件事（而不是靠"子空间是否相交"推理）：
///   1. `max|q·k|`：分数是否真的非零（0 ⇒ q·k 全零 ⇒ softmax 必然均匀）
///   2. softmax 的集中度：`max_prob` 与均匀值 1/T 的比（≈1 ⇒ 没有任何选择）
/// 返回 (max|q·k|, mean|q·k|, 非零比例, max_prob, 平均 max_prob/均匀)
#[allow(clippy::too_many_arguments)]
pub unsafe fn attn_qk_stats(
    q: *const f32,
    kk: *const f32,
    t: usize,
    d: usize,
    prob: *const f32,
) -> (f32, f32, f32, f32, f32) {
    let mut mx = 0.0f32;
    let mut sum = 0.0f32;
    let mut nz = 0usize;
    let mut n = 0usize;
    for i in 0..t {
        for j in 0..=i {
            let mut acc = 0.0f32;
            let qr = q.add(i * d);
            let kr = kk.add(j * d);
            for x in 0..d {
                acc += *qr.add(x) * *kr.add(x);
            }
            let a = acc.abs();
            if a > mx {
                mx = a;
            }
            if a > 0.0 {
                nz += 1;
            }
            sum += a;
            n += 1;
        }
    }
    // softmax 集中度：逐行 max prob 相对均匀值 1/T 的倍数（取平均）
    // ⚠️ 第 i 行的合法项数是 i+1（因果），所以"均匀"是 1/(i+1) 而不是 1/T。
    //    第一版用 1/T 做分母，把 i=0 行（只有 1 项，prob 恒 1.0）算成"高度集中"，
    //    得到 max_prob=1.0、集中度=4.06 的假象。这里只统计 i>=3 的行。
    let mut pmax = 0.0f32;
    let mut ratio_sum = 0.0f32;
    let mut ratio_n = 0usize;
    let mut n_sel = 0usize;
    for i in 0..t {
        let mut rmax = 0.0f32;
        for j in 0..=i {
            let a = *prob.add(i * t + j);
            if a > rmax {
                rmax = a;
            }
        }
        if rmax > pmax {
            pmax = rmax;
        }
        if i >= 3 {
            let ratio = rmax * (i + 1) as f32; // 1.0 = 完全均匀
            ratio_sum += ratio;
            ratio_n += 1;
            if ratio > 1.1 {
                n_sel += 1;
            }
        }
    }
    let _ = (n_sel, ratio_n);
    (
        mx,
        sum / n.max(1) as f32,
        nz as f32 / n.max(1) as f32,
        pmax,
        ratio_sum / ratio_n.max(1) as f32,
    )
}
