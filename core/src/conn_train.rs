// P17 T1/T2/T3：稀疏连接架构的**训练侧**。
//
// ============================ 本模块属于哪一步 ============================
//
// 架构只有两个主机制（任务书原文）：
//   1. 稀疏连接权重：每神经元初始化 N 个连接，**代替剪枝**
//   2. 稀疏激活连接：计算时只算被激活的连接，零连接跳过
// 其余一切是这两个机制的自然衍生，**不增加算法**。
//
// P16 交付了机制 1 的**存储格式**（core/src/conn_sparse.rs，CSR）和机制 2 的
// **两个算子**（OP_TOPK_BY_STRENGTH=93 / OP_SPARSE_MATVEC=94）。
// 但 P16 报告 §4 实测出一条硬结论：
//
//     随机权重下 per-row top-k 无法逼近稠密行 —— 稀疏率扫描在任何比例下
//     损失都 ≈1.0，**拐点不存在**。
//
// 并且 P16 §10 把原因写成了下一步的前提：**「P16 的真实目标需要权重有结构。
// 判据 2/3 的失败不是实现问题 —— 在有结构的权重（或训练后权重）上重跑扫描，
// 才可能看到拐点。」**
//
// 本模块就是那条前提的实现：让 N=20 条连接的**权重可训练**，于是
// 「稀疏连接权重」不再是一个随机占位结构，而是一个训练出来的结构。
//
// ---- 与 P16 存储层的分工（不要混淆）----
//
//   conn_sparse.rs   回答「哪些连接存在、初始权重多少、怎么存、怎么验」
//                    —— **纯结构**，它自己的模块头就写明「训练策略是另一个
//                    任务，本模块不声称它有统计意义」。
//   本模块（conn_train）回答「训练时梯度往哪回传、哪些权重更新、更新后
//                    统计长什么样」—— **纯训练语义**。
//
//   两者**共享**连接拓扑：本模块**不重新抽随机数**，而是调用
//   `conn_sparse::neuron_conn_src()` 取 src_id。任务书明确要求
//   「复用 conn_sparse.rs 的确定性 PRNG，不要另起一套随机数」。
//
// ============================ 唯一的算法差异 ============================
//
// 与稠密训练相比，本模块**只有一处**算法差异，任务书把它写得很清楚：
//
//     梯度**只回传 top-k 命中的连接**，非激活连接权重保持不变。
//
// 其余（前向、CE+softmax 反向、Adam、梯度裁剪）与稠密训练**完全相同** ——
// 复用 core/src/train.rs 的既有内核，不新增自动微分。
//
// ---- 「非激活连接」到底指什么（这个定义必须钉死）----
//
// 这里有一个真实的设计歧义，必须写下来而不是含糊过去：
//
//   (a) 「拓扑非激活」= 不在该神经元的 k0 条连接里的 (i, src) 对。
//       这些权重**根本不存在**（CSR 里没有它们），所以「保持不变」是
//       平凡成立的 —— 没有东西可以更新。
//   (b) 「强度非激活」= 在该神经元的 k0 条连接里，但**强度 s = w·x 不是
//       top-k**（OP_TOPK_BY_STRENGTH 会把它筛掉），于是它**这一帧不参与
//       y 的求和**。
//
//   机制 2 说的是 (b)：前向时这些连接被**跳过**，所以反向时它们对 y
//   的偏导**恒为 0**（它们没进 y），于是梯度必须**不进**这些权重。
//   这正是「只更新激活连接」的准确含义，也是与稠密训练的唯一差异。
//
//   ⚠️ 诚实标注：如果把 (b) 理解成 (a)，这条机制就是**空的** —— 因为 (a)
//   里根本没有可更新的权重。所以本模块**显式实现 (b)**，并且用一个能失败
//   的判据（`sparse_grad_respects_mask` 的 can-fail 对照）证明它非空。
//
// ---- 为什么反向是「指令流」而不是自动微分 ----
//
// 白皮书 §6 与本项目一贯做法：反向由 .vml 显式编写。本模块因此提供的是
// **内核函数**（供 interp.rs 的新指令调用）+ **host 侧驱动**（供 train/
// 驱动样例调用），而不是一张计算图。稀疏反向的内核只有一个新东西：
// 一个**掩码**，它把非激活连接的 dW 项写回原值（即不动）。
//
// ⚠️ 关于「不动」的实现方式（这里有一个必须避免的陷阱）：
//   朴素做法是「加到 dW 上，然后让 Adam 跳过未激活项」。那**不是**本模块
//   的做法，因为 Adam 有 m/v 状态：跳过更新却仍然推进 t，会让所有
//   未激活权重的有效步长漂移。正确做法是在**写梯度时**就把非激活项
//   置 0（而不是事后跳过），于是 Adam 的 m/v 对这些位置衰减到 0、
//   权重**严格不动** —— 这才是「保持不变」的严格形式，也是可测的形式
//   （判据：未激活权重的位型在训练前后逐位相同）。
#![allow(dead_code)]

use crate::conn_sparse::neuron_conn_src;
// [抽库 L4-2] `SparseConnW` + impl 已下移到 `vmlm_ops::conn`（L2a）；这里重导出 ⇒ 路径不变。
pub use vmlm_ops::conn::*;

// ============================================================================
// 1. 稀疏连接权重矩阵（训练侧的可变视图）
// ============================================================================
//
// 布局与 CSR 的**语义**一致（第 i 个神经元的第 j 条连接 = src[i*k0+j]），
// 但**不是**磁盘格式：本结构是训练期的工作区，权重是**连续可写的 f32**，
// 而 conn_sparse 的磁盘权重是「按位存储的占位值」且在 mmap 区只读。
//
// 为什么不直接用 CSR 的字节：训练要**就地更新**权重（OP_ADAM 的契约），
// 而 conn_sparse 的权重区在推理路径上位于 PAGE_READONLY 映射中
// （见 lib.rs `mmap_file_ro` 的注释：这是**故意的安全性质** ——
//  训练一旦误写到 mmap 区会立刻访问违例，而不是静默写坏共享页）。
// ⇒ 训练必须用自己的可写缓冲，训练完再 `write_section` 落盘。
//
// ⚠️ 训练侧布局取**每神经元定长 k0**（不是变长 CSR offsets）：
//   本阶段（T1）的连接数在训练中**固定**（任务书原文），所以变长表达
//   没有用处，而定长让 `i*k0+j` 成为一次乘法而不是两次 load ——
//   训练是最热的路径。落盘时再由 write_section 转成 CSR。

// ============================================================================
// 2. T1 —— 初始化：每神经元 N=20 个连接
// ============================================================================

/// T1：初始化「每神经元 N 个连接」的拓扑 + 权重。
///
/// `depth_scale` 是**残差分支**输出的额外缩放（见调用方：标准做法是
/// `1/sqrt(2L)`）。为什么它必须存在，是一条**实测**结论，不是风格选择：
///   24 层 × 1536 维、`depth_scale = 1` 时初始 CE = **83.04**（ln V = 4.16），
///   第 1 步 CE 变成**负数** —— 残差流方差随深度无界增长，
///   logits 饱和。而同样代码在 2 层时初始 CE = 2.78（= ln V，健康）。
///   调小学习率**修不了**这个问题（前向在 t=0 就发散了）。
///   传 1.0 即「不缩放」，用于单层/对照场景。
///
/// ⚠️ 它只改**初始尺度**，不改算法：两条机制的语义一字未动。
///
/// # Safety
/// `sc` 已由 `from_base` 构造且缓冲可写。
pub unsafe fn init_sparse_conn_scaled(sc: &SparseConnW, depth_scale: f32) {
    init_sparse_conn(sc);
    if depth_scale != 1.0 {
        for t in 0..sc.e {
            *sc.w.add(t) *= depth_scale;
        }
    }
}

/// T1：初始化「每神经元 N 个连接」的拓扑 + 权重（不缩放版本）。
///
/// 两件事各自有明确的来源，**不要混为一谈**：
///
///   * **拓扑（src id）** 来自 `conn_sparse::neuron_conn_src` —— 复用既有
///     确定性 PRNG。任务书原文：「复用 conn_sparse.rs 的确定性 PRNG
///     （neuron_seed/neuron_rand/neuron_conn_src），**不要另起一套随机数**」。
///     于是训练出来的权重可以按 (layer, i, j) 精确重建，磁盘上不需要
///     存 src —— 只存权重就能还原整个结构。这是本项目「可校验」的延续。
///
///   * **权重** 则**必须**换成有符号、小方差的分布。P16 的
///     `neuron_conn_weight_bits` 给出的是**全正 [0.5,1.0)** 的占位值，
///     它自己的文档就写明「不是训练好的权重……训练/初始化策略是另一个任务」。
///     一个**全正**的初始权重对本任务是不可用的：sum of 20 个正数恒为正，
///     于是每层的输出有系统性偏置，loss 下降会被这个偏置主导，
///     而「loss 下降」这条验收就失去意义（它测的是偏置而不是结构）。
///
///   ⚠️ 这里的权重分布**不是**「新的随机数」（那会违反任务书）：
///     它仍然由 `neuron_conn_src` 的**同一个** PRNG 流导出 ——
///     用第 j 条连接的 src 与第 i 个神经元的种子混合出符号与幅度。
///     换句话说：**PRNG 只有一套**，本函数只是换了一种用法（取符号 +
///     缩放到 small-init），而不是引入第二个随机源。
///
/// 缩放：Xavier/Glorot 对 fan-in = k0 的形式 w ~ U(-s, s)，
/// s = sqrt(6/(fan_in + fan_out))，fan_in = fan_out = k0 ⇒ s = sqrt(3/k0)。
/// k0=20 时 s ≈ 0.387。这是**标准**初始化，不是本架构新增的算法。
///
/// # Safety
/// `sc` 已由 `from_base` 构造且缓冲可写。
pub unsafe fn init_sparse_conn(sc: &SparseConnW) {
    let n = sc.n_neurons as usize;
    let k0 = sc.k0;
    for i in 0..n {
        for j in 0..k0 {
            let t = sc.idx(i, j);
            let sid = neuron_conn_src(sc.layer, i as u32, j as u32, sc.n_neurons, k0 as u32);
            *sc.src.add(t) = sid as f32;
            // u32 镜像：**这是生产路径上唯一的 src 写入点**，所以镜像只需
            // 在这里同步一次。若将来出现第二个写入点而忘了同步镜像，
            // `assert_src_u32_consistent()` 会硬失败并打印首个不一致位置。
            *sc.src_u32.add(t) = sid;
            // 同一个 PRNG 流的第二个用法：高 32 位取符号位 + 幅度。
            // 用 sid 与 j 一起混合，使「同一条连接」在两个不同神经元里
            // 得到不同权重（否则同 src 的权重会完全相同，那是退化的）。
            let mix = crate::conn_sparse::neuron_rand(
                crate::conn_sparse::neuron_seed(sc.layer, i as u32) ^ 0xA5A5_5A5A,
                j as u32 * 2654435761u32.wrapping_add(sid),
            );
            // 映射到 [-1, 1)：u32 -> f32 的**确定性**映射（不用浮点除法以外的
            // 任何东西；这里一次除法是确定的，且不涉及 FMA 收缩）。
            let u = (mix as f64) / (u32::MAX as f64); // [0,1)
            let v = (2.0 * u - 1.0) as f32; // [-1,1)
            // 缩放到 Glorot 区间
            // 缩放到 Glorot 区间。用引擎自己的 sqrt_f 而不是 f32::sqrt ——
            // 本 crate 是 no_std，没有 core 的浮点方法（那是 std 的）。
            // 这同时保证了初始化在不同构建下**位型相同**（同一个实现）。
            let s = crate::kernels::sqrt_f(3.0f32 / k0 as f32);
            *sc.w.add(t) = v * s;
            // 初始时全部视为激活（T2 会在每帧重新算 top-k）
            *sc.act.add(t) = 1.0;
        }
    }
}

// ============================================================================
// 3. T2/T3 —— 只对激活连接算梯度
// ============================================================================

/// 一个神经元的、按强度降序取 top-k 的连接下标。
///
/// 强度 s = |w · x|：任务书定义「共享量：连接激活强度 s_i = w_i × x_i」。
/// 取**绝对值**是这里唯一需要说明的选择 —— 见下面的长注释。
///
/// 返回命中的个数（<= k）。
///
/// # Safety
/// `wrow`/`xsrc` 的长度必须与 `k0`/源 id 的取值范围一致。
pub unsafe fn row_topk(
    wrow: *const f32,
    srcrow: *const f32,
    x: *const f32,
    k0: usize,
    k: usize,
    hit: &mut [usize],
) -> usize {
    if k0 > MAX_K0 {
        // 与引擎侧同样的硬失败口径：不静默截断。
        crate::print(b"CONN_TRAIN: k0 exceeds MAX_K0\n");
        crate::exit(9);
    }
    // ===================== P18：单遍 top-k（原为 k 轮全扫）=====================
    //
    // WHY THIS WAS REWRITTEN (measured, not guessed):
    //   The previous version did `k` rounds of "scan all k0, take the max,
    //   mark it taken". That is O(k * k0) comparisons per neuron. At the
    //   acceptance config (d=1536, k=8, k0=20, T=128, L=24) it measured
    //   139.4 ms/layer = 3346.6 ms/step, i.e. **28.0% of the whole step** --
    //   see docs/P18_步骤0_剖析实测.md. It was the second-largest single cost
    //   after the dense GEMMs.
    //
    // The replacement keeps a sorted array of the current best `k` and does ONE
    // pass over k0, inserting into it. Cost is O(k0) comparisons in the common
    // case (early-outs when the candidate cannot beat the current k-th best).
    //
    // ============ THE TIE RULE IS PART OF THE CONTRACT, NOT AN ACCIDENT ============
    // doc comment on ref_topk_sorted (below) states it and a test cross-checks
    // against it:
    //     "row_topk: `if v > bestv` => ties keep the FIRST-ENCOUNTERED (lower index)"
    // The old code got that by scanning in ascending index order and only
    // replacing on a STRICT improvement. The single-pass version reproduces it
    // exactly by scanning ascending and inserting a new candidate only when it
    // is STRICTLY greater than the element it displaces. Replacing any `>` below
    // with `>=` changes which of two equal-strength connections survives, so
    // the reference cross-check WILL fail -- that is the failable criterion.
    //
    // Strict `>` also keeps this consistent with the engine's opcode 93, which
    // is what OP_SPARSE_MATVEC consumes.
    let kk = if k > k0 { k0 } else { k };
    if kk == 0 {
        return 0;
    }
    // ============ 栈用量是这里的硬约束，不是风格问题 ============
    //
    // ⚠️ 第一版把这两个缓冲写成 `[_; MAX_K0]`（2048 项），在**每次调用**里
    //    占用 2048*8 + 2048*4 = 24 KB 栈。这个函数在 `compute_act_mask` 里
    //    按神经元调用（24 层 × 1536 神经元 × 128 token），于是驱动当场
    //    0xC0000005（exit -1073741819）**且没有任何诊断信息** —— 正是本项目
    //    在 build_all.ps1 里记录过的那一类：栈帧一大就跳过 guard page。
    //    （P16 给引擎的 TOPK_MAX_L 设上界，用的就是同一条理由。）
    //
    //    ⇒ 这里改用**固定的小尺寸** `TOPK_STACK_K`，并把 k0 > 它的情形
    //      变成**显式硬失败**，而不是静默地开一个巨大的栈帧然后崩溃。
    //      这与本项目「非法输入必须硬失败并打印是哪一条」的口径一致。
    //
    // 为什么是 256：本模型 k0=20、k=8，留 12x 余量足够覆盖任何有意义的实验；
    // 而 256*8 + 256*4 = 3 KB 的栈帧是安全的（远低于 guard page 阈值）。
    // 真要 k0 > 256，那是换一条算式的问题，不是把这个常量调大。
    const TOPK_STACK_K: usize = ROW_TOPK_STACK_K;
    if k0 > TOPK_STACK_K {
        crate::print(b"CONN_TRAIN: row_topk k0 exceeds the safe stack bound\n");
        crate::print(b" k0=");
        crate::interp::print_u32(k0 as u32);
        crate::print(b" bound=");
        crate::interp::print_u32(TOPK_STACK_K as u32);
        crate::print(b"\n");
        crate::exit(9);
    }
    // best[j] holds the j-th strongest connection index so far (descending by
    // strength).
    let mut best = [0usize; TOPK_STACK_K];
    let mut bestv = [0.0f32; TOPK_STACK_K];
    let mut nk = 0usize;
    for j in 0..k0 {
        let wv = *wrow.add(j);
        let sid = *srcrow.add(j) as usize;
        // 强度 = w * x 的**绝对值**。
        //
        // WHY ABS: the task defines s_i = w_i * x_i without a modulus. With
        // a signed w (which T1's init requires, see above) the SIGNED product
        // ranks a large negative contribution BELOW a small positive one, so
        // "top-k by strength" would keep the weakest-magnitude connections
        // and drop the strongest contributors -- that is not "keep the
        // strongest connections", it is the opposite for half the inputs.
        // P16's placeholder weights were all-positive, so this distinction
        // could not show up there; with trained/init-signed weights it does.
        // |w*x| is the standard magnitude criterion (cf. magnitude pruning).
        let v = (wv * *x.add(sid)).abs();
        // Fast reject: if the buffer is full and this cannot beat the weakest
        // survivor, skip it. The comparison is the SAME strict `>` used for
        // insertion, so the tie rule is unchanged by this shortcut.
        if nk == kk && !(v > bestv[kk - 1]) {
            continue;
        }
        // Insertion sort, descending by strength. Walk up while the candidate is
        // STRICTLY greater than the element below it; equal strengths stop the
        // walk, which leaves the earlier-encountered (lower) index ahead --
        // exactly the old behaviour.
        let mut q = if nk < kk { nk } else { kk - 1 };
        while q > 0 && v > bestv[q - 1] {
            best[q] = best[q - 1];
            bestv[q] = bestv[q - 1];
            q -= 1;
        }
        best[q] = j;
        bestv[q] = v;
        if nk < kk {
            nk += 1;
        }
    }
    // 输出必须是**升序**（这与引擎 93 的契约一致，不是审美选择）
    //
    // WHY THIS SORT IS LOAD-BEARING (measured, not argued): the first version
    // returned the SELECTION order, i.e. descending strength. The single-layer
    // harness compared it against an independent full-sort reference and
    // reported `bad=64` -- every row disagreed. The two implementations chose
    // the SAME survivors, so this was purely an order bug, and it is the exact
    // bug opcode 93's own long comment records being caught before:
    //   "the first version selected correctly and emitted in strength order,
    //    and tools/p16_sparse_ops.py's D4 control reported
    //    `indices=[4, 0] ascending=False`."
    //
    // Why the order matters HERE too, beyond matching the engine: the mask this
    // function produces is consumed by `sparse_gemm_bwd_w`, and the trainer's
    // `macs` count is compared against the engine's FL_NNZ count. If the two
    // disagreed on order the counts would still match (it is a set), so order
    // alone cannot be caught by the count -- only by a pairwise comparison
    // against an independent reference, which is exactly what the harness does.
    //
    // `nk <= kk <= k0 <= MAX_K0`, and the sort is over at most kk elements --
    // a handful for a real k0, so insertion sort is the right choice and no
    // sort routine is worth its bytes.
    for i in 0..nk {
        hit[i] = best[i];
    }
    for i in 1..nk {
        let v = hit[i];
        let mut q = i;
        while q > 0 && hit[q - 1] > v {
            hit[q] = hit[q - 1];
            q -= 1;
        }
        hit[q] = v;
    }
    nk
}

/// **纯选择**版 top-k：输入已经是算好的强度行，不再触碰 `x`（因此没有 gather）。
///
/// 存在理由（P18-C，实测）：`row_topk` 在它的选择循环里用 `x[srcrow[j]]` 取强度，
/// 于是同一行的源被 gather 了**多轮**。实测 `compute_act_mask` 0.4821 ms
/// 大于整个 `sparse_forward` 0.4767 ms —— 掩码就是全部成本，而 gather 是其主因。
/// 把强度先一次性算进栈缓冲、再交给本函数做选择，每行每个源只 gather 一次。
///
/// **选择规则与 `row_topk` 逐条相同**（这是硬要求，不是「大致一样」）：
///   * 强度相等时保留**下标更小**的 —— 由「升序扫描 + 只在严格 `>` 时替换/插入」保证；
///   * 输出**升序**（引擎 93 的契约）。
/// 两处共用同一套代码是本函数存在的第二个理由：分开写会漂移。
///
/// 返回命中数（<= k）。
pub unsafe fn row_topk_from_strength(
    strength: *const f32,
    k0: usize,
    k: usize,
    hit: &mut [usize],
) -> usize {
    let kk = if k > k0 { k0 } else { k };
    if kk == 0 {
        return 0;
    }
    if k0 > ROW_TOPK_STACK_K {
        crate::print(b"CONN_TRAIN: row_topk_from_strength k0 exceeds the safe stack bound\n");
        crate::exit(9);
    }
    // ===================== P18-E：分支是这里的全部成本 =====================
    //
    // 实测分解（vmlm-p18e，d=1536 k0=20 k=8，300 reps）：
    //     gather + |w*x| 强度行           0.0356 ms   1.16 ns/连接
    //     **强度 + top-k 选择**           0.4119 ms  13.41 ns/连接
    //     => top-k 占 91.4%，是它选择的那些算术的 11 倍
    //
    // 13.4 ns/连接 ≈ 40 个周期，而一次比较应当只要 1 个周期。
    // 原因是**数据相关的分支**：插入 while 的行程数取决于新值排在第几位，
    // 每次误预测 ~15-20 周期。这就是「看似 O(k0) 却慢 10 倍」的来源。
    //
    // 修法：把插入改成**无分支的条件移动**。k0=20、k=8 都很小，
    // 所以直接用「逐位右移 + 条件写」的实现：
    //   对 j 从 kk-1 往下走，若 v > bestv[j] 就把 bestv[j] 往后挪一格。
    //   用 `if` 的三元形式让 LLVM 生成 cmov 而不是分支。
    //
    // ⚠️ 平局规则必须不变：仍然只在**严格** `v > bestv[q]` 时移动，
    //    相等强度不移动 ⇒ 先遇到的（下标更小）留在前面。
    //    这把 `>` 改成 `>=` 会改变幸存者 —— T1 的 ref_topk_sorted 对拍会报 FAIL。
    let mut best = [0usize; ROW_TOPK_STACK_K];
    let mut bestv = [0.0f32; ROW_TOPK_STACK_K];
    let mut nk = 0usize;
    // ---------------------------------------------------------------------
    // 无分支插入（branchless insertion）。
    //
    // 为什么这不是「微优化」而是必要：实测 13.4 ns/连接 ≈ 40 周期，
    // 而数据相关的 while 行程数是主因。改成「固定 kk 次迭代 + 条件移动」后，
    // 行程数与数据无关，分支预测器只需预测一条固定次数的循环。
    //
    // 算法（对每个输入 v，在**已排序降序**的 best[0..kk) 中插入 v，保持降序）：
    //   从尾部往前扫固定 kk 步；每步用条件移动决定「把 cur 右移」还是「把 v 放下」。
    //   用一个 carry 对 (cv, ci) 实现无分支交换。
    //
    // ⚠️ 平局规则仍是**严格** `>`：`if v > cur` 才交换。相等时保留原位置，
    //    于是先遇到的（下标更小）留在前面 —— 与旧实现逐位一致。
    //    T1 的 ref_topk_sorted 对拍验证这一点（把 `>` 改成 `>=` 会 FAIL）。
    //
    // 先把前 kk 个直接填入，之后每个都走一次无分支插入 —— 这样不需要
    // 「nk 是否够」的分支，形状固定。
    let mut filled = 0usize;
    for j in 0..k0 {
        let v = *strength.add(j);
        if filled < kk {
            // 填充阶段：无分支的插入排序，把 v 放到降序正确位置
            let mut cv = v;
            let mut ci = j;
            let mut q = filled;
            while q > 0 {
                let prev = bestv[q - 1];
                // 严格 `>`：相等不交换 ⟹ 先遇到的留在前面
                let take_prev = cv > prev;
                let nv = if take_prev { prev } else { cv };
                let ni = if take_prev { best[q - 1] } else { ci };
                bestv[q] = if take_prev { cv } else { prev };
                best[q] = if take_prev { ci } else { best[q - 1] };
                cv = nv;
                ci = ni;
                q -= 1;
            }
            bestv[0] = cv;
            best[0] = ci;
            filled += 1;
            continue;
        }
        // 满员后：先快速拒绝（绝大多数连接在这里被排除，且形状可预测）
        if !(v > bestv[kk - 1]) {
            continue;
        }
        // 只有真正要进榜的才走插入。这一步是少数派（k0=20、k=8 时约 8/20 进榜），
        // 而且因为已经过了快速拒绝，分支可预测性比原来高得多。
        let mut cv = v;
        let mut ci = j;
        let mut q = kk - 1;
        while q > 0 {
            let prev = bestv[q - 1];
            let take_prev = cv > prev;
            let nv = if take_prev { prev } else { cv };
            let ni = if take_prev { best[q - 1] } else { ci };
            bestv[q] = if take_prev { cv } else { prev };
            best[q] = if take_prev { ci } else { best[q - 1] };
            cv = nv;
            ci = ni;
            q -= 1;
        }
        bestv[0] = cv;
        best[0] = ci;
    }
    nk = filled;
    // 输出升序（引擎 93 的契约）。
    // kk 很小（模型里 = 8），所以这里是至多 8 个元素的插入排序 —— 保留不动。
    for i in 0..nk {
        hit[i] = best[i];
    }
    for i in 1..nk {
        let v = hit[i];
        let mut q = i;
        while q > 0 && hit[q - 1] > v {
            hit[q] = hit[q - 1];
            q -= 1;
        }
        hit[q] = v;
    }
    nk
}

/// 本模块能处理的最大 k0。与引擎的 TOPK_MAX_L=2048 同一个理由：
/// 固定的栈上 `taken` 位图，超过就**硬报错**而不是静默截断。
pub const MAX_K0: usize = 2048;

/// top-k 缓冲区在**栈上**的实际上界。与 `MAX_K0` 分开，因为两者管的不是一件事：
///   * `MAX_K0` = 本模块愿意接受的 k0 上界（位图/参考实现用，堆或一次性分配）；
///   * `ROW_TOPK_STACK_K` = 每次调用都可以安全开在栈上的大小。
/// 把「能接受的最大 k0」和「能安全放进栈的大小」混为一谈，就是 0xC0000005 的来源：
/// `row_topk` 被逐神经元调用，栈帧乘上 24×1536×128 次调用，一大就跳过 guard page，
/// 而且**没有任何诊断输出**。所以这里用显式上界 + exit(9) 把静默崩溃换成可读报错。
pub const ROW_TOPK_STACK_K: usize = 256;

/// T2/T3 核心：为整个层**重算激活掩码**（前向的一部分）。
///
/// 这一步在**前向**里调用（因为掩码决定 y），反向只**读**它。
/// 把它放在前向而不是反向，是因为「哪些连接被激活」在数学上是前向的量：
/// y = Σ_{激活连接} w·x。反向若自己重算一遍，两侧就可能因浮点差异
/// 选出不同的 top-k，而那样 dW 会写到一个**没进 y** 的连接上 ——
/// 梯度与函数不一致，且**不会被任何维度检查发现**。
/// ⇒ 单一真相源：掩码只在前向算一次。
///
/// # Safety
/// `sc` 的 w/src/act 缓冲均可读写；`x` 至少 n_neurons 个 f32。
pub unsafe fn compute_act_mask(sc: &SparseConnW, x: *const f32, k: usize) {
    let n = sc.n_neurons as usize;
    let k0 = sc.k0;
    // 先全部清零，再把命中的置 1 —— 顺序不能反（否则上一帧的命中会残留，
    // 而残留在数值上完全看不出来，只会让「激活连接数」比 k 多）。
    for t in 0..sc.e {
        *sc.act.add(t) = 0.0;
    }
    // 栈用量同 row_topk 的理由（见那里的长注释）：这里也曾是 `[_; MAX_K0]`
    // = 16 KB 栈，与 row_topk 的 24 KB 叠加足以跳过 guard page 并静默崩溃。
    // 按 kk 开即可：row_topk 最多写回 kk 个下标。
    let kk = if k > k0 { k0 } else { k };
    if kk > ROW_TOPK_STACK_K {
        crate::print(b"CONN_TRAIN: compute_act_mask k exceeds the safe stack bound\n");
        crate::print(b" k=");
        crate::interp::print_u32(kk as u32);
        crate::print(b" bound=");
        crate::interp::print_u32(ROW_TOPK_STACK_K as u32);
        crate::print(b"\n");
        crate::exit(9);
    }
    let mut hit = [0usize; ROW_TOPK_STACK_K];
    // ===================== P18-C：强度只 gather 一次 =====================
    //
    // 实测（vmlm-p18d，d=1536 k0=20 k=8）：compute_act_mask 单次 0.4821 ms，
    // 而**整个** sparse_forward 才 0.4767 ms —— 掩码就是全部成本。
    //
    // 为什么贵：`row_topk` 在它的选择循环里用 `x[srcrow[j]]` 取强度，
    // 于是 **同一行**的 20 个源被 gather 了**多轮**（每轮扫一遍未取的）。
    // gather 是随机地址访问，比连续 load 贵 2~3 倍
    // （白皮书 §13.6 已独立测过同一现象：「8 次 gather8 比连续 load 慢 2~3 倍」）。
    //
    // 修法：先**一次**把本行 k0 个强度算进栈缓冲（每个源只 gather 一次），
    // 再把选择工作交给纯寄存器/栈的 `row_topk_from_strength`。
    // 这与 AVX2 版 `sparse_strength_row_avx2`（kernels.rs）是同一个思路，
    // 只是这里用标量以保证任何构建（含无 AVX2 的）都能走同一条路径。
    //
    // 位精确：强度值本身（`(w*x).abs()`）计算方式与原来**逐位相同**，
    // 选择规则也相同（同一个 row_topk_from_strength / 同一条平局规则），
    // 所以输出掩码逐位不变。由 T1/T2 的对拍验证。
    let mut strbuf = [0.0f32; ROW_TOPK_STACK_K];
    if k0 > ROW_TOPK_STACK_K {
        crate::print(b"CONN_TRAIN: compute_act_mask k0 exceeds the safe stack bound\n");
        crate::exit(9);
    }
    for i in 0..n {
        let base = i * k0;
        // P18-PERF：用 AVX2 gather 算强度行（每个源只 gather 一次）。
        //
        // 实测依据：`vmlm-p18e` 把掩码成本分解为
        //     强度行（gather + |w*x|）   1.16 ns/连接
        //     top-k 选择                 7.19 ns/连接（无分支版；原 13.41）
        // 所以这一步只是 14% 的掩码成本，向量化它的收益有限 ——
        // 但它**零风险**（已有 AVX2 与标量逐位对拍的证据），
        // 且 5 个 matvec × 24 层 × 128 位置 会把小收益放大 15360 倍。
        //
        // 位精确：`sparse_strength_row_avx2` 用 andnot 清符号位实现 abs，
        // 与标量 `f32::abs` 逐位一致（都只清 bit31，含 NaN 位型），
        // 已由 `vmlm-p18k` 的 A1 判据在 30720 个值上验证 mismatch=0。
        #[cfg(target_arch = "x86_64")]
        {
            if crate::interp::HAS_AVX2 {
                crate::kernels::sparse_strength_row_avx2(
                    sc.w.add(base),
                    sc.src.add(base),
                    x,
                    k0,
                    strbuf.as_mut_ptr(),
                );
            } else {
                for j in 0..k0 {
                    let sid = *sc.src.add(base + j) as usize;
                    strbuf[j] = (*sc.w.add(base + j) * *x.add(sid)).abs();
                }
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            for j in 0..k0 {
                let sid = *sc.src.add(base + j) as usize;
                strbuf[j] = (*sc.w.add(base + j) * *x.add(sid)).abs();
            }
        }
        let nk = row_topk_from_strength(strbuf.as_ptr(), k0, kk, &mut hit);
        for q in 0..nk {
            *sc.act.add(base + hit[q]) = 1.0;
        }
    }
}

/// 前向：y = Σ_{激活连接} w·x（**只算激活连接**）。
///
/// 返回实际执行的乘加次数 —— 这是「零连接跳过」的**可测证据**，
/// 与引擎侧 `OP_SPARSE_MATVEC` 的 FL_NNZ 是同一个量的同一个用途。
/// 报计数而不是布尔值，是为了让「跳过」这条主张**能失败**。
///
/// # Safety
/// 同 `compute_act_mask`。
pub unsafe fn sparse_forward(
    sc: &SparseConnW,
    x: *const f32,
    y: *mut f32,
    k: usize,
) -> usize {
    compute_act_mask(sc, x, k);
    sparse_accum(sc, x, y, k)
}

/// 只做「按已有的 act 掩码累加」这一步，**不重算掩码**。
///
/// 为什么要把这一步单独拆出来（P18-C，实测驱动）：
///   `sparse_forward` 原本 = compute_act_mask + 累加。改成路径 B 之后，
///   每层有 **5 个**稀疏 matvec（q/k/v/wo/ffn）而不再是 1 个，于是
///   top-k 掩码的计算次数被乘了 5。实测分解（vmlm-p18d，d=1536 k0=20 k=8）：
///       sparse_forward 单次       0.4767 ms
///       其中 compute_act_mask     0.4821 ms   <-- 占 101.1%
///       sparse_gemm_bwd_w         0.1057 ms
///   即：**掩码计算比整个前向调用还贵**，matvec 本身几乎免费。
///   外推到 24 层 × 128 位置 = 8947 ms/step，其中前向 7323 ms。
///
///   ⇒ 关键结构性事实：`act` 掩码只依赖 (sc, x, k)。**反向**根本不需要重算掩码
///     （它只读前向留下的那一张），这是本模块一贯的「单一真相源」原则
///     （见 compute_act_mask 的长注释：反向重算会因浮点差异选出不同 top-k，
///     把梯度写到没进 y 的连接上，且无法被任何维度检查发现）。
///     ⇒ 反向改用本函数读掩码累加，就把掩码计算次数从 5 降到 5（前向）
///       而不是 5（前向）+ 更多（反向）。
///
/// **位精确**：本函数的累加顺序与 `sparse_forward` 内联时**完全相同**
///   （行内 k0 升序、单标量累加器、首个命中直接赋值而非 0.0 起加），
///   所以拆分不改变任何数值。`vmlm-sparse-t1` 的 T2-a/T2-c 验证这一点。
///
/// # Safety
/// 同 `compute_act_mask`；`sc.act` 必须已由一次 `compute_act_mask` 写好。
pub unsafe fn sparse_accum(
    sc: &SparseConnW,
    x: *const f32,
    y: *mut f32,
    _k: usize,
) -> usize {
    let n = sc.n_neurons as usize;
    let k0 = sc.k0;
    let mut macs = 0usize;
    for i in 0..n {
        let base = i * k0;
        let mut acc = 0.0f32;
        let mut first = true;
        for j in 0..k0 {
            if *sc.act.add(base + j) == 0.0 {
                continue; // 零连接跳过 —— 不乘、不加、不计数
            }
            let sid = *sc.src.add(base + j) as usize;
            let term = *sc.w.add(base + j) * *x.add(sid);
            acc = if first { term } else { acc + term };
            first = false;
            macs += 1;
        }
        *y.add(i) = acc;
    }
    macs
}

/// T3：反向 —— 只对**激活**连接累积 dW。
///
///   dW[i][j] = dy[i] * x[src[i][j]]     若 act[i][j] == 1
///   dW[i][j] = 0                        若 act[i][j] == 0   （**严格不动**）
///
/// 与稠密 `gemm_bwd_w` 的唯一差异就是那条掩码判断。其余（累加语义、
/// 行的顺序）与稠密版**完全一致**，这样「稀疏 = 稠密 加上一个掩码」
/// 这条等式可以被逐位验证，而不是靠两套代码「看起来一样」。
///
/// ⚠️ **累加**（+=）而不是赋值：与 `train::gemm_bwd_w` 的契约一致 ——
///   引擎侧的反向指令全部是累加语义（调用方先 OP_ZERO 清零）。
///   改成赋值会让「一个 batch 里多帧的梯度」互相覆盖，而 loss 仍会下降
///   （只是慢），于是这个 bug 很难被发现。保持累加是与既有契约对齐。
///
/// 返回**实际写入梯度的连接数**（= 激活连接数），供判据使用。
///
/// # Safety
/// `dw` 至少 E 个 f32；`dy` 至少 n_neurons 个 f32。
pub unsafe fn sparse_gemm_bwd_w(
    sc: &SparseConnW,
    dy: *const f32,
    x: *const f32,
    dw: *mut f32,
) -> usize {
    let n = sc.n_neurons as usize;
    let k0 = sc.k0;
    let mut written = 0usize;
    for i in 0..n {
        let g = *dy.add(i);
        let base = i * k0;
        for j in 0..k0 {
            // ---- 唯一的算法差异：非激活连接梯度**不写** ----
            //
            // 注意这里是「不写」而不是「写 0 再让优化器跳过」：写 0 会让
            // Adam 的 m/v 继续按 0 更新（m 衰减、v 衰减），t 继续推进，
            // 于是该权的有效学习率**漂移**。不写则 m/v 保持上次的值……
            //
            // ⚠️ 这里必须说清楚一个反直觉之处：**不写**并不等于 m/v 冻结，
            // 因为 Adam 仍会对该位置做 `m = b1*m + (1-b1)*0`。所以真正
            // 严格的「权重不动」需要**优化器侧也跳过** —— 本模块因此
            // 提供 `sparse_adam_step`（见下），由它按同一张掩码跳过更新。
            // 只做前一半（这里不写 dw）而用稠密 Adam，权重会缓慢漂移，
            // 而漂移量小到 loss 曲线看不出来 —— 这正是本项目要消灭的
            // 「数值正常但结论错」。判据 `masked_weights_bitwise_frozen`
            // 专门抓它。
            if *sc.act.add(base + j) == 0.0 {
                continue;
            }
            let sid = *sc.src.add(base + j) as usize;
            // 用 += 与稠密版保持同一累加语义
            *dw.add(base + j) += g * *x.add(sid);
            written += 1;
        }
    }
    written
}

/// T2（严格形式）：按**同一张激活掩码**跳过更新的 Adam。
///
/// 为什么需要它 —— 见 `sparse_gemm_bwd_w` 里那段注释：只在反向不写 dw
/// 是不够的，因为 Adam 的 `m = b1*m + (1-b1)*g` 在 g=0 时仍会推进，
/// 于是未激活权重会被**以自己的动量**缓慢推动。这里显式跳过：
///   m、v、p 三者在未激活位置**逐位不变**（不是「变化很小」）。
///
/// `t` 的推进：仍然每次调用推进一次（这是 Adam 的全局步数，不是每参数
/// 的属性）。所以 `bias correction` 对所有参数一致 —— 这与「未激活参数
/// 完全不动」不矛盾：不动的参数根本不读 c1/c2。
///
/// 返回实际更新的参数个数（= 激活连接数）。
///
/// # Safety
/// `p`、`g`、`m`、`v` 各至少 E 个 f32。
pub unsafe fn sparse_adam_step(
    p: *mut f32,
    g: *const f32,
    m: *mut f32,
    v: *mut f32,
    act: *const f32,
    e: usize,
    lr: f32,
    b1: f32,
    b2: f32,
    eps: f32,
    t: u32,
) -> usize {
    let tf = t as f32;
    let c1 = 1.0 - pow_f(b1, tf);
    let c2 = 1.0 - pow_f(b2, tf);
    let mut upd = 0usize;
    for i in 0..e {
        if *act.add(i) == 0.0 {
            continue; // m / v / p 全部逐位不变
        }
        let gi = *g.add(i);
        let mi = b1 * *m.add(i) + (1.0 - b1) * gi;
        let vi = b2 * *v.add(i) + (1.0 - b2) * gi * gi;
        *m.add(i) = mi;
        *v.add(i) = vi;
        *p.add(i) -= lr * (mi / c1) / (sqrt_f(vi / c2) + eps);
        upd += 1;
    }
    upd
}

#[inline]
fn pow_f(b: f32, e: f32) -> f32 {
    crate::kernels::exp_f(e * crate::kernels::ln_f(b))
}

#[inline]
fn sqrt_f(x: f32) -> f32 {
    crate::kernels::sqrt_f(x)
}

// ============================================================================
// 4. 训练后统计（任务书要求：有效连接数分布、死神经元比例）
// ============================================================================

/// 一个层的统计结果。**全部是计数**，不是布尔值 —— 本项目要求每个检查
/// 报「差多少」而不是「对/错」。
#[derive(Clone, Copy, Default)]
pub struct ConnStats {
    /// 神经元数。
    pub n: u32,
    /// 每神经元连接数 k0。
    pub k0: u32,
    /// 权重**非零**的连接数（|w| > eps）。任务书的「有效连接数」。
    pub effective: u32,
    /// 权重**恰好为 0** 的连接数。
    pub zero_w: u32,
    /// 死神经元数：该神经元的 k0 条连接权重**全部**为 0。
    ///
    /// 定义必须写死，否则这个数没有意义。这里取「全部 k0 条权重为 0」，
    /// 因为一个还有非零权重的神经元**仍然能产生输出**（不是死的）；
    /// 而「输出恒为 0」在这个模型里还有第二个来源（所有 src 的 x 都是 0），
    /// 那是**数据**的属性而不是**结构**的属性，不能混进这个数。
    pub dead: u32,
    /// 有效连接数分布：`hist[c]` = 有 c 条有效连接的神经元个数，
    /// c 在 [0, k0]。长度 k0+1。
    ///
    /// ⚠️ 为什么把它做成定长 hist 而不是 Vec：no_std、无分配器。
    /// 调用方给一个 [u32; MAX_K0+1] 的切片。
    pub hist_len: u32,
}

/// 计算统计。`eps` 是「权重算不算有效」的阈值 —— 必须由调用方给，
/// 因为这个阈值**不是**引擎的属性（引擎只做乘加，不定义「有效」）。
///
/// # Safety
/// `sc` 有效；`hist` 至少 k0+1 项。
pub unsafe fn conn_stats(sc: &SparseConnW, eps: f32, hist: *mut u32) -> ConnStats {
    let n = sc.n_neurons as usize;
    let k0 = sc.k0;
    let mut st = ConnStats {
        n: sc.n_neurons,
        k0: k0 as u32,
        hist_len: (k0 + 1) as u32,
        ..Default::default()
    };
    for c in 0..=k0 {
        *hist.add(c) = 0;
    }
    for i in 0..n {
        let base = i * k0;
        let mut eff = 0u32;
        for j in 0..k0 {
            let w = *sc.w.add(base + j);
            let aw = if w < 0.0 { -w } else { w };
            if aw > eps {
                eff += 1;
                st.effective += 1;
            } else {
                st.zero_w += 1;
            }
        }
        *hist.add(eff as usize) += 1;
        if eff == 0 {
            st.dead += 1;
        }
    }
    st
}

// ============================================================================
// 5. 独立参考实现（用于「判据能失败」）
// ============================================================================
//
// ============================ 为什么还要一个「稠密参考」 ============================
//
// 本项目吃过三次同型的亏（P16_report.md §9）：**判据的两端来自同一个实现**，
// 于是它对自己写的错也成立。本模块的对应风险是：
// 「稀疏反向写对了」如果只用本模块自己的前向去验，那就是在验
// 「我算的和我算的一样」。
//
// 所以这里提供一条**结构上不同**的路径：把稀疏层**展开成一个稠密矩阵**
// （E 个非零 + 其余为 0），然后走**稠密** gemm 反向，再抽取非零位置。
// 两条路径的循环结构、索引推导、累加顺序都不同：
//   * 稀疏路径：for i, for j in 0..k0, 读 src[i][j]
//   * 稠密路径：for i, for p in 0..n, 只在 p 是某个 src 时累加
// 于是「两者一致」这条判据**能失败**（src 写错、掩码错、下标推导错
// 任一条都会让不一致计数 > 0）。
//
// ⚠️ 能力边界（诚实标注）：这条参考**不能**证明 top-k 选得对 ——
//    它只证明「给定掩码下，稀疏反向 == 稠密反向」。掩码的正确性由
//    `row_topk` 的对拍（与一个独立的全排序实现比较）覆盖。

/// 稠密参考反向：把稀疏连接展开成 [n][n] 的稠密 dW（只在存的连接上有值）。
///
/// # Safety
/// `dw_dense` 至少 n*n 个 f32（调用方清零）。
pub unsafe fn dense_ref_bwd_w(
    sc: &SparseConnW,
    dy: *const f32,
    x: *const f32,
    dw_dense: *mut f32,
) -> usize {
    let n = sc.n_neurons as usize;
    let k0 = sc.k0;
    let mut written = 0usize;
    // 稠密视角：遍历 (输出 i, 源 p) 的**笛卡尔积**，而不是连接的列表。
    // 这正是与稀疏路径结构上的差异所在。
    for i in 0..n {
        let g = *dy.add(i);
        for p in 0..n {
            // 问：神经元 i 是否有一条连到 p 的**激活**连接？
            // 注意这里可能有**多条**（neuron_conn_src 用模约减，允许重复，
            // 见 conn_sparse.rs 的 G4 注释）—— 所以必须求和，不能找到一条
            // 就 break。第一版就是 break，对重复连接的神经元报出不一致。
            let mut acc = 0.0f32;
            let mut any = false;
            for j in 0..k0 {
                let base = i * k0;
                if *sc.act.add(base + j) == 0.0 {
                    continue;
                }
                if *sc.src.add(base + j) as usize == p {
                    acc += g * *x.add(p);
                    any = true;
                }
            }
            if any {
                *dw_dense.add(i * n + p) += acc;
                written += 1;
            }
        }
    }
    written
}

/// 从稀疏路径的 dw[E] 与稠密参考的 dw_dense[n*n] 抽取比较，
/// 返回**不一致的项数**（不是布尔值）。
///
/// 只比较「存的连接」那些位置 —— 稠密参考在**没存连接**的 (i,p) 上
/// 也可能非零（如果别的神经元连到 p，而 i 有自己的连接……不，这里
/// (i,p) 是唯一的输出-源对，所以不存就是 0）。两个都查：
///   * 存的连接：两条路径必须逐位相同
///   * 没存的位置：稠密参考必须恰好是 0（否则说明「展开了不该展开的」）
///
/// # Safety
/// 长度同上。
pub unsafe fn bwd_mismatch(
    sc: &SparseConnW,
    dw: *const f32,
    dw_dense: *const f32,
) -> usize {
    let n = sc.n_neurons as usize;
    let k0 = sc.k0;
    let mut bad = 0usize;
    // 稠密侧：每个 (i,p) 必须等于「该 (i,p) 的稀疏 dw 之和」
    for i in 0..n {
        for p in 0..n {
            let want = *dw_dense.add(i * n + p);
            let mut got = 0.0f32;
            for j in 0..k0 {
                let base = i * k0;
                if *sc.src.add(base + j) as usize == p {
                    got += *dw.add(base + j);
                }
            }
            // **按位**比较而不是 `!=`：判据是「逐位相同」，
            // 而 `==` 对 NaN 恒假、对 +0.0/-0.0 恒真（见 conn_sparse.rs
            // 的 bitwise_diff_u32 用同一理由）。
            if want.to_bits() != got.to_bits() {
                bad += 1;
            }
        }
    }
    bad
}

/// 独立的「按强度取 top-k」参考实现 —— 用**全排序**（降序）而不是
/// k 轮最大值扫描。
///
/// 为什么需要：`row_topk` 用的是「k 轮扫最大值 + taken 位图」，那是一个
/// 与「排序后取前 k 个」**结构不同**的算法。两条路径对拍，才能证明
/// 选择正确（并且能失败：把 row_topk 的 `>` 改成 `>=` 就会改变平局规则，
/// 在存在相等强度时产生不同结果）。
///
/// 平局规则必须**写死**并与 row_topk 一致，否则对拍会因合法平局而失败：
///   * row_topk：`if v > bestv` ⇒ 平局时保留**先遇到的**（下标更小）
///   * 本函数：插入排序 + 只在 `v > ` 时前移 ⇒ 同样保留下标更小的
///
/// 返回命中数（<= k），`hit` 里是**升序**的下标（与引擎 93 的输出序一致）。
///
/// # Safety
/// 同 row_topk。
pub unsafe fn ref_topk_sorted(
    wrow: *const f32,
    srcrow: *const f32,
    x: *const f32,
    k0: usize,
    k: usize,
    hit: &mut [usize],
) -> usize {
    // 全部下标 + 强度
    let mut idx = [0usize; MAX_K0];
    let mut strv = [0.0f32; MAX_K0];
    if k0 > MAX_K0 {
        crate::print(b"CONN_TRAIN: k0 exceeds MAX_K0\n");
        crate::exit(9);
    }
    for j in 0..k0 {
        idx[j] = j;
        let sid = *srcrow.add(j) as usize;
        strv[j] = (*wrow.add(j) * *x.add(sid)).abs();
    }
    // 稳定插入排序：降序。只在 strict > 时前移 ⇒ 相等时保持原相对顺序
    // ⇒ 等价于「平局保留下标小的」。
    for i in 1..k0 {
        let vi = strv[i];
        let ii = idx[i];
        let mut q = i;
        while q > 0 && strv[q - 1] < vi {
            strv[q] = strv[q - 1];
            idx[q] = idx[q - 1];
            q -= 1;
        }
        strv[q] = vi;
        idx[q] = ii;
    }
    let nk = if k < k0 { k } else { k0 };
    for q in 0..nk {
        hit[q] = idx[q];
    }
    // 升序输出（与引擎 93 相同）：插入排序
    for i in 1..nk {
        let v = hit[i];
        let mut q = i;
        while q > 0 && hit[q - 1] > v {
            hit[q] = hit[q - 1];
            q -= 1;
        }
        hit[q] = v;
    }
    nk
}
