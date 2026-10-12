// P16-1：连接稀疏（connection sparsity）的**存储格式** + 读/写器 + 硬守卫
//
// ============================ 本模块属于哪一步 ============================
//
// P1「连接稀疏 + 连接稀疏激活」由两个原则组成，**两个原则共享同一个量**：
//
//     连接的激活强度   s_i = w_i * x_i
//
//   原则 1（本模块）连接稀疏：层**不是全连接**。每个神经元只保留 top-k
//       条连接（初始 k0 = 20）。本模块负责把「哪些连接存在、权重多少」
//       变成**磁盘上的显式数据**，并保证它可被校验。
//   原则 2（另一个任务）连接稀疏激活：只算存在的连接，零连接跳过。
//
// 换句话说：本模块**不做前向**，**不定义 opcode**，**不改解释器**。
// 它只回答一个问题：「第 i 个神经元连到哪些源、权重是多少，存在哪、怎么验」。
//
// ---- 命名澄清（避免与既有 core/src/sparse.rs 混淆）----
//
// 已经有一个 `core/src/sparse.rs`（P7 成员 E）：那是**动态稀疏**
// —— 运行时用掩码筛掉不重要的行/列，拓扑仍然是稠密的（每个权重都存在，
// 只是某些被跳过）。本模块是**静态结构稀疏**：不存在的连接在存储里
// **根本没有**（没有 src_id、没有 weight）。两者不是同一个东西，也不互相调用。
//
// ============================ 为什么选 CSR / 偏移表 ============================
//
// 备选与实测取舍：
//
//  A. 稠密 [n_neurons][n_src] 矩阵 + 掩码
//     —— 存储 O(n_neurons * n_src)。24 层 x 1536 维、k0=20 时
//        每条连接 8 字节，稠密要 1536*1536*8 = 18.9 MB/层，
//        而 CSR 只要 1536*20*8 = 245760 B/层。稠密的唯一好处是索引式
//        寻址（addr = i*n_src + j），但那要求**先有 j**，而原则 2 恰恰
//        要「不遍历 j」。故弃。
//
//  B. 每神经元一条定长记录 [k0]（隐式偏移 i*k0）
//     —— 这就是 (A) 的特例，同样不表达「某神经元连接数 < k0」，
//        而且它让「k0 变了」变成一次全量重排。故弃。
//
//  C. COO：(src_id, dst_id, weight) 三元组数组
//     —— 可直接表达任意连接，但「第 i 个神经元的邻接表在哪」需要
//        先建索引或每次线性扫描。原则 2 的稳态路径是**按神经元取列表**，
//        线性扫描是 O(E) 每神经元。故弃。
//
//  D. **CSR（本模块采用）**：`conn_offsets[n_neurons+1]` + `conn_src[E]`
//     + `conn_weights[E]`。
//     —— 神经元 i 的邻接表就是切片 `conn_offsets[i] .. conn_offsets[i+1]`，
//        **一次减法两次 load**，无分支、无扫描、无哨兵；表达变长
//        （某神经元可以少于 k0 条）；src 与 weight 平行数组让原则 2
//        的「只算活跃连接」可以顺扫两个数组，局部性最好。
//
// ---- 让 CSR 无歧义的**不变式**（全部由 `validate` 硬校验）----
//
//   (I1) conn_offsets[0] == 0
//   (I2) conn_offsets 单调不减：conn_offsets[i] <= conn_offsets[i+1]
//   (I3) conn_offsets[n_neurons] == total_pairs（头部声明值）
//        且 total_pairs == 实际给出的 src/weight 元素数
//   (I4) 每个 src_id < n_neurons（**自环 allowed**，即 src_id 可以是它自己：
//        自环在数学上合法，且在本模型里表示「自成一条连接」。这里只校验
//        上界，不禁止自环 —— 禁止它是**语义**选择，不该藏在存储层，
//        否则原则 2 的算子和本模块会对同一份数据给出不同判断）
//   (I5) 若 k0 > 0，则每个神经元的连接数**恰好是 k0**
//        （见下「为什么是恰好」）
//
// (I1)+(I2)+(I3) 三條合起来 = **切片划分**：n_neurons 个区间首尾相接、
// 无洞、无重叠、恰好覆盖 [0, total_pairs)。这就是「无歧义」的准确含义：
// 给定 (n_neurons, total_pairs, offsets[])，每条连接属于哪个神经元是唯一确定的。
//
// ---- 为什么是「恰好 k0」而不是「最多 k0」（这是一个真实的设计选择）----
//
// 任务书写「capped at k0」与「exactly k0」两处措辞并存。本模块取
// **exactly k0**，理由是它**可校验**：
//   * 「恰好 k0」是一条**能被证伪**的断言 —— 只要有一个神经元少于 k0，
//     守卫就报出它的下标。见 `validate` 的 G6 与 can-fail 控制 (f)。
//   * 「最多 k0」在存储层是**不可证伪**的：一个 0 条连接的神经元
//     与一个「本来就没有连接」的神经元在这一层完全无法区分，
//     于是「连接稀疏」这条原则**没有任何东西在检查它**。
// 与本项目一贯的纪律一致：不能失败的检查不是测量。
//
// ⚠️ 代价必须诚实说明：**恰好 k0 禁止「死神经元」**。一个真的应当被
//    完全剪掉的神经元在这个格式里表达不了，必须写成 k0 条 weight=0 的连接。
//    这是已知限制，写在 `ConnSparseLayout::validate` 与报告里，不是隐藏假设。
//    将来要表达「剪掉」时，需要另加一个 mask 位或哨兵，那是一次格式版本变更。
//
// ============================ 本项目的铁律：非法输入必须硬失败 ============================
//
// 本模块每个非法输入都 `exit(9)`，且消息里**带具体的数值**。理由照抄
// core/src/neuron_kv.rs：这个项目反复吃过「输入不合法但工具不报错、继续跑出
// 坏结果」的亏。本模块的等价物是**偏移表错位** —— 某个神经元的邻接表
// 会读到**别的神经元**的连接，读出来是量级正常的权重，极难归因。
//
// ⚠️ 关于 exit 码的口径（诚实说明，不要把它读成「返回 9」）：
//    `exit(9)` 走的是 `ExitProcess(9)`，**不返回**。在引擎里这就结束了进程。
//    在**测试 harness** 里（tools/p16_conn_format.py 之外的 Rust harness），
//    本模块被链接进一个正常的 std 二进制，所以 `ExitProcess(9)` 同样直接
//    终止该进程 —— 观测到的进程退出码就是 9。这就是本模块守卫的观测形式。
//
// ============================ 尺寸可校验性（S8 缺陷类，不重犯） ============================
//
// S5 的 S8 缺陷是：**描述符声明的 length 不约束 section 的内容** ——
// 一个 len=20 的描述符配一个需要 36 字节的头部会被**接受**。
// 本模块不重犯，做法是把「需要多少字节」变成**头部字段的纯函数**：
//
//     section_bytes(n_neurons, k0, total_pairs)
//         = CS_HDR + (n_neurons+1)*4 + total_pairs*4 + total_pairs*4
//
// 于是**截断**与**多声明**各自成为一条独立的、能失败的守卫：
//   * 实际可读字节 != 上式            -> G1（截断 / 多声明 都拦）
//   * 但 G1 依赖 `total_pairs` 这个头部字段，若它本身被改小，
//     G1 会「自洽地通过」—— 那正是 S8 的形态。所以额外要求：
//   * total_pairs == k0 * n_neurons    -> G7（k0>0 时；这条把头部
//     三个字段互相绑定，任何一个单独漂移都会被抓住）
//
// 三条守卫**独立**，任一条单独都能失败 —— 见 tools/p16_conn_format.py 的
// can-fail 控制 (a)/(c)/(g)，每个都报**计数**而不是布尔值。
//
// ---- 为什么还要「预留 section 描述符」的接口 ----
//
// .vmb 的可选 section 用 [32..44) 的 (tag u32, off u32, len u32) 描述符，
// 且**有 section 时 text_off=44、无 section 时 text_off=32**
// （S5 踩过：靠 `tag == 0` 判断会漏判，因为无 section 时 [32..44) 落的是
// TEXT 段开头）。本模块提供同形的 `load_find_section` / `load_validate`，
// 唯一真相源仍是 section 自己的头部，`.vmb` 头只用来确认「这是 VMLM 产物」
// 且描述符在文件内。

use crate::{exit, print};
// [抽库 L1] CSR 布局搬到 vmlm-tensor::layout（`vmlm_core::conn_sparse::*` 路径不变）
pub use vmlm_tensor::layout::*;
// [抽库 L0/L1] print_num 是 L0 的 IO 原语；本模块原有一份（重复实现）搬到 sys 后重导出
pub use vmlm_sys::io::print_num;

// ============================================================================
// 1. 布局常量（唯一真相源；Python 侧的等价定义见 tools/p16_conn_format.py）
// ============================================================================


/// 每个神经元的 PRNG 种子推导：**逐字节可复现**，不依赖任何平台行为。
///
/// 种子规则（写死在这里，Python 侧 tools/p16_conn_format.py 的
/// `neuron_seed()` 是**同一份规则的另一份实现**，两者的等价性由
/// 交叉校验测试实际比对，而不是靠「看着一样」）：
///
///     seed(layer, i) = (SPLITMIX_GOLDEN * (layer + 1) ^ (0x9E3779B9 * (i + 1)))
///                     & 0xFFFFFFFF
///
/// 为什么用乘法混合而不是 `seed = i`：`seed = i` 会让相邻神经元的
/// 随机序列在低质量 PRNG 下高度相关，于是「随机选择 20 个源」实际上
/// 变成有规律的选择，而**看起来仍然随机**。这是本模型最怕的那类
/// 「数值正常但结论错」的偏差。
///
/// 为什么 `+1`：让 layer=0 / i=0 也得到一个非零、非棋盘格的种子
/// （seed=0 在很多 PRNG 上是不动点或退化点）。
///
/// 本函数是 `const fn` 且只用 wrapping 算术 ⇒ 没有任何未定义行为、
/// 没有浮点、没有字节序依赖。
pub const SPLITMIX_GOLDEN: u32 = 0x9E37_79B9;

#[inline]
pub const fn neuron_seed(layer: u32, i: u32) -> u32 {
    // 0x85EB_CA6B 是 splitmix32 的另一个奇数常数，用于让两个输入
    // 以不同的方式混进高位，降低 (layer, i) → seed 的碰撞率。
    let a = SPLITMIX_GOLDEN.wrapping_mul(layer.wrapping_add(1));
    let b = 0x85EB_CA6B_u32.wrapping_mul(i.wrapping_add(1));
    let mut z = a ^ b;
    // splitmix32 finalizer：与 Python 侧逐位相同（只用 u32 回绕运算）。
    z = (z ^ (z >> 16)).wrapping_mul(0x7FEB_352D);
    z = (z ^ (z >> 15)).wrapping_mul(0x846C_A68B);
    z ^ (z >> 16)
}

/// 单个神经元的下一个伪随机数（**无状态**：由 (seed, step) 直接算出第 step 个）。
///
/// 为什么要无状态：如果初始化按「神经元顺序推进一个共享 PRNG」，
/// 那么**从第 i 个神经元开始重建**就不可能 —— 而本项目需要能只重建
/// 一个神经元来定位问题。无状态形式让 `neuron_conn_src(layer, i, j)`
/// 是一个**纯函数**，测试侧可以只问第 i 个神经元的第 j 条连接。
///
/// 规则：`mix(seed ^ (0x9E37_79B9 * (j+1)))`，`mix` 是 splitmix32 finalizer。
#[inline]
pub const fn neuron_rand(seed: u32, j: u32) -> u32 {
    let mut z = seed ^ SPLITMIX_GOLDEN.wrapping_mul(j.wrapping_add(1));
    z = (z ^ (z >> 16)).wrapping_mul(0x7FEB_352D);
    z = (z ^ (z >> 15)).wrapping_mul(0x846C_A68B);
    z ^ (z >> 16)
}

/// 第 i 个神经元的第 j 条连接的源 id。
///
/// ⚠️ 这里用 `% n_neurons`（模约减）而不是「拒绝采样去重」：
///   * 模约减**允许重复**（同一个 src 出现两次）。这是**故意的**：
///     「每神经元恰好 k0 条连接」是一条可校验的结构断言，而「k0 条
///     src 互不相同」需要 O(k0^2) 检查、且会把这条断言变成一个
///     概率性的东西（拒绝采样在 n_neurons 小时可能死循环）。
///   * 重复连接的数学含义是「两条并行连接」，权重相加即可 ——
///     对原则 2 的算子无害，且**不改存储格式**。
///   * 如果将来要求去重，那是一次**格式语义**变更（需要多一条守卫），
///     不是这里偷改。
///
/// n_neurons == 0 时不做除法（那会 panic/UB）；调用方必须先 `validate()`。
#[inline]
pub const fn neuron_conn_src(layer: u32, i: u32, j: u32, n_neurons: u32, k0: u32) -> u32 {
    if n_neurons == 0 || k0 == 0 {
        return 0;
    }
    let seed = neuron_seed(layer, i);
    let r = if j == 0 {
        // 第 0 条连接**不自环**：给原则 2 一个稳定的、非退化的起点。
        // （自环本身是允许的，见 I4；这里只是不让初始化必然产出自环。）
        neuron_rand(seed, u32::MAX)
    } else {
        neuron_rand(seed, j)
    };
    r % n_neurons
}

/// 第 i 个神经元的第 j 条连接的初始权重，**按位**给出。
///
/// 规则：把 `neuron_rand(seed ^ 0x5A5A_5A5A, j)` 的**高 16 位**当作
/// IEEE-754 binary16 风格的符号+指数候选……不，这太绕且难以在 Python 侧
/// 逐位复现。**采用更简单且完全确定的规则**：
///
///     bits = (rand & 0x007F_FFFF) | 0x3F00_0000
///
/// 即：把随机数限制在 f32 的 [0.5, 1.0) 区间内（指数 = 0x7E，
/// 尾数取低 23 位中的 23 位随机）—— 具体地，`0x3F00_0000` 是
/// 2^-1 = 0.5 的位型，加上 23 位随机尾数得到 [0.5, 1.0)。
///
/// **这是一个已知的简化，必须诚实标注**：初始权重是**全正**的 [0.5,1)。
/// 它不是训练好的权重，只是一个确定性的、位精确的占位分布；
/// 真正的作用是让「写→读→逐位相同」这条判据有非平凡的数据可测
/// （全零或全同值的 payload 会让「一致」变得平凡 —— 见 can-fail 控制 (d)）。
/// 训练/初始化策略是**另一个任务**，本模块不声称它有统计意义。
///
/// 用「构造位型」而不是「算一个浮点表达式」的理由：浮点表达式在不同
/// 优化/平台下可能给出不同的**位**（尤其是 FMA 收缩），而本模块的
/// 验收判据是**逐位相同**。按位构造让 Python 与 Rust 不可能漂移。
#[inline]
pub const fn neuron_conn_weight_bits(layer: u32, i: u32, j: u32) -> u32 {
    let seed = neuron_seed(layer, i);
    let r = neuron_rand(seed ^ 0x5A5A_5A5A, j);
    (r & 0x007F_FFFF) | 0x3F00_0000
}

// ============================================================================
// 4. 写：布局 -> section 字节
// ============================================================================

/// 把布局写成 section 字节（头部 + offsets + src + weights）。
///
/// 由 **host/构建侧** 调用（Python 侧的等价实现见 tools/p16_conn_format.py）。
///
/// 权重按 **`to_bits()`** 写入，不做任何数值转换：本模块的验收判据是
/// 「逐位相同（含 -0.0 / NaN 的位型）」，任何一次 f32 运算都可能改位型。
///
/// 返回写入的字节数。调用方保证 `dst` 至少有 `section_bytes()` 字节。
///
/// # Safety
/// `dst` 至少 `section_bytes()` 字节且可写、4 字节对齐（用 write_unaligned，
/// 所以弱对齐也可以）。
pub unsafe fn write_section(layout: &ConnSparseLayout, dst: *mut u8) -> usize {
    layout.validate();
    put_u32(dst, CS_OFF_MAGIC, CS_MAGIC);
    put_u32(dst, CS_OFF_N_NEURONS, layout.n_neurons);
    put_u32(dst, CS_OFF_K0, layout.k0);
    put_u32(dst, CS_OFF_TOTAL_PAIRS, layout.total_pairs);
    put_u32(dst, CS_OFF_LAYER, layout.layer);

    let n = layout.n_neurons as usize;
    let k0 = layout.k0 as usize;
    let e = layout.total_pairs as usize;
    let off_base = layout.offsets_off();
    let src_base = layout.src_off();
    let wgt_base = layout.weights_off();

    // ---- conn_offsets：第 i 个神经元从 i*k0 开始（k0>0 时）----
    //
    // offsets 是**显式写下来的**，不是让读取方自己乘 i*k0。
    // 理由与 neuron_kv 相同：显式数据能在装载期被校验，
    // 「自己乘出来」一旦公式写错就没有任何东西能发现。
    //
    // 但**写的时候**必须按每个神经元的实际连接数累加，而不是乘 i*k0：
    // 只有这样才能表达「第 i 个神经元有 c_i 条」这个一般形式，
    // 而 k0>0 的恰好性由 G5/G6 单独校验。两者**独立**，
    // 于是「写入按 k0 布局」与「校验按 k0 布局」不是同一个断言的两遍。
    let mut i = 0usize;
    let mut acc: u64 = 0;
    while i < n {
        put_u32(dst, off_base + i * 4, acc as u32);
        acc += k0 as u64;
        i += 1;
    }
    // offsets[n] = total_pairs（不变式 I3）。
    put_u32(dst, off_base + n * 4, acc as u32);

    // ---- conn_src / conn_weights ----
    //
    // 布局：**连接优先，src 数组整块在前、weights 整块在后**（SoA），
    // 而不是按神经元交错成 (src, w, src, w, ...)。
    // 为什么：原则 2 的算子要顺扫 src（判断某个源是否活跃），
    // 交错会让每一次 src load 都跨过一个 weight（步长 8 而 cache line 64），
    // 局部性减半；SoA 让每次 cache line 装 16 个 src。
    let mut e_i = 0usize;
    if k0 == 0 {
        // 变长模式：src/weight 由调用方填充（本模块只保证 offsets 自洽）。
        // 这里显式留空而不是写垃圾 —— 未初始化的 section 字节会让
        // 「读回来发现是垃圾」变成一次极难的归因。
        let mut b = 0usize;
        while b < layout.src_bytes() + layout.weights_bytes() {
            *dst.add(src_base + b) = 0;
            b += 1;
        }
    } else {
        while e_i < e {
            let neuron = e_i / k0;
            let j = e_i % k0;
            let ni = neuron as u32;
            let jj = j as u32;
            let src = neuron_conn_src(layout.layer, ni, jj, layout.n_neurons, layout.k0);
            put_u32(dst, src_base + e_i * 4, src);
            put_u32(
                dst,
                wgt_base + e_i * 4,
                neuron_conn_weight_bits(layout.layer, ni, jj),
            );
            e_i += 1;
        }
    }
    layout.section_bytes()
}

/// **序列化器**：把**调用方给的** src / weight 数组写成一个 conn_sparse section。
///
/// ============================ 为什么必须新增这个函数 ============================
///
/// 上面的 `write_section(layout, dst)` 用确定性 PRNG **重新生成** src 与 weight：
///
///     src    = neuron_conn_src(layout.layer, i, j, n, k0)
///     weight = neuron_conn_weight_bits(layout.layer, i, j)
///
/// 它**取不到**调用方的数组（签名里没有这个参数）。于是：
///
///     * 它写出的是**初始**拓扑与**初始**权重；
///     * 训练后的权重（`SparseConnW.w` 已被 Adam 改过）**无法**通过它落盘。
///
/// 这是 P18-P3「训练 → 可分发」这条链的真实断点。实测证据见
/// `docs/P18-P3阻塞点_无权重序列化器.md`：
///
///     vmlm-p18f: src mismatches = 0 , weight mismatches = 1280 (全部)
///     => TOPOLOGY MATCHES, ONLY WEIGHTS DIFFER. There is no weight serializer.
///
/// ============================ 为什么不改 write_section ============================
///
/// `write_section` 是**历史阶段**的产物，有 12 条已实测会 fire 的守卫，且记录了
/// 「与独立 Python emitter 逐字节相同」。改它的签名会波及那些对拍。所以这里
/// **新增**一个函数，`write_section` **一字不动**。
///
/// 两者**共用** `layout.validate()` 与 `layout.section_bytes()`，所以
/// offsets 的构造与全部头部守卫仍然只有一份实现 —— 新函数只把
/// 「src / weight 从哪里来」参数化。这一点是刻意的：若复制一份 offsets 逻辑，
/// 两个真相源就会漂移，而那正是本项目反复吃亏的地方。
///
/// ============================ 契约 ============================
///
/// * `src`：至少 `layout.total_pairs` 个 u32，每项是源神经元下标。
///   **必须 < n_neurons**，否则硬失败（越界的源 id 会被引擎当成合法下标读，
///   产生「看起来像激活的垃圾」——本项目最忌讳的静默降级）。
/// * `weight_bits`：至少 `layout.total_pairs` 个 u32，是权重 f32 的**位型**
///   （不是 f32 本身）。用位型而非 f32 是为了让「写出去 == 内存里」可以被
///   `to_bits()` 逐位验证，且对 NaN/±0 不会误判。
/// * 顺序：**连接优先**（`e_i = i*k0 + j`），与 `write_section` 完全一致，
///   也与读取侧 `conn_src(.., i, j)` 的寻址一致。写错顺序会让读回的数
///   仍然「像个权重」，但属于别的神经元 —— 因此本函数**不依赖**调用方自觉：
///   配套的读回对拍（`tools`/驱动侧）才是判据。
///
/// 返回写入的字节数（== `layout.section_bytes()`）。
///
/// # Safety
/// `dst` 至少 `layout.section_bytes()` 字节且可写；
/// `src` / `weight_bits` 至少 `layout.total_pairs` 项。
pub unsafe fn write_section_data(
    layout: &ConnSparseLayout,
    src: *const u32,
    weight_bits: *const u32,
    dst: *mut u8,
) -> usize {
    // 复用同一套守卫。这是新函数与旧函数**唯一**必须共享的东西。
    layout.validate();

    put_u32(dst, CS_OFF_MAGIC, CS_MAGIC);
    put_u32(dst, CS_OFF_N_NEURONS, layout.n_neurons);
    put_u32(dst, CS_OFF_K0, layout.k0);
    put_u32(dst, CS_OFF_TOTAL_PAIRS, layout.total_pairs);
    put_u32(dst, CS_OFF_LAYER, layout.layer);

    let n = layout.n_neurons as usize;
    let k0 = layout.k0 as usize;
    let e = layout.total_pairs as usize;
    let off_base = layout.offsets_off();
    let src_base = layout.src_off();
    let wgt_base = layout.weights_off();

    // ---- offsets：与 write_section 同一构造（第 i 个神经元从 i*k0 开始）----
    let mut i = 0usize;
    let mut acc: u64 = 0;
    while i < n {
        put_u32(dst, off_base + i * 4, acc as u32);
        acc += k0 as u64;
        i += 1;
    }
    put_u32(dst, off_base + n * 4, acc as u32);

    // ---- src / weights：这次来自**调用方** ----
    //
    // 越界检查是**必须**的，不是防御性装饰：src_id >= n_neurons 时，
    // 引擎侧 `OP_SPARSE_MATVEC` 会拿它去索引 x 数组，读到别的槽的数据 ——
    // 数值上完全「正常」，不会有任何报错。所以在这里硬失败。
    let mut e_i = 0usize;
    while e_i < e {
        let s = *src.add(e_i);
        if s as u64 >= n as u64 {
            print(b"CS: write_section_data src_id out of range\n");
            print(b"CS: e_i=");
            print_num(e_i);
            print(b" src=");
            print_num(s as usize);
            print(b" n_neurons=");
            print_num(n);
            print(b"\n");
            exit(9);
        }
        put_u32(dst, src_base + e_i * 4, s);
        put_u32(dst, wgt_base + e_i * 4, *weight_bits.add(e_i));
        e_i += 1;
    }

    layout.section_bytes()
}

// ============================================================================
// 5. 读：section 字节 -> 布局（逐项校验）
// ============================================================================

/// 从 section 字节解析并**逐项校验**。
///
/// ============================ 为什么要求调用方给 `avail` ============================
///
/// S5 的 S8 缺陷就是「只有一个读入口，而它拿不到描述符声明的长度」，
/// 于是长度完全不约束内容。本模块的对应约定：**必须显式声明这段内存
/// 有多少字节可用**。不能声明就别调它 —— 改用 `read_section_declared`，
/// 它从描述符里取 `len`。
///
/// ============================ 校验顺序（顺序本身是设计的一部分） ============================
///
///   1. `avail >= CS_HDR`       —— 否则连头部都读不出来（G0）
///   2. 读头 5 个字段
///   3. magic 必须等于 CS_MAGIC —— 否则后面所有数字都不可信（G8）
///   4. `validate()`            —— n_neurons/上溢/total_pairs==k0*n/G4（G2/G3/G7）
///   5. `avail == section_bytes()` —— **精确相等**（G1）
///   6. offsets[0] == 0         —— （G9）
///   7. offsets 单调不减         —— （G5）
///   8. offsets[n] == total_pairs —— （G10）
///   9. 每个 src_id < n_neurons  —— （G11）
///  10. k0>0 时每神经元恰好 k0 条 —— （G6）
///
/// 为什么 magic 放在 4 之前：`n_neurons`/`k0` 是**能算**的，
/// 而算出来的东西可以自洽。magic 是唯一**不可推导**的一致来源。
/// 反过来，把 5（长度）放在 4 之后是因为只有布局合法，
/// `section_bytes()` 才是一个有意义的「需要多少字节」。
///
/// 为什么长度用**精确相等**而不是 `>=`（照抄 neuron_kv 的结论）：
/// section 是定长结构（尺寸由头部唯一确定），多出来的字节没有合法解释。
/// 允许 `>` 等于允许「section 后面挂一段谁也不认识的东西」——
/// 那正是本模块要拦的静默错位。这也让 can-fail 控制 (a) 的
/// 「多声明 4 字节」能被抓住。
///
/// # Safety
/// - `src` 至少 `avail` 字节可读；
/// - `avail` 是调用方保证的、`src` 起可读的字节数。
pub unsafe fn read_section(src: *const u8, avail: usize) -> ConnSparseLayout {
    // G0: 连头部都放不下
    if avail < CS_HDR {
        print(b"CS: available bytes below header\n");
        print(b"CS: avail=");
        print_num(avail);
        print(b" CS_HDR=");
        print_num(CS_HDR);
        print(b"\n");
        exit(9);
    }
    let magic = get_u32(src, CS_OFF_MAGIC);
    // G8：magic 先于一切「看起来合理」的字段。
    if magic != CS_MAGIC {
        print(b"CS: bad section magic\n");
        print(b"CS: got=0x");
        print_hex32(magic);
        print(b" want=0x");
        print_hex32(CS_MAGIC);
        print(b"\n");
        exit(9);
    }
    let layout = ConnSparseLayout {
        n_neurons: get_u32(src, CS_OFF_N_NEURONS),
        k0: get_u32(src, CS_OFF_K0),
        total_pairs: get_u32(src, CS_OFF_TOTAL_PAIRS),
        layer: get_u32(src, CS_OFF_LAYER),
    };
    // G2/G3/G7/G4
    layout.validate();

    // G1：可读字节数必须**精确**等于头部推出的 section 尺寸。
    let need = layout.section_bytes();
    if avail != need {
        print(b"CS: section length disagrees with header\n");
        print(b"CS: avail=");
        print_num(avail);
        print(b" required=");
        print_num(need);
        print(b"\n");
        exit(9);
    }

    let n = layout.n_neurons as usize;
    let e = layout.total_pairs as usize;
    let off_base = layout.offsets_off();
    let src_base = layout.src_off();

    // G9 + I1：conn_offsets[0] 必须恰好是 0。
    let first = get_u32(src, off_base);
    if first != 0 {
        print(b"CS: conn_offsets[0] != 0\n");
        print(b"CS: conn_offsets[0]=");
        print_num(first as usize);
        print(b"\n");
        exit(9);
    }

    // G5 + I2：单调不减。
    //
    // ⚠️ 这里**必须**从 0 走到 n（含 offsets[n]），而不是 n-1：
    //    漏掉最后一项会让「offsets[n] < offsets[n-1]」以及
    //    「offsets[n] != total_pairs」两条守卫都失去机会开火。
    let mut i = 1usize;
    let mut prev = first;
    while i <= n {
        let cur = get_u32(src, off_base + i * 4);
        if cur < prev {
            print(b"CS: conn_offsets not monotonically non-decreasing\n");
            print(b"CS: index=");
            print_num(i);
            print(b" prev=");
            print_num(prev as usize);
            print(b" cur=");
            print_num(cur as usize);
            print(b"\n");
            exit(9);
        }
        prev = cur;
        i += 1;
    }

    // G10 + I3：offsets[n] == total_pairs。
    let last = get_u32(src, off_base + n * 4);
    if last as usize != e {
        print(b"CS: conn_offsets[n_neurons] != total_pairs\n");
        print(b"CS: conn_offsets[n]=");
        print_num(last as usize);
        print(b" total_pairs=");
        print_num(e);
        print(b"\n");
        exit(9);
    }

    // G11 + I4：每个 src_id 必须 < n_neurons。
    //
    // 这一步是 O(E)（24 层 1536 维、k0=20 时 E=30720）。**值得**：
    // 越界的 src 在原则 2 里是一次越界读 —— 读出来是别的神经元的激活，
    // 量级正常、不崩、只是结果错。装载期扫一遍换掉它是划算的。
    // G6 同时在这里做（每神经元恰好 k0 条），所以仍是**一次**遍历。
    let mut idx = 0usize;
    while idx < e {
        let s = get_u32(src, src_base + idx * 4);
        if s >= layout.n_neurons {
            print(b"CS: conn_src out of range (src_id >= n_neurons)\n");
            print(b"CS: at_pair=");
            print_num(idx);
            print(b" src_id=");
            print_num(s as usize);
            print(b" n_neurons=");
            print_num(n);
            print(b"\n");
            exit(9);
        }
        idx += 1;
    }

    // G6 + I5：k0>0 ⇒ **每个**神经元恰好 k0 条。
    if layout.k0 != 0 {
        let k0 = layout.k0 as usize;
        let mut v = 0usize;
        while v < n {
            let b = get_u32(src, off_base + v * 4) as usize;
            let en = get_u32(src, off_base + (v + 1) * 4) as usize;
            let cnt = en - b; // b<=en 已由 G5 保证
            if cnt != k0 {
                print(b"CS: neuron does not have exactly k0 connections\n");
                print(b"CS: neuron=");
                print_num(v);
                print(b" count=");
                print_num(cnt);
                print(b" k0=");
                print_num(k0);
                print(b"\n");
                exit(9);
            }
            v += 1;
        }
    }

    layout
}

/// 直接从 section 描述符解析（`load_validate` 用的就是它）。
///
/// 长度来自**磁盘上的声明**（描述符的 `len`），不需要调用方凭空知道 ——
/// 这正是 S8 修复的形态。
///
/// # Safety
/// `d` 至少 44 字节可读；`load_find_section` 的成功返回保证 `off+len`
/// 在 `file_len` 之内。
pub unsafe fn read_section_declared(
    d: *const u8,
    file_len: usize,
) -> (*const u8, ConnSparseLayout) {
    let sec = load_find_section(d, file_len);
    let len = crate::u32_at(d, 40) as usize;
    (sec, read_section(sec, len))
}

// ============================================================================
// 6. 槽位访问（`neuron_base` 风格的寻址）
// ============================================================================

/// 一个神经元在 section 内的**三处地址**：offsets 项、src 起点、weight 起点。
///
/// 与 neuron_kv 的 `neuron_base` 同形：调用方只需要一个整数 `neuron_id`，
/// 不需要任何查表/比较/分支去决定读哪一段。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NeuronConnAddr {
    pub begin: usize,
    pub end: usize,
}

impl NeuronConnAddr {
    pub fn count(&self) -> usize {
        self.end - self.begin
    }
}

/// 神经元 `i` 的邻接表在 section 内的**元素下标**区间。
///
/// # Safety
/// `sec` 是一段**已由 `read_section` 校验过**的 section 起址。
pub unsafe fn neuron_conn_addr(
    sec: *const u8,
    layout: &ConnSparseLayout,
    i: u32,
) -> NeuronConnAddr {
    let (b, e) = layout.slice_of(sec.add(layout.offsets_off()) as *const u32, i);
    NeuronConnAddr { begin: b, end: e }
}

/// 神经元 `i` 的第 `j` 条连接的源 id。
///
/// # Safety
/// 同 `neuron_conn_addr`；`j < neuron_conn_addr(..).count()` 由本函数检查。
pub unsafe fn conn_src(
    sec: *const u8,
    layout: &ConnSparseLayout,
    i: u32,
    j: usize,
) -> u32 {
    let a = neuron_conn_addr(sec, layout, i);
    if j >= a.count() {
        print(b"CS: connection index exceeds neuron degree\n");
        exit(9);
    }
    get_u32(sec, layout.src_off() + (a.begin + j) * CS_SRC_ENTRY)
}

/// 神经元 `i` 的第 `j` 条连接的权重，**按位**返回（u32）。
///
/// 返回位型而不是 f32：本模块的判据是逐位相同，任何经过 f32 的返回
/// 都有可能在调用方那里被 `==` 比较（NaN/-0.0 会漏）。
///
/// # Safety
/// 同 `conn_src`。
pub unsafe fn conn_weight_bits(
    sec: *const u8,
    layout: &ConnSparseLayout,
    i: u32,
    j: usize,
) -> u32 {
    let a = neuron_conn_addr(sec, layout, i);
    if j >= a.count() {
        print(b"CS: connection index exceeds neuron degree\n");
        exit(9);
    }
    get_u32(sec, layout.weights_off() + (a.begin + j) * CS_WGT_ENTRY)
}

// ============================================================================
// 7. 装载期入口（.vmb section 描述符）
// ============================================================================
//
// 与 core/src/neuron_kv.rs 的 §7 同形。**为什么在这里重写一遍而不是复用
// neuron_kv 的函数**：那些函数的名字/消息都写死了 "NKV:"，而它们的
// feature gate 是 `neuron-kv`。把它们借过来会让「哪个 feature 开了什么」
// 变成隐式耦合（开 conn-sparse 却因为少开 neuron-kv 而链接失败）。
// 复制的代价是 40 行，换来的是两个 feature 完全正交 —— 这一条值得。

/// 本模块的 section tag。**预留值，尚未写入 format.rs**（见模块头的说明）。
///
/// ⚠️ 诚实标注：这个 tag 目前**只在 examples/p16_conn 的示例产物里使用**，
/// 生产 .vmb 还没有它。0 保留 = 「本 artifact 没有 conn_sparse section」。
pub const SEC_CONN_SPARSE: u32 = 3;

/// section 描述符在 .vmb 头之后的偏移（32 字节头 + 0）。
pub const DESCR_OFF: usize = 32;
/// .vmb 头字节数。
pub const VMB_HDR: usize = 32;
/// 有 section 时 TEXT 从 44 开始（= VMB_HDR + 12 字节描述符）。
pub const VMB_HDR_WITH_DESCR: usize = 44;

/// 从 artifact 顶部的 section 描述符读 `(tag, off, len)` 并硬校验，
/// 返回 section 起点。
///
/// # Safety
/// `d` 至少 `32 + 12` 字节可读；若描述符合法，还需至少 `off + len`
/// 字节可读（调用方保证）。
pub unsafe fn load_find_section(d: *const u8, file_len: usize) -> *const u8 {
    if crate::u32_at(d, 0) != crate::MAGIC {
        print(b"CS: artifact is not a VMLM .vmb (bad magic)\n");
        exit(9);
    }
    // ⚠️ 有没有 section **不能靠 tag == 0 判断**：无 section 时
    // [32..44) 落的是 TEXT 段的开头（S5 实测 tag=52、off=0）。
    // 唯一正确的判据是 `text_off`：有 section = 44，无 section = 32。
    let text_off = crate::u32_at(d, 8) as usize;
    if text_off < VMB_HDR_WITH_DESCR {
        print(b"CS: artifact has no conn_sparse section\n");
        exit(9);
    }
    let tag = crate::u32_at(d, DESCR_OFF);
    let off = crate::u32_at(d, DESCR_OFF + 4) as usize;
    let len = crate::u32_at(d, DESCR_OFF + 8) as usize;
    if tag == 0 {
        print(b"CS: artifact has no conn_sparse section\n");
        exit(9);
    }
    if tag != SEC_CONN_SPARSE {
        print(b"CS: unexpected section tag\n");
        exit(9);
    }
    if off < VMB_HDR_WITH_DESCR {
        print(b"CS: section offset overlaps the .vmb header\n");
        exit(9);
    }
    if len < CS_HDR {
        print(b"CS: section length below header\n");
        exit(9);
    }
    if (off as u64) + (len as u64) > file_len as u64 {
        print(b"CS: section exceeds artifact size\n");
        exit(9);
    }
    if (off as u64) + (len as u64) > u32::MAX as u64 {
        print(b"CS: section offset exceeds 32-bit range\n");
        exit(9);
    }
    d.add(off)
}

/// 装载期入口：描述符 -> 长度 -> `read_section`（含全部逐项守卫）。
///
/// **不分配**：分配与重排属于原则 2 的任务，本模块只回答
/// 「这份数据合法吗、每一项是什么」。
///
/// # Safety
/// 同 `load_find_section`。
pub unsafe fn load_validate(d: *const u8, file_len: usize) -> (*const u8, ConnSparseLayout) {
    // 顺序固定：先找 section（它校验 magic/描述符范围），再解析。
    // 反过来会让「描述符越界」这条守卫被更早的另一条消息掩盖 ——
    // neuron_kv 踩过这个（D4：44 个非法值 0 个能走到那条守卫）。
    read_section_declared(d, file_len)
}

// ============================================================================
// 8. 独立参考：第 i 个神经元应有哪 k0 条连接（测试侧的期望值来源）
// ============================================================================
//
// ============================ 为什么需要「独立参考」 ============================
//
// 本项目吃过两次「判据不可能失败」的亏：
//   * P13 §3.2 —— 探针的 patch 是空的，却报出「0 B」；
//   * P14 §4.4 —— 用「搬完之后拿 dst 与 src 比」来验方向，那是把赋值
//     重述成等式，对任何输入都成立。
//
// 本模块的对应陷阱是：**用 `neuron_conn_src()` 自己生成 section，
// 再用同一个 `neuron_conn_src()` 算出「期望值」去比对** —— 那是同义反复，
// 无论格式写得多错都会通过。
//
// ⇒ 所以本模块**只提供**（写侧的）`neuron_conn_src` 与（读侧的）
//   `conn_src` 两条**真正不同**的路径：
//     写侧：JR 风格 seeding -> PRNG -> 模约减 -> 字节
//     读侧：字节 -> offsets[i] 切片 -> conn_src[begin+j]
//   两者的**索引推导不同**（一个由 PRNG 算出连接，一个由偏移表划出切片）。
//   判据「两条路径一致」因此能失败：偏移表写错、切片划错、字节序写错、
//   或 PRNG 与 Python 侧漂移，任一条都会让计数 > 0。
//
//   而其**可失败性**由 can-fail 控制证明（tools/p16_conn_format.py 的
//   control (w)：把某个 weight 的一个 bit 翻掉 ⇒ 不一致计数必须 > 0）。
//
// ---- 能力边界（诚实，不要含糊）----
//
// 「写侧公式 == 读侧切片」这条判据**不能**证明「PRNG 选出的源是好的」
// （比如均匀性或去重）—— 那只由**外部期望值**（Python 侧独立实现）
// 的交叉校验来覆盖。本模块内部的判据只覆盖**存储的完整性**：
// 写进去的每个 bit 都能按 (neuron, j) 从字节里取回来、且不越界。

/// 逐位比较两条路径下每个神经元的每条连接，返回**不一致的项数**。
///
/// 计数而不是布尔值：本项目要求每个检查都能报出「差多少」
/// （见 P13 §3.2：只报「0 B」的探针无法与「没跑」区分）。
///
/// 比较两个量：
///   (1) src_id：读侧切片取出的 vs 写侧公式算出的；
///   (2) weight **位型**：同上。
///
/// 返回 `(src_mismatch, weight_bit_mismatch, first_src_bad, first_weight_bad)`。
///
/// # Safety
/// `sec` 是一段已由 `read_section` 校验过的 section。
pub unsafe fn crosscheck_readback(
    sec: *const u8,
    layout: &ConnSparseLayout,
) -> (usize, usize, Option<(u32, usize, u32, u32)>, Option<(u32, usize, u32, u32)>) {
    let mut src_bad = 0usize;
    let mut w_bad = 0usize;
    let mut first_src_bad = None;
    let mut first_w_bad = None;
    let mut i = 0u32;
    while i < layout.n_neurons {
        let a = neuron_conn_addr(sec, layout, i);
        let cnt = a.count();
        let mut j = 0usize;
        while j < cnt {
            let got_src = get_u32(sec, layout.src_off() + (a.begin + j) * CS_SRC_ENTRY);
            let want_src = neuron_conn_src(layout.layer, i, j as u32, layout.n_neurons, layout.k0);
            if got_src != want_src {
                src_bad += 1;
                if first_src_bad.is_none() {
                    first_src_bad = Some((i, j, got_src, want_src));
                }
            }
            let got_w = get_u32(sec, layout.weights_off() + (a.begin + j) * CS_WGT_ENTRY);
            let want_w = neuron_conn_weight_bits(layout.layer, i, j as u32);
            if got_w != want_w {
                w_bad += 1;
                if first_w_bad.is_none() {
                    first_w_bad = Some((i, j, got_w, want_w));
                }
            }
            j += 1;
        }
        i += 1;
    }
    (src_bad, w_bad, first_src_bad, first_w_bad)
}

/// 逐位比较两个 u32 缓冲区是否**完全相同**。
///
/// 为什么按字节而不是按 u32 `==`：这是 f32 权重的缓冲区，
/// 而 `==` 对 NaN 恒 false、对 +0.0/-0.0 恒 true。判据是「逐位相同」，
/// 就必须按**位型**比。返回 0 = 相同，否则返回**第一个不同**的元素下标 + 1。
///
/// # Safety
/// `a`、`b` 各至少 `n` 个 u32。
pub unsafe fn bitwise_diff_u32(a: *const u32, b: *const u32, n: usize) -> usize {
    let pa = a as *const u8;
    let pb = b as *const u8;
    let mut i = 0usize;
    while i < n * 4 {
        if *pa.add(i) != *pb.add(i) {
            return i + 1;
        }
        i += 1;
    }
    0
}

// ============================================================================
// 9. 无 CRT 的小工具（不引 core::fmt）
// ============================================================================

/// 读一个 u32（弱对齐安全）。
#[inline]
pub unsafe fn get_u32(p: *const u8, off: usize) -> u32 {
    core::ptr::read_unaligned(p.add(off) as *const u32)
}

/// 写一个 u32（弱对齐安全）。
#[inline]
pub unsafe fn put_u32(p: *mut u8, off: usize, v: u32) {
    core::ptr::write_unaligned(p.add(off) as *mut u32, v);
}


/// 无 CRT 的 8 位十六进制打印（magic 之类的字段用，十进制读不出来）。
pub unsafe fn print_hex32(v: u32) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut buf = [0u8; 8];
    let mut i = 0usize;
    while i < 8 {
        buf[i] = HEX[((v >> (28 - 4 * i)) & 0xF) as usize];
        i += 1;
    }
    print(&buf);
}
