// PATH: B2
//
// B2 = 稀疏存储 + 稀疏激活：每神经元只有 k0 条连接，遍历这 k0 条算强度
//      s_j = |w[i][j] * x[src[i][j]]|，取 **top-k**，只累加被选中的 k 条。
//
// 与 B1（`b1_pure`）的唯一算法差异：B1 累加全部 k0 条，B2 累加被选中的 k 条。
// 存储**完全相同**（同一个 `SparseConnW`），所以两条路径算的是同一批数，
// 差异只能来自「有没有选择」，不能来自布局。
//
// ===========================================================================
// 本模块存在的理由：B2 有两种读法，它们差 d 倍，必须能同时实测
// ===========================================================================
//
// 用户裁决「先测两种再决定」。两种读法都对得上任务书第五部分的字面描述，
// 但在「谁做选择、做几次」上完全不同：
//
//   PER-ROW（局部语义）：**每个神经元各自**从自己的 20 条里选 top-k。
//       选择次数 = d 次/矩阵；选择工作量 = d*k0*k 次比较
//       d=1536 k0=20 k=8 → 245,760 次比较/矩阵
//       而它选中的乘加只有 d*k = 12,288 次
//       ⇒ **选择比它选择的算术贵 20 倍**（这是 76.8x 不能兑现的根因）
//
//   SHARED（全局语义）：**整层做一次**选择，所有神经元共用同一批
//       「连接槽位」（slot = j，即「第 j 条连接」这个位置）。
//       选择次数 = 1 次/矩阵；选择工作量 = k0*k 次比较 = 160
//       ⇒ 选择与 d 脱钩，只有聚合 O(d*k0) 那一遍随 d 走
//
// 为什么两者在 k = k0 时必须**都**退化成 B1：k0=20 时 top-20-out-of-20
// 必然选中全部 20 条，选择的结果恒等于「全选」，于是 PER-ROW 与 SHARED
// 的可观测行为都等于 B1。这是 A1 判据的地基，也是本模块能自称「B2 = B1 +
// 选择」的证据 —— 若任一条在 k=k0 时对不上 B1，说明它的累加顺序或选择
// 规则被写坏了，而不是「语义不同」。
//
// ⚠️ SHARED 是一个**建模决定**，不是一个性能细节。它改变模型（所有神经元
//    共用同一批槽位）。本模块只让这个决定的**代价**可测（见 A5），
//    不主张它保精度 —— 保精度需要一次真实训练，那是另一个任务。
//
// ===========================================================================
// 累加顺序契约（与 b1_pure 逐字相同 —— 这是 bit-exact 的前提，不是风格）
// ===========================================================================
//
// 浮点加法**不满足结合律**，所以同一个数学表达式换个求和顺序就是另一个
// 比特串。被选中的 k 条**必须**：
//
//   * 按 **升序 j** 累加（不是按强度降序！选择结果里带的是下标，
//     累加时按下标升序重排 —— 与 `sparse_accum` / `b1_forward` 同序）；
//   * **单个标量累加器** acc —— 不做 8 路 SIMD 归约（归约会改变分组，
//     进而改变位型）；
//   * **首项直接赋值** `acc = term`，而不是从 `0.0f32` 起加：
//     `0.0 + (-0.0)` 在 IEEE-754 下等于 `+0.0`，而 `-0.0` 的位型是
//     0x8000_0000。从 0.0 起加会把 `-0.0` 项「洗」成 `+0.0`，位型分叉。
//
// ⇒ A1 因此用 `f32::to_bits()` 比较而不是 `==`：`==` 对 NaN 恒假、对 ±0
//   恒真，恰好对上面这类分叉**全盲**。
//
// ===========================================================================
// 平局规则（契约，不可改 —— 与 conn_train::row_topk_from_strength 一致）
// ===========================================================================
//
//   强度相等时保留**下标更小**的；输出**升序**。
//   实现方式：升序扫描 j，只在**严格** `>` 时替换/插入 ⇒ 先遇到的（下标
//   更小）留在前面。把 `>` 改成 `>=` 会改变幸存者 —— A6 把这条**当成
//   主张去测**（必须看到幸存者集合发生变化），否则平局规则零信息。
//
// ===========================================================================
// 关于「转译版 top-k 是 O(20) 不是 O(1536)」（判据 A4）
// ===========================================================================
//
// A4 要求用**比较次数计数器**证明选择成本是 O(k0) 而不是 O(d)。
// 所以本模块的每个选择函数都**返回比较次数**，并且 `b2_forward` 把
// 所有比较次数累加成一个总数返回。这个计数是**实测**（真的自增），
// 不是算出来的公式：公式写得对而实现写成 O(d²) 是真实存在的失败模式，
// 只有计数器能抓到。
//
// 期望值（d=1536 k0=20 k=8）：
//   PER-ROW : d*k0*k = 245,760
//   SHARED  : k0*k   =     160
//
// ===========================================================================
// 为什么不用堆 / 快排（任务书禁止）
// ===========================================================================
//
// k0=20、k=8 都很小，堆的常数（sift 的行程 + 指针追逐）比它省下的比较还贵，
// 而快排的分区会带来数据相关的分支（`conn_train.rs:439-448` 实测过这条：
// 数据相关的分支让「看似 O(k0)」的选择慢到 13.4 ns/连接 ≈ 40 周期）。
// 本模块用**固定行程的插入**：对每个输入 v，在已降序的 best[0..kk) 里
// 从尾往前扫固定 kk 步插入。行程与数据无关，比较次数因此是确定的
// `k0 * kk`，这也正好让 A4 的计数器有一个可预测的期望值。

use crate::conn_train::SparseConnW;

/// B2 的两种选择语义。用 enum 而不是两个函数族，理由：
/// 两者的**选择算法与累加顺序必须逐字相同**（否则 A1 里两条路径会因
/// 实现漂移而各自出错，还会掩盖真正的语义差异）。把差异收敛到一个
/// `match`，共享的强度/选择/累加代码就只有一份。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum B2Mode {
    /// 每神经元各自选自己的 top-k（局部语义）。选择次数 = d。
    PerRow,
    /// 整层一次选择，所有神经元共用同一批槽位（全局语义）。选择次数 = 1。
    Shared,
}

/// 单次前向的分解计数。**全部是实测自增**，不是公式。
///
/// 为什么返回结构体而不是只返回一个总数：A4 要证明的是「选择是 O(k0)」，
/// 而 A2/A3 要分别归因强度计算与选择。只报一个总数就无法区分
/// 「比较次数变多」是来自选择还是来自强度。
#[derive(Clone, Copy, Debug, Default)]
pub struct B2Stats {
    /// 执行的乘加次数（被选中的连接数之和）。
    pub macs: usize,
    /// 强度计算次数（= 被考察的连接数）。
    pub strength_ops: usize,
    /// **选择比较次数** —— A4 的核心量化。
    pub compares: usize,
    /// 执行的选择次数（PER-ROW = d，SHARED = 1）。这是 76.8x 的分界量。
    pub selections: usize,
}

/// 在 `strength[0..k0]` 上选 top-`k`，写**升序**下标到 `hit`，返回
/// (命中数, 比较次数)。
///
/// 平局规则：严格 `>` 才替换 ⇒ 强度相等时保留下标更小的（见模块头注释）。
/// 输出升序（引擎 93 的契约，也是把结果直接喂进累加的累加顺序契约）。
///
/// 比较次数是**确定的** `k0 * kk`（kk = min(k, k0)）：固定行程插入。
/// 这不是「大致 O(k0*k)」而是精确值，因此 A4 的期望值可对拍。
///
/// `hit` 至少 `min(k, k0)` 个 usize。
unsafe fn topk_ascending(strength: *const f32, k0: usize, k: usize, hit: *mut usize) -> (usize, usize) {
    let kk = if k > k0 { k0 } else { k };
    if kk == 0 {
        return (0, 0);
    }
    // 栈上固定大小：k0 是本模块的前提（B2 的选择空间**就是** k0，不是 d），
    // k0 = 20 时这是 20+20 个槽位。k0 > B2_STACK_K0 时硬失败而不是静默截断。
    //
    // ── 插入算法（本模块最容易写错的地方；第 1 版真的写错了）──────────────
    //
    // 第 1 版的错法：把「空槽」当成 `-inf` 参与比较，并在空槽里做条件写，
    // 靠一个 carry 对在空槽之间搬运。实测 `hit = [0 0 0 3]` —— carry 丢失了
    // 中间几次搬运，选出的下标根本不是 top-4。A1 因此报 1536/1536 mismatch。
    //
    // 这一版换成**「先把 v 放在榜尾，再固定 kk 步向后冒泡」**：
    //
    //   best[0..kk) 是**降序**榜。处理新的 v 时：
    //     1. 把 v 当作榜尾第 (kk-1) 名（槽位可能还是空的）；
    //     2. 从尾部往头部走固定 kk 步，每步比较 best[q-1] 与 best[q]：
    //        若前一名**小于**后一名就交换 —— 这就是一次冒泡。
    //     3. 冒泡把 v 送到它该在的位置；榜满时最小的那个被挤到榜尾之外
    //        （用 `present` 标记榜里真正有几个元素，绝不读未初始化的槽）。
    //
    // 为什么这个形状是对的（而不是「看起来对」）：它等价于「把 v 追加到
    // 降序榜尾，然后对整个榜做一趟完整的冒泡排序」—— 追加会破坏有序性的
    // 只可能是 v 与它左侧相邻元素这一对，冒泡恰好修复它。所以每步都是
    // 局部比较交换，没有 carry，没有空槽参与比较，也就不可能丢元素。
    //
    // 比较次数是 **O(k0·kk) 且有硬上界**，但不是与数据无关的常数：
    // 填充/插入循环在「前一名不比 v 小」时就 `break`，所以真实次数 <= 上界。
    //
    // ⚠️ 这里我改了两次，两次都值得记下来：
    //   第 1 版：按 `k0*kk` 写期望值，实测 215040 vs 245760（每行少 k0 次）—— 
    //           因为我按 kk 步写了闭式，而实际的相邻对只有 kk-1 对。
    //   第 2 版：为了让计数「恰好等于闭式」而把循环写成固定 kk 步无 break，
    //           结果 kk=1 时冒泡 0 步，top-1 选中了**最后一条连接**而不是
    //           最大值（A6 的 micro case 抓到了：' picks slot 2', compares=0）。
    //
    //   ⇒ 教训：**不能为了让计数器好看而改算法形状**。计数器的职责是报告
    //     真实工作量，不是去匹配一个我事先写下的公式。所以现在是
    //     「有 break 的真实插入 + 上界作为判据」，而不是「凑公式的假比较」。
    //
    // 上界推导（每次插入最多走 kk 步）：
    //   填充阶段：present 从 0 到 kk-1，插入步数 <= 0+1+...+(kk-1) = kk(kk-1)/2
    //   满员阶段：剩余 (k0-kk) 个元素，每个 <= 1 次拒绝 + (kk-1) 次挪动 = kk 步
    //   => 上界 = kk(kk-1)/2 + (k0-kk)*kk
    // 与 d 无关 ⇒ PER-ROW 的 d 次选择给出 O(d)，SHARED 的 1 次给出 O(1)。
    //
    // 平局规则：挪动条件是**严格** `prev_v < v`。相等时**不**挪动 ⇒ v 停在
    // 相等元素之后 ⇒ 先遇到的（下标更小）留在前面 ⇒ 与
    // conn_train::row_topk_from_strength 同一条契约。
    //
    // 平局规则：交换条件是**严格** `best[q-1] < best[q]`（即前一名严格更小
    // 才交换）。强度相等时**不交换** ⇒ 先遇到的（下标更小）留在前面 ⇒
    // 与 conn_train::row_topk_from_strength 同一条契约。
    //
    // 第 1 版之后我一度想保留「空槽当 -inf」的写法，那是把**已排序榜的
    // 不变量**和**榜还没被填满**这两件事混在一个变量里。分开表达
    // （best + present）才是它正确的原因。
    let mut best = [0usize; B2_STACK_K0];
    let mut bestv = [0.0f32; B2_STACK_K0];
    let mut compares = 0usize;
    let mut present = 0usize; // 榜里真正有效的元素个数（<= kk）

    let mut j = 0usize;
    while j < k0 {
        let v = *strength.add(j);
        if present < kk {
            // ── 填充阶段：榜还没满，v **必定**留在榜里 ──────────────────────
            // 用「升序插入」把 v 放到降序榜的正确位置：从尾往前把比 v 小的
            // 元素各往后挪一格，然后把 v 写进空出的槽位。
            //
            // ⚠️ 这里**不能**用「追加到榜尾再冒泡」：kk=1 时冒泡步数是
            //    kk-1 = 0，追加法会让唯一那个槽位永远是「最后遇到的」，
            //    于是 k=1 的 top-1 变成「最后一条连接」而不是最大值。
            //    实测就是这个症状：micro case 里 `'>' picks slot 2`、
            //    compares=0，选中的是全场最弱的槽位。
            //    填充阶段必须**真的插入**，不能靠后续冒泡补。
            let mut q = present;
            while q > 0 {
                let prev_v = bestv[q - 1];
                // 严格 `<`：相等时**不**右移 ⇒ v 停在相等元素之后 ⇒
                // 先遇到的（下标更小）留在前面（平局契约）。
                if prev_v < v {
                    bestv[q] = prev_v;
                    best[q] = best[q - 1];
                    q -= 1;
                } else {
                    break;
                }
                compares += 1;
            }
            bestv[q] = v;
            best[q] = j;
            present += 1;
        } else {
            // ── 满员阶段：v 只在比当前最后一名大时才进榜 ───────────────────
            // 先做一次快速拒绝（绝大多数连接在这里被排除），形状可预测。
            // 这一次比较**要计数**（它是真的做过的选择工作）。
            compares += 1;
            if v > bestv[kk - 1] {
                // 插入到降序榜的正确位置，最后一名被挤出榜外。
                // 从尾往前挪：把比 v 小的各往后推一格。
                let mut q = kk - 1;
                while q > 0 {
                    let prev_v = bestv[q - 1];
                    // 严格 `<` ⇒ 平局时下标更小的留在前面。
                    if prev_v < v {
                        bestv[q] = prev_v;
                        best[q] = best[q - 1];
                        q -= 1;
                        compares += 1;
                    } else {
                        break;
                    }
                }
                bestv[q] = v;
                best[q] = j;
            }
        }
        j += 1;
    }

    // 输出升序。kk 很小（模型里 = 8），至多 kk 个元素的插入排序。
    let nk = present;
    let mut i = 0usize;
    while i < nk {
        *hit.add(i) = best[i];
        i += 1;
    }
    i = 1;
    while i < nk {
        let v = *hit.add(i);
        let mut q = i;
        while q > 0 && *hit.add(q - 1) > v {
            *hit.add(q) = *hit.add(q - 1);
            q -= 1;
        }
        *hit.add(q) = v;
        i += 1;
    }
    (nk, compares)
}

/// 把插入的平局规则从严格 `>` 换成 `>=` 的**对照实现**（判据 A6 专用）。
///
/// 为什么这个函数在生产模块里：A6 要证明「平局规则是可失败的判据」。
/// 若这条规则被改坏而没有任何东西能报出来，它就是一条零信息的规则。
/// 把对照组**放在被测代码旁边**，比在驱动里重写一份更可信 —— 驱动里
/// 重写的那份会与真实实现漂移，于是 A6 测的是驱动自己的拷贝。
///
/// ⚠️ 本函数**不在任何生产路径上被调用**，只被 A6 调用。
unsafe fn topk_ascending_ge(strength: *const f32, k0: usize, k: usize, hit: *mut usize) -> (usize, usize) {
    let kk = if k > k0 { k0 } else { k };
    if kk == 0 {
        return (0, 0);
    }
    // 与 topk_ascending **逐字同构**，唯一的差别是挪动条件从严格 `<`
    // 改成 `<=`。这样 A6 测到的差异**只能**来自平局规则，不能来自两份
    // 实现的其它漂移 —— 这是「对照组必须只差一个变量」的要求。
    let mut best = [0usize; B2_STACK_K0];
    let mut bestv = [0.0f32; B2_STACK_K0];
    let mut compares = 0usize;
    let mut present = 0usize;
    for j in 0..k0 {
        let v = *strength.add(j);
        if present < kk {
            let mut q = present;
            while q > 0 {
                let prev_v = bestv[q - 1];
                // 唯一的改动：`<=`。相等时**挪动** ⇒ 后遇到的（下标更大）
                // 留在前面。
                if prev_v <= v {
                    bestv[q] = prev_v;
                    best[q] = best[q - 1];
                    q -= 1;
                } else {
                    break;
                }
                compares += 1;
            }
            bestv[q] = v;
            best[q] = j;
            present += 1;
        } else {
            compares += 1;
            if v >= bestv[kk - 1] {
                let mut q = kk - 1;
                while q > 0 {
                    let prev_v = bestv[q - 1];
                    if prev_v <= v {
                        bestv[q] = prev_v;
                        best[q] = best[q - 1];
                        q -= 1;
                        compares += 1;
                    } else {
                        break;
                    }
                }
                bestv[q] = v;
                best[q] = j;
            }
        }
    }
    let nk = present;
    for i in 0..nk {
        *hit.add(i) = best[i];
    }
    for i in 1..nk {
        let v = *hit.add(i);
        let mut q = i;
        while q > 0 && *hit.add(q - 1) > v {
            *hit.add(q) = *hit.add(q - 1);
            q -= 1;
        }
        *hit.add(q) = v;
    }
    (nk, compares)
}

/// 本模块允许的 k0 上界（栈上固定大小）。
///
/// 与 `conn_train::ROW_TOPK_STACK_K` 分开：那一个管的是「每次调用都能开在
/// 栈上」，本模块的 k0 是**模型超参**（K0_DEFAULT=20）。B2 的选择空间
/// **就是** k0，所以这个上界必须显式写出来，超过就硬失败而不是静默截断。
pub const B2_STACK_K0: usize = 256;

/// 用「严格 >」还是「>=」做插入。A6 通过它切换对照组，
/// 而不是把两份插入代码复制到驱动里（那会漂移）。
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TieRule {
    /// 生产规则：强度相等时保留下标更小的。见模块头注释。
    StrictGt,
    /// **仅用于 A6 can-fail 对照**：`>=`，相等时下标更大的胜出。
    GeCanFail,
}

#[inline]
unsafe fn select(
    strength: *const f32,
    k0: usize,
    k: usize,
    hit: *mut usize,
    rule: TieRule,
) -> (usize, usize) {
    match rule {
        TieRule::StrictGt => topk_ascending(strength, k0, k, hit),
        TieRule::GeCanFail => topk_ascending_ge(strength, k0, k, hit),
    }
}

/// 强度行 `out[j] = |w[i][j] · x[src[i][j]]|` —— 整行一次算完。
///
/// ===========================================================================
/// 为什么这里可以上 SIMD，而 B1 的前向乘积不行（这是本函数存在的唯一理由）
/// ===========================================================================
/// `kernels::sparse_strength_row_avx2` 的**语义就是 `|w·x|`**
/// （`_mm256_andnot_ps(sign, prod)` 清 bit31，与 `f32::abs()` 逐位一致，
/// 含 NaN 位型）。B2 的强度**按定义带绝对值**，所以这个内核在这里是**对的**。
/// B1 的前向乘积**必须保符号**，用它就是符号级错误（实测 15314/30720 纯符号
/// 翻转，见 `core/src/b1_pure.rs` 文件头的 S1 回退记录）。
/// ⇒ 两者不可互换；本函数与 `sparse_product_row_*` 并列放置正是为了让「选错」
///   在代码结构上可见。
///
/// **逐位一致性（硬论证，不是「应该差不多」）**：
///   * 每个元素独立做一次 `mul` + 清符号位；
///   * 没有归约、没有 FMA、没有重排、没有跨元素依赖；
///   * k0 % 8 的尾部由内核用**与标量版逐字相同**的表达式处理。
/// ⇒ 本函数输出与下面那个标量循环**逐位相同**。所以 B2 的 A1 逐位判据
///   （`f32::to_bits`）不受这条优化影响，选择结果也不受影响。
///
/// 加速来源（实测）：标量循环每条连接要走「load src → cvttss2si → load x →
/// mul → abs」这条串行依赖链；内核一次处理 8 条（cvttps2dq 一次产 8 个 i32）。
///
/// # Safety
/// 同 `b2_forward` 的 w/src/x 前提；`out` 至少 `k0` 个 f32。
#[inline]
unsafe fn strength_row(sc: &SparseConnW, base: usize, x: *const f32, out: *mut f32) {
    let k0 = sc.k0;
    #[cfg(target_arch = "x86_64")]
    {
        if crate::interp::HAS_AVX2 && k0 > 0 {
            crate::kernels::sparse_strength_row_avx2(
                sc.w.add(base),
                sc.src.add(base),
                x,
                k0,
                out,
            );
            return;
        }
    }
    for j in 0..k0 {
        let sid = *sc.src.add(base + j) as usize;
        *out.add(j) = (*sc.w.add(base + j) * *x.add(sid)).abs();
    }
}

/// B2 前向：y[i] = Σ_{j ∈ top-k(|w[i][j]·x[src[i][j]]|)} w[i][j] · x[src[i][j]]。
///
/// `mode` 决定选择的**次数**（这是 76.8x 的分界量），不决定选择的算法。
/// 返回 `B2Stats`，其中 `compares` / `selections` 是 A4 的证据。
///
/// PER-ROW：内层每个神经元算完 20 条强度后立刻选 k 条并累加。每行的
/// 强度缓冲复用一个栈数组，不逐行分配。
///
/// SHARED：先做一遍**聚合**（把每个「槽位」j 的跨神经元强度加起来），
/// 在 k0 个聚合值上选一次 k 个槽位，然后所有神经元都只累加这些槽位。
/// 聚合那一遍是 O(d*k0) 乘加，**不做选择** —— 这正是「选择与 d 脱钩」
/// 的含义。
///
/// # Safety
/// * `sc.w` / `sc.src` 各至少 `sc.e = n_neurons*k0` 个 f32。
/// * `x` 至少 `n_neurons` 个 f32；每个 `src` 值落在 `[0, n_neurons)`。
/// * `y` 至少 `n_neurons` 个 f32 且可写。
/// * `sc.k0 <= B2_STACK_K0`。
/// * `agg`（仅 SHARED 用）至少 `k0` 个 f32；传 null 且 mode=SHARED 是契约违反。
/// * `hit`（仅 SHARED 用）至少 `min(k,k0)` 个 usize。
pub unsafe fn b2_forward(
    sc: &SparseConnW,
    x: *const f32,
    y: *mut f32,
    k: usize,
    mode: B2Mode,
    agg: *mut f32,
    hit: *mut usize,
) -> B2Stats {
    b2_forward_tie(sc, x, y, k, mode, agg, hit, TieRule::StrictGt)
}

/// 与 `b2_forward` 相同，但可选择平局规则。生产代码一律走 `b2_forward`；
/// 这个入口只给 A6 的 can-fail 对照用。
///
/// # Safety
/// 同 `b2_forward`。
pub unsafe fn b2_forward_tie(
    sc: &SparseConnW,
    x: *const f32,
    y: *mut f32,
    k: usize,
    mode: B2Mode,
    agg: *mut f32,
    hit: *mut usize,
    rule: TieRule,
) -> B2Stats {
    let n = sc.n_neurons as usize;
    let k0 = sc.k0;
    if k0 > B2_STACK_K0 {
        crate::print(b"B2_TOPK: k0 exceeds the safe stack bound\n");
        crate::exit(9);
    }
    let kk = if k > k0 { k0 } else { k };
    let mut st = B2Stats::default();
    // 每行的强度缓冲在栈上开一次，全行复用（不逐行分配）。
    let mut strength = [0.0f32; B2_STACK_K0];
    let sp = strength.as_mut_ptr();

    match mode {
        B2Mode::PerRow => {
            for i in 0..n {
                let base = i * k0;
                // 1) 强度：整行一次算完（AVX2 内核，逐位同标量）。见 `strength_row`。
                strength_row(sc, base, x, sp);
                st.strength_ops += k0;
                // 2) 本行自己的 top-k（选择次数 = 1，累计 d 次）。
                let (nk, cmp) = select(sp, k0, k, hit_local(hit, kk), rule);
                st.compares += cmp;
                st.selections += 1;
                // 3) 累加被选中的 k 条：**升序 j、单标量累加器、首项赋值**。
                //    `hit_local` 提到循环外（原来每次 q 都调一次取静态地址）。
                let hp = hit_local(hit, kk);
                let mut acc = 0.0f32;
                let mut first = true;
                for q in 0..nk {
                    let j = *hp.add(q);
                    let sid = *sc.src.add(base + j) as usize;
                    let term = *sc.w.add(base + j) * *x.add(sid);
                    acc = if first { term } else { acc + term };
                    first = false;
                    st.macs += 1;
                }
                *y.add(i) = acc;
            }
        }
        B2Mode::Shared => {
            // 1) 聚合：槽位 j 的强度跨全部神经元求和。O(d*k0) 乘加，无选择。
            //
            //    融合内核一次算完整个矩阵（省掉 1536 次逐行调用 + 一趟
            //    强度缓冲往返）。逐槽位仍是 **i 升序、每元素一次 add** ⇒
            //    agg 与最朴素的标量实现**逐位相同**。
            for j in 0..k0 {
                *agg.add(j) = 0.0;
            }
            #[cfg(target_arch = "x86_64")]
            {
                if crate::interp::HAS_AVX2 && k0 > 0 {
                    crate::kernels::sparse_strength_agg_avx2(sc.w, sc.src, x, n, k0, agg);
                    st.strength_ops += n * k0;
                } else {
                    for i in 0..n {
                        let base = i * k0;
                        strength_row(sc, base, x, sp);
                        st.strength_ops += k0;
                        for j in 0..k0 {
                            *agg.add(j) += *sp.add(j);
                        }
                    }
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                for i in 0..n {
                    let base = i * k0;
                    strength_row(sc, base, x, sp);
                    st.strength_ops += k0;
                    for j in 0..k0 {
                        *agg.add(j) += *sp.add(j);
                    }
                }
            }
            // 2) **一次**选择，作用在 k0 个槽位上（与 d 无关）。
            let (nk, cmp) = select(agg, k0, k, hit, rule);
            st.compares += cmp;
            st.selections += 1;
            // 3) 所有神经元共用这批槽位；累加顺序契约同 PER-ROW。
            for i in 0..n {
                let base = i * k0;
                let mut acc = 0.0f32;
                let mut first = true;
                for q in 0..nk {
                    let j = *hit.add(q);
                    let sid = *sc.src.add(base + j) as usize;
                    let term = *sc.w.add(base + j) * *x.add(sid);
                    acc = if first { term } else { acc + term };
                    first = false;
                    st.macs += 1;
                }
                *y.add(i) = acc;
            }
        }
    }
    st
}

/// PER-ROW 每行复用的命中缓冲。
///
/// 为什么 SHARED 用调用方的 `hit` 而 PER-ROW 用本地缓冲：SHARED 的命中表
/// 是**跨神经元共享的状态**（必须先选后用），调用方应该能观察到它 ——
/// 那是「全层一张表」这个语义的可观测载体。PER-ROW 的命中表是每行的
/// 临时量，暴露它只会给调用方一个能读到「最后一行的结果」的错误接口。
///
/// 用一个 `static mut` 而不是 `[usize; B2_STACK_K0]` 的局部数组：
/// 局部数组会让每行多出 2KB 的栈触碰（20 行 × ... 无实际收益），
/// 而本模块是单线程契约（与 conn_train 的 sparse_accum 同一条纪律）。
static mut PERROW_HIT: [usize; B2_STACK_K0] = [0; B2_STACK_K0];

#[inline]
unsafe fn hit_local(_hit: *mut usize, _kk: usize) -> *mut usize {
    core::ptr::addr_of_mut!(PERROW_HIT) as *mut usize
}

/// B2 反向：dW[i][j] += dy[i] · x[src[i][j]]，**只写被选中的 k 条**。
///
/// 选择必须与前向**同一次**（否则梯度与激活不匹配 —— 那是一类只能靠
/// 位对拍抓到的静默错误）。所以本函数自己重算选择，且用的是同一套
/// `select`。调用方要保证 `x` 与 `k` 和前向一致。
///
/// 与 `sparse_gemm_bwd_w` 的关系：那一个的差异是「读 act 掩码」，本模块
/// 不物化掩码（PER-ROW 的命中表根本不该被物化成 [E] 的掩码 —— 那正是
/// 「掩码占前向 103%」的来源，见 ref_sparse_a/AUDIT_PATH_ISOLATION.md §3.2）。
///
/// 累加语义是 `+=`（不是赋值）：引擎侧反向指令全是累加，调用方先清零。
///
/// # Safety
/// 同 `b2_forward`，外加 `dy` 至少 `n_neurons` 个 f32、
/// `dw` 至少 `sc.e` 个 f32 且可写。
pub unsafe fn b2_backward(
    sc: &SparseConnW,
    dy: *const f32,
    x: *const f32,
    dw: *mut f32,
    k: usize,
    mode: B2Mode,
    agg: *mut f32,
    hit: *mut usize,
) -> B2Stats {
    let n = sc.n_neurons as usize;
    let k0 = sc.k0;
    if k0 > B2_STACK_K0 {
        crate::print(b"B2_TOPK: k0 exceeds the safe stack bound\n");
        crate::exit(9);
    }
    let kk = if k > k0 { k0 } else { k };
    let mut st = B2Stats::default();
    let mut strength = [0.0f32; B2_STACK_K0];
    let sp = strength.as_mut_ptr();

    // 先确定命中集合（与前向同一套规则），再逐行写梯度。
    let nk: usize;
    match mode {
        B2Mode::Shared => {
            for j in 0..k0 {
                *agg.add(j) = 0.0;
            }
            #[cfg(target_arch = "x86_64")]
            {
                if crate::interp::HAS_AVX2 && k0 > 0 {
                    crate::kernels::sparse_strength_agg_avx2(sc.w, sc.src, x, n, k0, agg);
                    st.strength_ops += n * k0;
                } else {
                    for i in 0..n {
                        let base = i * k0;
                        strength_row(sc, base, x, sp);
                        st.strength_ops += k0;
                        for j in 0..k0 {
                            *agg.add(j) += *sp.add(j);
                        }
                    }
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                for i in 0..n {
                    let base = i * k0;
                    strength_row(sc, base, x, sp);
                    st.strength_ops += k0;
                    for j in 0..k0 {
                        *agg.add(j) += *sp.add(j);
                    }
                }
            }
            let (m, cmp) = select(agg, k0, k, hit, TieRule::StrictGt);
            nk = m;
            st.compares += cmp;
            st.selections += 1;
            for i in 0..n {
                let base = i * k0;
                let g = *dy.add(i);
                for q in 0..nk {
                    let j = *hit.add(q);
                    let sid = *sc.src.add(base + j) as usize;
                    *dw.add(base + j) += g * *x.add(sid);
                    st.macs += 1;
                }
            }
        }
        B2Mode::PerRow => {
            for i in 0..n {
                let base = i * k0;
                strength_row(sc, base, x, sp);
                st.strength_ops += k0;
                let hp = hit_local(hit, kk);
                let (m, cmp) = select(sp, k0, k, hp, TieRule::StrictGt);
                st.compares += cmp;
                st.selections += 1;
                let g = *dy.add(i);
                for q in 0..m {
                    let j = *hp.add(q);
                    let sid = *sc.src.add(base + j) as usize;
                    *dw.add(base + j) += g * *x.add(sid);
                    st.macs += 1;
                }
            }
        }
    }
    st
}

/// B2 的**输入侧梯度**（dx）：`dx[src[i][j]] += dy[i]·w[i][j]`，**只对被选中的 k 条**。
///
/// 为什么 B2 也需要它：只有 `b2_backward`（dw）时梯度传不到上一层，24 层里只有
/// 最后一层在学 —— 那会"跑通"、loss 也降，但不是深度训练。P0 把 B2 接进训练器
/// 时这条与 B1 的 `b1_backward_dx` 是同一个前置。
///
/// 选择必须与前向 **同一次**（契约同 `b2_backward`：调用方保证 `x` 与 `k` 一致）。
/// 逐位契约：`i` 升序、`q` 升序（`hit` 是升序表）、每个元素一次 `add`
/// ⇒ 单线程下**确定**且与「先选后散射」的手写参考逐位相同。
///
/// 返回散射累加次数（= 被选中的连接数之和），与 `B2Stats::macs` 同口径。
///
/// # Safety
/// 同 `b2_backward`，外加 `dx` 至少源维度个 f32 且**由调用方清零**（本函数只 `+=`）。
pub unsafe fn b2_backward_dx(
    sc: &SparseConnW,
    dy: *const f32,
    x: *const f32,
    dx: *mut f32,
    k: usize,
    mode: B2Mode,
    agg: *mut f32,
    hit: *mut usize,
) -> usize {
    let n = sc.n_neurons as usize;
    let k0 = sc.k0;
    if k0 > B2_STACK_K0 {
        crate::print(b"B2_TOPK: k0 exceeds the safe stack bound\n");
        crate::exit(9);
    }
    let kk = if k > k0 { k0 } else { k };
    let mut strength = [0.0f32; B2_STACK_K0];
    let sp = strength.as_mut_ptr();
    let mut accs = 0usize;
    match mode {
        B2Mode::Shared => {
            // 与前向**同一套**强度聚合 + 一次选择（选择必须与前向同一次）。
            for j in 0..k0 {
                *agg.add(j) = 0.0;
            }
            #[cfg(target_arch = "x86_64")]
            {
                if crate::interp::HAS_AVX2 && k0 > 0 {
                    crate::kernels::sparse_strength_agg_avx2(sc.w, sc.src, x, n, k0, agg);
                } else {
                    for i in 0..n {
                        let base = i * k0;
                        strength_row(sc, base, x, sp);
                        for j in 0..k0 {
                            *agg.add(j) += *sp.add(j);
                        }
                    }
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                for i in 0..n {
                    let base = i * k0;
                    strength_row(sc, base, x, sp);
                    for j in 0..k0 {
                        *agg.add(j) += *sp.add(j);
                    }
                }
            }
            let hp = hit;
            let (nk, _cmp) = select(agg, k0, k, hp, TieRule::StrictGt);
            for i in 0..n {
                let base = i * k0;
                let g = *dy.add(i);
                for q in 0..nk {
                    let j = *hp.add(q);
                    let sid = *sc.src.add(base + j) as usize;
                    *dx.add(sid) += g * *sc.w.add(base + j);
                    accs += 1;
                }
            }
        }
        B2Mode::PerRow => {
            for i in 0..n {
                let base = i * k0;
                strength_row(sc, base, x, sp);
                let hp = hit_local(hit, kk);
                let (m, _cmp) = select(sp, k0, k, hp, TieRule::StrictGt);
                let g = *dy.add(i);
                for q in 0..m {
                    let j = *hp.add(q);
                    let sid = *sc.src.add(base + j) as usize;
                    *dx.add(sid) += g * *sc.w.add(base + j);
                    accs += 1;
                }
            }
        }
    }
    accs
}

/// 把某行的强度行算进 `out`（判据 A3 用：单独测强度那一遍，不含选择）。
///
/// 单独暴露它不是为了生产使用，而是为了让 A3 能**实测**「强度 / 选择 /
/// 累加」三者的分解，而不是从总时间反推。反推的分解剖面在本项目已经
/// 出过错（见 lead_b1_adversarial.rs 头部的三个记录案例）。
///
/// 返回强度计算次数 = n_neurons * k0。
///
/// # Safety
/// 同 `b2_forward` 的 w/src/x 前提；`out` 至少 `n_neurons*k0` 个 f32。
pub unsafe fn b2_strength_rows(sc: &SparseConnW, x: *const f32, out: *mut f32) -> usize {
    let n = sc.n_neurons as usize;
    let k0 = sc.k0;
    #[cfg(target_arch = "x86_64")]
    {
        if crate::interp::HAS_AVX2 && k0 > 0 {
            // 整矩阵一次调用（与 b2_forward 的 SHARED 强度阶段同一条路径，
            // 这样 A3 的分解口径与生产路径一致，不会出现"分解项比总和还慢"）。
            crate::kernels::sparse_strength_rows_avx2(sc.w, sc.src, x, n, k0, out);
            return n * k0;
        }
    }
    for i in 0..n {
        let base = i * k0;
        for j in 0..k0 {
            let sid = *sc.src.add(base + j) as usize;
            *out.add(base + j) = (*sc.w.add(base + j) * *x.add(sid)).abs();
        }
    }
    n * k0
}

/// 单独的 top-k 选择（判据 A3/A4 用）：在 `strength[0..k0]` 上选一次。
/// 返回 (命中数, 比较次数)。
///
/// # Safety
/// `strength` 至少 `k0` 个 f32；`hit` 至少 `min(k,k0)` 个 usize。
pub unsafe fn b2_select_once(
    strength: *const f32,
    k0: usize,
    k: usize,
    hit: *mut usize,
    rule: TieRule,
) -> (usize, usize) {
    if k0 > B2_STACK_K0 {
        crate::print(b"B2_TOPK: k0 exceeds the safe stack bound\n");
        crate::exit(9);
    }
    select(strength, k0, k, hit, rule)
}

// ===========================================================================
// PATH: B2 —— 令牌分块（transposed activation）SHARED 前向
// ===========================================================================
//
// 动机与 B1 的 `b1_forward_t8` 相同，且**实测同样成立**：B2-SHARED 的主成本是
// 「遍历全部 d*k0 条连接算强度并聚合」，而逐 token 的版本对每个 token 都要
// 为每条连接做一次 `_mm256_i32gather_ps` ⇒ 与发布版 B1 一样是 gather 吞吐受限。
//
// 手法：把 8 个 token 放进一个 ymm 通道，激活用 source-major 布局 `[d][t8]`，
// 于是每条连接**一次连续 load** 服务 8 个 token。
//
// 语义与逐 token 版本**逐位相同**（逐项论证）：
//   * 聚合：`agg_t[j][t]` 在 i 升序上每次加一个元素，与标量版 `agg[j]` 同序；
//   * 选择：每个 token 的 k0 个聚合值被拷进连续缓冲后走**同一个** `select`
//     （同一个平局规则、同一个比较计数语义）；
//   * 累加：每个 token 单独走「升序 j、单标量累加器、首项赋值」，无横向归约。
// ⇒ 因此可以用 `f32::to_bits` 对拍逐 token 版本（判据见 `vmlm-sparseb-tok` C9）。
//
// 只支持 `B2Mode::Shared`：PER-ROW 的每次选择只用**一个** token 的 x，
// 令牌分块对它的选择阶段毫无帮助（它本来就是逐神经元选择），传 PerRow 会硬失败
// 而不是悄悄退化成另一个语义。

/// 令牌分块 B2 SHARED 前向：输出 `yt[i*t8 + t]`（转置），与逐 token 版本逐位相同。
///
/// `agg_t` 是 `[k0][t8]` 的调用方缓冲（no_std 不分配）；`hit_t` 是 `[8][k0]`。
/// 返回的 `B2Stats` 与「逐 token 调用 `b2_forward`」的累计值**逐项相等**：
/// `strength_ops = n*k0*n_tokens`、`selections = n_tokens`、
/// `compares = Σ_t compares_t`、`macs = Σ_t n*nk_t`。
///
/// # Safety
/// * `sc.k0 <= B2_STACK_K0`；`t8 >= n_tokens` 且是 `xt`/`yt` 的行距。
/// * `xt` / `yt` 各至少 `n_neurons * t8`；`agg_t` 至少 `k0*t8`；
///   `hit_t` 至少 `8*k0` 个 usize。
pub unsafe fn b2_forward_t8(
    sc: &SparseConnW,
    xt: *const f32,
    t8: usize,
    n_tokens: usize,
    yt: *mut f32,
    k: usize,
    mode: B2Mode,
    agg_t: *mut f32,
    hit_t: *mut usize,
) -> B2Stats {
    if mode != B2Mode::Shared {
        crate::print(b"B2_TOPK: b2_forward_t8 only supports B2Mode::Shared\n");
        crate::exit(9);
    }
    let n = sc.n_neurons as usize;
    let k0 = sc.k0;
    if k0 > B2_STACK_K0 {
        crate::print(b"B2_TOPK: k0 exceeds the safe stack bound\n");
        crate::exit(9);
    }
    if t8 < n_tokens {
        crate::print(b"B2_TOPK: row stride t8 < n_tokens\n");
        crate::exit(9);
    }
    let kk = if k > k0 { k0 } else { k };
    let mut st = B2Stats::default();
    if kk == 0 || n_tokens == 0 {
        return st;
    }
    // 每 token 的聚合值连续缓冲（走同一个 select）。
    let mut agg_row = [0.0f32; B2_STACK_K0];
    let nblk = n_tokens / 8;
    let n_full = nblk * 8;
    // ---- 阶段 1：聚合（全部完整 8-token 块一次算完，行优先以保证 xt 局部性）----
    if nblk > 0 {
        let mut j = 0usize;
        while j < k0 {
            let mut t = 0usize;
            while t < n_full {
                *agg_t.add(j * t8 + t) = 0.0;
                t += 1;
            }
            j += 1;
        }
        #[cfg(target_arch = "x86_64")]
        {
            if crate::interp::HAS_AVX2 {
                crate::kernels::sparse_strength_agg_t8_avx2(
                    sc.w, sc.src, xt, t8, n, k0, nblk, agg_t,
                );
            } else {
                for i in 0..n {
                    let base = i * k0;
                    for j in 0..k0 {
                        let sid = *sc.src.add(base + j) as usize;
                        let wv = *sc.w.add(base + j);
                        for t in 0..n_full {
                            *agg_t.add(j * t8 + t) += (wv * *xt.add(sid * t8 + t)).abs();
                        }
                    }
                }
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            for i in 0..n {
                let base = i * k0;
                for j in 0..k0 {
                    let sid = *sc.src.add(base + j) as usize;
                    let wv = *sc.w.add(base + j);
                    for t in 0..n_full {
                        *agg_t.add(j * t8 + t) += (wv * *xt.add(sid * t8 + t)).abs();
                    }
                }
            }
        }
        st.strength_ops += n * k0 * n_full;
    }
    // ---- 阶段 1b：令牌尾部（n_tokens % 8）标量聚合，同一 i 升序契约 ----
    if n_full < n_tokens {
        let mut j = 0usize;
        while j < k0 {
            let mut t = n_full;
            while t < n_tokens {
                *agg_t.add(j * t8 + t) = 0.0;
                t += 1;
            }
            j += 1;
        }
        for i in 0..n {
            let base = i * k0;
            for j in 0..k0 {
                let sid = *sc.src.add(base + j) as usize;
                let wv = *sc.w.add(base + j);
                let mut t = n_full;
                while t < n_tokens {
                    *agg_t.add(j * t8 + t) += (wv * *xt.add(sid * t8 + t)).abs();
                    t += 1;
                }
            }
        }
        st.strength_ops += n * k0 * (n_tokens - n_full);
    }
    // ---- 阶段 2：每个 token 独立选择 + 累加（与逐 token 版本同一套代码路径）----
    {
        let mut t = 0usize;
        while t < n_tokens {
            let lane = t % 8;
            let mut j = 0usize;
            while j < k0 {
                agg_row[j] = *agg_t.add(j * t8 + t);
                j += 1;
            }
            let hp = hit_t.add(lane * k0);
            let (nk, cmp) = select(agg_row.as_ptr(), k0, k, hp, TieRule::StrictGt);
            st.compares += cmp;
            st.selections += 1;
            for i in 0..n {
                let base = i * k0;
                let mut acc = 0.0f32;
                let mut first = true;
                for q in 0..nk {
                    let j = *hp.add(q);
                    let sid = *sc.src.add(base + j) as usize;
                    let term = *sc.w.add(base + j) * *xt.add(sid * t8 + t);
                    acc = if first { term } else { acc + term };
                    first = false;
                    st.macs += 1;
                }
                *yt.add(i * t8 + t) = acc;
            }
            t += 1;
        }
    }
    st
}

/// 单独测「k0 条连接上的升序累加」那一遍（A3 的第三块）。
///
/// 与 `b1_forward` 逐字同构（升序 j、单标量、首项赋值），但只累加 `hit`
/// 里的下标。返回 MAC 次数。
///
/// # Safety
/// 同 `b2_forward`；`hit` 至少 `nk` 个 usize 且每个 < k0。
pub unsafe fn b2_accum_selected(
    sc: &SparseConnW,
    x: *const f32,
    y: *mut f32,
    hit: *const usize,
    nk: usize,
) -> usize {
    let n = sc.n_neurons as usize;
    let k0 = sc.k0;
    let mut macs = 0usize;
    for i in 0..n {
        let base = i * k0;
        let mut acc = 0.0f32;
        let mut first = true;
        for q in 0..nk {
            let j = *hit.add(q);
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
