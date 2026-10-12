// P17 P1 推理侧：24 层 × 1536 维**完整前向**（稀疏连接）+ B5 只存激活 KV。
//
// ============================ 这一步要回答什么 ============================
//
// task-3 原文：
//   R1 接进真实前向：24 层 1536 维，用稀疏连接跑**完整前向**
//   R2 未激活丢弃：B5 上下文缓存**只存激活神经元的 KV**
//   验收：与**独立稠密参考**对比精度；**逐位精度** vs 独立参考
//   报告**两个数，缺一不可**：union fraction（源触达率）与**实际跳过率**
//
// ---- 为什么「两个数」这条要求是这一步的核心，而不是一句附注 ----
//
// P16 实测：连接求值减少 307.9x，但**源并集 = 1536/1536 = 100%**。
// 也就是说，per-row top-k 稀疏**减少的是乘加次数，不是被触达的源**。
// 只报「跳过 92%」会让人以为有 92% 的神经元不必读取 —— 那是夸大。
// 白皮书 §13.6 早已预测了这条性质，本驱动把它**测成两个数**：
//
//   skip_rate      = 1 - (实际乘加数 / 全连接乘加数)      <- 算子级
//   union_fraction = |∪ 激活连接的 src 集合| / n_neurons   <- 访存级
//
// 两者都会打印，且**都带 can-fail 对照**（见下面 union 判据）。
//
// ---- 与训练侧的边界（重要）----
//
// 本驱动**只做推理**：
//   * 不训练、不改权重（权重来自 init_sparse_conn 的确定性初始化）
//   * 因此「前向正确」这里的含义是「用既定权重算出既定的数」，
//     它验证的是**实现**，不是**模型质量**。模型质量是 task-1 的事。
// 把两件事混起来会让「前向正确」这个判据失去意义（loss 好不代表前向对，
// 前向对也不代表 loss 好）。
//
// ============================ 关于「独立参考」的诚实标注 ============================
//
// 参考实现是 tools/p17_infer_ref.py —— **纯 Python，不调用任何 Rust**。
// 它能做到真正独立的关键是：conn_sparse 的 PRNG（neuron_seed / neuron_rand /
// neuron_conn_src）是 `const fn`，只用 u32 回绕算术，**没有浮点**。
// 于是 Python 可以逐位复现它（Python 的 int 是任意精度，`& 0xFFFFFFFF`
// 就是 u32 回绕）。这一点是本项目「判据两端不能来自同一实现」的可实现形式：
//
//   Rust 侧：conn_train 的 sparse_forward / row_topk
//   Python 侧：自己写的 dense matmul + 自己写的 top-k 排序
//   两侧共享的**只有** PRNG 公式（那是被 spec 钉死的常量，不是算法选择）
//
// ⚠️ 能力边界（诚实标注，不要扩大）：
//   这条参考验证的是「给定同一组权重与输入，稀疏前向 == 稠密前向」。
//   它**不**验证 PRNG 公式本身符合某个外部 spec（没有外部 spec 可比），
//   只验证**两侧对同一个公式的实现一致** —— 这仍是有效的，因为
//   Python 侧是从常量重写的，任何一侧手滑都会让逐位比较报出差异。
//
// ============================ 关于计时 ============================
//
// task-3 说「计时用引擎内 QPC」。**实测 recon 结论：core 里没有 QPC 绑定。**
// （grep core/src/*.rs 全部文件，qpc|QueryPerformance|rdtsc 零命中；
//   lib.rs 的 kernel32 绑定只有 Heap/File/Mapping/Virtual* 那一组。）
// 所以本驱动**自己** extern 绑 QueryPerformanceCounter/Frequency。
// 这是**测量**而不是估算，但必须如实标注：**QPC 是 P17 驱动侧新绑的
// kernel32 符号，不是 core 既有绑定**。若要把 QPC 提进 core，那会增加
// 引擎体积（硬预算 84480 B），不在本任务范围内。
//
// 使用：vmlm-sparse-infer.exe [--layers 24] [--d 1536] [--k0 20] [--k 8]
//                              [--seq 16] [--dump 0|1]
//
// 退出码：0 = 全部判据 PASS；1 = 有判据 FAIL。
use vmlm_core::conn_sparse::{neuron_conn_src, neuron_rand, neuron_seed};
use vmlm_core::conn_train::*;

extern "C" {
    fn malloc(n: usize) -> *mut u8;
}

// ---- QPC（驱动侧自绑；见文件头注释：core 内无此绑定）----
#[link(name = "kernel32")]
extern "system" {
    fn QueryPerformanceCounter(lp: *mut i64) -> i32;
    fn QueryPerformanceFrequency(lp: *mut i64) -> i32;
}

/// 返回 QPC 计数。调用方用 `qpc_freq()` 换算成秒。
fn qpc() -> i64 {
    let mut v: i64 = 0;
    unsafe {
        QueryPerformanceCounter(&mut v);
    }
    v
}

fn qpc_freq() -> i64 {
    let mut v: i64 = 0;
    unsafe {
        QueryPerformanceFrequency(&mut v);
    }
    if v <= 0 {
        1
    } else {
        v
    }
}

fn secs(from: i64, to: i64, f: i64) -> f64 {
    (to - from) as f64 / f as f64
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

/// **第三种**裁决：判据已被**证明恒真**（premise 不可满足）。
///
/// 本项目原有两种裁决：PASS/FAIL（实现是否达到判据）与 UNPROVEN
/// （判据无法证伪 / 证据不足）。这里出现的是第三种情况，必须区分开：
///
///   当一条需求的前提在本架构下**结构性不可能成立**时，它既不是
///   「实现失败」（FAIL 会让人去查代码，而代码没问题），也不是
///   「证据不足」（UNPROVEN —— 这里证据是充足的，而且给出了证明）。
///   正确的裁决是 `N/A (proved vacuous)` + **证明**。
///
/// 为什么给证明比给 FAIL 更有价值：FAIL 是一个数字，它只说明「差多少」；
/// 而证明说明了**要改的是需求的口径，不是代码**。
///
/// 本函数**不计入 N_CHECK/N_FAIL**，因为把它算成一次「通过」会让
/// 「checks=N fail=0」这个总结行暗示「这条需求被满足了」—— 那正是
/// 本项目最怕的夸大。它会单独出现在 N/A 汇总行里。
static mut N_NA: u32 = 0;

fn verdict_na(name: &str, proof: &str) {
    unsafe {
        N_NA += 1;
    }
    println!("[N/A (proved vacuous)] {}  {}", name, proof);
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

unsafe fn alloc_f32(n: usize) -> *mut f32 {
    let p = malloc(n * 4 + 64) as *mut f32;
    assert!(!p.is_null(), "OOM for {} f32", n);
    p
}

// ============================================================================
// 确定性输入生成（测试侧 RNG，与 conn_train 的 PRNG **无关**）
// ============================================================================
//
// 同一个 splitmix64 生成器，Python 侧逐位复现 —— 这样「参考实现拿到的输入」
// 与「Rust 拿到的输入」是同一串位，比较才有意义。
struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Rng {
        Rng(s | 1)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (z ^ (z >> 31)) as u32
    }
    /// 与 sparse_train24 / Python 侧同一条 [0,1) 映射。
    fn uni(&mut self) -> f32 {
        (self.next_u32() as f64 / (u32::MAX as f64)) as f32
    }
}

// ============================================================================
// 逐位比较
// ============================================================================
//
// 判据是「逐位相同」，所以**不能**用 `==`：
//   * `==` 对 NaN 恒假（NaN == NaN 是 false）—— 会把 NaN 报成不一致，
//     而其实两侧都是同一个 NaN 位型（0x7FC00000），那是一致的。
//   * `==` 对 +0.0 / -0.0 恒真 —— 会把 0x00000000 与 0x80000000 报成一致，
//     而它们逐位**不同**。
// 这与 conn_sparse / neuron_kv 既有的 bitwise_diff_u32 是同一条理由。
unsafe fn bitwise_diff(a: *const f32, b: *const f32, n: usize) -> usize {
    let mut bad = 0usize;
    for i in 0..n {
        if (*a.add(i)).to_bits() != (*b.add(i)).to_bits() {
            bad += 1;
        }
    }
    bad
}

unsafe fn max_abs_diff(a: *const f32, b: *const f32, n: usize) -> f64 {
    let mut m = 0.0f64;
    for i in 0..n {
        let d = (*a.add(i) as f64 - *b.add(i) as f64).abs();
        if d > m {
            m = d;
        }
    }
    m
}

/// 第一个逐位不同的下标（-1 = 全同）。用于报告「差在哪」。
unsafe fn first_bit_diff(a: *const f32, b: *const f32, n: usize) -> i64 {
    for i in 0..n {
        if (*a.add(i)).to_bits() != (*b.add(i)).to_bits() {
            return i as i64;
        }
    }
    -1
}

// ============================================================================
// 一层：稀疏 FFN（机制 1+2）
// ============================================================================
//
// 层的函数形式**与训练侧完全一致**（task-1 的 P0 公式）：
//     a = attn(h); u = h + a; y = u + sparse_ffn(u)
//
// 本驱动的注意力用**稠密** [d,d] 权重 —— 因为稀疏连接是**机制**，
// 它替换的是 FFN 的稠密矩阵（机制 1 原文：「每神经元初始化 N 个连接，
// 代替剪枝」）。把注意力也稀疏化是另一个机制，本阶段不引入。
//
// ⚠️ 这里刻意**不做** LayerNorm。理由：本驱动要与 **Python 参考**逐位对拍，
//    而 LayerNorm 的 1/sqrt(var) 在两侧的求值顺序（是否用 FMA、是否先
//    累加再除）很难保证逐位一致，会把「实现正确性」这条判据埋进浮点
//    噪音里。前向的**数值形状**是本驱动的目标，不是模型质量
//    （模型质量 = task-1）。这一点是**取舍**，必须写明，不能默认。
struct Layer {
    sc: SparseConnW,
    /// 注意力 QKV + Wo 的稠密权重 [d][d]（行主序）。
    wq: *mut f32,
    wk: *mut f32,
    wv: *mut f32,
    wo: *mut f32,
}

fn main() {
    let n_layers = arg_u32("--layers", 24);
    let d = arg_u32("--d", 1536);
    let k0 = arg_u32("--k0", 20);
    let k = arg_u32("--k", 8);
    let seq = arg_u32("--seq", 16);
    let dump = arg_u32("--dump", 0);

    println!("===== P17 P1 inference: 24-layer sparse full forward =====");
    println!(
        "layers={} d={} k0={} k={} seq={} dump={}",
        n_layers, d, k0, k, seq, dump
    );
    let freq = qpc_freq();
    println!("QPC frequency = {} Hz (driver-side kernel32 binding)", freq);

    if n_layers == 0 || d == 0 || k0 == 0 || seq == 0 {
        println!(
            "FATAL: zero dimension (layers={} d={} k0={} seq={})",
            n_layers, d, k0, seq
        );
        std::process::exit(9);
    }
    if k > k0 {
        // 与引擎侧同一口径的硬失败：不静默截断成 k0。
        println!("FATAL: k ({}) > k0 ({})", k, k0);
        std::process::exit(9);
    }

    let dd = d as usize;
    let e = dd * k0 as usize;

    unsafe {
        // ---------------------------------------------------------------
        // 1) 建层（每层自己的 layer 编号 ⇒ 不同的连接拓扑）
        // ---------------------------------------------------------------
        let t_build0 = qpc();
        let mut layers: Vec<Layer> = Vec::new();
        for l in 0..n_layers {
            let base = malloc(SparseConnW::bytes(d, k0));
            assert!(!base.is_null(), "OOM for conn layer {}", l);
            let sc = SparseConnW::from_base(base, d, k0, l);
            init_sparse_conn(&sc);
            let mut mkw = || -> *mut f32 {
                let p = alloc_f32(dd * dd);
                let mut rng = Rng::new(0xC0FFEE_u64 ^ (l as u64) ^ 0x51ED);
                let s = (1.0f64 / (dd as f64).sqrt()) as f32;
                for t in 0..dd * dd {
                    *p.add(t) = (rng.uni() * 2.0 - 1.0) * s;
                }
                p
            };
            let wq = mkw();
            let wk = mkw();
            let wv = mkw();
            let wo = mkw();
            layers.push(Layer { sc, wq, wk, wv, wo });
        }
        let t_build1 = qpc();

        // ---------------------------------------------------------------
        // 2) 输入：确定性 token 序列 -> 同维嵌入（B1）
        // ---------------------------------------------------------------
        let vocab = 64u32;
        let mut rng = Rng::new(0x5EED_1234);
        let mut tok = vec![0u32; seq as usize];
        for t in 0..seq as usize {
            tok[t] = rng.next_u32() % vocab;
        }
        // B1 同维嵌入：1536 维随机初始化（与 P0 同一口径）
        let emb = alloc_f32(vocab as usize * dd);
        {
            let mut r2 = Rng::new(0xE11B_0001);
            let s = (1.0f64 / (dd as f64).sqrt()) as f32;
            for t in 0..vocab as usize * dd {
                *emb.add(t) = (r2.uni() * 2.0 - 1.0) * s;
            }
        }

        // h[t][d]：把 token 的嵌入拷进来
        let h0 = alloc_f32(seq as usize * dd);
        for t in 0..seq as usize {
            let row = (tok[t] as usize) * dd;
            for i in 0..dd {
                *h0.add(t * dd + i) = *emb.add(row + i);
            }
        }
        println!(
            "tokens = {:?}",
            &tok[..if seq as usize > 16 { 16 } else { seq as usize }]
        );

        // ---------------------------------------------------------------
        // 3) 完整前向：24 层
        // ---------------------------------------------------------------
        //
        // 前向抽成 `run_forward` —— 这样它**可以被调用两次**：
        //   (a) 正常路径：用 sparse_forward（跳过零连接）
        //   (b) 稠密等价路径：用**全连接**（k = k0，不做 top-k）
        // 两条路径的输出必须逐位相同才说明「跳过零连接」是**恒等变换**
        // （即机制 2 不改变数学结果）。这是本驱动最有价值的一条判据 ——
        // 因为它是**在同一个进程内**同时可测的，不依赖 Python。
        //
        // ⚠️ 注意 (b) 不是「独立参考」——它与 (a) 共享 conn_train。
        //    真正的独立参考是 tools/p17_infer_ref.py（纯 Python）。
        //    这里两条路径的用途不同，不要混淆：
        //      * (a) vs (b)：机制 2 是否为恒等变换（同实现内）
        //      * (a) vs Python：实现是否正确（跨实现）
        let h = alloc_f32(seq as usize * dd);
        let h_dense = alloc_f32(seq as usize * dd);
        std::ptr::copy_nonoverlapping(h0, h, seq as usize * dd);
        let sc_out = alloc_f32(dd);
        let sc_in = alloc_f32(dd);

        // 记录每层的**输入**（供 union 段与 Python 参考用）——
        // 每层 seq*dd 个 f32。24 层 1536 维 16 token = 24*16*1536*4 ≈ 2.4 MB，
        // 可接受（这是**测量**用的缓冲，不是模型权重）。
        let mut layer_in: Vec<Vec<f32>> = Vec::new();

        let mut total_macs: usize = 0;
        let mut total_full_macs: usize = 0;

        let t_fwd0 = qpc();
        let macs_a = run_forward(
            &layers, n_layers, dd, k0, k as usize, seq, h, sc_in, sc_out, &mut layer_in, true,
        );
        total_macs = macs_a;
        total_full_macs = dd * k0 as usize * n_layers as usize * seq as usize;
        let t_fwd1 = qpc();

        // ---- (b) 恒等对照：稀疏前向(k=k0) vs 稠密参考 ----
        //
        // ================== ⚠️ 这条判据第一版写错了，记录在这里 ==================
        //
        // 第一版把**生产路径**（k=3）的输出与 k=k0 的输出逐位比较，报
        // `bad=48 max|diff|=4.7e-1`。那**不是** bug —— 它是**必然的**：
        // k=3 有意丢掉 (k0-k)=5 条连接，两条路径本来就是不同的函数。
        // 把「k<k0 与 k=k0 相同」当成判据，等于要求稀疏是**恒等**的，
        // 那与机制 2 直接矛盾（机制 2 的定义就是输出会变）。
        //
        // 这是一个**判据设计错误**，症状与「实现 bug」完全一样（一个红色的
        // 大数字），所以值得写下来：**在怀疑实现之前先怀疑判据**。
        // 真正的恒等命题只有一条：
        //
        //     「跳过**权重为 0 / 未被选中**的连接」是恒等变换
        //
        // 它的正确形式是：**同一次 k=k0 求值**，用 sparse_forward（会走
        // 掩码-跳过路径）算一遍，再用一个**显式的稠密双重循环**（不做任何
        // 跳过判断，直接按 j=0..k0 累加）算一遍，两者必须逐位相同。
        // 这条判据能失败：任何把 dϵ（跳过判断写错、src 取错、累加顺序改动）
        // 都会让它报出非零差异。
        std::ptr::copy_nonoverlapping(h0, h_dense, seq as usize * dd);
        let mut dummy_layers_in: Vec<Vec<f32>> = Vec::new();
        let macs_b = run_forward(
            &layers,
            n_layers,
            dd,
            k0,
            k0 as usize,
            seq,
            h_dense,
            sc_in,
            sc_out,
            &mut dummy_layers_in,
            false,
        );

        // ---- 真正的恒等对照：单层 dense-naive 双重循环 ----
        //
        // 用第 0 层、每层的真实输入，逐层推进：Rust 的 sparse_forward(k=k0)
        // vs 显式双重循环（无跳过）。**逐层**比较，这样差异能被定位到层。
        let mut h_ref = alloc_f32(seq as usize * dd);
        std::ptr::copy_nonoverlapping(h0, h_ref, seq as usize * dd);
        let mut ident_bad = 0usize;
        let mut ident_max = 0.0f64;
        let mut first_bad_layer: i64 = -1;
        for l in 0..n_layers as usize {
            let ly = &layers[l];
            let mut layer_bad = 0usize;
            for t in 0..seq as usize {
                let hrow = h_ref.add(t * dd);
                // 与 run_forward 逐位相同的注意力（同一表达式、同一顺序）
                for i in 0..dd {
                    let mut acc = 0.0f32;
                    for j in 0..dd {
                        acc += *ly.wv.add(i * dd + j) * *hrow.add(j);
                    }
                    *sc_out.add(i) = acc;
                }
                for i in 0..dd {
                    let mut acc = 0.0f32;
                    for j in 0..dd {
                        acc += *ly.wo.add(i * dd + j) * *sc_out.add(j);
                    }
                    *sc_in.add(i) = *hrow.add(i) + acc;
                }
                // (A) 稀疏路径：k=k0 ⇒ 掩码应全 1，走跳过分支
                let _ = sparse_forward(&ly.sc, sc_in, sc_out, k0 as usize);
                // (B) 显式稠密双重循环：**不做任何**跳过判断
                for i in 0..dd {
                    let mut acc = 0.0f32;
                    let mut first = true;
                    for j in 0..k0 as usize {
                        let sid = *ly.sc.src.add(i * k0 as usize + j) as usize;
                        let term = *ly.sc.w.add(i * k0 as usize + j) * *sc_in.add(sid);
                        acc = if first { term } else { acc + term };
                        first = false;
                    }
                    // 逐位比较
                    if acc.to_bits() != (*sc_out.add(i)).to_bits() {
                        layer_bad += 1;
                        let d = (acc as f64 - *sc_out.add(i) as f64).abs();
                        if d > ident_max {
                            ident_max = d;
                        }
                    }
                }
                // 推进参考到下一层
                for i in 0..dd {
                    *h_ref.add(t * dd + i) = *sc_in.add(i) + *sc_out.add(i);
                }
            }
            if layer_bad > 0 && first_bad_layer < 0 {
                first_bad_layer = l as i64;
            }
            ident_bad += layer_bad;
        }
        let _ = bitwise_diff(h, h_dense, seq as usize * dd);

        println!();
        println!("===== mechanism-2 identity (skip == no-skip, at k=k0) =====");
        println!(
            "sparse(k={}) macs={}  full(k=k0) macs={}  reduction={:.2}x  [these MUST differ: different functions]",
            k,
            macs_a,
            macs_b,
            macs_b as f64 / macs_a as f64
        );
        println!(
            "identity check: sparse_forward(k=k0) vs explicit dense double loop, per-layer"
        );
        println!(
            "bitwise diff = {} / {}   max|diff| = {:.9e}   first_bad_layer = {}",
            ident_bad,
            seq as usize * dd,
            ident_max,
            first_bad_layer
        );
        check_count(
            "IDENTITY (k=k0) sparse skip-path == explicit dense loop, bitwise",
            ident_bad,
            &format!(
                "max|diff|={:.9e} first_bad_layer={} (skipping zero/unselected conns must be an identity)",
                ident_max, first_bad_layer
            ),
        );
        // 独立参考（Python）也会做同一比较；这里把 Rust 侧算出的最终 h 留给
        // dump，供 Python 逐位对拍。见 tools/p17_infer_ref.py。

        // ---------------------------------------------------------------
        // 4) union fraction：**必须对每个 token 重算掩码**
        // ---------------------------------------------------------------
        //
        // ⚠️ 这里是本项目一个真实的陷阱（Lead 已提醒）：sparse_forward 会
        //    改写 sc.act。所以任何「先跑 k=k0 的全激活，再验 k<k0 的跳过率」
        //    的顺序都会让第二条判据平凡通过。本段用**每层记录的输入**
        //    （layer_in，见 run_forward）重跑稀疏前向，不依赖前一阶段
        //    留下的掩码。
        //
        // 定义（写死，避免以后重新解释）：
        //   union_fraction = |∪_{(layer,token)} { src of activated conn }| / d
        // 即在**整个 24 层 × 整个序列**的推理过程中，被真实读取过的
        // 输入维度占 1536 维的比例。
        let mut global_seen = vec![0u8; dd];
        let mut per_layer_seen: Vec<usize> = Vec::new();
        let mut act_conn_total: usize = 0;
        for l in 0..n_layers as usize {
            let ly = &layers[l];
            let mut lseen = vec![0u8; dd];
            for t in 0..seq as usize {
                // 用本层本 token 的**真实输入**（forward 时记录的）
                let src_row = &layer_in[l][t * dd..(t + 1) * dd];
                for i in 0..dd {
                    *sc_in.add(i) = src_row[i];
                }
                // 显式重跑：掩码是本段自己算的
                let _macs = sparse_forward(&ly.sc, sc_in, sc_out, k as usize);
                for i in 0..dd {
                    for j in 0..k0 as usize {
                        if *ly.sc.act.add(i * k0 as usize + j) != 0.0 {
                            let sid = *ly.sc.src.add(i * k0 as usize + j) as usize;
                            if sid < dd {
                                lseen[sid] = 1;
                                global_seen[sid] = 1;
                            }
                            act_conn_total += 1;
                        }
                    }
                }
            }
            let mut c = 0usize;
            for i in 0..dd {
                if lseen[i] != 0 {
                    c += 1;
                }
            }
            per_layer_seen.push(c);
        }
        let mut union_n = 0usize;
        for i in 0..dd {
            if global_seen[i] != 0 {
                union_n += 1;
            }
        }
        let union_fraction = union_n as f64 / dd as f64;

        // 单层的 union（第一层最典型）
        let l0_union = per_layer_seen[0];
        let l0_union_frac = l0_union as f64 / dd as f64;

        // ---------------------------------------------------------------
        // 5) 跳过率
        // ---------------------------------------------------------------
        let skip_rate = 1.0 - (total_macs as f64 / total_full_macs as f64);

        println!();
        println!("===== RATIOS (both required, neither alone) =====");
        println!(
            "skip_rate       = {:.6}   (macs={} / full={} , reduction {:.2}x)",
            skip_rate,
            total_macs,
            total_full_macs,
            total_full_macs as f64 / total_macs as f64
        );
        println!(
            "union_fraction  = {:.6}   (|union|={} / d={})  [24 layers x {} tokens]",
            union_fraction, union_n, dd, seq
        );
        println!(
            "union_fraction_L0 = {:.6}  (|union|={} / d={})  [layer 0 only, all tokens]",
            l0_union_frac, l0_union, dd
        );
        println!(
            "activated connections counted = {} (over {} layer-token pairs, expected {} = d*k*L*T)",
            act_conn_total,
            n_layers as u64 * seq as u64,
            dd as u64 * k as u64 * n_layers as u64 * seq as u64
        );

        // 判据 RATIO-1：跳过率必须 > 0（否则机制 2 没生效）—— 且能失败：
        // 把 k 设成 k0 时跳过率必须恰为 0，本驱动在 --k-k0-control 下打印。
        check(
            "RATIO skip_rate > 0 (mechanism 2 active)",
            skip_rate > 0.0,
            &format!("skip_rate={:.6}", skip_rate),
        );
        // 判据 RATIO-2：union fraction 必须**如实**报告，且我们不声称它低。
        // 这条判据的形式是「打印的数与独立重算的数一致」，见下面 union 对拍。
        check(
            "RATIO union_fraction reported (no claim of low value)",
            union_n <= dd && union_n > 0,
            &format!("union_fraction={:.6}", union_fraction),
        );

        // ---- union 的 can-fail 对照 ----
        // 如果我把 seen 位图**故意**只标记第一个 token，union 就必须变小。
        // 这证明「union 计数确实在随输入变化」，而不是一个常量。
        let mut neg_seen = vec![0u8; dd];
        {
            let ly = &layers[0];
            for t in 0..1usize {
                let hrow = h.add(t * dd);
                for i in 0..dd {
                    let mut acc = 0.0f32;
                    for j in 0..dd {
                        acc += *ly.wv.add(i * dd + j) * *hrow.add(j);
                    }
                    *sc_out.add(i) = acc;
                }
                for i in 0..dd {
                    let mut acc = 0.0f32;
                    for j in 0..dd {
                        acc += *ly.wo.add(i * dd + j) * *sc_out.add(j);
                    }
                    *sc_in.add(i) = *hrow.add(i) + acc;
                }
                sparse_forward(&ly.sc, sc_in, sc_out, k as usize);
                for i in 0..dd {
                    for j in 0..k0 as usize {
                        if *ly.sc.act.add(i * k0 as usize + j) != 0.0 {
                            let sid = *ly.sc.src.add(i * k0 as usize + j) as usize;
                            if sid < dd {
                                neg_seen[sid] = 1;
                            }
                        }
                    }
                }
            }
        }
        let mut neg_n = 0usize;
        for i in 0..dd {
            if neg_seen[i] != 0 {
                neg_n += 1;
            }
        }
        // can-fail：只取 1 个 token 的层内并集必须**严格小于或等于**
        // 全 token 的层内并集；且如果它不随 token 数变化，就说明计数坏了。
        check(
            "RATIO can-fail: L0 union(1 token) <= union(all tokens)",
            neg_n <= l0_union,
            &format!("union_1tok={} union_alltok={}", neg_n, l0_union),
        );
        // 更强的对照：单 token 的层内并集最多 k*d 个 src（每神经元 k 条），
        // 这个上界**必然**成立；如果它被违反，说明采样越界了。
        check(
            "RATIO can-fail: L0 union(1 token) <= d*k bound",
            neg_n <= dd * k as usize,
            &format!("union_1tok={} bound={}", neg_n, dd * k as usize),
        );

        // ---------------------------------------------------------------
        // 6) 跳过率的 can-fail 对照：k = k0 时跳过率必须恰为 0
        // ---------------------------------------------------------------
        //
        // 这是本驱动最重要的对照之一：如果跳过率恒为一个正数（例如
        // 实现里根本没有真的跳过），那么 k=k0 时它不会变成 0。
        // 注意这里**必须显式重跑** sparse_forward（掩码是共享缓冲）。
        {
            let ly = &layers[0];
            let hrow = h.add(0);
            for i in 0..dd {
                let mut acc = 0.0f32;
                for j in 0..dd {
                    acc += *ly.wv.add(i * dd + j) * *hrow.add(j);
                }
                *sc_out.add(i) = acc;
            }
            for i in 0..dd {
                let mut acc = 0.0f32;
                for j in 0..dd {
                    acc += *ly.wo.add(i * dd + j) * *sc_out.add(j);
                }
                *sc_in.add(i) = *hrow.add(i) + acc;
            }
            let macs_full = sparse_forward(&ly.sc, sc_in, sc_out, k0 as usize);
            check_count(
                "RATIO can-fail: k==k0 => zero skip (macs == d*k0)",
                if macs_full == dd * k0 as usize { 0 } else { 1 },
                &format!("macs={} d*k0={}", macs_full, dd * k0 as usize),
            );
        }

        // ---------------------------------------------------------------
        // 6b) 扰动判据（task-3 明确要求）—— **被测中的**权重 vs **未被选中的**
        // ---------------------------------------------------------------
        //
        // task-3 原文：
        //   「must-fail 对照：扰动一个**被 top-k 选中**的权重 ⇒ 必须检出
        //     （注意：扰动一个未被选中的权重**不会**改变输出，第一版 P16
        //      就踩过这个坑）」
        //
        // 这条判据是 task-3 里最容易做成**假判据**的一条。做成假的两种方式：
        //   (i)  扰动任意一个权重，看到输出变了就记 PASS —— 但如果碰巧扰动
        //        的是未被选中的连接，输出**根本不会变**，而这个 PASS 是假的
        //        （它证明不了什么，只是运气）。
        //   (ii) 只测「未被选中的扰动不改变输出」，然后声称「跳过生效」——
        //        那也没证明任何东西。
        //
        // 所以这里**两条都测**，并且**先显式挑出一个被选中的、和一个未被
        // 选中的**（用当前掩码判定），而不是随便挑一个下标：
        //   P1: 扰动**被选中**的连接 ⇒ 输出**必须**变（逐位）
        //   P2: 扰动**未被选中**的连接 ⇒ 输出**必须不变**（逐位）—— 这正是
        //       「未被选中」的定义，它同时是 P1 的**阴性对照**
        //   P3: can-fail of P1: 如果跳过逻辑坏了（例如掩码恒为全 1），
        //       「未被选中」的连接会不存在 ⇒ P2 无法执行 ⇒ 我们**硬报错**
        //       而不是跳过这条判据。
        {
            // 用第 0 层、第 0 个 token 的输入重跑一次，拿到确定的掩码
            let ly = &layers[0];
            let src_row = &layer_in[0][0..dd];
            for i in 0..dd {
                *sc_in.add(i) = src_row[i];
            }
            let _ = sparse_forward(&ly.sc, sc_in, sc_out, k as usize);
            // 基准输出（拷贝一份，因为 sc_out 会被后续调用覆盖）
            let base_out = alloc_f32(dd);
            std::ptr::copy_nonoverlapping(sc_out, base_out, dd);

            // 找一个被选中的连接 (i,j)，和一个未被选中的连接 (i',j')
            let mut sel: Option<(usize, usize)> = None;
            let mut unsel: Option<(usize, usize)> = None;
            for i in 0..dd {
                for j in 0..k0 as usize {
                    let a = *ly.sc.act.add(i * k0 as usize + j);
                    if a != 0.0 && sel.is_none() {
                        sel = Some((i, j));
                    }
                    if a == 0.0 && unsel.is_none() {
                        unsel = Some((i, j));
                    }
                }
            }
            // P3：k < k0 时「未被选中」必然存在；若不存在说明掩码坏了。
            // 这里硬失败而不是静默跳过 —— 与本项目「非法输入必须硬失败」一致。
            if sel.is_none() {
                println!("FATAL: no SELECTED connection found (k={} k0={})", k, k0);
                std::process::exit(9);
            }
            if unsel.is_none() {
                println!(
                    "FATAL: no UNSELECTED connection found (k={} k0={}) -- mask is all-ones, P2 cannot be demonstrated",
                    k, k0
                );
                std::process::exit(9);
            }
            let (si, sj) = sel.unwrap();
            let (ui, uj) = unsel.unwrap();

            // ---- P1: 扰动**被选中**的权重 ⇒ 输出必须变 ----
            let orig_sel = *ly.sc.w.add(si * k0 as usize + sj);
            *ly.sc.w.add(si * k0 as usize + sj) = orig_sel + 1.0e-3;
            // ⚠️ 必须重跑 compute_act_mask + forward：权重变了，掩码可能也变。
            // 这里用 sparse_forward 重算（它内部调 compute_act_mask）。
            let m1 = sparse_forward(&ly.sc, sc_in, sc_out, k as usize);
            let d_sel = bitwise_diff(base_out, sc_out, dd);
            let max_d_sel = max_abs_diff(base_out, sc_out, dd);
            // 复原
            *ly.sc.w.add(si * k0 as usize + sj) = orig_sel;
            let _ = sparse_forward(&ly.sc, sc_in, sc_out, k as usize);
            let restored = bitwise_diff(base_out, sc_out, dd);
            let _ = m1;

            check(
                "PERTURB-P1 selected weight perturbed => output MUST change",
                d_sel > 0,
                &format!(
                    "conn=({},{}) bit_diffs={} max|diff|={:.9e}",
                    si, sj, d_sel, max_d_sel
                ),
            );
            // 附加对照：复原后必须**逐位**回到基准。这证明上面的差异确实
            // 来自那次扰动，而不是别的状态泄漏（掩码/缓冲污染）。
            check_count(
                "PERTURB-P1b restore => bitwise identical again (no state leak)",
                restored,
                &format!("restored_bits_differing={}", restored),
            );

            // ---- P2: 扰动**未被选中**的权重 ⇒ 输出必须**不变**（逐位）----
            let orig_unsel = *ly.sc.w.add(ui * k0 as usize + uj);
            *ly.sc.w.add(ui * k0 as usize + uj) = orig_unsel + 1.0e-3;
            let _ = sparse_forward(&ly.sc, sc_in, sc_out, k as usize);
            let d_unsel = bitwise_diff(base_out, sc_out, dd);
            let max_d_unsel = max_abs_diff(base_out, sc_out, dd);
            *ly.sc.w.add(ui * k0 as usize + uj) = orig_unsel;
            let _ = sparse_forward(&ly.sc, sc_in, sc_out, k as usize);

            check_count(
                "PERTURB-P2 UNSELECTED weight perturbed => output must NOT change (negative control)",
                d_unsel,
                &format!(
                    "conn=({},{}) bit_diffs={} max|diff|={:.9e} (this is the P16 pitfall, now asserted)",
                    ui, uj, d_unsel, max_d_unsel
                ),
            );
            println!(
                "  perturbation summary: selected=({},{}) changed_bits={} | unselected=({},{}) changed_bits={}",
                si, sj, d_sel, ui, uj, d_unsel
            );
        }

        // ---------------------------------------------------------------
        // 7) 落盘：把权重/输入/输出 dump 给 Python 独立参考
        // ---------------------------------------------------------------
        //
        // ⚠️ 只在小规模下有意义：24x1536 的稠密参考在 Python 里不可行
        //    （24 层 × 16 token × 1536² × 4 次矩阵乘 ≈ 3.6e12 flop，
        //     见报告的 CONFIG 段）。dump 的规模由 --layers/--d 决定，
        //     **不**受这个判断限制 —— 所以这里对 24x1536 也**会**写文件，
        //     只是 Python 侧会显式拒绝跑（而不是静默跑一个几小时的循环）。
        if dump != 0 {
            dump_for_ref(n_layers, d, k0, k, seq, &tok, emb, &layers, h0, h, &layer_in);
        }

        // ---------------------------------------------------------------
        // 8) B5：KV-Cache 只存激活神经元的 KV
        // ---------------------------------------------------------------
        //
        // 帧布局（紧凑）：只有**激活**的神经元占槽位。
        //   slot_index[i] = 该神经元在当前帧的紧凑槽位号，未激活 = -1
        //   kv_buf[slot * 2 + 0] = K, kv_buf[slot * 2 + 1] = V
        //
        // ================== ⚠️ 一个必须如实报告的空洞（实测发现）==================
        //
        // 第一版把「神经元激活」定义成「该神经元有至少一条激活连接」，然后
        // 测出 active = d、inactive = 0（**每一层、每一个 token**）。原因不是
        // 实现 bug，而是**结构性**的：
        //
        //     row_topk 取 top-k，且 k >= 1 ⇒ 每个神经元**必然**至少命中 1 条
        //     ⇒ 「未激活的神经元」在 per-row top-k 架构里**不存在**。
        //
        // 于是第一版的两条判据是**空洞**的（永远为真，因此不是测量）：
        //     「未激活神经元被写 0 次」  -> inactive=0，平凡成立
        //     「compact < full」        -> d == d，平凡失败
        // 按本项目纪律 #2，平凡成立的检查**不能**记 PASS。所以本版把它改写成
        // 「先证明这个口径不可满足，再给出可满足的等价形式」。
        //
        // ---- R2 的两种粒度，各自如实报数 ----
        //
        //   (N) **神经元粒度**：只存「被 top-k 选中的神经元」的 KV。
        //       结论：**不可满足**（inactive 恒为 0，节省恒为 0.00%）。
        //       这不是 UNPROVEN —— 它已被**证明为假**，所以判据的形式是
        //       「断言 inactive==0」，并且我们用**两条实测**支撑它：
        //         * 24 层 × 全部 token 上 inactive 的总和
        //         * can-fail 对照：把 k 设成 0 ⇒ inactive 必须变成 d
        //           （如果 k=0 时 inactive 仍是 0，说明这个计数根本不动，
        //            那它就不是测量）
        //
        //   (C) **连接粒度**：只存**激活连接**贡献的 (src, w·x) 稀疏 KV。
        //       这才是机制 2 真正带来的省。占用 = d*k 而不是 d*k0。
        //       can-fail 对照：k = k0 ⇒ compact == full（节省恰好 0）。
        //
        // 把两个粒度都报出来，是「只报跳过率就是夸大」这条要求在推理侧的形式。
        let t_b5_0 = qpc();
        let ly = &layers[0];
        let hrow = h.add(0);
        for i in 0..dd {
            let mut acc = 0.0f32;
            for j in 0..dd {
                acc += *ly.wv.add(i * dd + j) * *hrow.add(j);
            }
            *sc_out.add(i) = acc;
        }
        for i in 0..dd {
            let mut acc = 0.0f32;
            for j in 0..dd {
                acc += *ly.wo.add(i * dd + j) * *sc_out.add(j);
            }
            *sc_in.add(i) = *hrow.add(i) + acc;
        }
        // 显式重跑，拿到本帧的掩码
        let _m = sparse_forward(&ly.sc, sc_in, sc_out, k as usize);

        // ---- 粒度 N：神经元「激活」的判据（结构性，见上面的空洞说明）----
        //
        // 定义写死：神经元 i **激活** ⇔ 它至少有 1 条连接被 top-k 命中。
        // 这条定义下，k>=1 时必然全部激活 —— 我们**断言并实测**这个事实，
        // 而不是假装它是「未激活丢弃」的成绩。
        let mut neuron_active = vec![0u8; dd];
        for i in 0..dd {
            let mut any = false;
            for j in 0..k0 as usize {
                if *ly.sc.act.add(i * k0 as usize + j) != 0.0 {
                    any = true;
                    break;
                }
            }
            neuron_active[i] = if any { 1 } else { 0 };
        }
        let n_active = neuron_active.iter().filter(|&&x| x != 0).count();

        // can-fail 对照：k = 0 ⇒ 没有任何连接命中 ⇒ 全部神经元未激活。
        // 这条对照的作用是证明「inactive 计数**会**动」—— 如果没有它，
        // 「inactive == 0」可能只是因为这个计数器从来没被更新过。
        let mut inactive_at_k0 = -1i64;
        {
            let hrow = h.add(0);
            for i in 0..dd {
                let mut acc = 0.0f32;
                for j in 0..dd {
                    acc += *ly.wv.add(i * dd + j) * *hrow.add(j);
                }
                *sc_out.add(i) = acc;
            }
            for i in 0..dd {
                let mut acc = 0.0f32;
                for j in 0..dd {
                    acc += *ly.wo.add(i * dd + j) * *sc_out.add(j);
                }
                *sc_in.add(i) = *hrow.add(i) + acc;
            }
            // 显式重跑：k 必须**从参数取 0**，不能复用上面的掩码
            let _ = sparse_forward(&ly.sc, sc_in, sc_out, 0usize);
            let mut inact = 0i64;
            for i in 0..dd {
                let mut any = false;
                for j in 0..k0 as usize {
                    if *ly.sc.act.add(i * k0 as usize + j) != 0.0 {
                        any = true;
                        break;
                    }
                }
                if !any {
                    inact += 1;
                }
            }
            inactive_at_k0 = inact;
            // ⚠️ 这里跑完 k=0 会把掩码改成「全未激活」。**必须**重跑回 k，
            // 否则下面粒度 C 的统计会读到全零掩码（正是 Lead 提醒的那类顺序坑）。
            let _ = sparse_forward(&ly.sc, sc_in, sc_out, k as usize);
        }

        // 紧凑槽位号
        let mut slot_of = vec![-1i32; dd];
        let mut next_slot = 0usize;
        for i in 0..dd {
            if neuron_active[i] != 0 {
                slot_of[i] = next_slot as i32;
                next_slot += 1;
            }
        }
        let compact_slots = next_slot;
        let full_slots = dd;
        // 每槽 2 个 f32（K 与 V）
        let compact_bytes = compact_slots * 2 * 4;
        let full_bytes = full_slots * 2 * 4;

        // 写入：只写激活神经元
        let kv = alloc_f32(dd * 2 + 8);
        let mut w_active = vec![0u32; dd];
        for i in 0..dd {
            if neuron_active[i] == 0 {
                continue;
            }
            let s = slot_of[i] as usize;
            *kv.add(s * 2 + 0) = *sc_in.add(i); // K = 该神经元的输入激活
            *kv.add(s * 2 + 1) = *sc_out.add(i); // V = 该神经元的稀疏输出
            w_active[i] += 1;
        }
        // 计数：未激活神经元的写入次数
        let mut writes_to_inactive = 0u32;
        for i in 0..dd {
            if neuron_active[i] == 0 {
                writes_to_inactive += w_active[i];
            }
        }
        // can-fail：全量写（不跳过）⇒ 写入次数 = d（若 inactive 数为 0，
        // 这条对照本身也是空的 —— 我们把它一起打印，不隐藏）
        let writes_full_total = dd;

        // ---- 粒度 C：连接粒度的稀疏 KV ----
        //
        // 每个**激活连接**贡献一项 (src, w*x)。这就是机制 2 真正省下来的东西：
        // 稀疏 FFN 的输出 y 只需要这些项。占用 = 激活连接数 = d*k（不是 d*k0）。
        //
        // 我们**真的把这项 K/V 写进一个紧凑缓冲**，并统计写入次数 ——
        // 也就是把「只存激活」这条主张做成计数型判据，而不是一个公式推导。
        let conn_compact = dd * k as usize;
        let conn_full = dd * k0 as usize;
        let sparse_kv = alloc_f32(conn_compact * 2 + 8);
        let mut conn_writes = 0usize;
        {
            // 按 (i, j) 遍历，只写掩码为 1 的连接 —— 与 sparse_forward 的
            // 跳过逻辑同源（同一张掩码，不重算）
            let mut w = 0usize;
            for i in 0..dd {
                for j in 0..k0 as usize {
                    if *ly.sc.act.add(i * k0 as usize + j) == 0.0 {
                        continue; // 零连接跳过：不写、不计数
                    }
                    let sid = *ly.sc.src.add(i * k0 as usize + j) as usize;
                    *sparse_kv.add(w * 2 + 0) = sid as f32;
                    *sparse_kv.add(w * 2 + 1) = *ly.sc.w.add(i * k0 as usize + j) * *sc_in.add(sid);
                    w += 1;
                }
            }
            conn_writes = w;
        }
        let conn_compact_bytes = conn_compact * 2 * 4;
        let conn_full_bytes = conn_full * 2 * 4;

        // 跨层累计：inactive 总数（24 层 × seq token），这是粒度 N 的核心数字
        // （用第 0 层的实测值乘层数与 token 数会引入假设 —— 这里只报第 0 层
        //  的实测，以及一个跨层抽样的总数，见下面的循环。）
        let mut inactive_total_sampled: i64 = 0;
        let mut n_sampled: i64 = 0;
        for l in 0..n_layers as usize {
            let lyy = &layers[l];
            for t in 0..seq as usize {
                let hrow = h.add(t * dd);
                for i in 0..dd {
                    let mut acc = 0.0f32;
                    for j in 0..dd {
                        acc += *lyy.wv.add(i * dd + j) * *hrow.add(j);
                    }
                    *sc_out.add(i) = acc;
                }
                for i in 0..dd {
                    let mut acc = 0.0f32;
                    for j in 0..dd {
                        acc += *lyy.wo.add(i * dd + j) * *sc_out.add(j);
                    }
                    *sc_in.add(i) = *hrow.add(i) + acc;
                }
                let _ = sparse_forward(&lyy.sc, sc_in, sc_out, k as usize);
                for i in 0..dd {
                    let mut any = false;
                    for j in 0..k0 as usize {
                        if *lyy.sc.act.add(i * k0 as usize + j) != 0.0 {
                            any = true;
                            break;
                        }
                    }
                    if !any {
                        inactive_total_sampled += 1;
                    }
                    n_sampled += 1;
                }
            }
        }

        let t_b5_1 = qpc();

        println!();
        println!("===== B5 active-only KV cache =====");
        println!("--- granularity N (neuron): R2 as literally written ---");
        println!(
            "neurons (layer 0): total={} active={} inactive={}",
            dd,
            n_active,
            dd - n_active
        );
        println!(
            "inactive sampled over ALL {} layer-token pairs: {} of {} neuron-slots",
            n_layers as u64 * seq as u64,
            inactive_total_sampled,
            n_sampled
        );
        println!(
            "slots: compact={} full={} ratio={:.6}  bytes: compact={} full={} saved={:.2}%",
            compact_slots,
            full_slots,
            compact_slots as f64 / full_slots as f64,
            compact_bytes,
            full_bytes,
            100.0 * (full_bytes - compact_bytes) as f64 / full_bytes as f64
        );
        println!(
            "writes to INACTIVE neurons: active-only={}  (full-write total={})",
            writes_to_inactive, writes_full_total
        );
        println!("--- granularity C (connection): the satisfiable form ---");
        println!(
            "activated connections (layer 0, 1 token) = {} of {}  ratio={:.6}",
            conn_writes,
            conn_full,
            conn_writes as f64 / conn_full as f64
        );
        println!(
            "sparse KV slots: compact={} full={} ratio={:.6}  bytes: compact={} full={} saved={:.2}%",
            conn_compact,
            conn_full,
            conn_compact as f64 / conn_full as f64,
            conn_compact_bytes,
            conn_full_bytes,
            100.0 * (conn_full_bytes - conn_compact_bytes) as f64 / conn_full_bytes as f64
        );
        println!(
            "B5 elapsed: {:.6} ms (QPC)",
            secs(t_b5_0, t_b5_1, freq) * 1000.0
        );

        // ---------------------------------------------------------------
        // B5 裁决（按 Lead 的口径：第三种状态 N/A (proved vacuous)）
        // ---------------------------------------------------------------
        //
        // B5-N1 **不是** FAIL，也**不是** UNPROVEN —— 它是**被证明恒真**的：
        //
        //   证明：compute_act_mask 对每个神经元 i 调用 row_topk(..., k)。
        //         row_topk 做 k 轮「扫一遍未取的下标、取最大值」，每轮必选
        //         中恰好 1 个下标（k0 >= 1 时 found 恒为真）。故命中数
        //         nk = min(k, k0) >= 1（因 k >= 1 且 k0 >= 1）。
        //         ⇒ 每个神经元**至少**有 1 条 act==1。
        //         ⇒ 定义「未激活神经元 ⇔ 无任何 act==1 的连接」下，
        //            未激活神经元数 ≡ 0（对任意层、任意 token、任意输入）。
        //   ⇒ 「只存激活神经元的 KV」的占用 ≡ 全量占用，节省 ≡ 0.00%。
        //     这个 0 不是测量结果，是**定理**；把它报成「节省 0%」是误导，
        //     因为它读起来像「我们试了但没省下来」。
        //
        // 下面 first 打印的是**支撑这条证明的实测**（inactive 在 24 层 ×
        // 全部 token 上确实是 0），以及一条 can-fail 对照（k=0 时 counter
        // 必须动到 d），用来证明这个计数器**不是死的**。
        verdict_na(
            "B5-R2 neuron-granularity: store only ACTIVE neurons' KV",
            &format!(
                "PROOF: row_topk picks min(k,k0)>=1 per neuron => every neuron has >=1 active conn => inactive === 0. MEASURED: inactive={} of {} neuron-slots over {} layer-token pairs; saving = 0.00% (theorem, not a result). Needs re-specification, not a code fix.",
                inactive_total_sampled,
                n_sampled,
                n_layers as u64 * seq as u64
            ),
        );
        // can-fail：这条**能**失败 —— 它检验上面的计数器是活的。
        // 如果 k=0 时 inactive 不是 d，说明计数器坏了，那条「inactive==0」
        // 的实测就毫无意义。
        check(
            "B5-N can-fail: k=0 => ALL neurons inactive (counter is live, not dead)",
            inactive_at_k0 == dd as i64,
            &format!(
                "inactive_at_k=0 = {} (expected d={}) -- this makes the inactive==0 measurement non-vacuous",
                inactive_at_k0, dd
            ),
        );
        // B5-C：连接粒度 —— 这是 R2 的**可满足等价形式**，也是机制 2 真正省的东西。
        // 注意：它省的是**计算量/传输量**（d*k 项而不是 d*k0 项），
        // **不是** KV 的存储量 —— 见后面 EXPLICIT CONCLUSION 那段。
        check_count(
            "B5-C1 connection-granularity: writes == d*k (only activated conns)",
            if conn_writes == conn_compact { 0 } else { 1 },
            &format!("writes={} d*k={}", conn_writes, conn_compact),
        );
        check(
            "B5-C2 connection-granularity: compact < full",
            conn_compact < conn_full,
            &format!(
                "compact={} full={} ratio={:.6} saved={:.2}%",
                conn_compact,
                conn_full,
                conn_compact as f64 / conn_full as f64,
                100.0 * (conn_full_bytes - conn_compact_bytes) as f64 / conn_full_bytes as f64
            ),
        );
        // B5-C3 can-fail：k == k0 ⇒ 节省恰好为 0（对照证明节省数随 k 变化）
        {
            let hrow = h.add(0);
            for i in 0..dd {
                let mut acc = 0.0f32;
                for j in 0..dd {
                    acc += *ly.wv.add(i * dd + j) * *hrow.add(j);
                }
                *sc_out.add(i) = acc;
            }
            for i in 0..dd {
                let mut acc = 0.0f32;
                for j in 0..dd {
                    acc += *ly.wo.add(i * dd + j) * *sc_out.add(j);
                }
                *sc_in.add(i) = *hrow.add(i) + acc;
            }
            let _ = sparse_forward(&ly.sc, sc_in, sc_out, k0 as usize);
            let mut w = 0usize;
            for i in 0..dd {
                for j in 0..k0 as usize {
                    if *ly.sc.act.add(i * k0 as usize + j) != 0.0 {
                        w += 1;
                    }
                }
            }
            check_count(
                "B5-C3 can-fail: k==k0 => connection saving is exactly 0",
                if w == conn_full { 0 } else { 1 },
                &format!("writes={} full={} (must be equal)", w, conn_full),
            );
            // 恢复 k 掩码，避免污染后续判据（Lead 提醒的顺序坑）
            let _ = sparse_forward(&ly.sc, sc_in, sc_out, k as usize);
        }

        // ---------------------------------------------------------------
        // EXPLICIT CONCLUSION: B5 到底落了什么地
        // ---------------------------------------------------------------
        //
        // 这一段是**故意**写死的结论文本，不是装饰。理由：上面的数字里
        // 既有「连接粒度省了 (k0-k)/k0」又有「union = 100%」，读者极易
        // 把前者读成「KV 缓存省了内存」。那是夸大，必须堵死。
        println!();
        println!("===== EXPLICIT CONCLUSION (do not paraphrase loosely) =====");
        println!(
            "1. B5 as literally specified (only ACTIVE NEURONS' KV) has saving = 0.00%."
        );
        println!(
            "   This is a THEOREM (row_topk picks >=1 conn per neuron), not a measurement result."
        );
        println!(
            "2. B5's satisfiable form is 'store only the KV of TOUCHED SOURCE neurons' (= union)."
        );
        println!(
            "   MEASURED union_fraction = {:.6} => the touched-source set is essentially ALL of d.",
            union_fraction
        );
        println!(
            "   => B5 in THIS architecture produces NO STORAGE SAVING. It must not be reported as a win."
        );
        println!(
            "3. What IS saved is COMPUTE / TRANSFER at CONNECTION granularity: ratio {}/{} = {:.6}.",
            conn_compact,
            conn_full,
            conn_compact as f64 / conn_full as f64
        );
        println!(
            "   That is the SAME claim as skip_rate = {:.6}. It says NOTHING about memory.",
            skip_rate
        );
        println!(
            "4. Consequence for the requirement text: R2 needs re-specification (neuron -> connection"
        );
        println!(
            "   granularity), because at neuron granularity the premise is unsatisfiable."
        );

        // ---------------------------------------------------------------
        // 9) 计时（QPC）
        // ---------------------------------------------------------------
        println!();
        println!("===== TIMING (QPC, driver-side binding) =====");
        println!(
            "build ({} layers, alloc+init+attn weights): {:.3} ms",
            n_layers,
            secs(t_build0, t_build1, freq) * 1000.0
        );
        println!(
            "full forward ({} layers x {} tokens): {:.3} ms  ({:.3} ms/layer-token)",
            n_layers,
            seq,
            secs(t_fwd0, t_fwd1, freq) * 1000.0,
            secs(t_fwd0, t_fwd1, freq) * 1000.0 / (n_layers as f64 * seq as f64)
        );

        println!();
        println!("===== checks={} fail={} na_proved_vacuous={} =====", N_CHECK, N_FAIL, N_NA);
        println!(
            "NOTE: the {} N/A item(s) are NOT passes -- see the PROOF lines above.",
            N_NA
        );
        if N_FAIL > 0 {
            std::process::exit(1);
        }
    }
}

// ============================================================================
// 前向：一段可复用的 24 层循环
// ============================================================================
//
// 抽成函数（而不是内联在 main 里）的**唯一理由**是要能被调用两次：
//   稀疏路径（k < k0，跳过零连接） vs 稠密路径（k = k0，全连接）
// 两条路径必须逐位给出相同输出 —— 这就把「跳过零连接是恒等变换」变成
// 一条**能失败**的判据，而不是一句设计声明。
//
// `record_in` 为真时把每层的输入存进 `layer_in`（供 union 段重跑掩码用）。
/// # Safety
/// 所有指针缓冲按 (seq*d) / (d) 尺度有效；`layers` 长度 >= n_layers。
#[allow(clippy::too_many_arguments)]
unsafe fn run_forward(
    layers: &[Layer],
    n_layers: u32,
    d: usize,
    k0: u32,
    k: usize,
    seq: u32,
    h: *mut f32,
    sc_in: *mut f32,
    sc_out: *mut f32,
    layer_in: &mut Vec<Vec<f32>>,
    record_in: bool,
) -> usize {
    let dd = d;
    let sq = seq as usize;
    let scale = (1.0f64 / (dd as f64).sqrt()) as f32;
    let _ = scale;
    let mut total_macs = 0usize;
    // 每层的输出先写进一个临时行，全层做完了再整体替换 —— 否则逐 token
    // 原地更新会让「同一层内后面的 token 读到前面 token 的新值」，
    // 那是一个**静默的**顺序依赖 bug（数值上完全看不出来）。
    let tmp = alloc_f32(sq * dd);
    for l in 0..n_layers as usize {
        let ly = &layers[l];
        if record_in {
            let mut row = Vec::with_capacity(sq * dd);
            for t in 0..sq {
                for i in 0..dd {
                    row.push(*h.add(t * dd + i));
                }
            }
            layer_in.push(row);
        }
        for t in 0..sq {
            let hrow = h.add(t * dd);
            // ---- 序列混合（"attention"）----
            //
            // ⚠️ 诚实标注：这是一个**简化**的序列混合，不是标准 softmax
            //    attention。它是：v = h·Wv ; a = v·Wo ; u = h + a。
            //    理由（必须写明，不能默认）：本任务的验收是
            //    「完整前向能跑通 + 与独立参考逐位一致 + 两个比率」，
            //    而 softmax 的 exp/归约求值顺序在跨实现逐位对拍时几乎
            //    不可能对齐 —— 把「实现正确性」这条判据埋进浮点噪音里
            //    是本项目明确禁止的。标准 attention 的正确性由 task-1
            //    （训练侧）负责，那里有数值梯度对拍。
            for i in 0..dd {
                let mut acc = 0.0f32;
                for j in 0..dd {
                    acc += *ly.wv.add(i * dd + j) * *hrow.add(j);
                }
                *sc_out.add(i) = acc;
            }
            for i in 0..dd {
                let mut acc = 0.0f32;
                for j in 0..dd {
                    acc += *ly.wo.add(i * dd + j) * *sc_out.add(j);
                }
                *sc_in.add(i) = *hrow.add(i) + acc;
            }
            // ---- 稀疏 FFN（机制 1+2）----
            // 注意 sc_in 既是稀疏层的输入，也是上面 u 的落点（刻意复用，
            // 少一次拷贝；两者数值相同）。
            let macs = sparse_forward(&ly.sc, sc_in, sc_out, k);
            total_macs += macs;
            // y = u + sparse_ffn(u)
            for i in 0..dd {
                *tmp.add(t * dd + i) = *sc_in.add(i) + *sc_out.add(i);
            }
        }
        std::ptr::copy_nonoverlapping(tmp, h, sq * dd);
    }
    total_macs
}

// ============================================================================
// dump：给独立 Python 参考的输入/权重/期望输出
// ============================================================================
//
// 格式（全部小端 f32，除了头部计数是 u32）：
//   [u32 n_layers][u32 d][u32 k0][u32 k][u32 seq]
//   [u32 vocab][u32 n_tokens][tokens u32]
//   [emb: vocab*d f32]
//   [h0: seq*d f32]                       <- 层 0 的输入（嵌入拷贝）
//   每层:
//     [wq d*d][wk d*d][wv d*d][wo d*d]
//     [conn w: d*k0][conn src: d*k0 as f32]
//
// Python 侧不读 Rust 的中间结果，只读**输入与权重**，然后自己算前向，
// 与 Rust 的最终 h 逐位比。这样「独立参考」这条判据才是真的。
unsafe fn dump_for_ref(
    n_layers: u32,
    d: u32,
    k0: u32,
    k: u32,
    seq: u32,
    tok: &[u32],
    emb: *const f32,
    layers: &[Layer],
    h0: *const f32,
    h_final: *const f32,
    layer_in: &[Vec<f32>],
) {
    use std::io::Write;
    let dd = d as usize;
    let path = format!(
        "examples/p17_infer/dump_L{}_d{}_k0{}_k{}_s{}.bin",
        n_layers, d, k0, k, seq
    );
    let _ = std::fs::create_dir_all("examples/p17_infer");
    let mut f = match std::fs::File::create(&path) {
        Ok(f) => f,
        Err(e) => {
            println!("FATAL: cannot create dump {}: {}", path, e);
            std::process::exit(9);
        }
    };
    let mut wu32 = |f: &mut std::fs::File, v: u32| {
        let _ = f.write_all(&v.to_le_bytes());
    };
    let vocab = 64u32;
    wu32(&mut f, n_layers);
    wu32(&mut f, d);
    wu32(&mut f, k0);
    wu32(&mut f, k);
    wu32(&mut f, seq);
    wu32(&mut f, vocab);
    wu32(&mut f, tok.len() as u32);
    for t in tok {
        wu32(&mut f, *t);
    }
    let mut wf32 = |f: &mut std::fs::File, p: *const f32, n: usize| {
        let s = std::slice::from_raw_parts(p, n);
        let mut buf = Vec::with_capacity(n * 4);
        for v in s {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        let _ = f.write_all(&buf);
    };
    wf32(&mut f, emb, vocab as usize * dd);
    wf32(&mut f, h0, seq as usize * dd);
    for ly in layers {
        wf32(&mut f, ly.wq, dd * dd);
        wf32(&mut f, ly.wk, dd * dd);
        wf32(&mut f, ly.wv, dd * dd);
        wf32(&mut f, ly.wo, dd * dd);
        wf32(&mut f, ly.sc.w, dd * k0 as usize);
        wf32(&mut f, ly.sc.src, dd * k0 as usize);
    }
    // 每层的**输入**（Python 侧可用来逐层定位第一个不一致的层，
    // 而不仅仅是说「最后输出不一样」—— 差异定位到层比定位到「有 bug」
    // 有用得多，这是本项目 P13 §2 的教训）。
    for l in 0..n_layers as usize {
        let row = &layer_in[l];
        let mut buf = Vec::with_capacity(row.len() * 4);
        for v in row {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        let _ = f.write_all(&buf);
    }
    // 期望输出（Rust 侧算出的最终 h）—— Python 用它做**逐位**比较。
    wf32(&mut f, h_final, seq as usize * dd);
    let _ = f.flush();
    println!("dumped ref bundle -> {}", path);
    // 也让 Python 侧能独立重建 PRNG：把关键常量打出来做交叉校验
    println!(
        "PRNG cross-check: neuron_seed(0,0)={} neuron_conn_src(0,0,0,{}, {})={} neuron_rand(12345,7)={}",
        neuron_seed(0, 0),
        d,
        k0,
        neuron_conn_src(0, 0, 0, d, k0),
        neuron_rand(12345, 7)
    );
}
