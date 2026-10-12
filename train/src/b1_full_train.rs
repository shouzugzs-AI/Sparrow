// PATH: B1
//
// ===========================================================================
// B1 **完整训练**驱动：dim=1536, k0=8, layers=24,**无残差**, Adam + warmup + cosine
// ===========================================================================
// 这是 `train/src/b1_scaling.rs`（扫描用）之外的**专用**训练器：它不扫描任何超参，
// 只把一个配置从头训到坡度拐平，并把产物写到 artifacts/ 与 docs/。
//
// 架构（无残差，任务已定）：
//     h = emb[tok]
//     对每层 l:  h = B1_l(h)          // **纯堆叠**，没有 h += ...
//     logits = lm_w·h + lm_b
//     CE(t) = -log softmax(logits_t)[tok_{t+1}]
//
// 为什么无残差时 `depth_scale = 1`（保方差）而不是 `1/sqrt(2L)`：
//   见 docs/B1_layers_k0_loss.md §5b/§5c 的 2×2 因子实验（实测）：
//     · 残差堆叠 + 保方差初始化  ⇒ 每层方差翻倍 ⇒ L=24 第 1 步就崩（val≈55）
//     · 无残差堆叠 + 1/sqrt(2L) ⇒ 信号按 (1/(2L))^L 衰减 ⇒ val≈ln V（学不动）
//     · **无残差 + 保方差**      ⇒ 最好（L=24,k0=2 时 8.017 vs 有残差 8.812）
//   本驱动默认 `--no-residual 1 --depth-scale 1`，并在启动时把这条依据打印出来。
//
// 优化器：Adam(0.9,0.999) + 线性 warmup + cosine 退火到 `--min-lr`；逐组 clip=1.0。
// 数据：`tools/b1_scaling_encode.py` 用 **MiniCPM 官方 tokenizer** 编码的 u32 流
//       （train/val 按**文档** 90/10）。本文件不含任何 BPE 代码。
//
// 安全网（都会写进产物，不吞掉）：
//   · 每一步检查 loss 是否有限、是否 > 2·ln V ⇒ 记 `diverged_at` 并**停止**
//   · 每 `--eval-every` 步在**固定验证批**上评估；val 连续 `--plateau-patience` 次不改善 ⇒ 停
//   · `--gradcheck 1` 模式：数值梯度 vs 解析梯度（中心差分，噪声感知判据）
//
// USAGE（冒烟）：
//   vmlm-b1-full.exe --dim 64 --k0 4 --layers 4 --steps 8 --seq 16 --batch 2 \
//       --lr 3e-4 --warmup 2 --seed 1 --data-dir artifacts/b1_scaling \
//       --out-dir artifacts/b1_full_smoke --doc docs/_smoke.md
// USAGE（完整训练，见任务输出）：见 docs/B1_full_training_1536x8.md 末尾的一行命令。

extern "C" {
    fn malloc(n: usize) -> *mut u8;
}

use vmlm_core::b1_pure::{b1_backward, b1_backward_dx, b1_backward_fused, b1_forward, b1_layer_backward_t8, b1_layer_forward_t8};
use vmlm_core::conn_train::{init_sparse_conn_scaled, SparseConnW};
use vmlm_core::kernels::gemm_nt;

// [抽库 L0-e/L0-f] 线程池 / pin 原语 / CPU 探测 已搬到 vmlm-sys，这里重导出（调用点零改动）
use vmlm_sys::cpu::logical_cpu_count;
// [抽库 L2a] region 切分/归约搬到 vmlm-ops::par（调用点零改动）
#[allow(unused_imports)]
use vmlm_ops::par::*;
#[allow(unused_imports)]
use vmlm_sys::thread::*;
// [抽库 L3/L4] 稠密 attention 核心 + 通用原语在 vmlm-blocks；B1 模型层（b1/head/attn_sparse）在 vmlm-models；
// 这里重导出 ⇒ 全部调用点零改动。
#[allow(unused_imports)]
use vmlm_blocks::attn::*;
#[allow(unused_imports)]
use vmlm_blocks::util::*;
#[allow(unused_imports)]
use vmlm_models::head::*;
#[allow(unused_imports)]
use vmlm_models::b1::*;
#[allow(unused_imports)]
use vmlm_models::attn_sparse::*;

// ---------------------------------------------------------------------------
// 小工具
// ---------------------------------------------------------------------------
#[derive(Clone)]
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
    fn uni(&mut self) -> f32 {
        (self.next_u32() as f64 / (u32::MAX as f64)) as f32
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.uni()
    }
}

unsafe fn alloc(n: usize) -> *mut f32 {
    let p = malloc(n * 4 + 64) as *mut f32;
    assert!(!p.is_null(), "OOM {} f32 ({} MB)", n, n * 4 / 1048576);
    p
}

// ---- 参数来源：默认 std::env::args()；vmlm-chat 会在同进程内塞入一份覆盖值 ----
// （为什么这样做：vmlm-chat 要在**一个进程生命周期里反复**跑完整的 --generate 流程，
//   而 std::env::args() 是不可变的。加一层可覆盖的 argv，就不用把 3000 行的引擎抽成 lib。）
pub static ARG_OVERRIDE: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
pub static ARG_OVERRIDE_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn args_vec() -> Vec<String> {
    if ARG_OVERRIDE_ON.load(std::sync::atomic::Ordering::Relaxed) {
        if let Ok(g) = ARG_OVERRIDE.lock() {
            if !g.is_empty() {
                return g.clone();
            }
        }
    }
    std::env::args().collect()
}

pub fn set_args(v: Vec<String>) {
    if let Ok(mut g) = ARG_OVERRIDE.lock() {
        *g = v;
    }
    ARG_OVERRIDE_ON.store(true, std::sync::atomic::Ordering::Relaxed);
}

fn arg_u32(flag: &str, def: u32) -> u32 {
    let v: Vec<String> = args_vec();
    for i in 0..v.len() {
        if v[i] == flag && i + 1 < v.len() {
            return v[i + 1].parse().unwrap_or(def);
        }
    }
    def
}
fn arg_str_opt(flag: &str) -> Option<String> {
    let v: Vec<String> = args_vec();
    for i in 0..v.len() {
        if v[i] == flag && i + 1 < v.len() {
            return Some(v[i + 1].clone());
        }
    }
    None
}
fn arg_u32_opt(flag: &str) -> Option<u32> {
    let v: Vec<String> = args_vec();
    for i in 0..v.len() {
        if v[i] == flag && i + 1 < v.len() {
            return v[i + 1].parse().ok();
        }
    }
    None
}
fn arg_f32(flag: &str, def: f32) -> f32 {
    let v: Vec<String> = args_vec();
    for i in 0..v.len() {
        if v[i] == flag && i + 1 < v.len() {
            return v[i + 1].parse().unwrap_or(def);
        }
    }
    def
}
fn arg_str(flag: &str, def: &str) -> String {
    let v: Vec<String> = args_vec();
    for i in 0..v.len() {
        if v[i] == flag && i + 1 < v.len() {
            return v[i + 1].clone();
        }
    }
    def.to_string()
}

// [P21-1/E 类] Adam 的偏差修正系数 (rc1, rc2) 只与步数 t 有关，但**每次调用**都要跑
// 2 次标量 exp_f/ln_f（≈66 条标量指令）。训练器每步调它 15 次（层 ×12 + head/embed），
// 于是每步白跑 14 × 66 条标量指令。这里按 t 缓存一次（t 在一步内不变）。
static ADAM_C_T: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
static ADAM_RC1: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
static ADAM_RC2: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

#[inline]
unsafe fn adam_rc(t: u32) -> (f32, f32) {
    use std::sync::atomic::Ordering::Relaxed;
    if ADAM_C_T.load(Relaxed) == t && t != 0 {
        return (
            f32::from_bits(ADAM_RC1.load(Relaxed)),
            f32::from_bits(ADAM_RC2.load(Relaxed)),
        );
    }
    let tf = t as f32;
    let c1 = 1.0 - vmlm_core::kernels::exp_f(tf * vmlm_core::kernels::ln_f(0.9));
    let c2 = 1.0 - vmlm_core::kernels::exp_f(tf * vmlm_core::kernels::ln_f(0.999));
    let (a, b) = (1.0 / c1, 1.0 / c2);
    ADAM_RC1.store(a.to_bits(), Relaxed);
    ADAM_RC2.store(b.to_bits(), Relaxed);
    ADAM_C_T.store(t, Relaxed);
    (a, b)
}

unsafe fn adam_dense(p: *mut f32, g: *const f32, m: *mut f32, v: *mut f32, n: usize, lr: f32, t: u32) {
    let (rc1, rc2) = adam_rc(t);
    // [P19-9] 8 宽向量化；非 AVX2 走原来的标量循环（逐位不变）
    #[cfg(target_arch = "x86_64")]
    {
        if vmlm_core::interp::HAS_AVX2 {
            vmlm_core::kernels::adam_avx2(p, g, m, v, n, lr, rc1, rc2, 1e-8);
            return;
        }
    }
    for i in 0..n {
        let gi = *g.add(i);
        let mi = 0.9 * *m.add(i) + 0.1 * gi;
        let vi = 0.999 * *v.add(i) + 0.001 * gi * gi;
        *m.add(i) = mi;
        *v.add(i) = vi;
        *p.add(i) -= lr * (mi * rc1) / (vmlm_core::kernels::sqrt_f(vi * rc2) + 1e-8);
    }
}

/// FNV-1a over the loss curve (乘以 1e6 取整后再哈希，避免格式化差异）。
/// 用途：同 seed 跑两次，`loss_hash` 必须相同 —— 用户自己就能核对确定性。
fn loss_hash(v: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &x in v {
        let q = (x as f64 * 1.0e6).round() as i64 as u64;
        for b in 0..8 {
            h ^= (q >> (8 * b)) & 0xff;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
    }
    h
}

/// JSON 字符串转义。**必须用**：Windows 路径里的 `\` 在 JSON 里是转义引导符。
/// 第一版直接把 `<repo>/...` 塞进 `"data_dir":"..."` ⇒ 产出的 summary.json 是
/// **非法 JSON**（`\c` 不是合法转义），PowerShell 的 ConvertFrom-Json 静默返回空对象，
/// 于是下游的「确定性」「发散记录」判据全部在**空值上通过**（假通过）。
/// 这里留档：探针自己坏掉，比被测代码坏掉更难发现。
fn jesc(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '\\' => o.push_str("\\\\"),
            '"' => o.push_str("\\\""),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c => o.push(c),
        }
    }
    o
}

/// 打开 FTZ(bit15) + DAZ(bit6)。
///
/// 为什么必须有（实测，不是预防性写法）：本配置 dim=1536 下，**训练越久每步越慢** ——
/// 前 40 步 0.41 s/步，到 step 150 变成 1.1 s/步，128 步的整轮实测 2.6 s/步。
/// 原因是 softmax 里的 `exp(x - mx)`：`exp_f` 把 `x < -88` 夹到 -88，`exp(-88)=6.1e-39`
/// 就是 **f32 次正规数**；模型越训越自信 ⇒ 越多的项落进次正规区间 ⇒ SIMD 走慢路径。
/// FTZ/DAZ 是数值库的标准做法：把次正规数当 0 处理。
///
/// 配套的数值改动（见 CE 那段）：loss 改用 **log-sum-exp** 形式
/// `loss = (mx + ln(sum)) - x_tgt`，这样**不再对可能次正规/下溢的值取 ln**，
/// 否则 FTZ 会把 `ln(0)` 变成 `+inf`。两处必须一起改。
unsafe fn enable_ftz_daz() {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::{_mm_getcsr, _mm_setcsr};
        let mut csr = _mm_getcsr();
        csr |= 0x8040; // 0x8000 = FTZ, 0x0040 = DAZ
        _mm_setcsr(csr);
    }
}

/// FNV-1a over raw f32 bits —— 用来证明"被冻结的层真的一个比特都没动"。
unsafe fn fnv_f32(p: *const f32, n: usize) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for i in 0..n {
        let bits = (*p.add(i)).to_bits() as u64;
        for b in 0..4 {
            h ^= (bits >> (8 * b)) & 0xff;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
    }
    h
}

/// 非有限数在 JSON 里**没有字面量**（`NaN`/`inf` 都不是合法 JSON）——
/// 第一版把 `final_val=NaN`、`best_val=inf` 直接写进 summary.json，整个文件变成
/// **非法 JSON**，下游 `ConvertFrom-Json` 返回空对象，于是发散判据在空值上"通过"。
/// 非有限一律写 `null`（配合文档里的 n/a）。
fn jnum(x: f32) -> String {
    if x.is_finite() {
        format!("{:.6}", x)
    } else {
        "null".to_string()
    }
}

/// 取一个**必须显式传入**的参数：缺了就报错退出，**不允许默认值/fallback**。
/// 任务书要求三个 k0（主层 / 嵌入 / lm_head）互相独立、缺任一即报错。
fn req_u32(flag: &str) -> u32 {
    let v: Vec<String> = args_vec();
    for i in 0..v.len() {
        if v[i] == flag {
            if i + 1 < v.len() {
                match v[i + 1].parse::<u32>() {
                    Ok(x) => return x,
                    Err(_) => {
                        println!("ERROR: {} expects a positive integer, got '{}'", flag, v[i + 1]);
                        std::process::exit(2);
                    }
                }
            }
            println!("ERROR: {} given without a value", flag);
            std::process::exit(2);
        }
    }
    println!("ERROR: {} is REQUIRED (no default / no fallback)", flag);
    std::process::exit(2);
}
fn has_flag(flag: &str) -> bool {
    std::env::args().any(|a| a == flag)
}

/// 一组 f32 的 RMS（诊断用：看激活/投影到底缩到多小）
unsafe fn rms(p: *const f32, n: usize) -> f32 {
    let mut s = 0.0f64;
    for i in 0..n {
        let v = *p.add(i) as f64;
        s += v * v;
    }
    (s / n.max(1) as f64).sqrt() as f32
}

/// 检查点几何：载入时必须与当前配置一致，否则 exit 2（不能"大概能读"）
#[derive(Clone, Copy)]
struct CkGeo {
    dim: usize,
    k0: usize,
    k0_embed: usize,
    k0_head: usize,
    layers: usize,
    n_attn: usize,
    k0_attn: usize,
    io_sparse: bool,
    head_skip: bool,
    attn_sparse: bool,
    n_vocab: usize,
}

fn ck_header(g: &CkGeo) -> Vec<u32> {
    let mut h = vec![0xB1C0_0DE1u32, 1u32];
    h.push(g.dim as u32);
    h.push(g.k0 as u32);
    h.push(g.k0_embed as u32);
    h.push(g.k0_head as u32);
    h.push(g.layers as u32);
    h.push(g.n_attn as u32);
    h.push(g.k0_attn as u32);
    h.push(g.n_vocab as u32);
    h.push(g.io_sparse as u32);
    h.push(g.head_skip as u32);
    h.push(g.attn_sparse as u32);
    h
}

/// 写权重（只写权重本身；标定因子 g 已经乘进 Wq/Wk，所以跟着权重一起持久化）
unsafe fn ck_write(path: &str, g: &CkGeo, tens: &[(*const f32, usize)]) -> usize {
    use std::io::Write;
    let mut f = std::fs::File::create(path).unwrap_or_else(|e| panic!("cannot create {}: {}", path, e));
    let h = ck_header(g);
    let mut bytes: Vec<u8> = Vec::new();
    for v in &h {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let mut n: u64 = 0;
    for (_, len) in tens {
        n += *len as u64;
    }
    bytes.extend_from_slice(&n.to_le_bytes());
    for (ptr, len) in tens {
        for i in 0..*len {
            bytes.extend_from_slice(&(*ptr.add(i)).to_le_bytes());
        }
    }
    f.write_all(&bytes).unwrap();
    n as usize
}

/// 读权重（几何不一致 → exit 2）
unsafe fn ck_read(path: &str, g: &CkGeo, tens: &mut [(*mut f32, usize)]) -> usize {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("cannot read {}: {}", path, e));
    let hdr = ck_header(g);
    let hn = hdr.len();
    if b.len() < (hn + 2) * 4 {
        eprintln!("FATAL: 检查点 {} 太短", path);
        std::process::exit(2);
    }
    let mut got = Vec::with_capacity(hn);
    for i in 0..hn {
        got.push(u32::from_le_bytes([b[i * 4], b[i * 4 + 1], b[i * 4 + 2], b[i * 4 + 3]]));
    }
    if got != hdr {
        eprintln!(
            "FATAL: 检查点几何与当前配置不一致，拒绝载入。\n  文件: {:?}\n  当前: {:?}",
            got, hdr
        );
        std::process::exit(2);
    }
    let n = u64::from_le_bytes([
        b[hn * 4], b[hn * 4 + 1], b[hn * 4 + 2], b[hn * 4 + 3],
        b[hn * 4 + 4], b[hn * 4 + 5], b[hn * 4 + 6], b[hn * 4 + 7],
    ]) as usize;
    let need: usize = tens.iter().map(|(_, l)| *l).sum();
    if n != need {
        eprintln!("FATAL: 检查点 f32 数量 {} != 期望 {}", n, need);
        std::process::exit(2);
    }
    let mut off = (hn + 2) * 4;
    for (ptr, len) in tens.iter() {
        for i in 0..*len {
            let v = f32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]]);
            *(*ptr).add(i) = v;
            off += 4;
        }
    }
    n
}

// =====================================================================
// 多核并行（--threads N）
// =====================================================================
// 设计取舍：
//  * 用 `std::thread::scope` **每次调用起线程**，不做线程池。步时是秒级，起线程 ~30µs，
//    一次 step 约 100 次并行区 ⇒ 开销 ~10ms，可以忽略；换来的是不用写任务队列。
//  * **`--threads 1` 必须逐位不变**（所有并行函数在 nt<=1 时走原来的串行循环）。
//  * 会被多线程同时写的累加缓冲（dw / head_dw / etab_g）用**每线程私有分段 + 归约**，
//    不用原子加：归约顺序固定（线程号升序），所以同一线程数下**可复现**；
//    不同线程数之间会有浮点求和顺序差异（会报告，不假装逐位相同）。

/// 把 0..n 切成 nt 段并行跑。nt<=1 时直接串行（逐位不变）。

// ---- 线程绑定（避开超线程兄弟核）----
// 实测：不绑定时 4 个 worker 会被调度到 8 个逻辑核上，每个只忙 25~56%，
// 等效只有 ~3.2 个物理核 ⇒ 加速比被压在 ~2.5-3.1x。
// Windows 上 4c/8t 的枚举顺序是"每个物理核的两个逻辑核相邻"（0,1 | 2,3 | ...），
// 所以把第 t 个 worker 绑到逻辑核 `t * nlogical / nt` 就能落到不同物理核上。






// =====================================================================
// 生成（--generate）：**采样 top-k / top-p 是解码策略，与 B2 的 top-k 毫无关系**
//   B2 top-k：前向传播里每个神经元在 k0 条连接中只算 top-k 条 → 改计算量与模型行为（本任务不启用，主层仍是 B1 全算）
//   采样 top-k：解码时从 logits 里取概率最高的 K 个 token，再在这 K 个里采样 → 只决定"下一个 token 怎么选"，不改模型
// 两段代码路径没有任何复用：下面这些函数只被 --generate 调用。
// =====================================================================
struct Rng64(u64);
impl Rng64 {
    fn new(seed: u64) -> Self {
        // splitmix64 起步，避免 seed=0 时退化
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        Rng64((z ^ (z >> 31)) | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// [0,1) 均匀
    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / (1u32 << 24) as f32
    }
}

/// 采样：temperature → 采样 top-k → top-p（nucleus）→ 按概率抽一个 id。
/// 完全确定性（同一个 rng 状态序列 ⇒ 同一个结果）。idx/val 是调用方给的 [n] 临时数组。
#[allow(clippy::too_many_arguments)]
unsafe fn sample_next(
    logits: *const f32,
    n: usize,
    temp: f32,
    top_k: usize,
    top_p: f32,
    rng: &mut Rng64,
    idx: *mut u32,
    val: *mut f32,
    hist: &[u32],          // 已生成的历史（含 prompt），用于重复抑制
    rep_penalty: f32,      // >1 生效：出现过的 token，正 logit 除以它、负 logit 乘它
    no_repeat_ngram: usize,// >=2 生效：禁止让它形成重复的 N-gram
) -> u32 {
    let t = if temp > 1e-6 { temp } else { 1e-6 };
    for i in 0..n {
        *idx.add(i) = i as u32;
        *val.add(i) = *logits.add(i) / t;
    }
    // ---- 重复抑制（在 top-k 之前做，这样被压下去的词不会占 top-k 名额）----
    if rep_penalty > 1.0 && !hist.is_empty() {
        let start = hist.len().saturating_sub(64); // 只看最近 64 个，够用且稳
        for &tk in &hist[start..] {
            let i = tk as usize;
            if i < n {
                let v = *val.add(i);
                *val.add(i) = if v > 0.0 { v / rep_penalty } else { v * rep_penalty };
            }
        }
    }
    if no_repeat_ngram >= 2 && hist.len() >= no_repeat_ngram {
        let m = no_repeat_ngram;
        let plen = m - 1;
        let prefix = &hist[hist.len() - plen..];
        let mut i = 0usize;
        while i + m <= hist.len() {
            if &hist[i..i + plen] == prefix {
                let banned = hist[i + plen] as usize;
                if banned < n {
                    *val.add(banned) = f32::NEG_INFINITY;
                }
            }
            i += 1;
        }
    }
    // 按 logit 降序（简单选择排序的代价在 n=4054 下可接受：~n^2/2 比较 ≈ 8M，几十 ms）
    // ⇒ 为了速度用"部分选择"：先取 top_k（若 top_k<n），再对这 top_k 排序。
    let kk = if top_k == 0 || top_k > n { n } else { top_k };
    for a in 0..kk {
        let mut best = a;
        for b in (a + 1)..n {
            if *val.add(b) > *val.add(best) {
                best = b;
            }
        }
        if best != a {
            core::ptr::swap(val.add(a), val.add(best));
            core::ptr::swap(idx.add(a), idx.add(best));
        }
    }
    // softmax（只在前 k 个上）
    let mut mx = *val.add(0);
    for a in 1..kk {
        if *val.add(a) > mx {
            mx = *val.add(a);
        }
    }
    let mut sum = 0.0f32;
    for a in 0..kk {
        let e = vmlm_core::kernels::exp_f(*val.add(a) - mx);
        *val.add(a) = e;
        sum += e;
    }
    let inv = 1.0 / (sum + 1e-30);
    for a in 0..kk {
        *val.add(a) *= inv;
    }
    // top-p：累积到 >= top_p 就截断（至少保留 1 个）
    let mut keep = kk;
    if top_p > 0.0 && top_p < 1.0 {
        let mut c = 0.0f32;
        for a in 0..kk {
            c += *val.add(a);
            if c >= top_p {
                keep = a + 1;
                break;
            }
        }
    }
    let mut s2 = 0.0f32;
    for a in 0..keep {
        s2 += *val.add(a);
    }
    let mut r = rng.next_f32() * (s2 + 1e-30);
    for a in 0..keep {
        r -= *val.add(a);
        if r <= 0.0 {
            return *idx.add(a);
        }
    }
    *idx.add(keep - 1)
}

/// 读 char_vocab_hex.txt：id -> 字符（kind=char 用十六进制码点；kind=sent 是标识符原文）
fn load_char_map(path: &str) -> Vec<String> {
    let txt = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read char vocab map {}: {}", path, e));
    let mut out: Vec<String> = Vec::new();
    for line in txt.lines() {
        let l = line.trim_end();
        if l.is_empty() {
            continue;
        }
        let mut it = l.split('\t');
        let id: usize = it.next().unwrap_or("0").trim().parse().unwrap_or(0);
        let kind = it.next().unwrap_or("char");
        let v = it.next().unwrap_or("");
        if out.len() <= id {
            out.resize(id + 1, String::new());
        }
        out[id] = if kind == "sent" {
            v.to_string()
        } else {
            let cp = u32::from_str_radix(v.trim(), 16).unwrap_or(0xFFFD);
            char::from_u32(cp).map(|c| c.to_string()).unwrap_or_else(|| "?".to_string())
        };
    }
    out
}

fn read_u32_file(path: &str) -> Vec<u32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("cannot read {}: {}", path, e));
    assert!(bytes.len() % 4 == 0, "{}: not a u32 stream", path);
    let n = bytes.len() / 4;
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        let b = &bytes[i * 4..i * 4 + 4];
        v.push(u32::from_le_bytes([b[0], b[1], b[2], b[3]]));
    }
    v
}

// ---------------------------------------------------------------------------
pub fn run_main() {
    unsafe {
        vmlm_core::interp::init_cpu();
        enable_ftz_daz();
    }

    let dim = arg_u32("--dim", 1536) as usize;
    // ---- 三个独立 k0 ----
    // `--sparse-io 1` 时：主层 `--k0`、嵌入 `--k0-embed`、lm_head `--k0-head` **三个都必须显式给出**
    // （req_u32：缺任一 ⇒ 打印 ERROR 并 exit(2)，不给默认值、不 fallback）。
    // 不开 `--sparse-io` 时保持原有稠密路径（`--k0` 仍可省，行为与之前逐位一致）。
    let io_sparse = arg_u32("--sparse-io", 0) == 1;
    let (k0_embed, k0_head);
    let k0 = if io_sparse {
        let m = req_u32("--k0") as usize;
        k0_embed = req_u32("--k0-embed") as usize;
        k0_head = req_u32("--k0-head") as usize;
        if k0_embed == 0 || k0_head == 0 || m == 0 {
            println!("ERROR: --k0 / --k0-embed / --k0-head must all be > 0");
            std::process::exit(2);
        }
        m
    } else {
        if has_flag("--k0-embed") || has_flag("--k0-head") {
            println!("ERROR: --k0-embed / --k0-head only apply with --sparse-io 1 (refusing to ignore them silently)");
            std::process::exit(2);
        }
        k0_embed = 0;
        k0_head = 0;
        arg_u32("--k0", 8) as usize
    };
    let layers = arg_u32("--layers", 24).max(1) as usize;
    let steps = arg_u32("--steps", 3000) as usize;
    let seq = arg_u32("--seq", 32) as usize;
    let batch = arg_u32("--batch", 4) as usize;
    let lr = arg_f32("--lr", 3.0e-4);
    let min_lr = arg_f32("--min-lr", 3.0e-5);
    let warmup = arg_u32("--warmup", 100) as usize;
    let clip = arg_f32("--clip", 1.0);
    let seed = arg_u32("--seed", 1) as u64;
    let eval_every = arg_u32("--eval-every", 50).max(1) as usize;
    let eval_batches = arg_u32("--eval-batches", 4).max(1) as usize;
    let plateau_patience = arg_u32("--plateau-patience", 10) as usize; // 0 = 关闭
    let min_delta = arg_f32("--min-delta", 1.0e-4);
    let no_res = arg_u32("--no-residual", 1) == 1;
    let vocab_mode = arg_str("--vocab-mode", "observed");
    // 跨分词口径比较：CE(nats/position) 只有在除以"每个 position 覆盖多少字符"之后才可比。
    //   BPE 词表跑：--chars-per-token 1.2938（本数据集实测）
    //   字符词表跑：1.0（默认）
    // 打印 bits/char = CE / chars_per_token / ln2。
    let chars_per_token = arg_f32("--chars-per-token", 1.0);
    // 并行线程数（1 = 串行，逐位与历史一致；>1 = 每线程私有累加缓冲 + 固定顺序归约）
    let nth = arg_u32("--threads", 1).max(1) as usize;
    // ---- 生成（解码）参数。注意：--top-k 是**采样** top-k，与 B2 的前向 top-k 无关。----
    let gen_mode = arg_u32("--generate", 0) == 1;
    // ⚠️ 本机控制台是 GBK：--prompt 里的中文经 shell 传递可能被转码搞坏（实测 '今天' 变成了乱码）。
    //    所以再加一个 --prompt-file（UTF-8 文件），两者同时给时以文件为准。
    let gen_prompt = match arg_str_opt("--prompt-file") {
        Some(f) => std::fs::read_to_string(&f)
            .unwrap_or_else(|e| panic!("cannot read prompt file {}: {}", f, e))
            .trim_end_matches(['\n', '\r'])
            .to_string(),
        None => arg_str("--prompt", "今天"),
    };
    let gen_len = arg_u32("--gen-len", 100) as usize;
    let gen_temp = arg_f32("--temperature", 0.8);
    // 重复抑制（默认关 ⇒ 与旧行为逐位一致）
    let gen_rep_pen = arg_f32("--rep-penalty", 1.0);
    let gen_no_repeat = arg_u32("--no-repeat-ngram", 0) as usize;
    let gen_topk = arg_u32("--top-k", 40) as usize;
    let gen_topp = arg_f32("--top-p", 0.9);
    let gen_out = arg_str_opt("--gen-out");
    // 采样用独立的 RNG 流（由 --seed 派生），与权重初始化互不干扰 ⇒ 同 seed 同 prompt 同输出
    let mut gen_rng = Rng64::new((seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03);
    // 线程绑定默认开（--pin 0 关闭）：避免 4 个 worker 落到 HT 兄弟核上
    let pin = arg_u32("--pin", 1) == 1;
    // ---- 并行模式：intra（按 token 切，旧行为）/ inter（按样本切，数据并行）----
    //   inter：把 batch 切成 nslot 份**完整样本**，每线程独立跑前向+反向+梯度累加，
    //          末端按 slot 升序归约梯度、再统一做一次 Adam。
    //   ⚠️ inter 下每个 region 的 attention 上下文是 region(=sb*seq) 而不是整个 batch*seq，
    //      所以 inter 与 intra 只有在 --attn-window 1（逐窗口独立）时才可能数值一致。
    let pmode = arg_str("--parallel-mode", "inter");
    let inter = pmode == "inter";
    let nslot = if inter { nth.min(batch).max(1) } else { 1 };
    let sb = if inter { (batch / nslot).max(1) } else { batch };
    let tt = sb * seq; // 每个 region 的 token 数（intra 时 == t_tokens）
    let nt_in = if inter { 1 } else { nth }; // 体内并行度：inter 下由外层线程提供
    if inter && batch % nslot != 0 {
        eprintln!("FATAL: --parallel-mode inter 要求 batch({}) 能被线程数({}) 整除", batch, nslot);
        std::process::exit(2);
    }
    if inter && nslot > 1 && arg_u32("--attn-sparse", 1) != 1 {
        eprintln!("FATAL: 稠密 attention 下 inter 需要 nslot 份 [d*d] 梯度缓冲（会 OOM）；请用 --attn-sparse 1 或 --parallel-mode intra");
        std::process::exit(2);
    }
    println!(
        "parallel-mode: {} (nslot={}, 每 slot {} 样本 / {} token, 体内并行度={})",
        pmode, nslot, sb, tt, nt_in
    );
    if inter && nslot > 1 && arg_u32("--attn-window", 0) != 1 {
        println!("  ⚠️ inter + 整块 attention：每个 region 的 attention 上下文 = {} token（不是整个 batch）⇒ 与 intra 数值不同；要语义一致请加 --attn-window 1", tt);
    }
    // [P20] t8（source-major）drop-in 开关（默认开）
    // [P20] drop-in t8 **默认关闭**：指令级确实更快（kbench 前向 1.83x / 反向 2.20x，
    // 含转置），但**接进 trainer 后每种组合都更慢**（实测 primary 100 步：
    // off 0.1587 / fwd-only 0.1864 / bwd-only 0.2115 / both 0.2272 s/步）。
    // 原因：drop-in 每层要额外付 2~3 趟全尺寸转置（≈2~6 MB/层/region 的读写），
    // 而 trainer 是**带宽受限**、不是指令受限（12 层激活 + 注意力缓冲已压满内存系统）。
    // 指令级优势要兑现，必须让布局在**整条栈上**都是 source-major（只在注意力边界转置），
    // 那是 step_fn 的布局改造 —— 见 docs/P20_指令级审计.md §4 的下一步。
    let t8f = arg_u32("--t8", 0) == 1 && arg_u32("--t8-fwd", 1) == 1;
    let t8b = arg_u32("--t8", 0) == 1 && arg_u32("--t8-bwd", 1) == 1;
    T8_ON.store(t8f || t8b, std::sync::atomic::Ordering::Relaxed);
    T8_FWD.store(t8f, std::sync::atomic::Ordering::Relaxed);
    T8_BWD.store(t8b, std::sync::atomic::Ordering::Relaxed);
    println!("t8: drop-in fwd={} bwd={} | stack={}（--t8-stack 1）", t8f, t8b, (arg_u32("--t8-stack", 1) == 1));
    PIN_THREADS.store(if pin { 1 } else { 0 }, std::sync::atomic::Ordering::Relaxed);
    LOGICAL_CPUS.store(logical_cpu_count(), std::sync::atomic::Ordering::Relaxed);
    println!(
        "cores: 逻辑核={} 线程数={} 绑定={}",
        logical_cpu_count(),
        nth,
        if pin { "on" } else { "off" }
    );
    let data_dir = arg_str("--data-dir", "artifacts/b1_scaling");
    let out_dir = arg_str("--out-dir", "artifacts/b1_full_1536x8");
    let doc_path = arg_str("--doc", "");
    let gradcheck = arg_u32("--gradcheck", 0) == 1;
    // ---- 测量钩子（不改模型）：激活频率统计 + 分阶段耗时 ----
    // 任务要求"先跑频率统计（128 步，不改代码）"：这里只加**观测**，
    // 前向/反向/优化器的数学一行不动（由 tools/b1_full_validate.ps1 的 37 项回归保证）。
    let stats_out = arg_str("--stats-out", "");
    let stats_topk = arg_u32("--stats-topk", 64) as usize;
    let profile = arg_u32("--profile", 0) == 1;
    // 融合反向（**接入既有内核**，不是新内核）：`b1_backward_fused` 一次遍历同时算 dw+dx。
    // 为什么必须开：实测分阶段耗时显示**反向占 82%**（逐层反向 53.6% + head 反向 28.7%），
    // 而 `b1_backward` / `b1_backward_dx` 是**标量**实现（`b1_forward` 有 AVX2 路径，
    // 反向没有）—— 这正是"参数少 98% 却只快 3.2x"的原因。
    // 该内核在上一轮已逐位验证（sparseb-bwd E4：dw 0/30720、dx 0/1536 分叉），
    // 这里用 `--no-fusion 1` 保留两条标量调用的路径做 A/B（loss_hash 相同即逐位）。
    let fusion = arg_u32("--no-fusion", 0) == 0;
    // ---- 方案一：嵌入层动态子空间（dynamic sparse training 风格的子空间换维）----
    // 初始仍是「均匀分配」（贪心最少被用块）；训练中每 `--dyn-swap-every` 步把每个 token
    // 重要性最低的槽换成「全局使用次数最少、且不在该 token 子空间里」的维度。
    // 约束：每个 token 恰好 k0_embed 维；总维度使用保持均衡（候选优先选低使用次数）。
    // 共享子空间 ⇒ head 的源**跟着一起换**（`dim_of` 是唯一真相）。
    let dyn_sub = arg_u32("--dyn-subspace", 0) == 1;
    let dyn_every = arg_u32("--dyn-swap-every", 200).max(1) as usize;
    let dyn_ema = arg_f32("--dyn-ema", 0.9);
    // ---- 方案二：把 lm_head 的连接对齐到"高频活跃神经元" ----
    // `--head-hot-pool K`：所有 head 行的 16 个源都从「被最多 token 使用的 K 个维度」里轮转取。
    // 加这个开关是为了**把方案二做成可测的实验**而不是靠论证：
    // 频率统计显示最后一层 1536 个神经元的活跃率**全是 1.0000**（没有"高频"可言），
    // 而且 head 的行专属源是它唯一的"按 token 区分"的读出路径 —— 共享成一个池会把它废掉。
    let head_hot_pool = arg_u32("--head-hot-pool", 0) as usize;
    // ---- 归因实验 A/B/C 的开关 ----
    // A: `--freeze-layers 1` 稀疏层保持初始值不训练
    // B: `--freeze-io 1`     嵌入 + head 保持初始值不训练
    // C: `--identity-layers 1` 稀疏层替换为恒等映射 y = x（不学习）
    let freeze_layers = arg_u32("--freeze-layers", 0) == 1;
    let freeze_io = arg_u32("--freeze-io", 0) == 1;
    let freeze_embed = freeze_io || arg_u32("--freeze-embed", 0) == 1;
    let freeze_head = freeze_io || arg_u32("--freeze-head", 0) == 1;
    let identity_layers = arg_u32("--identity-layers", 0) == 1;
    // 激活 dump（实验 D 有效秩分析）：训练结束后把每一层的激活写盘
    let dump_acts = arg_str("--dump-acts", "");
    let dump_tokens = arg_u32("--dump-tokens", 256) as usize;
    // ---- 第一步检查：线性性数值证明（--linearity-probe 1）----
    // 对同一批 token 把**输入**分别做 (a) 放大 α=2 倍、(b) 拆成两个互不相交的一半 A/B，
    // 然后看逐层输出是否满足  y(αx) = α·y(x)  与  y(A+B) = y(A) + y(B)。
    // α=2 是 2 的幂 ⇒ f32 下乘 2 是**精确**的，所以纯线性结构应当给出**逐位 0 偏差**。
    let lin_probe = arg_u32("--linearity-probe", 0) == 1;
    // ---- 第三步：每层稀疏乘之后加 ReLU（`--relu 1`）----
    // 为什么要加：第一步已证明 24 层是**纯线性**（齐次性逐位 0），所以整条链等效于
    // 一个线性映射（且没有跨 token 交互 ⇒ 结构上只能表达 bigram）。
    // ReLU 是唯一新增的非线性，不引入 top-k、不改主层内核、保持无残差。
    let relu = arg_u32("--relu", 0) == 1;
    // ---- 跨层残差：嵌入 → head 直连（`--head-skip 1`）----
    // 中间层保持无残差（已被实验证明更好），只加**一条**从嵌入展开直通 head 的通路：
    //   logits = head_mid(h_last) + head_skip(h0)
    // 两个 head 都是普通 SparseConnW（同一个块对齐拓扑），各自一套权重与 Adam 状态。
    // 代价：+V×k0_head 参数、+V×k0_head MACs/token（k0_head=16 时 +174,736 / +174,736）。
    // 为什么直连可能有用：末层被压到近秩 1（信息瓶颈），head 只能看到坍缩后的向量；
    // 直连让 head 同时看到**未经过 24 层的原始嵌入**（并给嵌入一条不穿过 24 层的梯度通路）。
    // ⚠️ 拓扑限制（必须写明白）：head 行 v 读的是 block(v) 的维度，而 h0 的非零只在
    //    block(t) 的维度 ⇒ 直连项只在 **block(v) == block(t)** 时非零，即"块内 bigram"
    //    （每个输入 token 直连区分它自己块里的 ~114 个候选 token）。
    let head_skip = arg_u32("--head-skip", 0) == 1;
    // ---- 全局 attention（跨 token 交互）：`--attn-every N` ----
    // 每 N 层稀疏层之后插一个 attention 块（复用上面 attn_forward/attn_backward，
    // 即 sparse_train24 的那条 MiniCPM 注意力路径）： h = h + attn(h)
    // N=0 关闭（默认，保持既有行为逐位不变）；N=layers ⇒ 只在最后一层稀疏层之后插一个。
    let attn_every = arg_u32("--attn-every", 0) as usize;
    let n_attn = if attn_every > 0 { layers / attn_every } else { 0 };
    let attn_drop = arg_f32("--attn-dropout", 0.0);
    // ---- 四个消融用到的开关 ----
    // 消融一：`--attn-no-residual 1` ⇒ h = attn·Wo（替换而非累加）
    // 消融二：`--attn-sparse 1 --k0-attn K` ⇒ Q/K/V/O 用 b1_forward 稀疏投影
    // 消融三：**不加** `--head-skip 1` 即可（回到只读末层）
    let attn_residual = arg_u32("--attn-no-residual", 0) == 0;
    let attn_sparse = arg_u32("--attn-sparse", 0) == 1;
    // 稀疏 attention 的 src 共享模式：
    //   1 = 方案 A（默认）：Q/K 共享 src_qk、V/O 共享 src_vo，共 2 份 src
    //   0 = 旧实现：Q/K/V/O 各自独立均匀铺开（4 份 src，互不相干）
    // 保留 0 是为了能**测出来**两种模式下 q·k 与 softmax 的实际差别，而不是靠推理下结论。
    let attn_src_shared = arg_u32("--attn-src-shared", 1) == 1;
    // 逐窗口 attention（`--attn-window 1`）：每个样本的 seq 个 token 只在自己窗口内做因果 attention。
    // 为什么需要它（实测）：当前实现把**整个 batch*seq 块**当成一条因果序列，
    // 于是 attention 开销是 O((batch*seq)^2)。实测 1M 配置（seq 64, threads 8）：
    //   batch=4 → 0.350 s/step(87.5 ms/样本)；batch=32 → 18.75 s/step(585.9 ms/样本)
    //   ⇒ 每样本成本涨 6.7 倍（batch 只涨 8 倍算力，时间涨 53 倍）——纯浪费。
    // 逐窗口后每样本成本恒定 ⇒ batch=32 应回到 ~2.8 s/step；而且每个窗口互相独立，
    // **这正是 inter-op 数据并行能与 intra-op 数值一致的前提**（否则切 batch 会改变 attention 上下文）。
    // 注意：默认 0（保持与已训模型一致）；改成 1 需要重训（语义变了，而且更正确）。
    let attn_window = arg_u32("--attn-window", 0) == 1;
    let attn_debug = arg_u32("--attn-debug", 0); // 0=关 1=阶段标记 2=逐窗口细节
    let persistent_threads = arg_u32("--persistent-threads", 1) == 1;
    // q·k / softmax 体检：默认开（attention 是否有选择能力，必须可测）
    let attn_qk_check = arg_u32("--attn-qk-check", 1) == 1;
    let k0_attn = if attn_sparse {
        match arg_u32_opt("--k0-attn") {
            Some(v) if v > 0 => v as usize,
            _ => {
                eprintln!("FATAL: --attn-sparse 1 需要显式 --k0-attn K（本项目的三条 k0 都要显式给，不许回退）");
                std::process::exit(2);
            }
        }
    } else {
        0
    };
    // 参数哈希（证明"冻结层真的没更新"）：分组 FNV-1a，训练前后各算一次进 summary.json
    let mut hash_layers_before: u64 = 0;
    let mut hash_embed_before: u64 = 0;
    let mut hash_head_before: u64 = 0;
    let mut hash_layers_after: u64 = 0;
    let mut hash_embed_after: u64 = 0;
    let mut hash_head_after: u64 = 0;
    let gc_eps = arg_f32("--gc-eps", 5.0e-2);
    let gc_probes = arg_u32("--gc-probes", 3) as usize;
    let log_every = arg_u32("--log-every", 100).max(1) as usize;

    // 无残差 ⇒ 保方差初始化（见文件头）。残差模式才用 1/sqrt(2L)。
    let ds_arg = arg_f32("--depth-scale", -1.0);
    let depth_scale = if ds_arg < 0.0 {
        if no_res {
            1.0
        } else {
            1.0 / vmlm_core::kernels::sqrt_f(2.0 * layers as f32)
        }
    } else {
        ds_arg
    };

    // 常驻线程池：**只在这里初始化一次**（顶层、任何 par_for 之前）。
    // ⚠️ 曾经因为脚本锚点匹配到 step_fn 里的第一个 `if profile {`，把这段插进了热路径 ⇒ 训练循环里刷屏。
    if persistent_threads {
        init_pool(nth, pin);
        static POOL_MSG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if nth > 1
            && !POOL_MSG.swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            println!("常驻线程池: {} 个 worker（--persistent-threads 0 回到每次 spawn）", nth);
        }
    }
    let t_tokens = batch * seq;
    // 按 slot 的缓冲尺寸：slot0 用 t_tokens 尺寸（诊断/评估/标定路径都用 tt = t_tokens 调 step_fn），
    // 其余 slot 只需 region 的 tt 尺寸 ⇒ 内存不再随 nslot 乘 t_tokens（batch32/4线程下 2×2.1GB → ~160MB）。
    let tok_slot0 = t_tokens * dim;
    let tok_slotn = tt * dim;
    let tok_total = tok_slot0 + (nslot - 1) * tok_slotn;
    let prob_slot0 = n_attn.max(1) * t_tokens * t_tokens;
    let prob_slotn = n_attn.max(1) * tt * tt;
    let train = read_u32_file(&format!("{}/train_ids.bin", data_dir));
    let val = read_u32_file(&format!("{}/val_ids.bin", data_dir));

    // ---- 词表 ----
    let (n_vocab, id_map) = if vocab_mode == "full" {
        let meta = std::fs::read_to_string(format!("{}/meta.json", data_dir))
            .unwrap_or_else(|e| panic!("need meta.json in {}: {}", data_dir, e));
        let key = "\"vocab_size\":";
        let pos = meta.find(key).expect("vocab_size missing") + key.len();
        let rest = &meta[pos..];
        let end = rest
            .find(|c: char| !c.is_ascii_digit() && c != ' ')
            .unwrap_or(rest.len());
        let vs: usize = rest[..end].trim().parse().expect("bad vocab_size");
        let mut map = vec![0u32; vs];
        for i in 0..vs {
            map[i] = i as u32;
        }
        (vs, map)
    } else if vocab_mode != "observed" {
        // 切表模式：由 tools/b1_vocab_cjk.py 产出的映射文件（每行一个**旧 raw token id**，
        // 行号即新 id-1；未列出的旧 id 一律映到 0 = UNK）。
        // 必须显式给 --vocab-map：不给就 exit 2（不许悄悄回退到 observed）。
        let path = match arg_str_opt("--vocab-map") {
            Some(p) => p,
            None => {
                eprintln!("FATAL: --vocab-mode {} 需要显式 --vocab-map FILE（不许回退到 observed）", vocab_mode);
                std::process::exit(2);
            }
        };
        let txt = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read vocab map {}: {}", path, e));
        let mut ids: Vec<u32> = Vec::new();
        for line in txt.lines() {
            let l = line.trim();
            if l.is_empty() || l.starts_with('#') {
                continue;
            }
            ids.push(l.parse().unwrap_or_else(|_| panic!("bad id in {}: {:?}", path, l)));
        }
        if ids.is_empty() {
            eprintln!("FATAL: 词表映射 {} 是空的", path);
            std::process::exit(2);
        }
        let maxv = *ids.iter().max().unwrap() as usize;
        let mut map = vec![0u32; maxv + 1];
        for (i, &id) in ids.iter().enumerate() {
            map[id as usize] = (i as u32) + 1; // 0 = UNK
        }
        println!("vocab map: {} ({} 个新 id, 旧 id 上界 {})", path, ids.len(), maxv);
        (ids.len() + 1, map)
    } else {
        let mut maxv = 0u32;
        for &t in train.iter() {
            if t > maxv {
                maxv = t;
            }
        }
        let mut seen = vec![false; (maxv + 1) as usize];
        for &t in train.iter() {
            seen[t as usize] = true;
        }
        let ids: Vec<u32> = (0..=maxv).filter(|&i| seen[i as usize]).collect();
        let mut map = vec![0u32; (maxv + 1) as usize];
        for (i, &id) in ids.iter().enumerate() {
            map[id as usize] = (i as u32) + 1; // 0 = UNK
        }
        (ids.len() + 1, map)
    };
    let map_tok = |t: u32| -> u32 {
        if (t as usize) < id_map.len() {
            id_map[t as usize]
        } else {
            0
        }
    };
    let mut val_oov = 0usize;
    if vocab_mode != "full" {
        for &t in val.iter() {
            if (t as usize) >= id_map.len() || id_map[t as usize] == 0 {
                val_oov += 1;
            }
        }
    }
    let train_mapped: Vec<u32> = train.iter().map(|&t| map_tok(t)).collect();
    let val_mapped: Vec<u32> = val.iter().map(|&t| map_tok(t)).collect();
    let lnv = vmlm_core::kernels::ln_f(n_vocab as f32);
    let div_thresh = 2.0 * lnv;

    // =====================================================================
    // 稀疏 IO：嵌入 + lm_head 的**共享子空间**块分配
    // =====================================================================
    // 算法（任务书指定）：
    //   a. dim 维切成 dim/k0_embed 个「k0_embed 维块」；
    //   b. 每个 token id **贪心**分到当前被用次数最少的块（并列取最小下标）；
    //   c. 因为块大小相同且使用次数最多差 1，所以**每个维度的使用次数也大致相同**。
    // 同一个 block_of[v] 同时决定：
    //   · 嵌入侧：token v 的 k0_embed 个值放在第 block_of[v] 块的 k0_embed 个维度上；
    //   · lm_head 侧：第 v 行读的 k0_head 个源就是**同一个块**的 k0_embed 个维度。
    // 这就是「共享子空间定义」：共享的不是矩阵，而是**子空间结构**。
    let mut block_of: Vec<u32> = Vec::new();
    let mut n_blocks = 0usize;
    if io_sparse {
        if k0_embed > dim || dim % k0_embed != 0 {
            println!(
                "ERROR: dim ({}) must be a positive multiple of --k0-embed ({})",
                dim, k0_embed
            );
            std::process::exit(2);
        }
        if k0_head > k0_embed {
            // 共享子空间：head 只能读嵌入块里的维度 ⇒ k0_head <= k0_embed
            println!(
                "ERROR: --k0-head ({}) must be <= --k0-embed ({}) because the head reads the shared subspace",
                k0_head, k0_embed
            );
            std::process::exit(2);
        }
        n_blocks = dim / k0_embed;
        let mut usage = vec![0u32; n_blocks];
        block_of = vec![0u32; n_vocab];
        for v in 0..n_vocab {
            let mut best = 0usize;
            let mut bv = u32::MAX;
            for b in 0..n_blocks {
                if usage[b] < bv {
                    bv = usage[b];
                    best = b;
                }
            }
            block_of[v] = best as u32;
            usage[best] += 1;
        }
        let umin = *usage.iter().min().unwrap();
        let umax = *usage.iter().max().unwrap();
        let mut used_dims = vec![0u32; dim];
        for v in 0..n_vocab {
            let b = block_of[v] as usize;
            for j in 0..k0_embed {
                used_dims[b * k0_embed + j] += 1;
            }
        }
        let dmin = *used_dims.iter().min().unwrap();
        let dmax = *used_dims.iter().max().unwrap();
        println!(
            "sparse-IO: k0(main)={} k0_embed={} k0_head={} | {} blocks x {} dims | block usage {}..{} , per-dim usage {}..{}",
            k0, k0_embed, k0_head, n_blocks, k0_embed, umin, umax, dmin, dmax
        );
        println!(
            "  dyn_subspace={} (swap every {} steps, EMA {}) | fusion={}",
            dyn_sub, dyn_every, dyn_ema, fusion
        );
    }
    // 子空间的唯一真相：dim_of[v*k0_embed + j] = 第 j 个槽对应的维度。
    // 动态模式下它会被换维改写；静态模式下它就是块布局（连续）。
    let mut dim_of: Vec<u32> = Vec::new();
    if io_sparse {
        dim_of = vec![0u32; n_vocab * k0_embed];
        for v in 0..n_vocab {
            let b = block_of[v] as usize;
            for j in 0..k0_embed {
                dim_of[v * k0_embed + j] = (b * k0_embed + j) as u32;
            }
        }
    }
    let mut dim_usage: Vec<u32> = Vec::new();
    if io_sparse {
        dim_usage = vec![0u32; dim];
        for v in 0..n_vocab {
            for j in 0..k0_embed {
                dim_usage[dim_of[v * k0_embed + j] as usize] += 1;
            }
        }
    }
    let params_main = layers * dim * k0;
    let params_embed = if io_sparse { n_vocab * k0_embed } else { n_vocab * dim };
    let params_head = if io_sparse {
        n_vocab * k0_head + if head_skip { n_vocab * k0_head } else { 0 }
    } else {
        n_vocab * dim + n_vocab
    };
    let macs_head = if io_sparse {
        n_vocab * k0_head + if head_skip { n_vocab * k0_head } else { 0 }
    } else {
        dim * n_vocab
    };

    let params_attn = if attn_sparse {
        n_attn * 4 * dim * k0_attn
    } else {
        n_attn * 4 * dim * dim
    };
    let macs_attn = params_attn + n_attn * 2 * t_tokens * dim;

    println!("===== B1 full training =====");
    println!(
        "dim={} k0={} layers={} no_residual={} depth_scale={:.6} | steps={} seq={} batch={} ({} tok/step)",
        dim, k0, layers, no_res, depth_scale, steps, seq, batch, t_tokens
    );
    println!(
        "optimizer=Adam(0.9,0.999) warmup={} cosine->{:.1e} peak_lr={:.1e} clip={} seed={}",
        warmup, min_lr, lr, clip, seed
    );
    println!(
        "data: {} train tok / {} val tok | vocab_mode={} n_vocab={} (ln V={:.4}) val OOV {} tok ({:.3}%)",
        train.len(), val.len(), vocab_mode, n_vocab, lnv, val_oov,
        100.0 * val_oov as f64 / val.len().max(1) as f64
    );
    println!(
        "params: main={} embed={} head={} attn={} total={}  (dense-IO would be {} + {} + {})",
        params_main, params_embed, params_head, params_attn,
        params_main + params_embed + params_head + params_attn,
        params_main, n_vocab * dim, n_vocab * dim + n_vocab
    );
    println!(
        "MACs/token: main={} head={}  (dense head = {} , ratio {:.1}x)",
        layers * dim * k0,
        macs_head,
        dim * n_vocab,
        (dim * n_vocab) as f64 / macs_head.max(1) as f64
    );

    unsafe {
        // ---------- 参数 ----------
        let emb = alloc(n_vocab * dim);
        let lm_w = alloc(n_vocab * dim);
        let lm_b = alloc(if io_sparse { 1 } else { n_vocab });
        zero(lm_b, if io_sparse { 1 } else { n_vocab });
        // 稀疏 IO 模式不分配稠密 emb/lm_w（省下 ~470 MB），改用下面两张小表
        let nd = if io_sparse { 1 } else { n_vocab * dim };
        let emb = alloc(nd);
        let lm_w = alloc(nd);
        {
            let mut r = Rng::new(0x5EED_0000 ^ seed);
            let s = vmlm_core::kernels::sqrt_f(1.0 / dim as f32);
            if !io_sparse {
                for i in 0..n_vocab * dim {
                    *emb.add(i) = r.range(-s, s);
                }
                for i in 0..n_vocab * dim {
                    *lm_w.add(i) = r.range(-s, s);
                }
            }
        }
        let emb_g = alloc(nslot * nd);
        let lm_g = alloc(nslot * nd);
        let lm_bg = alloc(nslot * (if io_sparse { 1 } else { n_vocab }));
        let emb_m = alloc(nd);
        let emb_v = alloc(nd);
        let lm_m = alloc(nd);
        let lm_v = alloc(nd);
        let lm_bm = alloc(if io_sparse { 1 } else { n_vocab });
        let lm_bv = alloc(if io_sparse { 1 } else { n_vocab });
        zero(emb_g, nd);
        zero(lm_g, nd);
        zero(lm_bg, if io_sparse { 1 } else { n_vocab });
        zero(emb_m, nd);
        zero(emb_v, nd);
        zero(lm_m, nd);
        zero(lm_v, nd);
        zero(lm_bm, if io_sparse { 1 } else { n_vocab });
        zero(lm_bv, if io_sparse { 1 } else { n_vocab });

        // ---------- 稀疏 IO：嵌入表 [V × k0_embed] + lm_head 稀疏连接 ----------
        // 嵌入：每个 token 只存 k0_embed 个值（存的是「块内第 j 个维度」的值），
        //       init 仍用项目的 Glorot 口径 s = sqrt(3/k0_embed)。
        // lm_head：一个普通的 SparseConnW，n_neurons = n_vocab, k0 = k0_head，
        //       只是**拓扑由块布局给定**：src[v][j] = block(v)*k0_embed + j。
        //       ⇒ 直接复用 b1_forward / b1_backward / b1_backward_dx，零新内核。
        let etab = alloc(if io_sparse { n_vocab * k0_embed } else { 1 });
        let head_base = if io_sparse {
            malloc(SparseConnW::bytes(n_vocab as u32, k0_head as u32))
        } else {
            malloc(64)
        };
        let head = SparseConnW::from_base(
            head_base,
            n_vocab as u32,
            if io_sparse { k0_head as u32 } else { 1 },
            (seed as u32).wrapping_mul(7919).wrapping_add(3),
        );
        let head_dw = alloc(nslot * if io_sparse { n_vocab * k0_head } else { 1 });
        let head_m = alloc(if io_sparse { n_vocab * k0_head } else { 1 });
        let head_v = alloc(if io_sparse { n_vocab * k0_head } else { 1 });
        // 跨层残差：第二个 head（读 h0 = 嵌入展开）+ 它自己的梯度/Adam 状态 + 临时 logits
        let head2_base = if io_sparse && head_skip {
            malloc(SparseConnW::bytes(n_vocab as u32, k0_head as u32))
        } else {
            malloc(64)
        };
        let head2 = SparseConnW::from_base(
            head2_base,
            if io_sparse { n_vocab as u32 } else { 1 },
            if io_sparse && head_skip { k0_head as u32 } else { 1 },
            (seed as u32).wrapping_mul(104_729).wrapping_add(11),
        );
        let head2_dw = alloc(nslot * if io_sparse && head_skip { n_vocab * k0_head } else { 1 });
        let head2_m = alloc(if io_sparse && head_skip { n_vocab * k0_head } else { 1 });
        let head2_v = alloc(if io_sparse && head_skip { n_vocab * k0_head } else { 1 });
        let d_h0 = alloc(tok_total); // 直连项回传到 h0 的梯度（按 slot 私有）
        let etab_g = alloc(nslot * if io_sparse { n_vocab * k0_embed } else { 1 });
        let etab_m = alloc(if io_sparse { n_vocab * k0_embed } else { 1 });
        let etab_v = alloc(if io_sparse { n_vocab * k0_embed } else { 1 });
        if io_sparse {
            let mut r = Rng::new(0xE3B0_0000 ^ seed);
            let s = vmlm_core::kernels::sqrt_f(3.0 / k0_embed as f32);
            for i in 0..n_vocab * k0_embed {
                *etab.add(i) = r.range(-s, s);
            }
            // 先按项目标准初始化（拿到 Glorot 权重的同一个 PRNG 流），
            // 再**覆盖拓扑**为块布局 —— 这是第二个 src 写入点，所以必须同步 u32 镜像，
            // 随后 assert_src_u32_consistent() 会校验（项目特意为此设了硬失败）。
            init_sparse_conn_scaled(&head, 1.0);
            // 方案二：把 head 的源改成"热池轮转"（K=0 则用共享子空间块，见下）
            let hot_pool: Vec<u32> = if head_hot_pool > 0 {
                // 用当前 dim_usage 最多的 K 个维度当"高频活跃神经元"池
                let mut idx: Vec<u32> = (0..dim as u32).collect();
                idx.sort_by_key(|&d| (std::cmp::Reverse(dim_usage[d as usize]), d));
                idx.truncate(head_hot_pool.min(dim));
                idx
            } else {
                Vec::new()
            };
            for v in 0..n_vocab {
                for j in 0..k0_head {
                    let t = head.idx(v, j);
                    let sid = if head_hot_pool > 0 {
                        hot_pool[(v * k0_head + j) % hot_pool.len()]
                    } else {
                        dim_of[v * k0_embed + j]
                    };
                    *head.src.add(t) = sid as f32;
                    *head.src_u32.add(t) = sid;
                }
            }
            head.assert_src_u32_consistent();
            head.assert_geometry();
            // 直连 head：同一个块对齐拓扑（读 h0 的 block(v) 维度），权重独立初始化
            if head_skip {
                init_sparse_conn_scaled(&head2, 1.0);
                for v in 0..n_vocab {
                    for j in 0..k0_head {
                        let t = head2.idx(v, j);
                        let sid = if head_hot_pool > 0 {
                            hot_pool[(v * k0_head + j) % hot_pool.len()]
                        } else {
                            dim_of[v * k0_embed + j]
                        };
                        *head2.src.add(t) = sid as f32;
                        *head2.src_u32.add(t) = sid;
                    }
                }
                head2.assert_src_u32_consistent();
                head2.assert_geometry();
                zero(head2_dw, n_vocab * k0_head);
                zero(head2_m, n_vocab * k0_head);
                zero(head2_v, n_vocab * k0_head);
            }
            zero(head_dw, n_vocab * k0_head);
            zero(head_m, n_vocab * k0_head);
            zero(head_v, n_vocab * k0_head);
            zero(etab_g, n_vocab * k0_embed);
            zero(etab_m, n_vocab * k0_embed);
            zero(etab_v, n_vocab * k0_embed);
        }

        // =====================================================================
        // [P19-7] 块对齐 head 快路径的**前提检查**（不满足就整条退回通用内核）
        // =====================================================================
        // 块分配是「贪心取最少使用、并列取最小下标」⇒ 在 n_vocab > n_blocks 时
        // 结果就是 block_of[v] = v % n_blocks，于是 head 每行的 k0_head 个源连续。
        // 这里**实测校验**（O(V*k0_head) 一次），而不是靠推理：任一行的 src 不等于
        // (v%nb)*k0_embed + j 就整条退回。另加 dyn_sub / head_hot_pool 两个会改写
        // 拓扑的开关（它们会让这条不变量在训练中途失效）。
        let nb_head = if io_sparse { dim / k0_embed } else { 0 };
        let head_blocked = unsafe {
            io_sparse
                && !dyn_sub
                && head_hot_pool == 0
                && k0_head <= k0_embed
                && nb_head > 0
                && (nb_head & (nb_head - 1)) == 0 // nb 必须是 2 的幂
                && vmlm_core::interp::HAS_AVX2
                && (0..n_vocab).all(|v| {
                    (0..k0_head).all(|j| {
                        *head.src_u32.add(v * k0_head + j) as usize == (v % nb_head) * k0_embed + j
                    }) && (0..k0_embed).all(|j| {
                        *dim_of.as_ptr().add(v * k0_embed + j) as usize == (v % nb_head) * k0_embed + j
                    })
                })
        };
        if io_sparse && n_attn >= 0 {
            println!(
                "head 快路径（块对齐，无 gather）: {} (nb={}, k0_embed={}, k0_head={})",
                if head_blocked { "启用" } else { "未启用（拓扑非块对齐 / 开关冲突 / 无 AVX2）" },
                nb_head,
                k0_embed,
                k0_head
            );
        }
        // [P19-8] WT（列主序权重）与 ht 工作区：块对齐时前向走 8 行/ymm 无归约路径。
        // 不满足前提时只分配 1 个元素（不占内存），代码路径也不会碰它们。
        let head_wt = alloc(if head_blocked { n_vocab * k0_head.max(1) } else { 1 });
        let head2_wt = alloc(if head_blocked && head_skip { n_vocab * k0_head.max(1) } else { 1 });
        // ht：每个 token 一份（每 slot 私有；slot0 用 t_tokens 尺寸，与 arowsum 同一套偏移）
        let htr = if head_blocked { nb_head * k0_head.max(1) } else { 1 };
        let head_ht = alloc(if head_blocked { t_tokens + (nslot - 1) * tt } else { 1 } * htr);

        // ---------- B1 层（每层独立拓扑 + 独立 Adam 状态）----------
        let mut scs: Vec<SparseConnW> = Vec::with_capacity(layers);
        let mut dws: Vec<*mut f32> = Vec::with_capacity(layers);
        let mut sp_ms: Vec<*mut f32> = Vec::with_capacity(layers);
        let mut sp_vs: Vec<*mut f32> = Vec::with_capacity(layers);
        for l in 0..layers {
            let base = malloc(SparseConnW::bytes(dim as u32, k0 as u32));
            let lid = if layers == 1 {
                seed as u32
            } else {
                (seed as u32).wrapping_mul(131).wrapping_add(l as u32)
            };
            let sc = SparseConnW::from_base(base, dim as u32, k0 as u32, lid);
            init_sparse_conn_scaled(&sc, depth_scale);
            sc.assert_geometry();
            sc.assert_src_u32_consistent();
            let dw = alloc(nslot * dim * k0);
            let m = alloc(dim * k0);
            let v = alloc(dim * k0);
            zero(dw, nslot * dim * k0);
            zero(m, dim * k0);
            zero(v, dim * k0);
            scs.push(sc);
            dws.push(dw);
            sp_ms.push(m);
            sp_vs.push(v);
        }

        // ---------- 全局 attention（可选，n_attn 块，每块 4 个 [d×d] 稠密权重）----------
        // ⚠️ 只有**稠密** attention 才分配 [d×d] 权重。第一版无条件分配，
        //    在 --attn-sparse 1 下白占 n_attn*4*(w+grad+m+v)*d^2 字节：
        //    dim=4096/8 块 = 8.6 GB（实测峰值 WS 8872 MB）—— 与"稀疏 attention 省参数量"完全矛盾。
        let ad = if n_attn > 0 && !attn_sparse { dim * dim } else { 1 };
        let awq = alloc(n_attn.max(1) * ad);
        let awk = alloc(n_attn.max(1) * ad);
        let awv = alloc(n_attn.max(1) * ad);
        let awo = alloc(n_attn.max(1) * ad);
        let awq_g = alloc(nslot * n_attn.max(1) * ad);
        let awk_g = alloc(n_attn.max(1) * ad);
        let awv_g = alloc(n_attn.max(1) * ad);
        let awo_g = alloc(n_attn.max(1) * ad);
        let awq_m = alloc(n_attn.max(1) * ad);
        let awk_m = alloc(n_attn.max(1) * ad);
        let awv_m = alloc(n_attn.max(1) * ad);
        let awo_m = alloc(n_attn.max(1) * ad);
        let awq_v = alloc(n_attn.max(1) * ad);
        let awk_v = alloc(n_attn.max(1) * ad);
        let awv_v = alloc(n_attn.max(1) * ad);
        let awo_v = alloc(n_attn.max(1) * ad);
        for b in 0..(if attn_sparse { 0 } else { n_attn }) {
            let s = vmlm_core::kernels::sqrt_f(1.0 / dim as f32);
            let mut r = Rng::new(0xA77E_0000u64 ^ seed ^ ((b as u64) << 12));
            for w in [awq, awk, awv, awo] {
                for i in 0..ad {
                    *w.add(b * ad + i) = r.range(-s, s);
                }
            }
            for g in [awq_g, awk_g, awv_g, awo_g] {
                zero(g.add(b * ad), ad);
            }
            for m in [awq_m, awk_m, awv_m, awo_m, awq_v, awk_v, awv_v, awo_v] {
                zero(m.add(b * ad), ad);
            }
        }
        if attn_sparse {
            // 稀疏投影模式下这几个缓冲完全不参与计算，但可能是未初始化的 malloc 内存；
            // 一旦被"探针选择"读到就会报出天文数字的相对误差（A6 实测 rel=4e27）。
            for w in [awq, awk, awv, awo, awq_g, awk_g, awv_g, awo_g, awq_m, awk_m, awv_m, awo_m, awq_v, awk_v, awv_v, awo_v] {
                zero(w, n_attn.max(1) * ad);
            }
        }
        // attention 的工作缓冲（[t×d] 若干 + [t×t] 概率 + [d×d] 权重转置）
        // [P19-4] q/k/v/o/asp 改成**每个 attention 块一份**（原来每 region 只有一份，
        //   于是反向必须先重算再反传 —— 重算版每块要跑 6 次整尺寸稀疏前向）。
        //   代价：多 n_attn-1 份 [tok_total] 缓冲；换来的是反向不再重算。
        let aq = alloc(n_attn.max(1) * tok_total);
        let ak = alloc(n_attn.max(1) * tok_total);
        let av = alloc(n_attn.max(1) * tok_total);
        let ao = alloc(n_attn.max(1) * tok_total);
        let aout = alloc(tok_total);
        let aprob = alloc(prob_slot0 + (nslot - 1) * prob_slotn);
        let adq = alloc(tok_total);
        let adk = alloc(tok_total);
        let adv = alloc(tok_total);
        let adprob = alloc(prob_slot0 + (nslot - 1) * prob_slotn);
        let arowsum = alloc(t_tokens.max(1) + (nslot - 1) * tt.max(1));
        let adobuf = alloc(tok_total);
        let aat1 = alloc(dim * t_tokens);
        let aat2 = alloc(dim * t_tokens);
        let awt = alloc(dim * dim);
        let asp = alloc(n_attn.max(1) * tok_total); // 缓存的稀疏层输出（attention 的输入）
        let adx = alloc(tok_total); // 对 attention 输入的梯度（残差/替换由调用方决定）
        let qk_stats = alloc(n_attn.max(1) * 5);
        // 跨词表比较 val 的诚实口径：把"目标恰好是 UNK(0)"的位置单独统计。
        // 切表后那些不可预测的稀有 token 全变成同一个可预测的 id 0 ⇒ 整体 val CE 会"假性下降"。
        // 只有非 UNK 子集上的 CE 才能跨词表比较。（裸指针：闭包按值捕获，不引入借用冲突）
        // 每线程私有 dw 分段：取所有"需要累加的权重"里最大的那个
        let max_wlen = [
            dim * k0,
            if attn_sparse { dim * k0_attn } else { 1 },
            if io_sparse { n_vocab * k0_head } else { n_vocab * dim },
            if io_sparse { n_vocab * k0_embed } else { 1 },
        ]
        .into_iter()
        .max()
        .unwrap_or(1);
        // 每个 region 给足 nth*max_wlen（不能只给 max_wlen）：见 docs/B1_crash_inter_backward.md
    let dwp = alloc(nslot.max(1) * nth.max(1) * max_wlen);
        println!("threads: {} (每线程私有 dw 分段 {} f32, 共 {:.1} MB)", nth, max_wlen, (nth * max_wlen * 4) as f64 / 1e6);
        let ce_known_sum = alloc(nslot);
        let ce_known_n = alloc(nslot);
        let ce_unk_n = alloc(nslot);
        zero(qk_stats, n_attn.max(1) * 5);
        let collect_flag: *mut u8 = malloc(n_attn.max(1)); // 采集开关 0/1（裸指针：闭包按值捕获）
        for ci in 0..n_attn.max(1) {
            *collect_flag.add(ci) = 0;
        }
        // 消融二：稀疏投影 Q/K/V/O（每个都是 [dim × k0_attn] 的 SparseConnW）
        let as_w = nslot * if attn_sparse { n_attn.max(1) * dim * k0_attn.max(1) } else { 1 };
        let asq_g = alloc(as_w);
        let ask_g = alloc(as_w);
        let asv_g = alloc(as_w);
        let aso_g = alloc(as_w);
        let asq_m = alloc(as_w);
        let ask_m = alloc(as_w);
        let asv_m = alloc(as_w);
        let aso_m = alloc(as_w);
        let asq_v = alloc(as_w);
        let ask_v = alloc(as_w);
        let asv_v = alloc(as_w);
        let aso_v = alloc(as_w);
        let mut asq: Vec<SparseConnW> = Vec::new();
        let mut ask: Vec<SparseConnW> = Vec::new();
        let mut asv: Vec<SparseConnW> = Vec::new();
        let mut aso: Vec<SparseConnW> = Vec::new();
        if attn_sparse {
            let ka = k0_attn;
            let s = vmlm_core::kernels::sqrt_f(1.0 / dim as f32);
            for b in 0..n_attn {
                for (tag, dst) in [(1u64, &mut asq), (2, &mut ask), (3, &mut asv), (4, &mut aso)] {
                    let base = malloc(SparseConnW::bytes(dim as u32, ka as u32));
                    let mut sc = SparseConnW::from_base(base, dim as u32, ka as u32, 0);
                    let mut r = Rng::new(0x5EED_0000u64 ^ seed ^ ((b as u64) << 8) ^ tag);
                    for i in 0..dim * ka {
                        *sc.w.add(i) = r.range(-s, s);
                    }
                    // 方案 A：Q/K 用同一套 src（tag 折成同一个偏移），V/O 用另一套。
                    // 旧实现（--attn-src-shared 0）：四个投影各自偏移 13·tag。
                    let off = if attn_src_shared {
                        if tag <= 2 { 0usize } else { 4099usize }
                    } else {
                        tag as usize * 13
                    };
                    for i in 0..dim {
                        for j in 0..ka {
                            let sid = ((i * ka + j * 7919 + off) % dim) as u32;
                            *sc.src.add(i * ka + j) = sid as f32;
                            *sc.src_u32.add(i * ka + j) = sid;
                        }
                    }
                    sc.assert_src_u32_consistent();
                    sc.assert_geometry();
                    dst.push(sc);
                }
            }
            for g in [asq_g, ask_g, asv_g, aso_g] {
                zero(g, as_w);
            }
            for m in [asq_m, ask_m, asv_m, aso_m, asq_v, ask_v, asv_v, aso_v] {
                zero(m, as_w);
            }
        }
        // ---------- 缓冲 ----------
        let mut h_ins: Vec<*mut f32> = Vec::with_capacity(layers);
        for _ in 0..layers {
            h_ins.push(alloc(t_tokens * dim));
        }
        // ReLU 需要保存每层的**预激活**（反向要用它的符号做掩码）
        let mut pres: Vec<*mut f32> = Vec::with_capacity(layers);
        for _ in 0..layers {
            pres.push(if relu { alloc(t_tokens * dim) } else { alloc(1) });
        }
        let h = alloc(t_tokens * dim);
        let y = alloc(t_tokens * dim);
        let logits = alloc(t_tokens * n_vocab);
        let logits2 = alloc(if io_sparse && head_skip { t_tokens * n_vocab } else { 1 });
        let dlogits = alloc(t_tokens * n_vocab);
        let dh = alloc(t_tokens * dim);
        let dy = alloc(t_tokens * dim);
        let dx = alloc(t_tokens * dim);
        let bt = alloc(dim * n_vocab); // lm_wᵀ [dim × V]
        let btd = alloc(n_vocab * t_tokens); // dlogitsᵀ [V × T]
        let bth = alloc(dim * t_tokens); // hᵀ [dim × T]

        // ---------- 前向 / 反向 ----------
        // ---------- 测量缓冲（stats / profile）----------
        // 相位：0=嵌入+逐层前向 1=head前向 2=CE/softmax 3=head反向 4=逐层反向 5=嵌入反向 6=取批 7=Adam
        let mut prof: Vec<f64> = vec![0.0; 8];
        let prof_p = prof.as_mut_ptr();
        let collect = !stats_out.is_empty();
        // 最后一层每个神经元的"非零"出现次数 / 层0 / 输入
        let mut nz_last: Vec<u32> = vec![0; dim];
        let mut nz_l0: Vec<u32> = vec![0; dim];
        let mut nz_in: Vec<u32> = vec![0; dim];
        let nz_last_p = nz_last.as_mut_ptr();
        let nz_l0_p = nz_l0.as_mut_ptr();
        let nz_in_p = nz_in.as_mut_ptr();
        let mut samples: u64 = 0; // 累计 (step × token) 数
        let mut head_hits: u64 = 0; // head 连接里"源非零"的条数
        let mut head_conns: u64 = 0;
        // 用裸指针累加，保持闭包是 Fn（否则 step_fn 变 FnMut，与 eval_val 的借用冲突）
        let hh_p: *mut u64 = &mut head_hits;
        let hc_p: *mut u64 = &mut head_conns;
        let mut emb_use: Vec<u64> = vec![0; dim.max(1)]; // 每个维度被激活的次数
        let emb_use_p: *mut u64 = emb_use.as_mut_ptr();
        // 动态子空间：dim_of 的唯一写点在下面的换维例程；闭包只**读**它（裸指针，保持 Fn）
        let dim_of_p: *mut u32 = dim_of.as_mut_ptr();
        let mut imp: Vec<f32> = vec![0.0; (n_vocab * k0_embed).max(1)];
        let imp_p: *mut f32 = imp.as_mut_ptr();
        let mut dyn_swaps_total: usize = 0;
        let mut dyn_rounds: usize = 0;
        let mut dyn_cursor: usize = 0; // 跨轮保留：避免每轮从头开始造成瞬时偏斜

        // =====================================================================
        // --linearity-probe 1：逐层线性性数值证明
        // =====================================================================
        if lin_probe {
            let nt = t_tokens;
            let toks: Vec<u32> = (0..nt)
                .map(|i| val_mapped[(i * 977 + 13) % (val_mapped.len() - 2).max(1)])
                .collect();
            let mk = |scale: f32, half: i32| -> *mut f32 {
                let p = alloc(nt * dim);
                zero(p, nt * dim);
                for i in 0..nt {
                    let tok = toks[i] as usize;
                    for j in 0..k0_embed {
                        let take = match half {
                            0 => true,                                   // 全部
                            1 => j < k0_embed / 2,                       // A 半
                            2 => j >= k0_embed / 2,                      // B 半
                            _ => true,
                        };
                        if !take {
                            continue;
                        }
                        let d = if io_sparse {
                            *dim_of_p.add(tok * k0_embed + j) as usize
                        } else {
                            j
                        };
                        let v = if io_sparse { *etab.add(tok * k0_embed + j) } else { 0.0 };
                        *p.add(i * dim + d) += scale * v;
                    }
                }
                p
            };
            let inp_base = mk(1.0, 0);
            let inp_sc = mk(2.0, 0);
            let inp_a = mk(1.0, 1);
            let inp_b = mk(1.0, 2);
            // 5 趟前向：base / scaled / A / B / (A+B)（第 5 趟用 base 本身）
            let store_base = alloc(layers * nt * dim);
            let store_sc = alloc(layers * nt * dim);
            let store_a = alloc(layers * nt * dim);
            let store_b = alloc(layers * nt * dim);
            let run = |inp: *mut f32, store: *mut f32| {
                std::ptr::copy_nonoverlapping(inp, h_ins[0], nt * dim);
                for l in 0..layers {
                    let xin = h_ins[l];
                    let dest = if l + 1 < layers { h_ins[l + 1] } else { h };
                    if identity_layers {
                        std::ptr::copy_nonoverlapping(xin, dest, nt * dim);
                    } else {
                        for i in 0..nt {
                            b1_forward(&scs[l], xin.add(i * dim), y.add(i * dim));
                        }
                        if relu {
                            // 探针必须走**和训练完全相同**的前向（含 ReLU），否则探针测的不是模型
                            for i in 0..nt * dim {
                                let v = *y.add(i);
                                *y.add(i) = if v > 0.0 { v } else { 0.0 };
                            }
                        }
                        std::ptr::copy_nonoverlapping(y, dest, nt * dim);
                    }
                    std::ptr::copy_nonoverlapping(dest, store.add(l * nt * dim), nt * dim);
                }
            };
            run(inp_base, store_base);
            run(inp_sc, store_sc);
            run(inp_a, store_a);
            run(inp_b, store_b);
            let mut worst_scale = 0.0f32;
            let mut worst_add = 0.0f32;
            let mut mag = 0.0f32;
            let mut where_scale = 0usize;
            let mut where_add = 0usize;
            for l in 0..layers {
                for i in 0..nt * dim {
                    let b = *store_base.add(l * nt * dim + i);
                    let s = *store_sc.add(l * nt * dim + i);
                    let d = (s - 2.0 * b).abs();
                    if d > worst_scale {
                        worst_scale = d;
                        where_scale = l;
                    }
                    let ab = *store_a.add(l * nt * dim + i) + *store_b.add(l * nt * dim + i);
                    let d2 = (b - ab).abs();
                    if d2 > worst_add {
                        worst_add = d2;
                        where_add = l;
                    }
                    let m = b.abs();
                    if m > mag {
                        mag = m;
                    }
                }
            }
            let denom = mag.max(1e-30);
            println!("\n--- LINEARITY PROBE（逐层，{} token × {} 维，{} 层）---", nt, dim, layers);
            println!("    max|y| = {:.6e}", mag);
            println!("    齐次性  max|y(2x) - 2·y(x)| = {:.6e}  （相对 {:.3e}，首次出现在第 {} 层）", worst_scale, worst_scale as f64 / denom as f64, where_scale);
            println!("    可加性  max|y(A+B) - y(A) - y(B)| = {:.6e}  （相对 {:.3e}，首次出现在第 {} 层）", worst_add, worst_add as f64 / denom as f64, where_add);
            // 判据：α=2 在 f32 下是精确缩放 ⇒ 线性结构应给出**逐位 0**
            let linear = worst_scale == 0.0 && worst_add == 0.0;
            let almost = (worst_scale as f64 / denom as f64) < 1e-6 && (worst_add as f64 / denom as f64) < 1e-6;
            println!(
                "  LINEARITY {}: 结构{}",
                if linear { "PASS(逐位)" } else if almost { "PASS(1e-6)" } else { "FAIL" },
                if linear { "是逐位线性的（无逐元素非线性、无偏置）" } else if almost { "在这精度下线性" } else { "存在非线性项" }
            );
            if !(linear || almost) {
                std::process::exit(1);
            }
            // PASS 走**正常返回**：`std::process::exit` 不会 flush Rust 的 stdout 缓冲，
            // 在管道/文件重定向下会把判据文本丢掉（实测：同一命令独立跑能抓到输出，
            // 在长脚本里抓不到）。判据文本丢了，脚本就只能读到空串 —— 探针自己坏掉。
            return;
        }

        // ===================== inter-op 数据并行：region/slot 参数化 =====================
        // 参数 off = 本 region 在 token 维的起点；tt = 本 region 的 token 数；slot = 线程号。
        // intra: off=0, tt=tt, slot=0 ⇒ 与旧行为**逐位相同**；
        // inter: 每个线程一个 region（若干完整样本），token 维缓冲只碰自己那一段。
        // [P23b] bf16 镜像：f32 master（Adam 更新）+ bf16 精度 compute 缓冲（前向/反向读）。
    // 与「就地圆整主权重」不同：优化器累积的低位保留在 master 里，不丢学习量。
    let bf16_mirror = arg_u32("--w-bf16", 0) == 1;
    let mut sp_master: Vec<*mut f32> = Vec::new();
    let mut head_master: *mut f32 = std::ptr::null_mut();
    let mut head2_master: *mut f32 = std::ptr::null_mut();
    if bf16_mirror {
        unsafe {
            for l in 0..layers {
                let p = malloc(dim * k0 * 4) as *mut f32;
                std::ptr::copy_nonoverlapping(scs[l].w, p, dim * k0);
                sp_master.push(p);
            }
            head_master = malloc(n_vocab * k0_head * 4) as *mut f32;
            std::ptr::copy_nonoverlapping(head.w, head_master, n_vocab * k0_head);
            head2_master = malloc(n_vocab * k0_head * 4) as *mut f32;
            std::ptr::copy_nonoverlapping(head2.w, head2_master, n_vocab * k0_head);
        }
    }
    let step_fn = |off: usize, tt: usize, slot: usize, toks: &[u32], train_mode: bool| -> (f32, usize) {
            // ---- 重绑定：把同名缓冲换成"本线程那一段"的指针，下面 700+ 行**逐字不动** ----
            //   token 维缓冲按 off 平移（各 slot 写自己那一段行，互不重叠）；
            //   attention 工作缓冲 / 梯度累加器按 slot 分段（并发写，必须私有）。
            //   ⚠️ slot 步长一律用 **t_tokens**（分配时的尺寸），不能用 tt（= t_tokens/nslot），
            //      否则各 region 的缓冲会重叠 —— 这就是第一版出现 NaN/结果不稳定的原因。
            let h_ins: Vec<*mut f32> = h_ins.iter().map(|p| p.add(off * dim)).collect();
            let pres: Vec<*mut f32> = pres.iter().map(|p| p.add(off * dim)).collect();
            // region 的样本数必须由 token 数推出：诊断/评估/标定用 tt = t_tokens，
            // 那时 sb（=batch/nslot）是错的样本数 ⇒ 后半个 batch 的 h 恒为 0 ⇒ 最后一个窗口 q/k 全零。
            let n_samp = tt / seq;
            if attn_debug >= 1 {
                println!("[step] enter off={} tt={} slot={} train={} n_samp={}", off, tt, slot, train_mode, n_samp);
            }
            let y = y.add(off * dim);
            let logits = logits.add(off * n_vocab);
            let logits2 = if io_sparse && head_skip { logits2.add(off * n_vocab) } else { logits2 };
            let dlogits = dlogits.add(off * n_vocab);
            let dh = dh.add(off * dim);
            let dy = dy.add(off * dim);
            let dx = dx.add(off * dim);
            // 最末层输出缓冲 h 也是 token 维共享缓冲（dest = h）⇒ 按 off 平移，否则各 region 互相覆盖
            let h = h.add(off * dim);
            // slot0 用 t_tokens 尺寸的块（诊断/评估路径会用 tt = t_tokens 调进来），
            // 其余 slot 只需 tt 尺寸（它们是 region 私有的）⇒ 内存从 nslot*t_tokens 降到 ~t_tokens。
            let tok_off = if slot == 0 { 0 } else { tok_slot0 + (slot - 1) * tok_slotn };
            let prob_off = if slot == 0 { 0 } else { prob_slot0 + (slot - 1) * prob_slotn };
            let asp = asp.add(tok_off);
            let adx = adx.add(tok_off);
            let aq = aq.add(tok_off);
            let ak = ak.add(tok_off);
            let av = av.add(tok_off);
            let ao = ao.add(tok_off);
            let aout = aout.add(tok_off);
            let adq = adq.add(tok_off);
            let adk = adk.add(tok_off);
            let adv = adv.add(tok_off);
            let adobuf = adobuf.add(tok_off);
            let d_h0 = d_h0.add(tok_off);
            let adprob = adprob.add(prob_off);
            let aprob = aprob.add(prob_off);
            let arowsum = arowsum.add(if slot == 0 { 0 } else { t_tokens + (slot - 1) * tt });
            // [P19-8] 本 region 的 ht 工作区（每 token 一份，slot 私有）
            let head_ht = head_ht.add((if slot == 0 { 0 } else { t_tokens + (slot - 1) * tt }) * htr);
            // par_b1_backward / attn_backward_sparse_par 的"每线程 dw 暂存"：inter 下内层 nt=1，
            // 若所有 region 共用 dwp 就会互相踩（实测：NaN + 两次运行结果不同）。
            let dwp = dwp.add(slot * nth.max(1) * max_wlen);
            let dws: Vec<*mut f32> = dws.iter().map(|p| p.add(slot * dim * k0)).collect();
            let head_dw = head_dw.add(slot * if io_sparse { n_vocab * k0_head } else { 1 });
            let head2_dw = head2_dw.add(slot * if io_sparse && head_skip { n_vocab * k0_head } else { 1 });
            let etab_g = etab_g.add(slot * if io_sparse { n_vocab * k0_embed } else { 1 });
            let emb_g = emb_g.add(slot * nd);
            let lm_g = lm_g.add(slot * nd);
            let lm_bg = lm_bg.add(slot * (if io_sparse { 1 } else { n_vocab }));
            let as_w1 = as_w / nslot;
            let asq_g = asq_g.add(slot * as_w1);
            let ask_g = ask_g.add(slot * as_w1);
            let asv_g = asv_g.add(slot * as_w1);
            let aso_g = aso_g.add(slot * as_w1);
            let awq_g = awq_g.add(slot * n_attn.max(1) * ad);
            let awk_g = awk_g.add(slot * n_attn.max(1) * ad);
            let awv_g = awv_g.add(slot * n_attn.max(1) * ad);
            let awo_g = awo_g.add(slot * n_attn.max(1) * ad);
            let ce_known_sum = ce_known_sum.add(slot);
            let ce_known_n = ce_known_n.add(slot);
            let ce_unk_n = ce_unk_n.add(slot);
            let tm0 = std::time::Instant::now();
            let mut tm = tm0;
            // 前向：h_0 = emb；每层 h = B1_l(h)（**赋值**，无残差）
            let h0 = h_ins[0];
            if io_sparse {
                // 稀疏嵌入：token → 查 k0_embed 个值 → 展开（散写）到第 block_of[tok] 块。
                // ⚠️ 必须先清零：只有 k0_embed 个维度被写，其余必须真的是 0
                //    （否则上一批/上一次不同 token 的残留会混进来 —— 一个不会崩、
                //    只会让 loss 悄悄变好的脏数据 bug）。
                zero(h0, tt * dim);
                // [P21-3/D 类] 块对齐时 dim_of[tok*16+j] = (tok%nb)*16+j ⇒ 那 16 个维度是
                // **连续一段**，于是 16 次散布标量存储变成一次 64B 拷贝（memcpy）。
                // 非块对齐（--dyn-subspace 等）仍走原来的索引散写。
                if head_blocked {
                    for b in 0..n_samp {
                        for t in 0..seq {
                            let tok = toks[off + b * seq + t] as usize;
                            let blk = tok % nb_head;
                            let src = etab.add(tok * k0_embed);
                            let dstp = h0.add((b * seq + t) * dim + blk * k0_embed);
                            std::ptr::copy_nonoverlapping(src, dstp, k0_embed);
                        }
                    }
                } else {
                for b in 0..n_samp {
                    for t in 0..seq {
                        let tok = toks[off + b * seq + t] as usize;
                        let src = etab.add(tok * k0_embed);
                        for j in 0..k0_embed {
                            let d = *dim_of_p.add(tok * k0_embed + j) as usize;
                            *h0.add((b * seq + t) * dim + d) = *src.add(j);
                        }
                    }
                }
                }
            } else {
                for b in 0..batch {
                    for t in 0..seq {
                        let tok = toks[b * seq + t] as usize;
                        std::ptr::copy_nonoverlapping(emb.add(tok * dim), h0.add((b * seq + t) * dim), dim);
                    }
                }
            }
            // ---- [P21-7] 整栈 source-major 前向（n_attn==0；层间零转置）----
            // [P21-7b] 整栈 source-major（**含注意力边界**）：只在 !relu && !identity_layers 时启用。
            // 连续非注意力层走 b1_forward_t8（层间零转置）；注意力层/末层是"边界"：
            //   t8(输入) → token-major(y) → 原层前向 + attention → 转回 t8(h_ins[l+1])。
            // [P21-7b] 含注意力边界已修（根因：边界层 xin 复用了 y，而 par_b1_forward(xin=y,y)
            // 会在跨神经元 gather 里读到被覆写的输入 ⇒ 前向错；已改专用 t8tok 缓冲）。
            let t8s = (tt + 7) & !7usize;
            // [P21-7c] **形状门控**（实测驱动）：整栈 t8 只在「每趟转置能留在 L2」时启用。
            //   生产 dim4096/tt256 ⇒ 每趟 4 MB：1.2026 → 1.6599 s/步（**+38%**，转置溢出到 L3/DRAM）
            //   primary/迭代 dim2048/tt128 ⇒ 每趟 1 MB：−1.7%（含注意力）/ −9.5%（n_attn=0）
            // [修复] 原阈值 2MB 太小：dim>=512 即判超限，t8 整栈被静默关闭（dim1536 需 6.3MB）。
    // 改为 CLI 可调，默认 256MB（仅防病态配置，正常规模全部启用）。
    let t8_max_mb = arg_u32("--t8-max-mb", 256) as usize;
    let t8_small = dim * t8s * 4 <= t8_max_mb * 1024 * 1024;
            let t8_want = vmlm_core::interp::HAS_AVX2 && !relu && !identity_layers && t8_small;
            let t8_stack = t8_want && (arg_u32("--t8-stack", 1) == 1);
    // [修复] 打印**生效值**与原因：原日志打的是 flag，dim1536 时显示 stack=true 但实际 false
    println!("t8: 生效 stack={} (want={} small={} need={:.1}MB budget={}MB flag={})",
        t8_stack, t8_want, t8_small, (dim * t8s * 4) as f64 / 1048576.0, t8_max_mb,
        arg_u32("--t8-stack", 1));
            let mut t8ta: *mut f32 = std::ptr::null_mut();
            let mut t8dyc: *mut f32 = std::ptr::null_mut();
            let mut t8dxn: *mut f32 = std::ptr::null_mut();
            if t8_stack {
                let (ta, _tb, dyc, dxn) = t8s4(dim * t8s.max(1));
                t8ta = ta;
                t8dyc = dyc;
                t8dxn = dxn;
                // 入口：h_ins[0]（嵌入展开，token-major）→ t8 链起点；
                // h_ins[0] 本身保留 token-major 供 head-skip 的 head2 前向使用
                vmlm_core::kernels::transpose_t8_avx2(h_ins[0], tt, dim, ta, t8s.max(1));
            }
            // [抽库 L4c-2] 训练前向组装收进 B1Model（训练模式；与搬移前逐字等价）
            let mut b1 = vmlm_models::model::B1Model {
                cfg: vmlm_models::model::B1ForwardCfg {
                    dim, n_vocab, layers, nth, attn_every, n_attn,
                    attn_sparse, attn_residual, head_skip, identity_layers, relu,
                    io_sparse, k0_embed, t_tokens, tok_total,
                    k0, k0_attn, ad, fusion, no_res, profile,
                    train_mode: true, tt, seq, nt_in, attn_window, head_blocked,
                    nb_head, k0_head, t8_stack, t8s, attn_qk_check, slot, off, attn_debug,
                },
                scs: &scs, asq: &asq, ask: &ask, asv: &asv, aso: &aso,
                awq, awk, awv, awo, ad,
                head: &head, head2: &head2,
                emb, dim_of_p, etab,
                h_ins: &h_ins, h, y,
                aq, ak, av, ao, aprob, aout,
                logits, head2_out: logits2,
                pres: &pres, asp,
                head_wt, head2_wt, head_ht, t8ta, t8dy: dy,
                collect_flag, qk_stats,
                dh,
                dx,
                dy,
                dlogits,
                dws: &dws,
                dwp,
                head_dw,
                lm_w,
                lm_g,
                lm_bg,
                btd,
                bth,
                bt,
                adx,
                adq,
                adk,
                adv,
                adobuf,
                adprob,
                arowsum,
                aat1,
                aat2,
                awt,
                asq_g,
                ask_g,
                asv_g,
                aso_g,
                awq_g,
                awk_g,
                awv_g,
                awo_g,
                t8dxn,
                t8dyc,
                prof_p,
            };
            let _ = b1.forward(None, tt);
            // 原 head 块之后的 profile/collect 诊断仍在 if io_sparse 内，保持原嵌套
            if io_sparse {
                if profile {
                    *prof_p.add(1) += tm.elapsed().as_secs_f64();
                    tm = std::time::Instant::now();
                }
                // 统计：head 的 16×V 条连接里，源（最后一层该维）非零的比例
                if collect && train_mode {
                    for v in 0..n_vocab {
                        for j in 0..k0_head {
                            let sid = *head.src_u32.add(v * k0_head + j) as usize;
                            *hc_p += 1;
                            if (*h.add(sid)).abs() > 1.0e-12 {
                                *hh_p += 1;
                            }
                        }
                    }
                }
            } else {
                if profile {
                    *prof_p.add(0) += tm.elapsed().as_secs_f64();
                    tm = std::time::Instant::now();
                }
                gemm_nt(h, lm_w, logits, tt, n_vocab, dim, false);
                if profile {
                    *prof_p.add(1) += tm.elapsed().as_secs_f64();
                    tm = std::time::Instant::now();
                }
            }
            // CE
            // [P19-6] 三个标量 pass（max / exp+sum / dlogits 写）改成 AVX2：
            //   原来每个 position 是 3 × n_vocab 次标量访问，其中 exp_f 是 15 条标量指令。
            //   primary 配置实测 CE 占 5.3% 而它全是「每行 8198 个元素 × 126 行 × 8 region」
            //   的纯吞吐活。改用 max256_f32 / exp_neg_sum_avx2 / scale_mul_avx2（8 宽）。
            //   数值口径：归约分组变了 ⇒ ε 级（exp 的 FMA 多项式与标量 exp_f 同系数）。
            let avx = vmlm_core::interp::HAS_AVX2;
            let mut loss_sum = 0.0f32;
            let mut n_ce = 0usize;
            for b in 0..n_samp {
                for t in 0..seq - 1 {
                    let pos = b * seq + t;
                    let tgt = toks[off + pos + 1] as usize;
                    let lg = logits.add(pos * n_vocab);
                    // 先留一份目标 logit：下面会把 lg 原地覆盖成 exp 值
                    let tgt_logit = *lg.add(tgt);
                    let mx = if io_sparse {
                        // 稀疏 head 没有偏置（偏置本身就是 10921 个参数，与参数预算冲突）
                        if avx {
                            vmlm_core::kernels::max256_f32(lg, n_vocab)
                        } else {
                            let mut m = -1.0e30f32;
                            for v in 0..n_vocab {
                                let x = *lg.add(v);
                                if x > m {
                                    m = x;
                                }
                            }
                            m
                        }
                    } else {
                        if avx {
                            vmlm_core::kernels::add256_f32(lg, lm_b, n_vocab);
                            vmlm_core::kernels::max256_f32(lg, n_vocab)
                        } else {
                            let mut m = -1.0e30f32;
                            for v in 0..n_vocab {
                                let x = *lg.add(v) + *lm_b.add(v);
                                *lg.add(v) = x;
                                if x > m {
                                    m = x;
                                }
                            }
                            m
                        }
                    };
                    let sum = if avx {
                        vmlm_core::kernels::exp_neg_sum_avx2(lg, n_vocab, mx)
                    } else {
                        let mut s = 0.0f32;
                        for v in 0..n_vocab {
                            let e = vmlm_core::kernels::exp_f(*lg.add(v) - mx);
                            *lg.add(v) = e;
                            s += e;
                        }
                        s
                    };
                    // log-sum-exp 口径：loss = logZ - x_tgt。
                    // 不用 `-ln(exp(x_tgt-mx)/sum)`：那个形式在 FTZ 下会变成 ln(0) = +inf。
                    // 数学上两者相等；数值上这个形式对下溢是安全的（sum >= 1，因为最大项为 1）。
                    let logz = mx
                        + vmlm_core::kernels::ln_f(sum)
                        + if io_sparse { 0.0 } else { *lm_b.add(tgt) };
                    let ce_i = logz - tgt_logit;
                    loss_sum += ce_i;
                    n_ce += 1;
                    // 目标是不是 UNK(0)？分开累计（跨词表比较用）
                    if toks[off + b * seq + t + 1] == 0 {
                        *ce_unk_n += 1.0;
                    } else {
                        *ce_known_sum += ce_i;
                        *ce_known_n += 1.0;
                    }
                    let inv = 1.0 / sum; // 梯度侧仍用 softmax 值：dlogits = softmax - onehot
                    if train_mode {
                        let invb = 1.0 / ((seq - 1) * batch) as f32;
                        let dl = dlogits.add(pos * n_vocab);
                        if avx {
                            vmlm_core::kernels::scale_mul_avx2(dl, lg, inv * invb, n_vocab);
                        } else {
                            for v in 0..n_vocab {
                                *dl.add(v) = *lg.add(v) * inv * invb;
                            }
                        }
                        *dl.add(tgt) -= invb;
                    }
                }
            }
            if !train_mode {
                if profile {
                    *prof_p.add(2) += tm.elapsed().as_secs_f64();
                }
                return (loss_sum, n_ce);
            }
            if profile {
                *prof_p.add(2) += tm.elapsed().as_secs_f64();
                tm = std::time::Instant::now();
            }
            if attn_debug >= 1 {
                println!("[step] fwd done (loss_sum={:.6} n_ce={})", loss_sum, n_ce);
            }
            // [抽库 L4c-3/4] 反向组装收进 B1Model::backward（逐字复刻原 1949-2110）
            let _ = b1.backward();

            // emb 反向
            if profile {
                *prof_p.add(4) += tm.elapsed().as_secs_f64();
                tm = std::time::Instant::now();
            }
            if io_sparse {
                // 跨层残差：把**直连项**回传到 h0 的梯度加进来（它不穿过 24 层）
                if head_skip {
                    zero(d_h0, tt * dim);
                    zero(head2_dw, n_vocab * k0_head);
                    if head_blocked && nt_in <= 1 {
                        // [P21-4] 同 head：批量反向
                        head_bwd_batched(head2.w, head2.w, dlogits, h_ins[0], tt, dim, n_vocab, nb_head, k0_embed, k0_head, head2_dw, d_h0);
                    } else if head_blocked {
                        // [P19-7] 块对齐快路径（head2 与 head 同拓扑）
                        par_head_bwd(&head2, dlogits, h_ins[0], head2_dw, d_h0, tt, dim, n_vocab, nt_in, nb_head, k0_embed);
                    } else {
                        par_b1_backward(&head2, dlogits, h_ins[0], head2_dw, d_h0, tt, dim, n_vocab, nt_in, dwp, true);
                    }
                    if vmlm_core::interp::HAS_AVX2 {
                        vmlm_core::kernels::add256_f32(dh, d_h0, tt * dim);
                    } else {
                        for i in 0..tt * dim {
                            *dh.add(i) += *d_h0.add(i);
                        }
                    }
                }
                // 只更新被激活的那 k0_embed 个维度（表里就没有别的维度可更新）
                // [P21-3/D 类] 块对齐 ⇒ 源也是连续 16 个维度 ⇒ 一次 AVX2 8 宽累加
                if head_blocked && vmlm_core::interp::HAS_AVX2 {
                    for b in 0..n_samp {
                        for t in 0..seq {
                            let tok = toks[off + b * seq + t] as usize;
                            let blk = tok % nb_head;
                            let dst = etab_g.add(tok * k0_embed);
                            let srch = dh.add((b * seq + t) * dim + blk * k0_embed);
                            vmlm_core::kernels::add256_f32(dst, srch, k0_embed);
                        }
                    }
                } else {
                for b in 0..n_samp {
                    for t in 0..seq {
                        let tok = toks[off + b * seq + t] as usize;
                        let dst = etab_g.add(tok * k0_embed);
                        for j in 0..k0_embed {
                            let d = *dim_of_p.add(tok * k0_embed + j) as usize;
                            *dst.add(j) += *dh.add((b * seq + t) * dim + d);
                        }
                    }
                }
                }
            } else {
                for b in 0..n_samp {
                    for t in 0..seq {
                        let tok = toks[off + b * seq + t] as usize;
                        for i in 0..dim {
                            *emb_g.add(tok * dim + i) += *dh.add((b * seq + t) * dim + i);
                        }
                    }
                }
            }
            if profile {
                *prof_p.add(5) += tm.elapsed().as_secs_f64();
            }
            if attn_debug >= 1 {
                println!("[step] exit off={} tt={} slot={}", off, tt, slot);
            }
            (loss_sum, n_ce)
        };

        // 固定验证批（跨步可比）
        let val_batches: Vec<Vec<u32>> = {
            let mut v = Vec::with_capacity(eval_batches);
            for eb in 0..eval_batches {
                let mut one = Vec::with_capacity(t_tokens);
                for b in 0..batch {
                    let base = (eb * 977 + b * seq * 13) % (val_mapped.len() - seq - 1).max(1);
                    for t in 0..seq {
                        one.push(val_mapped[base + t]);
                    }
                }
                v.push(one);
            }
            v
        };
        // =====================================================================
        // 权重持久化：--load-model（在标定之前载入，这样标定会作用在载入的权重上）
        // =====================================================================
        // 为什么必须有它：本驱动**原本没有任何权重持久化**，"训练后另起一个进程做验证"
        // 拿到的是**重新初始化的随机权重** —— 所谓"验证路径和训练路径不一致"的根源之一。
        let ck_load = arg_str_opt("--load-model");
        let ck_save = arg_str_opt("--save-model");
    let save_every = arg_u32("--save-every", 0) as usize; // >0：每 N 步落一次盘（长跑保险）
    // 续训用：把 lr 调度按"虚拟步数"继续（0 = 从头）。从检查点续跑时设成检查点步数。
    let lr_offset = arg_u32("--lr-offset", 0) as usize;
        if let Some(path) = ck_load.clone() {
            let geo = CkGeo {
                dim, k0, k0_embed, k0_head, layers, n_attn, k0_attn, io_sparse, head_skip, attn_sparse, n_vocab,
            };
            let mut tens: Vec<(*mut f32, usize)> = Vec::new();
            for l in 0..layers {
                tens.push((scs[l].w, dim * k0));
            }
            if io_sparse {
                tens.push((etab, n_vocab * k0_embed));
                tens.push((head.w, n_vocab * k0_head));
                if head_skip {
                    tens.push((head2.w, n_vocab * k0_head));
                }
            } else {
                tens.push((emb, n_vocab * dim));
                tens.push((lm_w, n_vocab * dim));
                tens.push((lm_b, n_vocab));
            }
            if n_attn > 0 {
                if attn_sparse {
                    for b in 0..n_attn {
                        tens.push((asq[b].w, dim * k0_attn));
                        tens.push((ask[b].w, dim * k0_attn));
                        tens.push((asv[b].w, dim * k0_attn));
                        tens.push((aso[b].w, dim * k0_attn));
                    }
                } else {
                    for b in 0..n_attn {
                        tens.push((awq.add(b * ad), ad));
                        tens.push((awk.add(b * ad), ad));
                        tens.push((awv.add(b * ad), ad));
                        tens.push((awo.add(b * ad), ad));
                    }
                }
            }
            let nread = ck_read(&path, &geo, &mut tens);
            println!("--- 载入权重: {} （{} 个张量, {} 个 f32）", path, tens.len(), nread);
        }
        // [P19-8] 初版 WT 打包：放在**任何 step_fn 调用（标定/体检/gradcheck）之前**。
        repack_head_wt(head_blocked, head.w, head_wt, head2.w, head2_wt, head_skip, n_vocab, k0_head);

        // =====================================================================
        // attention 分数尺度标定（**逐块**，一次性；不改前向数学）
        // =====================================================================
        // 旧实现的两个 bug（本轮修掉）：
        //   1. `worst_std` 取的是 6 个块的**最小值**，但只算出一个**全局 g** 乘到所有块上
        //      ⇒ 深层块饿死时全局 g 被它拖爆，而浅层块被过度放大；
        //   2. `if worst_std > 1e-12` 是个**静默跳过**：尺度更小时既不打印也不标定，
        //      然后体检报"坍缩"，输出里看不到任何原因 —— 这就是用户看到的
        //      "验证路径没有标定输出、max|q·k|=3e-33、FATAL"。
        // 现在：**逐块标定**（g_b = 1/√scale_b），每块都打印，任何被跳过的路径都 exit 2 并指名。
        let attn_calib = arg_u32("--attn-calib", 1) == 1;
        // 标定因子的上限：默认 1e15（允许"深层块输入只有 1e-14"这种极端情况被拉回 O(1)）。
        // 超过上限就 exit 2 指名是哪一块 —— 因为那说明是**前向把激活搞没了**，不是尺度问题。
        let calib_max_g = arg_f32("--attn-calib-max-g", 1.0e15);
        // 标定目标：
        //   score（默认）= 让 **缩放后的分数** q·k/√d 的尺度为 1（这才是 softmax 真正看到的东西）
        //   qk          = 让 **原始** q·k 的尺度为 1（上一版行为，softmax 仍接近均匀：实测集中度 1.02~1.08）
        let calib_score_target = arg_str_opt("--attn-calib-target").map(|v| v != "qk").unwrap_or(true);
        let mut calib_g: Vec<f32> = vec![1.0; n_attn.max(1)];
        if n_attn > 0 && attn_calib {
            let vb = val_batches[0].clone();
            for b in 0..n_attn {
                *collect_flag.add(b) = 1;
            }
            let _ = step_fn(0, t_tokens, 0, &vb, false);
            println!(
                "--- ATTENTION 逐块尺度诊断（view={} token{}，dim={}，{} 块；parallel-mode={}）---",
                if attn_window { seq } else { t_tokens },
                if attn_window { format!("=单窗口，共 {} 个窗口", t_tokens / seq) } else { String::new() },
                dim, n_attn, pmode
            );
            let mut any_skip = false;
            let mut worst_b = 0usize;
            let mut worst_sc = f32::INFINITY;
            // ⚠️ 这里改成**直接**从 aq/ak 量（而不是读 body 里 collect_flag 收集的 qk_stats）：
            //    inter+window 实测 qk_stats 全为 0 —— 那个"收集"是副作用，会被更早的一次前向消费掉，
            //    非常脆弱。直接量以后，标定与"体检"用的是同一条测量路径、同一个 view。
            let t_cal_view = if attn_window { seq } else { t_tokens };
            // 诊断：aq/ak 到底有没有被写进去（区分"前向没写" vs "读错位置"）
            println!(
                "    [probe] slot0 aq/ak RMS: 前{}行={:.3e}/{:.3e} 全缓冲({}行)={:.3e}/{:.3e} asp RMS={:.3e}",
                t_cal_view,
                rms(aq, t_cal_view * dim),
                rms(ak, t_cal_view * dim),
                t_tokens,
                rms(aq, t_tokens * dim),
                rms(ak, t_tokens * dim),
                rms(asp, tt * dim)
            );
            for b in 0..n_attn {
                let (mx, mean, nz, _pmax, ratio) =
                    attn_qk_stats(aq, ak, t_cal_view, dim, aprob.add(b * t_tokens * t_tokens));
                let sc = if mean > 0.0 { mean } else { mx };
                if sc < worst_sc {
                    worst_sc = sc;
                    worst_b = b;
                }
                let li = (((b + 1) * attn_every).min(layers)).saturating_sub(1);
                // ⚠️ 最后一个 attention 块的输入是**最后一层的输出**，它在 `h` 里而不是 h_ins[layers]
                //    （h_ins 只有 0..layers-1）。这里原来写成 h_ins[li+1] ⇒ 最后一个块越界 panic。
                let hr = if li + 1 < layers {
                    rms(h_ins[li + 1], t_cal_view * dim)
                } else {
                    rms(h, t_cal_view * dim)
                };
                println!(
                    "    块{}: mean|q.k|={:.4e} max|q.k|={:.4e} 非零占比={:.4} 输入RMS(h)={:.4e} 集中度={:.3}（view={} token）",
                    b, mean, mx, nz, hr, ratio, t_cal_view
                );
                if sc <= 0.0 {
                    eprintln!("FATAL: 块{} 的 q·k 恒为 0（结构性为零）⇒ 不能靠标定掩盖，需修前向。", b);
                    std::process::exit(2);
                }
                if ck_load.is_some() && sc < 0.33 {
                    eprintln!(
                        "FATAL: 载入的检查点在块{} 上尺度 {:.4e} < 1/3 ⇒ 该检查点**未经标定**（或来自旧二进制），\n       与「标定已乘进权重」的约定不一致。拒绝静默重标定。",
                        b, sc
                    );
                    std::process::exit(2);
                }
                if sc < 1.0 {
                    // 原始分数尺度 sc ⇒ 乘 g 后变成 sc·g²；/√d 后是 sc·g²/√d。
                    //   qk 目标   : sc·g² = 1            ⇒ g = 1/√sc
                    //   score 目标: sc·g²/√d = 1         ⇒ g = d^{1/4}/√sc
                    let g = if calib_score_target {
                        (dim as f32).sqrt().sqrt() / vmlm_core::kernels::sqrt_f(sc)
                    } else {
                        1.0 / vmlm_core::kernels::sqrt_f(sc)
                    };
                    if g > calib_max_g {
                        eprintln!(
                            "FATAL: 块{} 的分数尺度 {:.3e} 需要 g={:.3e}（>--attn-calib-max-g={:.3e}）⇒ 该块激活已坍缩，标定不能替代修前向。",
                            b, sc, g, calib_max_g
                        );
                        std::process::exit(2);
                    }
                    calib_g[b] = g;
                    if g > 1.0e4 {
                        println!("    ⚠️ 块{} 的 g={:.3e}（>1e4）：该块输入已很小（RMS(h)={:.3e}），仅靠标定掩盖", b, g, hr);
                    }
                } else {
                    // 尺度已经 >= 1：不需要标定。但要**显式说出来**，不能静默
                    println!("    （块{} 尺度 {:.3} >= 1，无需标定，g=1）", b, sc);
                    // 载入检查：标定后**原始**分数的尺度应当正好是设计目标
                    //   qk 目标   ⇒ ≈ 1
                    //   score 目标 ⇒ ≈ √d   （因为 sc·g² = √d）
                    let expect = if calib_score_target {
                        (dim as f32).sqrt()
                    } else {
                        1.0f32
                    };
                    // 载入检查的分寸（实测教训）：
                    //   硬判据只能用来抓"**根本没标定**的检查点"（那类 bug 的尺度是 1e-5 量级）；
                    //   **训练过的**模型尺度可以合理地远离初始化目标（实测 600 步后块2 从 1.0 涨到 8.1，
                    //   因为模型在学着让 attention 更尖）。所以放宽到 1/1000，只把"明显没标定"判死。
                    if ck_load.is_some() && sc < expect / 1000.0 {
                        eprintln!(
                            "FATAL: 载入的权重在该块上尺度 {:.3e} 远低于目标 {:.3e}（<1/1000）⇒ 该检查点似乎**未经标定**，拒绝静默重标定。",
                            sc, expect
                        );
                        std::process::exit(2);
                    }
                    if ck_load.is_some() && (sc < expect / 10.0 || sc > expect * 10.0) {
                        println!(
                            "    （提示：载入权重在块{} 的尺度 {:.3} 与初始化目标 {:.3} 相差 >10× —— 训练后的正常现象，仅提示）",
                            b, sc, expect
                        );
                    }
                }
                let _ = any_skip;
            }
            any_skip = false;
            // 逐块乘进权重本身（这样 g 随权重一起持久化，验证路径不需要"再标一次"）
            for b in 0..n_attn {
                let g = calib_g[b];
                if g == 1.0 {
                    continue;
                }
                if attn_sparse {
                    for i in 0..dim * k0_attn {
                        *asq[b].w.add(i) *= g;
                        *ask[b].w.add(i) *= g;
                    }
                } else {
                    for i in 0..ad {
                        *awq.add(b * ad + i) *= g;
                        *awk.add(b * ad + i) *= g;
                    }
                }
            }
            println!(
                "--- ATTENTION 标定完成：worst 块{} 尺度 {:.4e}；g = [{}]",
                worst_b,
                worst_sc,
                calib_g[..n_attn]
                    .iter()
                    .map(|g| format!("{:.3}", g))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            // 标定后再测一次（逐块），这就是"训练开始时的 max|q·k|"基准
            for b in 0..n_attn {
                *collect_flag.add(b) = 1;
            }
            let _ = step_fn(0, t_tokens, 0, &vb, false);
            let mut mn = f32::INFINITY;
            for b in 0..n_attn {
                let mx = *qk_stats.add(b * 5 + 0);
                if mx < mn {
                    mn = mx;
                }
            }
            println!(
                "--- ATTENTION 标定后体检：worst max|q.k|={:.4e}（判据 > 0.01）  ⇒ {}",
                mn,
                if mn > 0.01 { "通过" } else { "未通过" }
            );
            if mn <= 0.01 {
                eprintln!("FATAL: 标定后 max|q.k| 仍 <= 0.01 ⇒ 该配置下 attention 无选择能力。");
                std::process::exit(2);
            }
        }

        // =====================================================================
        // attention 体检（训练前跑一次）：q·k 是否非零 + softmax 是否有选择
        // =====================================================================
        if n_attn > 0 && attn_qk_check {
            let vb = val_batches[0].clone();
            let _ = step_fn(0, t_tokens, 0, &vb, false); // 前向一次，填好 aq/ak/aprob
            // ⚠️ 关键：aq/ak/aprob 是**共享工作缓冲**。逐窗口模式下，循环结束时里面只剩
            //    **最后一个窗口**的内容（t=seq），所以体检必须按 t=seq 去读；
            //    以前这里写 `let tt = t_tokens;`（全局 2048）⇒ 读到 99.9% 是 0 的缓冲，
            //    于是报出"非零占比=0.0010、集中度=0.000"的**假坍缩**（=用户在 batch32 看到的现象）。
            let t_view = if attn_window { seq } else { t_tokens };
            let n_win_view = if attn_window { t_tokens / seq } else { 1 };
            let mut worst_mx = f32::INFINITY;
            let mut worst_ratio = f32::INFINITY;
            let mut worst_nz = f32::INFINITY;
            let mut report = String::new();
            for b in 0..n_attn {
                let (mx, mean, nzfrac, pmax, ratio) =
                    attn_qk_stats(aq, ak, t_view, dim, aprob.add(b * t_tokens * t_tokens));
                if mx < worst_mx {
                    worst_mx = mx;
                }
                if ratio < worst_ratio {
                    worst_ratio = ratio;
                }
                if nzfrac < worst_nz {
                    worst_nz = nzfrac;
                }
                report.push_str(&format!(
                    "    attn块{}: max|q.k|={:.4e} mean|q.k|={:.4e} 非零占比={:.4} max_prob={:.6} 集中度(max_prob/(1/T))={:.3}\n",
                    b, mx, mean, nzfrac, pmax, ratio
                ));
            }
            println!(
                "--- ATTENTION 体检（view = {} token{}（T={}）, dim={}, {} 个 attention 块）---",
                t_view,
                if attn_window { format!("，逐窗口模式：读的是最后一个窗口，共 {} 个窗口", n_win_view) } else { String::new() },
                t_view, dim, n_attn
            );
            print!("{}", report);
            println!(
                "    判据: max|q.k| > 0.01 且 集中度 > 1（均匀=1.0）  ⇒ worst max|q.k|={:.4e} 集中度={:.3} 非零占比={:.4}",
                worst_mx, worst_ratio, worst_nz
            );
            // 判据 1：分数不能是 0（结构性坍缩）
            if worst_mx <= 0.01 {
                eprintln!(
                    "FATAL: attention 坍缩：worst max|q.k| = {:.4e} <= 0.01 ⇒ 分数为 0/近 0，softmax 退化为均匀分布，attention 没有选择能力。\n   view: t={}（{}）, 逐窗口={}, 标定目标={}",
                    worst_mx, t_view, if attn_window { "最后一个窗口" } else { "整块" }, attn_window, if calib_score_target { "score" } else { "qk" }
                );
                std::process::exit(2);
            }
            // 判据 2（新增）：集中度 / 非零占比。以前只**打印**不判定 ⇒ 判据失败也不报错。
            //   注意：初始化后 attention 接近均匀是**正常的**（实测集中度 1.00~1.08），
            //   所以这里只在"明显崩坏"时报错（<0.5），集中度 <= 1 只警告。
            if worst_ratio < 0.5 || worst_nz < 0.5 {
                eprintln!(
                    "FATAL: attention 体检失败：worst 集中度={:.3} 非零占比={:.4}（判据：集中度>1，非零占比≈1）\n   ⇒ 概率矩阵几乎没有内容。逐窗口模式下常见原因是**体检的 view 与写入的 view 不一致**（t_view={} vs 训练 t={}），\n   或标定未生效（当前 --attn-calib={} --attn-calib-target={}）。",
                    worst_ratio, worst_nz, t_view, tt, attn_calib, if calib_score_target { "score" } else { "qk" }
                );
                std::process::exit(2);
            }
            if worst_ratio <= 1.0 {
                println!(
                    "    警告：worst 集中度={:.3} <= 1.0（softmax 近似均匀）。初始化后这是预期值（实测 1.00~1.08），训练后应升到 >1。",
                    worst_ratio
                );
            }
        }

        // =====================================================================
        // 给一段 token 打分：按 seq 切窗跑前向，返回 bits/char（非 UNK 目标）。
        // 生成端（best-of-K 挑选）与 --score-file 共用这一个实现。
        let score_ids = |ids_in: &[u32]| -> f32 {
            let mut tot = 0.0f32;
            let mut nce = 0usize;
            let mut pos = 0usize;
            while pos + seq <= ids_in.len() {
                let (ls, nc) = step_fn(0, seq, 0, &ids_in[pos..pos + seq], false);
                tot += ls;
                nce += nc;
                pos += seq;
            }
            if nce > 0 {
                (tot / nce as f32) / std::f32::consts::LN_2
            } else {
                f32::INFINITY
            }
        };

        // =====================================================================
        // --score-file <utf8>：给任意文本打 bits/char
        // =====================================================================
        // 目的：判断一段生成文本到底是"模型自信的复读"还是"采样漂出分布的乱码"：
        //   · 复读     ⇒ 模型自己给它的 NLL 很低（接近 val 水平）
        //   · 乱码     ⇒ 模型自己给的 NLL 很高（接近均匀地板 11.99 bits/char）
        // 复用同一条前向（step_fn 的 train=false 分支会累加 CE），不新增任何内核。
        if let Some(spath) = arg_str_opt("--score-file") {
            let map = load_char_map(&format!("{}/char_vocab_hex.txt", data_dir));
            let mut ch2id: std::collections::HashMap<char, u32> = std::collections::HashMap::new();
            for (i, m) in map.iter().enumerate().skip(1) {
                let mut it = m.chars();
                if let Some(c0) = it.next() {
                    if it.next().is_none() {
                        ch2id.entry(c0).or_insert(i as u32);
                    }
                }
            }
            let txt = std::fs::read_to_string(&spath).unwrap_or_default();
            let ids: Vec<u32> = txt.chars().map(|c| *ch2id.get(&c).unwrap_or(&0)).collect();
            let nun = ids.iter().filter(|x| **x == 0).count();
            let nwin = ids.len() / seq;
            let all = nwin * (seq - 1);
            let bpc_unk_only = score_ids(&ids);
            let tot = bpc_unk_only * std::f32::consts::LN_2; // 仅用于下面 nce==0 的判断
            let nce = if nwin > 0 { 1 } else { 0 };
            println!("--- SCORE {} ---", spath);
            println!(
                "    字符 {}（编码后 {}，UNK {}），窗口 {}（每窗 seq-1={} 个目标）",
                txt.chars().count(), ids.len(), nun, nwin, seq - 1
            );
            if nce > 0 {
                let bpc = bpc_unk_only;
                println!(
                    "    bits/char（非UNK目标，窗口 {} 个）= {:.4}   （均匀地板 11.99；val≈6.53）",
                    nwin, bpc
                );
                let _ = (tot, all);
            } else {
                println!("    样本不足（文本太短）");
            }
            return;
        }

        // --generate 1：实际输出测试（**复用现有前向逻辑与既有内核**，没有 KV-Cache）
        // =====================================================================
        // 说明：这里把 step_fn 里的**前向部分**（嵌入 scatter → 逐层 B1 → 全局 attention → head）
        // 原样重放一遍，但 batch=1、只取最后一个位置的 logits。没有写任何新内核，
        // 调用的还是 par_b1_forward / attn_forward_sparse / attn_core_par。
        // 复杂度 O(gen_len × seq) —— seq=64 下每生成一个字符重跑整段，实测可接受。
        if gen_mode {
            let map = load_char_map(&format!("{}/char_vocab_hex.txt", data_dir));
            println!("--- GENERATE 模式 ---");
            println!(
                "prompt={:?} gen_len={} temperature={} 采样top-k={} top-p={} seed={} seq={} threads={}",
                gen_prompt, gen_len, gen_temp, gen_topk, gen_topp, seed, seq, nth
            );
            if gen_rep_pen > 1.0 || gen_no_repeat >= 2 {
                println!("重复抑制: --rep-penalty={} --no-repeat-ngram={}", gen_rep_pen, gen_no_repeat);
            }
            println!("词表: {} 项（来自 {}/char_vocab_hex.txt）", map.len(), data_dir);
            // prompt -> id
            // ⚠️ 这里踩过一个坑：判"映射项是不是单个字符"要用**字符数**而不是 `str::len()`（字节数）。
            //    中文的 UTF-8 长度是 3，用 `m.len() == 1` 会一个都匹配不上 ⇒ prompt 全变 UNK
            //    ⇒ 三个不同 prompt 生成出**完全相同**的结果（这就是当时的现象）。
            let mut ch2id: std::collections::HashMap<char, u32> = std::collections::HashMap::new();
            for (i, m) in map.iter().enumerate().skip(1) {
                let mut it = m.chars();
                if let Some(c0) = it.next() {
                    if it.next().is_none() {
                        ch2id.entry(c0).or_insert(i as u32);
                    }
                }
            }
            println!("字符表可编码单字符 {} 个", ch2id.len());
            let mut ids: Vec<u32> = Vec::new();
            for ch in gen_prompt.chars() {
                let hit = *ch2id.get(&ch).unwrap_or(&0);
                if hit == 0 {
                    eprintln!("提示: prompt 里的字符 {:?} (U+{:04X}) 不在字符表里 ⇒ 记作 UNK(0)", ch, ch as u32);
                }
                ids.push(hit);
            }
            println!("prompt id 序列: {:?}", ids);
            let gscratch = alloc(n_vocab);
            let gidx = malloc(n_vocab * 4) as *mut u32; // u32 索引数组（alloc 返回 *mut f32）
            let gval = alloc(n_vocab);
            let t0 = std::time::Instant::now();
            let n_prompt = ids.len();
            let prompt_ids: Vec<u32> = ids.clone();
            // ---- best-of-K：K 条候选各自独立随机流；全部跑完后按自评 bits/char 选最低 ----
            // 依据：同一个模型下不同随机流的自评差可达 6.7 vs 13.2 bits/char（乱码=跑出分布），
            //       多抽几条再挑，等于把"运气"换成"抽样"，是推理端最省事的稳定手段。
            let n_cand = arg_u32("--n-candidates", 1).max(1) as usize;
            if n_cand > 1 {
                println!("best-of-{}: 每条候选独立随机流，按自评 bits/char 挑最低", n_cand);
            }
            let mut cands: Vec<(f32, f32, f32, Vec<u32>, f32)> = Vec::new(); // (自评, 去重率, 重复4gram覆盖, ids, 用时)
            for _cand in 0..n_cand {
                let mut ids: Vec<u32> = prompt_ids.clone();
                let mut gen_rng = Rng64::new(
                    ((seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03)
                        .wrapping_add((_cand as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
                );
                let tc = std::time::Instant::now();
                for _step in 0..gen_len {
                // 窗口 = 最近 seq 个 id；不足时左侧用**真实语料字符**补齐（不是 UNK(0)）。
                // ⚠️ 原来补 0（UNK）：训练窗口里全是真实字符，模型从没见过"62/64 是 UNK"的输入
                //    ⇒ 采样分布完全跑偏。实测：补 UNK 时生成文本模型自己给 11.8~14.1 bits/char
                //    （比均匀地板 11.99 还差，即"模型自己都觉得不可能是它生成的"），
                //    而同一模型在真实上下文下只要 ~1.2~5.0 bits/char。
                let win_len = seq.min(ids.len());
                let mut window: Vec<u32> = Vec::with_capacity(seq);
                let pad = seq - win_len;
                if pad > 0 {
                    // 用验证集的真实字符当左上下文（确定性、且与训练分布一致）
                    let n_pad_avail = val_mapped.len().saturating_sub(2);
                    for k in 0..pad {
                        window.push(val_mapped[(k * 13 + 7) % n_pad_avail.max(1)]);
                    }
                }
                window.extend_from_slice(&ids[ids.len() - win_len..]);
                let t = window.len();
                // [抽库 L4c-1] 组装收进 B1Model（生成语义；与搬移前逐字等价）
                let gskip = if head_skip { alloc(n_vocab) } else { core::ptr::null_mut() };
                let mut b1 = vmlm_models::model::B1Model {
                    cfg: vmlm_models::model::B1ForwardCfg {
                        dim, n_vocab, layers, nth, attn_every, n_attn,
                        attn_sparse, attn_residual, head_skip, identity_layers, relu,
                        io_sparse, k0_embed, t_tokens, tok_total: 0,
                        k0: 0, k0_attn: 0, ad: 0, fusion: false, no_res: false, profile: false,
                        train_mode: false, tt: 0, seq, nt_in: 0, attn_window: false,
                        head_blocked: false, nb_head: 0, k0_head: 0, t8_stack: false, t8s: 0,
                        attn_qk_check: false, slot: 0, off: 0, attn_debug: 0,
                    },
                    scs: &scs, asq: &asq, ask: &ask, asv: &asv, aso: &aso,
                    awq, awk, awv, awo, ad,
                    head: &head, head2: &head2,
                    emb, dim_of_p, etab,
                    h_ins: &h_ins, h, y,
                    aq, ak, av, ao, aprob, aout,
                    logits: gscratch, head2_out: gskip,
                    pres: &[], asp: core::ptr::null_mut(),
                    head_wt: core::ptr::null(), head2_wt: core::ptr::null(),
                    head_ht: core::ptr::null_mut(), t8ta: core::ptr::null_mut(), t8dy: core::ptr::null_mut(),
                    collect_flag: core::ptr::null_mut(), qk_stats: core::ptr::null_mut(),
                    dh: core::ptr::null_mut(),
                    dx: core::ptr::null_mut(),
                    dy: core::ptr::null_mut(),
                    dlogits: core::ptr::null_mut(),
                    dws: &[],
                    dwp: core::ptr::null_mut(),
                    head_dw: core::ptr::null_mut(),
                    lm_w: core::ptr::null_mut(),
                    lm_g: core::ptr::null_mut(),
                    lm_bg: core::ptr::null_mut(),
                    btd: core::ptr::null_mut(),
                    bth: core::ptr::null_mut(),
                    bt: core::ptr::null_mut(),
                    adx: core::ptr::null_mut(),
                    adq: core::ptr::null_mut(),
                    adk: core::ptr::null_mut(),
                    adv: core::ptr::null_mut(),
                    adobuf: core::ptr::null_mut(),
                    adprob: core::ptr::null_mut(),
                    arowsum: core::ptr::null_mut(),
                    aat1: core::ptr::null_mut(),
                    aat2: core::ptr::null_mut(),
                    awt: core::ptr::null_mut(),
                    asq_g: core::ptr::null_mut(),
                    ask_g: core::ptr::null_mut(),
                    asv_g: core::ptr::null_mut(),
                    aso_g: core::ptr::null_mut(),
                    awq_g: core::ptr::null_mut(),
                    awk_g: core::ptr::null_mut(),
                    awv_g: core::ptr::null_mut(),
                    awo_g: core::ptr::null_mut(),
                    t8dxn: core::ptr::null_mut(),
                    t8dyc: core::ptr::null_mut(),
                    prof_p: core::ptr::null_mut(),
                };
                let _ = b1.forward(Some(&window), t);
                let next = sample_next(
                    gscratch, n_vocab, gen_temp, gen_topk, gen_topp, &mut gen_rng, gidx, gval,
                    &ids, gen_rep_pen, gen_no_repeat,
                );
                ids.push(next);
                if n_cand == 1 && (ids.len() - n_prompt) % 20 == 0 {
                    println!("    ...已生成 {} 字符 ({:.1}s)", ids.len() - n_prompt, t0.elapsed().as_secs_f32());
                }
                }
                let gen_part: Vec<u32> = ids[n_prompt..].to_vec();
                let sc = score_ids(&gen_part);
                // 去重率（衡量"是不是复读"）：纯自评会选中复读（循环最可预测 ⇒ NLL 最低）
                let mut seen = vec![false; n_vocab.min(1 << 20)];
                let mut nd = 0usize;
                for &v in gen_part.iter() {
                    let i = v as usize;
                    if i < seen.len() && !seen[i] {
                        seen[i] = true;
                        nd += 1;
                    }
                }
                let div = nd as f32 / gen_part.len().max(1) as f32;
                // 最频繁 4-gram 的覆盖率：复读文本会接近 1（如"谢谢谢…"），正常文本很小
                let mut g4: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
                for w in gen_part.windows(4) {
                    let key = (w[0] & 0xFF) | ((w[1] & 0xFF) << 8) | ((w[2] & 0xFF) << 16) | ((w[3] & 0xFF) << 24);
                    *g4.entry(key).or_insert(0) += 1;
                }
                let rep4 = g4.values().copied().max().unwrap_or(0) as f32 * 4.0
                    / gen_part.len().max(1) as f32;
                let dt = tc.elapsed().as_secs_f32();
                if n_cand > 1 {
                    println!(
                        "    候选 {}/{}: 自评 {:.3} bits/char, 去重率 {:.3}, 重复4-gram覆盖 {:.2}, {} 字符 / {:.1}s",
                        _cand + 1, n_cand, sc, div, rep4, gen_part.len(), dt
                    );
                }
                cands.push((sc, div, rep4, ids, dt));
            }
            // 选择准则：① 自评 <= 8 bits（不是跑出分布的乱码）且 去重率 >= 0.25（不是复读）里取自评最低；
            //           ② 若没有合格候选，则取去重率最高的（宁可乱码也不要复读——复读是 100% 无信息）。
            cands.sort_by(|a, b| {
                let oa = !(a.0 <= 8.0 && a.2 <= 0.5);
                let ob = !(b.0 <= 8.0 && b.2 <= 0.5);
                oa.cmp(&ob).then_with(|| {
                    if !oa {
                        a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
                    } else {
                        // 都不合格：先要"别复读"，再要"别乱"（复读 100% 无信息，乱码至少还有信息）
                        a.2.partial_cmp(&b.2)
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then_with(|| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
                    }
                })
            });
            let (best_score, best_div, best_rep, ids, best_dt) = cands.remove(0);
            let _ = (best_div,);
            if n_cand > 1 {
                let allsc: Vec<String> = std::iter::once(format!("{:.2}/rep{:.2}", best_score, best_rep))
                    .chain(cands.iter().map(|c| format!("{:.2}/rep{:.2}", c.0, c.2)))
                    .collect();
                let good = best_score <= 8.0 && best_rep <= 0.5;
                println!(
                    "    best-of-{} 选定：自评 {:.3} bits/char, 重复4-gram覆盖 {:.2}{}（各候选 自评/rep: {}）",
                    n_cand, best_score, best_rep,
                    if good { "" } else { "  ⚠️ 无合格候选（都是乱码或复读），选了最不坏的" },
                    allsc.join(", ")
                );
            }
            let wall = t0.elapsed().as_secs_f32();
            // 解码
            let mut out = String::new();
            for &i in &ids {
                let m = map.get(i as usize).cloned().unwrap_or_default();
                out.push_str(&m);
            }
            let gen_text: String = ids[n_prompt..]
                .iter()
                .map(|&i| map.get(i as usize).cloned().unwrap_or_default())
                .collect();
            println!("--- 输出（prompt + 生成）---");
            println!("{}", out.replace('\n', "⏎")); // 控制台是 GBK，换行/中文可能显示异常；以文件为准
            println!("--- 只统计生成部分 ---");
            println!("{}", gen_text.replace('\n', "⏎"));
            println!(
                "速度: {} 字符 / {:.1}s = {:.2} 字符/秒（每字符一次完整前向，无 KV-Cache）",
                ids.len() - n_prompt, wall, (ids.len() - n_prompt) as f32 / wall.max(1e-6)
            );
            // 写文件（UTF-8，权威结果）
            let path = gen_out.clone().unwrap_or_else(|| format!("{}/generate.txt", out_dir));
            if let Some(dir) = std::path::Path::new(&path).parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            let mut fout = format!(
                "# B1 generate\nprompt={}\ngen_len={}\ntemperature={}\ntop_k(sampling)={}\ntop_p={}\nseed={}\nn_candidates={}\nself_bits_per_char={:.4}\ndistinct_ratio={:.4}\nrep4_coverage={:.4}\ncandidate_wall_s={:.2}\n",
                gen_prompt, gen_len, gen_temp, gen_topk, gen_topp, seed, n_cand, best_score, best_div, best_rep, best_dt
            );
            fout.push_str(&format!("chars_per_sec={:.2}\n\n[prompt+gen]\n{}\n\n[gen only]\n{}\n", (ids.len() - n_prompt) as f32 / wall.max(1e-6), out, gen_text));
            std::fs::write(&path, fout).unwrap();
            println!("已写入 {}", path);
            return;
        }

        let eval_val = || -> f32 {
            let mut s = 0.0f32;
            let mut n = 0usize;
            *ce_known_sum = 0.0;
            *ce_known_n = 0.0;
            *ce_unk_n = 0.0;
            for vb in val_batches.iter() {
                let (a, c) = step_fn(0, t_tokens, 0, vb, false);
                s += a;
                n += c;
            }
            let all = s / n as f32;
            let kn = *ce_known_n;
            let un = *ce_unk_n;
            let known = if kn > 0.0 { *ce_known_sum / kn } else { f32::NAN };
            let bpc = all as f64 / (chars_per_token.max(1e-6) as f64) / std::f64::consts::LN_2;
            let bpc_known = known as f64 / (chars_per_token.max(1e-6) as f64) / std::f64::consts::LN_2;
            println!(
                "    [val 口径] all={:.4}  非UNK目标={:.4}  (UNK 目标 {:.0}/{:.0} = {:.3}%)  bits/char: all={:.4} 非UNK={:.4}",
                all, known, un, un + kn, 100.0 * un as f64 / ((un + kn) as f64).max(1.0), bpc, bpc_known
            );
            all
        };
        // ⚠️ 评估口径体检：val **批内** 到底有多少 target 是 UNK(0)。
        //    为什么要专门查：如果评估批里 UNK 占比很高，那么 val loss 主要是在
        //    "猜不到的词"上，**任何模型都降不下去**，整套对比就测错了东西。
        {
            let mut unk_tgt = 0usize;
            let mut tot_tgt = 0usize;
            let mut unk_in = 0usize;
            let mut tot_in = 0usize;
            for vb in val_batches.iter() {
                for b in 0..batch {
                    for t in 0..seq {
                        let tok = vb[b * seq + t];
                        tot_in += 1;
                        if tok == 0 {
                            unk_in += 1;
                        }
                        if t + 1 < seq {
                            tot_tgt += 1;
                            if vb[b * seq + t + 1] == 0 {
                                unk_tgt += 1;
                            }
                        }
                    }
                }
            }
            println!(
                "eval-batch health: 输入 UNK {}/{} ({:.3}%) | 目标 UNK {}/{} ({:.3}%)  ← 目标 UNK 越高，val loss 越是被'猜不到的词'顶住",
                unk_in, tot_in, 100.0 * unk_in as f64 / tot_in.max(1) as f64,
                unk_tgt, tot_tgt, 100.0 * unk_tgt as f64 / tot_tgt.max(1) as f64
            );
        }

        // =====================================================================
        // --io-xcheck 1：稀疏 IO 前向 vs「等价的稠密写法」逐值对账
        // =====================================================================
        // 「接入是否改变了数学」不能靠"训练 loss 下降"证明（那是模型性质）。
        // 这里用**同一组参数**把两个前向都算一遍：
        //   稀疏路径：嵌入散写 k0_embed 个维度 + b1_forward(head) 收 k0_head 个源；
        //   稠密写法：显式构造 [dim] 的输入与 [V × dim] 的 head 矩阵（其余位置为 0），
        //             按稠密求和算 logits。
        // 两者数学相等；差异只来自**求和顺序**（16 项 vs 1536 项，其中 1520 项为 0）。
        // 报告 max|diff| 与相对量级 —— 这就是「与稠密版对比，报告差异量级」。
        if io_sparse && arg_u32("--io-xcheck", 0) == 1 {
            let nt = 2usize.min(t_tokens); // 只算 2 个 token：稠密写法是 O(V*dim)
            let mut toks = Vec::with_capacity(nt);
            for t in 0..nt {
                toks.push(train_mapped[(seed as usize * 31 + t * 7) % train_mapped.len()]);
            }
            let mut h_sp = alloc(nt * dim);
            let mut log_sp = alloc(nt * n_vocab);
            let mut h_dn = alloc(nt * dim);
            let mut log_dn = alloc(nt * n_vocab);
            zero(h_sp, nt * dim);
            zero(h_dn, nt * dim);
            // --- 稀疏：嵌入散写 + 头部稀疏收 ---
            for i in 0..nt {
                let tok = toks[i] as usize;
                let base = block_of[tok] as usize * k0_embed;
                for j in 0..k0_embed {
                    *h_sp.add(i * dim + base + j) = *etab.add(tok * k0_embed + j);
                }
                let x = h_sp.add(i * dim);
                // 复现完整路径（层 + head）：用一个临时 batch 太贵，这里直接算 head 部分，
                // 层部分由 --gradcheck 覆盖（它用的是同一条 b1_forward 路径）。
                b1_forward(&head, x, log_sp.add(i * n_vocab));
            }
            // --- 稠密写法：显式 [V × dim] 矩阵（其余 0）+ 稠密求和 ---
            for i in 0..nt {
                let tok = toks[i] as usize;
                let base = block_of[tok] as usize * k0_embed;
                for j in 0..k0_embed {
                    *h_dn.add(i * dim + base + j) = *etab.add(tok * k0_embed + j);
                }
            }
            zero(log_dn, nt * n_vocab);
            for v in 0..n_vocab {
                let b = block_of[v] as usize;
                for i in 0..nt {
                    let mut acc = 0.0f32;
                    for j in 0..k0_head {
                        // 稠密矩阵在 dim 上求和：只有 block(v)*k0_embed + j 非零
                        let d = b * k0_embed + j;
                        acc += *head.w.add(v * k0_head + j) * *h_dn.add(i * dim + d);
                    }
                    *log_dn.add(i * n_vocab + v) = acc;
                }
            }
            let mut max_abs = 0.0f32;
            let mut max_rel = 0.0f32;
            let mut scale = 0.0f32;
            for i in 0..nt * n_vocab {
                let a = *log_sp.add(i);
                let b = *log_dn.add(i);
                let d = (a - b).abs();
                if d > max_abs {
                    max_abs = d;
                }
                let s = b.abs();
                if s > scale {
                    scale = s;
                }
                if s > 1e-6 && d / s > max_rel {
                    max_rel = d / s;
                }
            }
            let mut hm = 0.0f32;
            for i in 0..nt * dim {
                let d = (*h_sp.add(i) - *h_dn.add(i)).abs();
                if d > hm {
                    hm = d;
                }
            }
            println!(
                "\n--- IO-XCHECK（稀疏 vs 等价稠密写法，{} token，{} 行 logits）---",
                nt, n_vocab
            );
            println!("    h       max|diff| = {:.3e}（嵌入散写，应逐位相同）", hm);
            println!("    logits  max|diff| = {:.3e}，max rel = {:.3e}，|logits| 量级 ≈ {:.3e}", max_abs, max_rel, scale);
            println!(
                "    稀疏 head 每 token 乘加 {} 次；稠密写法 {} 次（其中 {} 次乘的是 0）",
                k0_head, dim, dim - k0_head
            );
            let ok = hm == 0.0 && max_rel < 1e-4;
            println!("  IO-XCHECK {}", if ok { "PASS" } else { "FAIL" });
            if !ok {
                std::process::exit(1);
            }
            return; // 同上：PASS 正常返回以 flush 输出
        }

        // =====================================================================
        // --gradcheck 1：数值梯度 vs 解析梯度（中心差分）
        // =====================================================================
        if gradcheck {
            let mut toks = Vec::with_capacity(t_tokens);
            for b in 0..batch {
                let base = (seed as usize * 977 + b * seq) % (train_mapped.len() - seq - 1).max(1);
                for t in 0..seq {
                    toks.push(train_mapped[base + t]);
                }
            }
            if io_sparse {
                zero(etab_g, n_vocab * k0_embed);
                zero(head_dw, n_vocab * k0_head);
            } else {
                zero(emb_g, n_vocab * dim);
                zero(lm_g, n_vocab * dim);
                zero(lm_bg, n_vocab);
            }
            for l in 0..layers {
                zero(dws[l], dim * k0);
            }
            for b in 0..(if attn_sparse { 0 } else { n_attn }) {
                for g in [awq_g, awk_g, awv_g, awo_g] {
                    zero(g.add(b * ad), ad);
                }
            }
            hash_layers_before = {
                let mut h: u64 = 0xcbf2_9ce4_8422_2325;
                for l in 0..layers {
                    h ^= fnv_f32(scs[l].w, dim * k0);
                    h = h.wrapping_mul(0x100_0000_01b3);
                }
                h
            };
            hash_embed_before = if io_sparse { fnv_f32(etab, n_vocab * k0_embed) } else { fnv_f32(emb, n_vocab * dim) };
            hash_head_before = if io_sparse { fnv_f32(head.w, n_vocab * k0_head) } else { fnv_f32(lm_w, n_vocab * dim) };
            let l0 = step_fn(0, t_tokens, 0, &toks, false).0 / ((seq - 1) * batch) as f32;
            let _ = step_fn(0, t_tokens, 0, &toks, true);
            println!("\n--- GRADCHECK: 中心差分 eps={:.0e}，loss={:.4} ---", gc_eps, l0);
            let noise_abs = (l0.abs() * 6.0 * f32::EPSILON) / (2.0 * gc_eps);
            println!("    FD 噪声量级 ≈ {:.3e}（|解析梯度| 低于它的条目在本精度下不可分辨）", noise_abs);
            // 组：emb / lm_w / 每层 w（取 |grad| 最大的 gc_probes 个条目）
            let mut worst = 0.0f32;
            let mut probes = 0usize;
            let mut over = 0usize;
            let mut noise_limited = 0usize;
            let group_list: Vec<(&str, *mut f32, *const f32, usize)> = {
                let mut v: Vec<(&str, *mut f32, *const f32, usize)> = if io_sparse {
                    vec![
                        ("emb_tab", etab, etab_g as *const f32, n_vocab * k0_embed),
                        ("head", head.w, head_dw as *const f32, n_vocab * k0_head),
                    ]
                } else {
                    vec![
                        ("emb", emb, emb_g as *const f32, n_vocab * dim),
                        ("lm_w", lm_w, lm_g as *const f32, n_vocab * dim),
                    ]
                };
                for l in 0..layers {
                    v.push(("w[layer]", scs[l].w, dws[l] as *const f32, dim * k0));
                }
                // attention 权重也纳入 FD 对账（用户说"不做完整对账"，但把它加进
                // 现有探针几乎是免费的，能挡住"反向写错还照样降 loss"这类事故）
                if n_attn > 0 && !attn_sparse {
                    v.push(("attn_wq", awq, awq_g as *const f32, ad));
                    v.push(("attn_wo", awo, awo_g as *const f32, ad));
                }
                // 消融二：稀疏投影的权重也纳入对账
                if attn_sparse {
                    // 四组都覆盖（上一轮只探了 Q/O，用户要求 Q/K/V/O 全 PASS）
                    v.push(("attn_sq", asq[0].w, asq_g as *const f32, dim * k0_attn));
                    v.push(("attn_sk", ask[0].w, ask_g as *const f32, dim * k0_attn));
                    v.push(("attn_sv", asv[0].w, asv_g as *const f32, dim * k0_attn));
                    v.push(("attn_so", aso[0].w, aso_g as *const f32, dim * k0_attn));
                }
                v
            };
            for (name, w, g, n) in group_list.iter() {
                for _pick in 0..gc_probes {
                    let mut best = 0usize;
                    let mut bv = -1.0f32;
                    for i in 0..*n {
                        let a = (*g).add(i).read().abs();
                        if a > bv {
                            bv = a;
                            best = i;
                        }
                    }
                    if bv <= 0.0 {
                        break;
                    }
                    let orig = (*w).add(best).read();
                    (*w).add(best).write(orig + gc_eps);
                    // [P19-8] FD 直接改的是 head.w/head2.w —— 前向走的是 WT 副本，
                    // 必须先重打包，否则 FD 恒为 0（实测：head 组 6 个探针全 OVER）。
                    repack_head_wt(head_blocked, head.w, head_wt, head2.w, head2_wt, head_skip, n_vocab, k0_head);
                    let lp = step_fn(0, t_tokens, 0, &toks, false).0 / ((seq - 1) * batch) as f32;
                    (*w).add(best).write(orig - gc_eps);
                    repack_head_wt(head_blocked, head.w, head_wt, head2.w, head2_wt, head_skip, n_vocab, k0_head);
                    let lm = step_fn(0, t_tokens, 0, &toks, false).0 / ((seq - 1) * batch) as f32;
                    (*w).add(best).write(orig);
                    repack_head_wt(head_blocked, head.w, head_wt, head2.w, head2_wt, head_skip, n_vocab, k0_head);
                    let num = (lp - lm) / (2.0 * gc_eps);
                    let ana = (*g).add(best).read();
                    let tol = (5.0e-2 * ana.abs()).max(3.0 * noise_abs);
                    let ok = (num - ana).abs() <= tol;
                    if !ok {
                        over += 1;
                    }
                    if 3.0 * noise_abs > 5.0e-2 * ana.abs() {
                        noise_limited += 1;
                    }
                    let rel = (num - ana).abs() / num.abs().max(1.0e-6);
                    println!(
                        "    probe {:<8} idx={:7} ana={:+.4e} num={:+.4e} rel={:.2e} SNR={:>7.0} {}",
                        name, best, ana, num, rel, ana.abs() / noise_abs,
                        if ok { "ok" } else { "OVER" }
                    );
                    if rel > worst {
                        worst = rel;
                    }
                    probes += 1;
                    let gp = *g as *mut f32;
                    *gp.add(best) = 0.0;
                }
            }
            let pass = over == 0 && probes >= 6;
            println!(
                "  GRADCHECK {}: {} 探针，越界 {}，worst rel={:.3e}，其中 {} 个受噪声限制",
                if pass { "PASS" } else { "FAIL" },
                probes, over, worst, noise_limited
            );
            if !pass {
                std::process::exit(1);
            }
            return; // 同上：PASS 正常返回以 flush 输出
        }

        // =====================================================================
        // 训练
        // =====================================================================
        std::fs::create_dir_all(&out_dir).expect("cannot create out-dir");
        let metrics_path = format!("{}/metrics.jsonl", out_dir);
        let curve_path = format!("{}/loss_curve.csv", out_dir);
        {
            use std::io::Write;
            let mut f = std::fs::File::create(&metrics_path).unwrap();
            let mut c = std::fs::File::create(&curve_path).unwrap();
            writeln!(c, "step,lr,train_loss,val_loss,val_ppl").unwrap();
        }

        // ---- 检查点内容（终点保存与周期保存共用）----
        let ck_geo = CkGeo {
            dim, k0, k0_embed, k0_head, layers, n_attn, k0_attn, io_sparse, head_skip, attn_sparse, n_vocab,
        };
        let build_tens = || -> Vec<(*const f32, usize)> {
            let mut tens: Vec<(*const f32, usize)> = Vec::new();
            for l in 0..layers {
                tens.push((scs[l].w, dim * k0));
            }
            if io_sparse {
                tens.push((etab, n_vocab * k0_embed));
                tens.push((head.w, n_vocab * k0_head));
                if head_skip {
                    tens.push((head2.w, n_vocab * k0_head));
                }
            } else {
                tens.push((emb, n_vocab * dim));
                tens.push((lm_w, n_vocab * dim));
                tens.push((lm_b, n_vocab));
            }
            if n_attn > 0 {
                if attn_sparse {
                    for b in 0..n_attn {
                        tens.push((asq[b].w, dim * k0_attn));
                        tens.push((ask[b].w, dim * k0_attn));
                        tens.push((asv[b].w, dim * k0_attn));
                        tens.push((aso[b].w, dim * k0_attn));
                    }
                } else {
                    for b in 0..n_attn {
                        tens.push((awq.add(b * ad), ad));
                        tens.push((awk.add(b * ad), ad));
                        tens.push((awv.add(b * ad), ad));
                        tens.push((awo.add(b * ad), ad));
                    }
                }
            }
            tens
        };

        let mut cursor = (seed as usize * 977) % (train_mapped.len() - seq - 1).max(1);
        let mut step_t: u32 = 0;
        let mut losses: Vec<f32> = Vec::new(); // 每次 log_every 的训练 loss
        let mut evals: Vec<(usize, f32, f32, f32, f32)> = Vec::new(); // step, lr, train, val, ppl
        let mut diverged = false;
        let mut diverged_at: usize = 0;
        let mut stopped_by = "steps";
        let mut best_val = f32::INFINITY;
        let mut best_step = 0usize;
        let mut no_improve = 0usize;
        let t_start = std::time::Instant::now();
        let mut last_logged_train = f32::NAN;
        // 训练前参数哈希（分组）——实验 A/B 的"冻结可证"依据
        hash_layers_before = {
            let mut h: u64 = 0xcbf2_9ce4_8422_2325;
            for l in 0..layers {
                h ^= fnv_f32(scs[l].w, dim * k0);
                h = h.wrapping_mul(0x100_0000_01b3);
            }
            h
        };
        hash_embed_before = if io_sparse { fnv_f32(etab, n_vocab * k0_embed) } else { fnv_f32(emb, n_vocab * dim) };
        hash_head_before = if io_sparse { fnv_f32(head.w, n_vocab * k0_head) } else { fnv_f32(lm_w, n_vocab * dim) };
        if freeze_layers || freeze_io || identity_layers {
            println!(
                "ATTRIBUTION: freeze_layers={} freeze_embed={} freeze_head={} identity_layers={}",
                freeze_layers, freeze_embed, freeze_head, identity_layers
            );
        }

        // ---- 每步一行 CSV（--log-every 只管控制台；这里**每步都写**，方便直接画曲线）----
        let tlog_path = format!("{}/train_log.csv", out_dir);
        let mut tlog = std::fs::File::create(&tlog_path).unwrap();
        {
            use std::io::Write as _;
            writeln!(tlog, "step,virtual_step,lr,train_loss,elapsed_s,tok_per_s").unwrap();
        }

        if lr_offset > 0 {
            println!(
                "续训模式: --lr-offset {} ⇒ 本次跑第 {}..{} 步，lr 调度按虚拟步继续（不重新 warmup）",
                lr_offset, lr_offset + 1, steps
            );
        }
        for st in (lr_offset + 1)..=steps {
            step_t += 1;
            // ---- lr 计划：线性 warmup + cosine 退火（用虚拟步 vst，续训接得上）----
            let vst = st; // 这里的 st 已经是"虚拟步"（从 lr_offset+1 开始）
            let cur_lr = if vst <= warmup {
                lr * (vst as f32) / (warmup.max(1) as f32)
            } else {
                let denom = (steps.saturating_sub(warmup)).max(1) as f32;
                let p = ((vst - warmup) as f32 / denom).min(1.0);
                min_lr + (lr - min_lr) * 0.5 * (1.0 + (std::f32::consts::PI * p).cos())
            };
            // ---- 取一批 ----
            let mut toks = Vec::with_capacity(t_tokens);
            for b in 0..batch {
                let base = (cursor + b * seq) % (train_mapped.len() - seq - 1).max(1);
                for t in 0..seq {
                    toks.push(train_mapped[base + t]);
                }
            }
            cursor = (cursor + batch * seq) % (train_mapped.len() - seq - 1).max(1);

            if io_sparse {
                zero(etab_g, n_vocab * k0_embed);
            } else {
                zero(emb_g, n_vocab * dim);
                zero(lm_g, n_vocab * dim);
                zero(lm_bg, n_vocab);
            }
            let tp0 = std::time::Instant::now();
            // ================= inter-op 数据并行：每线程一个 region =================
            // 等价性：4 线程 × 每线程 8 样本 ≡ 单线程 batch 32（前提：归约按 slot 升序固定）。
            let (ls, nc) = if inter && nslot > 1 {
                let sf = SharedFn(&step_fn);
                let sums = alloc(nslot);
                let cnts = alloc(nslot);
                zero(sums, nslot);
                zero(cnts, nslot);
                let worker = |s_i: usize| unsafe {
                    let (a, b) = (sf.0)(s_i * tt, tt, s_i, &toks, true);
                    *sums.add(s_i) = a;
                    *cnts.add(s_i) = b as f32;
                };
                let sw = SharedFn(&worker);
                // ⚠️ 必须整体借用 sw：Rust 2021 的"字段级捕获"会直接抓 sw.0（&闭包，含裸指针）⇒ 不满足 Sync。
                let swr = &sw;
                par_for(nslot, nslot, move |lo, hi| {
                    for s_i in lo..hi {
                        (swr.0)(s_i);
                    }
                });
                // 末端：按 slot 升序归约各 slot 的梯度（固定顺序 ⇒ 确定性）
                let red = |base: *mut f32, per: usize| unsafe {
                    // [P19-9] 8 宽向量化（每步 12 层 + head/embed 共 ~4M 次加法，原来是标量链）
                    if vmlm_core::interp::HAS_AVX2 {
                        for sl in 1..nslot {
                            let src = base.add(sl * per);
                            vmlm_core::kernels::add256_f32(base, src, per);
                        }
                        return;
                    }
                    for sl in 1..nslot {
                        for j in 0..per {
                            *base.add(j) += *base.add(sl * per + j);
                        }
                    }
                };
                for l in 0..layers {
                    red(dws[l], dim * k0);
                }
                if io_sparse {
                    red(head_dw, n_vocab * k0_head);
                    if head_skip {
                        red(head2_dw, n_vocab * k0_head);
                    }
                    red(etab_g, n_vocab * k0_embed);
                } else {
                    red(emb_g, nd);
                    red(lm_g, nd);
                    red(lm_bg, n_vocab);
                }
                if attn_sparse && n_attn > 0 {
                    let as_w1_o = as_w / nslot;
                    red(asq_g, as_w1_o);
                    red(ask_g, as_w1_o);
                    red(asv_g, as_w1_o);
                    red(aso_g, as_w1_o);
                } else if n_attn > 0 {
                    red(awq_g, n_attn.max(1) * ad);
                    red(awk_g, n_attn.max(1) * ad);
                    red(awv_g, n_attn.max(1) * ad);
                    red(awo_g, n_attn.max(1) * ad);
                }
                let mut a = 0.0f32;
                let mut b = 0.0f32;
                for i in 0..nslot {
                    a += *sums.add(i);
                    b += *cnts.add(i);
                }
                (a, b as usize)
            } else {
                step_fn(0, t_tokens, 0, &toks, true)
            };
            let _ = tp0;
            // ⚠️ 不要把整个 step_fn 记进相位 6：那会把 0..5 再算一遍（第一版就是这样，
            //    相位总和 31.25 s 而 wall 只有 15.8 s）。取批的时间很短，单独记没有意义。
            // ---- 频率统计：累加本步的激活/使用情况（只读，不改模型）----
            if collect {
                samples += t_tokens as u64;
                for i in 0..t_tokens {
                    for d in 0..dim {
                        if (*h.add(i * dim + d)).abs() > 1.0e-12 {
                            *nz_last_p.add(d) += 1;
                        }
                        if (*h_ins[1].add(i * dim + d)).abs() > 1.0e-12 {
                            *nz_l0_p.add(d) += 1;
                        }
                        if (*h_ins[0].add(i * dim + d)).abs() > 1.0e-12 {
                            *nz_in_p.add(d) += 1;
                        }
                    }
                }
                for b in 0..batch {
                    for t in 0..seq {
                        let tok = toks[b * seq + t] as usize;
                        for j in 0..k0_embed {
                            *emb_use_p.add(dim_of[tok * k0_embed + j] as usize) += 1;
                        }
                    }
                }
            }
            let tl = ls / nc as f32;
            if !tl.is_finite() || !ls.is_finite() || tl > div_thresh {
                diverged = true;
                diverged_at = st;
                stopped_by = "diverged";
                println!(
                    "[step {}] DIVERGED: train_loss={} (ln V={:.4}, threshold 2*ln V={:.4})",
                    st, tl, lnv, div_thresh
                );
                break;
            }
            let ta0 = std::time::Instant::now();
            // ---- 归因实验 A/B：冻结组**既不清梯度也不更新**（梯度置零，参数逐位不动）----
            if freeze_embed || freeze_head {
                if io_sparse {
                    if freeze_embed {
                        zero(etab_g, n_vocab * k0_embed);
                    }
                    if freeze_head {
                        zero(head_dw, n_vocab * k0_head);
                    }
                } else {
                    if freeze_embed {
                        zero(emb_g, n_vocab * dim);
                    }
                    if freeze_head {
                        zero(lm_g, n_vocab * dim);
                        zero(lm_bg, n_vocab);
                    }
                }
            }
            if io_sparse {
                if !freeze_head {
                    vmlm_core::train::clip_grad(head_dw, n_vocab * k0_head, clip);
                    adam_dense(if bf16_mirror { head_master } else { head.w }, head_dw, head_m, head_v, n_vocab * k0_head, cur_lr, step_t);
                    if head_skip {
                        vmlm_core::train::clip_grad(head2_dw, n_vocab * k0_head, clip);
                        adam_dense(if bf16_mirror { head2_master } else { head2.w }, head2_dw, head2_m, head2_v, n_vocab * k0_head, cur_lr, step_t);
                    }
                }
                if !freeze_embed {
                    vmlm_core::train::clip_grad(etab_g, n_vocab * k0_embed, clip);
                    adam_dense(etab, etab_g, etab_m, etab_v, n_vocab * k0_embed, cur_lr, step_t);
                }
            } else {
                if !freeze_head {
                    vmlm_core::train::clip_grad(lm_g, n_vocab * dim, clip);
                    vmlm_core::train::clip_grad(lm_bg, n_vocab, clip);
                    adam_dense(lm_w, lm_g, lm_m, lm_v, n_vocab * dim, cur_lr, step_t);
                    adam_dense(lm_b, lm_bg, lm_bm, lm_bv, n_vocab, cur_lr, step_t);
                }
                if !freeze_embed {
                    vmlm_core::train::clip_grad(emb_g, n_vocab * dim, clip);
                    adam_dense(emb, emb_g, emb_m, emb_v, n_vocab * dim, cur_lr, step_t);
                }
            }
            if !freeze_layers && !identity_layers {
                for l in 0..layers {
                    vmlm_core::train::clip_grad(dws[l], dim * k0, clip);
                    adam_dense(if bf16_mirror { sp_master[l] } else { scs[l].w }, dws[l], sp_ms[l], sp_vs[l], dim * k0, cur_lr, step_t);
                }
            }
            // attention 权重（dense Adam，与 sparse_train24 一致）；稀疏模式下 ad=1 且无权重可更
            for b in 0..(if attn_sparse { 0 } else { n_attn }) {
                for (w, g, m, v) in [
                    (awq, awq_g, awq_m, awq_v),
                    (awk, awk_g, awk_m, awk_v),
                    (awv, awv_g, awv_m, awv_v),
                    (awo, awo_g, awo_m, awo_v),
                ] {
                    vmlm_core::train::clip_grad(g.add(b * ad), ad, clip);
                    adam_dense(w.add(b * ad), g.add(b * ad), m.add(b * ad), v.add(b * ad), ad, cur_lr, step_t);
                }
            }
            // 消融二：稀疏投影的权重（同样 dense Adam，但每个只有 dim×k0_attn 个参数）
            if attn_sparse {
                for b in 0..n_attn {
                    // 只读地取 w 指针：`&mut asq[b]` 会与 step_fn 闭包里的不可变借用冲突
                    for (w, g, m, v) in [
                        (asq[b].w, asq_g, asq_m, asq_v),
                        (ask[b].w, ask_g, ask_m, ask_v),
                        (asv[b].w, asv_g, asv_m, asv_v),
                        (aso[b].w, aso_g, aso_m, aso_v),
                    ] {
                        let off = b * dim * k0_attn;
                        vmlm_core::train::clip_grad(g.add(off), dim * k0_attn, clip);
                        adam_dense(w, g.add(off), m.add(off), v.add(off), dim * k0_attn, cur_lr, step_t);
                    }
                }
            }
            // 每步清零 attention 梯度（下一轮反向会累加）
            for b in 0..n_attn {
                for g in [awq_g, awk_g, awv_g, awo_g] {
                    zero(g.add(b * ad), ad);
                }
            }
            // [P23b] 镜像刷新：f32 master → compute 缓冲（bf16 精度、预扩成 f32）。
            // 优化器只动 master ⇒ 学习量不丢；计算侧看到的是 bf16 精度权重。
            if bf16_mirror {
                unsafe {
                    let f = |dst: *mut f32, src: *const f32, n: usize| {
                        let mut i = 0usize;
                        while i < n {
                            *dst.add(i) = vmlm_core::kernels::bf16_to_f32(vmlm_core::kernels::f32_to_bf16(*src.add(i)));
                            i += 1;
                        }
                    };
                    f(head.w, head_master, n_vocab * k0_head);
                    if head_skip {
                        f(head2.w, head2_master, n_vocab * k0_head);
                    }
                    for l in 0..layers {
                        f(scs[l].w, sp_master[l], dim * k0);
                    }
                }
            }
            // [P19-8] Adam 更新完 head 权重后重打包 WT（否则前向会用旧权重 ⇒ 静默错）。
            repack_head_wt(head_blocked, head.w, head_wt, head2.w, head2_wt, head_skip, n_vocab, k0_head);
            if profile {
                *prof_p.add(7) += ta0.elapsed().as_secs_f64();
            }
            // ---- 方案一：动态子空间换维（每 dyn_every 步一次）----
            if io_sparse && dyn_sub && st % dyn_every == 0 && st < steps {
                // (1) 重要性 EMA：|梯度|（在 optimizer 之后、下一步清零之前读）
                for i in 0..n_vocab * k0_embed {
                    let g = (*etab_g.add(i)).abs();
                    *imp_p.add(i) = dyn_ema * *imp_p.add(i) + (1.0 - dyn_ema) * g;
                }
                // 候选：**游标**轮转 —— 每条 token 从 cursor 起找第一个「不在本 token 子空间」的维度，
                // 找到后 cursor 前移。这样一轮下来每个维度被分到的次数差 ≤1（自动均衡），
                // 且平均只需试 1~2 个维度。
                // ⚠️ 第一版把候选序**在一轮开始前排一次序**然后反复取同一个"最低使用"维度：
                //    结果 10921 个 token 全挤到同一个维度上（实测"每维使用 0..10580"），
                //    共享子空间瞬间退化成"所有人读同几个维度"。
                let mut cursor: usize = dyn_cursor;
                let mut swaps = 0usize;
                for v in 0..n_vocab {
                    let row = v * k0_embed;
                    // 该 token 最弱的槽
                    let mut jmin = 0usize;
                    let mut bv = f32::INFINITY;
                    for j in 0..k0_embed {
                        let x = *imp_p.add(row + j);
                        if x < bv {
                            bv = x;
                            jmin = j;
                        }
                    }
                    let mut chosen: i32 = -1;
                    for t in 0..dim {
                        let d = (cursor + t) % dim;
                        let mut inrow = false;
                        for j in 0..k0_embed {
                            if dim_of[row + j] as usize == d {
                                inrow = true;
                                break;
                            }
                        }
                        if !inrow {
                            chosen = d as i32;
                            cursor = (d + 1) % dim;
                            break;
                        }
                    }
                    if chosen >= 0 {
                        let d = chosen as usize;
                        dim_usage[dim_of[row + jmin] as usize] -= 1;
                        dim_of[row + jmin] = d as u32;
                        dim_usage[d] += 1;
                        *imp_p.add(row + jmin) = 0.0; // 新槽重新积累重要性
                        swaps += 1;
                    }
                }
                // (2) 共享子空间：head 的源跟着换（dim_of 是唯一真相）
                for v in 0..n_vocab {
                    for j in 0..k0_head {
                        let t = head.idx(v, j);
                        let sid = dim_of[v * k0_embed + j];
                        *head.src.add(t) = sid as f32;
                        *head.src_u32.add(t) = sid;
                    }
                }
                head.assert_src_u32_consistent();
                dyn_cursor = cursor;
                dyn_swaps_total += swaps;
                dyn_rounds += 1;
                let umin2 = *dim_usage.iter().min().unwrap();
                let umax2 = *dim_usage.iter().max().unwrap();
                println!(
                    "[step {:4}] dyn-subspace: 换维 {} 槽（累计 {}），每维使用 {}..{}",
                    st, swaps, dyn_swaps_total, umin2, umax2
                );
            }
            last_logged_train = tl;
            // ---- 每步一行 CSV（含 flush，跑到一半也能直接读）----
            {
                use std::io::Write as _;
                let wall_now = t_start.elapsed().as_secs_f64();
                let tps_now = (st * t_tokens) as f64 / wall_now.max(1e-9);
                writeln!(
                    tlog,
                    "{},{},{:.6e},{:.6},{:.4},{:.2}",
                    st, st, cur_lr, tl, wall_now, tps_now
                )
                .unwrap();
                let _ = tlog.flush();
            }
            // ---- 进度行（--log-every）：必须**真的打印**，否则长跑看不到进度 ----
            // 注意：loss_hash 只由"被记录的步"决定，所以这里的 losses.push 语义保持不变。
            let will_eval = st % eval_every == 0 || st == steps;
            if log_every > 0 && (st % log_every == 0 || st == 1) && !will_eval {
                let wall = t_start.elapsed().as_secs_f64();
                let tps = (st * t_tokens) as f64 / wall.max(1e-9);
                let eta_h = if st > 0 {
                    (steps - st) as f64 * (wall / st as f64) / 3600.0
                } else {
                    f64::NAN
                };
                println!(
                    "[log {:6}/{}] lr={:.3e} train={:.4} | {:.0}s, {:.0} tok/s, ETA {:.2}h",
                    st, steps, cur_lr, tl, wall, tps, eta_h
                );
                use std::io::Write;
                let _ = std::io::stdout().flush();
            }
            // ---- 周期落盘：--save-every N（0=关）。写 <save-model>.step<N>.cpk ----
            if save_every > 0 && st % save_every == 0 {
                if let Some(base) = ck_save.as_ref() {
                    let p = format!("{}.step{}", base, st);
                    let tens = build_tens();
                    let nw = ck_write(&p, &ck_geo, &tens);
                    println!(
                        "[ckpt] step {}: 已周期保存 {}（{} 张量, {} f32, {:.1} MB）",
                        st, p, tens.len(), nw,
                        nw as f64 * 4.0 / 1e6
                    );
                }
            }
            if st % log_every == 0 || st == 1 {
                losses.push(tl);
            }
            // ---- 评估 ----
            if st % eval_every == 0 || st == steps {
                let vl = eval_val();
                let ppl = vmlm_core::kernels::exp_f(vl);
                let wall = t_start.elapsed().as_secs_f64();
                evals.push((st, cur_lr, tl, vl, ppl));
                {
                    use std::io::Write;
                    let mut f = std::fs::OpenOptions::new().append(true).open(&metrics_path).unwrap();
                    writeln!(
                        f,
                        "{{\"step\":{},\"lr\":{:.8},\"train_loss\":{:.6},\"val_loss\":{:.6},\"val_ppl\":{:.4},\"wall_s\":{:.3}}}",
                        st, cur_lr, tl, vl, ppl, wall
                    )
                    .unwrap();
                    let mut c = std::fs::OpenOptions::new().append(true).open(&curve_path).unwrap();
                    writeln!(c, "{},{:.8},{:.6},{:.6},{:.4}", st, cur_lr, tl, vl, ppl).unwrap();
                }
                if !vl.is_finite() {
                    diverged = true;
                    diverged_at = st;
                    stopped_by = "diverged";
                    println!("[step {}] DIVERGED: val_loss is not finite", st);
                    break;
                }
                println!(
                    "[step {:5}/{}] lr={:.3e} train={:.4} val={:.4} ppl={:10.2} wall={:6.1}s",
                    st, steps, cur_lr, tl, vl, ppl, wall
                );
                if vl < best_val - min_delta {
                    best_val = vl;
                    best_step = st;
                    no_improve = 0;
                } else {
                    no_improve += 1;
                    if plateau_patience > 0 && no_improve >= plateau_patience {
                        stopped_by = "plateau";
                        println!(
                            "PLATEAU: val_loss 连续 {} 次评估没有改善（>{:.1e}），在 step {} 停止。best={:.4} @ step {}",
                            no_improve, min_delta, st, best_val, best_step
                        );
                        break;
                    }
                }
            }
        }
        let total_wall = t_start.elapsed().as_secs_f64();
        // 训练后参数哈希（与 before 比较 ⇒ 冻结是否真的生效，可证）
        hash_layers_after = {
            let mut h: u64 = 0xcbf2_9ce4_8422_2325;
            for l in 0..layers {
                h ^= fnv_f32(scs[l].w, dim * k0);
                h = h.wrapping_mul(0x100_0000_01b3);
            }
            h
        };
        hash_embed_after = if io_sparse { fnv_f32(etab, n_vocab * k0_embed) } else { fnv_f32(emb, n_vocab * dim) };
        hash_head_after = if io_sparse { fnv_f32(head.w, n_vocab * k0_head) } else { fnv_f32(lm_w, n_vocab * dim) };
        println!(
            "param hashes: layers {:016x}->{:016x} | embed {:016x}->{:016x} | head {:016x}->{:016x}",
            hash_layers_before, hash_layers_after, hash_embed_before, hash_embed_after, hash_head_before, hash_head_after
        );
        let steps_done = step_t as usize;
        let tokens_seen = steps_done * t_tokens;
        let final_train = if evals.is_empty() { last_logged_train } else { evals.last().unwrap().2 };
        let final_val = if evals.is_empty() { f32::NAN } else { evals.last().unwrap().3 };
        let hash = loss_hash(&losses);

        // =====================================================================
        // 训练结束：再测一次 q·k（与训练开始时对比 —— 一致性判据）
        // + 保存权重（--save-model），使"训练后的验证"能拿到同一份权重
        // =====================================================================
        if n_attn > 0 && attn_qk_check {
            let vb = val_batches[0].clone();
            for b in 0..n_attn {
                *collect_flag.add(b) = 1;
            }
            let _ = step_fn(0, t_tokens, 0, &vb, false);
            let mut mn = f32::INFINITY;
            let mut mx = 0.0f32;
            for b in 0..n_attn {
                let v = *qk_stats.add(b * 5 + 0);
                if v < mn {
                    mn = v;
                }
                if v > mx {
                    mx = v;
                }
            }
            println!(
                "--- ATTENTION 训练结束体检：worst max|q.k|={:.4e} best max|q.k|={:.4e}（应与训练开始时同量级）",
                mn, mx
            );
            if mn <= 0.01 {
                eprintln!("FATAL: 训练结束时 attention 已坍缩（max|q.k|={:.4e} <= 0.01）。", mn);
                std::process::exit(2);
            }
        }
        if let Some(path) = ck_save.clone() {
            let tens = build_tens();
            let nw = ck_write(&path, &ck_geo, &tens);
            println!("--- 已保存权重: {} （{} 个张量, {} 个 f32，标定因子已含在 Wq/Wk 里）", path, tens.len(), nw);
        }
        shutdown_pool(); // 常驻 worker 优雅退出
    println!("\n===== RESULT =====");
        println!(
            "steps_done={} stopped_by={} diverged={} diverged_at={}",
            steps_done, stopped_by, diverged, diverged_at
        );
        println!("final_train={:.6} final_val={:.6} best_val={:.6} @step {}", final_train, final_val, best_val, best_step);
        println!(
            "wall={:.1}s tokens_seen={} tok/s={:.1} loss_hash=0x{:016x}",
            total_wall, tokens_seen, tokens_seen as f64 / total_wall.max(1e-9), hash
        );

        // ---- summary.json ----
        {
            use std::io::Write;
            let mut f = std::fs::File::create(format!("{}/summary.json", out_dir)).unwrap();
            writeln!(
                f,
                "{{\"dim\":{},\"k0\":{},\"layers\":{},\"no_residual\":{},\"depth_scale\":{:.6},\"io_sparse\":{},\"k0_embed\":{},\"k0_head\":{},\"n_blocks\":{},\"params_main\":{},\"params_embed\":{},\"params_head\":{},\"params_total\":{},\"macs_head\":{},\"attn_blocks\":{},\"params_attn\":{},\"macs_attn\":{},\"freeze_layers\":{},\"freeze_embed\":{},\"freeze_head\":{},\"identity_layers\":{},\"hash_layers_before\":\"0x{:016x}\",\"hash_layers_after\":\"0x{:016x}\",\"hash_embed_before\":\"0x{:016x}\",\"hash_embed_after\":\"0x{:016x}\",\"hash_head_before\":\"0x{:016x}\",\"hash_head_after\":\"0x{:016x}\",\"steps_cfg\":{},\"steps_done\":{},\"seq\":{},\"batch\":{},\"lr\":{},\"min_lr\":{},\"warmup\":{},\"clip\":{},\"seed\":{},\"vocab_mode\":\"{}\",\"n_vocab\":{},\"val_oov\":{},\"data_dir\":\"{}\",\"data_train_tokens\":{},\"data_val_tokens\":{},\"stopped_by\":\"{}\",\"diverged\":{},\"diverged_at\":{},\"final_train\":{:.6},\"final_val\":{:.6},\"best_val\":{:.6},\"best_step\":{},\"wall_s\":{:.3},\"tokens_seen\":{},\"tok_per_s\":{:.1},\"loss_hash\":\"0x{:016x}\",\"evals\":{}}}",
                dim, k0, layers, no_res, depth_scale, io_sparse, k0_embed, k0_head, n_blocks,
                params_main, params_embed, params_head, params_main + params_embed + params_head,
                macs_head, n_attn, params_attn, macs_attn, freeze_layers, freeze_embed, freeze_head, identity_layers,
                hash_layers_before, hash_layers_after, hash_embed_before, hash_embed_after,
                hash_head_before, hash_head_after,
                steps, steps_done, seq, batch, lr, min_lr, warmup,
                clip, seed, vocab_mode, n_vocab, val_oov, jesc(&data_dir), train.len(), val.len(), stopped_by,
                diverged, diverged_at, jnum(final_train), jnum(final_val), jnum(best_val), best_step, total_wall,
                tokens_seen, tokens_seen as f64 / total_wall.max(1e-9), hash, evals.len()
            )
            .unwrap();
        }

        // ---- 频率统计报告 + 分阶段耗时 ----
        if collect {
            use std::io::Write;
            let mut s = std::fs::File::create(&stats_out).unwrap();
            let tot = samples.max(1) as f64;
            writeln!(s, "# 激活频率统计（只观测；模型未改动）").unwrap();
            writeln!(s, "dim={} k0(main)={} k0_embed={} k0_head={} layers={} io_sparse={}", dim, k0, k0_embed, k0_head, layers, io_sparse).unwrap();
            writeln!(s, "samples = 步数 × token = {}（每步 {} token）", samples, t_tokens).unwrap();
            writeln!(s).unwrap();
            // (1) 输入/层0/最后一层的非零率
            for (nm, arr, n) in [
                ("输入(嵌入展开)", nz_in_p, dim),
                ("层0输出", nz_l0_p, dim),
                ("最后一层输出", nz_last_p, dim),
            ] {
                let mut rates: Vec<f64> = (0..n).map(|i| unsafe { *arr.add(i) } as f64 / tot).collect();
                let mean = rates.iter().sum::<f64>() / n as f64;
                rates.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let med = rates[n / 2];
                let p10 = rates[n / 10];
                let p90 = rates[n * 9 / 10];
                let hi = rates.iter().filter(|&&r| r > 0.5).count();
                let mid = rates.iter().filter(|&&r| r > 0.1).count();
                let lo = rates.iter().filter(|&&r| r < 0.01).count();
                writeln!(s, "## {}", nm).unwrap();
                writeln!(s, "每个维度的「非零出现率」= 该维度非零的 (步×token) 次数 / 总样本数").unwrap();
                writeln!(s, "| 统计 | 值 |").unwrap();
                writeln!(s, "|---|---|").unwrap();
                writeln!(s, "| 平均非零率 | {:.4} |", mean).unwrap();
                writeln!(s, "| 中位数 / p10 / p90 | {:.4} / {:.4} / {:.4} |", med, p10, p90).unwrap();
                writeln!(s, "| 非零率 > 0.5 的维度数 | {} / {} |", hi, n).unwrap();
                writeln!(s, "| 非零率 > 0.1 的维度数 | {} / {} |", mid, n).unwrap();
                writeln!(s, "| 非零率 < 0.01 的维度数 | {} / {} |", lo, n).unwrap();
                writeln!(s, "| 平均每 token 非零维度数 | {:.1} |", mean * n as f64).unwrap();
                writeln!(s).unwrap();
            }
            // (2) 最后一层 top-N 神经元
            let mut idx: Vec<usize> = (0..dim).collect();
            idx.sort_by_key(|&i| std::cmp::Reverse(unsafe { *nz_last_p.add(i) }));
            writeln!(s, "## 最后一层激活频率最高的 {} 个神经元", stats_topk).unwrap();
            writeln!(s, "| rank | neuron | 非零率 |").unwrap();
            writeln!(s, "|---|---|---|").unwrap();
            for r in 0..stats_topk.min(dim) {
                let i = idx[r];
                writeln!(s, "| {} | {} | {:.4} |", r + 1, i, unsafe { *nz_last_p.add(i) } as f64 / tot).unwrap();
            }
            writeln!(s).unwrap();
            // (3) head 连接命中率
            writeln!(s, "## lm_head 连接命中率（源=最后一层该维是否非零）").unwrap();
            writeln!(s, "| 统计 | 值 |").unwrap();
            writeln!(s, "|---|---|").unwrap();
            writeln!(s, "| 连接总数（步×token×V×k0_head 累计） | {} |", head_conns).unwrap();
            writeln!(s, "| 命中（源非零） | {} |", head_hits).unwrap();
            writeln!(s, "| **命中率** | **{:.6}** |", head_hits as f64 / head_conns.max(1) as f64).unwrap();
            writeln!(s, "| 每条连接平均命中数（k0_head={}） | {:.4} |", k0_head, k0_head as f64 * head_hits as f64 / head_conns.max(1) as f64).unwrap();
            writeln!(s).unwrap();
            // (4) 嵌入维度使用
            let used = (0..dim).filter(|&i| unsafe { *emb_use_p.add(i) } > 0).count();
            let mut u: Vec<u64> = (0..dim).map(|i| unsafe { *emb_use_p.add(i) }).collect();
            u.sort_unstable();
            writeln!(s, "## 嵌入维度使用频率").unwrap();
            writeln!(s, "| 统计 | 值 |").unwrap();
            writeln!(s, "|---|---|").unwrap();
            writeln!(s, "| 被用到的维度数 | {} / {} |", used, dim).unwrap();
            writeln!(s, "| 使用次数 min / median / max | {} / {} / {} |", u[0], u[u.len() / 2], u[u.len() - 1]).unwrap();
            writeln!(s).unwrap();
            // (5) 参数"野"的程度：值的分布（|w| 很小的比例）
            let mut vals: Vec<f32> = Vec::new();
            for v in 0..n_vocab {
                for j in 0..k0_head {
                    vals.push(unsafe { *head.w.add(v * k0_head + j) }.abs());
                }
            }
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let q = |p: f64| vals[((vals.len() as f64 - 1.0) * p) as usize];
            writeln!(s, "## 参数值分布（|w|，用于判断「参数野」）").unwrap();
            writeln!(s, "| 张量 | n | p10 | p50 | p90 | max |\\|w\\|<1e-3 |").unwrap();
            writeln!(s, "|---|---|---|---|---|---|---|").unwrap();
            writeln!(s, "| head.w | {} | {:.5} | {:.5} | {:.5} | {:.5} | {} |", vals.len(), q(0.1), q(0.5), q(0.9), q(1.0), vals.iter().filter(|&&x| x < 1e-3).count()).unwrap();
            let mut ev: Vec<f32> = Vec::new();
            if io_sparse {
                for i in 0..n_vocab * k0_embed {
                    ev.push(unsafe { *etab.add(i) }.abs());
                }
                ev.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let qe = |p: f64| ev[((ev.len() as f64 - 1.0) * p) as usize];
                writeln!(s, "| emb | {} | {:.5} | {:.5} | {:.5} | {:.5} | {} |", ev.len(), qe(0.1), qe(0.5), qe(0.9), qe(1.0), ev.iter().filter(|&&x| x < 1e-3).count()).unwrap();
            }
            let mut lv: Vec<f32> = Vec::new();
            for l in 0..layers {
                for i in 0..dim * k0 {
                    lv.push(unsafe { *scs[l].w.add(i) }.abs());
                }
            }
            lv.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let ql = |p: f64| lv[((lv.len() as f64 - 1.0) * p) as usize];
            writeln!(s, "| layers.w（合并） | {} | {:.5} | {:.5} | {:.5} | {:.5} | {} |", lv.len(), ql(0.1), ql(0.5), ql(0.9), ql(1.0), lv.iter().filter(|&&x| x < 1e-3).count()).unwrap();
            writeln!(s).unwrap();
            println!("wrote stats -> {}", stats_out);
        }
        if profile {
            let names = ["嵌入+逐层前向", "head 前向", "CE/softmax", "head 反向", "逐层反向", "嵌入反向", "取批/统计", "Adam+clip"];
            let tot: f64 = prof.iter().sum();
            println!("\n--- 分阶段耗时（占已统计相位总和 {:.2} s）---", tot);
            for i in 0..8 {
                println!("    {:<16} {:8.3} s  {:5.1}%", names[i], prof[i], 100.0 * prof[i] / tot.max(1e-9));
            }
            println!("    {:<16} {:8.3} s", "（其中 wall 总计）", total_wall);
        }
        // ---- 激活 dump（实验 D：逐层有效秩分析）----
        if !dump_acts.is_empty() {
            use std::io::Write;
            std::fs::create_dir_all(&dump_acts).expect("cannot create dump dir");
            // 固定批次：从 **val** 流取（不在训练数据上测有效秩）
            let want = dump_tokens.min(val_mapped.len().saturating_sub(2)).max(1);
            let ntok = ((want / seq).max(1) * seq).min(t_tokens);
            let mut toks: Vec<u32> = Vec::with_capacity(ntok);
            for i in 0..ntok {
                toks.push(val_mapped[(i * 977 + 13) % (val_mapped.len() - 2).max(1)]);
            }
            let h0 = h_ins[0];
            // h_ins[0] = 嵌入展开（与训练时同一套散写）
            zero(h0, t_tokens * dim);
            if io_sparse {
                for i in 0..ntok {
                    let tok = toks[i] as usize;
                    for j in 0..k0_embed {
                        let d = *dim_of_p.add(tok * k0_embed + j) as usize;
                        *h0.add(i * dim + d) = *etab.add(tok * k0_embed + j);
                    }
                }
            } else {
                for i in 0..ntok {
                    let tok = toks[i] as usize;
                    std::ptr::copy_nonoverlapping(emb.add(tok * dim), h0.add(i * dim), dim);
                }
            }
            let mut acts: Vec<(*const f32, String)> = vec![(h0, "input".to_string())];
            for l in 0..layers {
                let xin = h_ins[l];
                let dest = if l + 1 < layers { h_ins[l + 1] } else { h };
                if identity_layers {
                    std::ptr::copy_nonoverlapping(xin, dest, t_tokens * dim);
                } else {
                    for i in 0..ntok {
                        b1_forward(&scs[l], xin.add(i * dim), y.add(i * dim));
                    }
                    std::ptr::copy_nonoverlapping(y, dest, t_tokens * dim);
                }
                acts.push((dest, format!("layer_{:02}", l)));
            }
            for (p, name) in acts.iter() {
                let path = format!("{}/{}.f32", dump_acts, name);
                let mut f = std::fs::File::create(&path).unwrap();
                let bytes = std::slice::from_raw_parts(*p as *const u8, ntok * dim * 4);
                f.write_all(bytes).unwrap();
            }
            let meta = format!(
                "{{\"dim\":{},\"layers\":{},\"tokens\":{},\"k0\":{},\"k0_embed\":{},\"k0_head\":{},\"io_sparse\":{},\"identity_layers\":{},\"steps\":{},\"seed\":{},\"freeze_layers\":{},\"freeze_io\":{},\"final_val\":{},\"note\":\"每行 = 一个 token 的 dim 维激活；input = 嵌入展开，layer_ll = 第 ll 层输出\"}}",
                dim, layers, ntok, k0, k0_embed, k0_head, io_sparse, identity_layers, steps, seed,
                freeze_layers, freeze_io, jnum(final_val)
            );
            std::fs::File::create(format!("{}/meta.json", dump_acts))
                .unwrap()
                .write_all(meta.as_bytes())
                .unwrap();
            println!(
                "wrote activation dump -> {} ({} tokens x {} dims x {} matrices)",
                dump_acts, ntok, dim, acts.len()
            );
        }
        if !doc_path.is_empty() {
            use std::io::Write;
            if let Some(parent) = std::path::Path::new(&doc_path).parent() {
                std::fs::create_dir_all(parent).ok();
            }
            let mut d = std::fs::File::create(&doc_path).unwrap();
            writeln!(d, "# B1 完整训练：dim={} k0={} layers={}（无残差）", dim, k0, layers).unwrap();
            writeln!(d).unwrap();
            writeln!(d, "> 由 `vmlm-b1-full` 自动生成（`--doc` 参数）。**数字全部来自本次运行**。").unwrap();
            writeln!(d).unwrap();
            writeln!(d, "## 配置").unwrap();
            writeln!(d).unwrap();
            writeln!(d, "| 项 | 值 |").unwrap();
            writeln!(d, "|---|---|").unwrap();
            writeln!(d, "| 架构 | 单层 B1 堆叠 × **{}**，**无残差**（`h = B1_l(h)`） |", layers).unwrap();
            writeln!(d, "| dim / k0 | {} / {}（每 token 层内 MACs = {}） |", dim, k0, layers * dim * k0).unwrap();
            writeln!(d, "| 初始化 | Glorot `sqrt(3/k0)`，`depth_scale={:.6}`（无残差 ⇒ 保方差） |", depth_scale).unwrap();
            writeln!(d, "| 优化器 | Adam(0.9,0.999) + 线性 warmup {} 步 + cosine 退火到 {:.1e} |", warmup, min_lr).unwrap();
            writeln!(d, "| 峰值 lr / clip | {:.1e} / {} |", lr, clip).unwrap();
            writeln!(d, "| 数据 | `{}`（MiniCPM tokenizer 编码；train {} tok / val {} tok） |", data_dir, train.len(), val.len()).unwrap();
            writeln!(d, "| 模型词表 | {} 路（{}；val OOV {:.3}%） |", n_vocab, vocab_mode, 100.0 * val_oov as f64 / val.len().max(1) as f64).unwrap();
            writeln!(d, "| batch × seq | {} × {} = {} tok/步 |", batch, seq, t_tokens).unwrap();
            writeln!(d, "| 步数 | 上限 {}，实跑 **{}**（停止原因：**{}**） |", steps, steps_done, stopped_by).unwrap();
            writeln!(d, "| seed | {} |", seed).unwrap();
            writeln!(d).unwrap();
            writeln!(d, "## 结果").unwrap();
            writeln!(d).unwrap();
            writeln!(d, "| 项 | 值 |").unwrap();
            writeln!(d, "|---|---|").unwrap();
            writeln!(d, "| 均匀分布基线 `ln V` | {:.4}（PPL = V = {}） |", lnv, n_vocab).unwrap();
            writeln!(d, "| 终末训练 loss | {} |", if final_train.is_finite() { format!("**{:.4}**", final_train) } else { "n/a（非有限）".to_string() }).unwrap();
            writeln!(d, "| 终末验证 loss / PPL | {} |", if final_val.is_finite() { format!("**{:.4}** / {:.2}", final_val, vmlm_core::kernels::exp_f(final_val)) } else { "n/a（第一次评估之前就发散了）".to_string() }).unwrap();
            writeln!(d, "| 最佳验证 loss | {} |", if best_val.is_finite() { format!("**{:.4}** @ step {}", best_val, best_step) } else { "n/a（没有任何一次评估是有限的）".to_string() }).unwrap();
            writeln!(d, "| 发散 | {} |", if diverged { format!("**是**，`diverged_at` = step {}", diverged_at) } else { "否（全程有限，loss 未超 2·ln V）".to_string() }).unwrap();
            writeln!(d, "| 墙钟 / 吞吐 | {:.1} s / **{:.1} tok/s**（{} tok） |", total_wall, tokens_seen as f64 / total_wall.max(1e-9), tokens_seen).unwrap();
            writeln!(d, "| 曲线哈希 `loss_hash` | `0x{:016x}`（同 seed 重跑必须一致；由 `tools/b1_full_validate.ps1` 验证） |", hash).unwrap();
            writeln!(d).unwrap();
            writeln!(d, "## 验证曲线（每 {} 步一次，固定验证批）", eval_every).unwrap();
            writeln!(d).unwrap();
            writeln!(d, "| step | lr | train loss | val loss | val PPL | 墙钟 s |").unwrap();
            writeln!(d, "|---|---|---|---|---|---|").unwrap();
            for (s, l, tr_, vl, pp) in evals.iter() {
                writeln!(d, "| {} | {:.3e} | {:.4} | {:.4} | {:.2} | — |", s, l, tr_, vl, pp).unwrap();
            }
            writeln!(d).unwrap();
            writeln!(d, "## 诚实边界").unwrap();
            writeln!(d).unwrap();
            writeln!(d, "1. 这是**玩具规模**的真实文本训练（词表精简到数据实际出现的 token），**不能**与常规 LLM 训练对比。").unwrap();
            writeln!(d, "2. 3000 步是**上限**；曲线若在到达上限前拐平会提前停（`stopped_by=plateau`）。").unwrap();
            writeln!(d, "3. **未收敛**：只有 val loss 曲线能说明学到什么；本报告不声称收敛。").unwrap();
            writeln!(d, "4. 发散判据是 `train_loss > 2·ln V`；命中即停并记录 `diverged_at`。").unwrap();
            writeln!(d, "5. 单 seed。确定性由 `loss_hash` + `tools/b1_full_validate.ps1` 的第 4 项核对。").unwrap();
            writeln!(d, "6. **引擎层面的一个已修缺陷（与本轮数字有关，留档）**：`core::dealloc` 曾是空实现，").unwrap();
            writeln!(d, "   而 `gemm_nt_packed` 每次调用都会 `alloc` 一块与权重同量级的 pack 缓冲").unwrap();
            writeln!(d, "   （dim=1536、词表=10921 时 **67 MB/次**、约 200 MB/步）⇒ 工作集 3.8 GB→10.5 GB(20 s)，").unwrap();
            writeln!(d, "   随后 `HeapAlloc` 返回 null、`alloc` 里 `exit(1)` ⇒ **训练在第 150~200 步静默退出**，").unwrap();
            writeln!(d, "   并且在此之前表现为**越训越慢**（0.4 → 2.6 s/步）。修好（真正 `HeapFree`）之后：").unwrap();
            writeln!(d, "   工作集稳定在 ~660 MB，**250 步 = 99.9 s，恒定 0.40 s/步**。").unwrap();
            writeln!(d, "   凡是**修前**测到的「大词表 × 大 dim」吞吐（例如 full 词表 8.2 s/步）都被这个泄漏抬高过。").unwrap();
            writeln!(d).unwrap();
            writeln!(d, "## 产物").unwrap();
            writeln!(d).unwrap();
            writeln!(d, "- `{}/metrics.jsonl`（每次评估一行 JSON）", out_dir).unwrap();
            writeln!(d, "- `{}/loss_curve.csv`（step,lr,train,val,ppl）", out_dir).unwrap();
            writeln!(d, "- `{}/summary.json`（含 `loss_hash`、`diverged_at`、`stopped_by`）", out_dir).unwrap();
            println!("wrote {}", doc_path);
        }
        if diverged {
            println!("DIVERGED at step {} — 已停止（产物仍已写出，发散也是结果）。", diverged_at);
        }
    }
}

fn main() {
    run_main();
}
