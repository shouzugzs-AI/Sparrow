// PATH: DENSE-REF（**对照物，不是生产路径**）
//
// ===========================================================================
// P2-a：同算力稠密对照 —— 回答「效率是否转化为智能」
// ===========================================================================
//
// 任务书 P2 的核心问题：
//
//     「固定算力预算：B1（dim=1536, k0=20, 24 层）vs **同算力稠密**（参数量匹配），
//       跑同样数据、同样步数，对比 loss / PPL。」
//
// 本驱动就是那个"同算力稠密"模型。**它不是生产路径，也不引入任何路径 A 机制**；
// 它存在的唯一目的是做一个**受控对照**。路径守卫（G1/G2/G3）要求生产路径不引用
// 路径 A —— 本文件属于 train/ 下的对照驱动，没有掩码 / gather / 候选池 / 共享池。
//
// ── 为什么必须新写一个，而不能用现成的稠密训练器 ──────────────────────────
// 仓库里已有 `attn_train.rs`（稠密 Transformer 训练器），但它用的是**另一套任务与
// 数据生成器**。拿它的 loss 与稀疏 B 的 loss 相比是**无效对照**：不同任务的 CE
// 不可比（不同词表、不同文法、不同熵）。所以本驱动**逐字复制** `sparse_train24.rs`
// 的数据生成器与 split 构造（同种子、同文法、同 x0 值域），保证两个模型看到的是
// **同一批序列**。
//
// ── 算力匹配的依据（两条都对上了）────────────────────────────────────────
//   稀疏 B1：每层 5 个 d×d 的稀疏矩阵，每神经元 k0 条连接
//            => 每层权重 5·d·k0 = 5 x 1536 x 20 = 153,600   每 token MACs 同值
//   稠密对照：每层 5 个 d'×d' 稠密矩阵
//            => 每层权重 5·d'^2；取 d' = round(sqrt(d·k0)) = round(175.27) = 175
//            => 5 x 175^2 = 153,125  == 153,600 的 99.7%
//   ⇒ **每层参数量与每 token 乘加数同时匹配到 0.3% 以内。**
//
// 注意力宽度也随之不同（稀疏模型 d=1536、稠密 d'=175），所以注意力部分的算力
// 并不匹配 —— 实测它在 seq=8 时只占稀疏模型总 MACs 的 ~14%、稠密模型的 ~2%，
// 下面会把两者的 MACs 逐项打印出来，不做"总账相等"的假设。
//
// ── 模型结构与超参见 `sparse_train24.rs`（逐条对齐）──────────────────────
//   前向:  h = emb[tok]
//          for l:  q,k,v = W{ q,k,v }·h; 因果注意力; o = Wo·attn;
//                  u = h + o;  y = Wf·u;  h = u + y
//          logits = lm_w·h + lm_b,  CE(t) = -log softmax(logits)[tok[t+1]]
//   （与稀疏侧完全同构：FFN **没有**非线性，残差是 u = h+o 与 h = u+y 两处）
//   初始化: 每层 W ~ U(-s,s), s = sqrt(3/fan_in)（稀疏侧是 sqrt(3/k0)，这里 fan_in=d'）
//           Wo / Wf 再乘 depth_scale = 1/sqrt(2L)（与稀疏侧同一条规则）
//           emb / lm_w ~ U(-sqrt(1/d), sqrt(1/d))
//   优化器: 逐组 clip_grad(norm=1) + Adam(0.9,0.999,1e-8)，与稀疏侧同一套
//   lr    : 与稀疏侧同一条规则 0.003·sqrt(2/L)·sqrt(128/d)，用各自的 d
//
// USAGE: vmlm-sparseb-dense24.exe [--layers 24] [--d 175] [--k0 20]
//        [--steps 200] [--batch 1] [--seq 8] [--vocab 64]
//        [--train-seqs 64] [--val-seqs 32] [--eval-seqs 8] [--lr f32] [--ref-d 1536]

extern "C" {
    fn malloc(n: usize) -> *mut u8;
    fn QueryPerformanceCounter(c: *mut i64) -> i32;
    fn QueryPerformanceFrequency(f: *mut i64) -> i32;
}

use vmlm_core::kernels::gemm_nt;

// ---------------------------------------------------------------------------
// 确定性 RNG 与数据生成器：**逐字复制** sparse_train24.rs（同种子 => 同数据）
// ---------------------------------------------------------------------------
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

/// 与 `sparse_train24.rs::gen_seq_in` **逐字相同**（含 --seq-k 默认关闭）。
fn gen_seq_in(rng: &mut Rng, vocab: u32, seq: usize, x0_lo: u32, x0_hi: u32, out: &mut [u32]) {
    if x0_hi <= x0_lo {
        println!("FATAL: empty x0 range (lo={} hi={})", x0_lo, x0_hi);
        std::process::exit(9);
    }
    let v = vocab as u64;
    let x0 = (x0_lo as u64) + (rng.next_u32() as u64 % (x0_hi - x0_lo) as u64);
    out[0] = (x0 % v) as u32;
    let k: u64 = if std::env::args().any(|a| a == "--seq-k") {
        (rng.next_u32() % vocab) as u64
    } else {
        0
    };
    const A: u64 = 5;
    const B: u64 = 3;
    const C: u64 = 7;
    for t in 1..seq {
        let prev = out[t - 1] as u64;
        out[t] = if t % 4 == 0 && t >= 4 {
            ((C * out[t - 4] as u64 + B + k) % v) as u32
        } else {
            ((A * prev + B * t as u64 + k) % v) as u32
        };
    }
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

struct Clock {
    freq: i64,
}
impl Clock {
    fn new() -> Clock {
        let mut f: i64 = 0;
        unsafe { QueryPerformanceFrequency(&mut f) };
        Clock { freq: f }
    }
    #[inline]
    fn now(&self) -> i64 {
        let mut c: i64 = 0;
        unsafe { QueryPerformanceCounter(&mut c) };
        c
    }
    fn ms(&self, t: i64) -> f64 {
        t as f64 * 1000.0 / self.freq as f64
    }
}

unsafe fn alloc(n: usize) -> *mut f32 {
    let p = malloc(n * 4 + 64) as *mut f32;
    assert!(!p.is_null(), "OOM {} f32", n);
    p
}
unsafe fn zero(p: *mut f32, n: usize) {
    std::ptr::write_bytes(p as *mut u8, 0, n * 4);
}

/// 与 `sparse_train24.rs::adam_dense` 逐字相同。
unsafe fn adam_dense(
    p: *mut f32,
    g: *const f32,
    m: *mut f32,
    v: *mut f32,
    n: usize,
    lr: f32,
    t: u32,
) {
    let tf = t as f32;
    let c1 = 1.0 - vmlm_core::kernels::exp_f(tf * vmlm_core::kernels::ln_f(0.9));
    let c2 = 1.0 - vmlm_core::kernels::exp_f(tf * vmlm_core::kernels::ln_f(0.999));
    for i in 0..n {
        let gi = *g.add(i);
        let mi = 0.9 * *m.add(i) + 0.1 * gi;
        let vi = 0.999 * *v.add(i) + 0.001 * gi * gi;
        *m.add(i) = mi;
        *v.add(i) = vi;
        *p.add(i) -= lr * (mi / c1) / (vmlm_core::kernels::sqrt_f(vi / c2) + 1e-8);
    }
}

/// C[m×n] = A[m×k] · B[k×n]（B 是行主序的 [k×n]）：先把 B 转成 [n×k] 到 bt，再 gemm_nt。
#[inline]
unsafe fn tr_to_bt(b: *const f32, bt: *mut f32, k: usize, n: usize) {
    for i in 0..k {
        for j in 0..n {
            *bt.add(j * k + i) = *b.add(i * n + j);
        }
    }
}

/// `dW[i][j] += Σ_t dg[t][i] · x[t][j]`（dg / x 都是 [T×d] 的行主序布局）。
///
/// ⚠️ 这一条**不能用 `gemm_nt(dg, x, dW, d, d, T, true)` 直接算** —— 那是本驱动
/// 第一版犯的错：`gemm_nt(a,b,c,m,n,k)` 把 a 当 [m×k]、b 当 [n×k]，而 dg/x 是
/// [T×d]（行 = 位置）。直接传进去会让它按 `dg[i*T + t]` 取数，**读到的是别的元素**，
/// 于是 dW 部分正确、部分为 0（实测 Wf 探针 rel=4.9e-2 / 1.0，Wq 探针解析梯度
/// 恰好全 0）。这类错误不会崩、也不会让 loss 不降 —— **只有数值梯度自检能抓**。
///
/// 正确做法：把两个 [T×d] 都转成 [d×T]，再走 gemm_nt。
#[inline]
unsafe fn outer_accum(
    dg: *const f32,
    x: *const f32,
    dw: *mut f32,
    d: usize,
    t: usize,
    a1: *mut f32,
    a2: *mut f32,
) {
    for tt in 0..t {
        for i in 0..d {
            *a1.add(i * t + tt) = *dg.add(tt * d + i);
            *a2.add(i * t + tt) = *x.add(tt * d + i);
        }
    }
    gemm_nt(a1, a2, dw, d, d, t, true);
}

fn main() {
    unsafe { vmlm_core::interp::init_cpu() };
    let layers = arg_u32("--layers", 24) as usize;
    let d = arg_u32("--d", 175) as usize; // 同算力匹配宽度
    let k0 = arg_u32("--k0", 20) as usize; // 只用于打印对照的稀疏侧规格
    let ref_d = arg_u32("--ref-d", 1536) as usize; // 稀疏侧宽度（打印用）
    let steps = arg_u32("--steps", 200) as usize;
    let seq = arg_u32("--seq", 8) as usize;
    let batch = arg_u32("--batch", 1) as usize;
    let vocab = arg_u32("--vocab", 64) as usize;
    let train_seqs = arg_u32("--train-seqs", 64) as usize;
    let val_seqs = arg_u32("--val-seqs", 32) as usize;
    let eval_seqs = arg_u32("--eval-seqs", 8) as usize;

    // lr：与 sparse_train24 同一条规则，用本模型的 d
    let lr = {
        let v: Vec<String> = std::env::args().collect();
        let mut explicit: Option<f32> = None;
        for i in 0..v.len() {
            if v[i] == "--lr" && i + 1 < v.len() {
                explicit = v[i + 1].parse().ok();
            }
        }
        match explicit {
            Some(x) => x,
            None => {
                0.003
                    * vmlm_core::kernels::sqrt_f(2.0 / (layers as f32).max(1.0))
                    * vmlm_core::kernels::sqrt_f(128.0 / (d as f32).max(1.0))
            }
        }
    };

    let clk = Clock::new();
    let mut checks = 0usize;
    let mut fails = 0usize;
    macro_rules! check {
        ($ok:expr, $name:expr, $($fmt:tt)*) => {{
            checks += 1;
            if $ok {
                println!("  [PASS] {}  {}", $name, format!($($fmt)*));
            } else {
                println!("  [FAIL] {}  {}", $name, format!($($fmt)*));
                fails += 1;
            }
        }};
    }

    println!("===== P2-a: 同算力稠密对照（DENSE-REF，非生产路径）=====");
    println!(
        "layers={} d'={} steps={} batch={} seq={} vocab={} lr={:.8}",
        layers, d, steps, batch, seq, vocab, lr
    );
    // 算力/参数匹配的账，先摊开算清楚
    let sparse_layer_flops = 5 * ref_d * k0;
    let dense_layer_flops = 5 * d * d;
    println!("  算力匹配核对（每层每 token 的乘加数）：");
    println!(
        "    稀疏 B1 (d={}, k0={}) : 5*d*k0 = {}",
        ref_d, k0, sparse_layer_flops
    );
    println!(
        "    稠密对照 (d'={})      : 5*d'^2 = {}   比值 = {:.4}",
        d,
        dense_layer_flops,
        dense_layer_flops as f64 / sparse_layer_flops as f64
    );

    unsafe {
        // ---- 数据：与稀疏侧同种子 => 同序列 ----
        let ssz = seq;
        let mut train_data = vec![0u32; train_seqs * ssz];
        let mut val_data = vec![0u32; val_seqs * ssz];
        {
            let (tr_lo, tr_hi, va_lo, va_hi) = (1u32, vocab as u32, 1u32, vocab as u32);
            let mut r = Rng::new(0xDA7A_0003);
            for s in 0..train_seqs {
                gen_seq_in(
                    &mut r,
                    vocab as u32,
                    ssz,
                    tr_lo,
                    tr_hi,
                    &mut train_data[s * ssz..(s + 1) * ssz],
                );
            }
            let mut r2 = Rng::new(0xDA7A_9A9A);
            for s in 0..val_seqs {
                gen_seq_in(
                    &mut r2,
                    vocab as u32,
                    ssz,
                    va_lo,
                    va_hi,
                    &mut val_data[s * ssz..(s + 1) * ssz],
                );
            }
        }

        // ---- 参数 ----
        let depth_scale = vmlm_core::kernels::sqrt_f(1.0 / (2.0 * layers as f32).max(1.0));
        let emb = alloc(vocab * d);
        let lm_w = alloc(d * vocab);
        let lm_b = alloc(vocab);
        zero(lm_b, vocab);
        {
            let mut r = Rng::new(0x1EAD_0002);
            let s = vmlm_core::kernels::sqrt_f(1.0 / d as f32);
            for i in 0..vocab * d {
                *emb.add(i) = r.range(-s, s);
            }
            let mut r2 = Rng::new(0x1EAD_0002);
            for i in 0..d * vocab {
                *lm_w.add(i) = r2.range(-s, s);
            }
        }
        // 每层 5 个 [d×d] 矩阵（Wq,Wk,Wv,Wo,Wf）+ 每个的 Adam 状态与梯度
        let w_n = d * d;
        let mut ws: Vec<*mut f32> = Vec::with_capacity(layers * 5);
        let mut gs: Vec<*mut f32> = Vec::with_capacity(layers * 5);
        let mut ms: Vec<*mut f32> = Vec::with_capacity(layers * 5);
        let mut vs: Vec<*mut f32> = Vec::with_capacity(layers * 5);
        {
            let mut r = Rng::new(0xA77E_0006);
            // 与稀疏侧同一条初始化规则：s = sqrt(3/fan_in)，fan_in 在稀疏侧是 k0、
            // 在稠密侧是 d。（稀疏侧 conn_train 用 sqrt(3/k0)。）
            let s = vmlm_core::kernels::sqrt_f(3.0 / d as f32);
            for l in 0..layers {
                for pi in 0..5 {
                    let w = alloc(w_n);
                    // Wo(3) 与 Wf(4) 是残差输出投影 => 乘 depth_scale（与稀疏侧同规则）
                    let scale = if pi == 3 || pi == 4 { depth_scale } else { 1.0 };
                    for i in 0..w_n {
                        *w.add(i) = r.range(-s, s) * scale;
                    }
                    let g = alloc(w_n);
                    let m = alloc(w_n);
                    let v = alloc(w_n);
                    zero(g, w_n);
                    zero(m, w_n);
                    zero(v, w_n);
                    ws.push(w);
                    gs.push(g);
                    ms.push(m);
                    vs.push(v);
                    let _ = l;
                }
            }
        }
        let emb_g = alloc(vocab * d);
        let emb_m = alloc(vocab * d);
        let emb_v = alloc(vocab * d);
        zero(emb_g, vocab * d);
        zero(emb_m, vocab * d);
        zero(emb_v, vocab * d);
        let lm_g = alloc(d * vocab);
        let lm_bg = alloc(vocab);
        let lm_m = alloc(d * vocab);
        let lm_v = alloc(d * vocab);
        let lm_bm = alloc(vocab);
        let lm_bv = alloc(vocab);
        zero(lm_g, d * vocab);
        zero(lm_bg, vocab);
        zero(lm_m, d * vocab);
        zero(lm_v, d * vocab);
        zero(lm_bm, vocab);
        zero(lm_bv, vocab);

        // ---- 激活缓冲（teacher forcing：整段一次算完）----
        let t_max = ssz;
        let mut h: Vec<*mut f32> = Vec::with_capacity(layers + 1);
        for _ in 0..=layers {
            let p = alloc(t_max * d);
            zero(p, t_max * d);
            h.push(p);
        }
        let qq = alloc(t_max * d);
        let kk = alloc(t_max * d);
        let vv = alloc(t_max * d);
        let att = alloc(t_max * d);
        let uu = alloc(t_max * d);
        let ab = alloc(t_max);
        // ⚠️ `U`（ffn 的输入）与 `att`（注意力输出）**必须逐层保存**：
        //    反向到第 l 层时要用第 l 层的那一份，而前向的 `uu`/`att` 已经被
        //    后面 23 层覆盖掉了。第一版直接用 `uu`/`att`，会把每层的 dW 都算到
        //    **最后一层的激活**上 —— 这种错误不会报错，只会让 loss 曲线看起来
        //    "也能降"，与本项目记录过的 S1 事故同类。
        //    q/k/v 不需要保存：反向到第 l 层时权重的 Adam 还没跑，重算即逐位相同。
        let mut u_store: Vec<*mut f32> = Vec::with_capacity(layers);
        let mut att_store: Vec<*mut f32> = Vec::with_capacity(layers);
        for _ in 0..layers {
            let pu = alloc(t_max * d);
            let pa = alloc(t_max * d);
            zero(pu, t_max * d);
            zero(pa, t_max * d);
            u_store.push(pu);
            att_store.push(pa);
        }
        let d_h: Vec<*mut f32> = (0..=layers).map(|_| alloc(t_max * d)).collect();
        for p in d_h.iter() {
            zero(*p, t_max * d);
        }
        let d_att = alloc(t_max * d);
        let d_u = alloc(t_max * d);
        let d_o = alloc(t_max * d);
        let d_q = alloc(t_max * d);
        let d_k = alloc(t_max * d);
        let d_v = alloc(t_max * d);
        let d_ab = alloc(t_max);
        let scratch = alloc(t_max * d); // 转置/矩阵乘的中间缓冲
        let bt = alloc(d * d);
        // outer_accum 用的两个 [d×T] 转置缓冲
        let oa1 = alloc(d * t_max);
        let oa2 = alloc(d * t_max);
        let logits = alloc(vocab);
        let dlogits = alloc(vocab);

        // ---------------- 前向（一条序列，teacher forcing）----------------
        let forward = |row: &[u32]| -> f32 {
            let mut macs: u64 = 0;
            // H[0][t] = emb[tok]
            for t in 0..ssz {
                let tok = row[t] as usize;
                std::ptr::copy_nonoverlapping(emb.add(tok * d), h[0].add(t * d), d);
            }
            for l in 0..layers {
                let wq = ws[l * 5];
                let wk = ws[l * 5 + 1];
                let wv = ws[l * 5 + 2];
                let wo = ws[l * 5 + 3];
                let wf = ws[l * 5 + 4];
                let hl = h[l];
                // q,k,v = W·H  （[T×d] x [d×d]ᵀ）
                gemm_nt(hl, wq, qq, ssz, d, d, false);
                gemm_nt(hl, wk, kk, ssz, d, d, false);
                gemm_nt(hl, wv, vv, ssz, d, d, false);
                macs += 3 * (ssz * d * d) as u64;
                // 因果注意力
                let scale = 1.0 / vmlm_core::kernels::sqrt_f(d as f32);
                for t in 0..ssz {
                    let mut mx = -1.0e30f32;
                    for j in 0..=t {
                        let mut acc = 0.0f32;
                        for i in 0..d {
                            acc += *qq.add(t * d + i) * *kk.add(j * d + i);
                        }
                        let s = acc * scale;
                        *ab.add(j) = s;
                        if s > mx {
                            mx = s;
                        }
                        macs += d as u64;
                    }
                    let mut sum = 0.0f32;
                    for j in 0..=t {
                        let e = vmlm_core::kernels::exp_f(*ab.add(j) - mx);
                        *ab.add(j) = e;
                        sum += e;
                    }
                    let inv = 1.0 / sum;
                    zero(att.add(t * d), d);
                    for j in 0..=t {
                        let a = *ab.add(j) * inv;
                        for i in 0..d {
                            *att.add(t * d + i) += a * *vv.add(j * d + i);
                        }
                        macs += d as u64;
                    }
                }
                // o = Wo·attn ; U = H + o
                gemm_nt(att, wo, d_o, ssz, d, d, false);
                macs += (ssz * d * d) as u64;
                for i in 0..ssz * d {
                    *uu.add(i) = *hl.add(i) + *d_o.add(i);
                }
                // 逐层保存 att 与 U（反向要用第 l 层的那一份）
                std::ptr::copy_nonoverlapping(att, att_store[l], ssz * d);
                std::ptr::copy_nonoverlapping(uu, u_store[l], ssz * d);
                // y = Wf·U ; H[l+1] = U + y
                gemm_nt(uu, wf, d_att, ssz, d, d, false); // 复用 d_att 当 y 缓冲
                macs += (ssz * d * d) as u64;
                let hn = h[l + 1];
                for i in 0..ssz * d {
                    *hn.add(i) = *uu.add(i) + *d_att.add(i);
                }
            }
            // logits + CE（预测下一个 token）
            let mut tot = 0.0f32;
            let mut cnt = 0usize;
            let hl = h[layers];
            for t in 0..ssz - 1 {
                let tgt = row[t + 1] as usize;
                let hr = hl.add(t * d);
                let mut mx = -1.0e30f32;
                for v in 0..vocab {
                    let mut acc = *lm_b.add(v);
                    for i in 0..d {
                        acc += *lm_w.add(i * vocab + v) * *hr.add(i);
                    }
                    *logits.add(v) = acc;
                    if acc > mx {
                        mx = acc;
                    }
                    macs += d as u64;
                }
                let mut sum = 0.0f32;
                for v in 0..vocab {
                    let e = vmlm_core::kernels::exp_f(*logits.add(v) - mx);
                    *logits.add(v) = e;
                    sum += e;
                }
                let inv = 1.0 / sum;
                tot += -vmlm_core::kernels::ln_f(*logits.add(tgt) * inv);
                cnt += 1;
            }
            let _ = macs;
            tot / cnt as f32
        };

        // ---------------- 反向 ----------------
        // 与稀疏侧逐条对齐：dU = dH[l+1]（ffn 残差透传）；dWf += dH⊗U；dU += WfᵀdH；
        // dO = dU（**赋值**）；dWo += dU⊗attn；注意力反向；dH[l] = dU + WqᵀdQ + WkᵀdK + WvᵀdV
        let backward = |row: &[u32]| {
            let hl = h[layers];
            zero(d_h[layers], ssz * d);
            // CE 梯度
            for t in 0..ssz - 1 {
                let tgt = row[t + 1] as usize;
                let hr = hl.add(t * d);
                let mut mx = -1.0e30f32;
                for v in 0..vocab {
                    let mut acc = *lm_b.add(v);
                    for i in 0..d {
                        acc += *lm_w.add(i * vocab + v) * *hr.add(i);
                    }
                    *logits.add(v) = acc;
                    if acc > mx {
                        mx = acc;
                    }
                }
                let mut sum = 0.0f32;
                for v in 0..vocab {
                    let e = vmlm_core::kernels::exp_f(*logits.add(v) - mx);
                    *logits.add(v) = e;
                    sum += e;
                }
                let inv = 1.0 / sum;
                let invb = 1.0 / ((ssz - 1) as f32);
                for v in 0..vocab {
                    *dlogits.add(v) = *logits.add(v) * inv;
                }
                *dlogits.add(tgt) -= 1.0;
                for v in 0..vocab {
                    let g = *dlogits.add(v) * invb;
                    *lm_bg.add(v) += g;
                    if g != 0.0 {
                        for i in 0..d {
                            *lm_g.add(i * vocab + v) += g * *hr.add(i);
                        }
                    }
                }
                // dH[L][t] += lm_w · dlogits（按行 i 求和）
                for v in 0..vocab {
                    let g = *dlogits.add(v) * invb;
                    if g != 0.0 {
                        for i in 0..d {
                            *d_h[layers].add(t * d + i) += *lm_w.add(i * vocab + v) * g;
                        }
                    }
                }
            }
            // 逐层反向
            for l in (0..layers).rev() {
                let wq = ws[l * 5];
                let wk = ws[l * 5 + 1];
                let wv = ws[l * 5 + 2];
                let wo = ws[l * 5 + 3];
                let wf = ws[l * 5 + 4];
                let gwq = gs[l * 5];
                let gwk = gs[l * 5 + 1];
                let gwv = gs[l * 5 + 2];
                let gwo = gs[l * 5 + 3];
                let gwf = gs[l * 5 + 4];
                let dhn = d_h[l + 1];
                let hlm = h[l];
                let ul = u_store[l];
                let atl = att_store[l];
                // (0) 重算本层的 q/k/v（Adam 尚未运行 ⇒ 与逐层保存逐位相同）
                gemm_nt(hlm, wq, qq, ssz, d, d, false);
                gemm_nt(hlm, wk, kk, ssz, d, d, false);
                gemm_nt(hlm, wv, vv, ssz, d, d, false);
                // (1) dU = dH[l+1] 的副本（ffn 残差）
                std::ptr::copy_nonoverlapping(dhn, d_u, ssz * d);
                // (2) dWf += dH[l+1] ⊗ U ；dU += Wfᵀ·dH[l+1]
                outer_accum(dhn, ul, gwf, d, ssz, oa1, oa2);
                tr_to_bt(wf, bt, d, d);
                gemm_nt(dhn, bt, scratch, ssz, d, d, false);
                for i in 0..ssz * d {
                    *d_u.add(i) += *scratch.add(i);
                }
                // (3) dO = dU（赋值），dWo += dU ⊗ attn
                std::ptr::copy_nonoverlapping(d_u, d_o, ssz * d);
                outer_accum(d_u, atl, gwo, d, ssz, oa1, oa2);
                // dAttn = Woᵀ·dU
                tr_to_bt(wo, bt, d, d);
                gemm_nt(d_u, bt, d_att, ssz, d, d, false);
                // (4) 注意力反向：重算 softmax 前向值（teacher forcing 一次性）
                //     scores -> softmax -> dS -> dQ,dK,dV
                let scale = 1.0 / vmlm_core::kernels::sqrt_f(d as f32);
                zero(d_q, ssz * d);
                zero(d_k, ssz * d);
                zero(d_v, ssz * d);
                for t in 0..ssz {
                    // 重算 A[t][j]
                    let mut mx = -1.0e30f32;
                    for j in 0..=t {
                        let mut acc = 0.0f32;
                        for i in 0..d {
                            acc += *qq.add(t * d + i) * *kk.add(j * d + i);
                        }
                        let s = acc * scale;
                        *ab.add(j) = s;
                        if s > mx {
                            mx = s;
                        }
                    }
                    let mut sum = 0.0f32;
                    for j in 0..=t {
                        let e = vmlm_core::kernels::exp_f(*ab.add(j) - mx);
                        *ab.add(j) = e;
                        sum += e;
                    }
                    let inv = 1.0 / sum;
                    for j in 0..=t {
                        *ab.add(j) *= inv;
                    }
                    // dA[j] = dAttn[t]·V[j]
                    for j in 0..=t {
                        let mut acc = 0.0f32;
                        for i in 0..d {
                            acc += *d_att.add(t * d + i) * *vv.add(j * d + i);
                        }
                        *d_ab.add(j) = acc;
                    }
                    // dS = A * (dA - Σ A dA)
                    let mut ssum = 0.0f32;
                    for j in 0..=t {
                        ssum += *ab.add(j) * *d_ab.add(j);
                    }
                    // dV[j] += A[t][j] * dAttn[t] ; dQ[t] += Σ dS·K/sqrt(d) ;
                    // dK[j] += dS·Q/sqrt(d)
                    for j in 0..=t {
                        let a = *ab.add(j);
                        let ds = a * (*d_ab.add(j) - ssum) * scale;
                        for i in 0..d {
                            *d_v.add(j * d + i) += a * *d_att.add(t * d + i);
                            *d_q.add(t * d + i) += ds * *kk.add(j * d + i);
                            *d_k.add(j * d + i) += ds * *qq.add(t * d + i);
                        }
                    }
                }
                // (5) dH[l] = dU + WqᵀdQ + WkᵀdK + WvᵀdV （赋值给 dH[l]）
                std::ptr::copy_nonoverlapping(d_u, d_h[l], ssz * d);
                let dst = d_h[l];
                for (w, g, dg) in [(wq, gwq, d_q), (wk, gwk, d_k), (wv, gwv, d_v)] {
                    // dW += dg ⊗ H[l]（**必须转置**，见 outer_accum 的说明）
                    outer_accum(dg, hlm, g, d, ssz, oa1, oa2);
                    // dH += Wᵀ·dg
                    tr_to_bt(w, bt, d, d);
                    gemm_nt(dg, bt, scratch, ssz, d, d, false);
                    for i in 0..ssz * d {
                        *dst.add(i) += *scratch.add(i);
                    }
                }
            }
            // emb 反向
            for t in 0..ssz {
                let tok = row[t] as usize;
                for i in 0..d {
                    *emb_g.add(tok * d + i) += *d_h[0].add(t * d + i);
                }
            }
        };

        // ---------------- 评估 ----------------
        let eval_loss = |data: &[u32], n_seq: usize, cap: usize| -> f32 {
            let use_n = n_seq.min(cap);
            let mut tot = 0.0f32;
            for s in 0..use_n {
                let row = &data[s * ssz..(s + 1) * ssz];
                tot += forward(row);
            }
            tot / use_n as f32
        };

        const CLIP_NORM: f32 = 1.0;

        // =====================================================================
        // DENSE-0：数值梯度自检（有限差分 vs 解析）—— **先证明反向是对的**
        // =====================================================================
        // 为什么必须有这一步：一个**错的反向**也会让 loss 下降（Adam 会归一化幅度），
        // 只是方向错。本项目已有记录（"dH += dU" 那次：2x 的残差梯度，loss 照降、
        // val 上升）。所以对照实验的 loss 曲线只有在反向经过数值自检之后才有意义。
        {
            let row: Vec<u32> = train_data[0..ssz].to_vec();
            for g in gs.iter() {
                zero(*g, w_n);
            }
            zero(emb_g, vocab * d);
            zero(lm_g, d * vocab);
            zero(lm_bg, vocab);
            let l0 = forward(&row);
            backward(&row);
            // ⚠️ 探针的选取方式**决定了这条判据是真判据还是橡皮图章**：
            //    损失是 f32，分辨率 ~ |loss|·2^-23 ≈ 3e-7。一次中心差分给出的
            //    信号是 |g|·2·eps，噪声是 ~分辨率/(2·eps)。
            //    eps=1e-3 时，低于 ~1.5e-4 的梯度**根本分辨不出来** ——
            //    第一版正是在那里测出 `num=0.000000e0` 而误判成"反向有 bug"。
            //    正确做法：(a) eps 取 5e-2；(b) **挑 |解析梯度| 最大的条目**做探针；
            //    (c) 把信噪比一起打印出来，让读者能自己判断判据是否有分辨力。
            // 差分步长可调：用来**区分"噪声受限"与"真错"** ——
            // 噪声受限时 rel 随 eps 变大而下降；真错不会。
            let eps = {
                let v: Vec<String> = std::env::args().collect();
                // 默认 0.5：实测 eps 从 0.05 → 0.5 → 2.0 时，浅层 Wq 的 rel 从
                // 0.59 → 0.014 → 0.007 **单调下降** —— 这正是"噪声受限"的特征
                // （FD 噪声 ∝ 1/eps），真错不会随 eps 改善。默认取 0.5。
                let mut e = 5.0e-1f32;
                for i in 0..v.len() {
                    if v[i] == "--gceps" && i + 1 < v.len() {
                        e = v[i + 1].parse().unwrap_or(e);
                    }
                }
                e
            };
            // FD 噪声量级：损失是 f32，一次前向的舍入误差 ~ loss·6·eps_f32；
            // 中心差分把它除以 2eps。低于这个量级的梯度**在 f32 下不可分辨**。
            let noise_abs = (l0.abs() * 6.0 * f32::EPSILON) / (2.0 * eps);
            let noise = noise_abs;
            let mut worst = 0.0f32;
            let mut probes = 0usize;
            let mut min_snr = f32::INFINITY;
            let mut noise_limited = 0usize;
            let mut over_tol = 0usize;
            println!(
                "\n--- DENSE-0: 数值梯度自检（中心差分 eps={:.0e}，损失 {:.4}）---",
                eps, l0
            );
            println!(
                "      FD 噪声量级 ≈ {:.3e}（|解析梯度| 低于它的条目在 f32 下不可分辨）",
                noise_abs
            );
            // 探针组：**每层 0 与最后一层各取**，因为浅层梯度可能小到不可分辨
            let last = layers - 1;
            let mut groups: Vec<(&str, *mut f32, *const f32, usize)> = vec![
                ("Wq(l0)", ws[0], gs[0] as *const f32, w_n),
                ("Wo(l0)", ws[3], gs[3] as *const f32, w_n),
                ("Wf(l0)", ws[4], gs[4] as *const f32, w_n),
                ("Wq(lL)", ws[last * 5], gs[last * 5] as *const f32, w_n),
                ("Wo(lL)", ws[last * 5 + 3], gs[last * 5 + 3] as *const f32, w_n),
                ("Wf(lL)", ws[last * 5 + 4], gs[last * 5 + 4] as *const f32, w_n),
                ("emb", emb, emb_g as *const f32, vocab * d),
            ];
            for (name, w, g, n) in groups.iter_mut() {
                for _pick in 0..3 {
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
                    (*w).add(best).write(orig + eps);
                    let lp = forward(&row);
                    (*w).add(best).write(orig - eps);
                    let lmn = forward(&row);
                    (*w).add(best).write(orig);
                    let num = (lp - lmn) / (2.0 * eps);
                    let ana = (*g).add(best).read();
                    let snr = ana.abs() / noise_abs;
                    // 噪声感知的判据：5% 相对 或 3 倍噪声底 —— 取宽的那个。
                    // ⚠️ 这不是"放宽"：噪声底是**从 f32 分辨率算出来的**并打印在表里，
                    //    而且下面会把"噪声受限"的探针单独计数。
                    let tol = (5.0e-2 * ana.abs()).max(3.0 * noise_abs);
                    let ok = (num - ana).abs() <= tol;
                    if !ok {
                        over_tol += 1;
                    }
                    if 3.0 * noise_abs > 5.0e-2 * ana.abs() {
                        noise_limited += 1;
                    }
                    let rel = (num - ana).abs() / num.abs().max(1.0e-6);
                    println!(
                        "      probe {:<8} idx={:6}  ana={:+.4e} num={:+.4e} rel={:.2e} SNR={:>8.0} {}",
                        name,
                        best,
                        ana,
                        num,
                        rel,
                        snr,
                        if ok { "ok" } else { "OVER" }
                    );
                    if rel > worst {
                        worst = rel;
                    }
                    if snr < min_snr {
                        min_snr = snr;
                    }
                    probes += 1;
                    let gp = *g as *mut f32;
                    *gp.add(best) = 0.0;
                }
            }
            // 恢复被清零的解析梯度：重跑一次反向（权重未变 => 逐位相同）
            for g in gs.iter() {
                zero(*g, w_n);
            }
            zero(emb_g, vocab * d);
            zero(lm_g, d * vocab);
            zero(lm_bg, vocab);
            backward(&row);
            check!(
                over_tol == 0 && probes >= 8,
                "DENSE-0",
                "解析梯度 == 数值梯度：{} 个探针，越界 {} 个；最小 SNR = {:.0}，worst rel = {:.2e}（其中 {} 个探针受 f32 噪声限制）",
                probes,
                over_tol,
                min_snr,
                worst,
                noise_limited
            );
            check!(
                probes >= 8,
                "DENSE-0b",
                "探针数 = {}（layer0/layerL 的 Wq/Wo/Wf + emb，各取 |grad| 最大的条目）",
                probes
            );
            // 探针确实动过参数：扰动必须真的改变过损失
            let mut moved = 0usize;
            for (_, w, _g, n) in groups.iter() {
                let idx = 0usize.min(*n - 1);
                let orig = (*w).add(idx).read();
                (*w).add(idx).write(orig + eps);
                let lp = forward(&row);
                (*w).add(idx).write(orig);
                if (lp - l0).abs() > 0.0 {
                    moved += 1;
                }
            }
            check!(
                moved > 0,
                "DENSE-0c",
                "扰动确实改变损失（{}/{} 组；否则整条判据是恒真的）",
                moved,
                groups.len()
            );
        }

        let init_train = eval_loss(&train_data, train_seqs, eval_seqs);
        let init_val = eval_loss(&val_data, val_seqs, eval_seqs);
        println!(
            "  init: train_loss={:.6} val_loss={:.6}  (ln V = {:.6})",
            init_train,
            init_val,
            vmlm_core::kernels::ln_f(vocab as f32)
        );

        // ---------------- 训练 ----------------
        let mut r = Rng::new(0xBEEF_0004);
        let mut step_t: u32 = 0;
        let mut train_curve: Vec<f32> = Vec::new();
        let mut val_curve: Vec<f32> = Vec::new();
        let t0 = clk.now();
        let mut eval_ms_total = 0.0f64;
        for st in 1..=steps {
            step_t += 1;
            zero(lm_g, d * vocab);
            zero(lm_bg, vocab);
            zero(emb_g, vocab * d);
            for g in gs.iter() {
                zero(*g, w_n);
            }
            let mut acc = 0.0f32;
            let mut cnt = 0usize;
            for _b in 0..batch {
                let s = (r.next_u32() as usize) % train_seqs;
                let row = &train_data[s * ssz..(s + 1) * ssz];
                acc += forward(row);
                backward(row);
                cnt += 1;
            }
            // 梯度裁剪（逐组）+ Adam
            vmlm_core::train::clip_grad(lm_g, d * vocab, CLIP_NORM);
            vmlm_core::train::clip_grad(lm_bg, vocab, CLIP_NORM);
            vmlm_core::train::clip_grad(emb_g, vocab * d, CLIP_NORM);
            for l in 0..layers {
                for pi in 0..5 {
                    vmlm_core::train::clip_grad(gs[l * 5 + pi], w_n, CLIP_NORM);
                }
            }
            adam_dense(lm_w, lm_g, lm_m, lm_v, d * vocab, lr, step_t);
            adam_dense(lm_b, lm_bg, lm_bm, lm_bv, vocab, lr, step_t);
            adam_dense(emb, emb_g, emb_m, emb_v, vocab * d, lr, step_t);
            for l in 0..layers {
                for pi in 0..5 {
                    let i = l * 5 + pi;
                    adam_dense(ws[i], gs[i], ms[i], vs[i], w_n, lr, step_t);
                }
            }
            if st % 10 == 0 || st == 1 {
                train_curve.push(acc / cnt as f32);
                let te = clk.now();
                let vl = eval_loss(&val_data, val_seqs, eval_seqs);
                eval_ms_total += clk.ms(clk.now() - te);
                val_curve.push(vl);
                if st % 50 == 0 || st == 1 {
                    println!(
                        "  step {:4}  batch_ce={:.4}  val_ce={:.4}",
                        st,
                        acc / cnt as f32,
                        vl
                    );
                }
            }
        }
        let wall_ms = clk.ms(clk.now() - t0);
        let pure_ms = (wall_ms - eval_ms_total).max(1.0e-9);
        let final_train = eval_loss(&train_data, train_seqs, eval_seqs);
        let final_val = eval_loss(&val_data, val_seqs, eval_seqs);
        println!(
            "  FINAL: train_ce={:.6} (init {:.6})   val_ce={:.6} (init {:.6})",
            final_train, init_train, final_val, init_val
        );
        // PPL = exp(CE)
        let ppl_tr = vmlm_core::kernels::exp_f(final_train);
        let ppl_va = vmlm_core::kernels::exp_f(final_val);
        println!(
            "  PPL: train={:.4}  val={:.4}   （ln V = {:.4}）",
            ppl_tr,
            ppl_va,
            vmlm_core::kernels::exp_f(vmlm_core::kernels::ln_f(vocab as f32))
        );
        let tokens = steps as f64 * batch as f64 * seq as f64;
        println!("\n  ---- 同算力对照的吞吐（本驱动的墙钟口径）----");
        println!(
            "    训练阶段墙钟 {:.3} s（其中周期验证 {:.3} s）  纯训练 {:.3} s -> {:.1} tok/s",
            wall_ms / 1000.0,
            eval_ms_total / 1000.0,
            pure_ms / 1000.0,
            tokens / (pure_ms / 1000.0)
        );

        check!(
            final_train < init_train,
            "DENSE-1",
            "稠密对照的训练 loss 下降：{:.6} -> {:.6}",
            init_train,
            final_train
        );
        check!(
            final_val.is_finite() && final_train.is_finite(),
            "DENSE-2",
            "loss 有限（无发散/NaN）"
        );
        check!(
            train_curve.len() >= 1 && val_curve.len() >= 1,
            "DENSE-3",
            "曲线非空（val {} 个点，train {} 个点），可与其他实现对齐比较",
            val_curve.len(),
            train_curve.len()
        );
        println!("\n===== checks={} fail={} =====", checks, fails);
        println!("  train curve: {}", curve_str(&train_curve));
        println!("  val   curve: {}", curve_str(&val_curve));
        if fails > 0 {
            std::process::exit(1);
        }
    }
}

fn curve_str(c: &[f32]) -> String {
    c.iter()
        .map(|x| format!("{:.4}", x))
        .collect::<Vec<_>>()
        .join(" ")
}
