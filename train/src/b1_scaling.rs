// PATH: B1
//
// ===========================================================================
// B1 Scaling Law 扫描驱动（单层）
// ===========================================================================
// 固定 128 步的算力预算下测：
//   扫描 A：k0 增长（dim=512, k0 = 1..512）
//   扫描 B：dim 增长（k0=20, dim = 32..512）
//   扫描 C：总连接数固定 C = dim*k0 = 10240
//   稠密对照：每个 B1 配置配同层内 MACs 稠密，d' = round(sqrt(dim*k0))
//
// 数据：`tools/b1_scaling_encode.py` 用 **MiniCPM 官方 tokenizer** 编码的 u32 token 流
//       （train/val 按**文档** 90/10 切分）。本驱动不含任何 BPE 代码。
//
// 模型（单层，与项目 B1 语义一致）：
//     h0 = emb[tok]                     [D]
//     y  = B1_layer(h0)                 [D]   每输出神经元 k0 条连接，w ~ U(-s,s), s=sqrt(3/k0)
//     h  = h0 + y                       残差
//     logits = lm_w·h + lm_b            [V]
//     CE(t) = -log softmax(logits_t)[tok_{t+1}]
//   层内 MACs（任务定义）= dim*k0。emb/lm_head 是**未匹配**的背景成本（报告里写明）。
//
// 稠密对照：宽度 d'，把 B1 层换成 d'×d' 稠密矩阵（gemm_nt）；层内 MACs = d'^2 ≈ dim*k0。
//
// ⚠️ 两处第一版写错、必须留档的地方（都会"看起来能跑"）：
//   1. 稀疏层的 Adam 状态**必须跨步保持**。第一版每步重新分配并清零 m/v，等价于
//      每一步都丢弃动量 —— loss 照样降，但那不是 Adam，配置之间也不可比。
//   2. 层反向必须用**残差相加之前**的 h0 当输入。第一版把 `h += y` 就地写回 h，
//      再用 h 去算 dw = dy⊗x ⇒ 用的是**层的输出**当输入。loss 也会降（方向错但有信号）。
//
// USAGE:
//   vmlm-b1-scaling.exe --mode b1|dense --dim 512 --k0 20 --seed 1 --steps 128
//                       --data-dir artifacts/b1_scaling --vocab-mode observed|full
//                       [--seq 32] [--batch 2] [--lr 3e-4] [--out results.jsonl]

extern "C" {
    fn malloc(n: usize) -> *mut u8;
}

use vmlm_core::b1_pure::{b1_backward, b1_backward_dx, b1_forward};
use vmlm_core::conn_train::{init_sparse_conn_scaled, SparseConnW};
use vmlm_core::kernels::gemm_nt;

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
unsafe fn zero(p: *mut f32, n: usize) {
    std::ptr::write_bytes(p as *mut u8, 0, n * 4);
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
fn arg_f32(flag: &str, def: f32) -> f32 {
    let v: Vec<String> = std::env::args().collect();
    for i in 0..v.len() {
        if v[i] == flag && i + 1 < v.len() {
            return v[i + 1].parse().unwrap_or(def);
        }
    }
    def
}
fn arg_str(flag: &str, def: &str) -> String {
    let v: Vec<String> = std::env::args().collect();
    for i in 0..v.len() {
        if v[i] == flag && i + 1 < v.len() {
            return v[i + 1].clone();
        }
    }
    def.to_string()
}

unsafe fn adam_dense(p: *mut f32, g: *const f32, m: *mut f32, v: *mut f32, n: usize, lr: f32, t: u32) {
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

/// 非有限数在 JSON 里没有字面量（NaN/inf 都不是合法 JSON）⇒ 一律写 null。
/// 与 `b1_full_train.rs` 同一处缺陷（那边是发散路径上把 NaN/inf 写进 summary.json，
/// 让下游 ConvertFrom-Json 静默返回空对象）；这里先补上，避免 RESULTS 行变成非法 JSON。
fn jnum(x: f32) -> String {
    if x.is_finite() { format!("{:.6}", x) } else { "null".to_string() }
}

/// `dst[j*m + i] = src[i*n + j]`（[m×n] → [n×m]）
unsafe fn tr(src: *const f32, dst: *mut f32, m: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            *dst.add(j * m + i) = *src.add(i * n + j);
        }
    }
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

fn main() {
    unsafe { vmlm_core::interp::init_cpu() };
    let mode = arg_str("--mode", "b1");
    let dim_arg = arg_u32("--dim", 512) as usize;
    let k0 = arg_u32("--k0", 20) as usize;
    let seed = arg_u32("--seed", 1) as u64;
    let steps = arg_u32("--steps", 128) as usize;
    let layers = arg_u32("--layers", 1).max(1) as usize;
    let seq = arg_u32("--seq", 32) as usize;
    let batch = arg_u32("--batch", 2) as usize;
    let lr = arg_f32("--lr", 3.0e-4);
    let clip = arg_f32("--clip", 1.0);
    let vocab_mode = arg_str("--vocab-mode", "observed");
    let data_dir = arg_str("--data-dir", "artifacts/b1_scaling");
    let out_path = arg_str("--out", "");
    let eval_batches = arg_u32("--eval-batches", 4) as usize;
    let log_every = arg_u32("--log-every", 8) as usize;
    // 残差旁路：本实验的核心机制对照。
    //   默认 h = h + y_l（残差堆叠）⇒ 恒等路径把信号直通输出，**不经过**稀疏层。
    //   --no-residual 1 ⇒ h = y_l（纯堆叠）⇒ 信息必须逐层穿过 k0 条连接，
    //   这才是「层间信息损耗」假设所描述的结构。
    let no_res = arg_u32("--no-residual", 0) == 1;
    // 残差分支的深度缩放。
    // 为什么必须有它（实测教训，也是本实验最重要的一个坑）：
    //   模型是 `h = h + y_l`，而 y_l 的初始化是**保方差**的（var(y)=var(x)）
    //   ⇒ var(h_l) = 2·var(h_{l-1}) ⇒ **每层方差翻倍**，L 层后 var 涨 2^L 倍。
    //   实测：L=24 在**第 1 步**训练 loss 就 87.4（logits ~ ±60），整个配置是坏的；
    //   L=16 的 logits 标准差已经 3.8（还能训但不干净）。这不是「信息损耗」，
    //   是**初始化尺度**问题。
    //   项目自己的 24 层训练器对残差分支的输出矩阵乘 `1/sqrt(2L)`（depth_scale），
    //   这里采用同一做法：var(y) = var(x)/(2L) ⇒ var(h_L) ≈ var(emb)·e^{1/2}（有界）。
    // `--depth-scale 1` 关闭（复现旧数据）；默认 -1 = auto = 1/sqrt(2·layers)。
    let ds_arg = arg_f32("--depth-scale", -1.0);
    let depth_scale = if ds_arg < 0.0 {
        1.0 / vmlm_core::kernels::sqrt_f(2.0 * layers as f32)
    } else {
        ds_arg
    };

    let dprime = ((dim_arg * k0) as f64).sqrt().round() as usize;
    let d = if mode == "dense" { dprime.max(8) } else { dim_arg };
    let t_tokens = batch * seq;

    let train = read_u32_file(&format!("{}/train_ids.bin", data_dir));
    let val = read_u32_file(&format!("{}/val_ids.bin", data_dir));

    // ---- 词表 ----
    let (n_vocab, id_map) = if vocab_mode == "full" {
        let meta = std::fs::read_to_string(format!("{}/meta.json", data_dir))
            .unwrap_or_else(|e| panic!("need meta.json: {}", e));
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
            let m = id_map[t as usize];
            if vocab_mode == "full" {
                m
            } else if m == 0 {
                0
            } else {
                m
            }
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
    let layer_macs = if mode == "dense" { d * d } else { d * k0 };

    println!("===== B1 scaling: one config =====");
    println!(
        "mode={} layers={} dim={} k0={} dprime={} d={} seed={} steps={} seq={} batch={} ({} tok/step, {} tok total)",
        mode, layers, dim_arg, k0, dprime, d, seed, steps, seq, batch, t_tokens, steps * t_tokens
    );
    println!(
        "data: train {} tok / val {} tok  vocab_mode={} n_vocab={}  val OOV {} tok ({:.4}%)",
        train.len(),
        val.len(),
        vocab_mode,
        n_vocab,
        val_oov,
        100.0 * val_oov as f64 / val.len().max(1) as f64
    );
    println!("layer MACs = {} (task def dim*k0 = {})", layer_macs, dim_arg * k0);

    unsafe {
        // ---------- 参数 ----------
        let emb = alloc(n_vocab * d);
        let lm_w = alloc(n_vocab * d);
        let lm_b = alloc(n_vocab);
        zero(lm_b, n_vocab);
        {
            let mut r = Rng::new(0x5EED_0000 ^ seed);
            let s = vmlm_core::kernels::sqrt_f(1.0 / d as f32);
            for i in 0..n_vocab * d {
                *emb.add(i) = r.range(-s, s);
            }
            for i in 0..n_vocab * d {
                *lm_w.add(i) = r.range(-s, s);
            }
        }
        let emb_g = alloc(n_vocab * d);
        let lm_g = alloc(n_vocab * d);
        let lm_bg = alloc(n_vocab);
        let emb_m = alloc(n_vocab * d);
        let emb_v = alloc(n_vocab * d);
        let lm_m = alloc(n_vocab * d);
        let lm_v = alloc(n_vocab * d);
        let lm_bm = alloc(n_vocab);
        let lm_bv = alloc(n_vocab);
        zero(emb_g, n_vocab * d);
        zero(lm_g, n_vocab * d);
        zero(lm_bg, n_vocab);
        zero(emb_m, n_vocab * d);
        zero(emb_v, n_vocab * d);
        zero(lm_m, n_vocab * d);
        zero(lm_v, n_vocab * d);
        zero(lm_bm, n_vocab);
        zero(lm_bv, n_vocab);

        // ---------- B1 层（layers 层，每层独立的拓扑与 Adam 状态）----------
        let mut scs: Vec<SparseConnW> = Vec::with_capacity(layers);
        let mut dws: Vec<*mut f32> = Vec::with_capacity(layers);
        let mut sp_ms: Vec<*mut f32> = Vec::with_capacity(layers);
        let mut sp_vs: Vec<*mut f32> = Vec::with_capacity(layers);
        for l in 0..layers {
            let sc_base = malloc(SparseConnW::bytes(d as u32, k0 as u32));
            // 每层用不同的 layer id ⇒ 每层拓扑不同（与项目 24 层训练器的做法一致）。
            // ⚠️ layers==1 时**必须**用 seed 本身：这样单层配置与之前 80 次运行
            //    逐位可比，用来验证「多层化重构没有改动单层数值」。
            let lid = if layers == 1 {
                seed as u32
            } else {
                (seed as u32).wrapping_mul(131).wrapping_add(l as u32)
            };
            let sc = SparseConnW::from_base(sc_base, d as u32, k0 as u32, lid);
            init_sparse_conn_scaled(&sc, depth_scale); // Glorot sqrt(3/k0) × 1/sqrt(2L)
            sc.assert_geometry();
            sc.assert_src_u32_consistent();
            let dw = alloc(d * k0);
            let m = alloc(d * k0); // ⚠️ 跨步保持的 Adam 状态（见文件头第 1 条）
            let v = alloc(d * k0);
            zero(dw, d * k0);
            zero(m, d * k0);
            zero(v, d * k0);
            scs.push(sc);
            dws.push(dw);
            sp_ms.push(m);
            sp_vs.push(v);
        }

        // ---------- 稠密层（同样 layers 层）----------
        let mut wds: Vec<*mut f32> = Vec::with_capacity(layers);
        let mut wd_gs: Vec<*mut f32> = Vec::with_capacity(layers);
        let mut wd_ms: Vec<*mut f32> = Vec::with_capacity(layers);
        let mut wd_vs: Vec<*mut f32> = Vec::with_capacity(layers);
        for l in 0..layers {
            let w = alloc(d * d);
            let g = alloc(d * d);
            let m = alloc(d * d);
            let v = alloc(d * d);
            zero(g, d * d);
            zero(m, d * d);
            zero(v, d * d);
            {
                let mut r = Rng::new(0xD3A5_0000 ^ seed ^ ((l as u64) << 8));
                let s = vmlm_core::kernels::sqrt_f(3.0 / d as f32) * depth_scale;
                for i in 0..d * d {
                    *w.add(i) = r.range(-s, s);
                }
            }
            wds.push(w);
            wd_gs.push(g);
            wd_ms.push(m);
            wd_vs.push(v);
        }

        // ---------- 缓冲 ----------
        // ⚠️ 每层的**输入**都要留下（残差相加之前），反向算 dW = dy⊗x 用它；
        //    第一版单层时只有一个 h0，多层化后必须按层存（L × T × d）。
        let mut h_ins: Vec<*mut f32> = Vec::with_capacity(layers);
        for _ in 0..layers {
            h_ins.push(alloc(t_tokens * d));
        }
        let h = alloc(t_tokens * d);
        let y = alloc(t_tokens * d);
        let logits = alloc(t_tokens * n_vocab);
        let dlogits = alloc(t_tokens * n_vocab);
        let dh = alloc(t_tokens * d);
        let dy = alloc(t_tokens * d);
        let dx = alloc(t_tokens * d);
        let bt = alloc(d * n_vocab); // lm_wᵀ [d × V]
        let btd = alloc(n_vocab * t_tokens); // dlogitsᵀ [V × T]
        let bth = alloc(d * t_tokens); // hᵀ [d × T]
        let bt2 = alloc(d * t_tokens.max(d));
        let bt3 = alloc(d * t_tokens.max(d));

        let step_fn = |toks: &[u32], train_mode: bool| -> (f32, usize) {
            // --- 前向：h = emb[tok]；对每层 h_in = h（存下来）；y = layer(h_in)；h = h_in + y ---
            let h_cur = h_ins[0]; // 第 0 层的输入缓冲同时充当当前 h
            for b in 0..batch {
                for t in 0..seq {
                    let tok = toks[b * seq + t] as usize;
                    std::ptr::copy_nonoverlapping(
                        emb.add(tok * d),
                        h_cur.add((b * seq + t) * d),
                        d,
                    );
                }
            }
            for l in 0..layers {
                let xin = h_ins[l];
                if l + 1 < layers {
                    std::ptr::copy_nonoverlapping(xin, h_ins[l + 1], t_tokens * d);
                }
                let dest = if l + 1 < layers { h_ins[l + 1] } else { h };
                if mode == "b1" {
                    for i in 0..t_tokens {
                        b1_forward(&scs[l], xin.add(i * d), y.add(i * d));
                    }
                } else {
                    gemm_nt(xin, wds[l], y, t_tokens, d, d, false);
                }
                for i in 0..t_tokens * d {
                    if no_res {
                        *dest.add(i) = *y.add(i);
                    } else {
                        *dest.add(i) = *xin.add(i) + *y.add(i);
                    }
                }
            }
            gemm_nt(h, lm_w, logits, t_tokens, n_vocab, d, false);
            // --- CE ---
            let mut loss_sum = 0.0f32;
            let mut n_ce = 0usize;
            for b in 0..batch {
                for t in 0..seq - 1 {
                    let pos = b * seq + t;
                    let tgt = toks[pos + 1] as usize;
                    let lg = logits.add(pos * n_vocab);
                    let mut mx = -1.0e30f32;
                    for v in 0..n_vocab {
                        let x = *lg.add(v) + *lm_b.add(v);
                        *lg.add(v) = x;
                        if x > mx {
                            mx = x;
                        }
                    }
                    let mut sum = 0.0f32;
                    for v in 0..n_vocab {
                        let e = vmlm_core::kernels::exp_f(*lg.add(v) - mx);
                        *lg.add(v) = e;
                        sum += e;
                    }
                    let inv = 1.0 / sum;
                    loss_sum += -vmlm_core::kernels::ln_f(*lg.add(tgt) * inv);
                    n_ce += 1;
                    if train_mode {
                        let invb = 1.0 / ((seq - 1) * batch) as f32;
                        for v in 0..n_vocab {
                            *dlogits.add(pos * n_vocab + v) = *lg.add(v) * inv * invb;
                        }
                        *dlogits.add(pos * n_vocab + tgt) -= invb;
                    }
                }
            }
            if !train_mode {
                return (loss_sum, n_ce);
            }
            // --- 反向：lm_head ---
            // ⚠️ 第一版这里写成「逐 (pos,v) 再对 i 循环」的**标量三重循环**：
            //    64 × 10921 × 512 = 3.58e8 次标量 FMA = **每步 0.11 s**，
            //    实测把单配置从 14 s 拖到 31.7 s。改用 gemm_nt（两次转置 + 8x8 内核）。
            zero(dh, t_tokens * d);
            for pos in 0..t_tokens {
                for v in 0..n_vocab {
                    *lm_bg.add(v) += *dlogits.add(pos * n_vocab + v);
                }
            }
            // dlm_w[v][i] += Σ_pos dlogits[pos][v]·h[pos][i]
            tr(dlogits, btd, t_tokens, n_vocab); // btd = [V × T]
            tr(h, bth, t_tokens, d); // bth = [d × T]
            gemm_nt(btd, bth, lm_g, n_vocab, d, t_tokens, true);
            tr(lm_w, bt, n_vocab, d);
            gemm_nt(dlogits, bt, dh, t_tokens, d, n_vocab, true);
            // --- 反向：逐层（最后一层往前）。每层输入用 h_ins[l]（残差之前）---
            for l in (0..layers).rev() {
                let xin = h_ins[l];
                std::ptr::copy_nonoverlapping(dh, dy, t_tokens * d);
                if mode == "b1" {
                    zero(dx, t_tokens * d);
                    zero(dws[l], d * k0);
                    for i in 0..t_tokens {
                        b1_backward(&scs[l], dy.add(i * d), xin.add(i * d), dws[l]);
                        b1_backward_dx(&scs[l], dy.add(i * d), dx.add(i * d));
                    }
                    for i in 0..t_tokens * d {
                        if no_res {
                            *dh.add(i) = *dx.add(i);
                        } else {
                            *dh.add(i) += *dx.add(i);
                        }
                    }
                } else {
                    // dh += dy·W
                    for i in 0..d {
                        for j in 0..d {
                            *bt2.add(j * d + i) = *wds[l].add(i * d + j);
                        }
                    }
                    gemm_nt(dy, bt2, dx, t_tokens, d, d, false);
                    for i in 0..t_tokens * d {
                        if no_res {
                            *dh.add(i) = *dx.add(i);
                        } else {
                            *dh.add(i) += *dx.add(i);
                        }
                    }
                    // dW += dyᵀ⊗x
                    tr(dy, bt2, t_tokens, d);
                    tr(xin, bt3, t_tokens, d);
                    gemm_nt(bt2, bt3, wd_gs[l], d, d, t_tokens, true);
                }
            }
            // --- emb 反向 ---
            for b in 0..batch {
                for t in 0..seq {
                    let tok = toks[b * seq + t] as usize;
                    for i in 0..d {
                        *emb_g.add(tok * d + i) += *dh.add((b * seq + t) * d + i);
                    }
                }
            }
            (loss_sum, n_ce)
        };

        // 固定的 val 批（跨配置可比）：从 val 流里取 eval_batches 批
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
        let eval_val = || -> f32 {
            let mut s = 0.0f32;
            let mut n = 0usize;
            for vb in val_batches.iter() {
                let (a, c) = step_fn(vb, false);
                s += a;
                n += c;
            }
            s / n as f32
        };

        // ---------- 训练 ----------
        let mut cursor = (seed as usize * 977) % (train_mapped.len() - seq - 1).max(1);
        let mut step_t: u32 = 0;
        let mut losses: Vec<f32> = Vec::new();
        let mut vals: Vec<f32> = Vec::new();
        let mut diverged = false;
        let mut diverged_at: usize = 0;
        // 发散判据：训练 loss 超过 **2 倍均匀分布基线**（2·ln V ≈ 18.6）。
        // 为什么需要它（实测教训）：L=24, k0=2 会发散，而发散状态下每步要 3.7 s
        // （退化数值走慢路径），单个 128 步配置要 **474 s**。加早停后同样的信息
        // （"第几步炸、炸到多少"）几秒就能拿到，并且把"炸在第几步"变成数据。
        let div_thresh = vmlm_core::kernels::ln_f(n_vocab as f32) * 2.0;
        for st in 1..=steps {
            step_t += 1;
            // 取一批（连续窗口，确定性）
            let mut toks = Vec::with_capacity(t_tokens);
            for b in 0..batch {
                let base = (cursor + b * seq) % (train_mapped.len() - seq - 1).max(1);
                for t in 0..seq {
                    toks.push(train_mapped[base + t]);
                }
            }
            cursor = (cursor + batch * seq) % (train_mapped.len() - seq - 1).max(1);

            zero(emb_g, n_vocab * d);
            zero(lm_g, n_vocab * d);
            zero(lm_bg, n_vocab);
            if mode == "dense" {
                for l in 0..layers {
                    zero(wd_gs[l], d * d);
                }
            }
            let (ls, nc) = step_fn(&toks, true);
            let tl = ls / nc as f32;
            if !tl.is_finite() || tl > div_thresh {
                diverged = true;
                diverged_at = st;
            }
            vmlm_core::train::clip_grad(lm_g, n_vocab * d, clip);
            vmlm_core::train::clip_grad(lm_bg, n_vocab, clip);
            vmlm_core::train::clip_grad(emb_g, n_vocab * d, clip);
            adam_dense(lm_w, lm_g, lm_m, lm_v, n_vocab * d, lr, step_t);
            adam_dense(lm_b, lm_bg, lm_bm, lm_bv, n_vocab, lr, step_t);
            adam_dense(emb, emb_g, emb_m, emb_v, n_vocab * d, lr, step_t);
            if mode == "b1" {
                for l in 0..layers {
                    vmlm_core::train::clip_grad(dws[l], d * k0, clip);
                    adam_dense(scs[l].w, dws[l], sp_ms[l], sp_vs[l], d * k0, lr, step_t);
                }
            } else {
                for l in 0..layers {
                    vmlm_core::train::clip_grad(wd_gs[l], d * d, clip);
                    adam_dense(wds[l], wd_gs[l], wd_ms[l], wd_vs[l], d * d, lr, step_t);
                }
            }
            if st % log_every == 0 || st == 1 {
                losses.push(tl);
                vals.push(eval_val());
            }
            if diverged {
                // 发散也报，但不再继续烧算力
                break;
            }
        }
        let final_train = *losses.last().unwrap_or(&f32::NAN);
        let final_val = *vals.last().unwrap_or(&f32::NAN);
        let ppl_tr = vmlm_core::kernels::exp_f(final_train);
        let ppl_va = vmlm_core::kernels::exp_f(final_val);
        let lj = losses
            .iter()
            .map(|x| format!("{:.6}", x))
            .collect::<Vec<_>>()
            .join(",");
        let vj = vals
            .iter()
            .map(|x| format!("{:.6}", x))
            .collect::<Vec<_>>()
            .join(",");
        let line = format!(
            "{{\"mode\":\"{}\",\"layers\":{},\"no_residual\":{},\"depth_scale\":{:.6},\"dim\":{},\"k0\":{},\"dprime\":{},\"d\":{},\"seed\":{},\"steps\":{},\"seq\":{},\"batch\":{},\"vocab_mode\":\"{}\",\"n_vocab\":{},\"val_oov\":{},\"layer_macs\":{},\"lr\":{},\"logged\":{},\"loss\":[{}],\"val\":[{}],\"final_train\":{},\"final_val\":{},\"ppl_train\":{},\"ppl_val\":{},\"diverged\":{},\"diverged_at\":{}}}",
            mode, layers, no_res, depth_scale, dim_arg, k0, dprime, d, seed, steps, seq, batch, vocab_mode, n_vocab, val_oov,
            layer_macs, lr, losses.len(), lj, vj, jnum(final_train), jnum(final_val), jnum(ppl_tr), jnum(ppl_va), diverged, diverged_at
        );
        println!("RESULT {}", line);
        if !out_path.is_empty() {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&out_path)
                .unwrap();
            writeln!(f, "{}", line).unwrap();
        }
        // 自检（让 verify_all 能把它当回归项）：最终 loss 必须有限且没有发散。
        // 不发散的判据不是"看起来对"，而是 loss 在合理域内（< ln(V)+5）。
        let lnv = vmlm_core::kernels::ln_f(n_vocab as f32);
        if !final_train.is_finite() || diverged {
            println!(
                "SELFCHECK DIVERGED: final_train={} (ln V = {:.4}) diverged_at_step={} logged={}",
                final_train, lnv, diverged_at, losses.len()
            );
            // 发散**不是**错误：任务要求"发散也报"。退出码用 0，让 runner 继续跑完网格。
        } else if final_train > lnv + 5.0 {
            println!(
                "SELFCHECK FAIL: final_train={} > ln V + 5 (ln V = {:.4})",
                final_train, lnv
            );
            std::process::exit(1);
        }
        println!(
            "SELFCHECK OK: final_train={:.4} final_val={:.4} (ln V = {:.4}, {} logged points)",
            final_train, final_val, lnv, losses.len()
        );
    }
}
