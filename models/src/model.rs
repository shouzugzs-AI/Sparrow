//! [抽库 L4c] `B1Model`：把 `run_main` 里 `嵌入 → 逐层(B1/attn) → head` 的组装收成一个对象。
//!
//! **同一个 `B1Model::forward` 同时服务**
//!   * 训练前向（`b1_full_train.rs` step_fn，L4c-2 接入）
//!   * 推理/生成前向（`--generate`，`vmlm-chat` 走它，L4c-1 接入）
//!
//! 两个模式都在本文件内、逐字复刻原 `run_main` 的代码：
//!   * `forward_train`：原 1798-1945（逐层 + 逐窗口 attention + head/head-skip）
//!   * `forward_last_logits`：原 2769-2842（生成：全序列 attention + 只取最后位置 logits）
//!
//! 逐位一致约束：缓冲别名 `xin=y`、window 偏移与 `asp` 缓存、head/head2 双路径、
//! `aprob` 的 `b * t_tokens * t_tokens` 步长、t8 static 一致性 —— 全部按原样保留。

use crate::attn_sparse::attn_forward_sparse;
use crate::b1::par_b1_forward;
use vmlm_blocks::attn::{attn_forward, attn_qk_stats};
use vmlm_blocks::util::zero;
use vmlm_ops::b1_pure::b1_forward_t8;
use vmlm_ops::kernels;
use vmlm_ops::conn::SparseConnW;
use crate::attn_sparse::attn_backward_sparse_par;
use crate::b1::par_b1_backward;
use crate::head::{head_bwd_batched, par_head_bwd};
use vmlm_blocks::attn::attn_backward;
use vmlm_blocks::util::tr;
use vmlm_ops::kernels::gemm_nt;

/// 前向所需的全部标量配置（来自 CLI）。
#[derive(Clone, Copy)]
pub struct B1ForwardCfg {
    // 共用
    pub dim: usize,
    pub n_vocab: usize,
    pub layers: usize,
    pub nth: usize,
    pub attn_every: usize,
    pub n_attn: usize,
    pub attn_sparse: bool,
    pub attn_residual: bool,
    pub head_skip: bool,
    pub identity_layers: bool,
    pub relu: bool,
    pub io_sparse: bool,
    pub k0_embed: usize,
    /// 训练口径的 token 总数（生成路径 `aprob` 步长；训练路径 `t8tok` 用）
    pub t_tokens: usize,
    /// 训练路径 attention 缓冲的块内步长（原 `tok_total`，与 `t_tokens` 是两个不同口径）
    pub tok_total: usize,
    // 训练模式
    pub train_mode: bool,
    pub tt: usize,
    pub seq: usize,
    pub nt_in: usize,
    pub attn_window: bool,
    pub head_blocked: bool,
    pub nb_head: usize,
    pub k0_head: usize,
    pub t8_stack: bool,
    pub t8s: usize,
    pub attn_qk_check: bool,
    pub slot: usize,
    pub off: usize,
    pub k0: usize,
    pub k0_attn: usize,
    pub ad: usize,
    pub fusion: bool,
    pub no_res: bool,
    pub profile: bool,
    pub attn_debug: u32,
}

pub struct B1Model<'a> {
    pub cfg: B1ForwardCfg,
    // 权重
    pub scs: &'a [SparseConnW],
    pub asq: &'a [SparseConnW],
    pub ask: &'a [SparseConnW],
    pub asv: &'a [SparseConnW],
    pub aso: &'a [SparseConnW],
    pub awq: *const f32,
    pub awk: *const f32,
    pub awv: *const f32,
    pub awo: *const f32,
    pub ad: usize,
    pub head: &'a SparseConnW,
    pub head2: &'a SparseConnW,
    pub emb: *const f32,
    pub dim_of_p: *const u32,
    pub etab: *const f32,
    // 工作区（共用）
    pub h_ins: &'a [*mut f32],
    pub h: *mut f32,
    pub y: *mut f32,
    pub aq: *mut f32,
    pub ak: *mut f32,
    pub av: *mut f32,
    pub ao: *mut f32,
    pub aprob: *mut f32,
    pub aout: *mut f32,
    pub logits: *mut f32,
    /// head_skip 的 head2 输出暂存（生成路径由调用方提供；训练路径 = `logits2`）
    pub head2_out: *mut f32,
    // L4c-3 反向专用（原 run_main 的局部缓冲）
    pub dh: *mut f32,
    pub dx: *mut f32,
    pub dy: *mut f32,
    pub dlogits: *mut f32,
    pub dws: &'a [*mut f32],
    pub dwp: *mut f32,
    pub head_dw: *mut f32,
    pub lm_w: *mut f32,
    pub lm_g: *mut f32,
    pub lm_bg: *mut f32,
    pub btd: *mut f32,
    pub bth: *mut f32,
    pub bt: *mut f32,
    pub adx: *mut f32,
    pub adq: *mut f32,
    pub adk: *mut f32,
    pub adv: *mut f32,
    pub adobuf: *mut f32,
    pub adprob: *mut f32,
    pub arowsum: *mut f32,
    pub aat1: *mut f32,
    pub aat2: *mut f32,
    pub awt: *mut f32,
    pub asq_g: *mut f32,
    pub ask_g: *mut f32,
    pub asv_g: *mut f32,
    pub aso_g: *mut f32,
    pub awq_g: *mut f32,
    pub awk_g: *mut f32,
    pub awv_g: *mut f32,
    pub awo_g: *mut f32,
    pub t8dxn: *mut f32,
    pub t8dyc: *mut f32,
    pub prof_p: *mut f64,
    // 训练专用
    pub pres: &'a [*mut f32],
    pub asp: *mut f32,
    pub head_wt: *const f32,
    pub head2_wt: *const f32,
    pub head_ht: *mut f32,
    pub t8ta: *mut f32,
    /// t8 整栈的层间中间缓冲（原 trainer 的 \dy\；t8_stack 时 dest 用它，不是 h_ins[l+1]）
    pub t8dy: *mut f32,
    pub collect_flag: *mut u8,
    pub qk_stats: *mut f32,
}

/// 诊断用 RMS（原 trainer 的 `rms`，只在 `attn_debug >= 2` 分支用）。
unsafe fn rms(p: *const f32, n: usize) -> f32 {
    let mut s = 0.0f64;
    for i in 0..n {
        let v = *p.add(i) as f64;
        s += v * v;
    }
    (s / n.max(1) as f64).sqrt() as f32
}

impl B1Model<'_> {
    /// 唯一入口：训练模式跑 `forward_train`，生成模式跑 `forward_last_logits`。
    #[inline(always)]
    pub unsafe fn forward(&mut self, window: Option<&[u32]>, t: usize) -> *const f32 {
        if self.cfg.train_mode {
            self.forward_train()
        } else {
            self.forward_last_logits(window.expect("生成模式必须给 window"), t)
        }
    }

    /// 生成语义前向：`window` 是长度 `t` 的 token 序列，返回**最后一个位置**的 logits 指针。
    /// 与 `run_main` 原生成块（2769-2842）逐字等价（L4c-1）。
    #[inline(always)]
    pub unsafe fn forward_last_logits(&mut self, window: &[u32], t: usize) -> *const f32 {
        let c = self.cfg;
        // ---- 嵌入 scatter（与 step_fn 前向开头同一逻辑）----
        if c.io_sparse {
            zero(self.h_ins[0], t * c.dim);
            for (p_i, &tok) in window.iter().enumerate() {
                let tk = tok as usize;
                for j in 0..c.k0_embed {
                    let d = *self.dim_of_p.add(tk * c.k0_embed + j) as usize;
                    let v = *self.etab.add(tk * c.k0_embed + j);
                    *self.h_ins[0].add(p_i * c.dim + d) += v;
                }
            }
        } else {
            for (p_i, &tok) in window.iter().enumerate() {
                std::ptr::copy_nonoverlapping(
                    self.emb.add(tok as usize * c.dim),
                    self.h_ins[0].add(p_i * c.dim),
                    c.dim,
                );
            }
        }
        // ---- 逐层前向 + 全局 attention ----
        for l in 0..c.layers {
            let xin = self.h_ins[l];
            let dest = if l + 1 < c.layers { self.h_ins[l + 1] } else { self.h };
            if c.identity_layers {
                std::ptr::copy_nonoverlapping(xin, dest, t * c.dim);
            } else {
                par_b1_forward(&self.scs[l], xin, self.y, t, c.dim, c.dim, c.nth);
                if c.relu {
                    for i in 0..t * c.dim {
                        let v0 = *self.y.add(i);
                        *self.y.add(i) = if v0 > 0.0 { v0 } else { 0.0 };
                    }
                }
                std::ptr::copy_nonoverlapping(self.y, dest, t * c.dim);
            }
            if c.n_attn > 0 && (l + 1) % c.attn_every == 0 {
                let b = (l + 1) / c.attn_every - 1;
                if b < c.n_attn {
                    if c.attn_sparse {
                        attn_forward_sparse(
                            dest, &self.asq[b], &self.ask[b], &self.asv[b], &self.aso[b], t, c.dim,
                            self.aq, self.ak, self.av, self.ao,
                            self.aprob.add(b * c.t_tokens * c.t_tokens), self.aout, c.nth,
                        );
                    } else {
                        attn_forward(
                            dest, self.awq.add(b * self.ad), self.awk.add(b * self.ad),
                            self.awv.add(b * self.ad), self.awo.add(b * self.ad),
                            t, c.dim, self.aq, self.ak, self.av, self.ao,
                            self.aprob.add(b * c.t_tokens * c.t_tokens), self.aout, c.nth,
                        );
                    }
                    if c.attn_residual {
                        for i in 0..t * c.dim {
                            *dest.add(i) += *self.aout.add(i);
                        }
                    } else {
                        std::ptr::copy_nonoverlapping(self.aout, dest, t * c.dim);
                    }
                }
            }
        }
        // ---- head：只要最后一个位置的 logits ----
        let last = t - 1;
        par_b1_forward(&self.head, self.h.add(last * c.dim), self.logits, 1, c.dim, c.n_vocab, 1);
        if c.head_skip {
            par_b1_forward(
                &self.head2, self.h_ins[0].add(last * c.dim), self.head2_out, 1, c.dim, c.n_vocab, 1,
            );
            for i in 0..c.n_vocab {
                *self.logits.add(i) += *self.head2_out.add(i);
            }
        }
        self.logits as *const f32
    }

    /// 训练语义前向：逐层 + 逐窗口 attention（+ `asp` 缓存供反向用）+ head/head-skip。
    /// 与 `run_main` 原训练块（1798-1945）逐字等价（L4c-2）。
    #[inline(always)]
    unsafe fn forward_train(&mut self) -> *const f32 {
        let c = self.cfg;
        let (layers, tt, dim, n_vocab) = (c.layers, c.tt, c.dim, c.n_vocab);
        let (n_attn, attn_every, seq) = (c.n_attn, c.attn_every, c.seq);
        let nt_in = c.nt_in;
        for l in 0..layers {
            let t8_boundary =
                c.t8_stack && (l + 1 == layers || (n_attn > 0 && (l + 1) % attn_every == 0));
            let xin = if t8_boundary {
                let src8 = if l == 0 { self.t8ta } else { self.h_ins[l] };
                let xinb = crate::b1::t8tok(c.t_tokens * dim);
                kernels::transpose_t8_back_avx2(src8, dim, tt, xinb, dim, c.t8s.max(1));
                xinb
            } else {
                self.h_ins[l]
            };
            let dest = if c.t8_stack {
                if l + 1 < layers { self.t8dy } else { self.h }
            } else if l + 1 < layers {
                self.h_ins[l + 1]
            } else {
                self.h
            };
            if c.t8_stack && !t8_boundary && l + 1 < layers {
                b1_forward_t8(
                    &self.scs[l], if l == 0 { self.t8ta } else { self.h_ins[l] }, c.t8s.max(1), tt,
                    self.h_ins[l + 1],
                );
            } else if c.identity_layers {
                std::ptr::copy_nonoverlapping(xin, dest, tt * dim);
            } else {
                par_b1_forward(&self.scs[l], xin, self.y, tt, dim, dim, nt_in);
                if c.relu {
                    std::ptr::copy_nonoverlapping(self.y, self.pres[l], tt * dim);
                    for i in 0..tt * dim {
                        let v = *self.y.add(i);
                        *self.y.add(i) = if v > 0.0 { v } else { 0.0 };
                    }
                }
                std::ptr::copy_nonoverlapping(self.y, dest, tt * dim);
            }
            // ---- 全局 attention（跨 token 交互）：h = h + attn(h) ----
            if n_attn > 0 && (l + 1) % attn_every == 0 {
                let b = (l + 1) / attn_every - 1;
                if b < n_attn {
                    let n_win = if c.attn_window { tt / seq } else { 1 };
                    let t_attn = if c.attn_window { seq } else { tt };
                    std::ptr::copy_nonoverlapping(dest, self.asp.add(b * c.tok_total), tt * dim);
                    for w in 0..n_win {
                        let base = w * seq * dim;
                        let woff = (n_win - 1 - w) * t_attn * dim;
                        let poff = (n_win - 1 - w) * t_attn * t_attn;
                        let aqw = self.aq.add(b * c.tok_total + woff);
                        let akw = self.ak.add(b * c.tok_total + woff);
                        let avw = self.av.add(b * c.tok_total + woff);
                        let aow = self.ao.add(b * c.tok_total + woff);
                        let apr = self.aprob.add(b * tt * tt + poff);
                        if c.attn_sparse {
                            attn_forward_sparse(
                                dest.add(base), &self.asq[b], &self.ask[b], &self.asv[b], &self.aso[b],
                                t_attn, dim, aqw, akw, avw, aow, apr, self.aout, nt_in,
                            );
                        } else {
                            attn_forward(
                                dest.add(base), self.awq.add(b * self.ad), self.awk.add(b * self.ad),
                                self.awv.add(b * self.ad), self.awo.add(b * self.ad),
                                t_attn, dim, aqw, akw, avw, aow, apr, self.aout, nt_in,
                            );
                        }
                        if c.attn_debug >= 2 && (w < 2 || w + 1 == n_win) {
                            println!(
                                "    [fw] l={} b={} w={}/{} off={} tt={} slot={} base={} t_attn={} nt_in={}",
                                l, b, w, n_win, c.off, tt, c.slot, base, t_attn, nt_in
                            );
                            println!(
                                "         aq={:p} ak={:p} dest+base={:p} | RMS dest={:.3e} aq={:.3e} ak={:.3e} aout={:.3e} asq.w={:p}",
                                aqw, akw, dest.add(base),
                                rms(dest.add(base), t_attn * dim), rms(aqw, t_attn * dim),
                                rms(akw, t_attn * dim), rms(self.aout, t_attn * dim), self.asq[b].w
                            );
                        }
                        if c.attn_residual {
                            for i in 0..t_attn * dim {
                                *dest.add(base + i) += *self.aout.add(i);
                            }
                        } else {
                            std::ptr::copy_nonoverlapping(self.aout, dest.add(base), t_attn * dim);
                        }
                    }
                    // 逐块采集 q·k / softmax 统计（读的是**最后一个窗口**）
                    if c.attn_qk_check && c.slot == 0 && *self.collect_flag.add(b) == 1 {
                        let st = attn_qk_stats(
                            self.aq.add(b * c.tok_total),
                            self.ak.add(b * c.tok_total),
                            t_attn,
                            dim,
                            self.aprob.add(b * tt * tt),
                        );
                        *self.qk_stats.add(b * 5 + 0) = st.0;
                        *self.qk_stats.add(b * 5 + 1) = st.1;
                        *self.qk_stats.add(b * 5 + 2) = st.2;
                        *self.qk_stats.add(b * 5 + 3) = st.3;
                        *self.qk_stats.add(b * 5 + 4) = st.4;
                        *self.collect_flag.add(b) = 0;
                    }
                }
            }
            if c.t8_stack && t8_boundary && l + 1 < layers {
                kernels::transpose_t8_avx2(dest, tt, dim, self.h_ins[l + 1], c.t8s.max(1));
            }
        }
        if c.io_sparse {
            if c.head_blocked {
                crate::head::par_head_fwd_wt(
                    self.head_wt, self.h, self.logits, self.head_ht, tt, dim, n_vocab, nt_in,
                    n_vocab, c.nb_head, c.k0_embed, c.k0_head,
                );
            } else {
                par_b1_forward(&self.head, self.h, self.logits, tt, dim, n_vocab, nt_in);
            }
            if c.head_skip {
                if c.head_blocked {
                    crate::head::par_head_fwd_wt(
                        self.head2_wt, self.h_ins[0], self.head2_out, self.head_ht, tt, dim,
                        n_vocab, nt_in, n_vocab, c.nb_head, c.k0_embed, c.k0_head,
                    );
                } else {
                    par_b1_forward(&self.head2, self.h_ins[0], self.head2_out, tt, dim, n_vocab, nt_in);
                }
                if vmlm_sys::cpu::HAS_AVX2 {
                    kernels::add256_f32(self.logits, self.head2_out, tt * n_vocab);
                } else {
                    for i in 0..tt * n_vocab {
                        *self.logits.add(i) += *self.head2_out.add(i);
                    }
                }
            }
        }
        self.logits as *const f32
    }
    /// 训练反向：head 反向 + 逐层反向（t8 / 逐窗口 attention / 逐层）。
    /// **逐字复刻**原 `run_main` 1949-2110（L4c-3）；字段在开头绑成同名局部，函数体未改一字。
    #[inline(always)]
    pub unsafe fn backward(&mut self) {
        let c = self.cfg;
        let ad = c.ad;
        let attn_every = c.attn_every;
        let attn_residual = c.attn_residual;
        let attn_sparse = c.attn_sparse;
        let attn_window = c.attn_window;
        let dim = c.dim;
        let fusion = c.fusion;
        let head_blocked = c.head_blocked;
        let identity_layers = c.identity_layers;
        let io_sparse = c.io_sparse;
        let k0 = c.k0;
        let k0_attn = c.k0_attn;
        let k0_embed = c.k0_embed;
        let k0_head = c.k0_head;
        let layers = c.layers;
        let n_attn = c.n_attn;
        let n_vocab = c.n_vocab;
        let nb_head = c.nb_head;
        let no_res = c.no_res;
        let nt_in = c.nt_in;
        let profile = c.profile;
        let relu = c.relu;
        let seq = c.seq;
        let t8_stack = c.t8_stack;
        let t8s = c.t8s;
        let tok_total = c.tok_total;
        let tt = c.tt;
        let scs = self.scs;
        let asq = self.asq;
        let ask = self.ask;
        let asv = self.asv;
        let aso = self.aso;
        let awq = self.awq;
        let awk = self.awk;
        let awv = self.awv;
        let awo = self.awo;
        let head = self.head;
        let head2 = self.head2;
        let h_ins = self.h_ins;
        let h = self.h;
        let y = self.y;
        let aq = self.aq;
        let ak = self.ak;
        let av = self.av;
        let ao = self.ao;
        let aprob = self.aprob;
        let aout = self.aout;
        let pres = self.pres;
        let asp = self.asp;
        let t8ta = self.t8ta;
        let collect_flag = self.collect_flag;
        let qk_stats = self.qk_stats;
        let head_wt = self.head_wt;
        let head2_wt = self.head2_wt;
        let head_ht = self.head_ht;
        let logits = self.logits;
        let head2_out = self.head2_out;
        let emb = self.emb;
        let dim_of_p = self.dim_of_p;
        let etab = self.etab;
        let dh = self.dh;
        let dx = self.dx;
        let dy = self.dy;
        let dlogits = self.dlogits;
        let dws = self.dws;
        let dwp = self.dwp;
        let head_dw = self.head_dw;
        let lm_w = self.lm_w;
        let lm_g = self.lm_g;
        let lm_bg = self.lm_bg;
        let btd = self.btd;
        let bth = self.bth;
        let bt = self.bt;
        let adx = self.adx;
        let adq = self.adq;
        let adk = self.adk;
        let adv = self.adv;
        let adobuf = self.adobuf;
        let adprob = self.adprob;
        let arowsum = self.arowsum;
        let aat1 = self.aat1;
        let aat2 = self.aat2;
        let awt = self.awt;
        let asq_g = self.asq_g;
        let ask_g = self.ask_g;
        let asv_g = self.asv_g;
        let aso_g = self.aso_g;
        let awq_g = self.awq_g;
        let awk_g = self.awk_g;
        let awv_g = self.awv_g;
        let awo_g = self.awo_g;
        let prof_p = self.prof_p;
        let mut t8dyc = self.t8dyc;
        let mut t8dxn = self.t8dxn;
        // 原 trainer 的 `tm` 是剖析用 Instant；此处用新 Instant（只影响 --profile 分相，不影响数值）
        let mut tm = std::time::Instant::now();
            // 反向：lm_head / 稀疏 head
            zero(dh, tt * dim);
            if io_sparse {
                // 直接复用 b1_backward（对 w 累加）+ b1_backward_dx（对 dx 累加）。
                // 每个 token 的 head 只读它自己那一行 h[i*dim..]，所以：
                //   · dw 跨 token 累加 ⇒ 进入循环前清零一次；
                //   · dx 写进 dh 的第 i 段 ⇒ dh 已清零，直接累加即可。
                zero(head_dw, n_vocab * k0_head);
                if head_blocked && fusion && nt_in <= 1 {
                    // [P21-4] 按 token 批量（一次转置 + 一次 GEMM），不再逐 token RMW 整块 dw
                    head_bwd_batched(head.w, head.w, dlogits, h, tt, dim, n_vocab, nb_head, k0_embed, k0_head, head_dw, dh);
                } else if head_blocked && fusion {
                    par_head_bwd(&head, dlogits, h, head_dw, dh, tt, dim, n_vocab, nt_in, nb_head, k0_embed);
                } else {
                    par_b1_backward(&head, dlogits, h, head_dw, dh, tt, dim, n_vocab, nt_in, dwp, fusion);
                }
            } else {
                for pos in 0..tt {
                    for v in 0..n_vocab {
                        *lm_bg.add(v) += *dlogits.add(pos * n_vocab + v);
                    }
                }
                tr(dlogits, btd, tt, n_vocab);
                tr(h, bth, tt, dim);
                gemm_nt(btd, bth, lm_g, n_vocab, dim, tt, true);
                tr(lm_w, bt, n_vocab, dim);
                gemm_nt(dlogits, bt, dh, tt, dim, n_vocab, true);
            }
            // 反向：逐层（最后一层往前）。无残差 ⇒ dh 被**赋值**成 dx
            if profile {
                *prof_p.add(3) += tm.elapsed().as_secs_f64();
                tm = std::time::Instant::now();
            }
            // [P21-7b] 反向：t8dyc 始终持有"当前层输出梯度"的 t8 形式；
            // 注意力层先把 t8dyc 转成 token-major dh 供 attention 反向，再转回 t8。
            let mut t8_dh_token = true;
            for l in (0..layers).rev() {
                if t8_stack {
                    let attn_b = n_attn > 0 && (l + 1) % attn_every == 0;
                    if attn_b && !t8_dh_token {
                        vmlm_ops::kernels::transpose_t8_back_avx2(t8dyc, dim, tt, dh, dim, t8s.max(1));
                        t8_dh_token = true;
                    } else if !attn_b && t8_dh_token {
                        vmlm_ops::kernels::transpose_t8_avx2(dh, tt, dim, t8dyc, t8s.max(1));
                        t8_dh_token = false;
                    }
                }
                if identity_layers {
                    // 恒等映射的梯度就是恒等：dh 原样往前传（无 dw、无 dx）
                    continue;
                }
                // ---- attention 反向（先做：它作用在该层输出上）----
                // 该层输出（post-attention）在 h_ins[l+1] 或 h 里；dh 就是它的梯度。
                if n_attn > 0 && (l + 1) % attn_every == 0 {
                    let b = (l + 1) / attn_every - 1;
                    if b < n_attn {
                        // 重算 attention 的输入 = 稀疏层输出（带 ReLU）
                        // [P19-4] 这里原来是**重算**：先用 b1_forward 重跑一遍本层稀疏前向
                        //   （+ReLU），再用 attn_forward_sparse 重算 5 个稀疏投影（q/k/v/o）。
                        // 前向已经把 attention 的输入（asp）与逐窗口的 q/k/v/o/prob 缓存好了
                        //   ⇒ 反向直接用，省掉每块 6 次整尺寸稀疏前向（实测占 attention 反向 ~47%）。
                        // 数值上不再重算 ⇒ 与"重算版"逐位不同（ε 级；且重算版本来就依赖
                        //   缓冲不被覆写，逐窗口交替重算的那套 fragility 一并消失）。
                        zero(adx, tt * dim);
                        // 反向也按同样的窗口切分（否则 --attn-window 的梯度是错的）
                        // ⚠️ 梯度累加缓冲的清零必须在**循环外**：第一版放在循环里，
                        //    第二个窗口会把第一个窗口的梯度清零 ⇒ 只有 1/n_win 的梯度被保留，
                        //    FD 对账立刻抓到（batch=2/4 FAIL，batch=1 PASS）。
                        if attn_sparse {
                            zero(asq_g.add(b * dim * k0_attn), dim * k0_attn);
                            zero(ask_g.add(b * dim * k0_attn), dim * k0_attn);
                            zero(asv_g.add(b * dim * k0_attn), dim * k0_attn);
                            zero(aso_g.add(b * dim * k0_attn), dim * k0_attn);
                        }
                        let n_win_b = if attn_window { tt / seq } else { 1 };
                        let t_attn_b = if attn_window { seq } else { tt };
                        for w in 0..n_win_b {
                            let base = w * seq * dim;
                            let woff = (n_win_b - 1 - w) * t_attn_b * dim;
                            let poff = (n_win_b - 1 - w) * t_attn_b * t_attn_b;
                            let aqw = aq.add(b * tok_total + woff);
                            let akw = ak.add(b * tok_total + woff);
                            let avw = av.add(b * tok_total + woff);
                            let aow = ao.add(b * tok_total + woff);
                            let apr = aprob.add(b * tt * tt + poff);
                            if attn_sparse {
                                attn_backward_sparse_par(
                                    asp.add(b * tok_total).add(base), &asq[b], &ask[b], &asv[b], &aso[b],
                                    dh.add(base), t_attn_b, dim, aqw, akw, avw, aow,
                                    apr,
                                    adq, adk, adv, adx.add(base), adobuf, adprob, arowsum,
                                    asq_g.add(b * dim * k0_attn), ask_g.add(b * dim * k0_attn),
                                    asv_g.add(b * dim * k0_attn), aso_g.add(b * dim * k0_attn),
                                    // [P19-3] 这里原来传的是 `nth`（=进程总线程数），而不是
                                    // `nt_in`（本 region 的**体内**并行度）。inter 模式下
                                    // step_fn 本身已经跑在 8 个 pool worker 上，内层再按 nth=8
                                    // spawn 一层 ⇒ 每步 3 个 attention 层 × 8 个 region = 24 次
                                    // 「8 线程 spawn + 3 次 barrier」，4 物理核上跑 64 个线程。
                                    // 这就是 --profile 里「逐层反向」占到 74% 而单内核微基准
                                    // （core/examples/kbench）只有 0.59 ns/conn 的原因。
                                    nt_in, dwp,
                                );
                            } else {
                                attn_backward(
                                    asp.add(b * tok_total).add(base), awq.add(b * ad), awk.add(b * ad), awv.add(b * ad),
                                    awo.add(b * ad),
                                    dh.add(base), t_attn_b, dim, aqw, akw, avw, aow,
                                    apr,
                                    adq, adk, adv,
                                    awq_g.add(b * ad), awk_g.add(b * ad), awv_g.add(b * ad), awo_g.add(b * ad),
                                    adx.add(base), aat1, aat2, awt, adobuf, adprob,
                                );
                            }
                        }
                        // 残差：层输出梯度同时流向恒等通道；替换：只流向 attention 的输入
                        if attn_residual {
                            for i in 0..tt * dim {
                                *dh.add(i) += *adx.add(i);
                            }
                        } else {
                            std::ptr::copy_nonoverlapping(adx, dh, tt * dim);
                        }
                    }
                }
                if t8_stack {
                    if t8_dh_token {
                        vmlm_ops::kernels::transpose_t8_avx2(dh, tt, dim, t8dyc, t8s.max(1));
                        t8_dh_token = false;
                    }
                    zero(dws[l], dim * k0);
                    zero(t8dxn, dim * t8s.max(1));
                    let src8 = if l == 0 { t8ta } else { h_ins[l] };
                    vmlm_ops::kernels::b1_backward_t8_avx2(
                        scs[l].w, scs[l].src_u32, src8, t8dyc, t8s.max(1), dim, k0, t8dxn, dws[l],
                    );
                    std::mem::swap(&mut t8dyc, &mut t8dxn);
                    continue;
                }
                let xin = h_ins[l];
                std::ptr::copy_nonoverlapping(dh, dy, tt * dim);
                if relu {
                    // ReLU 反向：预激活 <= 0 的位置梯度置 0
                    for i in 0..tt * dim {
                        if *pres[l].add(i) <= 0.0 {
                            *dy.add(i) = 0.0;
                        }
                    }
                }
                zero(dx, tt * dim);
                zero(dws[l], dim * k0);
                par_b1_backward(&scs[l], dy, xin, dws[l], dx, tt, dim, dim, nt_in, dwp, fusion);
                if no_res {
                    std::ptr::copy_nonoverlapping(dx, dh, tt * dim);
                } else {
                    for i in 0..tt * dim {
                        *dh.add(i) += *dx.add(i);
                    }
                }
            }
            if t8_stack && !t8_dh_token {
                vmlm_ops::kernels::transpose_t8_back_avx2(t8dyc, dim, tt, dh, dim, t8s.max(1));
            }
        // 交换后的 t8 缓冲写回，保持与搬移前一致的对外状态
        self.t8dyc = t8dyc;
        self.t8dxn = t8dxn;
    }

}
