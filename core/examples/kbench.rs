//! `_opt` 内核微基准（**不是交付物**，只是逐条优化的 A/B 计时器）。
//!
//! 为什么单独做一个：端到端 100 步要 40 s，而单条内核改动的收益常常只有几个百分点，
//! 用端到端去分辨「内核慢在哪」既慢又容易被噪声淹没。这里把 B1 前向/反向按
//! **发布形态**（与 trainer 完全相同的 `SparseConnW` 初始化 + 相同的 per-token 调用方式）
//! 放在一起计时，直接给 ns/conn。
//!
//! 用法：
//!   cargo run --release -p vmlm-core --example kbench --features train,conn-sparse -- \
//!       --dim 2048 --k0 8 --t 128 --rep 40
//!
//! 参数：`--dim` `--k0` `--t`（每 region token 数）`--rep`（重复轮数）`--mode`（fwd/bwd/all）

extern "C" {
    fn malloc(n: usize) -> *mut u8;
}

use vmlm_core::b1_pure::{b1_backward_fused, b1_forward};
use vmlm_core::conn_train::{init_sparse_conn_scaled, SparseConnW};

unsafe fn alloc(n: usize) -> *mut f32 {
    let p = malloc(n * 4 + 64) as *mut f32;
    assert!(!p.is_null(), "OOM");
    p
}
unsafe fn zero(p: *mut f32, n: usize) {
    std::ptr::write_bytes(p as *mut u8, 0, n * 4);
}

/// splitmix64（与 trainer 的 Rng 同族，只为填数）
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uni(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u32 << 24) as f32) - 0.5
    }
}

fn arg_u32(flag: &str, def: usize) -> usize {
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

// =====================================================================
// 头（block 对齐稀疏 head）原型：验证「把 head 当稠密块矩阵」的收益
// =====================================================================
// 事实（trainer 的 init 路径）：block_of[v] = v % n_blocks（贪心取最小使用次数 ⇒ 就是模）。
//   于是 src[v][j] = (v % n_blocks)*k0_embed + j —— **每个 head 行的 k0_head 个源连续**。
// 本文原型把 head 前向/反向改成「连续 16 float 的乘加 + 8 行向量化」，
// 完全不 gather；并给出与生产内核（b1_forward / b1_backward_fused）的对比。

#[cfg(target_arch = "x86_64")]
mod hb {
    use core::arch::x86_64::*;

    #[inline(always)]
    unsafe fn hsum(v: __m256) -> f32 {
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps(v, 1);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_hadd_ps(s, s);
        let s = _mm_hadd_ps(s, s);
        _mm_cvtss_f32(s)
    }

    /// W[V][16] -> WT[16][V]（8x8 分块，行数按 8 对齐；尾部标量）
    #[target_feature(enable = "avx2")]
    pub unsafe fn pack_wt(w: *const f32, wt: *mut f32, nv: usize, k0e: usize) {
        let mut v0 = 0usize;
        while v0 + 8 <= nv {
            let mut j0 = 0usize;
            while j0 + 8 <= k0e {
                // 取 8 行 × 8 列，写成 8 列 × 8 行
                let mut tmp = [0.0f32; 64];
                for r in 0..8 {
                    for c in 0..8 {
                        tmp[r * 8 + c] = *w.add((v0 + r) * k0e + j0 + c);
                    }
                }
                for c in 0..8 {
                    for r in 0..8 {
                        *wt.add((j0 + c) * nv + v0 + r) = tmp[r * 8 + c];
                    }
                }
                j0 += 8;
            }
            while j0 < k0e {
                for r in 0..8 {
                    *wt.add(j0 * nv + v0 + r) = *w.add((v0 + r) * k0e + j0);
                }
                j0 += 1;
            }
            v0 += 8;
        }
        while v0 < nv {
            for j in 0..k0e {
                *wt.add(j * nv + v0) = *w.add(v0 * k0e + j);
            }
            v0 += 1;
        }
    }

    /// h[128][16] -> hT[16][128]
    #[target_feature(enable = "avx2")]
    pub unsafe fn pack_ht(h: *const f32, ht: *mut f32, nb: usize, k0e: usize) {
        for b in 0..nb {
            for j in 0..k0e {
                *ht.add(j * nb + b) = *h.add(b * k0e + j);
            }
        }
    }

    /// **行主序版**前向：块主序遍历（b 外、行内），h_b 常驻寄存器，
    /// 每行 = 2 次连续 w 载入 + 2 次 FMA + 一次横向归约。**不需要转置、不需要 WT**。
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn fwd_rm(w: *const f32, h: *const f32, y: *mut f32, nv: usize, nb: usize, k0e: usize) {
        let mut b = 0usize;
        while b < nb {
            let hrow = h.add(b * k0e);
            let mut v = b;
            while v < nv {
                let wrow = w.add(v * k0e);
                let mut acc = _mm256_setzero_ps();
                let mut j = 0usize;
                while j + 8 <= k0e {
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(wrow.add(j)), _mm256_loadu_ps(hrow.add(j)), acc);
                    j += 8;
                }
                let mut s = hsum(acc);
                while j < k0e {
                    s += *wrow.add(j) * *hrow.add(j);
                    j += 1;
                }
                *y.add(v) = s;
                v += nb;
            }
            b += 1;
        }
    }

    /// 行主序、**v 顺序**遍历（W 顺序读，prefetch 友好）：每行 h 块从 L1 取。
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn fwd_rm2(w: *const f32, h: *const f32, y: *mut f32, nv: usize, nb: usize, k0e: usize) {
        let mask = nb - 1;
        let mut v = 0usize;
        while v < nv {
            let hrow = h.add((v & mask) * k0e);
            let wrow = w.add(v * k0e);
            let mut acc = _mm256_setzero_ps();
            let mut j = 0usize;
            while j + 8 <= k0e {
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(wrow.add(j)), _mm256_loadu_ps(hrow.add(j)), acc);
                j += 8;
            }
            let mut s = hsum(acc);
            while j < k0e {
                s += *wrow.add(j) * *hrow.add(j);
                j += 1;
            }
            *y.add(v) = s;
            v += 1;
        }
    }
    /// `dh[b][0..k0e] += dy[v]·w[v][0..k0e]`（b 由 v 决定，连续 RMW）。
    /// 没有 gather、没有标量散写、没有转置。
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn bwd_rm(
        w: *const f32,
        h: *const f32,
        dy: *const f32,
        dw: *mut f32,
        dh: *mut f32,
        nv: usize,
        nb: usize,
        k0e: usize,
    ) {
        let mask = nb - 1;
        for v in 0..nv {
            let b = v & mask;
            let d = _mm256_set1_ps(*dy.add(v));
            let wrow = w.add(v * k0e);
            let hrow = h.add(b * k0e);
            let drow = dw.add(v * k0e);
            let dhrow = dh.add(b * k0e);
            let mut j = 0usize;
            while j + 8 <= k0e {
                let p = _mm256_fmadd_ps(d, _mm256_loadu_ps(hrow.add(j)), _mm256_loadu_ps(drow.add(j)));
                _mm256_storeu_ps(drow.add(j), p);
                let q = _mm256_fmadd_ps(d, _mm256_loadu_ps(wrow.add(j)), _mm256_loadu_ps(dhrow.add(j)));
                _mm256_storeu_ps(dhrow.add(j), q);
                j += 8;
            }
            while j < k0e {
                let dd = *dy.add(v);
                *drow.add(j) += dd * *hrow.add(j);
                *dhrow.add(j) += dd * *wrow.add(j);
                j += 1;
            }
        }
    }

    /// 单 token：logits[V] = W[V][k0e] · h[block(v)*k0e ..]，8 行向量化。
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn fwd(wt: *const f32, ht: *const f32, logits: *mut f32, nv: usize, nb: usize, k0e: usize) {
        // 行按 [g*128 + s] 分组：同一 s 段的 8 行共享 s..s+7 块
        let mut g = 0usize;
        while g * nb + nb <= nv {
            let mut s = 0usize;
            while s + 8 <= nb {
                let v0 = g * nb + s;
                let mut acc = _mm256_setzero_ps();
                let mut j = 0usize;
                while j < k0e {
                    let hv = _mm256_loadu_ps(ht.add(j * nb + s)); // 8 lane = 8 相邻块，hT 里连续
                    acc = _mm256_fmadd_ps(hv, _mm256_loadu_ps(wt.add(j * nv + v0)), acc);
                    j += 1;
                }
                _mm256_storeu_ps(logits.add(v0), acc);
                s += 8;
            }
            // s 尾部（nb % 8）
            while s < nb {
                let v = g * nb + s;
                let mut a = 0.0f32;
                for j in 0..k0e {
                    a += *wt.add(j * nv + v) * *ht.add(j * nb + s);
                }
                *logits.add(v) = a;
                s += 1;
            }
            g += 1;
        }
        // 行尾部
        let mut v = g * nb;
        while v < nv {
            let b = v % nb;
            let mut a = 0.0f32;
            for j in 0..k0e {
                a += *wt.add(j * nv + v) * *ht.add(j * nb + b);
            }
            *logits.add(v) = a;
            v += 1;
        }
    }

    /// 单 token：dWT[16][V] += h·dyᵀ（外积累加）；dh[128][16] += dy·W。
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn bwd(
        w: *const f32,
        ht: *const f32,
        dy: *const f32,
        dwt: *mut f32,
        dh: *mut f32,
        nv: usize,
        nb: usize,
        k0e: usize,
        part: u32,
    ) {
        let mask = nb - 1;
        if part & 1 != 0 {
        let mut g = 0usize;
        while g * nb + nb <= nv {
            let mut s = 0usize;
            while s + 8 <= nb {
                let v0 = g * nb + s;
                let dyv = _mm256_loadu_ps(dy.add(v0));
                let mut j = 0usize;
                while j < k0e {
                    let hv = _mm256_loadu_ps(ht.add(j * nb + s)); // 8 lane = 8 相邻块，hT 里连续
                    let p = _mm256_fmadd_ps(hv, dyv, _mm256_loadu_ps(dwt.add(j * nv + v0)));
                    _mm256_storeu_ps(dwt.add(j * nv + v0), p);
                    j += 1;
                }
                s += 8;
            }
            while s < nb {
                let v = g * nb + s;
                let d = *dy.add(v);
                for j in 0..k0e {
                    *dwt.add(j * nv + v) += *ht.add(j * nb + s) * d;
                }
                s += 1;
            }
            g += 1;
        }
        let mut v = g * nb;
        while v < nv {
            let b = v % nb;
            let d = *dy.add(v);
            for j in 0..k0e {
                *dwt.add(j * nv + v) += *ht.add(j * nb + b) * d;
            }
            v += 1;
        }
        } // part&1
        if part & 2 != 0 {
        // dh：逐行 16 float 连续加（W 行主序，dh 块主序）
        // ⚠️ nb 必须是 2 的幂：这里用位与替掉 `v % nb`（原型第一版用 % ，
        //    8198 行 × ~25 周期 = 每 token 200k 周期，占了原型耗时的大头）
        for v in 0..nv {
            let b = v & mask;
            let d = _mm256_set1_ps(*dy.add(v));
            let wrow = w.add(v * k0e);
            let drow = dh.add(b * k0e);
            let mut j = 0usize;
            while j + 8 <= k0e {
                let p = _mm256_fmadd_ps(d, _mm256_loadu_ps(wrow.add(j)), _mm256_loadu_ps(drow.add(j)));
                _mm256_storeu_ps(drow.add(j), p);
                j += 8;
            }
            while j < k0e {
                *drow.add(j) += *dy.add(v) * *wrow.add(j);
                j += 1;
            }
        }
        }
        let _ = hsum;
    }
}

#[cfg(target_arch = "x86_64")]
fn head_bench(dim: usize, t: usize, rep: usize, mode: &str) {
    use vmlm_core::b1_pure::{b1_backward_fused, b1_forward};
    let k0e = 16usize;
    let n_vocab = 8198usize;
    let nb = dim / k0e; // 128
    unsafe {
        // head 的 block 拓扑（与 trainer 的 io_sparse 路径逐字同构）
        let base = malloc(SparseConnW::bytes(n_vocab as u32, k0e as u32));
        let head = SparseConnW::from_base(base, n_vocab as u32, k0e as u32, 3);
        init_sparse_conn_scaled(&head, 1.0);
        for v in 0..n_vocab {
            for j in 0..k0e {
                let sid = ((v % nb) * k0e + j) as u32;
                let tt = head.idx(v, j);
                *head.src.add(tt) = sid as f32;
                *head.src_u32.add(tt) = sid;
            }
        }
        head.assert_src_u32_consistent();

        let h = alloc(t * dim);
        let dy = alloc(t * n_vocab); // dy 的行长是 n_vocab（head 的 dy 按"词表行"索引）
        let logits = alloc(t * n_vocab);
        let logits2 = alloc(t * n_vocab);
        let dwt = alloc(k0e * n_vocab);
        let dh = alloc(t * dim);
        let wt = alloc(k0e * n_vocab);
        let ht = alloc(k0e * nb); // 单 token 的 hT
        let mut r = Rng(0xABCD_1234);
        for i in 0..t * dim {
            *h.add(i) = r.uni();
        }
        for i in 0..t * n_vocab {
            *dy.add(i) = r.uni();
        }
        zero(dwt, k0e * n_vocab);
        zero(dh, t * dim);
        hb::pack_wt(head.w, wt, n_vocab, k0e);

        let conns = (t * n_vocab * k0e) as f64;
        println!(
            "head_bench vocab={} dim={} k0e={} blocks={} t={} | conns/round={}",
            n_vocab, dim, k0e, nb, t, conns
        );

        if mode == "headfwd" || mode == "headall" {
            let gen = |reps: usize| -> f64 {
                let t0 = std::time::Instant::now();
                for _ in 0..reps {
                    for i in 0..t {
                        b1_forward(&head, h.add(i * dim), logits.add(i * n_vocab));
                    }
                }
                t0.elapsed().as_secs_f64()
            };
            let fast = |reps: usize| -> f64 {
                let t0 = std::time::Instant::now();
                for _ in 0..reps {
                    for i in 0..t {
                        hb::pack_ht(h.add(i * dim), ht, nb, k0e);
                        hb::fwd(wt, ht, logits2.add(i * n_vocab), n_vocab, nb, k0e);
                    }
                }
                t0.elapsed().as_secs_f64()
            };
            // 行主序版（无转置、无 WT）
            let fast_rm = |reps: usize| -> f64 {
                let t0 = std::time::Instant::now();
                for _ in 0..reps {
                    for i in 0..t {
                        hb::fwd_rm(head.w, h.add(i * dim), logits2.add(i * n_vocab), n_vocab, nb, k0e);
                    }
                }
                t0.elapsed().as_secs_f64()
            };
            let _ = gen(2);
            let _ = fast(2);
            let _ = fast_rm(2);
            let fast_rm2 = |reps: usize| -> f64 {
                let t0 = std::time::Instant::now();
                for _ in 0..reps {
                    for i in 0..t {
                        hb::fwd_rm2(head.w, h.add(i * dim), logits2.add(i * n_vocab), n_vocab, nb, k0e);
                    }
                }
                t0.elapsed().as_secs_f64()
            };
            let _ = fast_rm2(2);
            let mut brm2 = f64::INFINITY;
            let mut brm = f64::INFINITY;
            let mut bg = f64::INFINITY;
            let mut bf = f64::INFINITY;
            for _ in 0..5 {
                let a = gen(rep) / rep as f64;
                if a < bg {
                    bg = a;
                }
                let b = fast(rep) / rep as f64;
                if b < bf {
                    bf = b;
                }
                let c = fast_rm(rep) / rep as f64;
                if c < brm {
                    brm = c;
                }
                let d = fast_rm2(rep) / rep as f64;
                if d < brm2 {
                    brm2 = d;
                }
            }
            println!(
                "  b1_forward(generic) {:9.3} ms/round {:7.4} ns/conn",
                bg * 1e3,
                bg * 1e9 / conns
            );
            println!(
                "  head_block fwd(WT) {:9.3} ms/round {:7.4} ns/conn  speedup {:.2}x",
                bf * 1e3,
                bf * 1e9 / conns,
                bg / bf
            );
            println!(
                "  head_rowmajor fwd  {:9.3} ms/round {:7.4} ns/conn  speedup {:.2}x",
                brm * 1e3,
                brm * 1e9 / conns,
                bg / brm
            );
            println!(
                "  head_rm2 fwd(v序)  {:9.3} ms/round {:7.4} ns/conn  speedup {:.2}x",
                brm2 * 1e3,
                brm2 * 1e9 / conns,
                bg / brm2
            );
            // 数值对账（ε）
            let mut mx = 0.0f32;
            let mut den = 0.0f32;
            for i in 0..t * n_vocab {
                let d = (*logits.add(i) - *logits2.add(i)).abs();
                if d > mx {
                    mx = d;
                }
                if (*logits.add(i)).abs() > den {
                    den = (*logits.add(i)).abs();
                }
            }
            println!("  max|generic-block| = {:.3e} (max|ref|={:.3e}, rel {:.2e})", mx, den, mx as f64 / den.max(1e-30) as f64);
        }

        if mode == "headbwd" || mode == "headall" {
            let gen = |reps: usize| -> f64 {
                let t0 = std::time::Instant::now();
                for _ in 0..reps {
                    zero(dwt, k0e * n_vocab);
                    for i in 0..t {
                        b1_backward_fused(
                            &head,
                            dy.add(i * n_vocab),
                            h.add(i * dim),
                            dwt,
                            dh.add(i * dim),
                        );
                    }
                }
                t0.elapsed().as_secs_f64()
            };
            let fast = |reps: usize| -> f64 {
                let part = arg_u32("--part", 3) as u32;
                let t0 = std::time::Instant::now();
                for _ in 0..reps {
                    zero(dwt, k0e * n_vocab);
                    for i in 0..t {
                        hb::pack_ht(h.add(i * dim), ht, nb, k0e);
                        hb::bwd(
                            head.w,
                            ht,
                            dy.add(i * n_vocab),
                            dwt,
                            dh.add(i * dim),
                            n_vocab,
                            nb,
                            k0e,
                            part,
                        );
                    }
                }
                t0.elapsed().as_secs_f64()
            };
            let _ = gen(2);
            let _ = fast(2);
            let mut bg = f64::INFINITY;
            let mut bf = f64::INFINITY;
            for _ in 0..5 {
                let a = gen(rep) / rep as f64;
                if a < bg {
                    bg = a;
                }
                let b = fast(rep) / rep as f64;
                if b < bf {
                    bf = b;
                }
            }
            println!(
                "  b1_backward_fused(generic) {:9.3} ms/round {:7.4} ns/conn",
                bg * 1e3,
                bg * 1e9 / conns
            );
            println!(
                "  head_block bwd     {:9.3} ms/round {:7.4} ns/conn  speedup {:.2}x",
                bf * 1e3,
                bf * 1e9 / conns,
                bg / bf
            );
        }

        if mode == "headbwdrm" || mode == "headallrm" || mode == "headall" {
            // 行主序反向：dw 行主序、dh 块主序，无 gather/无散写/无转置
            let generic = |reps: usize| -> f64 {
                let t0 = std::time::Instant::now();
                for _ in 0..reps {
                    zero(dwt, k0e * n_vocab);
                    for i in 0..t {
                        b1_backward_fused(&head, dy.add(i * n_vocab), h.add(i * dim), dwt, dh.add(i * dim));
                    }
                }
                t0.elapsed().as_secs_f64()
            };
            let rm = |reps: usize| -> f64 {
                let t0 = std::time::Instant::now();
                for _ in 0..reps {
                    zero(dwt, k0e * n_vocab);
                    for i in 0..t {
                        hb::bwd_rm(head.w, h.add(i * dim), dy.add(i * n_vocab), dwt, dh.add(i * dim), n_vocab, nb, k0e);
                    }
                }
                t0.elapsed().as_secs_f64()
            };
            let _ = generic(2);
            let _ = rm(2);
            let mut bg = f64::INFINITY;
            let mut bf = f64::INFINITY;
            for _ in 0..5 {
                let a = generic(rep) / rep as f64;
                if a < bg {
                    bg = a;
                }
                let b = rm(rep) / rep as f64;
                if b < bf {
                    bf = b;
                }
            }
            println!(
                "  b1_backward_fused(gen) {:9.3} ms/round {:7.4} ns/conn",
                bg * 1e3,
                bg * 1e9 / conns
            );
            println!(
                "  head_rowmajor bwd      {:9.3} ms/round {:7.4} ns/conn  speedup {:.2}x",
                bf * 1e3,
                bf * 1e9 / conns,
                bg / bf
            );
        }
        let mut guard = 0.0f32;
        for i in 0..t {
            guard += *logits.add(i * n_vocab) + *logits2.add(i * n_vocab) + *dh.add(i * dim);
        }
        if !guard.is_finite() {
            println!("  WARN guard {}", guard);
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn head_bench(_dim: usize, _t: usize, _rep: usize, _mode: &str) {}

/// t8（令牌分块 / source-major）前向 vs 发布版 gather 前向，含转置成本。
///
/// 参数含义：`--dim` = 神经元数 n；`--k0` = 每神经元连接数；`--t` = token 数。
fn t8_bench(n: usize, k0: usize, t: usize, rep: usize) {
    use vmlm_core::b1_pure::{b1_forward, b1_forward_t8, b1_transpose_activations};
    unsafe {
        // n 个神经元的源维度取 n（自连接拓扑，和 B1 层的形状一致）
        let d = n;
        let base = malloc(SparseConnW::bytes(n as u32, k0 as u32));
        let sc = SparseConnW::from_base(base, n as u32, k0 as u32, 7919);
        init_sparse_conn_scaled(&sc, 1.0);
        let t8 = (t + 7) & !7usize;
        let xoff = arg_u32("--xoff", 0);
        let yoff = arg_u32("--yoff", 0);
        let x = (alloc(t * d + xoff) as *mut f32).add(xoff);
        let y = (alloc(t * n + yoff) as *mut f32).add(yoff);
        let xt = alloc(d * t8);
        let yt = alloc(n * t8);
        let y2 = alloc(t * n);
        let mut r = Rng(0x77);
        for i in 0..t * d {
            *x.add(i) = r.uni();
        }
        let conns = (t * n * k0) as f64;
        println!("t8_bench n={} k0={} t={} t8={} | conns/round={}", n, k0, t, t8, conns);

        let gen = |reps: usize| -> f64 {
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                for i in 0..t {
                    b1_forward(&sc, x.add(i * d), y.add(i * n));
                }
            }
            t0.elapsed().as_secs_f64()
        };
        let fast = |reps: usize| -> f64 {
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                b1_transpose_activations(x, t, d, xt, t8);
                b1_forward_t8(&sc, xt, t8, t, yt);
                // 转回 token-major（下一层/CE 需要）—— **必须用 AVX2 分块转置**：
                // 第一版这里是标量双循环（262k 次跨步写），把 t8 的收益全吃掉了，
                // 得出「t8 慢 2 倍」的错误结论。
                b1_transpose_activations(yt, t, n, y2, t8);
            }
            t0.elapsed().as_secs_f64()
        };
        let _ = gen(2);
        let _ = fast(2);
        let mut bg = f64::INFINITY;
        let mut bf = f64::INFINITY;
        for _ in 0..5 {
            let a = gen(rep) / rep as f64;
            if a < bg {
                bg = a;
            }
            let b = fast(rep) / rep as f64;
            if b < bf {
                bf = b;
            }
        }
        println!("  gather   {:9.3} ms/round {:7.4} ns/conn", bg * 1e3, bg * 1e9 / conns);
        println!(
            "  t8(+2x转置) {:9.3} ms/round {:7.4} ns/conn  speedup {:.2}x",
            bf * 1e3,
            bf * 1e9 / conns,
            bg / bf
        );
        let mut mx = 0.0f32;
        let mut den = 0.0f32;
        for i in 0..t * n {
            let dd = (*y.add(i) - *y2.add(i)).abs();
            if dd > mx {
                mx = dd;
            }
            if (*y.add(i)).abs() > den {
                den = (*y.add(i)).abs();
            }
        }
        println!("  max|gather-t8| = {:.3e} (max|ref|={:.3e}, rel {:.2e})", mx, den, mx as f64 / den.max(1e-30) as f64);
    }
}

fn bwd_variants(dim: usize, k0: usize, t: usize, rep: usize) {
    use vmlm_core::b1_pure::b1_backward_fused;
    #[cfg(target_arch = "x86_64")]
    mod v {
        use core::arch::x86_64::*;
        /// V2：不再把 sid 存栈再标量读回，改用 vpextrd 直接取通道
        #[target_feature(enable = "avx2,fma")]
        pub unsafe fn v2(
            w: *const f32,
            src: *const u32,
            n: usize,
            k0: usize,
            dy: *const f32,
            x: *const f32,
            dw: *mut f32,
            dx: *mut f32,
        ) {
            let mut i = 0usize;
            while i < n {
                let base = i * k0;
                let g = *dy.add(i);
                let gb = _mm256_set1_ps(g);
                let mut j = 0usize;
                while j + 8 <= k0 {
                    let sid = _mm256_loadu_si256(src.add(base + j) as *const __m256i);
                    let xv = _mm256_i32gather_ps(x, sid, 4);
                    let wv = _mm256_loadu_ps(w.add(base + j));
                    let acc = _mm256_add_ps(_mm256_loadu_ps(dw.add(base + j)), _mm256_mul_ps(gb, xv));
                    _mm256_storeu_ps(dw.add(base + j), acc);
                    let _ = wv;
                    macro_rules! lane {
                        ($l:expr) => {{
                            let s = _mm256_extract_epi32(sid, $l) as usize;
                            *dx.add(s) += g * *w.add(base + j + $l);
                        }};
                    }
                    lane!(0);
                    lane!(1);
                    lane!(2);
                    lane!(3);
                    lane!(4);
                    lane!(5);
                    lane!(6);
                    lane!(7);
                    j += 8;
                }
                while j < k0 {
                    let s = *src.add(base + j) as usize;
                    let gx = g * *x.add(s);
                    *dw.add(base + j) += gx;
                    *dx.add(s) += g * *w.add(base + j);
                    j += 1;
                }
                i += 1;
            }
        }

        /// V3：2 行并行（两条独立 gather/散写链）
        #[target_feature(enable = "avx2,fma")]
        pub unsafe fn v3(
            w: *const f32,
            src: *const u32,
            n: usize,
            k0: usize,
            dy: *const f32,
            x: *const f32,
            dw: *mut f32,
            dx: *mut f32,
        ) {
            let mut i = 0usize;
            while i + 2 <= n {
                let b0 = i * k0;
                let b1 = b0 + k0;
                let g0 = *dy.add(i);
                let g1 = *dy.add(i + 1);
                let gb0 = _mm256_set1_ps(g0);
                let gb1 = _mm256_set1_ps(g1);
                let mut j = 0usize;
                while j + 8 <= k0 {
                    let s0 = _mm256_loadu_si256(src.add(b0 + j) as *const __m256i);
                    let s1 = _mm256_loadu_si256(src.add(b1 + j) as *const __m256i);
                    let x0 = _mm256_i32gather_ps(x, s0, 4);
                    let x1 = _mm256_i32gather_ps(x, s1, 4);
                    let a0 = _mm256_add_ps(_mm256_loadu_ps(dw.add(b0 + j)), _mm256_mul_ps(gb0, x0));
                    let a1 = _mm256_add_ps(_mm256_loadu_ps(dw.add(b1 + j)), _mm256_mul_ps(gb1, x1));
                    _mm256_storeu_ps(dw.add(b0 + j), a0);
                    _mm256_storeu_ps(dw.add(b1 + j), a1);
                    let mut tmp = [0u32; 8];
                    _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, s0);
                    for l in 0..8 {
                        *dx.add(tmp[l] as usize) += g0 * *w.add(b0 + j + l);
                    }
                    _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, s1);
                    for l in 0..8 {
                        *dx.add(tmp[l] as usize) += g1 * *w.add(b1 + j + l);
                    }
                    j += 8;
                }
                while j < k0 {
                    let s = *src.add(b0 + j) as usize;
                    *dw.add(b0 + j) += g0 * *x.add(s);
                    *dx.add(s) += g0 * *w.add(b0 + j);
                    let s = *src.add(b1 + j) as usize;
                    *dw.add(b1 + j) += g1 * *x.add(s);
                    *dx.add(s) += g1 * *w.add(b1 + j);
                    j += 1;
                }
                i += 2;
            }
            while i < n {
                let base = i * k0;
                let g = *dy.add(i);
                for j in 0..k0 {
                    let s = *src.add(base + j) as usize;
                    *dw.add(base + j) += g * *x.add(s);
                    *dx.add(s) += g * *w.add(base + j);
                }
                i += 1;
            }
        }
    }
    unsafe {
        let base = malloc(SparseConnW::bytes(dim as u32, k0 as u32));
        let sc = SparseConnW::from_base(base, dim as u32, k0 as u32, 7919);
        init_sparse_conn_scaled(&sc, 1.0);
        let x = alloc(t * dim);
        let dy = alloc(t * dim);
        let dx = alloc(t * dim);
        let dw = alloc(dim * k0);
        let mut r = Rng(0x5555);
        for i in 0..t * dim {
            *x.add(i) = r.uni();
            *dy.add(i) = r.uni();
        }
        let conns = (t * dim * k0) as f64;
        let run = |limit: i32, reps: usize| -> f64 {
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                let _ = limit;
                for i in 0..t {
                    b1_backward_fused(&sc, dy.add(i * dim), x.add(i * dim), dw, dx.add(i * dim));
                }
            }
            t0.elapsed().as_secs_f64()
        };
        let _ = run(0, 2);
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let s = run(0, rep) / rep as f64;
            if s < best {
                best = s;
            }
        }
        println!(
            "bwd_variants dim={} k0={} : {:.4} ns/conn ({:.3} ms/round)",
            dim,
            k0,
            best * 1e9 / conns,
            best * 1e3
        );
        #[cfg(target_arch = "x86_64")]
        {
            let run2 = |reps: usize, which: u32| -> f64 {
                let t0 = std::time::Instant::now();
                for _ in 0..reps {
                    for i in 0..t {
                        if which == 2 {
                            v::v2(sc.w, sc.src_u32, dim, k0, dy.add(i * dim), x.add(i * dim), dw, dx.add(i * dim));
                        } else {
                            v::v3(sc.w, sc.src_u32, dim, k0, dy.add(i * dim), x.add(i * dim), dw, dx.add(i * dim));
                        }
                    }
                }
                t0.elapsed().as_secs_f64()
            };
            for which in [2u32, 3u32] {
                let _ = run2(2, which);
                let mut b2 = f64::INFINITY;
                for _ in 0..5 {
                    let s = run2(rep, which) / rep as f64;
                    if s < b2 {
                        b2 = s;
                    }
                }
                println!(
                    "  variant V{}          : {:.4} ns/conn ({:.3} ms/round)  vs V1 {:.2}x",
                    which,
                    b2 * 1e9 / conns,
                    b2 * 1e3,
                    best / b2
                );
            }
        }
    }
}

/// 小 k0（1/2/4/8）前向：把 G = 8/k0 个**连续行**放进一个 ymm（每行 k0 个通道），
/// 一次 gather + 一次连续 w/src 载入覆盖 G 行 ⇒ 每行摊薄到 1/G 次 gather。
///
/// 为什么：实测 k0=4 的发布版前向是 **0.476 ns/conn**（k0=8 只要 0.19），
/// 因为 `while j+8<=k0` 一次都不成立 ⇒ 整行走标量尾部循环，每行 ~20 uops 全花在
/// 行开销（函数调用/累加链/横向归约）上。这里把行开销按 G 摊薄。
#[cfg(target_arch = "x86_64")]
mod smallk {
    use core::arch::x86_64::*;

    #[inline(always)]
    unsafe fn hsum128(v: __m128) -> f32 {
        let s = _mm_add_ps(v, _mm_movehl_ps(v, v));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
        _mm_cvtss_f32(s)
    }

    /// k0 ∈ {1,4,8}；返回处理的行数（调用方负责尾部）。
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn fwd(w: *const f32, src: *const u32, x: *const f32, k0: usize, n: usize, y: *mut f32) -> usize {
        if k0 == 4 {
            let mut i = 0usize;
            while i + 2 <= n {
                let base = i * k0;
                let sid = _mm256_loadu_si256(src.add(base) as *const __m256i);
                let xv = _mm256_i32gather_ps(x, sid, 4);
                let wv = _mm256_loadu_ps(w.add(base));
                let acc = _mm256_mul_ps(wv, xv);
                *y.add(i) = hsum128(_mm256_castps256_ps128(acc));
                *y.add(i + 1) = hsum128(_mm256_extractf128_ps(acc, 1));
                i += 2;
            }
            return i;
        }
        if k0 == 8 {
            let mut i = 0usize;
            while i + 1 <= n {
                let base = i * k0;
                let sid = _mm256_loadu_si256(src.add(base) as *const __m256i);
                let xv = _mm256_i32gather_ps(x, sid, 4);
                let wv = _mm256_loadu_ps(w.add(base));
                let acc = _mm256_mul_ps(wv, xv);
                *y.add(i) = hsum128(_mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1)));
                i += 1;
            }
            return i;
        }
        0
    }
}

fn smallk_bench(n: usize, k0: usize, t: usize, rep: usize) {
    use vmlm_core::b1_pure::b1_forward;
    unsafe {
        let base = malloc(SparseConnW::bytes(n as u32, k0 as u32));
        let sc = SparseConnW::from_base(base, n as u32, k0 as u32, 7919);
        init_sparse_conn_scaled(&sc, 1.0);
        let x = alloc(t * n);
        let y = alloc(t * n);
        let y2 = alloc(t * n);
        let mut r = Rng(0x99);
        for i in 0..t * n {
            *x.add(i) = r.uni();
        }
        let conns = (t * n * k0) as f64;
        let gen = |reps: usize| -> f64 {
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                for i in 0..t {
                    b1_forward(&sc, x.add(i * n), y.add(i * n));
                }
            }
            t0.elapsed().as_secs_f64()
        };
        let fast = |reps: usize| -> f64 {
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                for i in 0..t {
                    let done = smallk::fwd(sc.w, sc.src_u32, x.add(i * n), k0, n, y2.add(i * n));
                    for r2 in done..n {
                        *y2.add(i * n + r2) = vmlm_core::kernels::b1_dot_row_u32_scalar(
                            sc.w.add(r2 * k0),
                            sc.src_u32.add(r2 * k0),
                            x.add(i * n),
                            k0,
                        );
                    }
                }
            }
            t0.elapsed().as_secs_f64()
        };
        let _ = gen(2);
        let _ = fast(2);
        let mut bg = f64::INFINITY;
        let mut bf = f64::INFINITY;
        for _ in 0..5 {
            let a = gen(rep) / rep as f64;
            if a < bg {
                bg = a;
            }
            let b = fast(rep) / rep as f64;
            if b < bf {
                bf = b;
            }
        }
        println!("smallk_bench n={} k0={} t={} | conns/round={}", n, k0, t, conns);
        println!("  b1_forward  {:9.3} ms/round {:7.4} ns/conn", bg * 1e3, bg * 1e9 / conns);
        println!(
            "  smallk      {:9.3} ms/round {:7.4} ns/conn  speedup {:.2}x",
            bf * 1e3,
            bf * 1e9 / conns,
            bg / bf
        );
        let mut mx = 0.0f32;
        let mut den = 0.0f32;
        for i in 0..t * n {
            let dd = (*y.add(i) - *y2.add(i)).abs();
            if dd > mx {
                mx = dd;
            }
            if (*y.add(i)).abs() > den {
                den = (*y.add(i)).abs();
            }
        }
        println!("  max|diff| = {:.3e} (max|ref|={:.3e}, rel {:.2e})", mx, den, mx as f64 / den.max(1e-30) as f64);
    }
}

/// t8（source-major）反向原型：dw 按 token 维向量化 + dx 连续 RMW（无 gather、无散写）
///
/// 布局：`xt[d][t8]`（源主序激活）、`dyt[n][t8]`（源主序 dy）、`dxt[d][t8]`（源主序 dx）。
/// 每条连接：`dw[i][j] += Σ_t dyt[i][t]·xt[s][t]`（点积，t 维向量化）、
/// `dxt[s][:] += dyt[i][:]·w[i][j]`（连续 32B RMW）。
/// 8x8 分块 AVX2 反置：`out[i*os + s] = a[s*stride + i]`（用于 dxt[d][t8] -> dx[t][d]）
/// 直接对拍：t8 drop-in 反向 vs 发布版逐 token fused 反向（dw/dx）
fn t8_check(n: usize, k0: usize, t: usize) {
    use vmlm_core::b1_pure::{b1_backward_fused, b1_layer_backward_t8};
    unsafe {
        let d = n;
        let base = malloc(SparseConnW::bytes(n as u32, k0 as u32));
        let sc = SparseConnW::from_base(base, n as u32, k0 as u32, 7919);
        init_sparse_conn_scaled(&sc, 1.0);
        let x = alloc(t * d);
        let dy = alloc(t * n);
        let dw1 = alloc(n * k0);
        let dw2 = alloc(n * k0);
        let dx1 = alloc(t * d);
        let dx2 = alloc(t * d);
        let t8 = (t + 7) & !7usize;
        let xt = alloc(d * t8);
        let dyt = alloc(n * t8);
        let dxt = alloc(d * t8);
        let mut r = Rng(0xFEED);
        for i in 0..t * d {
            *x.add(i) = r.uni();
        }
        for i in 0..t * n {
            *dy.add(i) = r.uni();
        }
        zero(dw1, n * k0);
        zero(dw2, n * k0);
        zero(dx1, t * d);
        zero(dx2, t * d);
        for i in 0..t {
            b1_backward_fused(&sc, dy.add(i * n), x.add(i * d), dw1, dx1.add(i * d));
        }
        b1_layer_backward_t8(&sc, dy, n, x, d, t, dw2, dx2, xt, dyt, dxt, t8);
        let mut mw1 = 0.0f32;
        let mut mw2 = 0.0f32;
        let mut dwmax = 0.0f32;
        for i in 0..n * k0 {
            mw1 = mw1.max((*dw1.add(i)).abs());
            mw2 = mw2.max((*dw2.add(i)).abs());
            dwmax = dwmax.max((*dw1.add(i) - *dw2.add(i)).abs());
        }
        let mut mx1 = 0.0f32;
        let mut mx2 = 0.0f32;
        let mut dxmax = 0.0f32;
        for i in 0..t * d {
            mx1 = mx1.max((*dx1.add(i)).abs());
            mx2 = mx2.max((*dx2.add(i)).abs());
            dxmax = dxmax.max((*dx1.add(i) - *dx2.add(i)).abs());
        }
        println!("t8check n={} k0={} t={} t8={}", n, k0, t, t8);
        println!("  |dw|  fused={:.6e} t8={:.6e}  maxdiff={:.6e}  ratio={:.4}", mw1, mw2, dwmax, mw2 / mw1.max(1e-30));
        println!("  |dx|  fused={:.6e} t8={:.6e}  maxdiff={:.6e}  ratio={:.4}", mx1, mx2, dxmax, mx2 / mx1.max(1e-30));
        // 隔离：用**标量**把 dxt 反置成 dx3，与 fused 对拍（区分「内核算错」还是「AVX2 转置写错」）
        let dx3 = alloc(t * d);
        for i in 0..t {
            for s in 0..d {
                *dx3.add(i * d + s) = *dxt.add(s * t8 + i);
            }
        }
        let mut mx3 = 0.0f32;
        let mut d3 = 0.0f32;
        for i in 0..t * d {
            mx3 = mx3.max((*dx3.add(i)).abs());
            d3 = d3.max((*dx1.add(i) - *dx3.add(i)).abs());
        }
        println!("  scalar-transpose dxt: |dx|={:.6e} maxdiff_vs_fused={:.6e}", mx3, d3);
        println!("  dx[0..6] fused={:?}", (0..6).map(|k| *dx1.add(k)).collect::<Vec<_>>());
        println!("  dx[0..6] t8avx2={:?}", (0..6).map(|k| *dx2.add(k)).collect::<Vec<_>>());
        println!("  dx[0..6] scalar={:?}", (0..6).map(|k| *dx3.add(k)).collect::<Vec<_>>());
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn t8_back(a: *const f32, rows: usize, cols: usize, out: *mut f32, os: usize, stride: usize) {
    use core::arch::x86_64::*;
    let mut s0 = 0usize;
    while s0 < rows {
        let mut i0 = 0usize;
        while i0 < cols {
            let n_s = (rows - s0).min(8);
            let n_i = (cols - i0).min(8);
            if n_s == 8 && n_i == 8 {
                let r0 = _mm256_loadu_ps(a.add((s0 + 0) * stride + i0));
                let r1 = _mm256_loadu_ps(a.add((s0 + 1) * stride + i0));
                let r2 = _mm256_loadu_ps(a.add((s0 + 2) * stride + i0));
                let r3 = _mm256_loadu_ps(a.add((s0 + 3) * stride + i0));
                let r4 = _mm256_loadu_ps(a.add((s0 + 4) * stride + i0));
                let r5 = _mm256_loadu_ps(a.add((s0 + 5) * stride + i0));
                let r6 = _mm256_loadu_ps(a.add((s0 + 6) * stride + i0));
                let r7 = _mm256_loadu_ps(a.add((s0 + 7) * stride + i0));
                let a0 = _mm256_unpacklo_ps(r0, r1);
                let a1 = _mm256_unpackhi_ps(r0, r1);
                let a2 = _mm256_unpacklo_ps(r2, r3);
                let a3 = _mm256_unpackhi_ps(r2, r3);
                let a4 = _mm256_unpacklo_ps(r4, r5);
                let a5 = _mm256_unpackhi_ps(r4, r5);
                let a6 = _mm256_unpacklo_ps(r6, r7);
                let a7 = _mm256_unpackhi_ps(r6, r7);
                let b0 = _mm256_shuffle_ps(a0, a2, 0x44);
                let b1 = _mm256_shuffle_ps(a0, a2, 0xEE);
                let b2 = _mm256_shuffle_ps(a1, a3, 0x44);
                let b3 = _mm256_shuffle_ps(a1, a3, 0xEE);
                let b4 = _mm256_shuffle_ps(a4, a6, 0x44);
                let b5 = _mm256_shuffle_ps(a4, a6, 0xEE);
                let b6 = _mm256_shuffle_ps(a5, a7, 0x44);
                let b7 = _mm256_shuffle_ps(a5, a7, 0xEE);
                let c0 = _mm256_permute2f128_ps(b0, b4, 0x20);
                let c1 = _mm256_permute2f128_ps(b1, b5, 0x20);
                let c2 = _mm256_permute2f128_ps(b2, b6, 0x20);
                let c3 = _mm256_permute2f128_ps(b3, b7, 0x20);
                let c4 = _mm256_permute2f128_ps(b0, b4, 0x31);
                let c5 = _mm256_permute2f128_ps(b1, b5, 0x31);
                let c6 = _mm256_permute2f128_ps(b2, b6, 0x31);
                let c7 = _mm256_permute2f128_ps(b3, b7, 0x31);
                let cs = [c0, c1, c2, c3, c4, c5, c6, c7];
                for (cc, v) in cs.iter().enumerate() {
                    _mm256_storeu_ps(out.add((i0 + cc) * os + s0), *v);
                }
            } else {
                for r in 0..n_s {
                    for cc in 0..n_i {
                        *out.add((i0 + cc) * os + s0 + r) = *a.add((s0 + r) * stride + i0 + cc);
                    }
                }
            }
            i0 += 8;
        }
        s0 += 8;
    }
}

#[cfg(target_arch = "x86_64")]
fn t8bwd_bench(n: usize, k0: usize, t: usize, rep: usize) {
    use core::arch::x86_64::*;
    use vmlm_core::b1_pure::{b1_backward_fused, b1_transpose_activations};
    unsafe {
        let d = n;
        let base = malloc(SparseConnW::bytes(n as u32, k0 as u32));
        let sc = SparseConnW::from_base(base, n as u32, k0 as u32, 7919);
        init_sparse_conn_scaled(&sc, 1.0);
        let t8 = (t + 7) & !7usize;
        let x = alloc(t * d);
        let dy = alloc(t * n);
        let dx = alloc(t * d);
        let dw = alloc(n * k0);
        let xt = alloc(d * t8);
        let dyt = alloc(n * t8);
        let dxt = alloc(d * t8);
        let dw2 = alloc(n * k0);
        let mut r = Rng(0xACE);
        for i in 0..t * d {
            *x.add(i) = r.uni();
            *dx.add(i) = r.uni();
        }
        for i in 0..t * n {
            *dy.add(i) = r.uni();
        }
        let conns = (t * n * k0) as f64;

        #[target_feature(enable = "avx2,fma")]
        unsafe fn k(xt: *const f32, dyt: *const f32, src: *const u32, w: *const f32,
                    dxt: *mut f32, dw: *mut f32, n: usize, k0: usize, t8: usize) {
            #[inline(always)]
            unsafe fn hsum(v: __m256) -> f32 {
                let lo = _mm256_castps256_ps128(v);
                let hi = _mm256_extractf128_ps(v, 1);
                let s = _mm_add_ps(lo, hi);
                let s = _mm_hadd_ps(s, s);
                let s = _mm_hadd_ps(s, s);
                _mm_cvtss_f32(s)
            }
            for i in 0..n {
                let dy_row = dyt.add(i * t8);
                let b = i * k0;
                for j in 0..k0 {
                    let s = *src.add(b + j) as usize;
                    let wij = *w.add(b + j);
                    let xr = xt.add(s * t8);
                    let dr = dxt.add(s * t8);
                    let mut acc = _mm256_setzero_ps();
                    let mut p = 0usize;
                    while p + 8 <= t8 {
                        acc = _mm256_fmadd_ps(_mm256_loadu_ps(dy_row.add(p)), _mm256_loadu_ps(xr.add(p)), acc);
                        p += 8;
                    }
                    *dw.add(b + j) += hsum(acc);
                    let vw = _mm256_set1_ps(wij);
                    let mut p = 0usize;
                    while p + 8 <= t8 {
                        let v = _mm256_fmadd_ps(vw, _mm256_loadu_ps(xr.add(p)), _mm256_loadu_ps(dr.add(p)));
                        _mm256_storeu_ps(dr.add(p), v);
                        p += 8;
                    }
                }
            }
        }

        // 参考：发布版 fused（逐 token）
        let gen = |reps: usize| -> f64 {
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                zero(dw, n * k0);
                zero(dx, t * d);
                for i in 0..t {
                    b1_backward_fused(&sc, dy.add(i * n), x.add(i * d), dw, dx.add(i * d));
                }
            }
            t0.elapsed().as_secs_f64()
        };
        let fast = |reps: usize| -> f64 {
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                zero(dw2, n * k0);
                zero(dxt, d * t8);
                b1_transpose_activations(x, t, d, xt, t8);
                b1_transpose_activations(dy, t, n, dyt, t8);
                k(xt, dyt, sc.src_u32, sc.w, dxt, dw2, n, k0, t8);
                t8_back(dxt, d, t8, dx, d, t8);
            }
            t0.elapsed().as_secs_f64()
        };
        let _ = gen(2);
        let _ = fast(2);
        let mut bg = f64::INFINITY;
        let mut bf = f64::INFINITY;
        for _ in 0..5 {
            let a = gen(rep) / rep as f64;
            if a < bg { bg = a; }
            let b = fast(rep) / rep as f64;
            if b < bf { bf = b; }
        }
        println!("t8bwd n={} k0={} t={} t8={} | conns/round={}", n, k0, t, t8, conns);
        println!("  fused/token  {:9.3} ms/round {:7.4} ns/conn", bg * 1e3, bg * 1e9 / conns);
        println!("  t8(dx+dw)    {:9.3} ms/round {:7.4} ns/conn  speedup {:.2}x", bf * 1e3, bf * 1e9 / conns, bg / bf);
    }
}

/// [P21-4] 对拍 + 计时：批量 head 反向 vs 逐 token 版
#[cfg(target_arch = "x86_64")]
fn head_batch_bench(t: usize, rep: usize) {
    use vmlm_core::kernels::{head_bwd_batched_avx2, head_block_bwd_avx2, transpose_t8_avx2, transpose_t8_back_avx2};
    let k0e = 16usize;
    let k0h = 16usize;
    let nv = 8198usize;
    let dim = 2048usize;
    let nb = dim / k0e;
    unsafe {
        let base = malloc(SparseConnW::bytes(nv as u32, k0h as u32));
        let head = SparseConnW::from_base(base, nv as u32, k0h as u32, 3);
        init_sparse_conn_scaled(&head, 1.0);
        for v in 0..nv {
            for j in 0..k0h {
                let sid = ((v % nb) * k0e + j) as u32;
                let tt = head.idx(v, j);
                *head.src.add(tt) = sid as f32;
                *head.src_u32.add(tt) = sid;
            }
        }
        let tt = (t + 7) & !7usize;
        let h = alloc(t * dim);
        let dy = alloc(t * nv);
        let dw1 = alloc(nv * k0h);
        let dw2 = alloc(nv * k0h);
        let dh1 = alloc(t * dim);
        let dh2 = alloc(t * dim);
        let dyT = alloc(nv * tt);
        let htT = alloc(dim * tt);
        let dht = alloc(dim * tt);
        let mut r = Rng(0xBEEF);
        for i in 0..t * dim {
            *h.add(i) = r.uni();
        }
        for i in 0..t * nv {
            *dy.add(i) = r.uni();
        }
        zero(dw1, nv * k0h);
        zero(dw2, nv * k0h);
        zero(dh1, t * dim);
        zero(dh2, t * dim);
        // 参考：逐 token
        for i in 0..t {
            head_block_bwd_avx2(head.w, h.add(i * dim), dy.add(i * nv), dw1, dh1.add(i * dim), nv, nb, k0e, k0h);
        }
        // 批量
        transpose_t8_avx2(h, t, dim, htT, tt);
        transpose_t8_avx2(dy, t, nv, dyT, tt);
        zero(dht, dim * tt);
        head_bwd_batched_avx2(head.w, htT, dyT, tt, nv, nb, k0e, k0h, dw2, dht);
        transpose_t8_back_avx2(dht, dim, t, dh2, dim, tt);
        let mut m1 = 0.0f32;
        let mut m2 = 0.0f32;
        let mut md = 0.0f32;
        for i in 0..nv * k0h {
            m1 = m1.max((*dw1.add(i)).abs());
            m2 = m2.max((*dw2.add(i)).abs());
            md = md.max((*dw1.add(i) - *dw2.add(i)).abs());
        }
        let mut x1 = 0.0f32;
        let mut x2 = 0.0f32;
        let mut xd = 0.0f32;
        for i in 0..t * dim {
            x1 = x1.max((*dh1.add(i)).abs());
            x2 = x2.max((*dh2.add(i)).abs());
            xd = xd.max((*dh1.add(i) - *dh2.add(i)).abs());
        }
        println!("head_batch t={} tt={} nv={} dim={} nb={}", t, tt, nv, dim, nb);
        println!("  dw |ref|={:.6e} batch={:.6e} maxdiff={:.6e} ratio={:.5}", m1, m2, md, m2 / m1.max(1e-30));
        println!("  dh |ref|={:.6e} batch={:.6e} maxdiff={:.6e} ratio={:.5}", x1, x2, xd, x2 / x1.max(1e-30));
        let conns = (t * nv * k0h) as f64;
        let gen = |reps: usize| -> f64 {
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                zero(dw1, nv * k0h);
                zero(dh1, t * dim);
                for i in 0..t {
                    head_block_bwd_avx2(head.w, h.add(i * dim), dy.add(i * nv), dw1, dh1.add(i * dim), nv, nb, k0e, k0h);
                }
            }
            t0.elapsed().as_secs_f64()
        };
        let fast = |reps: usize| -> f64 {
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                zero(dw2, nv * k0h);
                zero(dht, dim * tt);
                transpose_t8_avx2(h, t, dim, htT, tt);
                transpose_t8_avx2(dy, t, nv, dyT, tt);
                head_bwd_batched_avx2(head.w, htT, dyT, tt, nv, nb, k0e, k0h, dw2, dht);
                transpose_t8_back_avx2(dht, dim, t, dh2, dim, tt);
            }
            t0.elapsed().as_secs_f64()
        };
        let _ = gen(2);
        let _ = fast(2);
        let mut bg = f64::INFINITY;
        let mut bf = f64::INFINITY;
        for _ in 0..5 {
            let a = gen(rep) / rep as f64;
            if a < bg { bg = a; }
            let b = fast(rep) / rep as f64;
            if b < bf { bf = b; }
        }
        println!("  per-token {:9.3} ms/round {:7.4} ns/conn", bg * 1e3, bg * 1e9 / conns);
        println!("  batched   {:9.3} ms/round {:7.4} ns/conn  speedup {:.2}x", bf * 1e3, bf * 1e9 / conns, bg / bf);
    }
}

/// [P21-6] 第四组（整栈 source-major）的**前提验证**：
/// 12 层 B1 在 t8 布局下**层间零转置**（只有整栈入口/出口各一次），
/// 对比现状（token-major、逐 token gather、零转置）。
/// 这条直接回答「整栈改造值不值」——不用先动 step_fn。
#[cfg(target_arch = "x86_64")]
fn t8stack_bench(n: usize, k0: usize, t: usize, layers: usize, rep: usize) {
    use vmlm_core::b1_pure::{b1_forward, b1_forward_t8, b1_transpose_activations};
    unsafe {
        let d = n;
        let t8 = (t + 7) & !7usize;
        let mut scs: Vec<SparseConnW> = Vec::new();
        for l in 0..layers {
            let base = malloc(SparseConnW::bytes(n as u32, k0 as u32));
            let sc = SparseConnW::from_base(base, n as u32, k0 as u32, 7919 + l as u32);
            init_sparse_conn_scaled(&sc, 1.0);
            scs.push(sc);
        }
        let x = alloc(t * d);
        let y = alloc(t * d);
        let ta = alloc(d * t8);
        let tb = alloc(d * t8);
        let mut r = Rng(0x5150);
        for i in 0..t * d {
            *x.add(i) = r.uni();
        }
        let conns = (t * n * k0 * layers) as f64;
        // 现状：12 层 token-major，层间 copy，无需转置
        let gen = |reps: usize| -> f64 {
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                for l in 0..layers {
                    for i in 0..t {
                        b1_forward(&scs[l], x.add(i * d), y.add(i * d));
                    }
                    std::ptr::copy_nonoverlapping(y, x, t * d);
                }
            }
            t0.elapsed().as_secs_f64()
        };
        // 整栈 t8：入口 1 次转置 + 层间零转置 + 出口 1 次转置
        let fast = |reps: usize| -> f64 {
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                b1_transpose_activations(x, t, d, ta, t8);
                for l in 0..layers {
                    if l % 2 == 0 {
                        b1_forward_t8(&scs[l], ta, t8, t, tb);
                    } else {
                        b1_forward_t8(&scs[l], tb, t8, t, ta);
                    }
                }
                if layers % 2 == 0 {
                    b1_transpose_activations(tb, t, d, y, t8);
                } else {
                    b1_transpose_activations(ta, t, d, y, t8);
                }
            }
            t0.elapsed().as_secs_f64()
        };
        let _ = gen(2);
        let _ = fast(2);
        let mut bg = f64::INFINITY;
        let mut bf = f64::INFINITY;
        for _ in 0..4 {
            let a = gen(rep) / rep as f64;
            if a < bg { bg = a; }
            let b = fast(rep) / rep as f64;
            if b < bf { bf = b; }
        }
        println!("t8stack n={} k0={} t={} layers={} t8={}", n, k0, t, layers, t8);
        println!("  现状 token-major 12 层   {:9.3} ms/round {:7.4} ns/conn", bg * 1e3, bg * 1e9 / conns);
        println!("  整栈 t8（2 次转置）       {:9.3} ms/round {:7.4} ns/conn  speedup {:.2}x",
                 bf * 1e3, bf * 1e9 / conns, bg / bf);
    }
}

/// [P23] bf16 A/B：生产形状（n_neurons=4096、k0=8 ⇒ 每 token 32768 个连接）下
/// 三种权重载入方式的 ns/conn：
///   (a) f32 权重（现状）
///   (b) bf16 存储 + **内层循环内联扩展**（vpmovzxwd + vpslld，端口 5 开销 O(每 8 权重)）
///   (c) bf16 存储 + **加载时一次性扩成 f32**（内存 ×2，内层零扩展 ⇒ 等价于 (a) 的内核）
#[cfg(target_arch = "x86_64")]
fn bf16_bench(n: usize, k0: usize, t: usize, rep: usize) {
    use vmlm_core::kernels::{dot256_bf16_avx2, dot256_f32, f32_to_bf16};
    unsafe {
        let per = n * k0;
        let w = alloc(per);
        let wh: *mut u16 = { let mut v = vec![0u16; per]; let p = v.as_mut_ptr(); std::mem::forget(v); p };
        let mut idx = Vec::with_capacity(per);
        let x = alloc(per);
        let mut r = Rng(0xBF16);
        for i in 0..per {
            *w.add(i) = r.uni() * 0.05 - 0.025;
            *x.add(i) = r.uni();
            idx.push((i * 7919) % per);
        }
        let wp = alloc(per);
        for i in 0..per {
            let g = *w.add(i);
            *wp.add(i) = vmlm_core::kernels::bf16_to_f32(f32_to_bf16(g));
            *wh.add(i) = f32_to_bf16(g);
        }
        let conns = (per * t) as f64;

        // 直接用闭包跑三种内核
        let f32v = |reps: usize| -> (f64, f32) {
            let t0 = std::time::Instant::now();
            let mut s = 0.0f32;
            for _ in 0..reps {
                for i in 0..t {
                    s += dot256_f32(w, x, per);
                }
            }
            (t0.elapsed().as_secs_f64(), s)
        };
        let b16v = |reps: usize| -> (f64, f32) {
            let t0 = std::time::Instant::now();
            let mut s = 0.0f32;
            for _ in 0..reps {
                for i in 0..t {
                    s += dot256_bf16_avx2(x, wh, per);
                }
            }
            (t0.elapsed().as_secs_f64(), s)
        };
        let prev = |reps: usize| -> (f64, f32) {
            let t0 = std::time::Instant::now();
            let mut s = 0.0f32;
            for _ in 0..reps {
                for i in 0..t {
                    s += dot256_f32(x, wp, per);
                }
            }
            (t0.elapsed().as_secs_f64(), s)
        };
        let (_, _) = f32v(1);
        let (_, _) = b16v(1);
        let (_, _) = prev(1);
        let mut best = [f64::INFINITY; 3];
        let mut sums = [0.0f32; 3];
        for _ in 0..4 {
            let a = f32v(rep); if a.0 < best[0] { best[0] = a.0; sums[0] = a.1; }
            let b = b16v(rep); if b.0 < best[1] { best[1] = b.0; sums[1] = b.1; }
            let c = prev(rep); if c.0 < best[2] { best[2] = c.0; sums[2] = c.1; }
        }
        println!("bf16 A/B  n={} k0={} per_token={} t={}", n, k0, per, t);
        for (i, name) in ["(a) f32 权重        ", "(b) bf16 内联扩展   ", "(c) bf16 预扩成 f32 "].iter().enumerate() {
            println!("  {} {:9.3} ms/round  {:7.4} ns/conn  sum={:.6e}", name, best[i] * 1e3,
                     best[i] * 1e9 / conns, sums[i]);
        }
        println!("  内联扩展相对 f32 代价 {:.2}x；预扩展相对 f32 {:.2}x",
                 best[1] / best[0], best[2] / best[0]);
        println!("  内联 vs 预扩展差值 {:+.3} ms/round（每 token {} 个权重 ⇒ 每 8 个权重多 2 条微指令，其中 1 条占端口 5）",
                 (best[1] - best[2]) * 1e3, per);
    }
}

/// [P23] A/B 1：地址计算负担的源级对照（Rust 里无法手写 `lea`，只能改源级形态）。
/// 两个变体做同一件事（t=256 行 × per_token=32768 个权重的点积），只有**取址方式**不同：
///   (a) 全程指针步进（跨行只 `add(per)`，列内 `add(16)`）
///   (b) 每次用 `row * per + x` 重算下标（迫使地址运算进入循环体）
#[cfg(target_arch = "x86_64")]
fn addr_bench(n: usize, k0: usize, t: usize, rep: usize) {
    #[inline]
    unsafe fn hs(v: std::arch::x86_64::__m256) -> f32 {
        use std::arch::x86_64::*;
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps(v, 1);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        _mm_cvtss_f32(s)
    }
    use std::arch::x86_64::*;
    unsafe {
        let per = n * k0;
        let tot = per * t;
        let a = alloc(tot);
        let b = alloc(tot);
        let mut r = Rng(0xADDE);
        for i in 0..tot {
            *a.add(i) = r.uni() * 0.05;
            *b.add(i) = r.uni();
        }
        let conns = tot as f64;
        // (a) 指针步进
        let va = |reps: usize| -> (f64, f32) {
            let t0 = std::time::Instant::now();
            let mut s = 0.0f32;
            for _ in 0..reps {
                let mut ap = a;
                let mut bp = b;
                for _ in 0..t {
                    let mut acc0 = _mm256_setzero_ps();
                    let mut acc1 = _mm256_setzero_ps();
                    let mut x = 0usize;
                    while x + 16 <= per {
                        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(x)), _mm256_loadu_ps(bp.add(x)), acc0);
                        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(x + 8)), _mm256_loadu_ps(bp.add(x + 8)), acc1);
                        x += 16;
                    }
                    s += hs(_mm256_add_ps(acc0, acc1));
                    ap = ap.add(per);
                    bp = bp.add(per);
                }
            }
            (t0.elapsed().as_secs_f64(), s)
        };
        // (b) 下标重算
        let vb = |reps: usize| -> (f64, f32) {
            let t0 = std::time::Instant::now();
            let mut s = 0.0f32;
            for _ in 0..reps {
                for row in 0..t {
                    let base = row * per;
                    let mut acc0 = _mm256_setzero_ps();
                    let mut acc1 = _mm256_setzero_ps();
                    let mut x = 0usize;
                    while x + 16 <= per {
                        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(base + x)), _mm256_loadu_ps(b.add(base + x)), acc0);
                        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(base + x + 8)), _mm256_loadu_ps(b.add(base + x + 8)), acc1);
                        x += 16;
                    }
                    s += hs(_mm256_add_ps(acc0, acc1));
                }
            }
            (t0.elapsed().as_secs_f64(), s)
        };
        let _ = va(1);
        let _ = vb(1);
        let mut best = [f64::INFINITY; 2];
        let mut sums = [0.0f32; 2];
        for _ in 0..4 {
            let x = va(rep); if x.0 < best[0] { best[0] = x.0; sums[0] = x.1; }
            let y = vb(rep); if y.0 < best[1] { best[1] = y.0; sums[1] = y.1; }
        }
        println!("addr A/B  per_token={} t={}", per, t);
        println!("  (a) 指针步进       {:8.3} ms/round {:7.4} ns/conn sum={:.6e}", best[0] * 1e3, best[0] * 1e9 / conns, sums[0]);
        println!("  (b) 下标重算 row*per {:8.3} ms/round {:7.4} ns/conn sum={:.6e}", best[1] * 1e3, best[1] * 1e9 / conns, sums[1]);
        println!("  (b)/(a) = {:.3}x", best[1] / best[0]);
    }
}
unsafe fn enable_ftz_daz() {
    use std::arch::x86_64::{_mm_getcsr, _mm_setcsr};
    let mut csr = _mm_getcsr();
    csr |= 0x8040;
    _mm_setcsr(csr);
}

fn main() {
    let dim = arg_u32("--dim", 2048);
    let k0 = arg_u32("--k0", 8);
    let t = arg_u32("--t", 128);
    let rep = arg_u32("--rep", 40);
    let mode = arg_str("--mode", "all");
    let lid = arg_u32("--lid", 7919) as u32;

    // ⚠️ 必须在任何内核实测**之前**初始化 CPU 分发与 FTZ/DAZ：
    //    第一版把 mode 分发放在 init_cpu() 之前 ⇒ head/t8 两个模式全程走
    //    **标量回退路径**（HAS_AVX2=false），把「gather 前向」测成 1.31 ns/conn
    //    （真值是 0.19），于是得出「t8 只快 1.6x」这种被污染 7 倍的结论。
    unsafe {
        vmlm_core::interp::init_cpu();
        enable_ftz_daz();
    }
    if mode.starts_with("head") {
        head_bench(dim, t, rep, &mode);
        return;
    }
    if mode == "bwdvar" {
        bwd_variants(dim, k0, t, rep);
        return;
    }
    if mode == "t8" {
        t8_bench(dim, k0, t, rep);
        return;
    }
    if mode == "addr" {
        addr_bench(dim, k0, t, rep);
        return;
    }
    if mode == "bf16" {
        bf16_bench(dim, k0, t, rep);
        return;
    }
    if mode == "t8stack" {
        t8stack_bench(dim, k0, t, 12, rep);
        return;
    }
    if mode == "hbatch" {
        head_batch_bench(128, 20);
        return;
    }
    if mode == "t8check" {
        t8_check(dim, k0, t);
        return;
    }
    if mode == "t8bwd" {
        t8bwd_bench(dim, k0, t, rep);
        return;
    }
    if mode == "smallk" {
        smallk_bench(dim, k0, t, rep);
        return;
    }

    unsafe {
        let base = malloc(SparseConnW::bytes(dim as u32, k0 as u32));
        let sc = SparseConnW::from_base(base, dim as u32, k0 as u32, lid);
        init_sparse_conn_scaled(&sc, 1.0);
        sc.assert_geometry();
        sc.assert_src_u32_consistent();

        let x = alloc(t * dim);
        let y = alloc(t * dim);
        let dy = alloc(t * dim);
        let dx = alloc(t * dim);
        let dw = alloc(dim * k0);
        let mut r = Rng(0x1234_5678);
        for i in 0..t * dim {
            *x.add(i) = r.uni();
            *dy.add(i) = r.uni();
        }
        zero(dx, t * dim);
        zero(dw, dim * k0);

        let conns = (t * dim * k0) as f64;
        let fwd = |reps: usize| -> f64 {
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                for i in 0..t {
                    b1_forward(&sc, x.add(i * dim), y.add(i * dim));
                }
            }
            t0.elapsed().as_secs_f64()
        };
        let part = arg_u32("--part", 3) as u32;
        let bwd = |reps: usize| -> f64 {
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                for i in 0..t {
                    b1_backward_fused(&sc, dy.add(i * dim), x.add(i * dim), dw, dx.add(i * dim));
                }
            }
            t0.elapsed().as_secs_f64()
        };

        // 预热（页错误 / 分支预测 / 缓存）
        let _ = fwd(3);
        let _ = bwd(3);

        println!("kbench dim={} k0={} t={} | conns/round={}", dim, k0, t, (t * dim * k0) as f64);

        if mode == "fwd" || mode == "all" {
            let mut best = f64::INFINITY;
            for _ in 0..5 {
                let s = fwd(rep) / rep as f64;
                if s < best {
                    best = s;
                }
            }
            println!(
                "  b1_forward        {:9.3} ms/round  {:7.4} ns/conn  {:7.4} Mconn/s",
                best * 1e3,
                best * 1e9 / conns,
                conns / best / 1e6
            );
        }
        if mode == "bwd" || mode == "all" {
            let mut best = f64::INFINITY;
            for _ in 0..5 {
                let s = bwd(rep) / rep as f64;
                if s < best {
                    best = s;
                }
            }
            println!(
                "  b1_backward_fused {:9.3} ms/round  {:7.4} ns/conn  {:7.4} Mconn/s",
                best * 1e3,
                best * 1e9 / conns,
                conns / best / 1e6
            );
        }
        // 防止被优化掉的假零守卫
        let mut guard = 0.0f32;
        for i in 0..t {
            guard += *y.add(i * dim) + *dx.add(i * dim);
        }
        if !guard.is_finite() {
            println!("  WARN: guard not finite: {}", guard);
        }
    }
}
