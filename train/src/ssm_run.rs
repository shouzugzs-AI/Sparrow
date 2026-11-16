//! P62：状态向量（选择性 SSM）模型的引擎侧推理。
//!
//! 这是 `tools/p36_attnonce.py --mix state --state-kind ssm` 训练出来的模型的**生产推理路径**：
//!   * 主干：每 token 的 RMSNorm + gated MLP + 残差（没有 qkv / 位置编码 / KV cache）
//!   * 混合器：每 `attn_every` 层注入一次，注入量由**常数大小的状态向量** h[d] 递推得到
//!         dt = softplus(W_d x + b_d)     A = softplus(logA)+1e-4（导出时已算好）
//!         a  = exp(-dt ⊙ A)              h ← a⊙h + (1-a)⊙(W x)
//!         out = W_o h
//!     ⇒ decode 的内存与计算都与 ctx 无关（实测 1 KB / 14 µs，见 docs/P61 §6）
//!
//! 用法：
//!   vmlm-ssm --model artifacts/ssm/<tag>.s5q --check       # 与 PyTorch 参考 logits 对拍
//!   vmlm-ssm --model ... --speed                            # prefill / decode 吞吐
//!   vmlm-ssm --model ... --gen --prompt "…" --len 80        # 生成
//!
//! 数值口径：权重全部 f32（与训练一致）；prefill 用 `ops::gemm_nt`（NT 布局 = nn.Linear），
//! decode 用本文件的 AVX2 GEMV（`gemm_nt` 在 M=1 会落到标量兜底，实测慢 2.4×）。
use std::time::Instant;

use vmlm_ops::kernels::{gemm_nt, rmsnorm};

const EPS: f32 = 1e-6;

// ---------------- 模型文件解析 ----------------

struct Tensor {
    name: String,
    off: usize,
    numel: usize,
}

struct Model {
    d: usize,
    layers: usize,
    vocab: usize,
    hidden: usize,
    attn_every: usize,
    data: Vec<f32>,
    tensors: Vec<Tensor>,
    /// bf16 权重副本（只存便于压缩流量的矩阵；norm/inj/偏置保持 f32）
    bf16: bool,
    w16: Vec<(String, Vec<u16>)>,
    /// int8 权重副本（每行一个 scale）
    int8: bool,
    q8: Vec<(String, QuantW)>,
}

/// f32 → bf16（就近舍入到偶数）
#[inline]
fn to_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    let r = b.wrapping_add(0x7fff + ((b >> 16) & 1));
    (r >> 16) as u16
}

impl Model {
    /// 把 decode 需要的大矩阵转成 bf16 副本：流量减半，decode 是纯带宽瓶颈
    fn enable_bf16(&mut self) {
        let mut names: Vec<String> = vec!["emb".into(), "state_w".into(), "state_wd".into(), "state_o".into()];
        for i in 0..self.layers {
            names.push(format!("l{}_gate", i));
            names.push(format!("l{}_up", i));
            names.push(format!("l{}_down", i));
        }
        for n in names {
            let (p, numel) = {
                let t = self.tensors.iter().find(|t| t.name == n).unwrap();
                (unsafe { self.data.as_ptr().add(t.off) }, t.numel)
            };
            let src = unsafe { std::slice::from_raw_parts(p, numel) };
            let v: Vec<u16> = src.iter().map(|&x| to_bf16(x)).collect();
            self.w16.push((n, v));
        }
        self.bf16 = true;
    }
    fn p16(&self, name: &str) -> *const u16 {
        let e = self.w16.iter().find(|(n, _)| n == name)
            .unwrap_or_else(|| panic!("bf16 副本缺少 {}", name));
        e.1.as_ptr()
    }

    fn load(path: &str) -> Model {
        let buf = std::fs::read(path).unwrap_or_else(|e| panic!("读不到 {}: {}", path, e));
        assert_eq!(&buf[0..4], b"S5Q1", "不是 .s5q 文件");
        let mut p = 4usize;
        let mut u32_at = |p: &mut usize| -> usize {
            let v = u32::from_le_bytes(buf[*p..*p + 4].try_into().unwrap()) as usize;
            *p += 4;
            v
        };
        let (d, layers, vocab, hidden, attn_every) =
            (u32_at(&mut p), u32_at(&mut p), u32_at(&mut p), u32_at(&mut p), u32_at(&mut p));
        let n = u32_at(&mut p);
        let mut tensors = Vec::with_capacity(n);
        for _ in 0..n {
            let nl = u32_at(&mut p);
            let name = String::from_utf8(buf[p..p + nl].to_vec()).unwrap();
            p += nl;
            let off = u32_at(&mut p);
            let numel = u32_at(&mut p);
            tensors.push(Tensor { name, off, numel });
        }
        let rest = &buf[p..];
        assert_eq!(rest.len() % 4, 0);
        let mut data = vec![0f32; rest.len() / 4];
        for (i, x) in data.iter_mut().enumerate() {
            *x = f32::from_le_bytes(rest[i * 4..i * 4 + 4].try_into().unwrap());
        }
        Model { d, layers, vocab, hidden, attn_every, data, tensors,
                bf16: false, w16: Vec::new(), int8: false, q8: Vec::new() }
    }

    fn ptr(&self, name: &str) -> *const f32 {
        let t = self.tensors.iter().find(|t| t.name == name)
            .unwrap_or_else(|| panic!("缺少张量 {}", name));
        unsafe { self.data.as_ptr().add(t.off) }
    }
}

// ---------------- 运行时状态 ----------------

struct Scratch {
    x: Vec<f32>,      // [T, d] 残差流
    xn: Vec<f32>,     // [T, d] norm 输出
    a_out: Vec<f32>,  // [T, d] 混合器输出
    act: Vec<f32>,    // [T, I] silu(gate)*up
    up: Vec<f32>,     // [T, I] up 投影（与 act 同尺寸，别复用 d 维缓冲）
    out: Vec<f32>,    // [T, d] 临时
    logits: Vec<f32>, // [T, V]
    h: Vec<f32>,      // [d] 状态（跨 token 常驻）
    t1: Vec<f32>,     // [d]
    t2: Vec<f32>,     // [d]
}

impl Scratch {
    fn new(m: &Model) -> Scratch {
        let (d, h, v) = (m.d, m.hidden, m.vocab);
        Scratch {
            x: vec![0f32; d], xn: vec![0f32; d], a_out: vec![0f32; d],
            act: vec![0f32; h], up: vec![0f32; h], out: vec![0f32; d], logits: vec![0f32; v],
            h: vec![0f32; d], t1: vec![0f32; d], t2: vec![0f32; d],
        }
    }
    fn reset(&mut self) {
        self.h.iter_mut().for_each(|v| *v = 0.0);
    }
    /// prefill 需要 T 行的缓冲
    fn ensure(&mut self, m: &Model, t: usize) {
        let (d, h, v) = (m.d, m.hidden, m.vocab);
        if self.x.len() < d * t {
            self.x = vec![0f32; d * t];
            self.xn = vec![0f32; d * t];
            self.a_out = vec![0f32; d * t];
            self.out = vec![0f32; d * t];
        }
        if self.act.len() < h * t {
            self.act = vec![0f32; h * t];
            self.up = vec![0f32; h * t];
        }
        if self.logits.len() < v * t { self.logits = vec![0f32; v * t]; }
    }
}

// ---------------- 向量数学 ----------------

#[inline]
fn softplus(x: f32) -> f32 {
    let ax = x.abs();
    if ax > 30.0 { x.max(0.0) } else { x.max(0.0) + (-ax).exp().ln_1p() }
}

/// y[o] = Σ_p x[p] · W[o,p]（W 为 nn.Linear 布局 [out, in]，每行连续 → 点积形式）
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemv(x: *const f32, w: *const f32, y: *mut f32, out_n: usize, in_n: usize) {
    use core::arch::x86_64::*;
    let mut o = 0usize;
    while o < out_n {
        let wp = w.add(o * in_n);
        let mut acc = _mm256_setzero_ps();
        let mut p = 0usize;
        while p + 8 <= in_n {
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(x.add(p)), _mm256_loadu_ps(wp.add(p)), acc);
            p += 8;
        }
        let lo = _mm256_castps256_ps128(acc);
        let hi = _mm256_extractf128_ps(acc, 1);
        let s4 = _mm_add_ps(lo, hi);
        let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
        let s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
        let mut sum = _mm_cvtss_f32(s1);
        while p < in_n { sum += *x.add(p) * *wp.add(p); p += 1; }
        *y.add(o) = sum;
        o += 1;
    }
}

/// int8 权重量化副本：每输出行一个 scale（decode 流量再减半；CE 代价见 docs/P62）
struct QuantW {
    q: Vec<i8>,
    scale: Vec<f32>,
}

impl Model {
    fn enable_int8(&mut self) {
        let mut names: Vec<String> = vec!["emb".into(), "state_w".into(), "state_wd".into(), "state_o".into()];
        for i in 0..self.layers {
            names.push(format!("l{}_gate", i));
            names.push(format!("l{}_up", i));
            names.push(format!("l{}_down", i));
        }
        for n in names {
            let (p, numel) = {
                let t = self.tensors.iter().find(|t| t.name == n).unwrap();
                (unsafe { self.data.as_ptr().add(t.off) }, t.numel)
            };
            // 形状按名字判定（.s5q 头里只有总量，没有逐张量形状）
            let in_n = if n.starts_with("l") && n.ends_with("_down") { self.hidden } else { self.d };
            let rows = numel / in_n;
            let src = unsafe { std::slice::from_raw_parts(p, numel) };
            let mut q = vec![0i8; numel];
            let mut scale = vec![0f32; rows];
            for r in 0..rows {
                let row = &src[r * in_n..(r + 1) * in_n];
                let amax = row.iter().fold(0f32, |a, b| a.max(b.abs())).max(1e-8);
                let s = amax / 127.0;
                scale[r] = s;
                for (j, &v) in row.iter().enumerate() {
                    q[r * in_n + j] = (v / s).round().clamp(-127.0, 127.0) as i8;
                }
            }
            self.q8.push((n, QuantW { q, scale }));
        }
        self.int8 = true;
    }
    fn p8(&self, name: &str) -> &QuantW {
        &self.q8.iter().find(|(n, _)| n == name)
            .unwrap_or_else(|| panic!("int8 副本缺少 {}", name)).1
    }
}

/// int8 权重 GEMV：一次 8 个 int8 → i32 → f32，再乘每行 scale
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemv_i8(x: *const f32, w: *const i8, scale: *const f32, y: *mut f32,
                 out_n: usize, in_n: usize) {
    use core::arch::x86_64::*;
    let mut o = 0usize;
    while o < out_n {
        let wp = w.add(o * in_n);
        let mut acc = _mm256_setzero_ps();
        let mut p = 0usize;
        while p + 8 <= in_n {
            let b8 = _mm_loadl_epi64(wp.add(p) as *const __m128i);
            let f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b8));
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(x.add(p)), f, acc);
            p += 8;
        }
        let lo = _mm256_castps256_ps128(acc);
        let hi = _mm256_extractf128_ps(acc, 1);
        let s4 = _mm_add_ps(lo, hi);
        let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
        let s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
        let mut sum = _mm_cvtss_f32(s1);
        while p < in_n {
            sum += *x.add(p) * (*wp.add(p) as f32);
            p += 1;
        }
        *y.add(o) = sum * *scale.add(o);
        o += 1;
    }
}

/// bf16 权重 GEMV：一次取 8 个 bf16（16 字节）→ 左移 16 位还原成 f32 的位型#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemv_bf16(x: *const f32, w: *const u16, y: *mut f32, out_n: usize, in_n: usize) {
    use core::arch::x86_64::*;
    let mut o = 0usize;
    while o < out_n {
        let wp = w.add(o * in_n);
        let mut acc = _mm256_setzero_ps();
        let mut p = 0usize;
        while p + 8 <= in_n {
            let h = _mm_loadu_si128(wp.add(p) as *const __m128i);
            let f = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16));
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(x.add(p)), f, acc);
            p += 8;
        }
        let lo = _mm256_castps256_ps128(acc);
        let hi = _mm256_extractf128_ps(acc, 1);
        let s4 = _mm_add_ps(lo, hi);
        let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
        let s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
        let mut sum = _mm_cvtss_f32(s1);
        while p < in_n {
            sum += *x.add(p) * f32::from_bits((*wp.add(p) as u32) << 16);
            p += 1;
        }
        *y.add(o) = sum;
        o += 1;
    }
}

/// 统一矩阵乘入口：T==1 走 GEMV，T>1 走 GEMM；bf16 只用于 decode（prefill 的 bf16 GEMM
/// 实测更慢，因为 ops 的 gemm_nt_bf16 没有打包复用）
unsafe fn mm(m: &Model, name: &str, x: *const f32, y: *mut f32, t: usize, n: usize, k: usize) {
    if t == 1 {
        if m.int8 {
            let w = m.p8(name);
            gemv_i8(x, w.q.as_ptr(), w.scale.as_ptr(), y, n, k)
        } else if m.bf16 {
            gemv_bf16(x, m.p16(name), y, n, k)
        } else {
            gemv(x, m.ptr(name), y, n, k)
        }
    } else {
        gemm_nt(x, m.ptr(name), y, t, n, k, false);
    }
}

/// 状态混合器里固定走 GEMV（每 token 一次，T 不参与）
unsafe fn mm1(m: &Model, name: &str, x: *const f32, y: *mut f32, n: usize, k: usize) {
    if m.int8 {
        let w = m.p8(name);
        gemv_i8(x, w.q.as_ptr(), w.scale.as_ptr(), y, n, k)
    } else if m.bf16 {
        gemv_bf16(x, m.p16(name), y, n, k)
    } else {
        gemv(x, m.ptr(name), y, n, k)
    }
}

/// AVX2 exp（2^f 多项式；与 bench_decode_opt.rs 同一实现）
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vexp(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use core::arch::x86_64::*;
    let y = _mm256_mul_ps(x, _mm256_set1_ps(1.4426950408889634));
    let magic = _mm256_set1_ps(12582912.0);
    let t = _mm256_add_ps(y, magic);
    let n = _mm256_sub_ps(t, magic);
    let f = _mm256_sub_ps(y, n);
    let mut p = _mm256_fmadd_ps(_mm256_set1_ps(0.009618129107628477), f, _mm256_set1_ps(0.05550410866482158));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(0.2402265069591007));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(0.6931471805599453));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.0));
    let ni = _mm256_cvtps_epi32(n);
    let bits = _mm256_slli_epi32(ni, 23);
    let scale = _mm256_castsi256_ps(_mm256_add_epi32(bits, _mm256_set1_epi32(0x3f800000)));
    _mm256_mul_ps(p, scale)
}

/// act = silu(gate) * up（I 维，原地覆盖 gate）
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn silu_mul(gate: *mut f32, up: *const f32, n: usize) {
    use core::arch::x86_64::*;
    let one = _mm256_set1_ps(1.0);
    let mut j = 0usize;
    while j + 8 <= n {
        let g = _mm256_loadu_ps(gate.add(j));
        let e = vexp(_mm256_sub_ps(_mm256_setzero_ps(), g));
        let sig = _mm256_div_ps(one, _mm256_add_ps(one, e));
        let u = _mm256_loadu_ps(up.add(j));
        _mm256_storeu_ps(gate.add(j), _mm256_mul_ps(_mm256_mul_ps(g, sig), u));
        j += 8;
    }
    while j < n {
        let g = *gate.add(j);
        *gate.add(j) = g / (1.0 + (-g).exp()) * *up.add(j);
        j += 1;
    }
}

// ---------------- 前向 ----------------

/// 状态混合器：对第 t 个 token 走一步（更新 h，写 a_out[t]）
unsafe fn ssm_step(m: &Model, s: &mut Scratch, x0: *const f32, row: usize) {
    let d = m.d;
    let (w, wd, wdb, o, arr) = (
        "state_w", "state_wd", m.ptr("state_wdb"), "state_o", m.ptr("state_A"),
    );
    mm1(m, wd, x0, s.t1.as_mut_ptr(), d, d);
    for j in 0..d { s.t1[j] += *wdb.add(j); }
    mm1(m, w, x0, s.t2.as_mut_ptr(), d, d);
    for j in 0..d {
        let a = (-softplus(s.t1[j]) * *arr.add(j)).exp();
        s.h[j] = a * s.h[j] + (1.0 - a) * s.t2[j];
    }
    mm1(m, o, s.h.as_ptr(), s.a_out.as_mut_ptr().add(row * d), d, d);
}

unsafe fn rmsnorm_row(dst: *mut f32, src: *const f32, g: *const f32, d: usize) {
    rmsnorm(dst, src, g, 1, d, EPS);
}

/// 跑 tokens（prefill T>1 / decode T=1 同一路径），返回最后一行的 logits 指针
unsafe fn forward(m: &Model, s: &mut Scratch, tokens: &[u32]) -> *const f32 {
    let (d, t) = (m.d, tokens.len());
    s.ensure(m, t);
    let emb = m.ptr("emb");
    // 1) embedding
    for (i, &tok) in tokens.iter().enumerate() {
        std::ptr::copy_nonoverlapping(emb.add(tok as usize * d), s.x.as_mut_ptr().add(i * d), d);
    }
    // 2) 混合器：状态递推（一次性，每 attn_every 层复用同一份输出）
    let ln0 = m.ptr("ln0");
    for i in 0..t {
        rmsnorm_row(s.xn.as_mut_ptr().add(i * d), s.x.as_ptr().add(i * d), ln0, d);
        ssm_step(m, s, s.xn.as_ptr().add(i * d), i);
    }
    // 3) 主干层
    for li in 0..m.layers {
        // 注意：p36 的 Block 对**每一层**都做 x += inj*A（A 只在刷新层重算），
        // 不是只在刷新层注入。这里必须照抄，否则从第 1 层开始就分叉。
        let inj = m.ptr(&format!("l{}_inj", li));
        for i in 0..t {
            let xp = s.x.as_mut_ptr().add(i * d);
            let ap = s.a_out.as_ptr().add(i * d);
            for j in 0..d { *xp.add(j) += *inj.add(j) * *ap.add(j); }
        }
        let ln1 = m.ptr(&format!("l{}_ln1", li));
        for i in 0..t {
            rmsnorm_row(s.xn.as_mut_ptr().add(i * d), s.x.as_ptr().add(i * d), ln1, d);
        }
        let i_n = m.hidden;
        let gate = format!("l{}_gate", li);
        let up = format!("l{}_up", li);
        let down = format!("l{}_down", li);
        mm(m, &gate, s.xn.as_ptr(), s.act.as_mut_ptr(), t, i_n, d);
        mm(m, &up, s.xn.as_ptr(), s.up.as_mut_ptr(), t, i_n, d);
        for i in 0..t {
            silu_mul(s.act.as_mut_ptr().add(i * i_n), s.up.as_ptr().add(i * i_n), i_n);
        }
        mm(m, &down, s.act.as_ptr(), s.out.as_mut_ptr(), t, d, i_n);
        for j in 0..t * d { s.x[j] += s.out[j]; }
    }
    // 4) 收尾 norm + logits（logits = xn @ embᵀ，与 embedding 共享）
    let norm = m.ptr("norm");
    for i in 0..t {
        rmsnorm_row(s.xn.as_mut_ptr().add(i * d), s.x.as_ptr().add(i * d), norm, d);
    }
    mm(m, "emb", s.xn.as_ptr(), s.logits.as_mut_ptr(), t, m.vocab, d);
    s.logits.as_ptr().add((t - 1) * m.vocab)
}

/// 逐段 dump 版本（仅供 --dbg 对拍；生产走 forward）
unsafe fn forward_dbg(m: &Model, s: &mut Scratch, tokens: &[u32]) -> Vec<(String, Vec<f32>)> {
    let (d, t) = (m.d, tokens.len());
    s.ensure(m, t);
    let emb = m.ptr("emb");
    for (i, &tok) in tokens.iter().enumerate() {
        std::ptr::copy_nonoverlapping(emb.add(tok as usize * d), s.x.as_mut_ptr().add(i * d), d);
    }
    let ln0 = m.ptr("ln0");
    for i in 0..t {
        rmsnorm_row(s.xn.as_mut_ptr().add(i * d), s.x.as_ptr().add(i * d), ln0, d);
        ssm_step(m, s, s.xn.as_ptr().add(i * d), i);
    }
    let mut out: Vec<(String, Vec<f32>)> = Vec::new();
    out.push(("x0".into(), s.x[..t * d].to_vec()));
    out.push(("A".into(), s.a_out[..t * d].to_vec()));
    for li in 0..m.layers {
        // 每层都注入（见 forward 里的同一处注释）
        let inj = m.ptr(&format!("l{}_inj", li));
        for i in 0..t {
            let xp = s.x.as_mut_ptr().add(i * d);
            let ap = s.a_out.as_ptr().add(i * d);
            for j in 0..d { *xp.add(j) += *inj.add(j) * *ap.add(j); }
        }
        let ln1 = m.ptr(&format!("l{}_ln1", li));
        for i in 0..t {
            rmsnorm_row(s.xn.as_mut_ptr().add(i * d), s.x.as_ptr().add(i * d), ln1, d);
        }
        let i_n = m.hidden;
        let gate = format!("l{}_gate", li);
        let up = format!("l{}_up", li);
        let down = format!("l{}_down", li);
        mm(m, &gate, s.xn.as_ptr(), s.act.as_mut_ptr(), t, i_n, d);
        mm(m, &up, s.xn.as_ptr(), s.up.as_mut_ptr(), t, i_n, d);
        if li == 0 { out.push(("h0".into(), s.xn[..t * d].to_vec())); }
        for i in 0..t {
            silu_mul(s.act.as_mut_ptr().add(i * i_n), s.up.as_ptr().add(i * i_n), i_n);
        }
        if li == 0 { out.push(("act0".into(), s.act[..t * i_n].to_vec())); }
        mm(m, &down, s.act.as_ptr(), s.out.as_mut_ptr(), t, d, i_n);
        if li == 0 { out.push(("d0".into(), s.out[..t * d].to_vec())); }
        for j in 0..t * d { s.x[j] += s.out[j]; }
        out.push((format!("blk{}", li), s.x[..t * d].to_vec()));
    }
    let norm = m.ptr("norm");
    for i in 0..t {
        rmsnorm_row(s.xn.as_mut_ptr().add(i * d), s.x.as_ptr().add(i * d), norm, d);
    }
    out.push(("xn".into(), s.xn[..t * d].to_vec()));
    mm(m, "emb", s.xn.as_ptr(), s.logits.as_mut_ptr(), t, m.vocab, d);
    out.push(("logits".into(), s.logits[..t * m.vocab].to_vec()));
    out
}

fn load_bin(path: &std::path::Path) -> Vec<f32> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("读不到 {:?}: {}", path, e));
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

// ---------------- 采样 ----------------

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32) / (1u32 << 24) as f32
    }
}

fn sample(logits: &[f32], temp: f32, top_k: usize, rng: &mut Rng) -> u32 {
    if temp <= 0.0 {
        let mut bi = 0usize;
        for (i, &v) in logits.iter().enumerate() { if v > logits[bi] { bi = i; } }
        return bi as u32;
    }
    let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<(f32, u32)> = logits.iter().enumerate()
        .map(|(i, &v)| (((v - mx) / temp).exp(), i as u32)).collect();
    if top_k > 0 && top_k < probs.len() {
        probs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        probs.truncate(top_k);
    }
    let sum: f32 = probs.iter().map(|p| p.0).sum();
    let mut r = rng.next() * sum;
    for (p, i) in &probs {
        r -= p;
        if r <= 0.0 { return *i; }
    }
    probs[0].1
}

// ---------------- 词表 ----------------

struct Vocab {
    id_to_char: Vec<Option<char>>,
}

impl Vocab {
    fn load(path: &str) -> Vocab {
        let txt = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("读不到词表 {}: {}", path, e));
        let mut id_to_char: Vec<Option<char>> = Vec::new();
        for line in txt.lines() {
            let mut it = line.split('\t');
            let id: usize = it.next().unwrap().parse().unwrap();
            let kind = it.next().unwrap_or("");
            let hex = it.next().unwrap_or("");
            if id_to_char.len() <= id { id_to_char.resize(id + 1, None); }
            if kind == "char" && !hex.is_empty() {
                if let Ok(cp) = u32::from_str_radix(hex, 16) {
                    id_to_char[id] = char::from_u32(cp);
                }
            }
        }
        Vocab { id_to_char }
    }
    fn decode(&self, ids: &[u32]) -> String {
        ids.iter().filter_map(|&i| self.id_to_char.get(i as usize).copied().flatten())
            .collect()
    }
}

// ---------------- main ----------------

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |name: &str| -> Option<String> {
        args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
    };
    let flag = |name: &str| args.iter().any(|a| a == name);
    let model_path = get("--model").unwrap_or_else(|| "artifacts/ssm/st_ssm_lr4e3_4000.s5q".into());
    let dir = std::path::Path::new(&model_path).parent().map(|p| p.to_path_buf()).unwrap_or_default();
    // 参考文件按 tag 命名（多个模型导出到同一目录，不能共用一个 ref 文件名）
    let stem = std::path::Path::new(&model_path).file_stem()
        .map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    let vocab_path = get("--vocab")
        .unwrap_or_else(|| "artifacts/zh_wiki/char_vocab_hex.txt".into());

    let t_load = Instant::now();
    // 必须显式探测 CPU：ops::gemm_nt 只有在 HAS_AVX2 为真时才走 8x8 向量内核，
    // 否则静默退回标量（实测慢 ~50x，而且不会有任何报错）
    unsafe { vmlm_core::interp::init_cpu() };
    let mut m = Model::load(&model_path);
    if flag("--bf16") {
        let t = Instant::now();
        m.enable_bf16();
        println!("# bf16 权重副本就绪（{:.1} ms，只用于 decode；prefill 仍 f32）",
            t.elapsed().as_secs_f64() * 1e3);
    }
    if flag("--int8") {
        let t = Instant::now();
        m.enable_int8();
        println!("# int8 权重副本就绪（{:.1} ms，只用于 decode；prefill 仍 f32）",
            t.elapsed().as_secs_f64() * 1e3);
    }
    let mut s = Scratch::new(&m);
    println!("# 模型 {}  d={} L={} I={} V={} attn_every={}  参数 {:.3}M  载入 {:.1} ms",
        model_path, m.d, m.layers, m.hidden, m.vocab, m.attn_every,
        m.data.len() as f64 / 1e6, t_load.elapsed().as_secs_f64() * 1e3);

    if flag("--dbg") {
        let toks: Vec<u32> = {
            let b = std::fs::read(dir.join(format!("{}.tokens.bin", stem))).expect("缺少 tokens 参考");
            b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
        };
        unsafe {
            let stages = forward_dbg(&m, &mut s, &toks);
            let dbg_dir = dir.join(format!("dbg_{}", stem));
            for (name, got) in &stages {
                let path = dbg_dir.join(format!("dbg_{}.bin", name));
                if !path.exists() { println!("  {:8} 参考缺失", name); continue; }
                let r = load_bin(&path);
                if r.len() != got.len() { println!("  {:8} 尺寸不符 {} vs {}", name, r.len(), got.len()); continue; }
                let (mut mx, mut k) = (0f32, 0usize);
                let mut scale = 0f32;
                for i in 0..r.len() {
                    let d = (got[i] - r[i]).abs();
                    if d > mx { mx = d; k = i; }
                    if r[i].abs() > scale { scale = r[i].abs(); }
                }
                println!("  {:8} 最大差 {:10.3e}（相对 {:8.2e}，@{}）", name, mx,
                    mx / (scale + 1e-12), k);
            }
        }
    }

    if flag("--check-decode") {
        // prefill 前半段，然后逐 token decode 后半段：这条路径用的是 gemv + 常驻状态，
        // 与 prefill 的 gemm_nt 不同，必须单独对拍（否则"引擎能跑"只是 prefill 能跑）
        let toks: Vec<u32> = {
            let b = std::fs::read(dir.join(format!("{}.tokens.bin", stem))).expect("缺少 tokens 参考");
            b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
        };
        let refb = std::fs::read(dir.join(format!("{}.logits.bin", stem))).expect("缺少 logits 参考");
        let refl: Vec<f32> = refb.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        let t = toks.len();
        let half = t / 2;
        unsafe {
            s.reset();
            let _ = forward(&m, &mut s, &toks[..half]);
            let mut mx = 0f32; let mut bad = 0;
            for i in half..t {
                let lg = forward(&m, &mut s, &toks[i..i + 1]);
                let row = std::slice::from_raw_parts(lg, m.vocab);
                let (mut bi, mut ri) = (0usize, 0usize);
                for j in 0..m.vocab {
                    let d = (row[j] - refl[i * m.vocab + j]).abs();
                    if d > mx { mx = d; }
                    if row[j] > row[bi] { bi = j; }
                    if refl[i * m.vocab + j] > refl[i * m.vocab + ri] { ri = j; }
                }
                if bi != ri { bad += 1; }
            }
            println!("# decode 对拍：prefill {} + 逐 token {}，logits 最大绝对差 {:.3e}，argmax 不一致 {}/{}",
                half, t - half, mx, bad, t - half);
            println!("# 判定：{}", if mx < 2e-3 && bad == 0 { "PASS" } else { "FAIL" });
        }
    }

    if flag("--check") {
        let toks: Vec<u32> = {
            let b = std::fs::read(dir.join(format!("{}.tokens.bin", stem))).expect("缺少 tokens 参考");
            b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
        };
        let refb = std::fs::read(dir.join(format!("{}.logits.bin", stem))).expect("缺少 logits 参考");
        let refl: Vec<f32> = refb.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        let t = toks.len();
        assert_eq!(refl.len(), t * m.vocab, "参考 logits 尺寸不符");
        unsafe {
            let _ = forward(&m, &mut s, &toks);
            let got = std::slice::from_raw_parts(s.logits.as_ptr(), t * m.vocab);
            let mut mx = 0f32; let mut mx_at = (0, 0); let mut top_mismatch = 0;
            for i in 0..t {
                let (mut bi, mut ri) = (0usize, 0usize);
                for j in 0..m.vocab {
                    let d = (got[i * m.vocab + j] - refl[i * m.vocab + j]).abs();
                    if d > mx { mx = d; mx_at = (i, j); }
                    if got[i * m.vocab + j] > got[i * m.vocab + bi] { bi = j; }
                    if refl[i * m.vocab + j] > refl[i * m.vocab + ri] { ri = j; }
                }
                if bi != ri { top_mismatch += 1; }
            }
            println!("# 对拍：{} token × {} 词表，logits 最大绝对差 {:.3e}（位置 {:?}），argmax 不一致 {}/{}",
                t, m.vocab, mx, mx_at, top_mismatch, t);
            println!("# 判定：{}", if mx < 2e-3 && top_mismatch == 0 { "PASS" } else { "FAIL" });
        }
    }

    if flag("--speed") {
        let prefill_ctx: usize = get("--ctx").and_then(|v| v.parse().ok()).unwrap_or(64);
        let toks: Vec<u32> = (0..prefill_ctx).map(|i| 100u32 + (i as u32 % 7000)).collect();
        unsafe {
            let _ = forward(&m, &mut s, &toks);      // warmup
            let t0 = Instant::now();
            let n = 5;
            for _ in 0..n { s.reset(); let _ = forward(&m, &mut s, &toks); }
            let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
            println!("# prefill ctx={:4}: {:8.2} ms  {:8.1} tok/s  ({:.2} ms/token)",
                prefill_ctx, ms, prefill_ctx as f64 / (ms / 1000.0), ms / prefill_ctx as f64);
        }
        // decode：一次 prefill 之后逐 token 走
        let mut state = Scratch::new(&m);
        unsafe {
            let pre: Vec<u32> = (0..32).collect();
            let _ = forward(&m, &mut state, &pre);
            let nd = 256usize;
            let t0 = Instant::now();
            let mut tok = 200u32;
            for _ in 0..nd {
                let lg = forward(&m, &mut state, &[tok]);
                let mut bi = 0usize;
                for j in 1..m.vocab { if *lg.add(j) > *lg.add(bi) { bi = j; } }
                tok = bi as u32;
            }
            let ms = t0.elapsed().as_secs_f64() * 1000.0 / nd as f64;
            println!("# decode（状态常数，无 KV）: {:8.3} ms/token  {:8.1} tok/s", ms, 1000.0 / ms);
        }
    }

    if flag("--gen") {
        let vocab = Vocab::load(&vocab_path);
        let prompt = get("--prompt").unwrap_or_default();
        let len: usize = get("--len").and_then(|v| v.parse().ok()).unwrap_or(80);
        let temp: f32 = get("--temp").and_then(|v| v.parse().ok()).unwrap_or(0.8);
        let top_k: usize = get("--top-k").and_then(|v| v.parse().ok()).unwrap_or(50);
        let seed: u64 = get("--seed").and_then(|v| v.parse().ok()).unwrap_or(42);
        let mut rng = Rng(seed.wrapping_mul(6364136223846793005).wrapping_add(1));
        // 词表只有 id→char；生成从 BOS=1 起步（与训练数据一致），prompt 仅作显示
        let mut ids: Vec<u32> = vec![1];
        let mut out_ids: Vec<u32> = Vec::new();
        unsafe {
            let _ = forward(&m, &mut s, &[1]);
            let mut nxt = sample(std::slice::from_raw_parts(s.logits.as_ptr(), m.vocab), temp, top_k, &mut rng);
            for _ in 0..len {
                out_ids.push(nxt);
                ids.push(nxt);
                let lg = forward(&m, &mut s, &[nxt]);
                nxt = sample(std::slice::from_raw_parts(lg, m.vocab), temp, top_k, &mut rng);
            }
        }
        println!("{}", if prompt.is_empty() { String::new() } else { format!("[prompt] {}", prompt) });
        println!("[gen] {}", vocab.decode(&out_ids));
    }

    if !flag("--check") && !flag("--speed") && !flag("--gen") {
        println!("（没有选模式：--check / --speed / --gen）");
    }
}
