// P7 反向算子扩展（成员 A 负责实现）
//
// 本文件由成员 A 独占。签名已冻结，见 docs/P7_分工与接口契约.md §2.2。
// 调用入口在 interp.rs，由 Lead 接线 —— A 不要改 interp.rs。
//
// 约定（与 P3 的 train.rs 保持一致）：
//   * 输出型梯度（dx / dv）**覆盖写**：不再读写目标原有值，等价于先清零，
//     所以不需要脚本额外 ZERO。与 `gemm_bwd_x`（覆盖）/`gemm_bwd_w`（累加）一致。
//   * 参数梯度（dtable / dg / db）**累加**，配合每步开头的 ZERO 使用。
//   * 循环顺序刻意写成「归约维在内」，使归约次序与 interp.rs / kernels.rs 的
//     前向 kernel 完全一致 —— 反向数值与前向严格自洽（见各函数注释）。
//
// 数学工具来自 crate::kernels（exp_f / ln_f / sqrt_f），不引 libm。
#![allow(dead_code, unused_variables)]

use crate::kernels::sqrt_f;

// 不依赖 core::fmt 的浮点输出（引擎是 no_std 且裁掉了 fmt）。
// 只在显式调 autograd::debug_dump 时才会被引用，interp 不接线就不会进镜像。
// 返回写入字节数。
pub unsafe fn f32_to_str(v: f32, out: *mut u8, cap: usize) -> usize {
    let mut n = 0usize;
    unsafe fn push(out: *mut u8, cap: usize, n: &mut usize, b: u8) {
        if *n < cap {
            *out.add(*n) = b;
            *n += 1;
        }
    }
    macro_rules! push {
        ($b:expr) => {
            push(out, cap, &mut n, $b)
        };
    }
    let mut x = v;
    if x.is_nan() {
        for b in b"nan" {
            push!(*b);
        }
        return n;
    }
    if x < 0.0 {
        push!(b'-');
        x = -x;
    }
    if x.is_infinite() {
        for b in b"inf" {
            push!(*b);
        }
        return n;
    }
    let ip = x as u32;
    // 整数部分（不用 fmt，也不用 core::fmt）
    let mut digs = [0u8; 10];
    let mut nd = 0usize;
    let mut q = ip;
    loop {
        digs[nd] = b'0' + (q % 10) as u8;
        nd += 1;
        q /= 10;
        if q == 0 || nd == 10 {
            break;
        }
    }
    while nd > 0 {
        nd -= 1;
        push!(digs[nd]);
    }
    push!(b'.');
    let mut frac = x - ip as f32;
    for _ in 0..6 {
        frac *= 10.0;
        let d = frac as u32;
        push!(b'0' + (d % 10) as u8);
        frac -= d as f32;
    }
    n
}

// ============================================================================
// 1. softmax 反向
//    y = softmax(x)（逐行）；dx = y ⊙ (dy − Σ_j dy_j·y_j)
//    causal 时第 r 行只有前 r+1 列参与（与前向 kernels::softmax 的 lim 一致）
// ============================================================================
pub unsafe fn softmax_bwd(
    dy: *const f32,
    y: *const f32,
    dx: *mut f32,
    rows: usize,
    cols: usize,
    causal: bool,
) {
    for r in 0..rows {
        let lim = if causal { r + 1 } else { cols };
        let lim = if lim > cols { cols } else { lim };
        let dry = dy.add(r * cols);
        let yr = y.add(r * cols);
        let dxr = dx.add(r * cols);
        let mut dot = 0.0f32;
        for j in 0..lim {
            dot += *dry.add(j) * *yr.add(j);
        }
        for i in 0..lim {
            *dxr.add(i) = *yr.add(i) * (*dry.add(i) - dot);
        }
        for i in lim..cols {
            *dxr.add(i) = 0.0;
        }
    }
}

// ============================================================================
// 2. LayerNorm 反向
//    前向：mu=mean(x), var=mean((x-mu)^2), inv=1/sqrt(var+eps), xs=(x-mu)*inv
//    db += Σ_rows dy        dg += Σ_rows dy⊙xs
//    dx_r = inv · ( g⊙dy − mean_cols(g⊙dy) − xs_r · mean_cols(g⊙dy⊙xs) )
//
//    与 kernels::layernorm 的归约次序严格一致（外积 + 按列 BLAS-1 归约），
//    这样解析梯度与前向 kernel 数值自洽到 ~1e-7，而不是 ~1e-3。
// ============================================================================
pub unsafe fn layernorm_bwd(
    dy: *const f32,
    x: *const f32,
    g: *const f32,
    dx: *mut f32,
    dg: *mut f32,
    db: *mut f32,
    rows: usize,
    cols: usize,
    eps: f32,
) {
    for r in 0..rows {
        let xr = x.add(r * cols);
        let dyr = dy.add(r * cols);
        let dxr = dx.add(r * cols);

        let mut mu = 0.0f32;
        for i in 0..cols {
            mu += *xr.add(i);
        }
        mu /= cols as f32;
        let mut var = 0.0f32;
        for i in 0..cols {
            let d = *xr.add(i) - mu;
            var += d * d;
        }
        var /= cols as f32;
        let inv = 1.0 / sqrt_f(var + eps);

        // 按列归约 g⊙dy 与 (g⊙dy)⊙xs（归约次序与 kernels::layernorm 的
        // 逐列 BLAS-1 一致，使解析梯度与前向 kernel 数值自洽）。
        // 注意：这里必须带上 g —— mean(g⊙dy) 与 mean(g⊙dy⊙xs)，漏掉 g 就是错的。
        let mut sum_dyg = 0.0f32;
        let mut sum_dygxs = 0.0f32;
        for i in 0..cols {
            let dyi = *dyr.add(i);
            let xs = (*xr.add(i) - mu) * inv;
            let dyg = dyi * *g.add(i);
            sum_dyg += dyg;
            sum_dygxs += dyg * xs;
            *dxr.add(i) = dyg; // 暂存 g⊙dy，后面直接复用
        }
        let mean_dyg = sum_dyg / cols as f32;
        let mean_dygxs = sum_dygxs / cols as f32;

        for i in 0..cols {
            let dyi = *dyr.add(i);
            let xs = (*xr.add(i) - mu) * inv;
            *dxr.add(i) = inv * (*dxr.add(i) - mean_dyg - xs * mean_dygxs);
            // 参数梯度：db[N] += Σ_rows dy，dg[N] += Σ_rows dy⊙xs（累加）
            if !db.is_null() {
                *db.add(i) += dyi;
            }
            if !dg.is_null() {
                *dg.add(i) += dyi * xs;
            }
        }
    }
}

// ============================================================================
// 3. outer 反向：S[i,j] += v[i]·k[j]  (i∈[0,m), j∈[0,n))
//    dv[i] += Σ_j ds[i,j]·k[j]
//    dk[j] += Σ_i ds[i,j]·v[i]
//    与 kernels::outer 的归约次序一致（v 外层、k 内层）。
// ============================================================================
pub unsafe fn outer_bwd(
    ds: *const f32,
    v: *const f32,
    k: *const f32,
    dv: *mut f32,
    dk: *mut f32,
    m: usize,
    n: usize,
) {
    // dv[i] = Σ_j ds[i,j]·k[j]（内层按 j 升序，与 outer 的 j 升序一致）
    for i in 0..m {
        let dsi = ds.add(i * n);
        let mut acc = 0.0f32;
        for j in 0..n {
            acc += *dsi.add(j) * *k.add(j);
        }
        *dv.add(i) = acc;
    }
    // dk[j] += Σ_i ds[i,j]·v[i]（累加；归约次序 i 升序）
    for j in 0..n {
        let mut acc = 0.0f32;
        for i in 0..m {
            acc += *ds.add(i * n + j) * *v.add(i);
        }
        *dk.add(j) += acc;
    }
}

// ============================================================================
// 4. 查表反向：dtable[row, :] += dy[0, :]
// ============================================================================
pub unsafe fn emb_bwd(
    dy: *const f32,
    table_shape: (usize, usize),
    row: usize,
    dtable: *mut f32,
) {
    let (_rows, cols) = table_shape;
    let dst = dtable.add(row * cols);
    for i in 0..cols {
        *dst.add(i) += *dy.add(i);
    }
}

// ============================================================================
// 5. AdamW —— decoupled weight decay
//    先做解耦衰减 p -= lr·wd·p，再做标准 Adam 偏差校正更新。
//    等价形式：p = p·(1 − lr·wd) − lr·m̂/(sqrt(v̂)+eps)
//    pow_f(β, t) 用 exp/ln 实现（与 train::Adam 的 pow_f 同一套数学工具）。
// ============================================================================
#[inline(always)]
fn pow_f(b: f32, e: f32) -> f32 {
    crate::kernels::exp_f(e * crate::kernels::ln_f(b))
}

pub struct AdamW {
    pub m: *mut f32,
    pub v: *mut f32,
    pub lr: f32,
    pub b1: f32,
    pub b2: f32,
    pub eps: f32,
    pub wd: f32,
    pub t: u32,
}

impl AdamW {
    pub unsafe fn step(&mut self, p: *mut f32, g: *const f32, n: usize) {
        self.t += 1;
        let t = self.t as f32;
        let c1 = 1.0 - pow_f(self.b1, t);
        let c2 = 1.0 - pow_f(self.b2, t);
        for i in 0..n {
            let gi = *g.add(i);
            let mi = self.b1 * *self.m.add(i) + (1.0 - self.b1) * gi;
            let vi = self.b2 * *self.v.add(i) + (1.0 - self.b2) * gi * gi;
            *self.m.add(i) = mi;
            *self.v.add(i) = vi;
            let mhat = mi / c1;
            let vhat = vi / c2;
            // 解耦衰减：直接用原 p，而不是把 wd·p 加进 g
            let pi = *p.add(i);
            let decayed = pi - self.lr * self.wd * pi;
            *p.add(i) = decayed - self.lr * mhat / (sqrt_f(vhat) + self.eps);
        }
    }
}

// ============================================================================
// 调试用：把 (标签, 缓冲区) 列表按 f32 打印到 stdout（no_std，无 fmt）。
// 需要 crate::print，只有显式调用才进镜像。
// ============================================================================
pub unsafe fn debug_dump(label: *const u8, p: *const f32, n: usize) {
    let mut lb = 0usize;
    while *label.add(lb) != 0 && lb < 64 {
        lb += 1;
    }
    crate::print(core::slice::from_raw_parts(label, lb));
    crate::print(b" [");
    let mut buf = [0u8; 32];
    for i in 0..n {
        if i > 0 {
            crate::print(b", ");
        }
        let k = f32_to_str(*p.add(i), buf.as_mut_ptr(), buf.len());
        crate::print(core::slice::from_raw_parts(buf.as_ptr(), k));
    }
    crate::print(b"]\n");
}

// ============================================================================
// 兼容：LayerNorm 反向的「逐列归约」变体（不做 stat 分解的朴素两遍实现）。
// interp 接线用哪个由 Lead 决定；两者数学等价、数值差异 ~1e-7。
// ============================================================================
pub unsafe fn layernorm_bwd_ref(
    dy: *const f32,
    x: *const f32,
    g: *const f32,
    dx: *mut f32,
    dg: *mut f32,
    db: *mut f32,
    rows: usize,
    cols: usize,
    eps: f32,
) {
    for r in 0..rows {
        let xr = x.add(r * cols);
        let dyr = dy.add(r * cols);
        let dxr = dx.add(r * cols);
        let mut mu = 0.0f32;
        for i in 0..cols {
            mu += *xr.add(i);
        }
        mu /= cols as f32;
        let mut var = 0.0f32;
        for i in 0..cols {
            let d = *xr.add(i) - mu;
            var += d * d;
        }
        var /= cols as f32;
        let inv = 1.0 / sqrt_f(var + eps);

        let mut s_dyg = 0.0f32;
        let mut s_dygxs = 0.0f32;
        for i in 0..cols {
            let xs = (*xr.add(i) - mu) * inv;
            let dyg = *dyr.add(i) * *g.add(i);
            s_dyg += dyg;
            s_dygxs += dyg * xs;
        }
        let mean_dyg = s_dyg / cols as f32;
        let mean_dygxs = s_dygxs / cols as f32;
        for i in 0..cols {
            let xs = (*xr.add(i) - mu) * inv;
            let gd = *dyr.add(i) * *g.add(i);
            *dxr.add(i) = inv * (gd - mean_dyg - xs * mean_dygxs);
            if !db.is_null() {
                *db.add(i) += *dyr.add(i);
            }
            if !dg.is_null() {
                *dg.add(i) += *dyr.add(i) * xs;
            }
        }
    }
}

// 小工具：把整块清零（interp 如果没有 ZERO，接线时可以用这个）
pub unsafe fn zero(p: *mut f32, n: usize) {
    for i in 0..n {
        *p.add(i) = 0.0;
    }
}

// 小工具：逐元素拷贝（clamp 到 [lo, hi] 的变体留给 D 的 kernels）
pub unsafe fn copy(dst: *mut f32, src: *const f32, n: usize) {
    for i in 0..n {
        *dst.add(i) = *src.add(i);
    }
}
