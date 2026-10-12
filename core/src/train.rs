// 训练：梯度区 + 反向内核 + 优化器
//
// 设计（白皮书 §6）：
//   反向传播也是「指令流」——由 .vml 脚本显式编写反向序列，
//   引擎不做自动微分。这样新架构不需要引擎支持 AD，只要脚本写清反向。
//
//   前向：y = x·Wᵀ + b
//   反向（脚本显式调用）：
//     GEMM_BWD_W  dW += dyᵀ·x      （累积到常量池的梯度区）
//     GEMM_BWD_X  dx  = dy·W
//     BIAS_BWD    db += sum(dy, 行)
//     ACT_BWD     dx  = dy ⊙ act'(x)
//
// 梯度区与常量池同布局：每个可训练常量对应一块同样大小的梯度缓冲。
// 训练时只写梯度区 → 优化器就地更新常量池 → 零重编译（需求 2 核心）。
#![allow(dead_code)]
use crate::kernels::{act_scalar, exp_f, ln_f, sqrt_f};

#[inline(always)]
pub fn pow_f(b: f32, e: f32) -> f32 {
    exp_f(e * ln_f(b))
}

// ================= 反向内核 =================

// dW[N,K] += dy[M,N]ᵀ · x[M,K]    （Linear 权重梯度）
pub unsafe fn gemm_bwd_w(
    dy: *const f32,
    x: *const f32,
    dw: *mut f32,
    m: usize,
    n: usize,
    k: usize,
) {
    // dw[j][p] += Σ_i dy[i][j] * x[i][p]
    for i in 0..m {
        let dyi = dy.add(i * n);
        let xi = x.add(i * k);
        for j in 0..n {
            let g = *dyi.add(j);
            if g != 0.0 {
                let row = dw.add(j * k);
                for p in 0..k {
                    *row.add(p) += g * *xi.add(p);
                }
            }
        }
    }
}

// dx[M,K] = dy[M,N] · W[N,K]     （输入梯度）
pub unsafe fn gemm_bwd_x(
    dy: *const f32,
    w: *const f32,
    dx: *mut f32,
    m: usize,
    n: usize,
    k: usize,
) {
    for i in 0..m {
        let dyi = dy.add(i * n);
        let dxi = dx.add(i * k);
        for p in 0..k {
            *dxi.add(p) = 0.0;
        }
        for j in 0..n {
            let g = *dyi.add(j);
            if g != 0.0 {
                let wrow = w.add(j * k);
                for p in 0..k {
                    *dxi.add(p) += g * *wrow.add(p);
                }
            }
        }
    }
}

// db[N] += Σ_i dy[i][j]
pub unsafe fn bias_bwd(dy: *const f32, db: *mut f32, m: usize, n: usize) {
    for i in 0..m {
        let dyi = dy.add(i * n);
        for j in 0..n {
            *db.add(j) += *dyi.add(j);
        }
    }
}

// dx = dy ⊙ act'(y_pre)   激活反向
pub unsafe fn act_bwd(dy: *const f32, ypre: *const f32, dx: *mut f32, n: usize, kind: u16) {
    for i in 0..n {
        let x = *ypre.add(i);
        let d = match kind {
            crate::format::ACT_RELU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            crate::format::ACT_SIGMOID => {
                let s = 1.0 / (1.0 + exp_f(-x));
                s * (1.0 - s)
            }
            crate::format::ACT_TANH => {
                let t = (exp_f(2.0 * x) - 1.0) / (exp_f(2.0 * x) + 1.0);
                1.0 - t * t
            }
            crate::format::ACT_SILU => {
                let s = 1.0 / (1.0 + exp_f(-x));
                s + x * s * (1.0 - s)
            }
            crate::format::ACT_RELU2 => {
                if x > 0.0 {
                    2.0 * x
                } else {
                    0.0
                }
            }
            _ => 1.0,
        };
        *dx.add(i) = *dy.add(i) * d;
    }
    let _ = act_scalar;
}

// 交叉熵 + softmax 反向：dlogits = (softmax - onehot) * scale
// 返回 loss（就地写入 dlogits）
pub unsafe fn ce_softmax_bwd(
    logits: *const f32,
    target: usize,
    dlogits: *mut f32,
    n: usize,
) -> f32 {
    // softmax
    let mut mx = f32::NEG_INFINITY;
    for i in 0..n {
        let v = *logits.add(i);
        if v > mx {
            mx = v;
        }
    }
    let mut sum = 0.0f32;
    for i in 0..n {
        let e = exp_f(*logits.add(i) - mx);
        *dlogits.add(i) = e;
        sum += e;
    }
    let inv = 1.0 / sum;
    for i in 0..n {
        *dlogits.add(i) *= inv;
    }
    // loss = -log(p[target])
    let p = *dlogits.add(target);
    let loss = -ln_f(p);
    // dlogits = p - onehot
    *dlogits.add(target) -= 1.0;
    loss
}

// MSE 反向：dx = 2*(y - t)/n，返回 loss
pub unsafe fn mse_bwd(y: *const f32, t: *const f32, dy: *mut f32, n: usize) -> f32 {
    let mut loss = 0.0f32;
    let k = 2.0 / n as f32;
    for i in 0..n {
        let d = *y.add(i) - *t.add(i);
        loss += d * d;
        *dy.add(i) = k * d;
    }
    loss / n as f32
}

// ================= 优化器 =================

// SGD：p -= lr * g
pub unsafe fn sgd(p: *mut f32, g: *const f32, n: usize, lr: f32) {
    for i in 0..n {
        *p.add(i) -= lr * *g.add(i);
    }
}

// 梯度裁剪（原地缩放），返回原范数
//
// [P19-9] 平方和与缩放都走 AVX2（`dot256_f32` / `scale_mul_avx2`）。
//   每步要对 15 组梯度调用它（层 16k × 12 + head/embed 131k × 4 ≈ 70 万个数），
//   原来是一条标量 FMA 链 + 一条标量缩放链。归约分组变了 ⇒ ε 级（只有 nrm>max 时才缩放）。
pub unsafe fn clip_grad(g: *mut f32, n: usize, max_norm: f32) -> f32 {
    #[allow(unused_mut)]
    let mut s = 0.0f32;
    #[cfg(target_arch = "x86_64")]
    {
        if crate::interp::HAS_AVX2 {
            s = crate::kernels::dot256_f32(g, g, n);
        } else {
            for i in 0..n {
                s += *g.add(i) * *g.add(i);
            }
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        for i in 0..n {
            s += *g.add(i) * *g.add(i);
        }
    }
    let nrm = sqrt_f(s);
    if nrm > max_norm && nrm > 0.0 {
        let sc = max_norm / nrm;
        #[cfg(target_arch = "x86_64")]
        {
            if crate::interp::HAS_AVX2 {
                crate::kernels::scale_mul_avx2(g, g, sc, n);
                return nrm;
            }
        }
        for i in 0..n {
            *g.add(i) *= sc;
        }
    }
    nrm
}

pub struct Adam {
    pub m: *mut f32,
    pub v: *mut f32,
    pub lr: f32,
    pub b1: f32,
    pub b2: f32,
    pub eps: f32,
    pub t: u32,
}

impl Adam {
    pub unsafe fn step(&mut self, p: *mut f32, g: *const f32, n: usize) {
        self.t += 1;
        let t = self.t as f32;
        let c1 = 1.0 - pow_f(self.b1, t);
        let c2 = 1.0 - pow_f(self.b2, t);
        for i in 0..n {
            let gi = *g.add(i);
            let m = self.b1 * *self.m.add(i) + (1.0 - self.b1) * gi;
            let v = self.b2 * *self.v.add(i) + (1.0 - self.b2) * gi * gi;
            *self.m.add(i) = m;
            *self.v.add(i) = v;
            *p.add(i) -= self.lr * (m / c1) / (sqrt_f(v / c2) + self.eps);
        }
    }
}

// ================= 数值梯度（L0 基准，用于校验解析梯度）=================
// 对常量池的每个元素做中心差分：df/dw ≈ (f(w+h) - f(w-h)) / 2h
// 这是唯一可信的梯度正确性判据（白皮书 §6.4）
pub unsafe fn numeric_grad<F: Fn() -> f32>(
    p: *mut f32,
    g: *mut f32,
    n: usize,
    h: f32,
    f: F,
) {
    for i in 0..n {
        let orig = *p.add(i);
        *p.add(i) = orig + h;
        let fp = f();
        *p.add(i) = orig - h;
        let fm = f();
        *p.add(i) = orig;
        *g.add(i) = (fp - fm) / (2.0 * h);
    }
}
