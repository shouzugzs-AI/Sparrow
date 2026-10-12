#!/usr/bin/env python3
"""
A2 / P8：RNN 训练闭环（BPTT）—— 权重生成 + PyTorch 同结构参考（前向 + autograd 梯度）

用途：
  1. 生成 VMLM `.vml` 汇编器要注入的权重/输入 `.bin`（f32 小端）
  2. 用 PyTorch 搭一个**逐算子完全相同**的 ESERNN 前向，取 loss.backward() 的参考梯度
  3. 输出参考 loss 序列（用于和 VMLM 前向对齐）

等价性关键（必须与 examples/rnn/ese_step.vml + core/src/kernels.rs 严格一致）：
  - LayerNorm：mean/var 都是 **biased**（除以 N，不是 N-1）；eps 用 LN_EPS 设定
  - GEMM：C[M,N] = A[M,K] · B[N,K]^T（NT 默认；B 是 nn.Linear 的 [out,in] 权重）
  - sigmoid/tanh/exp 在 VMLM 里走 kernels::exp_f（近似），
    这里脚本用精确 f64 exp 复算一份，用来区分"实现错" vs "exp 近似污染"
  - S ← S ⊙ decay + v ⊗ k ，decay 沿 **i 下标**（S 的行）作用
  - 残差：h = h_prev + W_o · o
  - logits = LN(h) · E^T
"""
import os
import sys
import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------- 配置（命令行可覆盖：T H D V seed） ----------------
T = int(sys.argv[1]) if len(sys.argv) > 1 else 2
H = int(sys.argv[2]) if len(sys.argv) > 2 else 16
D = int(sys.argv[3]) if len(sys.argv) > 3 else 8
V = int(sys.argv[4]) if len(sys.argv) > 4 else 4
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 7

rng = np.random.default_rng(SEED)


def rnd(*s):
    """小方差初始化，避免 RNN 在 T 步展开后激活爆炸"""
    return (rng.standard_normal(s) * 0.3).astype(np.float32)


F32 = np.float32

# ---------------- 可训练参数（VMLM 常量名 -> 数组，全部按 [rows, cols]） ----------------
P = {
    "W_r": rnd(D, H), "W_k": rnd(D, H), "W_v": rnd(D, H),
    "W_o": rnd(H, D),
    "W1d": rnd(32, H), "W2d": rnd(D, 32), "w0": rnd(1, D),
    # 逐通道条件向量（[1,H]）
    "xr": rnd(1, H), "xk": rnd(1, H), "xv": rnd(1, H), "xw": rnd(1, H),
    # LayerNorm 仿射
    "ln1_g": np.ones((1, H), dtype=np.float32), "ln1_b": np.zeros((1, H), dtype=np.float32),
    "ln_t_g": np.ones((1, D), dtype=np.float32), "ln_t_b": np.zeros((1, D), dtype=np.float32),
    "ln_out_g": np.ones((1, H), dtype=np.float32), "ln_out_b": np.zeros((1, H), dtype=np.float32),
    # 输出词表投影
    "E": rnd(V, H),
}

# ---------------- 输入序列：T 个 token 的嵌入 [T, H] ----------------
X = (rng.standard_normal((T, H)) * 0.5).astype(np.float32)

# ---------------- 目标：每个 step 一个类别（[T]） ----------------
Y = rng.integers(0, V, size=T).astype(np.int64)

# ---------------- 非参数槽（VMLM 侧需要，但这里也存一份便于对照） ----------------
HERE = {
    "eps1e6": np.array([1e-6], dtype=np.float32),
}

# 写入 .bin
for k, v in P.items():
    v.astype(np.float32).tofile(os.path.join(DIR, f"{k}.bin"))
for k, v in HERE.items():
    v.astype(np.float32).tofile(os.path.join(DIR, f"{k}.bin"))
# x 序列按 step 切片存（VMLM 每步喂一行）
X.astype(np.float32).tofile(os.path.join(DIR, "X.bin"))
for t in range(T):
    X[t].astype(np.float32).tofile(os.path.join(DIR, f"x_t{t}.bin"))
Y.astype(np.int64).tofile(os.path.join(DIR, "Y.bin"))
# 每个 step 一个目标（标量槽；CE_BWD 取第 0 个元素当 index）
for t in range(T):
    np.array([Y[t]], dtype=np.float32).tofile(os.path.join(DIR, f"y{t}.bin"))

# 超参：hparams = [lr, b1, b2, eps, wd]（ADAMW 用）
LR0 = 3e-3
np.array([LR0, 0.9, 0.999, 1e-8, 1e-4], dtype=np.float32).tofile(
    os.path.join(DIR, "hparams.bin"))
# 梯度裁剪阈值（0 或很大 = 关闭）
np.array([1e9], dtype=np.float32).tofile(os.path.join(DIR, "clip_norm.bin"))

print(f"配置: T={T} H={H} D={D} V={V} seed={SEED}")
print(f"目标 y = {Y.tolist()}")

# ============================================================================
# PyTorch 参考（相同前向 + autograd）
# ============================================================================
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("!! 无 PyTorch，跳过 autograd 参考")

if HAS_TORCH:
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    # 用 f64 做参考，避免"参考自身"的精度污染
    DT = torch.float64

    def t_(name):
        return torch.tensor(P[name].reshape(-1), dtype=DT, requires_grad=False)

    # 参数张量（叶子，需要梯度）
    names = ["W_r", "W_k", "W_v", "W_o", "W1d", "W2d", "w0",
             "xr", "xk", "xv", "xw", "E"]
    par = {}
    for n in names:
        par[n] = torch.tensor(P[n].reshape(-1), dtype=DT).clone().requires_grad_(True)

    def reshape(t, *s):
        return t.reshape(*s)

    def ln(x, g, b, eps):
        """与 kernels::layernorm 一致：biased var"""
        mu = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        return (x - mu) / torch.sqrt(var + eps) * g + b

    def forward(par, Xt, eps=1e-6):
        """Xt: [T,H] tensor. 返回 logits [T,V]"""
        S = torch.zeros(D, D, dtype=DT)
        h_prev = torch.zeros(H, dtype=DT)
        logits_all = []
        for t in range(T):
            x = Xt[t]
            u = ln(x, lg, lb, eps)
            d = h_prev - u
            r = torch.sigmoid(Wr(par) @ (u + d * par["xr"]))
            k = Wk(par) @ (u + d * par["xk"])
            v = Wv(par) @ (u + d * par["xv"])
            om = torch.tanh(W1(par) @ (u + d * par["xw"]))
            om = W2(par) @ om + par["w0"]
            decay = torch.exp(-torch.exp(om))
            S = S * decay.unsqueeze(1) + torch.outer(v, k)
            o = S @ r
            o = ln(o, lt_g, lt_b, eps)
            h = h_prev + Wo(par) @ o
            h_prev = h
            logits_all.append(ln(h, lo_g, lo_b, eps) @ E(par).T)
        return torch.stack(logits_all)

    def Wr(p): return reshape(p["W_r"], D, H)
    def Wk(p): return reshape(p["W_k"], D, H)
    def Wv(p): return reshape(p["W_v"], D, H)
    def Wo(p): return reshape(p["W_o"], H, D)
    def W1(p): return reshape(p["W1d"], 32, H)
    def W2(p): return reshape(p["W2d"], D, 32)
    def E(p): return reshape(p["E"], V, H)

    lg = torch.tensor(P["ln1_g"].reshape(-1), dtype=DT)
    lb = torch.tensor(P["ln1_b"].reshape(-1), dtype=DT)
    lt_g = torch.tensor(P["ln_t_g"].reshape(-1), dtype=DT)
    lt_b = torch.tensor(P["ln_t_b"].reshape(-1), dtype=DT)
    lo_g = torch.tensor(P["ln_out_g"].reshape(-1), dtype=DT)
    lo_b = torch.tensor(P["ln_out_b"].reshape(-1), dtype=DT)

    Xt = torch.tensor(X, dtype=DT)
    Yt = torch.tensor(Y, dtype=torch.long)
    lossf = nn.CrossEntropyLoss()

    logits = forward(par, Xt)
    loss = lossf(logits, Yt)
    loss.backward()

    print(f"\nPyTorch(f64) 参考 loss = {loss.item():.8f}")
    print("末步 logits =", [f"{z:.6f}" for z in logits[-1].detach().numpy()])

    # 导出参考梯度（与 VMLM 梯度槽同展开顺序，行主序）
    for n in names:
        g = par[n].grad.detach().numpy().astype(np.float32)
        g.tofile(os.path.join(DIR, f"refg_{n}.bin"))
    np.array([loss.item()], dtype=np.float32).tofile(os.path.join(DIR, "ref_loss.bin"))
    logits.detach().numpy().astype(np.float32).tofile(os.path.join(DIR, "ref_logits.bin"))
    print("已导出参考梯度: " + ", ".join(f"refg_{n}.bin" for n in names))
