#!/usr/bin/env python3
"""
A2: decide whether the W1d/W2d residual is "implementation wrong" or
"exp_f approximation pollution".

The task brief warns that kernels::exp_f has max relative error 6.51e-4, and that
anything downstream of it inherits that error. The decay gate is

    decay = exp(-exp(w))

i.e. TWO nested exps, and its derivative divides by nothing but multiplies by
exp(w) -- so the relative error of the forward decay is roughly doubled, and
d(decay)/dw is roughly 2x as sensitive.

This script re-implements exp_f EXACTLY as core/src/kernels.rs does (bit trick +
degree-6 polynomial), runs the SAME PyTorch autograd graph but with every exp
replaced by that approximate version, and then compares:

   VMLM analytic  vs  torch(f64, exact exp)     <- "is it right mathematically"
   VMLM analytic  vs  torch(f64, exp_f approx)  <- "is it right given the kernel"

If the second comparison is far better than the first, the residual is kernel
approximation, not a bug in the backward chain.
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn

DIR = os.path.dirname(os.path.abspath(__file__))
T = int(sys.argv[1]) if len(sys.argv) > 1 else 2
H, D, V = 16, 8, 4

PARAMS = ["W_r", "W_k", "W_v", "W_o", "W1d", "W2d", "w0",
          "xr", "xk", "xv", "xw", "ln1_g", "ln1_b",
          "ln_t_g", "ln_t_b", "ln_out_g", "ln_out_b", "E"]


def rd(p):
    return np.fromfile(p, dtype=np.float32)


def exp_f(x: torch.Tensor) -> torch.Tensor:
    """Bit-exact reimplementation of kernels::exp_f (f32, AVX-free)."""
    xc = torch.clamp(x, -88.0, 88.0)
    n = torch.floor(xc * 1.44269504 + 0.5)
    r = xc - n * 0.69314718
    r2 = r * r
    p = (1.0 + r + r2 * 0.5 + r2 * r * 0.16666667
         + r2 * r2 * 0.041666668 + r2 * r2 * r * 0.008333334
         + r2 * r2 * r2 * 0.001388889)
    bits = ((n.to(torch.int64) + 127) << 23).to(torch.int32)
    scale = bits.view(torch.float32)
    return p * scale


def build(use_expf):
    par = {}
    for n in PARAMS:
        par[n] = torch.tensor(rd(os.path.join(DIR, f"{n}.bin")).astype(np.float64),
                              dtype=torch.float64).clone().requires_grad_(True)
    return par


def forward(par, use_expf):
    eps = 1e-6

    def ln(x, g, b):
        mu = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        return (x - mu) / torch.sqrt(var + eps) * g + b

    def EXP(x):
        return exp_f(x) if use_expf else torch.exp(x)

    def SIGMOID(x):
        # kernels::act_scalar ACT_SIGMOID = 1/(1+exp_f(-x))
        return 1.0 / (1.0 + EXP(-x)) if use_expf else torch.sigmoid(x)

    def TANH(x):
        # kernels::tanh_f = (exp_f(2x)-1)/(exp_f(2x)+1), clamped at |x|>15
        if not use_expf:
            return torch.tanh(x)
        e = EXP(2.0 * x)
        t = (e - 1.0) / (e + 1.0)
        return torch.where(x > 15.0, torch.ones_like(t),
                           torch.where(x < -15.0, -torch.ones_like(t), t))

    S = torch.zeros(D, D, dtype=torch.float64)
    h_prev = torch.zeros(H, dtype=torch.float64)
    Xt = torch.tensor(np.stack([rd(os.path.join(DIR, f"x_t{t}.bin")) for t in range(T)]),
                      dtype=torch.float64)
    outs = []
    for t in range(T):
        x = Xt[t]
        u = ln(x, par["ln1_g"], par["ln1_b"])
        d = h_prev - u
        r = SIGMOID(par["W_r"].reshape(D, H) @ (u + d * par["xr"]))
        k = par["W_k"].reshape(D, H) @ (u + d * par["xk"])
        v = par["W_v"].reshape(D, H) @ (u + d * par["xv"])
        gd = TANH(par["W1d"].reshape(32, H) @ (u + d * par["xw"]))
        om = par["W2d"].reshape(D, 32) @ gd + par["w0"]
        decay = EXP(-EXP(om))
        S = S * decay.unsqueeze(1) + torch.outer(v, k)
        o = ln(S @ r, par["ln_t_g"], par["ln_t_b"])
        h = h_prev + par["W_o"].reshape(H, D) @ o
        h_prev = h
        outs.append(ln(h, par["ln_out_g"], par["ln_out_b"]) @ par["E"].reshape(V, H).T)
    return torch.stack(outs)


Y = np.fromfile(os.path.join(DIR, "grads", "vmlm_y.bin"), dtype=np.int32).astype(np.int64)
Yt = torch.tensor(Y)

print(f"A2 / P8 : exp_f pollution test   T={T}")
print(f"{'param':<9} {'vs exact-exp':>14} {'vs exp_f-emul':>14}   verdict")
res = {}
for mode, label in ((False, "exact"), (True, "expf")):
    par = build(mode)
    nn.CrossEntropyLoss(reduction="sum")(forward(par, mode), Yt).backward()
    res[label] = {n: par[n].grad.detach().numpy().ravel() for n in PARAMS}

worst_exact = worst_emul = 0.0
for n in PARAMS:
    a = rd(os.path.join(DIR, "grads", f"ana_{n}.bin")).astype(np.float64)
    e = res["exact"][n]
    f = res["expf"][n]
    rel_e = np.abs(a - e).sum() / (np.linalg.norm(e) + np.linalg.norm(a) + 1e-30)
    rel_f = np.abs(a - f).sum() / (np.linalg.norm(f) + np.linalg.norm(a) + 1e-30)
    worst_exact = max(worst_exact, rel_e)
    worst_emul = max(worst_emul, rel_f)
    tag = "PASS" if rel_f < 1e-3 else ("near" if rel_f < 5e-3 else "FAIL")
    print(f"{n:<9} {rel_e:>14.3e} {rel_f:>14.3e}   {tag}")

print(f"\nworst vs exact exp      = {worst_exact:.3e}")
print(f"worst vs exp_f emulation= {worst_emul:.3e}")
print(f"\ninterpretation: if the exp_f column is substantially better, the residual")
print(f"is the kernel's 6.51e-4 approximation amplified by the nested exp, not a")
print(f"bug in the backward chain.")
