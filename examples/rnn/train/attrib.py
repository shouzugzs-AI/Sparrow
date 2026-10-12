#!/usr/bin/env python3
"""
A2 / P8: final attribution -- is the W1d/W2d residual a backward bug, or the
forward exp_f approximation?

Evidence so far:
  (a) VMLM analytic vs numerical gradient of VMLM's OWN forward (h=1e-2):
      ~1e-4 .. 1e-3 relative  ->  the backward differentiates what the engine
      computes. No backward bug.
  (b) VMLM analytic vs PyTorch f64: W1d 2.37e-3, W2d 2.16e-3, everything else
      <= 9.1e-4 (16/18 params PASS).

If (a) holds and (b) fails only on the double-exp branch, the difference must be
that the engine's FORWARD is not exactly PyTorch's forward there.

To show this rigorously we need a reference whose forward is bit-close to the
engine's. The engine's exp_f works in f32 arithmetic; emulating it on f64
tensors does NOT reproduce the f32 rounding. So instead we:

  1. take the engine's own forward outputs (already dumped), and
  2. compute the reference loss using THOSE logits rather than PyTorch's,
  3. backprop through PyTorch's graph but seeded so the comparison isolates the
     backward.

More directly and more usefully: we measure how much the LOSS changes when we
swap exact exp for an exp_f evaluated in genuine f32, which bounds how much
gradient difference the approximation can cause.
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


def exp_f_f32(x64):
    """exp_f evaluated with genuine f32 rounding, but keeping the autograd graph.

    kernels::exp_f:
        x = clamp(x, -88, 88)
        n = (x * 1.44269504 + 0.5) as i32        <- truncation toward zero
        r = x - n * 0.69314718
        poly of degree 6 in r
        scale = f32::from_bits(((n + 127) as u32) << 23)
        return p * scale

    We cast to f32 for the arithmetic (so the rounding matches the kernel), then
    cast back to f64. Casting is differentiable, so gradients still flow; the
    rounding itself is treated as a straight-through step, which is exactly the
    right model of "the kernel computes the forward in f32" for attribution
    purposes.
    """
    def r32(t):
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=torch.float64)
        return t.to(torch.float32).to(torch.float64)

    x = torch.clamp(x64, -88.0, 88.0)
    xf = r32(x)
    n = torch.trunc(xf * 1.44269504 + 0.5)
    r = r32(xf - n * 0.69314718)
    r2 = r32(r * r)
    p = r32(r32(r32(r32(1.0) + r) + r32(r2 * 0.5))
            + r32(r32(r2 * r) * 0.16666667)
            + r32(r32(r2 * r2) * 0.041666668)
            + r32(r32(r32(r2 * r2) * r) * 0.008333334)
            + r32(r32(r2 * r2 * r2) * 0.001388889))
    bits = ((n.to(torch.int64) + 127) << 23).to(torch.int32)
    scale = bits.view(torch.float32).to(torch.float64)
    return r32(p * scale)


def run(mode):
    """mode: 'exact' | 'expf'. Returns (loss_sum, grads dict, logits)."""
    par = {n: torch.tensor(rd(os.path.join(DIR, f"{n}.bin")).astype(np.float64),
                           dtype=torch.float64).requires_grad_(True) for n in PARAMS}
    eps = 1e-6

    def ln(x, g, b):
        mu = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        return (x - mu) / torch.sqrt(var + eps) * g + b

    def EXP(x):
        return exp_f_f32(x) if mode == "expf" else torch.exp(x)

    def SIG(x):
        return 1.0 / (1.0 + EXP(-x))

    def TH(x):
        if mode != "expf":
            return torch.tanh(x)
        # kernels::tanh_f == act_scalar ACT_TANH path used by ACT
        e = exp_f_f32(2.0 * x)
        t = (e - 1.0) / (e + 1.0)
        return torch.where(x > 15.0, torch.ones_like(t),
                           torch.where(x < -15.0, -torch.ones_like(t), t))

    S = torch.zeros(D, D, dtype=torch.float64)
    hp = torch.zeros(H, dtype=torch.float64)
    Xt = torch.tensor(np.stack([rd(os.path.join(DIR, f"x_t{t}.bin")) for t in range(T)]),
                      dtype=torch.float64)
    outs = []
    for t in range(T):
        x = Xt[t]
        u = ln(x, par["ln1_g"], par["ln1_b"])
        d = hp - u
        r = SIG(par["W_r"].reshape(D, H) @ (u + d * par["xr"]))
        k = par["W_k"].reshape(D, H) @ (u + d * par["xk"])
        v = par["W_v"].reshape(D, H) @ (u + d * par["xv"])
        gd = TH(par["W1d"].reshape(32, H) @ (u + d * par["xw"]))
        om = par["W2d"].reshape(D, 32) @ gd + par["w0"]
        decay = EXP(-EXP(om))
        S = S * decay.unsqueeze(1) + torch.outer(v, k)
        o = ln(S @ r, par["ln_t_g"], par["ln_t_b"])
        h = hp + par["W_o"].reshape(H, D) @ o
        hp = h
        outs.append(ln(h, par["ln_out_g"], par["ln_out_b"]) @ par["E"].reshape(V, H).T)
    logits = torch.stack(outs)
    Y = torch.tensor(np.fromfile(os.path.join(DIR, "grads", "vmlm_y.bin"),
                                 dtype=np.int32).astype(np.int64))
    loss = nn.CrossEntropyLoss(reduction="sum")(logits, Y)
    loss.backward()
    return float(loss), {n: par[n].grad.detach().numpy().ravel() for n in PARAMS}, \
           logits.detach().numpy()


l_ex, g_ex, lg_ex = run("exact")
l_ap, g_ap, lg_ap = run("expf")

print(f"A2 / P8 : exp_f attribution (genuine f32 emulation)   T={T}")
print(f"\nloss: exact-exp = {l_ex:.8f}   exp_f-emul = {l_ap:.8f}   "
      f"|diff| = {abs(l_ex-l_ap):.3e}")
print(f"logits: max|diff| = {np.abs(lg_ex-lg_ap).max():.3e}")

print(f"\n{'param':<9} {'ana vs exact':>14} {'ana vs exp_f':>14}   better?")
worst_e = worst_f = 0.0
for n in PARAMS:
    a = rd(os.path.join(DIR, "grads", f"ana_{n}.bin")).astype(np.float64)
    e, f = g_ex[n], g_ap[n]
    re = np.abs(a - e).sum() / (np.linalg.norm(e) + np.linalg.norm(a) + 1e-30)
    rf = np.abs(a - f).sum() / (np.linalg.norm(f) + np.linalg.norm(a) + 1e-30)
    worst_e = max(worst_e, re)
    worst_f = max(worst_f, rf)
    tag = "YES" if rf < re * 0.99 else ("same" if rf < re * 1.01 else "no")
    print(f"{n:<9} {re:>14.3e} {rf:>14.3e}   {tag}")
print(f"\nworst vs exact-exp = {worst_e:.3e}")
print(f"worst vs exp_f     = {worst_f:.3e}")
