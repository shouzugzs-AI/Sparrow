#!/usr/bin/env python3
"""
A2 / P8: gradient check -- VMLM analytic BPTT gradients vs PyTorch autograd.

Criterion (contract section 3 / 5):
    sum|delta| / (||num|| + ||ana||) < 1e-3
NOT per-element relative error (that explodes when a gradient component is ~0).

Two references are produced:
  * f64 exact  : PyTorch float64, the mathematical ground truth
  * f32 emul   : PyTorch float32 (a sanity cross-check)

Why f64 matters here: kernels::exp_f has max relative error 6.51e-4, and both
sigmoid and exp (decay) route through it, so the forward itself carries ~1e-4
level error. Comparing against an f32 reference would conflate "my backward is
wrong" with "the engine's exp is approximate". Against f64 we can separate them:
a correctly-differentiated graph still lands well under 1e-3.

The VMLM gradients are dumped by `vmlm-rnn-train <dir> grad 1 <lr> <T>`.
"""
import os
import sys
import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))
T = int(sys.argv[1]) if len(sys.argv) > 1 else 2
H = int(sys.argv[2]) if len(sys.argv) > 2 else 16
D = int(sys.argv[3]) if len(sys.argv) > 3 else 8
V = int(sys.argv[4]) if len(sys.argv) > 4 else 4

PARAMS = ["W_r", "W_k", "W_v", "W_o", "W1d", "W2d", "w0",
          "xr", "xk", "xv", "xw", "ln1_g", "ln1_b",
          "ln_t_g", "ln_t_b", "ln_out_g", "ln_out_b", "E"]


def rd(name):
    p = os.path.join(DIR, f"{name}.bin")
    return np.fromfile(p, dtype=np.float32)


def rd_grad(name):
    p = os.path.join(DIR, "grads", f"ana_{name}.bin")
    if not os.path.exists(p):
        return None
    return np.fromfile(p, dtype=np.float32)


P = {n: rd(n) for n in PARAMS}
X = np.stack([rd(f"x_t{t}") for t in range(T)]).astype(np.float64)
Y = np.array([int(rd(f"y{t}")[0]) for t in range(T)], dtype=np.int64)

# Guard against comparing against a stale dump produced for a different T
# (the grads/ directory is reused between runs).
_gdir = os.path.join(DIR, "grads")
_tfile = os.path.join(_gdir, "vmlm_T.txt")
if not os.path.exists(_tfile):
    print(f"!! {_tfile} missing -- run: vmlm-rnn-train <dir> grad 1 <lr> {T}")
    sys.exit(2)
_dumped_t = int(open(_tfile).read().strip())
if _dumped_t != T:
    print(f"!! grads/ was dumped for T={_dumped_t} but this check is T={T}.")
    print(f"   re-run: vmlm-rnn-train examples/rnn/train grad 1 3e-3 {T}")
    sys.exit(2)
Y = np.fromfile(os.path.join(_gdir, "vmlm_y.bin"), dtype=np.int32).astype(np.int64)

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("PyTorch required for the reference; aborting")
    sys.exit(2)

torch.set_num_threads(1)


def build(dtype):
    np_dt = np.float64 if dtype == torch.float64 else np.float32
    par = {}
    for n in PARAMS:
        par[n] = torch.tensor(P[n].astype(np_dt), dtype=dtype).clone().requires_grad_(True)
    return par


def forward(par, dtype):
    """Identical to the VMLM forward, using eps=1e-6 (LN_EPS eps1e6)."""
    eps = 1e-6

    def ln(x, g, b):
        mu = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        return (x - mu) / torch.sqrt(var + eps) * g + b

    S = torch.zeros(D, D, dtype=dtype)
    h_prev = torch.zeros(H, dtype=dtype)
    Xt = torch.tensor(X, dtype=dtype)
    outs = []
    for t in range(T):
        x = Xt[t]
        u = ln(x, par["ln1_g"], par["ln1_b"])
        d = h_prev - u
        r = torch.sigmoid(par["W_r"].reshape(D, H) @ (u + d * par["xr"]))
        k = par["W_k"].reshape(D, H) @ (u + d * par["xk"])
        v = par["W_v"].reshape(D, H) @ (u + d * par["xv"])
        gd = torch.tanh(par["W1d"].reshape(32, H) @ (u + d * par["xw"]))
        om = par["W2d"].reshape(D, 32) @ gd + par["w0"]
        decay = torch.exp(-torch.exp(om))
        S = S * decay.unsqueeze(1) + torch.outer(v, k)
        o = S @ r
        o = ln(o, par["ln_t_g"], par["ln_t_b"])
        h = h_prev + par["W_o"].reshape(H, D) @ o
        h_prev = h
        outs.append(ln(h, par["ln_out_g"], par["ln_out_b"]) @ par["E"].reshape(V, H).T)
    return torch.stack(outs)


def ref_grads(dtype):
    """Reference gradients.

    NOTE on reduction: the VMLM backward issues one `CE_BWD` per step and lets
    the *_BWD operators accumulate, so its implicit objective is the **sum** of
    the per-step cross-entropies, not the mean. PyTorch's CrossEntropyLoss
    defaults to reduction='mean' over the flattened batch, which would scale
    every gradient by 1/T (T=2 gave an exact 2.0000 ratio). We therefore use
    reduction='sum' so the two objectives are actually the same function.
    """
    par = build(dtype)
    logits = forward(par, dtype)
    loss = nn.CrossEntropyLoss(reduction="sum")(logits, torch.tensor(Y))
    loss.backward()
    return {n: par[n].grad.detach().numpy().astype(np.float64).ravel() for n in PARAMS}, \
           float(loss.item()), logits.detach().numpy()


g64, loss64, lg64 = ref_grads(torch.float64)
g32, loss32, lg32 = ref_grads(torch.float32)

print("=" * 78)
print(f"A2 / P8  gradient check   T={T} H={H} D={D} V={V}")
print("=" * 78)

# ---- forward cross-check first (a wrong forward invalidates any gradcheck) ----
vmlm_loss = float(np.fromfile(os.path.join(DIR, "grads", "vmlm_loss.bin"),
                              dtype=np.float32)[0])
print()
maxlg = 0.0
for i in range(T):
    v = np.fromfile(os.path.join(DIR, "grads", f"vmlm_logits{i}.bin"), dtype=np.float32)
    d = np.abs(v - lg64[i]).max()
    maxlg = max(maxlg, d)
    print(f"[forward] step {i}  PyTorch(f64) = {np.round(lg64[i], 6).tolist()}")
    print(f"                   VMLM        = {np.round(v, 6).tolist()}")
    print(f"                   max |diff|  = {d:.3e}")
print(f"\n[forward] worst max|diff| over {T} steps = {maxlg:.3e}")
print(f"\n[loss]     PyTorch(f64) sum CE over {T} steps = {loss64:.8f}")
print(f"           PyTorch(f32) sum CE over {T} steps = {loss32:.8f}")
print(f"           VMLM         mean CE x T          = {vmlm_loss * T:.8f}")
print(f"           (VMLM reports the MEAN; the backward accumulates the SUM,")
print(f"            so mean x T is the quantity comparable to reduction='sum')")
print(f"           |VMLM*T - f64| = {abs(vmlm_loss * T - loss64):.3e}")

print("\n" + "-" * 78)
print(f"{'param':<10} {'n':>6} {'||num||':>11} {'||ana||':>11} "
      f"{'sum|d|':>11} {'rel':>10}  verdict")
print("-" * 78)

worst = 0.0
worst_name = ""
all_num, all_ana = [], []
rows = []
for n in PARAMS:
    ana = rd_grad(n)
    if ana is None:
        print(f"{n:<10} (no VMLM grad dumped)")
        continue
    num = g64[n]
    assert ana.size == num.size, f"{n}: size {ana.size} vs {num.size}"
    # criterion: sum|delta| / (||num|| + ||ana||)
    rel = np.abs(ana - num).sum() / (np.linalg.norm(num) + np.linalg.norm(ana) + 1e-30)
    all_num.append(num)
    all_ana.append(ana)
    ok = rel < 1e-3
    rows.append((n, num.size, np.linalg.norm(num), np.linalg.norm(ana),
                 np.abs(ana - num).sum(), rel, ok))
    if rel > worst:
        worst, worst_name = rel, n

for n, sz, nn_, na_, sd, rel, ok in rows:
    print(f"{n:<10} {sz:>6} {nn_:>11.5f} {na_:>11.5f} {sd:>11.5f} {rel:>10.3e}  "
          f"{'PASS' if ok else 'FAIL'}")

# global criterion over the concatenation of all gradients
NUM = np.concatenate(all_num)
ANA = np.concatenate(all_ana)
glob = np.abs(ANA - NUM).sum() / (np.linalg.norm(NUM) + np.linalg.norm(ANA))

print("-" * 78)
print(f"{'GLOBAL':<10} {NUM.size:>6} {np.linalg.norm(NUM):>11.5f} "
      f"{np.linalg.norm(ANA):>11.5f} {np.abs(ANA-NUM).sum():>11.5f} {glob:>10.3e}")
print("-" * 78)
print(f"\nworst param = {worst_name}  (rel = {worst:.3e})")
print(f"GLOBAL  sum|delta| / (||num||+||ana||) = {glob:.6e}")
print(f"criterion (< 1e-3): {'PASS' if glob < 1e-3 else 'FAIL'}")

# per-element worst offender, for debugging
if glob >= 1e-3:
    print("\nworst 10 elements (num vs ana):")
    order = np.argsort(-np.abs(ANA - NUM))[:10]
    print(f"{'idx':>7} {'num':>14} {'ana':>14} {'diff':>14}")
    for i in order:
        print(f"{i:>7} {NUM[i]:>14.6f} {ANA[i]:>14.6f} {ANA[i]-NUM[i]:>14.6f}")
sys.exit(0 if glob < 1e-3 else 1)
