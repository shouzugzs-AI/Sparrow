#!/usr/bin/env python3
"""
A2: isolate the decay branch gradient.

The only parameters still failing are W1d / W2d / (marginally) xw, i.e. exactly
the `decay = exp(-exp(w))` subgraph. Everything else passes at ~1e-5.

This script compares the VMLM analytic gradient of that subgraph against a
finite-difference (numerical) gradient computed with the SAME VMLM forward.
If numerical and analytic agree, the residual is kernel approximation noise
(exp_f has 6.51e-4 max relative error, and the double exp amplifies it).
If they disagree, there is a real bug in the chain.
"""
import os
import subprocess
import sys
import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(DIR, "..", "..", ".."))
DRV = os.path.join(REPO, "target", "release", "vmlm-rnn-train.exe")

T = int(sys.argv[1]) if len(sys.argv) > 1 else 2

PARAMS = ["W_r", "W_k", "W_v", "W_o", "W1d", "W2d", "w0",
          "xr", "xk", "xv", "xw", "ln1_g", "ln1_b",
          "ln_t_g", "ln_t_b", "ln_out_g", "ln_out_b", "E"]


def rd(p):
    return np.fromfile(p, dtype=np.float32)


def run_vmlm():
    """Run one train step with lr=0 (frozen params) and return mean CE."""
    r = subprocess.run([DRV, DIR, "grad", "1", "0", str(T)],
                       capture_output=True, text=True, cwd=REPO,
                       encoding="utf-8", errors="replace")
    loss = float(rd(os.path.join(DIR, "grads", "vmlm_loss.bin"))[0])
    return loss


def read_grads():
    return {n: rd(os.path.join(DIR, "grads", f"ana_{n}.bin")) for n in PARAMS}


# ---- analytic (lr=0 so weights are not modified) ----
run_vmlm()
ana = read_grads()

# ---- numerical: perturb the .bin files, re-assemble is NOT needed because the
#      weights live in the constant pool, loaded at Vm::load from the .vmb.
#      So we must rebuild the .vmb for each perturbation. That is slow but exact.
ASM = os.path.join(REPO, "target", "release", "vmlm-asm.exe")
WA = [x for x in open(os.path.join(DIR, "weights.args"), encoding="utf-8").read().split("\n") if x.strip()]

import os as _o; h = float(_o.environ.get('NUMH','1e-3'))
targets = ["W1d", "W2d", "xw", "W_v", "E"]
print(f"numerical gradient check on the VMLM forward itself (T={T}, h={h})")
print(f"{'param':<6} {'idx':>5} {'num':>13} {'ana':>13} {'|d|':>11} {'rel':>10}")
for name in targets:
    orig = rd(os.path.join(DIR, f"{name}.bin")).copy()
    n = orig.size
    step = max(1, n // 6)
    worst = 0.0
    for k in range(0, n, step):
        res = []
        for sign in (+1, -1):
            v = orig.copy()
            v[k] += sign * h
            v.astype(np.float32).tofile(os.path.join(DIR, f"{name}.bin"))
            r = subprocess.run([ASM, f"bptt_t{T}.vml", f"bptt_t{T}.vmb"] + WA,
                               capture_output=True, text=True, cwd=DIR,
                               encoding="utf-8", errors="replace")
            if r.returncode != 0:
                print("asm failed:", r.stderr[:200])
                sys.exit(2)
            res.append(run_vmlm())
        num = (res[0] - res[1]) / (2 * h)
        # The engine's backward ACCUMULATES the per-step CE (sum reduction),
        # while the driver reports the MEAN CE over T steps. Scale the numerical
        # derivative (which differentiates the reported mean) up by T so the two
        # quantities are the same function.
        num_sum = num * T
        a = ana[name][k]
        denom = abs(num_sum) + abs(a)
        rel = abs(num_sum - a) / denom if denom > 1e-9 else 0.0
        worst = max(worst, rel)
        print(f"{name:<6} {k:>5} {num_sum:>13.6f} {a:>13.6f} {abs(num_sum-a):>11.2e} {rel:>10.2e}")
    orig.astype(np.float32).tofile(os.path.join(DIR, f"{name}.bin"))
    print(f"  -> {name}: worst rel = {worst:.3e}")

# restore the canonical vmb
subprocess.run([ASM, f"bptt_t{T}.vml", f"bptt_t{T}.vmb"] + WA,
               capture_output=True, cwd=DIR)
print("\n(restored canonical .vmb)")
