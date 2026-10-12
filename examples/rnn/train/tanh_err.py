#!/usr/bin/env python3
"""
A2: is the W1d/W2d residual caused by tanh's exp_f approximation?

Reasoning that led here:
  * w0's gradient (= sum_t dom_t) matches the reference to ~6 significant
    figures  -> `dom`, the gradient w.r.t. the pre-decay logit, is correct.
  * The ROW SUMS of g_W2d (= sum_t dom_t[j] * sum_p gdact_t[p]) also match to
    ~4e-4 relative -> dom's per-step split is right too.
  * But g_W2d's individual elements differ by up to 2.2e-3 relative.
  Therefore the error must live in `gdact` per-element -- i.e. in tanh itself.

kernels::tanh_f(x) = (exp_f(2x) - 1) / (exp_f(2x) + 1)
So tanh inherits exp_f's error, and it is evaluated TWICE in the tanh path
(once in the forward ACT, once inside ACT_BWD), while its derivative
1 - t^2 amplifies that error near |t| -> 1.

This script measures kernels::tanh_f's actual error against f64 tanh, to see
whether the 2.2e-3 residual is explained.
"""
import numpy as np


def exp_f(x: np.ndarray) -> np.ndarray:
    """Bit-exact reimplementation of kernels::exp_f."""
    x = np.clip(x, -88.0, 88.0).astype(np.float32)
    n = np.floor(x * np.float32(1.44269504) + np.float32(0.5)).astype(np.int32)
    r = x - n.astype(np.float32) * np.float32(0.69314718)
    r2 = (r * r).astype(np.float32)
    p = (np.float32(1.0) + r + r2 * np.float32(0.5)
         + r2 * r * np.float32(0.16666667)
         + r2 * r2 * np.float32(0.041666668)
         + r2 * r2 * r * np.float32(0.008333334)
         + r2 * r2 * r2 * np.float32(0.001388889)).astype(np.float32)
    bits = ((n.astype(np.int64) + 127) << 23).astype(np.uint32)
    scale = bits.view(np.float32)
    return (p * scale).astype(np.float32)


def tanh_f(x: np.ndarray) -> np.ndarray:
    """kernels::tanh_f (clamped at |x|>15)."""
    x = x.astype(np.float32)
    e = exp_f(2.0 * x)
    t = (e - 1.0) / (e + 1.0)
    return np.where(x > 15.0, np.float32(1.0),
                    np.where(x < -15.0, np.float32(-1.0), t)).astype(np.float32)


rng = np.random.default_rng(0)
# the actual gdpre values in this model live in a modest range
xs = np.concatenate([rng.standard_normal(200000) * 1.5, np.linspace(-4, 4, 100000)])
exact = np.tanh(xs.astype(np.float64))
approx = tanh_f(xs).astype(np.float64)

relerr = np.abs(approx - exact) / (np.abs(exact) + 1e-12)
abserr = np.abs(approx - exact)

print("kernels::tanh_f error vs exact tanh (f64)")
print(f"  max |abs err| = {abserr.max():.3e}")
print(f"  max rel err   = {relerr.max():.3e}")
print(f"  mean rel err  = {relerr.mean():.3e}")

# derivative amplification: d(tanh)/dx = 1 - t^2 ; the gradient of a tanh layer
# carries this factor, so relative error in t is amplified by 2|t|/(1-t^2)
t = exact
amp = 1.0 / np.maximum(1e-9, 1.0 - t * t)
print(f"\n  max amplification 1/(1-t^2) over sampled range = {amp.max():.3e}")

# empirical: error in 1-t^2 (which is what ACT_BWD feeds forward)
d_exact = 1.0 - exact * exact
d_approx = 1.0 - approx * approx
drel = np.abs(d_approx - d_exact) / (np.abs(d_exact) + 1e-12)
print(f"  max rel err in (1 - t^2) = {drel.max():.3e}")
print(f"  mean rel err in (1-t^2)  = {drel.mean():.3e}")

# restrict to the range this model actually uses (|gdpre| < 3)
m = np.abs(xs) < 3.0
print(f"\n  restricted to |x|<3 (this model's regime):")
print(f"    max rel err in (1-t^2) = {drel[m].max():.3e}")
print(f"    mean rel err in (1-t^2)= {drel[m].mean():.3e}")
