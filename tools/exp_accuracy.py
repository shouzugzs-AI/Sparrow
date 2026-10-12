#!/usr/bin/env python3
"""
H / P9 : exp_f accuracy sweep (before vs after).

Purpose
-------
`core/src/kernels.rs::exp_f` is the single kernel that has twice caused a
*false* "gradient check FAILED" verdict (P7-A softmax_bwd, P8-A2 RNN BPTT
W1d/W2d).  This script measures its true relative error against a float64
(or, optionally, a correctly-rounded float32) reference.

Two modes
---------
* `--model` (default): reproduce the kernel *bit-exactly* in numpy float32.
  - `before` : a transcription of the P8 implementation (10-digit constants,
               Taylor truncated at 6 terms).
  - `after`  : the same float32 arithmetic as the Rust code currently in
               `kernels.rs` -- so the numbers printed here are the numbers the
               engine actually produces.  Useful for CI and for regression
               when someone edits kernels.rs again.
* `--dump FILE`: read a raw f32 dump of `exp_f(x_i)` produced by the Rust test
  binary (`vmlm-tests expdump <file>`), together with the `x_i` grid.  This is
  the authoritative check: it measures the *compiled* kernel, not a model.

Sampling
--------
The contract asks for x in [-20, 20] with >= 1e6 points.  A uniform grid can
*walk past* a local error peak, so we also run a fine scan around every
multiple of ln(2) inside the range (those are the range-reduction seams where
the error peaks) and report the worst point found over the union.

Usage
-----
    python tools/exp_accuracy.py                     # model, before vs after
    python tools/exp_accuracy.py --n 4000001
    python tools/exp_accuracy.py --model after --range -88 88
"""
import argparse
import numpy as np

# --------------------------------------------------------------------------
# float32 constants (exactly representable values written out in full)
# --------------------------------------------------------------------------
F32 = np.float32


def f32(v):
    return F32(v)


# log2(e) as the nearest f32: 1.44269502162933349609375
LOG2E_F32 = f32(1.4426950408889634)
# ln(2) as the nearest f32:  0.693147182464599609375
LN2_F32 = f32(0.6931471805599453)

# The 10-significant-digit literals the P8 code used.
LOG2E_OLD = f32(1.442_695_04)
LN2_OLD = f32(0.693_147_18)


# --------------------------------------------------------------------------
# BEFORE: transcription of the P8 kernel (kernels.rs, pre-P9)
# --------------------------------------------------------------------------
def exp_before(x):
    x = np.asarray(x, dtype=np.float32)
    xc = np.clip(x, f32(-88.0), f32(88.0))
    # (x * LOG2E_OLD + 0.5) as i32  -- Rust `as i32` truncates toward zero
    t = xc * LOG2E_OLD + f32(0.5)
    n = np.trunc(t).astype(np.int64)          # trunc == Rust `as i32` for finite
    n = np.clip(n, -127, 127)                 # keep the shift in range
    r = xc - n.astype(np.float32) * LN2_OLD
    r2 = r * r
    p = (f32(1.0) + r + r2 * f32(0.5)
         + r2 * r * f32(0.166_666_67)
         + r2 * r2 * f32(0.041_666_668)
         + r2 * r2 * r * f32(0.008_333_334)
         + r2 * r2 * r2 * f32(0.001_388_889))
    bits = ((n + 127).astype(np.uint32) << np.uint32(23))
    return p * bits.view(np.float32)


# --------------------------------------------------------------------------
# AFTER: mirrors the arithmetic of the P9 kernels.rs implementation.
# Keep in sync with kernels.rs.
# --------------------------------------------------------------------------
# Cody-Waite split of ln(2) in float32 (each part has few enough significant
# bits that n*LN2_HI etc. is exact).
LN2_HI = f32(0.693145751953125)      # 0x3F317200
LN2_LO = f32(1.42860677e-6)          # 0x35BFBE8E
INV_LN2 = LOG2E_F32

# Cephes expf minimax coefficients for r in [-ln2/2, ln2/2]
CEPHES = [f32(v) for v in (
    1.9875691500E-4,
    1.3981999507E-3,
    8.3334519073E-3,
    4.1665795894E-2,
    1.6666665459E-1,
    5.0000001201E-1,
)]


def exp_after(x):
    x = np.asarray(x, dtype=np.float32)
    xc = np.clip(x, f32(-88.722839), f32(88.722839))
    n = np.rint(xc * INV_LN2).astype(np.int64)
    n = np.clip(n, -126, 127)
    r = xc - n.astype(np.float32) * LN2_HI
    r = r - n.astype(np.float32) * LN2_LO
    # Horner, high -> low, the order the Cephes polynomial is written in
    y = CEPHES[0] * r + CEPHES[1]
    y = y * r + CEPHES[2]
    y = y * r + CEPHES[3]
    y = y * r + CEPHES[4]
    y = y * r + CEPHES[5]
    y = y * r + f32(1.0)
    y = y * r + f32(1.0)
    bits = ((n + 127).astype(np.uint32) << np.uint32(23))
    return y * bits.view(np.float32)


MODELS = {"before": exp_before, "after": exp_after}


# --------------------------------------------------------------------------
# reference & sweep
# --------------------------------------------------------------------------
def ref(x):
    """float64 reference (np.exp) -- good to ~1e-16 relative, far below f32.

    IMPORTANT: `x` is first rounded to float32, because that is what the kernel
    actually receives.  Measuring against exp(x_float64) while feeding the
    kernel float32(x) inflates the apparent error by the input-rounding term
    |exp(x) - exp(f32(x))| / exp(x) ~= |x - f32(x)|, which for |x| ~ 16 is
    ~9.4e-7 -- the same order as the target itself.  This bug produced a
    spurious 1.03e-6 "error" for a kernel that is actually at 0.8 ulp.
    """
    return np.exp(np.asarray(x, dtype=np.float32).astype(np.float64))


def grid(n, lo, hi, seam_points=0, seam_span=2e-3, seam_n=4001):
    """Uniform grid over [lo,hi] plus a dense comb around the ln(2) seams."""
    g = np.linspace(lo, hi, n, dtype=np.float64)
    if seam_points <= 0:
        return g
    # every k*ln2 inside [lo,hi] is a range-reduction seam
    k0 = int(np.floor(lo / np.log(2)))
    k1 = int(np.ceil(hi / np.log(2)))
    ks = np.arange(k0, k1 + 1)
    seams = ks * np.log(2)
    seams = seams[(seams >= lo) & (seams <= hi)]
    if seams.size == 0:
        return g
    # half-integers too: n = rint(x*log2e) flips there
    hs = (ks + 0.5) * np.log(2)
    hs = hs[(hs >= lo) & (hs <= hi)]
    centers = np.concatenate([seams, hs])
    off = np.linspace(-seam_span, seam_span, seam_n)
    comb = (centers[:, None] + off[None, :]).ravel()
    comb = comb[(comb >= lo) & (comb <= hi)]
    return np.concatenate([g, comb])


def sweep(name, x):
    # The kernel only ever sees float32 input; round once, up front, so that
    # both the model and the reference agree on the actual argument.
    x = np.asarray(x, dtype=np.float32)
    got = MODELS[name](x)
    r = ref(x)
    rel = np.abs(got.astype(np.float64) - r) / np.abs(r)
    rel = np.where(np.isfinite(rel), rel, 0.0)
    i = int(np.argmax(rel))
    return dict(name=name, n=x.size, max_rel=float(rel[i]), at=float(x[i]),
                got=float(got[i]), ref=float(r[i]), mean_rel=float(rel.mean()),
                rel=rel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2_000_001,
                    help="uniform grid points over the range (default 2e6+1)")
    ap.add_argument("--range", nargs=2, type=float, default=[-20.0, 20.0])
    ap.add_argument("--model", choices=["before", "after", "both"], default="both")
    ap.add_argument("--no-seams", action="store_true",
                    help="uniform grid only (no ln2 seam comb)")
    ap.add_argument("--dump", metavar="FILE",
                    help="raw f32 dump from `vmlm-tests expdump FILE`: "
                         "header int32 n, then n float32 x, then n float32 exp_f(x)")
    args = ap.parse_args()

    lo, hi = args.range
    print("=" * 78)
    print("VMLM P9/H  exp_f relative-error sweep")
    print("=" * 78)

    if args.dump:
        raw = np.fromfile(args.dump, dtype=np.uint8)
        n = int(np.frombuffer(raw[:4].tobytes(), dtype=np.int32)[0])
        xs = np.frombuffer(raw[4:4 + 4 * n].tobytes(), dtype=np.float32)
        ys = np.frombuffer(raw[4 + 4 * n:4 + 8 * n].tobytes(), dtype=np.float32)
        r = ref(xs)
        rel = np.abs(ys.astype(np.float64) - r) / np.abs(r)
        i = int(np.argmax(rel))
        print(f"source            : COMPILED kernel dump ({args.dump})")
        print(f"points            : {n}")
        print(f"x range           : [{xs.min():.6f}, {xs.max():.6f}]")
        print(f"max relative err  : {rel[i]:.6e}  @ x = {xs[i]:.9f}")
        print(f"  exp_f(x)        : {ys[i]:.9e}")
        print(f"  exp(x) f64      : {r[i]:.9e}")
        print(f"  ulp error       : {abs(float(ys[i])-r[i])/abs(r[i])/1.1920929e-7:.2f}")
        print(f"mean relative err : {rel.mean():.6e}")
        print(f"rms  relative err : {np.sqrt((rel**2).mean()):.6e}")
        worst = np.argpartition(-rel, 10)[:10]
        print("\ntop-10 offenders:")
        for j in sorted(worst.tolist(), key=lambda k: -rel[k]):
            print(f"  x={xs[j]:+.9f}  exp_f={ys[j]:.9e}  ref={r[j]:.9e}  rel={rel[j]:.3e}")

        # ---- normal vs subnormal split (see the note in the model path) ----
        SUB = -87.33654  # x below which exp(x) is subnormal in f32
        norm = xs >= SUB
        sub = ~norm
        print(f"\n  normal  result region x >= {SUB}: "
              f"{int(norm.sum())} pts, max rel = {rel[norm].max():.6e}")
        if sub.any():
            print(f"  subnormal result region x <  {SUB}: "
                  f"{int(sub.sum())} pts, max rel = {rel[sub].max():.6e}")
            print("    ^ limited by f32 subnormal representation (~1e-7 here),")
            print("      NOT by the kernel algorithm")
        ok = rel[norm].max() <= 1e-6
        print(f"\ntarget <= 1e-6 on the normal-result region : "
              f"{'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1

    x = grid(args.n, lo, hi,
             seam_points=0 if args.no_seams else 1)
    print(f"uniform grid      : {args.n} points on [{lo}, {hi}]")
    if not args.no_seams:
        print(f"plus seam comb    : +-2e-3 around every k*ln2 and (k+0.5)*ln2")
    print(f"total points      : {x.size}")

    names = ["before", "after"] if args.model == "both" else [args.model]
    res = [sweep(nm, x) for nm in names]

    print()
    hdr = f"{'model':<8} {'max rel err':>13} {'@ x':>16} {'mean rel':>13} {'rms rel':>13}"
    print(hdr)
    print("-" * len(hdr))
    for r in res:
        rms = float(np.sqrt((r['rel'] ** 2).mean()))
        print(f"{r['name']:<8} {r['max_rel']:>13.6e} {r['at']:>16.9f} "
              f"{r['mean_rel']:>13.6e} {rms:>13.6e}")

    if len(res) == 2:
        imp = res[0]["max_rel"] / res[1]["max_rel"]
        print(f"\nimprovement factor (max rel): {imp:.1f}x")
        print(f"target <= 1e-6 : "
              f"{'PASS' if res[1]['max_rel'] <= 1e-6 else 'FAIL'}")

    print("\ntop-10 offenders (worst model shown):")
    w = res[-1]
    rel = w["rel"]
    order = np.argsort(-rel)[:10]
    for j in order:
        print(f"  x={x[j]:+.9f}  {w['name']:<6} rel={rel[j]:.3e}")

    # ulp statistics are the honest way to state "how good is it really"
    print("\nulp interpretation (1 ulp = 1.192093e-7 relative):")
    for r in res:
        u = r["max_rel"] / 1.1920929e-7
        print(f"  {r['name']:<8} max = {u:.2f} ulp   mean = "
              f"{r['mean_rel']/1.1920929e-7:.4f} ulp")

    # ------------------------------------------------------------------
    # Subnormal caveat.
    #
    # exp(x) for x < ln(f32::MIN_POSITIVE) = -87.3365 is SUBNORMAL, where f32
    # carries fewer than 24 mantissa bits, so the achievable *relative* error
    # degrades no matter how good the algorithm is (at x=-88 only ~10 bits
    # remain -> ~6e-7 relative is the representational floor).
    # A full-range sweep therefore reports ~6.6e-7, which is NOT an algorithm
    # defect.  We report the normal-range figure separately so the two are
    # never conflated.
    # ------------------------------------------------------------------
    if lo < -87.3365:
        nlo = -87.3365
        xn = grid(args.n, nlo, hi, seam_points=0 if args.no_seams else 1)
        print(f"\nnormal-output sub-range [{nlo:.4f}, {hi}] "
              f"(exp >= f32 MIN_POSITIVE, {xn.size} pts):")
        for r in res:
            rr = sweep(r["name"], xn)
            print(f"  {r['name']:<8} max rel = {rr['max_rel']:.6e} "
                  f"({rr['max_rel']/1.1920929e-7:.2f} ulp) @ x = {rr['at']:.9f}   "
                  f"mean = {rr['mean_rel']:.4e}")
        print("  ^ above this range the result is subnormal: the residual")
        print("    ~6.6e-7 seen near x=-88 is the f32 representation floor,")
        print("    not the kernel.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
