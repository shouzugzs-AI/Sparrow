"""Is ~7e-3 the FUNDAMENTAL f32 limit for reducing x~1e5, or is my method bad?

Test: even with a PERFECT representation of n*(pi/2), the subtraction
x - n*(pi/2) is a catastrophic cancellation when both are ~1e5.
Measure how much precision that single subtraction can possibly keep.

If the answer is ~1e-3..1e-2, then NO range-reduction scheme can do better
in f32, and the correct engineering answer is NOT "write Payne-Hanek" but
"keep the RoPE phase small" -- i.e. compute the angle so that x stays small.
"""
import numpy as np
from decimal import Decimal, getcontext
getcontext().prec = 80
pi = Decimal('3.14159265358979323846264338327950288419716939937510582097494459230781640628620899')
half = pi / 2


def f32(x):
    return float(np.float32(x))


print("Cancellation analysis: x - n*(pi/2) in f32")
print("%10s %12s %14s %14s %12s" % ("x", "n", "f32(x)", "f32(n*half)", "gap"))
for xv in (100.0, 1000.0, 10000.0, 100000.0, 1000000.0):
    n = int(round(xv * 2 / float(pi)))
    a = f32(xv)
    b = f32(float(n) * float(half))
    gap = abs(a - b)
    # the ulp of the larger operand bounds the accuracy of the difference
    ulp = float(np.spacing(np.float32(max(abs(a), abs(b)))))
    print("%10.1f %12d %14.9f %14.9f %12.3e" % (xv, n, a, b, ulp))

print()
print("=> When x ~ 1e5, ulp(x) ~ 7.8e-3. The reduced remainder r is O(1).")
print("   So r can only be accurate to ~ulp(x) = 7.8e-3  --  REGARDLESS of")
print("   how good the n*(pi/2) product is.")
print()
print("CONCLUSION: ~7e-3 IS the f32 floor for |x| ~ 1e5.")
print("            Any scheme that first forms a big angle then reduces will")
print("            hit it. The fix is to NOT form the big angle.")
print()

# ---- the actual fix for RoPE: reduce the POSITION, not the angle ----
# angle = pos * inv_freq. Instead of computing pos*inv_freq (big) and then
# reducing, note we want cos/sin of it. We can reduce pos modulo the period
# of cos/sin for that component:
#     period_i = 2*pi / inv_freq_i   (in positions)
#     pos_i    = pos mod period_i
# then angle = pos_i * inv_freq_i is in [0, 2*pi) -- SMALL.
#
# pos is an INTEGER and can be reduced EXACTLY. That is the whole trick.
print("FIX: reduce pos modulo the component's period, using EXACT integer mod.")
print()
theta = 5e6
hd = 128
print("%4s %14s %16s %16s %14s" % ("i", "inv_freq", "period(pos)", "pos_red(131072)", "angle_after"))
for i in (0, 1, 2, 4, 8, 16, 32, 63):
    inv = theta ** (-2.0 * i / hd)
    period = 2 * float(pi) / inv
    pos = 131072
    if period < 2e9:
        pr = pos % period
    else:
        pr = float(pos)
    print("%4d %14.6e %16.4f %16.4f %14.6f" % (i, inv, period, pr, pr * inv))

print()
print("For i=0: period = 2*pi = 6.2832 positions, so pos mod period is exact")
print("and tiny. The resulting angle is < 2*pi, where our reduction is")
print("accurate to ~1e-7. That removes the cancellation entirely.")
print()
print("Cost: one f32 division/mod per (row, i) -- or precompute 1/period.")
print("Implementation detail: pos is an int, so `pos % period` in f32 is")
print("still a big-number op. Better: keep the modulo in INTEGER space by")
print("using a rational approx of period, OR simply compute")
print("    ang = (pos * inv) - 2*pi*floor(pos*inv/(2*pi))")
print("which reintroduces the cancellation... so the INTEGER route is the")
print("right one: precompute per-i integer period in units of 1/2^k.")
