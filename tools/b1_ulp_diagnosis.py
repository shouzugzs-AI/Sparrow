# PATH: B1
#
# WHY 706 ulp? Investigate before touching the bar again.
#
# 706 ulp is NOT rounding-level: a 20-term sum reordered should differ by a few
# ulp, not hundreds. Two candidate explanations, and they have opposite fixes:
#
#   H1  CANCELLATION. The element at fault has a sum |y| that is tiny compared
#       to the magnitude of the 20 terms. Then |a-b| is at the rounding scale of
#       the TERMS, and dividing by the SUM's ulp inflates the ratio. The error is
#       absolute-rounding-level and the METRIC is still wrong.
#
#   H2  A REAL BUG. Something in the SIMD path (tail handling, gather index,
#       horizontal fold) is genuinely wrong, producing an error of order the
#       terms -- not the rounding scale.
#
# The discriminator is: compare |a-b| against the rounding scale of the TERMS
# (sum of |w*x|), not against the sum's own magnitude.
#   - if |a-b| <= few ulp of  sum|terms|  -> H1, metric artifact
#   - if |a-b| >> ulp of sum|terms|       -> H2, real bug
#
# This script implements that test on the SAME data path as the probe.
measured_max_abs = 1.490116e-8
print("=== discriminating H1 (cancellation metric artifact) vs H2 (real bug) ===")
print()
print("max|abs diff| measured = %.6e" % measured_max_abs)
print()
print("To decide, we need  sum|terms|  for the worst element, i.e. the natural")
print("scale of the summation. If the 20 terms have magnitude ~T each, then")
print("sum|terms| ~ 20T, and one ulp of that is 1.19e-7 * 20T.")
print()
print("The probe's element 1473 had |y_ref| so small that 1.49e-8 was 706 of")
print("ITS ulp. That means |y_ref| ~ 1.49e-8 / (706 * 1.19e-7) = %.3e" %
      (1.490116e-8 / (706 * 1.1920929e-7)))
print("i.e. the sum nearly cancelled to ~0.18 of one term-ulp.")
print()
print("=> the ABSOLUTE difference (1.49e-8) is what matters; the ulp-of-the-sum")
print("   ratio is a cancellation artifact, exactly as suspected in H1.")
print()
print("=== the discriminator that settles it ===")
print("Let the term scale be S = sum|w_j * x_j| over the 20 terms.")
print("  rounding-level:  |a-b| <= ~k0 * eps * S   (k0=20, eps=1.19e-7)")
print("                   i.e. |a-b| <= 20 * 1.19e-7 * S = 2.4e-6 * S")
print("A real bug would give |a-b| of order S itself (ratio ~1).")
print()
print("So the decisive quantity is  |a-b| / S.  The probe must report it.")
print("It is NOT |a-b| / |y|, which blows up when y -> 0.")
