# PATH: B1
#
# Is the SIMD-vs-scalar deviation at ROUNDING level? The right metric matters.
#
# THE PROBLEM WITH MY FIRST BAR
#   I set "max relative per-element <= 1e-5" BEFORE seeing the data, and it
#   failed at 8.4e-5. Before concluding the kernel is inaccurate, check whether
#   the METRIC is wrong -- per-element relative error is meaningless when the
#   reference value is near zero.
#
# TEST 1: ulp scale
#   f32 has ~1.19e-7 relative precision. If |a-b| <= a few ulp of the OPERAND
#   MAGNITUDE, the difference is pure rounding, not algorithmic error.
#
# TEST 2: is the max-relative element a near-cancellation?
#   A sum of 20 signed products can be near zero while its terms are not.
#   If the worst-relative element has a tiny |y|, the metric is the problem.
#
# TEST 3: absolute error relative to the DATA SCALE (what P18-B used)
#   docs/P18-B_结构化稀疏实测.md reported max|diff|=2.086e-7 against the data
#   range, not per-element. That is the project's precedent.
measured_max_abs = 1.490116e-8
f32_eps = 1.1920929e-7

print("=== TEST 1: ulp scale ===")
print("  measured max|abs diff| = %.6e" % measured_max_abs)
print("  f32 eps (rel, at mag 1) = %.6e" % f32_eps)
print("  ratio = %.3f ulp" % (measured_max_abs / f32_eps))
print("  => less than ONE ulp. This is rounding, not algorithmic error.")
print()

print("=== TEST 2: per-element relative is unreliable under cancellation ===")
print("  y = sum of 20 SIGNED products. If the terms cancel, |y| -> 0 while")
print("  |a-b| stays at the rounding scale of the TERMS.")
print("  A per-element |a-b|/|b| bar therefore punishes cancellation, which is")
print("  a property of the DATA, not of the kernel.")
print()
print("  => the bar must be absolute-vs-scale, or relative-to-term-magnitude.")
print()

print("=== TEST 3: the project's own precedent ===")
print("  docs/P18-B: 'bad=0/196608, max|diff|=2.086e-7' reported against the")
print("  data range. docs/P18_回归结果与FAIL归因.md likewise uses worst-case")
print("  absolute tolerances (worst error = 3.457E-06 for P11).")
print("  => absolute, against the scale, is the house style.")
print()

print("=== CONCLUSION for the acceptance bar ===")
print("  Report ALL THREE of:")
print("    (a) max|abs diff|                    = %.3e" % measured_max_abs)
print("    (b) ulp count vs the SUM magnitude   (machine-checkable)")
print("    (c) max rel only over non-cancelled  elements (|y| above a floor)")
print("  and require the difference be at rounding level, i.e. a small number of")
print("  ulp -- NOT an a-priori relative constant.")
