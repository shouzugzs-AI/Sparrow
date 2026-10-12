# PATH: B1
# Final position of packed B1 against the stated goal.
#
# UNIT DISCIPLINE (this file had a 1000x error on its first run):
#   A4 QPC reports ms/op. To get ms/step: ms_per_op * calls.
#   calls = 5 matvec * 24 layers * 128 positions = 15360.
#   The first version multiplied by 1000 as well, reporting "557568 ms/step"
#   -- a number that is obviously absurd (9 minutes per step) yet formatted
#   plausibly. Caught by sanity-checking against the measured 219 s/step
#   end-to-end figure, not by re-reading the code.
calls = 5 * 24 * 128

ms_per_op_before = 0.0363
ms_per_op_after = 0.0144
before = ms_per_op_before * calls
after = ms_per_op_after * calls

print("=== B1 forward: BEFORE vs AFTER packing (5 matvec x 24 layers x 128 pos) ===")
print("  calls per step    = 5 * 24 * 128 = %d" % calls)
print("  BEFORE (scalar)   = %8.1f ms/step   (%.4f ms/op, 1.204 ns/conn)"
      % (before, ms_per_op_before))
print("  AFTER  (packed)   = %8.1f ms/step   (%.4f ms/op, 0.469 ns/conn)"
      % (after, ms_per_op_after))
print("  gain              = %8.2fx" % (before / after))
print()

d = 1536
k0 = 20
dm = 176
print("=== GOAL CHECK: faster than same-parameter dense? ===")
dp = 11.5
print("  same-parameter dense (d=176) = %8.1f ms/step" % dp)
print("  B1 packed                    = %8.1f ms/step" % after)
print("  => B1 is %.1fx SLOWER  -> GOAL NOT MET" % (after / dp))
print()

print("=== the comparison IS fair (same MAC count) ===")
print("  B1 MACs/matrix/token = d*k0 = %d" % (d * k0))
print("  d=176 dense          = dm^2 = %d" % (dm * dm))
print("  ratio = %.2fx  -> same work, times are directly comparable" % ((d * k0) / (dm * dm)))
print()
print("  per-MAC: B1 %.3f ns/MAC  vs  dense AVX2 %.4f ns/MAC  =  %.1fx slower"
      % (0.469, 0.0286, 0.469 / 0.0286))
print()

print("=== what IS achieved ===")
print("  B1 packed vs same-STRUCTURE dense (d=1536, 829.1 ms/step) = %.2fx FASTER"
      % (829.1 / after))
print()

print("=== where the remaining time goes (measured, vmlm-b1-reconcile) ===")
print("  products (vectorised gather)   0.303 ns/conn")
print("  scalar reduction (contract)    0.234 ns/conn   <- required by bit-exact")
print("  buffer residency               0.127 ns/conn   <- reduced by sharing")
print("  total shipped                  0.469 ns/conn")
print()
print("  The scalar reduction is ~44% and is NOT removable: the bit-exact")
print("  contract forbids reordering the sum (S2 with 8-way SIMD reduction")
print("  forks 841/1536 outputs).")
