# PATH: B1
#
# CURRENT speedup vs both baselines. All inputs freshly measured this run.
#
# ⚠️ UNIT BUG FOUND AND FIXED IN THIS SCRIPT'S FIRST VERSION
#   The first version computed "B1 per matrix = 89.1 / 5 = 17.82 ms" and then
#   divided by 30720 MACs to get "580 ns/MAC". That is 3000x off, and it
#   contradicted the directly measured 0.189 ns/conn right next to it.
#
#   Cause: 89.1 ms/step is the time for ALL 24 LAYERS (120 matrices), not for
#   the 5 matvecs of ONE layer. The probe builds calls = 5 * 24 * 128 and
#   multiplies a per-CALL time by it. So the correct divisor is 5*24 = 120
#   matrices, not 5.
#
#   Consistency check that catches it: 0.0058 ms/call * 15360 calls = 89.1.
#   And 0.189 ns/conn * 30720 conns = 5.8e-3 ms = the same 0.0058 ms/call.
#   Both agree; only my per-matrix line was wrong.
#
# MEASURED THIS ROUND (vmlm-b1-prod-eps, vmlm-bench-baseline)
#   B1 shipped (SIMD reduce + u32 mirror)   0.0058 ms/call -> 89.1 ms/step
#   B1 scalar reference (bit-exact)         0.0380 ms/call -> 583.7 ms/step
#   same-structure dense d=1536 (4 proj)                   -> 872.4 ms/step
#   same-parameter dense d=176 (4 proj)                    ->  10.1 ms/step

MATVECS_PER_LAYER = 5
PROJ_PER_LAYER_DENSE = 4     # qkv(3) + wo(1)
LAYERS = 24
TOKENS = 128

b1_ms = 89.1
b1_scalar_ms = 583.7
dense_struct_ms = 872.4
dense_param_ms = 10.1

# per-matrix (one projection/matvec, one layer)
b1_per_matrix = b1_ms / (MATVECS_PER_LAYER * LAYERS)
ds_per_matrix = dense_struct_ms / (PROJ_PER_LAYER_DENSE * LAYERS)
dp_per_matrix = dense_param_ms / (PROJ_PER_LAYER_DENSE * LAYERS)

print("=" * 72)
print("PER-MATRIX, ONE LAYER (like-for-like: one matvec vs one dense projection)")
print("=" * 72)
print("  B1 (sparse, d=1536 k0=20)      %9.4f ms per matrix" % b1_per_matrix)
print("  same-STRUCTURE dense (d=1536)  %9.4f ms per matrix" % ds_per_matrix)
print("  same-PARAMETER dense (d=176)   %9.4f ms per matrix" % dp_per_matrix)
print()
print("  => vs same-STRUCTURE : %.2fx FASTER" % (ds_per_matrix / b1_per_matrix))
g = b1_per_matrix / dp_per_matrix
print("  => vs same-PARAMETER : %.3fx   (B1 is %.1fx SLOWER)" % (dp_per_matrix / b1_per_matrix, g))
print()

print("=" * 72)
print("FULL-STEP TOTALS (coverage differs: B1 5 matvecs vs dense 4 projections)")
print("=" * 72)
print("  B1 shipped         %8.1f ms/step   (5 matvec x 24L)" % b1_ms)
print("  B1 scalar ref      %8.1f ms/step   (bit-exact contract version)" % b1_scalar_ms)
print("  same-structure     %8.1f ms/step   (4 proj x 24L)" % dense_struct_ms)
print("  same-parameter     %8.1f ms/step   (4 proj x 24L)" % dense_param_ms)
print()
print("  B1 vs same-structure (raw) = %.2fx" % (dense_struct_ms / b1_ms))
print("  B1 vs same-parameter (raw) = %.3fx" % (dense_param_ms / b1_ms))
print()

print("=" * 72)
print("THE OPTIMISATION'S OWN SPEEDUP (this session)")
print("=" * 72)
print("  B1 scalar reference -> B1 shipped : %.2fx  (%.1f -> %.1f ms/step)"
      % (b1_scalar_ms / b1_ms, b1_scalar_ms, b1_ms))
print("  per-call: 0.0380 -> 0.0058 ms; per-connection: 1.237 -> 0.189 ns")
print()

print("=" * 72)
print("SANITY: the three ways of expressing B1 must agree")
print("=" * 72)
per_call = b1_ms / (MATVECS_PER_LAYER * LAYERS * TOKENS)
conns = 1536 * 20
print("  from ms/step : %.6f ms/call" % per_call)
print("  from ns/conn : %.6f ms/call  (0.189 ns * %d conns)" % (0.189 * conns / 1e6, conns))
print("  -> agree = %s" % (abs(per_call - 0.189 * conns / 1e6) < 1e-4))
print()

print("=" * 72)
print("WHY same-PARAMETER IS A LOSS (structural, not tuning)")
print("=" * 72)
d, k0, dm = 1536, 20, 176
print("  B1 MACs/matrix/token = d*k0 = %d" % (d * k0))
print("  d=176 dense MACs     = dm^2 = %d" % (dm * dm))
print("  ratio = %.2fx  => the SAME work per matrix, so this is a fair fight"
      % ((d * k0) / (dm * dm)))
print()
print("  per-MAC: B1 %.4f ns/MAC vs dense AVX2 %.4f ns/MAC  => dense is %.1fx more efficient"
      % (b1_per_matrix * 1e6 / (d * k0 * TOKENS), dp_per_matrix * 1e6 / (dm * dm * TOKENS),
         (b1_per_matrix / (d * k0)) / (dp_per_matrix / (dm * dm))))
print("  (units: one LAYER does d*k0 MACs per TOKEN, and a layer has %d tokens," % TOKENS)
print("   so MACs per matrix = %d * %d = %d. Dividing by d*k0 alone is a 128x error.)"
      % (d * k0, TOKENS, d * k0 * TOKENS))
print("  check: 0.189 ns/conn measured directly == %.4f ns/MAC computed here"
      % (b1_per_matrix * 1e6 / (d * k0 * TOKENS)))
print()
print("  the dense GEMM reuses each loaded operand across the whole N dimension;")
print("  B1 uses each of its 30720 connections exactly once. No reuse = no amortisation.")
