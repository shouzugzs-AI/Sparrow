# PATH: B1
#
# Why "reach same-parameter dense" is not a B1 tuning target, with the arithmetic.
d = 1536
k0 = 20
T = 128
L = 24

print("=== B1 vs same-parameter dense: the WORK differs, so the times cannot match ===")
print()
print("B1 (d=1536, k0=20) per matrix per token : d*k0   = %d MACs" % (d * k0))
dm = 176  # same-parameter-count dense: sqrt(d*k0) = 175.3
print("same-param dense (d_model=176)          : dm*dm  = %d MACs" % (dm * dm))
print("  => B1 performs %.2fx MORE MACs than the same-parameter dense model" % ((d * k0) / (dm * dm)))
print()
print("So 'B1 packed should approach same-parameter dense' asks a %dx-bigger-work" % ((d * k0) / (dm * dm)))
print("kernel to run as fast as a smaller one. That is not a packing question.")
print()
print("=== the correct same-WORK comparison (same MAC count) ===")
print("  B1 packed measures 0.469 ns/connection (30720 conns per matrix).")
print()
print("  same-structure dense (d=1536) measured 829.1 ms/step for 4 proj x 24 layers x 128 tokens")
tokens = 4 * 24 * 128
macs_dense = d * d
ns_per_mac_dense = 829.1 * 1e6 / (tokens * macs_dense)
print("    -> %.4f ns/MAC" % ns_per_mac_dense)
print("  B1 packed = %.4f ns/MAC (per MAC: w*x is one MAC)" % 0.469)
print()
print("  => B1 packed is %.1fx slower PER MAC than the dense AVX2 GEMM." % (0.469 / ns_per_mac_dense))
print()
print("=== why: dense GEMM amortises, B1 cannot ===")
print("  * dense gemm_nt reuses each loaded A-row across the whole N dimension")
print("    (packed MR=8 kernel), so its loads are amortised ~N times.")
print("  * B1 loads one weight and one gathered x per MAC, with NO reuse at all:")
print("    each of the 30720 connections is used exactly once per matrix.")
print("  => B1 is load-bound by construction. Its ceiling is set by load throughput,")
print("     not by instruction count, which is why vectorising products gave only 2.5x.")
print()
print("=== what IS achievable and measured ===")
print("  B1 scalar  (before) = 557.6 ms/step   (1.204 ns/conn)")
print("  B1 packed  (after)  = 221.2 ms/step   (0.469 ns/conn)   -> 2.52x")
print("  B1 packed vs same-STRUCTURE dense (829.1) = 3.75x FASTER  <- the honest win")
