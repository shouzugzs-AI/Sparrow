# PATH: B1
#
# The b1_canfail crash: 0xC0000005 at wrong-5.
#
# THE CRASH
#   b1_canfail's wrong-5 deliberately constructs a SparseConnW with k0 = k0+1,
#   i.e. a wrong row pitch, to check the criterion catches it. That was safe for
#   the OLD scalar b1_forward: it read src[base + j] as f32 one element at a
#   time, and the only overrun was one f32 past the last row -- still inside the
#   allocation's slack.
#
#   The NEW kernel reads src_u32 in 8-wide vectors via _mm256_loadu_si256.
#   For the LAST row, base + k0.. reads 8 u32 = 32 bytes past the mirror's end.
#   src_u32 sits at the END of the allocation (w, src, act, src_u32), so that
#   vector load runs off the end of the buffer -> access violation.
#
# IS THIS A BUG IN THE KERNEL, OR IN THE PROBE?
#   Both, and the distinction matters:
#   * The PROBE is passing a struct whose k0 contradicts its own buffer size.
#     That is an invalid-SparseConnW, and the kernel's safety contract says the
#     row must be in bounds. So the probe is violating a documented precondition.
#   * But the OLD code survived it and the NEW code does not. A probe that used
#     to work now crashes -- that is a real behavioural regression in the
#     robustness envelope, and this project's whole discipline is that illegal
#     input must produce a READABLE ERROR, not 0xC0000005 with no diagnostic
#     (see build_all.ps1's notes on the guard-page class of failures).
#
# THE FIX (both sides)
#   1. KERNEL/PRODUCTION: add an explicit bound guard in b1_forward that
#      validates k0 against the allocated geometry and exits with a message,
#      rather than letting a vector load run off the end.
#      We cannot see the allocation size from SparseConnW (it has only pointers),
#      so the guard is: the struct must carry a consistent (e, k0, n_neurons).
#      Add a `assert_geometry()` that hard-fails on inconsistency.
#   2. PROBE: wrong-5 must exercise the wrong-pitch idea WITHOUT violating the
#      kernel's precondition -- e.g. build a genuinely k0+1-wide struct with a
#      matching larger buffer. That tests the pitch logic instead of testing
#      undefined behaviour.
print(__doc__)

n, k0 = 1536, 20
e = n * k0
print()
print("geometry: n_neurons=%d k0=%d e=%d" % (n, k0, e))
print("mirror bytes = e*4 = %d" % (e * 4))
print("8-wide load at the last row's end reads +32 bytes past e*4")
print("=> out of the src_u32 slice; src_u32 is the LAST field, so it exits the alloc")
print()
print("SparseConnW carries: e, k0, n_neurons (all derivable)")
print("  consistency: e == n_neurons * k0")
print("so an inconsistent struct IS detectable without the allocation size.")
