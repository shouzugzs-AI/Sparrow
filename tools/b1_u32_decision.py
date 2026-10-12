# PATH: B1
#
# Decision note: how to get u32 indices into the B1 SIMD kernel WITHOUT changing
# SparseConnW.src from *mut f32.
#
# THE TENSION
#   Measured: u32 indices give 1.43x on the shipped shape, bit-exact
#   (vmlm-b1-shipped-u32). But `SparseConnW.src` is `*mut f32` and is read at
#   92 call sites across core/, train/ (trainers, verifiers, probes).
#   Changing the field type rewrites all of them -- a large, risky edit whose
#   failure mode is silent index corruption ("plausible garbage numbers"),
#   which is the exact class this project keeps fighting.
#
# THE OPTIONS
#   A. Change SparseConnW.src to *mut u32.
#      + cleanest end state, removes the convert everywhere
#      - 92 call sites; every `as usize` becomes wrong; probes break
#      - the disk format and engine are unaffected (they are already u32)
#
#   B. Keep src as f32, ADD a parallel `src_u32` mirror maintained at init.
#      + zero blast radius: nothing existing changes
#      + the SIMD kernel gets u32 with no convert
#      - 4 bytes x E extra memory (30720 * 4 = 120 KB per matrix, 28.8 MB over
#        120 matrices) -- but this is a TRAINING-SIDE workspace, not the artifact
#      - the two copies can diverge => needs a consistency assertion
#
#   C. Convert f32->u32 ONCE into a reusable scratch buffer per forward call.
#      + no struct change, no extra persistent memory
#      - the convert still runs, but only E times per call instead of E
#        per-connection... which is the SAME count. No saving.
#      - actually worse: it adds a pass.
#
#   D. Do nothing to src; only add the SIMD reduction (no u32).
#      + smallest change
#      - leaves the single biggest measured win (1.43x) on the table
#
# DECISION: B, with an explicit divergence check.
#   Rationale: the measured win is real and bit-exact, but option A's blast
#   radius (92 sites, silent-corruption failure mode) is not justified by a
#   1.43x on a training-side workspace. Option B gets the win, keeps every
#   existing reader working unchanged, and the divergence risk is closed by a
#   cheap assertion in the init path and by a verification criterion.
#
# WHAT MUST BE VERIFIED FOR B TO BE SAFE
#   1. src_u32[i] == (src[i] as u32) for all i, asserted after init.
#   2. The SIMD/u32 path and the f32 scalar path agree within epsilon.
#   3. src is never mutated without recomputing src_u32 (checked: src is written
#      only in init_sparse_conn and in test probes; production init is the only
#      writer on the training path).
print(__doc__)

E = 30720
print("memory cost of the u32 mirror: E*4 = %d bytes per matrix" % (E * 4))
print("  over 120 matrices (24 layers x 5) = %.1f MiB (training workspace only)" % (120 * E * 4 / 1024 / 1024))
print("  the DISK artifact is unchanged: it already stores src as u32")
