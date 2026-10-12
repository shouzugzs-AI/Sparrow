# P18 profiling probe (design note)

Purpose: find where the 215 s/step at batch=4 seq=32 actually goes, BEFORE
optimising. Prior lessons in this repo: "measure, don't estimate" and the
215 s/step figure in `sparse_train24.rs` has no breakdown attached.

Two candidates, both quantified on paper first:

## A. Dense attention/MLP GEMMs  O(L * T * d^2)
Per layer, per token: q,k,v projections = 3*d^2 MACs; wo = d^2; sparse FFN is
gated by k0. With L=24, T=B*S=128, d=1536:
    4 * d^2 * T * L = 4 * 2359296 * 128 * 24 = 2.90e10 MACs
Attention score/AV: L * T^2/2 * d * 2 = 24 * 8192 * 1536 * 2 = 6.0e8  (negligible)

So the q/k/v/wo projections ARE the dominant term, not the sparse matvec.

## B. Sparse connection path  O(L * n_neurons * k0)
    24 * 1536 * 20 = 737k MACs per token -> x128 = 9.4e7  (negligible)

=> The task brief says "vectorise OP_SPARSE_MATVEC's inner loop first", but the
   arithmetic says the sparse matvec is ~300x smaller than the dense
   projections. Vectorising only the sparse ops CANNOT reach <5 s/step.

This note records that so the plan is not built on an unmeasured assumption.
The probe below measures the real split.
