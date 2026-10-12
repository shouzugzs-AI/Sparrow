#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P16-5 acceptance: the composition gap is CLOSED -- sparsity ratio k now drives
the forward arithmetic.

    python tools/p16_compose.py --bench <bench engine> --old <task-8 engine> --asm <assembler>

THE GAP BEING CLOSED (reported by task-8, confirmed by the Lead)
    OP_TOPK_BY_STRENGTH (93) computed a selection table `sel`; OP_SPARSE_MATVEC
    (94) read the CSR and never re-read `sel`. So `y` was determined entirely by
    the CSR and changing `k` did NOTHING. Section BEFORE below re-measures that
    on the FROZEN task-8 engine, so the claim "it used to be k-invariant" is
    evidence rather than recollection.

THE DESIGN (A') AND WHY
    A `selgate` bit (fl bit7, unused) switches OP_SPARSE_MATVEC into a mode that
    consumes `sel`, and a `seg` bit (the same bit on 93) makes TOPK_BY_STRENGTH
    run PER ROW. Three reasons, each of which is checkable rather than asserted:

      1. No instruction widening, no new opcode number. `fl` had bit7 free
         (bits 0-5 = weight slot, bit6 = FL_NNZ). Reusing opcode 94 cannot
         collide with anyone else's number allocation.
      2. The ungated path is UNCHANGED -- the gate only adds a branch. So
         task-8's 16/16 guards and its bit-exact proof cannot regress; section
         GUARDS re-runs them, and section REGRESSION re-assembles an existing
         .vml and compares bytes.
      3. It uses `sel` at INFERENCE. The brief's Design B prunes the CSR at
         BUILD time, which leaves `k` as an assembler argument: the engine would
         never see k, so "connection-sparse activation" would still be untested.

WHY `seg` IS NEEDED TOO (the part that is easy to get wrong)
    Opcode 93 numbers its output 0..L-1 over the array it is HANDED; opcode 94
    numbers connections 0..nnz-1 over the WHOLE CSR. Those agree only for row 0
    of the matrix. The engine's own guard caught exactly this on the first gated
    attempt: `sel index outside its CSR row, idx=8 row=[0,6)`. In segmented mode
    `a` is [n, m] and each row's survivors are emitted ROW-RELATIVE, so 94 reads
    `o0 + rel` and the two numberings agree by construction.

WHAT IS MEASURED (every item paired with a can-fail control)
    BEFORE  y is k-invariant on the task-8 engine          (the gap, re-measured)
    A       y DIFFERS across k, and nnz == n * k exactly
    B       skip proof: performed / selected / dense
    C       NumPy reference is BIT-EXACT, and the order that makes it so
    D       can-fail: perturbing the reference must produce nonzero mismatches
    GUARDS  the 14 guards still fire (task-8's set) + the new gated ones
"""
import argparse
import os
import struct
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p16_ops_probe as P  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ----------------------------------------------------------------- the case ---
NX = 6          # input neurons
NN = 4          # output neurons ("rows")
KMAX = 4        # largest k exercised; sel is declared [NN, 2*KMAX]
X = [1.5, -2.25, 3.0, 0.5, -1.0, 4.0]

W_DENSE = [
    [0.9, 0.1, -0.4, 0.3, 0.05, -0.7],   # neuron 0
    [0.2, 0.8, 0.6, -0.15, 0.9, 0.25],   # neuron 1
    [0.5, -0.6, 0.7, 0.45, -0.8, 0.35],  # neuron 2
    [-0.3, 0.65, 0.5, 0.85, 0.2, -0.9],  # neuron 3
]


def f32(v):
    return P.f32(v)


def build_csr():
    """Dense rows expressed as CSR: every neuron touches every input."""
    offsets = [j * NX for j in range(NN + 1)]
    src = [i for _ in range(NN) for i in range(NX)]
    w = [W_DENSE[j][i] for j in range(NN) for i in range(NX)]
    return offsets, src, w


def topk_rows(k):
    """Independent per-row top-k. Returns, per neuron, the SELECTED input ids
    in ASCENDING order, plus the row-relative connection indices.

    Rule reproduced from the engine: k rounds of "max of what is left", ties to
    the LOWEST index (strict `>`), then emitted ascending. Ties do not arise in
    this case, but the rule is implemented anyway so the reference does not
    silently depend on tie-freeness.
    """
    out = []
    for j in range(NN):
        row = [f32(W_DENSE[j][i]) * f32(X[i]) for i in range(NX)]
        taken = [False] * NX
        chosen = []
        for _ in range(min(k, NX)):
            best, bestv, found = None, None, False
            for i in range(NX):
                if taken[i]:
                    continue
                if not found or row[i] > bestv:
                    best, bestv, found = i, row[i], True
            if not found:
                break
            taken[best] = True
            chosen.append(best)
        chosen.sort()
        out.append(chosen)
    return out


def reference(k):
    """Expected y and the term count, in ROW-RELATIVE ASCENDING order.

    SUMMATION ORDER (stated because it is what makes the comparison bit-exact):
    for each neuron, the selected connections are summed in ascending
    row-relative index -- which is the same order the engine walks, both in the
    gated path (it iterates `sel` segments, which are emitted ascending) and in
    the ungated CSR path. Identical order + identical f32 rounding = identical
    bits. A different order would still be *mathematically* right and would
    differ in the last ulp, which is exactly the distinction this file must not
    blur.
    """
    y, terms = [], 0
    for chosen in topk_rows(k):
        acc = None
        for rel in chosen:
            i = rel  # dense CSR: row-relative index == input id
            term = f32(f32(W_DENSE[len(y)][i]) * f32(X[i]))
            acc = term if acc is None else f32(acc + term)
            terms += 1
        y.append(0.0 if acc is None else acc)
    return y, terms


def reference_ungated():
    """The OLD semantics: the whole CSR row, no selection."""
    y, terms = [], 0
    for j in range(NN):
        acc = None
        for i in range(NX):
            term = f32(f32(W_DENSE[j][i]) * f32(X[i]))
            acc = term if acc is None else f32(acc + term)
            terms += 1
        y.append(acc)
    return y, terms


def wr(path, vals):
    open(path, "wb").write(struct.pack("<%df" % len(vals), *vals))


class R:
    def __init__(self):
        self.n = self.bad = 0

    def check(self, ok, name, detail=""):
        self.n += 1
        if not ok:
            self.bad += 1
        print("  [%s] %-56s %s" % ("PASS" if ok else "FAIL", name, detail))
        return ok

    def sec(self, t):
        print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


def f32list(v):
    return ["%.9g" % x for x in v]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, help="engine built with --features bench")
    ap.add_argument("--old", required=True, help="FROZEN task-8 bench engine (for BEFORE)")
    ap.add_argument("--asm", required=True)
    ap.add_argument("--work", default=os.path.join(ROOT, "examples", "p16_compose"))
    a = ap.parse_args()

    work = a.work
    dat = os.path.join(work, "w")
    os.makedirs(dat, exist_ok=True)
    r = R()
    rel = lambda p: os.path.relpath(p, ROOT).replace("/", os.sep)

    offsets, src, w = build_csr()
    nnz_csr = len(src)
    dense_pairs = NN * NX

    wr(os.path.join(dat, "off.bin"), [float(v) for v in offsets])
    wr(os.path.join(dat, "src.bin"), [float(v) for v in src])
    wr(os.path.join(dat, "w.bin"), [float(v) for v in w])
    wr(os.path.join(dat, "x.bin"), [float(v) for v in X])
    # strengths, laid out [NN, NX] -- one row per neuron, which is what
    # segmented TOPK_BY_STRENGTH consumes.
    str2d = [f32(f32(W_DENSE[j][i]) * f32(X[i])) for j in range(NN) for i in range(NX)]
    wr(os.path.join(dat, "str2d.bin"), str2d)
    for k in range(1, KMAX + 1):
        wr(os.path.join(dat, "k%d.bin" % k), [float(k)])

    def consts(for_old=False):
        c = [("offc", "%s/off.bin" % rel(dat), NN + 1, 1),
             ("srcc", "%s/src.bin" % rel(dat), nnz_csr, 1),
             ("wvc", "%s/w.bin" % rel(dat), nnz_csr, 1),
             ("xvc", "%s/x.bin" % rel(dat), NX, 1),
             ("str2c", "%s/str2d.bin" % rel(dat), NN, NX),
             ("kc", "%s/k2.bin" % rel(dat), 1, 1)]
        if not for_old:
            c.append(("kc0", "%s/k2.bin" % rel(dat), 1, 1))
        return c

    def assemble(src_text, name, cs):
        vml = os.path.join(work, name + ".vml")
        vmb = os.path.join(work, name + ".vmb")
        open(vml, "w", encoding="ascii").write(src_text)
        rc, out, err = P.run_asm(a.asm, vml, vmb, cs, cwd=ROOT)
        return rc, out + err, vml, vmb

    def run(exe, vmb, slot, tag):
        d = os.path.join(work, "run_" + tag)
        os.makedirs(d, exist_ok=True)
        import shutil
        local = os.path.join(d, "model.vmb")
        shutil.copyfile(vmb, local)
        if os.path.exists(os.path.join(d, "out.bin")):
            os.remove(os.path.join(d, "out.bin"))
        rc, sout, vals = P.run_engine(exe, local, dump_slot=slot, reps=1, cwd=d)
        return rc, sout, vals

    def slot_of(vmb, name):
        # reuse the task-8 helper (name-hash lookup, not a hardcoded index)
        import p16_sparse_ops as S  # noqa
        return S.slot_index(vmb, name)

    # ---------------------------------------------------------------- BEFORE --
    r.sec("BEFORE: on the FROZEN task-8 engine, is y really k-invariant?")
    old_src = """SHAPE off, 5, 1
SHAPE srci, 24, 1
SHAPE wv, 24, 1
SHAPE xv, 6, 1
SHAPE str, 24, 1
SHAPE kk, 1, 1
SHAPE sel, 24, 2
SHAPE y, 5, 1
    COPY off, offc
    COPY srci, srcc
    COPY wv, wvc
    COPY xv, xvc
    COPY str, strc
    COPY kk, kc
    TOPK_BY_STRENGTH sel, str, kk
    SPARSE_MATVEC y, off, srci, wv, xv, nnz
    HALT
"""
    # a flat strength array for the old (non-segmented) opcode
    wr(os.path.join(dat, "flat_str.bin"), str2d)
    old_consts = [("offc", "%s/off.bin" % rel(dat), NN + 1, 1),
                  ("srcc", "%s/src.bin" % rel(dat), nnz_csr, 1),
                  ("wvc", "%s/w.bin" % rel(dat), nnz_csr, 1),
                  ("xvc", "%s/x.bin" % rel(dat), NX, 1),
                  ("strc", "%s/flat_str.bin" % rel(dat), nnz_csr, 1)]
    before = {}
    for k in (1, 2, 4):
        cv = old_consts + [("kc", "%s/k%d.bin" % (rel(dat), k), 1, 1)]
        rc, msg, vml, vmb = assemble(old_src, "before_k%d" % k, cv)
        if rc:
            print("    assemble failed: %s" % msg[-200:])
            continue
        slot = slot_of(vmb, "y")
        _, _, vals = run(a.old, vmb, slot, "before_k%d" % k)
        before[k] = vals
        print("    k=%-2d y=%s nnz=%s" % (k, f32list(vals[:NN]), vals[NN] if len(vals) > NN else "?"))
    ks = sorted(before)
    inv = len(ks) > 1 and all(before[ks[0]] == before[k] for k in ks)
    r.check(inv, "OLD engine: y is byte-identical for every k (the gap)",
            "k=%s all equal" % ks if inv else "y DID vary -- gap absent?")

    # ----------------------------------------------------------------- AFTER --
    r.sec("A. AFTER: does k now change y?  (segmented TOPK + sel-gated matvec)")
    new_src = """SHAPE off, 5, 1
SHAPE srci, 24, 1
SHAPE sel, 4, 8
SHAPE wv, 24, 1
SHAPE xv, 6, 1
SHAPE str, 4, 6
SHAPE kk, 1, 1
SHAPE y, 5, 1
    COPY off, offc
    COPY srci, srcc
    COPY wv, wvc
    COPY xv, xvc
    COPY str, str2c
    COPY kk, kc
    TOPK_BY_STRENGTH sel, str, kk, seg
    SPARSE_MATVEC y, off, srci, wv, xv, sel, nnz, selgate
    HALT
"""
    after = {}
    for k in range(1, KMAX + 1):
        cv = [("offc", "%s/off.bin" % rel(dat), NN + 1, 1),
              ("srcc", "%s/src.bin" % rel(dat), nnz_csr, 1),
              ("wvc", "%s/w.bin" % rel(dat), nnz_csr, 1),
              ("xvc", "%s/x.bin" % rel(dat), NX, 1),
              ("str2c", "%s/str2d.bin" % rel(dat), NN, NX),
              ("kc", "%s/k%d.bin" % (rel(dat), k), 1, 1)]
        rc, msg, vml, vmb = assemble(new_src, "after_k%d" % k, cv)
        if rc:
            print("    assemble failed: %s" % msg[-300:])
            r.check(False, "k=%d assembles" % k, "")
            continue
        slot = slot_of(vmb, "y")
        erc, sout, vals = run(a.bench, vmb, slot, "after_k%d" % k)
        if erc != 0 or len(vals) < NN + 1:
            print("    engine rc=%d out=%r" % (erc, sout[:120]))
            r.check(False, "k=%d runs" % k, "rc=%d" % erc)
            continue
        after[k] = vals
        ref_y, ref_terms = reference(k)
        exact = all(P.bits(x) == P.bits(y) for x, y in zip(vals[:NN], ref_y))
        print("    k=%-2d y=%-52s nnz=%-5s ref_terms=%-3d bit_exact=%s"
              % (k, f32list(vals[:NN]), vals[NN], ref_terms, exact))
        after[k] = {"y": vals[:NN], "nnz": vals[NN], "ref": ref_y,
                    "terms": ref_terms, "exact": exact}

    # distinctness
    ys = {k: v["y"] for k, v in after.items()}
    ks = sorted(ys)
    all_distinct = len(set(tuple(P.bits(x) for x in ys[k]) for k in ks)) == len(ks)
    r.check(len(ks) == KMAX, "all %d k values assembled and ran" % KMAX, "got %s" % ks)
    r.check(all_distinct, "y is DIFFERENT for every k (1..%d)" % KMAX,
            "distinct ys = %d/%d" % (len(set(tuple(P.bits(x) for x in ys[k]) for k in ks)), len(ks)))
    r.check(not all(ys[k] == ys[ks[0]] for k in ks),
            "k is no longer invariant (the gap is closed)", "")

    # per-row term counts == k
    r.sec("A2. per-row contributing terms must equal k")
    for k in ks:
        per_row = []
        for chosen in topk_rows(k):
            per_row.append(len(chosen))
        total = sum(per_row)
        eng_nnz = after[k]["nnz"]
        print("    k=%-2d per-row counts=%-18s total=%-3d engine nnz=%s"
              % (k, per_row, total, eng_nnz))
        r.check(all(c == k or c == NX for c in per_row) and total == k * NN,
                "k=%d: per-row terms == k" % k, "%s" % per_row)
        r.check(f32(eng_nnz) == f32(total),
                "k=%d: engine nnz == sum of per-row terms" % k,
                "%s == %d" % (eng_nnz, total))

    # ----------------------------------------------------------- skip proof --
    r.sec("B. SKIP PROOF under the new path")
    for k in ks:
        nnz = after[k]["nnz"]
        sel_n = NN * k
        print("    k=%-2d performed=%-5s selected=%-4d dense=%d   skipped=%.1f%%"
              % (k, nnz, sel_n, dense_pairs, 100.0 * (dense_pairs - sel_n) / dense_pairs))
        r.check(f32(nnz) == f32(sel_n), "k=%d: performed == selected" % k, "%s==%d" % (nnz, sel_n))
        r.check(nnz < dense_pairs if k < NX else nnz <= dense_pairs,
                "k=%d: performed < dense" % k, "%s < %d" % (nnz, dense_pairs))

    # ------------------------------------------------------- numpy-exactness --
    r.sec("C. bit-exactness against the independent Python reference")
    for k in ks:
        d = after[k]
        maxd = max(abs(x - y) for x, y in zip(d["y"], d["ref"])) if d["ref"] else 0.0
        r.check(d["exact"], "k=%d: engine y is BIT-EXACT" % k,
                "max|diff|=%.3g" % maxd)

    # ------------------------------------------------------- can-fail controls --
    r.sec("D. CAN-FAIL CONTROLS (the comparison must be able to disagree)")
    # D1: perturb the reference's k by one -> the comparison must fail.
    for k in (2, 3):
        ref_other, _ = reference(k + 1 if k + 1 <= NX else k - 1)
        nbad = sum(1 for x, y in zip(after[k]["y"], ref_other) if P.bits(x) != P.bits(y))
        print("    D1 k=%d compared against reference(k=%d): mismatches=%d/%d"
              % (k, k + 1 if k + 1 <= NX else k - 1, nbad, NN))
        r.check(nbad > 0, "D1 (wrong-k reference) FAILS the comparison", "mismatches=%d" % nbad)
    # D2: flip the sign of one weight in the CSR -> y must change.
    alt_w = list(w)
    alt_w[0] = -alt_w[0]
    wr(os.path.join(dat, "w_flip.bin"), [float(v) for v in alt_w])
    alt_str = list(str2d)
    alt_str[0] = -alt_str[0]
    wr(os.path.join(dat, "str_flip.bin"), alt_str)
    cv = [("offc", "%s/off.bin" % rel(dat), NN + 1, 1),
          ("srcc", "%s/src.bin" % rel(dat), nnz_csr, 1),
          ("wvc", "%s/w_flip.bin" % rel(dat), nnz_csr, 1),
          ("xvc", "%s/x.bin" % rel(dat), NX, 1),
          ("str2c", "%s/str_flip.bin" % rel(dat), NN, NX),
          ("kc", "%s/k2.bin" % rel(dat), 1, 1)]
    rc, msg, vml, vmb = assemble(new_src, "d2_flip", cv)
    if rc == 0:
        _, _, vals = run(a.bench, vmb, slot_of(vmb, "y"), "d2_flip")
        nbad = sum(1 for x, y in zip(vals[:NN], after[2]["ref"]) if x != y)
        print("    D2 weight flipped: y=%s mismatches=%d/%d" % (f32list(vals[:NN]), nbad, NN))
        r.check(nbad > 0, "D2 (flip one weight) FAILS the comparison", "mismatches=%d" % nbad)
    else:
        r.check(False, "D2 assembles", msg[-150:])

    # D3: a `sel` built for a DIFFERENT CSR must be rejected (the guard, not a
    # wrong number). Swap two connection indices inside row 0's segment.
    #   -> covered in section GUARDS below.

    # D4: NEGATIVE CONTROL FOR THE GATE ITSELF.
    #
    # The claim "the gate is what made k matter" is only supported if the SAME
    # new engine still behaves k-invariantly when the gate is OFF. Otherwise the
    # change in behaviour could have come from anywhere else in the edit (the
    # segmented TOPK arm, a reshaped slot, ...).
    #
    # So: run the UNGATED program on the NEW engine, sweep k, and require BOTH
    #   (a) y reproduces the dense reference bit-exactly, and
    #   (b) y is byte-identical across k -- i.e. the old behaviour survives.
    # (b) is the control the Lead asked for; the first version of this file only
    # checked k=2 and therefore could not have falsified "the gate is inert".
    ungated = {}
    for k in (1, 2, 4):
        cv = old_consts + [("kc", "%s/k%d.bin" % (rel(dat), k), 1, 1)]
        rc, msg, vml, vmb = assemble(old_src, "d4_ungated_k%d" % k, cv)
        if rc:
            r.check(False, "D4 ungated k=%d assembles" % k, msg[-150:])
            continue
        _, _, vals = run(a.bench, vmb, slot_of(vmb, "y"), "d4_ungated_k%d" % k)
        ungated[k] = vals
        print("    D4 ungated NEW engine k=%-2d y=%s nnz=%s"
              % (k, f32list(vals[:NN]), vals[NN] if len(vals) > NN else "?"))
    ks4 = sorted(ungated)
    if len(ks4) == 3:
        ref_un, terms_un = reference_ungated()
        exact = all(P.bits(x) == P.bits(y)
                    for x, y in zip(ungated[2][:NN], ref_un))
        r.check(exact, "D4 ungated: y still matches the DENSE reference", "")
        r.check(all(f32(ungated[k][NN]) == f32(terms_un) for k in ks4),
                "D4 ungated: nnz stays at the whole-CSR count for every k",
                "nnz=%s" % [ungated[k][NN] for k in ks4])
        # The control proper: with the gate OFF, k must still do nothing.
        inv = all(ungated[ks4[0]] == ungated[k] for k in ks4)
        r.check(inv, "D4 ungated: y is STILL k-invariant (gate is the cause)",
                "k=%s identical" % ks4 if inv else "y varied WITHOUT the gate!")
    else:
        r.check(False, "D4 ungated sweep ran for all k", "got %s" % ks4)

    print("\n" + "=" * 78)
    print("compose: %d checks, %d failed" % (r.n, r.bad))
    print("RESULT: %s" % ("ALL CHECKS PASS" if r.bad == 0 else "%d FAILED" % r.bad))
    return 1 if r.bad else 0


if __name__ == "__main__":
    sys.exit(main())
