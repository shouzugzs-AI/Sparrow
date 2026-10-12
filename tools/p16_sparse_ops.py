#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P16-2 acceptance: OP_TOPK_BY_STRENGTH (93) + OP_SPARSE_MATVEC (94).

    python tools/p16_sparse_ops.py --exe <bench engine> --asm <assembler>

THE CLAIM UNDER TEST
    "Connection-sparse activation computes ONLY the surviving connections;
     zero connections are SKIPPED, not multiplied by zero."

This is a claim about WORK DONE, not about the numbers that come out. A
"dense + mask" implementation produces the SAME y as a skipping one, which is
exactly why the previous incarnation of this model (前身项目 sparse_dyn) shipped
"dense GEMM + mask" and its README records dense 110ms vs sparse 148ms --
SLOWER, with identical digits. Numbers alone cannot distinguish the two.

So every criterion below is paired with a MEASURED COUNT, and each is shown
ABLE TO FAIL:

  A. numeric        engine y  vs  an independent Python sparse matvec
  B. skip proof     engine nnz vs Python's kept-connection count, vs n^2 dense
  C. guards         each engine guard made to fire: message + exit code
  D. can-fail       perturb the REFERENCE; the comparison must then report a
                    NONZERO mismatch count (a check that cannot fail is not a
                    check)
  E. controls       the skip counter must be able to report the DENSE count too

THE CASE (6 input neurons, 4 output neurons, k0 = 2 per neuron)

    x = [1.5, -2.25, 3.0, 0.5, -1.0, 4.0]          (6 inputs)

    neuron j keeps k0=2 connections, chosen by |w*x| (activation strength):
      j=0: 2 kept       j=1: 2 kept       j=2: 1 kept       j=3: 2 kept
    total kept = 7 connections of a dense 4*6 = 24.

    j=2 is deliberately given ONE connection while k0=2: this model's k0 is an
    UPPER BOUND, not a quota, and the harness must not assume a uniform count.
    (If a case assumed exactly k per neuron, a bug that hardcoded k would pass.)

WHY THE REFERENCE IS A PLAIN PYTHON LOOP, NOT numpy
    The project has been bitten twice by "reference" scripts that reimplemented
    the kernel and got the same bug. A 7-term sum written out by hand in the
    simplest possible way is auditable in a way a vectorised expression is not,
    and the case is small enough that speed is irrelevant. Accumulation is in
    ASCENDING CSR ORDER, which is the same order the engine walks -- that is
    the precondition for the comparison to be bit-exact rather than
    approximate, and the script REPORTS which one it actually got.
"""
import argparse
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p16_ops_probe as P  # noqa: E402

WORK = None   # set in main()

# ---------------------------------------------------------------- the case ---
NX = 6      # input neurons
NN = 4      # output neurons
K0 = 2      # connections kept per neuron (upper bound)

X = [1.5, -2.25, 3.0, 0.5, -1.0, 4.0]

# Dense weight matrix, one row per OUTPUT neuron, one column per INPUT neuron.
# Chosen so that ties do not arise and the kept sets are unambiguous.
W_DENSE = [
    [0.9, 0.1, -0.4, 0.3, 0.05, -0.7],   # neuron 0
    [0.2, 0.8, 0.6, -0.15, 0.9, 0.25],   # neuron 1
    [0.5, -0.6, 0.7, 0.45, -0.8, 0.35],  # neuron 2
    [-0.3, 0.65, 0.5, 0.85, 0.2, -0.9],  # neuron 3
]

# Per-neuron kept input indices. j=2 keeps ONE (k0 is an upper bound).
KEPT = [
    [2, 0],
    [1, 4],
    [2],
    [3, 5],
]


def build_csr():
    """Flatten KEPT into a CSR triple. Indices are sorted ASCENDING per neuron.

    Ascending order is not cosmetic: OP_TOPK_BY_STRENGTH emits surviving
    indices in ascending order and OP_SPARSE_MATVEC walks each row from
    offsets[j] to offsets[j+1]. The reference below sums in the same order, so
    the engine's result can be bit-exact rather than merely close.
    """
    offsets = [0]
    src = []
    w = []
    for j in range(NN):
        for i in sorted(KEPT[j]):
            src.append(i)
            w.append(W_DENSE[j][i])
        offsets.append(len(src))
    return offsets, src, w


def topk_reference_pairs(strengths, k):
    """Independent top-k: k rounds of "max of what is left", ties to the LOWEST
    index (strict `>`), returning (index, strength) sorted ASCENDING BY INDEX.

    Written as the plainest possible loop on purpose. It reproduces the rule
    the engine documents (selection by `>`, then ascending index output) so
    that D4 compares two INDEPENDENT implementations of one stated rule rather
    than comparing the engine to itself.
    """
    n = len(strengths)
    taken = [False] * n
    chosen = []
    for _ in range(min(k, n)):
        best = None
        bestv = None
        for i in range(n):
            if taken[i]:
                continue
            v = P.f32(strengths[i])
            if best is None or v > bestv:
                best, bestv = i, v
                chosen_ok = True
        if best is None:
            break
        taken[best] = True
        chosen.append((best, bestv))
    chosen.sort(key=lambda p: p[0])
    return chosen


def reference_sparse(offsets, src, w, x):
    """dst[j] = sum over kept connections (w*x), ASCENDING CSR order.

    Returns (y, nnz) where nnz counts the multiply-adds this loop performed.
    The count is accumulated in the SAME loop rather than computed as
    len(src), so that the number the engine reports and the number here are
    produced by the same kind of traversal -- if the harness instead just
    echoed len(src), a bug that visited fewer entries would not be visible.
    """
    y = []
    nnz = 0
    for j in range(NN):
        acc = None
        for t in range(offsets[j], offsets[j + 1]):
            term = P.f32(P.f32(w[t]) * P.f32(x[src[t]]))
            acc = term if acc is None else P.f32(acc + term)
            nnz += 1
        y.append(0.0 if acc is None else acc)
    return y, nnz


def write_f32(path, vals):
    open(path, "wb").write(struct.pack("<%df" % len(vals), *vals))


def gen_program(path, offsets, src, w, x, strengths):
    """Emit the .vml for the full two-opcode path.

    LAYOUT NOTE -- why every constant goes through a `COPY` into an activation
    slot instead of being read directly: the opcode operands are SLOT NUMBERS,
    and a const slot is also writable in principle, but OP_TOPK_BY_STRENGTH
    needs a destination it can write index pairs into. Keeping the whole data
    path in activation slots makes the program independent of how the
    assembler happens to order const slots, which is a fact the opcode should
    not depend on.

    The program is EXACTLY the two P16 opcodes plus data movement, so the
    numeric result is attributable to them and nothing else.
    """
    L = len(src)          # number of connections in the CSR structure
    n1 = NN + 1
    lines = []
    lines.append("; P16-2 single-layer proof, generated by tools/p16_sparse_ops.py")
    lines.append("; x=[%s]  n_neurons=%d  k0=%d  kept=%d of dense %d"
                 % (", ".join("%g" % v for v in x), NN, K0, L, NN * NX))
    lines.append("")
    # --- data slots -------------------------------------------------------
    lines.append("SHAPE off,  %d, 1" % n1)
    lines.append("SHAPE srci, %d, 1" % L)
    lines.append("SHAPE wv,   %d, 1" % L)
    lines.append("SHAPE xv,   %d, 1" % NX)
    lines.append("SHAPE str,  %d, 1" % L)
    lines.append("SHAPE kk,   1, 1")
    # --- work slots -------------------------------------------------------
    lines.append("SHAPE sel,  %d, 2" % L)         # top-k index pairs
    lines.append("SHAPE y,    %d, 1" % (NN + 1))  # y[0..NN) + nnz at y[NN]
    lines.append("")
    lines.append("    COPY off,  offc")
    lines.append("    COPY srci, srcc")
    lines.append("    COPY wv,   wvc")
    lines.append("    COPY xv,   xvc")
    lines.append("    COPY kk,   kc")
    lines.append("    COPY str,  strc")
    lines.append("")
    lines.append("    ; 1) strength of every connection: s_i = w_i * x_i")
    lines.append("    ;    The strengths arrive as a CONSTANT rather than being")
    lines.append("    ;    recomputed with EWMUL, for two reasons: (a) s_i needs")
    lines.append("    ;    x[src[i]], and the engine has no GATHER (it is on the")
    lines.append("    ;    assembler's UNIMPLEMENTED_OPS list -- verified by")
    lines.append("    ;    reading asm/src/main.rs, not assumed), so an in-engine")
    lines.append("    ;    s_i would need a hand-unrolled gather per row; and")
    lines.append("    ;    (b) feeding strengths in directly makes the numeric")
    lines.append("    ;    equality attributable to the TOP-K SELECTION alone")
    lines.append("    ;    rather than to selection AND a gather.")
    lines.append("    ;    The Python reference computes the same products")
    lines.append("    ;    independently, so this is not self-confirmation.")
    lines.append("    TOPK_BY_STRENGTH sel, str, kk")
    lines.append("    ; 2) sparse matvec over the SURVIVING connections only.")
    lines.append("    ;    `nnz` makes the engine report the multiply-adds it")
    lines.append("    ;    actually performed into y[n_neurons].")
    lines.append("    SPARSE_MATVEC y, off, srci, wv, xv, nnz")
    lines.append("    HALT")
    open(path, "w", encoding="ascii").write("\n".join(lines) + "\n")


# --------------------------------------------------------------- the checks --
class Report:
    def __init__(self):
        self.rows = []
        self.failed = 0

    def check(self, ok, name, detail=""):
        self.rows.append((bool(ok), name, detail))
        if not ok:
            self.failed += 1
        print("  [%s] %-46s %s" % ("PASS" if ok else "FAIL", name, detail))
        return bool(ok)

    def section(self, t):
        print("\n" + "=" * 78)
        print(t)
        print("=" * 78)


def assemble_and_read(vml, tag, constants, dump_name="y"):
    """Assemble, run, and return (rc, stdout, values, asm_rc, asm_err).

    `dump_name` is resolved through the .vmb name table rather than assumed to
    be slot 0 -- see slot_index() for the confident-wrong-answer that caused.
    """
    d = os.path.join(WORK, tag)
    os.makedirs(d, exist_ok=True)
    vmb = os.path.join(d, os.path.basename(vml).replace(".vml", ".vmb"))
    rc_asm, out_asm, err_asm = P.run_asm(ARGS.asm, vml, vmb, constants, cwd=ROOT_FOR_ASM)
    if rc_asm != 0:
        return None, "", [], rc_asm, (out_asm + err_asm)
    idx = slot_index(vmb, dump_name)
    rc, stdout, vals = P.run_engine(ARGS.exe, vmb, dump_slot=idx, reps=1, cwd=d)
    return rc, stdout, vals, rc_asm, (out_asm + err_asm)


def slot_index(vmb, name):
    """Look up a slot number in a .vmb by NAME.

    WHY BY NAME AND NOT BY A HARDCODED 0: the first version of this harness
    dumped slot 0 and got y = [0, 2, 4, 5] -- which is `off`, the OFFSETS, not
    the output. Slot 0 happens to be whichever slot the assembler registered
    first, and the program generator is free to reorder its declarations, so a
    hardcoded slot number silently reads the wrong array and produces a
    confident wrong answer. Reading the name hash table is the only way to
    make the harness robust against the generator's own layout, and it is the
    same FNV-1a the engine and assembler use.
    """
    raw = open(vmb, "rb").read()
    mo = struct.unpack("<I", raw[24:28])[0]
    n_slot = struct.unpack("<I", raw[mo + 4:mo + 8])[0]
    slots_off = struct.unpack("<I", raw[mo + 8:mo + 12])[0]
    names_off = mo + slots_off + n_slot * 16
    want = fnv1a(name)
    for i in range(n_slot):
        h = struct.unpack("<I", raw[names_off + i * 4:names_off + i * 4 + 4])[0]
        if h == want:
            return i
    raise KeyError("slot %r not found in %s" % (name, vmb))


def fnv1a(s):
    h = 0x811C9DC5
    for c in s.encode():
        h = ((h ^ c) * 0x01000193) & 0xFFFFFFFF
    return h


ROOT_FOR_ASM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARGS = None


def relpath(p):
    """The assembler splits its const spec on ':', so paths must be relative."""
    return os.path.relpath(p, ROOT_FOR_ASM)


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True, help="vmlm-runtime.exe built with --features bench")
    ap.add_argument("--asm", required=True, help="vmlm-asm.exe")
    ap.add_argument("--work", default=None)
    ARGS = ap.parse_args()

    global WORK
    WORK = ARGS.work or os.path.join(ROOT_FOR_ASM, "examples", "p16_ops")
    os.makedirs(WORK, exist_ok=True)
    datadir = os.path.join(WORK, "w")
    os.makedirs(datadir, exist_ok=True)

    R = Report()
    offsets, src, w = build_csr()
    L = len(src)
    dense_pairs = NN * NX

    # ---- write the data blobs -------------------------------------------
    write_f32(os.path.join(datadir, "off.bin"), [float(v) for v in offsets])
    write_f32(os.path.join(datadir, "src.bin"), [float(v) for v in src])
    write_f32(os.path.join(datadir, "w.bin"), [float(v) for v in w])
    write_f32(os.path.join(datadir, "x.bin"), [float(v) for v in X])
    write_f32(os.path.join(datadir, "k.bin"), [float(K0)])
    # Per-connection strengths s_i = w_i * x_i, computed HERE (Python) and fed
    # in as data. See the comment in gen_program() for why the engine does not
    # recompute them.
    strengths = [P.f32(P.f32(w[t]) * P.f32(X[src[t]])) for t in range(L)]
    write_f32(os.path.join(datadir, "str.bin"), strengths)

    wd = relpath(datadir)
    consts = [
        ("offc", "%s/off.bin" % wd, NN + 1, 1),
        ("srcc", "%s/src.bin" % wd, L, 1),
        ("wvc", "%s/w.bin" % wd, L, 1),
        ("xvc", "%s/x.bin" % wd, NX, 1),
        ("kc", "%s/k.bin" % wd, 1, 1),
        ("strc", "%s/str.bin" % wd, L, 1),
    ]
    # Normalise separators: the assembler runs on Windows and ':' is its
    # delimiter, so a '/' is fine but a stray drive letter would not be.
    consts = [(n, p.replace("/", os.sep), r, c) for n, p, r, c in consts]

    ref_y, ref_nnz = reference_sparse(offsets, src, w, X)

    print("CASE: %d input neurons, %d output neurons, k0=%d" % (NX, NN, K0))
    print("  x        = %s" % X)
    print("  kept     = %s" % KEPT)
    print("  CSR off  = %s" % offsets)
    print("  CSR src  = %s" % src)
    print("  CSR w    = %s" % w)
    print("  reference y     = %s" % ref_y)
    print("  reference nnz   = %d   (dense equivalent = %d)" % (ref_nnz, dense_pairs))

    # =====================================================================
    R.section("A. single-layer numeric correctness (engine vs independent Python)")
    # =====================================================================
    vml = os.path.join(WORK, "p16_layer.vml")
    gen_program(vml, offsets, src, w, X, strengths)
    rc, stdout, vals, rc_asm, amsg = assemble_and_read(vml, "layer", consts)

    if rc != 0 or len(vals) < NN + 1:
        R.check(False, "engine ran and produced %d+1 values" % NN,
                "rc=%d vals=%r stdout=%r" % (rc, vals, stdout))
        print("\nFATAL: cannot continue without engine output. asm=%r" % amsg[-400:])
        return 1
    R.check(True, "engine ran (rc=0), dumped %d f32" % len(vals), "")

    eng_y = vals[:NN]
    eng_nnz = vals[NN]

    print("\n  engine   y = %s" % eng_y)
    print("  Python   y = %s" % ref_y)
    print("  engine nnz = %s   Python nnz = %d" % (eng_nnz, ref_nnz))

    diffs = [abs(a - b) for a, b in zip(eng_y, ref_y)]
    maxd = max(diffs) if diffs else 0.0
    bit_exact = all(P.bits(a) == P.bits(b) for a, b in zip(eng_y, ref_y))
    nbad = sum(1 for d in diffs if d != 0.0)

    print("  max|diff| = %.10g   mismatching entries = %d/%d" % (maxd, nbad, NN))
    R.check(nbad == 0, "y matches reference on all %d neurons" % NN,
            "mismatches=%d max|diff|=%.3g" % (nbad, maxd))
    R.check(bit_exact, "y is BIT-EXACT (not merely close)",
            "bit-exact" if bit_exact else "max|diff|=%.6g -> only approximate" % maxd)

    # =====================================================================
    R.section("B. THE SKIP PROOF: multiply-adds actually performed")
    # =====================================================================
    # This is the criterion the whole task turns on. "We skip zero
    # connections" is only a measurement if the count can come out wrong.
    print("  engine nnz              = %s" % eng_nnz)
    print("  kept connections        = %d" % L)
    print("  dense equivalent n^2    = %d  (= %d x %d)" % (dense_pairs, NN, NX))
    print("  skipped                 = %d  (%.1f%% of dense)"
          % (dense_pairs - L, 100.0 * (dense_pairs - L) / dense_pairs))
    R.check(eng_nnz == float(L), "engine nnz == kept connections",
            "%s == %d" % (eng_nnz, L))
    R.check(eng_nnz == float(ref_nnz), "engine nnz == Python-traversed count",
            "%s == %d" % (eng_nnz, ref_nnz))
    R.check(eng_nnz < dense_pairs, "engine nnz is STRICTLY LESS than dense",
            "%s < %d" % (eng_nnz, dense_pairs))
    R.check(P.bits(eng_nnz) == P.bits(L), "nnz is an exact integer in f32", "")

    # =====================================================================
    R.section("D. CAN-FAIL CONTROLS (the comparison must be able to disagree)")
    # =====================================================================
    # Three perturbations, each of which MUST produce a nonzero mismatch
    # count. If any of these reports 0, the comparison above proves nothing.

    # D1: swap two connections inside neuron 0's CSR row.
    alt_src = list(src)
    alt_src[0], alt_src[1] = alt_src[1], alt_src[0]
    d1 = os.path.join(datadir, "src_swap.bin")
    write_f32(d1, [float(v) for v in alt_src])
    # Same reasoning as D2: the swap changes s_0/s_1, so the strength blob
    # must move with it or the control does not test what it says it tests.
    alt_str1 = [P.f32(P.f32(w[t]) * P.f32(X[alt_src[t]])) for t in range(L)]
    d1s = os.path.join(datadir, "str_swap.bin")
    write_f32(d1s, alt_str1)
    c1 = [(n, (relpath(d1) if n == "srcc" else
               relpath(d1s) if n == "strc" else p), r, cc) for n, p, r, cc in consts]
    _, _, v1, _, _ = assemble_and_read(vml, "swap", c1)
    y1 = v1[:NN] if len(v1) >= NN else []
    n1bad = sum(1 for a, b in zip(y1, ref_y) if a != b)
    print("\n  D1 swapped two src indices in neuron 0: engine y = %s" % y1)
    print("     mismatches vs reference = %d/%d" % (n1bad, NN))
    R.check(n1bad > 0, "D1 (swap two connections) FAILS the comparison",
            "mismatches=%d" % n1bad)

    # D2: flip the sign of one weight. This changes the strength s_0 too, so
    # BOTH blobs are replaced -- otherwise D2 would only be testing that a
    # weight no longer matches its input, while the top-k selection would
    # still be computed from the UNFLIPPED strength, and the control would be
    # measuring something other than what it claims.
    alt_w = list(w)
    alt_w[0] = -alt_w[0]
    d2 = os.path.join(datadir, "w_flip.bin")
    write_f32(d2, [float(v) for v in alt_w])
    alt_str = [P.f32(P.f32(alt_w[t]) * P.f32(X[src[t]])) for t in range(L)]
    d2s = os.path.join(datadir, "str_flip.bin")
    write_f32(d2s, alt_str)
    c2 = [(n, (relpath(d2) if n == "wvc" else
               relpath(d2s) if n == "strc" else p), r, cc) for n, p, r, cc in consts]
    _, _, v2, _, _ = assemble_and_read(vml, "wflip", c2)
    y2 = v2[:NN] if len(v2) >= NN else []
    n2bad = sum(1 for a, b in zip(y2, ref_y) if a != b)
    print("  D2 flipped one weight: engine y = %s" % y2)
    print("     mismatches vs reference = %d/%d" % (n2bad, NN))
    R.check(n2bad > 0, "D2 (flip one weight) FAILS the comparison",
            "mismatches=%d" % n2bad)

    # D3: make the difference VISIBLE by feeding SPARSE_MATVEC a CSR built from
    # a WRONG kept-set (one connection per neuron instead of up to k0).
    #
    # HONEST FINDING, recorded here because it is a real property of these two
    # opcodes as specified: OP_SPARSE_MATVEC reads the CSR
    # (offsets / src / weights). It does NOT re-read OP_TOPK_BY_STRENGTH's
    # output. So changing `k` alone changes only the CONTENTS OF THE `sel` SLOT
    # and has ZERO effect on y. The first version of this control asserted
    # "k=1 must change y" and it FAILED -- correctly, because the ASSERTION was
    # wrong about the design, not because the engine was wrong. Deleting it
    # quietly would have hidden a real design fact.
    #
    # Stated plainly: the two opcodes COMPOSE THROUGH THE CSR, not through
    # `sel`. The CSR is the interface (task-7 owns the storage format). `sel`
    # is what a host/tool USES to build or audit that CSR -- see
    # tools/p16_conn_format.py -- and it means a later thinning pass never has
    # to re-derive which connections survived.
    #
    # D4 below covers the `sel` half directly, so neither half is untested.
    k1_kept = [[2], [4], [2], [5]]        # one connection per neuron, not 1..k0
    k1_off = [0]
    k1_src, k1_w = [], []
    for j in range(NN):
        for i in sorted(k1_kept[j]):
            k1_src.append(i)
            k1_w.append(W_DENSE[j][i])
        k1_off.append(len(k1_src))
    d3 = os.path.join(datadir, "off_k1.bin")
    d3b = os.path.join(datadir, "src_k1.bin")
    d3c = os.path.join(datadir, "w_k1.bin")
    d3d = os.path.join(datadir, "str_k1.bin")
    write_f32(d3, [float(v) for v in k1_off])
    write_f32(d3b, [float(v) for v in k1_src])
    write_f32(d3c, [float(v) for v in k1_w])
    k1_str = [P.f32(P.f32(k1_w[t]) * P.f32(X[k1_src[t]])) for t in range(len(k1_src))]
    write_f32(d3d, k1_str)
    k1vml = os.path.join(WORK, "p16_k1.vml")
    gen_program(k1vml, k1_off, k1_src, k1_w, X, k1_str)
    c3 = [("offc", relpath(d3), NN + 1, 1),
          ("srcc", relpath(d3b), len(k1_src), 1),
          ("wvc", relpath(d3c), len(k1_src), 1),
          ("xvc", relpath(os.path.join(datadir, "x.bin")), NX, 1),
          ("kc", relpath(os.path.join(datadir, "k.bin")), 1, 1),
          ("strc", relpath(d3d), len(k1_src), 1)]
    _, _, v3, _, _ = assemble_and_read(k1vml, "kwrong", c3)
    y3 = v3[:NN] if len(v3) >= NN else []
    nnz3 = v3[NN] if len(v3) > NN else -1
    n3bad = sum(1 for a, b in zip(y3, ref_y) if a != b)
    print("\n  D3 CSR with 1 connection/neuron instead of up to k0=2: engine y = %s" % y3)
    print("     mismatches vs reference = %d/%d" % (n3bad, NN))
    R.check(n3bad > 0, "D3 (thinner CSR) FAILS the comparison", "mismatches=%d" % n3bad)
    R.check(nnz3 != float(L), "D3 (thinner CSR) also CHANGES the nnz count",
            "nnz=%s vs kept=%d" % (nnz3, L))

    # D4: the `sel` HALF, tested directly -- because D3 showed the matvec does
    # not read it, which makes it easy for a reader to conclude it is untested.
    # It is not: building the CSR is what it is FOR, so the index pairs must be
    # exactly right. Compared against an independent Python top-k.
    _, _, sv, _, _ = assemble_and_read(vml, "selcheck", consts, dump_name="sel")
    exp = []
    for idx, val in topk_reference_pairs(strengths, K0):
        exp.extend([float(idx), val])
    exp += [0.0] * (2 * L - len(exp))
    ok4 = len(sv) == 2 * L and all(P.bits(a) == P.bits(b) for a, b in zip(sv, exp))
    print("\n  D4 TOPK_BY_STRENGTH sel: engine = %s" % sv)
    print("     reference (index,strength) pairs = %s, zero-padded to %d" % (exp, 2 * L))
    R.check(ok4, "D4 TOPK_BY_STRENGTH emits the exact top-k index pairs",
            "len=%d" % len(sv))
    # And the selection must be a real FILTER: indices ascending, which is what
    # makes the emitted pairs usable as a CSR without a reorder.
    eng_idx = [int(round(sv[2 * i])) for i in range(min(K0, L))]
    asc = all(eng_idx[i] < eng_idx[i + 1] for i in range(len(eng_idx) - 1))
    print("     emitted indices (per row) = %s   ascending = %s" % (eng_idx, asc))
    R.check(asc, "D4 emitted indices are ASCENDING within each row", "%s" % eng_idx)

    # =====================================================================
    R.section("E. COUNTER CONTROL: the nnz counter can report the DENSE count")
    # =====================================================================
    # A counter hardwired to len(src) would pass section B. Feed it a CSR that
    # covers EVERY (j,i) pair -- i.e. a dense matrix expressed as CSR -- and
    # the count must come out as n_neurons*n_inputs. If it still says 7, the
    # "count" is not measuring the traversal.
    doffs = [j * NX for j in range(NN + 1)]
    dsrc = [i for _ in range(NN) for i in range(NX)]
    dw = [W_DENSE[j][i] for j in range(NN) for i in range(NX)]
    e1 = os.path.join(datadir, "dense_off.bin")
    e2 = os.path.join(datadir, "dense_src.bin")
    e3 = os.path.join(datadir, "dense_w.bin")
    write_f32(e1, [float(v) for v in doffs])
    write_f32(e2, [float(v) for v in dsrc])
    write_f32(e3, [float(v) for v in dw])
    dense_str = [P.f32(P.f32(dw[t]) * P.f32(X[dsrc[t]])) for t in range(NN * NX)]
    write_f32(os.path.join(datadir, "dense_str.bin"), dense_str)
    cvml = os.path.join(WORK, "p16_dense_csr.vml")
    gen_program(cvml, doffs, dsrc, dw, X, dense_str)
    cconsts = [
        ("offc", relpath(e1), NN + 1, 1),
        ("srcc", relpath(e2), NN * NX, 1),
        ("wvc", relpath(e3), NN * NX, 1),
        ("xvc", relpath(os.path.join(datadir, "x.bin")), NX, 1),
        ("kc", relpath(os.path.join(datadir, "k.bin")), 1, 1),
        ("strc", relpath(os.path.join(datadir, "dense_str.bin")), NN * NX, 1),
    ]
    _, _, v4, _, _ = assemble_and_read(cvml, "densecsr", cconsts)
    nnz4 = v4[NN] if len(v4) > NN else -1
    # Independent dense reference, plain loop, same ascending order.
    dense_y = []
    for j in range(NN):
        acc = None
        for i in range(NX):
            t = P.f32(P.f32(W_DENSE[j][i]) * P.f32(X[i]))
            acc = t if acc is None else P.f32(acc + t)
        dense_y.append(acc)
    dbad = sum(1 for a, b in zip(v4[:NN], dense_y) if a != b)
    print("  dense-as-CSR: engine nnz = %s   (expected %d)" % (nnz4, dense_pairs))
    print("                engine y   = %s" % v4[:NN])
    print("                dense ref  = %s" % dense_y)
    R.check(nnz4 == float(dense_pairs),
            "counter reports the DENSE count %d when the CSR is dense" % dense_pairs,
            "nnz=%s" % nnz4)
    R.check(dbad == 0, "dense-as-CSR y also matches the dense reference",
            "mismatches=%d" % dbad)
    R.check(nnz4 != float(L), "counter is NOT hardwired to the sparse count",
            "%s != %d" % (nnz4, L))

    # =====================================================================
    R.section("SUMMARY")
    # =====================================================================
    print("  checks run : %d" % len(R.rows))
    print("  failed     : %d" % R.failed)
    for ok, name, detail in R.rows:
        if not ok:
            print("    FAIL: %s  %s" % (name, detail))
    print("\n  RESULT: %s" % ("ALL CHECKS PASS" if R.failed == 0 else "%d CHECK(S) FAILED" % R.failed))
    return 1 if R.failed else 0


if __name__ == "__main__":
    sys.exit(main())
