#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P15-S5 D3 RECURRENCE: the `xref` check is a third tautology, and its comment
claims a discriminating power it does not have.

FINDING (new, distinct from the D3 that was fixed)
    runtime/src/s5_load.rs lines 249-266 implement an "xref" cross-check:

        let expect = (*compact.add(xp * n + xi)).to_bits();      // line 255
        let got    = (*per_neuron.add(base_elem + xp)).to_bits(); // line 256
        if expect == got { xref_ok += 1 } else { xref_bad += 1 }

    The reorder it is validating, neuron_kv::reorder_compact_to_neurons (Rust,
    verbatim):  *drow.add(p) = *src.add(p * n + i);   with drow = dst + i*stride

    i.e.  per_neuron[i * stride + p] = compact[p * n + i].

    Substituting xi=i, xp=p, the check's two sides are THE SAME TWO EXPRESSIONS
    as the assignment. Therefore expect == got for every cell the reorder wrote,
    for EVERY possible compact buffer. `xref_bad` is identically zero.

    This is the same defect class as the direction check D3 that was already
    reported and replaced: "把赋值重述成等式" (the assignment restated as an
    equality). It survived the D3 fix because it was written with different
    variable names, which is exactly how a tautology hides from a review that
    looks for the old text.

THE FALSE CLAIM
    runtime/src/s5_load.rs lines 233-235 say:

        "**行/列（方向）写反** —— 同样分叉。实测：故意反向重排时
         `xref_bad` 非零并以 exit(9) 拒绝（见 p15_s5_load_probe.py 的
         REVERSED 用例）。"

    ("a reversed row/column writing also diverges. Measured: with a deliberately
      reversed reorder `xref_bad` is nonzero and it is rejected with exit(9).")

    MEASURED: I built a deliberately reversed compact region and ran the engine.
        correct  -> exit=0  xref_ok=24576 xref_bad=0
        reversed -> exit=0  xref_ok=24576 xref_bad=0
    The claim is false. The comment also attributes the measurement to a
    test-side REVERSED case; a test-side check can be discriminating, but this
    in-engine predicate is not, and the comment says the in-engine counter is.

EVIDENCE
    - algebraic: the two expressions are the assignment's own sides.
    - brute force: 200000 random compact buffers, xref_bad total = 0.
    - engine: correct and reversed both report xref_bad=0, exit=0.
    - line 232 also claims it catches "stride 写错" (wrong stride). With a
      wrong stride the reorder and the check would use the SAME stride value
      (load_reorder passes layout.stride to the reorder; the check reads
      layout.stride as `stride`), so a wrong stride is equally undetectable here
      -- the same self-consistency hazard the project recorded in
      docs/P14_S5_O6_报告.md §4.4 item 2.

SEVERITY
    NOT a data-safety hole: no illegal input is ACCEPTED because of this; the
    tautology simply provides no evidence. It is a FALSE CLAIM in a comment plus
    a check that cannot fail -- the class this project treats as the cardinal
    sin (docs/P14_verification.md §0.1), and the same class as D3.

RECOMMENDED FIX (the implementer's, not mine)
    Either delete the xref counters and say plainly that no in-engine predicate
    can validate the direction (the honest position already written at
    s5_load.rs lines 236-244 and 326-335 of the earlier revision), or replace
    them with a genuinely independent expectation. Correct the line 233-235
    comment either way.

Run from <repo>:  python tools/p15_verify_d3_recurrence.py
"""
import os
import random
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p15_verify_lib as V  # noqa: E402
import p15_verify_d1 as D1   # noqa: E402

N = D1.N_NEURONS
PMAX = D1.PMAX


def build(n_pos, fill):
    sec = bytearray()
    for v in (D1.NL, D1.NKH, D1.HD, PMAX, PMAX):
        sec += struct.pack("<I", v)
    for i in range(N):
        sec += struct.pack("<I", i * PMAX)
    b = bytearray()
    b += struct.pack("<I", 0x4D4C4D56)
    b += struct.pack("<H", 1)
    b += struct.pack("<H", 0)
    while len(b) < 32:
        b += b"\x00"
    b += struct.pack("<I", 1)
    b += struct.pack("<I", 44)
    b += struct.pack("<I", len(sec))
    b += sec
    coff = (44 + len(sec) + 63) & ~63
    while len(b) < coff:
        b += b"\x00"
    for p in range(n_pos):
        for i in range(N):
            b += struct.pack("<f", fill(p, i))
    return bytes(b)


def field(txt, key):
    for l in txt.splitlines():
        if key in l:
            tok = l.split(key, 1)[1].strip().split()
            if tok and tok[0].isdigit():
                return int(tok[0])
    return None


def main():
    print("=" * 78)
    print("D3 RECURRENCE: the xref check is a tautology (third instance)")
    print("=" * 78)
    meta = V.MetaCheck()

    # ---- 1. algebraic / brute force ----
    print("\n--- 1. brute force: can xref_bad ever be nonzero? ---")
    random.seed(12345)
    n, npos, stride = 8, 3, 8
    total_bad = 0
    trials = 200000
    for _ in range(trials):
        compact = [random.randint(0, 5) for _ in range(npos * n)]
        per = [0] * (n * stride)
        for i in range(n):                     # reorder, verbatim
            for p in range(npos):
                per[i * stride + p] = compact[p * n + i]
        for xi in range(n):                    # xref, verbatim
            for xp in range(npos):
                if compact[xp * n + xi] != per[xi * stride + xp]:
                    total_bad += 1
    print("   %d random compact buffers -> xref_bad total = %d" % (trials,
                                                                    total_bad))
    meta.record("D3r the xref counter CAN be nonzero (i.e. it is not a "
                "tautology)",
                total_bad > 0,
                "if some input made xref_bad nonzero the check would be real; "
                "the brute force must find one",
                total_bad > 0,
                "%d mismatches over %d random buffers" % (total_bad, trials))

    # ---- 2. measured through the engine ----
    print("\n--- 2. measured through the engine ---")
    exe = os.path.join(V.ROOT, "examples", "p15_verify", "iso_nkv",
                       "x86_64-pc-windows-msvc", "release", "vmlm-runtime.exe")
    if not os.path.exists(exe):
        print("   engine not built"); return 2

    def f_correct(p, i):
        return float((p + 1) * 4096 + i)

    def f_reversed(p, i):
        return f_correct(i, p)

    rows = []
    for name, fn in (("correct", f_correct), ("reversed", f_reversed)):
        rc, txt = D1.run_engine(exe, build(4, fn))
        xo = field(txt, "xref_ok=")
        xb = field(txt, "xref_bad=")
        print("   %-9s exit=%-3d xref_ok=%-6s xref_bad=%s" % (name, rc, xo, xb))
        rows.append(dict(fill=name, exit=rc, xref_ok=xo, xref_bad=xb,
                         line=next((l.strip() for l in txt.splitlines()
                                    if "xref" in l), "")))
    rev_bad_zero = (rows[1]["xref_bad"] == 0)
    meta.record("D3r a reversed reorder makes xref_bad NONZERO as the comment "
                "claims (s5_load.rs:233-235)",
                not rev_bad_zero,
                "the comment claims the reversed case gives a nonzero xref_bad "
                "and exits 9; the measurement must show that for the claim to "
                "be true",
                rev_bad_zero,
                "reversed exit=%d xref_bad=%s (comment claims nonzero + exit 9)"
                % (rows[1]["exit"], rows[1]["xref_bad"]))

    # ---- 3. wrong stride: same hazard ----
    print("\n--- 3. does a wrong stride make xref_bad nonzero? ---")
    print("   The reorder and the xref check BOTH take `stride` from the same")
    print("   layout, so a wrong stride is used consistently by both and cannot")
    print("   diverge. This is the §4.4-self-consistency hazard, restated.")
    meta.record("D3r a wrong stride makes xref_bad nonzero",
                False,
                "both sides read layout.stride, so a wrong stride is shared and "
                "self-consistent; if this were satisfiable the comment's "
                "wrong-stride claim would hold",
                False,
                "structurally impossible: load_reorder passes layout.stride to "
                "the reorder and the check reads layout.stride as `stride`")

    v = V.ensure_outdir("d3")
    import json
    with open(os.path.join(v, "d3_recurrence.json"), "w", encoding="utf-8") as f:
        json.dump(dict(rows=rows, brute_bad=total_bad, trials=trials),
                  f, indent=2, ensure_ascii=False)
    print("\nwrote %s" % os.path.join(v, "d3_recurrence.json"))
    meta.report()
    print("\nFINDING: the xref predicate is a tautology (xref_bad identically 0)")
    print("         and the comment at s5_load.rs:233-235 overstates it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
