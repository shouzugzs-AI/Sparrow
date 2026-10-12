#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P15-S5 defect D3: the "direction check" in runtime/src/s5_load.rs cannot fail.

MINIMAL REPRODUCTION, and why it is a defect rather than a style issue.

THE CHECK (runtime/src/s5_load.rs, verify_bitexact, the direction block):
    want = compact[p * n + i]            (p = position/slot, i = neuron)
    got  = per_neuron[i * stride + p]
    require want == got, for every (i, p), else exit(9) with
        "NKV: reorder direction is wrong at neuron=.. slot=.."

THE ASSIGNMENT IT IS SUPPOSED TO VALIDATE
    neuron_kv::reorder_compact_to_neurons does, verbatim:
        *drow.add(p) = *src.add(p * n + i);     // drow = dst + i*stride
    i.e.  per_neuron[i * stride + p] = compact[p * n + i]

So the check compares the destination cell against the SOURCE CELL THE SAME
FUNCTION JUST COPIED FROM. It is the assignment restated as an equality. For ANY
input whatsoever -- correct, transposed, reversed, or random -- the comparison
holds, because the write made it hold one statement earlier.

THAT IS THE POINT: a check that cannot fail is not a measurement. This project
has that rule in writing (docs/P14_verification.md §0.1) and has been burned by
its violation twice (docs/P13_报告.md §3.2 "that 0 B was an empty measurement";
docs/P14_S5_O6_报告.md §4.4 item 2, the self-consistent wrong-stride meta-check).

WHY THE COMMENT IN THE SOURCE MISDESCRIBES THIS
    The source says (s5_load.rs:257-260):
      "要证明「行→列」这个**方向**对，判据必须是**逐格**的：
       per_neuron[i*stride + p] 必须逐位等于 compact[p*n + i]
       这一条对方向敏感：反向重排（列→行）会让它大面积不等"
    "This is direction-sensitive: a reversed reorder would make it largely
     unequal."
    That claim is FALSE as written, and I measure it: a reversed reorder DOES
    make `compact[p*n+i]` differ from `compact[i*n_pos+p]` -- but the check never
    compares those two. It compares the destination against `compact[p*n+i]`,
    which the reorder guarantees. The sentence compares the wrong pair.

WHAT *WOULD* BE DIRECTION-SENSITIVE
    Either of these, neither of which is present:
      (a) compare against an INDEPENDENT expectation, e.g. verify that
          per_neuron[i*stride+p] equals a value the check computes from a
          DIFFERENT source (such as re-reading through
          reorder_neurons_to_compact and requiring the ORIGINAL disk bytes at
          the ORIGINAL offsets -- which is the bit-exactness test, and it too is
          self-consistent); or
      (b) assert a PROPERTY that the transpose would violate, e.g. that the
          per-neuron rows are NOT constant when the compact data is
          position-varying (a real, falsifiable property); or
      (c) compare the produced layout against a REFERENCE layout computed
          independently on the host side and shipped/cross-checked.

HOW I MEASURED IT (three artifacts, same code path)
    All three are byte-built by tools/p15_verify_d1.py from primitives:
      correct  : compact[p*n+i] = (p+1)*1e6 + i
      reversed : compact[p*n+i] = fill(i, p)   (the transposed convention)
      identity : compact[p*n+i] = i            (position-invariant)

    Results (engine built with --features neuron-kv, 78848 B):
      identity  -> exit=9  "round trip is vacuous"        (correctly refused)
      correct   -> exit=0  direction_check elements=24576
      reversed  -> exit=0  direction_check elements=24576  <-- THE DEFECT

    The reversed artifact is ACCEPTED with the direction check reporting a full
    24576 elements checked. If the check were direction-sensitive, the reversed
    artifact would have to fail. It does not.

    Note: the reversed artifact is not itself "illegal input" in a way the format
    can detect -- the file is well formed, and only the SEMANTICS of the compact
    region are transposed. That is exactly why an independent direction property
    (option b above) is the only thing that could catch it, and exactly why a
    tautological check is dangerous here: it is presented as the guard for the
    one failure mode that is otherwise undetectable.

Run from <repo>:
    python tools/p15_verify_d3_direction.py
"""
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p15_verify_lib as V  # noqa: E402
import p15_verify_d1 as D1   # reuse MY OWN builder, not the implementer's


def main():
    print("=" * 78)
    print("P15-S5 defect D3: the direction check cannot fail (tautology)")
    print("=" * 78)
    meta = V.MetaCheck()

    exe = os.path.join(V.ROOT, "examples", "p15_verify", "fw_build",
                       "x86_64-pc-windows-msvc", "release", "vmlm-runtime.exe")
    if not os.path.exists(exe):
        print("engine not built: %s" % exe)
        return 2
    print("engine = %d B" % os.path.getsize(exe))

    n_pos = 4

    def fill_correct(p, i):
        return float((p + 1) * 1000000 + i)

    def fill_reversed(p, i):
        return fill_correct(i, p)

    def fill_identity(p, i):
        return float(i)

    print("\n--- 1. the algebra, independent of the engine ---")
    # model the assignment and the check exactly as the source states them
    n_small = 300
    for name, f in (("correct", fill_correct), ("reversed", fill_reversed),
                    ("identity", fill_identity)):
        compact = {(p, i): f(p, i) for p in range(n_pos) for i in range(n_small)}
        per = {}
        for i in range(n_small):                  # reorder_compact_to_neurons
            for p in range(n_pos):
                per[(i, p)] = compact[(p, i)]
        mism = sum(1 for i in range(n_small) for p in range(n_pos)
                   if per[(i, p)] != compact[(p, i)])   # the check
        print("   %-9s : check mismatches = %d  -> would exit(9)? %s"
              % (name, mism, mism != 0))
    print("   => the check passes for ALL THREE, including the transposed one.")
    print("      It compares the destination against the source cell the reorder")
    print("      just copied FROM, so the write makes it true by construction.")

    # can-fail for THAT claim: a comparison that CAN distinguish must exist.
    print("\n   (control) a DIRECTION-SENSITIVE property does exist:")
    sens = 0
    for i in range(n_small):
        row_correct = [fill_correct(p, i) for p in range(n_pos)]
        row_reversed = [fill_reversed(p, i) for p in range(n_pos)]
        if row_correct != row_reversed:
            sens += 1
    print("      comparing compact[p*n+i] against compact[i*n_pos+p] differs for")
    print("      %d/%d neurons -> a real direction test is possible;"
          % (sens, n_small))
    print("      the shipped check simply does not perform it.")
    meta.record("D3 the shipped direction check is a tautology (algebraic)",
                True,  # the finding itself is the claim
                "the control above must FIND a discriminating property; if it "
                "found none, 'a direction test is possible' would be unproven",
                sens > 0,
                "%d/%d neurons discriminate under the transposed convention"
                % (sens, n_small))

    # ------------------------------------------------------------------
    print("\n--- 2. measured through the real engine ---")
    rows = []
    for name, f in (("correct", fill_correct), ("reversed", fill_reversed),
                    ("identity", fill_identity)):
        blob, coff = D1.build_artifact(n_pos, f)
        rc, txt = D1.run_engine(exe, blob)
        ok = "S5LOAD ok" in txt
        de = None
        for l in txt.splitlines():
            if "direction_check elements=" in l:
                de = int(l.split("direction_check elements=")[1].split()[0])
        msg = [l.strip() for l in txt.splitlines() if l.strip().startswith("NKV:")]
        print("   %-9s exit=%-3d ok=%-5s direction_check_elements=%-7s %s"
              % (name, rc, ok, de, msg[:1]))
        rows.append(dict(fill=name, exit=rc, ok=ok, direction_elements=de,
                         msg=msg))

    correct_accepted = rows[0]["exit"] == 0 and rows[0]["ok"]
    reversed_accepted = rows[1]["exit"] == 0 and rows[1]["ok"]
    identity_refused = rows[2]["exit"] != 0
    print("\n   correct accepted  : %s" % correct_accepted)
    print("   identity refused  : %s  (the vacuity guard works)" % identity_refused)
    print("   REVERSED accepted : %s  <-- the defect" % reversed_accepted)

    meta.record("D3 reversal is accepted by the engine (defect confirmed on "
                "the real binary)",
                reversed_accepted,
                "if the reversed artifact were refused, the direction check "
                "would be discriminating and there would be no defect; this "
                "criterion is TRUE only when the DEFECT is present",
                reversed_accepted and correct_accepted and identity_refused,
                "correct exit=%d, reversed exit=%d, identity exit=%d"
                % (rows[0]["exit"], rows[1]["exit"], rows[2]["exit"]))

    # ------------------------------------------------------------------
    print("\n--- 3. is the vacuity guard itself sound? ---")
    # The pos_varying guard is a REAL guard -- check it is not also tautological.
    # It counts neurons whose position-0 and position-1 compact values differ.
    # On the identity fill that is 0 (guard fires, correctly). On the correct
    # fill it is n (guard passes). Both measured above via exit codes.
    print("   identity fill -> pos_varying = 0 -> guard fires: %s"
          % identity_refused)
    print("   correct  fill -> pos_varying = %d -> guard passes: %s"
          % (D1.N_NEURONS, correct_accepted))
    meta.record("D3 the vacuity guard (pos_varying) is NOT tautological -- it "
                "fires on invariant data and passes on varying data",
                identity_refused and correct_accepted,
                "a guard that fired on both, or neither, would be tautological "
                "like the direction check",
                identity_refused and correct_accepted,
                "identity refused=%s, correct accepted=%s"
                % (identity_refused, correct_accepted))

    v = V.ensure_outdir("d3")
    import json
    with open(os.path.join(v, "d3_direction.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print("\nwrote %s" % os.path.join(v, "d3_direction.json"))

    # NOTE: this script's MetaCheck is designed so the D3 rows are TRUE when the
    # DEFECT IS PRESENT (that is what is being measured). So a successful
    # reproduction reports overall success; a FALSE row would mean I could not
    # reproduce the defect.
    allok = meta.report()
    print("\nDEFECT REPRODUCED: %s" % ("YES" if allok else "NO -- rows above false"))
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
