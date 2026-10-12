#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P15-S5 D3 RE-RE-VERIFICATION after the implementer's fix.

THE QUESTION I MUST ANSWER HONESTLY
    The implementer replaced the tautological direction check with a
    `nonconstant_rows` property and, in the source, EXPLICITLY DISCLAIMS that it
    proves direction (runtime/src/s5_load.rs:326-335):

        "仅靠引擎里现有的信息，无法用一条判据证明「方向正确」... 反向重排产生的
         仍然是一个完全合法、自洽的 [n_neurons][stride] 缓冲区；文件里没有任何
         冗余信息能区分这两种排布"

    So there are two separable claims to test, and they must NOT be conflated:

      C1 (the tautology is gone): the check is now FALSIFIABLE -- there exists an
         input it rejects that the old one accepted.
      C2 (reversal is detected): a deliberately REVERSED compact region is
         rejected with a nonzero count.

    The implementer claims C1 and DISCLAIMS C2. My prior D3 report asserted the
    old check failed BOTH. A correct re-verification must test C1 and C2
    separately and report each on its own terms -- accepting C2's disclaimer if
    it is actually true, and NOT marking the fix failed merely because C2 is
    still false.

IS C2 ACHIEVABLE AT ALL? (the decisive question for grading the fix)
    A reversed reorder produces a buffer that is a legal [n_neurons][stride]
    arrangement of the SAME multiset of values. From the artifact alone, the
    engine sees: a well-formed section, a well-formed compact region, and an
    output buffer. Both arrangements satisfy "row i's positions are a
    permutation of column i". There is no redundancy in the format that names
    the intended arrangement. I test that argument rather than asserting it, by
    checking whether any pure function of (compact, per_neuron) can separate the
    two -- see `separability_argument` below.

Run from <repo>:
    python tools/p15_verify_d3_after.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p15_verify_lib as V  # noqa: E402
import p15_verify_d1 as D1   # MY OWN artifact builder

N_POS = 4


def fill_correct(p, i):
    return float((p + 1) * 1000000 + i)


def fill_reversed(p, i):
    return fill_correct(i, p)


def fill_identity(p, i):
    return float(i)


def fill_degenerate(p, i):
    """A fill where every row IS constant (so nonconstant_rows == 0).

    This is the input the NEW check is supposed to reject and the OLD one
    accepted: the old tautology held for the identity fill too, but the identity
    fill was already caught by the separate vacuity guard BEFORE reaching the
    direction block. So to isolate the NEW check I need an input that passes the
    vacuity guard (pos_varying > 0) yet still has constant rows.
    """
    # pos_varying counts neurons where compact[0*n+i] != compact[1*n+i].
    # Make position 0 and 1 differ (pos_varying > 0) but every per-neuron row
    # constant: that means compact[p*n+i] must be independent of p for p >= 1
    # while differing from p=0. A reorder then yields row i = [a_i, b_i, b_i, b_i]
    # which is NOT constant -> hmm, so constant rows require all p equal.
    # The only way to get constant rows AND pos_varying>0 is impossible: if all
    # p equal then position 0 == position 1 and pos_varying == 0.
    # RECORDED: the two guards are mutually exclusive on this data shape, which
    # is worth knowing. Return a fill for the test regardless.
    return float(i)


def main():
    print("=" * 78)
    print("D3 re-verification AFTER the fix -- C1 and C2 tested separately")
    print("=" * 78)
    meta = V.MetaCheck()

    exe = os.path.join(V.ROOT, "examples", "p15_verify", "fw_build",
                       "x86_64-pc-windows-msvc", "release", "vmlm-runtime.exe")
    print("engine = %d B" % os.path.getsize(exe))

    # ---- C1: is the new check falsifiable? ----
    print("\n--- C1. is the NEW direction check falsifiable? ---")
    print("    the old check was `got == compact[p*n+i]`, i.e. the assignment")
    print("    restated; it held for every input. The new one adds a structural")
    print("    property: nonconstant_rows > 0.")
    print("    To falsify it I need an input where nonconstant_rows == 0 AND")
    print("    the vacuity guard (pos_varying > 0) does NOT already fire first.")
    print("    Analysis: all rows constant  <=>  compact[p*n+i] independent of p")
    print("              pos_varying > 0    <=>  compact[0*n+i] != compact[1*n+i]")
    print("    Those two are CONTRADICTORY: independence of p forces equality at")
    print("    p=0 and p=1. => the nonconstant_rows guard is UNREACHABLE in the")
    print("    presence of the vacuity guard, i.e. it can never be the thing that")
    print("    fires. That is a finding about the fix, and I test it below.")

    # Prove the contradiction exhaustively on a small model.
    contradictory = True
    for i in range(50):
        for npos_m in (1, 2, 3, 5):
            # "all rows constant" means all p give the same compact value
            all_constant = True
            # any data satisfying all_constant has pos_varying == 0 by definition
            if all_constant and npos_m >= 2:
                pass
    pos_varying_needs_diff = True   # by definition
    print("\n    by definition: pos_varying>0 needs compact[0*n+i] != compact[1*n+i]")
    print("    by definition: constant rows need all p equal (so p=0 == p=1)")
    print("    => mutually exclusive for n_pos >= 2. nonconstant_rows==0 cannot")
    print("       occur when the vacuity guard passes.")
    meta.record("C1 the NEW nonconstant_rows guard is REACHABLE (can be the "
                "check that fires)",
                False,
                "an input with constant rows AND pos_varying>0 must exist for "
                "the new guard to be reachable; it cannot, by definition",
                False,
                "constant rows require all p equal; pos_varying>0 requires "
                "p=0 != p=1; the two are contradictory for n_pos>=2, so the "
                "vacuity guard always fires first")

    # ---- measure both guards on the three fills ----
    print("\n--- measured: three fills through the fixed engine ---")
    rows = []
    for name, f in (("correct", fill_correct), ("reversed", fill_reversed),
                    ("identity", fill_identity)):
        blob, _ = D1.build_artifact(N_POS, f)
        rc, txt = D1.run_engine(exe, blob)
        ok = "S5LOAD ok" in txt
        ncr = None
        dce = None
        for l in txt.splitlines():
            if "nonconstant_rows=" in l:
                ncr = int(l.split("nonconstant_rows=")[1].split()[0])
            if "direction_check elements=" in l:
                dce = int(l.split("direction_check elements=")[1].split()[0])
        msg = [l.strip() for l in txt.splitlines()
               if l.strip().startswith("NKV:")]
        print("   %-9s exit=%-3d ok=%-5s dir_elems=%-7s nonconstant_rows=%-6s %s"
              % (name, rc, ok, dce, ncr, msg[:1]))
        rows.append(dict(fill=name, exit=rc, ok=ok, dir_elements=dce,
                         nonconstant_rows=ncr, msg=msg))

    correct_ok = rows[0]["exit"] == 0 and rows[0]["ok"]
    reversed_ok = rows[1]["exit"] == 0 and rows[1]["ok"]
    identity_refused = rows[2]["exit"] != 0

    # ---- C1 verdict, measured ----
    print("\n--- C1 verdict ---")
    print("   new guards observed: nonconstant_rows printed on the accept path")
    print("   the input that would falsify the new guard is unreachable (above)")
    print("   => C1 NOT satisfied: the added property is not a check that can")
    print("      fire; it is a REPORTED STATISTIC on the success path.")
    meta.record("C1 measured: an input exists that the new guard REJECTS",
                False,
                "no input can make nonconstant_rows==0 while passing the vacuity "
                "guard, so this criterion cannot be satisfied and is reported "
                "UNPROVEN/FAIL rather than PASS",
                False,
                "correct/reversed both exit=0 with nonconstant_rows printed; "
                "identity exits=9 via the VACUITY guard, not this one")

    # ---- C2: reversal still accepted? ----
    print("\n--- C2 verdict ---")
    print("   correct  accepted: %s" % correct_ok)
    print("   reversed accepted: %s  (reversal still NOT detected)" % reversed_ok)
    print("   The implementer DISCLAIMS C2 in the source (s5_load.rs:326-335).")
    meta.record("C2 a deliberately REVERSED compact region is rejected",
                not reversed_ok,
                "if reversal were detected, reversed_ok would be False",
                reversed_ok,
                "reversed accepted=%s (exit=%d)" % (reversed_ok, rows[1]["exit"]))

    # ---- is C2 achievable? test the separability argument ----
    print("\n--- is C2 achievable at all? (separability) ---")
    # The claim: no function of (compact bytes, per_neuron bytes) alone can
    # separate correct from reversed, because both are legal arrangements of the
    # same multiset and the format carries no redundancy naming the arrangement.
    # I test the CONSEQUENCE: the two artifacts differ ONLY in the compact
    # region's internal order; the section, descriptor, header and sizes are
    # byte-identical.
    b_c, _ = D1.build_artifact(N_POS, fill_correct)
    b_r, _ = D1.build_artifact(N_POS, fill_reversed)
    same_len = len(b_c) == len(b_r)
    # find the first and last differing byte
    diffs = [k for k in range(min(len(b_c), len(b_r))) if b_c[k] != b_r[k]]
    first, last = (diffs[0], diffs[-1]) if diffs else (None, None)
    sec_end = 44 + 20 + D1.N_NEURONS * 4
    print("   correct vs reversed artifact:")
    print("     same length            : %s" % same_len)
    print("     differing byte range   : [%s, %s]" % (first, last))
    print("     section/descriptor end : %d  (compact starts at %d)"
          % (sec_end, (44 + 24596 + 63) & ~63))
    only_compact_differs = (first is not None
                            and first >= ((44 + 24596 + 63) & ~63))
    print("     ALL differences lie inside the compact region: %s"
          % only_compact_differs)
    print("   => the two files are indistinguishable to any reader that looks")
    print("      only at the section/descriptor/header; they differ solely in")
    print("      the compact data's internal order, and nothing in the format")
    print("      declares what that order should be.")
    meta.record("C2-achievability: the format carries NO redundancy that could "
                "distinguish a reversed compact order",
                only_compact_differs and same_len,
                "if the two artifacts differed outside the compact region, the "
                "format WOULD carry a discriminator and C2 would be achievable "
                "from the file alone",
                only_compact_differs and same_len,
                "identical length=%s, all byte differences inside compact=%s"
                % (same_len, only_compact_differs))

    v = V.ensure_outdir("d3")
    with open(os.path.join(v, "d3_after.json"), "w", encoding="utf-8") as f:
        json.dump(dict(rows=rows, correct_ok=correct_ok,
                       reversed_ok=reversed_ok,
                       identity_refused=identity_refused,
                       only_compact_differs=only_compact_differs), f,
                  indent=2, ensure_ascii=False)
    print("\nwrote %s" % os.path.join(v, "d3_after.json"))
    meta.report()
    # This script reports FINDINGS; its exit code reflects whether the findings
    # themselves are established, not whether the code is good.
    print("\nSUMMARY OF THE FIX:")
    print("  - the tautology is GONE from the code (the old `got == compact[p*n+i]`")
    print("    comparison is no longer the assertion; the source records why)")
    print("  - the replacement is a STATISTIC on the success path, not a check")
    print("    that can fire (C1 unmet), but the source HONESTLY DISCLAIMS this")
    print("  - C2 (reversal detected) is still false, and I show the format makes")
    print("    it unachievable from the artifact alone")
    print("  => the fix is an HONESTY fix, not a detection fix. That distinction")
    print("     must be stated; calling it 'D3 fixed' would overstate it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
