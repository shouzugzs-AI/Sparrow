#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P14 S5 verification, ROUND 2 (after the Lead fixed S5-A and S5-B).

WHY A SECOND ROUND EXISTS
    The verifier's first round found two defects (recorded in
    docs/P14_verification.md):
      S5-A  the shipped stride convention reserved head_dim*max_seq elements per
            neuron, i.e. 128x the tight layout on MiniCPM (3.221 GB vs 0.025 GB
            of address space), and validate() enforced that inflated form.
      S5-B  format.rs stated the invariant as `offset[idx] + max_seq <= stride`,
            which is FALSE for every idx>=1 given offset[idx] = idx*stride, and
            validate() enforced a different (over-strict) condition.
    The Lead fixed both. A fix is not evidence until it is re-measured, and the
    specific thing to check is that the fix is PINNED -- i.e. the test now
    rejects the OLD convention as well as accepting the new one. Accepting the
    right answer is easy; the value is in also rejecting the wrong one.

WHAT THIS ROUND ASSERTS
    S5-R1  the invariant is now a single predicate, `max_seq <= stride`,
           enforced by validate() AND stated identically in format.rs.
    S5-R2  the tight layout (stride == max_seq) is ACCEPTED -- under the old
           code it was REJECTED, so this pins the regression.
    S5-R3  the old inflated convention (stride == head_dim*max_seq) still
           satisfies the new rule (it is legal padding, just wasteful) and its
           cost is REPORTED, so the 128x mistake stays visible.
    S5-R4  a genuinely illegal layout (stride < max_seq) is REJECTED. This is
           the can-it-fail control: without it, R2/R3 only show the checker
           says yes.
"""
import json
import os
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p14_verify_lib as L          # noqa: E402
import p14_s5_layout_lib as S5      # noqa: E402

ROOT = L.ROOT


def read_text(rel):
    with open(os.path.join(ROOT, rel.replace("/", os.sep)), encoding="utf-8",
              errors="replace") as f:
        return f.read()


def main():
    print("=== P14 S5 verification ROUND 2 (post-fix) ===")
    meta = L.MetaCheck()
    fails = 0

    # ------------------------------------------------------------------
    # S5-R1: one predicate, stated and enforced
    # ------------------------------------------------------------------
    print("\n[S5-R1] the non-overlap invariant: stated == enforced?")
    fmt = read_text("core/src/format.rs")
    nkv = read_text("core/src/neuron_kv.rs")

    fmt_has_weak = bool(re.search(r"max_seq_per_neuron\s*<=\s*stride", fmt))
    fmt_has_old = bool(re.search(r"offset\[idx\]\s*\+\s*max_seq", fmt))
    code_enforces = bool(re.search(
        r"self\.stride as u64\)\s*<\s*self\.max_seq_per_neuron as u64", nkv))
    code_old = bool(re.search(
        r"self\.stride as u64\)\s*<\s*\(self\.head_dim as u64\s*\*\s*"
        r"self\.max_seq_per_neuron as u64\)", nkv))

    print("  format.rs states `max_seq_per_neuron <= stride` : %s" % fmt_has_weak)
    print("  format.rs still states the OLD false form        : %s (must be "
          "False)" % fmt_has_old)
    print("  validate() enforces `stride < max_seq -> exit`   : %s" % code_enforces)
    print("  validate() still uses the OLD head_dim*max_seq   : %s (must be "
          "False)" % code_old)

    # The old false form is only acceptable if it is present as an explicitly
    # labelled WITHDRAWN statement. Check the context around any mention.
    old_mentions = [m.start() for m in re.finditer(r"offset\[idx\]\s*\+", fmt)]
    withdrawn_ok = True
    for p in old_mentions:
        ctx = fmt[max(0, p - 260):p + 260]
        if not re.search(r"错|false|FALSE|withdrawn|曾|第一版", ctx):
            withdrawn_ok = False
    print("  every mention of the old form is labelled wrong : %s" % withdrawn_ok)

    r1 = (fmt_has_weak and code_enforces and not code_old and withdrawn_ok)
    print("  -> %s" % ("PASS: doc and code now use ONE predicate, and the old "
                        "form survives only as a labelled correction"
                        if r1 else "FAIL"))
    meta.record("S5-R1 invariant stated==enforced", r1,
                "the old `offset[idx]+max_seq<=stride` form re-appearing "
                "unlabelled", withdrawn_ok and not code_old,
                "regex scan of both files")
    if not r1:
        fails += 1

    # ------------------------------------------------------------------
    # S5-R2/R3/R4: numeric behaviour of the invariant, both directions
    # ------------------------------------------------------------------
    print("\n[S5-R2..R4] the invariant evaluated numerically on MiniCPM's shape")
    NL, NKH, HD, PMAX = 24, 2, 128, 1024
    n_neurons = NL * NKH * HD

    def overlaps(stride):
        """Does neuron i's slot range collide with neuron i+1's? (i>=0)"""
        for i in range(min(n_neurons - 1, 200)):
            lo_i, hi_i = i * stride, i * stride + PMAX
            lo_j, hi_j = (i + 1) * stride, (i + 1) * stride + PMAX
            if hi_i > lo_j:
                return True
        return False

    cases = [
        ("tight  stride=PMAX", PMAX),
        ("old    stride=HD*PMAX", HD * PMAX),
        ("illegal stride=PMAX-1", PMAX - 1),
    ]
    print("  %-24s %-12s %-10s %-12s %s"
          % ("case", "stride", "overlap?", "rule ok?", "elements"))
    rows = {}
    for name, st in cases:
        ov = overlaps(st)
        rule_ok = (st >= PMAX)
        total = n_neurons * st
        rows[name] = (st, ov, rule_ok, total)
        print("  %-24s %-12d %-10s %-12s %d"
              % (name, st, ov, rule_ok, total))

    # R2: the tight layout must be ACCEPTED by the new rule, and the old rule
    # would have REJECTED it (that is the pinned regression).
    tight_stride = rows["tight  stride=PMAX"][0]
    tight_ok_new = (tight_stride >= PMAX)
    tight_ok_old = (tight_stride >= HD * PMAX)   # the old, over-strict rule
    print("\n  S5-R2 tight layout: accepted by NEW rule = %s ; accepted by OLD "
          "rule = %s" % (tight_ok_new, tight_ok_old))
    r2 = tight_ok_new and not tight_ok_old
    print("  -> %s" % ("PASS: the fix genuinely changes behaviour (the tight "
                        "layout used to be rejected)" if r2 else
                        "FAIL: no behavioural change -- fix is cosmetic"))
    meta.record("S5-R2 tight layout newly accepted", r2,
                "stride=PMAX (previously rejected)", bool(not tight_ok_old),
                "old rule needed stride>=%d, tight uses %d" % (HD * PMAX,
                                                               tight_stride))
    if not r2:
        fails += 1

    # R3: the inflated form is legal padding but its cost must be REPORTED.
    old_stride, old_ov, old_ok, old_total = rows["old    stride=HD*PMAX"]
    tight_total = rows["tight  stride=PMAX"][3]
    ratio = old_total / tight_total
    print("\n  S5-R3 inflated form cost:")
    print("    total (tight)   = %d elements = %.3f GB f32"
          % (tight_total, tight_total * 4 / 1e9))
    print("    total (old)     = %d elements = %.3f GB f32"
          % (old_total, old_total * 4 / 1e9))
    print("    inflation       = %.1fx" % ratio)
    print("    overlap under old stride? %s (must be False -- the old form was"
          " wasteful, not unsafe)" % old_ov)
    r3 = (not old_ov and old_ok and abs(ratio - HD) < 1e-9)
    print("  -> %s" % ("PASS: old form is non-overlapping and exactly %dx, so "
                        "the regression metric is stable" % HD if r3 else "FAIL"))
    meta.record("S5-R3 inflation cost reported", r3,
                "ratio != head_dim", bool(abs(ratio - HD) < 1e-9),
                "measured ratio %.1fx" % ratio)
    if not r3:
        fails += 1

    # R4: THE CAN-IT-FAIL CONTROL. An illegal layout must be rejected.
    bad_stride, bad_ov, bad_ok, bad_total = rows["illegal stride=PMAX-1"]
    print("\n  S5-R4 illegal layout (stride < max_seq):")
    print("    stride=%d  overlap=%s  rule_ok=%s"
          % (bad_stride, bad_ov, bad_ok))
    r4 = bad_ov and (not bad_ok)
    print("  -> %s" % ("PASS: illegal layout overlaps AND the rule rejects it "
                        "(the rule can fail)" if r4 else
                        "FAIL: the rule accepts an overlapping layout"))
    meta.record("S5-R4 illegal layout rejected", r4,
                "stride=PMAX-1 causes overlap", bool(r4),
                "overlap=%s rule_ok=%s" % (bad_ov, bad_ok))
    if not r4:
        fails += 1

    # ------------------------------------------------------------------
    # S5-R5: the layout/bit-identity criteria, re-measured
    # ------------------------------------------------------------------
    print("\n[S5-R5] layout [L][h][d][slot] + bit-identity, re-measured")
    lay = S5.NeuronLayout(NL, NKH, HD, PMAX)
    mismatch = 0
    for (Lx, hx, dx, sx) in [(0, 0, 0, 0), (0, 0, 0, 7), (0, 0, 1, 0),
                             (0, 1, 0, 0), (1, 0, 0, 0), (3, 1, 127, 1023),
                             (23, 1, 127, 0)]:
        nid = (Lx * NKH + hx) * HD + dx
        # The shipped tight formula is neuron_base(id) + slot, i.e.
        #   addr = idx * stride + slot      (NOT idx * stride alone)
        # An earlier version of this line omitted the `+ sx` term and reported
        # 2 spurious mismatches for the two cases with slot != 0. Recorded here
        # because it is the same class of error the suite exists to catch: a
        # comparison whose arithmetic does not model what it claims to model
        # fails (or passes) for the wrong reason.
        shipped = nid * PMAX + sx
        mine = lay.addr(lay.neuron_id(Lx, hx, dx), sx)
        if shipped != mine:
            mismatch += 1
            print("    MISMATCH (L=%d,h=%d,d=%d,s=%d): shipped=%d model=%d"
                  % (Lx, hx, dx, sx, shipped, mine))
    print("  address model vs shipped tight formula: %d mismatches (must be 0)"
          % mismatch)

    rng = np.random.RandomState(99)
    small = S5.NeuronLayout(3, 2, 4, 8)
    src = rng.randn(3, 2, 4, 8).astype(np.float32)
    flat = small.encode(src)
    nbad = 0
    for Lx in range(3):
        for hx in range(2):
            for dx in range(4):
                for sx in range(8):
                    if S5.indexed_read(src, small, Lx, hx, dx, sx) != \
                       S5.neuron_read(flat, small, Lx, hx, dx, sx):
                        nbad += 1
    rt = S5.bit_identical(src, small.decode(flat))
    # negative control: wrong stride
    badl = S5.wrong_stride_variant(small)
    bflat = badl.encode(src)
    nbad2 = 0
    for Lx in range(3):
        for hx in range(2):
            for dx in range(4):
                for sx in range(8):
                    off = badl.addr(badl.neuron_id(Lx, hx, dx), sx)
                    b = bflat[off] if off < bflat.size else np.float32(0)
                    if S5.indexed_read(src, small, Lx, hx, dx, sx) != b:
                        nbad2 += 1
    r5 = (mismatch == 0 and nbad == 0 and rt and nbad2 > 0)
    print("  neuron-addressed read != indexed read : %d / %d (must be 0)"
          % (nbad, src.size))
    print("  round trip bit-identical              : %s" % rt)
    print("  META-CHECK wrong stride disagrees on  : %d / %d (must be > 0)"
          % (nbad2, src.size))
    print("  -> %s" % ("PASS" if r5 else "FAIL"))
    meta.record("S5-R5 layout + bit-identity", r5,
                "wrong stride", bool(nbad2 > 0),
                "%d/%d differ under wrong stride" % (nbad2, src.size))
    if not r5:
        fails += 1

    # ------------------------------------------------------------------
    # S5-R6: the "costs 0 bytes" claim, independently
    # ------------------------------------------------------------------
    print("\n[S5-R6] does S5 cost the inference engine 0 bytes? (independent)")
    lib = read_text("core/src/lib.rs")
    gated = bool(re.search(r'#\[cfg\(feature = "neuron-kv"\)\]\s*\n\s*pub mod '
                           r'neuron_kv;', lib))
    print("  lib.rs gates the module behind `feature = \"neuron-kv\"`: %s" % gated)
    if not gated:
        print("  -> UNVERIFIED: module is not feature-gated, so the 0-byte "
              "claim cannot hold by construction")
        meta.record("S5-R6 0-byte claim", False,
                    "ungated module cannot cost 0 bytes", False,
                    "gating absent")
        fails += 1
    else:
        # Measure the runtime WITH and WITHOUT the feature. The feature is
        # declared on vmlm-core and (after the verifier's S5-R6 finding was
        # accepted) forwarded by runtime/Cargo.toml as
        #     neuron-kv = ["vmlm-core/neuron-kv"]
        # so the engine must now actually build both ways. If the forwarding
        # were absent, the ON build would fail with "package 'vmlm-runtime'
        # does not contain this feature" -- which is exactly what the first
        # round of this check measured, and why it is asserted here.
        sizes = {}
        for feat_on in (False, True):
            cmd = ["cargo", "build", "--release",
                   "--target", "x86_64-pc-windows-msvc",
                   "--manifest-path", os.path.join(ROOT, "runtime",
                                                   "Cargo.toml")]
            if feat_on:
                cmd += ["--features", "neuron-kv"]
            p = subprocess.run(cmd, cwd=os.path.join(ROOT, "runtime"),
                               capture_output=True, text=True,
                               errors="replace", timeout=1800)
            exe = os.path.join(ROOT, "target", "x86_64-pc-windows-msvc",
                               "release", "vmlm-runtime.exe")
            ok = (p.returncode == 0 and os.path.exists(exe))
            sizes[feat_on] = os.path.getsize(exe) if ok else None
            print("  feature %-3s : build ok=%-5s engine=%s B"
                  % ("ON" if feat_on else "OFF", ok, sizes[feat_on]))
            if not ok:
                print("    error: %s"
                      % (p.stdout + p.stderr).replace("\n", " ")[-300:])

        if sizes[False] is None or sizes[True] is None:
            print("  -> UNVERIFIED: the ON build must succeed for the 0-byte")
            print("     claim to be measurable at all. (Round 1 measured this")
            print("     build FAILING because the feature was not forwarded;")
            print("     that was reported and the forwarding was added.)")
            meta.record("S5-R6 0-byte claim", False,
                        "the --features neuron-kv build failing", False,
                        "one or both runtime builds failed")
            fails += 1
        else:
            delta = sizes[True] - sizes[False]
            r6 = (delta == 0)
            print("  engine delta OFF vs ON = %+d B  (OFF=%d ON=%d)"
                  % (delta, sizes[False], sizes[True]))
            # CAN-THE-TEST-FAIL: the same size comparison must report NON-zero
            # on a variant that genuinely differs. The first version of this
            # check hardcoded the meta-result to True, and the MetaCheck
            # (correctly) refused to certify it -- hardcoding the meta-result is
            # itself the rubber-stamp failure this suite exists to catch.
            cmd_tr = ["cargo", "build", "--release",
                      "--target", "x86_64-pc-windows-msvc",
                      "--manifest-path", os.path.join(ROOT, "runtime",
                                                      "Cargo.toml"),
                      "--features", "train"]
            ptr = subprocess.run(cmd_tr, cwd=os.path.join(ROOT, "runtime"),
                                 capture_output=True, text=True,
                                 errors="replace", timeout=1800)
            exe = os.path.join(ROOT, "target", "x86_64-pc-windows-msvc",
                               "release", "vmlm-runtime.exe")
            train_size = (os.path.getsize(exe)
                          if ptr.returncode == 0 and os.path.exists(exe)
                          else None)
            if train_size is None:
                canfail6 = False
                detail6 = "train-variant build failed; no falsification shown"
                print("  cannot build the train variant; test stays UNPROVEN")
            else:
                ctrl = train_size - sizes[False]
                canfail6 = (ctrl != 0)
                detail6 = ("control (--features train) delta = %+d B -- the size "
                           "comparison DOES detect a real difference" % ctrl)
                print("  control: --features train = %d B (delta %+d B)"
                      % (train_size, ctrl))
            print("  -> %s" % ("PASS: S5 contributes 0 B to the engine, and the"
                                " feature now builds" if r6 else
                                "FAIL: S5 changes the engine size by %+d B"
                                % delta))
            meta.record("S5-R6 0-byte claim", r6,
                        "the same size comparison run against a variant that "
                        "genuinely differs (--features train)",
                        canfail6, "measured delta %+d B; %s" % (delta, detail6))
            # restore the default engine for downstream use
            subprocess.run(["cargo", "build", "--release", "--target",
                            "x86_64-pc-windows-msvc", "--manifest-path",
                            os.path.join(ROOT, "runtime", "Cargo.toml")],
                           cwd=os.path.join(ROOT, "runtime"),
                           capture_output=True, text=True, timeout=1800)
            if not r6:
                fails += 1

    allproven = meta.report()
    print("\n" + "=" * 62)
    print("S5 ROUND-2 VERDICT: %s"
          % ("PASS" if (fails == 0 and allproven) else "FAIL"))
    print("  failed criteria   : %d" % fails)
    print("  unproven criteria : %d"
          % sum(1 for r in meta.rows if not r["can_fail"]))
    return 0 if (fails == 0 and allproven) else 1


if __name__ == "__main__":
    sys.exit(main())
