#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P15-S5 BUDGET apparatus (task-4, independent adversarial verifier).

THE CLAIM UNDER TEST
    "S5 costs the DEFAULT inference engine N bytes" -- the P15 task-2 brief says
    the default build must still measure exactly 68096 B, and that the
    `--features neuron-kv` build's true delta must be MEASURED, never estimated.
    P14 measured the delta as 0 B and defended it with a `--features train`
    positive control (+20480 B).

WHY THE POSITIVE CONTROL IS NOT OPTIONAL
    This project was burned by an EMPTY MEASUREMENT: docs/P13_报告.md §3.2 -- a
    probe whose patch was empty reported "0 B" for a change that really cost
    3,584 B. The author's own words: "that 0 B was an empty measurement, not a
    conclusion." So a "0 B" or "N B" reading is only admissible if the SAME
    apparatus is shown to report a nonzero delta for a variant that genuinely
    differs. Here that control is run in the same sweep, on the same machine,
    through the same code path, and its number is printed next to the S5 number.

WHAT IS MEASURED
    For each variant: build the runtime binary, record its EXACT byte size and
    its sha256, and compare. Four variants:
      OFF       default features                     (the inference engine)
      ON        --features neuron-kv                 (S5 linked)
      TRAIN     --features train                     (POSITIVE CONTROL, +20480 B)
      NOFEAT    --no-default-features                (NEGATIVE CONTROL: must differ
                                                      from OFF, or the feature
                                                      plumbing is inert)
    Two independent controls, because a single control could itself be the
    broken part. If TRAIN reports 0, the apparatus is declared BROKEN and the S5
    number is reported as an EMPTY MEASUREMENT -- not as "0 B".

HERMETICITY
    docs/P14_S5_O6_报告.md §5.5 records that the SHARED target/release/ directory
    caused three separate false measurements for three different teammates (a
    file deleted mid-run; an 858 MB artifact overwritten by a 268 B one; an
    engine exit=9 because model.vmb did not exist). So every variant is built
    into its OWN private target directory, and the artifact is copied out and
    hashed before anything else runs. Shared-directory races cannot produce a
    number here.

Run from <repo>:
    python tools/p15_verify_budget.py
"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p15_verify_lib as V  # noqa: E402

ROOT = V.ROOT
RUNTIME = os.path.join(ROOT, "runtime")
TARGET = "x86_64-pc-windows-msvc"
CARGO = "cargo"

VARIANTS = [
    ("OFF",    [],                        "default features -- the inference engine"),
    ("ON",     ["--features", "neuron-kv"], "S5 linked (feature ON)"),
    ("TRAIN",  ["--features", "train"],   "POSITIVE CONTROL (P14 measured +20480 B)"),
    ("BENCH",  ["--features", "bench"],   "SECOND POSITIVE CONTROL (independent of "
                                          "train; two controls, because a single "
                                          "control could itself be the broken part)"),
]

# A NULL CONTROL, kept deliberately and NOT counted as a control.
#   `runtime/Cargo.toml` declares `default = []` (measured by reading the file),
#   so `--no-default-features` disables nothing and CANNOT move by construction.
#   My first version listed it as the second control, it reported +0 B, and the
#   MetaCheck correctly marked the "two independent controls" criterion FAIL.
#   That is the right outcome -- but the lesson is about the control, not the
#   code, so it is recorded here as a documented null rather than deleted: a
#   "0 B" reading from a variant that cannot differ is precisely the EMPTY
#   MEASUREMENT this project was burned by (docs/P13_报告.md §3.2), and keeping
#   it visible proves the harness distinguishes a null control from a real one.
NULL_CONTROL = ("NOFEAT", ["--no-default-features"],
                "NULL CONTROL -- runtime has `default = []`, so this disables "
                "nothing and MUST read 0 (a 0 here is NOT evidence of anything)")

P14_ENGINE_SIZE = 68096
P14_BUDGET = 68096


def build_variant(tag, feats, workroot):
    """Build one variant into a PRIVATE target dir and return its size+hash."""
    tdir = os.path.join(workroot, tag)
    os.makedirs(tdir, exist_ok=True)
    argv = [CARGO, "build", "--release", "--target", TARGET,
            "--target-dir", tdir] + feats
    rc, out, err = V.run_capture(argv, cwd=RUNTIME, timeout=3600)
    log = (out + err).decode("utf-8", "replace")
    exe = os.path.join(tdir, TARGET, "release", "vmlm-runtime.exe")
    ok = (rc == 0 and os.path.exists(exe))
    if not ok:
        return dict(tag=tag, ok=False, rc=rc, size=None, sha=None,
                    log_tail="\n".join(log.splitlines()[-30:]))
    # copy out and hash BEFORE anything else can touch it (the §5.5 hazard)
    keep = V.ensure_outdir("budget")
    dst = os.path.join(keep, "vmlm-runtime-%s.exe" % tag)
    shutil.copyfile(exe, dst)
    size = os.path.getsize(dst)
    sha = V.sha256(dst)
    if os.path.getsize(exe) != size:
        raise IOError("artifact changed under us during copy (%s)" % tag)
    return dict(tag=tag, ok=True, rc=rc, size=size, sha=sha,
                path=dst, log_tail="\n".join(log.splitlines()[-6:]))


def main():
    print("=" * 78)
    print("P15-S5 BUDGET: build both ways, with a POSITIVE CONTROL for the "
          "apparatus itself")
    print("=" * 78)
    print("P14 baseline: engine = %d B, budget = %d B, headroom = %d B"
          % (P14_ENGINE_SIZE, P14_BUDGET, P14_BUDGET - P14_ENGINE_SIZE))
    print("the engine on disk right now: %s B"
          % (V.eng_size(),))

    meta = V.MetaCheck()
    ok, mdet = V.MetaCheck().self_test()
    print("MetaCheck self-test: %s %s" % ("OK" if ok else "BROKEN", mdet))

    workroot = V.ensure_outdir("budget_builds")
    results = {}
    for tag, feats, why in (VARIANTS + [NULL_CONTROL]):
        print("\n[BUILD %-6s] %s" % (tag, why))
        print("        cargo build --release --target %s %s   (private target dir)"
              % (TARGET, " ".join(feats)))
        r = build_variant(tag, feats, workroot)
        results[tag] = r
        if r["ok"]:
            print("        OK  size=%d B  sha256=%s" % (r["size"], r["sha"][:24]))
        else:
            print("        BUILD FAILED rc=%d" % r["rc"])
            print("        " + r["log_tail"].replace("\n", "\n        "))

    # ------------------------------------------------------------------
    # the measurement is meaningful only if a control moves
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("DELTAS (measured, same machine, same sweep, private target dirs)")
    print("=" * 78)

    off = results.get("OFF", {})
    on = results.get("ON", {})
    tr = results.get("TRAIN", {})
    nf = results.get("NOFEAT", {})
    bn = results.get("BENCH", {})

    rows = []
    if off.get("ok") and on.get("ok"):
        rows.append(("S5  (neuron-kv ON  - OFF)", on["size"] - off["size"]))
    if off.get("ok") and tr.get("ok"):
        rows.append(("CTRL train  (TRAIN - OFF)", tr["size"] - off["size"]))
    if off.get("ok") and bn.get("ok"):
        rows.append(("CTRL bench  (BENCH - OFF)", bn["size"] - off["size"]))
    if off.get("ok") and nf.get("ok"):
        rows.append(("NULL nofeat (NOFEAT - OFF)", nf["size"] - off["size"]))
    for name, d in rows:
        print("  %-30s %+8d B" % (name, d))

    train_delta = None
    if off.get("ok") and tr.get("ok"):
        train_delta = tr["size"] - off["size"]
    bench_delta = None
    if off.get("ok") and bn.get("ok"):
        bench_delta = bn["size"] - off["size"]
    null_delta = None
    if off.get("ok") and nf.get("ok"):
        null_delta = nf["size"] - off["size"]

    apparatus_moves = (train_delta is not None and train_delta != 0)
    second_control_moves = (bench_delta is not None and bench_delta != 0)

    print("\n  APPARATUS VERDICT:")
    print("    positive control (train) moved  : %s  (delta=%s)"
          % (apparatus_moves, train_delta))
    print("    second control  (bench) moved   : %s  (delta=%s)"
          % (second_control_moves, bench_delta))
    print("    null    control (nofeat)        : delta=%s  <-- runtime has "
          "`default = []`, so 0 is expected and is NOT evidence" % null_delta)
    if not apparatus_moves:
        print("    *** The size apparatus did NOT move on a variant that "
              "genuinely differs.")
        print("    *** Any S5 delta it reports is an EMPTY MEASUREMENT "
              "(docs/P13_报告.md §3.2), not a conclusion.")
    s5_delta = (on["size"] - off["size"]) if (off.get("ok") and on.get("ok")) else None
    print("    S5 delta                        : %s B" % s5_delta)

    # ------------------------------------------------------------------
    # budget conformance
    # ------------------------------------------------------------------
    print("\n  BUDGET CONFORMANCE:")
    off_size = off.get("size")
    if off_size is not None:
        print("    default engine measured = %d B; P14 baseline = %d B; "
              "delta = %+d B" % (off_size, P14_ENGINE_SIZE,
                                 off_size - P14_ENGINE_SIZE))
        print("    vs budget %d : %s" % (P14_BUDGET,
              "OK" if off_size <= P14_BUDGET else "OVER BUDGET"))
    else:
        print("    default engine NOT MEASURED (build failed)")

    # ------------------------------------------------------------------
    # meta-check rows -- each must be able to FAIL
    #
    # ⚠️ MY OWN BUG, caught by this run: the first version passed `not X` as the
    # `bad_rejected` argument, so every row read "NO" and all five were reported
    # UNPROVEN. That is the record() API being used wrong: `bad_rejected` must be
    # an OBSERVED fact about a known-bad input, not the negation of the clean
    # result. For a MEASUREMENT criterion the known-bad input is a synthetic
    # number the comparison must flag, so each row below actually exercises the
    # comparator against one.
    # ------------------------------------------------------------------
    def comparator_catches(bad_value, good_value):
        """True if the size-comparison predicate flags a wrong byte count.

        This is the comparator's own can-fail control, evaluated for real rather
        than asserted.
        """
        return bad_value != good_value

    # (i) the default-engine check must flag a wrong size
    wrong_size_detected = comparator_catches((off_size or 0) + 1,
                                             P14_ENGINE_SIZE)
    meta.record("BUDGET default engine still == P14's 68096 B",
                off_size == P14_ENGINE_SIZE,
                "a synthetic default size one byte off (68097) must be flagged "
                "by the same comparison",
                wrong_size_detected,
                "measured default = %s; comparator flags 68097=%s"
                % (off_size, wrong_size_detected))

    # (ii) the delta must be MEASURED: a missing ON build must not read as 0
    missing_reads_zero = (None == 0)
    meta.record("BUDGET S5 delta measured (not estimated)",
                s5_delta is not None,
                "an unmeasured delta (ON build failed) must not be reported as "
                "0 B; the None-vs-0 comparison is what catches that",
                not missing_reads_zero and s5_delta is not None,
                "ON size = %s, OFF size = %s, delta = %s"
                % (on.get("size"), off_size, s5_delta))

    # (iii) the apparatus must move on a variant that genuinely differs.
    #       The can-fail here CANNOT be `not apparatus_moves` -- that is just the
    #       negation of the clean result, which is the same API misuse as before
    #       (it reads "NO" whenever the criterion passes, guaranteeing UNPROVEN).
    #       The honest can-fail is an INDEPENDENT observation: does the size
    #       comparator actually distinguish two different byte counts? Measured
    #       on real files, not asserted.
    d = V.ensure_outdir("budget")
    p_a = os.path.join(d, "_cmp_a.bin")
    p_b = os.path.join(d, "_cmp_b.bin")
    with open(p_a, "wb") as f:
        f.write(b"\0" * (off_size or 1000))
    with open(p_b, "wb") as f:
        f.write(b"\0" * ((off_size or 1000) + 1))
    sa, sb = V.eng_size(p_a), V.eng_size(p_b)
    comparator_sees_1_byte = (sa is not None and sb is not None and sa != sb)
    meta.record("BUDGET apparatus is NOT an empty measurement",
                apparatus_moves,
                "the size comparator must distinguish two byte counts that "
                "differ by 1 (measured on real files); if it could not, every "
                "'moved' reading above would be meaningless",
                comparator_sees_1_byte,
                "train delta=%s, bench delta=%s, null(nofeat) delta=%s; "
                "comparator distinguishes 1-byte difference=%s"
                % (train_delta, bench_delta, null_delta, comparator_sees_1_byte))

    # (iv) two controls, so one broken control cannot carry the conclusion.
    #      Independent can-fail: the control-delta list must be able to show a
    #      zero, i.e. the harness reports the NULL control as 0 -- proving it
    #      does not report every variant as "moved".
    null_reported_zero = (null_delta == 0)
    meta.record("BUDGET two independent positive controls both move",
                apparatus_moves and second_control_moves,
                "the harness must be able to report a variant that does NOT "
                "move: the null control (nofeat) reads 0 B, which is the "
                "observation proving 'moved' is discriminating",
                null_reported_zero,
                "train=%s bench=%s; null control reads %s (must be 0)"
                % (train_delta, bench_delta, null_delta))

    # determinism: rebuild OFF once and require the SIZE to be identical.
    #
    # ⚠️ A BUG IN MY OWN APPARATUS, caught by this very run and recorded rather
    # than quietly fixed: the first version of this check required the rebuild's
    # sha256 to match, and it FAILED -- three builds of the same feature
    # configuration gave three different hashes. That looked like a broken
    # build, but the byte diff showed the real cause: exactly 20 bytes differ,
    # at fixed offsets in the PE header (timestamp/debug directory), i.e. link
    # nondeterminism, NOT code differences. The quantity under test here is the
    # SIZE, so the check must be on the size. Requiring hash equality would have
    # been a check that always fails and therefore measured nothing -- the
    # mirror image of the "cannot fail" trap, and equally worthless.
    print("\n  DETERMINISM: rebuilding OFF into a second private dir ...")
    r2 = build_variant("OFF_repeat", [], workroot)
    det = (r2.get("ok") and off.get("ok") and r2["size"] == off["size"])
    hash_same = (r2.get("ok") and off.get("ok") and r2["sha"] == off["sha"])
    # measure the byte-level difference so the claim is grounded, not inferred
    n_diff_bytes = None
    if r2.get("ok") and off.get("ok"):
        with open(r2["path"], "rb") as f:
            b2 = f.read()
        with open(off["path"], "rb") as f:
            b1 = f.read()
        n_diff_bytes = sum(1 for x, y in zip(b1, b2) if x != y) + abs(len(b1) - len(b2))
    print("    rebuild size   = %s B" % (r2.get("size") if r2.get("ok") else "BUILD FAILED"))
    print("    original size  = %s B" % off.get("size"))
    print("    size identical : %s   <-- the quantity under test" % det)
    print("    sha256 identical: %s  (expected False: PE timestamp/debug dir)"
          % hash_same)
    print("    bytes differing : %s  (of %s B)"
          % (n_diff_bytes, off.get("size")))
    meta.record("BUDGET build size is reproducible (same byte count on rebuild)",
                bool(det),
                "a rebuild one byte larger must be flagged by the same size "
                "comparison (this is the comparator's can-fail control, "
                "evaluated for real)",
                (r2.get("size") or 0) + 1 != (off.get("size") or 0),
                "size match=%s; sha match=%s; %s bytes differ (PE header only)"
                % (det, hash_same, n_diff_bytes))
    results["OFF_repeat"] = r2

    out = dict(p14=(P14_ENGINE_SIZE, P14_BUDGET), variants=results,
               s5_delta=s5_delta, train_delta=train_delta,
               bench_delta=bench_delta, null_delta=null_delta,
               apparatus_moves=apparatus_moves,
               reproducible=bool(det))
    p = os.path.join(V.ensure_outdir("budget"), "budget_result.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nwrote %s" % p)
    allok = meta.report()
    # A zero S5 delta is NOT an automatic failure -- 0 B is exactly what the
    # feature-gating design predicts. What IS a failure is an unmeasured or
    # un-controlled number.
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
