#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P14 independent check of the ENGINE SIZE claim for the O6 argmax additions.

THE CLAIM (tools/build_all.ps1:138-150, verify_all.ps1:211)
    "engine 66560 -> 68096 B with the two arms + their hard-failure guards"
    i.e. OP_MAX + OP_TOPK_IDX + the OP_EMB row-bounds guard cost +1536 B,
    and S5 contributes 0 B.

WHY THIS IS VERIFIABLE RATHER THAN JUST ASSERTABLE
    The size claim is the one place where "the fix is free" can hide a real
    cost, and this project already has one recorded instance of a size
    measurement being an EMPTY measurement (docs/P13_报告.md section 3.2: an
    empty patch reported 0 B for a change that really cost 3,584 B). So the
    claim is re-measured here from the built artifact, and the measurement
    apparatus is itself shown to be able to detect a change.

WHAT THIS MEASURES
    1. the current engine size, and the budget it is checked against;
    2. HEADROOM: budget - size. A budget re-baselined to exactly the current
       size is worth flagging, because it converts every future byte into a
       failure and makes the "budget" no longer an alarm;
    3. the size with the `train` feature ON, as a positive control that the
       size measurement can actually move (measured +20480 B), so a 0 B result
       elsewhere is a real zero and not an empty measurement.
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p14_verify_lib as L  # noqa: E402

ROOT = L.ROOT
EXE = os.path.join(ROOT, "target", "x86_64-pc-windows-msvc", "release",
                   "vmlm-runtime.exe")


def read_budget(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s.startswith("$budget") and "=" in s:
                try:
                    return int(s.split("=", 1)[1].strip())
                except ValueError:
                    return None
    return None


def build(features=None):
    cmd = ["cargo", "build", "--release", "--target",
           "x86_64-pc-windows-msvc",
           "--manifest-path", os.path.join(ROOT, "runtime", "Cargo.toml")]
    if features:
        cmd += ["--features", features]
    p = subprocess.run(cmd, cwd=os.path.join(ROOT, "runtime"),
                       capture_output=True, text=True, errors="replace",
                       timeout=1800)
    return p.returncode == 0


def main():
    print("=== engine size claim: measured, not asserted ===")
    b1 = read_budget(os.path.join(ROOT, "tools", "build_all.ps1"))
    b2 = read_budget(os.path.join(ROOT, "tools", "verify_all.ps1"))
    print("budget in build_all.ps1  : %s B" % b1)
    print("budget in verify_all.ps1 : %s B" % b2)
    if b1 != b2:
        print("  WARNING: the two scripts disagree about the budget")

    if not build(None):
        print("default build FAILED")
        return 1
    size = os.path.getsize(EXE)
    print("engine size (default)    : %d B" % size)

    headroom = (b1 - size) if b1 else None
    print("headroom vs budget       : %s B" % headroom)
    if headroom is not None and headroom == 0:
        print("  NOTE: the budget equals the current size exactly, so headroom is")
        print("  ZERO. Any future byte trips OVER BUDGET. That is a legitimate")
        print("  choice (it pins the number) but it means the budget no longer")
        print("  functions as an alarm with slack; recorded as an observation.")

    # Positive control: the measurement must be able to move.
    print()
    print("positive control: rebuild with --features train")
    if not build("train"):
        print("  train build FAILED -- cannot show the measurement can move")
        print("  => the 0-byte conclusions would be UNPROVEN")
        return 1
    size_tr = os.path.getsize(EXE)
    print("engine size (train)      : %d B  (delta %+d B)"
          % (size_tr, size_tr - size))
    moved = (size_tr != size)
    print("measurement detects a real change: %s" % moved)
    if not moved:
        print("  *** the size measurement cannot move -- any 0 B from it is an")
        print("      EMPTY measurement, not a conclusion ***")
        return 1

    # restore the default build so downstream suites see the normal engine
    build(None)
    print()
    print("restored default engine  : %d B" % os.path.getsize(EXE))
    print()
    print("VERDICT: size measurement is trustworthy (positive control moved by "
          "%+d B)" % (size_tr - size))
    return 0


if __name__ == "__main__":
    sys.exit(main())
