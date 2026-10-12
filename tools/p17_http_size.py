#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P17-P4: prove the server crate does not change the ENGINE's code or data.

WHAT THIS FOUND, AND WHY THE FIRST CRITERION WAS WRONG
------------------------------------------------------
The Lead asked for sha256 equality (not just size) because "same size with
different bytes is possible". That instinct was right, and it immediately FAILED:

    BEFORE (workspace WITH server/)    : 74752 B  79cfc95b...
    AFTER  (workspace WITHOUT server/) : 74752 B  a17db0bc...

Same size, different bytes -- which looks exactly like a leak. It is not. A
control run settled it: building the engine TWICE from two clean, separate target
dirs with the SAME source produced

    build A : CE7D0CEE...
    build B : D5E4D174...

So `vmlm-runtime.exe` is **not byte-reproducible at all**, with or without the
server crate. The diff between two builds of identical sources is 20 bytes, in
two places:

  * offset 200 -- the PE/COFF `TimeDateStamp` field
  * offset 72560 -- an `RSDS` record, i.e. the CodeView PDB signature, followed
    by the literal string `vmlm_runtime.pdb`

MSVC's linker stamps the current time and a fresh PDB GUID into every link. Those
are build-clock artifacts with no relation to the generated code.

THE CORRECT CRITERION, therefore, is not "the files are identical" but
**"the code and data are identical once the linker's build stamps are removed"**.
This script implements that:

  1. Build the engine WITH server/ in the workspace  (isolated target dir)
  2. Build it again, same sources, second isolated dir  -> measures the NOISE
     FLOOR of the comparison. Without this step there is no way to know whether
     a difference is caused by the server crate or by the build clock.
  3. Build it from a pruned copy WITHOUT server/     -> the actual experiment
  4. Compare with build stamps normalized, and require the *set of differing
     byte offsets* to be confined to the same header/PDB regions in both
     comparisons.

Step 2 is the part that makes the result falsifiable: if the noise floor were 0
and the server comparison were non-zero, that WOULD be a real leak, and the
script reports FAIL.

This is the same discipline as the rest of the project: measure the instrument
before measuring the thing.

USAGE
-----
    python tools/p17_http_size.py
    python tools/p17_http_size.py --quick     # skip the root-build control

EXIT CODE: 0 all criteria PASS; 1 any FAIL.
"""
import argparse
import hashlib
import os
import shutil
import struct
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE_REL = os.path.join("x86_64-pc-windows-msvc", "release", "vmlm-runtime.exe")
LINES = []

# The two build-stamp regions the linker rewrites on every link. Anything outside
# these ranges that differs is a REAL difference in generated code or data.
STAMP_REGIONS = [
    ("PE TimeDateStamp", 0x00, 0x200),          # COFF header + data directories
    ("CodeView RSDS/PDB", 0x11B00, 0x11C00),    # located empirically at ~72560
]
# A generous bound on where the RSDS record can appear: the debug directory sits
# near the end of the file for this build. We compute it from the actual file
# rather than hardcoding, but keep a window as a fallback.
RSDS_WINDOW = 0x400


def emit(s=""):
    print(s)
    LINES.append(s)


def hr(t):
    emit()
    emit("=" * 78)
    emit(t)
    emit("=" * 78)


def read(path):
    with open(path, "rb") as f:
        return f.read()


def find_rsds(b):
    """Offset of the CodeView RSDS record, or None."""
    i = b.find(b"RSDS")
    return i if i >= 0 else None


def diff_offsets(a, b):
    if len(a) != len(b):
        return None
    return [i for i in range(len(a)) if a[i] != b[i]]


def classify(offsets, rsds_off, size):
    """Split differing offsets into 'linker build stamp' vs 'real'.

    # The regions, and how they were determined

    MEASURED, by diffing two builds of identical sources (build A vs build A2 in
    the output): 20 bytes differ, and they fall into three groups. The first
    version of this function only knew about the first two, which made a
    same-source pair look like a "REAL code/data difference" and would have
    produced a bogus FAIL for the server comparison.

        1. offset 200-203          -- PE/COFF `TimeDateStamp` (the link clock)
        2. offsets 72456, 72484,
           72512 (2 bytes each)    -- the three IMAGE_DEBUG_DIRECTORY entries.
                                      Their TimeDateStamp and PointerToRawData
                                      are rewritten on every link; this is the
                                      SAME kind of stamp as (1), just relocated.
        3. offset 72564..72579     -- the CodeView `RSDS` record: a fresh 16-byte
                                      PDB GUID per link, inside the record whose
                                      signature sits at 72560 and which is
                                      followed by the literal `vmlm_runtime.pdb`.

    All three are build-clock artifacts. Group 3 is found by searching for the
    `RSDS` magic rather than hardcoding the offset, so the classifier follows the
    file if the layout shifts; groups 1 and 2 are structural PE fields.

    # What is deliberately NOT excluded

    Anything else -- every other byte of .text, .rdata, the constant pool, the
    import table, the section headers -- is treated as REAL and any difference
    there fails the check. Excluding only the three known stamp kinds keeps the
    criterion narrow: it can still detect a genuine code change (which the
    positive control, `--features train`, demonstrates by ADDING 21504 B).
    """
    stamp, real = [], []
    for o in offsets:
        in_coff = o < 0x200
        in_pdb = rsds_off is not None and rsds_off <= o < rsds_off + 0x100
        # The IMAGE_DEBUG_DIRECTORY entries live immediately before the RSDS
        # record for this build; a 0x200-byte window before it covers all three.
        in_debugdir = rsds_off is not None and rsds_off - 0x200 <= o < rsds_off
        if in_coff or in_pdb or in_debugdir:
            stamp.append(o)
        else:
            real.append(o)
    return stamp, real


def build_engine(workdir, target_dir, extra=None):
    argv = ["cargo", "build", "--release", "-p", "vmlm-runtime",
            "--target", "x86_64-pc-windows-msvc", "--target-dir", target_dir]
    if extra:
        argv += extra
    p = subprocess.run(argv, capture_output=True, cwd=workdir)
    log = ((p.stdout or b"") + (p.stderr or b"")).decode("utf-8", "replace")
    exe = os.path.join(target_dir, ENGINE_REL)
    return (p.returncode == 0 and os.path.exists(exe)), exe, log, argv, p.returncode


def report_pair(label, ba, bb):
    """Compare two engine binaries, separating stamps from real differences.

    Returns (code_and_data_identical, differing_offsets).

    A SIZE difference returns `False` (not `None`): a different size IS a real
    difference, and it must not be confused with "no difference found". The first
    version returned `None` here, and the positive control -- which changes the
    size by +21504 B -- was then misread as "the control produced no real
    difference", producing a bogus FAIL on the control rather than a PASS.
    """
    emit()
    emit("  %s" % label)
    emit("    size A = %d B   size B = %d B" % (len(ba), len(bb)))
    if len(ba) != len(bb):
        emit("    SIZE DIFFERS by %+d B -> a real difference" % (len(bb) - len(ba)))
        return False, None
    offs = diff_offsets(ba, bb)
    rsds = find_rsds(ba)
    emit("    RSDS (PDB signature) at offset %s" % rsds)
    emit("    differing bytes: %d" % len(offs))
    if not offs:
        emit("    -> byte-identical")
        return True, offs
    stamp, real = classify(offs, rsds, len(ba))
    emit("    in linker build-stamp regions: %d  (offsets %s%s)"
         % (len(stamp), stamp[:8], " ..." if len(stamp) > 8 else ""))
    emit("    OUTSIDE those regions         : %d  (offsets %s)"
         % (len(real), real[:16]))
    ok = len(real) == 0
    emit("    -> %s" % ("only build stamps differ (code+data identical)"
                        if ok else "REAL code/data difference"))
    return ok, offs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-root-control", action="store_true")
    a = ap.parse_args()

    tag = time.strftime("%H%M%S")
    runtime = os.path.join(ROOT, "runtime")
    fails = []

    hr("P17-P4 ENGINE SIZE -- does the server crate change the engine?")
    emit("workspace root : %s" % ROOT)
    emit("engine build cwd: %s" % runtime)
    members = sorted(n for n in os.listdir(ROOT)
                     if os.path.exists(os.path.join(ROOT, n, "Cargo.toml")))
    emit("workspace members: %s" % ", ".join(members))

    # ---- 0. the reproducible failure -------------------------------------
    if not (a.quick or a.skip_root_control):
        hr("0. REPRODUCIBLE FAILURE -- building the engine from the workspace ROOT")
        emit("Recorded because cargo's config lookup starts at the CWD and walks")
        emit("UP only; the engine's link flags live in runtime/.cargo/config.toml.")
        ctl = os.path.join(ROOT, "target", "t_p17_rootctl_%s" % tag)
        argv = ["cargo", "build", "--release", "-p", "vmlm-runtime",
                "--target", "x86_64-pc-windows-msvc", "--target-dir", ctl]
        emit()
        emit("    cwd = %s" % ROOT)
        emit("    $ %s" % " ".join(argv))
        p = subprocess.run(argv, capture_output=True, cwd=ROOT)
        log = ((p.stdout or b"") + (p.stderr or b"")).decode("utf-8", "replace")
        emit("    exit code = %d" % p.returncode)
        for line in log.splitlines():
            if any(k in line for k in ("LNK1561", "must define", "必须定义",
                                       "error: linking")):
                emit("    | %s" % line.strip())
        if p.returncode != 0:
            emit("    => CONFIRMED: this command fails, so every engine build in")
            emit("       this project must state its working directory.")
        else:
            emit("    => NOT reproduced this run (recorded as observed).")
        shutil.rmtree(ctl, ignore_errors=True)

    # ---- 1. baseline WITH server -----------------------------------------
    hr("1. BUILD A -- workspace WITH server/")
    da = os.path.join(ROOT, "target", "t_p17_size_a_%s" % tag)
    ok, exe_a, log, argv, rc = build_engine(runtime, da)
    emit("    $ %s" % " ".join(argv))
    if not ok:
        for line in log.splitlines()[-20:]:
            emit("    | %s" % line)
        fails.append("build A failed")
        return finish(fails)
    ba = read(exe_a)
    emit("    exit code = 0")
    emit("    size   = %d B" % len(ba))
    emit("    sha256 = %s" % hashlib.sha256(ba).hexdigest())

    # ---- 2. THE INSTRUMENT: noise floor ----------------------------------
    hr("2. BUILD A2 -- SAME sources, second clean target dir  == NOISE FLOOR")
    emit("    This is the control the first version of this script was missing.")
    emit("    If two builds of IDENTICAL source differ, then a difference between")
    emit("    'with server' and 'without server' proves nothing on its own.")
    da2 = os.path.join(ROOT, "target", "t_p17_size_a2_%s" % tag)
    ok, exe_a2, log, argv, rc = build_engine(runtime, da2)
    if not ok:
        for line in log.splitlines()[-20:]:
            emit("    | %s" % line)
        fails.append("build A2 (noise floor) failed")
        return finish(fails)
    ba2 = read(exe_a2)
    emit("    size   = %d B" % len(ba2))
    emit("    sha256 = %s" % hashlib.sha256(ba2).hexdigest())
    floor_ok, floor_offs = report_pair("A vs A2 (identical sources)", ba, ba2)
    emit()
    emit("    INTERPRETATION: %s"
         % ("the build is not byte-reproducible; the comparison must ignore the "
            "linker's build stamps" if floor_offs else
            "the build IS byte-reproducible; a plain sha256 comparison is valid"))

    # ---- 3. the experiment: WITHOUT server -------------------------------
    hr("3. BUILD B -- pruned workspace WITHOUT server/")
    emit("    Only server/ is removed (and its members entry); everything else is")
    emit("    byte-identical source, so any code difference is attributable to it.")
    prune = os.path.join(ROOT, "target", "_p17_pruned_%s" % tag)
    shutil.rmtree(prune, ignore_errors=True)
    os.makedirs(prune)
    for name in os.listdir(ROOT):
        if name in ("target", "server"):
            continue
        src, dst = os.path.join(ROOT, name), os.path.join(prune, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst, symlinks=True,
                            ignore=shutil.ignore_patterns("target"))
        else:
            shutil.copy2(src, dst)
    ct = os.path.join(prune, "Cargo.toml")
    txt = open(ct, encoding="utf-8").read()
    new = txt.replace(', "server"]', ']')
    if new == txt:
        emit("    !! could not find \", \\\"server\\\"]\" in the members list;")
        emit("       the pruned tree still contains server, so this control is VOID.")
        fails.append("could not prune server from members")
    else:
        open(ct, "w", encoding="utf-8", newline="\n").write(new)
        emit("    pruned: %s" % [l for l in new.splitlines() if l.startswith("members")][0])
    db = os.path.join(prune, "target", "t_p17_size_b")
    ok, exe_b, log, argv, rc = build_engine(os.path.join(prune, "runtime"), db)
    if not ok:
        for line in log.splitlines()[-20:]:
            emit("    | %s" % line)
        fails.append("build B failed")
        shutil.rmtree(prune, ignore_errors=True)
        return finish(fails)
    bb = read(exe_b)
    emit("    size   = %d B" % len(bb))
    emit("    sha256 = %s" % hashlib.sha256(bb).hexdigest())
    exp_ok, exp_offs = report_pair("A (with server) vs B (without server)", ba, bb)

    # ---- 4. positive control ---------------------------------------------
    hr("4. POSITIVE CONTROL -- can this comparison detect a REAL change?")
    emit("    A comparison that always says 'identical' proves nothing. Rebuilding")
    emit("    WITH --features train must produce a real, non-stamp difference.")
    dc = os.path.join(ROOT, "target", "t_p17_size_pos_%s" % tag)
    ok, exe_c, log, argv, rc = build_engine(runtime, dc, extra=["--features", "train"])
    pos_ok = False
    if not ok:
        for line in log.splitlines()[-20:]:
            emit("    | %s" % line)
        fails.append("positive control build failed")
    else:
        bc = read(exe_c)
        emit("    size   = %d B   (vs %d B for the default build)" % (len(bc), len(ba)))
        # A SIZE difference IS the evidence here: the control must change the
        # engine, and `--features train` adds a whole backward path. Reading
        # `report_pair`'s return as "the control worked" was the bug -- a
        # size-differing pair returns False ("code+data identical" is false),
        # which is exactly what a working positive control looks like.
        report_pair("A (default) vs C (--features train)", ba, bc)
        delta = len(bc) - len(ba)
        emit("    delta  = %+d B" % delta)
        pos_ok = delta != 0
        if not pos_ok:
            emit()
            emit("    FAIL: the control produced no real difference, so a 'no real")
            emit("    difference' result for the server comparison would be vacuous.")
            fails.append("positive control indistinguishable")
        else:
            emit()
            emit("    PASS: the comparison detects a genuine change.")
    shutil.rmtree(dc, ignore_errors=True)

    # ---- 5. the criterion -------------------------------------------------
    hr("5. CRITERION")
    emit("  A  (workspace WITH server/)      : %6d B  %s"
         % (len(ba), hashlib.sha256(ba).hexdigest()))
    emit("  A2 (same sources, other dir)     : %6d B  %s"
         % (len(ba2), hashlib.sha256(ba2).hexdigest()))
    emit("  B  (workspace WITHOUT server/)   : %6d B  %s"
         % (len(bb), hashlib.sha256(bb).hexdigest()))
    emit("  engine budget                    :  84480 B")
    emit("  headroom                         : %d B" % (84480 - len(ba)))
    emit()
    emit("  naive sha256 A == B              : %s   <- would FAIL, but is the"
         % (hashlib.sha256(ba).hexdigest() == hashlib.sha256(bb).hexdigest()))
    emit("                                          wrong criterion (see line 2)")
    emit("  noise floor (A vs A2) real diffs : %s" % (len(floor_offs) and "yes" or "no"))
    emit()
    emit("  NOTE ON THE CRITERION. The naive requirement 'sha256(A) == sha256(B)'")
    emit("  is FALSE here -- and it is false for a reason that has nothing to do")
    emit("  with the server crate: builds A and A2 come from IDENTICAL sources and")
    emit("  still differ in %d bytes. MSVC's linker stamps a fresh PE" % len(floor_offs or []))
    emit("  TimeDateStamp and a fresh PDB GUID into every link. So the criterion")
    emit("  used is the strongest one that is actually well-posed:")
    emit()
    emit("      every differing byte lies in a linker build-stamp region,")
    emit("      i.e. the emitted CODE AND DATA are identical.")
    emit()
    if exp_ok:
        emit("  [PASS] removing the server crate from the workspace changes NOTHING")
        emit("         in the engine's code or data. Every differing byte between A")
        emit("         and B is a linker build stamp; the emitted code and data are")
        emit("         identical. Size 74752 B and headroom %d B are unaffected."
             % (84480 - len(ba)))
    else:
        emit("  [FAIL] the engine's code/data differs when server/ is present.")
        emit("         This is NOT explainable as a build stamp; it needs")
        emit("         investigation before this task can be called done.")
        fails.append("engine code/data differs with server present")

    if len(ba) > 84480:
        emit("  [FAIL] engine exceeds the 84480 B budget")
        fails.append("engine over budget")

    shutil.rmtree(prune, ignore_errors=True)
    shutil.rmtree(da, ignore_errors=True)
    shutil.rmtree(da2, ignore_errors=True)
    return finish(fails)


def finish(fails):
    hr("SUMMARY")
    emit("  checks failed: %d" % len(fails))
    for f in fails:
        emit("    - %s" % f)
    emit("  [%s]" % ("PASS" if not fails else "FAIL"))
    return 1 if fails else 0


if __name__ == "__main__":
    out = os.path.join(ROOT, "examples", "p17_http", "size_evidence.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rc = main()
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")
    print("\nEvidence written to examples/p17_http/size_evidence.txt")
    sys.exit(rc)
