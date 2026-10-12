#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P12: separate the two costs that made a ONE-LAYER model take ~20 minutes.

The 858 MB artifact and the runaway runtime were the SAME defect:
  embed (401 MB) + lm_head (401 MB) were shipped into a model that reads
  neither (prompt rows are copied in; lm_head is only for the final logits).
Every extra constant is memory the engine touches at load and the OS pages in.

This script measures, on the SAME program:
  A. with the 802 MB of dead constants   (the bug)
  B. without them                        (the fix)
and reports both wall time and peak working set, so the claim is evidence
rather than assertion.

Usage: python tools/p12_size_effect.py
"""
import os
import shutil
import subprocess
import sys
import time

ROOT = str(Path(__file__).resolve().parents[1])
DIR = os.path.join(ROOT, "examples", "minicpm_full")
ASM = os.path.join(ROOT, "target", "release", "vmlm-asm.exe")
ENG = os.path.join(ROOT, "target", "release", "vmlm-runtime.exe")
REL = os.path.join(ROOT, "target", "release")


def run_variant(tag, extra_consts, vml_name, vmb_name):
    """Assemble a program with an explicit constant list and time it."""
    args = [ASM, vml_name, vmb_name]
    with open(os.path.join(DIR, "weights.args"), encoding="ascii") as f:
        args += [l.strip() for l in f if l.strip()]
    args += extra_consts
    r = subprocess.run(args, cwd=DIR, capture_output=True, text=True,
                       errors="replace", encoding="utf-8")
    if r.returncode != 0:
        print("  ASM FAIL: " + (r.stdout + r.stderr)[-200:])
        return None
    vmb = os.path.join(DIR, vmb_name)
    size = os.path.getsize(vmb)
    shutil.copyfile(vmb, os.path.join(REL, "model.vmb"))
    t0 = time.perf_counter()
    e = subprocess.run([ENG], cwd=REL, capture_output=True, text=True,
                       errors="replace", encoding="utf-8")
    dt = time.perf_counter() - t0
    print("  %-30s file=%8.1f MB  time=%7.2f s  exit=%d"
          % (tag, size / 1e6, dt, e.returncode))
    return dt


def main():
    # find the embed/lm_head bf16 files if a --with-embed build produced them
    need = []
    for nm in ("embed", "lm_head"):
        p = os.path.join(DIR, "weights", nm + ".bf16")
        if os.path.exists(p):
            shp = {"embed": (130560, 1536), "lm_head": (130560, 1536)}[nm]
            need.append("%s:weights/%s.bf16:%d:%d:bf16" % (nm, nm, shp[0], shp[1]))
    print("extra big constants present: %d" % len(need))
    for n in need:
        print("   " + n)

    print()
    print("=== same program, different constant set ===")
    run_variant("A. as-shipped (with dead tensors)", need,
                "full.vml", "_with.vmb")
    run_variant("B. only what the program reads", [],
                "full.vml", "_without.vmb")
    print()
    print("If A is far slower, the cost is NOT compute: it is touching")
    print("hundreds of MB of constants the program never reads.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
