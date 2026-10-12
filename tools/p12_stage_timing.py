#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P12: time the 1-layer MiniCPM model in the engine, stage by stage.

WHY THIS EXISTS
    A 1-layer MiniCPM run took ~20 minutes. Two hypotheses were tested and
    BOTH were wrong:
      - "the naive bf16 GEMM order is 100000x slow"  -> measured: whole layer
        is ~10 ms, lm_head ~117 ms (tests/src/bf16diag.rs)
      - "PRINT does one WriteFile per element"       -> measured: 3.1 us/call,
        so 522240 logits = ~1.6 s (tools/_wfprobe.py)
    Neither accounts for 20 minutes, so the cause is somewhere else and must
    be measured, not guessed.

METHOD
    Build variants of the .vml that stop after N instructions and time them.
    That localises the cost to an instruction range instead of a hunch.
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


def load_args():
    with open(os.path.join(DIR, "weights.args"), encoding="ascii") as f:
        return [l.strip() for l in f if l.strip()]


def build_and_run(vml_text, tag):
    p = os.path.join(DIR, "_t_%s.vml" % tag)
    with open(p, "w", encoding="ascii", newline="\n") as f:
        f.write(vml_text)
    a = [ASM, os.path.basename(p), "_t_%s.vmb" % tag] + load_args()
    r = subprocess.run(a, cwd=DIR, capture_output=True, text=True,
                       errors="replace", encoding="utf-8")
    if r.returncode != 0:
        return None, r.stdout[-300:] + r.stderr[-300:]
    vmb = os.path.join(DIR, "_t_%s.vmb" % tag)
    shutil.copyfile(vmb, os.path.join(REL, "model.vmb"))
    t0 = time.perf_counter()
    e = subprocess.run([ENG], cwd=REL, capture_output=True, text=True,
                       errors="replace", encoding="utf-8")
    t1 = time.perf_counter()
    return (t1 - t0, e.returncode, len(e.stdout)), None


def main():
    src = open(os.path.join(DIR, "full.vml"), encoding="utf-8").read().split("\n")
    shapes = [l for l in src if l.strip().startswith("SHAPE")]
    body = [l for l in src if l.strip() and not l.strip().startswith(";")
            and not l.strip().startswith("SHAPE")]
    print("shapes=%d body=%d instructions" % (len(shapes), len(body)))

    # find the boundaries of interest
    def upto(pred):
        idx = [i for i, l in enumerate(body) if pred(l)]
        return idx

    marks = []
    marks.append(("shapes only + HALT", 0))
    for i, l in enumerate(body):
        s = l.strip()
        if s.startswith("GEMM") and "L0_q" in s:
            marks.append(("through L0_q GEMM", i + 1))
            break
    # after the attention block (first WRITECOL of head 15 is hard to find;
    # use the o_proj GEMM)
    for i, l in enumerate(body):
        if "L0_o" in l:
            marks.append(("through o_proj GEMM", i + 1))
            break
    for i, l in enumerate(body):
        if "L0_down" in l:
            marks.append(("through down GEMM (layer done)", i + 1))
            break
    marks.append(("whole program incl PRINT", len(body)))

    print()
    print("%-34s %8s %10s %s" % ("variant", "instrs", "seconds", "exit"))
    for name, n in marks:
        prog = "\n".join(shapes + [""] + body[:n] + ["HALT"]) + "\n"
        res, err = build_and_run(prog, "m%d" % n)
        if res is None:
            print("%-34s %8d  ASM FAIL %s" % (name, n, err[:120]))
            continue
        dt, rc, ob = res
        print("%-34s %8d %10.2f %d  (stdout %d B)" % (name, n, dt, rc, ob))
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
