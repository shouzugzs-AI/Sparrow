#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P13 task C: trustworthy attention-cost measurement using the ENGINE'S OWN clock.

WHY THE PREVIOUS ATTEMPT WAS NOT GOOD ENOUGH (measured, and it invalidated my
first conclusion):
    Timing the engine as a subprocess measures wall clock from Python. At S=64
    that spread was 0.270 s across 7 identical runs (min 1.151, max 1.421) --
    because each process re-faults pages of the 2.16 GB mmap. The attention term
    I was trying to isolate is of the same order. So the earlier table
    ("attention = 0.8% at S=64, 7.0% at S=256") was NOT resolvable by that method
    and must not be trusted.

WHAT THIS DOES INSTEAD
    Uses the `bench` feature build, which times with QueryPerformanceCounter
    INSIDE the process, after a warm-up run, over N iterations (runtime/src/
    bench.rs). Warm-up means the weights are already faulted in, so the
    measurement is the interpreter's work rather than page-fault behaviour.

    Caveat handled explicitly: the bench entry point runs `model.vmb` and does
    NOT fill `st_tok` from tokens.bin (that lives in the default entry point).
    So the token ids here come from... nothing: st_tok stays zero, i.e. every
    position embeds token 0. That is FINE for a timing measurement (identical
    work either way, same shapes) but it means this harness must never be used
    for numerics.

Usage:
    python tools/p13_bench_attn.py --seqs 16,64,256 --reps 20
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = str(Path(__file__).resolve().parents[1])
RT = os.path.join(ROOT, "runtime")
REL = os.path.join(ROOT, "target", "release")
TRIPLE_REL = os.path.join(ROOT, "target", "x86_64-pc-windows-msvc", "release")
WORK = os.path.join(ROOT, "examples", "p13_attn")


def build_bench():
    p = subprocess.run(["cargo", "build", "--features", "bench", "--release",
                        "--target", "x86_64-pc-windows-msvc"],
                       cwd=RT, capture_output=True, text=True, errors="replace")
    if p.returncode != 0:
        print("bench build failed")
        print((p.stdout + p.stderr)[-1500:])
        return None
    exe = os.path.join(TRIPLE_REL, "vmlm-runtime.exe")
    dst = os.path.join(REL, "vmlm-runtime-bench.exe")
    shutil.copyfile(exe, dst)
    return dst


def bench_run(exe, vmb, reps):
    """Run `vmlm-runtime-bench.exe <vmb> <reps>` and parse avg us/iter."""
    tmp = os.path.join(REL, "_bench_model.vmb")
    shutil.copyfile(vmb, tmp)
    p = subprocess.run([exe, tmp, str(reps)], cwd=REL,
                       capture_output=True, text=True, errors="replace")
    m = re.search(r"avg us/iter =\s*(\d+)", p.stdout)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default="16,64,256")
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()

    exe = build_bench()
    if not exe:
        return 1
    print("=== P13 task C: attention cost via the engine's own QPC timer "
          "(%d reps, warm) ===\n" % args.reps)
    print("%5s %14s %14s %14s %8s" %
          ("S", "full us", "no-attn us", "attn us", "share"))
    print("-" * 62)
    for seq in [int(x) for x in args.seqs.split(",")]:
        d = os.path.join(WORK, "s%d" % seq)
        full = os.path.join(d, "full.vmb")
        noattn = os.path.join(d, "noattn.vmb")
        if not (os.path.exists(full) and os.path.exists(noattn)):
            print("  S=%d: missing artifacts (run p13_attn_isolate.py first)" % seq)
            continue
        uf = bench_run(exe, full, args.reps)
        un = bench_run(exe, noattn, args.reps)
        if uf is None or un is None:
            print("  S=%d: bench failed" % seq)
            continue
        print("%5d %13d us %13d us %13d us %7.1f%%"
              % (seq, uf, un, uf - un, 100.0 * (uf - un) / uf))
    print()
    print("Measured INSIDE the process after warm-up, so page faults are excluded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
