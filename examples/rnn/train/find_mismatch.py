#!/usr/bin/env python3
"""
A2 debug helper v2: find the instruction that trips "EW shape mismatch" in the
FULL program, without truncating (truncation breaks gradient-slot resolution).

Approach: build a copy of the real .vml where the first K *real instructions* are
kept and the rest replaced by HALT, but ALWAYS keep all SHAPE lines and all ZERO
lines (so slot resolution and shapes are identical).

This isolates the first K instructions and reports the smallest K that fails.
"""
import os
import shutil
import subprocess
import sys
import tempfile

DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(DIR, "..", "..", ".."))
ASM = os.path.join(REPO, "target", "release", "vmlm-asm.exe")
DRV = os.path.join(os.environ.get("TEMP", "/tmp"), "a2drv", "target", "release", "a2drv.exe")

T = sys.argv[1] if len(sys.argv) > 1 else "2"
vml_path = os.path.join(DIR, f"bptt_t{T}.vml")
lines = open(vml_path, encoding="utf-8").read().split("\n")

KEEP_OPS = {"SHAPE", "LN_EPS", "ZERO", "STACK_INIT"}
keep_lines, instr_lines = [], []
for l in lines:
    s = l.strip()
    if not s or s.startswith(";") or s.startswith("."):
        continue
    op = s.split()[0].upper()
    if op in KEEP_OPS:
        keep_lines.append(l)
    else:
        instr_lines.append(l)

print(f"structural={len(keep_lines)}  instructions={len(instr_lines)}")
wa = [x for x in open(os.path.join(DIR, "weights.args"), encoding="utf-8").read().split("\n") if x.strip()]

staged = os.path.join(DIR, f"bptt_t{T}.vmb")
backup = staged + ".bak"
shutil.copyfile(staged, backup)


def try_k(k):
    src = "\n".join(keep_lines) + "\n" + "\n".join(instr_lines[:k]) + "\n    HALT\n"
    with tempfile.TemporaryDirectory() as td:
        vml = os.path.join(td, "p.vml")
        vmb = os.path.join(td, "p.vmb")
        open(vml, "w", encoding="utf-8", newline="\n").write(src)
        r = subprocess.run([ASM, vml, vmb] + wa, capture_output=True, text=True,
                           cwd=DIR, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            return None, "asm: " + r.stderr.strip()[:160]
        shutil.copyfile(vmb, staged)
        # use the plain "train 1 step" path: it only needs the constants that
        # exist, and reports the engine's own crash (e.g. EW shape mismatch)
        rr = subprocess.run([DRV, DIR, "run", "1", "3e-3", T], capture_output=True,
                            text=True, cwd=REPO, encoding="utf-8", errors="replace")
        out = (rr.stdout + rr.stderr)
        bad = ("shape mismatch" in out) or ("slot not found" in out) or rr.returncode != 0
        return (not bad), out.strip()[:160]


try:
    # exponential probe to find an upper bound quickly
    lo, hi = 0, len(instr_lines)
    ok0, msg0 = try_k(0)
    print(f"k=0 -> {ok0} {msg0}")
    # binary search for the smallest failing k (assuming monotone)
    lo, hi = 1, len(instr_lines)
    while lo < hi:
        mid = (lo + hi) // 2
        ok, msg = try_k(mid)
        if ok:
            lo = mid + 1
        else:
            hi = mid
    ok, msg = try_k(lo)
    print(f"\nsmallest failing k = {lo}  ok={ok}  msg={msg}")
    print(f"  offending instruction: {instr_lines[lo-1].strip()}")
    if lo >= 2:
        print(f"  previous instruction : {instr_lines[lo-2].strip()}")
finally:
    shutil.copyfile(backup, staged)
    os.remove(backup)
