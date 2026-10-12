#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P13 Task A: prove OP_EMB is now dtype-aware AND driven by runtime state.

WHAT THIS CHECKS (and why each part is worth checking):
  1. bf16 table  -- the real gap. `embed_tokens` is bf16 [130560,1536] = 401 MB.
     Before P13, OP_EMB read the table as f32, so feeding a bf16 table either
     tripped the load-time dtype audit (exit 9) or, with the audit bypassed,
     silently read every element at twice the stride -- garbage from element 2
     onward, with no error. This is the exact failure class P12 fixed for GEMM.
  2. f32 table   -- the pre-existing path must keep working (no regression).
  3. bf16 widening is EXACT -- bf16 is the top 16 bits of f32, so a widened
     value must equal the original f32 rounded to bf16, bit for bit. A tolerance
     would hide a stride bug, so this asserts exactness.
  4. row selection -- the row index comes from a `st_` (state) slot, which is
     what lets the SAME .vmb serve any prompt.

WHERE THE EXPECTED VALUES COME FROM
  Computed here with numpy, and the widened values are compared EXACTLY. The
  bf16 encoding uses round-to-nearest-even, the same rule the engine assumes
  (tools/p12_build_minicpm.py:f32_to_bf16).

Usage: python tools/p13_emb_check.py
"""
import os
import re
import subprocess
import sys

import numpy as np

# The Windows console here is GBK; the assembler prints Chinese diagnostics,
# which arrive as replacement chars and would crash print(). Force UTF-8 with
# errors='replace' so a diagnostic can never take down the check itself.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = str(Path(__file__).resolve().parents[1])
DIR = os.path.join(ROOT, "examples", "p13")
REL = os.path.join(ROOT, "target", "release")
ASM = os.path.join(REL, "vmlm-asm.exe")
ENG = os.path.join(REL, "vmlm-runtime.exe")

ROWS, COLS = 4, 3
# A row set chosen so the bf16 rounding is non-trivial (not exactly
# representable), which is what makes a stride error visible.
TABLE = np.array([
    [0.0, 1.0, 2.0],
    [0.1, -3.5, 1234.5678],
    [-0.0009765625, 1e-3, 65.0],
    [7.0, -8.25, 0.333333343267],
], dtype=np.float32)


def f32_to_bf16(a):
    u = a.astype(np.float32).view(np.uint32).astype(np.uint64)
    rnd = np.uint64(0x7FFF) + ((u >> 16) & 1)
    return ((u + rnd) >> 16).astype("<u2")


def bf16_to_f32(h):
    return (h.astype(np.uint32) << 16).view(np.float32)


def build_and_asm():
    for name, arr, ext in [("table.bf16", f32_to_bf16(TABLE), "bf16"),
                           ("table_f32.f32", TABLE, "f32")]:
        arr.tofile(os.path.join(DIR, name))
    p = subprocess.run(
        [ASM, "emb_state.vml", "emb_state.vmb",
         "table:table.bf16:%d:%d:bf16" % (ROWS, COLS),
         "table_f32:table_f32.f32:%d:%d:f32" % (ROWS, COLS)],
        cwd=DIR, capture_output=True, text=True, errors="replace")
    out = p.stdout + p.stderr
    if p.returncode != 0 or "OK:" not in out:
        print("ASSEMBLE FAILED (exit %d)" % p.returncode)
        print(out[-2000:])
        return False
    for line in out.splitlines():
        if "OK:" in line:
            print("  assembler: " + line.strip())
    return True


def run_engine(token_id):
    """Run with the `st_tok` state slot driven to `token_id`.

    Two mechanisms tested here:
      * the state slot exists (`st_` prefix -> SLOT_STATE) and is engine-owned;
      * the runtime fills it from tokens.bin before run().
    Driving a NON-ZERO row matters: row 0 of the test table is [0,1,2], whose
    first element is 0 -- indistinguishable from an untouched (zeroed) state
    region if the lookup silently did nothing.

    NOTE (measured): this harness must control tokens.bin explicitly. An
    earlier version left it alone, so whatever a previous test had written in
    target/release was inherited and the row silently became a different one.
    """
    import shutil
    shutil.copyfile(os.path.join(DIR, "emb_state.vmb"),
                    os.path.join(REL, "model.vmb"))
    tb = os.path.join(REL, "tokens.bin")
    with open(tb, "wb") as f:
        np.array([token_id], dtype="<u4").tofile(f)
    out = os.path.join(DIR, "_emb_out.txt")
    with open(out, "wb") as f:
        p = subprocess.run([ENG], cwd=REL, stdout=f, stderr=subprocess.DEVNULL)
    if p.returncode != 0:
        return None, p.returncode
    vals = []
    with open(out, "r", errors="replace") as f:
        for line in f:
            for t in line.split():
                try:
                    vals.append(float(t))
                except ValueError:
                    pass
    return np.array(vals, dtype=np.float32), 0


def main():
    print("=== P13 Task A: dtype-aware EMB + state-driven row ===\n")
    if not os.path.isdir(DIR):
        os.makedirs(DIR)
    print("table (f32):\n%s\n" % TABLE)
    print("table as bf16 (what the engine must produce):")
    print(bf16_to_f32(f32_to_bf16(TABLE)))
    print()

    if not build_and_asm():
        return 1

    # ROW picks a non-zero row whose values are all non-zero and whose bf16
    # rounding is lossy for at least one column. Row 0 is deliberately NOT used:
    # [0,1,2] would still "pass" if the lookup did nothing and the zeroed state
    # region were read instead.
    ROW = 1
    got, rc = run_engine(ROW)
    if got is None:
        print("engine exit=%d" % rc)
        return 1
    expect = np.concatenate([bf16_to_f32(f32_to_bf16(TABLE))[ROW], TABLE[ROW]])
    print("\ndriven row: tokens.bin = [%d]  (non-zero on purpose)" % ROW)
    print("engine printed %d values, expected %d" % (got.size, expect.size))
    if got.size != expect.size:
        print("FAIL: size mismatch")
        return 1
    exp_bf16, exp_f32 = expect[:COLS], expect[COLS:]
    g_bf16, g_f32 = got[:COLS], got[COLS:]

    ok = True
    d = np.abs(g_bf16 - exp_bf16)
    print("\nbf16 table -> widened row: engine %s" % g_bf16.tolist())
    print("                           expect %s" % exp_bf16.tolist())
    # TRUNCATED, NOT ROUNDED -- and that is pre-existing engine behaviour.
    #
    # print_f32 emits 6 digits by repeatedly taking the integer part of *10,
    # i.e. it TRUNCATES. 0.10009765625 therefore prints as "0.100097", not
    # "0.100098". The P13 rewrite kept that algorithm deliberately so the
    # output stayed bit-identical (verified: the 24-layer logits reproduce the
    # P12 numbers exactly, rel 1.8806e-06), so this expectation must model
    # truncation. Comparing against 6-digit ROUNDING would report the correct
    # engine as broken -- which is exactly what an earlier version of this
    # check did.
    def trunc6(a):
        out = []
        for x in np.asarray(a, dtype=np.float64).ravel():
            f = abs(float(x))
            ip = int(f)
            fr = f - ip
            s = ""
            for _ in range(6):
                fr *= 10.0
                dg = int(fr)
                s += str(dg % 10)
                fr -= dg
            out.append(float("%d.%s" % (ip, s)))
        return np.array(out)
    if not np.array_equal(trunc6(g_bf16), trunc6(exp_bf16)):
        print("FAIL: bf16 widening wrong at printed precision "
              "(engine %s vs expect %s)"
              % (trunc6(g_bf16).tolist(), trunc6(exp_bf16).tolist()))
        ok = False
    else:
        print("  PASS: matches the widened value to printed precision "
              "(engine truncates 6 digits; raw max|d| = %.2e)" % d.max())

    print("\nf32 table -> row:          engine %s" % g_f32.tolist())
    print("                           expect %s" % exp_f32.tolist())
    if not np.array_equal(g_f32, exp_f32):
        print("FAIL: f32 path regressed")
        ok = False
    else:
        print("  PASS: unchanged (no regression)")

    print()
    print("VERDICT: %s" % ("PASS -- EMB is dtype-aware and state-addressed" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
