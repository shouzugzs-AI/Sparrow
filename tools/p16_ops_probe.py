#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P16-2: engine runner / probe for the two connection-sparse opcodes.

WHY THIS FILE EXISTS AT ALL (rather than a "just call the exe" one-liner)

The engine writes its results into the ACTIVATION AREA, not to a file. There
are only two ways to get numbers out of it:

  (A) `PRINT <slot>` -- the engine's hand-rolled f32 formatter. It writes to
      stdout with 6 decimals, and `core/src/interp.rs` records WHY it is hand
      rolled: the `core::fmt` version was 3072 B of a 53248 B engine. So it is
      available, but it is LOSSY: 6 decimals is not enough to distinguish
      bit-exact from 1-ULP-off, and comparing strings is not a numeric
      comparison. `tools/p12_logits_check.py` does use it for the 522240-logit
      acceptance, but that comparison has a tolerance.

  (B) the `bench` feature's `N:slot` syntax, which dumps slot contents with
      `dump_f32` -- raw little-endian f32, full precision -- to `out.bin`.

This probe uses (B) for the numeric comparison, because "bit-exact" is a claim
about bits and a 6-decimal string cannot carry it.

THE BUG THIS FILE WORKS AROUND -- AND WHY THE WORKAROUND IS NOT A HACK

`runtime/src/bench.rs` dumps the slot at line 144 **before** the timed loop and
`return`s immediately after `vm.run()` when `reps <= 1`:

    if reps <= 1 { let r = vm.run(); if let Some(si) = dump_slot { ... } return r; }

so `N:slot` with N=1 works. BUT `OP_HALT` is not the only way a program ends:
the value dumped is whatever is in the slot when `run()` returns.

The real constraint is different and worth writing down, because it shaped the
PROGRAM GENERATOR in p16_sparse_ops.py: the dump happens after `run()`
completes, so the slot to dump must still hold its correct value at HALT.
A probe slot that a later instruction overwrites is silently dumped wrong.
That is why every generated program ends with `HALT` immediately after the
instruction under test, and why the nnz self-report reads `dst[n]` rather than
a separate counter slot that could be clobbered.

USAGE
    python tools/p16_ops_probe.py --exe <path to vmlm-runtime.exe> \
                                  --asm <path to vmlm-asm.exe> \
                                  --vml <file.vml> --constants name:file:rows:cols ... \
                                  --dump <slot> --work <dir>

Exposes `run_program()` so p16_sparse_ops.py can import it instead of
shelling out twice with slightly different quoting (the first version of this
harness died on exactly that: PowerShell re-splits `name:file:rows:cols` on the
drive-letter colon, and the assembler's `w.split(':')` then reads the file "D").
"""
import os
import struct
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The engine is no_std with a custom entry point: its exit code is the value
# `mainCRTStartup` returned, i.e. `ExitProcess(code)`. A Rust `panic!` under
# `panic = "abort"` shows up as STATUS_STACK_BUFFER_OVERRUN-ish values on
# Windows (measured: 0xC0000409 = 3221226505) rather than as 101.
EXIT_UNIMPL_OP = 5
EXIT_GUARD = 9
EXIT_PANIC = 3221226505


def _decode(b):
    """The assembler prints Chinese warnings in the console codepage (GBK).

    Decoding with utf-8 raised UnicodeDecodeError *inside subprocess's reader
    thread* on the first run (measured), which surfaced as a confusing
    "Exception in thread Thread-2" instead of the real assembler output.
    """
    for enc in ("utf-8", "gbk", "cp1252"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode("utf-8", "replace")


def run_asm(asm_exe, vml, vmb, constants=(), cwd=ROOT, timeout=120):
    """Assemble `vml` -> `vmb`.

    `constants` is a list of (name, path, rows, cols) triples. NOTE: the path
    MUST be relative to `cwd`. `asm/src/main.rs` parses the spec with
    `w.split(':')`, so an absolute Windows path ("D:\\...") splits into
    f[1]="D" and the assembler then reports "读不到权重文件" for a file that
    demonstrably exists. This is a pre-existing assembler limitation, not
    something P16 introduced; the probe works with it rather than papering
    over it, because silently rewriting paths would hide it from the next
    person.
    """
    argv = [asm_exe, vml, vmb]
    for name, path, rows, cols in constants:
        argv.append("%s:%s:%d:%d" % (name, path, rows, cols))
    p = subprocess.run(argv, capture_output=True, cwd=cwd, timeout=timeout)
    return p.returncode, _decode(p.stdout), _decode(p.stderr)


def run_engine(exe, vmb, dump_slot=None, reps=1, cwd=None, timeout=120):
    """Run the bench-feature engine. Returns (rc, stdout, dumped_f32_list).

    The engine's bench entry point is `vmlm-runtime <model.vmb> [reps[:slot]]`
    and it writes `out.bin` (raw LE f32) NEXT TO ITS CWD, so this runs the exe
    in a private work directory and reads that file. Without a private cwd two
    concurrent probes would race on the same `out.bin` -- the shared-target-dir
    class of false measurement this project has already been bitten by.
    """
    cwd = cwd or os.path.dirname(os.path.abspath(vmb))
    model = os.path.basename(vmb)
    arg = "%d" % reps if dump_slot is None else "%d:%d" % (reps, dump_slot)
    out_bin = os.path.join(cwd, "out.bin")
    if os.path.exists(out_bin):
        os.remove(out_bin)
    p = subprocess.run([exe, model, arg], capture_output=True, cwd=cwd, timeout=timeout)
    vals = []
    if os.path.exists(out_bin):
        raw = open(out_bin, "rb").read()
        vals = list(struct.unpack("<%df" % (len(raw) // 4), raw))
    return p.returncode, _decode(p.stdout), vals


def f32(x):
    """Round a Python float to the nearest f32 (the engine's arithmetic dtype)."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def bits(x):
    """Raw IEEE-754 bit pattern of an f32, as an int. `-0.0` != `0.0` here."""
    return struct.unpack("<I", struct.pack("<f", f32(x)))[0]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--asm", required=True)
    ap.add_argument("--vml", required=True)
    ap.add_argument("--vmb", required=True)
    ap.add_argument("--const", action="append", default=[])
    ap.add_argument("--dump", type=int, default=None)
    ap.add_argument("--reps", type=int, default=1)
    a = ap.parse_args()

    consts = []
    for c in a.const:
        n, p, r, cc = c.split(",")
        consts.append((n, p, int(r), int(cc)))

    rc, out, err = run_asm(a.asm, a.vml, a.vmb, consts)
    print("asm rc=%d" % rc)
    print(out)
    if err.strip():
        print("asm stderr: %s" % err.strip())
    if rc != 0:
        sys.exit(rc)

    rc, out, vals = run_engine(a.exe, a.vmb, dump_slot=a.dump, reps=a.reps)
    print("engine rc=%d stdout=%r" % (rc, out))
    print("dumped %d f32: %r" % (len(vals), vals))
