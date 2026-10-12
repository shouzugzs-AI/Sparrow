#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""O6-A: measure the REAL byte cost of the two engine arms.

WHY THIS EXISTS
    The O6-A arms took the engine from 65536 B to 67584 B against a 66560 B
    budget (OVER BUDGET). The project rule is "measure, never estimate", so the
    guards are stubbed out and the engine rebuilt to find out how many of those
    bytes are the loud-failure checks rather than the reductions.

METHOD
    Same method as tools/p13_sizeprobe.py: patch interp.rs, rebuild with the
    canonical tools/build_all.ps1, read `Engine core size`, restore. The file is
    restored from an in-memory copy in a finally block and re-verified.

    interp.rs is read/written as BYTES with encoding='utf-8', newline='' --
    NEVER a Get-Content -Raw | Set-Content round trip, which in P10 destroyed
    `#[repr(C)]` and made every instruction decode as op=0 (HALT).

Usage:
    python tools/o6_sizeprobe.py
    python tools/o6_sizeprobe.py --region guards
"""
import argparse
import io
import os
import re
import subprocess
import sys

ROOT = str(Path(__file__).resolve().parents[1])
INTERP = os.path.join(ROOT, "core", "src", "interp.rs")
ASM = os.path.join(ROOT, "asm", "src", "main.rs")

# Anchors are the PURE CODE spans (no comments), so they survive comment edits
# and match exactly once. Verified by assert below rather than by eye.

MAX_GUARDS = (
    '                    let (dr, dc) = self.slot_shape(d);\n'
    '                    if c == 0 || dr == 0 || dc == 0 {\n'
    '                        crate::print(b"MAX: src or dst slot has width 0\\n");\n'
    '                        exit_i(9);\n'
    '                    }\n'
    '                    if dr != r || dc != 1 {\n'
    '                        crate::print(b"MAX: dst must be [rows,1]\\n");\n'
    '                        crate::print(b" src=");\n'
    '                        print_u32(r as u32);\n'
    '                        crate::print(b"x");\n'
    '                        print_u32(c as u32);\n'
    '                        crate::print(b" dst=");\n'
    '                        print_u32(dr as u32);\n'
    '                        crate::print(b"x");\n'
    '                        print_u32(dc as u32);\n'
    '                        crate::print(b"\\n");\n'
    '                        exit_i(9);\n'
    '                    }\n'
)

MAX_BODY = (
    '                    let ca = self.slot_mut(a);\n'
    '                    let cd = self.slot_mut(d);\n'
    '                    for i in 0..r {\n'
    '                        let row = ca.add(i * c);\n'
    '                        let mut m = *row;\n'
    '                        for j in 1..c {\n'
    '                            let v = *row.add(j);\n'
    '                            if v > m {\n'
    '                                m = v;\n'
    '                            }\n'
    '                        }\n'
    '                        *cd.add(i) = m;\n'
    '                    }\n'
)

TOPK_GUARDS = (
    '                    let (dr, dc) = self.slot_shape(d);\n'
    '                    if r == 0 || c == 0 || dr == 0 || dc == 0 {\n'
    '                        crate::print(b"TOPK_IDX: src or dst slot has width 0\\n");\n'
    '                        exit_i(9);\n'
    '                    }\n'
    '                    if dr != 1 || dc != 1 {\n'
    '                        crate::print(b"TOPK_IDX: dst must be [1,1]\\n");\n'
    '                        crate::print(b" src=");\n'
    '                        print_u32(r as u32);\n'
    '                        crate::print(b"x");\n'
    '                        print_u32(c as u32);\n'
    '                        crate::print(b" dst=");\n'
    '                        print_u32(dr as u32);\n'
    '                        crate::print(b"x");\n'
    '                        print_u32(dc as u32);\n'
    '                        crate::print(b"\\n");\n'
    '                        exit_i(9);\n'
    '                    }\n'
)

TOPK_BODY = (
    '                    let ca = self.slot_mut(a);\n'
    '                    let cd = self.slot_mut(d);\n'
    '                    let mut best = 0usize;\n'
    '                    let mut mv = *ca;\n'
    '                    for j in 1..c {\n'
    '                        let v = *ca.add(j);\n'
    '                        if v > mv {\n'
    '                            mv = v;\n'
    '                            best = j;\n'
    '                        }\n'
    '                    }\n'
    '                    *cd = best as f32;\n'
)

# The guards bind `(dr, dc)` and are preceded by `let (r, c) = ...(a);`.
# Dropping the guards alone leaves `dr`/`dc` unused (fine) but the stubbed
# region must still type-check, so we drop whole coherent spans.
REGIONS = {
    "max_guards": (MAX_GUARDS, ""),
    "max_all": (MAX_GUARDS + MAX_BODY, ""),
    "topk_guards": (TOPK_GUARDS, ""),
    "topk_all": (TOPK_GUARDS + TOPK_BODY, ""),
}


def read_text(path):
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:3] == b"\xef\xbb\xbf":
        raise RuntimeError("refusing to touch a BOM file: %s" % path)
    return raw.decode("utf-8")


def write_text(path, text):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    # Probe methodology (measured the hard way in P13 Task B): writing content
    # byte-identical to what is already on disk may not advance the mtime, and
    # cargo then reuses the stale object, so `Engine core size` reports a stale
    # number. Touch the file so every reported number provably comes from the
    # source currently on disk.
    os.utime(path, None)


def engine_size(out):
    m = re.search(r"Engine core size:\s*(\d+)\s*bytes", out)
    return int(m.group(1)) if m else None


def build():
    p = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-File", os.path.join(ROOT, "tools", "build_all.ps1")],
        cwd=ROOT, capture_output=True, text=True, errors="replace")
    return p.stdout + p.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region")
    args = ap.parse_args()

    # Assert every anchor matches exactly once BEFORE building anything.
    text0 = read_text(INTERP)
    for name, (span, _repl) in REGIONS.items():
        n = text0.count(span)
        if n != 1:
            print("ANCHOR BROKEN: %s matched %d times (expected 1)" % (name, n))
            return 1
    print("anchors: all %d match exactly once" % len(REGIONS))

    regions = [args.region] if args.region else list(REGIONS)
    for r in regions:
        if r not in REGIONS:
            print("unknown region %s (have: %s)" % (r, ", ".join(REGIONS)))
            return 1

    base_out = build()
    base = engine_size(base_out)
    print("baseline engine core size: %s B" % base)
    if base is None:
        print(base_out[-3000:])
        return 1

    print()
    print("%-14s %9s %8s   %s" % ("region", "stubbed", "saved", "what was removed"))
    print("-" * 78)
    for name in regions:
        span, repl = REGIONS[name]
        original = read_text(INTERP)
        try:
            assert original.count(span) == 1
            write_text(INTERP, original.replace(span, repl))
            out = build()
            if "FAILED" in out:
                print("%-14s %9s %8s   BUILD FAILED" % (name, "-", "-"))
                continue
            sz = engine_size(out)
            print("%-14s %9d %7d B" % (name, sz, base - sz))
        finally:
            write_text(INTERP, original)

    again = engine_size(build())
    print()
    print("re-check after restore: %s  %s" % (again, "OK" if again == base else "MISMATCH!"))
    return 0 if again == base else 1


if __name__ == "__main__":
    sys.exit(main())
