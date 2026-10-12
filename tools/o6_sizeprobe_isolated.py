#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""O6-A: measure the engine size with the O6-A arms, in isolation from S5.

WHY THIS IS NEEDED
    At the time task-1 landed, `core/src/neuron_kv.rs` (another teammate's S5
    file) did not compile -- 12 x E0133 "call to unsafe function outside an
    unsafe block". That made tools/build_all.ps1 report 8 failed steps, so
    `vmlm-runtime.exe` was NEVER REBUILT. The "Engine core size" line still
    printed 67584, but that number came from the LAST SUCCESSFUL link, and the
    concurrent S5 work made it impossible to tell whether 67584 reflected the
    O6-A arms or something else.

    This script removes that ambiguity WITHOUT touching S5's source: it
    temporarily comments out the single `pub mod neuron_kv;` line in
    core/src/lib.rs, builds, reads the size, and restores lib.rs byte-for-byte
    in a finally block. That isolates "bytes my arms cost" from "someone else's
    module does not compile".

    It also does the same for the O6-A arms themselves, so the cost is
    ATTRIBUTED rather than guessed.

SAFETY
    * lib.rs and interp.rs are read/written as bytes with encoding='utf-8',
      newline='' -- never a Get-Content -Raw | Set-Content round trip (P10
      destroyed `#[repr(C)]` that way and made every instruction decode as
      op=0/HALT).
    * Every anchor must match EXACTLY ONCE or the script raises before building.
    * Every modified file is restored from an in-memory copy in a `finally`.
    * No `cargo clean` anywhere (it discards expensive prebuilt artifacts).

Usage:
    python tools/o6_sizeprobe_isolated.py
"""
import os
import re
import subprocess
import sys

ROOT = str(Path(__file__).resolve().parents[1])
LIB = os.path.join(ROOT, "core", "src", "lib.rs")
INTERP = os.path.join(ROOT, "core", "src", "interp.rs")

NEURON_MOD = "pub mod neuron_kv;\n"

# Pure-code spans (no comments), so they survive comment edits.
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

MAX_ALL = MAX_GUARDS + (
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

TOPK_ALL = TOPK_GUARDS + (
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


def read_bytes_text(path):
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:3] == b"\xef\xbb\xbf":
        raise RuntimeError("refusing to touch a BOM file: %s" % path)
    return raw.decode("utf-8")


def write_bytes_text(path, text):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    # P13 Task B methodology: a write that leaves the content byte-identical
    # may not advance mtime, cargo then reuses the stale object, and the size
    # line reports a number from the PREVIOUS build.
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


def measure(lib_text, interp_text, label):
    """Build once with these exact source states; return (size, ok)."""
    lib_o, interp_o = read_bytes_text(LIB), read_bytes_text(INTERP)
    try:
        write_bytes_text(LIB, lib_text)
        write_bytes_text(INTERP, interp_text)
        out = build()
        if "FAILED" in out:
            return None, False
        return engine_size(out), True
    finally:
        write_bytes_text(LIB, lib_o)
        write_bytes_text(INTERP, interp_o)


def main():
    interp0 = read_bytes_text(INTERP)
    lib0 = read_bytes_text(LIB)

    # --- preconditions: everything must match exactly once ---
    if lib0.count(NEURON_MOD) != 1:
        print("lib.rs anchor 'pub mod neuron_kv;' matched %d times (want 1)"
              % lib0.count(NEURON_MOD))
        return 1
    for nm, span in (("MAX_GUARDS", MAX_GUARDS), ("TOPK_GUARDS", TOPK_GUARDS),
                     ("MAX_ALL", MAX_ALL), ("TOPK_ALL", TOPK_ALL)):
        if interp0.count(span) != 1:
            print("interp anchor %s matched %d times (want 1)"
                  % (nm, interp0.count(span)))
            return 1
    print("preconditions OK: lib.rs anchor x1, 4 interp anchors x1 each")

    # S5's module is broken at the moment of writing; neutralise it for the
    # measurement only. This is a MEASUREMENT, not a fix, and it is restored.
    lib_no_s5 = lib0.replace(NEURON_MOD, "// [o6_sizeprobe_isolated] " + NEURON_MOD)

    print()
    print("=== engine size, isolating the S5 module compile failure ===")
    print()

    size_with_s5, ok = measure(lib0, interp0, "as-is")
    print("A. current tree as-is (S5 neuron_kv present) : %s"
          % ("%d B" % size_with_s5 if ok else "BUILD FAILED (as expected)"))

    size_base, ok = measure(lib_no_s5, interp0, "no-s5")
    if not ok:
        print("B. S5 module commented out                    : BUILD FAILED")
        print("   -> cannot attribute the O6-A cost; report this.")
        return 1
    print("B. S5 module commented out (O6-A arms present): %d B" % size_base)

    size_no_max, ok = measure(lib_no_s5, interp0.replace(MAX_ALL, ""), "no-max")
    print("C. + OP_MAX arm removed                       : %s  (OP_MAX costs %s)"
          % ("%d B" % size_no_max if ok else "BUILD FAILED",
             "%d B" % (size_base - size_no_max) if ok else "?"))

    size_no_both, ok = measure(lib_no_s5, interp0.replace(MAX_ALL, "")
                               .replace(TOPK_ALL, ""), "no-both")
    print("D. + OP_MAX and OP_TOPK_IDX both removed      : %s  (OP_TOPK_IDX costs %s,"
          " both cost %s)"
          % ("%d B" % size_no_both if ok else "BUILD FAILED",
             "%d B" % (size_no_max - size_no_both) if ok else "?",
             "%d B" % (size_base - size_no_both) if ok else "?"))

    size_no_guards, ok = measure(lib_no_s5, interp0.replace(MAX_GUARDS, "")
                                 .replace(TOPK_GUARDS, ""), "no-guards")
    print("E. guards removed, reductions kept            : %s  (guards cost %s)"
          % ("%d B" % size_no_guards if ok else "BUILD FAILED",
             "%d B" % (size_base - size_no_guards) if ok else "?"))

    # Confirm restoration is byte-exact.
    now_i, now_l = read_bytes_text(INTERP), read_bytes_text(LIB)
    print()
    print("restore check: interp.rs %s, lib.rs %s"
          % ("OK" if now_i == interp0 else "MISMATCH!",
             "OK" if now_l == lib0 else "MISMATCH!"))
    return 0 if (now_i == interp0 and now_l == lib0) else 1


if __name__ == "__main__":
    sys.exit(main())
