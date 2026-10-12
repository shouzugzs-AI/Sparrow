#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P13: line-by-line attribution of what load_tokens() costs the engine budget.

MEASURED SO FAR
    engine without the token feed ........ 53248 B
    engine with it ....................... 53760 B
    => the feed costs 512 B net today (1024 B before print_f32 was shrunk).
    budget 53312 B => 448 B still to find.

Rather than guess which part is expensive, this replaces successive pieces of
load_tokens with stubs and rebuilds. Each probe is a real measurement.

Usage: python tools/p13_attrib.py
"""
import os
import re
import subprocess
import sys

ROOT = str(Path(__file__).resolve().parents[1])
MAIN = os.path.join(ROOT, "runtime", "src", "main.rs")

STORAGE = """    let dst = vm.slot_mut(slot);
    let mut i = 0usize;
    while i < n {
        let b0 = *buf.add(i * 4) as u32;
        let b1 = *buf.add(i * 4 + 1) as u32;
        let b2 = *buf.add(i * 4 + 2) as u32;
        let b3 = *buf.add(i * 4 + 3) as u32;
        *dst.add(i) = (b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)) as f32;
        i += 1;
    }"""

READ = """    let (buf, len) = vmlm_core::read_file(tp.as_ptr());
    let n = len / 4;"""

GUARD = """    if n == 0 {
        vmlm_core::print(b"tokens.bin is empty\\n");
        vmlm_core::exit(9);
    }
    let (rows, cols) = vm.slot_shape(slot);
    if n > rows * cols {
        vmlm_core::print(b"tokens.bin is longer than the st_tok slot\\n");
        vmlm_core::exit(9);
    }"""

PROBES = [
    ("storage loop -> memcpy-equivalent stub", STORAGE,
     "    let dst = vm.slot_mut(slot);\n    let _ = (dst, n);"),
    ("read_file call removed", READ, "    let n = 8usize;"),
    ("error guards removed", GUARD, ""),
]


def build():
    p = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-File", os.path.join(ROOT, "tools", "build_all.ps1")],
        cwd=ROOT, capture_output=True, text=True, errors="replace")
    out = p.stdout + p.stderr
    m = re.search(r"Engine core size:\s*(\d+)", out)
    return int(m.group(1)) if m else None


def main():
    base = build()
    print("baseline (full token feed): %s\n" % base)
    orig = open(MAIN, "rb").read().decode("utf-8")
    for name, find, repl in PROBES:
        if orig.count(find) != 1:
            print("%-42s ANCHOR MISS (%d)" % (name, orig.count(find)))
            continue
        try:
            open(MAIN, "wb").write(orig.replace(find, repl).encode("utf-8"))
            os.utime(MAIN, None)
            sz = build()
            print("%-42s %s  (delta %+d)" % (name, sz, (sz or 0) - base))
        finally:
            open(MAIN, "wb").write(orig.encode("utf-8"))
            os.utime(MAIN, None)
    print("\nrestored:", build())
    return 0


if __name__ == "__main__":
    sys.exit(main())
