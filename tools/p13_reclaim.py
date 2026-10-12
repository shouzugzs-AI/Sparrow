#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P13: reclaim the 448 B that Task A's runtime token feed costs.

STATE OF PLAY (all measured):
    engine with print_f32 rewrite (B) + dtype-aware EMB (A)   = 53248 B
    + runtime tokens.bin feed                                 = 53760 B
    budget                                                    = 53312 B
    => 448 B must come back.

WHAT IS BEING PROBED
    The token feed is: probe for st_tok, probe for tokens.bin, read it, store
    S u32 ids as f32. Each removal is measured by rebuilding.

Usage:
    python tools/p13_reclaim.py
"""
import os
import re
import subprocess
import sys

ROOT = str(Path(__file__).resolve().parents[1])
MAIN = os.path.join(ROOT, "runtime", "src", "main.rs")


def build():
    p = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-File", os.path.join(ROOT, "tools", "build_all.ps1")],
        cwd=ROOT, capture_output=True, text=True, errors="replace")
    out = p.stdout + p.stderr
    m = re.search(r"Engine core size:\s*(\d+)", out)
    return int(m.group(1)) if m else None


def probe(name, fn):
    orig = open(MAIN, "rb").read().decode("utf-8")
    try:
        new = fn(orig)
        if new == orig:
            return "NO-OP"
        open(MAIN, "wb").write(new.encode("utf-8"))
        os.utime(MAIN, None)
        return build()
    finally:
        open(MAIN, "wb").write(orig.encode("utf-8"))
        os.utime(MAIN, None)


def drop_slot_index(orig):
    """Remove slot_index() and call the engine's find_slot directly."""
    i = orig.index("unsafe fn slot_index(")
    # back up over the doc comment
    j = orig.rindex("/// True if the program", 0, i)
    # find end: next "\n}\n" after i
    k = orig.index("\n}\n", i) + 3
    out = orig[:j] + orig[k:]
    out = out.replace(
        """    let slot = match slot_index(vm, b"st_tok") {
        Some(s) => s,
        None => return, // program is not prompt-driven
    };""",
        """    // The engine's own find_slot exits(7) on a miss, so gate on a cheap
    // name scan first. Reusing the engine's linked helper keeps this small.
    let slot = vm.find_slot("st_tok");""")
    return out


def drop_bounds_check(orig):
    return orig.replace(
        """    if n > rows * cols {
        vmlm_core::print(b"tokens.bin is longer than the st_tok slot\\n");
        vmlm_core::exit(9);
    }""", "")


def main():
    base = build()
    print("current: %s  (budget 53312, over by %s)\n"
          % (base, (base or 0) - 53312))
    for name, fn in [("slot_index removed (use engine find_slot)", drop_slot_index),
                     ("drop the capacity guard", drop_bounds_check)]:
        r = probe(name, fn)
        print("%-44s -> %s" % (name, r))
    print("\nrestored:", build())
    return 0


if __name__ == "__main__":
    sys.exit(main())
