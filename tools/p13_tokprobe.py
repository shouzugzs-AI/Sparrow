#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P13: attribute the 1024 B that load_tokens() added to the engine binary."""
import os
import re
import subprocess
import sys

ROOT = str(Path(__file__).resolve().parents[1])
P = os.path.join(ROOT, "runtime", "src", "main.rs")


def build():
    p = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-File", os.path.join(ROOT, "tools", "build_all.ps1")],
        cwd=ROOT, capture_output=True, text=True, errors="replace")
    out = p.stdout + p.stderr
    m = re.search(r"Engine core size:\s*(\d+)", out)
    return int(m.group(1)) if m else None


def main():
    orig = open(P, "rb").read().decode("utf-8")
    print("current:", build())

    # Variant 1: drop only the call, keep the functions (dead-code elim should
    # remove them entirely -> tells us the marginal cost of the FEATURE).
    call = "            load_tokens(&mut vm);\n"
    if orig.count(call) == 1:
        try:
            open(P, "wb").write(orig.replace(call, "").encode("utf-8"))
            os.utime(P, None)
            print("without load_tokens call:", build())
        finally:
            open(P, "wb").write(orig.encode("utf-8"))
            os.utime(P, None)

    # Variant 2: replace the body with an empty fn (keeps the extern block).
    print("restored:", build())
    return 0


if __name__ == "__main__":
    sys.exit(main())
