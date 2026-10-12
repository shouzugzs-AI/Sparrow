#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P13: measure the AVX2 store path of OP_EMB by removing it, then restore."""
import os
import re
import subprocess
import sys

ROOT = str(Path(__file__).resolve().parents[1])
P = os.path.join(ROOT, "core", "src", "interp.rs")

AVX_BLOCK = """                        #[cfg(target_arch = "x86_64")]
                        {
                            if HAS_AVX2 {
                                while i + 16 <= c {
                                    let (w0, w1) = kernels::bf16_widen_16(src.add(i));
                                    core::arch::x86_64::_mm256_storeu_ps(cd.add(i), w0);
                                    core::arch::x86_64::_mm256_storeu_ps(cd.add(i + 8), w1);
                                    i += 16;
                                }
                            }
                        }
"""


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
    base = build()
    print("current (EMB with AVX2):", base)

    if orig.count(AVX_BLOCK) != 1:
        print("anchor count = %d, cannot probe" % orig.count(AVX_BLOCK))
        return 1
    try:
        open(P, "wb").write(orig.replace(AVX_BLOCK, "").encode("utf-8"))
        os.utime(P, None)
        sz = build()
        print("EMB scalar-only       :", sz, " saved = %s B" % (base - sz))
        if sz is not None and sz <= 53312:
            print("=> scalar-only FITS the 53312 budget (headroom %d B)" % (53312 - sz))
    finally:
        open(P, "wb").write(orig.encode("utf-8"))
        os.utime(P, None)
        print("restored:", build())
    return 0


if __name__ == "__main__":
    sys.exit(main())
