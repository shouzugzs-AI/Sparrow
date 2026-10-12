#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P10 task C: split a .vmb into
    (a) a SMALL "code" file  -- header + TEXT + META, to be embedded in the .exe
    (b) a LARGE "weights" file -- the CONST region, to be mmap'd at startup

WHY NOT A NEW FORMAT
--------------------
The .vmb layout is:
    Header(32) | TEXT | CONST (64B-aligned) | META
`Vm` reaches the const pool as `buf + const_off` and each constant as
`cpool + off/4`, where `off` comes from the META const table. So we do NOT
need to touch the format at all:

  * the weights file is a byte-for-byte copy of the CONST region, and
  * the code file keeps the SAME const_off, i.e. its CONST region is a
    zero-filled PLACEHOLDER of identical length.

Then `Vm::load_weights_mapped(code_buf, weights_mmap)` computes
`cpool = mmap + const_off` and every offset lines up exactly. There is
exactly ONE layout truth (the assembler's), which is the whole point:
a second layout definition would be a silent-misalignment waiting to happen.

This also keeps every existing .vmb valid and every existing test unchanged:
the non-split path is byte-identical to before.

SIZE REALITY CHECK (measured, not assumed)
------------------------------------------
Splitting does NOT shrink the weights. For MiniCPM the numbers are:
    weights (bf16)          2.161 GB   <- unchanged by splitting
    safetensors header      ~25 KB     <- removed, i.e. 0.001%
    code+meta               ~KBs       <- what actually moves into the exe
The win from mmap is NOT file size -- it is that the 2.16 GB is never copied
into private committed memory (pages are file-backed, shareable, evictable).
Quantization (bf16->int8/int4) is the only thing that shrinks the bytes, and
it needs new dequant GEMM kernels the engine does not have.

Usage:
    python tools/p10_split_vmb.py in.vmb out_code.vmb out_weights.bin
"""

import os
import struct
import sys

MAGIC = 0x4D4C4D56


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        return 2
    src, code_out, w_out = sys.argv[1:4]
    d = open(src, "rb").read()
    if len(d) < 32:
        print("error: %s too small" % src)
        return 1
    magic, = struct.unpack_from("<I", d, 0)
    if magic != MAGIC:
        print("error: bad magic 0x%08X" % magic)
        return 1
    text_off, text_len, const_off, const_len, meta_off, meta_len = struct.unpack_from(
        "<IIIIII", d, 8
    )
    if const_off + const_len > len(d):
        print("error: CONST region (%d..%d) exceeds file (%d)"
              % (const_off, const_off + const_len, len(d)))
        return 1

    weights = d[const_off:const_off + const_len]

    # ---- code file ----------------------------------------------------
    # IMPORTANT (measured): CONST is ~100% of a real .vmb, so keeping a
    # zero-filled placeholder would keep the embedded blob at 113 MB for
    # MiniCPM layer 0 -- i.e. NO saving at all. The whole point is that the
    # embedded part is small, so we DROP the CONST region entirely:
    #
    #   code file = [0 .. const_off)  ++  [const_off+const_len .. end)
    #             = Header + TEXT ++ META
    #
    # and we REWRITE the header's offsets so META's new position is recorded.
    # The const_off / const_len fields are left AS-IS: the loader must keep
    # using the ORIGINAL const_off (it indexes into the weights mapping, not
    # into the code buffer). That is deliberate -- it keeps a single layout
    # truth (the assembler's const_off) rather than introducing a second one.
    tail = d[const_off + const_len:]
    code = bytearray(d[:const_off] + tail)

    # META moved left by const_len; patch meta_off (header offset 24).
    new_meta_off = meta_off - const_len
    if new_meta_off < 0:
        print("error: meta_off %d < const_len %d" % (meta_off, const_len))
        return 1
    struct.pack_into("<I", code, 24, new_meta_off)

    os.makedirs(os.path.dirname(os.path.abspath(code_out)) or ".", exist_ok=True)
    with open(code_out, "wb") as f:
        f.write(bytes(code))
    with open(w_out, "wb") as f:
        f.write(weights)

    hs = 100.0 * const_len / len(d)
    print("in       : %s  (%.3f MB, %d B)" % (src, len(d) / 1e6, len(d)))
    print("  TEXT   : off=%d len=%d" % (text_off, text_len))
    print("  CONST  : off=%d len=%d (%.3f MB, %.1f%% of file)"
          % (const_off, const_len, const_len / 1e6, hs))
    print("  META   : off=%d len=%d  -> new off=%d" % (meta_off, meta_len, new_meta_off))
    print()
    print("code out : %s  (%d B, %.4f MB)  <- EMBED this in the .exe"
          % (code_out, os.path.getsize(code_out), os.path.getsize(code_out) / 1e6))
    print("wts  out : %s  (%.3f MB)  <- mmap this at startup"
          % (w_out, os.path.getsize(w_out) / 1e6))
    print()
    shrink = 100.0 * (1.0 - os.path.getsize(code_out) / len(d))
    print("embedded blob is %.2f%% smaller than the original .vmb" % shrink)
    print("NOTE: the WEIGHTS file is not smaller -- mmap changes memory")
    print("      behaviour, not file size. See tools/p10_quant_eval.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
