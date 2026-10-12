#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P13 task E: minimal rehearsal of the LOOPED-layer mechanism.

WHY A MINIMAL PROBE
    The real 2-layer loop reads layer 1's weights wrongly, yet every piece
    checks out by inspection: the stride matches the measured artifact layout,
    LAYERB's encoding is correct, LOOP_TOP is the body's first instruction, and
    the counter runs exactly twice. Debugging that on a 0.6 GB artifact costs a
    lot of disk I/O, so this reproduces the same control flow on tiny constants.

WHAT IT MIMICS
    Exactly the real program's structure:
        COPY lcnt, layer_count        (NL)
        COPY lstride, layer_stride    (stride bytes)
      LOOP_TOP:
        <body: read per-layer w and g>
        LAYERB lstride
        LOOP_CNT lcnt, @LOOP_TOP
        EWMUL lneg, lstride, neg_one
        LAYERB lneg
        <read a NON-per-layer constant>

    The constants are padded so that the "layer 1" copies sit exactly `stride`
    bytes after the "layer 0" copies, mirroring the real artifact. A correct
    implementation prints layer 0's values first, then layer 1's, and finally
    the unshifted tail constant.

Usage: python tools/p13_loop_min.py
"""
import os
import subprocess
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = str(Path(__file__).resolve().parents[1])
DIR = os.path.join(ROOT, "examples", "p13")
REL = os.path.join(ROOT, "target", "release")
ASM = os.path.join(REL, "vmlm-asm.exe")
ENG = os.path.join(REL, "vmlm-runtime.exe")


def main():
    print("=== P13 task E: minimal LOOPED-layer rehearsal ===\n")

    # "Layer 0" constants and their "layer 1" counterparts.
    w0 = np.array([1, 2, 3, 4], dtype=np.float32)
    w1 = np.array([11, 12, 13, 14], dtype=np.float32)
    g0 = np.array([100.0], dtype=np.float32)
    g1 = np.array([200.0], dtype=np.float32)
    tail = np.array([777, 778, 779, 780], dtype=np.float32)
    one = np.array([1.0], dtype=np.float32)
    negone = np.array([-1.0], dtype=np.float32)
    lcount = np.array([2.0], dtype=np.float32)

    for nm, a in [("w0", w0), ("w1", w1), ("g0", g0), ("g1", g1), ("tail", tail),
                  ("one", one), ("negone", negone), ("lcount", lcount)]:
        np.ascontiguousarray(a).tofile(os.path.join(DIR, "lm_%s.f32" % nm))

    # Layout: emit w0, g0, tail-ish then w1, g1. The SAME (rows, cols, dtype)
    # sequence for both "layers", so the byte stride between w0 and w1 equals
    # the stride between g0 and g1 -- exactly the property the real artifact
    # relies on. The assembler pads every constant to 64 bytes.
    #   layer 0 block: w0 [1,4] f32 (16B -> 64), g0 [1,1] f32 (4B -> 64) = 128
    #   layer 1 block: w1, g1                                              = 128
    stride = 128
    np.array([float(stride)], dtype=np.float32).tofile(
        os.path.join(DIR, "lm_stride.f32"))

    consts = [
        "w:lm_w0.f32:1:4:f32",
        "g:lm_g0.f32:1:1:f32",
        # the "layer 1" names are placeholders so they occupy the next 128 bytes
        "_pad_w:lm_w1.f32:1:4:f32",
        "_pad_g:lm_g1.f32:1:1:f32",
        # non-per-layer constants AFTER the two blocks, so the restore matters
        "tail_const:lm_tail.f32:1:4:f32",
        "one1:lm_one.f32:1:1:f32",
        "neg_one:lm_negone.f32:1:1:f32",
        "layer_count:lm_lcount.f32:1:1:f32",
        "layer_stride:lm_stride.f32:1:1:f32",
        # -NL, so the post-loop restore subtracts the FULL NL*stride the loop
        # accumulated (LAYERB fires once per iteration, including the last).
        # Missing this declaration made the probe read an uninitialised
        # activation, which is a probe bug -- the engine was fine.
        "neg_layer_count:lm_neglcount.f32:1:1:f32",
    ]
    p = subprocess.run([ASM, "loop_min.vml", "loop_min.vmb"] + consts,
                       cwd=DIR, capture_output=True, text=True, errors="replace")
    if p.returncode != 0:
        print("ASSEMBLE FAILED:\n%s" % (p.stdout + p.stderr)[-1200:])
        return 1

    # Verify the stride assumption against the ACTUAL assembled offsets before
    # trusting the probe -- otherwise this test could pass for the wrong reason.
    import struct
    vmb = os.path.join(DIR, "loop_min.vmb")
    f = open(vmb, "rb")
    hdr = f.read(32)
    u = lambda o: struct.unpack_from("<I", hdr, o)[0]
    meta_off, meta_len = u(24), u(28)
    f.seek(meta_off)
    mb = f.read(meta_len)
    f.close()
    n_const = struct.unpack_from("<I", mb, 0)[0]
    offs = [struct.unpack_from("<5I", mb, 28 + i * 24)[0] for i in range(n_const)]
    got_stride = offs[2] - offs[0]          # _pad_w - w
    print("measured stride (w -> _pad_w) = %d, probe declares %d  %s"
          % (got_stride, stride, "OK" if got_stride == stride else "MISMATCH"))
    if got_stride != stride:
        print("adjusting stride constant to the measured value")
        np.array([float(got_stride)], dtype=np.float32).tofile(
            os.path.join(DIR, "lm_stride.f32"))
        p = subprocess.run([ASM, "loop_min.vml", "loop_min.vmb"] + consts,
                           cwd=DIR, capture_output=True, text=True,
                           errors="replace")
        if p.returncode != 0:
            return 1
        stride = got_stride

    import shutil
    shutil.copyfile(vmb, os.path.join(REL, "model.vmb"))
    for nm in ("tokens.bin", "gen_tokens.bin"):
        fp = os.path.join(REL, nm)
        if os.path.exists(fp):
            os.remove(fp)
    out = os.path.join(DIR, "_loop_min.txt")
    with open(out, "wb") as fh:
        r = subprocess.run([ENG], cwd=REL, stdout=fh, stderr=subprocess.DEVNULL)
    if r.returncode != 0:
        q = subprocess.run([ENG], cwd=REL, capture_output=True)
        print("engine exit=%d  %s" % (r.returncode,
                                      (q.stdout + q.stderr).decode("utf-8", "replace")[:160]))
        return 1

    vals = []
    for line in open(out, errors="replace"):
        vals.extend(line.split())
    print("\nprinted sequence:")
    for i, v in enumerate(vals):
        print("   [%d] %s" % (i, v))

    # Expected: w0, g0, w1, g1, tail
    exp = [w0, g0, w1, g1, tail]
    expf = [x for a in exp for x in a.tolist()]
    got = [float(v) for v in vals]
    ok = len(got) == len(expf) and all(
        abs(g - e) < 1e-4 for g, e in zip(got, expf))

    print("\nexpected: %s" % expf)
    print("got     : %s" % got)
    print()
    if ok:
        print("VERDICT: PASS -- base advances by the stride, the body sees")
        print("         layer 1 on iteration 2, and the restore works.")
        return 0
    print("VERDICT: FAIL")
    if len(got) >= 10:
        print("  iteration 1 printed %s (expect %s)" % (got[:2], [w0.tolist(), g0.tolist()]))
        print("  iteration 2 printed %s (expect %s)" % (got[2:4], [w1.tolist(), g1.tolist()]))
        print("  post-loop tail    %s (expect %s)" % (got[4:5], [tail.tolist()]))
    return 1


if __name__ == "__main__":
    sys.exit(main())
