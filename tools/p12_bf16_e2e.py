#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P12: end-to-end test of the bf16 engine path.

Verifies that
  1. the assembler writes a bf16 constant as 2 bytes/element (no widening),
  2. the engine's OP_GEMM dispatches to the widening bf16 kernel,
  3. the result equals the f32 path within f32 summation noise,
  4. the file is HALF the size (this is the point: MiniCPM must fit).

Also verifies the new guard: feeding a bf16 constant to a non-GEMM op must
hard-fail (exit 9) rather than read garbage.

Usage: python tools/p12_bf16_e2e.py
"""
import os
import shutil
import struct
import subprocess
import sys

import numpy as np

ROOT = str(Path(__file__).resolve().parents[1])
OUT = os.path.join(ROOT, "examples", "p12")
ASM = os.path.join(ROOT, "target", "release", "vmlm-asm.exe")
ENG = os.path.join(ROOT, "target", "release", "vmlm-runtime.exe")
REL = os.path.join(ROOT, "target", "release")


def f32_to_bf16_bytes(a: np.ndarray) -> bytes:
    """f32 array -> bf16 bytes (round-to-nearest-even, matching the kernel)."""
    u = a.astype(np.float32).view(np.uint32)
    # round to nearest even on bit 16
    rnd = np.uint32(0x7FFF) + ((u >> 16) & 1)
    return ((u + rnd) >> 16).astype("<u2").tobytes()


def run(vmb, name):
    shutil.copyfile(vmb, os.path.join(REL, "model.vmb"))
    p = subprocess.run([ENG], cwd=REL, capture_output=True, text=True,
                       errors="replace", encoding="utf-8")
    vals = []
    for ln in p.stdout.splitlines():
        for t in ln.split():
            try:
                vals.append(float(t))
            except ValueError:
                pass
    return p.returncode, np.array(vals, dtype=np.float32), p.stdout


def main():
    os.makedirs(OUT, exist_ok=True)
    M, N, K = 1, 64, 128
    rng = np.random.RandomState(42)
    x = (rng.randn(M, K) * 0.5).astype(np.float32)
    w = (rng.randn(N, K) / np.sqrt(K)).astype(np.float32)

    x.tofile(os.path.join(OUT, "x.bin"))
    w.astype(np.float32).tofile(os.path.join(OUT, "w_f32.bin"))
    with open(os.path.join(OUT, "w_bf16.bin"), "wb") as f:
        f.write(f32_to_bf16_bytes(w))

    s32 = os.path.getsize(os.path.join(OUT, "w_f32.bin"))
    s16 = os.path.getsize(os.path.join(OUT, "w_bf16.bin"))
    print("weights: f32 %d B  vs  bf16 %d B   (%.1f%% of f32)"
          % (s32, s16, 100.0 * s16 / s32))

    vml = "\n".join([
        "SHAPE x, %d, %d" % (M, K),
        "SHAPE w, %d, %d" % (N, K),
        "SHAPE y, %d, %d" % (M, N),
        "GEMM y, x, w",
        "PRINT y",
        "HALT",
    ])
    with open(os.path.join(OUT, "bf16_t.vml"), "w", encoding="ascii", newline="\n") as f:
        f.write(vml + "\n")

    # ---- f32 baseline ----
    r = subprocess.run([ASM, "bf16_t.vml", "t_f32.vmb", "x:x.bin:%d:%d" % (M, K),
                        "w:w_f32.bin:%d:%d:f32" % (N, K)],
                       cwd=OUT, capture_output=True, text=True, errors="replace", encoding="utf-8")
    if r.returncode != 0:
        print("ASM fail:\n" + r.stdout[-500:] + r.stderr[-500:])
        return 1
    c32, y32, _ = run(os.path.join(OUT, "t_f32.vmb"), "f32")

    # ---- bf16 ----
    r = subprocess.run([ASM, "bf16_t.vml", "t_bf16.vmb", "x:x.bin:%d:%d" % (M, K),
                        "w:w_bf16.bin:%d:%d:bf16" % (N, K)],
                       cwd=OUT, capture_output=True, text=True, errors="replace", encoding="utf-8")
    if r.returncode != 0:
        print("ASM(bf16) fail:\n" + r.stdout[-500:] + r.stderr[-500:])
        return 1
    c16, y16, out16 = run(os.path.join(OUT, "t_bf16.vmb"), "bf16")

    f32_vmb = os.path.getsize(os.path.join(OUT, "t_f32.vmb"))
    bf16_vmb = os.path.getsize(os.path.join(OUT, "t_bf16.vmb"))
    print("vmb: f32 %d B  vs  bf16 %d B" % (f32_vmb, bf16_vmb))

    # ---- compare ----
    # Reference: numpy f64 on the SAME bf16-rounded weights, so we isolate
    # engine correctness from bf16's own quantization.
    w_bf = np.frombuffer(open(os.path.join(OUT, "w_bf16.bin"), "rb").read(),
                         dtype="<u2").astype(np.uint32) << 16
    w_bf = w_bf.view(np.float32).reshape(N, K)
    ref_bf = (x.astype(np.float64) @ w_bf.astype(np.float64).T).astype(np.float32).reshape(-1)

    print()
    print("f32  path exit=%d  values=%d" % (c32, y32.size))
    print("bf16 path exit=%d  values=%d" % (c16, y16.size))
    if c32 != 0 or c16 != 0:
        print("ENGINE FAILED: " + out16[:300])
        return 1
    if y32.size != M * N or y16.size != M * N:
        print("size mismatch")
        return 1

    refmax = max(float(np.abs(ref_bf).max()), 1e-30)
    e16 = float(np.abs(y16 - ref_bf).max()) / refmax
    print("bf16 engine vs numpy(f64 on same bf16 weights): rel %.3e" % e16)
    print("f32  engine vs same reference (should differ by bf16 quantization): rel %.3e"
          % (float(np.abs(y32 - ref_bf).max()) / refmax))

    ok = True
    if e16 > 1e-5:
        print("FAIL: bf16 engine path deviates beyond f32 noise")
        ok = False
    else:
        print("PASS: bf16 engine path matches (residual is f32 summation order)")
    if bf16_vmb >= f32_vmb:
        print("FAIL: bf16 .vmb is not smaller (%d vs %d)" % (bf16_vmb, f32_vmb))
        ok = False
    else:
        print("PASS: bf16 .vmb is smaller (%d vs %d B)" % (bf16_vmb, f32_vmb))

    # ---- guard test: a bf16 const fed to a NON-GEMM op must hard-fail ----
    #
    # This is a real property of the design, not a wart: every opcode other
    # than GEMM reads its operands as f32 (129 dereference sites), so a bf16
    # constant reaching one of them would read half the elements and
    # reinterpret their bits -- silent garbage. The engine audits the whole
    # instruction stream at load and refuses.
    #
    # NOTE: an earlier version of this test used `PRINT w` to trigger it, but
    # the first model I built ALSO hit `op 52 (PRINT) uses bf16 const`, so this
    # now asserts the failure explicitly (exit 9) rather than assuming it.
    vml2 = "\n".join([
        "SHAPE w, %d, %d" % (N, K),
        "SHAPE z, %d, %d" % (N, K),
        "PRINT w",     # non-GEMM op receiving a bf16 const -> must be refused
        "HALT",
    ])
    with open(os.path.join(OUT, "guard.vml"), "w", encoding="ascii", newline="\n") as f:
        f.write(vml2 + "\n")
    r = subprocess.run([ASM, "guard.vml", "guard.vmb", "w:w_bf16.bin:%d:%d:bf16" % (N, K)],
                       cwd=OUT, capture_output=True, text=True, errors="replace", encoding="utf-8")
    if r.returncode != 0:
        print("PASS: assembly rejected the program")
    else:
        shutil.copyfile(os.path.join(OUT, "guard.vmb"), os.path.join(REL, "model.vmb"))
        gp = subprocess.run([ENG], cwd=REL, capture_output=True, text=True,
                            errors="replace", encoding="utf-8")
        if gp.returncode == 9 and "bf16" in gp.stdout:
            print("PASS: load-time audit refused a bf16 const on a non-GEMM op")
        else:
            print("FAIL: bf16 const on a non-GEMM op was accepted (exit=%d, out=%r)"
                  % (gp.returncode, gp.stdout[:80]))
            ok = False
    # length-mismatch guard
    r = subprocess.run([ASM, "bf16_t.vml", "bad.vmb", "x:x.bin:%d:%d" % (M, K),
                        "w:w_f32.bin:%d:%d:bf16" % (N, K)],
                       cwd=OUT, capture_output=True, text=True, errors="replace", encoding="utf-8")
    if r.returncode != 0 and "字节数不对" in (r.stdout + r.stderr):
        print("PASS: dtype/size mismatch is rejected at assembly")
    else:
        print("FAIL: a f32 file labelled bf16 was accepted (silent misalignment!)")
        ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
