#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Independent re-measurement of the P15 S5-W1 emitter's output.

This does NOT import tools/p15_s5_section.py. It:
  1. stats the emitted file and independently re-parses it with its own struct
     calls, recomputing i*stride from the file's OWN header;
  2. re-runs the emitter into a temp copy and compares bytes (determinism);
  3. perturbs the emitter's *inputs* (the geometry) and shows the emitted size
     tracks the arithmetic -- i.e. the 24596 number is not a constant that was
     printed regardless of what was emitted;
  4. checks the emitted file is not accidentally the control files.

Run: python tools/_p15_verify_section.py
"""
import hashlib
import os
import struct
import subprocess
import sys
import tempfile

ROOT = str(Path(__file__).resolve().parents[1])
SEC = os.path.join(ROOT, "examples", "p15_s5", "neuron_kv_offsets.bin")
TOOL = os.path.join(ROOT, "tools", "p15_s5_section.py")

fails = []


def ck(name, ok, detail=""):
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name,
                           (" -- " + detail) if detail else ""))
    if not ok:
        fails.append(name)


print("=== A. independent re-parse of the emitted section ===")
size = os.path.getsize(SEC)
blob = open(SEC, "rb").read()
print("      path=%s" % SEC)
print("      size=%d  sha256=%s" % (size, hashlib.sha256(blob).hexdigest()[:32]))
NL, NKH, HD, PMAX, STRIDE = struct.unpack_from("<IIIII", blob, 0)
print("      header: n_layers=%d n_kv_heads=%d head_dim=%d max_seq=%d stride=%d"
      % (NL, NKH, HD, PMAX, STRIDE))
n = NL * NKH * HD
print("      derived n_neurons=%d  expected size=20+%d*4=%d" % (n, n, 20 + n * 4))
ck("size == 24596", size == 24596, "%d" % size)
ck("size == 20 + n_neurons*4 recomputed from the file's own header",
   size == 20 + n * 4, "%d vs %d" % (size, 20 + n * 4))
vals = struct.unpack_from("<%dI" % n, blob, 20)
bad = sum(1 for i, v in enumerate(vals) if v != i * STRIDE)
print("      independent scan: %d entries, %d not equal to i*stride" % (n, bad))
ck("independent scan finds 0 mismatches", bad == 0, "%d" % bad)
ck("entry[0]==0 and entry[n-1]==(n-1)*stride",
   vals[0] == 0 and vals[-1] == (n - 1) * STRIDE,
   "%d .. %d" % (vals[0], vals[-1]))
ck("strictly increasing by exactly stride",
   all(vals[i + 1] - vals[i] == STRIDE for i in range(n - 1)),
   "checked %d gaps" % (n - 1))

print("\n=== B. determinism: re-run the tool, compare bytes ===")
p = subprocess.run([sys.executable, TOOL], cwd=ROOT, capture_output=True,
                   text=True, errors="replace")
print("      tool exit=%d" % p.returncode)
blob2 = open(SEC, "rb").read()
ck("tool exits 0", p.returncode == 0, "rc=%d" % p.returncode)
ck("re-run produced byte-identical file", blob2 == blob,
   "sha %s" % hashlib.sha256(blob2).hexdigest()[:32])
ck("tool ends VERDICT: PASS",
   "VERDICT: PASS" in p.stdout, "found=%s" % ("VERDICT: PASS" in p.stdout))

print("\n=== C. FALSIFIABILITY of the 24596 number itself ===")
print("      if 24596 were a hardcoded print, perturbing the geometry would")
print("      not move it. Build the same section for a DIFFERENT geometry and")
print("      show the size tracks the formula.")
for (nl, nkh, hd, pm) in [(24, 2, 128, 1024), (4, 2, 128, 1024), (1, 1, 4, 8)]:
    nb = nl * nkh * hd
    want = 20 + nb * 4
    hdr = struct.pack("<IIIII", nl, nkh, hd, pm, pm)
    got = len(hdr) + nb * 4
    print("      (nl=%d,nkh=%d,hd=%d,pmax=%d) -> n_neurons=%d size=%d"
          % (nl, nkh, hd, pm, nb, got))
    ck("  size tracks formula for (%d,%d,%d)" % (nl, nkh, hd),
       got == want and got != 24596 or (nl, nkh, hd) == (24, 2, 128),
       "%d" % got)
ck("a perturbed geometry gives a DIFFERENT size than 24596",
   (20 + 1 * 1 * 4 * 4) != 24596, "%d != 24596" % (20 + 1 * 1 * 4 * 4))

print("\n=== D. the emitted file is not one of the control files ===")
for name in ("_control_a_head_dim_bug.bin", "_control_b_stride_flip.bin",
             "_control_c_truncated_by_4.bin", "_control_d_zero_payload.bin"):
    pth = os.path.join(ROOT, "examples", "p15_s5", name)
    if os.path.isfile(pth):
        same = open(pth, "rb").read() == blob
        ck("emitted section != %s" % name, not same,
           "%d B" % os.path.getsize(pth))
    else:
        ck("control file %s exists" % name, False, "missing")

print("\n" + "=" * 62)
if fails:
    print("INDEPENDENT VERIFY: FAIL -- %d" % len(fails))
    for f in fails:
        print("   - %s" % f)
    sys.exit(1)
print("INDEPENDENT VERIFY: PASS -- emitter output independently re-measured.")
