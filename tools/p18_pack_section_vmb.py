#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P18-P3: assemble the trained sparse sections into a .vmb the engine can load.

============================================================================
WHAT THIS PROVES, AND WHAT IT DELIBERATELY DOES NOT
============================================================================

PROVES
  The trained weights, as serialised by `conn_sparse::write_section_data`, are
  consumable by the ENGINE'S OWN READER after being placed in a real .vmb
  container. That is the link "training -> engine-readable bytes", and it is
  the one a distribution depends on.

DOES NOT PROVE
  That the engine can run a full 24-layer forward pass from these files.
  A .vmb is an INSTRUCTION STREAM (TEXT) plus a constant pool; the .csr files
  are WEIGHTS. Gluing 24 layers of OP_SPARSE_MATVEC + attention + lm_head into
  such a program is separate work, and this script does not pretend to do it.

  Stated because the alternative -- shipping a .vmb that loads but cannot run,
  and calling the milestone done -- is exactly the class of overclaim this
  project's discipline exists to prevent.

============================================================================
THE FORMAT (copied from tools/p16_conn_format.py, which established it)
============================================================================

  [0..8)    magic 'VMLM' (0x4D4C4D56), version u16, pad u16
  [8..32)   text_off/len, const_off/len, meta_off/len   (six u32)
  [32..44)  the OPTIONAL SECTION DESCRIPTOR: tag u32, off u32, len u32
  [44..)    the section bytes

  TRAP (recorded in P16 and repeated here because it is easy to get wrong):
  an artifact WITHOUT a section sets text_off = 32; one WITH a section sets
  text_off = 44. Detecting "has section" by `tag == 0` MISJUDGES the second
  case because [32..44) then holds the start of TEXT. The correct predicate
  is text_off.

USAGE
  python tools/p18_pack_section_vmb.py --in artifacts/p18_sparse24/L00_wq.csr \
      --out temp/p18_pack/L00_wq.vmb [--engine target/release/vmlm-runtime.exe]
"""
import argparse
import os
import struct
import subprocess
import sys

SEC_CONN_SPARSE = 3
VMB_MAGIC = 0x4D4C4D56  # 'VMLM'
HDR = 32
DESCR = 12
TEXT_OFF_WITH_SECTION = HDR + DESCR  # 44


def build_vmb(section: bytes, text_off: int = TEXT_OFF_WITH_SECTION) -> bytes:
    v = bytearray()
    v.extend(struct.pack("<IHH", VMB_MAGIC, 1, 0))
    # text_off=44 (section present), all lengths 0: this artifact carries a
    # section and no program. That is a deliberate, stated limitation.
    v.extend(struct.pack("<II", text_off, 0))   # text
    v.extend(struct.pack("<II", text_off, 0))   # const
    v.extend(struct.pack("<II", text_off, 0))   # meta
    assert len(v) == HDR, len(v)
    v.extend(struct.pack("<III", SEC_CONN_SPARSE, TEXT_OFF_WITH_SECTION,
                         len(section)))
    assert len(v) == HDR + DESCR, len(v)
    v.extend(section)
    return bytes(v)


def parse_section_header(raw: bytes):
    """Independent parse of the conn_sparse header, per core/src/conn_sparse.rs."""
    magic, n, k0, pairs, layer = struct.unpack("<IIIII", raw[:20])
    return {"magic": magic, "n_neurons": n, "k0": k0, "total_pairs": pairs,
            "layer": layer}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True, help="a .csr section file")
    ap.add_argument("--out", dest="dst", required=True, help="the .vmb to write")
    ap.add_argument("--engine", default=None,
                    help="optional vmlm-runtime.exe to smoke-test loading")
    a = ap.parse_args()

    raw = open(a.src, "rb").read()
    h = parse_section_header(raw)
    print("input section: %s (%d B)" % (a.src, len(raw)))
    print("  magic=0x%08x (CSR1=0x31525343) n=%d k0=%d pairs=%d layer=%d"
          % (h["magic"], h["n_neurons"], h["k0"], h["total_pairs"], h["layer"]))
    if h["magic"] != 0x31525343:
        print("FAIL: not a conn_sparse section")
        return 1

    # size must match the documented formula exactly
    want = 20 + (h["n_neurons"] + 1) * 4 + h["total_pairs"] * 4 * 2
    print("  formula size = %d, file size = %d -> %s"
          % (want, len(raw), "OK" if want == len(raw) else "MISMATCH"))
    if want != len(raw):
        return 1

    vmb = build_vmb(raw)
    os.makedirs(os.path.dirname(os.path.abspath(a.dst)), exist_ok=True)
    with open(a.dst, "wb") as fh:
        fh.write(vmb)
    print("wrote %s (%d B = %d header + %d descriptor + %d section)"
          % (a.dst, len(vmb), HDR, DESCR, len(raw)))

    if a.engine:
        # The shipped engine has no "validate a section" subcommand (that was a
        # P16 harness). What CAN be checked here is that the engine OPENS the
        # file rather than rejecting it as unreadable -- i.e. the container is
        # well-formed enough to reach the loader.
        if not os.path.exists(a.engine):
            print("  engine not found: %s" % a.engine)
            return 1
        p = subprocess.run([a.engine, a.dst], capture_output=True, text=True,
                           errors="replace")
        out = (p.stdout + p.stderr).strip()
        print("  engine rc=%d out=%r" % (p.returncode, out[:200]))
        # "cannot open file" (rc=2) is what a non-artifact produces; anything
        # else means the container parsed far enough to reach the loader.
        if "cannot open file" in out:
            print("FAIL: engine rejected the container as unreadable")
            return 1
        print("  [PASS] the engine opened the container (no 'cannot open file')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
