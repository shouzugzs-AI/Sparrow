#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P15-S5 verifier instrument SELF-TEST (task-4, independent verifier).

This script does NOT test the S5 implementation. It tests MY INSTRUMENTS, and
it does so before task-1/task-2 land, because an instrument that has never been
shown to fail is not an instrument. The project has that lesson on record twice:
  * docs/P13_报告.md §3.2 -- a probe with an EMPTY PATCH reported "0 B" for a
    change that really cost 3,584 B. "That 0 B was an empty measurement, not a
    conclusion."
  * docs/P14_S5_O6_报告.md §4.4 -- the implementer's wrong-stride meta-check was
    self-consistent (same wrong stride on writer and reader), PASSED, and
    proved nothing. Corrected, it showed 115/120 differences.

WHAT IS EXERCISED HERE
    1. rdu32 / parse_section_primitives  -- on SYNTHETIC bytes I construct, so
       I know the ground truth exactly, and on the real format.rs constants.
    2. LayoutPrimitives                  -- address arithmetic against
       hand-computed values, including the 128x trap and the off-by-one.
    3. contiguity_negative_controls      -- the contiguity predicate must reject
       a split range, a gather list, and a wrong-stride reader.
    4. MetaCheck.self_test               -- the meta-check must refuse to
       certify a rubber stamp and must refuse to certify a reject-everything
       predicate, while certifying correct usage.
    5. build_both_ways measurement       -- the size apparatus must report a
       NONZERO delta for a variant that genuinely differs, or its "0 B" reading
       is an empty measurement. The positive control is --features train, which
       P14 measured at +20480 B.

Run from <repo>:
    python tools/p15_verify_instruments_selftest.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import p15_verify_lib as V  # noqa: E402

ROWS = []


def row(name, ok, detail):
    ROWS.append((name, bool(ok), detail))
    print("  [%s] %-58s %s" % ("PASS" if ok else "FAIL", name, detail))
    return bool(ok)


# ---------------------------------------------------------------------------
# 0. source-text cross-check: the constants I re-derive from must be the real ones
# ---------------------------------------------------------------------------
def t0_source_text():
    print("\n=== 0. constants re-read from the Rust SOURCE TEXT (not imported) ===")
    cc = V.cross_check_source_text()
    print("      " + str(cc))
    ok = row("format.rs NKV_HDR == 20", cc["NKV_HDR"] == 20,
             "read %s" % cc["NKV_HDR"])
    ok &= row("format.rs NKV_ENTRY == 4", cc["NKV_ENTRY"] == 4,
              "read %s" % cc["NKV_ENTRY"])
    ok &= row("format.rs SEC_NEURON_KV == 1", cc["SEC_NEURON_KV"] == 1,
              "read %s" % cc["SEC_NEURON_KV"])
    ok &= row("neuron_kv.rs still has neuron_base = idx*stride",
              cc["has_base_formula"], "regex on the Rust text")
    ok &= row("neuron_kv.rs still matches the 128x BUG pattern (stride >= "
              "head_dim*max_seq)", cc["has_bad_head_dim_stride"],
              "the corrected source is supposed to KEEP this as a "
              "documented-wrong record; if this goes False the record was "
              "deleted, which is a different problem")
    ok &= row("my module constants equal the parsed ones",
              V.NKV_HDR == cc["NKV_HDR"] and V.NKV_ENTRY == cc["NKV_ENTRY"],
              "mine %d/%d" % (V.NKV_HDR, V.NKV_ENTRY))
    return ok


# ---------------------------------------------------------------------------
# 1. byte parser on synthetic ground truth, then its can-fail controls
# ---------------------------------------------------------------------------
def synth_section(nl, nkh, hd, maxseq, stride, entry_fn=None):
    import struct
    n = nl * nkh * hd
    b = bytearray()
    for v in (nl, nkh, hd, maxseq, stride):
        b += struct.pack("<I", v)
    for i in range(n):
        v = (i * stride) if entry_fn is None else entry_fn(i, stride)
        b += struct.pack("<I", v & 0xFFFFFFFF)
    return bytes(b)


def t1_parser():
    print("\n=== 1. section byte parser: known ground truth, then can-fail ===")
    blob = synth_section(3, 2, 4, 16, 16)
    p = V.parse_section_primitives(blob)
    ok = row("synthetic 3x2x4x16 section parses to the bytes I wrote",
             p["header"] == {"n_layers": 3, "n_kv_heads": 2, "head_dim": 4,
                             "max_seq_per_neuron": 16, "stride": 16}
             and p["n_neurons"] == 24 and len(p["offsets"]) == 24,
             "n_neurons=%d hdr=%s" % (p["n_neurons"], p["header"]))
    ok &= row("expected size = 20 + 24*4 = 116", p["expected_section_bytes"] == 116,
              "got %d" % p["expected_section_bytes"])
    ok &= row("all 24 offsets == i*stride", p["offsets"] == [i * 16 for i in range(24)],
              "first=%s last=%s" % (p["offsets"][:3], p["offsets"][-1]))

    # CAN-FAIL 1: the 128x bug, i.e. entry[i] = i*head_dim*stride
    bad = synth_section(3, 2, 4, 16, 16, entry_fn=lambda i, s: i * 4 * s)
    pb = V.parse_section_primitives(bad)
    mc = V.section_mismatch_counts(pb)
    ok &= row("CAN-FAIL: entry[i]=i*head_dim*stride is DETECTED",
              mc["n_mismatch"] > 0,
              "%d/%d mismatched, first=%s" % (mc["n_mismatch"], mc["n_compared"],
                                              mc["first"]))

    # CAN-FAIL 2: off-by-one on a SINGLE entry
    hd_entries = [i * 16 for i in range(24)]
    hd_entries[7] += 1
    blob3 = bytearray(synth_section(3, 2, 4, 16, 16))
    import struct
    struct.pack_into("<I", blob3, 20 + 7 * 4, hd_entries[7])
    p3 = V.parse_section_primitives(bytes(blob3))
    mc3 = V.section_mismatch_counts(p3)
    ok &= row("CAN-FAIL: a SINGLE off-by-one entry (index 7, +1) is DETECTED",
              mc3["n_mismatch"] == 1 and mc3["first"][0] == 7,
              "%d/%d, first=%s" % (mc3["n_mismatch"], mc3["n_compared"],
                                   mc3["first"]))

    # CAN-FAIL 3: stride header field perturbed
    blob4 = bytearray(synth_section(3, 2, 4, 16, 16))
    struct.pack_into("<I", blob4, 16, 17)
    p4 = V.parse_section_primitives(bytes(blob4))
    mc4 = V.section_mismatch_counts(p4)
    ok &= row("CAN-FAIL: stride header flipped 16->17 is DETECTED",
              mc4["n_mismatch"] > 0,
              "%d/%d mismatched" % (mc4["n_mismatch"], mc4["n_compared"]))

    # CAN-FAIL 4: truncated file must raise, not silently shorten
    trunc = blob[:-4]
    try:
        V.parse_section_primitives(trunc)
        ok &= row("CAN-FAIL: 4-byte truncation must RAISE", False,
                  "parser accepted a truncated section -- silent degradation")
    except (ValueError, IndexError) as e:
        ok &= row("CAN-FAIL: 4-byte truncation RAISES", True, repr(e))

    # CAN-FAIL 5: a section that is all zeros must NOT look 'fine'
    zeros = synth_section(0, 0, 0, 0, 0)
    pz = V.parse_section_primitives(zeros)
    ok &= row("CAN-FAIL: an all-zero header yields n_neurons==0 and the "
              "mirror validate() REJECTS it",
              pz["n_neurons"] == 0 and
              len(V.LayoutPrimitives(0, 0, 0, 0, 0).validate_reasons()) > 0,
              "reasons=%s" % V.LayoutPrimitives(0, 0, 0, 0, 0).validate_reasons())
    return ok


# ---------------------------------------------------------------------------
# 2. address arithmetic against hand-computed ground truth
# ---------------------------------------------------------------------------
def t2_arithmetic():
    print("\n=== 2. address arithmetic vs hand-computed ground truth ===")
    L = V.LayoutPrimitives(24, 2, 128, 1024, 1024)
    ok = row("n_neurons(24,2,128) == 6144", L.n_neurons() == 6144,
             "got %d" % L.n_neurons())
    ok &= row("idx(L=0,h=0,d=0) == 0", L.linear_index(0, 0, 0) == 0, "0")
    ok &= row("idx(L=0,h=0,d=1) == 1 (head_dim innermost)",
              L.linear_index(0, 0, 1) == 1, "got %d" % L.linear_index(0, 0, 1))
    ok &= row("idx(L=0,h=1,d=0) == 128", L.linear_index(0, 1, 0) == 128,
              "got %d" % L.linear_index(0, 1, 0))
    ok &= row("idx(L=1,h=0,d=0) == 256", L.linear_index(1, 0, 0) == 256,
              "got %d" % L.linear_index(1, 0, 0))
    ok &= row("idx(L=23,h=1,d=127) == 6143", L.linear_index(23, 1, 127) == 6143,
              "got %d" % L.linear_index(23, 1, 127))
    ok &= row("tight: base(d=1)-base(d=0) == 1024 (NOT head_dim*1024)",
              L.neuron_base(0, 0, 1) - L.neuron_base(0, 0, 0) == 1024,
              "got %d" % (L.neuron_base(0, 0, 1) - L.neuron_base(0, 0, 0)))
    ok &= row("the 128x trap: with stride=HD*PMAX the spacing would be 131072",
              131072 == 128 * 1024, "arithmetic identity")
    ok &= row("slot_addr(d=0,s=0) == 0 and (d=0,s=1023) == 1023",
              L.slot_addr(0, 0, 0, 0) == 0 and L.slot_addr(0, 0, 0, 1023) == 1023,
              "%d %d" % (L.slot_addr(0, 0, 0, 0), L.slot_addr(0, 0, 0, 1023)))
    # the off-by-one that would land in the NEXT neuron
    ok &= row("CAN-FAIL: slot=1024 concatenates onto neuron idx=1 "
              "(this is WHY slot>=max_seq is a hard fail)",
              L.slot_addr(0, 0, 0, 1024) == L.neuron_base(0, 0, 1),
              "%d == %d" % (L.slot_addr(0, 0, 0, 1024), L.neuron_base(0, 0, 1)))
    # u32 overflow — the masked comparison must catch a wrap
    big = V.LayoutPrimitives(24, 2, 128, 1024, 1024 * 1024)
    ok &= row("CAN-FAIL: a layout whose entries exceed u32 is REJECTED by the "
              "mirror (wrap must not silently agree)",
              "NKV: layout exceeds 32-bit offsets" in big.validate_reasons(),
              "reasons=%s" % big.validate_reasons())
    return ok


# ---------------------------------------------------------------------------
# 3. contiguity predicate and its negative controls
# ---------------------------------------------------------------------------
def t3_contiguity():
    print("\n=== 3. contiguity predicate + negative controls ===")
    ok = row("CAN-FAIL: empty address list is NOT 'contiguous' "
             "(vacuous truth is the empty measurement)",
             not V.is_contiguous([]), "len<2 -> False by construction")
    ok &= row("CAN-FAIL: single address is NOT 'contiguous'",
              not V.is_contiguous([5]), "len<2 -> False")
    ok &= row("a genuine unit-stride run IS contiguous",
              V.is_contiguous([0, 1, 2, 3]), "gaps=[1,1,1]")
    okc, rows = V.contiguity_negative_controls()
    for r in rows:
        extra = ""
        if r["control"] == "wrong_stride_reader":
            extra = (" value_mismatches(self-consistent trap)=%d"
                     % r["value_mismatches_self_consistent_trap"])
        ok &= row("CAN-FAIL control %s detected" % r["control"], r["detected"],
                  "bad_gaps=%s%s" % (r.get("bad_gaps"), extra))
    ok &= okc

    # The self-consistent wrong-stride trap, reproduced so it can be avoided:
    # writer and reader share stride 9; the round trip is PERFECT and proves
    # nothing. This row documents the trap, it does not test the implementation.
    st, n, npos = 9, 4, 9
    data = list(range(n * st))
    enc = {i * st + s: data[i * st + s] for i in range(n) for s in range(npos)}
    dec = [enc[i * st + s] for i in range(n) for s in range(npos)]
    trap_perfect = (dec == data[:n * npos])
    ok &= row("TRAP DOCUMENTED: self-consistent wrong stride round-trips "
              "perfectly (%s) -> cannot discriminate" % trap_perfect,
              trap_perfect, "this is why the real control lays out data with "
                            "the CORRECT stride and gives only the READER a "
                            "wrong one")
    return ok


# ---------------------------------------------------------------------------
# 4. MetaCheck self-test
# ---------------------------------------------------------------------------
def t4_metacheck():
    print("\n=== 4. MetaCheck refuses rubber stamps (its own can-fail control) ===")
    ok, detail = V.MetaCheck().self_test()
    print("      " + str(detail))
    ok &= row("MetaCheck: rubber stamp is NOT certified",
              detail["rubber_stamp_certified"] is False,
              "report() -> %s" % detail["rubber_stamp_certified"])
    ok &= row("MetaCheck: reject-everything is NOT certified",
              detail["reject_all_certified"] is False,
              "report() -> %s" % detail["reject_all_certified"])
    ok &= row("MetaCheck: correct usage IS certified",
              detail["correct_certified"] is True,
              "report() -> %s" % detail["correct_certified"])
    return ok


# ---------------------------------------------------------------------------
# 5. size-measurement apparatus must be able to see a real difference
# ---------------------------------------------------------------------------
def t5_size_apparatus():
    print("\n=== 5. engine-size apparatus: positive control ===")
    print("      engine on disk now: %s bytes  sha256=%s"
          % (V.eng_size(), (V.sha256(V.ENG)[:16] if os.path.exists(V.ENG) else "n/a")))
    # This self-test deliberately does NOT build: the Lead owns the serialized
    # acceptance gate and the shared target dir is a documented measurement
    # hazard. What it verifies is that the MEASUREMENT PRIMITIVE distinguishes
    # two files of different length -- the smallest possible positive control.
    d = V.ensure_outdir("selftest")
    a = os.path.join(d, "_size_probe_a.bin")
    b = os.path.join(d, "_size_probe_b.bin")
    with open(a, "wb") as f:
        f.write(b"\0" * 1000)
    with open(b, "wb") as f:
        f.write(b"\0" * 1000)
        f.write(b"\0" * 20480)
    sa, sb = V.eng_size(a), V.eng_size(b)
    ok = row("size primitive sees 1000 vs 21480", (sa, sb) == (1000, 21480),
             "got %d vs %d" % (sa, sb))
    ok &= row("delta is NOT zero for a genuinely different file (the +20480 "
              "shape of the P14 train positive control)", (sb - sa) == 20480,
              "delta=%d" % (sb - sa))
    ok &= row("CAN-FAIL: the primitive reports None for a missing file "
              "(a missing build must not read as '0 bytes')",
              V.eng_size(os.path.join(d, "does_not_exist.bin")) is None,
              "None")
    return ok


def main():
    print("=" * 74)
    print("P15 verifier instrument SELF-TEST (task-4). No implementation is "
          "touched.")
    print("=" * 74)
    results = [t0_source_text(), t1_parser(), t2_arithmetic(),
               t3_contiguity(), t4_metacheck(), t5_size_apparatus()]
    print("\n" + "=" * 74)
    n_ok = sum(1 for _, ok, _ in ROWS if ok)
    print("instrument self-test: %d/%d PASS" % (n_ok, len(ROWS)))
    allok = all(results)
    print("INSTRUMENTS SOUND: %s" % ("YES" if allok else "NO -- fix before use"))
    V.ensure_outdir()
    import json
    with open(os.path.join(V.OUTDIR, "instrument_selftest.json"), "w",
              encoding="utf-8") as f:
        json.dump([{"name": n, "ok": o, "detail": d} for n, o, d in ROWS],
                  f, indent=2, ensure_ascii=False)
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
