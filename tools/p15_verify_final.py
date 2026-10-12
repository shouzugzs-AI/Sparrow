#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P15-S5 FINAL verification round (task-4 wrap-up). Covers the Lead's 4 points
plus the two boundary rulings.

1. GUARDS NOT WEAKENED. validate_artifact_header no longer reads the layout from
   [0..20) (that region is magic/version/text_off in a real .vmb); it now checks
   only magic==VMLM and section.off>=44, and the layout comes solely from the
   section's own header. So: is the 128x wrong stride still caught, and did any
   guard become dead code?

2. INDEX-ONLY HONESTY. A real production .vmb has its section at EOF with no
   compact payload. The engine must print "index-only ... zero elements moved"
   and NOT print ok, exit 0. And the "no payload" vs "truncated" distinction
   must be genuinely falsifiable.

3. TAUTOLOGY LABELLING. `diff` -> `inverse_bytes_compared`,
   `slot0_ok` -> `slot0_ok_selfconsistent`. Do the comments honestly say these
   verify "the two functions are mutually inverse / the cell we just wrote"
   rather than "the file was read correctly"? Any remaining false claim (e.g.
   direction sensitivity) must be flagged with file:line.

4. SIZES in ISOLATED target dirs (a reused target-dir gives a bogus delta 0).

Boundary rulings:
   B1 "file content corrupted but structurally self-consistent" -> still exit=0.
      Real boundary (needs an external reference the format does not carry)
      => UNVERIFIED, not PASS.
   B2 examples/p15_s5_e2e/e2e_24layer_neuron_kv.vmb magic is 0x18, not VMLM.

Run from <repo>:  python tools/p15_verify_final.py
"""
import json
import os
import re
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p15_verify_lib as V  # noqa: E402
import p15_verify_d1 as D1  # noqa: E402

N = D1.N_NEURONS              # 6144
SEC_BYTES = D1.SEC_BYTES      # 24596
MAGIC_VMLM = 0x4D4C4D56


# ---------------------------------------------------------------------------
# a real-shaped .vmb: 32-byte header (magic/version/offsets), descriptor at 32,
# section after 44. This mirrors what the production builder emits.
# ---------------------------------------------------------------------------
def build_vmb(sec_off=44, n_pos=0, fill=None, magic=MAGIC_VMLM,
              off_field=None, tail=None):
    if fill is None:
        def fill(p, i):
            return float((p + 1) * 4096 + i)
    sec = bytearray()
    for v in (D1.NL, D1.NKH, D1.HD, D1.PMAX, D1.PMAX):
        sec += struct.pack("<I", v)
    for i in range(N):
        sec += struct.pack("<I", i * D1.PMAX)
    assert len(sec) == SEC_BYTES

    b = bytearray()
    b += struct.pack("<I", magic)          # [0] magic
    b += struct.pack("<H", 1)              # [4] version
    b += struct.pack("<H", 0)              # [6] flags
    while len(b) < 32:
        b += b"\x00"
    b += struct.pack("<I", 1)              # [32] tag
    b += struct.pack("<I", sec_off if off_field is None else off_field)
    b += struct.pack("<I", len(sec))       # [40] len
    assert len(b) == sec_off, (len(b), sec_off)
    b += sec
    if n_pos:
        coff = (sec_off + len(sec) + 63) & ~63
        while len(b) < coff:
            b += b"\x00"
        for p in range(n_pos):
            for i in range(N):
                b += struct.pack("<f", fill(p, i))
    if tail is not None:
        b += tail
    return bytes(b)


def run(exe, blob):
    return D1.run_engine(exe, blob)


def parse(txt, key):
    for l in txt.splitlines():
        if key in l:
            tok = l.split(key, 1)[1].strip().split()
            if tok and tok[0].isdigit():
                return int(tok[0])
    return None


def main():
    print("=" * 78)
    print("P15-S5 FINAL verification round")
    print("=" * 78)
    meta = V.MetaCheck()
    ok, d = V.MetaCheck().self_test()
    print("MetaCheck self-test: %s %s" % ("OK" if ok else "BROKEN", d))

    base = os.path.join(V.ROOT, "examples", "p15_verify")
    def exe_for(tag):
        return os.path.join(base, tag, "x86_64-pc-windows-msvc", "release",
                            "vmlm-runtime.exe")
    ON = exe_for("iso_nkv")
    OFF = exe_for("iso_off")
    TR = exe_for("iso_train")
    if not os.path.exists(ON):
        print("build iso_nkv first"); return 2

    nkv_src = open(os.path.join(V.ROOT, "core", "src", "neuron_kv.rs"),
                   encoding="utf-8", errors="replace").read()
    s5_src = open(os.path.join(V.ROOT, "runtime", "src", "s5_load.rs"),
                  encoding="utf-8", errors="replace").read()
    results = {}

    # =====================================================================
    # POINT 1: guards not weakened
    # =====================================================================
    print("\n=== POINT 1. guards not weakened by the header change ===")
    # (a) 128x wrong stride injected into a REAL-shaped vmb.
    b128 = bytearray(build_vmb())
    so = 44
    for i in range(N):
        struct.pack_into("<I", b128, so + 20 + i * 4, (i * 128 * D1.PMAX) & 0xFFFFFFFF)
    rc128, t128 = run(ON, bytes(b128))
    m128 = [l.strip() for l in t128.splitlines() if l.strip().startswith("NKV:")]
    print("  128x stride entries -> exit=%d %s" % (rc128, m128[:1]))
    # (b) bad magic
    rc_m, t_m = run(ON, build_vmb(magic=0x00000018))
    mm = [l.strip() for l in t_m.splitlines() if l.strip().startswith("NKV:")]
    print("  bad magic (0x18)   -> exit=%d %s" % (rc_m, mm[:1]))
    # (c) section off overlapping the header
    rc_o, t_o = run(ON, build_vmb(off_field=16))
    mo = [l.strip() for l in t_o.splitlines() if l.strip().startswith("NKV:")]
    print("  section.off=16     -> exit=%d %s" % (rc_o, mo[:1]))
    # (d) stride < max_seq (the invariant)
    bs = bytearray(build_vmb())
    struct.pack_into("<I", bs, so + 16, 1023)
    rc_s, t_s = run(ON, bytes(bs))
    ms = [l.strip() for l in t_s.splitlines() if l.strip().startswith("NKV:")]
    print("  stride<max_seq     -> exit=%d %s" % (rc_s, ms[:1]))
    # (e) exact-length descriptor guard (S8) still live
    rc_l, t_l = run(ON, build_vmb(sec_off=44, tail=b"\x00" * 4))
    # note: 4 extra bytes with n_pos=0 -> payload=4, not a whole position
    ml = [l.strip() for l in t_l.splitlines() if l.strip().startswith("NKV:")]
    print("  4-byte tail (n_pos=0) -> exit=%d %s" % (rc_l, ml[:1]))

    g_128 = (rc128 == 9 and "offsets[] disagrees with stride" in t128)
    g_magic = (rc_m == 9 and "bad magic" in t_m)
    g_off = (rc_o == 9 and "section offset overlaps" in t_o)
    g_stride = (rc_s == 9 and "stride < max_seq" in t_s)
    print("  128x caught=%s magic caught=%s off caught=%s stride caught=%s"
          % (g_128, g_magic, g_off, g_stride))
    # is validate_artifact_header still CALLED (not dead)?
    called = "validate_artifact_header(d);" in nkv_src
    print("  validate_artifact_header still CALLED: %s" % called)
    meta.record("P1 the 128x wrong stride is still caught after the header change",
                g_128,
                "if the 128x form were accepted, the S5-A defect class would be "
                "re-opened by this change",
                g_128, "exit=%d msg=%s" % (rc128, m128[:1]))
    meta.record("P1 the new header guards fire (magic, off>=44) and no guard "
                "became dead code",
                g_magic and g_off and g_stride and called,
                "each of the three must exit 9 with its own message; if "
                "validate_artifact_header were unreferenced it would be dead",
                g_magic and g_off and g_stride,
                "magic=%s off=%s stride=%s still_called=%s"
                % (g_magic, g_off, g_stride, called))
    results["point1"] = dict(rc128=rc128, m128=m128[:3], magic=g_magic,
                             off=g_off, stride=g_stride, called=called)

    # =====================================================================
    # POINT 2: index-only honesty; no-payload vs truncated distinguishable
    # =====================================================================
    print("\n=== POINT 2. index-only path: honest, and falsifiable ===")
    # a real production shape: section at EOF, no payload at all
    b_idx = build_vmb(sec_off=44, n_pos=0)
    rc_i, t_i = run(ON, b_idx)
    print("  index-only (section at EOF, no payload): exit=%d" % rc_i)
    for l in t_i.splitlines():
        if l.strip():
            print("     | %s" % l.strip()[:150])
    idx_ok = ("index-only" in t_i and "S5LOAD ok" not in t_i)
    print("  prints index-only and NOT ok: %s" % idx_ok)

    # the distinction must be falsifiable: trailing NON-zero bytes are not
    # padding and must hard-fail (the payload becomes a non-whole position
    # count, or a bogus compact region -- either way it must not be blessed)
    b_nz = build_vmb(sec_off=44, n_pos=0, tail=b"\x01\x02\x03\x04")
    rc_nz, t_nz = run(ON, b_nz)
    mnz = [l.strip() for l in t_nz.splitlines() if l.strip().startswith("NKV:")]
    print("  4 NON-zero trailing bytes -> exit=%d %s" % (rc_nz, mnz[:1]))
    nz_refused = (rc_nz == 9)
    # and a full extra position of zero padding is treated as a payload
    b_z = build_vmb(sec_off=44, n_pos=1, fill=lambda p, i: 0.0)
    rc_z, t_z = run(ON, b_z)
    mz = [l.strip() for l in t_z.splitlines() if l.strip().startswith("NKV:")]
    print("  1 position of ZEROS        -> exit=%d %s" % (rc_z, mz[:1]))
    print("  (zeros => pos_varying=0 => vacuity guard, which is the honest read:")
    print("   an all-zero payload carries no position information)")
    meta.record("P2 the index-only path prints index-only and does NOT print ok",
                idx_ok,
                "a path that printed ok with no payload would be the D1 pattern "
                "again; if 'ok' appeared here this criterion must fail",
                idx_ok, "exit=%d, contains ok=%s" % (rc_i, "S5LOAD ok" in t_i))
    meta.record("P2 'no payload' vs 'truncated/non-zero tail' is distinguishable",
                nz_refused and idx_ok,
                "a non-zero tail must not be silently accepted as padding; if "
                "rc were 0 here the distinction would be a rubber stamp",
                nz_refused,
                "index-only exit=%d (accepted, honest); non-zero tail exit=%d "
                "(refused)" % (rc_i, rc_nz))
    results["point2"] = dict(rc_index_only=rc_i, idx_ok=idx_ok,
                             rc_nonzero_tail=rc_nz, m_nonzero=mnz[:3],
                             rc_zero_payload=rc_z, m_zero=mz[:3])

    # =====================================================================
    # POINT 3: tautology labelling honest?
    # =====================================================================
    print("\n=== POINT 3. are the self-consistent predicates labelled honestly? ===")
    # what the source now says about inverse_bytes_compared
    def ctx(src, pat, before=14, after=14):
        m = re.search(pat, src)
        if not m:
            return None
        ls = src[:m.start()].splitlines()
        start = max(0, len(ls) - before)
        return "\n".join((ls + src[m.start():].splitlines()[:after])[start:])
    inv = ctx(nkv_src, r"inverse_bytes_compared", 16, 6)
    print("  --- inverse_bytes_compared context (neuron_kv.rs) ---")
    for l in (inv or "(not found)").splitlines():
        print("     %s" % l.strip()[:120])
    # does it DISCLAIM proving direction?
    disclaim_inv = bool(re.search(
        r"inverse|互逆|自洽|不能证明|not.*proof|不是.*证明|恒等",
        inv or ""))
    # does the source still claim direction sensitivity anywhere?
    false_dir = []
    for i, l in enumerate(s5_src.splitlines(), 1):
        if re.search(r"方向敏感|direction-sensitive|大面积不等|largely unequal", l):
            false_dir.append((i, l.strip()))
    print("\n  remaining 'direction-sensitive' claims in s5_load.rs: %d" % len(false_dir))
    for i, l in false_dir[:5]:
        print("     s5_load.rs:%d %s" % (i, l[:120]))
    # slot0_ok naming
    s0 = "slot0_ok_selfconsistent" in s5_src or "slot0_ok_selfconsistent" in nkv_src
    print("  slot0 predicate renamed to *selfconsistent: %s" % s0)
    meta.record("P3 the inverse/slot0 predicates are labelled as self-consistent "
                "(not as 'the file was read correctly')",
                disclaim_inv and s0,
                "if the source claimed these prove the FILE was read correctly, "
                "that would be a false claim; the disclaimer must be present",
                disclaim_inv and s0,
                "disclaimer present=%s, renamed=%s" % (disclaim_inv, s0))
    meta.record("P3 no residual 'direction-sensitive' false claim remains in "
                "s5_load.rs",
                len(false_dir) == 0,
                "a surviving claim that the check is direction-sensitive would "
                "be the same false comment D3 reported",
                len(false_dir) == 0,
                "found %d: %s" % (len(false_dir), false_dir[:3]))
    results["point3"] = dict(disclaim_inv=disclaim_inv, slot0_renamed=s0,
                             false_dir_claims=false_dir[:5])

    # =====================================================================
    # POINT 4: sizes in isolated target dirs
    # =====================================================================
    print("\n=== POINT 4. sizes, isolated target dirs ===")
    sz = {}
    for tag, p in (("OFF", OFF), ("NKV", ON), ("TRAIN", TR)):
        sz[tag] = os.path.getsize(p) if os.path.exists(p) else None
    print("  OFF   = %s B" % sz["OFF"])
    print("  NKV   = %s B" % sz["NKV"])
    print("  TRAIN = %s B" % sz["TRAIN"])
    print("  NKV - OFF   = %s" % (None if None in (sz["OFF"], sz["NKV"])
                                  else sz["NKV"] - sz["OFF"]))
    print("  TRAIN - OFF = %s  (positive control)" % (None if None in (sz["OFF"], sz["TRAIN"])
                                                      else sz["TRAIN"] - sz["OFF"]))
    delta = None if None in (sz["OFF"], sz["NKV"]) else sz["NKV"] - sz["OFF"]
    ctrl = None if None in (sz["OFF"], sz["TRAIN"]) else sz["TRAIN"] - sz["OFF"]
    print("  Lead's independent numbers: OFF 68096, NKV 81408 (+13312)")
    meta.record("P4 default engine is EXACTLY 68096 B (isolated target dir)",
                sz["OFF"] == 68096,
                "a synthetic 68097 must be distinguishable by the same comparison",
                sz["OFF"] is not None and sz["OFF"] + 1 != 68096,
                "measured OFF=%s" % sz["OFF"])
    meta.record("P4 NKV delta is +13312 B as claimed (isolated target dir)",
                delta == 13312,
                "if the delta were 0 the target dir was reused and the "
                "measurement would be an artefact (s5-engine hit exactly that)",
                delta is not None and delta != 0,
                "NKV-OFF=%s" % delta)
    meta.record("P4 positive control (train) moves, so the apparatus is live",
                ctrl is not None and ctrl != 0,
                "a control that did not move would make the NKV delta "
                "meaningless",
                ctrl is not None and ctrl != 0, "TRAIN-OFF=%s" % ctrl)
    results["point4"] = dict(off=sz["OFF"], nkv=sz["NKV"], train=sz["TRAIN"],
                             delta=delta, control=ctrl)

    # =====================================================================
    # BOUNDARY B1: content corrupted but structurally self-consistent
    # =====================================================================
    print("\n=== BOUNDARY B1. corrupted content, structurally self-consistent ===")
    # take the good artifact and flip some payload bits, keeping all structure
    good = build_vmb(n_pos=4)
    bad = bytearray(good)
    coff = (44 + SEC_BYTES + 63) & ~63
    for k in range(coff, coff + 4000, 4):        # scribble in the payload
        struct.pack_into("<f", bad, k, 12345.5)
    rc_c, t_c = run(ON, bytes(bad))
    first = parse(t_c, "first_differing_byte=")
    print("  payload scribbled (structure intact): exit=%d first_differing=%s"
          % (rc_c, first))
    b1_undetected = (rc_c == 0)
    print("  => corruption is UNDETECTED: %s" % b1_undetected)
    meta.record("B1 content corruption with intact structure is DETECTED",
                not b1_undetected,
                "detecting it would require an external reference the format "
                "does not carry (no checksum, no expected values); this "
                "criterion is reported UNVERIFIED, never PASS",
                False,
                "exit=%d -> the format has no redundancy to detect it; "
                "UNVERIFIED boundary, not a defect" % rc_c)
    results["boundary_B1"] = dict(exit=rc_c, undetected=b1_undetected)

    # =====================================================================
    # BOUNDARY B2: the e2e artifact's magic
    # =====================================================================
    print("\n=== BOUNDARY B2. e2e artifact magic ===")
    e2e = os.path.join(V.ROOT, "examples", "p15_s5_e2e",
                       "e2e_24layer_neuron_kv.vmb")
    if os.path.exists(e2e):
        with open(e2e, "rb") as f:
            head = f.read(48)
        mg = V.rdu32(head, 0)
        tag = V.rdu32(head, 32)
        off = V.rdu32(head, 36)
        ln = V.rdu32(head, 40)
        print("  %s" % os.path.relpath(e2e, V.ROOT))
        print("  magic = 0x%08X (VMLM would be 0x%08X) -> is VMLM: %s"
              % (mg, MAGIC_VMLM, mg == MAGIC_VMLM))
        print("  descriptor tag=%d off=%d len=%d" % (tag, off, ln))
        # and the engine's ruling on it
        rc_b2, t_b2 = run(ON, open(e2e, "rb").read())
        mb2 = [l.strip() for l in t_b2.splitlines()
               if l.strip().startswith("NKV:")]
        print("  engine on that blob: exit=%d %s" % (rc_b2, mb2[:1]))
        b2_nonvmlm = (mg != MAGIC_VMLM)
        b2_refused = (rc_b2 != 0)
        meta.record("B2 the e2e artifact is NOT a VMLM .vmb and the engine "
                    "refuses it",
                    b2_nonvmlm and b2_refused,
                    "if the engine accepted a non-VMLM blob the magic guard "
                    "would be ineffective",
                    b2_refused,
                    "magic=0x%08X exit=%d msg=%s" % (mg, rc_b2, mb2[:1]))
        results["boundary_B2"] = dict(magic=mg, is_vmlm=mg == MAGIC_VMLM,
                                      exit=rc_b2, msg=mb2[:2])
    else:
        print("  e2e artifact absent")
        meta.record("B2 the e2e artifact is NOT a VMLM .vmb", False,
                    "file absent", False, "not found")

    out = os.path.join(V.ensure_outdir("final"), "final_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nwrote %s" % out)
    return 0 if meta.report() else 1


if __name__ == "__main__":
    sys.exit(main())
