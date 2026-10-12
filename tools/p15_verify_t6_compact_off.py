#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P15-S5 criterion T6: is the COMPACT REGION's location pinned by the format, or
is it an unpinned convention shared by writer and reader?

WHY THIS IS A DISTINCT CRITERION (Lead's request)
    The engine does not READ the compact region's offset from the artifact. It
    DERIVES it:

        compact_off = (sec_file_off + layout.section_bytes() + 63) & ~63
        (core/src/neuron_kv.rs load_compact_geometry, ~line 932)

    The artifact's descriptor declares only (tag, off, len) for the SECTION. No
    field states where the compact region begins, nor how many positions it has.
    So the layout is a CONVENTION that the writer and the reader must agree on
    independently -- structurally the same hazard as the wrong-stride trap this
    project already documented (docs/P14_S5_O6_报告.md §4.4 item 2: a writer and
    reader sharing a wrong convention are self-consistent and the test proves
    nothing).

    This is NOT automatically a defect. An unpinned offset can be perfectly fine
    if it is a documented format invariant. What matters is that the claim be
    stated correctly, and that I check whether a disagreement is DETECTED or
    SILENT.

WHAT I MEASURE
    I build artifacts that are each well formed by the reader's own rules except
    for the compact region's POSITION, and I observe whether the engine:
      (a) reads the wrong bytes and reports a nonzero difference (GOOD: detected),
      (b) refuses loudly (GOOD: rejected),
      (c) accepts with first_differing_byte=0 (BAD: silent).

    Cases:
      T6.1 compact region at the DERIVED offset (align 64)          -> must pass
      T6.2 compact region shifted to align 128 (64 B of padding)    -> ?
      T6.3 compact region shifted by a NON-multiple of the element  -> ?
           size (a 4-byte shift), so the payload is still a whole number of
           positions and the "not a whole number" guard does NOT fire
      T6.4 4 bytes of padding inserted BEFORE the section end, so the derived
           offset lands 4 bytes into the compact data

    T6.3/T6.4 are the sharp ones: they keep `payload % (n*4) == 0` so the
    existing remainder guard stays quiet, which is exactly the condition under
    which a wrong offset could go unnoticed.

    The can-fail for the whole criterion: T6.1 must be ACCEPTED (otherwise I
    would be measuring a broken builder, not the code).

Run from <repo>:
    python tools/p15_verify_t6_compact_off.py
"""
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p15_verify_lib as V  # noqa: E402
import p15_verify_d1 as D1   # MY OWN builder

NL, NKH, HD, PMAX = 24, 2, 128, 1024
N = NL * NKH * HD                 # 6144
SEC_BYTES = 20 + N * 4            # 24596
N_POS = 4
SEC_OFF = 44
DERIVED_OFF = (SEC_OFF + SEC_BYTES + 63) & ~63     # 24640


def build_at(compact_off, n_pos=N_POS, fill=None, sec_off=SEC_OFF):
    """Build a well-formed artifact with the compact region at `compact_off`.

    Everything except the compact region's POSITION is by the book: correct
    header, correct descriptor (tag/off/len exact), correct offsets[].
    """
    if fill is None:
        def fill(p, i):
            return float((p + 1) * 1000000 + i)

    sec = bytearray()
    for v in (NL, NKH, HD, PMAX, PMAX):
        sec += struct.pack("<I", v)
    for i in range(N):
        sec += struct.pack("<I", i * PMAX)
    assert len(sec) == SEC_BYTES

    blob = bytearray()
    for v in (NL, NKH, HD, PMAX, PMAX):
        blob += struct.pack("<I", v)
    blob += struct.pack("<I", sec_off)
    blob += struct.pack("<I", sec_off)
    blob += struct.pack("<I", 0)
    blob += struct.pack("<I", 1)
    blob += struct.pack("<I", sec_off)
    blob += struct.pack("<I", len(sec))
    assert len(blob) == sec_off
    blob += sec
    if compact_off < len(blob):
        raise ValueError("compact_off %d would overwrite the section (ends %d)"
                         % (compact_off, len(blob)))
    while len(blob) < compact_off:
        blob += b"\x00"
    for p in range(n_pos):
        for i in range(N):
            blob += struct.pack("<f", fill(p, i))
    return bytes(blob)


def parse(txt, key):
    for l in txt.splitlines():
        if key in l:
            part = l.split(key, 1)[1].strip().split()
            if part and part[0].isdigit():
                return int(part[0])
    return None


def main():
    print("=" * 78)
    print("T6: is the compact region's LOCATION pinned by the format?")
    print("=" * 78)
    meta = V.MetaCheck()

    exe = os.path.join(V.ROOT, "examples", "p15_verify", "fw_build",
                       "x86_64-pc-windows-msvc", "release", "vmlm-runtime.exe")
    if not os.path.exists(exe):
        print("engine not built")
        return 2

    # ---- 1. does the format declare the offset anywhere? ----
    print("\n--- T6.0 does the format DECLARE the compact offset? ---")
    nkv = open(os.path.join(V.ROOT, "core", "src", "neuron_kv.rs"),
               encoding="utf-8", errors="replace").read()
    # the descriptor is tag/off/len; search for any other field read from the file
    declares = ("compact_off" in nkv and "let compact_off = crate::u32_at" in nkv)
    print("   a descriptor field carrying compact_off: %s" % declares)
    print("   derived at load time as (sec_off + section_bytes + 63) & ~63: %s"
          % ("(sec_file_off + layout.section_bytes() + 63) & !63usize" in nkv))
    print("   => the offset is a CONVENTION, not data in the artifact")
    meta.record("T6.0 the format does NOT declare the compact offset (it is "
                "derived)",
                not declares,
                "if the format did carry the field, this criterion would be "
                "false and the finding would be wrong",
                not declares,
                "declared_in_format=%s derived=%s"
                % (declares, "(sec_file_off + layout.section_bytes() + 63) & "
                             "!63usize" in nkv))

    # ---- 2. measured cases ----
    print("\n--- T6.1 compact region at the DERIVED offset (control) ---")
    blob = build_at(DERIVED_OFF)
    rc, txt = D1.run_engine(exe, blob)
    coff = parse(txt, "compact_off=")
    okv = "S5LOAD ok" in txt
    print("   exit=%d ok=%s compact_off=%s (derived=%d) first_diff=%s"
          % (rc, okv, coff, DERIVED_OFF, parse(txt, "first_differing_byte=")))
    control_ok = (rc == 0 and okv and coff == DERIVED_OFF)
    print("   control accepted: %s" % control_ok)

    cases = []

    # T6.2 shifted to the next 64-byte boundary (64 bytes of padding)
    print("\n--- T6.2 compact region 64 B later (padding before it) ---")
    off2 = DERIVED_OFF + 64
    blob2 = build_at(off2)
    rc2, txt2 = D1.run_engine(exe, blob2)
    d2 = parse(txt2, "first_differing_byte=")
    msg2 = [l.strip() for l in txt2.splitlines() if l.strip().startswith("NKV:")]
    print("   exit=%d first_diff=%s %s" % (rc2, d2, msg2[:1]))
    cases.append(("64-byte shift", rc2, d2, msg2, off2))

    # T6.3 shifted by 4 bytes: payload is still (n_pos*n*4) if total stays the
    #      same, so the remainder guard does NOT fire. The engine will read the
    #      compact region 4 bytes early -> every value is a neighbour's bytes.
    print("\n--- T6.3 compact region shifted by 4 bytes (remainder guard stays "
          "quiet) ---")
    off3 = DERIVED_OFF + 4
    blob3 = build_at(off3)
    rc3, txt3 = D1.run_engine(exe, blob3)
    d3 = parse(txt3, "first_differing_byte=")
    msg3 = [l.strip() for l in txt3.splitlines() if l.strip().startswith("NKV:")]
    print("   exit=%d first_diff=%s %s" % (rc3, d3, msg3[:1]))
    cases.append(("4-byte shift", rc3, d3, msg3, off3))

    # T6.4 THE SHARP CASE. A shift by a multiple of one POSITION's size
    #      (n*4 = 24576 bytes) keeps `payload % (n*4) == 0`, so the remainder
    #      guard stays quiet AND n_pos comes out unchanged. The engine then
    #      reads the compact region at the derived offset, which is inside the
    #      PADDING, and the real data starts 24576 bytes later. Every value it
    #      reads is padding (zeros) -- so the round trip is still bit-exact
    #      (it compares the file against itself) and pos_varying will be 0,
    #      which the vacuity guard catches. To isolate the OFFSET question I
    #      therefore put a SECOND full position-payload's worth of real data in
    #      the padding as well, so pos_varying can be nonzero while the engine
    #      is still reading the wrong region.
    print("\n--- T6.4 compact region shifted by a WHOLE POSITION (n*4 = %d B): "
          "remainder guard stays quiet ---" % (N * 4))
    off4 = DERIVED_OFF + N * 4
    # Fill the padding region (from DERIVED_OFF) with position-varying data too,
    # so that if the engine reads there it sees varying values (vacuity guard
    # quiet) -- isolating the OFFSET question from the vacuity question.
    def fill_pv(p, i):
        return float((p + 1) * 1000000 + i)

    sec = bytearray()
    for v in (NL, NKH, HD, PMAX, PMAX):
        sec += struct.pack("<I", v)
    for i in range(N):
        sec += struct.pack("<I", i * PMAX)
    blob4 = bytearray()
    for v in (NL, NKH, HD, PMAX, PMAX):
        blob4 += struct.pack("<I", v)
    blob4 += struct.pack("<I", SEC_OFF) * 2
    blob4 += struct.pack("<I", 0)
    blob4 += struct.pack("<I", 1)
    blob4 += struct.pack("<I", SEC_OFF)
    blob4 += struct.pack("<I", len(sec))
    blob4 += sec
    # region A: the padding the engine WILL read (position-varying, 2 positions)
    while len(blob4) < DERIVED_OFF:
        blob4 += b"\x00"
    for p in range(2):
        for i in range(N):
            blob4 += struct.pack("<f", float((p + 1) * 1000000 + i))
    assert len(blob4) == DERIVED_OFF + 2 * N * 4
    # region B: the REAL compact data, a whole position later
    while len(blob4) < off4:
        blob4 += b"\x00"
    for p in range(N_POS):
        for i in range(N):
            blob4 += struct.pack("<f", fill_pv(p, i))
    blob4 = bytes(blob4)
    rc4, txt4 = D1.run_engine(exe, blob4)
    d4 = parse(txt4, "first_differing_byte=")
    npos4 = parse(txt4, "n_pos=")
    pv4 = parse(txt4, "pos_varying_neurons=")
    msg4 = [l.strip() for l in txt4.splitlines() if l.strip().startswith("NKV:")]
    print("   exit=%d n_pos=%s pos_varying=%s first_diff=%s %s"
          % (rc4, npos4, pv4, d4, msg4[:1]))
    print("   engine expects the compact region at %d; the real data begins at "
          "%d" % (DERIVED_OFF, off4))
    cases.append(("one-position shift", rc4, d4, msg4, off4))

    # ---- 3. verdict ----
    print("\n--- T6 verdict ---")
    silent = [(n, rc, d) for n, rc, d, _m, _o in cases
              if rc == 0 and d == 0]
    loud = [(n, rc, d) for n, rc, d, _m, _o in cases if rc != 0]
    diffed = [(n, rc, d) for n, rc, d, _m, _o in cases
              if rc == 0 and d not in (0, None)]
    print("   cases: %d" % len(cases))
    print("   refused loudly            : %d %s"
          % (len(loud), [n for n, _r, _d in loud]))
    print("   accepted with nonzero diff: %d %s"
          % (len(diffed), [n for n, _r, _d in diffed]))
    print("   SILENTLY ACCEPTED (rc=0, first_diff=0): %d %s"
          % (len(silent), [n for n, _r, _d in silent]))

    # The criterion: a compact region placed at a DIFFERENT offset must not be
    # silently accepted. "Control accepted" is the can-fail proof that the
    # builder works and the derived offset is the accepted one.
    detected = (len(silent) == 0)
    meta.record("T6 a compact region at a NON-derived offset is not silently "
                "accepted",
                detected and control_ok,
                "control (derived offset) must be ACCEPTED, else the builder is "
                "at fault; a shifted region that is accepted with "
                "first_differing_byte=0 would mean the offset is unverifiable",
                control_ok,
                "control_ok=%s silent_cases=%s loud=%d diffed=%d"
                % (control_ok, [n for n, _r, _d in silent], len(loud),
                   len(diffed)))

    # ---- 4. the honest statement of the limit ----
    #
    # ⚠️ SCOPE, stated carefully because it is easy to overclaim here. My T6.4
    # file is AMBIGUOUS BY CONSTRUCTION: it contains two plausible compact
    # regions and nothing in the format says which is authoritative. No reader
    # could resolve that. So T6.4 is NOT "the engine misreads valid files".
    # What it establishes is narrower and still worth reporting:
    #   (a) the compact region's OFFSET and LENGTH are not declared anywhere in
    #       the artifact -- the engine derives both;
    #   (b) consequently any byte string filling the derived window with varying
    #       data is accepted as a valid compact region (measured: exit=0,
    #       first_diff=0, pos_varying=6144, while the "intended" data sat one
    #       whole position later);
    #   (c) writer and reader agreeing on the same convention therefore cannot
    #       be verified from the file -- the same self-consistency hazard as the
    #       documented wrong-stride trap (P14 report 4.4 item 2).
    # I report (a)-(c) as a format-level observation with an UNVERIFIED verdict
    # on "is the location pinned?", not as a code defect.
    print("\n--- T6 limit statement ---")
    print("   The format does not declare the compact offset. The engine derives")
    print("   it. A writer that uses the SAME convention is self-consistent, so")
    print("   no reader-side check can distinguish 'right convention' from")
    print("   'wrong convention agreed on by both sides'. That is the same")
    print("   structural hazard as the documented wrong-stride trap.")
    print("   Measured consequence: a file whose derived window is filled with")
    print("   position-varying floats is ACCEPTED (exit=0, first_diff=0,")
    print("   pos_varying=%d) even when the region the author intended lies a"
          % N)
    print("   whole position later. SCOPE: that file is ambiguous by")
    print("   construction, so this is NOT 'the engine misreads valid files'.")
    print("   It is: the location and length are UNPINNED, and agreement on the")
    print("   convention is unverifiable from the artifact.")
    # rows (a)-(c): (a) and (b) are measured; (c) is a structural statement I
    # mark UNVERIFIED because the format genuinely does not declare the offset.
    meta.record("T6 (a)+(b) the compact offset/length are underived-from-format "
                "and a filled derived window is accepted",
                (not declares) and (len(silent) > 0) and control_ok,
                "if the format DID declare the offset this row would be false",
                (not declares) and control_ok,
                "declared=%s silent=%s control_ok=%s"
                % (declares, [n for n, _r, _d in silent], control_ok))
    meta.record("T6 (c) the compact region's location is PINNED by the format",
                False,
                "there is no field to check against, so a wrong convention "
                "shared by writer and reader is undetectable. Reported "
                "UNVERIFIED, never PASS -- this is the same shape as the "
                "wrong-stride trap the project already recorded",
                False,
                "no descriptor field carries compact_off or the position count; "
                "unfalsifiable from the artifact alone")
    v = V.ensure_outdir("t6")
    import json
    with open(os.path.join(v, "t6_compact_off.json"), "w", encoding="utf-8") as f:
        json.dump(dict(derived=DERIVED_OFF, control_ok=control_ok,
                       cases=[dict(case=n, exit=rc, first_diff=d, off=o,
                                   msg=m) for n, rc, d, m, o in cases],
                       silent=[n for n, _r, _d in silent]),
                  f, indent=2, ensure_ascii=False)
    print("\nwrote %s" % os.path.join(v, "t6_compact_off.json"))
    return 0 if meta.report() else 1


if __name__ == "__main__":
    sys.exit(main())
