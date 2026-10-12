#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P15-S5 D6: the D4 reordering made the no-section case report the WRONG reason.

WHAT CHANGED
    D4's fix moved validate_artifact_header(d) BEFORE read_section_declared in
    load_validate (core/src/neuron_kv.rs). That makes the `off >= 44` guard
    REACHABLE, which is what I asked for. But validate_artifact_header runs
    before load_find_section, and it checks `off < 44` on a descriptor that is
    only meaningful when a section EXISTS.

A PRODUCTION ARTIFACT WITH NO SECTION STILL HAS A DESCRIPTOR REGION
    The .vmb container reserves [32..44) for the descriptor. A build that emits
    no neuron_kv section leaves those bytes as ZERO:
        tag = 0, off = 0, len = 0
    tag == 0 is the DOCUMENTED "no section" value (format.rs: SEC_NEURON_KV's 0
    is reserved for "no such section"). The correct message for that input is
    `NKV: artifact has no neuron_kv section`.

MEASURED (the real production control artifact)
    examples/p15_s5_prod/prod2/m.vmb.nosection, 520210944 bytes
      tag=0  off=0  len=0
    Engine: exit=9  "NKV: section offset overlaps the .vmb header"
    The real reason is "there is no section"; `off=0` is not an overlap, it is
    the absence of a descriptor. The message points the reader at the wrong
    place -- exactly the class of cost docs/P13_报告.md §2 records ("报错但不说
    哪里错" costs an extra bisection round).

SCOPE / SEVERITY
    NOT a safety hole: the input IS refused (exit 9), which is right. This is a
    DIAGNOSTIC regression introduced by the D4 fix: D4 made one guard reachable
    and, as a side effect, pre-empted a different guard's message for a
    different, more common condition. Both my earlier D4 report and this one are
    about which SPECIFIC message the reader sees -- the project's own discipline
    is that a guard must name its own condition.

RECOMMENDED FIX (implementer's)
    In validate_artifact_header, check the tag/absence case first and emit the
    "no section" message, or gate the off<44 check on `tag != 0`; alternatively
    call load_find_section (which already distinguishes tag==0) before
    validate_artifact_header and keep only the magic check in the header
    validator.

Run from <repo>:  python tools/p15_verify_d6_nosection.py
"""
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p15_verify_lib as V  # noqa: E402
import p15_verify_d1 as D1  # noqa: E402

MAGIC = 0x4D4C4D56


def build(off_field, tag, magic=MAGIC, with_section=True, sec_at_off=True):
    """Build a .vmb with a chosen descriptor and optionally a real section."""
    sec = bytearray()
    for v in (D1.NL, D1.NKH, D1.HD, D1.PMAX, D1.PMAX):
        sec += struct.pack("<I", v)
    for i in range(D1.N_NEURONS):
        sec += struct.pack("<I", i * D1.PMAX)
    b = bytearray(44 + len(sec) + 128)
    struct.pack_into("<I", b, 0, magic)
    struct.pack_into("<H", b, 4, 1)
    struct.pack_into("<H", b, 6, 0)
    struct.pack_into("<I", b, 32, tag)
    struct.pack_into("<I", b, 36, off_field)
    struct.pack_into("<I", b, 40, len(sec) if with_section else 0)
    if with_section and sec_at_off and off_field + len(sec) <= len(b):
        b[off_field:off_field + len(sec)] = sec
    return bytes(b)


def main():
    print("=" * 78)
    print("D6: the no-section case reports the WRONG reason after the D4 fix")
    print("=" * 78)
    meta = V.MetaCheck()

    exe = os.path.join(V.ROOT, "examples", "p15_verify", "iso_nkv",
                       "x86_64-pc-windows-msvc", "release", "vmlm-runtime.exe")
    if not os.path.exists(exe):
        print("engine not built"); return 2

    # ---- 1. the real production control artifact ----
    print("\n--- 1. the REAL production no-section artifact ---")
    prod = os.path.join(V.ROOT, "examples", "p15_s5_prod", "prod2",
                        "m.vmb.nosection")
    prod_row = None
    if os.path.exists(prod):
        with open(prod, "rb") as f:
            head = f.read(44)
        tag = V.rdu32(head, 32)
        off = V.rdu32(head, 36)
        ln = V.rdu32(head, 40)
        print("   %s" % os.path.relpath(prod, V.ROOT))
        print("   size=%d  tag=%d  off=%d  len=%d"
              % (os.path.getsize(prod), tag, off, ln))
        rc, txt = D1.run_engine(exe, open(prod, "rb").read())
        msgs = [l.strip() for l in txt.splitlines()
                if l.strip().startswith("NKV:")]
        print("   engine: exit=%d  %s" % (rc, msgs[:1]))
        correct_msg = "has no neuron_kv section"
        wrong_msg = "overlaps the .vmb header"
        prod_row = dict(size=os.path.getsize(prod), tag=tag, off=off, len=ln,
                        exit=rc, msg=msgs[0] if msgs else None)
        print("   correct message for tag=0 would be: %r" % correct_msg)
        print("   actual message                    : %r"
              % (msgs[0] if msgs else None))
        meta.record("D6 the no-section production artifact reports 'has no "
                    "neuron_kv section' (its own condition)",
                    any(correct_msg in m for m in msgs),
                    "tag=0 is the documented 'no section' value; the message "
                    "must name THAT condition, not an offset overlap",
                    any(correct_msg in m for m in msgs),
                    "exit=%d msg=%s" % (rc, msgs[:1]))
    else:
        print("   production control artifact not found: %s" % prod)

    # ---- 2. isolate: is it the tag, or the off value? ----
    print("\n--- 2. isolation: which field decides the message? ---")
    cases = [
        ("tag=0 off=0 (no section; prod shape)", build(0, 0, with_section=False)),
        ("tag=0 off=44 (no section, off legal)", build(44, 0, with_section=False)),
        ("tag=0 off=100 (no section, off high)", build(100, 0, with_section=False)),
        ("tag=1 off=0 (wrong tag, off illegal)", build(0, 1)),
        ("tag=7 off=44 (unknown tag)", build(44, 7)),
        ("tag=1 off=44 (legal section)", build(44, 1)),
    ]
    rows = []
    for name, blob in cases:
        rc, txt = D1.run_engine(exe, blob)
        msgs = [l.strip() for l in txt.splitlines()
                if l.strip().startswith("NKV:")]
        print("   %-38s exit=%-3d %s" % (name, rc, msgs[:1]))
        rows.append(dict(case=name, exit=rc, msg=msgs[0] if msgs else None))
    # the two I care about: the no-section shapes must say "no section"
    nosec_ok = all(rows[i]["msg"] and "has no neuron_kv section" in rows[i]["msg"]
                   for i in (0, 1, 2))
    print("\n   all three tag=0 shapes report 'has no neuron_kv section': %s"
          % nosec_ok)
    print("   (measured above; the prod shape is case 0)")
    meta.record("D6 every tag=0 shape reports 'has no neuron_kv section'",
                nosec_ok,
                "if a tag=0 shape reports a different message, the guard "
                "ordering still pre-empts the correct one",
                not nosec_ok,
                "tag=0 x {off=0, off=44, off=100}: msgs=%s"
                % [rows[i]["msg"] for i in (0, 1, 2)])

    # ---- 3. what the CORRECT behaviour would be (can-fail reference) ----
    print("\n--- 3. can-fail reference: an off<44 WITH a real section ---")
    # this is the case the D4 fix was FOR: a genuine overlap with a section
    # actually present, which must report the overlap message.
    real_overlap = None
    for off in range(7, 44):
        blob = build(off, 1)
        rc, txt = D1.run_engine(exe, blob)
        msgs = [l.strip() for l in txt.splitlines()
                if l.strip().startswith("NKV:")]
        if msgs and "overlaps the .vmb header" in msgs[0]:
            real_overlap = (off, rc, msgs[0])
            break
    print("   first off<44 that reaches the overlap message: %s" % (real_overlap,))
    print("   => the overlap message is CORRECT for a real overlapping section,")
    print("      and WRONG for an absent one. Both conditions share one guard")
    print("      position, which is why tag=0 must be checked first.")
    meta.record("D6 the overlap message IS reachable and correct for a genuine "
                "overlap (D4 fix intact)",
                real_overlap is not None,
                "if no off<44 reached the overlap message, the D4 fix would "
                "have regressed",
                real_overlap is not None,
                "first reaching off=%s" % (real_overlap[0] if real_overlap
                                           else None))

    v = V.ensure_outdir("d6")
    import json
    with open(os.path.join(v, "d6_nosection.json"), "w", encoding="utf-8") as f:
        json.dump(dict(prod=prod_row, rows=rows, real_overlap=real_overlap),
                  f, indent=2, ensure_ascii=False)
    print("\nwrote %s" % os.path.join(v, "d6_nosection.json"))
    return 0 if meta.report() else 1


if __name__ == "__main__":
    sys.exit(main())
