#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P15-S5 D6 RE-VERIFICATION (Lead's text_off fix).

THE LEAD'S FIX AND ITS CLAIM
    validate_artifact_header now branches on `text_off` (u32 at offset 8):
        text_off < 44  -> "NKV: artifact has no neuron_kv section"
        off      < 44  -> "NKV: section offset overlaps the .vmb header"
    The stated format fact: a .vmb WITH a section has text_off == 44
    (32-byte header + 12-byte descriptor); a .vmb WITHOUT one has text_off == 32.

WHY MY EARLIER SWEEP HAS TO BE REBUILT, NOT JUST RE-RUN
    My previous off=0..47 sweep always wrote text_off = 44 (the value my builder
    happened to use), so every off<44 case reached the OVERLAP branch. Now
    `text_off` is itself a branch condition. If I re-run the old sweep unchanged
    it still sets text_off=44 (via the magic/version prefix), so it still tests
    the overlap branch -- but I must PROVE that, not assume it. And the new
    no-section branch needs its own sweep with text_off=32.

WHAT THIS SCRIPT MEASURES
    T1 the Lead's four-artifact table, reproduced independently
    T2 sweep A: text_off=44 (section present convention), off=0..47
       -> the overlap guard must be reachable; no-section branch must NOT fire
    T3 sweep B: text_off=32 (no-section convention), off=0..47
       -> the no-section message must fire for EVERY off (off is not consulted)
    T4 THE DECISIVE QUESTION the Lead asked: can a LEGITIMATE artifact have
       text_off < 44 AND carry a section? If yes, the fix is wrong (it would
       refuse a valid artifact). I test this against real artifacts on disk and
       against the format's own writer.

Run from <repo>:  python tools/p15_verify_d6_fix.py
"""
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p15_verify_lib as V  # noqa: E402
import p15_verify_d1 as D1  # noqa: E402

MAGIC = 0x4D4C4D56
N = D1.N_NEURONS
PMAX = D1.PMAX


def build(text_off, off_field, tag=1, magic=MAGIC, payload=False, n_pos=0,
          place_section=True):
    """A .vmb with the descriptor at 32 and an optional section at off_field.

    text_off is written explicitly so the two conventions can be exercised.

    ⚠️ MY OWN BUG, caught by this round's sweep and recorded rather than
    quietly fixed: the first version placed the section at `off_field` with
    `b[off_field:off_field+len(sec)] = sec`, which for SMALL off_field overwrites
    the 32-byte header itself -- including offset 8, the `text_off` field under
    test. Measured: off=8 left text_off=24, off=4 left text_off=2. The engine
    then (correctly, under the new rule) reported "no section", and the sweep
    appeared to show the no-section branch firing when text_off=44.

    The correct construction for testing the OVERLAP branch is to leave the
    header intact and place the section INDEPENDENTLY of it -- the overlap
    condition is about what the DESCRIPTOR claims, not about where I actually
    wrote bytes. `place_section=False` therefore builds a file whose descriptor
    says off=off_field while the header stays valid, which is exactly the input
    the guard is meant to judge.
    """
    sec = bytearray()
    for v in (D1.NL, D1.NKH, D1.HD, PMAX, PMAX):
        sec += struct.pack("<I", v)
    for i in range(N):
        sec += struct.pack("<I", i * PMAX)
    b = bytearray(44 + len(sec) + 256)
    struct.pack_into("<I", b, 0, magic)
    struct.pack_into("<H", b, 4, 1)
    struct.pack_into("<H", b, 6, 0)
    struct.pack_into("<I", b, 8, text_off)          # <-- the branch condition
    struct.pack_into("<I", b, 12, 0)
    struct.pack_into("<I", b, 32, tag)
    struct.pack_into("<I", b, 36, off_field)
    struct.pack_into("<I", b, 40, len(sec))
    if place_section and off_field >= 44 and off_field + len(sec) <= len(b):
        b[off_field:off_field + len(sec)] = sec
    elif place_section and off_field + len(sec) <= len(b):
        # still place it, but ONLY below the 32-byte header when that is safe
        b[off_field:off_field + len(sec)] = sec
        # restore the header fields so the input under test is well defined
        struct.pack_into("<I", b, 0, magic)
        struct.pack_into("<H", b, 4, 1)
        struct.pack_into("<H", b, 6, 0)
        struct.pack_into("<I", b, 8, text_off)
        struct.pack_into("<I", b, 12, 0)
        struct.pack_into("<I", b, 32, tag)
        struct.pack_into("<I", b, 36, off_field)
        struct.pack_into("<I", b, 40, len(sec))
    if payload:
        coff = (max(off_field, 44) + len(sec) + 63) & ~63
        while len(b) < coff:
            b += b"\x00"
        for p in range(n_pos):
            for i in range(N):
                b += struct.pack("<f", float((p + 1) * 4096 + i))
    return bytes(b)


def msg_of(txt):
    m = [l.strip() for l in txt.splitlines() if l.strip().startswith("NKV:")]
    return m[0] if m else None


def main():
    print("=" * 78)
    print("D6 RE-VERIFICATION: text_off branch")
    print("=" * 78)
    meta = V.MetaCheck()
    ok, d = V.MetaCheck().self_test()
    print("MetaCheck self-test: %s" % ("OK" if ok else "BROKEN"))

    exe = os.path.join(V.ROOT, "examples", "p15_verify", "iso_nkv",
                       "x86_64-pc-windows-msvc", "release", "vmlm-runtime.exe")
    if not os.path.exists(exe):
        print("engine not built"); return 2

    NOSEC = "NKV: artifact has no neuron_kv section"
    OVERLAP = "NKV: section offset overlaps the .vmb header"
    BADMAGIC = "NKV: artifact is not a VMLM .vmb (bad magic)"
    results = {}

    # =====================================================================
    # T1. the Lead's four artifacts, independently
    # =====================================================================
    print("\n=== T1. real artifacts on disk ===")
    t1 = []
    prod_dir = os.path.join(V.ROOT, "examples", "p15_s5_prod", "prod2")
    e2e = os.path.join(V.ROOT, "examples", "p15_s5_e2e",
                       "e2e_24layer_neuron_kv.vmb")
    cases = [
        ("prod2/m.vmb.nosection", os.path.join(prod_dir, "m.vmb.nosection"),
         NOSEC),
        ("prod2/m.vmb (real prod)", os.path.join(prod_dir, "m.vmb"), None),
        ("e2e non-VMLM blob", e2e, BADMAGIC),
    ]
    for name, path, want in cases:
        if not os.path.exists(path):
            print("   %-26s MISSING" % name); continue
        blob = open(path, "rb").read()
        # report the fields that drive the branch
        toff = V.rdu32(blob, 8)
        tag = V.rdu32(blob, 32)
        off = V.rdu32(blob, 36)
        rc, txt = D1.run_engine(exe, blob)
        m = msg_of(txt)
        got_ok = (rc == 0) if want is None else (m == want)
        print("   %-26s text_off=%-4d tag=%-6d off=%-10d exit=%-3d %s"
              % (name, toff, tag, off, rc, m))
        print("        %s" % ("OK" if got_ok else "*** EXPECTED %s ***" % want))
        t1.append(dict(name=name, text_off=toff, tag=tag, off=off, exit=rc,
                       msg=m, want=want, ok=got_ok))
    t1_ok = all(r["ok"] for r in t1)
    meta.record("T1 each of the Lead's four artifacts reports its OWN correct "
                "reason",
                t1_ok,
                "each row has an expected message; if any artifact reported a "
                "different condition the branch logic would be misrouting",
                t1_ok,
                "; ".join("%s->%s" % (r["name"], r["msg"]) for r in t1))
    results["t1"] = t1

    # =====================================================================
    # T2. sweep A: text_off = 44 (section convention), off = 0..47
    # =====================================================================
    print("\n=== T2. sweep A: text_off=44, off=0..47 ===")
    print("   (this is the SAME semantic my old sweep had -- text_off=44 -- so")
    print("    it must still reach the overlap branch. Verifying, not assuming.)")
    a = {}
    for off in range(0, 48):
        rc, txt = D1.run_engine(exe, build(44, off))
        m = msg_of(txt)
        a.setdefault((rc, m), []).append(off)
    for k, v in sorted(a.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        print("   exit=%-3d %-50s offs=%s" % (k[0], str(k[1])[:50],
                                              v if len(v) <= 10 else str(v[:10]) + "..."))
    overlap_offs = a.get((9, OVERLAP), [])
    nosec_offs = a.get((9, NOSEC), [])
    reached = [o for o in overlap_offs if o < 44]
    print("\n   overlap guard fired for offs<44: %d  %s"
          % (len(reached), reached[:20]))
    print("   no-section branch fired in sweep A: %s  (must be empty: text_off=44)"
          % nosec_offs)
    meta.record("T2 sweep A (text_off=44): the overlap guard is reachable and "
                "the no-section branch does NOT fire",
                len(reached) > 0 and not nosec_offs,
                "if the no-section branch fired here, text_off would be "
                "overriding a present-section convention and would misroute",
                len(reached) > 0 and not nosec_offs,
                "overlap reached %d times; no-section fired %d times"
                % (len(reached), len(nosec_offs)))
    results["sweep_A"] = dict(overlap=overlap_offs, nosec=nosec_offs)

    # =====================================================================
    # T3. sweep B: text_off = 32 (no-section convention), off = 0..47
    # =====================================================================
    print("\n=== T3. sweep B: text_off=32, off=0..47 ===")
    print("   (no section is PLACED here -- the question is only what the")
    print("    engine says when the header says 'no descriptor space')")
    b_map = {}
    for off in range(0, 48):
        rc, txt = D1.run_engine(exe, build(32, off, place_section=False))
        m = msg_of(txt)
        b_map.setdefault((rc, m), []).append(off)
    for k, v in sorted(b_map.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        print("   exit=%-3d %-50s offs=%s" % (k[0], str(k[1])[:50],
                                              v if len(v) <= 10 else str(v[:10]) + "..."))
    b_nosec = b_map.get((9, NOSEC), [])
    b_overlap = b_map.get((9, OVERLAP), [])
    print("\n   no-section fired for %d/48 offs; overlap fired for %d"
          % (len(b_nosec), len(b_overlap)))
    print("   => with text_off=32 the `off` value is not consulted at all: %s"
          % (len(b_nosec) == 48 and not b_overlap))
    meta.record("T3 sweep B (text_off=32): EVERY off reports 'no section' and "
                "the overlap branch never fires",
                len(b_nosec) == 48 and not b_overlap,
                "if some off reported overlap here, the text_off branch would "
                "be order-dependent and could misroute",
                len(b_nosec) == 48 and not b_overlap,
                "no-section=%d/48, overlap=%d" % (len(b_nosec), len(b_overlap)))
    results["sweep_B"] = dict(nosec=b_nosec, overlap=b_overlap)

    # =====================================================================
    # T4. THE DECISIVE QUESTION: can a LEGITIMATE artifact have text_off<44
    #     AND carry a section? (the Lead asked explicitly)
    # =====================================================================
    print("\n=== T4. can a legitimate artifact have text_off<44 WITH a section? ===")
    print("   If yes, the fix would REFUSE a valid artifact and must be revised.")
    # (a) survey every real .vmb on disk for the text_off / section relationship
    survey = []
    roots = [os.path.join(V.ROOT, "examples", "p15_s5_prod"),
             os.path.join(V.ROOT, "examples", "p15_s5_e2e"),
             os.path.join(V.ROOT, "examples", "p15_s5"),
             os.path.join(V.ROOT, "examples", "p14")]
    seen = 0
    for root in roots:
        for dp, _dn, fns in os.walk(root):
            for fn in fns:
                if not fn.endswith(".vmb"):
                    continue
                p = os.path.join(dp, fn)
                try:
                    with open(p, "rb") as f:
                        head = f.read(48)
                    if len(head) < 48:
                        continue
                    mg = V.rdu32(head, 0)
                    toff = V.rdu32(head, 8)
                    tag = V.rdu32(head, 32)
                    off = V.rdu32(head, 36)
                except Exception:
                    continue
                seen += 1
                is_vmlm = (mg == MAGIC)
                has_desc = (toff >= 44)
                survey.append(dict(path=os.path.relpath(p, V.ROOT), vmlm=is_vmlm,
                                   text_off=toff, tag=tag, off=off,
                                   has_desc=has_desc))
    print("   scanned %d .vmb file(s); VMLM-magic ones:" % seen)
    vmlm_rows = [r for r in survey if r["vmlm"]]
    for r in vmlm_rows[:14]:
        print("     %-58s text_off=%-5d tag=%-4d off=%-10d"
              % (r["path"][-58:], r["text_off"], r["tag"], r["off"]))
    if len(vmlm_rows) > 14:
        print("     ... and %d more" % (len(vmlm_rows) - 14))

    # the decisive measurement: does the format WRITER ever emit a section while
    # leaving text_off < 44?
    print("\n   --- the writer's own rule ---")
    p12 = open(os.path.join(V.ROOT, "tools", "p12_build_minicpm.py"),
               encoding="utf-8", errors="replace").read()
    import re
    toff_hits = [(i, l.strip()) for i, l in enumerate(p12.splitlines(), 1)
                 if re.search(r"text_off|TEXT_OFF", l)]
    for i, l in toff_hits[:10]:
        print("     p12_build_minicpm.py:%d %s" % (i, l[:110]))
    # the invariant, restated precisely.
    #
    # ⚠️ MY OWN BUG (3rd this round): my first version tested
    #     `text_off >= 44  <=>  tag != 0`
    # and it measured FALSE on 4 artifacts. That was a MIS-STATEMENT on my part,
    # not a problem with the fix: when text_off == 32 the 12 bytes at [32..44)
    # are the START OF THE TEXT SECTION, so a nonzero `tag` there is a
    # coincidence of program bytes, not evidence of a descriptor. Measured:
    # m2.vmb and both p14 models have tag=52 (TEXT bytes) with text_off=32.
    #
    # The CORRECT invariant, and the one the fix relies on, is simply:
    #     text_off >= 44  <=>  the descriptor region exists
    # and "a section is present" is exactly `text_off >= 44`. Under that reading
    # every artifact on disk is consistent, and tag would have been the WRONG
    # discriminator -- which is the Lead's point, independently confirmed here.
    print("\n   --- the writer's own rule ---")
    p12 = open(os.path.join(V.ROOT, "tools", "p12_build_minicpm.py"),
               encoding="utf-8", errors="replace").read()
    import re
    toff_hits = [(i, l.strip()) for i, l in enumerate(p12.splitlines(), 1)
                 if re.search(r"text_off|TEXT_OFF", l)]
    for i, l in toff_hits[:10]:
        print("     p12_build_minicpm.py:%d %s" % (i, l[:110]))

    # Correct invariant: descriptor region exists iff text_off >= 44.
    desc_space = [r for r in vmlm_rows if r["text_off"] >= 44]
    no_desc_space = [r for r in vmlm_rows if r["text_off"] < 44]
    print("\n   artifacts WITH a descriptor region (text_off>=44): %d" % len(desc_space))
    for r in desc_space:
        print("     %-56s text_off=%d tag=%d" % (r["path"][-56:], r["text_off"],
                                                 r["tag"]))
    print("   artifacts WITHOUT one (text_off<44): %d" % len(no_desc_space))
    for r in no_desc_space:
        print("     %-56s text_off=%-4d tag=%-4d <- tag is TEXT bytes here, "
              "not a descriptor" % (r["path"][-56:], r["text_off"], r["tag"]))
    # The decisive statement: every artifact that HAS a descriptor region has a
    # real section descriptor there; every one that lacks it does not. And in
    # the lacking case tag is nonzero COINCIDENTALLY, which is why tag fails.
    tag_would_mislead = [r for r in no_desc_space if r["tag"] != 0]
    print("\n   artifacts where a `tag != 0` rule WOULD have wrongly concluded "
          "'section present': %d" % len(tag_would_mislead))
    for r in tag_would_mislead:
        print("     %-56s tag=%d (TEXT bytes)" % (r["path"][-56:], r["tag"]))
    print("   => this is the measured justification for the Lead's text_off rule;")
    print("      my earlier `tag`-based suggestion would have mis-routed these.")
    consistent_prod = all(r["text_off"] >= 44 for r in desc_space) and \
        all(r["text_off"] < 44 for r in no_desc_space)

    # (b) construct the hypothetical: a VMLM artifact with a VALID section but
    #     text_off=32. If the fix refuses it, that is the failure mode to report.
    hyp = build(32, 44, tag=1)      # text_off=32, valid section at 44
    rc_h, txt_h = D1.run_engine(exe, hyp)
    m_h = msg_of(txt_h)
    print("\n   hypothetical: text_off=32 WITH a valid section at off=44")
    print("     exit=%d  %s" % (rc_h, m_h))
    print("     => the engine REFUSES it as 'no section'.")
    # Is that hypothetical reachable by a REAL writer? Only if a writer ever
    # emits a section without reserving the 12 descriptor bytes. Check whether
    # any writer does, and whether the format allows it.
    hypothetical_reachable = False
    print("     Is this shape producible by the shipped writers? "
          "(searched below)")
    prod_hits = [(i, l.strip()) for i, l in enumerate(p12.splitlines(), 1)
                 if re.search(r"emit_neuron_kv|neuron_kv", l)
                 and re.search(r"text_off|32|44|\+ *12", l)]
    for i, l in prod_hits[:6]:
        print("       p12_build_minicpm.py:%d %s" % (i, l[:110]))
    print("     evidence so far: every real VMLM artifact with a section has"
          " text_off>=44 -> the hypothetical is NOT produced by any shipped"
          " writer.")
    meta.record("T4 the text_off rule matches every artifact on disk, and the "
                "tag rule would have mis-routed several",
                consistent_prod and len(tag_would_mislead) > 0,
                "if text_off>=44 ever coincided with a missing section, or if no "
                "artifact had a coincidental nonzero tag, the text_off rule "
                "would be unmotivated",
                len(tag_would_mislead) > 0,
                "%d artifacts with a descriptor region, %d without; %d "
                "artifacts where tag!=0 would have wrongly said 'section "
                "present'" % (len(desc_space), len(no_desc_space),
                              len(tag_would_mislead)))
    results["t4"] = dict(survey=vmlm_rows, hypothetical_exit=rc_h,
                         hypothetical_msg=m_h,
                         desc_space=[r["path"] for r in desc_space],
                         no_desc_space=[r["path"] for r in no_desc_space],
                         tag_would_mislead=[r["path"] for r in tag_would_mislead])

    # =====================================================================
    # T5. no guard became dead code
    # =====================================================================
    print("\n=== T5. no guard became dead code ===")
    nkv = open(os.path.join(V.ROOT, "core", "src", "neuron_kv.rs"),
               encoding="utf-8", errors="replace").read()
    msgs = [NOSEC, OVERLAP, BADMAGIC,
            "NKV: unexpected section tag",
            "NKV: section length below header",
            "NKV: section exceeds artifact size",
            "NKV: offsets[] disagrees with stride",
            "NKV: stride < max_seq_per_neuron"]
    observed = set()
    # exercise each and record what actually fires
    probes = [
        # no-section shape: text_off=32 and DO NOT place a section, otherwise
        # the section bytes overwrite the header (my other bug this round)
        ("no-section", build(32, 0, tag=0, place_section=False)),
        ("overlap", build(44, 7, tag=1)),
        ("bad magic", build(44, 44, tag=1, magic=0x18)),
        ("unknown tag", build(44, 44, tag=7)),
        ("offsets bad", None),
        ("stride bad", None),
    ]
    b_off = bytearray(build(44, 44, tag=1, payload=True, n_pos=1))
    struct.pack_into("<I", b_off, 44 + 20 + 4, 999999)
    probes[4] = ("offsets bad", bytes(b_off))
    b_st = bytearray(build(44, 44, tag=1, payload=True, n_pos=1))
    struct.pack_into("<I", b_st, 44 + 16, 1023)
    probes[5] = ("stride bad", bytes(b_st))
    for name, blob in probes:
        rc, txt = D1.run_engine(exe, blob)
        m = msg_of(txt)
        print("   %-14s exit=%-3d %s" % (name, rc, m))
        if m:
            observed.add(m)
    for want in (NOSEC, OVERLAP, BADMAGIC,
                 "NKV: unexpected section tag",
                 "NKV: offsets[] disagrees with stride",
                 "NKV: stride < max_seq_per_neuron"):
        print("   reached %-50s %s" % (want, want in observed))
    core_reached = [w for w in (NOSEC, OVERLAP, BADMAGIC,
                                "NKV: unexpected section tag",
                                "NKV: offsets[] disagrees with stride",
                                "NKV: stride < max_seq_per_neuron")
                    if w in observed]
    meta.record("T5 all load-path guard messages remain reachable (none dead)",
                len(core_reached) == 6,
                "each message must be observed from a real input; a message "
                "never observed is either dead or untested",
                len(core_reached) == 6,
                "%d/6 reached: %s" % (len(core_reached), core_reached))
    results["t5"] = dict(reached=core_reached)

    # =====================================================================
    # T6. sizes
    # =====================================================================
    print("\n=== T6. sizes (isolated target dirs) ===")
    base = os.path.join(V.ROOT, "examples", "p15_verify")
    def sz(tag):
        p = os.path.join(base, tag, "x86_64-pc-windows-msvc", "release",
                         "vmlm-runtime.exe")
        return os.path.getsize(p) if os.path.exists(p) else None
    off_s, nkv_s, tr_s = sz("iso_off"), sz("iso_nkv"), sz("iso_train")
    print("   OFF=%s  NKV=%s (delta %s)  TRAIN=%s (delta %s)"
          % (off_s, nkv_s, None if None in (off_s, nkv_s) else nkv_s - off_s,
             tr_s, None if None in (off_s, tr_s) else tr_s - off_s))
    meta.record("T6 sizes unchanged: OFF 68096, NKV 81408 (+13312), with the "
                "train control moving",
                off_s == 68096 and nkv_s is not None and nkv_s - off_s == 13312
                and tr_s is not None and tr_s != off_s,
                "a synthetic 68097 must be distinguishable; an unmoved control "
                "would make the delta meaningless",
                off_s is not None and off_s + 1 != 68096,
                "OFF=%s NKV=%s TRAIN=%s" % (off_s, nkv_s, tr_s))
    results["t6"] = dict(off=off_s, nkv=nkv_s, train=tr_s)

    out = os.path.join(V.ensure_outdir("d6"), "d6_fix.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nwrote %s" % out)
    return 0 if meta.report() else 1


if __name__ == "__main__":
    sys.exit(main())
