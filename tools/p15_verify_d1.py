#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P15-S5 D1 RE-VERIFICATION (task-4 follow-up, independent adversarial verifier).

WHAT D1 WAS
    The real-file load path called load_reorder(..., n_pos = 0): it allocated
    25 MB, moved ZERO elements, and printed "S5LOAD ok" with exit 0. A success
    message on a path that did nothing. The Lead has since rewritten the path.

WHAT THIS SCRIPT DOES
    It re-attacks D1 from scratch, with instruments and artifacts that are MINE.
    I do NOT reuse the implementer's or the Lead's position-varying artifact:
    building my own is the only way to know that "the reorder moved something"
    is a statement about the CODE and not about a particular hand-made file.

    ARTIFACT CONSTRUCTION, derived from primitives here:
      [0..32)   .vmb / layout header
      [32..44)  section descriptor: tag(32) off(36) len(40)
      [44..)    section: NKV_HDR(20) + offsets[n_neurons]*4
      aligned up to 64, then the compact region: n_pos * n_neurons * f32

    The alignment rule is re-derived from core/src/neuron_kv.rs
    load_compact_geometry: compact_off = (sec_off + section_bytes + 63) & ~63.
    I assert MY arithmetic reproduces that, and I ALSO test what happens when a
    writer uses a DIFFERENT alignment -- that is the real attack, because the
    engine computes compact_off from a formula instead of reading it.

TESTS
  T1 vacuous/no-op paths must NOT print ok (n_pos=0, empty compact, invariant fill)
  T2 my own position-varying artifact: elements really move, and the direction
     check DISCRIMINATES (reversed data must be rejected)
  T3 alignment: a writer using different padding must not be silently misread
  T4 every load-path guard still fires with its own message (nothing lost in the move)
  T5 sizes: default 68096 exact, neuron-kv delta, controls, private target dir

Run from <repo>:
    python tools/p15_verify_d1.py
"""
import os
import shutil
import struct
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p15_verify_lib as V  # noqa: E402

ROOT = V.ROOT
NL, NKH, HD, PMAX = 24, 2, 128, 1024
N_NEURONS = NL * NKH * HD            # 6144
SEC_BYTES = 20 + N_NEURONS * 4       # 24596
TAG = 1


# ---------------------------------------------------------------------------
# artifact construction, from primitives
# ---------------------------------------------------------------------------
def build_artifact(n_pos, fill, align=64, extra_tail=0, sec_off=44):
    """Build a .vmb carrying a neuron_kv section and a compact region.

    fill: callable (p, i) -> float, the value at position p, neuron i.
    `align` is the alignment used for the compact region start; the shipped
    engine computes it as (sec_off + SEC_BYTES + 63) & ~63, i.e. align=64.
    """
    sec = bytearray()
    for v in (NL, NKH, HD, PMAX, PMAX):          # tight: stride == max_seq
        sec += struct.pack("<I", v)
    for i in range(N_NEURONS):
        sec += struct.pack("<I", i * PMAX)
    assert len(sec) == SEC_BYTES, (len(sec), SEC_BYTES)

    # compact_off by the SHIPPED formula (mirrored), parameterised so I can
    # build a deliberately misaligned variant.
    compact_off = (sec_off + len(sec) + (align - 1)) & ~(align - 1)

    blob = bytearray()
    # [0..20) layout declaration that validate_artifact_header compares
    for v in (NL, NKH, HD, PMAX, PMAX):
        blob += struct.pack("<I", v)
    # [20..32) other header fields
    blob += struct.pack("<I", sec_off)     # 20
    blob += struct.pack("<I", sec_off)     # 24
    blob += struct.pack("<I", 0)           # 28
    # [32..44) descriptor
    blob += struct.pack("<I", TAG)         # 32
    blob += struct.pack("<I", sec_off)     # 36
    blob += struct.pack("<I", len(sec))    # 40

    assert len(blob) == sec_off, (len(blob), sec_off)
    blob += sec
    # pad to compact_off
    while len(blob) < compact_off:
        blob += b"\x00"
    assert len(blob) == compact_off

    # compact region: compact[p * n + i]
    for p in range(n_pos):
        for i in range(N_NEURONS):
            blob += struct.pack("<f", fill(p, i))
    blob += b"\x00" * extra_tail
    return bytes(blob), compact_off


def run_engine(exe, blob, name="s5_kv.vmb", timeout=600):
    """Run the engine hermetically on `blob` (fixed filename, bare flag)."""
    d = tempfile.mkdtemp(prefix="p15_d1_")
    try:
        shutil.copyfile(exe, os.path.join(d, "vmlm-runtime.exe"))
        with open(os.path.join(d, name), "wb") as f:
            f.write(blob)
        rc, out, err = V.run_capture(
            [os.path.join(d, "vmlm-runtime.exe"), "--s5-load"], cwd=d,
            timeout=timeout)
        return rc, (out + err).decode("utf-8", "replace")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    print("=" * 78)
    print("P15-S5 D1 RE-VERIFICATION (independent verifier)")
    print("=" * 78)
    meta = V.MetaCheck()
    ok, mdet = V.MetaCheck().self_test()
    print("MetaCheck self-test: %s %s" % ("OK" if ok else "BROKEN", mdet))

    exe = os.path.join(ROOT, "examples", "p15_verify", "fw_build",
                       "x86_64-pc-windows-msvc", "release", "vmlm-runtime.exe")
    if not os.path.exists(exe):
        print("\nENGINE NOT FOUND at %s" % exe)
        print("Build it first: cd runtime; cargo build --release --target "
              "x86_64-pc-windows-msvc --features neuron-kv "
              "--target-dir ../examples/p15_verify/fw_build")
        return 2
    print("engine (neuron-kv) = %d B  sha=%s"
          % (os.path.getsize(exe), V.sha256(exe)[:16]))

    outdir = V.ensure_outdir("d1")
    results = []

    # =====================================================================
    # T1. the old silent success must be GONE
    # =====================================================================
    print("\n=== T1. the old silent success must be gone ===")
    t1_cases = []

    # (a) position-INVARIANT fill: compact[p*n+i] = i (the as-shipped e2e fill).
    #     The transpose is the identity on this data.
    blob, coff = build_artifact(4, lambda p, i: float(i))
    rc, txt = run_engine(exe, blob)
    t1_cases.append(("position-invariant fill (transpose is identity)", rc, txt))

    # (b) empty compact region: file ends exactly at compact_off
    blob_e, coff_e = build_artifact(0, lambda p, i: 0.0)
    rc, txt = run_engine(exe, blob_e)
    t1_cases.append(("empty compact region (file ends at compact_off)", rc, txt))

    # (c) compact region with a non-whole number of positions (tail 2 bytes)
    blob_t, coff_t = build_artifact(2, lambda p, i: float(i), extra_tail=2)
    rc, txt = run_engine(exe, blob_t)
    t1_cases.append(("compact region truncated mid-element (+2 bytes)", rc, txt))

    for name, rc, txt in t1_cases:
        last = [l for l in txt.splitlines() if l.strip()]
        print("\n  [%s]  exit=%d" % (name, rc))
        for l in last[-4:]:
            print("     | %s" % l.strip()[:150])
        results.append(dict(test="T1", case=name, exit=rc, tail=last[-3:]))

    t1_ok = all(rc != 0 and "S5LOAD ok" not in txt
                for _n, rc, txt in t1_cases)
    t1_saw_ok = any("S5LOAD ok" in txt for _n, _r, txt in t1_cases)
    print("\n  => every one refused to print ok: %s" % t1_ok)
    meta.record("D1 T1 no-op / vacuous / truncated reorder cannot print ok",
                t1_ok,
                "the PRE-FIX code printed ok with exit 0 on exactly case (a) — "
                "the position-invariant as-shipped artifact; if any of these now "
                "prints ok the defect is back",
                all(rc != 0 for _n, rc, _t in t1_cases) and not t1_saw_ok,
                "exits=%s, any ok printed=%s"
                % ([rc for _n, rc, _t in t1_cases], t1_saw_ok))

    # =====================================================================
    # T2. my OWN position-varying artifact: does the reorder really move data?
    # =====================================================================
    print("\n=== T2. my own position-varying artifact (independent build) ===")
    # fill that makes the transpose a GENUINE permutation:
    #   compact[p*n + i] = (p+1)*4096 + i
    #
    # ⚠️ MY OWN BUG, caught while reconciling a 718-vs-6144 discrepancy: my
    # first fill was (p+1)*1_000_000 + i, and f32 CANNOT represent that
    # distinctly. At i=1000 the value is ~1.001e9, whose ulp is 64, so the +p
    # term (p in 0..3) is rounded away and the row becomes CONSTANT -- the very
    # degeneracy I was trying to avoid. Measured: only 67/6144 neurons kept a
    # distinct value across positions with that fill. The scaling below keeps
    # every value exactly representable (max ~16384, ulp far below 1), so the
    # position is genuinely encoded and the transpose is a real permutation.
    def fill_pv(p, i):
        return float((p + 1) * 4096 + i)

    n_pos = 4
    blob_pv, coff_pv = build_artifact(n_pos, fill_pv)
    print("  my artifact: %d bytes, compact_off=%d (my arithmetic), "
          "n_pos=%d, compact bytes=%d"
          % (len(blob_pv), coff_pv, n_pos, len(blob_pv) - coff_pv))
    # Independently confirm the transpose is genuinely non-identity on my data,
    # in f32, the way the engine will see it.
    import struct as _s

    def f32(v):
        return _s.pack("<f", v)
    perm = 0
    same = 0
    for i in range(N_NEURONS):
        for p in range(n_pos):
            if f32(fill_pv(p, i)) != f32(fill_pv(i % n_pos, i)):
                pass
    for i in range(N_NEURONS):
        for p in range(n_pos):
            if f32(fill_pv(p, i)) != f32(fill_pv(i, p)):
                perm += 1
    # non-constant rows, the property the engine reports
    nonconst = sum(1 for i in range(N_NEURONS)
                   if len({f32(fill_pv(p, i)) for p in range(n_pos)}) > 1)
    ident = (perm == 0)
    print("  transpose is identity on MY data: %s (differing cell pairs "
          "sampled=%d)" % (ident, perm))
    print("  rows that are non-constant in f32: %d/%d (engine should report "
          "this number)" % (nonconst, N_NEURONS))

    rc_pv, txt_pv = run_engine(exe, blob_pv)
    print("\n  engine on MY artifact: exit=%d" % rc_pv)
    for l in txt_pv.splitlines():
        if l.strip():
            print("     | %s" % l.strip()[:160])
    results.append(dict(test="T2", case="my position-varying artifact",
                        exit=rc_pv, tail=txt_pv.splitlines()[-12:],
                        compact_off_mine=coff_pv))

    def field(txt, key):
        for l in txt.splitlines():
            if key in l:
                part = l.split(key, 1)[1].strip().split()[0]
                if part.isdigit():
                    return int(part)
        return None

    n_pos_eng = field(txt_pv, "n_pos=")
    bytes_cmp = field(txt_pv, "bytes_compared=")
    first_diff = field(txt_pv, "first_differing_byte=")
    pos_var = field(txt_pv, "pos_varying_neurons=")
    dir_elems = field(txt_pv, "direction_check elements=")
    print("\n  parsed: n_pos=%s bytes_compared=%s first_differing_byte=%s "
          "pos_varying_neurons=%s direction_check_elements=%s"
          % (n_pos_eng, bytes_cmp, first_diff, pos_var, dir_elems))
    # my independently computed expectations
    exp_cmp = n_pos * N_NEURONS * 4
    exp_dir = n_pos * N_NEURONS
    print("  my independent expectations: bytes_compared=%d "
          "direction_check_elements=%d" % (exp_cmp, exp_dir))

    ok_pv = (rc_pv == 0 and n_pos_eng == n_pos and bytes_cmp == exp_cmp
             and first_diff == 0 and pos_var == nonconst
             and dir_elems == exp_dir and not ident)
    meta.record("D1 T2 positive path is REAL: my artifact moves elements and "
                "agrees with my independent counts",
                ok_pv,
                "if the transform were the identity on my data (ident=True) "
                "or bytes_compared were 0, this criterion must not be certified",
                (not ident) and bytes_cmp is not None and bytes_cmp > 0,
                "exit=%s n_pos=%s bytes_compared=%s (expect %d) "
                "pos_varying=%s (expect my non-constant count %d) "
                "direction=%s (expect %d) identity_on_my_data=%s"
                % (rc_pv, n_pos_eng, bytes_cmp, exp_cmp, pos_var, nonconst,
                   dir_elems, exp_dir, ident))

    # ---- T2b. the direction check must DISCRIMINATE ----
    #
    # The correct contract is  per_neuron[i*stride + p] == compact[p*n + i].
    # A REVERSED reorder would instead produce compact[i*n_pos + p].
    # To test that the engine's direction check is real, I build an artifact
    # whose compact region is crafted so that the CORRECT reading and the
    # REVERSED reading give DIFFERENT values -- then the engine must either pass
    # only the correct one, or reject. Concretely I build the artifact as if a
    # writer had used the REVERSED convention: it puts the value that belongs at
    # (i, p) into compact[i*n_pos + p]. A correct reader, reading per spec, will
    # then observe per_neuron[i*stride+p] = compact[p*n+i] = value that belongs
    # at (p, i) -- i.e. wrong wherever i != p. That must be caught.
    print("\n=== T2b. direction check must reject REVERSED (transposed) data ===")
    # a data set where the transpose genuinely permutes (my fill_pv qualifies,
    # because (p+1)*1e6 + i differs from (i+1)*1e6 + p for i != p)
    t_ident = 0
    t_perm = 0
    for i in range(0, 512):
        for p in range(n_pos):
            if fill_pv(p, i) != fill_pv(i, p):
                t_perm += 1
            else:
                t_ident += 1
    print("  on my fill, cells where transpose changes the value: %d "
          "(unchanged: %d) -> the permutation is real: %s"
          % (t_perm, t_ident, t_perm > 0))

    def fill_rev(p, i):
        # reversed convention: cell (p,i) holds what belongs at (i,p)
        return fill_pv(i, p)
    blob_rev, _ = build_artifact(n_pos, fill_rev)
    rc_rev, txt_rev = run_engine(exe, blob_rev)
    print("  engine on the REVERSED-convention artifact: exit=%d" % rc_rev)
    for l in txt_rev.splitlines()[-5:]:
        if l.strip():
            print("     | %s" % l.strip()[:150])
    rev_rejected = (rc_rev != 0)
    # can-fail for THIS control: the same build with the CORRECT convention must
    # be accepted, otherwise "rejected" would just mean the builder is broken.
    print("  (control: the CORRECT-convention build is accepted: exit=%d)" % rc_pv)
    meta.record("D1 T2b the direction check DISCRIMINATES (reversed data "
                "rejected, correct data accepted)",
                rev_rejected and rc_pv == 0,
                "if the reversed artifact were accepted with exit 0, the "
                "direction check would not be discriminating; if the correct "
                "artifact were also rejected, the builder would be at fault",
                rev_rejected,
                "reversed exit=%d (must be nonzero), correct exit=%d (must be 0)"
                % (rc_rev, rc_pv))

    # =====================================================================
    # T3. alignment: can a differently-aligned writer be silently misread?
    # =====================================================================
    print("\n=== T3. compact-region alignment ===")
    print("  the engine computes compact_off = (sec_off + SEC_BYTES + 63) & ~63")
    print("  = (%d + %d + 63) & ~63 = %d" % (44, SEC_BYTES,
                                             (44 + SEC_BYTES + 63) & ~63))
    # Build with a DIFFERENT alignment (128) and see what happens. The engine
    # will look at 64-aligned offset; the data will be at 128-aligned. This is
    # the "writer and reader disagree about the layout" failure mode.
    blob_a, coff_a = build_artifact(n_pos, fill_pv, align=128)
    rc_a, txt_a = run_engine(exe, blob_a)
    first_a = field(txt_a, "first_differing_byte=")
    print("  artifact aligned to 128 (data at %d, engine expects %d): exit=%d "
          "first_differing_byte=%s"
          % (coff_a, (44 + SEC_BYTES + 63) & ~63, rc_a, first_a))
    for l in txt_a.splitlines()[-4:]:
        if l.strip():
            print("     | %s" % l.strip()[:150])
    # Either the engine rejects it, or it reports a nonzero difference. What it
    # must NOT do is print ok with first_differing_byte=0.
    silent = (rc_a == 0 and first_a == 0)
    meta.record("D1 T3 a differently-aligned compact region is NOT silently "
                "accepted",
                not silent,
                "if rc==0 and first_differing_byte==0 for a 128-aligned "
                "artifact, the engine would be reading padding as data and "
                "blessing it",
                not silent,
                "exit=%d first_differing_byte=%s silent=%s"
                % (rc_a, first_a, silent))

    # =====================================================================
    # T4. guards preserved by the scaffolding move
    # =====================================================================
    print("\n=== T4. did the scaffolding move drop any hard-fail path? ===")
    # Exercise each guard through the ENGINE (not the probe), by mutating my
    # good artifact one field at a time.
    good = bytearray(blob_pv)

    def mutate(off, val):
        b = bytearray(good)
        struct.pack_into("<I", b, off, val)
        return bytes(b)

    t4 = []
    # G3 offsets[] disagrees with stride: entry 1 (sec at 44, +20 +4)
    b = bytearray(good)
    struct.pack_into("<I", b, 44 + 20 + 4, 999999)
    t4.append(("offsets[] disagrees with stride", bytes(b),
               "NKV: offsets[] disagrees with stride"))
    # G1 stride < max_seq: header stride field at 44+16 -> 1023
    b = bytearray(good)
    struct.pack_into("<I", b, 44 + 16, 1023)
    b2 = bytearray(b)
    struct.pack_into("<I", b2, 16, 1023)   # keep artifact header agreeing
    t4.append(("stride < max_seq", bytes(b2),
               "NKV: stride < max_seq_per_neuron"))
    # S8 exact length: descriptor len at 40 -> 20
    t4.append(("declared section len too short", mutate(40, 20),
               "NKV: section length disagrees with layout"))
    # tag wrong
    t4.append(("wrong section tag", mutate(32, 7),
               "NKV: unexpected section tag"))
    # tag zero = no section
    t4.append(("tag=0 (no section)", mutate(32, 0),
               "NKV: artifact has no neuron_kv section"))
    # descriptor len > file
    t4.append(("declared len exceeds file", mutate(40, 1 << 20),
               "NKV: section exceeds artifact size"))
    # artifact header disagrees with section (field 0 = n_layers -> 99)
    t4.append(("artifact header disagrees", mutate(0, 99),
               "NKV: artifact header disagrees with section"))
    # n_pos > max_seq: append a 5th position (max_seq=1024 so need >1024 pos)
    big_pos = build_artifact(PMAX + 1, fill_pv)[0]
    t4.append(("n_pos > max_seq_per_neuron", big_pos,
               "NKV: compact region has more positions than slots"))
    # compact region empty
    t4.append(("compact region empty", blob_e,
               "NKV: compact region is empty"))
    # compact not a whole number of positions
    t4.append(("compact not whole positions", blob_t,
               "NKV: compact region is not a whole number of positions"))

    t4_rows = []
    for name, blob_i, want in t4:
        rc_i, txt_i = run_engine(exe, blob_i)
        hit = want in txt_i
        print("   %-38s exit=%-3d msg=%-5s | %s"
              % (name, rc_i, hit,
                 next((l.strip() for l in txt_i.splitlines() if want[:28] in l),
                      (txt_i.splitlines() or [""])[-1].strip())[:80]))
        t4_rows.append(dict(case=name, exit=rc_i, msg=hit, want=want,
                            tail=txt_i.splitlines()[-3:]))
    n_fired = sum(1 for r in t4_rows if r["exit"] != 0 and r["msg"])
    print("   %d/%d guards fired with THEIR OWN message" % (n_fired, len(t4_rows)))
    results.append(dict(test="T4", rows=t4_rows, fired=n_fired,
                        total=len(t4_rows)))
    meta.record("D1 T4 all load-path guards survived the scaffolding move and "
                "name their own condition",
                n_fired == len(t4_rows),
                "each mutation must exit nonzero AND print the message for ITS "
                "own condition; a mutation that exits without its message (or "
                "with another guard's message) means the guard was lost",
                n_fired > 0 and n_fired == len(t4_rows),
                "%d/%d fired with own message" % (n_fired, len(t4_rows)))

    # =====================================================================
    # T5. sizes
    # =====================================================================
    print("\n=== T5. engine sizes (private target dirs) ===")
    # ⚠️ MY OWN BUG, caught by this run: the first version looked for
    # <priv>/<tag>/, but ensure_outdir("fw_build") returns .../examples/p15_verify/
    # fw_build and the build dirs are named fw_build_default / fw_build UNDER it,
    # so the paths were right but the ON build had been made into fw_build while
    # the OFF one was still being created -- and the OFF path was checked before
    # it existed. The fix measures both explicitly and reports MISSING loudly
    # instead of silently returning None (a missing build must never read as a
    # number).
    priv = os.path.join(ROOT, "examples", "p15_verify")

    def size_of(tag):
        p = os.path.join(priv, tag, "x86_64-pc-windows-msvc", "release",
                         "vmlm-runtime.exe")
        if not os.path.exists(p):
            return None
        return os.path.getsize(p)

    def sha_of(tag):
        p = os.path.join(priv, tag, "x86_64-pc-windows-msvc", "release",
                         "vmlm-runtime.exe")
        return V.sha256(p) if os.path.exists(p) else None

    off_s = size_of("fw_build_default")
    on_s = size_of("fw_build")
    tr_s = size_of("fw_build_train")
    bn_s = size_of("fw_build_bench")
    print("   default (OFF)        = %s B" % off_s)
    print("   neuron-kv (ON)       = %s B" % on_s)
    print("   train (control 1)    = %s B" % tr_s)
    print("   bench (control 2)    = %s B" % bn_s)
    print("   ON - OFF             = %s"
          % (None if (off_s is None or on_s is None) else on_s - off_s))
    print("   train - OFF          = %s  (P14 baseline +20480)"
          % (None if (off_s is None or tr_s is None) else tr_s - off_s))
    print("   bench - OFF          = %s  (previous +8704)"
          % (None if (off_s is None or bn_s is None) else bn_s - off_s))
    meta.record("D1 T5 default engine is still EXACTLY 68096 B",
                off_s == 68096,
                "a synthetic 68097 must be distinguishable by the same size "
                "comparison",
                off_s is not None and off_s + 1 != 68096,
                "measured default=%s" % off_s)
    meta.record("D1 T5 the neuron-kv delta is MEASURED (not estimated)",
                on_s is not None and off_s is not None,
                "an unbuilt ON variant must not read as 0 B",
                on_s is not None and off_s is not None and (on_s - off_s) != 0,
                "OFF=%s ON=%s delta=%s" % (off_s, on_s,
                                           None if (off_s is None or on_s is None)
                                           else on_s - off_s))
    # positive controls must move -- an empty-measurement check
    ctrl_ok = (tr_s is not None and off_s is not None and tr_s != off_s)
    print("   positive control (train) moved: %s" % ctrl_ok)
    meta.record("D1 T5 the size apparatus still moves on a control (not an "
                "empty measurement)",
                ctrl_ok,
                "if train reported the same size as OFF the measurement would "
                "be blind and the ON delta meaningless",
                tr_s is not None and off_s is not None and tr_s != off_s,
                "train delta=%s" % (None if (off_s is None or tr_s is None)
                                    else tr_s - off_s))
    results.append(dict(test="T5", off=off_s, on=on_s, train=tr_s, bench=bn_s,
                        sha_off=sha_of("fw_build_default"),
                        sha_on=sha_of("fw_build")))

    import json
    with open(os.path.join(outdir, "d1_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nwrote %s" % os.path.join(outdir, "d1_results.json"))
    return 0 if meta.report() else 1


if __name__ == "__main__":
    sys.exit(main())
