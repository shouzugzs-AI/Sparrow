#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P15-S5 LAYOUT + WRONG-STRIDE + TRANSPOSE discriminator (independent verifier).

WHAT THIS ATTACKS (task-4 attack surface #1 and #2)
    1. The byte layout of the ACTUAL EMITTED section file, re-derived from
       primitives (little-endian u32 reads) and compared against an independent
       reconstruction from the Rust source text. The implementer's Python
       helpers are NOT imported: a wrong formula used by both the writer and the
       checker agrees with itself. That is stated policy in this project
       (docs/P14_verification.md §1) and it is the whole reason this file exists.
    2. The wrong-stride discriminator, built the way the project finally got it
       right after getting it wrong once:
         * docs/P14_S5_O6_报告.md §4.4 item 2 -- the implementer's first
           wrong-stride meta-check used the SAME wrong stride on the writer and
           the reader. It was self-consistent, it PASSED, and it proved nothing.
           Corrected, it showed 115/120 differences.
         * docs/P14_verification.md §0.1 item 2 -- the previous verifier's
           transpose negative control had a decode that was the exact inverse of
           its own encode, so a self-consistently WRONG layout round-tripped
           perfectly and the control proved nothing. Fixed by encoding per spec
           and decoding per spec.
       So EVERY discriminator here lays the DATA out correctly and gives only
       the READER (or only the SPEC) the wrong convention. The self-consistent
       variant is included ONLY as a documented trap, and the harness asserts
       that the trap does NOT discriminate -- if the trap ever started
       discriminating, the control would be measuring something else.
    3. A [dim]/[slot] transpose ON DISK, read back per spec.

EVERY claim here is paired with a can-fail control (MetaCheck). A row that
cannot fail is reported UNPROVEN.

Run from <repo>:
    python tools/p15_verify_layout.py
"""
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p15_verify_lib as V  # noqa: E402

NL, NKH, HD, PMAX = 24, 2, 128, 1024


# ===========================================================================
# 1. the emitted section file, found without being told where it is
# ===========================================================================
def find_section_files():
    """Locate every candidate emitted neuron_kv section under examples/p15_s5.

    Deliberately NOT given the implementer's filename: the verifier should find
    the artifact from the filesystem, so that "the file exists at the path the
    task named" is itself an observation. Candidates are ranked by whether they
    parse as a section with the expected geometry.
    """
    base = os.path.join(V.ROOT, "examples", "p15_s5")
    out = []
    if not os.path.isdir(base):
        return out
    for dirpath, _dirs, files in os.walk(base):
        for fn in files:
            p = os.path.join(dirpath, fn)
            try:
                sz = os.path.getsize(p)
            except OSError:
                continue
            if sz == 0 or sz > 64 * 1024 * 1024:
                continue
            if sz < V.NKV_HDR:
                continue
            try:
                with open(p, "rb") as f:
                    head = f.read(V.NKV_HDR)
                h = dict(n_layers=V.rdu32(head, 0), n_kv_heads=V.rdu32(head, 4),
                         head_dim=V.rdu32(head, 8),
                         max_seq=V.rdu32(head, 12), stride=V.rdu32(head, 16))
            except Exception:  # noqa: BLE001
                continue
            plausible = (h["n_layers"] == NL and h["n_kv_heads"] == NKH
                         and h["head_dim"] == HD)
            exact = (sz == V.NKV_HDR + NL * NKH * HD * V.NKV_ENTRY)
            out.append(dict(path=p, size=sz, header=h, plausible=plausible,
                            exact_size=exact))
    out.sort(key=lambda r: (not r["plausible"], not r["exact_size"], -r["size"]))
    return out


def main():
    print("=" * 78)
    print("P15-S5 LAYOUT / WRONG-STRIDE / TRANSPOSE (independent verifier)")
    print("=" * 78)
    meta = V.MetaCheck()
    ok, mdet = V.MetaCheck().self_test()
    print("MetaCheck self-test: %s %s" % ("OK" if ok else "BROKEN", mdet))

    cc = V.cross_check_source_text()
    print("\n[SRC] constants re-read from the Rust SOURCE TEXT: %s" % cc)
    src_ok = (cc["NKV_HDR"] == 20 and cc["NKV_ENTRY"] == 4
              and cc["SEC_NEURON_KV"] == 1)
    # CAN-FAIL control: the SAME comparison must reject a source that declares
    # different constants. Exercised against a synthetic text, so the control
    # cannot be satisfied by the real file happening to be right.
    import tempfile
    fd, tmp = tempfile.mkstemp(suffix=".rs", text=True)
    os.close(fd)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("pub const SEC_NEURON_KV: u32 = 7;\n"
                "pub const NKV_ENTRY: usize = 8;\n"
                "pub const NKV_HDR: usize = 32;\n")
    cc_bad = V.cross_check_source_text(path=tmp)
    os.remove(tmp)
    src_control = not (cc_bad["NKV_HDR"] == 20 and cc_bad["NKV_ENTRY"] == 4
                       and cc_bad["SEC_NEURON_KV"] == 1)
    print("  CAN-FAIL synthetic format.rs with 32/8/7 rejected: %s (%s)"
          % (src_control, {k: cc_bad[k] for k in ("NKV_HDR", "NKV_ENTRY",
                                                  "SEC_NEURON_KV")}))
    meta.record("LAYOUT source constants are NKV_HDR=20 / NKV_ENTRY=4 / "
                "SEC_NEURON_KV=1",
                src_ok,
                "a synthetic format.rs declaring 32/8/7 must be rejected by the "
                "same comparison",
                src_control, "real=%s synthetic_rejected=%s" % (cc, src_control))

    # -----------------------------------------------------------------------
    # 2. find the emitted section
    # -----------------------------------------------------------------------
    print("\n[FIND] candidate section files under examples/p15_s5/")
    cands = find_section_files()
    if not cands:
        print("  NONE FOUND. examples/p15_s5/ does not exist or holds no section.")
        meta.record("SECTION exists on disk", False,
                    "no candidate file found", False,
                    "nothing to parse -- UNVERIFIED, not PASS")
        meta.report()
        return 2
    for c in cands[:8]:
        print("  %-56s size=%-8d hdr=%s plausible=%s exact=%s"
              % (os.path.relpath(c["path"], V.ROOT), c["size"], c["header"],
                 c["plausible"], c["exact_size"]))
    real = [c for c in cands if c["plausible"]]
    # CAN-FAIL: the same "is this a section with the right geometry" predicate
    # must reject the candidate files that are NOT one. The emitter's own control
    # blobs (_control_*) are in this directory, so there is real counter-material
    # to reject: the truncated one has the wrong size, and the .vmb / .json are
    # not sections at all.
    rejected = [c for c in cands if not c["plausible"] or not c["exact_size"]]
    section_predicate_ok = bool(real) and bool(rejected)
    meta.record("SECTION exists on disk with the MiniCPM geometry",
                bool(real),
                "the same predicate must reject the non-section candidates "
                "(the .vmb blob, the .json dump, the truncated control)",
                section_predicate_ok,
                "%d plausible of %d candidates; %d rejected"
                % (len(real), len(cands), len(rejected)))
    if not real:
        meta.report()
        return 2
    pick = real[0]
    print("  -> using %s" % os.path.relpath(pick["path"], V.ROOT))

    with open(pick["path"], "rb") as f:
        blob = f.read()

    # -----------------------------------------------------------------------
    # 3. byte-level re-parse from primitives
    # -----------------------------------------------------------------------
    print("\n[BYTES] re-parse from primitives (struct.unpack_from, LE)")
    p = V.parse_section_primitives(blob)
    print("  header            : %s" % p["header"])
    print("  n_neurons         : %d" % p["n_neurons"])
    print("  expected bytes    : %d" % p["expected_section_bytes"])
    print("  actual bytes      : %d" % p["actual_bytes"])
    print("  trailing bytes    : %d" % p["trailing_bytes"])
    print("  computed n_neurons= 24*2*128 = %d (independent arithmetic)"
          % (NL * NKH * HD))
    print("  computed bytes    = 20 + %d*4 = %d (independent arithmetic)"
          % (NL * NKH * HD, 20 + NL * NKH * HD * 4))

    hdr_ok = (p["header"]["n_layers"] == NL and p["header"]["n_kv_heads"] == NKH
              and p["header"]["head_dim"] == HD
              and p["header"]["max_seq_per_neuron"] == PMAX)
    size_ok = (p["actual_bytes"] == 20 + NL * NKH * HD * 4)
    stride_ok = (p["header"]["stride"] == PMAX)
    n_ok = (p["n_neurons"] == NL * NKH * HD)
    print("  header matches 24/2/128/PMAX : %s" % hdr_ok)
    print("  size == 24596                : %s (%d)" % (size_ok, p["actual_bytes"]))
    print("  stride == PMAX (tight, NOT HD*PMAX) : %s (%d)"
          % (stride_ok, p["header"]["stride"]))
    print("  n_neurons matches            : %s (%d)" % (n_ok, p["n_neurons"]))
    dirty_ok = (hdr_ok and size_ok and stride_ok and n_ok)

    # can-fail: perturb the header in MEMORY and require the predicate to reject
    bad_hdr_ok = False
    for field, off, val in (("stride", 16, PMAX * 128), ("max_seq", 12, 1),
                            ("head_dim", 8, 127)):
        b2 = bytearray(blob)
        struct.pack_into("<I", b2, off, val)
        p2 = V.parse_section_primitives(bytes(b2))
        good2 = (p2["header"]["n_layers"] == NL
                 and p2["header"]["n_kv_heads"] == NKH
                 and p2["header"]["head_dim"] == HD
                 and p2["header"]["max_seq_per_neuron"] == PMAX
                 and p2["header"]["stride"] == PMAX
                 and p2["actual_bytes"] == 20 + NL * NKH * HD * 4
                 and p2["n_neurons"] == NL * NKH * HD)
        if not good2:
            bad_hdr_ok = True
    meta.record("LAYOUT header/size/stride/n_neurons all match the spec",
                dirty_ok,
                "header fields and size perturbed in memory must be rejected",
                bad_hdr_ok,
                "3 perturbations (stride->128*PMAX, max_seq->1, head_dim->127) "
                "all rejected=%s" % bad_hdr_ok)

    # -----------------------------------------------------------------------
    # 4. offsets[] == i*stride, counted
    # -----------------------------------------------------------------------
    print("\n[OFFSETS] entry i == i*stride ?  (counted, not a boolean)")
    mc = V.section_mismatch_counts(p)
    print("  compared %d entries, %d mismatch, first=%s"
          % (mc["n_compared"], mc["n_mismatch"], mc["first"]))

    # can-fail A: the 128x S5-A shape injected into the REAL bytes
    b3 = bytearray(blob)
    stride = p["header"]["stride"]
    for i in range(p["n_neurons"]):
        struct.pack_into("<I", b3, V.NKV_HDR + i * V.NKV_ENTRY,
                         (i * HD * stride) & 0xFFFFFFFF)
    p3 = V.parse_section_primitives(bytes(b3))
    mc3 = V.section_mismatch_counts(p3)
    print("  CAN-FAIL i*head_dim*stride injected -> %d/%d mismatch, first=%s"
          % (mc3["n_mismatch"], mc3["n_compared"], mc3["first"]))

    # can-fail B: a SINGLE off-by-one deep in the array (the one most likely to
    # be missed by a spot-check)
    idx = p["n_neurons"] - 1
    b4 = bytearray(blob)
    orig = V.rdu32(blob, V.NKV_HDR + idx * V.NKV_ENTRY)
    struct.pack_into("<I", b4, V.NKV_HDR + idx * V.NKV_ENTRY, orig + 1)
    p4 = V.parse_section_primitives(bytes(b4))
    mc4 = V.section_mismatch_counts(p4)
    print("  CAN-FAIL last entry (%d) +1 -> %d/%d mismatch, first=%s"
          % (idx, mc4["n_mismatch"], mc4["n_compared"], mc4["first"]))

    # can-fail C: u32 wrap. entry = i*stride computed in u32 would wrap; the
    # checker must count a wrap as a mismatch, not as agreement.
    #
    # ⚠️ MY OWN BUG, caught by this run and recorded rather than deleted: the
    # first version declared n_neurons=1, so `i*stride` for i=0 was 0 and no
    # overflow could occur -- the control reported "0/1 mismatch (must be >0)"
    # and made its own criterion UNPROVEN. A can-fail control that CANNOT fail
    # is the same empty-measurement defect as a criterion that cannot fail. The
    # fix is to declare enough neurons that i*stride genuinely exceeds u32::MAX,
    # and to verify the overflow actually happens before relying on the count.
    for (nl_, nkh_, hd_, stride_) in ((2, 1, 2, 1 << 30), (3, 1, 1, 1 << 31)):
        total = nl_ * nkh_ * hd_ * stride_
        if total <= 0xFFFFFFFF:
            continue
        wrap_blob = bytearray()
        for v in (nl_, nkh_, hd_, 1, stride_):
            wrap_blob += struct.pack("<I", v)
        for i in range(nl_ * nkh_ * hd_):
            wrap_blob += struct.pack("<I", (i * stride_) & 0xFFFFFFFF)
        pw = V.parse_section_primitives(bytes(wrap_blob))
        mcw = V.section_mismatch_counts(pw)
        print("  CAN-FAIL u32-overflow layout (n_neurons=%d, stride=%d, total=%d "
              "> 2^32-1) -> %d/%d mismatch"
              % (pw["n_neurons"], stride_, total, mcw["n_mismatch"],
                 mcw["n_compared"]))
        if mcw["n_mismatch"] > 0:
            break
    wrap_detected = mcw["n_mismatch"] > 0

    offsets_ok = (mc["n_mismatch"] == 0 and mc["n_compared"] == NL * NKH * HD)
    meta.record("OFFSETS every entry == i*stride",
                offsets_ok,
                "i*head_dim*stride injection, a single last-entry +1, AND a "
                "u32-wrapping layout must all be counted as mismatches",
                (mc3["n_mismatch"] > 0 and mc4["n_mismatch"] == 1
                 and wrap_detected),
                "clean %d/%d mismatch; 128x inject %d/%d; single +1 %d/%d; "
                "u32-wrap %d/%d"
                % (mc["n_mismatch"], mc["n_compared"], mc3["n_mismatch"],
                   mc3["n_compared"], mc4["n_mismatch"], mc4["n_compared"],
                   mcw["n_mismatch"], mcw["n_compared"]))

    # -----------------------------------------------------------------------
    # 5. the disk index vs the shipped ADDRESS ARITHMETIC
    # -----------------------------------------------------------------------
    print("\n[INDEX-VS-ADDR] does offsets[i] equal the address the shipped "
          "formula computes?")
    L = V.LayoutPrimitives(NL, NKH, HD, PMAX, stride)
    mism = []
    # sample the corners and a deterministic sweep, not just the ends
    sample = [0, 1, HD - 1, HD, NKH * HD - 1, NKH * HD,
              NL * NKH * HD // 2, NL * NKH * HD - 1]
    sample += list(range(0, min(64, NL * NKH * HD)))
    for i in sorted(set(sample)):
        # invert idx -> (L, h, d) and recompute the base from the shipped formula
        d = i % HD
        h = (i // HD) % NKH
        lay = i // (HD * NKH)
        base = L.neuron_base(lay, h, d)
        if base != p["offsets"][i]:
            mism.append((i, (lay, h, d), base, p["offsets"][i]))
    print("  sampled %d neuron indices; %d disagreements"
          % (len(set(sample)), len(mism)))
    if mism:
        for m in mism[:6]:
            print("    idx=%d (L,h,d)=%s shipped_base=%d disk_offsets=%d" % m)

    # can-fail for THIS comparison: recompute the base with head_dim*stride,
    # i.e. the pre-fix convention, and require disagreements to appear.
    bad_mism = []
    for i in sorted(set(sample)):
        d = i % HD
        h = (i // HD) % NKH
        lay = i // (HD * NKH)
        idx = (lay * NKH + h) * HD + d
        wrong = idx * (HD * stride)
        if wrong != p["offsets"][i]:
            bad_mism.append(i)
    print("  CAN-FAIL with base=idx*(head_dim*stride): %d/%d disagree"
          % (len(bad_mism), len(set(sample))))
    meta.record("INDEX-VS-ADDR disk offsets[] == shipped neuron_base()",
                not mism,
                "the pre-fix base=idx*head_dim*stride must disagree on the "
                "majority of sampled indices",
                len(bad_mism) > 0,
                "clean disagreements=%d, wrong-convention disagreements=%d/%d"
                % (len(mism), len(bad_mism), len(set(sample))))

    # -----------------------------------------------------------------------
    # 6. contiguity at the two granularities, on the SHIPPED stride
    # -----------------------------------------------------------------------
    print("\n[CONTIG] address gaps, measured per granularity (shipped stride=%d)"
          % stride)
    # (a) one neuron's whole slot range: must be unit stride
    a_neuron = [L.slot_addr(0, 0, 0, s) for s in range(PMAX)]
    g_neuron = V.gaps(a_neuron)
    neuron_contig = V.is_contiguous(a_neuron)
    print("  (a) one NEURON, all %d slots : contiguous=%s gap-set=%s"
          % (PMAX, neuron_contig, sorted(set(g_neuron))))
    # (b) one head, dims+slots walked together
    a_head = [L.slot_addr(0, 0, d, s) for d in range(4) for s in range(PMAX)]
    g_head = V.gaps(a_head)
    head_contig = V.is_contiguous(a_head)
    print("  (b) one HEAD, dims+slots     : contiguous=%s gap-set=%s"
          % (head_contig, sorted(set(g_head))))
    # (c) one head, ONE slot across dims -- THE HONEST NON-CONTIGUOUS CASE
    a_col = [L.slot_addr(0, 0, d, 0) for d in range(HD)]
    g_col = V.gaps(a_col)
    col_contig = V.is_contiguous(a_col)
    print("  (c) one HEAD, one slot x dims: contiguous=%s gap-set=%s   <-- the "
          "real stride access" % (col_contig, sorted(set(g_col))))

    print("  g_neuron first 8 gaps: %s" % g_neuron[:8])
    print("  g_col    first 8 gaps: %s" % g_col[:8])

    # (c) is the known trap: the PHENOMENON is a real strided access, but with a
    # TIGHT stride the numbers happen to look like... let's measure what they
    # actually are rather than assert.
    meta.record("CONTIG one neuron's slots are ONE contiguous load "
                "(unit stride)",
                neuron_contig and all(x == 1 for x in g_neuron),
                "a split neuron range must be rejected by the same predicate",
                not V.is_contiguous(a_neuron[:-1] + [a_neuron[-1] + 7]),
                "gaps=%s" % sorted(set(g_neuron)))
    meta.record("CONTIG per-slot-across-dims is NOT contiguous (reported "
                "honestly, not weakened)",
                not col_contig,
                "the SAME predicate must return contiguous=True for a genuine "
                "unit-stride run, otherwise it is a stub that always answers "
                "'not contiguous' and proves nothing",
                neuron_contig and all(x == 1 for x in g_neuron),
                "gaps_col=%s ; predicate returns True for one neuron's slots=%s"
                % (sorted(set(g_col)), neuron_contig))

    # -----------------------------------------------------------------------
    # 7. WRONG-STRIDE discriminator -- correct DATA, wrong READER
    # -----------------------------------------------------------------------
    print("\n[WRONG-STRIDE] data laid out per spec; only the READER is wrong")
    # A small, fully checkable case: 3 layers x 2 heads x 4 dims, PMAX=8.
    nL, nH, nD, ms = 3, 2, 4, 8
    st = ms                      # tight
    n = nL * nH * nD
    src = [float(i) for i in range(nL * nH * nD * ms)]  # unique values
    # write per spec: per_neuron[i*st + s] = value for (i, s)
    per_neuron = [None] * (n * st)
    for i in range(n):
        for s in range(ms):
            per_neuron[i * st + s] = src[i * ms + s]
    # reader per spec
    read_ok = [per_neuron[i * st + s] for i in range(n) for s in range(ms)]
    # reader with WRONG stride (st+1) over the SAME correctly-laid-out data
    read_bad = []
    for i in range(n):
        for s in range(ms):
            a = i * (st + 1) + s
            read_bad.append(per_neuron[a] if a < len(per_neuron) else None)
    diff = sum(1 for a, b in zip(read_ok, read_bad) if a != b)
    print("  correct data + wrong reader stride (%d vs %d): %d/%d values "
          "differ" % (st, st + 1, diff, len(read_ok)))

    # THE TRAP, reproduced so it is documented and can never be mistaken for
    # evidence: writer and reader SHARE the wrong stride -> perfect round trip.
    #
    # ⚠️ MY OWN BUG, caught by this run: the first version compared the decoded
    # sequence against `data_bad[:n*ms]`, which is the CONTIGUOUS prefix, not the
    # strided layout the round trip actually reproduces. It therefore reported
    # "trap perfect = False" and made the criterion UNPROVEN. The ground truth
    # for a round trip is the layout itself: data_bad[i*st_bad + s]. With the
    # correct ground truth the trap round-trips PERFECTLY, which is the whole
    # point -- it is self-consistent and discriminates nothing.
    st_bad = st + 1
    data_bad = [float(i) for i in range(n * st_bad)]
    enc_trap = {i * st_bad + s: data_bad[i * st_bad + s]
                for i in range(n) for s in range(ms)}
    dec_trap = [enc_trap[i * st_bad + s] for i in range(n) for s in range(ms)]
    trap_truth = [data_bad[i * st_bad + s] for i in range(n) for s in range(ms)]
    trap_perfect = (dec_trap == trap_truth)
    print("  TRAP (writer AND reader both wrong stride %d): round trip perfect "
          "= %s -> discriminates NOTHING" % (st_bad, trap_perfect))

    meta.record("WRONG-STRIDE discriminator: correct data + wrong reader is "
                "DETECTED",
                diff > 0,
                "the self-consistent trap must NOT be counted as detection: a "
                "detector that fires on the trap would be measuring the wrong "
                "thing",
                diff > 0 and trap_perfect,
                "%d/%d values differ; self-consistent trap perfect=%s"
                % (diff, len(read_ok), trap_perfect))

    # and the trap's own can-fail: if writer and reader both use the CORRECT
    # stride the round trip must also be perfect (so 'perfect' alone proves
    # nothing) -- documented as a row so nobody reads perfect==True as success.
    data_ok = [float(i) for i in range(n * st)]
    enc_ok = {i * st + s: data_ok[i * st + s] for i in range(n) for s in range(ms)}
    dec_ok = [enc_ok[i * st + s] for i in range(n) for s in range(ms)]
    ok_truth = [data_ok[i * st + s] for i in range(n) for s in range(ms)]
    ok_perfect = (dec_ok == ok_truth)
    meta.record("WRONG-STRIDE: a perfect round trip alone proves NOTHING "
                "(both conventions round-trip)",
                bool(ok_perfect and trap_perfect),
                "if either round trip were imperfect this row would be claiming "
                "something other than what it says",
                bool(ok_perfect and trap_perfect),
                "correct-stride perfect=%s AND wrong-stride self-consistent "
                "perfect=%s -> round-trip identity cannot be the discriminator"
                % (ok_perfect, trap_perfect))

    # -----------------------------------------------------------------------
    # 8. [dim]/[slot] TRANSPOSE read back per spec
    # -----------------------------------------------------------------------
    print("\n[TRANSPOSE] a [dim]/[slot] transposed layout read back per spec")
    # Spec layout for one head: per_neuron[d*ms + s]. A transposed writer wrote
    # per_neuron[d*ms + s] = value(s, d) instead of value(d, s) -- i.e. it
    # swapped the two inner axes. Reading it per spec then yields a permutation.
    spec = [None] * (nD * ms)
    for d in range(nD):
        for s in range(ms):
            spec[d * ms + s] = (d, s)
    trans = [None] * (nD * ms)
    for d in range(nD):
        for s in range(ms):
            # transposed: the writer stored (s, d) at the (d, s) address
            trans[d * ms + s] = (s, d)
    perm_mismatch = sum(1 for a, b in zip(spec, trans) if a != b)
    print("  spec vs transposed, read back per spec: %d/%d positions wrong"
          % (perm_mismatch, len(spec)))
    # can-fail: the IDENTITY (untransposed) layout must show 0 mismatches, so
    # the comparison is not simply always-failing.
    ident_mismatch = sum(1 for a, b in zip(spec, spec) if a != b)
    print("  CAN-FAIL control: un-transposed vs itself -> %d mismatches (must "
          "be 0)" % ident_mismatch)
    meta.record("TRANSPOSE [dim]/[slot] swap is detected when read per spec",
                perm_mismatch > 0,
                "the un-transposed layout must show 0 mismatches, otherwise "
                "the comparison always fails and proves nothing",
                perm_mismatch > 0 and ident_mismatch == 0,
                "%d/%d wrong; identity control=%d" % (perm_mismatch, len(spec),
                                                      ident_mismatch))

    # and on the REAL data: does the emitted section describe a TRANSPOSED
    # index (head_dim outermost, layer innermost -- i.e. the [L][h][d] axes
    # written in the wrong nesting order)? offsets[i] must equal i*stride for the
    # SPEC order, and must therefore DIFFER from the transposed order's base for
    # nearly every i. A count near zero would mean the two orders coincide,
    # which is what a degenerate test looks like.
    #
    # ⚠️ MY OWN BUG, caught by this run: the first version's expected value was
    # "0 mismatches", i.e. it asserted the emitted index MATCHES a transposed
    # reading -- the exact opposite of the claim, and it made the criterion FAIL
    # while the underlying data was correct. The `and i != idx_t` clause also
    # silently skipped the one index where the two orders coincide, hiding the
    # degeneracy. Corrected: the SPEC order must be the one that matches, and the
    # transposed order must be the one that disagrees.
    print("\n  on the REAL emitted section: spec order vs transposed order")
    spec_match = 0
    trans_match = 0
    n_sampled = min(p["n_neurons"], 4096)
    coincident = 0
    for i in range(n_sampled):
        d = i % HD
        h = (i // HD) % NKH
        lay = i // (HD * NKH)
        idx_t = (d * NKH + h) * NL + lay        # head_dim outermost
        if idx_t == i:
            coincident += 1
        if p["offsets"][i] == i * stride:
            spec_match += 1
        if p["offsets"][i] == idx_t * stride:
            trans_match += 1
    print("  sampled %d entries: spec-order matches=%d, transposed-order "
          "matches=%d, orders coincide at %d index(es)"
          % (n_sampled, spec_match, trans_match, coincident))
    meta.record("TRANSPOSE the emitted index is the SPEC order, not a "
                "[dim]/[layer] swap",
                spec_match == n_sampled and trans_match < n_sampled,
                "the transposed order must NOT match all entries; if it did, "
                "the two orders would be indistinguishable and the test would "
                "prove nothing",
                trans_match < n_sampled and n_sampled > 1,
                "spec=%d/%d, transposed=%d/%d, coincident=%d"
                % (spec_match, n_sampled, trans_match, n_sampled, coincident))

    v = V.ensure_outdir("layout")
    import json
    with open(os.path.join(v, "layout_result.json"), "w", encoding="utf-8") as f:
        json.dump(dict(section=os.path.relpath(pick["path"], V.ROOT),
                       header=p["header"], n_neurons=p["n_neurons"],
                       size=p["actual_bytes"],
                       offsets_mismatch=mc, gaps_neuron=sorted(set(g_neuron)),
                       gaps_head=sorted(set(g_head)),
                       gaps_col=sorted(set(g_col)),
                       neuron_contiguous=neuron_contig,
                       head_contiguous=head_contig,
                       col_contiguous=col_contig,
                       wrong_stride_diff=diff, trap_perfect=trap_perfect,
                       transpose_mismatch=perm_mismatch),
                  f, indent=2, ensure_ascii=False)
    print("\nwrote %s" % os.path.join(v, "layout_result.json"))
    return 0 if meta.report() else 1


if __name__ == "__main__":
    sys.exit(main())
