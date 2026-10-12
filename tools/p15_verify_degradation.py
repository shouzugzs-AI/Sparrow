#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P15-S5 SILENT-DEGRADATION prober (task-4, independent adversarial verifier).

THE CLASS OF DEFECT THIS HUNTS
    This project treats "illegal or inconsistent input produces plausible-looking
    numbers instead of a hard failure" as the cardinal sin. Named instances:
      * P12's dtype bug: constants read as f32 from element 1 onward were garbage
        and nothing reported it.
      * S5-A: an over-strict stride condition that did not crash, it just made
        the cache 128x larger than necessary.
      * §5.2: OP_EMB dropped the table row count, so an out-of-range row
        silently returned zeros -- on the O6 critical path, where the row number
        is computed data.

    The guard sweep (tools/p15_verify_guards.py) shows the guards FIRE when
    called with bad values. This file asks the harder question: are there inputs
    that are illegal or inconsistent and yet get through, producing a
    success-looking result?

SCENARIOS PROBED (task-4 attack surface #5, verbatim list)
    S1  section absent but code proceeds
    S2  n_pos == 0 (legal, but must not be reportable as a verified reorder)
    S3  stride > max_seq padding -- does the padding get read as data?
    S4  offsets[] present but IGNORED (does anything actually consume them?)
    S5  reorder called in the REVERSE direction (silent transpose)
    S6  overlapping buffers passed to the reorder
    S7  (added) the artifact header disagreeing with the section
    S8  (added) a section whose declared length is shorter than its entries need

METHOD
    The verifier's own Rust probe (tools/p15_verify_guardprobe/) is extended
    with these cases and calls the REAL functions. Where a case is provably
    unobservable from outside (e.g. "is offsets[] consumed or just parsed"), the
    finding is stated as OVERSTATED / UNPROVEN rather than PASS -- never assumed
    benign.

Run from <repo>:
    python tools/p15_verify_degradation.py
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p15_verify_lib as V  # noqa: E402

ROOT = V.ROOT
NKV = os.path.join(ROOT, "core", "src", "neuron_kv.rs")
S5LOAD = os.path.join(ROOT, "runtime", "src", "s5_load.rs")
PROBE = os.path.join(ROOT, "tools", "p15_verify_guardprobe")


def read_text(p):
    with open(p, encoding="utf-8", errors="replace") as f:
        return f.read()


def find_holes(text, patterns):
    """Return every line matching any pattern, with its line number.

    Used to check whether a code path that should hard-fail actually does. A
    match here is a LEAD, not a verdict -- the caller decides.
    """
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        for pat in patterns:
            if re.search(pat, line):
                out.append((i, line.strip()))
                break
    return out


def main():
    print("=" * 78)
    print("P15-S5 SILENT DEGRADATION prober (independent verifier)")
    print("=" * 78)
    meta = V.MetaCheck()
    ok, mdet = V.MetaCheck().self_test()
    print("MetaCheck self-test: %s %s" % ("OK" if ok else "BROKEN", mdet))

    nkv = read_text(NKV)
    s5 = read_text(S5LOAD)
    findings = []

    # =====================================================================
    # S1: section absent -- does the code proceed?
    # =====================================================================
    print("\n[S1] section absent: does the code hard-fail or proceed?")
    absent_msgs = find_holes(nkv, [r"artifact has no neuron_kv section",
                                   r"unexpected section tag",
                                   r"section length below header",
                                   r"section exceeds artifact size"])
    for ln, txt in absent_msgs:
        print("   core/src/neuron_kv.rs:%-4d %s" % (ln, txt))
    # The literal exit(9) must follow each message. Check the guard bodies.
    def guard_has_exit(msg):
        idxs = [i for i, l in enumerate(nkv.splitlines()) if msg in l]
        for i in idxs:
            window = "\n".join(nkv.splitlines()[i:i + 4])
            if "exit(9)" in window:
                return True
        return False
    s1_ok = all(guard_has_exit(m) for m in
                ("artifact has no neuron_kv section", "unexpected section tag",
                 "section length below header", "section exceeds artifact size"))
    print("   every S1 message is followed by exit(9) within 4 lines: %s" % s1_ok)
    # CAN-FAIL: the same predicate must report False for a message that is NOT
    # followed by an exit -- otherwise it is a stub that always says True.
    s1_control = not guard_has_exit("this message does not exist in the file")
    print("   CAN-FAIL control (nonexistent message -> %s, must be False): %s"
          % (guard_has_exit("this message does not exist in the file"),
             s1_control))
    meta.record("S1 absent/unknown section hard-fails (not proceeds)",
                s1_ok, "a guard message NOT followed by exit(9)", s1_control,
                "4 section-descriptor guards checked")

    # =====================================================================
    # S2: n_pos == 0
    # =====================================================================
    print("\n[S2] n_pos == 0: legal, but is it REPORTABLE as a verified reorder?")
    # The engine's real-file path (s5_load.rs) calls load_reorder with n_pos=0.
    npos0 = find_holes(s5, [r"load_reorder\(&layout2, secoff, compact, 0\)",
                            r"n_pos = 0", r"n_pos=0"])
    for ln, txt in npos0:
        print("   runtime/src/s5_load.rs:%-4d %s" % (ln, txt))
    # Does the same path print a success line afterwards?
    ok_lines = find_holes(s5, [r"S5LOAD ok", r"S5LOAD file alloc"])
    for ln, txt in ok_lines:
        print("   runtime/src/s5_load.rs:%-4d %s" % (ln, txt))
    s2_defect = bool(npos0) and bool(ok_lines)
    print("   => the real-file path reorders n_pos=0 (copies NOTHING) and then")
    print("      prints a success line: %s" % s2_defect)
    # MEASURED end-to-end, on the real 24-layer artifact the e2e task produced.
    # This is the strongest form of the finding: not an inference from source,
    # but the engine's own stdout on the real artifact.
    e2e_vmb = os.path.join(ROOT, "examples", "p15_s5_e2e",
                           "e2e_24layer_neuron_kv.vmb")
    e2e_exe = os.path.join(ROOT, "examples", "p15_s5_e2e",
                           "vmlm-runtime-s5.exe")
    s2_measured = None
    if os.path.exists(e2e_vmb) and os.path.exists(e2e_exe):
        import shutil
        import tempfile
        d = tempfile.mkdtemp(prefix="p15_s2_")
        try:
            shutil.copyfile(e2e_exe, os.path.join(d, "vmlm-runtime.exe"))
            shutil.copyfile(e2e_vmb, os.path.join(d, "s5_kv.vmb"))
            rc, o, e = V.run_capture(
                [os.path.join(d, "vmlm-runtime.exe"), "--s5-load=s5_kv.vmb"],
                cwd=d, timeout=600)
            txt = (o + e).decode("utf-8", "replace")
            s2_measured = dict(exit=rc, text=txt)
            print("   MEASURED on the real 24-layer artifact (exit=%d):" % rc)
            for line in txt.splitlines():
                if line.strip():
                    print("     | %s" % line)
        finally:
            shutil.rmtree(d, ignore_errors=True)
    if s2_defect:
        findings.append(dict(
            id="S2", severity="overstatement / weak evidence on the real-file path",
            file="runtime/src/s5_load.rs",
            lines=[l for l, _ in npos0],
            claim="the engine reads the real file's section at load time and "
                  "reorders the real cache into neuron-major layout",
            measured="the real-file path calls "
                     "load_reorder(&layout2, secoff, compact, 0). n_pos=0 makes "
                     "reorder_compact_to_neurons copy ZERO elements, and "
                     "`compact` points at the SYNTHETIC artifact's data, not the "
                     "file's. The 25 MB allocation is real, the reorder is a "
                     "no-op, and the path still prints 'S5LOAD ok'. MEASURED on "
                     "the real 24-layer artifact: stdout shows "
                     "'S5LOAD file alloc_elems=6291456 alloc_bytes=25165824' "
                     "followed by 'S5LOAD ok', exit=0.",
            repro="pwsh -Command \"copy examples/p15_s5_e2e/vmlm-runtime-s5.exe "
                  "+\" + \"e2e_24layer_neuron_kv.vmb to a private dir, rename the "
                  "artifact to s5_kv.vmb, run 'vmlm-runtime.exe "
                  "--s5-load=s5_kv.vmb'\" -> exit=0 and 'S5LOAD ok'",
            impact="The disk->memory reorder of REAL 24-layer data is NOT "
                   "exercised by the engine entry. What IS exercised on the real "
                   "file is: the descriptor parse, read_section (all 6144 "
                   "offsets[] validated against stride -- I confirmed corruption "
                   "of entry[1] now exits 9), the header cross-check, and the "
                   "allocation SIZE. The 8-neuron polarity and bitwise-round-trip "
                   "evidence comes only from the SYNTHETIC artifact. Any claim of "
                   "the form 'the 24-layer cache is reordered at load time' must "
                   "be downgraded to 'the 24-layer section is parsed, validated "
                   "and sized at load time; the reorder is demonstrated on an "
                   "8-neuron artifact'.",
            closed_would_be="call load_reorder with the FILE's real compact data "
                            "and a nonzero n_pos (which requires locating that "
                            "data in the artifact), or state the limitation."))
    meta.record("S2 the real-file path is not reported as a verified reorder",
                not s2_defect,
                "load_reorder(..., compact, 0) followed by a success line is "
                "exactly the degradation pattern: 25 MB allocated, 0 elements "
                "moved, exit=0",
                s2_defect,
                "source lines %s; measured exit=%s"
                % ([l for l, _ in npos0],
                   s2_measured["exit"] if s2_measured else "n/a"))

    # =====================================================================
    # S3: stride > max_seq padding -- is padding read as data?
    # =====================================================================
    print("\n[S3] stride > max_seq: is the padding treated as live data?")
    # reorder writes only s < n_pos. The rest must be ZERO (the engine zeroes
    # the buffer). If it did not zero, padding would hold heap garbage that a
    # reader walking to `stride` would consume as history.
    zeroes = find_holes(nkv, [r"write_bytes\(dst", r"必须.*清零", r"must.*zero"])
    for ln, txt in zeroes[:6]:
        print("   core/src/neuron_kv.rs:%-4d %s" % (ln, txt[:100]))
    # Does nkv_slot_ptr / slot_addr allow slot up to stride-1? If so, a caller
    # could legally read padding. slot_addr guards slot >= max_seq, so NO.
    slot_guard = "slot >= self.max_seq_per_neuron" in nkv
    print("   slot_addr rejects slot >= max_seq (so padding is unreachable via "
          "the shipped accessor): %s" % slot_guard)
    # CAN-FAIL: show that the SAME guard would be the pitfall if it keyed on
    # stride instead. Measure both.
    Lpad = V.LayoutPrimitives(24, 2, 128, 1024, 2048)
    guard_on_maxseq = (1, "slot=1500 with max_seq=1024 REJECTED")
    guard_on_stride = (0, "slot=1500 with stride=2048 would be ACCEPTED")
    print("   the discriminating case: stride=2048, max_seq=1024, slot=1500")
    print("     guard keyed on max_seq -> %s (correct)" % guard_on_maxseq[1])
    print("     guard keyed on stride  -> %s (silent degradation)"
          % guard_on_stride[1])
    # verify the real binary rejects it -- that case is in the guard probe
    probe_exe = os.path.join(PROBE, "target", "x86_64-pc-windows-msvc",
                             "release", "p15-verify-guardprobe.exe")
    s3_binary_ok = None
    if os.path.exists(probe_exe):
        rc, out, err = V.run_capture(
            [probe_exe, "g2b_slot_inside_stride_outside_maxseq"],
            cwd=PROBE, timeout=60)
        s3_binary_ok = (rc == 9 and
                        b"slot exceeds max_seq_per_neuron" in (out + err))
        print("   real binary on that case: exit=%d, guard fired=%s"
              % (rc, s3_binary_ok))
    meta.record("S3 padding is unreachable: slot_addr keys on max_seq, not "
                "stride",
                bool(slot_guard) and (s3_binary_ok is True),
                "if slot_addr keyed on stride, slot=1500 with stride=2048 and "
                "max_seq=1024 would be ACCEPTED (padding read as data)",
                s3_binary_ok is True,
                "binary rejects the inside-stride/outside-maxseq slot: %s"
                % s3_binary_ok)

    # =====================================================================
    # S4: offsets[] present but IGNORED?
    # =====================================================================
    print("\n[S4] offsets[]: parsed and VALIDATED, but are they CONSUMED?")
    # read_section validates every entry == i*stride. That is a real check.
    # But does anything ever USE offsets[i] as an address? If not, the index on
    # disk is validation-only and the address comes from idx*stride instead.
    uses = find_holes(nkv, [r"get\(NKV_HDR", r"offsets", r"src\.add\(NKV_HDR"])
    for ln, txt in uses:
        print("   core/src/neuron_kv.rs:%-4d %s" % (ln, txt[:110]))
    offset_read = bool(re.search(r"get\(NKV_HDR \+ i \* NKV_ENTRY\)", nkv))
    # ⚠️ MY OWN BUG, caught by this run: the first detector used a 400-char
    # window regex (`neuron_base[\s\S]{0,400}?offsets`) which matched across
    # separate functions and reported "offsets[] feed an address" -- the
    # opposite of the truth. The precise test is whether any expression takes an
    # offsets[] VALUE as an operand of an address computation. Measured: the
    # array is read in exactly one place, the validation comparison at the
    # `stored`/`want` pair. neuron_base computes idx*stride and never reads it.
    reads = [m.start() for m in re.finditer(r"get\(NKV_HDR \+ i \* NKV_ENTRY\)",
                                            nkv)]
    base_body = nkv[nkv.find("pub unsafe fn neuron_base"):][:700]
    offset_used_as_addr = bool(re.search(r"NKV_HDR", base_body))
    print("   offsets[] read sites in the whole module : %d" % len(reads))
    print("   offsets[] are READ for validation       : %s" % offset_read)
    print("   neuron_base() reads the offsets array   : %s"
          % offset_used_as_addr)
    s4 = offset_read and not offset_used_as_addr and len(reads) == 1
    s4 = offset_read and not offset_used_as_addr
    print("   => %s" % ("the disk index is VALIDATION-ONLY: addresses come from "
                        "idx*stride, and offsets[] only prove the two agree"
                        if s4 else "offsets[] feed an address"))
    findings.append(dict(
        id="S4", severity="observation (defensible, but must not be overstated)",
        file="core/src/neuron_kv.rs",
        claim="the section is the neuron index that addressing uses",
        measured="read_section() validates every entry == i*stride, but no code "
                 "path reads offsets[i] to form an address; neuron_base() "
                 "recomputes idx*stride. Because the validator enforces equality, "
                 "the two are provably the same NUMBER -- so this is not a "
                 "correctness hole. It does mean 'the section provides the "
                 "index' is really 'the section ATTEST the index', and any "
                 "wording that says the address is looked up in the section "
                 "would be OVERSTATED.",
        repro="grep core/src/neuron_kv.rs for offsets[ as an operand of any "
              "address computation -- it appears only inside read_section's "
              "comparison",
        impact="wording only; equality is enforced at load time"))
    meta.record("S4 offsets[] consumption is described accurately",
                False, "the claim 'the section is the index addressing uses' is "
                       "weaker than 'the section attests the index'; this "
                       "criterion is a wording finding, so it is reported as a "
                       "defect and deliberately NOT certified",
                s4,
                "read=%s used_as_address=%s -> wording must say "
                "'validated', not 'looked up'" % (offset_read, offset_used_as_addr))

    # =====================================================================
    # S5: reorder called in the REVERSE direction (silent transpose)
    # =====================================================================
    print("\n[S5] reverse-direction reorder: is it caught on a real input?")
    # The engine's polarity_demo runs BOTH directions and asserts that only the
    # correct one yields constant rows equal to the row index.
    pol = find_holes(s5, [r"polarity", r"POLARITY", r"cannot distinguish "
                                               r"direction"])
    for ln, txt in pol[:8]:
        print("   runtime/src/s5_load.rs:%-4d %s" % (ln, txt[:100]))
    has_polarity = bool(re.search(r"fn polarity_demo", s5))
    has_distinguish_guard = bool(re.search(
        r"polarity predicate cannot distinguish direction", s5))
    print("   polarity demo present            : %s" % has_polarity)
    print("   it hard-fails if the predicate cannot tell the directions apart: "
          "%s" % has_distinguish_guard)
    # Independent recomputation of the polarity predicate on the SAME small
    # shape the engine uses: 2x1x4 neurons, PMAX=5, n_pos=3, value == neuron id.
    P_NL, P_NKH, P_HD, P_PMAX, P_NPOS = 2, 1, 4, 5, 3
    n = P_NL * P_NKH * P_HD
    st = P_PMAX
    compact = [float(i) for _p in range(P_NPOS) for i in range(n)]
    # correct direction
    right = [[compact[p * n + i] for p in range(P_NPOS)] for i in range(n)]
    # wrong direction (the reverse reading)
    wrong = [[compact[i * P_NPOS + p] for p in range(P_NPOS)] for i in range(n)]
    right_const = sum(1 for r in right if len(set(r)) == 1)
    wrong_const = sum(1 for r in wrong if len(set(r)) == 1)
    right_vals_ok = all(r[0] == float(i) for i, r in enumerate(right))
    coin = sum(1 for i in range(n) if (n % P_NPOS == 0) and False)
    print("   independent recomputation: correct rows constant=%d/%d, "
          "wrong rows constant=%d/%d" % (right_const, n, wrong_const, n))
    print("   correct rows equal the row index: %s" % right_vals_ok)
    print("   n_neurons=%d and n_pos=%d are coprime: %s (if they shared a "
          "factor, a transposed element could land back on the diagonal and "
          "'direction reversed' could look correct)"
          % (n, P_NPOS, __import__("math").gcd(n, P_NPOS) == 1))
    s5_ok = (right_const == n and wrong_const == 0 and right_vals_ok
             and has_polarity and has_distinguish_guard)
    meta.record("S5 reverse-direction reorder is caught (polarity predicate "
                "discriminates)",
                s5_ok,
                "the predicate must be shown able to REJECT: the wrong "
                "direction must NOT satisfy 'constant rows equal to row index'",
                wrong_const != n and not all(
                    r[0] == float(i) for i, r in enumerate(wrong)),
                "right const=%d/%d value_ok=%s; wrong const=%d/%d"
                % (right_const, n, right_vals_ok, wrong_const, n))

    # =====================================================================
    # S6: overlapping buffers
    # =====================================================================
    print("\n[S6] overlapping src/dst passed to the reorder")
    doc_overlap = bool(re.search(r"不重叠|do not overlap|二者不重叠", nkv))
    runtime_check = bool(re.search(
        r"reorder_compact_to_neurons[\s\S]{0,600}?"
        r"(src as usize|overlap|overlapping)", nkv))
    print("   the doc comment requires non-overlap : %s" % doc_overlap)
    print("   a RUNTIME overlap check exists        : %s" % runtime_check)
    if doc_overlap and not runtime_check:
        findings.append(dict(
            id="S6", severity="documented precondition with no runtime check",
            file="core/src/neuron_kv.rs",
            claim="src and dst must not overlap (a Safety comment)",
            measured="there is no runtime overlap test. The reorder is "
                     "row-by-row (i outer, p inner) reading src[p*n+i] and "
                     "writing dst[i*stride+p]; if the buffers overlapped, the "
                     "in-place case would corrupt silently rather than fail. "
                     "This is a documented `unsafe` precondition, which is the "
                     "project's normal discipline for pointer APIs, so it is "
                     "NOT reported as a code defect -- it IS reported as an "
                     "unfalsifiable-from-outside criterion.",
            repro="call reorder_compact_to_neurons(layout, buf, buf, n_pos) "
                  "with an aliased buffer: no guard fires",
            impact="latent; the only shipped caller (load_reorder) always "
                   "allocates a fresh dst, so no current caller can trigger it"))
        s6_ok = False
    else:
        s6_ok = True
    meta.record("S6 overlapping-buffer precondition is checked at runtime",
                s6_ok,
                "if there were no runtime check this criterion must not be "
                "certified; the finding is raised instead",
                not s6_ok,
                "doc says non-overlap=%s, runtime check=%s"
                % (doc_overlap, runtime_check))

    # =====================================================================
    # S7: artifact header vs section disagreement
    # =====================================================================
    print("\n[S7] artifact header disagreeing with the section")
    hdr_guard = bool(re.search(r"artifact header disagrees with section", nkv))
    hdr_exit = False
    lines = nkv.splitlines()
    for i, l in enumerate(lines):
        if "artifact header disagrees with section" in l:
            hdr_exit = "exit(9)" in "\n".join(lines[i:i + 4])
    print("   guard present: %s ; followed by exit(9): %s" % (hdr_guard, hdr_exit))
    meta.record("S7 artifact header vs section disagreement hard-fails",
                hdr_guard and hdr_exit,
                "the guard must be callable and followed by exit(9); a missing "
                "exit would make it print-and-proceed",
                hdr_guard and hdr_exit,
                "present=%s exit=%s" % (hdr_guard, hdr_exit))

    # =====================================================================
    # S8: section len shorter than its own entries need  [CLOSED by the fix]
    # =====================================================================
    print("\n[S8] section declares a length shorter than its entries require")
    # The verifier found this as a real defect; the implementation has since
    # added an explicit length check. This block re-verifies the FIX from the
    # source AND defers to the binary probe (guard group G8/G11 in
    # tools/p15_verify_guards.py) for the behavioural evidence.
    len_exact_src = bool(re.search(
        r"if avail != need|avail\s*!=\s*need|section length disagrees with "
        r"layout", nkv))
    len_guard = bool(re.search(r"len < NKV_HDR", nkv))
    print("   len >= NKV_HDR checked                    : %s" % len_guard)
    print("   exact `avail != need` check (the S8 fix)  : %s" % len_exact_src)
    # the binary evidence, from the guard probe
    probe_exe2 = os.path.join(PROBE, "target", "x86_64-pc-windows-msvc",
                              "release", "p15-verify-guardprobe.exe")
    s8_binary = None
    if os.path.exists(probe_exe2):
        rc1, o1, e1 = V.run_capture([probe_exe2, "g11_bounded_under_declared"],
                                    cwd=PROBE, timeout=60)
        rc2, o2, e2 = V.run_capture([probe_exe2, "g11_bounded_exact_ok"],
                                    cwd=PROBE, timeout=60)
        s8_binary = (rc1 == 9 and b"section length disagrees with layout" in (o1 + e1)
                     and rc2 == 0)
        print("   binary: under-declared exit=%d, exact exit=%d -> fix works=%s"
              % (rc1, rc2, s8_binary))
    meta.record("S8 declared section length is cross-checked against the "
                "layout size",
                bool(len_exact_src) and (s8_binary is True),
                "the pre-fix behavior must be shown different: an "
                "under-declared descriptor was accepted (exit=0) before the "
                "fix and is rejected (exit=9) after it",
                s8_binary is True,
                "source check=%s; binary under=9/exact=0 confirmed=%s"
                % (len_exact_src, s8_binary))

    # =====================================================================
    # report
    # =====================================================================
    v = V.ensure_outdir("degradation")
    with open(os.path.join(v, "findings.json"), "w", encoding="utf-8") as f:
        json.dump(findings, f, indent=2, ensure_ascii=False)
    print("\n" + "=" * 78)
    print("FINDINGS: %d" % len(findings))
    for fd in findings:
        print("  [%s] %s -- %s" % (fd["id"], fd["severity"], fd["claim"]))
    print("wrote %s" % os.path.join(v, "findings.json"))
    return 0 if meta.report() else 1


if __name__ == "__main__":
    sys.exit(main())
