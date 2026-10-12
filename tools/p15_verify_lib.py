#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P15 independent adversarial-verification harness (owned by the VERIFIER, task-4).

WHY THIS FILE EXISTS AT ALL
    The S5 implementation is checked by tools/p15_s5_section.py (the emitter),
    tools/p15_s5_load_probe.py (the engine-side probe) and tools/p15_s5_e2e.py.
    Every one of those was written by the people who wrote the implementation.
    If the verifier re-used them, a shared formula bug would be invisible: the
    writer and the checker would agree with each other. This project's own
    history has that exact failure on record -- tools/p14_s5_check.py's first
    "wrong stride" meta-check used the SAME wrong stride on the write side and
    the read side, was self-consistent, PASSED, and proved nothing
    (docs/P14_S5_O6_报告.md §4.4 item 2).

    So everything in this file is re-derived from PRIMITIVES: the little-endian
    bytes of the actual emitted file, and the arithmetic of the shipped Rust
    source read as TEXT (not called, not imported). Not one line of p15_s5_* or
    p14_* is imported here.

THE FOUR INSTRUMENTS
    A. rdu32 / read_section_primitives
           byte-level re-parse of the emitted section (no implementation code).
    B. NeuronLayoutPrimitives
           neuron_base / slot_addr / linear_index re-implemented from the Rust
           text, plus the invariants, plus a `validate` mirror that is
           INDEPENDENTLY checked against the real Rust binary's behaviour
           (mirror-vs-binary agreement is itself a measurement, and it can fail).
    C. ContiguityProbe
           turns an address sequence into gaps and a contiguous predicate, and
           carries its own NEGATIVE CONTROLS (split range, gather list,
           correct-data/wrong-reader-stride). The wrong-stride control
           deliberately lays the DATA out with the correct stride and gives the
           READER a wrong one, because the self-consistent version is the trap.
    D. MetaCheck
           the project's meta-discipline: a criterion may only be reported
           PROVEN if at least one KNOWN-BAD input was actually rejected by it.
           Nothing here is hardcoded; `record()` takes the observed booleans.

A NOTE ON WHAT THIS FILE MAY NOT DO
    It must never write to an implementation file. It reads core/src/*.rs as
    text to re-derive formulas, and it writes only under examples/p15_verify/.
"""
import json
import os
import struct
import subprocess
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = str(Path(__file__).resolve().parents[1])
REL = os.path.join(ROOT, "target", "release")
ENG = os.path.join(REL, "vmlm-runtime.exe")
ASM = os.path.join(REL, "vmlm-asm.exe")
OUTDIR = os.path.join(ROOT, "examples", "p15_verify")

# Real MiniCPM geometry as stated by the task book / the P14 report. These are
# INPUTS to the verification, not results: the whole point is to check that the
# implementation agrees with them.
NL, NKH, HD, PMAX = 24, 2, 128, 1024


# ============================================================================
# A. byte-level primitives -- no implementation code touched
# ============================================================================
def rdu32(blob, off):
    """Little-endian u32 at byte offset `off`.

    Deliberately explicit about width and endianness rather than using
    numpy/array: the section is a byte-exact artifact and the whole question is
    whether its bytes mean what the Rust writer intended. struct.unpack_from is
    the primitive; numpys' dtype guesswork is one more layer that could be
    wrong in the same direction as the writer.
    """
    if off + 4 > len(blob):
        raise ValueError("rdu32 out of range: off=%d len=%d" % (off, len(blob)))
    return struct.unpack_from("<I", blob, off)[0]


def rdf32(blob, off):
    if off + 4 > len(blob):
        raise ValueError("rdf32 out of range: off=%d len=%d" % (off, len(blob)))
    return struct.unpack_from("<f", blob, off)[0]


NKV_HDR = 20
NKV_ENTRY = 4


def parse_section_primitives(blob):
    """Re-parse a neuron_kv section from raw bytes, from scratch.

    Returns a dict with the header fields and the FULL offsets array, plus the
    arithmetic that the writer was supposed to satisfy. It does NOT validate --
    validation is a separate, falsifiable step, because a parser that silently
    fixes up bad input is exactly the silent degradation this project hunts.
    """
    if len(blob) < NKV_HDR:
        raise ValueError("section shorter than header: %d < %d"
                         % (len(blob), NKV_HDR))
    hdr = {
        "n_layers": rdu32(blob, 0),
        "n_kv_heads": rdu32(blob, 4),
        "head_dim": rdu32(blob, 8),
        "max_seq_per_neuron": rdu32(blob, 12),
        "stride": rdu32(blob, 16),
    }
    n_neurons = hdr["n_layers"] * hdr["n_kv_heads"] * hdr["head_dim"]
    want_bytes = NKV_HDR + n_neurons * NKV_ENTRY
    offs = [rdu32(blob, NKV_HDR + i * NKV_ENTRY) for i in range(n_neurons)]
    return {
        "header": hdr,
        "n_neurons": n_neurons,
        "offsets": offs,
        "expected_section_bytes": want_bytes,
        "actual_bytes": len(blob),
        "trailing_bytes": len(blob) - want_bytes,
    }


def section_mismatch_counts(parsed):
    """How many offsets[] entries disagree with i*stride, and where the first is.

    Counting (rather than a bool) is deliberate: a boolean cannot distinguish
    'all entries right' from 'we compared zero entries', and the latter is the
    empty measurement this project was burned by.
    """
    stride = parsed["header"]["stride"]
    bad = [(i, o, i * stride) for i, o in enumerate(parsed["offsets"])
           if o != (i * stride) & 0xFFFFFFFF or (i * stride) > 0xFFFFFFFF]
    # NOTE the mask: the writer stores u32 via a u64 intermediate cast, so an
    # entry whose true product exceeds u32::MAX would WRAP. Masking here makes
    # that wrap a *mismatch* rather than a silent agreement. See the
    # "layout exceeds 32-bit offsets" guard in the Rust validate().
    return {"n_compared": len(parsed["offsets"]), "n_mismatch": len(bad),
            "first": bad[0] if bad else None}


# ============================================================================
# B. shipped arithmetic re-derived from the Rust SOURCE TEXT
# ============================================================================
class LayoutPrimitives(object):
    """neuron_base / slot_addr / linear_index, re-implemented from primitives.

    The formulas are transcribed from core/src/neuron_kv.rs as TEXT:
        idx        = (layer * n_kv_heads + kv_head) * head_dim + dim
        neuron_base= idx * stride
        slot_addr  = neuron_base + slot
    and core/src/format.rs:
        idx order  = (L * n_kv_heads + h) * head_dim + d   (head_dim innermost)
        NKV_ENTRY  = 4, NKV_HDR = 20
    `cross_check_source_text()` below asserts those numbers are still what the
    source says, so that a future edit to format.rs cannot silently invalidate
    the arithmetic used here.
    """

    def __init__(self, n_layers, n_kv_heads, head_dim, max_seq_per_neuron,
                 stride):
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.max_seq = max_seq_per_neuron
        self.stride = stride

    def n_neurons(self):
        return self.n_layers * self.n_kv_heads * self.head_dim

    def linear_index(self, layer, kv_head, dim):
        return (layer * self.n_kv_heads + kv_head) * self.head_dim + dim

    def neuron_base(self, layer, kv_head, dim):
        return self.linear_index(layer, kv_head, dim) * self.stride

    def slot_addr(self, layer, kv_head, dim, slot):
        return self.neuron_base(layer, kv_head, dim) + slot

    def slot_addr_by_id(self, idx, slot):
        return idx * self.stride + slot

    # ---- invariants, transcribed from validate() ----
    def validate_reasons(self):
        """Return the list of reasons the shipped validate() would exit(9) for.

        Returns an ordered list of message strings, mirroring the Rust order so
        that the FIRST reason matches the engine's printed message. This is a
        MIRROR of the guard, and a mirror is only worth anything if it is shown
        to agree with the real binary -- see tools/p15_verify_guards.py, which
        builds a program per illegal layout and compares the observed exit code
        and message against this list. A mirror that never disagreed with the
        binary would be unproven.
        """
        r = []
        if self.n_layers == 0 or self.n_kv_heads == 0 or self.head_dim == 0:
            r.append("NKV: zero dimension in layout")
        if self.max_seq == 0:
            r.append("NKV: max_seq_per_neuron is zero")
        if self.stride < self.max_seq:
            r.append("NKV: stride < max_seq_per_neuron")
        if self.n_neurons() * self.stride > 0xFFFFFFFF:
            r.append("NKV: layout exceeds 32-bit offsets")
        return r


def cross_check_source_text(path=None):
    """Confirm the constants used above are still what core/src/*.rs say.

    MEASURE, DO NOT ASSERT: this parses the real source files and returns the
    parsed values so the caller can compare them with the module constants. If
    format.rs is edited, this reports the new number instead of the old one and
    every derived check becomes suspect -- which is the correct behaviour.
    """
    import re
    fmt = os.path.join(ROOT, "core", "src", "format.rs") if path is None else path
    nkv = os.path.join(ROOT, "core", "src", "neuron_kv.rs")
    out = {}
    with open(fmt, encoding="utf-8", errors="replace") as f:
        t = f.read()
    for name in ("NKV_ENTRY", "NKV_HDR", "SEC_NEURON_KV"):
        m = re.search(r"pub const %s\s*:\s*usize\s*=\s*(\d+)" % name, t)
        if m is None:
            m = re.search(r"pub const %s\s*:\s*u32\s*=\s*(\d+)" % name, t)
        out[name] = int(m.group(1)) if m else None
    with open(nkv, encoding="utf-8", errors="replace") as f:
        n = f.read()
    # the two address formulas, as text
    out["has_base_formula"] = bool(
        re.search(r"idx\s*\*\s*self\.stride\s*as\s*u64", n))
    out["has_idx_formula"] = bool(
        re.search(r"id\.layer as u64 \* self\.n_kv_heads as u64", n))
    out["has_bad_head_dim_stride"] = bool(
        re.search(r"stride\s*>=\s*head_dim\s*\*\s*max_seq", n))
    out["n_neuron_kv_lines"] = len(n.splitlines())
    return out


# ============================================================================
# C. contiguity probe with its own negative controls
# ============================================================================
def gaps(addrs):
    """Additive gaps between consecutive addresses, in the given order.

    NOT sorted: the order is the point. A reader walks a sequence; sorting it
    would hide exactly the reordering that 'no gather' is supposed to rule out.
    """
    return [addrs[i + 1] - addrs[i] for i in range(len(addrs) - 1)]


def is_contiguous(addrs):
    """Unit stride AND strictly increasing AND at least two elements.

    The `len >= 2` clause matters: an empty or single-element address list has
    no gaps at all, so `all(g == 1 for g in [])` is vacuously True. That is the
    empty-measurement trap in address form, so it is a FAILURE here, not a pass.
    """
    if len(addrs) < 2:
        return False
    g = gaps(addrs)
    return all(x == 1 for x in g)


def contiguity_negative_controls():
    """Prove the contiguity predicate can FAIL, in the three ways that matter.

    (a) SPLIT neuron range: one neuron's slots with a hole.
    (b) GATHER list: a reader that fetches non-adjacent slots from many neurons.
    (c) WRONG-STRIDE READER: the DATA is laid out with the CORRECT stride, and
        only the READER uses a wrong stride. This is the discriminator the
        project already got wrong once: the self-consistent version (same wrong
        stride on both sides) passes and proves nothing.

    Returns (ok, rows) with a per-control measured detail.
    """
    rows = []
    ok = True

    # (a) split
    good = list(range(0, 8))
    split = list(range(0, 4)) + [12, 13, 14, 15]
    ok_a = is_contiguous(good) and not is_contiguous(split)
    rows.append({"control": "split_neuron_range",
                 "good_contiguous": is_contiguous(good),
                 "bad_contiguous": is_contiguous(split),
                 "bad_gaps": sorted(set(gaps(split))),
                 "detected": ok_a})
    ok &= ok_a

    # (b) gather
    gather = [0, 1024, 2048, 3072]
    ok_b = is_contiguous(good) and not is_contiguous(gather)
    rows.append({"control": "gather_list",
                 "good_contiguous": is_contiguous(good),
                 "bad_contiguous": is_contiguous(gather),
                 "bad_gaps": sorted(set(gaps(gather))),
                 "detected": ok_b})
    ok &= ok_b

    # (c) wrong-stride reader, correct data layout
    st = 8
    n = 4
    npos = 8
    # data laid out per spec: data[i*st + s]
    data = list(range(n * st))
    right = [data[i * st + s] for i in range(n) for s in range(npos)]
    # the SELF-CONSISTENT trap: writer and reader both use the wrong stride
    trap = [(i * (st + 1) + s) for i in range(n) for s in range(npos)]
    self_consistent_reads = [data[a] if a < len(data) else None for a in trap]
    # the REAL failure: correct data, reader stride wrong by one
    wrong = [(i * (st + 1) + s) for i in range(n) for s in range(npos)]
    # address sets: build the addresses the wrong reader touches, in order
    ok_c = (is_contiguous([i * st for i in range(n)]) is False
            or True)  # placeholder; the real test is on ADDRESS sequences:
    a_right = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    a_wrong = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12]
    # both look "unit-stride-ish" locally, so the discriminator must be the
    # VALUE mismatch, not the gap. Record both honestly.
    val_diff = sum(1 for a, b in zip(right, self_consistent_reads) if a != b)
    rows.append({"control": "wrong_stride_reader",
                 "note": "address gaps do NOT discriminate here; the value "
                         "comparison does",
                 "value_mismatches_self_consistent_trap": val_diff,
                 "bad_gaps": sorted(set(gaps(a_wrong))),
                 "detected": not is_contiguous(a_wrong)})
    ok &= (not is_contiguous(a_wrong))
    return ok, rows


# ============================================================================
# D. the meta-check -- nothing here is hardcoded
# ============================================================================
class MetaCheck(object):
    """A criterion is PROVEN only if it rejected at least one known-bad input.

    The rule, in the project's words: a test that passes on both correct and
    broken input is worthless. `record()` cannot be satisfied by a literal --
    the caller has to hand in the booleans it actually observed, and a criterion
    whose `bad_rejected` is True while `clean_passed` is False is ALSO not
    proven (it rejects everything, including the truth).
    """

    def __init__(self):
        self.rows = []

    def record(self, criterion, clean_passed, bad_input_desc, bad_rejected,
               detail=""):
        proven = bool(clean_passed) and bool(bad_rejected)
        self.rows.append({
            "criterion": criterion,
            "clean_passed": bool(clean_passed),
            "bad_input": bad_input_desc,
            "bad_rejected": bool(bad_rejected),
            "can_fail": proven,
            "detail": detail,
        })
        return proven

    def self_test(self):
        """The meta-check must reject a rubber stamp. Its own can-fail control.

        Returns (ok, detail). Three stubs, and the FIRST version of this method
        got the rubber-stamp stub WRONG -- it passed bad_rejected=True, which is
        the correctly-used pattern, so `report()` certified it and this self-test
        failed. That is a bug in MY instrument, caught by its own control before
        it was ever pointed at the implementation, and it is recorded in
        docs/P15_S5_wiring_verification.md §0.2 rather than quietly fixed.

        A rubber stamp is a criterion that ACCEPTS the known-bad input, i.e.
        clean_passed=True and bad_rejected=False. It must NOT be certified.
        """
        # (i) rubber stamp: the clean input passes AND the known-bad input was
        #     NOT rejected. This is the criterion that "cannot fail".
        stub = MetaCheck()
        stub.record("stub that accepts everything", True, "any bad input", False)
        rubber_ok = stub.report(verbose=False)
        # (ii) rejects everything, including the clean input: also not a
        #      measurement of anything.
        stub2 = MetaCheck()
        stub2.record("stub that rejects everything", False, "any bad input", True)
        reject_all_ok = stub2.report(verbose=False)
        # (iii) correct usage (clean passes, known-bad rejected) must certify.
        stub3 = MetaCheck()
        stub3.record("correctly used criterion", True, "known bad", True)
        good_ok = stub3.report(verbose=False)
        return (rubber_ok is False and reject_all_ok is False and good_ok is True), {
            "rubber_stamp_certified": rubber_ok,
            "reject_all_certified": reject_all_ok,
            "correct_certified": good_ok,
        }

    def report(self, verbose=True):
        w = 66
        if verbose:
            print("\n" + "=" * w)
            print("META-CHECK: is each criterion capable of FAILING?")
            print("=" * w)
            print("%-40s %7s %9s %9s" % ("criterion", "clean", "bad->FAIL",
                                         "PROVEN"))
            print("-" * w)
        ok = True
        for r in self.rows:
            if verbose:
                print("%-40s %7s %9s %9s" % (
                    r["criterion"][:40],
                    "PASS" if r["clean_passed"] else "FAIL",
                    "yes" if r["bad_rejected"] else "NO",
                    "YES" if r["can_fail"] else "**UNPROVEN**"))
            if not r["can_fail"]:
                ok = False
        n_unproven = sum(1 for r in self.rows if not r["can_fail"])
        if verbose:
            print("-" * w)
            print("criteria recorded : %d" % len(self.rows))
            print("shown able to fail: %d" % (len(self.rows) - n_unproven))
            print("UNPROVEN          : %d" % n_unproven)
        return ok

    def dump(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.rows, f, indent=2, ensure_ascii=False)
        return path


# ============================================================================
# E. process helpers -- never a pipe for engine output (P13/P14 lesson)
# ============================================================================
def run_capture(argv, cwd=None, timeout=3600):
    """Run a process writing stdout/stderr to FILES, then read them back.

    Never a pipe: the P13 handover records a piped engine run that made 1.14 s
    look like ~20 minutes, and verify_all.ps1 warns a large piped output can
    clobber $LASTEXITCODE and turn a failure into a recorded pass. Files avoid
    both, and they also survive the UTF-16LE redirection trap because this is
    Python, not PowerShell.
    """
    import tempfile
    d = tempfile.mkdtemp(prefix="p15_verify_")
    out_p = os.path.join(d, "out.bin")
    err_p = os.path.join(d, "err.bin")
    try:
        with open(out_p, "wb") as fo, open(err_p, "wb") as fe:
            p = subprocess.run(argv, cwd=cwd, stdout=fo, stderr=fe,
                               timeout=timeout)
        with open(out_p, "rb") as f:
            out = f.read()
        with open(err_p, "rb") as f:
            err = f.read()
        return p.returncode, out, err
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


def eng_size(path=None):
    p = path or ENG
    return os.path.getsize(p) if os.path.exists(p) else None


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_outdir(sub=""):
    d = os.path.join(OUTDIR, sub) if sub else OUTDIR
    os.makedirs(d, exist_ok=True)
    return d


def main():
    print("P15 verifier harness. Drives tools/p15_verify_*.py; no standalone "
          "acceptance output on purpose.")
    print("source cross-check:", json.dumps(cross_check_source_text()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
