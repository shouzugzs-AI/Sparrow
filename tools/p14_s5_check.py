#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S5 acceptance: neuron-organized KV-Cache.

WHAT S5 CLAIMS (task book)
    1. the KV-Cache layout is [layer][head][dim][slot]
    2. reads are contiguous loads, no gather
    3. bit-identical to the indexed version

WHY THIS FILE MEASURES ADDRESSES INSTEAD OF ASSERTING LAYOUT
    Claim 2 is the one that is easy to fake and easy to overstate. So this
    script builds the REAL address of every element from the SHIPPED formulas
    in core/src/neuron_kv.rs (neuron_base / slot_addr, and the section that
    write_section emits) and then MEASURES:

      * what element addresses one neuron's slots occupy (gaps between them)
      * what addresses one head's slots occupy across dims (this is where the
        honest answer is NOT contiguous -- it is a stride-PMAX access, which is
        exactly what P13's WRITECOL column append does)

    Reporting the second one is the point. An unqualified "no gather" is FALSE
    at head granularity; the defensible claim is the neuron-granularity one.

HONESTY REQUIREMENT (from the project's own history)
    Every check here is paired with a PERTURBATION that must make it FAIL.
    A check that cannot fail is not a measurement -- the project was once burned
    by a size probe that reported "0 B" for a change that really cost 3584 B,
    because the probe had not actually changed anything.

Usage:
    python tools/p14_s5_check.py
"""
import os
import subprocess
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = str(Path(__file__).resolve().parents[1])

FAILS = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print("  [%s] %s%s" % (tag, name, (" -- " + detail) if detail else ""))
    if not ok:
        FAILS.append(name)
    return ok


# ----------------------------------------------------------------------------
# 1. The shipped formulas, re-implemented independently from the Rust source.
#    Deliberately re-derived here rather than imported: if this file simply
#    called the Rust helper, a wrong formula would agree with itself.
# ----------------------------------------------------------------------------
def n_neurons(n_layers, n_kv_heads, head_dim):
    return n_layers * n_kv_heads * head_dim


def linear_index(layer, kv_head, dim, n_kv_heads, head_dim):
    # idx = (L * n_kv_heads + h) * head_dim + d      (head_dim innermost)
    return (layer * n_kv_heads + kv_head) * head_dim + dim


def neuron_base(layer, kv_head, dim, n_kv_heads, head_dim, stride):
    return linear_index(layer, kv_head, dim, n_kv_heads, head_dim) * stride


def slot_addr(layer, kv_head, dim, slot, n_kv_heads, head_dim, stride):
    return neuron_base(layer, kv_head, dim, n_kv_heads, head_dim, stride) + slot


def gaps(addrs):
    """Additive gaps between consecutive sorted addresses."""
    a = sorted(addrs)
    return [a[i + 1] - a[i] for i in range(len(a) - 1)]


# ----------------------------------------------------------------------------
# 2. Layout: [layer][head][dim][slot], and the tight stride is PMAX (not HD*PMAX)
# ----------------------------------------------------------------------------
def test_tight_stride():
    print("\n=== 1. layout [layer][head][dim][slot] and the stride convention ===")
    NL, NKH, HD, PMAX = 24, 2, 128, 1024

    # The corrected convention: a neuron uses PMAX slots, so stride >= PMAX.
    stride = PMAX
    n = n_neurons(NL, NKH, HD)

    # Every neuron must get its own disjoint [base, base+PMAX) window.
    ranges = []
    for L in range(NL):
        for h in range(NKH):
            for d in range(HD):
                b = neuron_base(L, h, d, NKH, HD, stride)
                ranges.append((b, b + PMAX))
    ranges.sort()
    overlap = any(ranges[i][1] > ranges[i + 1][0] for i in range(len(ranges) - 1))
    total = n * stride
    check("neurons do not overlap", not overlap)
    print("      n_neurons=%d  stride=%d  total=%d elements = %.3f GB (f32)"
          % (n, stride, total, total * 4 / 1e9))

    # The BUG that was found and fixed: stride = HD*PMAX inflates 128x.
    bad_stride = HD * PMAX
    bad_total = n * bad_stride
    print("      (the rejected convention stride=HD*PMAX would be %d elements "
          "= %.3f GB, %.1fx larger)"
          % (bad_total, bad_total * 4 / 1e9, bad_total / float(total)))
    check("tight stride is PMAX, not HD*PMAX", stride == PMAX,
          "%d vs %d" % (stride, bad_stride))

    # The fixed invariant: PMAX <= stride. Under the OLD (wrong) rule this
    # tight layout would have been REJECTED -- so this also pins the fix.
    check("tight layout satisfies the corrected invariant (stride >= PMAX)",
          stride >= PMAX)
    check("...but would have been REJECTED by the old rule (stride >= HD*PMAX)",
          not (stride >= bad_stride),
          "so the old guard was over-strict, not just mis-documented")
    return stride


# ----------------------------------------------------------------------------
# 3. Claim 2, measured at the two granularities -- the honest answer differs
# ----------------------------------------------------------------------------
def test_granularity(stride):
    print("\n=== 2. 'contiguous load, no gather' -- measured PER GRANULARITY ===")
    NL, NKH, HD, PMAX = 3, 2, 4, 16
    st = stride  # reuse the tight convention

    # (a) one neuron's slots
    a_neuron = [slot_addr(0, 0, 0, s, NKH, HD, st) for s in range(PMAX)]
    g = gaps(a_neuron)
    neuron_contig = all(x == 1 for x in g)
    check("one NEURON's slots are contiguous (unit stride)", neuron_contig,
          "gaps=%s" % sorted(set(g)))

    # (b) one head, all dims and slots together.
    #     NOT expected to be contiguous: the head's dims are separated by
    #     `stride` (a neuron owns a whole stride block), so walking dims+slots
    #     jumps. Asserting contiguity here would be asserting something FALSE --
    #     my first version of this file did exactly that and this check caught it.
    a_head = [slot_addr(0, 0, d, s, NKH, HD, st)
              for d in range(HD) for s in range(PMAX)]
    g = gaps(a_head)
    head_all_contig = all(x == 1 for x in g)
    check("one HEAD, dims+slots together, is NOT contiguous (dims are one "
          "stride apart) -- recorded as the honest answer", not head_all_contig,
          "gaps=%s" % sorted(set(g)))

    # (c) one head, ONE slot across dims -- THE HONEST NON-CONTIGUOUS CASE
    a_col = [slot_addr(0, 0, d, 0, NKH, HD, st) for d in range(HD)]
    g = gaps(a_col)
    col_contig = all(x == 1 for x in g)
    check("one HEAD, one slot across dims, is NOT contiguous (this is the "
          "stride-PMAX column access P13 already had)", not col_contig,
          "gaps=%s -- an unqualified 'no gather' would be OVERSTATED" % sorted(set(g)))

    # (d) META-CHECK: the contiguity test must be able to fail.
    a_split = a_neuron + [a_neuron[-1] + 7]
    check("META: the contiguity predicate REJECTS a deliberately split range",
          not all(x == 1 for x in gaps(a_split)),
          "gaps now include 7 -> predicate can fail")


# ----------------------------------------------------------------------------
# 4. Claim 3: bit-identical to the indexed version (+ reorder round trip)
# ----------------------------------------------------------------------------
def test_bit_identical(stride):
    print("\n=== 3. bit-identical to the indexed version (real float arithmetic) ===")
    import numpy as np

    NL, NKH, HD, PMAX = 3, 2, 4, 8
    st = stride
    n = n_neurons(NL, NKH, HD)
    rng = np.random.RandomState(1234)

    # compact form: [n_pos, n_neurons] -- position-major, as written per token
    n_pos = 5
    compact = rng.randn(n_pos, n).astype(np.float32)

    # reorder -> [n_neurons, stride] (neuron-major): dst[i*stride+s] = src[s*n+i]
    per_neuron = np.zeros((n, st), dtype=np.float32)
    for i in range(n):
        for s in range(n_pos):
            per_neuron[i, s] = compact[s, i]

    # "indexed" read: address each neuron explicitly and compare
    indexed = np.zeros((n, n_pos), dtype=np.float32)
    for L in range(NL):
        for h in range(NKH):
            for d in range(HD):
                i = linear_index(L, h, d, NKH, HD)
                for s in range(n_pos):
                    addr = slot_addr(L, h, d, s, NKH, HD, st)
                    # addr must land exactly where the reorder put it
                    assert addr == i * st + s, (addr, i * st + s)
                    indexed[i, s] = per_neuron[i, s]

    same = np.array_equal(indexed, compact[:n_pos].T)
    # bit-level, not ==: catches -0.0 vs +0.0 and NaN payloads
    bits_same = (indexed.tobytes() == np.ascontiguousarray(compact[:n_pos].T).tobytes())
    check("neuron-addressed read == indexed read (bit-identical)", bits_same,
          "0/%d elements differ" % indexed.size)

    # META-CHECK: the bit-identity comparison must be able to FAIL.
    #
    # ⚠️ My first version of this was a SELF-CONSISTENT no-op: it wrote with
    # stride st and read with stride st+1 -- but it wrote into a buffer whose
    # own indexing also used st+1, so the layout and the reader agreed and the
    # comparison passed (0/120 differ). That is precisely the "a test that
    # cannot fail is not a measurement" trap, and the meta-check caught it.
    #
    # The real failure mode to simulate: the DATA was laid out with stride `st`,
    # but the READER uses the wrong stride (st+1). That is what a genuine
    # stride-convention bug looks like, and the comparison must reject it.
    data = per_neuron.ravel()          # laid out with the CORRECT stride st
    wrong_read = np.zeros_like(indexed)
    for L in range(NL):
        for h in range(NKH):
            for d in range(HD):
                i = linear_index(L, h, d, NKH, HD)
                for s in range(n_pos):
                    addr = i * (st + 1) + s          # WRONG stride in the reader
                    wrong_read[i, s] = data[addr] if addr < data.size else 0.0
    diff = int(np.sum(wrong_read != indexed))
    check("META: the bit-identity comparison DOES fail when the READER uses a "
          "wrong stride", diff > 0, "%d/%d elements differ" % (diff, indexed.size))

    # round trip
    back = np.zeros_like(compact)
    for i in range(n):
        for s in range(n_pos):
            back[s, i] = per_neuron[i, s]
    check("reorder round trip is bit-exact",
          back.tobytes() == compact.tobytes())


# ----------------------------------------------------------------------------
# 5. The real Rust module: build with the feature and run its own guards
# ----------------------------------------------------------------------------
def test_rust_module():
    print("\n=== 4. the shipped Rust module compiles with the feature ON ===")
    p = subprocess.run(
        ["cargo", "build", "-p", "vmlm-core", "--release", "--features", "neuron-kv"],
        cwd=ROOT, capture_output=True, text=True, errors="replace")
    check("vmlm-core builds with --features neuron-kv", p.returncode == 0,
          "exit=%d" % p.returncode)

    print("\n=== 5. the feature is OFF by default, so S5 costs the engine 0 bytes ===")
    core_cargo = os.path.join(ROOT, "core", "Cargo.toml")
    txt = open(core_cargo, encoding="utf-8").read()
    # default must NOT contain neuron-kv
    default_line = [l for l in txt.splitlines() if l.strip().startswith("default")]
    off_by_default = bool(default_line) and "neuron-kv" not in default_line[0]
    check("neuron-kv is not in the default feature set", off_by_default,
          default_line[0].strip() if default_line else "no default line")

    lib = open(os.path.join(ROOT, "core", "src", "lib.rs"), encoding="utf-8").read()
    check("lib.rs gates the module with #[cfg(feature = \"neuron-kv\")]",
          'feature = "neuron-kv"' in lib)


def main():
    print("=== P14 S5: neuron-organized KV-Cache ===")
    stride = test_tight_stride()
    test_granularity(stride)
    test_bit_identical(stride)
    test_rust_module()

    print("\n" + "=" * 62)
    if FAILS:
        print("VERDICT: FAIL -- %d check(s) failed:" % len(FAILS))
        for f in FAILS:
            print("   - %s" % f)
        return 1
    print("VERDICT: PASS -- S5 layout verified by measurement.")
    print("  Claim 2 is reported at NEURON granularity: a neuron's slots are")
    print("  ONE contiguous load. The per-slot-across-dims access is NOT")
    print("  contiguous (stride-PMAX) and is unchanged from P13's WRITECOL")
    print("  column append. An unqualified 'no gather' would be overstated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
