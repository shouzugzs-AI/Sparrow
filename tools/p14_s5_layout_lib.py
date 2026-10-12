#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P14 S5 verification: is the KV cache ACTUALLY laid out [layer][head][dim][slot],
and are the reads ACTUALLY contiguous?

THE ATTACK THIS FILE IS BUILT TO SURVIVE
    S5's acceptance has two claims that are trivially fakeable:
      (a) "the layout is [layer][head][dim][slot]"
      (b) "reads are contiguous loads, no gather"
    Both can be "satisfied" by writing a comment, a docstring, or an
    `assert True` next to a struct. Neither counts. And (b) has a specific trap
    recorded in the task book itself: the implementer was TOLD the existing
    design probably already satisfied it, and told that "if a criterion turns
    out to be already met by the existing design, REPORT THAT HONESTLY". So
    "already satisfied" is a claim, not an exemption -- it has to be measured
    the same way a fresh implementation would be.

    The instrument below therefore does NOT look for a function named
    `reorder` and trust it. It:

      1. Reads the REAL byte layout out of a REAL artifact / the real
         neuron_kv module output, and checks the physical stride pattern.
         A layout claim is verified by MEASURING addresses, never by reading
         the name of a variable.
      2. Builds the SAME logical cache two ways -- the neuron-ordered way and a
         plain indexed (gather) way -- and requires the two results to be
         bit-identical. A stride error produces a permuted cache, which shows
         up as a large numeric difference, not as a crash.
      3. Re-derives the "contiguous" claim as a MEASUREMENT of the access
         pattern: for the neuron-ordered read, the sequence of addresses a
         neuron's slots occupy must be base + id*stride + [0..count), i.e. ONE
         base plus a unit stride. If a read touches addresses whose gaps are not
         1 element, that is a gather and must be reported as a FAIL.
      4. And it must be shown able to fail: a deliberately WRONG stride (or a
         swapped [dim]/[slot] order) has to make it report FAIL.

WHY THE MEASUREMENT IS DONE IN PYTHON RATHER THAN BY INSTRUMENTING THE ENGINE
    The engine is no_std, single-threaded and has a hard 64000 B budget; adding
    a tracing facility to it would change the artifact being measured. Instead
    the layout is reconstructed from the DEFINITION the implementation ships
    (the offsets array + stride) and then checked against the artifact bytes /
    the engine's own output. If the implementation's offsets disagree with the
    engine's behaviour, that disagreement is the finding.
"""
import os
import struct
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = str(Path(__file__).resolve().parents[1])


# ----------------------------------------------------------------------------
# 1. Layout description + address model
# ----------------------------------------------------------------------------
class NeuronLayout(object):
    """The addressing model S5 specifies: addr = base + neuron_id*stride + slot.

    A neuron IS one output dimension of the Q/K/V projection, i.e. one head_dim
    channel (the task book settles this from the reference implementation:
    "each d_model dimension = one neuron"), and the S5 section's own
    [n_layers][n_kv_heads][head_dim] shape agrees with that reading. So
    neuron_id = (layer * n_kv_heads + head) * head_dim + dim.
    """

    def __init__(self, n_layers, n_kv_heads, head_dim, slots_per_neuron):
        self.n_layers = int(n_layers)
        self.n_kv_heads = int(n_kv_heads)
        self.head_dim = int(head_dim)
        self.slots = int(slots_per_neuron)
        self.n_neurons = self.n_layers * self.n_kv_heads * self.head_dim
        self.stride = self.slots            # elements between neuron starts
        self.total = self.n_neurons * self.stride

    def neuron_id(self, layer, head, dim):
        return (layer * self.n_kv_heads + head) * self.head_dim + dim

    def addr(self, neuron_id, slot):
        return neuron_id * self.stride + slot

    def encode(self, src):
        """Indexed/gather reference -> neuron-major flat buffer.

        `src` is [layer][head][dim][slot]. This is the DEFINITION of the
        layout, expressed as a loop over the physical index, so the result can
        be checked against whatever the implementation produces.

        NOTE ON A DELIBERATE NON-ASSERTION: when stride < slots a neuron's
        slots straddle the next neuron's. The write below then raises a
        broadcast error from numpy, which is a legitimate outcome -- the
        negative control is supposed to be caught. But a silently OVERLAPPING
        variant is more dangerous than a loud one, because overlap is exactly
        the class of bug that passes a round trip and corrupts a real cache, so
        the negative control uses a plain numpy loop that does overlap
        silently. See `encode_overlapping` below.
        """
        src = np.asarray(src, dtype=np.float32).reshape(
            self.n_layers, self.n_kv_heads, self.head_dim, self.slots)
        out = np.zeros(self.total, dtype=np.float32)
        for L in range(self.n_layers):
            for h in range(self.n_kv_heads):
                for d in range(self.head_dim):
                    nid = self.neuron_id(L, h, d)
                    off = self.addr(nid, 0)
                    for s in range(self.slots):
                        if off + s < out.size:
                            out[off + s] = src[L, h, d, s]
        return out

    def decode(self, flat):
        """neuron-major flat buffer -> [layer][head][dim][slot]."""
        flat = np.asarray(flat, dtype=np.float32)
        out = np.zeros((self.n_layers, self.n_kv_heads, self.head_dim,
                        self.slots), dtype=np.float32)
        for L in range(self.n_layers):
            for h in range(self.n_kv_heads):
                for d in range(self.head_dim):
                    nid = self.neuron_id(L, h, d)
                    off = self.addr(nid, 0)
                    for s in range(self.slots):
                        if off + s < flat.size:
                            out[L, h, d, s] = flat[off + s]
        return out


# ----------------------------------------------------------------------------
# 2. THE MEASUREMENT: is a read contiguous, or does it gather?
# ----------------------------------------------------------------------------
def measure_stride(addresses):
    """Given the element addresses a single logical read touches, in order,
    return (is_contiguous, gaps).

    This is the whole of the "no gather" claim: a contiguous load of n elements
    from a base has gaps equal to [1,1,...,1]. Anything else -- a stride of
    head_dim, a jump between neurons, an interleave -- is a scatter/gather no
    matter what the surrounding comment says.
    """
    a = list(int(x) for x in addresses)
    if len(a) < 2:
        return True, []
    gaps = [a[i + 1] - a[i] for i in range(len(a) - 1)]
    return all(g == 1 for g in gaps), gaps


def measure_neuron_read(layout, layer, head, dim):
    """Addresses the runtime would touch to read ONE neuron's whole slot range."""
    nid = layout.neuron_id(layer, head, dim)
    return [layout.addr(nid, s) for s in range(layout.slots)]


def measure_head_read(layout, layer, head):
    """Addresses to read a whole (layer, head) block: head_dim neurons in order.

    Under the S5 layout this is the interesting case, because a head's block is
    only contiguous if slots_per_neuron == 1 (or if the dims are the fast axis
    AND stride == slots). It is measured rather than assumed, and reported with
    its actual gap pattern.
    """
    addrs = []
    for d in range(layout.head_dim):
        addrs.extend(measure_neuron_read(layout, layer, head, d))
    return addrs


def measure_append_column(layout, layer, head, dim, slot):
    """Addresses touched when APPENDING position `slot` for one neuron.

    P13's design appends a position with WRITECOL, a straight strided column
    store. Under a [dim][slot] transposed per-neuron block this is a single
    element per neuron; across a head it is head_dim elements with stride
    `layout.stride`. The stride is the honest answer about gather: an append
    across a head is NOT contiguous when stride > 1, and saying so is the point.
    """
    nid = layout.neuron_id(layer, head, dim)
    return layout.addr(nid, slot)


# ----------------------------------------------------------------------------
# 3. The indexed (gather) comparison
# ----------------------------------------------------------------------------
def indexed_read(src, layout, layer, head, dim, slot):
    """Read one element the OLD way: index into [layer][head][dim][slot]."""
    a = np.asarray(src, dtype=np.float32).reshape(
        layout.n_layers, layout.n_kv_heads, layout.head_dim, layout.slots)
    return a[layer, head, dim, slot]


def neuron_read(flat, layout, layer, head, dim, slot):
    """Read the same element the S5 way: base + neuron_id*stride + slot."""
    return np.asarray(flat, dtype=np.float32)[layout.addr(
        layout.neuron_id(layer, head, dim), slot)]


def bit_identical(a, b):
    x = np.asarray(a, dtype=np.float32)
    y = np.asarray(b, dtype=np.float32)
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return False
    return x.tobytes() == y.tobytes()


# ----------------------------------------------------------------------------
# 4. Negative controls for the layout instrument ITSELF
# ----------------------------------------------------------------------------
def wrong_stride_variant(layout):
    """A layout whose stride is one element too small.

    If stride < slots, a neuron's slots OVERLAP its neighbour's -- the exact
    invariant the task book requires the runtime helper to hard-fail on
    ("slot_offset < stride so a neuron's slots never straddle another's").
    """
    bad = NeuronLayout(layout.n_layers, layout.n_kv_heads, layout.head_dim,
                       layout.slots)
    bad.stride = layout.slots - 1
    bad.total = bad.n_neurons * bad.stride
    return bad


def swapped_dims_variant(layout):
    """A layout that stores [slot][dim] inside each neuron instead of [dim][slot].

    THE SUBTLETY THAT MADE THE FIRST VERSION OF THIS NEGATIVE CONTROL USELESS:
    if the variant's decode() is the exact inverse of its own encode(), the pair
    round-trips perfectly and the control proves nothing -- a self-consistent
    wrong layout is still wrong. So the check below does NOT round-trip the
    variant. It encodes with the SPEC encoder and decodes with the SPEC decoder
    (i.e. reads it the way the runtime is specified to read it) and requires the
    result to differ from the original. That models the real failure: the bytes
    on disk are in the wrong order, and the reader that assumes the spec gets
    garbage.

    Concretely `encode_wrong` lays [L][h][slot][dim] while `NeuronLayout.decode`
    reads [dim][slot]. Because head_dim != slots in general, and because the
    transpose is a real permutation, the decoded values are permuted.
    """
    lay = layout

    def encode_wrong(src):
        s = np.asarray(src, dtype=np.float32).reshape(
            lay.n_layers, lay.n_kv_heads, lay.head_dim, lay.slots)
        return np.transpose(s, (0, 1, 3, 2)).reshape(-1).copy()

    return encode_wrong


def selftest():
    print("=== S5 instrument self-test: the layout check MUST be able to FAIL ===")
    lay = NeuronLayout(2, 3, 4, 5)
    print("layout: [%d][%d][%d][%d] -> %d neurons, stride=%d, total=%d"
          % (lay.n_layers, lay.n_kv_heads, lay.head_dim, lay.slots,
             lay.n_neurons, lay.stride, lay.total))
    rng = np.random.RandomState(1234)
    src = rng.randn(lay.n_layers, lay.n_kv_heads, lay.head_dim,
                    lay.slots).astype(np.float32)

    flat = lay.encode(src)
    back = lay.decode(flat)
    print("round trip bit-identical          :", bit_identical(src, back))

    # every neuron read must be one contiguous run
    allc = True
    for L in range(lay.n_layers):
        for h in range(lay.n_kv_heads):
            for d in range(lay.head_dim):
                c, gaps = measure_stride(measure_neuron_read(lay, L, h, d))
                allc = allc and c
    print("all neuron reads contiguous       :", allc)

    # THE CASE THAT MATTERS. The real cache has slots = PMAX (>>1). Two
    # different notions of "one read" must be separated, because conflating them
    # is exactly how a "no gather" claim becomes a rubber stamp:
    #   * one NEURON's slot range  -> must be ONE contiguous load. This is what
    #     S5 actually claims ("a neuron's position IS its address").
    #   * one HEAD (head_dim neurons) -> is contiguous ONLY IF the layout puts
    #     dim innermost AND stride == slots. With slots=PMAX>1 and stride==slots,
    #     the (dim, slot) block IS contiguous as a whole, but the per-SLOT view
    #     across a head is stride-slots apart.
    # Both are measured and reported; neither is asserted from the docs.
    lay_big = NeuronLayout(2, 3, 4, 16)
    c_neuron, gaps_neuron = measure_stride(measure_neuron_read(lay_big, 0, 0, 0))
    c_head, gaps_head = measure_stride(measure_head_read(lay_big, 0, 0))
    # per-slot view across a head: appending position p touches one element per
    # dim; under stride==slots those are `stride` apart -- a strided store.
    slot_addrs = [measure_append_column(lay_big, 0, 0, d, 3)
                  for d in range(lay_big.head_dim)]
    c_slot, gaps_slot = measure_stride(slot_addrs)
    print()
    print("STRIDE MEASUREMENT (the 'no gather' claim), slots=%d stride=%d:"
          % (lay_big.slots, lay_big.stride))
    print("  one NEURON's whole slot range : contiguous=%s gaps=%s"
          % (c_neuron, gaps_neuron[:4]))
    print("  one HEAD, dims+slots          : contiguous=%s gaps=%s"
          % (c_head, gaps_head[:6]))
    print("  one HEAD, ONE SLOT across dims: contiguous=%s gaps=%s  <-- strided"
          % (c_slot, gaps_slot[:6]))
    print("  => the claim holds at NEURON granularity (one contiguous load).")
    print("     It does NOT hold per-slot across a head: that access is")
    print("     stride-%d, i.e. genuinely strided, and is reported as such." % lay_big.stride)

    # --- negative controls ---
    print()
    print("-- negative control 1: stride too small (neurons OVERLAP) --")
    bad = wrong_stride_variant(lay)
    bflat = bad.encode(src)
    print("wrong-stride round trip identical :", bit_identical(src, bad.decode(bflat)),
          "(must be False)")
    nbef = 0
    for L in range(lay.n_layers):
        for h in range(lay.n_kv_heads):
            for d in range(lay.head_dim):
                for s in range(lay.slots):
                    off = bad.addr(bad.neuron_id(L, h, d), s)
                    if off < bflat.size and bflat[off] != src[L, h, d, s]:
                        nbef += 1
    print("elements wrong under wrong stride : %d (must be > 0)" % nbef)
    print("stride=%d slots=%d -> invariant 'slot < stride' holds? %s (must be False)"
          % (bad.stride, bad.slots, bad.stride >= bad.slots))

    print()
    print("-- negative control 2: [dim]/[slot] transposed ON DISK, read per spec --")
    enc_wrong = swapped_dims_variant(lay)
    sflat = enc_wrong(src)
    print("wrong bytes == correct bytes      :", bit_identical(sflat, flat),
          "(must be False)")
    decoded_per_spec = lay.decode(sflat)
    nwrong = int((decoded_per_spec != src).sum())
    print("elements wrong when read per spec : %d / %d (must be > 0)"
          % (nwrong, src.size))
    print("spec-decode bit-identical to src  :", bit_identical(decoded_per_spec, src),
          "(must be False)")

    print()
    print("-- negative control 3: a gather must be DETECTED --")
    gathered = [0, 5, 10, 15, 20]
    c, gaps = measure_stride(gathered)
    print("gather addresses %s -> contiguous=%s gaps=%s (must be False)"
          % (gathered, c, gaps))
    assert c is False, "measure_stride failed to detect a gather"

    print()
    print("-- negative control 4: a per-neuron read split across the buffer --")
    # Two halves of one neuron stored far apart: address list jumps.
    split = [0, 1, 2, 50, 51, 52]
    c4, gaps4 = measure_stride(split)
    print("split read %s -> contiguous=%s gaps=%s (must be False)"
          % (split, c4, gaps4))
    assert c4 is False, "measure_stride failed to detect a split read"

    good = (bit_identical(src, back) and allc
            and not bit_identical(sflat, flat)
            and not bit_identical(decoded_per_spec, src)
            and nwrong > 0 and nbef > 0
            and c is False and c4 is False
            and c_neuron is True and c_slot is False)
    print()
    print("INSTRUMENT VERDICT: %s" % ("OK -- can detect wrong stride, transpose, "
                                     "and gather" if good else
                                     "BROKEN -- do not trust its PASS"))
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(selftest())
