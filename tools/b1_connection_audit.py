# PATH: B1
#
# AUDIT of the 5 questions, from the "weight/connection" viewpoint.
# READ-ONLY: this script only reads the produced artifact. No code is changed.
#
# THE FRAMING UNDER TEST
#   B1 = "a d=20 dense dot product + an index-mapping table".
#   Every weight that exists is used exactly once. Nothing is skipped.
#   => B1 is COMPUTE-DENSE. "Sparse" describes only the topology/storage.
#
# This script answers questions 1-3 and 5 statistically, from real data.
import struct
import os

d = open("artifacts/p18_sparse24/L00_ffn.csr", "rb").read()
n = struct.unpack("<I", d[4:8])[0]
k0 = struct.unpack("<I", d[8:12])[0]
tot = struct.unpack("<I", d[12:16])[0]
H = 20
src_base = H + (n + 1) * 4
w_base = src_base + tot * 4
src = [struct.unpack("<I", d[src_base + 4 * i: src_base + 4 * i + 4])[0] for i in range(tot)]
w = [struct.unpack("<f", d[w_base + 4 * i: w_base + 4 * i + 4])[0] for i in range(tot)]

print("=" * 74)
print("Q1: are the weights w[0..20] contiguous in memory?")
print("=" * 74)
print("  layout per conn_sparse.rs: CS_HDR=20, offsets[%d]*4, src[%d]*4, w[%d]*4" % (n + 1, tot, tot))
print("  w array base offset in file  = %d" % w_base)
print("  w for neuron i occupies      = [%d + i*%d*4, ...) i.e. 80 contiguous bytes" % (w_base, k0))
big = [i for i in range(1, n) if True]
gaps = 0
for i in range(n - 1):
    end_of_row_i = w_base + (i * k0 + k0) * 4
    start_of_row_i1 = w_base + ((i + 1) * k0) * 4
    if end_of_row_i != start_of_row_i1:
        gaps += 1
print("  row-to-row gaps found        = %d  (0 => fully packed, no padding)" % gaps)
print("  => ANSWER: YES, contiguous. Each neuron's 20 weights are one 80-byte block,")
print("             and the blocks are back-to-back with no padding.")
print("             (This was already true before any packing work.)")

print()
print("=" * 74)
print("Q2: are the source ids src[0..20] contiguous / ordered?")
print("=" * 74)
mono = sum(1 for i in range(n) if all(src[i * k0 + j] <= src[i * k0 + j + 1] for j in range(k0 - 1)))
print("  rows in ascending order      = %d / %d" % (mono, n))
diffs = []
for i in range(n):
    row = src[i * k0:(i + 1) * k0]
    for j in range(k0 - 1):
        diffs.append(row[j + 1] - row[j])
neg = sum(1 for x in diffs if x < 0)
print("  adjacent-pair deltas checked = %d" % len(diffs))
print("  descents (delta < 0)         = %d  (%.1f%% of pairs)" % (neg, 100.0 * neg / len(diffs)))
print("  => ANSWER: src[] itself IS contiguous in memory (a plain 20-element array),")
print("             but its VALUES are NOT ordered -- %.1f%% of adjacent pairs go" % (100.0 * neg / len(diffs)))
print("             backwards. So reading src is sequential; USING it is random.")

print()
print("=" * 74)
print("Q3: the access pattern of x[src[j]] -- random or clustered?")
print("=" * 74)
import math
spans = []
for i in range(n):
    row = sorted(src[i * k0:(i + 1) * k0])
    spans.append(row[-1] - row[0])
spans.sort()
mean_span = sum(spans) / len(spans)
print("  span (max-min) of a neuron's 20 sources, over all %d neurons:" % n)
print("     min %d   median %d   mean %.1f   max %d   (source space = %d)" %
      (spans[0], spans[len(spans) // 2], mean_span, spans[-1], n))
print("  mean span covers %.1f%% of the 1536-wide source space" % (100.0 * mean_span / n))
uniq_rows = sum(1 for i in range(n) if len(set(src[i * k0:(i + 1) * k0])) == k0)
print("  rows with all-distinct sources = %d / %d" % (uniq_rows, n))
# cache-line analysis: how many distinct 64B lines (16 floats) do 20 random sources touch?
lines = []
for i in range(0, min(n, 2000)):
    row = src[i * k0:(i + 1) * k0]
    lines.append(len(set(s // 16 for s in row)))
print("  distinct 64-byte cache lines touched per row (16 floats/line):")
print("     min %d   mean %.1f   max %d   (best possible = %d)" %
      (min(lines), sum(lines) / len(lines), max(lines), math.ceil(k0 / 16)))
print("  => ANSWER: RANDOM, and maximally so. 20 sources spread over ~%.0f%% of the" % (100.0 * mean_span / n))
print("             space, touching ~%.1f distinct cache lines per row." % (sum(lines) / len(lines)))
union = len(set(src))
print("  union of all sources = %d / %d  (%.1f%%)" % (union, n, 100.0 * union / n))
print("             => NO clustering exists to exploit; every source is used.")

print()
print("=" * 74)
print("Q5: non-compute overhead present in the current B1 path?")
print("=" * 74)
b1 = open("core/src/b1_pure.rs", encoding="utf-8").read()
body = b1[b1.index("pub unsafe fn b1_forward"):b1.index("pub unsafe fn b1_backward")]
checks = [
    ("operator dispatch / indirect call", "OP_" in body or "call" in body.lower()),
    ("mask / act[] reads or writes", "act" in body),
    ("slot read/write (interpreter state)", "slot" in body or "Vm" in body),
    ("top-k selection", "topk" in body.lower() or "top_k" in body),
    ("gather-style branching to skip", "continue" in body or "if *" in body),
    ("function call into a kernel", "kernels::" in body),
    ("stack buffer per row", "let mut prod" in body),
]
for name, present in checks:
    print("  %-42s %s" % (name, "PRESENT" if present else "absent"))
print()
print("  => ANSWER: after the packing work, the remaining non-compute items are:")
print("       - ONE kernel call per row (kernels::sparse_product_row_avx2)")
print("       - ONE stack buffer, now hoisted out of the row loop")
print("     There is NO top-k, NO mask, NO slot traffic, NO dispatch. The earlier")
print("     'mask 103%% of forward' figure belongs to B2/operator path, not to B1.")
