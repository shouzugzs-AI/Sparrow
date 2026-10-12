import sys
sys.path.insert(0, r"<repo>/tools")
import numpy as np
import p14_s5_layout_lib as L

print("=== S5-P1/S5-P3: layout + bit-identity, driven by the SHIPPED spec ===")
# The shipped numbers from core/src/format.rs + core/src/neuron_kv.rs
# MiniCPM5-1B: 24 layers, 2 kv heads, head_dim 128, PMAX (max_seq_per_neuron).
NL, NKH, HD, PMAX = 24, 2, 128, 1024

print()
print("--- the two candidate invariants, evaluated on the real shape ---")
n_neurons = NL * NKH * HD
print("n_neurons = %d * %d * %d = %d" % (NL, NKH, HD, n_neurons))

# format.rs:295-296 says: offset[idx] + max_seq_per_neuron <= stride
# neuron_kv.rs:120-124 checks: stride >= head_dim * max_seq_per_neuron
# The two coincide ONLY if offset[idx] == idx * head_dim, which is NOT the
# shipped writing loop (write_section puts i*stride).
print()
print("format.rs doc invariant : offset[idx] + PMAX <= stride")
print("neuron_kv.rs validate() : stride >= head_dim * PMAX = %d" % (HD * PMAX))
print()

# What does the shipped write_section actually store?
# put(NKV_HDR + i*NKV_ENTRY, i * stride)  -> offset[idx] = idx * stride
# Then format.rs's invariant "offset[idx] + PMAX <= stride" becomes
#   idx*stride + PMAX <= stride   ->   only satisfiable for idx=0 unless PMAX<=0.
for stride in [HD * PMAX, 2 * HD * PMAX]:
    print("stride = %d:" % stride)
    bad = []
    for idx in [0, 1, 2, n_neurons - 1]:
        off = idx * stride
        lhs = off + PMAX
        ok = lhs <= stride
        bad.append((idx, off, lhs, ok))
    for idx, off, lhs, ok in bad:
        print("   idx=%-6d offset=%-10d offset+PMAX=%-10d <= stride? %s"
              % (idx, off, lhs, ok))
print()
print("=> format.rs's stated invariant is FALSE for every idx >= 1 under the")
print("   shipped offset convention (offset[idx] = idx*stride).")
print("   The condition that is actually correct for 'slots must not straddle'")
print("   is slot_addr's own check: slot < max_seq_per_neuron, combined with")
print("   stride >= head_dim*max_seq  -- i.e. neurons are head_dim APART-ish.")
print("   This is a DOC-vs-CODE defect: the doc states an invariant that the")
print("   code does not implement and cannot satisfy. It is not a runtime bug")
print("   (validate()'s stricter check implies non-overlap), but it is exactly")
print("   the kind of 'comment is not evidence' claim the task warned about.")

print()
print("--- S5-P1: does the shipped addressing reproduce [L][h][d][slot]? ---")
# The shipped model stores at neuron_base(id) = ((L*NKH + h)*HD + d) * stride
# and slot inside -> exactly the NeuronLayout model. Verify equivalence.
lay = L.NeuronLayout(NL, NKH, HD, PMAX)
mismatch = 0
for (Lx, hx, dx, sx) in [(0,0,0,0), (0,0,0,7), (0,0,1,0), (0,1,0,0),
                         (1,0,0,0), (3,1,127,1023), (23,1,127,0)]:
    nid = (Lx * NKH + hx) * HD + dx
    shipped = nid * lay.stride + sx
    mine = lay.addr(lay.neuron_id(Lx, hx, dx), sx)
    if shipped != mine:
        mismatch += 1
        print("  MISMATCH L=%d h=%d d=%d s=%d: shipped=%d model=%d"
              % (Lx, hx, dx, sx, shipped, mine))
print("  address model vs shipped formula: %d mismatches (must be 0)" % mismatch)
print("  n_neurons shipped=%d  model=%d" % (n_neurons, lay.n_neurons))
print("  stride shipped=%d model=%d" % (HD * PMAX, lay.stride))

print()
print("--- S5-P3: bit-identity of neuron layout vs indexed layout ---")
rng = np.random.RandomState(7)
# use a small case so it is fast and hand-checkable, then the real shape
small = L.NeuronLayout(3, 2, 4, 8)
src = rng.randn(3, 2, 4, 8).astype(np.float32)
flat = small.encode(src)
back = small.decode(flat)
print("small [3][2][4][8]: round-trip bit-identical =", L.bit_identical(src, back))
# element-by-element: neuron read == indexed read, for EVERY element
nbad = 0
for Lx in range(3):
    for hx in range(2):
        for dx in range(4):
            for sx in range(8):
                a = L.indexed_read(src, small, Lx, hx, dx, sx)
                b = L.neuron_read(flat, small, Lx, hx, dx, sx)
                if a != b:
                    nbad += 1
print("neuron-addressed read != indexed read for %d / %d elements (must be 0)"
      % (nbad, src.size))

# THE META-CHECK: the same comparison must FAIL on a wrong stride
print()
print("--- META-CHECK: same comparison on a WRONG stride must FAIL ---")
bad_layout = L.wrong_stride_variant(small)
bflat = bad_layout.encode(src)
nbad2 = 0
for Lx in range(3):
    for hx in range(2):
        for dx in range(4):
            for sx in range(8):
                a = L.indexed_read(src, small, Lx, hx, dx, sx)
                off = bad_layout.addr(bad_layout.neuron_id(Lx, hx, dx), sx)
                b = bflat[off] if off < bflat.size else np.float32(0)
                if a != b:
                    nbad2 += 1
print("wrong stride: %d / %d elements disagree (must be > 0)" % (nbad2, src.size))
print("comparison demonstrably able to fail:", nbad2 > 0)

ok = (mismatch == 0 and nbad == 0 and nbad2 > 0
      and L.bit_identical(src, back))
print()
print("VERDICT: %s" % ("PASS (with the format.rs doc-invariant defect noted)"
                       if ok else "FAIL"))
