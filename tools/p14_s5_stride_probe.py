"""
S5 follow-up: characterise the SHIPPED stride convention precisely.

The shipped layout reports stride = head_dim * max_seq_per_neuron = 131072 for
MiniCPM (24x2x128 neurons, PMAX=1024). My NeuronLayout model used stride =
slots_per_neuron = 1024. Both give the SAME neuron_id -> base mapping scale-
wise, but they mean different things and the difference matters for the
    "offset[idx] + max_seq_per_neuron <= stride"
invariant that format.rs states.

Establish, by arithmetic on the shipped formulas only:
  1. what neuron_base() actually returns,
  2. the spacing between adjacent neurons,
  3. whether a neuron's slots can reach into the next neuron,
  4. what the true "slots fit" condition is.
"""
NL, NKH, HD, PMAX = 24, 2, 128, 1024
n_neurons = NL * NKH * HD

def neuron_base(layer, kv_head, dim, stride):
    idx = (layer * NKH + kv_head) * HD + dim
    return idx * stride

print("=== SHIPPED formula: neuron_base = ((L*NKH + h)*HD + d) * stride ===")
print("n_neurons =", n_neurons)
print("stride   = HD * PMAX =", HD * PMAX)
print()

s = HD * PMAX
print("adjacent-neuron spacing (same L,h; d and d+1):")
print("  base(d=0) =", neuron_base(0, 0, 0, s))
print("  base(d=1) =", neuron_base(0, 0, 1, s))
print("  spacing   =", neuron_base(0, 0, 1, s) - neuron_base(0, 0, 0, s))
print()
print("=> spacing is HD*PMAX = %d, i.e. each neuron gets a WHOLE head's worth" % s)
print("   of address space (HD*PMAX elements), not just PMAX.")
print()
print("A neuron uses slots 0..PMAX-1 = %d elements, in a %d-element reservation." % (PMAX, s))
print("slack per neuron =", s - PMAX, "elements")
print()
print("Does neuron d's slot range overlap neuron d+1's?")
print("  d   range = [%d, %d)" % (neuron_base(0,0,0,s), neuron_base(0,0,0,s)+PMAX))
print("  d+1 range = [%d, %d)" % (neuron_base(0,0,1,s), neuron_base(0,0,1,s)+PMAX))
print("  overlap   =", neuron_base(0,0,0,s)+PMAX > neuron_base(0,0,1,s))
print()
print("=== the TRUE non-overlap condition, given base = idx*stride ===")
print("need  slot < max_seq_per_neuron  (checked in slot_addr)  AND")
print("      idx*stride + max_seq_per_neuron <= (idx+1)*stride  for all idx")
print("  <=>  max_seq_per_neuron <= stride")
print()
print("shipped validate() requires the STRICTER: stride >= HD * PMAX")
print("  HD*PMAX >= PMAX  <=>  HD >= 1, always true")
print("  so validate() is SUFFICIENT but not NECESSARY: stride=PMAX would also")
print("  be non-overlapping and validate() would REJECT it.")
print()
print("=== is that a bug? ===")
print("No memory-safety bug: validate()'s condition implies non-overlap, so the")
print("shipped code cannot corrupt a neighbour. But it is:")
print("  (a) an over-strict guard that REJECTS valid layouts (e.g. stride=PMAX),")
print("  (b) inconsistent with format.rs:295-296 which states the weak form,")
print("  (c) result: 131072/1024 = %d x more memory than the tight layout." % (s // PMAX))
print()
total_shipped = n_neurons * s
total_tight = n_neurons * PMAX
print("bytes shipped (f32): %d elements = %.3f GB" % (total_shipped, total_shipped*4/1e9))
print("bytes if tight      : %d elements = %.3f GB" % (total_tight, total_tight*4/1e9))
print("ratio               : %.1fx" % (total_shipped / total_tight))
print()
print("NOTE this is a LAYOUT/API observation, not a measured runtime allocation:")
print("p14_s5_check.py / the engine may never allocate the padded form. I report")
print("the arithmetic of the shipped formulas and flag the doc inconsistency.")
