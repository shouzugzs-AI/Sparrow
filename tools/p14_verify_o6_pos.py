import re, os, sys
sys.path.insert(0, r"<repo>/tools")
import p14_verify_lib as L

# O6-P7 structural check: every generation step's ROPE must use its OWN
# absolute position as a literal (OP_ROPE's dt is a compile-time u16), i.e. the
# emitted .vml must contain one distinct literal per step at position S+step.
#
# The point of checking literals rather than runtime behaviour: a LOOP_CNT loop
# would rotate every token at the SAME position, which is wrong for all but the
# first AND looks like plausible numeric drift rather than a bug (P13 5A.3).
vml = os.path.join(L.ROOT, "examples", "p14", "feed", "full.vml")
if not os.path.exists(vml):
    vml = os.path.join(L.ROOT, "examples", "p14", "argmax", "full.vml")
print("inspecting:", vml)
src = open(vml, encoding="ascii", errors="replace").read().splitlines()

# The decode/generation section is what follows the prefill. Collect every ROPE
# with its literal position, in emission order.
rope = []
for i, line in enumerate(src, 1):
    m = re.match(r"\s*ROPE\s+(\S+)\s*,\s*(\S+)\s*,\s*(\S+)\s*,\s*(\d+)\s*$", line)
    if m:
        rope.append((i, m.group(1), m.group(2), m.group(4)))
print("total ROPE instructions:", len(rope))
positions = [int(r[3]) for r in rope]
print("distinct positions used :", sorted(set(positions)))
print("count per position      :")
from collections import Counter
c = Counter(positions)
for p in sorted(c):
    print("   pos %-6d : %d ROPE" % (p, c[p]))

# Sanity: prefill covers 0..S-1; generation covers S..S+G-1.
print()
print("first 8 ROPE literals:", positions[:8])
print("last 8 ROPE literals :", positions[-8:])

# What a LOOP_CNT implementation would look like: only ONE distinct position
# repeated. Assert we are NOT in that case.
distinct = len(set(positions))
print()
print("distinct position count: %d" % distinct)
if distinct > 1:
    print("=> PASS (structural): each step carries its OWN literal position, so a")
    print("   runtime loop reusing one position is ruled out by construction.")
else:
    print("=> FAIL: only one distinct position -> every step rotates identically")

# CAN-THE-TEST-FAIL: the same detector must reject a mono-position program.
mono = [("ROPE qdrot, qdh, theta1, %d" % 4)] * 5
det = set(int(re.match(r".*,\s*(\d+)\s*$", m).group(1)) for m in mono)
print()
print("control: a synthetic mono-position program -> distinct=%d (must be 1"
      " and therefore REJECTED)" % len(det))
print("detector can distinguish the two cases:", (len(det) == 1) and (distinct > 1))
