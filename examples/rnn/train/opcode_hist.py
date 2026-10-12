#!/usr/bin/env python3
"""A2: dump the opcode histogram of a .vmb, mapped to names from format.rs."""
import re
import struct
import sys

VMB = sys.argv[1] if len(sys.argv) > 1 else r"bptt_t2.vmb"
FMT = r"<repo>/core/src/format.rs"

ops = {}
for line in open(FMT, encoding="utf-8"):
    m = re.match(r"pub const (OP_\w+): u8 = (\d+);", line.strip())
    if m:
        ops[int(m.group(2))] = m.group(1)

d = open(VMB, "rb").read()
to = struct.unpack_from("<I", d, 8)[0]
tl = struct.unpack_from("<I", d, 12)[0]
n = tl // 16
seen = {}
for i in range(n):
    op = struct.unpack_from("<B", d, to + i * 16)[0]
    seen.setdefault(op, []).append(i)

print(f"{VMB}: {n} instructions, {len(seen)} distinct opcodes")
for o in sorted(seen):
    name = ops.get(o, "???UNKNOWN")
    print(f"  op {o:4}  {name:16} x{len(seen[o]):4}   first@{seen[o][0]}")
