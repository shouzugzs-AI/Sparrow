import re, io, os

ROOT = str(Path(__file__).resolve().parents[1])
fmt = io.open(os.path.join(root, 'core', 'src', 'format.rs'), encoding='utf-8').read()
interp = io.open(os.path.join(root, 'core', 'src', 'interp.rs'), encoding='utf-8').read()
asm = io.open(os.path.join(root, 'asm', 'src', 'main.rs'), encoding='utf-8').read()

ops = {}
for m in re.finditer(r'pub const (OP_\w+): u8 = (\d+);', fmt):
    ops[m.group(1)] = int(m.group(2))

rows = []
for name, num in sorted(ops.items(), key=lambda kv: kv[1]):
    in_interp = re.search(r'\b' + name + r'\b', interp) is not None
    # assembler exposes it if some "NAME" => OP_X mapping exists
    in_asm = re.search(r'"([A-Z0-9_]+)"\s*=>\s*' + name + r'\b', asm) is not None
    if not in_interp:
        rows.append((num, name, in_asm))

print('opcodes defined in format.rs but ABSENT from interp.rs')
print('%4s  %-22s %s' % ('num', 'name', 'assembler-exposed?'))
for num, name, in_asm in rows:
    print('%4d  %-22s %s' % (num, name, 'YES  <-- user-visible trap' if in_asm else 'no'))
print()
print('total: %d' % len(rows))
