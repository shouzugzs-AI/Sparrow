import capstone, sys

path = sys.argv[1]
data = open(path, 'rb').read()
md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
n = 0
for ins in md.disasm(data, 0):
    print(f"{ins.address:04x}: {ins.mnemonic:8s} {ins.op_str}")
    n += 1
    if n > 40:
        print("... (truncated)")
        break
print(f"\ntotal bytes={len(data)}")
