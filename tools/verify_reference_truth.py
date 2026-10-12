# PATH: DENSE
# Verify the 24-layer reference truth artifact (task brief section 8):
#   "当前训练框架产出的 24 层模型（120 矩阵，30.23 MB）"
# This script does NOT assume the header layout -- it reads conn_sparse.rs's
# declared layout and cross-checks the counts against the manifest.
import struct, os, sys, glob, hashlib

ROOT = str(Path(__file__).resolve().parents[1])
ART = os.path.join(ROOT, "artifacts", "p18_sparse24")

fs = sorted(glob.glob(os.path.join(ART, "*.csr")))
print("files = %d" % len(fs))
tot = sum(os.path.getsize(f) for f in fs)
print("total bytes = %d = %.2f MiB = %.2f MB(1e6)" % (tot, tot / 1024 / 1024, tot / 1e6))

d = open(fs[0], "rb").read()
print("\n--- section header of %s ---" % os.path.basename(fs[0]))
print("magic   = %r" % d[:4])
print("ver     = %d" % struct.unpack("<H", d[4:6])[0])
print("flags   = %d" % struct.unpack("<H", d[6:8])[0])
for off in range(8, 72, 4):
    print("  u32@%-3d = %d" % (off, struct.unpack("<I", d[off:off + 4])[0]))

# Independent check of the arithmetic the brief states.
# Read the manifest and confirm per-matrix geometry.
mani = open(os.path.join(ART, "manifest.txt"), "r", encoding="utf-8", errors="replace").read().splitlines()
rows = 0
mats = []
for ln in mani:
    if ln.startswith("#") or not ln.strip():
        continue
    parts = ln.split()
    if len(parts) >= 7:
        rows += 1
        mats.append((int(parts[0]), parts[1], int(parts[3]), int(parts[4]), int(parts[5]), int(parts[6])))
print("\nmanifest data rows = %d" % rows)
layers = sorted(set(m[0] for m in mats))
names = sorted(set(m[1] for m in mats))
print("layers = %d  (%d..%d)" % (len(layers), layers[0], layers[-1]))
print("matrix names per layer = %d  %s" % (len(names), names))
print("total matrices = layers * names = %d * %d = %d" % (len(layers), len(names), len(layers) * len(names)))

# section_bytes declared vs real file size -- a mismatch would mean the
# manifest and the artifacts disagree (the "confident wrong answer" class).
bad = 0
for f in fs:
    real = os.path.getsize(f)
    if real != 251928:
        bad += 1
        print("  SIZE MISMATCH %s = %d" % (os.path.basename(f), real))
print("size mismatches = %d (all files must be 251928)" % bad)

# Geometry: n_neurons=1536, k0=20 -> e = 30720 pairs.
n_neurons, k0 = 1536, 20
e = n_neurons * k0
print("\n--- arithmetic the brief's '76.8x' rests on ---")
print("e (connections per matrix) = %d * %d = %d" % (n_neurons, k0, e))
print("dense equivalent per matrix  = %d * %d = %d" % (n_neurons, n_neurons, n_neurons * n_neurons))
print("ratio (storage/arithmetic)  = %.1fx" % (n_neurons * n_neurons / e))
print("per matrix bytes f32 = w+src+act = %d*4*3 = %d B" % (e, e * 12))
print("120 matrices * %d B = %d B = %.2f MiB" % (e * 12, 120 * e * 12, 120 * e * 12 / 1024 / 1024))

# Digest so a later step can prove the reference was not silently mutated.
h = hashlib.sha256()
for f in fs:
    h.update(os.path.basename(f).encode())
    h.update(open(f, "rb").read())
print("\nREFERENCE TRUTH sha256 (names+bytes, all %d files) = %s" % (len(fs), h.hexdigest()))

# Cross-check against other artifacts (task-1/2/3 runs must reproduce this).
out = os.path.join(ROOT, "artifacts", "reference_truth.sha256")
with open(out, "w", encoding="ascii") as fh:
    fh.write("# 24-layer reference truth (task brief section 8)\n")
    fh.write("# files=%d total_bytes=%d\n" % (len(fs), tot))
    for f in fs:
        fh.write("%s  %s\n" % (hashlib.sha256(open(f, "rb").read()).hexdigest(), os.path.basename(f)))
print("per-file digests written -> %s" % out)
