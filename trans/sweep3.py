import subprocess, re

exe = r".\target\release\vmlm-trans-verify.exe"

def run(args):
    return subprocess.run([exe] + args, capture_output=True, text=True,
                          cwd=r"<repo>").stdout

print("=== fixed rows=1536, vary k0 -> vary code footprint (packed layout) ===")
print(f"{'k0':>4} {'KB code':>9} {'op ms':>10} {'packed ms':>11} {'x':>7}")
for k0 in [2, 4, 8, 12, 16, 20]:
    o = run(["--d", "1536", "--k0", str(k0), "--k", "8", "--rows", "1536", "--reps", "11"])
    m_bytes = re.search(r"total code bytes\s+=\s+(\d+)", o)
    m_bytes2 = re.search(r"packed back-to-back,\s+(\d+)\s+bytes", o)
    secb = o.split("--- (b) WHOLE layer")[1].split("--- (c)")[0] if "--- (b)" in o else ""
    secc = o.split("--- (c) CONTIGUOUS")[1] if "--- (c) CONTIGUOUS" in o else ""
    m_op = re.search(r"op_accum_layer median\s+([\d.]+)", secb)
    m_c = re.search(r"translated\s+\d+\s+calls\s+median\s+([\d.]+)", secc)
    if m_op and m_c:
        kb = int(m_bytes2.group(1)) / 1024 if m_bytes2 else float('nan')
        op = float(m_op.group(1)); c = float(m_c.group(1))
        print(f"{k0:>4} {kb:>9.0f} {op:>10.5f} {c:>11.5f} {op/c:>7.3f}")
    else:
        print(f"{k0:>4} parse failed")
