import subprocess, re

exe = r".\target\release\vmlm-trans-verify.exe"

def run(args):
    return subprocess.run([exe] + args, capture_output=True, text=True,
                          cwd=r"<repo>").stdout

print("=== layout (a) one row | (b) page-per-row | (c) contiguous, same row count ===")
print(f"{'rows':>6} {'op ms':>10} {'b: page/row':>12} {'c: packed':>11} {'b x':>8} {'c x':>8}")
for rows in [64, 128, 256, 512, 768, 1024, 1536]:
    o = run(["--d", "1536", "--k0", "20", "--k", "8", "--rows", str(rows), "--reps", "11"])
    secb = o.split("--- (b) WHOLE layer")[1].split("--- (c)")[0] if "--- (b)" in o else ""
    secc = o.split("--- (c) CONTIGUOUS")[1] if "--- (c) CONTIGUOUS" in o else ""
    m_op = re.search(r"op_accum_layer median\s+([\d.]+)", secb)
    m_b = re.search(r"translated\s+\d+\s+calls\s+median\s+([\d.]+)", secb)
    m_c = re.search(r"translated\s+\d+\s+calls\s+median\s+([\d.]+)", secc)
    if m_op and m_b and m_c:
        op = float(m_op.group(1)); b = float(m_b.group(1)); c = float(m_c.group(1))
        print(f"{rows:>6} {op:>10.5f} {b:>12.5f} {c:>11.5f} {op/b:>8.3f} {op/c:>8.3f}")
    else:
        print(f"{rows:>6} parse failed")
