import subprocess, re

exe = r".\target\release\vmlm-trans-verify.exe"

def run(args):
    return subprocess.run([exe] + args, capture_output=True, text=True,
                          cwd=r"<repo>").stdout

print("=== rows sweep: whole-layer speedup vs row count (d=1536 k0=20 k=8) ===")
print(f"{'rows':>6} {'op_layer ms':>12} {'tr_layer ms':>12} {'speedup':>9}")

for rows in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1536]:
    o = run(["--d", "1536", "--k0", "20", "--k", "8", "--rows", str(rows), "--reps", "9"])
    # grab the (b) section only
    sec = o.split("--- (b) WHOLE layer")[1] if "--- (b) WHOLE layer" in o else ""
    m_op = re.search(r"op_accum_layer median\s+([\d.]+)", sec)
    m_tr = re.search(r"translated\s+\d+\s+calls\s+median\s+([\d.]+)", sec)
    if m_op and m_tr:
        op = float(m_op.group(1)); tr = float(m_tr.group(1))
        sp = op / tr if tr else float('nan')
        print(f"{rows:>6} {op:>12.6f} {tr:>12.6f} {sp:>9.3f}")
    else:
        print(f"{rows:>6}  <parse failed>")
