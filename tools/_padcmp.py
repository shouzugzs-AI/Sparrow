# -*- coding: utf-8 -*-
"""补真实字符 vs 补 UNK 的对照：生成 + 打分（一次跑完，避免 PowerShell 引号问题）。"""
import subprocess, os, re

ROOT = str(Path(__file__).resolve().parents[1])
EXE = os.path.join(ROOT, r"target/release/vmlm-b1-full.exe")
GEN = os.path.join(ROOT, r"artifacts/gen")
BASE = ("--sparse-io 1 --dim 4096 --k0 8 --k0-embed 16 --k0-head 16 --k0-attn 8 --layers 32 "
        "--no-residual 1 --attn-every 4 --attn-sparse 1 --attn-src-shared 1 --attn-calib 0 "
        "--head-skip 1 --seq 64 --batch 4 --threads 4 --data-dir artifacts/b1_char --vocab-mode full "
        "--chars-per-token 1.0 --steps 1 --eval-every 999999 --log-every 999999 --plateau-patience 0 "
        "--load-model artifacts/01.cpk")

def run(args):
    p = subprocess.run([EXE] + args.split(), cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return p.returncode, p.stdout + p.stderr

rows = []
for sd in (1, 2, 3, 4):
    out = os.path.join(GEN, "fixed_s%d.txt" % sd)
    rc, _ = run(BASE + " --generate 1 --seed %d --temperature 0.7 --top-k 30 --top-p 0.85 "
                "--gen-len 100 --prompt-file artifacts/gen/p1.txt --gen-out artifacts/gen/fixed_s%d.txt" % (sd, sd))
    txt = open(out, encoding="utf-8").read()
    body = txt.split("[gen only]")[1].strip()
    clean = os.path.join(GEN, "cleanfix_s%d.txt" % sd)
    open(clean, "w", encoding="utf-8").write(body)
    rc2, so2 = run(BASE + " --score-file artifacts/gen/cleanfix_s%d.txt" % sd)
    m = re.search(r"bits/char（非UNK目标[^=]*= ([0-9.]+)", so2)
    score = float(m.group(1)) if m else float("nan")
    distinct = len(set(body.replace("\n", "")))
    rows.append((sd, distinct, score, body.replace("\n", " ")[:46]))

print("seed | 不同字/100 | 模型自评 bits/char | 开头")
for sd, d, s, head in rows:
    print("%4d | %10d | %17.4f | %s" % (sd, d, s, head))
print()
print("（参照：val=6.53；均匀地板=11.99；补 UNK 时的旧值 = seed1 2.00 / seed2 14.11 / seed3 11.78）")
