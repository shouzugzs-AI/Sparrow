# -*- coding: utf-8 -*-
"""判决性测试：全支撑采样（k=4054,p=1.0,T=1.0）。若采样器正确，输出应当 ≈ 模型自身熵（~7-9 bits）；
   若输出仍 12+ bits（比均匀地板还差），则采样器没有按模型分布抽。"""
import subprocess, os, re

ROOT = str(Path(__file__).resolve().parents[1])
EXE = os.path.join(ROOT, r"target/release/vmlm-b1-full.exe")
BASE = ("--sparse-io 1 --dim 4096 --k0 8 --k0-embed 16 --k0-head 16 --k0-attn 8 --layers 32 "
        "--no-residual 1 --attn-every 4 --attn-sparse 1 --attn-src-shared 1 --attn-calib 0 "
        "--head-skip 1 --seq 64 --batch 4 --threads 4 --data-dir artifacts/b1_char --vocab-mode full "
        "--chars-per-token 1.0 --steps 1 --eval-every 999999 --log-every 999999 --plateau-patience 0 "
        "--load-model artifacts/01.cpk")

def run(a):
    p = subprocess.run([EXE] + a.split(), cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return p.stdout + p.stderr

def score(path):
    so = run(BASE + " --score-file " + path)
    m = re.search(r"=\s*([0-9.]+)\s+bits/char", so) or re.search(r"([0-9.]+)\s*（均匀地板", so)
    m2 = re.search(r"([0-9]+\.[0-9]+)\s+bits/char", so)
    return (m2.group(1) if m2 else "?"), so

cases = [("full_k4054_T1.0", "--temperature 1.0 --top-k 4054 --top-p 1.0 --seed 1"),
         ("full_k4054_T1.0_s2", "--temperature 1.0 --top-k 4054 --top-p 1.0 --seed 2"),
         ("k30_T0.7_s2", "--temperature 0.7 --top-k 30 --top-p 0.85 --seed 2"),
         ("greedy_k1_s2", "--temperature 0.7 --top-k 1 --top-p 1.0 --seed 2")]
for tag, opt in cases:
    out = r"artifacts/gen/dec_%s.txt" % tag
    run(BASE + " --generate 1 --gen-len 100 --prompt-file artifacts/gen/p1.txt --gen-out " + out + " " + opt)
    txt = open(os.path.join(ROOT, out), encoding="utf-8").read()
    body = txt.split("[gen only]")[1].strip()
    clean_rel = r"artifacts/gen/clean_dec_%s.txt" % tag
    open(os.path.join(ROOT, clean_rel), "w", encoding="utf-8").write(body)
    s, _ = score(clean_rel)
    print("%-20s 不同字=%3d  模型自评=%s bits/char   开头=%s" % (tag, len(set(body.replace("\n", ""))), s, body.replace("\n", " ")[:34]))
print()
print("参照：val=6.53  均匀地板=11.99")
