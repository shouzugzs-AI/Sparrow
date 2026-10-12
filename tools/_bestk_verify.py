# -*- coding: utf-8 -*-
"""验证 best-of-K：K=1 应与旧行为一致（同 seed 同输出）；K=4 应打印各候选自评并选最低。"""
import subprocess, os, re

ROOT = str(Path(__file__).resolve().parents[1])
EXE = os.path.join(ROOT, r"target/release/vmlm-b1-full.exe")
BASE = ("--sparse-io 1 --dim 4096 --k0 8 --k0-embed 16 --k0-head 16 --k0-attn 8 --layers 32 "
        "--no-residual 1 --attn-every 4 --attn-sparse 1 --attn-src-shared 1 --attn-calib 0 "
        "--head-skip 1 --seq 64 --batch 4 --threads 4 --data-dir artifacts/b1_char --vocab-mode full "
        "--chars-per-token 1.0 --steps 1 --eval-every 999999 --log-every 999999 --plateau-patience 0 "
        "--load-model artifacts/01.cpk --generate 1 --gen-len 100 --temperature 0.7 --top-k 30 --top-p 0.85 "
        "--prompt-file artifacts/gen/p1.txt")

def run(a):
    p = subprocess.run([EXE] + a.split(), cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return p.stdout + p.stderr

for tag, extra in [("k1_seed1", "--seed 1 --n-candidates 1"), ("k1_seed2", "--seed 2 --n-candidates 1"),
                   ("k4_seed2", "--seed 2 --n-candidates 4"), ("k4_seed5", "--seed 5 --n-candidates 4")]:
    out = "artifacts/gen/best_%s.txt" % tag
    so = run(BASE + " --gen-out " + out + " " + extra)
    cand = [l.strip() for l in so.splitlines() if l.strip().startswith("候选")]
    sel = [l.strip() for l in so.splitlines() if "选定" in l]
    txt = open(os.path.join(ROOT, out), encoding="utf-8").read()
    body = txt.split("[gen only]")[1].strip()
    sc = re.search(r"self_bits_per_char=([0-9.]+)", txt)
    print("=== %s ===" % tag)
    for c in cand:
        print("   ", c)
    for l in sel:
        print("   ", l)
    print("    写入文件自评 = %s ; 不同字 = %d ; 开头 = %s" %
          (sc.group(1) if sc else "?", len(set(body.replace("\n", ""))), body.replace("\n", " ")[:40]))
