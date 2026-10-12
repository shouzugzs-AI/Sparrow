# -*- coding: utf-8 -*-
"""验证重复抑制：只打印 ASCII 安全字段，正文写 UTF-8 文件供 read 查看。"""
import subprocess, os, re
from collections import Counter

ROOT = str(Path(__file__).resolve().parents[1])
EXE = os.path.join(ROOT, r"target/release/vmlm-b1-full.exe")
BASE = ("--sparse-io 1 --dim 4096 --k0 8 --k0-embed 16 --k0-head 16 --k0-attn 8 --layers 32 "
        "--no-residual 1 --attn-every 4 --attn-sparse 1 --attn-src-shared 1 --attn-calib 0 "
        "--head-skip 1 --seq 64 --batch 4 --threads 4 --data-dir artifacts/b1_char --vocab-mode full "
        "--chars-per-token 1.0 --steps 1 --eval-every 999999 --log-every 999999 --plateau-patience 0 "
        "--load-model artifacts/01.cpk --generate 1 --gen-len 100 --temperature 0.7 --top-k 30 --top-p 0.85 "
        "--prompt-file artifacts/gen/p1.txt --n-candidates 1")

def run(a):
    p = subprocess.run([EXE] + a.split(), cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return p.stdout + p.stderr

rows = []
cases = [("off_seed1", "--seed 1"), ("off_seed2", "--seed 2"), ("off_seed5", "--seed 5"),
         ("on_seed1", "--seed 1 --rep-penalty 1.2 --no-repeat-ngram 4"),
         ("on_seed2", "--seed 2 --rep-penalty 1.2 --no-repeat-ngram 4"),
         ("on_seed5", "--seed 5 --rep-penalty 1.2 --no-repeat-ngram 4"),
         ("on_seed7", "--seed 7 --rep-penalty 1.2 --no-repeat-ngram 4"),
         ("on_seed9", "--seed 9 --rep-penalty 1.2 --no-repeat-ngram 4")]
for tag, extra in cases:
    out = "artifacts/gen/rep_%s.txt" % tag
    run(BASE + " --gen-out " + out + " " + extra)
    txt = open(os.path.join(ROOT, out), encoding="utf-8").read()
    body = txt.split("[gen only]")[1].strip().replace("\n", "")
    sc = re.search(r"self_bits_per_char=([0-9.]+)", txt)
    c = Counter(body[i:i+4] for i in range(max(0, len(body)-3)))
    rep4 = (c.most_common(1)[0][1] * 4 / max(1, len(body))) if c else 0.0
    rows.append((tag, sc.group(1) if sc else "?", len(set(body)), rep4, body))
    print("%-10s self=%6s distinct=%3d rep4=%.2f len=%d" % (tag, rows[-1][1], rows[-1][2], rep4, len(body)))

doc = ["# 重复抑制（--rep-penalty / --no-repeat-ngram）验证\n",
       "| 运行 | 自评 bits/char | 不同字 | 最频繁4-gram覆盖 |", "|---|---|---|---|"]
for tag, sc, d, r, b in rows:
    doc.append("| %s | %s | %d | %.2f |" % (tag, sc, d, r))
doc.append("")
for tag, sc, d, r, b in rows:
    doc.append("### %s（自评 %s，4-gram 覆盖 %.2f）" % (tag, sc, r))
    doc.append(b)
    doc.append("")
open(os.path.join(ROOT, r"artifacts/gen/REP_REPORT.md"), "w", encoding="utf-8").write("\n".join(doc))
print("report written")
