# -*- coding: utf-8 -*-
"""验证重复抑制：默认关（旧输出不变）；开了之后 seed 2/5 应不再复读且自评下降。"""
import subprocess, os, re

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

def analyze(tag, extra):
    out = "artifacts/gen/rep_%s.txt" % tag
    run(BASE + " --gen-out " + out + " " + extra)
    txt = open(os.path.join(ROOT, out), encoding="utf-8").read()
    body = txt.split("[gen only]")[1].strip().replace("\n", "")
    sc = re.search(r"self_bits_per_char=([0-9.]+)", txt)
    # 最频繁 4-gram 覆盖
    from collections import Counter
    c = Counter(body[i:i+4] for i in range(len(body)-3))
    rep4 = (c.most_common(1)[0][1] * 4 / max(1, len(body))) if c else 0.0
    print("%-26s 自评=%s  不同字=%3d  4gram覆盖=%.2f  开头=%s" %
          (tag, sc.group(1) if sc else "?", len(set(body)), rep4, body[:34]))

analyze("off_seed2", "--seed 2")
analyze("off_seed5", "--seed 5")
analyze("on_seed2", "--seed 2 --rep-penalty 1.2 --no-repeat-ngram 4")
analyze("on_seed5", "--seed 5 --rep-penalty 1.2 --no-repeat-ngram 4")
analyze("on_seed1", "--seed 1 --rep-penalty 1.2 --no-repeat-ngram 4")
print()
print("参照：val=6.53；可读≈6-8；乱码≈11+；复读 ⇒ 4gram覆盖≈1.0")
