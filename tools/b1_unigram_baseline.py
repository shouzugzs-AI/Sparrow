#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reference points for "did the model learn anything beyond token frequencies?"

Computes, on EXACTLY the positions the trainer evaluates (same indexing scheme),
two parameter-free baselines:
  * uniform   : CE = ln(V)                       (V = number of distinct train tokens + 1)
  * unigram   : CE = -mean log p_train(target)   (frequencies from the TRAIN split only)

If a trained model's val CE is at/near the unigram number, then whatever it learned is
(almost) just token frequencies -- i.e. the 24-layer stack added ~nothing on top of the
embedding/head. If it is clearly below, it learned real context (here: bigram) structure.

Usage:
  python tools/b1_unigram_baseline.py --data-dir artifacts/b1_scaling \
      --seq 32 --batch 4 --eval-batches 4
"""

import argparse
import math
import os
import struct
import sys

import numpy as np


def read_u32(path):
    b = open(path, "rb").read()
    assert len(b) % 4 == 0, path
    return np.frombuffer(b, dtype="<u4").astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(r"<repo>", "artifacts", "b1_scaling"))
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--eval-batches", type=int, default=4)
    args = ap.parse_args()

    tr = read_u32(os.path.join(args.data_dir, "train_ids.bin"))
    va = read_u32(os.path.join(args.data_dir, "val_ids.bin"))

    # 与驱动完全相同的 val 批构造：base = (eb*977 + b*seq*13) % (len-seq-1)
    # ⚠️ 这里必须取 **va[base+t+1]**（真正的 token id）。第一版把**下标**当成了 token id
    #    （收集 (base+t, base+t+1) 后直接当 target 用），于是"OOV 率"算成 81%，
    #    而驱动自己的 `eval-batch health` 报的是 1.41%。跨工具对账把这个 bug 抓出来了。
    tgt = []
    for eb in range(args.eval_batches):
        for b in range(args.batch):
            base = (eb * 977 + b * args.seq * 13) % (len(va) - args.seq - 1)
            for t in range(args.seq - 1):
                tgt.append(int(va[base + t + 1]))
    tgt = np.array(tgt, dtype=np.int64)

    # 训练集词频（只统计 train；target 空间 = train 出现过的 token）
    uniq, cnt = np.unique(tr, return_counts=True)
    V = len(uniq)
    p = cnt / cnt.sum()
    # 映射 val 的 target 到 unigram 分布；未在 train 出现过的 → 概率 0（驱动里映射成 UNK，
    # 所以这里也给 UNK 一个极小的概率：用 1/(2*total) 的下限，避免 log(0)）
    look = {int(u): float(pp) for u, pp in zip(uniq, p)}
    floor = 1.0 / (2.0 * float(cnt.sum()))
    probs = np.array([look.get(int(t), 0.0) for t in tgt])
    n_oov = int((probs == 0).sum())
    probs = np.maximum(probs, floor)
    ce_uni = float(-np.mean(np.log(probs)))
    ce_uni_raw = float(-np.mean(np.log(np.array([max(look.get(int(t), 0.0), 1e-12) for t in tgt]))))
    ce_uniform = float(math.log(V))

    print("val positions evaluated : %d  (identical to the trainer's scheme)" % len(tgt))
    print("train tokens            : %d   distinct = %d   (UNK id 0 => V_model = %d)" % (len(tr), V, V + 1))
    print("val target OOV (not in train): %d (%.3f%%)" % (n_oov, 100.0 * n_oov / len(tgt)))
    print("")
    print("uniform baseline  CE = ln(V)            = %.4f   (PPL %.1f)" % (ce_uniform, math.exp(ce_uniform)))
    print("unigram baseline  CE (train freq)       = %.4f   (PPL %.1f)" % (ce_uni, math.exp(ce_uni)))
    print("unigram baseline  (OOV floored at 1e-12)= %.4f   (PPL %.1f)" % (ce_uni_raw, math.exp(ce_uni_raw)))
    print("")
    print("=> any model's val CE gap vs the unigram number is what it learned BEYOND token frequency:")
    print("   val ~ %.4f  => essentially only token frequency (the 24-layer stack added nothing)" % ce_uni)
    print("   val <  %.4f  => learned structure beyond frequency (at most bigram here)" % ce_uni)
    # 最常见的 10 个 token 的占比，让"词频有多集中"可见
    order = np.argsort(-cnt)[:10]
    print("")
    print("top-10 train tokens (id:share): " + ", ".join("%d:%.4f" % (uniq[i], cnt[i] / cnt.sum()) for i in order))
    return 0


if __name__ == "__main__":
    sys.exit(main())
