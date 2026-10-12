#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B1 — STEP 2: cut the vocabulary down to CJK (+ necessary punctuation/digits).

WHY: `--vocab-mode observed` keeps every token that occurs in the train split
(10,920 distinct ids), including a lot of foreign-language fragments / rare symbols.
Those consume embedding + head parameters (2*V*k0) that a Chinese task never uses.

HOW: the data was encoded with the official MiniCPM tokenizer, so we decode each
observed id back to its string and keep it only if the string is
  * CJK           (U+4E00-9FFF, U+3400-4DBF, U+F900-FAFF, U+20000-2EBEF, 3000-303F, FF00-FFEF, 2E80-2EFF, 31C0-31EF)
  * punctuation / digits / whitespace  (necessary structure)
  * a special token
and (optionally) the `--extra-top-n` most frequent remaining tokens, to control OOV.

OUTPUT (for the Rust driver, which must not need a tokenizer):
  <out>.txt   one OLD raw token id per line, in NEW-id order (new id = line_index + 1, 0 = UNK)
  <out>.json  the same list + the statistics below (for humans / reports)

Usage:
  python tools/b1_vocab_cjk.py ^
      --data-dir artifacts/b1_scaling ^
      --tokenizer "<外部模型目录>/MiniCPM5-1B/tokenizer.json" ^
      --out artifacts/b1_scaling/vocab_cjk
"""

import argparse
import json
import os
import sys
import unicodedata

ROOT = str(Path(__file__).resolve().parents[1])

# ---- CJK 范围 ----
CJK_RANGES = [
    (0x4E00, 0x9FFF),    # 基本汉字
    (0x3400, 0x4DBF),    # 扩展 A
    (0xF900, 0xFAFF),    # 兼容汉字
    (0x20000, 0x2A6DF),  # 扩展 B
    (0x2A700, 0x2EBEF),  # 扩展 C/D/E/F
    (0x2E80, 0x2EFF),    # 部首
    (0x3000, 0x303F),    # CJK 标点
    (0x31C0, 0x31EF),    # 笔画
    (0xFF00, 0xFFEF),    # 全角
]
# ---- 标点/数字/空白 ----
PUNCT_EXTRA = set("！？。，、；：（）《》「」『』【】—…·～“”‘’％＃＠＆＊＋－／＝＜＞｜＄＾｀｛｝［］＼｜")


def is_cjk(ch):
    o = ord(ch)
    for a, b in CJK_RANGES:
        if a <= o <= b:
            return True
    return False


def classify(s):
    """返回 'cjk' / 'punct' / 'special' / 'other'"""
    if s == "":
        return "other"
    if all(is_cjk(c) for c in s):
        return "cjk"
    ok = True
    for c in s:
        o = ord(c)
        cat = unicodedata.category(c)
        if is_cjk(c):
            continue
        if c in PUNCT_EXTRA:
            continue
        if cat.startswith("P"):          # 标点
            continue
        if cat in ("Nd", "Nl", "No"):    # 数字
            continue
        if cat in ("Zs", "Zl", "Zp") or c in "\r\n\t ":  # 空白
            continue
        ok = False
        break
    return "punct" if ok else "other"


def read_u32(path):
    import numpy as np
    b = open(path, "rb").read()
    assert len(b) % 4 == 0, path
    return np.frombuffer(b, dtype="<u4").astype("int64")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "artifacts", "b1_scaling"))
    ap.add_argument("--tokenizer", default=r"<外部模型目录>/MiniCPM5-1B/tokenizer.json")
    ap.add_argument("--out", default=os.path.join(ROOT, "artifacts", "b1_scaling", "vocab_cjk"))
    ap.add_argument("--mode", default="cjk", choices=["cjk", "freq"],
                    help="cjk=只留中文+标点/数字（实测只砍 3.8%%）；freq=按训练频次留前 --keep-n 个")
    ap.add_argument("--keep-n", type=int, default=5000, help="--mode freq 时保留的类型数")
    ap.add_argument("--extra-top-n", type=int, default=0,
                    help="再补上训练集里最高频的 N 个非 CJK/标点 token（用来压 OOV）")
    ap.add_argument("--max-oov", type=float, default=0.05, help="OOV 上限（超过就报警）")
    args = ap.parse_args()

    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(args.tokenizer)

    tr = read_u32(os.path.join(args.data_dir, "train_ids.bin"))
    va = read_u32(os.path.join(args.data_dir, "val_ids.bin"))

    uniq, cnt = __import__("numpy").unique(tr, return_counts=True)
    freq = {int(u): int(c) for u, c in zip(uniq, cnt)}
    specials = set(tk.get_added_tokens_decoder().keys()) if hasattr(tk, "get_added_tokens_decoder") else set()

    kinds = {}
    kept_cjk, kept_punct, kept_special, other = [], [], [], []
    for i in freq:
        try:
            s = tk.decode([i], skip_special_tokens=False)
        except Exception:
            s = ""
        k = "special" if (i in specials or (s.startswith("<") and s.endswith(">"))) else classify(s)
        kinds[i] = (k, s)
        if k == "cjk":
            kept_cjk.append(i)
        elif k == "punct":
            kept_punct.append(i)
        elif k == "special":
            kept_special.append(i)
        else:
            other.append(i)

    # 高频非中文 token（按训练频率降序，id 升序打破并列 —— 完全确定性）
    other_sorted = sorted(other, key=lambda i: (-freq[i], i))
    extra = other_sorted[: args.extra_top_n]

    if args.mode == "freq":
        # 按训练频次取前 N（id 升序打破并列 —— 完全确定性），与 CJK 过滤无关
        by_f = sorted(freq.keys(), key=lambda i: (-freq[i], i))
        kept = sorted(by_f[: args.keep_n])
        print("MODE=freq  keep_n=%d" % args.keep_n)
    else:
        kept = sorted(set(kept_cjk + kept_punct + kept_special + extra))
        print("MODE=cjk")
    n_new = len(kept) + 1  # +1: UNK
    kepset = set(kept)

    def oov_rate(stream):
        tot = len(stream)
        bad = sum(1 for t in stream.tolist() if t not in kepset)
        return bad, tot, (bad / max(1, tot))

    oov_tr = oov_rate(tr)
    oov_va = oov_rate(va)

    # 丢掉的高频 token（让人看清损失了什么）
    dropped_top = [(i, freq[i], kinds[i][1]) for i in other_sorted[:20]]

    print("tokenizer        : %s" % args.tokenizer)
    print("distinct train ids: %d   (train %d tok / val %d tok)" % (len(freq), len(tr), len(va)))
    print("kinds: CJK=%d punct/digit/space=%d special=%d other=%d" % (len(kept_cjk), len(kept_punct), len(kept_special), len(other)))
    print("extra high-freq non-CJK kept: %d (--extra-top-n %d)" % (len(extra), args.extra_top_n))
    print("")
    print("n_vocab NEW = %d   OLD = %d   => %.1f%%" % (n_new, len(freq) + 1, 100.0 * n_new / (len(freq) + 1)))
    print("OOV train: %d/%d = %.3f%%" % (oov_tr[0], oov_tr[1], 100 * oov_tr[2]))
    print("OOV val  : %d/%d = %.3f%%" % (oov_va[0], oov_va[1], 100 * oov_va[2]))
    print("")
    print("top dropped tokens (old_id, train_freq, text):")
    for i, c, s in dropped_top:
        print("    %6d  %7d  %s" % (i, c, ascii(s[:24])))
    print("")
    # 参数影响：embed + head 都是 V*k0
    for label, ke, kh in [("当前最优 k0e=32/k0h=32", 32, 32), ("扩规模 k0e=48/k0h=48", 48, 48)]:
        old_p = (len(freq) + 1) * (ke + kh)
        new_p = n_new * (ke + kh)
        print("params (embed+head, %s): %d -> %d  (saved %d, %.1f%%)"
              % (label, old_p, new_p, old_p - new_p, 100.0 * (old_p - new_p) / old_p))

    if oov_va[2] > args.max_oov:
        print("")
        print("WARNING: val OOV %.3f%% > limit %.1f%% -- must be reported; use --extra-top-n or a larger keep set"
              % (100 * oov_va[2], 100 * args.max_oov))

    # ------------------------------------------------------------------
    # 关键决策数据：**按训练频次裁剪**能把词表压到多小、代价是多少 OOV
    # （纯 CJK 过滤只能砍掉 3.8% —— 因为观测词表本来就有 94.5% 是 CJK）
    # ------------------------------------------------------------------
    print("")
    print("frequency-pruning tradeoff (keep N most frequent TRAIN tokens):")
    print("    keep_N    n_vocab   OOV_train   OOV_val   params(k0e32+k0h32)")
    by_freq = sorted(freq.keys(), key=lambda i: (-freq[i], i))
    for n in [1500, 2000, 3000, 4000, 5000, 6000, 8000, 10920]:
        keep = set(by_freq[:n])
        bt = sum(1 for t in tr.tolist() if t not in keep)
        bv = sum(1 for t in va.tolist() if t not in keep)
        print("    %6d    %6d    %7.3f%%   %7.3f%%   %9d"
              % (n, n + 1, 100.0 * bt / len(tr), 100.0 * bv / len(va), (n + 1) * 64))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out + ".txt", "w", encoding="utf-8") as f:
        for i in kept:
            f.write("%d\n" % i)
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump({
            "tokenizer": args.tokenizer,
            "data_dir": args.data_dir,
            "policy": {"extra_top_n": args.extra_top_n, "cjk_ranges": [[hex(a), hex(b)] for a, b in CJK_RANGES]},
            "n_vocab_new": n_new,
            "n_vocab_old": len(freq) + 1,
            "counts": {"cjk": len(kept_cjk), "punct": len(kept_punct), "special": len(kept_special),
                       "other_dropped": len(other), "extra_kept": len(extra)},
            "oov_train": {"bad": oov_tr[0], "total": oov_tr[1], "rate": oov_tr[2]},
            "oov_val": {"bad": oov_va[0], "total": oov_va[1], "rate": oov_va[2]},
            "dropped_top20": [{"old_id": i, "freq": c, "text": s} for i, c, s in dropped_top],
            "kept_old_ids": kept,
        }, f, ensure_ascii=False, indent=1)
    print("")
    print("wrote %s.txt (%d ids) and %s.json" % (args.out, len(kept), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
