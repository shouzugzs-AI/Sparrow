#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B1 — STEP 2b: **character-level** vocabulary cut straight from the dataset.

WHY (and how this differs from the BPE cut in `b1_vocab_cjk.py`):
  The previous attempt filtered *BPE token types*, and measured that the observed
  token vocabulary is already 94.5% CJK -> only a 3.8% cut. But the user's point is
  different and correct: the dataset contains fewer than ~4k distinct **characters**.
  So instead of pruning BPE types, build a **character table** from the data:

      id 0        = <unk>
      id 1..S     = special markers actually present (</s>, <end>, </br>, ...)
      id S+1..    = every character observed, by frequency

  and RE-ENCODE the token streams into character-id streams. The result is consumable
  by the existing driver with **no driver change**: `--vocab-mode full --data-dir <out>`
  (the ids are already 1:1).

CONSEQUENCES that are reported, not hidden:
  * sequence length grows: measured chars/token = 1.294, so `--seq 32` now covers ~25 BPE
    tokens' worth of text; use `--seq 40` for the same text span.
  * CE per position is NOT comparable across tokenizations -> the report gives
    **bits/character** = CE / chars-per-token / ln2.

Usage:
  python tools/b1_char_vocab.py --data-dir artifacts/b1_scaling ^
      --tokenizer "<外部模型目录>/MiniCPM5-1B/tokenizer.json" --out artifacts/b1_char
"""

import argparse
import json
import os
import sys
import unicodedata
from collections import Counter

import numpy as np
from tokenizers import Tokenizer

ROOT = str(Path(__file__).resolve().parents[1])

# 内部哨兵：把多字符的"特殊标识符"折叠成**单个**字符级 id
SENTINELS = {
    "</s>": 0xE000,
    "<end>": 0xE001,
    "</br>": 0xE002,
    "<s>": 0xE003,
    "<pad>": 0xE004,
}
SENT_NAME = {v: k for k, v in SENTINELS.items()}


def cjk_kind(ch):
    o = ord(ch)
    if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0xF900 <= o <= 0xFAFF or 0x20000 <= o <= 0x2EBEF:
        return "CJK"
    if 0x3000 <= o <= 0x303F or 0xFF00 <= o <= 0xFFEF or 0x2E80 <= o <= 0x2EFF:
        return "CJK-punct/fullwidth"
    if 0xE000 <= o <= 0xE00F:
        return "special"
    return unicodedata.category(ch)


def read_u32(path):
    b = open(path, "rb").read()
    return np.frombuffer(b, dtype="<u4").astype("int64")


def write_u32(path, arr):
    open(path, "wb").write(np.asarray(arr, dtype="<u4").tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "artifacts", "b1_scaling"))
    ap.add_argument("--tokenizer", default=r"<外部模型目录>/MiniCPM5-1B/tokenizer.json")
    ap.add_argument("--out", default=os.path.join(ROOT, "artifacts", "b1_char"))
    ap.add_argument("--from", dest="src", default="train", choices=["train", "all"],
                    help="字表从哪来：train（诚实，val 可能有 OOV）/ all（train+val，OOV=0 但轻微泄漏）")
    args = ap.parse_args()

    tk = Tokenizer.from_file(args.tokenizer)
    tr_tok = read_u32(os.path.join(args.data_dir, "train_ids.bin"))
    va_tok = read_u32(os.path.join(args.data_dir, "val_ids.bin"))

    # token 流 -> 文本（meta 里 roundtrip_bad_docs=0，所以这一步是无损的）
    tr_txt = tk.decode(tr_tok.tolist(), skip_special_tokens=False)
    va_txt = tk.decode(va_tok.tolist(), skip_special_tokens=False)

    # 多字符标识符 -> 哨兵字符（保持"原子性"：一个标识符 = 一个 id）
    for s, cp in SENTINELS.items():
        tr_txt = tr_txt.replace(s, chr(cp))
        va_txt = va_txt.replace(s, chr(cp))

    src_txt = tr_txt if args.src == "train" else tr_txt + va_txt
    freq = Counter(src_txt)

    # 字表：0=UNK，然后 1.. 按频次降序（并列按码点升序 —— 完全确定性）
    chars = sorted(freq.keys(), key=lambda c: (-freq[c], ord(c)))
    vocab = ["<unk>"] + chars          # new id = 下标
    ch2id = {c: i for i, c in enumerate(vocab) if i > 0}
    unk = 0

    def encode(txt):
        return np.array([ch2id.get(c, unk) for c in txt], dtype="<u4")

    tr_ids = encode(tr_txt)
    va_ids = encode(va_txt)

    # OOV（只在 val 上可能非零）
    va_oov = int((va_ids == unk).sum())
    tr_oov = int((tr_ids == unk).sum())

    os.makedirs(args.out, exist_ok=True)
    write_u32(os.path.join(args.out, "train_ids.bin"), tr_ids)
    write_u32(os.path.join(args.out, "val_ids.bin"), va_ids)
    meta = {
        "source_data_dir": args.data_dir,
        "tokenizer": args.tokenizer,
        "level": "char",
        "vocab_from": args.src,
        "vocab_size": len(vocab),
        "train_chars": len(tr_ids),
        "val_chars": len(va_ids),
        "chars_per_token_train": len(tr_txt) / len(tr_tok),
        "chars_per_token_val": len(va_txt) / len(va_tok),
        "train_oov": tr_oov,
        "val_oov": va_oov,
        "sentinel_markers": {hex(v): k for k, v in SENTINELS.items()},
        "add_special_tokens": False,
    }
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump({"vocab": [("<unk>" if i == 0 else (SENT_NAME.get(ord(c), c))) for i, c in enumerate(vocab)],
               "freq": {c: freq[c] for c in chars},
               "meta": meta},
              open(os.path.join(args.out, "char_vocab.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    # ---- 给 Rust 的**无转义**映射文件（生成模式要 id -> 文本）----
    # 每行: <id>/t<kind>/t<value>   kind=char 时 value 是十六进制码点；kind=sent 时是标识符原文。
    # 为什么不用 JSON：驱动里没有 serde，而 JSON 字符串转义（引号/反斜杠/换行）很容易解析错。
    hexmap = os.path.join(args.out, "char_vocab_hex.txt")
    with open(hexmap, "w", encoding="utf-8") as f:
        f.write("0\tsent\t<unk>/n")
        for i, c in enumerate(vocab):
            if i == 0:
                continue
            name = SENT_NAME.get(ord(c))
            if name is not None:
                f.write("%d\tsent\t%s\n" % (i, name))
            else:
                f.write("%d\tchar\t%X\n" % (i, ord(c)))

    # ---- 报告 ----
    kinds = Counter(cjk_kind(c) for c in chars)
    print("source            : %s (%s)" % (args.data_dir, args.src))
    print("token -> char     : train %d tok -> %d char (x%.4f) | val %d -> %d (x%.4f)"
          % (len(tr_tok), len(tr_txt), len(tr_txt) / len(tr_tok),
             len(va_tok), len(va_txt), len(va_txt) / len(va_tok)))
    print("distinct chars    : train-only %d | train+val %d" % (len(set(tr_txt)), len(set(tr_txt) | set(va_txt))))
    print("char categories   : %s" % dict(sorted(kinds.items(), key=lambda kv: -kv[1])))
    print("")
    print("n_vocab (char)    : %d   (= 1 UNK + %d chars)" % (len(vocab), len(chars)))
    print("OOV train         : %d/%d = %.5f%%" % (tr_oov, len(tr_ids), 100.0 * tr_oov / max(1, len(tr_ids))))
    print("OOV val           : %d/%d = %.5f%%" % (va_oov, len(va_ids), 100.0 * va_oov / max(1, len(va_ids))))
    print("")
    print("top 20 chars by freq:")
    for c in chars[:20]:
        print("    %7d  U+%04X  %s" % (freq[c], ord(c), ascii(c)))
    if va_oov:
        vo = Counter(c for c in va_txt if c not in ch2id)
        print("")
        print("val chars NOT in the train-derived table (%d types): %s"
              % (len(vo), ascii("".join(list(vo.keys())[:60]))))
    print("")
    # 参数量对比（当前最优 k0e=32/k0h=32 + head-skip ⇒ 3*V*32；main/attention 与词表无关）
    for label, ke, kh in [("k0e=32/k0h=32", 32, 32), ("扩规模 k0e=48/k0h=48", 48, 48)]:
        p_new = 3 * len(vocab) * ke
        p_old = 3 * 10921 * ke
        print("params embed+head+skip (%s): BPE %d -> CHAR %d  (saved %d, %.1f%%)"
              % (label, p_old, p_new, p_old - p_new, 100.0 * (p_old - p_new) / p_old))
    print("")
    print("wrote %s\\{train_ids.bin,val_ids.bin,meta.json,char_vocab.json,char_vocab_hex.txt}" % args.out)
    print("driver usage: --vocab-mode full --data-dir %s" % args.out)
    # 参考：BPE 口径的 bits/char 换算系数
    print("")
    print("cross-tokenization metric: bits/char = CE_nats / chars_per_token / ln2")
    print("    BPE  run: divide by %.4f ; CHAR run: divide by 1.0" % (len(tr_txt) / len(tr_tok)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
