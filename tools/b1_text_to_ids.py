#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用某个字符词表把**任意原始文本**编码成 trainer 需要的 id 流（以后换语料直接用）。

用法：
  # 1) 编码单个文本文件 -> ids.bin（一行一个文件也行，按需要）
  python tools\\b1_text_to_ids.py --vocab artifacts\\char_general\\char_vocab_hex.txt ^
      --in some_corpus.txt --out artifacts\\mydata\\train_ids.bin

  # 2) 直接从 JSONL 语料切 train/val（等价于 general_vocab 的第 5 步，但不重新统计字频）
  python tools\\b1_text_to_ids.py --vocab artifacts\\char_general\\char_vocab_hex.txt ^
      --jsonl data\\other.jsonl --text-key text --out-dir artifacts\\other_data --val-frac 0.002

  # 3) 只看 OOV（不写文件）
  python tools\\b1_text_to_ids.py --vocab ... --in corpus.txt --stats-only

统一约定：UNK=0；换行 → </br>（若词表里有）；每个文档末尾加 </s>（若词表里有）。
"""
import argparse
import json
import os
import struct
import sys


def load_vocab(path):
    ch2id = {}
    sent = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            i = int(p[0])
            if p[1] == "char":
                ch2id[chr(int(p[2], 16))] = i
            elif p[1] == "sent":
                sent[p[2]] = i
    return ch2id, sent


def iter_texts(path, text_key):
    keys = [text_key] if text_key else ["completion", "text", "content", "prompt", "document"]
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line[0] == "{":
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                got = None
                for k in keys:
                    if isinstance(o.get(k), str):
                        got = o[k]
                        break
                if got is None:
                    got = " ".join(v for v in o.values() if isinstance(v, str))
                yield got
            else:
                yield line


def encode(txt, ch2id, br, eos):
    out = []
    for ch in txt:
        cid = ch2id.get(ch)
        if cid is None:
            cid = br if (ch == "\n" and br is not None) else 0
        out.append(cid)
    if eos is not None:
        out.append(eos)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--in", dest="inp", default="")
    ap.add_argument("--jsonl", default="")
    ap.add_argument("--text-key", default="text")
    ap.add_argument("--out", default="")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--val-frac", type=float, default=0.002)
    ap.add_argument("--stats-only", action="store_true")
    a = ap.parse_args()
    ch2id, sent = load_vocab(a.vocab)
    br, eos = sent.get("</br>"), sent.get("</s>")
    print("词表: %d 个字符 + %d 个 sentinel" % (len(ch2id), len(sent)), file=sys.stderr)

    if a.inp:
        txt = open(a.inp, "r", encoding="utf-8", errors="replace").read()
        ids = encode(txt, ch2id, br, eos)
        n_oov = sum(1 for i in ids if i == 0)
        print("字符 %d → id %d, UNK=%d (%.4f%%)" % (len(txt), len(ids), n_oov, 100.0 * n_oov / max(1, len(ids))))
        if not a.stats_only and a.out:
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
            with open(a.out, "wb") as f:
                f.write(struct.pack("<%dI" % len(ids), *ids))
            print("已写 %s (%.1f MB)" % (a.out, os.path.getsize(a.out) / 1e6))
        return

    if a.jsonl and (a.out_dir or a.stats_only):
        n_lines = sum(1 for _ in open(a.jsonl, "r", encoding="utf-8", errors="replace"))
        split = max(1, n_lines - max(1, int(n_lines * a.val_frac)))
        tr = va = None
        if a.out_dir and not a.stats_only:
            os.makedirs(a.out_dir, exist_ok=True)
            tr = open(os.path.join(a.out_dir, "train_ids.bin"), "wb")
            va = open(os.path.join(a.out_dir, "val_ids.bin"), "wb")
            # 词表随数据目录放一份（驱动按 <data-dir>/char_vocab_hex.txt 找）
            with open(a.vocab, "r", encoding="utf-8") as fi, \
                 open(os.path.join(a.out_dir, "char_vocab_hex.txt"), "w", encoding="utf-8", newline="\n") as fo:
                fo.write(fi.read())
        tot = unk = 0
        for i, txt in enumerate(iter_texts(a.jsonl, a.text_key)):
            ids = encode(txt, ch2id, br, eos)
            tot += len(ids)
            unk += sum(1 for x in ids if x == 0)
            if tr is not None:
                (tr if i < split else va).write(struct.pack("<%dI" % len(ids), *ids))
        if tr is not None:
            tr.close()
            va.close()
            meta = {"vocab_size": len(ch2id) + len(sent), "chars_per_token": 1.0,
                    "source": os.path.abspath(a.jsonl), "unk_ids": unk,
                    "train_bytes": os.path.getsize(os.path.join(a.out_dir, "train_ids.bin")),
                    "val_bytes": os.path.getsize(os.path.join(a.out_dir, "val_ids.bin"))}
            with open(os.path.join(a.out_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        print("行 %d, id %d, UNK=%d (%.5f%%)" % (n_lines, tot, unk, 100.0 * unk / max(1, tot)))
        return
    print("请给 --in 或 (--jsonl 且 --out-dir)", file=sys.stderr)


if __name__ == "__main__":
    main()
