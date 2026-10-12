#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B1 scaling law — STEP 1: encode the dataset with the MiniCPM vocabulary.

WHY THIS FILE EXISTS AND WHAT IT DOES **NOT** DO
------------------------------------------------
The task says "MiniCPM 词表（已有 BPE，直接用，不用重新实现）". This script therefore
does **not** implement any BPE. It loads the official MiniCPM tokenizer
(`MiniCPM5-1B/tokenizer.json`, the HF `tokenizers` implementation) and uses it.

Why not `core/src/bpe.rs`: that path is the project's OWN BPE1 container, whose
ids are u16 (`n_pieces = 256 + n_merges`, max 65535) and whose pieces are a pure
function of the merge table. MiniCPM's vocab is ~73k pieces, and no tool in the
repo reads `tokenizer.json` (grep: 0 hits under tools/ and server/src). So there
is no existing path that can consume MiniCPM's real vocabulary; the honest move
is to use the official tokenizer for the ENCODING step and hand the resulting
token ids to the Rust trainer (which then needs no tokenizer at all).

WHAT IS MEASURED / CHECKED HERE (so the numbers can be trusted)
---------------------------------------------------------------
  * vocab size actually loaded from tokenizer.json
  * per-line and total token counts
  * the 90/10 split is **by document (line)**, not by token -- a token-level split
    would leak the continuation of every training document into validation.
  * round-trip: decode(encode(text)) must reproduce the text byte-for-byte
    (reported token-by-token as well: count of mismatching documents)

Usage:
  python tools/b1_scaling_encode.py \
      --jsonl "<数据集目录>/这是数据集但是格式要转换.jsonl" \
      --tokenizer "<外部模型目录>/MiniCPM5-1B/tokenizer.json" \
      --out-dir <repo>/artifacts/b1_scaling
"""

import argparse
import json
import os
import struct
import sys


def load_tokenizer(path):
    from tokenizers import Tokenizer

    tk = Tokenizer.from_file(path)
    return tk


def documents(jsonl_path):
    """One document per JSONL line: the turn contents in order, joined by '\\n'."""
    docs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print("FATAL: line %d is not JSON: %s" % (ln, e))
                sys.exit(9)
            if "turns" not in obj:
                print("FATAL: line %d has no 'turns'" % ln)
                sys.exit(9)
            parts = []
            for turn in obj["turns"]:
                c = turn.get("content")
                if c is None:
                    continue
                parts.append(c)
            docs.append("\n".join(parts))
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ratio", type=float, default=0.9)
    args = ap.parse_args()

    tk = load_tokenizer(args.tokenizer)
    vocab_size = tk.get_vocab_size()
    print("tokenizer        : %s" % args.tokenizer)
    print("vocab size       : %d" % vocab_size)

    docs = documents(args.jsonl)
    print("documents        : %d" % len(docs))
    print("chars            : %d" % sum(len(d) for d in docs))

    # ---- document-level 90/10 split (no token-level leakage) ----
    n_train = int(round(len(docs) * args.ratio))
    # deterministic: first n_train documents (file order), no shuffling
    train_docs = docs[:n_train]
    val_docs = docs[n_train:]
    print(
        "split            : train %d docs / val %d docs (document-level, ratio %.2f)"
        % (len(train_docs), len(val_docs), args.ratio)
    )

    def encode_all(ds):
        ids_all = []
        per_doc = []
        for d in ds:
            enc = tk.encode(d, add_special_tokens=False)
            per_doc.append(len(enc.ids))
            ids_all.extend(enc.ids)
        return ids_all, per_doc

    train_ids, train_per = encode_all(train_docs)
    val_ids, val_per = encode_all(val_docs)

    # ---- round-trip check (the tokenizer must be information-preserving) ----
    bad = 0
    for d in docs:
        enc = tk.encode(d, add_special_tokens=False)
        if tk.decode(enc.ids, skip_special_tokens=False) != d:
            bad += 1
    print("round-trip       : %d/%d documents byte-identical" % (len(docs) - bad, len(docs)))

    total = len(train_ids) + len(val_ids)
    print("tokens total     : %d" % total)
    print("tokens train     : %d  (%.2f%% of all tokens)" % (len(train_ids), 100.0 * len(train_ids) / total))
    print("tokens val       : %d  (%.2f%% of all tokens)" % (len(val_ids), 100.0 * len(val_ids) / total))
    print(
        "tokens/doc       : train mean %.1f min %d max %d | val mean %.1f min %d max %d"
        % (
            sum(train_per) / max(1, len(train_per)),
            min(train_per) if train_per else 0,
            max(train_per) if train_per else 0,
            sum(val_per) / max(1, len(val_per)),
            min(val_per) if val_per else 0,
            max(val_per) if val_per else 0,
        )
    )
    # distinct ids actually used -> how much of the 73k vocab this dataset touches
    used = set(train_ids) | set(val_ids)
    print("distinct ids used: %d / %d (%.2f%% of the vocab)" % (len(used), vocab_size, 100.0 * len(used) / vocab_size))
    idmax = max(used) if used else 0
    print("max id           : %d" % idmax)

    os.makedirs(args.out_dir, exist_ok=True)
    for name, ids in (("train_ids.bin", train_ids), ("val_ids.bin", val_ids)):
        p = os.path.join(args.out_dir, name)
        with open(p, "wb") as f:
            f.write(struct.pack("<%dI" % len(ids), *ids))
        print("wrote %s (%d ids, %d bytes)" % (p, len(ids), 4 * len(ids)))
    meta = {
        "tokenizer": args.tokenizer,
        "vocab_size": vocab_size,
        "jsonl": args.jsonl,
        "docs_total": len(docs),
        "docs_train": len(train_docs),
        "docs_val": len(val_docs),
        "tokens_total": total,
        "tokens_train": len(train_ids),
        "tokens_val": len(val_ids),
        "distinct_ids_used": len(used),
        "roundtrip_bad_docs": bad,
        "add_special_tokens": False,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("wrote meta.json")
    if bad:
        print("FATAL: round-trip failed for %d documents" % bad)
        return 9
    return 0


if __name__ == "__main__":
    sys.exit(main())
