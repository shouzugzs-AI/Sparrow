#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从**原始文本语料**（JSONL / 纯文本）构建「通用字符词表」，并顺带产出可直接训练的 id 数据包。

为什么需要它：
  * 现有 `b1_char_vocab.py` 是从 MiniCPM tokenizer 流**解码**出字，只适用于那套数据；
  * 换预训练语料（原始文本）后，需要一个**通用词表** + 一个「原文 → char id」的编码器。

产物（`--out-vocab` 下）：
    char_vocab_hex.txt   标准格式 `id<TAB>kind<TAB>value`（id 0 = UNK，按频率降序；与现有驱动完全兼容）
    char_freq.tsv        rank / hex / 字符 / 次数 / 累计覆盖率（以后要重截词表直接从它来）
    meta.json            来源 / 总字数 / 词表大小 / 覆盖率 / **留出段 OOV**
（`--out-data` 下）训练器直接可用的一包：
    train_ids.bin, val_ids.bin, meta.json, char_vocab_hex.txt

用法：
  python tools\\b1_general_vocab.py --jsonl data\\wiki_zh.jsonl --text-key completion ^
      --out-vocab artifacts\\char_general --out-data artifacts\\zh_wiki
"""
import argparse
import collections
import json
import os
import struct
import sys
import time

# 与 trainer 的约定保持一致：UNK = 0；这些 sentinel 放高 id（不占常用字区间）
SENTINELS = ["<unk>", "</s>", "<end>", "</br>", "<s>", "<pad>"]


def iter_texts(path, text_key):
    """流式产出文本片段：JSONL 取 text_key（自动兼容 completion/text/content/...）；否则整行当文本。"""
    keys = [text_key] if text_key else ["completion", "text", "content", "prompt", "document"]
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
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
                    # 兜底：把所有字符串字段拼起来
                    got = " ".join(v for v in o.values() if isinstance(v, str))
                yield got
            else:
                yield line
            if i % 200000 == 0 and i:
                print("  ...已读 %d 行" % i, file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="语料（.jsonl 或纯文本）")
    ap.add_argument("--text-key", default="completion")
    ap.add_argument("--out-vocab", required=True)
    ap.add_argument("--out-data", default="")
    ap.add_argument("--target-coverage", type=float, default=0.9999, help="取到多少覆盖率（默认 99.99%%）")
    ap.add_argument("--max-chars", type=int, default=8192, help="字符数上限")
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--val-frac", type=float, default=0.002, help="留出做 OOV 估计/验证的比例（按行）")
    ap.add_argument("--holdout-lines", type=int, default=0, help="用末尾 N 行做留出（0 = 按 val-frac 推算）")
    ap.add_argument("--data-max-lines", type=int, default=0, help="id 数据包只取前 N 行（0=全部）；词表统计不受影响")
    a = ap.parse_args()

    t0 = time.time()
    # 先数总行数以便切分留出段（文件 500MB，数一遍行很快；也可直接按 val_frac 计数）
    print("[1/5] 统计语料规模 ...", file=sys.stderr)
    n_lines = 0
    total_chars = 0
    with open(a.jsonl, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            n_lines += 1
            if line.strip():
                total_chars += len(line)
    print("    行数=%d  原始字节≈%.1f MB" % (n_lines, total_chars / 1e6), file=sys.stderr)

    hold = a.holdout_lines or max(1, int(n_lines * a.val_frac))
    split_at = max(1, n_lines - hold)
    print("[2/5] 只对前 %d 行统计字频（末尾 %d 行留出，用于诚实估计 OOV） ..." % (split_at, hold), file=sys.stderr)

    freq = collections.Counter()
    for i, txt in enumerate(iter_texts(a.jsonl, a.text_key)):
        if i >= split_at:
            break
        freq.update(txt)
    tot = sum(freq.values())
    print("    统计到 %d 个不同字符 / %d 个字符实例" % (len(freq), tot), file=sys.stderr)

    # 选字：按频率降序取到目标覆盖率（并受 max-chars 限制）
    ordered = [c for c, _ in freq.most_common()]
    keep, cum = [], 0
    for c in ordered:
        if len(keep) >= a.max_chars:
            break
        keep.append(c)
        cum += freq[c]
        if cum / tot >= a.target_coverage:
            break
    print("[3/5] 选定 %d 个字符（覆盖 %.5f%%）" % (len(keep), 100.0 * cum / tot), file=sys.stderr)

    os.makedirs(a.out_vocab, exist_ok=True)
    # char_vocab_hex.txt：id 0 = <unk>，然后 sentinel，然后按频率降序的字
    lines_v = ["0\tsent\t<unk>"]
    nid = 1
    for s in SENTINELS[1:]:
        lines_v.append("%d\tsent\t%s" % (nid, s))
        nid += 1
    ch2id = {}
    for c in keep:
        ch2id[c] = nid
        cp = ord(c)
        if cp <= 0xFFFF:
            lines_v.append("%d\tchar\t%04X" % (nid, cp))
        else:
            # 扩展平面（emoji 等）用 8 位十六进制，驱动按 u32 解析
            lines_v.append("%d\tchar\t%08X" % (nid, cp))
        nid += 1
    vocab_size = nid
    with open(os.path.join(a.out_vocab, "char_vocab_hex.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines_v) + "\n")

    with open(os.path.join(a.out_vocab, "char_freq.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("rank\tchar\tcodepoint\tcount\tcum_coverage\n")
        run = 0
        for r, c in enumerate(keep, 1):
            run += freq[c]
            f.write("%d\t%s\tU+%04X\t%d\t%.6f\n" % (r, c, ord(c), freq[c], run / tot))

    # 留出段：OOV 率（这是"通用词表"是否够用的关键指标）
    print("[4/5] 在留出段上估计 OOV ...", file=sys.stderr)
    oov = 0
    nchr = 0
    for i, txt in enumerate(iter_texts(a.jsonl, a.text_key)):
        if i < split_at:
            continue
        for ch in txt:
            nchr += 1
            if ch not in ch2id and ch != "\n":
                oov += 1
    oov_rate = (oov / nchr) if nchr else 0.0
    print("    留出 %d 行 / %d 字符：OOV=%d (%.5f%%)" % (hold, nchr, oov, 100 * oov_rate), file=sys.stderr)

    meta = {
        "source": os.path.abspath(a.jsonl),
        "text_key": a.text_key,
        "lines_total": n_lines,
        "lines_counted": split_at,
        "lines_holdout": hold,
        "chars_counted": tot,
        "distinct_chars": len(freq),
        "vocab_chars": len(keep),
        "vocab_size": vocab_size,
        "target_coverage": a.target_coverage,
        "coverage": cum / tot,
        "holdout_chars": nchr,
        "holdout_oov": oov,
        "holdout_oov_rate": oov_rate,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(a.out_vocab, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 可选：产出训练器直接可用的 id 数据包
    if a.out_data:
        print("[5/5] 生成 train_ids.bin / val_ids.bin ...", file=sys.stderr)
        os.makedirs(a.out_data, exist_ok=True)
        tr = open(os.path.join(a.out_data, "train_ids.bin"), "wb")
        va = open(os.path.join(a.out_data, "val_ids.bin"), "wb")
        unk = 0
        # 数据包切分：默认全量（train=前 split_at 行，val=末尾 hold 行）；
        # 给了 --data-max-lines 则只用前 N 行，并按 val_frac 在其中切 train/val。
        if a.data_max_lines > 0:
            cap = a.data_max_lines
            tr_end = max(1, int(cap * (1.0 - a.val_frac)))
            take_until = cap
        else:
            tr_end = split_at
            take_until = 1 << 62
        for i, txt in enumerate(iter_texts(a.jsonl, a.text_key)):
            if i >= take_until:
                break
            ids = []
            for ch in txt:
                cid = ch2id.get(ch)
                if cid is None:
                    if ch == "\n":
                        cid = ch2id.get("</br>", 0)
                    else:
                        cid = 0
                        unk += 1
                ids.append(cid)
            ids.append(ch2id.get("</s>", 0))
            if i < tr_end:
                tr.write(struct.pack("<%dI" % len(ids), *ids))
            else:
                va.write(struct.pack("<%dI" % len(ids), *ids))
        tr.close()
        va.close()
        dmeta = {
            "vocab_size": vocab_size,
            "chars_per_token": 1.0,
            "source": os.path.abspath(a.jsonl),
            "train_bytes": os.path.getsize(os.path.join(a.out_data, "train_ids.bin")),
            "val_bytes": os.path.getsize(os.path.join(a.out_data, "val_ids.bin")),
            "unk_written": unk,
            "vocab_file": os.path.relpath(os.path.join(a.out_vocab, "char_vocab_hex.txt"), a.out_data)
            if False
            else "char_vocab_hex.txt",
        }
        with open(os.path.join(a.out_data, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(dmeta, f, ensure_ascii=False, indent=2)
        # 词表也复制一份进数据目录（驱动按 <data-dir>/char_vocab_hex.txt 找）
        with open(os.path.join(a.out_data, "char_vocab_hex.txt"), "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines_v) + "\n")
        print("    train=%.1f MB  val=%.1f MB  UNK写出的个数=%d" % (
            dmeta["train_bytes"] / 1e6, dmeta["val_bytes"] / 1e6, unk), file=sys.stderr)

    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print("耗时 %.1f s" % (time.time() - t0), file=sys.stderr)


if __name__ == "__main__":
    main()
