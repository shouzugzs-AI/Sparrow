#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""读 .cpk 头部 → 打印驱动需要的结构参数（供打包脚本组装"载入并评估"的命令行）。

用法：
  python tools/cpk_info.py --model artifacts/b1_C3_10k.cpk
  python tools/cpk_info.py --model x.cpk --flags     # 只打印 flags 一行
"""
import argparse
import os
import struct
import sys

NAMES = ["magic", "version", "dim", "k0", "k0_embed", "k0_head", "layers", "n_attn",
         "k0_attn", "n_vocab", "io_sparse", "head_skip", "attn_sparse"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--flags", action="store_true")
    a = ap.parse_args()
    if not os.path.isfile(a.model):
        print("FATAL: 找不到 %s" % a.model)
        return 2
    with open(a.model, "rb") as f:
        head = f.read(13 * 4)
    v = struct.unpack("<13I", head)
    g = dict(zip(NAMES, v))
    if g["magic"] != 0xB1C00DE1:
        print("FATAL: 魔数不对 0x%08X（不是本项目的 .cpk）" % g["magic"])
        return 2
    attn_every = (g["layers"] // g["n_attn"]) if g["n_attn"] > 0 else 0
    flags = ("--sparse-io %d --dim %d --k0 %d --layers %d --no-residual 1 --head-skip %d "
             "--seq 64 --batch 4 --chars-per-token 1.0 --vocab-mode full --attn-every %d "
             "--attn-src-shared 1 --attn-calib 0 --attn-calib-target qk --steps 1 --warmup 1 "
             "--eval-every 1 --log-every 999999 --plateau-patience 0") % (
        g["io_sparse"], g["dim"], g["k0"], g["layers"], g["head_skip"], attn_every)
    if g["io_sparse"]:
        flags += " --k0-embed %d --k0-head %d" % (g["k0_embed"], g["k0_head"])
    if g["n_attn"] > 0:
        flags += " --attn-sparse %d" % g["attn_sparse"]
        if g["attn_sparse"]:
            flags += " --k0-attn %d" % g["k0_attn"]
    if a.flags:
        print(flags)
        return 0
    sz = os.path.getsize(a.model)
    n_f32 = (sz - (13 + 2) * 4) // 4
    print("file      : %s" % a.model)
    print("size      : %d bytes (%.2f MB)" % (sz, sz / 1e6))
    print("geometry  : " + ", ".join("%s=%d" % (k, g[k]) for k in NAMES[2:]))
    print("attn_every: %d" % attn_every)
    print("f32 count : %d" % n_f32)
    print("flags     : %s" % flags)
    return 0


if __name__ == "__main__":
    sys.exit(main())
