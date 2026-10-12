#!/usr/bin/env python3
"""
.pt → .vmb 转换器（决策 1：工具层体积不受限，怎么方便怎么来）

产物是「极简二进制数值对文件」：{name_hash, shape, dtype, bytes} 数组。
引擎侧不需要任何 pickle/ZIP 解析逻辑 —— 这正是把重活放在工具层的红利。

用法:
    python pt2vmb.py model.pt out.vmb
    python pt2vmb.py model.pt out.vmb --text asm_out.vmb   # 合并已有汇编产物
"""
import argparse
import struct
import sys
import zlib

import torch

MAGIC = 0x4D4C_4D56
VERSION = 1
INS_SIZE = 16
CONST_ALIGN = 64

DT_F32, DT_F16, DT_I8, DT_I32 = 0, 1, 2, 3


def hash32(name: str) -> int:
    """FNV-1a 32，必须与 Rust format.rs::hash32 一致"""
    h = 0x811C_9DC5
    for b in name.encode():
        h ^= b
        h = (h * 0x0100_0193) & 0xFFFF_FFFF
    return h


def flatten(obj, prefix=""):
    """把嵌套 dict/list 压平成 (名字, 张量)"""
    if isinstance(obj, torch.Tensor):
        yield prefix.rstrip("."), obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from flatten(v, f"{prefix}{k}.")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from flatten(v, f"{prefix}{i}.")
    # 标量/其他类型直接忽略（引擎只吃张量）


DTYPE_MAP = {
    torch.float32: (DT_F32, 4),
    torch.float16: (DT_F16, 2),
    torch.float64: (DT_F32, 4),  # 降精度到 f32
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pt")
    ap.add_argument("out")
    ap.add_argument("--text", help="已有 .vmb 的 TEXT 段（来自 vmlm-asm）")
    ap.add_argument("--f32", action="store_true", default=True, help="统一转 f32")
    args = ap.parse_args()

    sd = torch.load(args.pt, map_location="cpu", weights_only=False)
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()

    consts = []
    for name, t in flatten(sd):
        if not t.is_floating_point():
            continue
        t = t.detach().to(torch.float32).contiguous()
        shape = list(t.shape)
        rows = shape[0] if len(shape) >= 1 else 1
        cols = 1
        for d in shape[1:]:
            cols *= d
        consts.append((name, rows, cols, DT_F32, t.numpy().tobytes()))

    print(f"读到 {len(consts)} 个张量", file=sys.stderr)

    # TEXT 段
    text = b""
    if args.text:
        with open(args.text, "rb") as f:
            blob = f.read()
        t_off, t_len = struct.unpack_from("<II", blob, 8)
        text = blob[t_off:t_off + t_len]

    # CONST 段
    const_data = bytearray()
    offsets = []
    for _, _, _, _, data in consts:
        offsets.append(len(const_data))
        const_data.extend(data)
        while len(const_data) % CONST_ALIGN:
            const_data.append(0)

    # META 段
    meta = bytearray()
    meta.extend(struct.pack("<I", len(consts)))
    for i, (name, rows, cols, dtype, data) in enumerate(consts):
        meta.extend(struct.pack("<IIIII", offsets[i], len(data), rows, cols, dtype))
        meta.extend(struct.pack("<I", hash32(name)))

    header = 32
    text_off = header
    const_off = text_off + len(text)
    meta_off = const_off + len(const_data)

    out = bytearray()
    out.extend(struct.pack("<IHH", MAGIC, VERSION, 0))
    out.extend(struct.pack("<IIIIII", text_off, len(text), const_off,
                           len(const_data), meta_off, len(meta)))
    out.extend(text)
    out.extend(const_data)
    out.extend(meta)

    with open(args.out, "wb") as f:
        f.write(out)
    print(f"OK: {len(consts)} 常量, {len(out)} 字节 → {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
