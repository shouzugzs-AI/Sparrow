#!/usr/bin/env python3
"""aot_build.py —— 把 .vmb 编译成 Rust 常量数组（VMLM P7 成员 B 的工具）

产物是一个 .rs 文件，里面是一段 `#[repr(C, align(64))] struct AotModel` +
`pub static MODEL_BYTES: AotModel = ...`。runtime/src/main.rs 在
`--features aot` 下 `include!` 它，于是 .vmb 的字节被烧进 .exe 的只读段。

为什么不能直接用 `include_bytes!`？
    `include_bytes!` 产出 `&[u8; N]`，只有 1 字节对齐；而 `Vm::load` 把
    常量池按 64B 对齐解释。更要紧的是 .rdata 是只读页，`Vm::load` 会把
    cpool 当可写区用（train/Adam 就地更新参数）—— 所以运行时还得拷一份到
    `vmlm_core::alloc` 拿到的可写堆里。这里生成 align(64) 静态只是为了
    让「构建期」也保持对齐，方便核对。

用法:
    python tools/aot_build.py <in.vmb> <out.rs> [--symbol NAME]
"""
import sys
import os
import argparse

PER_LINE = 16


def main() -> int:
    ap = argparse.ArgumentParser(description=".vmb -> Rust byte-constant source")
    ap.add_argument("vmb", help="input .vmb path")
    ap.add_argument("rs", help="output .rs path")
    ap.add_argument("--symbol", default="AOT_MODEL", help="static symbol name")
    args = ap.parse_args()

    data = open(args.vmb, "rb").read()
    if len(data) < 32:
        print(f"error: {args.vmb} too small ({len(data)} B), not a .vmb", file=sys.stderr)
        return 1
    magic = int.from_bytes(data[0:4], "little")
    if magic != 0x4D4C4D56:  # "VMLM" LE
        print(f"error: {args.vmb} bad magic 0x{magic:08X}, expected 0x4D4C4D56", file=sys.stderr)
        return 1

    rel = os.path.relpath(os.path.abspath(args.vmb), os.path.dirname(os.path.abspath(args.rs)))
    lines = []
    lines.append("// 本文件由 tools/aot_build.py 自动生成 —— 请勿手改。")
    lines.append(f"// 来源: {rel}  ({len(data)} 字节)")
    lines.append("// 用途: runtime --features aot 时被 include! 进 main.rs，把模型烧进 .exe")
    lines.append("")
    lines.append("#[repr(C, align(64))]")
    lines.append("pub struct AotModel {")
    lines.append("    pub bytes: [u8; %d]," % len(data))
    lines.append("}")
    lines.append("")
    lines.append("#[allow(dead_code)]")
    lines.append(f"pub static {args.symbol}: AotModel = AotModel {{")
    lines.append("    bytes: [")
    for i in range(0, len(data), PER_LINE):
        chunk = data[i:i + PER_LINE]
        lines.append("        " + " ".join("0x%02x," % b for b in chunk))
    lines.append("    ],")
    lines.append("};")
    lines.append("")
    lines.append(f"pub const AOT_MODEL_LEN: usize = {len(data)};")
    lines.append("")

    out = os.path.dirname(os.path.abspath(args.rs))
    if out:
        os.makedirs(out, exist_ok=True)
    with open(args.rs, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"OK: {args.vmb} ({len(data)} B) -> {args.rs} "
          f"({os.path.getsize(args.rs)} B source, {len(data)} B payload)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
