#!/usr/bin/env python3
"""aot_embed.py —— 把「数据集 + 元信息」烧成 Rust 常量（VMLM P8 成员 G）

为什么需要它
------------
AOT 的意义是**不读任何外部文件**。P7-B 只把 `.vmb`（模型/指令流）烧进了 exe，
但训练还需要数据集。`--features "aot train"` 下如果只跑一个训练步、
外层循环还在 std 驱动里，那就不是闭环（实测：跑完 exit=0、无任何输出、
也没有训练发生）。

本工具把数据集也烧成 align(64) 常量，配合 `runtime/src/aot_train.rs`
就能在**完全不读文件**的前提下跑真正的训练闭环。

产物格式（runtime/src/aot_train.rs 消费）
----------------------------------------
    pub struct AotData {
        pub name: &'static [u8],   // NUL 结尾
        pub bytes: &'static [u8],  // 小端 f32 原始字节
    }
    pub static AOT_DATA: &[AotData] = &[ ... ];

与 aot_build.py / aot_multi_build.py 一致的 align(64) + 名字表风格。

用法:
    python tools/aot_embed.py <out.rs> <name>=<file.bin> [<name>=<file.bin> ...]

例（MNIST 训练闭环）:
    python tools/aot_embed.py runtime/src/data_aot.rs ^
        META=examples/mnist/meta_f32.bin ^
        X=examples/mnist/X.bin Y=examples/mnist/Y.bin ^
        XTE=examples/mnist/Xte.bin YTE=examples/mnist/Yte.bin
"""
import sys
import os
import argparse

PER_LINE = 16


def die(msg: str) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return 1


def ident(name: str) -> str:
    out = []
    for ch in name:
        out.append(ch.upper() if ch.isalnum() else "_")
    s = "".join(out)
    if not s or s[0].isdigit():
        s = "D_" + s
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description=".bin datasets -> Rust constant source")
    ap.add_argument("out_rs")
    ap.add_argument("specs", nargs="*", help="name=path.bin")
    args = ap.parse_args()

    if not args.specs:
        return die("no data given; usage: aot_embed.py out.rs name=file.bin ...")

    items = []
    seen = set()
    for s in args.specs:
        if "=" not in s:
            return die(f"spec must be name=path, got {s!r}")
        name, path = s.split("=", 1)
        name, path = name.strip(), path.strip().strip('"')
        if name in seen:
            return die(f"duplicate name {name!r}")
        seen.add(name)
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError as e:
            return die(str(e))
        if len(data) % 4 != 0:
            return die(f"{path}: size {len(data)} not a multiple of 4 (f32 data)")
        items.append((name, data, os.path.abspath(path)))

    outdir = os.path.dirname(os.path.abspath(args.out_rs))
    lines = []
    lines.append("// 本文件由 tools/aot_embed.py 自动生成 —— 请勿手改。")
    lines.append(f"// 内嵌数据块: {len(items)}，总载荷 {sum(len(d) for _, d, _ in items)} 字节")
    for n, d, p in items:
        lines.append(f"//   {n:<8} {len(d):>10} B ({len(d)//4} f32)  <- {os.path.relpath(p, outdir)}")
    lines.append("//")
    lines.append("// 用途: runtime --features aot-train 时被 aot_train.rs include! 进 main.rs，")
    lines.append("// 让训练闭环**不读任何外部文件**。")
    lines.append("")
    lines.append("pub struct AotData {")
    lines.append("    /// NUL 结尾的名字")
    lines.append("    pub name: &'static [u8],")
    lines.append("    /// 小端 f32 原始字节")
    lines.append("    pub bytes: &'static [u8],")
    lines.append("}")
    lines.append("")

    syms = []
    for idx, (name, data, _) in enumerate(items):
        sym = f"AOT_DATA_{ident(name)}"
        lines.append(f"// [{idx}] {name}  ({len(data)} 字节)")
        lines.append("#[repr(C, align(64))]")
        lines.append(f"pub struct AotDataBlob{idx} {{")
        lines.append(f"    pub bytes: [u8; {len(data)}],")
        lines.append("}")
        lines.append("")
        lines.append(f"static {sym}_BLOB: AotDataBlob{idx} = AotDataBlob{idx} {{")
        lines.append("    bytes: [")
        for i in range(0, len(data), PER_LINE):
            chunk = data[i:i + PER_LINE]
            lines.append("        " + " ".join("0x%02x," % b for b in chunk))
        lines.append("    ],")
        lines.append("};")
        lines.append("")
        lines.append(f"static {sym}: &[u8] = &{sym}_BLOB.bytes;")
        lines.append("")
        syms.append((name, sym))

    lines.append("/// 数据表；按名字查找。")
    lines.append("pub static AOT_DATA: &[AotData] = &[")
    for name, sym in syms:
        lines.append(f'    AotData {{ name: b"{name}\\0", bytes: {sym} }},')
    lines.append("];")
    lines.append("")

    if outdir:
        os.makedirs(outdir, exist_ok=True)
    with open(args.out_rs, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    total = sum(len(d) for _, d, _ in items)
    print(f"OK: {len(items)} data blocks, {total} B -> {args.out_rs} "
          f"({os.path.getsize(args.out_rs)} B source)")
    for n, d, _ in items:
        print(f"    {n:<8} {len(d):>10} B ({len(d)//4} f32)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
