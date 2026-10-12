#!/usr/bin/env python3
"""aot_multi_build.py —— 把 N 个 .vmb 打包成一份可切换的 Rust 常量源（VMLM P8 成员 G）

背景 / 为什么需要它
-------------------
P7-B 的 `aot_build.py` 一次只能内嵌**一个**模型：它生成 `model_aot.rs`，
里面是单个 `AOT_MODEL`。换模型必须重跑脚本 + 重新 `cargo build`。
本工具生成 `models_aot.rs`：多个 `align(64)` 常量数组 + 一张名字表，
运行时可**按名字选择**，一个 exe 内嵌 N 个模型。

与 aot_build.py 的关系
----------------------
`aot_build.py` 保持原样（单模型路径仍可用，向后兼容，B 的验收脚本不动）。
本工具是**增量**，不是替换。

输出格式（runtime/src/aot_multi.rs 消费）
----------------------------------------
    pub struct AotEntry {
        pub name: &'static [u8],   // NUL 结尾，便于 no_std 下比较
        pub bytes: &'static [u8],
    }
    pub static AOT_MODELS: &[AotEntry] = &[ ... ];

每个模型字节放在自己的 `#[repr(C, align(64))] static` 里，
再由 `AotEntry` 用 `&'static [u8]` 引用 —— 这样既保留 64B 对齐，
又能用切片表达「长度可变」的多个模型。

为什么单个大数组不行
--------------------
一个 `[u8; N]` 常量数组无法表达 N 个**长度不同**的模型；
用「拼接 + 偏移表」可以，但会让每个模型的对齐只保证在第一个上
（第一个对齐不代表后续偏移都对齐）。逐模型独立 `align(64)` 更稳。

用法:
    python tools/aot_multi_build.py <out.rs> <name>=<in.vmb> [<name>=<in.vmb> ...]
    python tools/aot_multi_build.py <out.rs> <in.vmb> ...        # 名字取文件名
    python tools/aot_multi_build.py <out.rs> -- <manifest.txt>   # 每行 name=path
选项:
    --order-first NAME   把 NAME 放到表首位（运行时无参数时的默认模型）
"""
import sys
import os
import argparse

PER_LINE = 16


def die(msg: str) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return 1


def read_vmb(path: str) -> bytes:
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 32:
        raise ValueError(f"{path} too small ({len(data)} B), not a .vmb")
    magic = int.from_bytes(data[0:4], "little")
    if magic != 0x4D4C4D56:  # "VMLM" LE
        raise ValueError(f"{path} bad magic 0x{magic:08X}, expected 0x4D4C4D56")
    return data


def safe_relpath(path: str, start: str) -> str:
    """相对路径；跨盘符时 os.path.relpath 会抛 ValueError，这里回落到绝对路径。

    实测踩到：把输出写到 C:\\...\\Temp 而输入在 D:\\ 时，
    `os.path.relpath` 直接 ValueError("path is on mount 'D:', start on mount 'C:'")。
    生成物里的这行只是给人看的注释，跨盘符时用绝对路径即可。
    """
    try:
        return os.path.relpath(path, start)
    except ValueError:
        return path


def ident(name: str) -> str:
    """把模型名转成合法的 Rust 标识符片段（大写、非字母数字 -> _）。"""
    out = []
    for ch in name:
        out.append(ch.upper() if ch.isalnum() else "_")
    s = "".join(out)
    if not s or s[0].isdigit():
        s = "M_" + s
    return s


def emit_model(lines, name: str, data: bytes, rel: str, idx: int) -> str:
    """生成单个模型的 static 数组，返回其符号名。"""
    sym = f"AOT_BYTES_{ident(name)}"
    lines.append(f"// [{idx}] {name}  <- {rel}  ({len(data)} 字节)")
    lines.append("#[repr(C, align(64))]")
    lines.append(f"pub struct AotBlob{idx} {{")
    lines.append(f"    pub bytes: [u8; {len(data)}],")
    lines.append("}")
    lines.append("")
    lines.append(f"static {sym}_BLOB: AotBlob{idx} = AotBlob{idx} {{")
    lines.append("    bytes: [")
    for i in range(0, len(data), PER_LINE):
        chunk = data[i:i + PER_LINE]
        lines.append("        " + " ".join("0x%02x," % b for b in chunk))
    lines.append("    ],")
    lines.append("};")
    lines.append("")
    lines.append(f"static {sym}: &[u8] = &{sym}_BLOB.bytes;")
    lines.append("")
    return sym


def main() -> int:
    ap = argparse.ArgumentParser(description="multiple .vmb -> one Rust source")
    ap.add_argument("out_rs", help="output .rs path")
    ap.add_argument("specs", nargs="*", help="name=path.vmb  |  path.vmb  |  @manifest")
    ap.add_argument("--order-first", default=None,
                    help="model name to place first (default model at runtime)")
    args = ap.parse_args()

    specs = list(args.specs)

    # @manifest：每行 "name=path"，# 开头是注释
    expanded = []
    for s in specs:
        if s.startswith("@"):
            mf = s[1:]
            with open(mf, "r", encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln or ln.startswith("#"):
                        continue
                    expanded.append(ln)
        else:
            expanded.append(s)
    specs = expanded

    if not specs:
        return die("no models given; usage: aot_multi_build.py out.rs name=in.vmb ...")

    models = []          # (name, data, abspath)
    seen = set()
    for s in specs:
        if "=" in s:
            name, path = s.split("=", 1)
        else:
            path = s
            name = os.path.splitext(os.path.basename(path))[0]
        name = name.strip()
        path = path.strip().strip('"')
        if not name:
            return die(f"empty model name in spec {s!r}")
        if name in seen:
            return die(f"duplicate model name {name!r}")
        seen.add(name)
        try:
            data = read_vmb(path)
        except (OSError, ValueError) as e:
            return die(str(e))
        models.append((name, data, os.path.abspath(path)))

    names = [m[0] for m in models]
    if args.order_first:
        if args.order_first not in names:
            return die(f"--order-first {args.order_first!r} not among {names}")
        i = names.index(args.order_first)
        models.insert(0, models.pop(i))

    outdir = os.path.dirname(os.path.abspath(args.out_rs))
    lines = []
    lines.append("// 本文件由 tools/aot_multi_build.py 自动生成 —— 请勿手改。")
    lines.append(f"// 内嵌模型数: {len(models)}，总载荷 {sum(len(d) for _, d, _ in models)} 字节")
    for n, d, p in models:
        lines.append(f"//   {n:<16} {len(d):>8} B   <- {safe_relpath(p, outdir)}")
    lines.append("//")
    lines.append("// 用途: runtime --features aot 时被 aot_multi.rs include! 进 main.rs。")
    lines.append("// 运行时用 --model=<name> 选择；无参数则用表首项。")
    lines.append("")
    lines.append("pub struct AotEntry {")
    lines.append("    /// NUL 结尾的名字，no_std 下便于按字节比较")
    lines.append("    pub name: &'static [u8],")
    lines.append("    pub bytes: &'static [u8],")
    lines.append("}")
    lines.append("")

    syms = []
    for idx, (name, data, path) in enumerate(models):
        rel = safe_relpath(path, outdir)
        syms.append((name, emit_model(lines, name, data, rel, idx)))

    lines.append("/// 名字表 + 载荷表；顺序即 `--model` 的匹配顺序。")
    lines.append("pub static AOT_MODELS: &[AotEntry] = &[")
    for name, sym in syms:
        lines.append(f'    AotEntry {{ name: b"{name}\\0", bytes: {sym} }},')
    lines.append("];")
    lines.append("")
    lines.append(f"pub const AOT_MODEL_COUNT: usize = {len(models)};")
    lines.append(
        "pub const AOT_TOTAL_BYTES: usize = "
        + " + ".join(str(len(d)) for _, d, _ in models)
        + ";"
    )
    lines.append("")

    if outdir:
        os.makedirs(outdir, exist_ok=True)
    with open(args.out_rs, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    total = sum(len(d) for _, d, _ in models)
    print(f"OK: {len(models)} models, {total} B payload -> {args.out_rs} "
          f"({os.path.getsize(args.out_rs)} B source)")
    for n, d, _ in models:
        print(f"    {n:<16} {len(d):>8} B")
    return 0


if __name__ == "__main__":
    sys.exit(main())
