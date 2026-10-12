# -*- coding: utf-8 -*-
"""
One-shot filler for section 2.2 of docs/P17_baseline_verify.md.

WHY THIS EXISTS
---------------
The 27-item P17 baseline has to be transcribed from a raw verify_all log into a
Markdown table. Doing that by hand invites the exact failure this project keeps
hitting: a number that was retyped rather than copied, or a FAIL quietly
becoming a PASS. So: parse the log mechanically, diff it against the P16
comparison baseline, and rewrite ONLY the marked region of the report.

WHAT IT WILL NOT DO
-------------------
It refuses to touch the report unless the log actually reached the Summary
block. A truncated run (which is exactly what happened to the Lead's first
attempt at `_p17_baseline_verify.txt`) would otherwise look like "0 failures"
and could be mistaken for a clean baseline. That is the single most dangerous
outcome here, so it is a hard precondition, not a warning.

Usage:
    python tools/p17_fill_baseline.py <verify_all_log.txt>
    python tools/p17_fill_baseline.py            # uses _p17_baseline_run.txt
"""

import datetime
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = os.path.join(ROOT, "docs", "P17_baseline_verify.md")
DEFAULT_LOG = os.path.join(ROOT, "_p17_baseline_run.txt")

BEGIN = "<!-- BEGIN P17-BASELINE-AUTO -->"
END = "<!-- END P17-BASELINE-AUTO -->"

# The P16 comparison column, copied verbatim from the report's own 2.1 table.
# Kept here so the diff is computed against the same values a reader sees.
P16 = [
    ("P1 GEMM kernel regression", "PASS"),
    ("P1 exp_f accuracy", "PASS"),
    ("P3 MNIST training", "PASS"),
    ("P4 RNN recurrent state", "PASS"),
    ("P5 Transformer attention", "PASS"),
    ("P6 dynamic sparsity", "PASS"),
    ("P8 RNN BPTT training", "PASS"),
    ("P8 Transformer training", "PASS"),
    ("P7-A backward ops + AdamW", "PASS"),
    ("P7-A lead cross-check", "PASS"),
    ("P7-C ESERNN migration", "PASS"),
    ("P7-D GEMM benchmark", "PASS"),
    ("P7-D packed crash guard", "PASS"),
    ("P7-E sparse thin-GEMM", "PASS"),
    ("P10 runtime loop control", "PASS"),
    ("P10 MiniCPM layer0 vs HF", "PASS"),
    ("P10 AOT embed + weight mmap", "PASS"),
    ("P10 RoPE + sin/cos", "PASS"),
    ("P11 optimization (elim + quant)", "PASS"),
    ("P12 MiniCPM5-1B full model", "PASS"),
    ("P13 EMB dtype-aware + state row", "PASS"),
    ("P13 one .vmb serves any prompt", "PASS"),
    ("P13 dtype audit still fails loud", "PASS"),
    ("P13 incremental decode (KV cache)", "PASS"),
    ("P13 int8 weights + dequant GEMM", "FAIL(exit=1)"),
    ("P13 looped layers (runtime base)", "PASS"),
    ("P7-B AOT standalone", "FAIL(exit=1)"),
]

ROW_RE = re.compile(
    r"^(P\d+.*?)\s+(PASS|FAIL\(exit=-?\d+\)|SKIP|MISSING)\s*(.*)$")


def read_log_text(path):
    """Read a verify_all log, decoding whatever the capture actually produced.

    MEASURED, and it cost a false alarm: `Tee-Object` under Windows PowerShell
    5.1 writes **UTF-16LE with a BOM**, not UTF-8. The first version of this
    reader opened with `encoding="utf-8"`, which decoded 21642 bytes into 21604
    replacement characters and then reported

        "log has NO '===== Summary =====' block -- the run did not finish"

    -- i.e. a perfectly good, COMPLETE baseline was announced as truncated. The
    refusal logic was right and load-bearing; the reader underneath it was
    wrong, and a wrong reader makes a correct guard produce a wrong answer.

    So: sniff the BOM. A BOM is the only self-describing signal here, and both
    PowerShell 5.1 (UTF-16LE) and pwsh 7 (UTF-8, often no BOM) are in play in
    this project. Do NOT guess beyond that.
    """
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:2] == b"\xff\xfe":
        return raw.decode("utf-16", errors="replace"), "utf-16 (BOM fffe)"
    if raw[:2] == b"\xfe\xff":
        return raw.decode("utf-16-be", errors="replace"), "utf-16-be (BOM feff)"
    if raw[:3] == b"\xef\xbb\xbf":
        return raw.decode("utf-8-sig", errors="replace"), "utf-8 (BOM efbbbf)"
    # No BOM: try strict UTF-8 first; a NUL-laden stream is UTF-16LE without BOM.
    try:
        return raw.decode("utf-8"), "utf-8 (no BOM)"
    except UnicodeDecodeError:
        if raw.count(b"\x00") > len(raw) // 8:
            return raw.decode("utf-16-le", errors="replace"), "utf-16-le (no BOM)"
        return raw.decode("utf-8", errors="replace"), "utf-8 (lossy)"


def parse_summary(text):
    """Return (rows, non_passing, engine_size) from a verify_all log.

    Raises ValueError if the log did not complete -- see the module docstring.

    NOTE on the separator whitespace: the first version of this required TWO or
    more spaces (`\\s{2,}`) between the item name and the result, on the theory
    that the table is column-aligned. That theory is false at exactly one width:
    `P13 incremental decode (KV cache)` is 33 characters, so its row has the
    result starting one column later and only ONE space separates them. The
    parser silently returned 26 of 27 rows -- i.e. it would have dropped a real
    PASS from the table, which is the same class of silent omission this project
    hard-fails on. A single `\\s+` is the correct, width-independent separator.
    (Guarded by the explicit row-count check in main(), which is what caught it.)
    """
    idx = text.find("===== Summary =====")
    if idx < 0:
        raise ValueError(
            "log has NO '===== Summary =====' block -- the run did not finish. "
            "A truncated run must NEVER be reported as a baseline (the Lead's "
            "first attempt failed exactly this way). Refusing to fill.")
    block = text[idx:]
    rows = []
    for line in block.splitlines():
        m = ROW_RE.match(line.rstrip())
        if m:
            rows.append((m.group(1).strip(), m.group(2), m.group(3).strip()))
    np_ = re.search(r"Non-passing items:\s*(\d+)", block)
    sz = re.search(r"Engine core size:\s*(\d+)\s*bytes", block)
    if len(rows) == 0:
        raise ValueError("Summary block present but 0 rows parsed -- refusing "
                         "to report an empty table as a baseline")
    return rows, (int(np_.group(1)) if np_ else None), (int(sz.group(1)) if sz else None)


def clean_note(note):
    """Make a log note safe to embed in the Markdown table.

    The verify_all log contains the ASSEMBLER's Chinese diagnostics, which the
    Windows console (GBK) has already mangled into mojibake + U+FFFD before it
    ever reached the log file. Reproducing that mojibake in the report would
    (a) be unreadable and (b) inject replacement characters into a UTF-8 doc.

    So: keep the ASCII/diagnostic part, which is what carries the meaning
    (`NotFound`, `code: 2`, the .vml filename), and replace the mojibake run
    with an explicit marker rather than silently dropping it. Silently dropping
    would be the same "quiet omission" failure this tool already guards against.
    """
    # U+FFFD, or the Latin-1/CJK garbage that GBK-mangled UTF-8 produces.
    if "\ufffd" in note or re.search(r"[\u4e00-\u9fff][\u4e00-\u9fff]{2,}", note):
        # Collapse runs of CJK/mojibake into one marker, keep the ASCII skeleton.
        note = re.sub(r"[^\x20-\x7e]+", " <console-GBK mojibake omitted> ", note)
        note = re.sub(r"( <console-GBK mojibake omitted> )+",
                      " <console-GBK mojibake omitted> ", note)
        note = note.strip()
    note = note.replace("|", "\\|")
    return note


def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOG
    if not os.path.isfile(log_path):
        print("FATAL: log not found: %s" % log_path)
        print("Run the suite first, e.g. (ONLY in a quiet window):")
        print("  powershell -NoProfile -ExecutionPolicy Bypass "
              "-File tools\\verify_all.ps1 -SkipAot *>&1 | "
              "Tee-Object _p17_baseline_run.txt")
        return 2

    with open(log_path, "rb") as _f:
        pass
    text, how = read_log_text(log_path)
    print("log decoded as %s" % how)

    try:
        rows, non_passing, engine_size = parse_summary(text)
    except ValueError as e:
        print("FATAL: %s" % e)
        return 2

    print("parsed %d item(s) from %s" % (len(rows), os.path.basename(log_path)))
    print("Non-passing items (from log): %s" % non_passing)
    print("Engine core size (from log): %s" % engine_size)

    if len(rows) != len(P16):
        print("FATAL: expected %d items, parsed %d -- the suite's item list "
              "changed. Refusing to fill a table that would silently drop or "
              "add a row." % (len(P16), len(rows)))
        return 2

    # Diff by POSITION as well as by NAME: a positional shift means a row was
    # reordered, which would make a name-only diff mis-attribute regressions.
    regressions, fixed, same = [], [], 0
    for i, ((p16_name, p16_res), (name, res, _note)) in enumerate(zip(P16, rows), 1):
        if name != p16_name:
            print("WARN: row %d name differs: report has %r, log has %r"
                  % (i, p16_name, name))
        if p16_res.startswith("PASS") and not res.startswith("PASS"):
            regressions.append((i, name, p16_res, res))
        elif not p16_res.startswith("PASS") and res.startswith("PASS"):
            fixed.append((i, name, p16_res, res))
        else:
            same += 1

    counts = {}
    for _n, res, _note in rows:
        key = "PASS" if res.startswith("PASS") else res
        counts[key] = counts.get(key, 0) + 1

    print("\nPASS=%d FAIL=%d SKIP=%d MISSING=%d  (total %d)"
          % (counts.get("PASS", 0),
             sum(v for k, v in counts.items() if k.startswith("FAIL")),
             counts.get("SKIP", 0), counts.get("MISSING", 0), len(rows)))
    if regressions:
        print("\nREGRESSIONS (P16 PASS -> P17 non-PASS): %d" % len(regressions))
        for i, n, a, b in regressions:
            print("   #%d %s: %s -> %s" % (i, n, a, b))
    else:
        print("\nREGRESSIONS: 0")
    if fixed:
        print("\nNow-passing (was non-PASS in P16): %d" % len(fixed))
        for i, n, a, b in fixed:
            print("   #%d %s: %s -> %s" % (i, n, a, b))

    # ---- build the replacement block ------------------------------------
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_mtime = datetime.datetime.fromtimestamp(
        os.path.getmtime(log_path)).strftime("%Y-%m-%d %H:%M:%S")
    out = []
    out.append(BEGIN)
    out.append("")
    out.append("**P17 基线（P17 改动之后的树）** — 由 "
               "`tools/p17_fill_baseline.py` 机械填入，非手工转录。")
    out.append("")
    out.append("- 日志：`%s`（文件 mtime %s）"
               % (os.path.basename(log_path), log_mtime))
    out.append("- 填表时间：%s" % stamp)
    out.append("- 命令：`tools\\verify_all.ps1 -SkipAot`（串行，静默窗口内）")
    out.append("")
    out.append("| # | 项目 | P16 对照（§2.1） | **P17 实测** | 是否回归 |")
    out.append("|---:|---|---|---|---|")
    for i, ((p16_name, p16_res), (name, res, note)) in enumerate(
            zip(P16, rows), 1):
        if p16_res.startswith("PASS") and not res.startswith("PASS"):
            verdict = "**回归**"
        elif not p16_res.startswith("PASS") and res.startswith("PASS"):
            verdict = "已修复"
        elif not p16_res.startswith("PASS"):
            verdict = "非回归（预先存在）"
        else:
            verdict = "—"
        note = clean_note(note or "")
        if len(note) > 90:
            note = note[:90] + "…"
        out.append("| %d | %s | %s | **%s** | %s |"
                   % (i, name, p16_res, res, verdict))
        if note:
            out.append("| | | | <sub>%s</sub> | |" % note)
    out.append("")
    out.append("**P17 汇总**：PASS %d / FAIL %d / SKIP %d / MISSING %d（共 %d）"
               % (counts.get("PASS", 0),
                  sum(v for k, v in counts.items() if k.startswith("FAIL")),
                  counts.get("SKIP", 0), counts.get("MISSING", 0), len(rows)))
    if engine_size is not None:
        out.append("")
        out.append("**Engine core size**：%d bytes（上限 84480；P16 对照 74752）"
                   % engine_size)
    out.append("")
    out.append("**回归判定**：P16 PASS 而 P17 非 PASS 的项 = %d。" % len(regressions))
    out.append("")
    out.append(END)

    with io.open(REPORT, "r", encoding="utf-8") as f:
        report = f.read()
    if BEGIN not in report or END not in report:
        print("FATAL: report lacks the %s / %s markers -- refusing to guess "
              "where the P17 table goes." % (BEGIN, END))
        return 2
    pre, rest = report.split(BEGIN, 1)
    _old, post = rest.split(END, 1)
    new_report = pre + "\n".join(out) + post
    with io.open(REPORT, "w", encoding="utf-8") as f:
        f.write(new_report)
    print("\nwrote P17 baseline block into %s" % os.path.relpath(REPORT, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
