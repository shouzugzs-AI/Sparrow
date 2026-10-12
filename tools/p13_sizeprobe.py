#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P13 Task B: measure the REAL byte cost of engine regions.

WHY THIS TOOL EXISTS
    The engine budget is 53,312 B and the current build is 53,248 B: 64 B of
    headroom. Any new feature needs bytes, and guessing at their cost is how
    you end up rewriting a half-finished patch. The handover note records the
    method: stub the region out, rebuild, read `Engine core size`, and take the
    difference. This script mechanizes exactly that, one region at a time, and
    restores the source byte-for-byte afterwards.

WHY IT EDITS WITH PYTHON (encoding='utf-8', newline preserved)
    A `Get-Content -Raw | Set-Content` round trip on interp.rs destroyed
    `#[repr(C)]` in P10 (GBK misdecode + BOM + eaten newlines) and made every
    instruction read as op=0. So: read bytes, decode utf-8 explicitly, write
    utf-8 with newline='' so Python never rewrites line endings.

    Every patch is asserted to match exactly once, and the file is restored
    from an in-memory copy in a finally block. A mis-targeted patch raises
    instead of silently measuring the wrong thing.

Usage:
    python tools/p13_sizeprobe.py --list
    python tools/p13_sizeprobe.py --probe trig_rope
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = str(Path(__file__).resolve().parents[1])
INTERP = os.path.join(ROOT, "core", "src", "interp.rs")
KERNELS = os.path.join(ROOT, "core", "src", "kernels.rs")

# ---------------------------------------------------------------------------
# Probes. Each entry: (name, file, description, [(find, replace), ...])
# The replacement must be a syntactically-valid stub that still type-checks.
# ---------------------------------------------------------------------------
TRIG_ROPE_OLD = """                            let period = TWO_PI / inv;
                            let pr = if period >= 1.0e6 {
                                // Period far beyond the position range: the
                                // angle is already small, skip the modulo.
                                pos
                            } else {
                                pos - period * kernels::floor_f(pos / period)
                            };
                            let ang = pr * inv;
                            let co = kernels::cos_f(ang);
                            let si = kernels::sin_f(ang);
                            let x1 = *sr.add(i);
                            let x2 = *sr.add(i + half);
                            *dr.add(i) = x1 * co - x2 * si;
                            *dr.add(i + half) = x1 * si + x2 * co;"""

# Stub: keep the loop shape (so the loop itself is still measured as present)
# but remove the reduction + both transcendental calls.
TRIG_KEEP_LOOP = """                            let ang = pos * inv;
                            let co = 1.0f32;
                            let si = ang;
                            let x1 = *sr.add(i);
                            let x2 = *sr.add(i + half);
                            *dr.add(i) = x1 * co - x2 * si;
                            *dr.add(i + half) = x1 * si + x2 * co;"""

# Stub: whole RoPE body becomes an empty loop -> measures reduction+trig.
def stub_fn(text, sig, ret):
    """Replace a function body with a trivial one, by brace counting.

    `sig` must be the literal opening line (up to and including its `{`).
    Raises unless it occurs exactly once.
    """
    if text.count(sig) != 1:
        raise RuntimeError("fn anchor matched %d times: %r" % (text.count(sig), sig))
    j = text.index(sig) + len(sig) - 1  # index of '{'
    d = 0
    k = j
    while k < len(text):
        if text[k] == "{":
            d += 1
        elif text[k] == "}":
            d -= 1
            if d == 0:
                break
        k += 1
    else:
        raise RuntimeError("unbalanced braces for %r" % sig)
    return text[:j + 1] + "\n    let _ = x;\n    " + ret + "\n}" + text[k + 1:]


def apply_sincos_stub(text):
    text = stub_fn(text, "pub fn sin_f(x: f32) -> f32 {", "0.0f32")
    text = stub_fn(text, "pub fn cos_f(x: f32) -> f32 {", "1.0f32")
    return text


def apply_print_stub(text):
    """Stub print_u32 / print_f32 (different signatures, so not stub_fn)."""
    return _stub_braced(text, "pub unsafe fn print_u32(v: u32) {", "let _ = v;", "print_u32")


def _stub_braced(text, sig, body, what):
    if text.count(sig) != 1:
        raise RuntimeError("%s anchor count %d" % (what, text.count(sig)))
    j = text.index(sig) + len(sig) - 1
    d = 0
    k = j
    while True:
        if text[k] == "{":
            d += 1
        elif text[k] == "}":
            d -= 1
            if d == 0:
                break
        k += 1
    return text[:j + 1] + "\n    " + body + "\n}" + text[k + 1:]


def apply_print_f32_stub(text):
    return _stub_braced(text, "pub unsafe fn print_f32(p: *const f32, n: usize) {",
                        "let _ = (p, n);", "print_f32")


def apply_print_u32_stub(text):
    return _stub_braced(text, "pub unsafe fn print_u32(v: u32) {",
                        "let _ = v;", "print_u32")


def apply_print_both_stub(text):
    return apply_print_f32_stub(apply_print_u32_stub(text))


PROBES = [
    dict(
        name="rope_trig",
        file=INTERP,
        desc="RoPE: Cody-Waite-period reduction + sin_f/cos_f calls (keep loop)",
        subs=[(TRIG_ROPE_OLD, TRIG_KEEP_LOOP)],
    ),
    dict(
        name="rope_whole",
        file=INTERP,
        desc="RoPE: entire per-element body (reduction + sin + cos)",
        subs=[(TRIG_ROPE_OLD, "                            let ang = pos * inv;\n                            let _ = ang;")],
    ),
    dict(
        name="print_bodies",
        file=INTERP,
        desc="print_f32 + print_u32 bodies (debug-only path)",
        subs=[],
        fn=apply_print_both_stub,
    ),
    dict(
        name="print_f32_only",
        file=INTERP,
        desc="print_f32 body only",
        subs=[],
        fn=apply_print_f32_stub,
    ),
    dict(
        name="print_u32_only",
        file=INTERP,
        desc="print_u32 body only (used by the load-time guards too)",
        subs=[],
        fn=apply_print_u32_stub,
    ),
    # --- attribution for the 2048 B: which part actually costs bytes? ---
    # rope_trig (trig replaced, loop kept) saved 0 B, but rope_whole (loop body
    # emptied) saved 2048 B. So the cost is NOT the transcendental calls --
    # but "0 B" for removing two function calls is itself surprising, and a
    # surprising number is a measurement problem until proven otherwise.
    # These two probes bracket it.
    dict(
        name="rope_body_only",
        file=INTERP,
        desc="RoPE: empty the per-element body, keep BOTH loops (rows x half)",
        subs=[(TRIG_ROPE_OLD, "")],
    ),
    dict(
        name="rope_no_rowloop",
        file=INTERP,
        desc="RoPE: keep inner loop only, drop outer row loop",
        subs=[("""                    for row in 0..r {
                        let pos = pos0 + row as f32;
                        let sr = cs.add(row * hd);
                        let dr = cd.add(row * hd);
                        for i in 0..half {""",
               """                    for row in 0..r {
                        let pos = pos0 + row as f32;
                        let sr = cs.add(row * hd);
                        let dr = cd.add(row * hd);
                        for i in 0..half {
                            let _ = (dr, sr);""")],
    ),
    # --- sin_f / cos_f THEMSELVES (this is where the bytes actually are) ---
    # Measured: replacing the CALL SITES saves 0 B; stubbing the FUNCTION
    # BODIES saves 1536 B. That kills the "inline the trig away" idea and
    # makes "RoPE must keep real sin/cos" a hard constraint (white paper
    # 16.4b: f32 large-angle reduction has a physical floor).
    dict(
        name="sincos_bodies",
        file=KERNELS,
        desc="sin_f + cos_f bodies only (callers untouched)",
        subs=[],
        fn=apply_sincos_stub,
    ),
]


def read_text(path):
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:3] == b"\xef\xbb\xbf":
        raise RuntimeError("refusing to touch a BOM file: %s" % path)
    return raw.decode("utf-8")


def write_text(path, text):
    # newline='' => no translation of the line endings we already hold.
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    # METHODOLOGY (measured the hard way): if the written content is
    # byte-identical to what is already on disk, the mtime may not advance,
    # cargo then reuses the previous object and `Engine core size` reports a
    # STALE number. During Task B a restore right after the sin_f/cos_f stub
    # reported 51712 B (the stubbed size) because of exactly this. Touching
    # the file removes the ambiguity; every probe below then reports a number
    # that provably came from the source currently on disk.
    os.utime(path, None)


def engine_size(out):
    m = re.search(r"Engine core size:\s*(\d+)\s*bytes", out)
    return int(m.group(1)) if m else None


def build():
    p = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-File", os.path.join(ROOT, "tools", "build_all.ps1")],
        cwd=ROOT, capture_output=True, text=True, errors="replace",
    )
    return p.stdout + p.stderr


def probe(entry):
    path = entry["file"]
    original = read_text(path)
    try:
        text = original
        for find, repl in entry["subs"]:
            n = text.count(find)
            if n != 1:
                raise RuntimeError(
                    "patch anchor matched %d times (expected 1) for probe %s"
                    % (n, entry["name"]))
            text = text.replace(find, repl)
        if entry.get("fn"):
            text = entry["fn"](text)
        write_text(path, text)
        out = build()
        if "FAILED" in out:
            return None, "BUILD FAILED"
        return engine_size(out), None
    finally:
        write_text(path, original)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--probe")
    args = ap.parse_args()

    if args.list:
        for e in PROBES:
            print("%-14s %s" % (e["name"], e["desc"]))
        return 0

    base_out = build()
    base = engine_size(base_out)
    print("baseline engine core size: %s" % base)
    if base is None:
        print(base_out[-3000:])
        return 1

    if args.probe:
        chosen = [e for e in PROBES if e["name"] == args.probe]
    else:
        chosen = PROBES

    print()
    print("%-16s %8s %8s   %s" % ("probe", "stubbed", "saved", "what was removed"))
    print("-" * 96)
    for e in chosen:
        sz, err = probe(e)
        if sz is None:
            print("%-16s %8s %8s   %s" % (e["name"], "BUILD-FAIL", "-", e["desc"]))
            continue
        print("%-16s %8d %7d B   %s" % (e["name"], sz, base - sz, e["desc"]))
    print()

    # confirm the restore is byte-exact
    again = engine_size(build())
    print("re-check after restore: %s  %s" % (again, "OK" if again == base else "MISMATCH!"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
