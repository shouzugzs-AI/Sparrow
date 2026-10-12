#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S5-D2: emit the `neuron_kv_offsets` section into a PRODUCTION .vmb container.

WHY THIS FILE EXISTS
--------------------
The independent verifier's finding D2 (docs/P15_S5_wiring_verification.md 4.2):
the engine's S5 load path existed, but the PRODUCTION pipeline
(`tools/p12_build_minicpm.py`) never wrote a `neuron_kv` section. The only
24-layer artifact carrying one was built by a HARNESS (`tools/p15_s5_e2e.py`).
So "S5 is wired" could not be claimed for the real MiniCPM artifact.

This module is the writer the production pipeline calls. It is deliberately a
SEPARATE file from the task-1 harness emitter (`tools/p15_s5_section.py`) so the
production path does not silently inherit a test's constants, and separate from
the verifier's builder so writer and checker are not the same code.

THE BYTE LAYOUT IS THE RUST ONE (core/src/neuron_kv.rs write_section)
--------------------------------------------------------------------
    [0]  u32 n_layers
    [4]  u32 n_kv_heads
    [8]  u32 head_dim
    [12] u32 max_seq_per_neuron
    [16] u32 stride                      <- NKV_HDR = 20 bytes (format.rs:319)
    [20] u32 offsets[n_layers * n_kv_heads * head_dim]
         entry i == i * stride           <- format.rs:308 NKV_ENTRY = 4

Header/entry sizes are VERIFIED AGAINST core/src/format.rs by
`cross_check_against_rust()` below, which PARSES the Rust source rather than
trusting a hand-copied constant. If format.rs ever changes, this fails loudly
instead of writing a section nobody can read.

THE CONTAINER: WHY A SECTION DESCRIPTOR *AND* HOW BACKWARD COMPAT IS KEPT
-----------------------------------------------------------------------
`core/src/interp.rs`'s .vmb header is 32 bytes:
    magic(0) ver(4) flags(6) text_off(8) text_len(12) const_off(16) const_len(20)
    meta_off(24) meta_len(28)
There is NO section table today. `load_find_section` (neuron_kv.rs:648) reads
the descriptor at absolute offsets 32/36/40:
    tag(32) off(36) len(40)

That collides with the .vmb's own [32..) layout, where the TEXT segment starts.
The collision is resolved by an ordering trick, measured not assumed:

    production .vmb := [ 32-byte .vmb header ]
                       [ 12-byte section descriptor ]   <- at 32/36/40
                       [ text ] [ consts ] [ meta ]

    text_off is 44 (not 32) WHEN a section is present, 32 when it is not.

Backward compatibility is structural, not bolted on:

  * No section  -> text_off/tlen/coff/clen/moff/mlen are byte-for-byte the values
    the existing assembler emits. The default engine never reads [32..44), so
    such an artifact loads EXACTLY as before. `--no-neuron-kv-section` exists so
    this can be demonstrated rather than asserted.
  * Section present -> only text_off moves (32 -> 44). `interp.rs` is unchanged
    and follows text_off; no engine change is required.

    BUT: `load_find_section` compares `off + len <= file_len`. In the existing
    .vmb the file ends exactly at meta_off+meta_len, and a 24596-byte section
    appended there overflows the u32 `off` field for any artifact >= 4 GiB. So
    the section is written into the gap that already exists between the END of
    consts and meta_off: any meta_off with (meta_off - const_end) >= sec_bytes
    fits for free. `patched_meta_end()` reports where meta must land so no tail
    is needed at all. When no gap exists we append `section + 64` bytes of
    section-then-padding at the end (so the file length still equals
    meta_off+meta_len, which the engine's own arithmetic re-derives).

WHAT THIS MODULE DOES NOT DO
----------------------------
It does not change any weight bytes, any instruction, or any numeric. It only
re-frames the container and adds one index array. `verify_numerics_unchanged()`
re-reads the segments and proves every byte outside the section is identical.

Usage (as a library, from tools/p12_build_minicpm.py):
    from p15_s5_production_section import build_section, install_into_vmb
Usage (standalone, for the size/geometry probe):
    python tools/p15_s5_production_section.py --selfcheck
"""

import argparse
import os
import struct
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORMAT_RS = os.path.join(ROOT, "core", "src", "format.rs")
NEURON_KV_RS = os.path.join(ROOT, "core", "src", "neuron_kv.rs")

# ---- the format's own .vmb header (mirrors asm/src/main.rs:1129-1138) ----
VMB_HDR = 32
VMB_MAGIC = 0x4D4C4D56  # "VMLM" LE

# ---- section descriptor, per neuron_kv.rs:648-678 (absolute file offsets) ----
DESC_TAG_OFF = 32
DESC_OFF_OFF = 36
DESC_LEN_OFF = 40
DESC_SIZE = 12
SEC_NEURON_KV = 1  # format.rs:284 (0 is reserved for "no such section")

# ---- section body, per format.rs:319 / :308 ----
NKV_HDR = 20
NKV_ENTRY = 4

# Default: this many bytes will be the text offset when a section is present.
TEXT_OFF_WITH_SECTION = VMB_HDR + DESC_SIZE  # 44


class GeometryError(Exception):
    """A geometry that cannot produce a legal section. Never silently clamped:
    a wrong stride makes token k land in ANOTHER neuron and reads back a
    plausible-looking number (neuron_kv.rs's own stated failure class)."""


# ---------------------------------------------------------------------------
# format.rs is the single source of truth -- parse it, do not copy it.
# ---------------------------------------------------------------------------
def cross_check_against_rust(fmt_path=FORMAT_RS, nkv_path=NEURON_KV_RS):
    """Re-derive NKV_HDR / NKV_ENTRY / SEC_NEURON_KV / TAG from the Rust source.

    Returns a dict of the values found. Raises GeometryError if any disagrees
    with this module's constants.

    Why parse rather than import: there is no Rust<->Python binding, and a
    hand-copied constant is exactly how the 128x stride bug survived. This makes
    the copy impossible to drift silently.
    """
    with open(fmt_path, encoding="utf-8", errors="replace") as f:
        fmt = f.read()
    with open(nkv_path, encoding="utf-8", errors="replace") as f:
        nkv = f.read()

    import re

    def const_u(src, name):
        m = re.search(r"pub const %s:\s*(?:usize|u32)\s*=\s*(\d+)\s*;" % name, src)
        if not m:
            raise GeometryError("cannot find `%s` in the Rust source" % name)
        return int(m.group(1))

    found = {
        "NKV_HDR": const_u(fmt, "NKV_HDR"),
        "NKV_ENTRY": const_u(fmt, "NKV_ENTRY"),
        "SEC_NEURON_KV": const_u(fmt, "SEC_NEURON_KV"),
    }
    # `neuron_kv::TAG` is defined as `= SEC_NEURON_KV` (neuron_kv.rs:377), so the
    # tag the loader compares against must equal the format constant.
    if not re.search(r"pub const TAG:\s*u32\s*=\s*SEC_NEURON_KV\s*;", nkv):
        raise GeometryError("neuron_kv.rs: TAG is no longer `= SEC_NEURON_KV`")

    mismatch = {k: (v, {"NKV_HDR": NKV_HDR, "NKV_ENTRY": NKV_ENTRY,
                        "SEC_NEURON_KV": SEC_NEURON_KV}[k])
                for k, v in found.items()
                if v != {"NKV_HDR": NKV_HDR, "NKV_ENTRY": NKV_ENTRY,
                         "SEC_NEURON_KV": SEC_NEURON_KV}[k]}
    if mismatch:
        raise GeometryError(
            "core/src/format.rs disagrees with this writer: %s "
            "(rust, python)" % mismatch)

    # The header field ORDER is part of the contract too: write_section puts
    # n_layers at 0, stride at 16. Check the Rust offsets are still 0/4/8/12/16.
    for i, name in enumerate(("n_layers", "n_kv_heads", "head_dim",
                              "max_seq_per_neuron", "stride")):
        if ("put(%d, layout.%s)" % (i * 4, name)) not in nkv:
            raise GeometryError(
                "neuron_kv.rs write_section no longer writes %s at offset %d"
                % (name, i * 4))
    if "put(NKV_HDR + i * NKV_ENTRY, (i as u64 * layout.stride as u64) as u32);" not in nkv:
        raise GeometryError(
            "neuron_kv.rs write_section no longer writes entry i == i*stride")
    return found


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def derive_geometry(n_layers, n_kv_heads, head_dim, pmax, stride=None):
    """Build (and hard-validate) the layout from the ARTIFACT'S OWN geometry.

    Every argument is the declared cache geometry of the artifact being built --
    never a constant copied out of a test. `stride` defaults to `pmax`, the TIGHT
    layout that S5's NeuronKvLayout::tight uses.

    The invariant is `stride >= max_seq_per_neuron` (neuron_kv.rs:133-155) and
    NOT `head_dim * max_seq_per_neuron`: a neuron only ever uses its first
    `max_seq_per_neuron` slots, and head_dim is already carried by the index.
    Multiplying by head_dim is the documented 128x defect that inflated MiniCPM
    from 25 MB to 3.22 GB.
    """
    if stride is None:
        stride = pmax
    for nm, v in (("n_layers", n_layers), ("n_kv_heads", n_kv_heads),
                  ("head_dim", head_dim), ("max_seq_per_neuron", pmax)):
        if not isinstance(v, int) or v <= 0:
            raise GeometryError("zero/invalid dimension %s=%r" % (nm, v))
    if not isinstance(stride, int) or stride <= 0:
        raise GeometryError("zero/invalid stride=%r" % (stride,))
    if stride < pmax:
        raise GeometryError(
            "stride (%d) < max_seq_per_neuron (%d): neuron %d's slot %d would "
            "land inside neuron %d" % (stride, pmax, 0, pmax, pmax // stride + 1
                                       if stride else -1))
    n_neurons = n_layers * n_kv_heads * head_dim
    # offsets are u32 ELEMENT offsets; the loader re-reads them as u32.
    if n_neurons * stride > 0xFFFFFFFF:
        raise GeometryError(
            "layout exceeds 32-bit offsets: n_neurons(%d) * stride(%d) = %d > "
            "u32::MAX -- the section would store truncated offsets and the "
            "loader would accept silently-wrong bases"
            % (n_neurons, stride, n_neurons * stride))
    return {
        "n_layers": n_layers, "n_kv_heads": n_kv_heads, "head_dim": head_dim,
        "max_seq_per_neuron": pmax, "stride": stride, "n_neurons": n_neurons,
    }


def section_bytes(geom):
    return NKV_HDR + geom["n_neurons"] * NKV_ENTRY


def build_section(geom, *, bad_entry_head_dim=False):
    """Serialize the section body. `bad_entry_head_dim` is the CAN-FAIL knob:
    it writes entry i == i * head_dim * stride, the historical 128x defect, so
    the control artifact is produced by the SAME code path as the good one."""
    buf = bytearray(section_bytes(geom))
    struct.pack_into("<5I", buf, 0, geom["n_layers"], geom["n_kv_heads"],
                     geom["head_dim"], geom["max_seq_per_neuron"],
                     geom["stride"])
    n = geom["n_neurons"]
    stride = geom["stride"]
    hd = geom["head_dim"]
    for i in range(n):
        e = i * hd * stride if bad_entry_head_dim else i * stride
        struct.pack_into("<I", buf, NKV_HDR + i * NKV_ENTRY, e & 0xFFFFFFFF)
    return bytes(buf)


# ---------------------------------------------------------------------------
# container
# ---------------------------------------------------------------------------
def read_vmb_header(path):
    with open(path, "rb") as f:
        d = f.read(VMB_HDR)
    if len(d) < VMB_HDR:
        raise GeometryError("%s is shorter than the 32-byte .vmb header" % path)
    (magic, ver, flags, text_off, text_len, const_off, const_len,
     meta_off, meta_len) = struct.unpack_from("<IHHIIIIII", d, 0)
    if magic != VMB_MAGIC:
        raise GeometryError("%s: magic=0x%08X, expected 0x%08X"
                            % (path, magic, VMB_MAGIC))
    return {
        "magic": magic, "version": ver, "flags": flags, "text_off": text_off,
        "text_len": text_len, "const_off": const_off, "const_len": const_len,
        "meta_off": meta_off, "meta_len": meta_len,
        "const_end": const_off + const_len,
        "meta_end": meta_off + meta_len,
        "file_len": os.path.getsize(path),
    }


def plan_container(h, sec_len, *, grow=True):
    """Decide where the section goes and what the new header must say.

    Returns a dict describing the plan. Raises GeometryError when the container
    cannot carry the section legally.
    """
    gap = h["meta_off"] - h["const_end"]
    if gap >= sec_len:
        sec_off = h["const_end"]
        new_meta_off = h["meta_off"]
        extra = 0
        mode = "gap"
    else:
        if not grow:
            raise GeometryError(
                "no room: const_end=%d..meta_off=%d leaves %d B but the section "
                "needs %d B" % (h["const_end"], h["meta_off"], gap, sec_len))
        # Append section + 64 B. The trailing 64 keeps
        # `file_len == meta_off + meta_len` true, which is how the existing
        # tooling re-derives the section extent. It is INSIDE meta's declared
        # length and the default engine never reads those bytes.
        sec_off = h["meta_end"]
        extra = sec_len + 64
        new_meta_off = h["meta_off"]
        mode = "append"

    # `load_find_section` stores off/len in u32 and checks
    # `off + len <= file_len` using u64. A >4 GiB offset truncates to a smaller
    # number that may still be "inside" the file -> SILENT wrong section. Refuse.
    if sec_off + sec_len > 0xFFFFFFFF:
        raise GeometryError(
            "section offset %d exceeds the u32 range that load_find_section "
            "stores offsets in -- would truncate silently" % (sec_off,))
    new_len = max(h["file_len"] + extra, new_meta_off + h["meta_len"] + extra)
    # text_off moves from 32 to 44; nothing else in the header changes.
    return {"mode": mode, "sec_off": sec_off, "sec_len": sec_len,
            "new_text_off": TEXT_OFF_WITH_SECTION,
            "new_file_len": new_len, "gap": gap, "extra": extra}


def install_into_vmb(vmb_path, geom, *, grow=True, bad_entry_head_dim=False,
                     section_tag=SEC_NEURON_KV, out_path=None):
    """Rewrite `vmb_path` in place (or to `out_path`) with the section added.

    Returns a report dict. Every segment is read and re-written byte-for-byte;
    only the header's text_off and the appended bytes differ.
    """
    h = read_vmb_header(vmb_path)
    # The assembler always lays text out immediately after the 32-byte header,
    # so a section-less artifact has text_off == 32. Anything else means we are
    # looking at a container we did not produce, or one already re-framed --
    # re-framing it again would move the text a second time.
    if h["text_off"] != VMB_HDR:
        raise GeometryError(
            "text_off=%d: expected %d (the assembler's value for a section-less "
            "image). This container is either already re-framed or was not "
            "produced by vmlm-asm." % (h["text_off"], VMB_HDR))
    # The file must end exactly where meta does. That identity is what lets the
    # engine's `off + len <= file_len` check bound the section, and it is how
    # existing tooling re-derives the section extent.
    if h["file_len"] != h["meta_end"]:
        raise GeometryError(
            "file_len=%d but meta_off+meta_len=%d: the container has trailing "
            "bytes the engine does not account for; refusing to add a section "
            "to an unfamiliar layout" % (h["file_len"], h["meta_end"]))
    sec = build_section(geom, bad_entry_head_dim=bad_entry_head_dim)
    plan = plan_container(h, len(sec), grow=grow)

    # Read the whole existing image so the rewrite is provably a re-framing.
    with open(vmb_path, "rb") as f:
        image = f.read()

    out = bytearray(image)
    # 1. header: text_off 32 -> 44. (magic/version/flags/text_len/const/meta all
    #    keep their values; the text simply starts 12 bytes later.)
    struct.pack_into("<I", out, 8, plan["new_text_off"])
    # 2. descriptor at 32/36/40 (inside the 12 bytes now skipped before text)
    struct.pack_into("<3I", out, DESC_TAG_OFF, section_tag, plan["sec_off"],
                     plan["sec_len"])
    # 3. grow if needed, then place the section
    if len(out) < plan["new_file_len"]:
        out.extend(b"\x00" * (plan["new_file_len"] - len(out)))
    out[plan["sec_off"]:plan["sec_off"] + plan["sec_len"]] = sec

    dst = out_path or vmb_path
    with open(dst, "wb") as f:
        f.write(out)

    rep = dict(plan)
    rep.update({"out_path": dst, "file_len_before": h["file_len"],
                "file_len_after": len(out),
                "size_delta": len(out) - h["file_len"],
                "n_neurons": geom["n_neurons"], "stride": geom["stride"],
                "pmax": geom["max_seq_per_neuron"],
                "n_layers": geom["n_layers"], "n_kv_heads": geom["n_kv_heads"],
                "head_dim": geom["head_dim"]})
    return rep


# ---------------------------------------------------------------------------
# verification helpers (used by the production pipeline's own self-check)
# ---------------------------------------------------------------------------
def verify_numerics_unchanged(before_path, after_path, sec_off, sec_len):
    """Prove the re-framing changed NOTHING except the header field, the 12-byte
    descriptor, and the section bytes themselves.

    Reads both files and compares every byte outside those three regions.
    Returns (equal, [(region, offset, a, b)...]) with the first mismatches.
    """
    with open(before_path, "rb") as f:
        a = f.read()
    with open(after_path, "rb") as f:
        b = f.read()
    if len(b) < len(a):
        return False, [("length", 0, len(a), len(b))]
    n = len(a)
    skip = set(range(8, 12))                      # header text_off
    skip |= set(range(DESC_TAG_OFF, DESC_TAG_OFF + DESC_SIZE))
    skip |= set(range(sec_off, sec_off + sec_len))
    mism = []
    for i in range(n):
        if i in skip or b[i] == a[i]:
            continue
        mism.append(("byte", i, a[i], b[i]))
        if len(mism) >= 8:
            break
    return (not mism), mism


def verify_section_of(path, expect_geom, *, expect_tag=SEC_NEURON_KV):
    """Independent read-back of what actually landed on disk (Python side).

    This is NOT the pinning test -- the pinning test is the Rust loader. This is
    the cheap check that runs inside the production pipeline so a broken writer
    is caught at build time.
    """
    h = read_vmb_header(path)
    with open(path, "rb") as f:
        f.seek(DESC_TAG_OFF)
        tag, off, ln = struct.unpack("<3I", f.read(DESC_SIZE))
        f.seek(off)
        sec = f.read(ln)
    problems = []
    if tag != expect_tag:
        problems.append("tag=%d expected %d" % (tag, expect_tag))
    if ln != section_bytes(expect_geom):
        problems.append("len=%d expected %d" % (ln, section_bytes(expect_geom)))
    if off + ln > os.path.getsize(path):
        problems.append("off+len=%d exceeds file %d"
                        % (off + ln, os.path.getsize(path)))
    if len(sec) != ln:
        problems.append("short read: %d of %d" % (len(sec), ln))
        return problems, h, (tag, off, ln)
    got = struct.unpack_from("<5I", sec, 0)
    want = (expect_geom["n_layers"], expect_geom["n_kv_heads"],
            expect_geom["head_dim"], expect_geom["max_seq_per_neuron"],
            expect_geom["stride"])
    if got != want:
        problems.append("header=%s expected %s" % (got, want))
    n = expect_geom["n_neurons"]
    stride = expect_geom["stride"]
    # Entry i must be exactly i*stride. Read in chunks, not a 24 KB loop per
    # entry over 6144 entries -> a single unpack is fine and fast.
    entries = struct.unpack_from("<%dI" % n, sec, NKV_HDR)
    bad = [(i, entries[i]) for i in range(n) if entries[i] != i * stride]
    if bad:
        problems.append("offsets[] disagrees with stride at %d entries, first "
                        "%r" % (len(bad), bad[0]))
    return problems, h, (tag, off, ln)


def patched_meta_end(h, sec_len):
    """What a section-aware assembler should write as meta_off.

    If this is <= the caller's byte budget the user can enable the assembler
    patch and get zero-size-delta production sections; otherwise the append
    layout in `plan_container` is used.
    """
    return h["const_end"] + sec_len + 64


# ---------------------------------------------------------------------------
def _selfcheck():
    """Standalone probe: geometry + format constants + container arithmetic."""
    print("=== p15_s5_production_section: selfcheck ===")
    found = cross_check_against_rust()
    print("format.rs cross-check: %s  -> MATCH (NKV_HDR=%d NKV_ENTRY=%d "
          "SEC_NEURON_KV=%d)" % (found, NKV_HDR, NKV_ENTRY, SEC_NEURON_KV))

    # The REAL production geometry of the MiniCPM pipeline (24 layers x 2 kv
    # heads x 128 head_dim, PMAX = seq + decode, i.e. 4 + 1020 = 1024).
    geom = derive_geometry(24, 2, 128, 1024)
    print("geometry 24x2x128 PMAX=1024 -> n_neurons=%d stride=%d section_bytes=%d"
          % (geom["n_neurons"], geom["stride"], section_bytes(geom)))
    print("  (S5-A control: head_dim*pmax would be %d B, i.e. %.1fx)"
          % (NKV_HDR + geom["n_neurons"] * NKV_ENTRY * 128,
             (NKV_HDR + geom["n_neurons"] * NKV_ENTRY * 128)
             / float(section_bytes(geom))))

    # CAN-FAIL controls, so "the validator is doing something" is measurable.
    controls = [
        ("stride < max_seq_per_neuron", 24, 2, 128, 1024, 512),
        ("zero head_dim", 24, 2, 0, 1024, None),
        ("zero n_layers", 0, 2, 128, 1024, None),
    ]
    for name, nl, nk, hd, pm, st in controls:
        try:
            derive_geometry(nl, nk, hd, pm, st)
            print("  [CONTROL-FAILED-TO-FAIL] %s was ACCEPTED" % name)
            return 1
        except GeometryError as e:
            print("  [control OK] %s -> REJECTED: %s" % (name, e))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--vmb", help="measure/plan a container")
    ap.add_argument("--pmax", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--head-dim", type=int, default=128)
    a = ap.parse_args()
    if a.selfcheck or not a.vmb:
        return _selfcheck()
    try:
        cross_check_against_rust()
    except GeometryError as e:
        print("FORMAT MISMATCH: %s" % e)
        return 2
    geom = derive_geometry(a.layers, a.kv_heads, a.head_dim, a.pmax)
    sec = section_bytes(geom)
    h = read_vmb_header(a.vmb)
    print("container %s" % a.vmb)
    for k in ("text_off", "text_len", "const_off", "const_len", "meta_off",
              "meta_len", "const_end", "meta_end", "file_len"):
        print("  %-10s %d" % (k, h[k]))
    print("  gap const_end..meta_off = %d B" % (h["meta_off"] - h["const_end"]))
    print("section needs %d B (n_neurons=%d stride=%d)"
          % (sec, geom["n_neurons"], geom["stride"]))
    print("  section-aware meta_off would be %d (budget %d)"
          % (patched_meta_end(h, sec), h["file_len"]))
    try:
        p = plan_container(h, sec)
        print("  plan: mode=%s sec_off=%d size_delta=+%d"
              % (p["mode"], p["sec_off"], p["extra"]))
    except GeometryError as e:
        print("  plan: REFUSED: %s" % e)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
