#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
O6-A: OP_MAX (34) and OP_TOPK_IDX (42) exercised through the REAL engine.

WHY THIS FILE EXISTS
    O6 is autoregressive generation, and generation is "feed decode's argmax
    back as the next token". Before this work there was NO argmax in the engine
    at all. The gap had three independent layers, and a test that only proves
    the happy path would have caught none of the interesting ones:

      1. core/src/interp.rs had no arm for op 34 / op 42, so a program using
         them fell into the catch-all `exit(5) unimpl op`.
      2. asm/src/main.rs listed MAX and TOPK_IDX in UNIMPLEMENTED_OPS and
         PANICKED at assembly time -- so the assembler exposed the opcode names
         in its table (main.rs "MAX" => OP_MAX) while refusing to emit them.
         `tools/p10_opcode_gaps.py` calls this a "user-visible trap".
      3. The assembler had no shape inference for them, so the output slot
         stayed 0x0 and the reduction result was written NOWHERE -- silently.
         Layer 3 does not fail; it makes a generator emit the same token
         forever. That is the dangerous one.

    So this test runs real .vml programs through vmlm-asm -> vmlm-runtime and
    compares against HAND-COMPUTED values. It is not a unit test against a
    reimplementation of the kernel; the numbers below were computed by hand in
    the comments next to each case.

WHAT IS DELIBERATELY COVERED (each case exists because of a specific bug it
would catch -- a test that cannot fail is not evidence):

  * ALL-NEGATIVE row for MAX. Catches the `m = 0.0` seed. That one-liner is
    tempting and it reports 0.0 as the max of a row like [-5,-3,-9], i.e. a
    plausible wrong number with no error.
  * TIE for TOPK_IDX. Catches `>=` instead of `>` (or any unstable scan), which
    picks the HIGHEST index. Generation must be reproducible, so ties go to the
    LOWEST index.
  * ALL-NEGATIVE row for TOPK_IDX. Catches argmax done on raw i32 BIT PATTERNS
    instead of numeric f32: IEEE-754 is monotonic over non-negatives, so bit
    comparison looks equivalent while inverting the entire negative range --
    argmax would return the WORST element. Here the correct answer is index 1
    and bit-comparison would return index 0.
  * NEGATIVE and NaN and FRACTIONAL row indices into OP_EMB. `as usize` on a
    negative or NaN f32 SATURATES TO 0 in Rust, and the old OP_EMB discarded the
    table's row count entirely, so an out-of-range index silently read garbage
    (or row 0) instead of failing. The verifier reproduced the silent case on
    the real engine: token id 99 against a 4-row table exited 0 and printed
    zeros. Every one of those must now be exit 9.
  * 0x0 output slot -> exit 9 rather than a silently-dropped result (layer 3).
  * MAX is SEPARATE from TOPK_IDX (value vs index) -- asserted, because O6-B
    uses MAX as a degenerate-generation confidence probe.

Usage:
    python tools/o6_argmax_check.py

Exit 0 = every case matched its hand-computed expectation.
"""
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = str(Path(__file__).resolve().parents[1])
REL = os.path.join(ROOT, "target", "release")
ASM = os.path.join(REL, "vmlm-asm.exe")
ENG = os.path.join(REL, "vmlm-runtime.exe")
DIR = os.path.join(ROOT, "examples", "o6")

# Engine exit codes this project uses.
EXIT_OK = 0
EXIT_UNIMPL_OP = 5
EXIT_GUARD = 9

_fails = []
_passes = []


def write_const(name, arr):
    """Write a little-endian f32 constant, byte-exact (no BOM, no text mode)."""
    import numpy as np
    path = os.path.join(DIR, name)
    data = np.ascontiguousarray(np.asarray(arr, dtype="<f4"))
    for _ in range(40):
        try:
            with open(path, "wb") as f:
                f.write(data.tobytes())
            return path
        except PermissionError:
            time.sleep(0.25)
    raise RuntimeError("could not write const %s (locked by another process)" % path)


def _fingerprint(path):
    """Identify the exact binary a result came from.

    WHY THIS IS NOT DECORATION: this test runs on a SHARED workspace where
    several people rebuild the engine concurrently. An earlier run of this file
    reported PASS 14 / FAIL 4 against CORRECT engine code, purely because the
    engine was relinked mid-run. That near-miss cost the Lead a diagnosis
    cycle, and the failure report named no binary, so it was not reproducible
    from the output alone. Every run now prints which asm and engine produced
    the numbers, so a stale-binary artifact is visible instead of mysterious.
    """
    if not os.path.exists(path):
        return "%s: MISSING" % path
    import hashlib
    with open(path, "rb") as f:
        data = f.read()
    st = os.stat(path)
    return "%s size=%d sha256=%s mtime=%s" % (
        os.path.basename(path), len(data),
        hashlib.sha256(data).hexdigest()[:16],
        time.strftime("%H:%M:%S", time.localtime(st.st_mtime)))


def assemble(tag, vml_src, consts=()):
    vml = "%s.vml" % tag
    vmb = "%s.vmb" % tag
    # Pre-check every constant file. The assembler PANICS with a bare exit code
    # (3221226505) when a const file is missing -- reproduced by the Lead -- so
    # without this a typo'd filename reports as an unexplained panic dump
    # instead of a named error. Cheap to check, and it keeps this harness's
    # failures self-explanatory.
    for spec in consts:
        parts = spec.split(":")
        if len(parts) >= 2:
            src = os.path.join(DIR, parts[1])
            if not os.path.exists(src):
                return None, ("const file MISSING for `%s`: %s (the assembler "
                              "panics on this rather than reporting it)"
                              % (parts[0], src))
    with open(os.path.join(DIR, vml), "w", encoding="ascii", newline="\n") as f:
        f.write(vml_src)
    # Retry: another session's vmlm-asm.exe can hold its output .vmb open, and
    # a transient "拒绝访问 (os error 5)" from that is contention, not a failure.
    for _ in range(40):
        p = subprocess.run([ASM, vml, vmb] + list(consts), cwd=DIR,
                           capture_output=True)
        msg = (p.stdout + p.stderr).decode("utf-8", "replace")
        if p.returncode != 0 and ("os error 5" in msg or "拒绝访问" in msg
                                  or "Access is denied" in msg):
            time.sleep(0.25)
            continue
        break
    if p.returncode != 0:
        return None, msg
    return os.path.join(DIR, vmb), ""


def _retry(fn, what, tries=40, delay=0.25):
    """Retry a filesystem op that another session's engine may be holding.

    WHY: target/release/model.vmb is a SHARED scratch file -- several teammates
    run the engine from this directory. A concurrent run holds model.vmb /
    tokens.bin open, and a plain copy/open then dies with PermissionError
    (measured: this happened mid-run here). That is contention, not a result,
    so it must not be reported as a test FAILURE -- but it also must not be
    silently swallowed, hence a bounded retry with a loud final error.
    """
    last = None
    for _ in range(tries):
        try:
            return fn()
        except PermissionError as e:
            last = e
            time.sleep(delay)
    raise RuntimeError("%s: still locked after %d tries (%s)" % (what, tries, last))


def run_engine(vmb, gen_tokens=None):
    """Run the engine in a PRIVATE temp dir, not the shared target/release/.

    WHY THIS IS NOT BELT-AND-BRACES, IT IS THE FIX (measured twice):
    the engine hardcodes `model.vmb` (runtime/src/main.rs:108), so the obvious
    thing is to cwd into target/release/ and drop the program there. But that
    directory is SHARED with other teammates, who overwrite `model.vmb`,
    delete it, or leave their own `tokens.bin` behind. Observed consequences:
      * "cannot open file" (exit 2) on every case -- someone deleted model.vmb
        between our copy and our exec, which no retry can fix reliably;
      * a full sweep of failures against CORRECT engine code (FAIL 20/PASS 2);
      * in tools/p13_audit_check.py, "row=0.000000" printed by another
        program's model.vmb.
    Copying the exe in alongside the artifact makes each run hermetic, so a
    result can only mean something about OUR program. Returns (exit, out, vals).
    """
    d = tempfile.mkdtemp(prefix="o6_argmax_")
    try:
        _retry(lambda: shutil.copyfile(ENG, os.path.join(d, "vmlm-runtime.exe")),
               "copy engine")
        _retry(lambda: shutil.copyfile(vmb, os.path.join(d, "model.vmb")),
               "copy model.vmb")
        if gen_tokens is not None:
            def _write():
                with open(os.path.join(d, "gen_tokens.bin"), "wb") as f:
                    for t in gen_tokens:
                        f.write(struct.pack("<I", t & 0xFFFFFFFF))
            _retry(_write, "write gen_tokens.bin")
        q = subprocess.run([os.path.join(d, "vmlm-runtime.exe")], cwd=d,
                           capture_output=True)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    out = (q.stdout + q.stderr).decode("utf-8", "replace")
    vals = []
    for line in out.splitlines():
        for tok in line.split():
            try:
                vals.append(float(tok))
            except ValueError:
                pass
    return q.returncode, out, vals


def slot_shapes(vmb):
    """Read (rows, cols) for every slot from a .vmb META section.

    Used to prove the ASSEMBLER's shape inference independently of whether the
    engine happens to be defensive. Header layout (asm/src/main.rs emit_binary):
      off 8  u32 text_off      off 24 u32 meta_off
      off 12 u32 n_ins         off 28 u32 meta_len
      off 16 u32 text_len
    META: [0] n_const, [4] n_slot, [8] slots_off, then the const table, then
          n_slot x 16B { kind, idx, rows, cols }.
    """
    with open(vmb, "rb") as f:
        hdr = f.read(32)
        u = lambda o: struct.unpack_from("<I", hdr, o)[0]
        text_len, meta_off, meta_len = u(16), u(24), u(28)
        f.seek(meta_off)
        mb = f.read(meta_len)
    n_const = struct.unpack_from("<I", mb, 0)[0]
    n_slot = struct.unpack_from("<I", mb, 4)[0]
    slots_off = struct.unpack_from("<I", mb, 8)[0]
    del text_len
    out = []
    for s in range(n_slot):
        _kind, _idx, r, c = struct.unpack_from("<4I", mb, slots_off + s * 16)
        out.append((r, c))
    del n_const
    return out


def check(name, cond, detail=""):
    if cond:
        _passes.append(name)
        print("  PASS  %-52s %s" % (name, detail))
    else:
        _fails.append(name)
        print("  FAIL  %-52s %s" % (name, detail))


def case(name, vml_src, expect_vals, consts=(), expect_exit=EXIT_OK,
         needle=None, gen_tokens=None):
    """Assemble, run on the real engine, compare against hand-computed values."""
    tag = "o6_" + name
    vmb, err = assemble(tag, vml_src, consts)
    if vmb is None:
        check(name, False, "ASSEMBLE FAILED: %s" % err.strip()[-160:])
        return
    code, out, vals = run_engine(vmb, gen_tokens)
    if code != expect_exit:
        check(name, False, "exit=%d (want %d) out=%r" % (code, expect_exit, out[:90]))
        return
    if needle is not None:
        # STRICT, not advisory: on a failure-to-fail case the message is the
        # only evidence that the RIGHT guard fired rather than some unrelated
        # exit(9). It is checked before the value comparison for that reason.
        if needle not in out:
            check(name, False, "exit=%d ok but reason %r missing from %r"
                  % (code, needle, out[:90]))
            return
    if expect_vals is not None:
        if len(vals) != len(expect_vals):
            check(name, False, "got %d values %s, want %d %s"
                  % (len(vals), vals, len(expect_vals), expect_vals))
            return
        bad = [(g, e) for g, e in zip(vals, expect_vals) if abs(g - e) > 1e-4]
        if bad:
            check(name, False, "got %s want %s" % (vals, expect_vals))
            return
        check(name, True, "exit=%d vals=%s" % (code, vals))
    elif needle is not None:
        check(name, True, "exit=%d reason=%r" % (code, needle))
    else:
        check(name, True, "exit=%d" % code)


def main():
    os.makedirs(DIR, exist_ok=True)

    print("=" * 78)
    print("O6-A: OP_MAX (34) / OP_TOPK_IDX (42) through the REAL engine")
    print("  asm  %s" % _fingerprint(ASM))
    print("  eng  %s" % _fingerprint(ENG))
    print("=" * 78)

    for p in (ASM, ENG):
        if not os.path.exists(p):
            print("MISSING BINARY: %s -- run tools/build_all.ps1" % p)
            return 1

    write_const("neg_row.f32", [-5.0, -3.0, -9.0])          # all negative
    write_const("tie_row.f32", [7.0, 9.0, 9.0, 2.0])        # max ties at 1 and 2
    write_const("neg_idx.f32", [-1.0, -7.0, -2.0])          # all negative, max at 1
    write_const("logits.f32", [0.5, -1.25, 3.0, 0.25])      # mixed signs
    write_const("tbl.f32", [10.0, 11.0, 12.0])              # EMB table row 0
    write_const("tbl2.f32", [10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42])
    write_const("row0.f32", [0.0])
    write_const("row2.f32", [2.0])
    write_const("row4.f32", [4.0])
    write_const("row99.f32", [99.0])
    write_const("rowneg.f32", [-1.0])

    # ------------------------------------------------------------------
    print("\n[1] OP_MAX: value semantics")
    # Hand-computed on a REAL constant (an activation slot starts zeroed, so a
    # test that only SHAPEs the input and expects non-zero output is a test
    # bug -- measured, this file's first version did exactly that):
    #   row 0 [1,-2,3,-4] -> 3 ; row 1 [-5,-3,-9,-6] -> -3
    write_const("two_rows.f32", [1.0, -2.0, 3.0, -4.0, -5.0, -3.0, -9.0, -6.0])
    case("max_basic",
         "SHAPE x, 2, 4\n"
         "SHAPE o, 2, 1\n"
         "MAX o, x\n"
         "PRINT o\nHALT\n",
         [3.0, -3.0],
         consts=["x:two_rows.f32:2:4:f32"])

    # ALL-NEGATIVE, single row. Hand-computed max = -3.0.
    # A `0.0` seed reports 0.0 here; a `-inf` seed reports -3.0 correctly, so
    # this case specifically pins down the no-constant-seed rule.
    case("max_all_negative",
         "SHAPE x, 1, 3\n"
         "SHAPE o, 1, 1\n"
         "MAX o, x\n"
         "PRINT o\nHALT\n",
         [-3.0],
         consts=["x:neg_row.f32:1:3:f32"])

    # Ties must not change the VALUE.
    case("max_ties_value",
         "SHAPE x, 1, 4\n"
         "SHAPE o, 1, 1\n"
         "MAX o, x\n"
         "PRINT o\nHALT\n",
         [9.0],
         consts=["x:tie_row.f32:1:4:f32"])

    # ------------------------------------------------------------------
    print("\n[2] OP_TOPK_IDX: argmax, ties -> LOWEST index")
    # Hand-computed: [0.5,-1.25,3.0,0.25] -> index 2 (value 3.0).
    case("topk_basic",
         "SHAPE x, 1, 4\n"
         "SHAPE o, 1, 1\n"
         "TOPK_IDX o, x\n"
         "PRINT o\nHALT\n",
         [2.0],
         consts=["x:logits.f32:1:4:f32"])

    # TIE: [7,9,9,2] -> indices 1 and 2 both hold the max 9.
    # Lowest index wins => 1. A `>=` scan returns 2.
    case("topk_tie_lowest_index",
         "SHAPE x, 1, 4\n"
         "SHAPE o, 1, 1\n"
         "TOPK_IDX o, x\n"
         "PRINT o\nHALT\n",
         [1.0],
         consts=["x:tie_row.f32:1:4:f32"])

    # ALL-NEGATIVE: [-1,-7,-2] -> argmax is index 0 (value -1.0).
    # Chosen so that numeric compare gives 0 and a BIT-PATTERN compare gives 2
    # ... so this case alone cannot separate them. The next case does.
    case("topk_all_negative",
         "SHAPE x, 1, 3\n"
         "SHAPE o, 1, 1\n"
         "TOPK_IDX o, x\n"
         "PRINT o\nHALT\n",
         [0.0],
         consts=["x:neg_idx.f32:1:3:f32"])

    # THE BIT-PATTERN CASE. Values [-1.0, -7.0, -2.0] are written with the
    # LARGEST magnitude first on purpose... but the discriminating row is
    # [-7.0, -1.0, -2.0]: numeric max is -1.0 at INDEX 1, while the raw-i32
    # bit pattern of -1.0 (0xBF800000) is SMALLER as unsigned than -7.0's
    # (0xC0E00000), so a bit-pattern argmax returns index 0. Expected: 1.
    write_const("neg_disc.f32", [-7.0, -1.0, -2.0])
    case("topk_negative_bitpattern_trap",
         "SHAPE x, 1, 3\n"
         "SHAPE o, 1, 1\n"
         "TOPK_IDX o, x\n"
         "PRINT o\nHALT\n",
         [1.0],
         consts=["x:neg_disc.f32:1:3:f32"])

    # Index must be exact for a realistic vocab width (MiniCPM5-1B is 73,448).
    # A f32 holds integers exactly below 2^24, so this is the real bound.
    big = [float(i % 7) for i in range(100000)]
    big[73447] = 999.0                      # unique max at the LAST index
    write_const("big.f32", big)
    case("topk_vocab_width_73448",
         "SHAPE x, 1, 100000\n"
         "SHAPE o, 1, 1\n"
         "TOPK_IDX o, x\n"
         "PRINT o\nHALT\n",
         [73447.0],
         consts=["x:big.f32:1:100000:f32"])

    # ------------------------------------------------------------------
    print("\n[3] MAX and TOPK_IDX agree on the same row (O6 consistency)")
    # [0.5,-1.25,3.0,0.25]: MAX -> 3.0, TOPK_IDX -> 2.
    case("max_and_topk_agree",
         "SHAPE x, 1, 4\n"
         "SHAPE v, 1, 1\n"
         "SHAPE i, 1, 1\n"
         "MAX v, x\n"
         "TOPK_IDX i, x\n"
         "PRINT v\n"
         "PRINT i\n"
         "HALT\n",
         [3.0, 2.0],
         consts=["x:logits.f32:1:4:f32"])

    # ------------------------------------------------------------------
    print("\n[4] Illegal input must fail LOUDLY (the whole point)")
    # Layer 3 of the trap: without shape inference the output slot stays 0x0
    # and the reduction result is written nowhere -- SILENTLY. Two independent
    # defences now exist, and this section proves BOTH:
    #   (a) the assembler infers dst's shape (so the common case cannot be 0x0);
    #   (b) if a slot is still 0x0, the ENGINE exits 9 rather than doing nothing.
    #
    # (a) is asserted by reading the assembled slot table: a 1x1 / 2x1 dst.
    vmb, err = assemble("o6_shape_infer",
                        "SHAPE x, 2, 4\nSHAPE o, 2, 1\nMAX o, x\nHALT\n",
                        ["x:two_rows.f32:2:4:f32"])
    if vmb is None:
        check("asm_infers_max_dst_shape", False, err[-120:])
    else:
        shapes = slot_shapes(vmb)
        check("asm_infers_max_dst_shape", shapes == [(2, 4), (2, 1)],
              "slot shapes %s (want [(2,4),(2,1)])" % shapes)

    vmb, err = assemble("o6_shape_infer_topk",
                        "SHAPE x, 1, 4\nTOPK_IDX o, x\nHALT\n",
                        ["x:logits.f32:1:4:f32"])
    if vmb is None:
        check("asm_infers_topk_dst_shape", False, err[-120:])
    else:
        shapes = slot_shapes(vmb)
        check("asm_infers_topk_dst_shape", shapes == [(1, 4), (1, 1)],
              "slot shapes %s (want [(1,4),(1,1)])" % shapes)

    # (b) No operand at all is declared, so there is nothing to infer and both
    # slots really are 0x0. The engine must exit 9 with a named message -- the
    # old behaviour was to loop zero times and leave dst holding a stale
    # activation, i.e. a max of "whatever was there".
    case("max_zero_width_engine_guard",
         "MAX o, x\nHALT\n",
         None, expect_exit=EXIT_GUARD, needle="width 0")

    case("topk_zero_width_engine_guard",
         "TOPK_IDX o, x\nHALT\n",
         None, expect_exit=EXIT_GUARD, needle="width 0")

    # Wrong dst shape: MAX writes [rows,1]; a [1,1] dst for a 2-row src would
    # silently write only the first row (or overflow). Must be exit 9.
    case("max_wrong_dst_shape",
         "SHAPE x, 2, 4\nSHAPE o, 1, 1\nMAX o, x\nHALT\n",
         None, expect_exit=EXIT_GUARD, needle="dst must be",
         consts=["x:two_rows.f32:2:4:f32"])

    case("topk_wrong_dst_shape",
         "SHAPE x, 1, 4\nSHAPE o, 1, 2\nTOPK_IDX o, x\nHALT\n",
         None, expect_exit=EXIT_GUARD, needle="dst must be",
         consts=["x:logits.f32:1:4:f32"])

    # ------------------------------------------------------------------
    print("\n[5] OP_EMB row bounds -- the O6 feedback path")
    # The generation loop feeds TOPK_IDX's result into st_gen, which SLICE/EMB
    # then read as a table row. That makes the row index a COMPUTED FLOAT, so
    # EMB must validate it. These cases drive EMB with an f32 CONSTANT row so
    # the index is exactly what we choose (no file plumbing in the way).
    #
    # Controls first: valid rows must still work, so the guard cannot be a
    # blanket rejection that breaks normal prompts.
    case("emb_valid_row2_control",
         "SHAPE t, 4, 3\nSHAPE o, 1, 3\nEMB o, t, r\nPRINT o\nHALT\n",
         [30.0, 31.0, 32.0],
         consts=["t:tbl2.f32:4:3:f32", "r:row2.f32:1:1:f32"])
    case("emb_valid_row0_control",
         "SHAPE t, 4, 3\nSHAPE o, 1, 3\nEMB o, t, r\nPRINT o\nHALT\n",
         [10.0, 11.0, 12.0],
         consts=["t:tbl2.f32:4:3:f32", "r:row0.f32:1:1:f32"])

    # OUT OF RANGE: 99 against a 4-row table.
    # Before the guard this EXITED 0 AND PRINTED ZEROS (verifier, real engine).
    case("emb_row_out_of_range_99",
         "SHAPE t, 4, 3\nSHAPE o, 1, 3\nEMB o, t, r\nPRINT o\nHALT\n",
         None, expect_exit=EXIT_GUARD, needle="EMB: row index out of range",
         consts=["t:tbl2.f32:4:3:f32", "r:row99.f32:1:1:f32"])

    # Exactly at the boundary: row == rows (4) is out of range (valid are 0..3).
    case("emb_row_exactly_rows",
         "SHAPE t, 4, 3\nSHAPE o, 1, 3\nEMB o, t, r\nPRINT o\nHALT\n",
         None, expect_exit=EXIT_GUARD, needle="EMB: row index out of range",
         consts=["t:tbl2.f32:4:3:f32", "r:row4.f32:1:1:f32"])

    # NEGATIVE: `as usize` SATURATES to 0 in Rust, so -1 would silently read
    # row 0 -- a wrong embedding with no error. Must be exit 9.
    case("emb_row_negative_saturates_to_0",
         "SHAPE t, 4, 3\nSHAPE o, 1, 3\nEMB o, t, r\nPRINT o\nHALT\n",
         None, expect_exit=EXIT_GUARD, needle="EMB: row index out of range",
         consts=["t:tbl2.f32:4:3:f32", "r:rowneg.f32:1:1:f32"])

    # FRACTIONAL: TOPK_IDX returns an exact integer, but nothing stops a program
    # from handing EMB a fractional row. Truncation to row 2 would be a silent
    # off-by-one; require the integral form instead.
    write_const("rowfrac.f32", [2.5])
    case("emb_row_fractional",
         "SHAPE t, 4, 3\nSHAPE o, 1, 3\nEMB o, t, r\nPRINT o\nHALT\n",
         None, expect_exit=EXIT_GUARD, needle="not an integer",
         consts=["t:tbl2.f32:4:3:f32", "r:rowfrac.f32:1:1:f32"])

    # NaN: every ordered comparison is false for NaN, so a `row < 0.0` test
    # would MISS it and `as usize` would saturate to row 0. The guard uses the
    # negated form `!(row >= 0.0)` precisely to catch this.
    write_const("rownan.f32", [float("nan")])
    case("emb_row_nan",
         "SHAPE t, 4, 3\nSHAPE o, 1, 3\nEMB o, t, r\nPRINT o\nHALT\n",
         None, expect_exit=EXIT_GUARD, needle="EMB: row index out of range",
         consts=["t:tbl2.f32:4:3:f32", "r:rownan.f32:1:1:f32"])

    print()
    print("=" * 78)
    print("PASS %d   FAIL %d" % (len(_passes), len(_fails)))
    if _fails:
        print("failed cases:")
        for f in _fails:
            print("   - %s" % f)
        # Re-print the fingerprints on failure: a result that does not name the
        # binary that produced it cannot be reproduced, and on a shared
        # workspace a stale engine is a real cause of a false FAIL.
        print()
        print("binaries that produced this result (re-run after rebuild):")
        print("  asm  %s" % _fingerprint(ASM))
        print("  eng  %s" % _fingerprint(ENG))
    print("=" * 78)
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
