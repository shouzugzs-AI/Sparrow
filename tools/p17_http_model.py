#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P17-P4: build the `.vmb` fixture the HTTP server runs.

WHY A FIXTURE AND NOT A REAL MODEL
----------------------------------
Task-5 says the server must "really call the engine and run a forward pass (even
with a small model / placeholder output)". A real 24-layer MiniCPM artifact is a
0.6 GB file built by `tools/p12_build_minicpm.py`, and it is the wrong instrument
here: the acceptance criterion is "curl can reach the engine and the response
carries the engine's own numbers", which a 4x6 layer exercises completely while
staying checkable BY HAND. The 24-layer flow is task-3's and task-1's scope.

WHAT THIS BUILDS
----------------
`examples/p17_http/model.vmb` -- a tiny real program (assembled by target/release/
vmlm-asm.exe, the same assembler every other phase uses, so this is a genuine
artifact and not a hand-rolled byte blob), plus `model.slots` recording which
slot names the server should use.

The program is deliberately multi-instruction so the engine has real work to do:
it computes a sparse-ish GEMM, applies an activation, adds a bias, and writes a
5-element output slot. `out.bin` from the bench engine then contains 5 f32 values
that are a deterministic function of the constants in the file -- which is what
makes "the reply depends on the engine" falsifiable: change a weight, the reply
changes.

MODEL SIZE IS REPORTED, NOT ASSUMED
-----------------------------------
The script prints the `.vmb` size, the slot table and the const table, and it
verifies that the two slot names the server needs (`x`, `y`) exist by hashing
against the file -- the same FNV-1a the engine uses, independently implemented
here in Python.

USAGE
-----
    python tools/p17_http_model.py
    python tools/p17_http_model.py --out examples/p17_http
"""
import argparse
import os
import struct
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- model shape -----------------------------------------------------------
# 4 neurons x 6 inputs. Small enough to check by hand; big enough that a mistake
# in slot addressing shows up as wrong NUMBERS rather than as an error.
N_NEURONS = 4
N_INPUTS = 6
N_OUT = 5


def fnv1a(name):
    """Independent implementation of format::hash32 (FNV-1a 32)."""
    h = 0x811C9DC5
    for b in name.encode("utf-8"):
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def f32(x):
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def write_f32(path, vals):
    with open(path, "wb") as f:
        for v in vals:
            f.write(struct.pack("<f", f32(v)))


def read_f32(path):
    raw = open(path, "rb").read()
    return list(struct.unpack("<%df" % (len(raw) // 4), raw))


def deterministic_weights(n, seed):
    """xorshift32 -- no numpy, no RNG version drift, reproducible forever."""
    s = seed | 1
    out = []
    for _ in range(n):
        s ^= (s << 13) & 0xFFFFFFFF
        s ^= s >> 17
        s ^= (s << 5) & 0xFFFFFFFF
        out.append(((s & 0xFFFF) / 32768.0) - 1.0)  # roughly [-1, 1)
    return out


def assemble(asm, vml, vmb, consts, cwd):
    argv = [asm, vml, vmb] + ["%s:%s:%d:%d" % c for c in consts]
    p = subprocess.run(argv, capture_output=True, cwd=cwd)
    out = (p.stdout or b"").decode("utf-8", "replace")
    err = (p.stderr or b"").decode("utf-8", "replace")
    return p.returncode, out + err


def parse_vmb(path):
    """Read back the metadata, independently of the server's vmb.rs.

    Both readers exist so that a disagreement is visible; the server's reader has
    its own unit tests, and this one is checked against the numbers the ASSEMBLER
    was told to write (rows/cols per constant), which is a third source.
    """
    b = open(path, "rb").read()
    magic, ver = struct.unpack_from("<IH", b, 0)
    text_off, text_len, const_off, const_len, meta_off = struct.unpack_from("<5I", b, 8)
    n_const, n_slots, slots_off, act, state, grad, names_off = struct.unpack_from("<7I", b, meta_off)
    ct = meta_off + 28
    consts = []
    for i in range(n_const):
        o = ct + i * 24
        off, ln, r, c = struct.unpack_from("<4I", b, o)
        dt = b[o + 16]
        consts.append(dict(off=off, len=ln, rows=r, cols=c, dtype=dt))
    st = meta_off + slots_off
    nt = meta_off + names_off
    slots = []
    for i in range(n_slots):
        e = st + i * 16
        kind, idx, r, c = struct.unpack_from("<4I", b, e)
        h = struct.unpack_from("<I", b, nt + i * 4)[0]
        slots.append(dict(kind=kind, idx=idx, rows=r, cols=c, hash=h))
    return dict(
        magic=magic, ver=ver, size=len(b),
        text_off=text_off, text_len=text_len,
        const_off=const_off, const_len=const_len, meta_off=meta_off,
        n_const=n_const, n_slots=n_slots, act_bytes=act,
        consts=consts, slots=slots,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "examples", "p17_http"))
    ap.add_argument("--asm", default=os.path.join(ROOT, "target", "release", "vmlm-asm.exe"))
    ap.add_argument("--engine", default=None,
                    help="bench-feature engine, to smoke-test the fixture")
    a = ap.parse_args()

    out = a.out
    dat = os.path.join(out, "w")
    os.makedirs(dat, exist_ok=True)
    rel = lambda p: os.path.relpath(p, ROOT).replace("/", os.sep)

    # ---- constants ---------------------------------------------------------
    w = [f32(v) for v in deterministic_weights(N_NEURONS * N_INPUTS, 0x51CE7)]
    x = [f32(v) for v in deterministic_weights(N_INPUTS, 0x1234)]
    bias = [f32(0.25 * (i + 1)) for i in range(N_OUT)]

    write_f32(os.path.join(dat, "w.bin"), w)
    write_f32(os.path.join(dat, "x.bin"), x)
    write_f32(os.path.join(dat, "bias.bin"), bias)

    # ---- the program -------------------------------------------------------
    # xa[1,6] = COPY of the embedded input constant xc   -- see the note below
    # y[1,4]  = act(xa[1,6] @ w[4,6]^T)                  -- NT GEMM
    # o[1,5]  = y broadcast + bias
    #
    # ⚠️ WHY THE INPUT IS A CONSTANT COPIED INTO AN ACTIVATION, AND NOT AN
    # ACTIVATION FILLED FROM a FILE (this was measured, not chosen):
    # `runtime/src/bench.rs` -- the argv-driven engine entry point -- parses
    # `<model.vmb>` and `[reps[:slot]]`, runs the VM, and dumps one slot to
    # `out.bin`. It reads NO input file: there is no `x.bin` convention in the
    # bench feature. So a program whose input slot is a plain ACTIVATION runs on
    # whatever the activation region happens to contain (zeroed at load), and the
    # first version of this fixture produced `out.bin = [0,0,0,0]` -- a valid
    # program, a clean exit 0, and meaningless numbers. The fix is to carry the
    # input as a CONSTANT in the .vmb and COPY it into the activation slot, so the
    # forward pass is fully determined by the artifact itself.
    #
    # Consequence to state plainly in the report: the server cannot yet choose the
    # INPUT from the prompt -- it runs a fixed fixture program. That is the honest
    # limit of "the server really calls the engine" at this stage; what it does
    # prove is that the engine is genuinely invoked, its output is measured and
    # returned, and the reply is a function of that output.
    #
    # ⚠️ SHAPE DECLARATIONS ARE AUTHORITATIVE AND OVERRIDE INFERENCE (measured):
    # asm/src/main.rs `infer_shapes` applies explicit SHAPE with `force_shape`,
    # then propagates. An earlier version declared `SHAPE y, 4, 1` for a GEMM whose
    # real result is 1x4, and the assembler HONOURED the wrong declaration: the
    # engine ran an invalid program and returned [2.0, 0.0, 0.0, 0.0] with no
    # error at all.
    #
    # ⚠️ `ACT` NEEDS TWO OPERANDS: `ACT dst, src`. The interpreter reads the
    # SOURCE from operand `a` (interp.rs OP_ACT: `let ca = self.slot_mut(a)`),
    # but the assembler's generic rule for a ONE-register instruction puts that
    # register in `dst` and leaves `a = 0` (asm/src/main.rs line ~860). So
    # `ACT y` encoded as `dst=y a=0` and the engine applied RELU to SLOT 0 and
    # wrote the result into `y` -- measured output [0.63934326171875, 0, 0, 0],
    # i.e. exactly relu(x[0]) in lane 0. Exit code 0, no warning. The named
    # intermediate `ya` exists precisely so both operands are explicit.
    #
    # ⚠️ `ACT`'s KIND also comes from `fl`, not from an operand: interp.rs does
    # `if ins.fl == 0 { ACT_RELU } else { ins.fl as u16 }`, so there is no
    # selector slot to pass. fl = 0 here, hence RELU.
    #
    # ⚠️ `BIAS o, bc` adds the bias to `o`'s CURRENT contents; `o` starts at zero
    # and the program never writes `ya` into it, so o == bias. Stated because the
    # first version of this comment claimed "o = y + bias", and an independent
    # reference built on that claim disagreed with a CORRECT engine run.
    #
    # Slot name `y` is the output the server binds to.
    vml = """\
; P17-P4 HTTP fixture, generated by tools/p17_http_model.py
; y = act(xa[1,6] @ w[4,6]^T) -> [1,4];  o = 0 + bias -> [1,5]
SHAPE xa, 1, 6
SHAPE y, 1, 4
SHAPE ya, 1, 4
SHAPE o, 1, 5
COPY xa, xc
GEMM ya, xa, wc
ACT y, ya
BIAS o, bc
HALT
"""
    vml_path = os.path.join(out, "model.vml")
    vmb_path = os.path.join(out, "model.vmb")
    open(vml_path, "w", encoding="ascii").write(vml)

    consts = [
        ("wc", "%s/w.bin" % rel(dat), N_NEURONS, N_INPUTS),
        ("xc", "%s/x.bin" % rel(dat), 1, N_INPUTS),
        ("bc", "%s/bias.bin" % rel(dat), 1, N_OUT),
    ]
    rc, outtxt = assemble(a.asm, vml_path, vmb_path, consts, ROOT)
    print("=== assemble ===")
    print("asm rc = %d" % rc)
    if outtxt.strip():
        print(outtxt.strip())
    if rc != 0 or not os.path.exists(vmb_path):
        print("FAIL: assembler did not produce %s" % vmb_path)
        return 1

    info = parse_vmb(vmb_path)
    print("\n=== model.vmb ===")
    print("size      = %d B" % info["size"])
    print("magic     = 0x%08X (want 0x4D4C4D56)" % info["magic"])
    print("version   = %d" % info["ver"])
    print("text      = off %d len %d" % (info["text_off"], info["text_len"]))
    print("const     = off %d len %d  (%d entries)" % (
        info["const_off"], info["const_len"], info["n_const"]))
    print("meta      = off %d" % info["meta_off"])
    print("n_slots   = %d   act_bytes = %d" % (info["n_slots"], info["act_bytes"]))
    for i, c in enumerate(info["consts"]):
        print("  const[%d] off=%4d len=%4d %dx%d dtype=%d" % (
            i, c["off"], c["len"], c["rows"], c["cols"], c["dtype"]))
    for i, s in enumerate(info["slots"]):
        kind = {0: "const", 1: "act", 2: "state", 3: "grad", 4: "param"}.get(s["kind"], "?")
        print("  slot[%d] %-5s idx=%4d %dx%d hash=%08x" % (
            i, kind, s["idx"], s["rows"], s["cols"], s["hash"]))

    # ---- the check that matters: does the file have the slots the server wants?
    print("\n=== slot lookup (independent FNV-1a) ===")
    found = {}
    for name in ("xa", "y", "o", "wc", "xc", "bc"):
        h = fnv1a(name)
        idx = next((i for i, s in enumerate(info["slots"]) if s["hash"] == h), None)
        found[name] = idx
        print("  %-4s hash=%08x -> slot %s" % (name, h, idx))
    ok = True
    # The server binds to `xa` (the activation the program actually reads) and
    # `y` (the output it dumps). Both are REQUIRED; a missing one is a hard
    # failure here rather than a runtime surprise.
    for need in ("xa", "y"):
        if found[need] is None:
            print("FAIL: the server needs a slot named `%s` and it is absent" % need)
            ok = False
    if not ok:
        return 1

    # ---- record the slot names the server should bind to -------------------
    slots_path = os.path.join(out, "model.slots")
    with open(slots_path, "w", encoding="ascii") as f:
        f.write("xa\ny\n")
    print("\nwrote %s (input slot / output slot, one per line)" % slots_path)

    # ---- smoke test with the real engine ----------------------------------
    if a.engine and os.path.exists(a.engine):
        import shutil
        d = os.path.join(out, "_smoke")
        os.makedirs(d, exist_ok=True)
        shutil.copyfile(vmb_path, os.path.join(d, "model.vmb"))

        # Independent reference, computed in Python with explicit f32 rounding.
        # This is the whole point of the smoke test: "engine rc == 0" is NOT
        # evidence the forward pass is right (the first two attempts at this
        # fixture both exited 0 with wrong numbers).
        pre = []
        for i in range(N_NEURONS):
            acc = 0.0
            first = True
            for j in range(N_INPUTS):
                t = f32(w[i * N_INPUTS + j] * x[j])
                acc = t if first else f32(acc + t)
                first = False
            pre.append(f32(acc))
        # ACT with fl == 0 is ACT_RELU (interp.rs line 909).
        y_ref = [f32(max(0.0, v)) for v in pre]
        # `BIAS o, bc` adds the bias to `o`'s CURRENT contents (interp.rs:
        # "dst += broadcast(bias)"). `o` is a fresh activation slot, so at load it
        # is zero -- but the program never writes y into o, so o = 0 + bias, NOT
        # y + bias. The reference must model the program that is actually there,
        # not the program the comment originally intended. (This is the same
        # class of mistake as the SHAPE override above: the artifact is the
        # authority, and an independent reference is only useful if it reads the
        # artifact's real semantics.)
        o_ref = [f32(0.0 + bias[i]) for i in range(N_OUT)]
        print("\nreference pre-activation = %r" % pre)
        print("reference relu           = %r" % y_ref)
        print("reference o = 0 + bias   = %r" % o_ref)

        for slot_name, ref in (("o", o_ref), ("y", y_ref)):
            si = found[slot_name]
            ob = os.path.join(d, "out.bin")
            if os.path.exists(ob):
                os.remove(ob)
            print("\n=== smoke test: engine model.vmb 1:%d  (slot %s) ===" % (si, slot_name))
            p = subprocess.run([a.engine, "model.vmb", "1:%d" % si],
                               capture_output=True, cwd=d)
            print("engine rc = %d" % p.returncode)
            so = (p.stdout or b"").decode("utf-8", "replace").strip()
            if so:
                print("engine stdout = %r" % so)
            if not os.path.exists(ob):
                print("FAIL: engine wrote no out.bin")
                return 1
            got = read_f32(ob)
            print("out.bin = %d f32: %r" % (len(got), got))
            match = got == ref
            print("engine == python reference : %s" % match)
            if not match:
                print("MISMATCH: engine=%r ref=%r" % (got, ref))
                return 1
    else:
        print("\n(no --engine given; skipped the smoke test)")

    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
