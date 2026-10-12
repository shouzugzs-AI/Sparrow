#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P16-2 GUARD PROBE: make every new engine guard fire, and capture its exact
message plus exit code.

    python tools/p16_guardprobe.py --exe <bench engine> --asm <assembler>

WHY A SEPARATE PROBE FROM p16_sparse_ops.py
    p16_sparse_ops.py answers "is the arithmetic right". This one answers "when
    the program is WRONG, does the engine say so". Those are different failure
    modes and the second is the one this project has been burned by repeatedly:
    a guard that never executes is indistinguishable from no guard, and a guard
    that fires with a message nobody can find is barely better.

    The pattern is copied from tools/p15_verify_guardprobe/, which exists for
    the same reason (S5's guards). Each case here asserts THREE things:
      1. the engine does NOT exit 0,
      2. it exits with the code documented for that guard (9),
      3. the message NAMES THE PROBLEM (a substring match, so a generic
         "unimpl op" or an empty stdout fails the check).

    (3) is the part that is easy to skip and the part that matters: the
    original sin this project keeps re-fixing is a failure that is *detected*
    but not *diagnosed*.

NEGATIVE CONTROL
    Section Z runs a CORRECT program and asserts exit 0. Without it, a probe
    that always saw exit 9 -- e.g. because the exe path was wrong -- would
    report every guard as working.
"""
import argparse
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p16_ops_probe as P  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fnv1a(s):
    h = 0x811C9DC5
    for c in s.encode():
        h = ((h ^ c) * 0x01000193) & 0xFFFFFFFF
    return h


def wr(path, vals):
    open(path, "wb").write(struct.pack("<%df" % len(vals), *vals))


class Probe:
    def __init__(self, exe, asm, work):
        self.exe, self.asm, self.work = exe, asm, work
        self.data = os.path.join(work, "w")
        os.makedirs(self.data, exist_ok=True)
        self.npass = self.nfail = 0

    def rel(self, p):
        return os.path.relpath(p, ROOT).replace("/", os.sep)

    def build(self, name, src, consts):
        vml = os.path.join(self.work, name + ".vml")
        vmb = os.path.join(self.work, name + ".vmb")
        open(vml, "w", encoding="ascii").write(src)
        rc, out, err = P.run_asm(self.asm, vml, vmb, consts, cwd=ROOT)
        return rc, out + err, vml, vmb

    def run(self, vmb, slot=None, reps=1):
        d = os.path.join(self.work, "run_" + os.path.basename(vmb))
        os.makedirs(d, exist_ok=True)
        # The engine's bench entry point takes a PATH to the .vmb, but it
        # resolves it relative to its OWN cwd, and it also writes out.bin
        # there. Copy the artifact in so each run is hermetic -- sharing one
        # directory made two runs race on out.bin in the main harness.
        local = os.path.join(d, os.path.basename(vmb))
        if os.path.abspath(local) != os.path.abspath(vmb):
            import shutil
            shutil.copyfile(vmb, local)
        return P.run_engine(self.exe, local, dump_slot=slot, reps=reps, cwd=d)

    def fire(self, label, expect_code, expect_msg, src, consts, slot=None):
        """Assemble+run a program that is SUPPOSED to fail. Check all 3 parts."""
        # Windows forbids < > : " / \ | ? * in file names, so a case label
        # ("dst rows < L") cannot be used verbatim.
        safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in label)
        safe = safe.strip("_")[:60]
        rc_asm, amsg, vml, vmb = self.build(safe, src, consts)
        if rc_asm != 0:
            print("  [ASM-REJECTED] %-44s (assembler panic, not an engine guard)"
                  % label)
            print("      %s" % amsg.strip().splitlines()[-1][:150])
            # An assembler rejection is a legitimate outcome for the two
            # encodings that can only be caught statically; count it as a pass
            # for its own expectation.
            if expect_code == "ASM":
                self.npass += 1
                return True
            self.nfail += 1
            return False
        rc, stdout, _ = self.run(vmb, slot=slot)
        # `run_engine` decodes stdout to str; compare as str.
        want = expect_msg.decode() if isinstance(expect_msg, bytes) else expect_msg
        got_msg = want in stdout
        ok = (rc == expect_code) and got_msg
        tag = "PASS" if ok else "FAIL"
        if ok:
            self.npass += 1
        else:
            self.nfail += 1
        print("  [%s] %-44s rc=%-11s msg=%r" % (tag, label, rc,
                                                stdout.strip()[:90]))
        if not got_msg:
            print("      expected message containing %r" % expect_msg)
        return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--asm", required=True)
    ap.add_argument("--work", default=os.path.join(ROOT, "examples", "p16_ops", "guards"))
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    pr = Probe(a.exe, a.asm, a.work)

    # ---- shared data -----------------------------------------------------
    # well-formed 4-neuron / 4-connection CSR (same shape as the main harness)
    wr(os.path.join(pr.data, "off.bin"), [0.0, 2.0, 3.0, 3.0, 4.0])
    wr(os.path.join(pr.data, "src.bin"), [0.0, 1.0, 3.0, 2.0])
    wr(os.path.join(pr.data, "w.bin"), [1.0, 2.0, 3.0, 4.0])
    wr(os.path.join(pr.data, "x.bin"), [10.0, 20.0, 30.0, 40.0])
    wr(os.path.join(pr.data, "k2.bin"), [2.0])
    wr(os.path.join(pr.data, "str.bin"), [1.0, 2.0, 3.0, 4.0])
    # k=99 with L=4  ->  TOPK_BY_STRENGTH: k > L
    wr(os.path.join(pr.data, "k99.bin"), [99.0])
    # k = -1        ->  TOPK_BY_STRENGTH: k is negative or NaN
    wr(os.path.join(pr.data, "kneg.bin"), [-1.0])
    # src id 9 with only 4 inputs ->  SPARSE_MATVEC: src_id out of range
    wr(os.path.join(pr.data, "srcbad.bin"), [0.0, 1.0, 9.0, 2.0])
    # offsets that walk past the src array ->  SPARSE_MATVEC: offsets out of range
    wr(os.path.join(pr.data, "offbad.bin"), [0.0, 2.0, 3.0, 3.0, 99.0])

    D = pr.rel(pr.data)
    OK = [("offc", "%s/off.bin" % D, 5, 1), ("srcc", "%s/src.bin" % D, 4, 1),
          ("wvc", "%s/w.bin" % D, 4, 1), ("xvc", "%s/x.bin" % D, 4, 1),
          ("k2c", "%s/k2.bin" % D, 1, 1), ("strc", "%s/str.bin" % D, 4, 1)]

    def C(**over):
        """Return the shared constant list with some entries re-pointed.

        The first version of this did the substitution inside a nested
        comprehension whose `for key in [...]` clause never matched, so the
        constants were NOT swapped -- and the two guards it was meant to
        exercise ("src_id out of range", "offsets out of range") both reported
        rc=0, i.e. the program SUCCEEDED because it was still running on the
        good data. Two probes silently testing nothing. Written as an explicit
        loop so the substitution is inspectable.
        """
        out = []
        for (n, p, r, c) in OK:
            if n in over:
                out.append((n, "%s/%s" % (D, over[n]), r, c))
            else:
                out.append((n, p, r, c))
        return out

    HEAD = """SHAPE off, 5, 1
SHAPE srci, 4, 1
SHAPE wv, 4, 1
SHAPE xv, 4, 1
SHAPE str, 4, 1
SHAPE kk, 1, 1
SHAPE sel, 4, 2
SHAPE y, 5, 1
    COPY off, offc
    COPY srci, srcc
    COPY wv, wvc
    COPY xv, xvc
    COPY kk, k2c
    COPY str, strc
"""

    print("=" * 78)
    print("GUARD PROBE -- each case must (a) not exit 0, (b) exit 9,")
    print("(c) print a message naming the problem")
    print("=" * 78)

    # --- guard 1: zero-width SOURCE slot on TOPK_BY_STRENGTH --------------
    # `zz` is declared but never given a shape, so it stays 0x0.
    print("\n-- TOPK_BY_STRENGTH --")
    pr.fire("k > L", 9, b"k > L",
            HEAD + "    TOPK_BY_STRENGTH sel, str, k99c\n    HALT\n",
            OK + [("k99c", "%s/k99.bin" % D, 1, 1)])
    pr.fire("k negative", 9, b"negative or NaN",
            HEAD + "    TOPK_BY_STRENGTH sel, str, knegc\n    HALT\n",
            OK + [("knegc", "%s/kneg.bin" % D, 1, 1)])
    pr.fire("src slot zero width", 9, b"width 0",
            HEAD + "    TOPK_BY_STRENGTH sel, zz, k2c\n    HALT\n", OK)
    # dst must be [L,2]: give it [4,1] -> dc != 2
    pr.fire("dst not [L,2]", 9, b"dst must be [L,2]",
            HEAD.replace("SHAPE sel, 4, 2", "SHAPE sel, 4, 1")
            + "    TOPK_BY_STRENGTH sel, str, k2c\n    HALT\n", OK)
    # dst too SHORT in rows: [2,2] with L=4
    pr.fire("dst rows < L", 9, b"dst needs L rows",
            HEAD.replace("SHAPE sel, 4, 2", "SHAPE sel, 2, 2")
            + "    TOPK_BY_STRENGTH sel, str, k2c\n    HALT\n", OK)

    # --- guard 3: P16-5 sel-gated mode (added by task-11) ----------------
    #
    # These exist because the gate introduces a NEW way for a program to be
    # wrong that the ungated path cannot express: `sel` can disagree with the
    # CSR it is supposed to index. That must be a hard failure, because the
    # alternative -- reading a weight that belongs to another neuron -- yields a
    # number that looks exactly like a real activation.
    print("\n-- P16-5 selgate (new in task-11) --")
    GSEL = ("SHAPE off, 5, 1\nSHAPE srci, 4, 1\nSHAPE sel, 4, 8\nSHAPE wv, 4, 1\n"
            "SHAPE xv, 4, 1\nSHAPE str, 4, 4\nSHAPE kk, 1, 1\nSHAPE y, 5, 1\n"
            "    COPY off, offc\n    COPY srci, srcc\n    COPY wv, wvc\n"
            "    COPY xv, xvc\n    COPY str, strc\n    COPY kk, k2c\n")
    SELOK = [("offc", "%s/off.bin" % D, 5, 1), ("srcc", "%s/src.bin" % D, 4, 1),
             ("wvc", "%s/w.bin" % D, 4, 1), ("xvc", "%s/x.bin" % D, 4, 1),
             ("strc", "%s/str.bin" % D, 4, 1), ("k2c", "%s/k2.bin" % D, 1, 1)]

    pr.fire("selgate: odd sel width", 9, b"selgate needs an even sel width",
            GSEL.replace("SHAPE sel, 4, 8", "SHAPE sel, 4, 7")
            + "    SPARSE_MATVEC y, off, srci, wv, xv, sel, nnz, selgate\n    HALT\n",
            SELOK)
    pr.fire("selgate: sel rows < n_neurons", 9, b"sel table too small",
            GSEL.replace("SHAPE sel, 4, 8", "SHAPE sel, 2, 8")
            + "    SPARSE_MATVEC y, off, srci, wv, xv, sel, nnz, selgate\n    HALT\n",
            SELOK)
    pr.fire("selgate: TOPK k > segment width", 9, b"k > dst segment width",
            GSEL.replace("SHAPE sel, 4, 8", "SHAPE sel, 4, 2")
            + "    TOPK_BY_STRENGTH sel, str, k2c, seg\n    HALT\n", SELOK)
    pr.fire("selgate: TOPK segmented dst rows != src rows", 9,
            b"segmented dst rows != src rows",
            GSEL.replace("SHAPE sel, 4, 8", "SHAPE sel, 3, 8")
            + "    TOPK_BY_STRENGTH sel, str, k2c, seg\n    HALT\n", SELOK)
    # The composition guard itself: a `sel` whose entries point outside their
    # own CSR row must be REJECTED, not summed. Reached by giving each row a
    # 1-connection CSR span while sel still claims 8-wide rows.
    wr(os.path.join(pr.data, "off_tiny.bin"), [0.0, 1.0, 1.0, 1.0, 1.0])
    wr(os.path.join(pr.data, "str_tiny.bin"), [1.0, 0.5, 0.25, 0.125])
    pr.fire("selgate: sel index outside its CSR row", 9,
            b"sel index outside its CSR row",
            GSEL + "    TOPK_BY_STRENGTH sel, str, k2c, seg\n"
                   "    SPARSE_MATVEC y, off, srci, wv, xv, sel, nnz, selgate\n    HALT\n",
            [("offc", "%s/off_tiny.bin" % D, 5, 1), ("srcc", "%s/src.bin" % D, 4, 1),
             ("wvc", "%s/w.bin" % D, 4, 1), ("xvc", "%s/x.bin" % D, 4, 1),
             ("strc", "%s/str_tiny.bin" % D, 4, 1), ("k2c", "%s/k2.bin" % D, 1, 1)])

    # --- guard 2: SPARSE_MATVEC ------------------------------------------
    print("\n-- SPARSE_MATVEC --")
    pr.fire("src_id out of range", 9, b"src_id out of range",
            HEAD + "    SPARSE_MATVEC y, off, srci, wv, xv, nnz\n    HALT\n",
            C(srcc="srcbad.bin"))
    pr.fire("offsets out of range", 9, b"offsets out of range",
            HEAD + "    SPARSE_MATVEC y, off, srci, wv, xv, nnz\n    HALT\n",
            C(offc="offbad.bin"))
    pr.fire("offsets zero width", 9, b"offsets or src slot has width 0",
            HEAD + "    SPARSE_MATVEC y, zz, srci, wv, xv, nnz\n    HALT\n", OK)
    pr.fire("x (dt) zero width", 9, b"x slot (dt) has width 0",
            HEAD + "    SPARSE_MATVEC y, off, srci, wv, zz, nnz\n    HALT\n", OK)
    # dst zero width. TWO THINGS HAD TO BE LEARNED TO REACH THIS GUARD, both
    # recorded because the first attempts silently tested nothing:
    #
    # 1. With `nnz` the dst-width check is SHADOWED by the stricter
    #    "nnz needs dst to be [n+1]" guard that runs first (measured: the first
    #    version expected "dst slot has width 0" and got the nnz message). So
    #    the no-`nnz` form is required to reach it.
    # 2. Even then it is UNREACHABLE if the offsets slot has a known shape:
    #    this opcode's own shape inference (asm/src/main.rs, OP_SPARSE_MATVEC)
    #    GIVES the destination [n_neurons,1], so a never-declared `zz` is no
    #    longer 0x0 by the time the engine sees it -- measured, the assembler
    #    printed `slot14 zz [act] 4x1`. That inference is CORRECT and
    #    desirable; it is exactly the OP_MAX/OP_TOPK_IDX trap being avoided.
    #    The guard therefore only becomes reachable when the OFFSETS are also
    #    unshaped, i.e. when inference has nothing to derive n from.
    #
    # So this case leaves BOTH the destination and the offsets undeclared. It
    # still exits 9, just with the offsets message -- which is the guard that
    # legitimately wins. The dst-specific message is then exercised by the
    # "dst too small for n" case below, where n IS known.
    pr.fire("dst and offsets both unshaped", 9, b"offsets or src slot has width 0",
            HEAD + "    SPARSE_MATVEC zz, zo, srci, wv, xv\n    HALT\n", OK)
    pr.fire("dst too small for n", 9, b"dst must be [n_neurons",
            HEAD.replace("SHAPE y, 5, 1", "SHAPE y, 2, 1")
            + "    SPARSE_MATVEC y, off, srci, wv, xv\n    HALT\n", OK)
    pr.fire("nnz needs one more element", 9, b"nnz needs dst to be",
            HEAD.replace("SHAPE y, 5, 1", "SHAPE y, 4, 1")
            + "    SPARSE_MATVEC y, off, srci, wv, xv, nnz\n    HALT\n", OK)
    # offsets must have n+1 entries: 1 row is not enough
    pr.fire("offsets need n+1 entries", 9, b"offsets must have n_neurons",
            HEAD.replace("SHAPE off, 5, 1", "SHAPE off, 1, 1")
                .replace("COPY off, offc", "")
            + "    SPARSE_MATVEC y, off, srci, wv, xv, nnz\n    HALT\n", OK)

    # --- assembler-side (static) rejections ------------------------------
    print("\n-- assembler-side hard failures (exit via panic, not an engine guard) --")
    # weight slot number > 63 cannot fit in fl's low 6 bits
    # Build a program with 70 dummy slots so `wz` lands above 63.
    dummies = "\n".join("SHAPE d%02d, 1, 1" % i for i in range(70))
    pr.fire("weight slot number > 63", "ASM", b"",
            dummies + "\n" + HEAD
            + "    SPARSE_MATVEC y, off, srci, wz, xv, nnz\n    HALT\n", OK)
    pr.fire("SPARSE_MATVEC missing x operand", "ASM", b"",
            HEAD + "    SPARSE_MATVEC y, off, srci, wv, nnz\n    HALT\n", OK)

    # --- Z: NEGATIVE CONTROL ---------------------------------------------
    print("\n-- Z: NEGATIVE CONTROL (a correct program must exit 0) --")
    rc_asm, amsg, vml, vmb = pr.build("z_ok", HEAD
                                      + "    SPARSE_MATVEC y, off, srci, wv, xv, nnz\n    HALT\n",
                                      OK)
    if rc_asm != 0:
        print("  [FAIL] correct program did not assemble: %s" % amsg[-200:])
        pr.nfail += 1
    else:
        rc, stdout, vals = pr.run(vmb, slot=0)
        ok = (rc == 0)
        print("  [%s] correct program exits 0                rc=%d vals=%r"
              % ("PASS" if ok else "FAIL", rc, vals))
        if ok:
            pr.npass += 1
        else:
            pr.nfail += 1

    print("\n" + "=" * 78)
    print("guard probe: %d pass, %d fail" % (pr.npass, pr.nfail))
    return 1 if pr.nfail else 0


if __name__ == "__main__":
    sys.exit(main())
