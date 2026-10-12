#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P14 independent verification harness (owned by the VERIFIER, task-4).

WHY THIS FILE EXISTS RATHER THAN REUSING p14_gen_check.py
    The implementer of O6 (task-2) owns tools/p14_gen_check.py. If the verifier
    re-used it, a bug in that checker would be invisible: the checker and the
    thing it checks would share an author, a code path, and a blind spot. This
    project has already been burned once by exactly that pattern -- a probe with
    an EMPTY PATCH reported "0 B" for a change that really cost 3,584 B (see
    docs/P13_report section 3.2: "that 0 B was an empty measurement, not a
    conclusion"). So the verifier's instruments are written from scratch here,
    by someone who did not write the implementation, and every one of them is
    shown to be CAPABLE OF FAILING before its PASS is believed.

THE FOUR INSTRUMENTS IN THIS FILE
    A. IdsTokenSource   -- write a chosen token chain to gen_tokens.bin
    B. perturb          -- mutate a token chain (the negative-control engine)
    C. compare_bits     -- exact bit comparison that reports its own margin
    D. check_fails_on   -- THE META-CHECK: run a predicate against a KNOWN-BAD
                           input and require it to REJECT. A predicate that
                           accepts the bad input is worthless and is reported
                           as UNPROVEN, never as PASS.

THE ONE FACT EVERY O6 NEGATIVE CONTROL LEANS ON (measured, not assumed)
    runtime/src/main.rs::load_tokens calls
        load_ids(vm, b"st_gen", b"gen_tokens.bin", ...)
    for every program that DECLARES the st_gen state slot, BEFORE vm.run().
    Measured behaviour of that call (real engine, examples/p13-style probe):
        file absent        -> exit=2  "cannot open file"
        0 bytes            -> exit=9  "gen_tokens.bin is empty"
        2 bytes (half u32) -> exit=9  "gen_tokens.bin is empty"   (n = len/4 = 0)
        more u32 than slot -> exit=9  "gen_tokens.bin is longer than its slot"
        valid              -> tokens land in st_gen BEFORE run()
    Consequences for the O6 verification:
      * A gen_tokens.bin that is ABSENT must make an O6 run FAIL LOUDLY. If the
        loop were a no-op that merely echoed the file, removing the file could
        not be survivable -- so "remove the file and require a loud failure" is
        a cheap necessary condition for "the loop is real".
      * A gen_tokens.bin present but DELIBERATELY WRONG must NOT determine the
        output. If it does, the loop is a no-op and the acceptance is fake.
"""
import os
import shutil
import subprocess
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = str(Path(__file__).resolve().parents[1])
REL = os.path.join(ROOT, "target", "release")
ASM = os.path.join(REL, "vmlm-asm.exe")
ENG = os.path.join(REL, "vmlm-runtime.exe")

GEN_TOKENS = os.path.join(REL, "gen_tokens.bin")
TOKENS = os.path.join(REL, "tokens.bin")
MODEL = os.path.join(REL, "model.vmb")
RUN_OUT = os.path.join(REL, "_p14_verify_out.txt")
RUN_ERR = os.path.join(REL, "_p14_verify_err.txt")


# ----------------------------------------------------------------------------
# A. token-chain source
# ----------------------------------------------------------------------------
def write_ids(path, ids):
    """Write ids as little-endian u32, exactly as load_ids expects.

    A trailing partial element is deliberately allowed so the caller can build
    the 'file is not a whole number of u32' perturbation.
    """
    with open(path, "wb") as f:
        for v in ids:
            f.write(int(v).to_bytes(4, "little", signed=False))


def write_ids_raw(path, blob):
    with open(path, "wb") as f:
        f.write(blob)


def clear(path):
    if os.path.exists(path):
        os.remove(path)


# ----------------------------------------------------------------------------
# B. perturbation operators -- the negative-control engine
#
#    Every entry returns a NEW chain. `kind` documents what class of wrongness
#    it injects, because the point is not "some other number" but "the specific
#    thing this acceptance criterion claims to rule out".
# ----------------------------------------------------------------------------
def perturb(chain, kind, rng=None, **kw):
    c = list(int(v) for v in chain)
    if kind == "identity":
        return c
    if kind == "off_by_one":
        # Wrong token id: the single most likely real bug in a loop that feeds
        # argmax back (off-by-one on the feedback slot).
        return [max(0, v + 1) for v in c]
    if kind == "reverse":
        return list(reversed(c))
    if kind == "shift_left":
        # Feeds step k the token that belonged to step k+1: a loop whose
        # feedback slot is one step early looks exactly like this.
        return c[1:] + [c[-1] if c else 0]
    if kind == "shift_right":
        return [c[0] if c else 0] + c[:-1]
    if kind == "wrong_position":
        # Right tokens, wrong RoPE position: the class P13 warned about (the
        # failure looks like plausible numeric drift, not a crash).
        return [max(0, v - 1) for v in c]
    if kind == "all_zero":
        return [0] * len(c)
    if kind == "truncate":
        return c[: max(1, len(c) // 2)]
    if kind == "extend":
        return c + [0]
    if kind == "corrupt_one":
        i = int(kw.get("index", len(c) // 2))
        c[i] = (c[i] + 12345) % 130560  # inside MiniCPM vocab, definitely wrong
        return c
    raise ValueError("unknown perturbation %r" % kind)


# ----------------------------------------------------------------------------
# C. exact comparison that reports its own margin
# ----------------------------------------------------------------------------
def compare_bits(a, b, label="A vs B"):
    """Exact f32 bit comparison.

    Returns (identical, report dict). `identical` is True only if the raw bytes
    match -- NOT a tolerance. We also report max|d| and the number of differing
    elements so that a PASS cannot hide behind an empty comparison: if the two
    arrays have different lengths, or either is empty, that is a FAILURE of the
    comparison itself, not a pass.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    rep = {"label": label, "n_a": int(a.size), "n_b": int(b.size)}
    if a.size == 0 or b.size == 0:
        rep["note"] = "EMPTY INPUT -- comparison is not a measurement"
        return False, rep
    if a.size != b.size:
        rep["note"] = "LENGTH MISMATCH"
        return False, rep
    ab = a.tobytes()
    bb = b.tobytes()
    rep["n_diff_bytes"] = int(sum(1 for x, y in zip(ab, bb) if x != y))
    rep["n_diff_elems"] = int((a != b).sum())
    rep["max_abs_d"] = float(np.abs(a - b).max())
    rep["identical"] = bool(ab == bb)
    return bool(ab == bb), rep


def compare_chains(claimed, observed, label="chain"):
    """Exact INTEGER-chain comparison, reporting the FIRST divergence index.

    Integer chains (token ids) must be compared for exact equality -- a
    tolerance is meaningless for a token id, and "close" ids are simply
    different tokens. An empty chain is a FAILED measurement, never a pass:
    that is the "0 B was an empty measurement" lesson applied to sequences.
    """
    a = [int(x) for x in claimed]
    b = [int(x) for x in observed]
    rep = {"label": label, "n_a": len(a), "n_b": len(b)}
    if not a or not b:
        rep["note"] = "EMPTY CHAIN -- comparison is not a measurement"
        return False, rep
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            rep["first_diff"] = i
            rep["claimed_at_diff"] = a[i]
            rep["observed_at_diff"] = b[i]
            rep["n_compared"] = n
            return False, rep
    if len(a) != len(b):
        rep["note"] = "LENGTH MISMATCH"
        return False, rep
    rep["identical"] = True
    rep["n_compared"] = n
    return True, rep


def demonstrate_can_fail(chain, stages=8):
    """THE META-CHECK for a chain comparison: show it REJECTS wrong chains.

    Returns (all_rejected, rows) where rows is a list of
    (perturbation, verdict, detail). The unperturbed chain must be ACCEPTED,
    and every perturbation must be REJECTED; a perturbation that is a no-op
    (identical to the input) is reported as an empty measurement, because that
    is the exact failure mode this project was burned by before.
    """
    rows = []
    ok = True
    good, _ = compare_chains(chain, chain, "identity")
    if not good:
        rows.append(("identity", "REJECTED",
                     "*** comparison rejects the unperturbed chain ***"))
        ok = False
    else:
        rows.append(("identity", "ACCEPTED", "correct: must accept"))
    kinds = ["off_by_one", "reverse", "shift_left", "shift_right",
             "wrong_position", "all_zero", "truncate", "extend"][:stages]
    for kind in kinds:
        p = perturb(chain, kind)
        if p == list(int(x) for x in chain):
            rows.append((kind, "SAME AS INPUT",
                         "*** perturbation was a no-op -- empty measurement "
                         "***"))
            ok = False
            continue
        acc, det = compare_chains(chain, p, kind)
        if acc:
            rows.append((kind, "ACCEPTED",
                         "*** NOT DETECTED -- comparison is a rubber stamp "
                         "***"))
            ok = False
        else:
            rows.append((kind, "REJECTED",
                         "first_diff=%s" % det.get("first_diff",
                                                   det.get("note"))))
    return ok, rows


def argmax_chain(logits, vocab=None):
    """Argmax per row of a [n, vocab] logits block, as the ENGINE would see it.

    Ties break to the LOWEST index, matching the OP_TOPK_IDX contract that
    task-1 was required to implement. numpy's argmax also returns the first
    maximum, so this is the same convention -- and the verifier checks that
    convention separately with a hand-built tie, rather than trusting the
    agreement of two libraries.
    """
    lg = np.asarray(logits, dtype=np.float32)
    if vocab is not None:
        lg = lg.reshape(-1, vocab)
    return [int(i) for i in np.argmax(lg, axis=1)]


# ----------------------------------------------------------------------------
# D. THE META-CHECK
# ----------------------------------------------------------------------------
class MetaCheck:
    """Records, for every acceptance test, whether it was shown able to FAIL.

    The rule this enforces, in the project's own words: a test that passes on
    both correct and broken input is worthless. So each acceptance predicate
    must be exercised against at least one input we KNOW is broken, and must
    REJECT it. If it accepts the broken input, the criterion is reported
    UNPROVEN no matter how green the clean run looked.
    """

    def __init__(self):
        self.rows = []

    def record(self, criterion, clean_passed, bad_input_desc,
               bad_rejected, detail=""):
        proven = bool(clean_passed and bad_rejected)
        self.rows.append({
            "criterion": criterion,
            "clean_passed": bool(clean_passed),
            "bad_input": bad_input_desc,
            "bad_rejected": bool(bad_rejected),
            "can_fail": proven,
            "detail": detail,
        })
        return proven

    def report(self):
        w = 62
        print("\n" + "=" * w)
        print("META-CHECK: is each acceptance test capable of FAILING?")
        print("=" * w)
        print("%-38s %8s %9s %8s" % ("criterion", "clean", "bad->FAIL", "PROVEN"))
        print("-" * w)
        ok = True
        for r in self.rows:
            print("%-38s %8s %9s %8s" % (
                r["criterion"][:38],
                "PASS" if r["clean_passed"] else "FAIL",
                "yes" if r["bad_rejected"] else "NO",
                "YES" if r["can_fail"] else "**UNPROVEN**"))
            if not r["can_fail"]:
                ok = False
        print("-" * w)
        n_unproven = sum(1 for r in self.rows if not r["can_fail"])
        print("criteria recorded : %d" % len(self.rows))
        print("shown able to fail: %d" % (len(self.rows) - n_unproven))
        print("UNPROVEN          : %d" % n_unproven)
        return ok


def run_engine(cwd=None, expect_exit=0):
    """Run the engine with stdout/stderr to REAL FILES.

    Never a pipe: draining engine stdout through a PowerShell pipe once made a
    1.14 s run look like ~20 minutes (documented in the P13 handover), and
    verify_all.ps1 warns that a large piped output can even clobber
    $LASTEXITCODE and turn a failure into a recorded pass.
    """
    cwd = cwd or REL
    out_p = os.path.join(cwd, "_p14_verify_out.txt")
    err_p = os.path.join(cwd, "_p14_verify_err.txt")
    for p in (out_p, err_p):
        clear(p)
    with open(out_p, "wb") as fo, open(err_p, "wb") as fe:
        p = subprocess.run([ENG], cwd=cwd, stdout=fo, stderr=fe)
    with open(out_p, "rb") as f:
        out = f.read()
    with open(err_p, "rb") as f:
        err = f.read()
    return p.returncode, out, err


def run_engine_hermetic(vmb_path, tokens=None, gen_tokens=None,
                        gen_tokens_raw=None, omit_tokens=False,
                        omit_gen=False, keep=None):
    """Run the engine in a PRIVATE temp dir instead of the shared target/release.

    WHY THIS IS THE FIX AND NOT DEFENSIVE PADDING (measured, and it cost the
    team real cycles): `runtime/src/main.rs` hardcodes the RELATIVE name
    `model.vmb`, so the obvious staging approach is to cwd into
    `target/release/` and drop the artifact there. But that directory is SHARED
    by every teammate, and the observed failures were all of this shape:
      * the verifier's 858 MB artifact was replaced by another member's 268-byte
        model.vmb mid-run;
      * another member DELETED model.vmb between a copy and an exec, so retrying
        the copy cannot help -- the file disappears after the copy;
      * a correct engine produced a full sweep of failures (FAIL 20/PASS 2)
        purely from concurrent runs interleaving on that one file.
    Because the name is RELATIVE, copying the engine into a private directory
    and running it with cwd=that directory is enough to make each run hermetic:
    the result can only be about OUR program. This matters most for a 20-30
    minute run, which is exposed to the race for its whole duration.

    Returns (exit_code, stdout_bytes, stderr_bytes). The temp dir is removed.
    """
    import tempfile
    d = tempfile.mkdtemp(prefix="p14_verify_")
    try:
        shutil.copyfile(ENG, os.path.join(d, "vmlm-runtime.exe"))
        shutil.copyfile(vmb_path, os.path.join(d, "model.vmb"))
        # Verify the artifact really landed, at the right size: a truncated copy
        # here would otherwise look like an engine bug.
        if os.path.getsize(os.path.join(d, "model.vmb")) != \
                os.path.getsize(vmb_path):
            raise IOError("model.vmb copy is truncated")
        if not omit_tokens:
            if tokens is not None:
                write_ids(os.path.join(d, "tokens.bin"), tokens)
        if not omit_gen:
            if gen_tokens_raw is not None:
                with open(os.path.join(d, "gen_tokens.bin"), "wb") as f:
                    f.write(gen_tokens_raw)
            elif gen_tokens is not None:
                write_ids(os.path.join(d, "gen_tokens.bin"), gen_tokens)
        with open(os.path.join(d, "out.txt"), "wb") as fo, \
                open(os.path.join(d, "err.txt"), "wb") as fe:
            p = subprocess.run([os.path.join(d, "vmlm-runtime.exe")], cwd=d,
                               stdout=fo, stderr=fe)
        with open(os.path.join(d, "out.txt"), "rb") as f:
            out = f.read()
        with open(os.path.join(d, "err.txt"), "rb") as f:
            err = f.read()
        return p.returncode, out, err
    finally:
        if keep:
            shutil.copytree(d, keep, dirs_exist_ok=True)
        shutil.rmtree(d, ignore_errors=True)


def parse_floats(blob):
    vals = []
    for t in blob.decode("ascii", "replace").split():
        try:
            vals.append(float(t))
        except ValueError:
            pass
    return np.array(vals, dtype=np.float32)


def assemble(workdir, vml, vmb, consts):
    """Assemble, returning (ok, stdout+stderr).

    NOTE the relative-const trick: PowerShell cannot pass an assembler weight
    spec containing a drive-letter path (`name:D:\\...:rows:cols:f32`) -- it
    parses the `D:` as a drive qualifier and the assembler then panics with
    "cannot read weights file" for a file that exists. Forward slashes do not
    help. The fix is cwd=workdir plus RELATIVE specs in consts. Code that shells
    out from a .ps1 must do the same; code in Python is unaffected.
    """
    p = subprocess.run([ASM, vml, vmb] + list(consts), cwd=workdir,
                       capture_output=True, text=True, errors="replace")
    return (p.returncode == 0 and os.path.exists(os.path.join(workdir, vmb)),
            (p.stdout or "") + (p.stderr or ""))


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("This module is the verifier's harness; it is driven by "
          "tools/p14_verify.ps1 and the per-area check scripts.")
    print("It has no standalone acceptance output on purpose: every number it "
          "reports must come with the perturbation that could have failed it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
