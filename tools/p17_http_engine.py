#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P17-P4: build the bench-feature engine the HTTP server drives, at a stable path.

WHY THIS IS A SCRIPT AND NOT A LINE IN THE REPORT
-------------------------------------------------
`vmlm-server` drives the engine through `vmlm-runtime <model.vmb> reps:slot`, and
**only the `bench` feature provides that argv interface**. The default inference
build takes no arguments: measured, it exits **0** and writes **no `out.bin`** even
when handed a valid model and a slot index.

That makes the bench build a real dependency of the P4 acceptance run, not an
incidental artifact. Leaving it in an ad-hoc `target/t_p17_*` directory caused a
regression during this task: a cleanup pass removed that directory, and the very
next verification run reported 4 FAILs (the 200-path cases became 503). The
verifier's guard said `NO BENCH-FEATURE ENGINE FOUND` and was correct -- the
reported statuses were honest -- but the failure looked like a server defect.

So the bench engine gets a documented, stable location:

    target/bench/x86_64-pc-windows-msvc/release/vmlm-runtime.exe

which is also the FIRST path `server/src/engine.rs` probes. Building it there means
the server's auto-detection works without `--engine`.

WHY THE BUILD RUNS WITH cwd = runtime/
--------------------------------------
Measured on this tree: from the workspace root the same command fails with
`LNK1561: 必须定义入口点`, because cargo reads `.cargo/config.toml` from the cwd
upward only and the engine's `/ENTRY:mainCRTStartup` + `/NODEFAULTLIB` live in
`runtime/.cargo/config.toml`. This script always sets the cwd and prints it.

USAGE
-----
    python tools/p17_http_engine.py
    python tools/p17_http_engine.py --out <dir>    # custom target dir

EXIT CODE: 0 the engine is present and provably slot-dumping; 1 otherwise.
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REL = os.path.join("x86_64-pc-windows-msvc", "release", "vmlm-runtime.exe")
LINES = []


def emit(s=""):
    print(s)
    LINES.append(s)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def dumps_a_slot(exe, model):
    """True iff `exe` runs the model and writes a non-empty out.bin.

    This is the ONLY reliable test. Measured: with a bad model path the default
    build and the bench build are indistinguishable (both print "cannot open file"
    and exit 2, because argv[1] is read as a model path before any feature-gated
    code runs). So the probe needs a model that loads.
    """
    if not model or not os.path.exists(model):
        return False, "no model available to probe with"
    import tempfile
    with tempfile.TemporaryDirectory(prefix="vmlm_bench_probe_") as d:
        shutil.copyfile(model, os.path.join(d, "model.vmb"))
        ob = os.path.join(d, "out.bin")
        p = subprocess.run([exe, "model.vmb", "1:0"], capture_output=True, cwd=d,
                           timeout=120)
        if os.path.exists(ob) and os.path.getsize(ob) > 0:
            return True, "%d B out.bin" % os.path.getsize(ob)
        text = ((p.stdout or b"") + (p.stderr or b"")).decode("utf-8", "replace")
        return False, "rc=%d, no out.bin; %s" % (p.returncode, text.strip()[:200])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "target", "bench"),
                    help="target dir for the bench build (default target/bench, "
                         "which is where the server looks first)")
    a = ap.parse_args()

    emit("=" * 78)
    emit("P17-P4 BENCH ENGINE -- the artifact the HTTP server drives")
    emit("=" * 78)
    emit("target dir : %s" % a.out)
    emit("build cwd  : %s   (required: see the module docstring)"
         % os.path.join(ROOT, "runtime"))
    emit()
    emit("Only the `bench` feature gives vmlm-runtime its argv interface")
    emit("(`<model.vmb> [reps[:slot]]` + an out.bin dump). The default build")
    emit("ignores argv entirely: exit 0, no output file.")

    runtime = os.path.join(ROOT, "runtime")
    argv = ["cargo", "build", "--release", "-p", "vmlm-runtime",
            "--features", "bench",
            "--target", "x86_64-pc-windows-msvc",
            "--target-dir", a.out]
    emit()
    emit("$ %s" % " ".join(argv))
    p = subprocess.run(argv, capture_output=True, cwd=runtime)
    log = ((p.stdout or b"") + (p.stderr or b"")).decode("utf-8", "replace")
    emit("  exit code = %d" % p.returncode)
    if p.returncode != 0:
        for line in log.splitlines()[-25:]:
            emit("  | %s" % line)
        return 1

    exe = os.path.join(a.out, REL)
    if not os.path.exists(exe):
        emit("  !! build succeeded but %s is missing" % exe)
        return 1
    emit("  size   = %d B" % os.path.getsize(exe))
    emit("  sha256 = %s" % sha256(exe))
    emit("  (note: the engine is NOT byte-reproducible across links -- MSVC stamps")
    emit("   a PE TimeDateStamp and a fresh PDB GUID. Different sha256 across two")
    emit("   builds is expected; see tools/p17_http_size.py section 2.)")

    # ---- prove it dumps a slot ------------------------------------------
    model = os.path.join(ROOT, "examples", "p17_http", "model.vmb")
    if not os.path.exists(model):
        # CHICKEN-AND-EGG -- and a CLEAN-CHECKOUT simulation caught this:
        # the capability probe needs a model, and `tools/p17_http_model.py` wants an
        # ENGINE (for its own smoke test). Running this script first on a clean tree
        # therefore printed
        #
        #     result: NO (no model available to probe with)
        #     FAIL: this engine cannot dump a slot
        #
        # which is doubly wrong: the engine was fine, and the message blamed the
        # engine for a missing fixture. The engine EXISTS by this point in the
        # script, so the fixture can just be built now.
        emit()
        emit("=== fixture model.vmb is absent -- building it now ===")
        emit("(the capability probe needs a model, and the model's smoke test needs")
        emit(" an engine; this script builds the engine first, so it builds the")
        emit(" fixture here instead of reporting a misleading engine failure)")
        gen = os.path.join(ROOT, "tools", "p17_http_model.py")
        gargv = [sys.executable, gen, "--engine", exe]
        emit("$ %s" % " ".join(gargv))
        g = subprocess.run(gargv, capture_output=True, cwd=ROOT)
        gout = ((g.stdout or b"") + (g.stderr or b"")).decode("utf-8", "replace")
        for line in gout.splitlines()[-8:]:
            emit("  | %s" % line)
        emit("  exit code = %d" % g.returncode)
        if g.returncode != 0 or not os.path.exists(model):
            emit()
            emit("  FAIL: could not build the fixture, so the engine's dump")
            emit("  capability cannot be proven.")
            return 1

    emit()
    emit("=== capability probe: does it write out.bin? ===")
    ok, why = dumps_a_slot(exe, model)
    emit("  model: %s" % model)
    emit("  result: %s (%s)" % ("YES" if ok else "NO", why))
    if not ok:
        emit()
        emit("  FAIL: this engine cannot dump a slot, so the server's success path")
        emit("  cannot work. Build the fixture first:")
        emit("    python tools/p17_http_model.py --engine \"%s\"" % exe)
        return 1

    emit()
    emit("  PASS: the engine accepts `model.vmb 1:0` and writes out.bin.")
    emit("  The server will find it at this path automatically (it is the first")
    emit("  candidate server/src/engine.rs probes).")
    emit()
    emit("[PASS]")
    return 0


if __name__ == "__main__":
    rc = main()
    out = os.path.join(ROOT, "examples", "p17_http", "engine_evidence.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")
    print("\nEvidence written to examples/p17_http/engine_evidence.txt")
    sys.exit(rc)
