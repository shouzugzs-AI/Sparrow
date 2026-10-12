#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P17-P4: the packaging options -- server only, cli only, or both.

THE REQUIREMENT
---------------
"打包时可选 server / cli / 两者" -- the build must be able to produce just the
HTTP server, just the CLI, or both. Cargo gives this for free via `--bin`:

    cargo build --release -p vmlm-server --bin vmlm-server   # server only
    cargo build --release -p vmlm-server --bin vmlm-cli      # cli only
    cargo build --release -p vmlm-server                     # both

"Free" is the claim that needs checking, though, and there are two ways it can be
false here:

  1. **The bins might share the library.** They do: both link `vmlm-server` (the
     lib), which holds json/http/auth/engine/vmb/tokenizer/chat. That is intended
     -- and it means "cli only" still compiles the whole library. Worth stating so
     nobody thinks `--bin vmlm-cli` produces a small binary.

  2. **`--bin` might silently build nothing, or build the wrong thing.** This
     script therefore asserts on the ARTIFACTS: after each invocation, exactly the
     expected .exe files must exist, and each must run its own self-test.

WHAT IS VERIFIED PER OPTION
---------------------------
  * the expected binaries exist and no unexpected ones appear
  * each binary actually EXECUTES (not just links): `vmlm-cli --selftest` runs the
    JSON/tokenizer/auth criteria offline, and `vmlm-server --help` prints usage
  * the CLI's `--selftest` exit code is 0, which means its embedded copies of the
    JSON and auth criteria pass

NOTE ON WHAT THIS DOES *NOT* CLAIM
----------------------------------
It does not claim the two binaries are independent of each other's CODE -- they
share a library by design. And it does not test the CLI's engine path here, because
that needs a bench engine and is covered by the CLI run in the report.

USAGE
-----
    python tools/p17_http_packaging.py

EXIT CODE: 0 all options PASS; 1 any FAIL.
"""
import hashlib
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RELEASE = os.path.join("x86_64-pc-windows-msvc", "release")
LINES = []


def emit(s=""):
    print(s)
    LINES.append(s)


def hr(t):
    emit()
    emit("=" * 78)
    emit(t)
    emit("=" * 78)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def build(target_dir, extra):
    argv = ["cargo", "build", "--release", "-p", "vmlm-server",
            "--target", "x86_64-pc-windows-msvc", "--target-dir", target_dir]
    if extra:
        argv += extra
    p = subprocess.run(argv, capture_output=True, cwd=ROOT)
    log = ((p.stdout or b"") + (p.stderr or b"")).decode("utf-8", "replace")
    return p.returncode, log, argv


def exes_in(target_dir):
    d = os.path.join(target_dir, RELEASE)
    if not os.path.isdir(d):
        return set()
    return {n for n in os.listdir(d) if n.endswith(".exe")}


OPTIONS = [
    ("server only", ["--bin", "vmlm-server"], {"vmlm-server.exe"}),
    ("cli only", ["--bin", "vmlm-cli"], {"vmlm-cli.exe"}),
    ("both", [], {"vmlm-server.exe", "vmlm-cli.exe"}),
]


def main():
    fails = []

    hr("P17-P4 PACKAGING OPTIONS -- server / cli / both")
    emit("root   = %s" % ROOT)
    emit("crate  = vmlm-server  (bins: vmlm-server, vmlm-cli; lib: vmlm_server)")
    emit()
    emit("Both binaries link the SAME library, by design: json.rs, http.rs,")
    emit("auth.rs, engine.rs, vmb.rs, tokenizer.rs, chat.rs live in the lib so")
    emit("the CLI and the server cannot drift apart in their JSON or validation")
    emit("behaviour. Consequence to state plainly: `--bin vmlm-cli` still")
    emit("compiles the whole library, so 'cli only' is not a smaller build --")
    emit("it is a build that produces one executable instead of two.")

    for label, extra, want in OPTIONS:
        hr("OPTION: %s" % label)
        td = os.path.join(ROOT, "target", "t_p17_pkg_%s" % label.replace(" ", "_"))
        shutil.rmtree(td, ignore_errors=True)
        rc, log, argv = build(td, extra)
        emit("  $ %s" % " ".join(argv))
        emit("  exit code = %d" % rc)
        if rc != 0:
            for line in log.splitlines()[-20:]:
                emit("  | %s" % line)
            fails.append("%s: build failed" % label)
            continue

        got = exes_in(td)
        emit("  executables produced: %s" % (sorted(got) or "NONE"))
        if got != want:
            emit("  !! expected exactly %s" % sorted(want))
            fails.append("%s: artifact set is %s, expected %s"
                         % (label, sorted(got), sorted(want)))
            continue

        for name in sorted(got):
            p = os.path.join(td, RELEASE, name)
            emit("    %-18s %8d B  %s" % (name, os.path.getsize(p), sha256(p)[:32] + ".."))

        # --- each binary must actually RUN, not merely link -----------------
        cli = os.path.join(td, RELEASE, "vmlm-cli.exe")
        if os.path.exists(cli):
            emit()
            emit("  executing: vmlm-cli.exe --selftest")
            p = subprocess.run([cli, "--selftest"], capture_output=True, cwd=ROOT)
            out = ((p.stdout or b"") + (p.stderr or b"")).decode("utf-8", "replace")
            for line in out.splitlines()[-6:]:
                emit("    | %s" % line)
            emit("  exit code = %d" % p.returncode)
            if p.returncode != 0:
                fails.append("%s: vmlm-cli --selftest exited %d" % (label, p.returncode))
            else:
                emit("    -> PASS: the CLI's offline criteria (JSON strictness,")
                emit("       validation, tokenizer round-trip, auth) all pass")
        else:
            emit()
            emit("  (vmlm-cli not in this option; nothing to execute)")

        srv = os.path.join(td, RELEASE, "vmlm-server.exe")
        if os.path.exists(srv):
            emit()
            emit("  executing: vmlm-server.exe --help")
            p = subprocess.run([srv, "--help"], capture_output=True, cwd=ROOT)
            out = ((p.stdout or b"") + (p.stderr or b"")).decode("utf-8", "replace")
            first = [l for l in out.splitlines() if l.strip()][:3]
            for line in first:
                emit("    | %s" % line)
            emit("  exit code = %d" % p.returncode)
            if p.returncode != 0 or "vmlm-server" not in out:
                fails.append("%s: vmlm-server --help did not print usage" % label)
            else:
                emit("    -> PASS: usage printed")

            # The refusal-to-start-without-a-token path is a packaging-relevant
            # property: the binary must not be runnable as an open server by
            # accident. Exit code 3 is the documented value.
            emit()
            emit("  executing: vmlm-server.exe --port 9   (no token)")
            p = subprocess.run([srv, "--port", "9"], capture_output=True, cwd=ROOT)
            out = ((p.stdout or b"") + (p.stderr or b"")).decode("utf-8", "replace")
            for line in out.splitlines()[:3]:
                emit("    | %s" % line)
            emit("  exit code = %d (documented: 3)" % p.returncode)
            if p.returncode != 3:
                fails.append("%s: server without a token exited %d, expected 3"
                             % (label, p.returncode))
            else:
                emit("    -> PASS: refuses to start unauthenticated")
        else:
            emit()
            emit("  (vmlm-server not in this option; nothing to execute)")

        shutil.rmtree(td, ignore_errors=True)

    hr("SUMMARY")
    emit("  options checked: %d" % len(OPTIONS))
    emit("  failed checks  : %d" % len(fails))
    for f in fails:
        emit("    - %s" % f)
    emit("  [%s]" % ("PASS" if not fails else "FAIL"))
    return 1 if fails else 0


if __name__ == "__main__":
    rc = main()
    out = os.path.join(ROOT, "examples", "p17_http", "packaging_evidence.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")
    print("\nEvidence written to examples/p17_http/packaging_evidence.txt")
    sys.exit(rc)
