#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P17-P4: prove `server/.cargo/config.toml` is either read or harmless.

THE PROBLEM BEING CHECKED
-------------------------
Cargo reads `.cargo/config.toml` from the **process working directory** upward.
It does not merge configs found further down the tree. So a config committed next
to a crate is only consulted when cargo is invoked from inside that crate's
directory -- which is easy to get wrong, and produces a file that LOOKS
authoritative while having no effect. The Lead flagged this explicitly and asked
for it to be either deleted or genuinely verified.

This script takes the "verified" route and measures two things:

  1. **Both build paths work.** `cargo build --release -p vmlm-server` from the
     workspace root, and `cargo build --release` from `server/`, must both
     succeed. (They must also be run in separate target dirs, so neither can be
     a no-op that reuses the other's artifacts -- the false-delta-0 trap.)

  2. **The server exe is not affected by the engine's link flags.** The engine's
     config carries `/NODEFAULTLIB`, which a std binary cannot link with. If
     those flags ever leaked into the server build, the link would fail with
     unresolved CRT symbols. A successful link is therefore the evidence that the
     boundary holds -- and the report shows the failure mode rather than just
     asserting the success.

WHY BOTH PATHS AND NOT JUST ONE
-------------------------------
Because "the file is dead" and "the file is load-bearing" are different facts and
the report should not have to guess which. Comparing the two outputs settles it.

⚠️ The comparison CANNOT be a raw sha256 equality, and the first version of this
script got that wrong. Measured: two builds of the SERVER from the same source
also differ byte-for-byte, because MSVC stamps a PE `TimeDateStamp` and a fresh
CodeView PDB GUID into every link. (The same effect is demonstrated as an explicit
noise-floor control for the engine in `tools/p17_http_size.py`, section 2.) So the
criterion here is the same one used there: every differing byte must lie inside a
linker build-stamp region, i.e. the emitted code and data must be identical. Any
difference outside those regions means the config really is changing codegen, and
this script then FAILS rather than reporting a conclusion either way.

USAGE
-----
    python tools/p17_http_config_check.py

EXIT CODE: 0 both paths build and produce identical code+data; 1 otherwise.
"""
import hashlib
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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


def read(p):
    with open(p, "rb") as f:
        return f.read()


def find_rsds(b):
    i = b.find(b"RSDS")
    return i if i >= 0 else None


def compare(ba, bb):
    """Compare two server binaries ignoring the linker's build stamps.

    Returns (code_and_data_identical, n_stamp_bytes, n_real_bytes, rsds_off).

    # Why this is not a plain sha256 comparison

    Measured on this tree: two builds of the SERVER from the same source also
    differ byte-for-byte, because MSVC stamps a PE `TimeDateStamp` and a fresh PDB
    GUID into every link (the identical-sources noise floor is established by
    `tools/p17_http_size.py`, section 2, for the engine). So a raw sha256
    inequality between build A and build B is NOT evidence that the cargo config
    was read on one path and not the other -- it is the expected result either way.

    The useful question is whether the differing bytes are confined to the
    linker's stamp regions. If they are, the emitted code and data are the same
    and the config is inert for codegen. If any byte outside those regions
    differs, the config genuinely changed the output and the report must say so.
    """
    if len(ba) != len(bb):
        return False, 0, abs(len(ba) - len(bb)), find_rsds(ba)
    rsds = find_rsds(ba)
    offs = [i for i in range(len(ba)) if ba[i] != bb[i]]
    stamp, real = [], []
    for o in offs:
        in_coff = o < 0x200
        in_pdb = rsds is not None and rsds <= o < rsds + 0x100
        in_debugdir = rsds is not None and rsds - 0x200 <= o < rsds
        (stamp if (in_coff or in_pdb or in_debugdir) else real).append(o)
    return len(real) == 0, len(stamp), len(real), rsds


def build(cwd, target_dir, extra=None):
    argv = ["cargo", "build", "--release", "-p", "vmlm-server",
            "--target", "x86_64-pc-windows-msvc", "--target-dir", target_dir]
    if extra:
        argv += extra
    p = subprocess.run(argv, capture_output=True, cwd=cwd)
    log = ((p.stdout or b"") + (p.stderr or b"")).decode("utf-8", "replace")
    return p.returncode, log, argv


def main():
    server_src = os.path.join(ROOT, "server")
    cfg = os.path.join(server_src, ".cargo", "config.toml")

    hr("P17-P4 CONFIG CHECK -- is server/.cargo/config.toml read, or inert?")
    emit("workspace root : %s" % ROOT)
    emit("server dir     : %s" % server_src)
    emit("config file    : %s (%s)" % (cfg, "present" if os.path.exists(cfg) else "ABSENT"))
    emit()
    emit("Cargo's rule, measured on this tree in the opposite direction: a")
    emit("config is read from the CWD upward and NOT merged from below. The")
    emit("engine build from the workspace root fails LNK1561 for exactly this")
    emit("reason (see tools/p17_http_size.py section 0).")

    fails = []
    results = {}

    for label, cwd in (("A. from workspace root", ROOT),
                       ("B. from server/", server_src)):
        hr("%s   (cwd=%s)" % (label, cwd))
        td = os.path.join(ROOT, "target", "t_p17_cfg_%s" % label[0])
        shutil.rmtree(td, ignore_errors=True)
        rc, log, argv = build(cwd, td)
        emit("  $ %s" % " ".join(argv))
        emit("  exit code = %d" % rc)
        exe = os.path.join(td, "x86_64-pc-windows-msvc", "release", "vmlm-server.exe")
        if rc != 0:
            for line in log.splitlines()[-20:]:
                emit("  | %s" % line)
            fails.append("%s did not build" % label)
            continue
        if not os.path.exists(exe):
            emit("  !! build reported success but %s is missing" % exe)
            fails.append("%s produced no exe" % label)
            continue
        s, h = os.path.getsize(exe), sha256(exe)
        results[label] = (s, h, read(exe))
        emit("  size   = %d B" % s)
        emit("  sha256 = %s" % h)
        # The link succeeding IS the boundary evidence: /NODEFAULTLIB would have
        # produced unresolved CRT symbols here.
        emit("  link succeeded -> the engine's /NODEFAULTLIB did NOT leak into")
        emit("  this std build (it would have failed with `unresolved external")
        emit("  symbol _beginthreadex` / `_errno` style errors).")
        shutil.rmtree(td, ignore_errors=True)

    hr("VERDICT")
    if len(results) == 2:
        s1, h1, b1 = results["A. from workspace root"]
        s2, h2, b2 = results["B. from server/"]
        emit("  A (root)     : %d B  %s" % (s1, h1))
        emit("  B (server/)  : %d B  %s" % (s2, h2))
        emit("  raw sha256 identical : %s" % (h1 == h2))
        same, nstamp, nreal, rsds = compare(b1, b2)
        emit("  RSDS (PDB) at offset : %s" % rsds)
        emit("  differing bytes      : %d stamp, %d outside stamp regions"
             % (nstamp, nreal))
        emit("  code+data identical  : %s" % same)
        emit()
        if same:
            emit("  => server/.cargo/config.toml does NOT change the generated code")
            emit("     or data on either build path. The raw sha256 disparity is the")
            emit("     linker's PE TimeDateStamp / PDB GUID, which differs between")
            emit("     ANY two links of the same source -- not evidence about the")
            emit("     config.")
            emit()
            emit("     It is kept as DOCUMENTATION OF THE BOUNDARY: it states why the")
            emit("     engine's /NODEFAULTLIB must not be copied into a std crate,")
            emit("     and this script is the check that keeps the claim honest. If a")
            emit("     future edit makes the paths differ outside the stamp regions,")
            emit("     this script FAILS and the file has become load-bearing.")
        else:
            emit("  => The two paths DIFFER OUTSIDE the linker stamp regions, so the")
            emit("     config IS read on path B and changes the output. The file is")
            emit("     load-bearing and the report must say so.")
            fails.append("config changes generated code (paths differ outside stamps)")
    elif not fails:
        fails.append("not all build paths produced a binary")

    hr("SUMMARY")
    emit("  failed checks: %d" % len(fails))
    for f in fails:
        emit("    - %s" % f)
    emit("  [%s]" % ("PASS" if not fails else "FAIL"))
    return 1 if fails else 0


if __name__ == "__main__":
    rc = main()
    out = os.path.join(ROOT, "examples", "p17_http", "config_evidence.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")
    print("\nEvidence written to examples/p17_http/config_evidence.txt")
    sys.exit(rc)
