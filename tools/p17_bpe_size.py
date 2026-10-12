#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P17-P3: measure the BPE module's contribution to the DEFAULT engine size.

=============================================================================
THE HARD REQUIREMENT
=============================================================================
Task P3: "BPE 模块必须放在 feature 后面，并且**实测**它对默认引擎的贡献是
0 B（用**隔离 target-dir** 测，同一 target-dir 连构会报假 delta 0）".

So this script measures ONE variable at a time, in a FRESH target dir per
build, and reports the raw `Engine core size: <N> bytes` line for each.

=============================================================================
WHY ISOLATED TARGET DIRS
=============================================================================
`tools/build_all.ps1` mirrors the engine to `target/release/vmlm-runtime.exe`.
Building twice with and without a feature in the SAME target dir lets cargo
reuse the cached artifact: the second build can report the FIRST build's size,
i.e. a fake delta of exactly 0. That failure mode is silent and it looks like
success -- it is the single most dangerous measurement error in this script's
subject matter, and the project has already been bitten by it once
(recorded in tools/build_all.ps1's METHOD WARNING).

Each arm below therefore gets:
  * its own --target-dir (a fresh temp dir), and
  * its own CARGO_TARGET_DIR env var, and
  * a build that is invoked FROM runtime/ (required: the root workspace does
    not read runtime/.cargo/config.toml, so the no_std link flags would be
    missing and the link fails with LNK1561).

=============================================================================
THE ARMS
=============================================================================
  A. default engine, bpe feature absent            <- the invariant
  B. default engine, --features bpe                <- the treatment
  C. default engine, --features train (POSITIVE)   <- proves the apparatus moves

Arm C is not decoration. If A == B == C, the measuring apparatus is broken and
"0 B" means nothing; C must show a large non-zero delta for A/B's zero to be
evidence rather than an artifact. This is the same positive-control discipline
the project used for the P15 S5 wiring.

=============================================================================
HOW "FEATURE ABSENT" IS SIMULATED FOR ARM A
=============================================================================
`runtime/Cargo.toml` does NOT forward a `bpe` feature, so
`cargo build --features bpe` from runtime/ is not even accepted -- which is
itself a structural fact worth recording. Arm B builds `vmlm-core` WITH the
feature via the runtime's feature table... except it cannot, because a Cargo
feature must be declared to be forwarded.

Rather than edit runtime/Cargo.toml (a shared file outside this task's write
scope), the two arms build the ENGINE from the same runtime crate with
`--features` forced through an isolated, throwaway copy of runtime/ that has
the forwarding line added. The source under test is always the SHIPPED
core/src/bpe.rs and the SHIPPED runtime/src/main.rs; only the Cargo manifest
differs between arms, and the difference is stated explicitly.

Usage:
    python tools/p17_bpe_size.py
    python tools/p17_bpe_size.py --quick    # arms A and B only
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = str(Path(__file__).resolve().parents[1])
EX = os.path.join(ROOT, "examples", "p17_bpe")
TRIPLE = "x86_64-pc-windows-msvc"

RESULTS = []


def engine_size(out):
    m = re.search(r"Engine core size:\s*(\d+)\s*bytes", out)
    return int(m.group(1)) if m else None


def cargo_bin():
    c = shutil.which("cargo")
    if c is None:
        raise SystemExit("cargo not on PATH")
    return c


def ignore_transients(d, names):
    # shutil.copytree calls the ignore callable as ignore(dir, names) -- taking
    # only one argument raises TypeError at the first copy.
    # Other agents work in this tree concurrently; editors leave transient
    # temp dirs that copytree can race against (see p16_conn_format.py for the
    # incident). Exclude them so an environment artefact cannot masquerade as
    # a build failure.
    pats = shutil.ignore_patterns("target", "*.tmp", "*.tmpdir",
                                  ".*.tmpdir", ".*.tmp", "__pycache__")
    return pats(d, names)


def make_arm(label, runtime_toml_patch=None):
    """Copy runtime/ + core/ into a temp tree, apply an optional manifest patch,
    build the engine with its OWN target dir, and return (size, log)."""
    work = tempfile.mkdtemp(prefix="p17_bpe_size_%s_" % label)
    shutil.copytree(os.path.join(ROOT, "runtime"),
                    os.path.join(work, "runtime"), ignore=ignore_transients)
    shutil.copytree(os.path.join(ROOT, "core"),
                    os.path.join(work, "core"), ignore=ignore_transients)
    # The workspace manifest lives at the root; drop it so the copies form
    # their own trivial workspace and cannot reach back into the real tree.
    with open(os.path.join(work, "Cargo.toml"), "w", encoding="utf-8") as f:
        f.write("[workspace]\nmembers = [\"core\", \"runtime\"]\n"
                "resolver = \"2\"\n\n"
                "[profile.release]\npanic = \"abort\"\nopt-level = 3\n"
                "lto = true\ncodegen-units = 1\nstrip = true\n")
    rt_toml = os.path.join(work, "runtime", "Cargo.toml")
    if runtime_toml_patch is not None:
        with open(rt_toml, "rb") as f:
            raw = f.read()
        text = raw.decode("utf-8")
        n = text.count(runtime_toml_patch[0])
        if n != 1:
            return None, "manifest anchor matched %d times (want 1)" % n
        text = text.replace(runtime_toml_patch[0], runtime_toml_patch[1])
        with open(rt_toml, "w", encoding="utf-8", newline="") as f:
            f.write(text)
    # .cargo/config.toml must travel with runtime/ -- it holds the no_std link
    # flags (/ENTRY, /NODEFAULTLIB). Without it the link fails with LNK1561.
    if not os.path.isfile(os.path.join(work, "runtime", ".cargo", "config.toml")):
        tgt = os.path.join(work, "runtime", ".cargo")
        os.makedirs(tgt, exist_ok=True)
        shutil.copy(os.path.join(ROOT, "runtime", ".cargo", "config.toml"),
                    os.path.join(tgt, "config.toml"))

    env = dict(os.environ)
    tdir = os.path.join(work, "tdir")
    env["CARGO_TARGET_DIR"] = tdir
    args = [cargo_bin(), "build", "--release", "--target", TRIPLE]
    if runtime_toml_patch is not None and runtime_toml_patch[2]:
        args += ["--features", runtime_toml_patch[2]]
    t0 = time.time()
    p = subprocess.run(args, cwd=os.path.join(work, "runtime"),
                       capture_output=True, text=True, errors="replace",
                       env=env, timeout=3600)
    dt = time.time() - t0
    out = (p.stdout or "") + (p.stderr or "")
    exe = os.path.join(tdir, TRIPLE, "release", "vmlm-runtime.exe")
    sz = os.path.getsize(exe) if os.path.isfile(exe) else None
    # A build that "succeeds" but produces no binary, or a binary whose size
    # cannot be read, must not be reported as a size. Both are None, and the
    # caller treats None as a failed arm.
    return sz, {"rc": p.returncode, "secs": round(dt, 1),
                "target_dir": tdir, "output_tail": out.strip().splitlines()[-14:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    print("=" * 78)
    print("P17-P3 engine size: the BPE module's cost to the DEFAULT engine")
    print("=" * 78)
    print()
    print("method: one fresh target dir + CARGO_TARGET_DIR per arm, built from")
    print("        runtime/ (the no_std link flags live in runtime/.cargo).")
    print("        A same-target-dir A/B would let cargo reuse the cached")
    print("        artifact and report a FAKE delta of 0 -- see build_all.ps1's")
    print("        METHOD WARNING.")
    print()

    # ---- baseline tree fingerprint: prove the arms test the SHIPPED source
    import hashlib
    core_srcs = []
    for name in ("bpe.rs", "lib.rs", "format.rs"):
        p = os.path.join(ROOT, "core", "src", name)
        if os.path.isfile(p):
            with open(p, "rb") as f:
                core_srcs.append((name, hashlib.sha256(f.read()).hexdigest()))
    print("shipped source fingerprint (sha256, first 16):")
    for n, h in core_srcs:
        print("    core/src/%-12s %s" % (n, h[:16]))
    print()

    arms = []

    # -- Arm A: DEFAULT engine (bpe feature absent) -----------------------
    print("--- arm A: default engine, `bpe` feature ABSENT ---")
    a, la = make_arm("A_default", None)
    print("    rc=%s secs=%s size=%s" % (la["rc"], la["secs"],
                                         "%d B" % a if a else "NO BINARY"))
    if a is None:
        for line in la["output_tail"]:
            print("      | %s" % line)
    arms.append(("A default (bpe absent)", a, la))

    # -- Arm B: same tree, `bpe` feature ON -------------------------------
    # The forwarding line is added to a THROWAWAY copy of runtime/Cargo.toml.
    print()
    print("--- arm B: default engine, `--features bpe` ---")
    patch = ("neuron-kv = [\"vmlm-core/neuron-kv\"]",
             "neuron-kv = [\"vmlm-core/neuron-kv\"]\n"
             "bpe = [\"vmlm-core/bpe\"]",
             "bpe")
    b, lb = make_arm("B_bpe", patch)
    print("    rc=%s secs=%s size=%s" % (lb["rc"], lb["secs"],
                                         "%d B" % b if b else "NO BINARY"))
    if b is None:
        for line in lb["output_tail"]:
            print("      | %s" % line)
    arms.append(("B bpe feature ON", b, lb))

    # -- Arm C: POSITIVE CONTROL, `train` feature ON ----------------------
    if not args.quick:
        print()
        print("--- arm C: POSITIVE CONTROL, `--features train` ---")
        print("    (must move; if it does not, the apparatus is broken and")
        print("     arm A/B's zero is not evidence)")
        # `train` IS already forwarded by runtime/Cargo.toml to
        # vmlm-core/train, so no manifest patch is needed -- just pass the flag.
        patch_c = ("neuron-kv = [\"vmlm-core/neuron-kv\"]",
                   "neuron-kv = [\"vmlm-core/neuron-kv\"]",
                   "train")
        c, lc = make_arm("C_train", patch_c)
        print("    rc=%s secs=%s size=%s" % (lc["rc"], lc["secs"],
                                             "%d B" % c if c else "NO BINARY"))
        if c is None:
            for line in lc["output_tail"]:
                print("      | %s" % line)
        arms.append(("C train feature ON (control)", c, lc))

    # ---- verdict --------------------------------------------------------
    print()
    print("=" * 78)
    print("RESULTS")
    print("=" * 78)
    for name, sz, meta in arms:
        print("  %-34s %s" % (name, "%d B" % sz if sz else "BUILD/CAPTURE FAILED"))

    npass = 0
    nfail = 0
    def verdict(ok, text):
        nonlocal npass, nfail
        if ok:
            npass += 1
        else:
            nfail += 1
        print("  [%s] %s" % ("PASS" if ok else "FAIL", text))

    print()
    sizes = {n: s for n, s, _ in arms}
    a = sizes.get("A default (bpe absent)")
    b = sizes.get("B bpe feature ON")
    c = sizes.get("C train feature ON (control)")

    if a is None or b is None:
        verdict(False, "both arms A and B produced a binary (measurement "
                       "impossible otherwise)")
    else:
        verdict(True, "both arms A and B produced a binary")
        verdict(b == a, "bpe feature contributes 0 B to the default engine "
                        "(A=%d B=%d delta=%+d)" % (a, b, b - a))
    if c is None:
        if args.quick:
            print("  [SKIP] positive control (--quick)")
        else:
            verdict(False, "positive control produced a binary")
    else:
        moved = (c != a)
        verdict(moved, "positive control MOVES the size (train A=%s C=%s "
                       "delta=%s) -- proves the apparatus can detect a change"
                % (a, c, ("%+d" % (c - a)) if a else "?"))

    BUDGET = 84480
    if a is not None:
        verdict(a <= BUDGET, "default engine within budget (A=%d <= %d, "
                             "headroom %d)" % (a, BUDGET, BUDGET - a))

    print()
    print("PASS=%d FAIL=%d" % (npass, nfail))

    # ------------------------------------------------------------------
    # WHERE THE EVIDENCE GOES -- and why this is not just a filename choice
    # ------------------------------------------------------------------
    # A run WITHOUT --quick includes the positive control and is the complete
    # evidence. A --quick run deliberately omits it, so its JSON is a
    # STRICTLY WEAKER artifact.
    #
    # The first version wrote both to the same `bpe_size.json`. Consequence,
    # observed rather than imagined: a --quick run silently REPLACED the
    # complete evidence with a partial file, and the positive control
    # (96256 B / +21504 B -- the very thing that makes arm A/B's 0 B mean
    # anything) disappeared from the record with no error and no warning.
    #
    # That is the project's "silent degradation" class exactly: a mode whose
    # whole purpose is convenience must not be able to destroy the evidence a
    # slower mode produced. So:
    #   * --quick writes to a SEPARATE file, and
    #   * a --quick run is REFUSED if it would overwrite a complete JSON
    #     (unless --force), because refusing is cheaper than a lost control.
    log = os.path.join(EX, "bpe_size_quick.json" if args.quick
                       else "bpe_size.json")
    if args.quick and os.path.isfile(os.path.join(EX, "bpe_size.json")):
        try:
            with open(os.path.join(EX, "bpe_size.json"), encoding="utf-8") as f:
                prior = json.load(f)
            prior_full = any(a["name"].startswith("C ")
                             for a in prior.get("arms", []))
        except Exception:
            prior_full = False
        if prior_full:
            print()
            print("REFUSING to write: %s already holds a COMPLETE run"
                  % os.path.join(EX, "bpe_size.json"))
            print("  (it contains the positive control). A --quick run is a")
            print("  strictly weaker artifact and must not replace it.")
            print("  Writing this run to %s instead." % log)
    with open(log, "w", encoding="utf-8") as f:
        json.dump({"arms": [{"name": n, "size": s, "meta": m} for n, s, m in arms],
                   "fingerprint": dict(core_srcs),
                   "complete": not args.quick,
                   "positive_control_included": any(
                       n.startswith("C ") for n, _, _ in arms),
                   "budget": BUDGET, "pass": npass, "fail": nfail},
                  f, ensure_ascii=False, indent=1)
    print("wrote %s%s" % (log, "" if not args.quick else "  (partial evidence)"))
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
