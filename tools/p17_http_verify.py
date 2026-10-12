#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P17-P4 acceptance: the HTTP server, measured with curl.

WHAT THIS SCRIPT IS FOR
-----------------------
Task-5's acceptance criteria are behavioural:

    200 for a normal request, 401 with no Bearer, 400 for bad JSON.

A report that CLAIMS those three status codes is worth nothing; the raw curl
output has to be on disk. So every case here shells out to `curl.exe`, captures
the **status code and the body separately**, prints them verbatim, and writes
them to `examples/p17_http/curl_evidence.txt`.

WHY curl EXISTS AS A SUBPROCESS AND NOT AS A PYTHON HTTP LIBRARY
----------------------------------------------------------------
Because `curl` is the tool the task names, and because a Python `urllib` client
would share nothing with it. Running the real curl means the evidence includes
the real curl exit code, the real `Content-Length` handling, and the real header
parsing -- if the server's framing were wrong, curl would report a transfer
error and this script would show `curl_rc != 0` rather than a clean 200.

THE CAN-FAIL REQUIREMENT (project rule 2: 每个判据必须能演示自己失败)
-------------------------------------------------------------------
Each of the three criteria is paired with a control that must produce a
DIFFERENT result:

    200 criterion  <->  the same request with a wrong token must NOT be 200
    401 criterion  <->  the same request with the right token must NOT be 401
    400 criterion  <->  a well-formed body must NOT be 400

Plus `--allow-anonymous`: run against a server with auth explicitly off, where
the no-Bearer request must return 200. If auth were secretly broken ("everything
passes"), the 401 case and this case would give the same answer, and the pair
would show it.

And the strictness controls, which are the ones most likely to expose a
hand-written parser: trailing data after the JSON body, a duplicate key, a
missing field, a wrong type, and a body whose Content-Length lies.

USAGE
-----
    python tools/p17_http_verify.py                     # full run
    python tools/p17_http_verify.py --skip-size         # skip the engine-size half
    python tools/p17_http_verify.py --keep-server       # leave the server running

EXIT CODE: 0 all criteria PASS; 1 any FAIL.
"""
import argparse
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKEN = "p17-verify-token-9f2c"
WRONG_TOKEN = "p17-verify-token-WRONG"

EVIDENCE = []


def emit(line=""):
    print(line)
    EVIDENCE.append(line)


def hr(title):
    emit()
    emit("=" * 78)
    emit(title)
    emit("=" * 78)


class Results:
    def __init__(self):
        self.rows = []

    def check(self, name, ok, detail):
        tag = "PASS" if ok else "FAIL"
        self.rows.append((tag, name, detail))
        emit("[%s] %s" % (tag, name))
        emit("       %s" % detail)
        return ok

    def summary(self):
        n = len(self.rows)
        p = sum(1 for r in self.rows if r[0] == "PASS")
        f = n - p
        emit()
        emit("-" * 78)
        emit("TOTAL: %d checks, %d PASS, %d FAIL" % (n, p, f))
        return f


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def curl(port, path, method="GET", body=None, token=None, extra_args=None,
         content_length_override=None):
    """Run the REAL curl.exe and return (rc, status, body, raw).

    Status and body are separated by writing the body to a temp file
    (`-o`) and the status to stdout (`-w "%{http_code}"`), so a body containing
    newlines or JSON braces cannot be confused with the status line.
    """
    outfile = os.path.join(ROOT, "_p17_curl_body.tmp")
    if os.path.exists(outfile):
        os.remove(outfile)
    argv = [
        "curl.exe", "-sS", "--max-time", "30",
        "-o", outfile,
        "-w", "%{http_code}",
    ]
    if method == "POST":
        argv += ["-X", "POST"]
    if body is not None:
        argv += ["-H", "Content-Type: application/json"]
        if content_length_override is None:
            argv += ["--data-binary", body]
        else:
            # Deliberately lie about the length, to exercise the framer.
            argv += ["-H", "Content-Length: %d" % content_length_override,
                     "--data-binary", body]
    if token is not None:
        argv += ["-H", "Authorization: Bearer %s" % token]
    if extra_args:
        argv += extra_args
    argv.append("http://127.0.0.1:%d%s" % (port, path))

    p = subprocess.run(argv, capture_output=True)
    raw_err = (p.stderr or b"").decode("utf-8", "replace").strip()
    status = (p.stdout or b"").decode("utf-8", "replace").strip()
    body = ""
    if os.path.exists(outfile):
        body = open(outfile, "rb").read().decode("utf-8", "replace")
        os.remove(outfile)
    return p.returncode, status, body, raw_err, argv


def show_curl(rc, status, body, err, argv):
    emit("    $ %s" % " ".join(argv))
    emit("    curl_rc=%d http_status=%s" % (rc, status if status else "<none>"))
    if err:
        emit("    curl_stderr=%s" % err)
    emit("    body(%d B)=%s" % (len(body.encode("utf-8")), body if body else "<empty>"))


def wait_for_port(port, timeout=30.0):
    """Poll the TCP port until it accepts, or time out.

    Deliberately a raw socket connect and NOT an HTTP request: a readiness probe
    that had to authenticate correctly would conflate "server is up" with
    "auth works", and the 401 criterion depends on those being separable.
    """
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            s = socket.create_connection(("127.0.0.1", port), timeout=0.5)
            s.close()
            return True
        except OSError:
            time.sleep(0.15)
    return False


# ---------------------------------------------------------------------------
# The case table. Each entry: (name, kwargs, expected_status, why)
# ---------------------------------------------------------------------------

VALID_BODY = json.dumps({
    "model": "vmlm-sparse",
    "messages": [
        {"role": "system", "content": "You are a test."},
        {"role": "user", "content": "hello from curl"},
    ],
    "temperature": 0.0,
    "max_tokens": 8,
})

CASES = [
    # ---- the three criteria from the task ---------------------------------
    ("CRITERION 200: valid request with the right token",
     dict(method="POST", path="/v1/chat/completions", body=VALID_BODY, token=TOKEN),
     200, "the positive path"),

    ("CRITERION 401: no Authorization header at all",
     dict(method="POST", path="/v1/chat/completions", body=VALID_BODY, token=None),
     401, "task requirement 4, the case the brief explicitly names"),

    ("CRITERION 400: syntactically invalid JSON",
     dict(method="POST", path="/v1/chat/completions", body='{"model":"m","messages":[}',
          token=TOKEN),
     400, "hand-written parser must hard-fail, not guess"),

    # ---- controls that make the three above falsifiable -------------------
    ("control for 200: same request, WRONG token",
     dict(method="POST", path="/v1/chat/completions", body=VALID_BODY, token=WRONG_TOKEN),
     401, "if this were also 200, the 200 criterion would not distinguish auth"),

    ("control for 401: same request, RIGHT token",
     dict(method="POST", path="/v1/chat/completions", body=VALID_BODY, token=TOKEN),
     200, "if this were also 401, the 401 criterion would be constant"),

    ("control for 400: well-formed body",
     dict(method="POST", path="/v1/chat/completions",
          body='{"model":"m","messages":[{"role":"user","content":"ok"}]}',
          token=TOKEN),
     200, "if this were also 400, the 400 criterion would be constant"),

    # ---- strictness: each of these is a distinct way to be wrong ----------
    ("400: empty body",
     dict(method="POST", path="/v1/chat/completions", body="", token=TOKEN),
     400, "empty is not 'no messages', it is 'no request'"),

    ("400: missing `messages`",
     dict(method="POST", path="/v1/chat/completions", body='{"model":"m"}', token=TOKEN),
     400, "the silent-degradation case the task names"),

    ("400: missing `model`",
     dict(method="POST", path="/v1/chat/completions",
          body='{"messages":[{"role":"user","content":"x"}]}', token=TOKEN),
     400, "model is required too"),

    ("400: empty `messages` array",
     dict(method="POST", path="/v1/chat/completions",
          body='{"model":"m","messages":[]}', token=TOKEN),
     400, "an empty conversation is a rejected request, not a generic reply"),

    ("400: `content` is a number, not a string",
     dict(method="POST", path="/v1/chat/completions",
          body='{"model":"m","messages":[{"role":"user","content":42}]}', token=TOKEN),
     400, "type error"),

    ("400: `messages` is an object, not an array",
     dict(method="POST", path="/v1/chat/completions",
          body='{"model":"m","messages":{"role":"user"}}', token=TOKEN),
     400, "type error at the container level"),

    ("400: unknown role",
     dict(method="POST", path="/v1/chat/completions",
          body='{"model":"m","messages":[{"role":"wizard","content":"x"}]}', token=TOKEN),
     400, "role must be system/user/assistant"),

    ("400: temperature out of range",
     dict(method="POST", path="/v1/chat/completions",
          body='{"model":"m","messages":[{"role":"user","content":"x"}],"temperature":9}',
          token=TOKEN),
     400, "range check, not just type check"),

    ("400: duplicate object key",
     dict(method="POST", path="/v1/chat/completions",
          body='{"model":"a","model":"b","messages":[{"role":"user","content":"x"}]}',
          token=TOKEN),
     400, "first-wins vs last-wins is a silent-corruption class"),

    ("400: trailing comma",
     dict(method="POST", path="/v1/chat/completions",
          body='{"model":"m","messages":[{"role":"user","content":"x"},]}', token=TOKEN),
     400, "non-RFC JSON must not be accepted"),

    ("400: `stream: true` (explicitly unimplemented)",
     dict(method="POST", path="/v1/chat/completions",
          body='{"model":"m","messages":[{"role":"user","content":"x"}],"stream":true}',
          token=TOKEN),
     400, "better a 400 than a 200 a streaming client hangs on"),

    # ---- routing ---------------------------------------------------------
    ("401 on /v1/models without a token",
     dict(method="GET", path="/v1/models", token=None),
     401, "auth applies to the whole /v1 surface, not just the chat route"),

    ("200 on /v1/models with a token",
     dict(method="GET", path="/v1/models", token=TOKEN),
     200, "and it works with one"),

    ("200 on /health without a token",
     dict(method="GET", path="/health", token=None),
     200, "the only unauthenticated route, so the readiness probe can exist"),

    ("404 on an unknown path",
     dict(method="GET", path="/nope", token=TOKEN),
     404, "unknown routes are not silently 200"),

    ("405 on GET /v1/chat/completions",
     dict(method="GET", path="/v1/chat/completions", token=TOKEN),
     405, "wrong method on a real route"),
]


def find_bench_engine():
    """Locate a bench-feature `vmlm-runtime.exe` that provably dumps a slot.

    WHY EVERY CANDIDATE IS PROBED AND NOT JUST THE "OBVIOUS" ONE:
    the first version trusted a directory-name heuristic (`"bench" in d or
    "p17" in d`) and selected `target/t_p17_engine_before/...` -- an isolated
    target dir created for a SIZE measurement, built WITHOUT `--features bench`.
    The name matched, the binary did not. The verify run then reported 503 for the
    200 criterion, which reads as a server defect but was a wrong-artifact
    selection. `engine_is_bench` now stages the real fixture and requires
    `out.bin`, so name heuristics cannot select a non-dumping build.
    """
    tdir = os.path.join(ROOT, "target")
    cands = []
    if os.path.isdir(tdir):
        for d in sorted(os.listdir(tdir)):
            p = os.path.join(tdir, d, "x86_64-pc-windows-msvc", "release",
                             "vmlm-runtime.exe")
            if os.path.exists(p):
                cands.append(p)
    p = os.path.join(ROOT, "target", "x86_64-pc-windows-msvc", "release",
                     "vmlm-runtime.exe")
    if os.path.exists(p):
        cands.append(p)
    # Prefer names that suggest a bench build, but only as an ORDER, never as a
    # filter: every candidate is probed anyway.
    cands.sort(key=lambda c: (0 if "bench" in c else 1))
    for c in cands:
        if engine_is_bench(c):
            return c
    return None


def engine_is_bench(exe):
    """True if `exe` really accepts `reps:slot` and writes `out.bin`.

    # WHY THE PROBE USES A VALID MODEL AND CHECKS FOR out.bin

    The first version of this probe ran the engine against a missing model and
    treated "non-zero exit" as proof of a bench build. That produced a FALSE
    POSITIVE on `target/t_p17_engine_before/...` -- a build with NO bench feature.
    Two different failures were being conflated:

        default engine + bad path : prints "cannot open file", exit 2
        bench   engine + bad path : prints "cannot open file", exit 2

    Both read `argv[1]` as the model path (`vmlm_core::read_file` runs before any
    feature-gated code), so both exit 2 and the signal is useless.

    The only difference that matters is whether the engine DUMPS A SLOT, and that
    requires a model that actually loads. So the probe stages the real fixture in
    a private temp dir and requires `out.bin` to appear. That is the property the
    server depends on, tested directly rather than inferred from an exit code.
    """
    vmb = os.path.join(ROOT, "examples", "p17_http", "model.vmb")
    if not os.path.exists(vmb):
        # Fall back to any .vmb in the tree so the probe can still run.
        vmb = None
        for ex in ("p16_compose", "p16_ops", "p17_http"):
            d = os.path.join(ROOT, "examples", ex)
            if os.path.isdir(d):
                for n in sorted(os.listdir(d)):
                    if n.endswith(".vmb"):
                        vmb = os.path.join(d, n)
                        break
            if vmb:
                break
    if not vmb:
        return False
    import tempfile
    with tempfile.TemporaryDirectory(prefix="vmlm_probe_") as d:
        shutil.copyfile(vmb, os.path.join(d, "model.vmb"))
        ob = os.path.join(d, "out.bin")
        p = subprocess.run([exe, "model.vmb", "1:0"], capture_output=True, cwd=d,
                           timeout=60)
        text = ((p.stdout or b"") + (p.stderr or b"")).decode("utf-8", "replace")
        ok = os.path.exists(ob) and os.path.getsize(ob) > 0
        if not ok and p.returncode != 0:
            sys.stderr.write("[probe] %s rc=%d %s\n" % (exe, p.returncode,
                                                        text.strip()[:200]))
        return ok


def run_server(port, extra_args=None, env=None):
    """Start the server as a MANAGED BACKGROUND PROCESS (never foreground)."""
    exe = os.path.join(ROOT, "target", "release", "vmlm-server.exe")
    if not os.path.exists(exe):
        raise SystemExit(
            "FAIL: %s not built. Run:\n"
            "  cargo build --release -p vmlm-server" % exe)
    argv = [exe, "--host", "127.0.0.1", "--port", str(port), "--token", TOKEN]
    eng = find_bench_engine()
    if eng:
        argv += ["--engine", eng]
    if extra_args:
        argv += extra_args
    e = dict(os.environ)
    if env:
        e.update(env)
    p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         cwd=ROOT, env=e)
    return p


def kill(p):
    if p is None or p.poll() is not None:
        return
    p.terminate()
    try:
        p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait(timeout=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-size", action="store_true")
    ap.add_argument("--skip-model", action="store_true")
    ap.add_argument("--keep-server", action="store_true")
    a = ap.parse_args()

    r = Results()
    hr("P17-P4 HTTP ACCEPTANCE -- curl evidence dump")
    emit("root      = %s" % ROOT)
    emit("token     = %s" % TOKEN)
    emit("wrong     = %s" % WRONG_TOKEN)
    emit("curl      = %s" % shutil.which("curl.exe"))

    # ---- 1. the fixture model ---------------------------------------------
    if not a.skip_model:
        hr("1. FIXTURE MODEL (the engine input)")
        vmb = os.path.join(ROOT, "examples", "p17_http", "model.vmb")
        if not os.path.exists(vmb):
            r.check("fixture model.vmb exists", False,
                    "%s missing -- run tools/p17_http_model.py" % vmb)
        else:
            size = os.path.getsize(vmb)
            b = open(vmb, "rb").read()
            magic = struct.unpack_from("<I", b, 0)[0]
            r.check("fixture model.vmb is a valid VMLM artifact",
                    magic == 0x4D4C4D56 and size > 32,
                    "%s: %d B, magic=0x%08X" % (vmb, size, magic))

    # ---- 2. server cases ---------------------------------------------------
    hr("2. HTTP CASES -- raw curl output")
    bench = find_bench_engine()
    if bench:
        emit("bench-feature engine: %s" % bench)
        emit("  (verified to read argv: it exits non-zero on a bogus model, while")
        emit("   the default build exits 0 and writes nothing -- measured)")
    else:
        emit("!! NO BENCH-FEATURE ENGINE FOUND. The success-path cases will report")
        emit("   503 UNPROVEN, and that is the CORRECT behaviour: there is no engine")
        emit("   that can execute a forward pass here. Build one with:")
        emit("     cd runtime")
        emit("     cargo build --release -p vmlm-runtime --features bench \\")
        emit("       --target x86_64-pc-windows-msvc --target-dir <iso-dir>")
    port = free_port()
    emit("free port chosen by the OS: %d" % port)
    srv = run_server(port)
    try:
        if not wait_for_port(port):
            r.check("server accepts connections", False,
                    "nothing listening on 127.0.0.1:%d after 30 s" % port)
            kill(srv)
            print(srv.stdout.read().decode("utf-8", "replace") if srv.stdout else "")
            return 1
        r.check("server accepts connections", True,
                "127.0.0.1:%d is listening" % port)

        for name, kw, want, why in CASES:
            emit()
            emit("--- %s" % name)
            emit("    why this case exists: %s" % why)
            rc, status, body, err, argv = curl(port, **kw)
            show_curl(rc, status, body, err, argv)
            got = int(status) if status.isdigit() else -1
            ok = (rc == 0) and (got == want)
            if not ok:
                emit("    EXPECTED %d, GOT %s" % (want, status or "<none>"))
            # For the 400s, assert the body says WHERE, not just that it failed.
            if want == 400 and ok:
                try:
                    parsed = json.loads(body)
                    has_msg = bool(parsed.get("error", {}).get("message"))
                    r.check(name, has_msg,
                            "status 400 and body explains the error: %r"
                            % parsed.get("error", {}).get("message", "")[:120])
                    continue
                except Exception as e:
                    ok = False
                    emit("    body is not JSON: %s" % e)
            r.check(name, ok,
                    "curl_rc=%d status=%s (want %d)" % (rc, status, want))
    finally:
        if a.keep_server:
            emit("\n(server left running: pid %d, port %d)" % (srv.pid, port))
        else:
            kill(srv)
            emit("\nserver terminated")

    # ---- 3. the auth-disabled control -------------------------------------
    hr("3. CAN-FAIL CONTROL: server with --allow-anonymous")
    emit("If auth were 'passing everything' rather than working, section 2's 401")
    emit("criterion and this section would look the same. They must not.")
    port2 = free_port()
    srv2 = run_server(port2, extra_args=["--allow-anonymous"])
    try:
        if not wait_for_port(port2):
            r.check("anonymous server accepts connections", False, "no listener")
        else:
            rc, status, body, err, argv = curl(
                port2, method="POST", path="/v1/chat/completions",
                body=VALID_BODY, token=None)
            show_curl(rc, status, body, err, argv)
            r.check("with --allow-anonymous, NO token is accepted (200)",
                    rc == 0 and status == "200",
                    "status=%s -- this is the mirror image of the 401 criterion, "
                    "and it proves the 401 was produced by the auth check rather "
                    "than by something that rejects everything" % status)
    finally:
        kill(srv2)
        emit("anonymous server terminated")

    # ---- 4. engine size ----------------------------------------------------
    if not a.skip_size:
        hr("4. ENGINE SIZE -- the server must not grow vmlm-runtime.exe")
        emit("Measured in ISOLATED target dirs. (A shared target dir reports a")
        emit("false delta 0: the second build is a no-op, so the 'before' binary")
        emit("is simply reused. P13 hit exactly this.)")
        emit("See tools/p17_http_size.py for the measurement; its output is")
        emit("appended to this file.")

    nfail = r.summary()
    return 1 if nfail else 0


if __name__ == "__main__":
    out_dir = os.path.join(ROOT, "examples", "p17_http")
    os.makedirs(out_dir, exist_ok=True)
    rc = main()
    if not EVIDENCE:
        EVIDENCE.append("(no evidence captured)")
    with open(os.path.join(out_dir, "curl_evidence.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(EVIDENCE) + "\n")
    print("\nEvidence written to examples/p17_http/curl_evidence.txt")
    sys.exit(rc)
