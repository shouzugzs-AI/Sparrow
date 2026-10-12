#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P10 B2: bisect the FIRST divergent stage inside the MiniCPM layer-0 attention.

Context: tools/minicpm_layer0_run.py shows the engine matches HF to ~1e-6
for xn/q/k/v, then breaks at att_out (rel 0.46). The numpy path
(minicpm_layer0_check_ref.py) matches HF everywhere, so this is an ENGINE
gap inside the per-head attention block.

The block is: SLICE q/k/v head -> RoPE (4 EWMUL + 2 EWSUB/EWADD + CONCAT)
-> GEMM scores -> SCALE -> SOFTMAX causal -> GEMM ctx -> WRITECOL.

Rather than guess, generate a *minimal* .vml that dumps each of those
sub-stages for ONE head and diff against values computed in numpy from the
engine's own q/k/v (so any upstream error is factored out -- this is the
"differential oracle" idea from white paper 15.6-2).
"""

import os
import subprocess
import sys

import numpy as np

ROOT = str(Path(__file__).resolve().parents[1])
REF = os.path.join(ROOT, "examples", "minicpm", "layer0_ref")
OUT = os.path.join(ROOT, "examples", "minicpm")
ASM = os.path.join(ROOT, "target", "release", "vmlm-asm.exe")
ENG = os.path.join(ROOT, "target", "release", "vmlm-runtime.exe")
REL = os.path.join(ROOT, "target", "release")


def L(n):
    return np.load(os.path.join(REF, n + ".npy")).astype(np.float32)


def wr(name, arr):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    arr.tofile(os.path.join(OUT, name + ".bin"))
    return name, arr.shape


def main():
    meta = __import__("json").load(open(os.path.join(REF, "meta.json"), encoding="utf-8"))
    cfg = meta["config"]
    H, S = cfg["hidden_size"], meta["seq"]
    nh, nkv, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    h = 0                      # head under test
    j = h // (nh // nkv)       # its kv head
    print("bisecting head %d (kv head %d), hd=%d S=%d" % (h, j, hd, S))

    # q/k/v as the ENGINE computed them (== HF to 1e-6, so upstream is fine)
    q = L("q_proj_out").reshape(S, nh, hd)[:, h, :]    # [S,hd]
    k = L("k_proj_out").reshape(S, nkv, hd)[:, j, :]
    v = L("v_proj_out").reshape(S, nkv, hd)[:, j, :]
    cos, sin = L("rope_cos"), L("rope_sin")

    consts = [
        wr("qh_in", q),
        wr("kh_in", k),
        wr("vh_in", v),
        wr("cos", cos),
        wr("sin", sin),
        wr("inv_sqrt_hd", np.array([1.0 / np.sqrt(hd)], dtype=np.float32)),
    ]

    half = hd // 2
    lines = [
        "; minimal one-head attention dump (P10 bisect)",
        "SHAPE qh_in, %d, %d" % (S, hd),
        "SHAPE kh_in, %d, %d" % (S, hd),
        "SHAPE vh_in, %d, %d" % (S, hd),
        "SHAPE cos,   %d, %d" % (S, hd),
        "SHAPE sin,   %d, %d" % (S, hd),
        "SHAPE c1,    %d, %d" % (S, half),
        "SHAPE c2,    %d, %d" % (S, half),
        "SHAPE s1,    %d, %d" % (S, half),
        "SHAPE s2,    %d, %d" % (S, half),
        "SHAPE q1,    %d, %d" % (S, half),
        "SHAPE q2,    %d, %d" % (S, half),
        "SHAPE o1,    %d, %d" % (S, half),
        "SHAPE o2,    %d, %d" % (S, half),
        "SHAPE tmp,   %d, %d" % (S, half),
        "SHAPE qrot,  %d, %d" % (S, hd),
        "SHAPE krot,  %d, %d" % (S, hd),
        "SHAPE sc,    %d, %d" % (S, S),
        "SHAPE ctx,   %d, %d" % (S, hd),
        "SHAPE inv_sqrt_hd, 1, 1",
        "",
        "SLICE c1, cos, 0, %d" % half,
        "SLICE c2, cos, %d, %d" % (half, half),
        "SLICE s1, sin, 0, %d" % half,
        "SLICE s2, sin, %d, %d" % (half, half),
        "",
        "; ---- q rope ----",
        "SLICE q1, qh_in, 0, %d" % half,
        "SLICE q2, qh_in, %d, %d" % (half, half),
        "EWMUL o1, q1, c1",
        "EWMUL tmp, q2, s1",
        "EWSUB o1, o1, tmp",
        "EWMUL o2, q1, s1",
        "EWMUL tmp, q2, c2",
        "EWADD o2, o2, tmp",
        "CONCAT qrot, o1, o2",
        "; ---- k rope ----",
        "SLICE q1, kh_in, 0, %d" % half,
        "SLICE q2, kh_in, %d, %d" % (half, half),
        "EWMUL o1, q1, c1",
        "EWMUL tmp, q2, s1",
        "EWSUB o1, o1, tmp",
        "EWMUL o2, q1, s1",
        "EWMUL tmp, q2, c2",
        "EWADD o2, o2, tmp",
        "CONCAT krot, o1, o2",
        "; ---- scores / softmax / ctx ----",
        "; scores[S,S] = qrot[S,hd] @ krot[S,hd]^T  -> NT (the DEFAULT):",
        ";   NT treats B as [N,K], so N = rows(krot) = S. Correct here.",
        "; (Using `nn` here treats B as [K,N] => N = cols(krot) = hd = 128,",
        ";  which overruns the [S,S] destination -> ACCESS VIOLATION.)",
        "GEMM sc, qrot, krot",
        "SCALE sc, sc, inv_sqrt_hd",
        "SOFTMAX sc, sc, causal",
        "; ctx[S,hd] = att[S,S] @ vh_in[S,hd] -> B is [K,N] here, so NN.",
        "GEMM ctx, sc, vh_in, nn",
        "",
        "PRINT q1",
        "PRINT q2",
        "PRINT o1",
        "PRINT o2",
        "PRINT qrot",
        "PRINT krot",
        "PRINT sc",
        "PRINT ctx",
        "HALT",
    ]
    p = os.path.join(OUT, "att_bisect.vml")
    with open(p, "w", encoding="ascii", newline="\n") as f:
        f.write("\n".join(lines) + "\n")

    args = [ASM, "att_bisect.vml", "att_bisect.vmb"]
    for n, sh in consts:
        args.append("%s:%s.bin:%d:%d" % (n, n, sh[0], sh[1]) if len(sh) == 2
                    else "%s:%s.bin:1:%d" % (n, n, sh[0]))
    r = subprocess.run(args, cwd=OUT, capture_output=True, text=True,
                       errors="replace", encoding="utf-8")
    if r.returncode != 0:
        print("ASM FAILED:", r.stdout[-800:], r.stderr[-800:])
        return 1
    print("assembled")

    import shutil
    shutil.copyfile(os.path.join(OUT, "att_bisect.vmb"), os.path.join(REL, "model.vmb"))
    e = subprocess.run([ENG], cwd=REL, capture_output=True, text=True,
                       errors="replace", encoding="utf-8")
    if e.returncode != 0:
        print("ENGINE exit=%d out=%s" % (e.returncode, e.stdout[:400]))
        return 1
    vals = []
    for ln in e.stdout.splitlines():
        for t in ln.split():
            try:
                vals.append(float(t))
            except ValueError:
                pass
    got = np.array(vals, dtype=np.float32)

    # ---- numpy expectations, computed from the SAME q/k/v the engine saw ----
    def rope_ref(t):
        t1, t2 = t[:, :half], t[:, half:]
        c1, c2 = cos[:, :half], cos[:, half:]
        s1, s2 = sin[:, :half], sin[:, half:]
        return np.concatenate([t1 * c1 - t2 * s1, t1 * s1 + t2 * c2], axis=-1)

    q1e, q2e = q[:, :half], q[:, half:]
    c1e, c2e = cos[:, :half], cos[:, half:]
    s1e, s2e = sin[:, :half], sin[:, half:]
    o1e = q1e * c1e - q2e * s1e
    o2e = q1e * s1e + q2e * c2e
    qrot_e = rope_ref(q)
    krot_e = rope_ref(k)
    sc_e = (qrot_e @ krot_e.T) / np.float32(np.sqrt(hd))
    mask = np.triu(np.ones((S, S), dtype=np.float32), 1) * np.float32(-1e30)
    sc_e = sc_e + mask
    mx = sc_e.max(axis=-1, keepdims=True)
    ex = np.exp((sc_e - mx).astype(np.float64)).astype(np.float32)
    att_e = (ex / ex.sum(axis=-1, keepdims=True)).astype(np.float32)
    # SOFTMAX in-engine writes the probabilities back into sc
    ctx_e = (att_e @ v).astype(np.float32)

    order = [
        ("q1", q1e), ("q2", q2e), ("o1", o1e), ("o2", o2e),
        ("qrot", qrot_e), ("krot", krot_e),
        ("sc (after softmax)", att_e),   # engine's sc is softmax output
        ("ctx", ctx_e),
    ]
    print()
    print("%-20s %12s %12s %10s  %s" % ("stage", "absmax", "refmax", "rel", "verdict"))
    off = 0
    first_bad = None
    for name, ref in order:
        n = ref.size
        g = got[off:off + n]
        off += n
        if g.size != n:
            print("%-20s SIZE MISMATCH got %d want %d" % (name, g.size, n))
            return 1
        d = np.abs(g - ref.reshape(-1))
        rm = max(float(np.abs(ref).max()), 1e-30)
        rel = float(d.max()) / rm
        ok = rel < 1e-3
        if not ok and first_bad is None:
            first_bad = name
        print("%-20s %12.4e %12.4e %10.3e  %s"
              % (name, d.max(), rm, rel, "OK" if ok else "DIVERGES <--"))

    print()
    if first_bad is None:
        print("no divergence found in this isolated block")
    else:
        print("FIRST DIVERGENT SUB-STAGE: %s" % first_bad)
    return 0


if __name__ == "__main__":
    sys.exit(main())
