#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P13 task C: verify the cached K against an INDEPENDENT numpy computation.

WHY
    Decode matches a full re-prefill on the last step of every configuration but
    is wrong on earlier steps, error decaying ~5.5x per step. Verified so far:
      * the reference forward IS causal (shared rows agree exactly, 0.0)
      * the cache append changes exactly ONE column per step and preserves the
        rest byte-for-byte
      * all four GEMM operand layouts are correct on paper
    So the remaining unknown is the CONTENT of the cached key: is column p the
    RoPE-rotated key at position p? This computes it from the safetensors
    weights and compares.

WHAT IS COMPUTED
    For layer 0, kv head 0, position p:
        x    = rmsnorm(embed[token_p])                     [H]
        k    = W_k @ x                                     [NKV*HD] -> head 0 block
        krot = rope_half(k, p, theta)                      [HD]
    with the same rotate_half convention OP_ROPE uses:
        out[i]      = k[i]*cos(a) - k[i+half]*sin(a)
        out[i+half] = k[i]*sin(a) + k[i+half]*cos(a)
        a = p * theta^(-2i/hd)
    and the engine's rmsnorm: x / sqrt(mean(x^2) + eps) * weight.

Usage: python tools/p13_kv_verify_content.py --seq 2
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = str(Path(__file__).resolve().parents[1])
MODEL = os.environ.get("MINICPM_DIR", "<外部模型目录>/MiniCPM5-1B")
REL = os.path.join(ROOT, "target", "release")
ASM = os.path.join(REL, "vmlm-asm.exe")
ENG = os.path.join(REL, "vlm-runtime.exe")
WORK = os.path.join(ROOT, "examples", "p13_kvdump")
IDS = [3, 117, 2501, 88, 42, 7, 900, 15]

sys.path.insert(0, os.path.join(ROOT, "tools"))


def bf16_roundtrip(a):
    u = a.astype(np.float32).view(np.uint32).astype(np.uint64)
    rnd = np.uint64(0x7FFF) + ((u >> 16) & 1)
    h = ((u + rnd) >> 16).astype(np.uint32)
    return (h << 16).astype(np.uint32).view(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=2)
    args = ap.parse_args()
    S = args.seq

    from safetensors_reader import SafeTensors
    st = SafeTensors(os.path.join(MODEL, "model-00000-of-00001.safetensors"))
    with open(os.path.join(MODEL, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    H, NKV, HD = cfg["hidden_size"], cfg["num_key_value_heads"], cfg["head_dim"]
    EPS, THETA = cfg["rms_norm_eps"], cfg["rope_theta"]

    embed = bf16_roundtrip(np.asarray(st.get("model.embed_tokens.weight"), dtype=np.float32))
    in_ln = np.asarray(st.get("model.layers.0.input_layernorm.weight"), dtype=np.float32)
    Wk = bf16_roundtrip(np.asarray(st.get("model.layers.0.self_attn.k_proj.weight"), dtype=np.float32))

    def rmsnorm(x, g):
        m = float(np.mean(x.astype(np.float64) ** 2))
        return (x / np.sqrt(m + EPS) * g).astype(np.float32)

    def rope_half(k, pos):
        half = HD // 2
        out = np.zeros_like(k)
        for i in range(half):
            inv = float(np.float32(THETA) ** np.float32(-2.0 * i / HD))
            period = 2.0 * np.pi / inv
            pr = pos if period >= 1e6 else pos - period * np.floor(pos / period)
            ang = np.float32(pr * inv)
            co, si = np.float32(np.cos(ang)), np.float32(np.sin(ang))
            x1, x2 = k[i], k[i + half]
            out[i] = np.float32(x1 * co) - np.float32(x2 * si) if False else np.float32(x1 * co - x2 * si)
            out[i + half] = np.float32(x1 * si + x2 * co)
        return out

    # Build the expected cache: [HD, S] with column p = rotated key at pos p.
    exp = np.zeros((HD, S), dtype=np.float32)
    for p in range(S):
        x = rmsnorm(embed[IDS[p]], in_ln)
        kfull = (Wk @ x).astype(np.float32)          # [NKV*HD]
        kh = kfull[:HD]                              # kv head 0 block
        exp[:, p] = rope_half(kh, p)

    print("=== P13 task C: cached K vs independent numpy RoPE ===")
    print("S=%d, layer 0, kv head 0; column p must be the rotated key at pos p\n" % S)

    # Read the engine's cache: rebuild the dump program for this S.
    p = subprocess.run(
        [sys.executable, os.path.join(ROOT, "tools", "p12_build_minicpm.py"),
         "--layers", "1", "--seq", str(S), "--decode", "1", "--skip-head",
         "--print-hidden", "--prompt-mode", "tokens", "--out", WORK],
        cwd=ROOT, capture_output=True, text=True, errors="replace")
    if p.returncode != 0:
        print((p.stdout + p.stderr)[-1000:])
        return 1
    t = open(os.path.join(WORK, "full.vml"), encoding="ascii").read()
    t = t.replace("; ================ P13 task C: incremental decode",
                  "PRINT st_kc0_0\n; ================ P13 task C: incremental decode", 1)
    open(os.path.join(WORK, "kv1.vml"), "w", encoding="ascii", newline="\n").write(t)
    consts = [l.strip() for l in open(os.path.join(WORK, "weights.args"),
                                      encoding="utf-8") if l.strip()]
    q = subprocess.run([ASM, "kv1.vml", "kv1.vmb"] + consts, cwd=WORK,
                       capture_output=True, text=True, errors="replace")
    if q.returncode != 0:
        print("assemble failed")
        print((q.stdout + q.stderr)[-800:])
        return 1

    import shutil
    shutil.copyfile(os.path.join(WORK, "kv1.vmb"), os.path.join(REL, "model.vmb"))
    with open(os.path.join(REL, "tokens.bin"), "wb") as f:
        np.array(IDS[:S], dtype="<u4").tofile(f)
    # decode builds declare st_gen, and a declared-but-missing input is a hard
    # failure by design (exit 2, "cannot open file"). Supply it.
    with open(os.path.join(REL, "gen_tokens.bin"), "wb") as f:
        np.array(IDS[S:S + 1], dtype="<u4").tofile(f)
    res = os.path.join(WORK, "_kv.txt")
    with open(res, "wb") as f:
        r = subprocess.run([os.path.join(REL, "vmlm-runtime.exe")], cwd=REL,
                           stdout=f, stderr=subprocess.DEVNULL)
    if r.returncode != 0:
        print("engine exit=%d" % r.returncode)
        return 1
    vals = []
    for line in open(res, "r", errors="replace"):
        for tok in line.split():
            try:
                vals.append(float(tok))
            except ValueError:
                pass
    # The cache slot is [HD, PMAX] where PMAX = S + decode, NOT [HD, S]. Using
    # S here misaligned the whole read and made correct engine output look like
    # a stride-3 corruption (the "every third element is zero" pattern was
    # really just a reshape across the wrong row width). PMAX is what the
    # builder declares.
    PMAX = S + 1  # this script builds with --decode 1
    pos = S * H
    cm = np.array(vals[pos:pos + HD * PMAX], dtype=np.float32).reshape(HD, PMAX)
    print("cache slot is [HD=%d, PMAX=%d]; comparing columns 0..%d (prefill)"
          % (HD, PMAX, S - 1))

    print("col | max|engine - numpy| | engine norm2 | numpy norm2")
    print("-" * 62)
    worst = 0.0
    for c in range(S):
        d = float(np.abs(cm[:, c] - exp[:, c]).max())
        worst = max(worst, d)
        print("%3d | %18.4e | %11.5f | %11.5f"
              % (c, d, float(np.linalg.norm(cm[:, c])), float(np.linalg.norm(exp[:, c]))))
    print()
    # The engine prints 6 decimals, so agreement is bounded by ~1e-6 relative
    # to the values involved (norms ~20-50 here).
    tol = 1e-4
    if worst <= tol:
        print("VERDICT: PASS -- the cached key matches an independent RoPE")
        return 0
    print("VERDICT: FAIL -- worst %.3e > %.1e: the cached key is NOT the" % (worst, tol))
    print("         rotated key at that position.")
    print("\nfirst 5 components, column 0:")
    print("  engine:", np.round(cm[:5, 0], 6).tolist())
    print("  numpy :", np.round(exp[:5, 0], 6).tolist())
    return 1


if __name__ == "__main__":
    sys.exit(main())
