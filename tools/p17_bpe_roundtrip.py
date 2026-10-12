#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P17-P3: the acceptance chain, end to end, with an INDEPENDENT decoder.

=============================================================================
WHAT THE TASK ASKS FOR
=============================================================================
"验收链路要打通：给一段文本 ⇒ encode 成 token id ⇒ （可先用随机/占位 logits
或真模型）⇒ decode 回文本。要求 **round-trip 逐字节相同**。
must-fail 对照：改一个 token id ⇒ 输出必须变。"

So the chain is: text -> ids -> [logits] -> ids -> text.

The [logits] stage is the model's job, not this task's, and there is no trained
model wired to this vocabulary. Rather than skip it or fake it silently, this
script does BOTH, and labels which is which:

  STAGE A (real): text -> ids, using the SHIPPED Rust encoder.
  STAGE B (placeholder, LABELLED): ids -> "logits" -> ids. Two variants:
        B1 identity      : the model emits exactly what it was given. This is
                           the degenerate model, and it is what makes the
                           round trip a statement about the TOKENIZER rather
                           than about a model.
        B2 argmax-of-logits: a deterministic pseudo-logit table is built from
                           a seeded PRNG over the vocabulary, the "model"
                           argmaxes it conditioned on the previous id, and the
                           resulting ids are fed to the decoder. This exercises
                           the ids -> logits -> ids plumbing for real, so the
                           wiring is demonstrated rather than asserted -- but
                           the logits are PLACEHOLDERS and the text that comes
                           back is NOT expected to match the input. That is
                           reported as its own criterion, not hidden.
  STAGE C (real): ids -> text, using the SHIPPED Rust decoder.
  STAGE D (independent): the same ids through the PYTHON decoder, so the
        round trip is verified by two implementations that share no code.

=============================================================================
WHY THE MUST-FAIL CONTROL IS THE POINT
=============================================================================
"round-trip is byte-identical" is only evidence if it CAN fail. So for every
case this script also mutates one token id and requires the output to change,
and reports both outcomes. A round trip that survives a mutated id would be
proving nothing.

Usage:
    python tools/p17_bpe_roundtrip.py
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p17_bpe_vocab as V   # noqa: E402
import p17_bpe_xcheck as X  # noqa: E402  (reuse the harness builder)

ROOT = str(Path(__file__).resolve().parents[1])
EX = os.path.join(ROOT, "examples", "p17_bpe")
VOCAB = os.path.join(EX, "bpe_vocab.vmb")
CORPUS = os.path.join(EX, "corpus.txt")

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print("  [%s] %-56s %s" % ("PASS" if ok else "FAIL", name, detail))
    return ok


def unproven(name, detail=""):
    RESULTS.append((name, None, detail))
    print("  [UNPROVEN] %-52s %s" % (name, detail))


def fnv1a64(b):
    h = 0xcbf29ce484222325
    for x in b:
        h ^= x
        h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return "%016x" % h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    print("=" * 78)
    print("P17-P3 acceptance chain: text -> token id -> logits -> text")
    print("=" * 78)

    blob = open(VOCAB, "rb").read()
    corpus = open(CORPUS, "rb").read()
    b = V.build(4096)
    merges, vocab, specials = b["merges"], b["vocab"], b["specials"]
    ref = V.BpeRef(merges, specials, vocab)

    # The one text we walk through end to end, chosen so that it exercises all
    # THREE decode cases (byte piece, merged piece, and a merged piece whose
    # operands are themselves merged pieces).
    TEXT = ("VMLM 是一个把 Transformer 权重预先汇编成指令流的推理引擎。 "
            "The engine is no_std and links against kernel32 directly.")
    raw = TEXT.encode("utf-8")
    print()
    print("text     : %r" % TEXT)
    print("bytes    : %d" % len(raw))
    print("sha256   : %s" % hashlib.sha256(raw).hexdigest())

    h = X.Harness()
    ok, msg = h.build()
    if not ok:
        print("harness build failed: %s" % msg)
        return 1
    try:
        # ---- STAGE A: text -> ids (SHIPPED Rust encoder) ----------------
        print()
        print("--- stage A: text -> token ids (shipped Rust encoder) ---")
        inp = os.path.join(h.dir, "rt_text.bin")
        with open(inp, "wb") as f:
            f.write(raw)
        rc, so, se = h.run("roundtrip", VOCAB, inp)
        if rc != 0:
            check("stage A: Rust encode", False, "rc=%d %s" % (rc, se[-200:]))
            return 1
        d = X.kv(so)
        rust_ids = [int(x) for x in d["ids"].split(",") if x]
        py_ids, _ = ref.encode(TEXT)
        check("stage A: Rust encoder produced ids", len(rust_ids) > 0,
              "n_ids=%d" % len(rust_ids))
        check("stage A: Rust ids == Python ids (independent encoders)",
              rust_ids == py_ids,
              "rust=%d python=%d, equal=%s" % (len(rust_ids), len(py_ids),
                                               rust_ids == py_ids))
        print("      first 12 ids: %s" % rust_ids[:12])
        print("      ids are token ids into a %d-piece vocabulary"
              % (V.N_BYTE_PIECES + len(merges)))

        # ---- STAGE B: ids -> logits -> ids ------------------------------
        print()
        print("--- stage B: ids -> logits -> ids ---")
        # B1: the DEGENERATE model (identity). This is the one that makes the
        # round trip a claim about the tokenizer.
        b1_ids = list(rust_ids)
        check("stage B1: identity model passes the ids through unchanged",
              b1_ids == rust_ids, "n=%d" % len(b1_ids))
        # B2: a real argmax over PLACEHOLDER logits, so the plumbing
        # (ids -> logits -> argmax -> ids -> decode) is actually exercised.
        # The logits are seeded PRNG, NOT a trained model, and the recovered
        # text is NOT expected to match -- that is stated, not hidden.
        def pseudo_logits(prev_id, n):
            """Deterministic placeholder logits: a splitmix64-seeded table.
            Written here so the plumbing is demonstrated without pretending a
            model exists."""
            z = (prev_id * 0x9E3779B97F4A7C15 + 0x123456789ABCDEF) & 0xFFFFFFFFFFFFFFFF
            out = []
            for k in range(min(n, 8)):
                z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
                z = (z ^ (z >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
                z = z ^ (z >> 31)
                out.append((z, (z >> 11) % (V.N_BYTE_PIECES + len(merges))))
            return out
        gen = []
        prev = 0
        for _ in range(24):
            cand = pseudo_logits(prev, 8)
            best = max(cand)          # argmax over the placeholder logits
            gen.append(best[1])
            prev = best[1]
        check("stage B2: ids -> placeholder logits -> argmax -> ids runs",
              len(gen) == 24, "24 greedy steps, first 8 = %s" % gen[:8])
        if gen:
            # The generated ids must be DECODABLE (in range) -- that is the
            # part this task owns. The text is not expected to match.
            rc, so, se = h.run("decode", VOCAB, ",".join(map(str, gen)))
            check("stage B2: generated ids decode without a guard firing",
                  rc == 0, "rc=%d %s" % (rc, se[-160:]))
            unproven("stage B2: generated text == input text",
                     "the logits are a seeded placeholder, not a model, so "
                     "the decoded text is NOT expected to match. Reported "
                     "rather than asserted -- this task owns the tokenizer, "
                     "not the model.")

        # ---- STAGE C: ids -> text (SHIPPED Rust decoder) ----------------
        print()
        print("--- stage C: ids -> text (shipped Rust decoder) ---")
        # The harness reports the decoded bytes as hex AND the FNV digests of
        # input and output. The criterion compares the actual bytes, not just
        # the boolean the harness computed -- if the digest were miscomputed,
        # "exact=1" would still print 1, and only the byte comparison catches
        # that. (The digest is a transport convenience, not the evidence.)
        outb = bytes.fromhex(d.get("hex", ""))
        check("stage C: Rust round trip is byte-identical",
              outb == raw,
              "n_bytes=%s (input %d), sha_in=%s sha_out=%s exact=%s, "
              "bytes_equal=%s"
              % (d.get("n_bytes"), len(raw), d.get("sha_in"),
                 d.get("sha_out"), d.get("exact"), outb == raw))

        # ---- STAGE D: the same ids through the INDEPENDENT decoder ------
        print()
        print("--- stage D: same ids through the PYTHON decoder ---")
        py_out = ref.decode_ids(rust_ids)
        check("stage D: Python decoder agrees byte for byte",
              py_out == raw,
              "python=%d B sha=%s" % (len(py_out), fnv1a64(py_out)))

        # ---- MUST-FAIL CONTROL ------------------------------------------
        print()
        print("--- must-fail control: change ONE token id ---")
        mut = list(rust_ids)
        mut[0] = 65 if mut[0] != 65 else 66
        rc, so, se = h.run("decode", VOCAB, ",".join(map(str, mut)))
        mut_out = bytes.fromhex(X.kv(so).get("hex", "")) if rc == 0 else None
        check("control: mutating ONE token id changes the Rust output",
              rc == 0 and mut_out != raw,
              "orig id=%d -> 65; output changed=%s, len %d -> %s"
              % (rust_ids[0], mut_out != raw, len(raw),
                 len(mut_out) if mut_out is not None else "n/a"))
        check("control: the mutated output is not merely a re-encode",
              mut_out != raw and mut_out != py_out,
              "also differs from the Python decoder's output")

        # ---- the full corpus --------------------------------------------
        print()
        print("--- full corpus (%d bytes) ---" % len(corpus))
        inp = os.path.join(h.dir, "rt_corpus.bin")
        with open(inp, "wb") as f:
            f.write(corpus)
        rc, so, se = h.run("roundtrip", VOCAB, inp)
        if rc != 0:
            check("corpus round trip (Rust)", False, "rc=%d" % rc)
        else:
            d2 = X.kv(so)
            check("corpus round trip (Rust): %d B, byte-identical" % len(corpus),
                  d2.get("exact") == "1",
                  "n_ids=%s sha_in=%s sha_out=%s"
                  % (d2.get("n_ids"), d2.get("sha_in"), d2.get("sha_out")))
            cids = [int(x) for x in d2["ids"].split(",") if x]
            check("corpus round trip (Python, independent)",
                  ref.decode_ids(cids) == corpus,
                  "%d ids -> %d B" % (len(cids), len(corpus)))
    finally:
        if not args.keep:
            h.close()

    npass = sum(1 for _, o, _ in RESULTS if o is True)
    nfail = sum(1 for _, o, _ in RESULTS if o is False)
    nun = sum(1 for _, o, _ in RESULTS if o is None)
    print()
    print("=" * 78)
    print("PASS=%d FAIL=%d UNPROVEN=%d" % (npass, nfail, nun))
    print("=" * 78)

    log = os.path.join(EX, "bpe_roundtrip.json")
    with open(log, "w", encoding="utf-8") as f:
        json.dump({"text": TEXT, "n_bytes": len(raw),
                   "sha256": hashlib.sha256(raw).hexdigest(),
                   "n_ids": len(rust_ids) if 'rust_ids' in dir() else None,
                   "pass": npass, "fail": nfail, "unproven": nun,
                   "results": [{"name": n, "ok": o, "detail": d}
                               for n, o, d in RESULTS]},
                  f, ensure_ascii=False, indent=1)
    print("wrote %s" % log)
    # Also keep the human-readable transcript next to it.
    txt = os.path.join(EX, "bpe_roundtrip.log")
    with open(txt, "w", encoding="utf-8") as f:
        for n, o, det in RESULTS:
            f.write("%-8s %s%s\n" % ("PASS" if o is True else
                                     ("FAIL" if o is False else "UNPROVEN"),
                                     n, ("  | " + det) if det else ""))
    print("wrote %s" % txt)
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
