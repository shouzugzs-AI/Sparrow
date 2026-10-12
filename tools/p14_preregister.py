#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P14 pre-registration: freeze the O6/S5 acceptance predicates BEFORE the
implementations land, so they cannot be retrofitted to whatever got built.

WHY PRE-REGISTRATION, AND WHY A HASH
    The failure mode this guards against is subtle and it is NOT dishonesty: an
    implementer who has already built something will, quite reasonably, describe
    the acceptance criterion in terms of what they built. By then the criterion
    has stopped being a test and become a description. The way to keep it a test
    is to write it down while the implementation does not exist yet, and to make
    the record tamper-evident.

    There is no git in this environment (measured: `git` is not on PATH), so the
    record is a JSON file whose own sha256 is printed, plus the sha256 of every
    file it names. Changing a predicate afterwards changes the hash.

WHAT "FROZEN" MEANS HERE
    The predicates below are stated in terms of INPUT -> EXPECTED OUTPUT and the
    PERTURBATION that must make them fail. They deliberately do NOT name any
    symbol, function, slot, or file that task-2/task-3 might create, because
    doing so would let the implementation define the test.

    Each predicate carries:
      id           stable name
      claim        the acceptance criterion, as the task book states it
      test         exactly what I will run
      must_reject  the KNOWN-BAD input the test must reject (can-it-fail proof)
"""
import hashlib
import json
import os
import sys
import time

ROOT = str(Path(__file__).resolve().parents[1])
OUT = os.path.join(ROOT, "docs", "P14_preregistration.json")

PREDICATES = [
    # ---------------- O6 (task-2), needs task-1 ----------------
    {
        "id": "O6-P1",
        "area": "O6",
        "claim": "Generate 16 tokens; the sequence is bit-identical to a "
                 "one-shot forward over the same 16 positions.",
        "test": "Run the O6 generation artifact with temperature=0 and read the "
                "emitted token chain C_loop. Independently obtain the argmax "
                "chain C_one_shot from a one-shot forward over the same "
                "positions. Require C_loop == C_one_shot element-wise, exactly "
                "(integer equality, no tolerance).",
        "must_reject": "A chain perturbed by off-by-one, reversal, a one-step "
                       "shift, truncation, or extension must be rejected. "
                       "Demonstrated by tools/p14_o6_harness.py before use.",
    },
    {
        "id": "O6-P2",
        "area": "O6",
        "claim": "The generation loop is not a no-op: it really feeds its own "
                 "argmax back.",
        "test": "Two routes must agree: (a) generate with the O6 loop "
                "(argmax fed back in-engine); (b) give the engine the SAME "
                "chain as pre-given ids via gen_tokens.bin and run the P13 "
                "decode path with no feedback. Require the two chains equal. "
                "Plus: seed gen_tokens.bin with a DELIBERATELY WRONG chain and "
                "require the O6 output to follow its own argmax, not the file.",
        "must_reject": "If gen_tokens.bin contents determined the output, the "
                       "wrong-chain run would reproduce the wrong chain. That "
                       "outcome is a FAIL and is exactly the fake being hunted.",
    },
    {
        "id": "O6-P3",
        "area": "O6",
        "claim": "External anchor: the first generated token is fixed by an "
                 "oracle outside our implementation.",
        "test": "On the model's own prompt ids [3,117,2501,88], require the "
                "first generated token == 5, which is the HuggingFace argmax "
                "re-derived here from examples/minicpm_full/hf_logits.npy.",
        "must_reject": "Any other first token. The anchor is re-read from disk "
                       "at test time, not hard-coded, so a changed .npy is "
                       "detected rather than hidden.",
    },
    {
        "id": "O6-P4",
        "area": "O6",
        "claim": "The loop cannot be an echo of its input file.",
        "test": "Remove gen_tokens.bin and run the O6 artifact. Require a "
                "NON-ZERO exit. (Measured on the existing engine: a declared "
                "st_gen with no file exits 2 'cannot open file'.)",
        "must_reject": "exit==0 with a full token chain would mean the artifact "
                       "never needed the file -- i.e. it is not the loop under "
                       "test, or the guard was weakened.",
    },
    {
        "id": "O6-P5",
        "area": "O6",
        "claim": "temperature=0 degrades EXACTLY to argmax.",
        "test": "Run temperature=0 and temperature=0 expressed by an equivalent "
                "control; require the chains equal exactly. Then require "
                "temperature>0 to produce a DIFFERENT chain at least once "
                "(otherwise the temperature parameter is inert).",
        "must_reject": "temperature>0 giving an identical chain to temperature=0 "
                       "means temperature is not wired in. That is a FAIL, not a "
                       "pass with a note.",
    },
    {
        "id": "O6-P6",
        "area": "O6",
        "claim": "Determinism: the same inputs give byte-identical output.",
        "test": "Run the same artifact twice with identical inputs; require the "
                "raw stdout bytes and the token chain to be identical.",
        "must_reject": "Any byte difference. Also: an EMPTY output is not a "
                       "pass -- a zero-length comparison is reported as a failed "
                       "measurement.",
    },
    {
        "id": "O6-P7",
        "area": "O6",
        "claim": "Position correctness: each step rotates at its own absolute "
                 "position.",
        "test": "Perturb the RoPE start position / the live range of one step "
                "and require the chain to change; and require a build whose "
                "steps all use ONE position to be rejected by O6-P1.",
        "must_reject": "A chain that does not change when a position changes "
                       "means positions are not reaching the rotation.",
    },
    # ---------------- S5 (task-3) ----------------
    {
        "id": "S5-P1",
        "area": "S5",
        "claim": "The KV-Cache memory layout is [layer][head][dim][slot].",
        "test": "Derive the ACTUAL address of every (layer, head, dim, slot) "
                "from the shipped offsets/stride and compare against the bytes "
                "physically present in the artifact / the engine's own output. "
                "A comment, a docstring, or a struct name is NOT evidence; the "
                "check is on addresses and values.",
        "must_reject": "[dim] and [slot] transposed on disk, read per the spec, "
                       "must be detected. Demonstrated in "
                       "tools/p14_s5_layout_lib.py: 108/120 elements wrong.",
    },
    {
        "id": "S5-P2",
        "area": "S5",
        "claim": "Reads are contiguous loads with no gather.",
        "test": "MEASURE the address sequence for (a) one neuron's whole slot "
                "range and (b) one slot across a head's dims, and report the gap "
                "pattern for both. Contiguity means gaps are all 1.",
        "must_reject": "A stride != 1 gap pattern must be reported as NOT "
                       "contiguous. Measured gap=[5,5,5,5] for a gather is "
                       "rejected by the same code path.",
        "caveat": "PRE-MEASURED BEFORE THE IMPLEMENTATION LANDED: at NEURON "
                  "granularity gaps are all 1 (contiguous). At HEAD granularity, "
                  "one slot across dims has gaps = stride (NOT contiguous). The "
                  "criterion as literally written is therefore only true at "
                  "neuron granularity, and the report will say which. If task-3 "
                  "claims flat contiguity it is overstated.",
    },
    {
        "id": "S5-P3",
        "area": "S5",
        "claim": "The neuron-organized version is bit-identical to the indexed "
                 "(gather) version.",
        "test": "Build the same logical cache both ways -- neuron-addressed and "
                "plain indexed -- and require the resulting byte strings to be "
                "EQUAL. Not a tolerance, and not a same-code-path comparison.",
        "must_reject": "A wrong stride (23/120 elements wrong) and a "
                       "transposed store (108/120 wrong) must both be detected. "
                       "Demonstrated in tools/p14_s5_layout_lib.py self-test.",
    },
    {
        "id": "S5-P4",
        "area": "S5",
        "claim": "Illegal input fails loudly rather than producing plausible "
                 "garbage.",
        "test": "Feed an offset out of range, a stride too small for the slot "
                "count, and a count mismatch; require each to be REJECTED "
                "(non-zero exit / error), never silently accepted.",
        "must_reject": "Any of the three being accepted is the "
                       "silent-degradation class this project converts to hard "
                       "failures, and is reported as a FAIL.",
    },
]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    rec = {
        "what": "P14 O6/S5 acceptance predicates, pre-registered BEFORE the "
                "implementations land",
        "why": "so an acceptance criterion cannot be retrofitted to describe "
               "whatever was built",
        "author": "verifier (task-4), who did not write the implementations",
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note_no_git": "git is not installed in this environment (measured: "
                       "'git' is not recognized); integrity is by sha256.",
        "predicates": PREDICATES,
        "instrument_hashes": {},
    }
    for name in ["tools/p14_verify_lib.py", "tools/p14_s5_layout_lib.py",
                 "tools/p14_o6_harness.py"]:
        p = os.path.join(ROOT, name.replace("/", os.sep))
        if os.path.exists(p):
            rec["instrument_hashes"][name] = sha256_file(p)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)
        f.write("\n")

    blob = open(OUT, "rb").read()
    print("wrote %s (%d bytes)" % (OUT, len(blob)))
    print("sha256(record) = %s" % hashlib.sha256(blob).hexdigest())
    print("predicates frozen: %d" % len(PREDICATES))
    for p in PREDICATES:
        print("  %-8s %s" % (p["id"], p["claim"][:66]))
    print()
    print("instrument hashes:")
    for k, v in rec["instrument_hashes"].items():
        print("  %-34s %s" % (k, v[:16]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
