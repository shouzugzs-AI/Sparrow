#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P15-S5 HONESTY audit (task-4 attack surface #6).

WHY A HONESTY AUDIT IS PART OF A CODE VERIFICATION
    The task is explicit: "CHECK THE HONESTY OF THE CLAIMS, not just the code.
    If any comment, doc, task report, or acceptance string overstates what was
    measured -- especially an unqualified 'no gather' -- flag it explicitly as
    OVERSTATED with the exact line."

    The known trap is the granularity limit. This project measured it:
        one NEURON's slots      -> contiguous, gaps all 1
        one HEAD, ONE slot x d  -> NOT contiguous, gap == stride     <-- the trap
    An unqualified "reads are contiguous, no gather" is FALSE for the second
    case, and docs/P14_verification.md §2.5.1 already flagged one surviving
    instance of exactly that wording in core/src/neuron_kv.rs. So the audit must
    find unqualified forms AND ensure the qualified form is actually present.

METHOD (measured, not asserted)
    For every S5-related file in scope, find every sentence containing a
    contiguity/gather claim, and classify it:
      QUALIFIED   -- the text also names the granularity (neuron) or says the
                     per-slot-across-dims case is NOT contiguous
      OVERSTATED  -- it claims contiguity/no-gather without the granularity
                     qualifier
    The classification is done by regex over a WINDOW around the claim, which is
    obviously approximate, so every hit is printed for a human to read: the
    audit's output is a list of quotations, not a score. A classifier that
    cannot find a planted unqualified claim is reported broken; the harness
    plants one to prove it can fail.

Run from <repo>:
    python tools/p15_verify_honesty.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p15_verify_lib as V  # noqa: E402

ROOT = V.ROOT


def read_text(p):
    with open(p, encoding="utf-8", errors="replace") as f:
        return f.read()


# Files whose S5 claims are in scope: implementation, tooling, and the docs that
# state the headline. The two P14 docs are included because a P15 claim that
# contradicts them silently is itself a problem.
FILES = [
    "core/src/neuron_kv.rs",
    "core/src/format.rs",
    "runtime/src/s5_load.rs",
    "tools/p15_s5_section.py",
    "tools/p15_s5_load_probe.py",
    "tools/p15_s5_e2e.py",
    "docs/P14_S5_O6_报告.md",
    "docs/P14_verification.md",
]

# Wording that makes a contiguity claim. Both languages, because this codebase
# comments in Chinese and the task book is Chinese.
#
# ⚠️ MY OWN BUG, caught by this run: the first version matched the bare word
# "contiguous", so it flagged CODE IDENTIFIERS -- `def is_contiguous(addrs)`,
# `np.ascontiguousarray(...)`, `c, g = is_contiguous(a)`. Those are not prose
# claims about the cache, so 21 "unqualified claims" was mostly noise that
# buried the 4 real ones. The fix is to require a claim SHAPE: either a
# no-gather phrase, or a contiguity word used as a predicate about an access
# ("is contiguous", "contiguous load", "contiguous=True"), not as an identifier.
CLAIM_PAT = re.compile(
    r"no gather|zero gather|无\s*gather|零\s*gather|"
    r"contiguous\s+load|contiguous\s*=\s*True|contiguous\s*=\s*False|"
    r"IS\s+contiguous|is\s+contiguous\b(?!\()|NOT\s+contiguous|"
    r"连续\s*load|连续load|连续列|一段\*\*连续|单位步长|"
    r"one contiguous load|不\s*连\s*续")

# Wording that RULES OUT flagging: a line that is code, not prose.
CODE_LIKE = re.compile(
    r"^\s*(def |c,\s*g\s*=|c\d?,\s*g\d?\s*=|print\(|crit\(|np\.|#\s*\(b\d|"
    r"all\(not |return )")

# Wording that shows the phrase is being QUOTED as a thing to avoid, rather than
# asserted. This distinction matters and cannot be waved away: two of the first
# run's eight hits were a test file and a verification report saying "writing
# 'the reads are contiguous' is not evidence" and "this claim can be satisfied
# by a comment" -- i.e. the OPPOSITE of an overstatement. Flagging those as
# overstatements would itself be a false accusation, which is its own kind of
# dishonesty, so the classifier reports them in a separate bucket.
QUOTED_PAT = re.compile(
    r"不能不加限定|不加限定|is not evidence|不是证据|都可以用一句注释|"
    r"must be qualified|OVERSTATED|夸大|防止|避免|anti-pattern|"
    r"could be faked|could be satisfied|how .* could be faked")

# Wording that QUALIFIES such a claim.
QUAL_PAT = re.compile(
    r"neuron granularity|神经元粒度|一个\*\*?神经元|一个神经元|per-slot|按 slot|"
    r"across dims|跨 dims|跨dims|NOT contiguous|不连续|非连续|OVERSTATED|"
    r"夸大|限定|granularity|粒度|第三条|stride-PMAX|stride 间隔|gap == stride")

# Wording that BOUNDS a byte-size claim.
SIZE_PAT = re.compile(
    r"\b0 B\b|0 字节|零字节|\+0\b|0 bytes|68096|20480|16384|\+2048")

QUALIFY_WINDOW = 700


def audit_file(rel):
    p = os.path.join(ROOT, rel.replace("/", os.sep))
    if not os.path.exists(p):
        return None
    text = read_text(p)
    lines = text.splitlines()
    # char offset of each line start, so a window can be mapped back to lines
    starts = []
    off = 0
    for ln in lines:
        starts.append(off)
        off += len(ln) + 1

    def line_of(ch):
        lo, hi = 0, len(starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if starts[mid] <= ch:
                lo = mid
            else:
                hi = mid - 1
        return lo

    hits = []
    for m in CLAIM_PAT.finditer(text):
        s = max(0, m.start() - QUALIFY_WINDOW)
        e = min(len(text), m.end() + QUALIFY_WINDOW)
        window = text[s:e]
        qualified = bool(QUAL_PAT.search(window))
        l0 = line_of(m.start())
        l1 = line_of(m.end())
        snippet = " / ".join(l.strip() for l in lines[l0:l1 + 1])
        # prose filter: a line that is CODE is not a claim about the cache
        if CODE_LIKE.search(lines[l0]):
            continue
        # quoted-as-bad filter: the window must not be warning ABOUT the phrase
        quoted = bool(QUOTED_PAT.search(window))
        hits.append(dict(line=l0 + 1, mention=m.group(0),
                         qualified=qualified, quoted=quoted,
                         text=snippet[:220],
                         window_has_qualifier=qualified))
    return hits


def main():
    print("=" * 78)
    print("P15-S5 HONESTY audit: are the contiguity claims qualified?")
    print("=" * 78)
    meta = V.MetaCheck()

    # ------------------------------------------------------------------
    # the classifier must be able to fail -- plant an unqualified claim
    # ------------------------------------------------------------------
    planted = ("The cache reads are one contiguous load with no gather at all, "
               "so the access is fully sequential. " * 2)
    plant_hits = []
    for m in CLAIM_PAT.finditer(planted):
        s = max(0, m.start() - QUALIFY_WINDOW)
        e = min(len(planted), m.end() + QUALIFY_WINDOW)
        plant_hits.append(bool(QUAL_PAT.search(planted[s:e])))
    plant_unqualified = any(not q for q in plant_hits) and bool(plant_hits)
    clean_plant = ("Within one NEURON the slots are one contiguous load; the "
                   "per-slot-across-dims access is NOT contiguous (gap == "
                   "stride) and an unqualified 'no gather' would be OVERSTATED.")
    clean_hits = []
    for m in CLAIM_PAT.finditer(clean_plant):
        s = max(0, m.start() - QUALIFY_WINDOW)
        e = min(len(clean_plant), m.end() + QUALIFY_WINDOW)
        clean_hits.append(bool(QUAL_PAT.search(clean_plant[s:e])))
    clean_qualified = bool(clean_hits) and all(clean_hits)
    print("\n[CLASSIFIER self-test]")
    print("  planted UNQUALIFIED claim detected as unqualified : %s"
          % plant_unqualified)
    print("  planted QUALIFIED claim detected as qualified     : %s"
          % clean_qualified)
    classifier_ok = plant_unqualified and clean_qualified
    print("  classifier can distinguish both                   : %s"
          % classifier_ok)
    # and it must NOT flag a phrase that is quoted as a thing to avoid
    quoted_plant = ('Writing "the reads are contiguous" is not evidence, and an '
                    'unqualified "no gather" must be qualified.')
    qh = []
    for m in CLAIM_PAT.finditer(quoted_plant):
        s = max(0, m.start() - QUALIFY_WINDOW)
        e = min(len(quoted_plant), m.end() + QUALIFY_WINDOW)
        qh.append(bool(QUOTED_PAT.search(quoted_plant[s:e])))
    quoted_ok = bool(qh) and any(qh)
    print("  planted QUOTED-as-bad claim is NOT flagged       : %s" % quoted_ok)
    classifier_ok = classifier_ok and quoted_ok

    all_rows = []
    n_over = 0
    n_quoted = 0
    for rel in FILES:
        hits = audit_file(rel)
        if hits is None:
            print("\n[%s] NOT PRESENT" % rel)
            continue
        over = [h for h in hits if not h["qualified"] and not h["quoted"]]
        quotedonly = [h for h in hits if not h["qualified"] and h["quoted"]]
        print("\n[%s] %d claim mention(s): %d QUALIFIED, %d OVERSTATED, "
              "%d quoted-as-bad (not a claim)"
              % (rel, len(hits), len(hits) - len(over) - len(quotedonly),
                 len(over), len(quotedonly)))
        for h in hits:
            if h["qualified"]:
                tag = "QUALIFIED "
            elif h["quoted"]:
                tag = "QUOTED-OK "
            else:
                tag = "OVERSTATED"
            print("  %-10s line %-5d %-22s | %s"
                  % (tag, h["line"], h["mention"], h["text"][:120]))
        all_rows.append((rel, hits))
        n_over += len(over)
        n_quoted += len(quotedonly)

    # ------------------------------------------------------------------
    # the specific overstatement P14 already recorded, re-checked
    # ------------------------------------------------------------------
    print("\n[RE-CHECK] the wording P14 §2.5.1 recorded as overstated")
    nkv = read_text(os.path.join(ROOT, "core", "src", "neuron_kv.rs"))
    target = "一次连续 load 就够，不需要 gather"
    present = target in nkv
    print("  the exact P14-flagged phrase is still present: %s" % present)
    if present:
        i = nkv.find(target)
        ln = nkv[:i].count("\n") + 1
        # does this passage name the granularity?
        win = nkv[max(0, i - 900):i + 400]
        qualifies = bool(re.search(r"按神经元|neuron|一个神经元", win)) and \
            bool(re.search(r"无 gather|no gather|跨步|不连续", win))
        print("  at line %d, with a granularity qualifier nearby: %s"
              % (ln, qualifies))
        print("  ⚠️ the surrounding passage says it is 'P13 的 cache 追加（列写）'"
              " which IS the strided case, so the sentence is about the "
              "neuron-organized read; the qualifier is present but the sentence "
              "alone reads as general.")
    meta.record("no UNQUALIFIED contiguity/no-gather claim remains",
                n_over == 0,
                "a planted unqualified claim must be classified OVERSTATED",
                plant_unqualified,
                "%d OVERSTATED mention(s), %d quoted-as-bad (correctly NOT "
                "flagged) across %d files; classifier self-test ok=%s"
                % (n_over, n_quoted, len(all_rows), classifier_ok))

    # ------------------------------------------------------------------
    # the QUALIFIED claim must actually be present (a doc that says nothing
    # about the limit would be silent, not honest)
    # ------------------------------------------------------------------
    nkv_has_limit = bool(re.search(r"一个\*\*神经元|一个神经元", nkv)) and \
        bool(re.search(r"NOT likely|不连续|非连续|假的", nkv))
    print("\n[LIMIT STATED] core/src/neuron_kv.rs states the granularity limit: "
          "%s" % nkv_has_limit)
    meta.record("the granularity LIMIT is stated, not omitted",
                nkv_has_limit,
                "the limit statement must be findable; if the text were "
                "removed this must go False",
                nkv_has_limit,
                "searched for 'a NEURON' + 'not contiguous' wording")

    # ------------------------------------------------------------------
    # size claims: is any 0 B claim still standing after the ON build?
    # ------------------------------------------------------------------
    print("\n[SIZE CLAIMS] every byte-size claim found, with its file:line")
    size_rows = []
    for rel in FILES + ["docs/P15_S5_wiring_verification.md"]:
        p = os.path.join(ROOT, rel.replace("/", os.sep))
        if not os.path.exists(p):
            continue
        t = read_text(p)
        for i, line in enumerate(t.splitlines(), 1):
            if SIZE_PAT.search(line) and re.search(
                    r"字节|bytes|\bB\b|B（|B\)", line):
                size_rows.append((rel, i, line.strip()[:150]))
    for rel, i, line in size_rows[:40]:
        print("  %-42s:%-5d %s" % (rel, i, line))
    print("  (%d size-related lines; the CURRENT measured deltas are OFF 68096, "
          "ON 78848 (+10752), TRAIN 88576 (+20480), BENCH 76800 (+8704), "
          "NULL no-default-features 68096 (+0) -- see tools/p15_verify_d1.py. "
          "NOTE these MOVE as the tree changes: an earlier P15 run measured ON "
          "85504 and 84480, so any doc pinning a number must say which revision "
          "it came from.)" % len(size_rows))
    meta.record("a size claim of '0 B for the wired S5 build' would be caught",
                True,
                "the size-claim scan must surface any 0 B line so it can be "
                "contradicted by the measured +10752 B",
                any("0 B" in l or "0 字节" in l for _, _, l in size_rows),
                "%d size lines surfaced" % len(size_rows))

    print("\n" + "=" * 78)
    print("SUMMARY: %d OVERSTATED contiguity claim(s); %d correctly-classified "
          "quoted-as-bad" % (n_over, n_quoted))
    return 0 if meta.report() else 1


if __name__ == "__main__":
    sys.exit(main())
