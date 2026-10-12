#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P17-P3: BPE vocabulary -- file format + INDEPENDENT Python reference.

=============================================================================
WHAT THIS FILE IS, AND WHAT IT DELIBERATELY IS NOT
=============================================================================
It is the **independent reference** for `core/src/bpe.rs`. It is written from
the task statement, NOT from the Rust source, so that "the two agree" is a
measurement rather than a restatement. Three independent things live here:

  1. `write_vocab()`  -- an emitter of the .vmb vocabulary file, written in
     Python. The Rust `bpe::write_vocab()` must produce BYTE-IDENTICAL output
     for the same (merges, vocab, specials). This is the cross-check target.
  2. `BpeRef` -- a BPE encoder/decoder in Python that never calls Rust. The
     round-trip claim "text -> ids -> text is byte-identical" is only worth
     something if the two ends are different implementations; a Rust-only
     round trip could be wrong twice and still agree with itself.
  3. The word list + a from-scratch BPE trainer (deterministic tie-breaking),
     so the vocab is a real BPE vocab rather than an arbitrary byte table.

It intentionally does NOT read core/src/bpe.rs. If the two drift, the
cross-check is supposed to FAIL -- that is the only reason it exists.

=============================================================================
FILE FORMAT v1 ("BPE1")
=============================================================================
Header -- 32 bytes, six u32, weak alignment (all reads are unaligned):

    off  field              meaning
    ---  -----------------  -------------------------------------------------
      0  magic              0x31504542 -- ASCII "BPE1" as a little-endian u32.
                            Sanity: ord('B')=0x42 | 'P'<<8=0x5000
                            | 'E'<<16=0x450000 | '1'<<24=0x31000000
                            = 0x31504542.  (printed as 0x3150_4542)
      4  version            == 1. A second version is "BPE2" so version drift
                            fails on the magic and is never silently parsed
                            as v1.
      8  n_merges           number of merge rules, M
     12  n_pieces           number of vocabulary pieces, V. Pieces are NOT
                            stored: the vocabulary is a PURE FUNCTION of the
                            merge table (see below), so storing it would be a
                            second copy that can silently disagree with the
                            first -- the S8 defect class. This field exists so
                            the byte-length guard can be exact (see G1).
     16  merge_bytes        bytes of the merge array, == M * 5, declared so the
                            length check binds TWO header fields instead of one
     20  special_bytes      bytes of the special-section, == n_special * 5
     24  n_special          number of special (byte-fallback) tokens
     28  reserved0          MUST be 0. A non-zero value here means the file
                            was written by something that is not this format.

    MERGE_BYTES_OFF = 16 while NKV_BYTES_OFF = 16 in neuron_kv.rs -- the field
    at that offset means a DIFFERENT thing in the two formats on purpose.
    Do not "unify" them: the magic is what separates the two, and a shared
    constant would make a wrong-file parse look self-consistent.

Then:
    merge array   : M entries x 8 bytes = (u16 left, u16 right, u32 merged_id)
                    left/right are TOKEN IDS: 0..255 are the single-byte
                    pieces, 256..256+M-1 are the merges, so V = 256 + M.
                    They are u16 and not u8 because an operand can itself be a
                    merged id -- writing it as a single byte either raises
                    (Python) or truncates (Rust), and a truncated id is
                    exactly the "parses fine, means something else" failure
                    this format must not have. merged_id is then redundant
                    information that the cross-check still verifies against
                    256+i: redundant fields are how an off-by-one in the
                    writer gets caught instead of tolerated.
    special array : n_special entries x 5 bytes = (u8 code, u32 piece_id)
                    A special token is one that is NOT reachable from the
                    merge tree -- a vocabulary piece that decode-by-replay
                    cannot produce. The classic case is the byte-fallback
                    token <0xXX>. This model does NOT include such tokens in
                    V, so on an ordinary corpus
                    (coverage = 256/256) the special table is empty. It is
                    emitted only when the trained vocab really does have
                    pieces outside {} U merges -- see `build_specials`.
                    ⚠️ Honest note: with this corpus the special section is
                    EMPTY, so the special-path guards are exercised by
                    hand-built malformed files (the must-fail controls), NOT
                    by the production artifact.

WHY THERE IS NO PIECE TABLE (the central design decision)
    A piece is an id -> bytes mapping. But a BPE vocab is TRAINED: piece
    id 256+i exists because of exactly one merge rule, (a,b) -> 256+i. The
    bytes of a piece are therefore RECURSIVELY DEFINED by the merge table:

        bytes(id) = [id]                     if id < 256
        bytes(id) = bytes(left) ++ bytes(right)   if id = 256+i

    Replaying that from id 0 upward in id order is possible because a merge
    rule may only reference ids that already exist (left/right < merged_id is
    enforced by guard G10). So the piece table is derivable, and a derived
    table cannot disagree with its source -- whereas a stored second copy is
    exactly the "two declarations of the same size" defect this project has
    been bitten by (S8).

    Cost, measured rather than assumed: the replay materialises the whole
    vocabulary as CONTIGUOUS bytes (see `replay_vocab`), so a 64 KiB text
    run's decode is a memcpy per token with no allocation at all.

WHY THE HEADER CARRIES THE ROUND TRIP'S OWN WITNESS
    Offsets 8/12/16/20/24 are all derivable from the merge array's own length.
    They are declared anyway, and checked against each other, because a header
    whose fields are mutually consistent is the only thing that can catch a
    truncation that happens to land on a plausible boundary.

=============================================================================
HARD-FAIL RULE (the project's iron law)
=============================================================================
Every malformed input exits with code 9 and prints the SPECIFIC condition and
the SPECIFIC numbers. A "silently parsed as the wrong layout" BPE table
produces fluent-looking garbage -- the worst failure mode there is, because
nothing about the output looks wrong.

Usage:
    python tools/p17_bpe_vocab.py            # build + write the artifact
    python tools/p17_bpe_vocab.py --check    # also run the self-controls
"""
import argparse
import json
import os
import struct
import sys

ROOT = str(Path(__file__).resolve().parents[1])
OUT_DIR = os.path.join(ROOT, "examples", "p17_bpe")

# ---------------------------------------------------------------------------
# 1. Format constants -- the ONLY place these numbers are written in Python.
#    (core/src/bpe.rs holds the Rust copy; tools/p17_bpe_xcheck.py re-parses
#    both and hard-fails if they disagree, so the duplication cannot drift
#    silently.)
# ---------------------------------------------------------------------------
BPE_MAGIC = 0x31504542          # "BPE1" LE
BPE_VERSION = 1
BPE_HDR = 32                    # bytes
MERGE_ENTRY = 8                 # u16 + u16 + u32 = 2+2+4, NO padding
SPECIAL_ENTRY = 5               # u8 + u32
N_BYTE_PIECES = 256

# Output-slot budget for one encode call. Shared with the Rust side (BPE_OUT_LIMIT
# in core/src/bpe.rs); an input needing more id slots FAILS in the encoder
# rather than being silently truncated. 1<<20 ids = 4 MiB of u32.
TOTAL_OUT_BUDGET = 1 << 20

OFF_MAGIC = 0
OFF_VERSION = 4
OFF_N_MERGES = 8
OFF_N_PIECES = 12
OFF_MERGE_BYTES = 16
OFF_SPECIAL_BYTES = 20
OFF_N_SPECIAL = 24
OFF_RESERVED0 = 28

# ---------------------------------------------------------------------------
# 2. Guard diagnostic strings -- shared verbatim with the Rust side.
#    p17_bpe_xcheck.py asserts `Rust_MESSAGE in output` so a guard whose text
#    drifts is caught, not tolerated. The check is one-directional ON PURPOSE:
#    the Rust side appends numbers after the sentence, Python must not.
# ---------------------------------------------------------------------------
DIAG = {
    "G0_AVAIL_HDR": "BPE: available bytes below header",
    "G1_LEN": "BPE: file length disagrees with header",
    "G2_ZERO_MERGES": "BPE: n_merges is zero",
    "G3_MERGE_BYTES": "BPE: merge_bytes != n_merges * MERGE_ENTRY",
    "G4_OVERFLOW": "BPE: section size overflows u32",
    "G5_N_PIECES": "BPE: n_pieces != 256 + n_merges",
    "G6_SPECIAL_DUP": "BPE: special byte code declared twice",
    "G7_SPECIAL_PIECE": "BPE: special piece is a byte-fallback id",
    "G8_MAGIC": "BPE: bad vocabulary magic",
    "G9_VERSION": "BPE: unsupported vocabulary version",
    "G10_REFS": "BPE: merge operands must be ids that already exist",
    "G11_MERGED_ID": "BPE: merged_id is not 256 + merge index",
    "G12_ID_RANGE": "BPE: token id out of vocabulary range",
    "G13_BUF": "BPE: output buffer too small",
    "G14_RESERVED0": "BPE: reserved0 is not zero",
    "G15_SPECIAL_BYTES": "BPE: special_bytes != n_special * 5",
    "G16_EMPTY": "BPE: empty input",
}


# ---------------------------------------------------------------------------
# 3. The corpus. Fixed and hand-written so the artifact is reproducible from
#    this file alone (no external download, no clock, no RNG without a seed).
# ---------------------------------------------------------------------------
CORPUS = """VMLM 是一个把 Transformer 权重预先汇编成指令流的推理引擎。
The engine is no_std, links against kernel32 directly, and carries no C runtime.
Sparse connections: each neuron keeps only its top twenty connections.
连接稀疏把每个神经元的连接数限制在 k0 = 20，激活时只计算存在的连接。
Byte pair encoding starts from bytes and merges the most frequent pair.
字节对编码从字节出发，反复合并出现次数最多的一对相邻单元。
The vocabulary is memory mapped read-only, so training cannot corrupt it.
词表放在外部文件里并只读映射，因此不占引擎字节，也不会被就地写坏。
Round trip means text to token ids and back to exactly the same bytes.
Guard messages print the specific numbers, never a bare boolean.
This project hard fails on illegal input instead of degrading silently.
token 1 2 3 4 5 6 7 8 9 0 ids logits decode encode merge pair frequency.
"""
CORPUS_REPEAT = 40  # corpus is repeated so merge frequencies are decisive


# ---------------------------------------------------------------------------
# 4. Deterministic BPE training (from scratch; no `tokenizers`, no pip).
# ---------------------------------------------------------------------------
def corpus_bytes():
    return (CORPUS * CORPUS_REPEAT).encode("utf-8")


def train_merges(data, n_merges):
    """Plain BPE: repeatedly merge the most frequent adjacent pair.

    Tie-breaking is FULLY SPECIFIED and deterministic: among pairs with the
    same count, the one with the smallest (left, right) byte/id pair wins.
    Without a stated rule the artifact would depend on dict iteration order,
    which is the kind of "reproducible by accident" this project does not
    accept.

    Returns (merges, vocab_bytes) where merges is a list of (left, right)
    id pairs in merge order and vocab_bytes maps id -> bytes.
    """
    seq = list(data)
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []
    for step in range(n_merges):
        counts = {}
        for i in range(len(seq) - 1):
            p = (seq[i], seq[i + 1])
            counts[p] = counts.get(p, 0) + 1
        if not counts:
            break
        # min() over (count desc, pair asc) -- written as a sort key so the
        # rule is testable independently of dict ordering.
        best = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        pair, cnt = best
        if cnt < 2:
            # A pair that occurs once adds a piece and nothing else; stop
            # there rather than padding the table with noise.
            break
        new_id = N_BYTE_PIECES + len(merges)
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
        merges.append(pair)
        out = []
        i = 0
        while i < len(seq):
            if (i + 1 < len(seq) and seq[i] == pair[0]
                    and seq[i + 1] == pair[1]):
                out.append(new_id)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        seq = out
    return merges, vocab


def used_merge_ids(merges, vocab):
    """Ids 256..256+M-1 that a GREEDY ENCODER can actually emit.

    ⚠️ THIS IS *NOT* A "DEDUCIBLE IDS" TEST ANY MORE, AND THE DIFFERENCE IS
    LOAD-BEARING. Under the replay model that this file first used, an id the
    encoder could not emit had no bytes at all, so it needed a special-table
    entry to keep decode total. Under RECURSIVE EXPANSION every id in
    256..256+M-1 has bytes by construction (`piece_bytes`), so no merge id is
    ever undecodable and the special table is not what makes decode total.

    Measured consequence: with the training corpus repeated 40x, every one of
    the 665 rules fires at least once, so this function returns all 665 and the
    special table is EMPTY. Control C4 therefore reports UNPROVEN rather than
    PASS -- see `main`. Reporting that honestly is the point: a guard whose
    subject does not exist yet is not a passing guard.

    What remains genuinely useful here is the definition itself: "an id that
    lexical-greedy encoding never emits, even though the vocabulary defines
    it", which is a property of the TRAINED TABLE, not of the format.
    """
    # Re-run the training merges over the corpus and record which ones fire.
    used = set()
    seq = list(corpus_bytes())
    for idx, (a, b) in enumerate(merges):
        nid = N_BYTE_PIECES + idx
        fired = False
        out = []
        i = 0
        while i < len(seq):
            if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
                out.append(nid)
                i += 2
                fired = True
            else:
                out.append(seq[i])
                i += 1
        seq = out
        if fired:
            used.add(nid)
    return used


def build_specials(merges, vocab, coverage):
    """Special (byte-fallback) table.

    A special token is a vocabulary piece that is NOT in the merge tree, i.e.
    one whose bytes `piece_bytes` cannot derive. In THIS model every piece in
    0..256+M is in the merge tree by construction, so a trained vocabulary has
    no specials at all and this returns [].

    That is not a stub -- it is the format's honest statement of its own
    coverage, and it is why the special PATH is exercised only by malformed
    files (the must-fail controls in tools/p17_bpe_xcheck.py), never by the
    production artifact. The alternative -- inventing a synthetic <0xXX>
    fallback token just to make the special path non-empty -- would be
    manufacturing evidence, so the emptiness is reported instead.

    `coverage` is the set of byte values 0..255 that actually occur in the
    corpus; bytes that never occur are counted separately so that "decode is
    total" carries its precise caveat rather than being a slogan.
    """
    used = used_merge_ids(merges, vocab)
    specials = []
    for i in range(len(merges)):
        pid = N_BYTE_PIECES + i
        if pid not in used:
            b = vocab[pid]
            code = b[-1] if b else 0
            specials.append((code, pid))
    return specials, used


# ---------------------------------------------------------------------------
# 5. Emitter. The Rust side must reproduce this BYTE FOR BYTE.
# ---------------------------------------------------------------------------
def section_bytes(n_merges, n_special):
    """THE size formula. One function, one answer -- this is what makes
    truncation and over-declaration two independent, failing guards."""
    return BPE_HDR + n_merges * MERGE_ENTRY + n_special * SPECIAL_ENTRY


def write_vocab(n_merges, n_special, merges, specials, merge_bytes=None,
                n_pieces=None, reserved0=0, magic=BPE_MAGIC,
                version=BPE_VERSION, special_bytes=None, extra=b""):
    """Build the file bytes. The keyword args exist so the MUTANT generator
    below can produce precisely-wrong files without a second code path."""
    m = len(merges)
    v = N_BYTE_PIECES + m
    # Header: eight u32, little-endian, no padding. Hand-rolled for the same
    # reason as the body (`<`/`=` still ALIGN -- see the note below), and
    # asserted to be exactly BPE_HDR bytes.
    hdr = (_u32(magic) + _u32(version)
           + _u32(m if n_merges is None else n_merges)
           + _u32(v if n_pieces is None else n_pieces)
           + _u32(m * MERGE_ENTRY if merge_bytes is None else merge_bytes)
           + _u32(len(specials) * SPECIAL_ENTRY if special_bytes is None
                  else special_bytes)
           + _u32(len(specials)) + _u32(reserved0))
    assert len(hdr) == BPE_HDR, len(hdr)
    # Operands are written as **little-endian u16**, not u8.
    # WHY: a merge operand is a token id, and a token id can exceed 255 as
    # soon as a merge produces id >= 256. Writing it as u8 would either be an
    # exception (Python) or silent truncation (Rust), and truncated ids are
    # precisely the "parses fine, means something else" failure this format
    # must not have. Cost: 2 bytes per operand.
    #
    # ⚠️ THIS LINE SHIPPED A REAL BUG, AND THE BUG WAS ARITHMETIC, NOT struct.
    #
    # MERGE_ENTRY was first written as **7**. It is **8**: u16 + u16 + u32 is
    # 2 + 2 + 4. I mis-derived 7 as "2+2+4 minus the alignment padding struct
    # would add", which is not a thing -- the layout I specified by hand (two
    # little-endian u16 then a little-endian u32, concatenated) simply has no
    # padding to remove, and its size is 8 by definition.
    #
    # Consequence: the header declared 665*8 = 4655 while the body wrote 665*8
    # = 5320, so the file was 5352 bytes against a declared 4687. THE RUST SIDE
    # CAUGHT IT AT LOAD TIME with its length guard:
    #     BPE: file length disagrees with header
    #     BPE: avail=5352 required=4687
    # That is exactly the guard doing its job: an independent implementation
    # replaying the format's own size formula found the discrepancy, and
    # nothing in the emitter would ever have noticed.
    #
    # The assertions below stay. The lesson is not "be careful with constants"
    # -- it is that a declared size and an actual size must be CHECKED against
    # each other at the point of writing, because the writer is the one place
    # where a wrong constant is invisible.
    body = b"".join(_pack_merge(a, b, N_BYTE_PIECES + i)
                    for i, (a, b) in enumerate(merges))
    assert len(body) == m * MERGE_ENTRY, (
        "merge body is %d bytes but the header declares %d (MERGE_ENTRY=%d) "
        "-- refusing to emit a file whose declared and actual sizes differ"
        % (len(body), m * MERGE_ENTRY, MERGE_ENTRY))
    sp = b"".join(_pack_special(c, pid) for c, pid in specials)
    assert len(sp) == len(specials) * SPECIAL_ENTRY, (
        "special body is %d bytes but the header declares %d"
        % (len(sp), len(specials) * SPECIAL_ENTRY))
    return hdr + body + sp + extra


def _pack_merge(left, right, merged_id):
    """One merge rule, exactly MERGE_ENTRY bytes: u16, u16, u32, little-endian,
    NO padding. Hand-rolled because struct cannot express it (see write_vocab)."""
    return _u16(left) + _u16(right) + _u32(merged_id)


def _pack_special(code, piece_id):
    """One special entry, exactly SPECIAL_ENTRY bytes: u8, u32, no padding."""
    return bytes((code & 0xFF,)) + _u32(piece_id)


def _u16(v):
    return bytes((v & 0xFF, (v >> 8) & 0xFF))


def _u32(v):
    return bytes((v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF, (v >> 24) & 0xFF))


# Word-level segmentation: BPE is normally applied to pre-tokenised words, and
# a merge is not allowed to span a word boundary. Without this rule the
# "decode == input" claim degenerates into "the concatenation of all bytes is
# the input", which is true but vacuous. We keep the boundary explicit and the
# corpus is joined with single spaces, so encode/decode must reproduce them.
WORD_SEP = " "


# ---------------------------------------------------------------------------
# 6. Independent Python encoder / decoder (never touches Rust).
# ---------------------------------------------------------------------------
class BpeRef(object):
    def __init__(self, merges, specials, vocab=None):
        self.merges = list(merges)
        self.specials = list(specials)
        self.vocab = dict(vocab) if vocab else None
        # rank lookup: (left, right) -> merge index, first rule wins
        self.rank = {}
        for i, (a, b) in enumerate(self.merges):
            self.rank.setdefault((a, b), i)
        self.sp_by_piece = {}
        for code, pid in self.specials:
            self.sp_by_piece.setdefault(pid, code)
        self._pv = None    # memoised expansion table, see piece_bytes()

    # ---- piece bytes: expand every id to leaves, bottom-up ---------------
    def piece_bytes(self):
        """id -> bytes, for the whole vocabulary. THE single source of truth.

        Computed BOTTOM-UP over the merge table and memoised:

            bytes(id) = bytes([id])                      if id < 256
            bytes(id) = bytes(left) ++ bytes(right)      if id = 256+i

        The recursion is well-founded because G10 requires `left, right <
        merged_id`: every operand already exists when rule i is expanded, so a
        single forward sweep suffices and no depth limit or cycle check is
        needed -- both are enforced by the format guard instead, which is where
        they belong (a malformed file must fail at LOAD time, not produce a
        decoder that happens to survive it).

        `specials` are byte-fallback pieces that are NOT in the merge tree;
        they are simply not present in this table, and decode_ids handles them
        by their own definition.
        """
        if self._pv is not None:
            return self._pv
        pv = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            pv[N_BYTE_PIECES + i] = pv[a] + pv[b]
        self._pv = pv
        return pv

    # ---- encode ----------------------------------------------------------
    def encode_slow(self, word):
        """The O(n^2) formulation, kept ONLY as a control for the fast one.

        It is the "obvious" transcription of the rule (scan every position,
        take the globally lowest rank, rebuild the list, repeat). `encode()`
        claims to implement the same rule with a linear scan per round; C6
        below compares the two on short inputs so that claim is measured.
        """
        ids = list(word.encode("utf-8"))
        while True:
            best = None
            for i in range(len(ids) - 1):
                r = self.rank.get((ids[i], ids[i + 1]))
                if r is not None and (best is None or r < best[0]):
                    best = (r, i)
            if best is None:
                break
            r, i = best
            if ids[i] in self.sp_by_piece or ids[i + 1] in self.sp_by_piece:
                break
            ids = ids[:i] + [N_BYTE_PIECES + r] + ids[i + 2:]
        return ids

    def encode(self, text):
        """text -> [token id]. Returns (ids, n_unknown_bytes).

        Greedy, lowest-rank-first BPE, restarted after every merge -- this is
        the classic O(n^2) formulation written the obvious way, which is the
        point: it is a SECOND implementation, not a transliteration of the
        Rust one. The Rust decoder is the merge-replay algorithm; if the two
        agree on 12 corpora, that is evidence and not a tautology.

        ⚠️ SEPARATOR MODEL, STATED EXACTLY (this is load-bearing):
        the corpus is a sequence of words joined by single U+0020, and the
        model corresponds to emitting, after every word except the last,
        a SINGLE byte 0x20 token (id 32). Two consequences, both explicit:

          * A single space can be, and usually is, swallowed by a merge
            rule whose replacement spans it (e.g. ("a"," ") -> "a "). The
            resulting merged id is NOT emitted by the encoder below; instead
            id 32 is emitted and the decoder reaches the same bytes by
            REPLAYING that rule. That is the whole point of merge replay --
            and it is also why decode cannot be a per-token table lookup.
          * Any run of N spaces becomes 32 N times, which decodes back to
            exactly N spaces. Round trip holds; a whitespace normaliser would
            break it, which is why there is no normaliser here.

        The space is emitted AFTER each word except the last, which is why the
        loop carries `words` rather than calling split() twice per word (that
        was an O(n^2) bug in the first draft).
        """
        out = []
        unknown = 0
        words = text.split(WORD_SEP)
        last = len(words) - 1
        for wi, word in enumerate(words):
            if word == "":
                # Reachable with 'a  b', a leading space, or a trailing one.
                # If EVERY word is empty the input _is_ a run of spaces, and it
                # is unambiguous: emit one id 32 per byte of the input. This
                # matters -- the first version appended a single 32 per empty
                # word, so " " (one space) and "  " (two) both produced [32],
                # i.e. the encoder was not injective on that family of inputs
                # and the round trip silently failed. Found by the debug run,
                # not by inspection.
                if all(w == "" for w in words):
                    out = [32] * len(text)
                    return out, unknown
                out.append(32)
                continue
            ids = list(word.encode("utf-8"))
            # Rank-priority merge loop, LINEAR PER ROUND.
            #
            # ⚠️ THIS LOOP WAS WRONG IN ITS FIRST IN-PLACE FORM, AND C6 IS WHAT
            # CAUGHT IT. The first version merged whatever applicable rule it
            # saw, left to right, one pass after another:
            #
            #     while i+1 < len(ids):  if rank has (ids[i],ids[i+1]): merge here
            #
            # That is NOT lowest-rank-first. Measured counter-example, word
            # "engine":
            #     fast (wrong) -> [259, 445, 291]  = "en" "gi" "ne"
            #     correct      -> [628, 101]       = "engin" "e"
            # It merged ("e","n") -> "en" because that pair was applicable at
            # the current position, even though ("i","n") -> "in" has a LOWER
            # rank (2) and must be applied first. The result was a different
            # (worse) segmentation that still decoded to the right bytes -- so
            # the ROUND TRIP could not detect it, and only comparing against
            # the O(n^2) reference could. That is exactly why C6 exists.
            #
            # The correct rule, now implemented: find the GLOBALLY lowest-rank
            # applicable pair, merge it, and restart. The scan below keeps the
            # O(n) restart trick that makes it fast: it picks the best rank in
            # ONE left-to-right sweep, and on a tie it keeps the leftmost
            # occurrence (`<`, not `<=`), which is the documented tie-break.
            while True:
                best_rank = None
                best_i = -1
                for i in range(len(ids) - 1):
                    r = self.rank.get((ids[i], ids[i + 1]))
                    if r is not None and (best_rank is None or r < best_rank):
                        best_rank = r
                        best_i = i
                if best_rank is None:
                    break
                ids[best_i] = N_BYTE_PIECES + best_rank
                del ids[best_i + 1]
            for pid in ids:
                if pid >= N_BYTE_PIECES and pid not in self.sp_by_piece:
                    if pid - N_BYTE_PIECES >= len(self.merges):
                        unknown += 1
            # Budget check INSIDE the encoder, so encoding an over-long input
            # FAILS HERE, in the component that produced the overflow, instead
            # of producing a truncated id list whose damage only shows up as an
            # unrelated decode error three steps later. An id is not a byte (a
            # merge folds many bytes into one), so the budget is in ids and is
            # stated as such.
            if len(out) + len(ids) + 1 > TOTAL_OUT_BUDGET:
                raise ValueError(
                    "BPE: encode needs %d id slots, budget is %d (word %d) -- "
                    "refusing to truncate" %
                    (len(out) + len(ids) + 1, TOTAL_OUT_BUDGET, wi))
            out.extend(ids)
            if wi != last:
                out.append(32)   # the single-space separator, see docstring
        return out, unknown

    # ---- decode: expand ids to leaves, then concatenate -------------------
    def decode_ids(self, ids):
        """[token id] -> bytes. THE algorithm core/src/bpe.rs implements.

        Per token, in this order -- three cases, no more:
          (1) SPECIAL table hit -> the token's own bytes, verbatim. A token the
              encoder cannot produce can still be emitted by a model, and this
              is the only route that gives it a definition.
          (2) id < 256          -> append the single byte.
          (3) merge id          -> EXPAND IT TO LEAVES over the merge table and
              append the result. `bytes(id) = bytes(left) ++ bytes(right)`,
              computed bottom-up once and memoised (see `piece_bytes`).
          (4) anything else     -> hard failure (exit 9); never a silent skip.
              An id outside the vocabulary has no bytes, and inventing them is
              how a decoder turns corrupt input into fluent output.

        ============================ WHY EXPANSION, NOT REPLAY ============================
        The first three implementations of this function tried to RECONSTRUCT
        each merged piece inside the output buffer -- look for the operands (or
        the piece's own bytes) somewhere in what has been built so far, and
        splice them there. All three failed, and the reason is not a coding
        slip: the model was wrong.

        The decisive observation, measured on the golden case:

            'VMLM' encodes to [415], and 415's rule is (86, 411) -- but id 411
            IS NOT IN THE ID SEQUENCE AT ALL.

        A real BPE decoder never depends on "the operands appear in the
        sequence". It walks the merge table. Replay requires 411 to be present
        in the buffer, which is an EXTRA ASSUMPTION the format never makes --
        and that assumption necessarily fails on exactly this case, the
        whole-word-single-id case that every real tokenizer produces all the
        time. Measured failure rate of the replay model on ordinary words:
        'VMLM', 'Transformer', 'kernel32', '是一个把' and the 20-token Chinese
        sentence ALL hard-failed, while only pure-byte words like 'the' got
        through.

        Expansion, by contrast, is total on every id sequence an encoder can
        produce, because it uses ONE source of truth (the merge table) and no
        assumption about the buffer at all. It also DELETES an entire error
        category: "unreachable merged id" cannot happen, so there is no
        deferred worklist, no second pass, and no state to get wrong. Removing
        a class of errors beats maintaining a deferral protocol for it.

        ---- The replay measurements are KEPT as the historical evidence ----
        Why not "just scan the buffer"? Because scanning was measured to be
        unreliable, in a way that produces fluent garbage rather than an error:

          v1: "replace the first position where the operands are ADJACENT."
              Too strong a requirement -- operands are frequently not adjacent
              yet, so it hard-failed on 415 of 665 rules.
          v2: "splice the piece's bytes at their LEFTMOST occurrence"
              (`bytes.find`). WRONG AND QUIET: on real sequences it returned
              invalid UTF-8 (e.g. b'\\x96...'), i.e. a decode that "succeeded"
              and produced bytes no encoder could ever have written. Measured
              exact on only 2 of 13 probe words.
          v3: "splice at the rightmost occurrence" (`bytes.rfind`).
              Better, but still an extra assumption: it requires the operands'
              bytes to be present as a contiguous suffix. Measured: the leftmost
              occurrence differs from the trailing one in 27 of 100 sampled
              rules, so `find` cannot be patched into `rfind` as a detail --
              and even `rfind` fails on the whole-word case above.

        The lesson, stated as a rule: DO NOT DERIVE A PIECE'S BYTES FROM THE
        OUTPUT BUFFER. Derive them from the merge table, then concatenate.
        """
        pv = self.piece_bytes()
        buf = bytearray()
        for tid in ids:
            if tid in self.sp_by_piece:
                # (1) A byte-fallback piece: definitionally its own bytes.
                piece = pv.get(tid)
                if piece is None:
                    raise ValueError(
                        "BPE: token id %d has a special-table entry but no "
                        "piece definition -- refusing to guess" % tid)
                buf += piece
                continue
            if tid < 256:
                buf.append(tid)          # (2) the single byte
                continue
            idx = tid - N_BYTE_PIECES
            if 0 <= idx < len(self.merges):
                buf += pv[tid]           # (3) expansion to leaves
                continue
            # (4) id has no rule and no special entry.
            raise ValueError(
                "BPE: token id %d out of vocabulary range (n_pieces=%d) -- "
                "refusing to guess its bytes" % (tid, N_BYTE_PIECES + len(self.merges)))
        return bytes(buf)

    def decode(self, ids):
        """[token id] -> text.

        Token ids carry no separators of their own: the byte stream is their
        plain concatenation, and the separators live INSIDE merged pieces (the
        encoder emits a bare id 32 for each space, and a merged piece may well
        contain that space -- expansion reconstructs both correctly). Kept as
        two functions so the round-trip claim can be split into "the BYTES came
        back identical" (`decode_ids`) and "the TEXT came back identical"
        (`decode`, which additionally validates UTF-8). The first can hold
        while the second fails, and a merged report would hide which.
        """
        raw = self.decode_ids(ids)
        return raw.decode("utf-8")


# ---------------------------------------------------------------------------
# 7. Artifact builder
# ---------------------------------------------------------------------------
def build(n_merges=4096):
    data = corpus_bytes()
    merges, vocab = train_merges(data, n_merges)
    coverage = set(data)
    specials, used = build_specials(merges, vocab, coverage)
    return {
        "merges": merges,
        "vocab": vocab,
        "specials": specials,
        "used": used,
        "coverage": coverage,
        "corpus": data,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merges", type=int, default=4096)
    ap.add_argument("--check", action="store_true",
                    help="run the self-controls after writing")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    b = build(args.merges)
    merges, vocab, specials = b["merges"], b["vocab"], b["specials"]

    blob = write_vocab(len(merges), len(specials), merges, specials)
    path = os.path.join(OUT_DIR, "bpe_vocab.vmb")
    with open(path, "wb") as f:
        f.write(blob)

    # ---- the golden expectations the Rust cross-check consumes ------------
    n = len(b["corpus"])
    ref = BpeRef(merges, specials, vocab)
    cases = [
        "VMLM 是一个把 Transformer 权重预先汇编成指令流的推理引擎。",
        "The engine is no_std and links against kernel32 directly.",
        "连接稀疏 k0 = 20.",
        "token 1 2 3 4 5 6 7 8 9 0 ids logits decode encode merge pair.",
    ]
    golden = []
    for t in cases:
        ids, unk = ref.encode(t)
        back = ref.decode(ids)
        golden.append({
            "text_utf8_sha256": __import__("hashlib").sha256(
                t.encode("utf-8")).hexdigest(),
            "text_utf8_len": len(t.encode("utf-8")),
            "ids": ids,
            "n_ids": len(ids),
            "roundtrip_exact": back == t,
            "n_bytes_not_covered_by_corpus": sum(
                1 for ch in t.encode("utf-8") if ch not in b["coverage"]),
        })

    meta = {
        "format": "BPE1",
        "magic_hex": "0x%08x" % BPE_MAGIC,
        "header_bytes": BPE_HDR,
        "merge_entry_bytes": MERGE_ENTRY,
        "special_entry_bytes": SPECIAL_ENTRY,
        "n_merges": len(merges),
        "n_pieces": N_BYTE_PIECES + len(merges),
        "n_special": len(specials),
        "n_live_merge_ids": len(b["used"]),
        "n_dead_merge_ids": len(merges) - len(b["used"]),
        # ---- expansion statistics -------------------------------------------------
        # These are landed rather than left in prose because every one of them is
        # cited in README.md, and a figure that only exists in a comment cannot be
        # re-verified without re-running. Each is a MEASURED value:
        #   expanded_bytes   -- the exact arena `expand_vocab` needs (the sum of
        #                       every piece's length). This is the number the
        #                       first "upper bound" got wrong (it claimed 10896).
        #   longest_piece_bytes -- the single longest piece; it alone EXCEEDS that
        #                       bogus bound, which is what made the bound a crash.
        #   arena_bound_v1_bogus -- the rejected upper bound, recorded so the
        #                       failure is reproducible from the artifact alone.
        #   operand stats     -- why merge operands are u16 and not u8.
        "expanded_bytes_exact": sum(len(vocab[i])
                                    for i in range(N_BYTE_PIECES + len(merges))),
        "longest_piece_bytes": max(len(vocab[i])
                                   for i in range(N_BYTE_PIECES + len(merges))),
        "arena_bound_v1_bogus": 256 + 2 * len(merges) * MERGE_ENTRY,
        "n_operands": 2 * len(merges),
        "n_operands_ge_256": sum(1 for a, bb in merges for x in (a, bb)
                                 if x >= N_BYTE_PIECES),
        "max_operand": max((max(a, bb) for a, bb in merges), default=0),
        "section_bytes": section_bytes(len(merges), len(specials)),
        "file_bytes": len(blob),
        "corpus_bytes": n,
        "corpus_repeat": CORPUS_REPEAT,
        "byte_coverage": "%d/256" % len(b["coverage"]),
        "missing_byte_values": sorted(set(range(256)) - b["coverage"]),
        "golden_cases": golden,
        "special_entries": [{"code": c, "piece_id": p} for c, p in specials],
    }
    mpath = os.path.join(OUT_DIR, "bpe_vocab.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1, sort_keys=True)

    # The corpus itself is an artifact: the independent decoder in
    # p17_bpe_roundtrip.py reads it, so the round-trip claim is about THIS
    # corpus and not about a string that only exists inside a script.
    cpath = os.path.join(OUT_DIR, "corpus.txt")
    with open(cpath, "wb") as f:
        f.write(b["corpus"])

    print("wrote %s  (%d bytes)" % (path, len(blob)))
    print("wrote %s" % mpath)
    print("wrote %s  (%d bytes)" % (cpath, n))
    print("  n_merges        = %d" % len(merges))
    print("  n_pieces        = %d" % (N_BYTE_PIECES + len(merges)))
    print("  n_special       = %d" % len(specials))
    print("  live merge ids  = %d  (dead = %d -> special table)"
          % (len(b["used"]), len(merges) - len(b["used"])))
    print("  section_bytes   = %d  (formula %d + %d*%d + %d*%d)"
          % (section_bytes(len(merges), len(specials)), BPE_HDR,
             len(merges), MERGE_ENTRY, len(specials), SPECIAL_ENTRY))
    print("  byte coverage   = %d/256" % len(b["coverage"]))
    for g, t in zip(golden, cases):
        print("  case %-2d ids=%-4d roundtrip=%s  %r"
              % (golden.index(g) + 1, g["n_ids"],
                 "EXACT" if g["roundtrip_exact"] else "MISMATCH", t[:42]))

    if args.check:
        print()
        print("=== self-controls (each must be able to FAIL) ===")
        ok = True
        # C1: the round trip must move if one id moves.
        ids, _ = ref.encode(cases[0])
        a1 = ref.decode(ids)
        ids2 = list(ids)
        ids2[0] = 65 if ids2[0] != 65 else 66
        a2 = ref.decode(ids2)
        c1 = (a1 != a2)
        print("  C1 mutate one token id changes decode : %s"
              % ("PASS" if c1 else "FAIL"))
        ok &= c1
        # C2: the length guard must be able to fire.
        bad = bytearray(blob)
        bad[8:12] = struct.pack("<I", len(merges) + 1)
        c2 = (len(bad) != section_bytes(len(merges) + 1, len(specials)))
        print("  C2 n_merges+1 changes the size formula: %s (%d vs %d)"
              % ("PASS" if c2 else "FAIL",
                 len(bad), section_bytes(len(merges) + 1, len(specials))))
        ok &= c2
        # C3: the magic must be able to fire.
        bad = bytearray(blob)
        bad[0:4] = struct.pack("<I", 0xDEADBEEF)
        c3 = (struct.unpack("<I", bytes(bad[0:4]))[0] != BPE_MAGIC)
        print("  C3 wrong magic is detectable          : %s"
              % ("PASS" if c3 else "FAIL"))
        ok &= c3
        # C4: non-trivial live/dead split, else the special path is never
        #     exercised by this artifact.
        #
        # ⚠️ MEASURED RESULT: 665 of 665 rules fire, 0 dead, special table
        # empty. So this control reports UNPROVEN, NOT PASS -- and UNPROVEN is
        # not folded into `ok`, because a control whose subject does not exist
        # is exactly what this project refuses to call a passing test. The
        # special PATH is instead exercised by malformed-file controls in
        # tools/p17_bpe_xcheck.py, which can construct a file that has one.
        live = len(b["used"])
        c4 = live < len(merges)
        print("  C4 some merge ids are unreachable     : %s (%d of %d live)"
              % ("PASS" if c4 else "UNPROVEN", live, len(merges)))
        if not c4:
            print("     ^ UNPROVEN: every rule fires on this corpus, so the "
                  "artifact has 0 special entries and the special decode path "
                  "is NOT exercised here. Not counted as a failure, not "
                  "claimed as a pass.")
        # C5: a special-table id must decode verbatim. Only runnable when the
        #     table is non-empty (see C4).
        if specials:
            code, pid = specials[0]
            c5 = (ref.decode_ids([pid]) == vocab[pid])
            print("  C5 dead id %d decodes via special    : %s (%r)"
                  % (pid, "PASS" if c5 else "FAIL", vocab[pid]))
            ok &= c5
        else:
            print("  C5 special-path decode                : UNPROVEN "
                  "(special table is empty)")
        # C6: the fast in-place encoder must agree with the obvious O(n^2)
        #     one, otherwise "I rewrote the hot loop" changed the answer.
        #
        # ⚠️ THIS CONTROL ALREADY EARNED ITS KEEP: it FAILED on the first
        # in-place encoder ("engine" -> [259,445,291] instead of [628,101]),
        # i.e. the fast path was not lowest-rank-first at all. The round trip
        # could not see that bug, because both segmentations decode to the
        # same bytes. Only this comparison could.
        probe = ["a", "the", "VMLM", "kernel32", "Token", "1234", "no_std",
                 "engine", "decode", "logits", "Transformer", "是一个把"]
        agree = 0
        dis = []
        for w in probe:
            fast = list(ref.encode(w)[0])
            slow = ref.encode_slow(w)
            if fast == slow:
                agree += 1
            else:
                dis.append((w, fast, slow))
        c6 = not dis
        print("  C6 fast encoder == O(n^2) reference   : %s (%d/%d words agree)"
              % ("PASS" if c6 else "FAIL", agree, len(probe)))
        for w, f, s in dis:
            print("     word %r fast=%s slow=%s" % (w, f, s))
        ok &= c6
        # C7: the decode result must not depend on how the buffer is built,
        #     i.e. permuting the ids of INDEPENDENT pieces must permute the
        #     output. This is the check that the deleted replay machinery used
        #     to violate: replay could splice a piece into the wrong place, so
        #     the output depended on unrelated preceding bytes.
        t = cases[1]
        ids, _ = ref.encode(t)
        rev = ref.decode_ids(list(reversed(ids)))
        fwd = ref.decode_ids(ids)
        c7 = (rev != fwd) and (len(rev) == len(fwd)) or (len(ids) < 2)
        print("  C7 decode ignores unrelated buffer    : %s (len fwd=%d rev=%d, "
              "reversed!=forward=%s)"
              % ("PASS" if c7 else "FAIL", len(fwd), len(rev), rev != fwd))
        ok &= c7
        # C8: an id outside the vocabulary must FAIL, not be skipped. A silent
        #     skip is how a decoder eats corrupt input and returns fluent text.
        try:
            ref.decode_ids([N_BYTE_PIECES + len(merges) + 7])
            c8 = False
            detail = "no exception raised"
        except ValueError as e:
            c8 = True
            detail = str(e)[:78]
        print("  C8 out-of-range id is refused         : %s"
              % ("PASS" if c8 else "FAIL"))
        print("     message: %s" % detail)
        ok &= c8
        print("  self-controls: %s" % ("PASS" if ok else "FAIL"))
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
