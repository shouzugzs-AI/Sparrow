#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P10 B1: safetensors + bf16 reader.

WHY THIS EXISTS
---------------
`tools/pt2vmb.py` only reads `.pt`. MiniCPM5-1B ships a single
`model-00000-of-00001.safetensors` (2.16 GB, bf16). There was no reader.

WHY NOT USE THE `safetensors` / `torch` PACKAGES
-----------------------------------------------
Numpy alone is enough, and it keeps this repo's tooling dependency-free
(the same reason `pt2vmb.py` avoids exotic deps). The format is trivial:

    [0:8]   u64 little-endian  = N  (byte length of the JSON header)
    [8:8+N] UTF-8 JSON:  { "__metadata__": {...},
                           "tensor.name": {"dtype":"BF16",
                                           "shape":[d0,d1,...],
                                           "data_offsets":[beg,end]} }
    [8+N:]  raw tensor bytes, offsets are RELATIVE to 8+N

Endianness / dtype notes that actually bite:
  * BF16 is the TOP 16 BITS of an f32. Decoding "shift left 16" as uint32
    then view as float32 is EXACT and needs no math library.
  * F16 needs a real conversion (numpy has float16, so it is free).
  * The format is little-endian and offsets are relative to the END of the
    header, NOT to the file start. Getting this wrong yields plausible but
    garbage numbers (no exception) -- the classic silent-degradation shape.

USAGE
-----
    from safetensors_reader import SafeTensors
    st = SafeTensors(r"<外部模型目录>/MiniCPM5-1B/model-00000-of-00001.safetensors")
    st.names()                    # -> sorted list of tensor names
    st.meta("model.layers.0.x")   # -> dict(dtype, shape, data_offsets)
    arr = st.get("model.embed_tokens.weight")   # -> float32 numpy array
    st.get_raw("...")             # -> (dtype_str, shape, raw bytes)
"""

import json
import os
import struct

import numpy as np


# ---------------------------------------------------------------- dtype table
# name -> (numpy dtype used to VIEW the raw bytes, bytes per element)
_VIEW = {
    "F32": (np.dtype("<u4"), 4),
    "F16": (np.dtype("<f2"), 2),
    "BF16": (np.dtype("<u2"), 2),
    "I8": (np.dtype("i1"), 1),
    "U8": (np.dtype("u1"), 1),
    "I16": (np.dtype("<i2"), 2),
    "I32": (np.dtype("<i4"), 4),
    "I64": (np.dtype("<i8"), 8),
    "F64": (np.dtype("<f8"), 8),
    "BOOL": (np.dtype("u1"), 1),
}


def bf16_to_f32(raw: bytes) -> np.ndarray:
    """BF16 bytes -> float32 array. Exact (bf16 is the high half of f32)."""
    u16 = np.frombuffer(raw, dtype="<u2")
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def decode(raw: bytes, dtype: str, shape) -> np.ndarray:
    """Raw bytes -> float32 numpy array of `shape`."""
    if dtype == "BF16":
        a = bf16_to_f32(raw)
    elif dtype == "F16":
        a = np.frombuffer(raw, dtype="<f2").astype(np.float32)
    elif dtype == "F32":
        a = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    elif dtype in ("I8", "U8", "BOOL"):
        a = np.frombuffer(raw, dtype="u1" if dtype == "BOOL" else dtype.lower()).astype(np.float32)
    elif dtype == "I32":
        a = np.frombuffer(raw, dtype="<i4").astype(np.float32)
    elif dtype == "I64":
        a = np.frombuffer(raw, dtype="<i8").astype(np.float32)
    elif dtype == "F64":
        a = np.frombuffer(raw, dtype="<f8").astype(np.float32)
    else:
        raise ValueError("unsupported dtype: %s" % dtype)
    n = int(np.prod(shape)) if shape else 1
    if a.size != n:
        raise ValueError(
            "size mismatch for dtype %s shape %s: got %d elems, want %d"
            % (dtype, list(shape), a.size, n)
        )
    return a.reshape(shape)


class SafeTensors(object):
    """Reader for a single .safetensors file."""

    def __init__(self, path):
        self.path = path
        self.f = open(path, "rb")
        head = self.f.read(8)
        if len(head) != 8:
            raise ValueError("not a safetensors file (too short): %s" % path)
        (n,) = struct.unpack("<Q", head)
        # Guard: a bogus length would otherwise be read as a huge string,
        # or silently succeed on a truncated file.
        fsz = os.path.getsize(path)
        if n <= 0 or 8 + n > fsz:
            raise ValueError(
                "bad header length %d (file is %d bytes) -- not safetensors?" % (n, fsz)
            )
        self.header = json.loads(self.f.read(n).decode("utf-8"))
        self.data_off = 8 + n
        self._meta = {k: v for k, v in self.header.items() if k != "__metadata__"}

    # ---------------------------------------------------------------- queries
    def names(self):
        return sorted(self._meta.keys())

    def meta(self, name):
        if name not in self._meta:
            raise KeyError(name)
        return self._meta[name]

    def has(self, name):
        return name in self._meta

    def raw_bytes(self, name):
        m = self.meta(name)
        beg, end = m["data_offsets"]
        # Guard against a corrupt header pointing outside the file: a bad
        # offset otherwise returns short data and reshapes it silently.
        if beg < 0 or end < beg or self.data_off + end > os.path.getsize(self.path):
            raise ValueError(
                "tensor %s data_offsets %s outside file (data starts at %d, file %d)"
                % (name, [beg, end], self.data_off, os.path.getsize(self.path))
            )
        self.f.seek(self.data_off + beg)
        buf = self.f.read(end - beg)
        if len(buf) != end - beg:
            raise ValueError("short read for %s: %d != %d" % (name, len(buf), end - beg))
        return buf

    def get_raw(self, name):
        m = self.meta(name)
        return m["dtype"], m["shape"], self.raw_bytes(name)

    def get(self, name):
        """-> float32 numpy array (dequantized from whatever dtype)."""
        dt, shape, buf = self.get_raw(name)
        return decode(buf, dt, shape)

    # ------------------------------------------------------------ diagnostics
    def summary(self):
        tot = 0
        by_dtype = {}
        for name in self._meta:
            m = self._meta[name]
            beg, end = m["data_offsets"]
            tot += end - beg
            by_dtype[m["dtype"]] = by_dtype.get(m["dtype"], 0) + (end - beg)
        return {
            "n_tensors": len(self._meta),
            "total_bytes": tot,
            "by_dtype": by_dtype,
            "header_bytes": self.data_off,
        }


# ------------------------------------------------------------------- selftest
def _selftest():
    """Round-trip a synthetic file: catches offset/endianness mistakes."""
    import tempfile

    a = (np.arange(12, dtype=np.float32).reshape(3, 4) - 5.0)
    b = np.array([1.5, -2.25], dtype=np.float32)
    # encode a as bf16 by truncating the low 16 bits (exact for values like these)
    a_bf16 = (a.view(np.uint32) >> 16).astype("<u2")

    hdr = {}
    blob = b""
    for nm, arr_bytes, dt, shp in (
        ("a", a_bf16.tobytes(), "BF16", [3, 4]),
        ("b", b.astype("<f4").tobytes(), "F32", [2]),
    ):
        hdr[nm] = {"dtype": dt, "shape": shp, "data_offsets": [len(blob), len(blob) + len(arr_bytes)]}
        blob += arr_bytes
    js = json.dumps(hdr).encode("utf-8")

    fd, p = tempfile.mkstemp(suffix=".safetensors")
    os.close(fd)
    try:
        with open(p, "wb") as fh:
            fh.write(struct.pack("<Q", len(js)))
            fh.write(js)
            fh.write(blob)
        st = SafeTensors(p)
        assert st.names() == ["a", "b"], st.names()
        ga = st.get("a")
        assert ga.shape == (3, 4)
        # round-trip exactness: bf16 truncation of these small ints is exact
        assert np.array_equal(ga, a), (ga, a)
        gb = st.get("b")
        assert np.allclose(gb, b), (gb, b)
        print("[selftest] OK: %d tensors, header %d B, dtype mix %s"
              % (st.summary()["n_tensors"], st.summary()["header_bytes"],
                 sorted(st.summary()["by_dtype"])))
        st.f.close()
    finally:
        # Windows keeps the file locked while the handle is open; close first.
        try:
            st.f.close()
        except Exception:
            pass
        try:
            os.unlink(p)
        except OSError:
            pass


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] != "-":
        st = SafeTensors(sys.argv[1])
        s = st.summary()
        print("file      : %s" % sys.argv[1])
        print("tensors   : %d" % s["n_tensors"])
        print("header    : %d B (%.4f%% of file)"
              % (s["header_bytes"], 100.0 * s["header_bytes"] / os.path.getsize(sys.argv[1])))
        print("data      : %.3f GB" % (s["total_bytes"] / 1e9))
        print("by dtype  : %s" % {k: "%.3f GB" % (v / 1e9) for k, v in sorted(s["by_dtype"].items())})
        print()
        print("first 15 names:")
        for nm in st.names()[:15]:
            m = st.meta(nm)
            print("  %-60s %-5s %s" % (nm, m["dtype"], m["shape"]))
    else:
        _selftest()
