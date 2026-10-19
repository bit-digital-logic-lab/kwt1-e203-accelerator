#!/usr/bin/env python3
"""KWT1-PACK v1 — reference byte packer / unpacker for the KWT-1 accelerator.

This module defines and implements the byte-level layout used to move
(a) the 12 encoder layers' parameters and (b) the 99x64 layer activations
between the E203/SoC memory and the accelerator.

Container rule
--------------
Every multi-byte scalar is stored **little-endian (LE)**.  This matches the
RISC-V/E203 byte order and the frozen design decision
(`docs/KWT1_整体设计决策_20261018.md` §数值协议: "字节序小端").  A big-endian
decode is deliberately *not* accepted: it is used as a negative control.

Alignment rule
--------------
* blob header        : 64 bytes, 16-byte aligned at offset 0
* segment directory  : 48-byte entries, padded to a 16-byte boundary
* payload / segments : each segment starts at a 16-byte boundary
* element alignment  : int16 @ +0..1, int32/fp32 @ 4-byte aligned,
                       packed bit streams @ byte aligned
The 16-byte payload alignment matches the `SUBMIT` descriptor byte address
contract already frozen for the E203 path.

Streams (dtype)
---------------
  INT16   : 16-bit signed two's complement, 2 bytes/value, LE
  INT32   : 32-bit signed two's complement, 4 bytes/value, LE
  FP32    : IEEE-754 binary32, 4 bytes/value, LE
  PACK10  : signed 10-bit two's complement, packed 4 values / 5 bytes
  PACK12  : signed 12-bit two's complement, packed 2 values / 3 bytes

Packed bit stream definition: value *i* occupies stream bits
[bits*i, bits*i+bits).  Stream bit *j* is bit (j % 8) of byte (j // 8), i.e.
**LSB-first** bit order inside each byte (same "little-endian" spirit as the
word order).  Negative values use two's complement in the low `bits` bits.

Value order
-----------
Row-major (C order) for every tensor.  Shape is carried in the directory, so
the decoder restores the exact same index mapping the integer reference model
uses (e.g. FFN1 W1 index `c*D + j`).

This file is independent of the RTL; the RTL byte-lane is compared against the
byte streams this module produces (see `rtl/`).
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field

import numpy as np

MAGIC = b"KWTP"
VERSION = 1
HEADER_BYTES = 64
DIRENT_BYTES = 48
ALIGN = 16
NAME_BYTES = 16

DT_INT16, DT_INT32, DT_FP32, DT_PACK10, DT_PACK12 = 0, 1, 2, 3, 4
DT_NAMES = {
    DT_INT16: "int16", DT_INT32: "int32", DT_FP32: "fp32",
    DT_PACK10: "pack10", DT_PACK12: "pack12",
}
DT_BITS = {DT_INT16: 16, DT_INT32: 32, DT_FP32: 32, DT_PACK10: 10, DT_PACK12: 12}


# --------------------------------------------------------------------------- #
# bit-level helpers (signed two's complement, LSB-first stream)
# --------------------------------------------------------------------------- #
def pack_bits(values, bits: int) -> bytes:
    out = bytearray()
    acc = 0
    n = 0
    mask = (1 << bits) - 1
    for v in values:
        acc |= (int(v) & mask) << n
        n += bits
        while n >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            n -= 8
    if n:
        out.append(acc & 0xFF)
    return bytes(out)


def unpack_bits(buf, bits: int, count: int):
    acc = 0
    n = 0
    i = 0
    mask = (1 << bits) - 1
    sign = 1 << (bits - 1)
    out = []
    for _ in range(count):
        while n < bits:
            if i >= len(buf):
                raise ValueError("bit stream truncated")
            acc |= buf[i] << n
            i += 1
            n += 8
        u = acc & mask
        acc >>= bits
        n -= bits
        out.append(u - (1 << bits) if (u & sign) else u)
    return out


def _le_bytes(values, width_bytes: int) -> bytes:
    """Vectorised little-endian encode of signed/unsigned integers."""
    a = np.asarray(values)
    return a.astype(f"<i{width_bytes}").tobytes()


def _le_from_bytes(buf, width_bytes: int, dtype_sign=True):
    dt = f"<i{width_bytes}" if dtype_sign else f"<u{width_bytes}"
    return np.frombuffer(buf, dtype=dt).astype(np.int64)


# --------------------------------------------------------------------------- #
# segment description
# --------------------------------------------------------------------------- #
@dataclass
class Segment:
    name: str
    dtype: int
    shape: tuple
    data: np.ndarray = field(default=None)

    @property
    def nelem(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return int(n)

    def encode(self) -> bytes:
        if self.dtype == DT_INT16:
            return _le_bytes(self.data, 2)
        if self.dtype == DT_INT32:
            return _le_bytes(self.data, 4)
        if self.dtype == DT_FP32:
            return np.asarray(self.data, dtype="<f4").tobytes()
        if self.dtype == DT_PACK10:
            return pack_bits(np.asarray(self.data).ravel(), 10)
        if self.dtype == DT_PACK12:
            return pack_bits(np.asarray(self.data).ravel(), 12)
        raise ValueError(f"unknown dtype {self.dtype}")

    def byte_len(self) -> int:
        bits = DT_BITS[self.dtype]
        return (self.nelem * bits + 7) // 8


def encode_segment(dtype: int, shape, data) -> bytes:
    return Segment(name="", dtype=dtype, shape=tuple(shape), data=data).encode()


def decode_segment(dtype: int, shape, payload: bytes, swap_endian: bool = False):
    n = 1
    for d in shape:
        n *= int(d)
    if dtype in (DT_INT16, DT_INT32):
        w = 2 if dtype == DT_INT16 else 4
        need = n * w
        if len(payload) < need:
            raise ValueError("segment truncated")
        if swap_endian:
            a = np.frombuffer(payload[:need], dtype=f">i{w}").astype(np.int64)
        else:
            a = _le_from_bytes(payload[:need], w)
        return a.reshape(shape)
    if dtype == DT_FP32:
        b = payload[: n * 4]
        if len(b) < n * 4:
            raise ValueError("segment truncated")
        return np.frombuffer(b, dtype=("<f4" if not swap_endian else ">f4")).copy().reshape(shape)
    if dtype == DT_PACK10:
        vals = unpack_bits(payload, 10, n)
        return np.asarray(vals, dtype=np.int64).reshape(shape)
    if dtype == DT_PACK12:
        vals = unpack_bits(payload, 12, n)
        return np.asarray(vals, dtype=np.int64).reshape(shape)
    raise ValueError(f"unknown dtype {dtype}")


# --------------------------------------------------------------------------- #
# blob pack / unpack / validate
# --------------------------------------------------------------------------- #
def _dirent(name, dtype, shape, offset, nbytes) -> bytes:
    nm = name.encode("ascii")[:NAME_BYTES].ljust(NAME_BYTES, b"\x00")
    dims = [int(d) for d in shape] + [0, 0, 0]
    return struct.pack(
        "<16sBBBBIIIQII",
        nm, dtype, DT_BITS[dtype], len(shape), 0,
        dims[0], dims[1], dims[2], offset, nbytes, 0,
    )


def pack(segments) -> bytes:
    """segments: list of Segment (with data) -> bytes blob."""
    nseg = len(segments)
    dir_bytes = nseg * DIRENT_BYTES
    payload_start = HEADER_BYTES + dir_bytes
    payload_start = (payload_start + ALIGN - 1) // ALIGN * ALIGN

    payload = bytearray()
    offsets = []
    for seg in segments:
        raw = seg.encode()
        off = payload_start + len(payload)
        # each segment starts 16-byte aligned
        pad = (-len(payload)) % ALIGN
        payload += b"\x00" * pad
        off = payload_start + len(payload)
        offsets.append((off, len(raw)))
        payload += raw
    payload_bytes = len(payload)
    total = payload_start + payload_bytes

    out = bytearray(total)
    out[0:4] = MAGIC
    struct.pack_into("<HHI I I Q", out, 4,
                     VERSION, 0, nseg, HEADER_BYTES, payload_start, payload_bytes)
    pos = HEADER_BYTES
    for seg, (off, nb) in zip(segments, offsets):
        out[pos:pos + DIRENT_BYTES] = _dirent(seg.name, seg.dtype, seg.shape, off, nb)
        pos += DIRENT_BYTES
    out[payload_start:payload_start + payload_bytes] = payload
    return bytes(out)


@dataclass
class Dirent:
    name: str
    dtype: int
    rank: int
    shape: tuple
    offset: int
    nbytes: int


def read_directory(blob: bytes):
    if len(blob) < HEADER_BYTES:
        raise ValueError("blob shorter than header")
    if blob[0:4] != MAGIC:
        raise ValueError(f"bad magic {blob[0:4]!r}")
    version, flags, nseg, hbytes, payload_start, payload_bytes = struct.unpack_from(
        "<HHIIIQ", blob, 4)
    entries = []
    pos = HEADER_BYTES
    for _ in range(nseg):
        (nm, dtype, bits, rank, _r, d0, d1, d2, off, nb, _f) = struct.unpack_from(
            "<16sBBBBIIIQII", blob, pos)
        pos += DIRENT_BYTES
        name = nm.split(b"\x00", 1)[0].decode("ascii")
        if bits != DT_BITS.get(dtype, -1):
            raise ValueError(f"dir {name}: elem_bits {bits} inconsistent with dtype {dtype}")
        if off % ALIGN != 0:
            raise ValueError(f"dir {name}: offset {off} not {ALIGN}-byte aligned")
        if off + nb > len(blob):
            raise ValueError(f"dir {name}: segment exceeds blob")
        shape = tuple([d0, d1, d2][:rank])
        entries.append(Dirent(name, dtype, rank, shape, off, nb))
    return dict(version=version, flags=flags, nseg=nseg, header_bytes=hbytes,
                payload_start=payload_start, payload_bytes=payload_bytes), entries


def unpack(blob: bytes, swap_endian: bool = False):
    meta, entries = read_directory(blob)
    out = {}
    for e in entries:
        payload = blob[e.offset:e.offset + e.nbytes]
        out[e.name] = decode_segment(e.dtype, e.shape, payload, swap_endian=swap_endian)
    return meta, entries, out


def validate(blob: bytes) -> None:
    meta, entries = read_directory(blob)
    if meta["header_bytes"] != HEADER_BYTES:
        raise ValueError("unexpected header_bytes")
    if meta["payload_start"] % ALIGN != 0:
        raise ValueError("payload_start not aligned")
    if meta["nseg"] != len(entries):
        raise ValueError("nseg mismatch")
    if meta["payload_start"] + meta["payload_bytes"] != len(blob):
        raise ValueError("blob length != payload_start + payload_bytes")
    names = [e.name for e in entries]
    if len(set(names)) != len(names):
        raise ValueError("duplicate segment names")
    for e in entries:
        if e.offset + e.nbytes > len(blob):
            raise ValueError("segment outside blob")
    # header/directory must not overlap payload
    dir_end = HEADER_BYTES + meta["nseg"] * DIRENT_BYTES
    if dir_end > meta["payload_start"]:
        raise ValueError("directory overlaps payload")


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()
