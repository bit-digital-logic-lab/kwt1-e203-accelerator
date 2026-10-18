#!/usr/bin/env python3
"""Frozen PC -> RCL byte-transport ABI for the KWT-1 FFN1 first linear layer.

This module is the *only* place that defines how PC-side integer operands for
``Y = X W + b`` (``99x64 @ 64x256 + 256 -> 99x256``) are laid out as bytes in
the accelerator's external/shared memory, and how the 32-byte NICE-like tile
descriptor is encoded.

Evidence base (see ``README.md`` for file/line pointers):

* ``hardware/rtl/dma/kwt_dma.sv``            - 128-bit little-endian bus, 8x16-bit lanes/beat
* ``hardware/rtl/sram/kwt_memory_defs.svh``  - 16-bit x 1024 SRAM, 512 KiB external memory
* ``hardware/rtl/pulse_array/kwt_pingpong_cacc_engine.sv`` - A[k][r], B[k][c] addressing
* ``hardware/rtl/control/kwt_tile_system_model.sv`` - per-tile DMA word counts
* ``hardware/rtl/control/kwt_nice_system_model.sv`` - 2-beat descriptor fetch fields
* ``hardware/rtl/control/tb/prepare_real_e203_vectors.py`` - real a16/w16 hex layout
* ``software/driver/kwt_nice_driver.{h,c}``  - descriptor struct and tile strides

Nothing here is a SoC pin/register map. The register-level ABI of the official
competition SoC is not present in this repository, so this ABI deliberately
stops at (a) the external-memory byte stream and (b) the descriptor the RTL
models actually consume.

Two format versions are defined:

* ``V1`` is the frozen/observed format: raw tiled INT16 A/W, INT32 bias/Y, no
  header, no scale, no checksum. It matches the current RTL and driver exactly.
* ``V2`` is an explicit, versioned *candidate* container that wraps the same
  frozen payload tilings with magic/version/dtype/scale/CRC metadata, plus an
  optional tightly packed W10 section. V2 is not implemented in the RTL; it is
  a proposal, clearly separated from V1.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------
# Frozen V1 constants -- all derived from the RTL, not invented.
# --------------------------------------------------------------------------

BUS_BYTES = 16                 # kwt_memory_defs.svh KWT_DMA_BUS_W = 128
BUS_LANES = BUS_BYTES // 2     # 8 x 16-bit lanes per bus beat
EXT_MEM_BYTES = 512 * 1024     # KWT_EXT_MEM_BYTES
SRAM_DATA_W = 16               # KWT_SRAM_DATA_W
SRAM_DEPTH = 1024              # KWT_SRAM_DEPTH
BANK_A, BANK_B, BANK_OUT = 0, 1, 2   # writeback only allowed on BANK_OUT

M, K, N = 99, 64, 256          # FFN1 problem shape
ROW_TILE, COL_TILE = 8, 16     # compute array ROWS x COLS
ROW_TILES = (M + ROW_TILE - 1) // ROW_TILE   # 13
COL_TILES = (N + COL_TILE - 1) // COL_TILE   # 16
TILES = ROW_TILES * COL_TILES                 # 208

A_TILE_BYTES = ROW_TILE * K * 2    # 1024
W_TILE_BYTES = K * COL_TILE * 2    # 2048
BIAS_TILE_BYTES = COL_TILE * 4     # 64
OUT_TILE_BYTES = ROW_TILE * COL_TILE * 4  # 512

A_BYTES = ROW_TILES * A_TILE_BYTES       # 13312
W_BYTES = COL_TILES * W_TILE_BYTES       # 32768
BIAS_BYTES = N * 4                        # 1024
OUT_BYTES = TILES * OUT_TILE_BYTES        # 106496

A12_QMAX = 2047   # signed 12-bit activation container
W10_QMAX = 511    # signed 10-bit weight container

DESCRIPTOR_BYTES = 32

# V1 has no header at all; the byte stream is exactly the tiled payload.
V1 = 1
V2 = 2


def _i16(values) -> np.ndarray:
    return np.asarray(values).astype("<i2", copy=False)


def _i32(values) -> np.ndarray:
    return np.asarray(values).astype("<i4", copy=False)


# --------------------------------------------------------------------------
# V1 payload pack / unpack.  Round-trip is byte-exact by construction.
# --------------------------------------------------------------------------

def pack_a(qx: np.ndarray) -> bytes:
    """A[M,K] signed 16-bit -> 13 row-tiles of [k][r], row-major in tile."""
    qx = np.asarray(qx)
    if qx.shape != (M, K):
        raise ValueError(f"A must be {(M, K)}, got {qx.shape}")
    if qx.dtype.kind not in "iu":
        raise ValueError("A must be integer")
    out = np.zeros((ROW_TILES, K, ROW_TILE), dtype="<i2")
    for rt in range(ROW_TILES):
        rows = min(ROW_TILE, M - rt * ROW_TILE)
        out[rt, :, :rows] = qx[rt * ROW_TILE:rt * ROW_TILE + rows].T
    return out.tobytes()


def unpack_a(blob: bytes) -> np.ndarray:
    if len(blob) != A_BYTES:
        raise ValueError(f"A blob must be {A_BYTES} bytes, got {len(blob)}")
    tiles = np.frombuffer(blob, dtype="<i2").reshape(ROW_TILES, K, ROW_TILE)
    qx = np.zeros((M, K), dtype=np.int16)
    for rt in range(ROW_TILES):
        rows = min(ROW_TILE, M - rt * ROW_TILE)
        qx[rt * ROW_TILE:rt * ROW_TILE + rows] = tiles[rt, :, :rows].T
    return qx


def pack_w(qw: np.ndarray) -> bytes:
    """W[K,N] signed 16-bit -> 16 col-tiles of [k][c], little-endian lanes."""
    qw = np.asarray(qw)
    if qw.shape != (K, N):
        raise ValueError(f"W must be {(K, N)}, got {qw.shape}")
    if qw.dtype.kind not in "iu":
        raise ValueError("W must be integer")
    out = np.zeros((COL_TILES, K, COL_TILE), dtype="<i2")
    for ct in range(COL_TILES):
        out[ct] = qw[:, ct * COL_TILE:(ct + 1) * COL_TILE]
    return out.tobytes()


def unpack_w(blob: bytes) -> np.ndarray:
    if len(blob) != W_BYTES:
        raise ValueError(f"W blob must be {W_BYTES} bytes, got {len(blob)}")
    tiles = np.frombuffer(blob, dtype="<i2").reshape(COL_TILES, K, COL_TILE)
    qw = np.empty((K, N), dtype=np.int16)
    for ct in range(COL_TILES):
        qw[:, ct * COL_TILE:(ct + 1) * COL_TILE] = tiles[ct]
    return qw


def pack_bias(qb: np.ndarray) -> bytes:
    qb = np.asarray(qb)
    if qb.shape != (N,):
        raise ValueError(f"bias must be {(N,)}, got {qb.shape}")
    if qb.dtype.kind not in "iu":
        raise ValueError("bias must be integer")
    return _i32(qb).tobytes()


def unpack_bias(blob: bytes) -> np.ndarray:
    if len(blob) != BIAS_BYTES:
        raise ValueError(f"bias blob must be {BIAS_BYTES} bytes, got {len(blob)}")
    return np.frombuffer(blob, dtype="<i4").copy()


def pack_out(y: np.ndarray) -> bytes:
    """Y[M,N] signed 32-bit -> 208 tiles, row-major [r][c] inside each tile."""
    y = np.asarray(y)
    if y.shape != (M, N):
        raise ValueError(f"Y must be {(M, N)}, got {y.shape}")
    if y.dtype.kind not in "iu":
        raise ValueError("Y must be integer")
    out = np.zeros((TILES, ROW_TILE, COL_TILE), dtype="<i4")
    for rt in range(ROW_TILES):
        rows = min(ROW_TILE, M - rt * ROW_TILE)
        for ct in range(COL_TILES):
            tile = rt * COL_TILES + ct
            out[tile, :rows] = y[rt * ROW_TILE:rt * ROW_TILE + rows,
                                 ct * COL_TILE:(ct + 1) * COL_TILE]
    return out.tobytes()


def unpack_out(blob: bytes) -> np.ndarray:
    if len(blob) != OUT_BYTES:
        raise ValueError(f"Y blob must be {OUT_BYTES} bytes, got {len(blob)}")
    tiles = np.frombuffer(blob, dtype="<i4").reshape(TILES, ROW_TILE, COL_TILE)
    y = np.zeros((M, N), dtype=np.int32)
    for rt in range(ROW_TILES):
        rows = min(ROW_TILE, M - rt * ROW_TILE)
        for ct in range(COL_TILES):
            tile = rt * COL_TILES + ct
            y[rt * ROW_TILE:rt * ROW_TILE + rows,
              ct * COL_TILE:(ct + 1) * COL_TILE] = tiles[tile, :rows]
    return y


# --------------------------------------------------------------------------
# Tile descriptors (V1).  Field positions match kwt_nice_system_model.sv.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class TileDescriptor:
    a_addr: int
    b_addr: int      # = weight address in the driver struct
    bias_addr: int
    out_addr: int
    k_len: int = K
    relu_en: bool = False
    row_mask: int = 0xFF
    col_mask: int = 0xFFFF

    def pack(self) -> bytes:
        for name, value in (("a_addr", self.a_addr), ("b_addr", self.b_addr),
                            ("bias_addr", self.bias_addr), ("out_addr", self.out_addr)):
            if value & 0xF:
                raise ValueError(f"{name} must be 16-byte aligned: {value:#x}")
        if not 1 <= self.k_len <= 64:
            raise ValueError("k_len must be in 1..64")
        if not 0 <= self.row_mask <= 0xFF:
            raise ValueError("row_mask must be 8 bits")
        if not 0 <= self.col_mask <= 0xFFFF:
            raise ValueError("col_mask must be 16 bits")
        control = self.k_len | (0x80 if self.relu_en else 0)
        beat1 = struct.pack("<I", control) + bytes(
            (self.row_mask, self.col_mask & 0xFF, (self.col_mask >> 8) & 0xFF, 0)
        ) + b"\x00" * 8
        return struct.pack("<IIII", self.a_addr, self.b_addr,
                           self.bias_addr, self.out_addr) + beat1

    @classmethod
    def unpack(cls, blob: bytes) -> "TileDescriptor":
        if len(blob) != DESCRIPTOR_BYTES:
            raise ValueError(f"descriptor must be {DESCRIPTOR_BYTES} bytes")
        a, b, bias, out = struct.unpack("<IIII", blob[:16])
        control = blob[16]
        return cls(a, b, bias, out, k_len=control & 0x7F,
                   relu_en=bool(control & 0x80), row_mask=blob[20],
                   col_mask=blob[21] | (blob[22] << 8))


def descriptor_tiles(a_base: int, w_base: int, bias_base: int, out_base: int):
    """Yield the 208 (rt, ct) descriptors exactly as the driver prepares them."""
    for rt in range(ROW_TILES):
        row_mask = 0x07 if rt == ROW_TILES - 1 else 0xFF
        for ct in range(COL_TILES):
            yield rt, ct, TileDescriptor(
                a_addr=a_base + rt * A_TILE_BYTES,
                b_addr=w_base + ct * W_TILE_BYTES,
                bias_addr=bias_base + ct * BIAS_TILE_BYTES,
                out_addr=out_base + (rt * COL_TILES + ct) * OUT_TILE_BYTES,
                row_mask=row_mask, col_mask=0xFFFF)


# --------------------------------------------------------------------------
# Address / alignment preflight (mirrors kwt_dma.sv and driver checks).
# --------------------------------------------------------------------------

def check_region(name: str, base: int, length: int, limit: int = EXT_MEM_BYTES):
    if base % BUS_BYTES:
        raise ValueError(f"{name} base {base:#x} not {BUS_BYTES}-byte aligned")
    if base < 0 or base + length > limit:
        raise ValueError(f"{name} region [{base:#x},{base + length:#x}) outside 0..{limit:#x}")
    return True


def check_layout_bounds(a_base, w_base, bias_base, out_base, out_base_next=None):
    """All tile bases/sizes must be 16-byte aligned and inside the 512 KiB model."""
    for rt in range(ROW_TILES):
        check_region(f"A tile {rt}", a_base + rt * A_TILE_BYTES, A_TILE_BYTES)
    for ct in range(COL_TILES):
        check_region(f"W tile {ct}", w_base + ct * W_TILE_BYTES, W_TILE_BYTES)
        check_region(f"bias tile {ct}", bias_base + ct * BIAS_TILE_BYTES, BIAS_TILE_BYTES)
    end = out_base + OUT_BYTES
    if out_base % BUS_BYTES:
        raise ValueError("out base not aligned")
    if end > EXT_MEM_BYTES:
        raise ValueError(f"out region ends at {end:#x} beyond {EXT_MEM_BYTES:#x}")
    return end


# --------------------------------------------------------------------------
# V2 candidate container: same frozen payload tilings, self-describing header.
# --------------------------------------------------------------------------

V2_MAGIC = b"KWT2"
V2_HEADER_BYTES = 64
V2_HEADER_FIELDS_BYTES = 36
V2_SECTION_BYTES = 48
V2_SECTION_FIELDS_BYTES = 40
# section dtype codes
DT_INT16 = 0
DT_INT32 = 1
DT_W10_PACKED = 2
DT_FLOAT64 = 3


def _w10_pack(qw: np.ndarray) -> bytes:
    """Signed 10-bit two's-complement packing, LSB-first, tightly packed."""
    vals = np.asarray(qw, dtype=np.int64).reshape(-1)
    if np.any(vals < -W10_QMAX) or np.any(vals > W10_QMAX):
        raise ValueError("weight outside W10 range for packing")
    out = bytearray()
    acc = 0
    nbits = 0
    for value in vals:
        acc |= (int(value) & 0x3FF) << nbits
        nbits += 10
        while nbits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            nbits -= 8
    if nbits:
        out.append(acc & 0xFF)
    return bytes(out)


def _w10_unpack(blob: bytes, count: int) -> np.ndarray:
    values = []
    acc = 0
    nbits = 0
    index = 0
    for byte in blob:
        acc |= byte << nbits
        nbits += 8
        while nbits >= 10 and len(values) < count:
            raw = acc & 0x3FF
            values.append(raw - 1024 if raw & 0x200 else raw)
            acc >>= 10
            nbits -= 10
    if len(values) != count:
        raise ValueError("packed W10 section too short")
    return np.asarray(values, dtype=np.int16)


@dataclass
class StreamV2:
    layer: int
    a: bytes
    w: bytes
    bias: bytes
    activation_scale: float
    weight_scales: np.ndarray
    w10_packed: bytes | None = None

    def pack(self) -> bytes:
        sections = [("A", DT_INT16, M * K, self.a),
                    ("W", DT_INT16, K * N, self.w),
                    ("BIAS", DT_INT32, N, self.bias),
                    ("SCALEX", DT_FLOAT64, 1, struct.pack("<d", self.activation_scale)),
                    ("SCALEW", DT_FLOAT64, N, np.asarray(self.weight_scales, "<f8").tobytes())]
        if self.w10_packed is not None:
            sections.append(("W10", DT_W10_PACKED, K * N, self.w10_packed))
        table = bytearray()
        payload = bytearray()
        offset = V2_HEADER_BYTES + V2_SECTION_BYTES * len(sections)
        for name, dtype, count, data in sections:
            entry = struct.pack("<8sBBHQQQI", name.encode("ascii"), dtype, 0, 0,
                                count, offset, len(data), zlib.crc32(data))
            table += entry.ljust(V2_SECTION_BYTES, b"\x00")
            payload += data
            offset += len(data)
        total = V2_HEADER_BYTES + len(table) + len(payload)
        body = bytes(table) + bytes(payload)
        header = struct.pack(
            "<4sHHIHHIIQI", V2_MAGIC, V2, V2_HEADER_BYTES, 0x3, self.layer,
            len(sections), 0, zlib.crc32(body), total, V2_HEADER_BYTES)
        header = header.ljust(V2_HEADER_BYTES, b"\x00")
        return header + body

    @classmethod
    def unpack(cls, blob: bytes) -> "StreamV2":
        if len(blob) < V2_HEADER_BYTES:
            raise ValueError("v2 stream shorter than header")
        magic, version, header_bytes, _flags, layer, nsec, _res, crc, total, table_off = \
            struct.unpack("<4sHHIHHIIQI", blob[:V2_HEADER_FIELDS_BYTES])
        if magic != V2_MAGIC:
            raise ValueError("bad v2 magic")
        if version != V2:
            raise ValueError(f"unsupported v2 version {version}")
        if total != len(blob):
            raise ValueError(f"v2 length mismatch: header {total} vs actual {len(blob)}")
        body = blob[header_bytes:]
        if zlib.crc32(body) != crc:
            raise ValueError("v2 body CRC mismatch")
        sections = {}
        for i in range(nsec):
            base = table_off + i * V2_SECTION_BYTES
            name, dtype, _r0, _r1, count, off, length, scrc = \
                struct.unpack("<8sBBHQQQI",
                              blob[base:base + V2_SECTION_FIELDS_BYTES])
            data = blob[off:off + length]
            if len(data) != length:
                raise ValueError(f"section {name!r} truncated")
            if zlib.crc32(data) != scrc:
                raise ValueError(f"section {name!r} CRC mismatch")
            sections[name.decode("ascii").rstrip("\x00")] = (dtype, count, data)
        sx = struct.unpack("<d", sections["SCALEX"][2])[0]
        sw = np.frombuffer(sections["SCALEW"][2], dtype="<f8").copy()
        w10 = sections.get("W10")
        return cls(layer=layer, a=sections["A"][2], w=sections["W"][2],
                   bias=sections["BIAS"][2], activation_scale=sx, weight_scales=sw,
                   w10_packed=None if w10 is None else w10[2])


# --------------------------------------------------------------------------
# Convenience: quantize with the frozen A12W10 rule (ties-to-even, clip).
# --------------------------------------------------------------------------

def quantize_activation(x: np.ndarray, sx: float) -> np.ndarray:
    q = np.rint(np.asarray(x, dtype=np.float64) / sx)
    clipped = int(np.count_nonzero(np.abs(q) > A12_QMAX))
    q = np.clip(q, -A12_QMAX, A12_QMAX).astype(np.int16)
    return q, clipped


def sha256_bytes(blob: bytes) -> str:
    import hashlib
    return hashlib.sha256(blob).hexdigest()
