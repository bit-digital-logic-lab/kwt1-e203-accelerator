#!/usr/bin/env python3
"""Emit RTL byte-lane vectors for the little-endian / bit-order RTL check.

Writes, into the build dir (default /tmp/opencode/r2_pack_format/rtl):
  lane_bytes16.hex        parameter-pack bytes (16-bit unpack test input)
  lane_words16.hex        expected sign-extended 16-bit words
  lane_bytes32.hex        parameter-pack bytes (32-bit unpack test input)
  lane_words32.hex        expected sign-extended 32-bit words
  pack12_bytes.hex        PACK12 stream of sample_00 activations
  pack12_vals.hex         expected signed 12-bit values

All data comes from the same frozen assets and the same Python codec used by
run_pack.py, so this file is only a *re-serialiser* for Verilator; it selects
no scale or threshold.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pack_format as pf  # noqa: E402
import run_pack as rp  # noqa: E402

N16_BYTES = 8192
N32_BYTES = 8192


def hex_bytes(path, data: bytes):
    path.write_text("".join(f"{b:02x}\n" for b in data))


def hex_words32(path, words):
    path.write_text("".join(f"{int(w) & 0xFFFFFFFF:08x}\n" for w in words))


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/opencode/r2_pack_format/rtl")
    out.mkdir(parents=True, exist_ok=True)

    layers = rp.load_manifest()
    blob = pf.pack(rp.build_param_segments(layers))

    b16 = blob[:N16_BYTES]
    hex_bytes(out / "lane_bytes16.hex", b16)
    hex_words32(out / "lane_words16.hex",
                np.frombuffer(b16, dtype="<i2").astype(np.int64))

    b32 = blob[:N32_BYTES]
    hex_bytes(out / "lane_bytes32.hex", b32)
    hex_words32(out / "lane_words32.hex",
                np.frombuffer(b32, dtype="<i4").astype(np.int64))

    samples = rp.load_samples()
    s0 = next(s for s in samples if s["file"] == "sample_00_encoder.npy")
    train_max = max(float(np.abs(s["x"]).max()) for s in samples if s["split"] == "training")
    scale = train_max / rp.ACT_QMAX
    q = rp.quant_sym(s0["x"], scale, rp.ACT_QMAX)
    p12 = pf.pack_bits(q.ravel(), 12)
    hex_bytes(out / "pack12_bytes.hex", p12)
    hex_words32(out / "pack12_vals.hex", q.ravel())

    print(f"EMIT_RTL_VECTORS ok dir={out}")
    print(f"  bytes16={len(b16)} words16={N16_BYTES//2}")
    print(f"  bytes32={len(b32)} words32={N32_BYTES//4}")
    print(f"  pack12_bytes={len(p12)} vals={q.size} max|q|={int(np.abs(q).max())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
