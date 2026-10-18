#!/usr/bin/env python3
"""Execute the PC -> RCL transport ABI check for one real FFN1-like layer.

Run:  python3 software/transport/run_transport_check.py

It performs, on real data present in this worktree:

* pack -> unpack -> byte equality for A/W/bias/Y in the frozen V1 layout;
* an independent INT64 GEMM reference (not derived from the packed bytes) and a
  per-tile INT32 accumulation over the *unpacked* bytes, compared bit-exactly;
* an FP64 decode vs the frozen FP32 weights (NRMSE), so the integer contract is
  checked against an independent float reference, not against itself;
* endianness, stride, alignment and 512 KiB address-boundary checks;
* a wrapping/overflow demonstration showing the transport does not saturate;
* negative cases: single-byte flip, truncation, and V2 CRC corruption;
* SHA-256 of every blob.

Exit code is non-zero if any assertion fails. Large blobs are written under
/tmp and are not committed; only hashes and counts land in results/.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transport_abi as abi  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = Path(__file__).resolve().parent / "results"
BLOB_DIR = Path("/tmp/opencode/transport_abi")

LAYER = 1
SAMPLE = ROOT / "software/validation/real_input_chain_20261017/sample_00_encoder.npy"
PARAMS = ROOT / "software/results/ffn1_a12w10_static_sym.npz"
METADATA = ROOT / "software/results/ffn1_a12w10_static_sym.json"
FP32_WEIGHTS = ROOT / "software/weights/kwt1_fp32.npz"
PREFIX = "layer_with_weights-0"
FFN = f"{PREFIX}/enc_layers/{LAYER - 1}/ffn/layer_with_weights-0"

checks: list[dict] = []
failures: list[str] = []


def record(name: str, ok: bool, detail: str) -> None:
    checks.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        failures.append(name)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    BLOB_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- frozen inputs ---------------------------------------------------
    x = np.load(SAMPLE).astype(np.float32)
    if x.shape != (abi.M, abi.K):
        raise SystemExit(f"real sample shape {x.shape} != {(abi.M, abi.K)}")
    with np.load(PARAMS, allow_pickle=False) as params:
        stem = f"layer{LAYER:02d}"
        qw = params[stem + "_weight"].astype(np.int16)
        qb = params[stem + "_bias"].astype(np.int32)
        sx = float(params[stem + "_activation_scale"])
        sw = params[stem + "_weight_scales"].astype(np.float64)
    meta = json.loads(METADATA.read_text())
    with np.load(FP32_WEIGHTS, allow_pickle=False) as fp32:
        w_ref = fp32[FFN + "/kernel"].astype(np.float64)
        b_ref = fp32[FFN + "/bias"].astype(np.float64)

    record("frozen_scheme", meta.get("scheme") == "A12W10", f"scheme={meta.get('scheme')}")
    record("parameter_sha256_matches_metadata",
           meta.get("parameter_sha256") == sha256(PARAMS),
           f"npz sha256={sha256(PARAMS)}")
    record("weight_sha256_matches_metadata",
           meta.get("weight_sha256") == sha256(FP32_WEIGHTS),
           f"fp32 sha256={sha256(FP32_WEIGHTS)}")
    record("frozen_weight_range",
           int(qw.min()) >= -abi.W10_QMAX and int(qw.max()) <= abi.W10_QMAX,
           f"W range=({int(qw.min())},{int(qw.max())}) within +-{abi.W10_QMAX}")

    # ---- quantize activation (ties-to-even, clip) and pack ---------------
    qx, clipped = abi.quantize_activation(x, sx)
    a_blob, w_blob, bias_blob = abi.pack_a(qx), abi.pack_w(qw), abi.pack_bias(qb)

    record("A_size", len(a_blob) == abi.A_BYTES,
           f"{len(a_blob)} bytes = 13x64x8x2")
    record("W_size", len(w_blob) == abi.W_BYTES,
           f"{len(w_blob)} bytes = 16x64x16x2")
    record("bias_size", len(bias_blob) == abi.BIAS_BYTES, f"{len(bias_blob)} bytes = 256x4")

    # endianness: lowest-address two bytes are the little-endian INT16 of A[0,0]
    le0 = struct_le_i16(int(qx[0, 0]))
    record("A_little_endian_first_element", a_blob[:2] == le0,
           f"blob[0:2]={a_blob[:2].hex()} le(int16 {int(qx[0,0])})={le0.hex()}")

    # ---- pack -> unpack -> byte equality --------------------------------
    record("A_roundtrip", np.array_equal(abi.unpack_a(a_blob), qx),
           f"max|diff|={int(np.max(np.abs(abi.unpack_a(a_blob).astype(np.int64) - qx)))}")
    record("W_roundtrip", np.array_equal(abi.unpack_w(w_blob), qw), "bit-identical")
    record("bias_roundtrip", np.array_equal(abi.unpack_bias(bias_blob), qb), "bit-identical")
    record("A_repack_stable", abi.pack_a(abi.unpack_a(a_blob)) == a_blob, "repack == original")

    # ---- independent numeric reference (INT64, from source arrays) ------
    ref64 = qx.astype(np.int64) @ qw.astype(np.int64) + qb.astype(np.int64)
    bound = abi.K * abi.A12_QMAX * abi.W10_QMAX + int(np.max(np.abs(qb.astype(np.int64))))
    record("conservative_bound_fits_int32", bound < (1 << 31),
           f"bound={bound} (metadata={meta['layers'][0]['conservative_abs_accumulator_bound']})")

    # per-tile INT32 accumulation over the unpacked bytes
    ua, uw, ub = abi.unpack_a(a_blob), abi.unpack_w(w_blob), abi.unpack_bias(bias_blob)
    y_tile = np.zeros((abi.M, abi.N), dtype=np.int32)
    for rt in range(abi.ROW_TILES):
        rows = min(abi.ROW_TILE, abi.M - rt * abi.ROW_TILE)
        for ct in range(abi.COL_TILES):
            a_tile = np.zeros((abi.ROW_TILE, abi.K), dtype=np.int32)
            a_tile[:rows] = ua[rt * abi.ROW_TILE:rt * abi.ROW_TILE + rows]
            acc = np.zeros((abi.ROW_TILE, abi.COL_TILE), dtype=np.int32)
            for k in range(abi.K):
                acc = acc + np.outer(a_tile[:, k],
                                     uw[k, ct * abi.COL_TILE:(ct + 1) * abi.COL_TILE]).astype(np.int32)
            acc = acc + ub[ct * abi.COL_TILE:(ct + 1) * abi.COL_TILE]
            y_tile[rt * abi.ROW_TILE:rt * abi.ROW_TILE + rows,
                   ct * abi.COL_TILE:(ct + 1) * abi.COL_TILE] = acc[:rows]
    record("tiled_int32_equals_independent_int64",
           np.array_equal(y_tile.astype(np.int64), ref64),
           f"max|diff|={int(np.max(np.abs(y_tile.astype(np.int64) - ref64)))}")

    # ---- output transport round-trip ------------------------------------
    y_blob = abi.pack_out(y_tile)
    record("Y_size", len(y_blob) == abi.OUT_BYTES, f"{len(y_blob)} bytes = 208x8x16x4")
    record("Y_roundtrip", np.array_equal(abi.unpack_out(y_blob), y_tile), "bit-identical")

    # ---- independent FP64 decode vs FP32 reference ----------------------
    decoded = (y_tile.astype(np.float64)) * (sx * sw)[None, :]
    reference = x.astype(np.float64) @ w_ref + b_ref
    err = float(np.sqrt(np.mean((decoded - reference) ** 2)))
    rms = float(np.sqrt(np.mean(reference ** 2)))
    nrmse = 100.0 * err / max(rms, 1e-8)
    record("decode_nrmse_below_3pct", nrmse <= 3.0,
           f"nrmse={nrmse:.6f}%  clipped_activations={clipped}")

    # ---- descriptor ABI --------------------------------------------------
    a_base, w_base, bias_base, out_base = 0x3000, 0x7000, 0xF000, 0x10000
    descs = list(abi.descriptor_tiles(a_base, w_base, bias_base, out_base))
    ok_desc = True
    first = descs[0][2].pack()
    for rt, ct, desc in descs:
        blob = desc.pack()
        if len(blob) != abi.DESCRIPTOR_BYTES or abi.TileDescriptor.unpack(blob) != desc:
            ok_desc = False
            break
    record("descriptor_208_roundtrip", ok_desc and len(descs) == abi.TILES,
           f"tiles={len(descs)} size={len(first)}")
    record("descriptor_first_matches_driver_contract",
           first[16] == abi.K and first[20] == 0xFF and first[21] == 0xFF and first[22] == 0xFF,
           f"control=0x{first[16]:02x} rowmask=0x{first[20]:02x} colmask=0x{first[21]:02x}{first[22]:02x}")
    record("last_row_tile_mask_is_7", descs[-1][2].row_mask == 0x07,
           f"rt={descs[-1][0]} row_mask={descs[-1][2].row_mask}")

    # address / alignment boundaries
    end = abi.check_layout_bounds(a_base, w_base, bias_base, out_base)
    record("layout_inside_512KiB", end <= abi.EXT_MEM_BYTES,
           f"out end={end:#x} <= {abi.EXT_MEM_BYTES:#x}")
    over_ok = raises(lambda: abi.check_layout_bounds(a_base, w_base, bias_base,
                                                     abi.EXT_MEM_BYTES - abi.OUT_BYTES + 16))
    record("out_overflow_rejected", over_ok, "out base past 512 KiB raises")
    mis_ok = raises(lambda: abi.TileDescriptor(0x3001, w_base, bias_base, out_base).pack())
    record("misaligned_descriptor_rejected", mis_ok, "a_addr=0x3001 raises")

    # ---- wrapping / no-saturation evidence ------------------------------
    wrap_src = np.array([np.iinfo(np.int32).max, np.iinfo(np.int32).min], dtype=np.int64)
    wrap_hw = wrap_src.astype(np.int32) + np.int32(1)   # 2's-complement wrap
    y_wrap = np.array([[wrap_hw[0]] * abi.N] * abi.M, dtype=np.int32)
    record("transport_is_twos_complement_not_saturating",
           int(wrap_hw[0]) == np.iinfo(np.int32).min and
           np.array_equal(abi.unpack_out(abi.pack_out(y_wrap)), y_wrap),
           f"INT32_MAX+1 wraps to {int(wrap_hw[0])}; container preserves it")

    # ---- negative cases --------------------------------------------------
    flat = bytearray(a_blob)
    flat[len(flat) // 2] ^= 0x01
    flipped = bytes(flat)
    diff = int(np.count_nonzero(abi.unpack_a(flipped) != qx))
    record("byte_flip_changes_sha_and_value",
           hashlib.sha256(flipped).hexdigest() != hashlib.sha256(a_blob).hexdigest() and diff > 0,
           f"differing elements={diff} sha changed=True")
    record("truncation_rejected", raises(lambda: abi.unpack_a(a_blob[:-1])),
           "A blob with one byte removed raises ValueError")

    # ---- V2 candidate container -----------------------------------------
    v2 = abi.StreamV2(LAYER, a_blob, w_blob, bias_blob, sx, sw,
                      w10_packed=abi._w10_pack(qw)).pack()
    back = abi.StreamV2.unpack(v2)
    record("v2_roundtrip_payload_and_scales",
           back.a == a_blob and back.w == w_blob and back.bias == bias_blob and
           back.activation_scale == sx and np.array_equal(back.weight_scales, sw),
           f"v2 bytes={len(v2)} (header {abi.V2_HEADER_BYTES}/section {abi.V2_SECTION_BYTES})")
    record("v2_packed_w10_exact",
           np.array_equal(abi._w10_unpack(back.w10_packed, abi.K * abi.N).reshape(qw.shape), qw),
           f"packed W10 bytes={len(back.w10_packed)} vs int16 {abi.W_BYTES}")
    corrupted = bytearray(v2)
    corrupted[-1] ^= 0xFF
    record("v2_crc_detects_corruption", raises(lambda: abi.StreamV2.unpack(bytes(corrupted))),
           "last payload byte flip raises CRC mismatch")

    # ---- hashes ----------------------------------------------------------
    for name, payload in (("A", a_blob), ("W", w_blob), ("BIAS", bias_blob), ("Y", y_blob)):
        (BLOB_DIR / f"layer{LAYER:02d}_{name}.bin").write_bytes(payload)
    hashes = {name: hashlib.sha256(blob).hexdigest() for name, blob in
              (("A", a_blob), ("W", w_blob), ("BIAS", bias_blob), ("Y", y_blob),
               ("V2", v2))}

    report = {
        "format": "KWT-1 PC->RCL transport ABI check v1",
        "layer": LAYER,
        "real_input": {"path": str(SAMPLE.relative_to(ROOT)), "shape": list(x.shape),
                       "sha256": sha256(SAMPLE)},
        "parameters": {"path": str(PARAMS.relative_to(ROOT)), "sha256": sha256(PARAMS),
                       "scheme": meta.get("scheme")},
        "fp32_weights_sha256": sha256(FP32_WEIGHTS),
        "quantization": {"activation_scale": sx, "clipped_activations": clipped,
                         "a_qmax": abi.A12_QMAX, "w_qmax": abi.W10_QMAX},
        "sizes": {"A": abi.A_BYTES, "W": abi.W_BYTES, "BIAS": abi.BIAS_BYTES,
                  "Y": abi.OUT_BYTES, "descriptor": abi.DESCRIPTOR_BYTES},
        "hashes_sha256": hashes,
        "nrmse_pct": nrmse,
        "conservative_abs_bound": bound,
        "boundary": ("real input is the committed 99x64 encoder input; the true "
                     "post-attention FFN1 activation needs the full FP32 forward "
                     "and is not recomputed here"),
        "checks": checks,
        "failures": failures,
    }
    (RESULT_DIR / "transport_abi_v1_layer01.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"\nblobs -> {BLOB_DIR}")
    print(f"report -> {RESULT_DIR / 'transport_abi_v1_layer01.json'}")
    print("RESULT", "PASS" if not failures else "FAIL", f"({len(checks)} checks)")
    return 0 if not failures else 1


def struct_le_i16(value: int) -> bytes:
    import struct
    return struct.pack("<h", value)


def raises(fn) -> bool:
    try:
        fn()
    except (ValueError, OverflowError):
        return True
    return False


if __name__ == "__main__":
    raise SystemExit(main())
