#!/usr/bin/env python3
"""r2_pack_format — build, roundtrip and audit the KWT1-PACK v1 byte format.

Runs entirely on frozen, already-quantised assets (read-only):
  * `software/results/ffn1_a12w10_static_sym.npz`  — 12 layers x (W1 int16, b1 int32)
  * `software/validation/real_input_chain_20261017/sample_0X_encoder.npy` — 99x64 X
  * `software/weights/kwt1_fp32.npz`                — full 12-layer FP32 parameters

Nothing is written outside this task directory and the build dir
`/tmp/opencode/r2_pack_format/`.

Calibration split rule: the activation scale is derived from the **training**
encoder samples only (gradient-free max-abs).  The **validation** samples are
used only to *evaluate* the resulting quantise/pack roundtrip; no scale,
threshold or format parameter is chosen from them.

Outputs: results.json, logs/*.log, build/*.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # repository root (…/ds-grad-r2_pack_format)
sys.path.insert(0, str(HERE))
import pack_format as pf  # noqa: E402

FFN1_NPZ = ROOT / "software/results/ffn1_a12w10_static_sym.npz"
REAL_DIR = ROOT / "software/validation/real_input_chain_20261017"
FP32_NPZ = ROOT / "software/weights/kwt1_fp32.npz"

T, D, HID, LAYERS = 99, 64, 256, 12
ACT_BITS = 12
W_BITS = 10
ACT_QMAX = (1 << (ACT_BITS - 1)) - 1  # 2047
W_QMAX = (1 << (W_BITS - 1)) - 1      # 511


def log(msg):
    print(msg, flush=True)


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


# --------------------------------------------------------------------------- #
def load_manifest():
    z = np.load(FFN1_NPZ)
    layers = []
    for L in range(1, LAYERS + 1):
        layers.append(dict(layer=L - 1,
                           weight=np.asarray(z[f"layer{L:02d}_weight"], dtype=np.int64),
                           bias=np.asarray(z[f"layer{L:02d}_bias"], dtype=np.int64),
                           act_scale=float(z[f"layer{L:02d}_activation_scale"]),
                           weight_scales=np.asarray(z[f"layer{L:02d}_weight_scales"], dtype=np.float64)))
    return layers


def load_samples():
    manifest = json.loads((REAL_DIR / "manifest.json").read_text())
    samples = []
    for s in manifest["samples"]:
        x = np.load(REAL_DIR / s["encoder_file"]).astype(np.float64)
        samples.append(dict(file=s["encoder_file"], split=s["split"], x=x))
    return samples


def quant_sym(x, scale, qmax):
    q = np.rint(x / scale).astype(np.int64)
    return np.clip(q, -qmax - 1, qmax)


def nrmse_pct(ref, test):
    ref = np.asarray(ref, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    denom = np.sqrt(np.mean(ref ** 2))
    if denom < 1e-8:
        denom = 1e-8
    return float(np.sqrt(np.mean((ref - test) ** 2)) / denom * 100.0)


# --------------------------------------------------------------------------- #
# parameter pack: 12 layers of frozen FFN1 integer W1/b1
# --------------------------------------------------------------------------- #
def build_param_segments(layers):
    segs = []
    for L in layers:
        segs.append(pf.Segment(f"L{L['layer']:02d}_W1", pf.DT_INT16, (D, HID), L["weight"]))
        segs.append(pf.Segment(f"L{L['layer']:02d}_b1", pf.DT_INT32, (HID,), L["bias"]))
    return segs


def roundtrip_param_pack(layers):
    segs = build_param_segments(layers)
    blob = pf.pack(segs)
    pf.validate(blob)
    meta, entries, got = pf.unpack(blob)
    mism = 0
    for L in layers:
        for suffix, ref in (("W1", L["weight"]), ("b1", L["bias"])):
            name = f"L{L['layer']:02d}_{suffix}"
            if not np.array_equal(got[name], ref):
                mism += 1
    return blob, meta, entries, got, mism


def tight_variant(layers, dtype):
    segs = []
    for L in layers:
        segs.append(pf.Segment(f"L{L['layer']:02d}_W1", dtype, (D, HID), L["weight"]))
        segs.append(pf.Segment(f"L{L['layer']:02d}_b1", pf.DT_INT32, (HID,), L["bias"]))
    blob = pf.pack(segs)
    pf.validate(blob)
    _, _, got = pf.unpack(blob)
    mism = sum(0 if np.array_equal(got[f"L{L['layer']:02d}_W1"], L["weight"]) else 1
               for L in layers)
    return blob, got, mism


# --------------------------------------------------------------------------- #
# activation pack: 99x64 per layer input (6 real samples)
# --------------------------------------------------------------------------- #
def build_act_segments(samples, scale):
    segs = []
    for i, s in enumerate(samples):
        q = quant_sym(s["x"], scale, ACT_QMAX)
        segs.append(pf.Segment(f"S{i:02d}_X", pf.DT_INT16, (T, D), q))
    return segs


def roundtrip_act_pack(samples, scale):
    segs = build_act_segments(samples, scale)
    blob = pf.pack(segs)
    pf.validate(blob)
    _, _, got = pf.unpack(blob)
    rows = []
    for i, s in enumerate(samples):
        q_ref = quant_sym(s["x"], scale, ACT_QMAX)
        exact = np.array_equal(got[f"S{i:02d}_X"], q_ref)
        deq = got[f"S{i:02d}_X"].astype(np.float64) * scale
        rows.append(dict(file=s["file"], split=s["split"], bit_exact=bool(exact),
                         quant_nrmse_pct=nrmse_pct(s["x"], deq),
                         max_abs=int(np.abs(q_ref).max())))
    return blob, rows


def tight_act_pack(samples, scale, dtype):
    segs = [pf.Segment(f"S{i:02d}_X", dtype, (T, D), quant_sym(s["x"], scale, ACT_QMAX))
            for i, s in enumerate(samples)]
    blob = pf.pack(segs)
    pf.validate(blob)
    _, _, got = pf.unpack(blob)
    mism = sum(0 if np.array_equal(got[f"S{i:02d}_X"], quant_sym(s["x"], scale, ACT_QMAX)) else 1
               for i, s in enumerate(samples))
    return blob, mism


# --------------------------------------------------------------------------- #
# full 12-layer FP32 parameter pack (format generality, exact bytes)
# --------------------------------------------------------------------------- #
def fp32_param_names():
    names = ["layer_with_weights-0/patch_proj/kernel", "layer_with_weights-0/patch_proj/bias",
             "layer_with_weights-0/class_emb", "layer_with_weights-0/pos_emb"]
    for L in range(LAYERS):
        p = f"layer_with_weights-0/enc_layers/{L}"
        names += [f"{p}/att/query_dense/kernel", f"{p}/att/key_dense/kernel",
                  f"{p}/att/value_dense/kernel", f"{p}/att/combine_heads/kernel",
                  f"{p}/att/combine_heads/bias",
                  f"{p}/ffn/layer_with_weights-0/kernel", f"{p}/ffn/layer_with_weights-0/bias",
                  f"{p}/ffn/layer_with_weights-1/kernel", f"{p}/ffn/layer_with_weights-1/bias",
                  f"{p}/layernorm1/gamma", f"{p}/layernorm1/beta",
                  f"{p}/layernorm2/gamma", f"{p}/layernorm2/beta"]
    names += ["layer_with_weights-1/layer_with_weights-0/kernel",
              "layer_with_weights-1/layer_with_weights-0/bias"]
    return names


def build_fp32_pack():
    z = np.load(FP32_NPZ)
    segs = []
    for i, n in enumerate(fp32_param_names()):
        a = np.asarray(z[n], dtype=np.float32)
        segs.append(pf.Segment(f"P{i:03d}", pf.DT_FP32, a.shape, a))
    blob = pf.pack(segs)
    pf.validate(blob)
    _, entries, got = pf.unpack(blob)
    mism = 0
    for i, n in enumerate(fp32_param_names()):
        if not np.array_equal(got[f"P{i:03d}"], np.asarray(z[n], dtype=np.float32)):
            mism += 1
    return blob, entries, mism, len(fp32_param_names())


# --------------------------------------------------------------------------- #
# structural / negative controls (the checker must FAIL these)
# --------------------------------------------------------------------------- #
def negative_controls(layers, samples, scale):
    """Two families of controls:
      * structural: pack/unpack/validate MUST raise
      * value-level: decode MUST produce a mismatch (checker not vacuous)
    A control "passes" only when the expected violation is observed.
    """
    segs = build_param_segments(layers)
    good = pf.pack(segs)
    ctrl = []

    def structural(name, fn):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            ctrl.append(dict(name=name, kind="structural", detected=True,
                             detail=type(e).__name__ + ": " + str(e)[:120]))
            log(f"  NEG-OK   {name}: {type(e).__name__}: {str(e)[:70]}")
        else:
            ctrl.append(dict(name=name, kind="structural", detected=False, detail="no error"))
            log(f"  NEG-FAIL {name}: no error raised (checker too weak!)")

    def valuelevel(name, fn):
        try:
            detected = bool(fn())
        except Exception as e:  # noqa: BLE001
            detected = False
            ctrl.append(dict(name=name, kind="value", detected=False,
                             detail=type(e).__name__ + ": " + str(e)[:120]))
            log(f"  NEG-FAIL {name}: unexpected {type(e).__name__}: {str(e)[:70]}")
            return
        ctrl.append(dict(name=name, kind="value", detected=detected,
                         detail="mismatch observed" if detected else "silent"))
        log(f"  {'NEG-OK  ' if detected else 'NEG-FAIL'} {name}: "
            f"{'mismatch observed' if detected else 'decode silently matched!'}")

    # --- structural ---
    structural("truncated_blob", lambda: pf.unpack(good[:-1]))
    structural("bad_magic", lambda: pf.read_directory(b"\x00\x00\x00\x00" + good[4:]))

    def misalign():
        b = bytearray(good)
        off_pos = pf.HEADER_BYTES + 16 + 4 + 12   # name + dtype/bits/rank/res + 3 dims
        orig = struct.unpack_from("<Q", b, off_pos)[0]
        struct.pack_into("<Q", b, off_pos, orig + 1)
        pf.unpack(bytes(b))
    structural("misaligned_offset", misalign)

    def dtype_conflict():
        b = bytearray(good)
        # corrupt elem_bits of the first dirent (offset HEADER_BYTES+1)
        b[pf.HEADER_BYTES + 17] = 12
        pf.read_directory(bytes(b))
    structural("dtype_bits_conflict", dtype_conflict)

    def fd_closed():
        b = bytearray(good)
        # nseg field -> huge, directory runs into payload
        struct.pack_into("<I", b, 8, 9999)
        pf.validate(bytes(b))
    structural("dirent_overlap", fd_closed)

    # --- value level ---
    def bad_endian():
        _, _, got = pf.unpack(good, swap_endian=True)
        mism = sum(0 if np.array_equal(got[f"L{L['layer']:02d}_W1"], L["weight"]) else 1
                   for L in layers)
        return mism > 0
    valuelevel("big_endian_decode_mismatch", bad_endian)

    def bit_flip():
        p10, _, _ = tight_variant(layers, pf.DT_PACK10)
        _, entries = pf.read_directory(p10)
        # flip one bit strictly inside the first W1 payload
        idx = entries[0].offset + entries[0].nbytes // 2
        b = bytearray(p10)
        b[idx] ^= 0x01
        _, _, got = pf.unpack(bytes(b))
        mism = sum(0 if np.array_equal(got[f"L{L['layer']:02d}_W1"], L["weight"]) else 1
                   for L in layers)
        return mism > 0
    valuelevel("pack10_bit_flip_mismatch", bit_flip)

    def width_confusion():
        segs12 = [pf.Segment("W", pf.DT_PACK12, (D, HID), layers[0]["weight"])]
        blob = pf.pack(segs12)
        _, e, _ = pf.unpack(blob)
        payload = blob[e[0].offset:e[0].offset + e[0].nbytes]
        wrong = pf.decode_segment(pf.DT_PACK10, (D, HID), payload)
        return not np.array_equal(wrong, layers[0]["weight"])
    valuelevel("dtype_width_confusion_mismatch", width_confusion)

    def truncate_packed():
        segs12 = [pf.Segment("W", pf.DT_PACK12, (D, HID), layers[0]["weight"])]
        blob = pf.pack(segs12)
        _, e = pf.read_directory(blob)
        payload = blob[e[0].offset:e[0].offset + e[0].nbytes - 4]
        try:
            pf.decode_segment(pf.DT_PACK12, (D, HID), payload)
        except ValueError:
            return True
        return False
    valuelevel("packed_stream_truncation_detected", truncate_packed)

    return ctrl, good


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default="/tmp/opencode/r2_pack_format")
    ap.add_argument("--out", default=str(HERE / "results.json"))
    ap.add_argument("--emit-rtl", action="store_true", default=True)
    args = ap.parse_args()
    build = Path(args.build_dir)
    build.mkdir(parents=True, exist_ok=True)

    log("== r2_pack_format: KWT1-PACK v1 ==")
    layers = load_manifest()
    samples = load_samples()

    # ---- training-only activation scale (gradient-free) -----------------
    train = [s for s in samples if s["split"] == "training"]
    val = [s for s in samples if s["split"] == "validation"]
    train_max = max(float(np.abs(s["x"]).max()) for s in train)
    scale = train_max / ACT_QMAX
    log(f"train samples={len(train)} val samples={len(val)}")
    log(f"train max|X|={train_max:.6f} -> s_x={scale:.9e} (2^-{np.log2(1/scale):.4f})")

    # ---- 1. 12-layer parameter pack (int16 container) -------------------
    blob, meta, entries, got, mism = roundtrip_param_pack(layers)
    log(f"PARAM_PACK bytes={len(blob)} nseg={meta['nseg']} roundtrip_mismatch={mism}")
    (build / "params_a12w10_container.kwtp").write_bytes(blob)

    blob10, got10, mism10 = tight_variant(layers, pf.DT_PACK10)
    blob12, got12, mism12 = tight_variant(layers, pf.DT_PACK12)
    log(f"PARAM_PACK tight10 bytes={len(blob10)} mismatch={mism10}")
    log(f"PARAM_PACK tight12 bytes={len(blob12)} mismatch={mism12}")

    # ---- 2. activation pack ---------------------------------------------
    act_blob, act_rows = roundtrip_act_pack(samples, scale)
    log(f"ACT_PACK bytes={len(act_blob)} samples={len(act_rows)}")
    for r in act_rows:
        log(f"  {r['file']:22s} {r['split']:10s} exact={r['bit_exact']} "
            f"quant_nrmse={r['quant_nrmse_pct']:.4f}% max|q|={r['max_abs']}")
    (build / "act_int16_container.kwtp").write_bytes(act_blob)
    act10, am10 = tight_act_pack(samples, scale, pf.DT_PACK10)
    act12, am12 = tight_act_pack(samples, scale, pf.DT_PACK12)
    log(f"ACT_PACK tight12 bytes={len(act12)} mismatch={am12}")
    log(f"ACT_PACK tight10 bytes={len(act10)} mismatch={am10} "
        f"(EXPECTED >0: 12-bit activations do not fit a 10-bit stream)")
    # A 10-bit activation stream cannot represent |q|>511 -> treat as an
    # *incompatibility control*, not a roundtrip success criterion.
    act10_incompatible = am10 > 0
    am10_for_gate = 0 if act10_incompatible else 1

    # ---- 3. full FP32 parameter pack ------------------------------------
    fp_blob, fp_entries, fp_mism, fp_nseg = build_fp32_pack()
    log(f"FP32_PACK bytes={len(fp_blob)} nseg={fp_nseg} roundtrip_mismatch={fp_mism}")

    # ---- 4. negative controls -------------------------------------------
    log("negative controls:")
    ctrl, good = negative_controls(layers, samples, scale)

    # ---- 5. bandwidth table ---------------------------------------------
    def bw(nbytes):
        return dict(bytes=int(nbytes),
                    words16=int((nbytes + 1) // 2),
                    words32=int((nbytes + 3) // 4),
                    beats128=int((nbytes + 15) // 16))

    w1_c = D * HID * 2
    b1_c = HID * 4
    x_c = T * D * 2
    y_c = T * HID * 2
    bandwidth = {
        "per_layer": {
            "W1_int16_container": bw(w1_c),
            "W1_pack10": bw((D * HID * 10 + 7) // 8),
            "W1_pack12": bw((D * HID * 12 + 7) // 8),
            "b1_int32": bw(b1_c),
            "staged_int16": bw(w1_c + b1_c),
            "staged_pack10": bw((D * HID * 10 + 7) // 8 + b1_c),
            "X_99x64_int16": bw(x_c),
            "X_99x64_pack12": bw((T * D * 12 + 7) // 8),
            "Y_99x256_int16": bw(y_c),
        },
        "twelve_layers": {
            "params_int16_container": bw(LAYERS * (w1_c + b1_c)),
            "params_pack10": bw(LAYERS * ((D * HID * 10 + 7) // 8 + b1_c)),
            "params_pack12": bw(LAYERS * ((D * HID * 12 + 7) // 8 + b1_c)),
            "activations_12x99x64_int16": bw(LAYERS * x_c),
        },
        "full_fp32_12layer_pack_bytes": len(fp_blob),
    }

    # ---- summary --------------------------------------------------------
    all_exact = (mism == 0 and mism10 == 0 and mism12 == 0 and am12 == 0
                 and fp_mism == 0 and all(r["bit_exact"] for r in act_rows)
                 and act10_incompatible
                 and all(c["detected"] for c in ctrl))
    val_rows = [r for r in act_rows if r["split"] == "validation"]
    worst_val = max(r["quant_nrmse_pct"] for r in val_rows)
    results = dict(
        task="r2_pack_format",
        format=dict(magic=pf.MAGIC.decode(), version=pf.VERSION,
                    endianness="little",
                    header_bytes=pf.HEADER_BYTES, dirent_bytes=pf.DIRENT_BYTES,
                    align_bytes=pf.ALIGN,
                    dtypes={k: pf.DT_NAMES[k] for k in pf.DT_NAMES}),
        inputs=dict(ffn1_npz=str(FFN1_NPZ.relative_to(ROOT)),
                    ffn1_npz_sha256=sha_bytes(FFN1_NPZ.read_bytes()),
                    fp32_npz=str(FP32_NPZ.relative_to(ROOT)),
                    fp32_npz_sha256=sha_bytes(FP32_NPZ.read_bytes()),
                    real_dir=str(REAL_DIR.relative_to(ROOT)),
                    train_samples=[s["file"] for s in train],
                    val_samples=[s["file"] for s in val]),
        calibration=dict(partition="training only", method="max-abs (gradient free)",
                         train_max_abs=train_max, activation_scale=scale,
                         act_bits=ACT_BITS, act_qmax=ACT_QMAX,
                         weight_bits=W_BITS, weight_qmax=W_QMAX,
                         validation_used_for_calibration=False),
        params_pack=dict(blob_bytes=len(blob), nseg=meta["nseg"],
                         sha256=sha_bytes(blob), roundtrip_mismatch=mism,
                         tight10_bytes=len(blob10), tight10_mismatch=mism10,
                         tight12_bytes=len(blob12), tight12_mismatch=mism12,
                         entries=[dict(name=e.name, dtype=pf.DT_NAMES[e.dtype],
                                       shape=list(e.shape), offset=e.offset,
                                       nbytes=e.nbytes) for e in entries]),
        act_pack=dict(blob_bytes=len(act_blob), sha256=sha_bytes(act_blob),
                      rows=act_rows, tight10_bytes=len(act10), tight10_mismatch=am10,
                      tight10_incompatible=bool(act10_incompatible),
                      tight12_bytes=len(act12), tight12_mismatch=am12,
                      worst_val_quant_nrmse_pct=worst_val),
        fp32_pack=dict(blob_bytes=len(fp_blob), nseg=fp_nseg,
                       sha256=sha_bytes(fp_blob), roundtrip_mismatch=fp_mism),
        bandwidth=bandwidth,
        negative_controls=ctrl,
        all_bit_exact=bool(all_exact),
        evidence_levels=dict(
            L1_software_reference="PROVEN: pack/unpack roundtrip + validator + negative controls",
            L2_rtl_bit_exact="see run.sh (Verilator byte-lane) — status recorded separately",
            L3_e203_soc="MISSING: no official E203/NICE/System-Bus run",
            L4_synthesis="see run.sh (generic yosys) — not PDK/150MHz",
            L5_official_acceptance="MISSING: no official 600-sample evaluator",
        ),
    )
    Path(args.out).write_text(json.dumps(results, indent=2))
    log(f"ALL_BIT_EXACT={all_exact} worst_val_quant_nrmse={worst_val:.4f}%")
    log(f"results -> {args.out}")
    log("R2_PACK_FORMAT_DONE")
    return 0 if all_exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
