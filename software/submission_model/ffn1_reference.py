#!/usr/bin/env python3
"""Independent integer software reference for the frozen undergraduate FFN1.

Config ID : KWT1-FFN1-A12W10-SYM-v1
Object    : Y[99,256] = X[99,64] @ W[64,256] + b[256]   (12 independent layers)
Protocol  : see PROTOCOL.md.  A12W10 symmetric, zero point 0, INT32 wrap.

This is the submittable "software numerical model": it encodes the same
integer path the RTL must reproduce bit for bit.  It never recomputes the
frozen calibration; it only consumes a frozen parameter pack and, for real
samples, the frozen FP32 KWT-1 checkpoint.

Subcommands
-----------
  emit           real samples -> per-layer INT32 expected32 + NRMSE + manifest
  verify-vectors cross-check existing E203/RTL vectors bit for bit with NumPy
  selftest       negative injections must be detected (wrong weight/rounding)
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOFTWARE = REPO / "software"
sys.path.insert(0, str(SOFTWARE))

CONFIG_ID = "KWT1-FFN1-A12W10-SYM-v1"
M, K, N, LAYERS = 99, 64, 256, 12
A_BITS, W_BITS = 12, 10
A_QMAX, W_QMAX = (1 << (A_BITS - 1)) - 1, (1 << (W_BITS - 1)) - 1
ROW_TILES, COL_TILES = 13, 16
ROW_LANES, COL_LANES = 8, 16
RMS_FLOOR = 1e-8

DEFAULT_PARAMS = SOFTWARE / "results/ffn1_a12w10_static_sym.npz"
DEFAULT_WEIGHTS = SOFTWARE / "weights/kwt1_fp32.npz"
DEFAULT_SAMPLES = SOFTWARE / ".build/quant_samples_12labels_20.npz"


# --------------------------------------------------------------------------- #
# Frozen integer protocol (pure NumPy, DUT independent)
# --------------------------------------------------------------------------- #
def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_out(path, name):
    """Accept either a file path or a directory; return a writable JSON path."""
    path = Path(path)
    if path.suffix == "":
        path.mkdir(parents=True, exist_ok=True)
        return path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def wrap_int32(values):
    """Signed INT32 two's-complement wraparound (hardware accumulator result)."""
    return (((np.asarray(values, dtype=np.int64) + (1 << 31)) & 0xFFFFFFFF)
            - (1 << 31)).astype(np.int32)


def quantize_symmetric(values, scale, qmax):
    """Static symmetric, zero point 0, round-to-nearest ties-to-even, clip."""
    scaled = np.rint(np.asarray(values, dtype=np.float64) / np.float64(scale))
    return np.clip(scaled, -qmax, qmax).astype(np.int64)


def integer_ffn1(qx, qw, qb):
    """Exact integer MAC + bias, then INT32 wrap.  qx[...,64] qw[64,256]."""
    acc = np.asarray(qx, dtype=np.int64) @ np.asarray(qw, dtype=np.int64)
    acc = acc + np.asarray(qb, dtype=np.int64)
    return wrap_int32(acc)


def decode(expected32, sx, sw):
    """FP64 dequantization: y = expected32 * (sx * sw[channel])."""
    return expected32.astype(np.float64) * (np.float64(sx) * sw[None, :])


def nrmse_pct(expected32, sx, sw, reference):
    """Contest NRMSE: pooled RMS(err)/max(RMS(ref),1e-8), in percent."""
    decoded = decode(expected32, sx, sw)
    ref = np.asarray(reference, dtype=np.float64)
    err_rms = float(np.sqrt(np.mean((decoded - ref) ** 2)))
    ref_rms = float(np.sqrt(np.mean(ref ** 2)))
    return 100.0 * err_rms / max(ref_rms, RMS_FLOOR), ref_rms


def load_pack(params_path):
    """Return list of 12 layer dicts {sx, sw, qw, qb} from a frozen NPZ."""
    layers = []
    with np.load(params_path, allow_pickle=False) as data:
        for layer in range(LAYERS):
            stem = f"layer{layer + 1:02}"
            layers.append({
                "sx": float(data[stem + "_activation_scale"]),
                "sw": data[stem + "_weight_scales"].astype(np.float64),
                "qw": data[stem + "_weight"].astype(np.int64),
                "qb": data[stem + "_bias"].astype(np.int64),
            })
    return layers


def read_hex(path, bits):
    tokens = Path(path).read_text().split()
    raw = np.array([int(token, 16) for token in tokens], dtype=np.uint64)
    if bits == 16:
        return raw.astype(np.uint16).view(np.int16).astype(np.int64)
    if bits == 32:
        return raw.astype(np.uint32).view(np.int32).astype(np.int64)
    raise ValueError(f"unsupported width {bits}")


def reconstruct_vectors(vectors_dir, layer):
    """Rebuild X[99,64], W[64,256], b[256] from the frozen E203 hex pack."""
    stem = f"layer{layer:02}"
    a = read_hex(vectors_dir / f"{stem}_a16.hex", 16)
    w = read_hex(vectors_dir / f"{stem}_w16.hex", 16)
    b = read_hex(vectors_dir / f"{stem}_bias32.hex", 32)
    expected = read_hex(vectors_dir / f"{stem}_expected32.hex", 32)
    if a.size != ROW_TILES * K * ROW_LANES or w.size != COL_TILES * K * COL_LANES:
        raise ValueError(f"{stem}: unexpected operand element count")
    x = np.zeros((M, K), dtype=np.int64)
    a_tiles = a.reshape(ROW_TILES, K, ROW_LANES)  # flat rt,k,r
    for rt in range(ROW_TILES):
        for r in range(ROW_LANES):
            row = rt * ROW_LANES + r
            if row < M:
                x[row, :] = a_tiles[rt, :, r]
    w_tiles = w.reshape(COL_TILES, K, COL_LANES)  # flat ct,k,c
    wm = np.zeros((K, N), dtype=np.int64)
    for ct in range(COL_TILES):
        wm[:, ct * COL_LANES:(ct + 1) * COL_LANES] = w_tiles[ct]
    if expected.size != M * N:
        raise ValueError(f"{stem}: expected32 must hold {M * N} values")
    return x, wm, b, expected.reshape(M, N), a, w


def config_record(params_path, weights_path, samples_path):
    return {
        "config_id": CONFIG_ID,
        "algorithm": "signed symmetric A12W10, zero point 0, per-layer activation "
                     "scale, per-output-channel weight scale, exact integer MAC + "
                     "INT32 bias, round-to-nearest ties-to-even clip at encode, "
                     "INT32 two's-complement wrap, FP64 decode only for NRMSE",
        "a_bits": A_BITS, "w_bits": W_BITS, "a_qmax": A_QMAX, "w_qmax": W_QMAX,
        "shapes": {"X": [M, K], "W": [K, N], "b": [N], "Y": [M, N]},
        "rms_denominator_floor": RMS_FLOOR,
        "sha256": {
            "params": sha256_file(params_path),
            "weights": sha256_file(weights_path),
            "samples": sha256_file(samples_path) if Path(samples_path).exists() else None,
        },
    }


# --------------------------------------------------------------------------- #
# emit: real samples -> integer expected32 + NRMSE
# --------------------------------------------------------------------------- #
def emit(args):
    import torch
    from quantize_ffn1 import collect_activations
    from run import KWT1, PREFIX

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    with np.load(args.samples, allow_pickle=False) as data:
        features, partitions = data["features"], data["partitions"]
        paths, words = data["paths"], data["words"]
    if np.any((partitions != "training") & (partitions != "validation")):
        raise ValueError("testing/other partitions are forbidden")
    validation = np.flatnonzero(partitions == "validation")
    selected = []
    seen = set()
    for index in validation:
        if words[index] not in seen:
            seen.add(words[index])
            selected.append(int(index))
        if len(selected) == args.sample_count:
            break
    if len(selected) < args.sample_count:
        raise ValueError("not enough distinct validation labels")
    if set(paths[selected]).intersection(paths[np.flatnonzero(partitions == "training")]):
        raise ValueError("selected validation path overlaps training")

    model = KWT1(args.weights, device)
    activations = collect_activations(features[selected], model)
    if activations.shape != (len(selected), LAYERS, M, K):
        raise ValueError(f"unexpected activation tensor {activations.shape}")

    pack = load_pack(args.params)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result_dir = out / "expected32"
    result_dir.mkdir(exist_ok=True)
    layer_records = []
    for layer in range(LAYERS):
        sx, sw, qw, qb = (pack[layer]["sx"], pack[layer]["sw"],
                          pack[layer]["qw"], pack[layer]["qb"])
        expected_all, ref_all, wrap_events = [], [], 0
        for slot, index in enumerate(selected):
            x = activations[slot, layer].astype(np.float64)
            weight = model.weight(
                f"{PREFIX}/enc_layers/{layer}/ffn/layer_with_weights-0/kernel"
            ).cpu().numpy().astype(np.float64)
            bias = model.weight(
                f"{PREFIX}/enc_layers/{layer}/ffn/layer_with_weights-0/bias"
            ).cpu().numpy().astype(np.float64)
            qx = quantize_symmetric(x, sx, A_QMAX)
            exact = qx @ qw + qb
            wrap_events += int(np.count_nonzero(
                (exact < np.iinfo(np.int32).min) | (exact > np.iinfo(np.int32).max)))
            expected_all.append(wrap_int32(exact))
            ref_all.append(x @ weight + bias)
        expected = np.stack(expected_all)
        reference = np.stack(ref_all)
        nrmse, ref_rms = nrmse_pct(expected, sx, sw, reference)
        path = result_dir / f"layer{layer + 1:02}_expected32.npy"
        np.save(path, expected)
        layer_records.append({
            "layer": layer + 1,
            "activation_scale": sx,
            "weight_scale_min": float(sw.min()), "weight_scale_max": float(sw.max()),
            "expected32_sha256": sha256_file(path),
            "expected32_dtype": "int32 little-endian",
            "checked_outputs": int(expected.size),
            "min_int32": int(expected.min()), "max_int32": int(expected.max()),
            "int32_wrap_events": wrap_events,
            "nrmse_pct": nrmse, "fp32_reference_rms": ref_rms,
            "conservative_abs_accumulator_bound": K * A_QMAX * W_QMAX + int(np.max(np.abs(qb))),
        })
        print(f"layer={layer + 1:02} NRMSE={nrmse:.6f}% outputs={expected.size}",
              flush=True)
    manifest = config_record(args.params, args.weights, args.samples)
    manifest.update({
        "mode": "emit",
        "sample_indices": selected,
        "sample_paths": [str(paths[i]) for i in selected],
        "sample_labels": [str(words[i]) for i in selected],
        "validation_only": True,
        "used_for_calibration": False,
        "layers": layer_records,
        "worst_layer": max(layer_records, key=lambda r: r["nrmse_pct"])["layer"],
        "worst_nrmse_pct": max(r["nrmse_pct"] for r in layer_records),
    })
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"PASS emit config={CONFIG_ID} worst={manifest['worst_nrmse_pct']:.6f}% -> {out/'manifest.json'}")
    return 0


# --------------------------------------------------------------------------- #
# verify-vectors: RTL/E203 golden vectors vs independent NumPy integer GEMM
# --------------------------------------------------------------------------- #
def verify_vectors(args):
    vectors = Path(args.vectors)
    manifest_path = vectors / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    pinned = load_pack(args.params) if args.params else None
    records = []
    for layer in range(1, LAYERS + 1):
        x, wm, b, expected, a, w = reconstruct_vectors(vectors, layer)
        got = integer_ffn1(x, wm, b)
        mismatch = int(np.count_nonzero(got != expected))
        if mismatch:
            raise AssertionError(f"layer{layer:02}: {mismatch} outputs differ from vectors")
        weight_equal = None
        if pinned is not None:
            weight_equal = bool(np.array_equal(wm, pinned[layer - 1]["qw"]))
        style_round = None
        if manifest is not None:
            entry = manifest["layers"][layer - 1]
            style_round = True
            for key, rec in entry["files"].items():
                candidate = Path(rec["path"])
                if not candidate.is_file():
                    candidate = vectors / f"layer{layer:02}_{key}.hex"
                if sha256_file(candidate) != rec["sha256"]:
                    style_round = False
        records.append({"layer": layer, "outputs": int(expected.size),
                        "bit_exact": mismatch == 0, "weight_matches_pack": weight_equal,
                        "vector_sha_ok": style_round})
        print(f"layer={layer:02} bit_exact={mismatch == 0} outputs={expected.size} "
              f"weights_match_config={weight_equal}", flush=True)
    result = {
        "mode": "verify-vectors",
        "config_id": CONFIG_ID,
        "vectors_dir": str(vectors),
        "vectors_manifest_sha256": sha256_file(manifest_path) if manifest else None,
        "rtl_pack_parameter_sha256": manifest.get("parameter_archive_sha256") if manifest else None,
        "note": "independent NumPy INT64 GEMM + INT32 wrap reproduces the frozen "
                "E203 full-layer golden vectors bit for bit",
        "layers": records,
        "all_bit_exact": all(r["bit_exact"] for r in records),
    }
    out = resolve_out(args.out, "verify_vectors.json")
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"PASS verify-vectors all_bit_exact={result['all_bit_exact']} -> {out}")
    return 0


# --------------------------------------------------------------------------- #
# selftest: negative injections must be detected
# --------------------------------------------------------------------------- #
def selftest(args):
    vectors = Path(args.vectors)
    checks = []

    # (1) wrong weight: flip one W element, expect a bitwise mismatch.
    x, wm, b, expected, _, _ = reconstruct_vectors(vectors, args.layer)
    bad = wm.copy()
    bad[0, 0] += 1
    checks.append(("wrong_weight_detected",
                   bool(np.any(integer_ffn1(x, bad, b) != expected))))

    # (2) wrong bias: flip one bias element, expect a mismatch.
    bad_b = b.copy()
    bad_b[0] += 1
    checks.append(("wrong_bias_detected",
                   bool(np.any(integer_ffn1(x, wm, bad_b) != expected))))

    # (3) wrong rounding: truncation instead of ties-to-even changes the encode.
    sx = 0.0024374177762223966
    probe = np.array([0.5 * sx, 1.5 * sx, 2.5 * sx, -0.5 * sx, -1.5 * sx])
    ties = quantize_symmetric(probe, sx, A_QMAX)
    trunc = np.trunc(probe / sx).astype(np.int64)
    checks.append(("wrong_rounding_detected", bool(np.any(ties != trunc))))

    # (4) correct path is self-consistent: truncation really differs from rint.
    checks.append(("ties_to_even_used", bool(np.array_equal(ties, [0, 2, 2, 0, -2]))))

    failures = [name for name, ok in checks if not ok]
    result = {"mode": "selftest", "config_id": CONFIG_ID, "layer": args.layer,
              "checks": [{"name": n, "passed": ok} for n, ok in checks],
              "result": "PASS" if not failures else "FAIL", "failures": failures}
    if args.out:
        resolve_out(args.out, "selftest.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'} {name}")
    print(f"{result['result']} selftest")
    return 0 if not failures else 1


def config(args):
    record = config_record(args.params, args.weights, args.samples)
    out = resolve_out(args.out, "config_record.json")
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    print(f"PASS config -> {out}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    common.add_argument("--out", type=Path, default=SOFTWARE / ".build/submission_model")

    p_emit = sub.add_parser("emit", parents=[common], help="real samples -> expected32")
    p_emit.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p_emit.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    p_emit.add_argument("--sample-count", type=int, default=2,
                        help="number of validation samples, one per distinct label")
    p_emit.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p_emit.set_defaults(func=emit)

    p_vec = sub.add_parser("verify-vectors", parents=[common],
                           help="RTL golden vectors vs independent GEMM")
    p_vec.add_argument("--vectors", type=Path, required=True)
    p_vec.set_defaults(func=verify_vectors)

    p_self = sub.add_parser("selftest", parents=[common], help="negative injections")
    p_self.add_argument("--vectors", type=Path, required=True)
    p_self.add_argument("--layer", type=int, default=11)
    p_self.set_defaults(func=selftest)

    p_cfg = sub.add_parser("config", parents=[common],
                           help="print the frozen config record")
    p_cfg.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p_cfg.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    p_cfg.set_defaults(func=config)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
