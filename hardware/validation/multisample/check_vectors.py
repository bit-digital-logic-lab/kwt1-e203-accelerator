#!/usr/bin/env python3
"""Independently re-check generated real-layer E203 vectors.

This checker does not import the generator.  It re-parses every hex file,
re-hashes every manifest-referenced file, rebuilds the (99,64) activation and
(64,256) weight matrices from the E203 tile layout, and recomputes the signed
INT32 GEMM+bias result in int64 before comparing with expected32.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

M, K, N, LAYERS = 99, 64, 256, 12
ROW_TILES, COL_TILES = 13, 16
A_QMAX, W_QMAX = 2047, 511
EXPECTED_LINES = {"a16": 13 * 64 * 8, "w16": 16 * 64 * 16,
                  "bias32": 256, "expected32": 99 * 256}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_hex(path, bits):
    tokens = path.read_text(encoding="ascii").split()
    mask = (1 << bits) - 1
    sign = 1 << (bits - 1)
    values = []
    for token in tokens:
        value = int(token, 16) & mask
        values.append(value - (1 << bits) if value & sign else value)
    return np.asarray(values, dtype=np.int64)


def check_sample(vector_dir, indices, archive, params, weights):
    manifest_path = vector_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []

    if manifest.get("validation_index") not in indices:
        errors.append(f"unexpected validation_index {manifest.get('validation_index')}")
    if manifest.get("tensor_shapes") != {"X": [M, K], "W": [K, N],
                                         "bias": [N], "Y": [M, N]}:
        errors.append("manifest tensor_shapes is not X99x64/W64x256/bias256/Y99x256")
    sources = {
        "sample_archive_sha256": archive,
        "parameter_archive_sha256": params,
        "fp32_weight_archive_sha256": weights,
        "parameter_metadata_sha256": params.with_suffix(".json"),
    }
    for key, path in sources.items():
        actual = sha256(path)
        if manifest.get(key) != actual:
            errors.append(f"{key} mismatch: manifest={manifest.get(key)} actual={actual}")

    with np.load(archive, allow_pickle=False) as data:
        features = data["features"]
        partitions = data["partitions"]
        paths = data["paths"]
        words = data["words"]
    validation_ids = np.flatnonzero(partitions == "validation")
    index = int(manifest["validation_index"])
    sample_id = int(validation_ids[index])
    if manifest.get("sample_path") != str(paths[sample_id]):
        errors.append(f"sample_path mismatch: {manifest.get('sample_path')} vs {paths[sample_id]}")
    if manifest.get("label") != str(words[sample_id]):
        errors.append(f"label mismatch: {manifest.get('label')} vs {words[sample_id]}")
    feature_sha = hashlib.sha256(features[sample_id].tobytes()).hexdigest()
    if manifest.get("sample_feature_sha256") not in (None, feature_sha):
        errors.append("sample_feature_sha256 mismatch")

    records = []
    by_layer = {record["layer"]: record for record in manifest.get("layers", [])}
    if sorted(by_layer) != list(range(1, LAYERS + 1)):
        errors.append(f"manifest layers are not 1..12: {sorted(by_layer)}")
    for layer in range(1, LAYERS + 1):
        stem = f"layer{layer:02}"
        record = by_layer[layer]
        arrays = {}
        for key, lines in EXPECTED_LINES.items():
            path = vector_dir / f"{stem}_{key}.hex"
            if not path.is_file():
                errors.append(f"missing {path}")
                continue
            if sum(1 for _ in path.open("rb")) != lines:
                errors.append(f"{path.name} line count != {lines}")
            if record["files"][key]["sha256"] != sha256(path):
                errors.append(f"{path.name} sha256 mismatch")
            if record["files"][key]["elements"] != lines:
                errors.append(f"{path.name} manifest element count mismatch")
            arrays[key] = read_hex(path, record["files"][key]["element_bits"])

        a = arrays["a16"].reshape(ROW_TILES, K, 8)
        w = arrays["w16"].reshape(COL_TILES, K, 16)
        bias = arrays["bias32"].reshape(N)
        expected = arrays["expected32"].reshape(M, N)

        if not np.all(a[-1, :, 3:] == 0):
            errors.append(f"{stem}: last A tile rows 3..7 are not zero padding")
        if np.any(np.abs(a) > A_QMAX) or np.any(np.abs(w) > W_QMAX):
            errors.append(f"{stem}: operand exceeds A12/W10 ranges")

        qx = np.zeros((M, K), dtype=np.int64)
        for rt in range(ROW_TILES):
            rows = min(8, M - rt * 8)
            qx[rt * 8:rt * 8 + rows] = a[rt, :, :rows].T
        qw = w.transpose(1, 0, 2).reshape(K, N)
        recomputed = qx @ qw + bias
        if not np.array_equal(recomputed, expected):
            bad = int(np.count_nonzero(recomputed != expected))
            errors.append(f"{stem}: recomputed expected32 differs in {bad} elements")
        if recomputed.min() < np.iinfo(np.int32).min or recomputed.max() > np.iinfo(np.int32).max:
            errors.append(f"{stem}: int32 overflow")

        records.append({
            "layer": layer,
            "a16_sha256": record["files"]["a16"]["sha256"],
            "w16_sha256": record["files"]["w16"]["sha256"],
            "bias32_sha256": record["files"]["bias32"]["sha256"],
            "expected32_sha256": record["files"]["expected32"]["sha256"],
            "shapes": {"X": [M, K], "W": [K, N], "bias": [N], "Y": [M, N]},
            "hex_elements": {key: int(value.size) for key, value in arrays.items()},
            "output_elements": int(expected.size),
            "max_abs_expected_int32": int(np.max(np.abs(expected))),
            "recomputed_bit_exact": bool(np.array_equal(recomputed, expected)),
            "layer_nrmse_pct": float(record["nrmse_pct"]),
        })

    return {
        "validation_index": index,
        "sample_id": sample_id,
        "label": manifest.get("label"),
        "sample_path": manifest.get("sample_path"),
        "sample_feature_sha256": feature_sha,
        "sample_archive_sha256": manifest.get("sample_archive_sha256"),
        "parameter_archive_sha256": manifest.get("parameter_archive_sha256"),
        "fp32_weight_archive_sha256": manifest.get("fp32_weight_archive_sha256"),
        "worst_layer_nrmse_pct": manifest.get("worst_layer_nrmse_pct"),
        "layers": records,
        "errors": errors,
        "ok": not errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector-root", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    reports = []
    for index in args.indices:
        dirs = sorted(args.vector_root.glob(f"sample_idx{index:03d}"))
        if len(dirs) != 1:
            reports.append({"validation_index": index, "ok": False,
                            "errors": [f"expected one sample_idx{index:03d} dir"]})
            continue
        reports.append(check_sample(dirs[0], args.indices, args.samples,
                                    args.params, args.weights))
    result = {"format": "KWT-1 A12W10 multisample vector independent check v1",
              "samples": reports,
              "ok": all(item["ok"] for item in reports)}
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    for item in reports:
        status = "PASS" if item["ok"] else "FAIL"
        print(f"{status} idx={item['validation_index']} label={item.get('label')} "
              f"layers={len(item.get('layers', []))} errors={len(item.get('errors', []))}")
        for error in item.get("errors", []):
            print(f"  {error}", file=sys.stderr)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
