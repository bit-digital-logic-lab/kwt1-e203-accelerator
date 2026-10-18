#!/usr/bin/env python3
"""Generate real A12W10 FFN1 vectors for the legacy E203 full-layer testbench.

One hex token is one signed two's-complement element: 4 digits for 16-bit
operands, 8 digits for 32-bit bias and expected output. Files are per layer.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[4]
SOFTWARE = REPO / "software"
sys.path.insert(0, str(SOFTWARE))
from quantize_ffn1 import collect_activations  # noqa: E402
from run import KWT1, PREFIX  # noqa: E402


M, K, N, LAYERS = 99, 64, 256, 12
ROW_TILES, COL_TILES = 13, 16
A_QMAX, W_QMAX = 2047, 511


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_hex(path, values, width):
    mask = (1 << width) - 1
    digits = width // 4
    with path.open("w", encoding="ascii") as destination:
        for value in np.asarray(values).reshape(-1):
            destination.write(f"{int(value) & mask:0{digits}x}\n")
    return {"path": str(path.resolve()), "sha256": sha256(path),
            "elements": int(np.asarray(values).size), "element_bits": width}


def load_validation_sample(path, index):
    with np.load(path, allow_pickle=False) as data:
        features = data["features"]
        partitions = data["partitions"]
        paths = data["paths"]
        words = data["words"]
    if features.dtype != np.float32 or features.ndim != 3 or features.shape[1:] != (98, 40):
        raise ValueError("expected FP32 MFCC [samples,98,40]")
    if not np.isfinite(features).all():
        raise ValueError("nonfinite MFCC feature")
    if np.any((partitions != "training") & (partitions != "validation")):
        raise ValueError("testing/other partitions are forbidden")
    train_ids = np.flatnonzero(partitions == "training")
    validation_ids = np.flatnonzero(partitions == "validation")
    if not len(train_ids) or not len(validation_ids):
        raise ValueError("training and validation samples are required")
    if set(paths[train_ids]).intersection(paths[validation_ids]):
        raise ValueError("training and validation paths overlap")
    if not 0 <= index < len(validation_ids):
        raise ValueError(f"validation index must be 0..{len(validation_ids)-1}")
    sample_id = int(validation_ids[index])
    return (features[sample_id:sample_id + 1].copy(),
            str(paths[sample_id]), str(words[sample_id]),
            int(len(train_ids)), int(len(validation_ids)))


def encode_layer(x, weight, bias, sx, sw, qw, qb):
    if x.shape != (M, K) or weight.shape != (K, N) or bias.shape != (N,):
        raise ValueError("unexpected FFN1 tensor shape")
    if qw.shape != (K, N) or qb.shape != (N,) or sw.shape != (N,):
        raise ValueError("unexpected quantized parameter shape")
    if not np.isfinite(x).all() or not np.isfinite(weight).all() or not np.isfinite(bias).all():
        raise ValueError("nonfinite FFN1 input or reference parameter")
    if not np.isfinite(sx) or sx <= 0 or not np.isfinite(sw).all() or np.any(sw <= 0):
        raise ValueError("invalid quantization scale")
    qw = qw.astype(np.int64)
    qb = qb.astype(np.int64)
    if np.any(np.abs(qw) > W_QMAX):
        raise ValueError("frozen weights exceed symmetric W10 range")
    qx = np.clip(np.rint(x.astype(np.float64) / sx), -A_QMAX, A_QMAX).astype(np.int64)
    expected = qx @ qw + qb
    if expected.min() < np.iinfo(np.int32).min or expected.max() > np.iinfo(np.int32).max:
        raise OverflowError("integer result exceeds INT32")
    bound = K * A_QMAX * W_QMAX + int(np.max(np.abs(qb)))
    if bound > np.iinfo(np.int32).max:
        raise OverflowError("full quantized-range bound exceeds INT32")

    # Exactly matches tb_kwt_full_layer_e203.sv: A shared-RAM words are
    # indexed by (rt*64+k), then 8 little-lane 16-bit elements r.
    a_tiles = np.zeros((ROW_TILES, K, 8), dtype=np.int64)
    for rt in range(ROW_TILES):
        rows = min(8, M - rt * 8)
        a_tiles[rt, :, :rows] = qx[rt * 8:rt * 8 + rows].T
    # B shared-RAM words are indexed by (ct*128+k*2+c//8), with c%8
    # selecting a 16-bit lane. The flattened ct,k,c file follows that order.
    w_tiles = np.transpose(qw.reshape(K, COL_TILES, 16), (1, 0, 2))
    if not np.all(a_tiles[-1, :, 3:] == 0):
        raise AssertionError("last A tile must have five zero-padded rows")
    if not np.array_equal(w_tiles.transpose(1, 0, 2).reshape(K, N), qw):
        raise AssertionError("B tiling changed column order")

    reference = x @ weight + bias
    decoded = expected.astype(np.float64) * (sx * sw)
    ref64 = reference.astype(np.float64)
    error_rms = float(np.sqrt(np.mean((decoded - ref64) ** 2)))
    reference_rms = float(np.sqrt(np.mean(ref64 ** 2)))
    nrmse_pct = 100 * error_rms / max(reference_rms, 1e-8)
    return a_tiles, w_tiles, qb, expected, nrmse_pct, bound


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path,
                        default=SOFTWARE / ".build/quant_samples_12labels_40.npz")
    parser.add_argument("--params", type=Path,
                        default=SOFTWARE / "results/ffn1_a12w10_static_sym_12labels_40.npz")
    parser.add_argument("--weights", type=Path,
                        default=SOFTWARE / "weights/kwt1_fp32.npz")
    parser.add_argument("--validation-index", type=int, default=0,
                        help="zero-based index within the validation partition")
    parser.add_argument("--layers", nargs="+", type=int,
                        default=list(range(1, LAYERS + 1)))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if len(set(args.layers)) != len(args.layers) or any(not 1 <= item <= LAYERS for item in args.layers):
        parser.error("--layers must contain unique layer numbers in 1..12")
    if not args.layers:
        parser.error("--layers cannot be empty")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)

    sample, sample_path, label, train_count, validation_count = load_validation_sample(
        args.samples, args.validation_index)
    params_meta_path = args.params.with_suffix(".json")
    params_meta = json.loads(params_meta_path.read_text())
    if params_meta.get("scheme") != "A12W10":
        raise ValueError("parameter metadata must declare A12W10")
    if params_meta.get("sample_sha256") != sha256(args.samples):
        raise ValueError("parameter calibration archive differs from input sample archive")
    if params_meta.get("weight_sha256") != sha256(args.weights):
        raise ValueError("parameter FP32 weight archive differs from model archive")
    model = KWT1(args.weights, device)
    activations = collect_activations(sample, model)
    if activations.shape != (1, LAYERS, M, K):
        raise ValueError(f"unexpected FFN1 activations {activations.shape}")
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    with np.load(args.params, allow_pickle=False) as params:
        for layer in args.layers:
            stem = f"layer{layer:02}"
            prefix = f"{PREFIX}/enc_layers/{layer-1}/ffn/layer_with_weights-0"
            weight = model.weight(prefix + "/kernel").cpu().numpy()
            bias = model.weight(prefix + "/bias").cpu().numpy()
            sx = float(params[stem + "_activation_scale"])
            sw = params[stem + "_weight_scales"].astype(np.float64)
            qw = params[stem + "_weight"]
            qb = params[stem + "_bias"]
            a_tiles, w_tiles, qbias, expected, nrmse, bound = encode_layer(
                activations[0, layer - 1], weight, bias, sx, sw, qw, qb)
            files = {
                "a16": write_hex(args.out / f"{stem}_a16.hex", a_tiles, 16),
                "w16": write_hex(args.out / f"{stem}_w16.hex", w_tiles, 16),
                "bias32": write_hex(args.out / f"{stem}_bias32.hex", qbias, 32),
                "expected32": write_hex(args.out / f"{stem}_expected32.hex", expected, 32),
            }
            records.append({"layer": layer, "activation_scale": sx,
                            "weight_scale_min": float(sw.min()),
                            "weight_scale_max": float(sw.max()),
                            "nrmse_pct": nrmse,
                            "max_abs_expected_int32": int(np.max(np.abs(expected))),
                            "conservative_abs_accumulator_bound": bound,
                            "files": files})
            print(f"layer={layer:02} nrmse={nrmse:.6f}% "
                  f"A={files['a16']['elements']} W={files['w16']['elements']} "
                  f"bias={files['bias32']['elements']} expected={files['expected32']['elements']}",
                  flush=True)

    manifest = {
        "format": "KWT-1 A12W10 per-layer E203 full-layer vectors v1",
        "generation_device": device.type,
        "validation_index": args.validation_index,
        "sample_path": sample_path, "label": label,
        "training_count": train_count, "validation_count": validation_count,
        "sample_archive_sha256": sha256(args.samples),
        "parameter_archive_sha256": sha256(args.params),
        "parameter_metadata_sha256": sha256(params_meta_path),
        "fp32_weight_archive_sha256": sha256(args.weights),
        "tensor_shapes": {"X": [M, K], "W": [K, N], "bias": [N], "Y": [M, N]},
        "hex_layout": {
            "a16": "13x64x8, flat rt,k,r; last tile rows 3..7 zero",
            "w16": "16x64x16, flat ct,k,c; physical 128-bit word = 8 consecutive c lanes",
            "bias32": "256 values, flat output column",
            "expected32": "99x256, flat row-major; valid rows only",
            "negative_values": "two's-complement hexadecimal, one element per line",
        },
        "numerical_protocol": "A12W10 symmetric zero-point 0, rint ties-to-even, signed INT32 GEMM+bias; FP64 final decode for NRMSE only",
        "layers": records,
        "worst_layer_nrmse_pct": max(record["nrmse_pct"] for record in records),
    }
    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"saved {manifest_path} worst={manifest['worst_layer_nrmse_pct']:.6f}%")


if __name__ == "__main__":
    main()
