#!/usr/bin/env python3
"""Freeze a static symmetric FFN1 integer parameter pack from training data."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from quantize_ffn1 import collect_activations
from run import KWT1, PREFIX, ROOT

LAYERS, M, K, N = 12, 99, 64, 256


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path,
                        default=ROOT / ".build/quant_samples_12labels_20.npz")
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/kwt1_fp32.npz")
    parser.add_argument("--activation-bits", type=int, default=12)
    parser.add_argument("--weight-bits", type=int, default=10)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/ffn1_a12w10_static_sym")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if not (2 <= args.activation_bits <= 15 and 2 <= args.weight_bits <= 15):
        parser.error("bit widths must be in 2..15")
    qmax_a = (1 << (args.activation_bits - 1)) - 1
    qmax_w = (1 << (args.weight_bits - 1)) - 1
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")

    with np.load(args.samples, allow_pickle=False) as dataset:
        features, partitions = dataset["features"], dataset["partitions"]
        paths = dataset["paths"]
    train_ids = np.flatnonzero(partitions == "training")
    validation_ids = np.flatnonzero(partitions == "validation")
    if not len(train_ids) or not len(validation_ids):
        raise ValueError("dataset must contain training and validation splits")
    if set(paths[train_ids]).intersection(paths[validation_ids]):
        raise ValueError("training and validation paths overlap")

    model = KWT1(args.weights, device)
    activations = collect_activations(features[train_ids], model)
    if activations.shape != (len(train_ids), LAYERS, M, K):
        raise ValueError(f"unexpected FFN1 input tensor {activations.shape}")

    packed, layer_records = {}, []
    for layer in range(LAYERS):
        stem = f"layer{layer+1:02}"
        prefix = f"{PREFIX}/enc_layers/{layer}/ffn/layer_with_weights-0"
        x = activations[:, layer].astype(np.float64)
        weight = model.weight(prefix + "/kernel").cpu().numpy().astype(np.float64)
        bias = model.weight(prefix + "/bias").cpu().numpy().astype(np.float64)
        sx = max(float(np.max(np.abs(x))) / qmax_a, 1e-12)
        sw = np.maximum(np.max(np.abs(weight), axis=0) / qmax_w, 1e-12)
        qw = np.clip(np.rint(weight / sw), -qmax_w, qmax_w).astype(np.int16)
        qb64 = np.rint(bias / (sx * sw)).astype(np.int64)
        if np.any(qb64 < np.iinfo(np.int32).min) or np.any(qb64 > np.iinfo(np.int32).max):
            raise OverflowError(f"layer {layer+1}: quantized bias exceeds INT32")
        bound = K * qmax_a * qmax_w + int(np.max(np.abs(qb64)))
        if bound >= (1 << 31):
            raise OverflowError(f"layer {layer+1}: conservative accumulator bound {bound} exceeds INT32")
        packed[stem + "_weight"] = qw
        packed[stem + "_bias"] = qb64.astype(np.int32)
        packed[stem + "_activation_scale"] = np.array(sx, dtype=np.float64)
        packed[stem + "_weight_scales"] = sw.astype(np.float64)
        layer_records.append({"layer": layer+1, "activation_scale": sx,
                              "weight_scale_min": float(sw.min()),
                              "weight_scale_max": float(sw.max()),
                              "weight_qmin": int(qw.min()), "weight_qmax": int(qw.max()),
                              "bias_abs_max": int(np.max(np.abs(qb64))),
                              "conservative_abs_accumulator_bound": int(bound)})
        print(f"layer={layer+1:02} A{args.activation_bits}W{args.weight_bits} "
              f"bound={bound}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"), **packed)
    metadata = {
        "scheme": f"A{args.activation_bits}W{args.weight_bits}",
        "activation_quantization": "per-layer static symmetric, zero point 0, training maximum",
        "weight_quantization": "per-output-channel symmetric, zero point 0, frozen FP32 max",
        "bias_quantization": "signed INT32 round-to-nearest ties-to-even at activation_scale*weight_scale",
        "training_count": int(len(train_ids)), "validation_count": int(len(validation_ids)),
        "validation_used_for_calibration": False,
        "rounding": "nearest, ties to even; clip to [-qmax,+qmax]",
        "accumulator": "signed INT32, conservative bound checked for all layers",
        "sample_sha256": sha256(args.samples), "weight_sha256": sha256(args.weights),
        "parameter_archive": str(args.output.with_suffix(".npz").resolve()),
        "parameter_sha256": sha256(args.output.with_suffix(".npz")),
        "packed_weight_kib_12layers": LAYERS*K*N*args.weight_bits/8/1024,
        "container_weight_bytes": LAYERS*K*N*2,
        "layers": layer_records,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(f"PASS exported {metadata['scheme']} to {metadata['parameter_archive']}")


if __name__ == "__main__":
    main()
