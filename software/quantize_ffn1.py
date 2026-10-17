"""Undergraduate KWT-1 FFN1 whole-tensor INT8/INT16 PTQ experiment."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from run import KWT1, PREFIX, ROOT


SCHEMES = ("global_max", "layer_max", "layer_p999", "layer_pow2", "dynamic_max")


def scale_from_abs(values, qmax, percentile=None, pow2=False):
    magnitude = np.abs(values.astype(np.float64))
    limit = float(np.max(magnitude) if percentile is None else np.percentile(magnitude, percentile))
    if limit <= 0:
        return 1.0
    scale = limit / qmax
    return float(2.0 ** math.ceil(math.log2(scale))) if pow2 else scale


def quantize(values, scale, qmax, dtype):
    return np.clip(np.rint(values.astype(np.float64) / scale), -qmax, qmax).astype(dtype)


def collect_activations(features, model):
    outputs = []
    with torch.inference_mode():
        for sample in features:
            encoder_input = model.encoder_input(torch.from_numpy(sample).to(model.device))
            _, _, ffn_inputs = model.forward(encoder_input, capture_ffn=True)
            outputs.append(np.stack([a.cpu().numpy() for a in ffn_inputs]))
    return np.stack(outputs)


def evaluate(bits, scheme, train, test, weights, biases):
    qmax = 2 ** (bits - 1) - 1
    dtype = np.int32 if bits == 8 else np.int64
    global_x = scale_from_abs(train, qmax)
    global_w = scale_from_abs(np.concatenate([w.ravel() for w in weights]), qmax)
    records = []
    for layer in range(12):
        x = test[:, layer]  # [samples, 99, 64]
        weight, bias = weights[layer], biases[layer]
        reference = x @ weight + bias
        if scheme == "global_max":
            sw = global_w
            sx = np.full(len(x), global_x)
        elif scheme == "layer_max":
            sw = scale_from_abs(weight, qmax)
            sx = np.full(len(x), scale_from_abs(train[:, layer], qmax))
        elif scheme == "layer_p999":
            sw = scale_from_abs(weight, qmax, percentile=99.9)
            sx = np.full(len(x), scale_from_abs(train[:, layer], qmax, percentile=99.9))
        elif scheme == "layer_pow2":
            sw = scale_from_abs(weight, qmax, pow2=True)
            sx = np.full(len(x), scale_from_abs(train[:, layer], qmax, pow2=True))
        elif scheme == "dynamic_max":
            sw = scale_from_abs(weight, qmax)
            sx = np.array([scale_from_abs(sample, qmax) for sample in x])
        else:
            raise ValueError(scheme)

        qw = quantize(weight, sw, qmax, dtype)
        approximate = np.empty_like(reference)
        max_acc = 0
        clipped_x = 0
        for i in range(len(x)):
            qx = quantize(x[i], sx[i], qmax, dtype)
            qb = quantize(bias, sx[i] * sw, np.iinfo(dtype).max, dtype)
            acc = qx @ qw + qb  # Genuine signed integer MAC and integer bias.
            max_acc = max(max_acc, int(np.max(np.abs(acc))))
            approximate[i] = acc.astype(np.float32) * np.float32(sx[i] * sw)
            clipped_x += int(np.count_nonzero(np.abs(x[i]) > sx[i] * qmax))
        if bits == 8 and max_acc > np.iinfo(np.int32).max:
            raise OverflowError("INT8 accumulator exceeds int32")
        error = approximate.astype(np.float64) - reference.astype(np.float64)
        nrmse = float(np.sqrt(np.sum(error**2) / np.sum(reference.astype(np.float64)**2)))
        records.append({
            "bits": bits, "scheme": scheme, "layer": layer + 1,
            "nrmse_pct": 100 * nrmse, "max_abs_error": float(np.max(np.abs(error))),
            "x_scale_min": float(np.min(sx)), "x_scale_max": float(np.max(sx)),
            "w_scale": sw, "x_clip_pct": 100 * clipped_x / x.size,
            "w_clip_pct": 100 * np.count_nonzero(np.abs(weight) > sw * qmax) / weight.size,
            "max_abs_accumulator": max_acc,
        })
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=ROOT / ".build/quant_samples.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "results/ffn1_ptq")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    model = KWT1(ROOT / "weights/kwt1_fp32.npz", device)
    with np.load(args.samples, allow_pickle=False) as data:
        features = data["features"]
        partitions = data["partitions"]
        paths = data["paths"]
    if features.dtype != np.float32 or features.shape[1:] != (98, 40):
        raise ValueError("Expected FP32 MFCC features [N,98,40]")
    if len(set(paths.tolist())) != len(paths):
        raise ValueError("Duplicate WAV paths")
    train_ids = np.flatnonzero(partitions == "training")
    test_ids = np.flatnonzero(partitions == "validation")
    if not len(train_ids) or not len(test_ids):
        raise ValueError("Need disjoint training and validation samples")
    inputs = collect_activations(features, model)
    train = inputs[train_ids]
    test = inputs[test_ids]
    weights, biases = [], []
    for i in range(12):
        base = f"{PREFIX}/enc_layers/{i}/ffn/layer_with_weights-0"
        weights.append(model.weight(base + "/kernel").cpu().numpy())
        biases.append(model.weight(base + "/bias").cpu().numpy())

    records = []
    for bits in (8, 16):
        for scheme in SCHEMES:
            rows = evaluate(bits, scheme, train, test, weights, biases)
            records.extend(rows)
            worst = max(rows, key=lambda row: row["nrmse_pct"])
            print(f"INT{bits:2} {scheme:12} worst_layer={worst['layer']:2} "
                  f"NRMSE={worst['nrmse_pct']:.5f}% "
                  f"max_clip_x={max(row['x_clip_pct'] for row in rows):.3f}%")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (args.output.with_suffix(".csv")).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "source_checkpoint": str(ROOT.parent / "资料/ARM_KWT-1_237e4097/models_data_v2_12_labels/kwt1/best_weights"),
        "dataset_archive": str(ROOT.parent / "资料/数据集/speech_commands_v0.02.tar.gz"),
        "metric": "per-layer sqrt(sum((quantized-fp32)^2)/sum(fp32^2)) over all validation samples and 99x256 outputs; report max of 12 layers",
        "calibration_paths": paths[train_ids].tolist(),
        "validation_paths": paths[test_ids].tolist(),
        "rounding": "nearest, ties to even; signed symmetric range [-qmax,qmax]",
        "integer_path": "qx @ qw + qb in int32 for INT8 and int64 for INT16; dequantization only after integer accumulation",
        "schemes": SCHEMES,
        "results": [{"bits": bits, "scheme": scheme,
                     "worst_layer": max((r for r in records if r["bits"] == bits and r["scheme"] == scheme),
                                        key=lambda r: r["nrmse_pct"])["layer"],
                     "worst_nrmse_pct": max(r["nrmse_pct"] for r in records if r["bits"] == bits and r["scheme"] == scheme)}
                    for bits in (8, 16) for scheme in SCHEMES],
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {args.output.with_suffix('.csv')} and {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
