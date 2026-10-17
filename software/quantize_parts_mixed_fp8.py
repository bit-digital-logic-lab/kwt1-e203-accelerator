"""Partitioned INT8, mixed INT8/INT16 and FP8 FFN1 numeric experiments."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from quantize_ffn1 import collect_activations, quantize, scale_from_abs
from run import KWT1, PREFIX, ROOT


METHODS = (
    "int8_layer", "int8_out16", "int8_out1", "int8_k16",
    "mix_layer2", "mix_layer4", "mix_out10", "mix_out25",
    "fp8_e4m3_raw", "fp8_e4m3_scaled", "fp8_e4m3_out16", "fp8_e4m3_out1",
    "fp8_e5m2_raw", "fp8_e5m2_scaled", "fp8_e5m2_out1",
)


def int_linear(x, w, b, sx, sw, bits):
    """Integer operands, integer bias and integer dot; float only at output."""
    qmax = 2 ** (bits - 1) - 1
    dtype = np.int32 if bits == 8 else np.int64
    qx = quantize(x, sx, qmax, dtype)
    qw = quantize(w, sw, qmax, dtype)
    qb = quantize(b, sx * sw, np.iinfo(dtype).max, dtype)
    acc = qx.reshape(-1, 64) @ qw + qb
    peak = int(np.max(np.abs(acc)))
    if bits == 8 and peak > np.iinfo(np.int32).max:
        raise OverflowError("int32 accumulator overflow")
    y = acc.astype(np.float32).reshape(len(x), 99, 256) * np.asarray(sx * sw, dtype=np.float32)
    return y, peak


def int_k16(x, w, b, train_x):
    """Four integer K=16 partial sums, then float32 cross-group reduction."""
    output = np.zeros((len(x), 99, 256), dtype=np.float32)
    peak = 0
    for start in range(0, 64, 16):
        xs, ws = x[:, :, start:start + 16], w[start:start + 16]
        sx = scale_from_abs(train_x[:, :, start:start + 16], 127)
        sw = scale_from_abs(ws, 127)
        qx = quantize(xs, sx, 127, np.int32)
        qw = quantize(ws, sw, 127, np.int32)
        acc = qx.reshape(-1, 16) @ qw
        peak = max(peak, int(np.max(np.abs(acc))))
        output += acc.reshape(len(x), 99, 256).astype(np.float32) * np.float32(sx * sw)
    return output + b, peak


def fp8_cast_scaled(values, dtype, scale, limit):
    normalized = np.clip(values.astype(np.float32) / np.float32(scale), -limit, limit)
    encoded = torch.from_numpy(np.ascontiguousarray(normalized)).to(dtype)
    decoded = encoded.to(torch.float32).numpy() * np.float32(scale)
    return decoded


def fp8_linear(x, w, b, train_x, fmt, scaled, group_size=None):
    dtype, limit = (torch.float8_e4m3fn, 448.0) if fmt == "e4m3" else (torch.float8_e5m2, 57344.0)
    sx = scale_from_abs(train_x, limit) if scaled else 1.0
    if group_size is not None:
        sw = np.repeat([scale_from_abs(w[:, j:j + group_size], limit)
                        for j in range(0, 256, group_size)], group_size)
    else:
        sw = scale_from_abs(w, limit) if scaled else 1.0
    qx = fp8_cast_scaled(x, dtype, sx, limit)
    qw = fp8_cast_scaled(w, dtype, sw, limit)
    # FP8 operands, FP32 multiply/accumulate and FP32 bias; no native FP8 GEMM.
    return qx @ qw + b


def nrmse_pct(y, reference):
    diff = y.astype(np.float64) - reference.astype(np.float64)
    base = reference.astype(np.float64)
    return float(100 * np.sqrt(np.sum(diff * diff) / np.sum(base * base)))


def column_error(y, reference):
    diff = y.astype(np.float64) - reference.astype(np.float64)
    base = reference.astype(np.float64)
    return np.sum(diff * diff, axis=(0, 1)) / np.maximum(np.sum(base * base, axis=(0, 1)), 1e-30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=ROOT / ".build/quant_samples_10.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "results/ffn1_parts_mixed_fp8")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    model = KWT1(ROOT / "weights/kwt1_fp32.npz", device)
    with np.load(args.samples, allow_pickle=False) as data:
        features, partitions, paths = data["features"], data["partitions"], data["paths"]
    train_ids = np.flatnonzero(partitions == "training")
    validation_ids = np.flatnonzero(partitions == "validation")
    if features.dtype != np.float32 or features.shape[1:] != (98, 40) or not len(train_ids) or not len(validation_ids):
        raise ValueError("Expected nonempty training and validation FP32 MFCC samples")
    if set(paths[train_ids]).intersection(paths[validation_ids]):
        raise ValueError("Training/validation overlap")

    activations = collect_activations(features, model)
    train, validation = activations[train_ids], activations[validation_ids]
    per_layer = []
    training_layer_scores = []
    column_choices = {"mix_out10": [], "mix_out25": []}
    layer_peaks = {}

    for i in range(12):
        p = f"{PREFIX}/enc_layers/{i}/ffn/layer_with_weights-0"
        w = model.weight(p + "/kernel").cpu().numpy()
        b = model.weight(p + "/bias").cpu().numpy()
        xt, xv = train[:, i], validation[:, i]
        ref_t, ref_v = xt @ w + b, xv @ w + b
        sx8, sx16 = scale_from_abs(xt, 127), scale_from_abs(xt, 32767)
        sw8, sw16 = scale_from_abs(w, 127), scale_from_abs(w, 32767)
        base_train, _ = int_linear(xt, w, b, sx8, sw8, 8)
        base_val, peak8 = int_linear(xv, w, b, sx8, sw8, 8)
        val16, peak16 = int_linear(xv, w, b, sx16, sw16, 16)
        training_layer_scores.append(nrmse_pct(base_train, ref_t))
        layer_peaks[str(i + 1)] = {"int8": peak8, "int16": peak16}

        # Whole output-column groups share one weight scale; input remains one INT8 tensor.
        sw_out16 = np.repeat([scale_from_abs(w[:, j:j + 16], 127) for j in range(0, 256, 16)], 16)
        sw_out1 = np.array([scale_from_abs(w[:, j], 127) for j in range(256)])
        out16, peak_out16 = int_linear(xv, w, b, sx8, sw_out16, 8)
        out1, peak_out1 = int_linear(xv, w, b, sx8, sw_out1, 8)
        k16, peak_k16 = int_k16(xv, w, b, xt)

        # Select 10% / 25% columns from training-only INT8 errors.
        ranking = np.argsort(-column_error(base_train, ref_t), kind="stable")
        mixed_columns = {}
        for name, ncols in (("mix_out10", 26), ("mix_out25", 64)):
            selected = np.sort(ranking[:ncols])
            column_choices[name].append(selected.tolist())
            result = base_val.copy()
            result[:, :, selected] = val16[:, :, selected]
            mixed_columns[name] = result

        results = {
            "int8_layer": base_val, "int8_out16": out16, "int8_out1": out1,
            "int8_k16": k16, **mixed_columns,
        }
        for fmt in ("e4m3", "e5m2"):
            for scaled in (False, True):
                suffix = "scaled" if scaled else "raw"
                results[f"fp8_{fmt}_{suffix}"] = fp8_linear(xv, w, b, xt, fmt, scaled)
        results["fp8_e4m3_out16"] = fp8_linear(xv, w, b, xt, "e4m3", True, group_size=16)
        results["fp8_e4m3_out1"] = fp8_linear(xv, w, b, xt, "e4m3", True, group_size=1)
        results["fp8_e5m2_out1"] = fp8_linear(xv, w, b, xt, "e5m2", True, group_size=1)
        layer_row = {name: nrmse_pct(value, ref_v) for name, value in results.items()}
        layer_row["int16_layer"] = nrmse_pct(val16, ref_v)
        per_layer.append(layer_row)
        layer_peaks[str(i + 1)].update({"out16": peak_out16, "out1": peak_out1, "k16_partial": peak_k16})
        print(f"layer={i + 1:2} int8={layer_row['int8_layer']:.4f}% "
              f"out1={layer_row['int8_out1']:.4f}% fp8e4={layer_row['fp8_e4m3_scaled']:.4f}%", flush=True)

    layer_rank = np.argsort(-np.array(training_layer_scores), kind="stable")
    layer_choices = {"mix_layer2": np.sort(layer_rank[:2]).tolist(),
                     "mix_layer4": np.sort(layer_rank[:4]).tolist()}
    for name, selected in layer_choices.items():
        for i in range(12):
            per_layer[i][name] = per_layer[i]["int16_layer"] if i in selected else per_layer[i]["int8_layer"]

    records = []
    for method in METHODS:
        for i, scores in enumerate(per_layer):
            records.append({"method": method, "layer": i + 1, "nrmse_pct": scores[method]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("method", "layer", "nrmse_pct"))
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "metric": "per-layer sqrt(sum((approx-fp32)^2)/sum(fp32^2)) over all validation samples and 99x256 outputs; worst layer reported",
        "calibration_paths": paths[train_ids].tolist(),
        "validation_paths": paths[validation_ids].tolist(),
        "selection": "mixed layers and columns selected only from training INT8 output error",
        "layer_choices_1based": {name: [i + 1 for i in idx] for name, idx in layer_choices.items()},
        "column_choices_0based": column_choices,
        "observed_accumulator_peaks": layer_peaks,
        "arithmetic": {
            "int8_out16/int8_out1": "int32 MAC, per-output-group/column weight scale, integer bias, output dequantization",
            "int8_k16": "four int32 partial MACs, FP32 cross-group reduction and FP32 bias",
            "mix_layer/mix_out": "INT8 int32 MAC or INT16 int64 MAC by selected layer/column, integer bias, output dequantization",
            "fp8": "PyTorch FP8 operand cast and decode, FP32 matrix multiply/accumulate and FP32 bias; no native FP8 GEMM",
        },
        "results": [{"method": method, "worst_layer": int(np.argmax([r[method] for r in per_layer]) + 1),
                     "worst_nrmse_pct": float(max(r[method] for r in per_layer))} for method in METHODS],
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    for item in summary["results"]:
        print(f"{item['method']:18} worst_layer={item['worst_layer']:2} NRMSE={item['worst_nrmse_pct']:.5f}%")
    print(f"Wrote {args.output.with_suffix('.csv')} and {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
