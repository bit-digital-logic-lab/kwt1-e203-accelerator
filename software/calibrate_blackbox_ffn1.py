"""Derivative-free, training-only scale calibration for the contest FFN1."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from quantize_ffn1 import collect_activations
from run import KWT1, PREFIX, ROOT


LAYERS, M, K, N = 12, 99, 64, 256
DEFAULT_FACTORS = (0.80, 0.90, 1.00, 1.10, 1.20)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantize_symmetric(values, scale, qmax):
    return np.clip(np.rint(np.asarray(values, dtype=np.float64) / scale),
                   -qmax, qmax).astype(np.int64)


def base_scales(training_x, weight, a_bits, w_bits):
    # All scales depend only on training activations and the frozen weights.
    qmax_a, qmax_w = (1 << (a_bits - 1)) - 1, (1 << (w_bits - 1)) - 1
    sx = max(float(np.max(np.abs(training_x.astype(np.float64)))) / qmax_a, 1e-12)
    sw = np.maximum(np.max(np.abs(weight.astype(np.float64)), axis=0) / qmax_w,
                    1e-12)
    return sx, sw


def encode_params(weight, bias, sx, sw, a_bits, w_bits):
    qmax_w = (1 << (w_bits - 1)) - 1
    qw = quantize_symmetric(weight, sw, qmax_w)
    qb = np.rint(bias.astype(np.float64) / (sx * sw)).astype(np.int64)
    if np.any(qb < np.iinfo(np.int32).min) or np.any(qb > np.iinfo(np.int32).max):
        raise OverflowError("quantized bias does not fit signed INT32")
    return qw, qb


def black_box_error(x, reference, weight, bias, sx, sw, a_bits, w_bits,
                    batch_size=32):
    """Return a real integer-GEMM error; the search never sees gradients."""
    qw, qb = encode_params(weight, bias, sx, sw, a_bits, w_bits)
    qmax_a = (1 << (a_bits - 1)) - 1
    output_scale = sx * sw
    error_square = 0.0
    reference_square = 0.0
    peak = 0
    min_acc, max_acc = 0, 0
    for begin in range(0, len(x), batch_size):
        end = min(begin + batch_size, len(x))
        qx = quantize_symmetric(x[begin:end], sx, qmax_a)
        acc = (qx.reshape(-1, K) @ qw).reshape(end - begin, M, N) + qb
        min_acc = min(min_acc, int(acc.min()))
        max_acc = max(max_acc, int(acc.max()))
        peak = max(peak, int(np.max(np.abs(acc))))
        # Only this final decode is floating point; MAC and bias are integers.
        approximate = acc.astype(np.float64) * output_scale
        ref = reference[begin:end].astype(np.float64)
        error_square += float(np.sum((approximate - ref) ** 2))
        reference_square += float(np.sum(ref ** 2))
    if min_acc < np.iinfo(np.int32).min or max_acc > np.iinfo(np.int32).max:
        raise OverflowError("integer MAC plus bias exceeds signed INT32")
    count = x.size // K * N
    denom = max(np.sqrt(reference_square / count), 1e-8)
    return {"nrmse_pct": 100 * np.sqrt(error_square / count) / denom,
            "max_abs_accumulator": peak,
            "min_accumulator": min_acc, "max_accumulator": max_acc}


def tune_layer(training_x, weight, bias, a_bits, w_bits, factors, batch_size):
    reference = training_x @ weight + bias
    sx_base, sw_base = base_scales(training_x, weight, a_bits, w_bits)
    history = []

    def probe(ax, wf, stage):
        result = black_box_error(training_x, reference, weight, bias,
                                 sx_base * ax, sw_base * wf, a_bits, w_bits,
                                 batch_size)
        history.append({"stage": stage, "activation_factor": ax,
                        "weight_factor": wf, "training_nrmse_pct": result["nrmse_pct"]})
        return result["nrmse_pct"]

    baseline = probe(1.0, 1.0, "baseline")
    # One coordinate round: tune activation scale, then weight scales. The
    # entire candidate choice is made on the training partition.
    ax = min(factors, key=lambda factor: (probe(factor, 1.0, "activation"),
                                          abs(factor - 1.0)))
    wf = min(factors, key=lambda factor: (probe(ax, factor, "weight"),
                                          abs(factor - 1.0)))
    selected = probe(ax, wf, "selected")
    return {"sx": sx_base * ax, "sw": sw_base * wf, "ax_factor": ax,
            "w_factor": wf, "baseline_train_nrmse_pct": baseline,
            "tuned_train_nrmse_pct": selected,
            "search": history}


def load_dataset(path):
    with np.load(path, allow_pickle=False) as data:
        features = data["features"].copy()
        partitions = data["partitions"].copy()
        paths = data["paths"].copy()
        words = data["words"].copy() if "words" in data else None
    if features.ndim != 3 or features.shape[1:] != (98, 40) or not np.isfinite(features).all():
        raise ValueError("dataset needs finite FP32 MFCC [samples,98,40]")
    train_ids = np.flatnonzero(partitions == "training")
    validation_ids = np.flatnonzero(partitions == "validation")
    if not len(train_ids) or not len(validation_ids):
        raise ValueError("dataset needs training and validation partitions")
    if np.any((partitions != "training") & (partitions != "validation")):
        raise ValueError("testing/other partitions must not be supplied")
    if set(paths[train_ids]).intersection(paths[validation_ids]):
        raise ValueError("training and validation paths overlap")
    return features, paths, words, train_ids, validation_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path,
                        default=ROOT / ".build/quant_samples_12labels_20.npz")
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/kwt1_fp32.npz")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/ffn1_blackbox_a11w10")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--activation-bits", type=int, default=11)
    parser.add_argument("--weight-bits", type=int, default=10)
    parser.add_argument("--factors", type=float, nargs="+", default=DEFAULT_FACTORS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--target", type=float, default=3.0, help="contest worst-layer NRMSE percent limit")
    args = parser.parse_args()
    if not (2 <= args.activation_bits <= 16 and 2 <= args.weight_bits <= 16):
        parser.error("bits must be in 2..16")
    if args.batch_size < 1 or not args.factors or any(f <= 0 for f in args.factors):
        parser.error("batch size and scale factors must be positive")
    factors = sorted(set([1.0] + list(args.factors)))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    features, paths, words, train_ids, validation_ids = load_dataset(args.samples)
    model = KWT1(args.weights, device)
    activations = collect_activations(features, model)
    if activations.shape != (len(features), LAYERS, M, K):
        raise ValueError(f"unexpected FFN1 inputs: {activations.shape}")

    records = []
    artifacts = {}
    search_traces = []
    for layer in range(LAYERS):
        base = f"{PREFIX}/enc_layers/{layer}/ffn/layer_with_weights-0"
        weight = model.weight(base + "/kernel").cpu().numpy()
        bias = model.weight(base + "/bias").cpu().numpy()
        if weight.shape != (K, N) or bias.shape != (N,):
            raise ValueError(f"unexpected layer {layer + 1} parameter shape")
        train_x = activations[train_ids, layer]
        tuning = tune_layer(train_x, weight, bias, args.activation_bits,
                            args.weight_bits, factors, args.batch_size)
        sx, sw = tuning["sx"], tuning["sw"]
        qw, qb = encode_params(weight, bias, sx, sw, args.activation_bits,
                               args.weight_bits)
        # Freeze the selected training parameters before accessing validation.
        validation_x = activations[validation_ids, layer]
        reference = validation_x @ weight + bias
        result = black_box_error(validation_x, reference, weight, bias, sx, sw,
                                 args.activation_bits, args.weight_bits, args.batch_size)
        name = f"layer{layer + 1:02}"
        artifacts[name + "_weight"] = qw.astype(np.int16)
        artifacts[name + "_bias"] = qb.astype(np.int32)
        artifacts[name + "_activation_scale"] = np.array(sx, dtype=np.float64)
        artifacts[name + "_weight_scales"] = sw.astype(np.float64)
        record = {"layer": layer + 1, "activation_factor": tuning["ax_factor"],
                  "weight_factor": tuning["w_factor"],
                  "baseline_training_nrmse_pct": tuning["baseline_train_nrmse_pct"],
                  "selected_training_nrmse_pct": tuning["tuned_train_nrmse_pct"],
                  "validation_nrmse_pct": result["nrmse_pct"],
                  "max_abs_accumulator": result["max_abs_accumulator"],
                  "min_accumulator": result["min_accumulator"],
                  "max_accumulator": result["max_accumulator"]}
        records.append(record)
        search_traces.append({"layer": layer + 1, "candidates": tuning["search"]})
        print(f"layer={layer + 1:02} train={record['selected_training_nrmse_pct']:.5f}% "
              f"validation={result['nrmse_pct']:.5f}% "
              f"factors=({tuning['ax_factor']:.3f},{tuning['w_factor']:.3f})",
              flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    np.savez_compressed(args.output.with_suffix(".npz"), **artifacts)
    worst = max(records, key=lambda item: item["validation_nrmse_pct"])
    meta = {
        "experiment": "one derivative-free coordinate round, training NRMSE objective",
        "contest_metric": "per layer RMS(error)/max(RMS(FP32 reference),1e-8), aggregate all validation samples and 99x256 outputs; report worst of 12 layers",
        "contest_target_nrmse_pct": args.target,
        "validation_pass": bool(worst["validation_nrmse_pct"] <= args.target),
        "worst_layer": worst["layer"],
        "worst_validation_nrmse_pct": worst["validation_nrmse_pct"],
        "activation_bits": args.activation_bits, "weight_bits": args.weight_bits,
        "factors": factors, "device": str(device), "batch_size": args.batch_size,
        "sample_archive": str(args.samples.resolve()), "sample_sha256": sha256(args.samples),
        "weight_archive": str(args.weights.resolve()), "weight_sha256": sha256(args.weights),
        "training_paths": paths[train_ids].tolist(),
        "validation_paths": paths[validation_ids].tolist(),
        "training_labels": sorted(set(words[train_ids].tolist())) if words is not None else None,
        "validation_labels": sorted(set(words[validation_ids].tolist())) if words is not None else None,
        "numerical_protocol": "static signed-symmetric activation scale/layer; signed-symmetric weight scale/output column; ties-to-even and clip to [-qmax,qmax]; INT64 host matmul checked to fit INT32 including INT32 bias; dequantize with activation_scale*weight_scale only for NRMSE; no gradient, weight correction, dynamic input scan or test partition",
        "artifact": "NPZ stores 16-bit containers for weights, signed INT32 biases, FP64 scales; 10-bit weights are not bit packed",
        "search_trace": search_traces,
        "layers": records,
    }
    args.output.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    print(f"WORST validation={worst['validation_nrmse_pct']:.5f}% "
          f"layer={worst['layer']} target={args.target:.3f}% "
          f"{'PASS' if meta['validation_pass'] else 'FAIL'}")
    print(f"Saved {args.output.with_suffix('.csv')}, {args.output.with_suffix('.json')}, "
          f"{args.output.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
