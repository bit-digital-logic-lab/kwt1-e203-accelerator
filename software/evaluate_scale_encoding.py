"""Evaluate compact output-scale encoding for the frozen A11W10 FFN1 model."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from calibrate_blackbox_ffn1 import sha256
from quantize_ffn1 import collect_activations
from run import KWT1, PREFIX, ROOT


def encode_scale(scale):
    mantissa, exponent = np.frexp(np.asarray(scale, dtype=np.float64))
    code = np.rint(mantissa * 32768).astype(np.int64)
    carry = code == 32768
    code[carry] = 16384
    exponent[carry] += 1
    if np.any(code < 16384) or np.any(code > 32767):
        raise ValueError("invalid Q15 mantissa")
    if np.any(exponent < -128) or np.any(exponent > 127):
        raise OverflowError("scale exponent does not fit int8")
    return code.astype(np.uint16), exponent.astype(np.int8)


def decode_scale(code, exponent):
    return np.ldexp(code.astype(np.float64), exponent.astype(np.int32) - 15)


def evaluate(x, weight, bias, sx, qw, qb, scale, batch_size):
    qx_max = (1 << 10) - 1
    reference = x @ weight + bias
    numerator = denominator = 0.0
    acc_peak = 0
    for begin in range(0, len(x), batch_size):
        end = min(len(x), begin + batch_size)
        qx = np.clip(np.rint(x[begin:end].astype(np.float64) / sx),
                     -qx_max, qx_max).astype(np.int64)
        acc = (qx.reshape(-1, 64) @ qw).reshape(end - begin, 99, 256) + qb
        if acc.min() < np.iinfo(np.int32).min or acc.max() > np.iinfo(np.int32).max:
            raise OverflowError("INT32 accumulation overflow")
        acc_peak = max(acc_peak, int(np.max(np.abs(acc))))
        decoded = acc.astype(np.float64) * scale
        ref = reference[begin:end].astype(np.float64)
        numerator += float(np.sum((decoded - ref) ** 2))
        denominator += float(np.sum(ref ** 2))
    count = len(x) * 99 * 256
    return 100 * np.sqrt(numerator / max(denominator, count * 1e-16)), acc_peak


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path,
                        default=ROOT / ".build/quant_samples_12labels_20.npz")
    parser.add_argument("--quantized", type=Path,
                        default=ROOT / "results/ffn1_blackbox_a11w10.npz")
    parser.add_argument("--weights", type=Path,
                        default=ROOT / "weights/kwt1_fp32.npz")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/ffn1_a11w10_q15_scales")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    with np.load(args.samples, allow_pickle=False) as source:
        features = source["features"]
        partitions = source["partitions"]
        paths = source["paths"]
    if features.ndim != 3 or features.shape[1:] != (98, 40) or not np.isfinite(features).all():
        raise ValueError("expected finite [samples,98,40] features")
    if np.any((partitions != "training") & (partitions != "validation")):
        raise ValueError("test partition is forbidden")
    train_ids = np.flatnonzero(partitions == "training")
    validation_ids = np.flatnonzero(partitions == "validation")
    if len(train_ids) == 0 or len(validation_ids) == 0:
        raise ValueError("missing training/validation partition")
    if set(paths[train_ids]).intersection(paths[validation_ids]):
        raise ValueError("training/validation paths overlap")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device)
    model = KWT1(args.weights, torch.device(device))
    activations = collect_activations(features[validation_ids], model)
    records = []
    encoded = {}
    with np.load(args.quantized, allow_pickle=False) as frozen:
        for layer in range(12):
            base = f"{PREFIX}/enc_layers/{layer}/ffn/layer_with_weights-0"
            weight = model.weight(base + "/kernel").cpu().numpy()
            bias = model.weight(base + "/bias").cpu().numpy()
            prefix = f"layer{layer + 1:02}"
            qw = frozen[prefix + "_weight"].astype(np.int64)
            qb = frozen[prefix + "_bias"].astype(np.int64)
            sx = float(frozen[prefix + "_activation_scale"])
            sw = frozen[prefix + "_weight_scales"]
            if qw.shape != (64, 256) or qb.shape != (256,) or sw.shape != (256,):
                raise ValueError("frozen parameter shape mismatch")
            original_scale = sx * sw
            code, exponent = encode_scale(original_scale)
            compact_scale = decode_scale(code, exponent)
            x = activations[:, layer]
            baseline, peak_a = evaluate(x, weight, bias, sx, qw, qb,
                                        original_scale, args.batch_size)
            compact, peak_b = evaluate(x, weight, bias, sx, qw, qb,
                                       compact_scale, args.batch_size)
            if peak_a != peak_b:
                raise AssertionError("integer accumulator changed")
            records.append({"layer": layer + 1, "baseline_nrmse_pct": baseline,
                            "q15_nrmse_pct": compact, "delta_pp": compact - baseline,
                            "max_scale_relative_error_pct": float(np.max(np.abs(
                                compact_scale / original_scale - 1)) * 100),
                            "max_abs_accumulator": peak_a})
            encoded[prefix + "_scale_mantissa_q15"] = code
            encoded[prefix + "_scale_exponent_i8"] = exponent
            print(f"layer={layer + 1:02} baseline={baseline:.5f}% q15={compact:.5f}%", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    np.savez_compressed(args.output.with_suffix(".npz"), **encoded)
    worst_base = max(records, key=lambda r: r["baseline_nrmse_pct"])
    worst_q15 = max(records, key=lambda r: r["q15_nrmse_pct"])
    meta = {"experiment": "frozen A11W10 integer operands/bias, Q15 mantissa plus int8 exponent output scale",
            "scale_decode": "scale = mantissa_q15 * 2 ** (exponent_i8 - 15)",
            "rounding": "numpy rint, nearest ties to even",
            "samples_sha256": sha256(args.samples),
            "frozen_quantized_sha256": sha256(args.quantized),
            "weights_sha256": sha256(args.weights),
            "validation_samples": len(validation_ids),
            "worst_baseline_nrmse_pct": worst_base["baseline_nrmse_pct"],
            "worst_q15_nrmse_pct": worst_q15["q15_nrmse_pct"],
            "worst_q15_layer": worst_q15["layer"],
            "scale_storage_bytes_all_layers": 12 * 256 * 3,
            "float32_scale_storage_bytes_all_layers": 12 * 256 * 4,
            "layers": records}
    args.output.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    print(f"WORST baseline={worst_base['baseline_nrmse_pct']:.5f}% "
          f"q15={worst_q15['q15_nrmse_pct']:.5f}% layer={worst_q15['layer']}")


if __name__ == "__main__":
    main()
