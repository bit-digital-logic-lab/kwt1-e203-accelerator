"""No-gradient A/W bit-width and activation-scale search for FFN1 NRMSE."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from quantize_ffn1 import collect_activations, quantize, scale_from_abs
from quantize_parts_mixed_fp8 import nrmse_pct
from run import KWT1, PREFIX, ROOT


METHODS = (
    (8, 8, "static_sym"), (8, 8, "static_affine"),
    (8, 8, "row_sym"), (8, 8, "row_affine"),
    (9, 8, "static_sym"), (10, 8, "static_sym"),
    (9, 8, "row_affine"), (10, 8, "row_affine"),
    (8, 9, "row_affine"), (8, 10, "row_affine"),
    (9, 9, "row_affine"), (8, 16, "row_affine"),
    (16, 8, "static_sym"),
    (10, 9, "static_sym"), (10, 10, "static_sym"),
    (11, 8, "static_sym"), (12, 8, "static_sym"),
    (9, 10, "row_affine"), (10, 9, "row_affine"),
    (10, 10, "row_affine"), (11, 9, "static_sym"),
    (12, 9, "static_sym"), (9, 12, "row_affine"),
    (11, 10, "static_sym"), (12, 10, "static_sym"),
    (10, 11, "static_sym"),
)
CALIBRATED_METHODS = ((8, 8, "row_affine"), (9, 8, "row_affine"),
                      (10, 8, "static_sym"), (9, 9, "row_affine"),
                      (10, 9, "static_sym"), (10, 10, "static_sym"))


def activation_params(x, calibration, bits, mode):
    qmax = 2 ** (bits - 1) - 1
    qmin = -(2 ** (bits - 1))
    if mode == "static_sym":
        return np.float32(scale_from_abs(calibration, qmax)), np.int32(0)
    if mode == "static_affine":
        low, high = float(np.min(calibration)), float(np.max(calibration))
        scale = np.float32(max((high - low) / (qmax - qmin), 1e-12))
        zero = np.int32(np.clip(np.rint(qmin - low / scale), qmin, qmax))
        return scale, zero
    if mode == "row_sym":
        scale = np.maximum(np.max(np.abs(x), axis=-1) / qmax, 1e-12).astype(np.float32)
        return scale[..., None], np.zeros((*scale.shape, 1), dtype=np.int32)
    if mode == "row_affine":
        low, high = np.min(x, axis=-1), np.max(x, axis=-1)
        scale = np.maximum((high - low) / (qmax - qmin), 1e-12).astype(np.float32)
        zero = np.clip(np.rint(qmin - low / scale), qmin, qmax).astype(np.int32)
        return scale[..., None], zero[..., None]
    raise ValueError(mode)


def quant_activation(x, scale, zero, bits):
    qmax = 2 ** (bits - 1) - 1
    qmin = -(2 ** (bits - 1))
    return np.clip(np.rint(x.astype(np.float64) / scale) + zero, qmin, qmax).astype(np.int32)


def calibrate_weight_scales(w, calibration, bits):
    """Select per-column scales minimizing training X-weight-error energy."""
    qmax = 2 ** (bits - 1) - 1
    base = np.array([scale_from_abs(w[:, j], qmax) for j in range(256)])
    factors = np.linspace(0.70, 1.10, 17)
    scales = base[None, :] * factors[:, None]
    quantized = np.clip(np.rint(w[None, :, :] / scales[:, None, :]), -qmax, qmax)
    error = quantized * scales[:, None, :] - w[None, :, :]
    x = calibration.reshape(-1, 64).astype(np.float64)
    covariance = (x.T @ x) / len(x)
    loss = np.einsum("fkc,kl,flc->fc", error, covariance, error, optimize=True)
    return scales[np.argmin(loss, axis=0), np.arange(256)].astype(np.float32)


def integer_linear(x, w, b, calibration, a_bits, w_bits, mode, weight_scales=None):
    """Integer GEMM, zero-point correction and integer bias; dequant at output."""
    scale_x, zero = activation_params(x, calibration, a_bits, mode)
    qx = quant_activation(x, scale_x, zero, a_bits)
    qmax_w = 2 ** (w_bits - 1) - 1
    scale_w = (np.array([scale_from_abs(w[:, j], qmax_w) for j in range(256)], dtype=np.float32)
               if weight_scales is None else weight_scales)
    accumulator_dtype = np.int64 if max(a_bits, w_bits) >= 16 else np.int32
    qw = quantize(w, scale_w, qmax_w, accumulator_dtype)
    qx = qx.astype(accumulator_dtype)
    zero = np.asarray(zero, dtype=accumulator_dtype)
    product = (qx.reshape(-1, 64) @ qw).reshape(len(x), 99, 256)
    correction = zero * np.sum(qw, axis=0)
    scale_out = scale_x * scale_w
    qb = np.rint(b / scale_out).astype(accumulator_dtype)
    acc = product - correction + qb
    peak = int(np.max(np.abs(acc)))
    if accumulator_dtype == np.int32 and peak > np.iinfo(np.int32).max:
        raise OverflowError("int32 accumulator overflow")
    y = acc.astype(np.float32) * scale_out
    return y, peak


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=ROOT / ".build/quant_samples_10.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "results/ffn1_target_06")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--target", type=float, default=0.3,
                        help="NRMSE percent target for the printed pass/fail result")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Optional method names, e.g. A11W10:static_sym:max")
    args = parser.parse_args()
    candidates = ([(a, w, m, "max") for a, w, m in METHODS] +
                  [(a, w, m, "train_covariance") for a, w, m in CALIBRATED_METHODS])
    if args.only:
        selected = set(args.only)
        candidates = [c for c in candidates
                      if f"A{c[0]}W{c[1]}:{c[2]}:{c[3]}" in selected]
        if len(candidates) != len(selected):
            raise ValueError("Unknown method in --only")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    model = KWT1(ROOT / "weights/kwt1_fp32.npz", device)
    with np.load(args.samples, allow_pickle=False) as data:
        features, partitions, paths = data["features"], data["partitions"], data["paths"]
    train_ids = np.flatnonzero(partitions == "training")
    validation_ids = np.flatnonzero(partitions == "validation")
    if not len(train_ids) or not len(validation_ids) or set(paths[train_ids]).intersection(paths[validation_ids]):
        raise ValueError("Need disjoint training and validation samples")
    inputs = collect_activations(features, model)
    train, validation = inputs[train_ids], inputs[validation_ids]
    records = []
    for layer in range(12):
        p = f"{PREFIX}/enc_layers/{layer}/ffn/layer_with_weights-0"
        w = model.weight(p + "/kernel").cpu().numpy()
        b = model.weight(p + "/bias").cpu().numpy()
        x = validation[:, layer]
        reference = x @ w + b
        for a_bits, w_bits, mode, weight_mode in candidates:
            if weight_mode != "max":
                continue
            y, peak = integer_linear(x, w, b, train[:, layer], a_bits, w_bits, mode)
            records.append({"a_bits": a_bits, "w_bits": w_bits, "activation_mode": mode,
                            "weight_scale_mode": "max",
                            "layer": layer + 1, "nrmse_pct": nrmse_pct(y, reference),
                            "max_abs_accumulator": peak})
        calibrated_scales = {bits: calibrate_weight_scales(w, train[:, layer], bits)
                             for bits in {w_bits for _, w_bits, _, weight_mode in candidates
                                          if weight_mode == "train_covariance"}}
        for a_bits, w_bits, mode, weight_mode in candidates:
            if weight_mode != "train_covariance":
                continue
            y, peak = integer_linear(x, w, b, train[:, layer], a_bits, w_bits, mode,
                                     weight_scales=calibrated_scales[w_bits])
            records.append({"a_bits": a_bits, "w_bits": w_bits, "activation_mode": mode,
                            "weight_scale_mode": "train_covariance",
                            "layer": layer + 1, "nrmse_pct": nrmse_pct(y, reference),
                            "max_abs_accumulator": peak})
        print(f"layer={layer + 1:2} done", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    summary = []
    for a_bits, w_bits, mode, weight_mode in candidates:
        subset = [r for r in records if (r["a_bits"], r["w_bits"], r["activation_mode"],
                                         r["weight_scale_mode"]) == (a_bits, w_bits, mode, weight_mode)]
        worst = max(subset, key=lambda r: r["nrmse_pct"])
        summary.append({"a_bits": a_bits, "w_bits": w_bits, "activation_mode": mode,
                        "weight_scale_mode": weight_mode,
                        "worst_layer": worst["layer"], "worst_nrmse_pct": worst["nrmse_pct"],
                        "max_abs_accumulator": max(r["max_abs_accumulator"] for r in subset)})
    meta = {"target_nrmse_pct": args.target,
            "metric": "per-layer sqrt(sum((quantized-fp32)^2)/sum(fp32^2)) over all validation samples and 99x256 outputs; worst layer reported",
            "training_paths": paths[train_ids].tolist(), "validation_paths": paths[validation_ids].tolist(),
            "weights": "original FP32 KWT-1 weights quantized per output column; max or training-covariance scale selection; no weight or bias correction",
            "rounding": "nearest, ties to even; signed symmetric weight quantization; activation symmetric or signed affine",
            "arithmetic": "integer matrix multiply, integer activation zero-point correction, integer bias, final FP32 dequantization for error measurement",
            "results": summary}
    args.output.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    for r in summary:
        print(f"A{r['a_bits']:2}W{r['w_bits']:2} {r['activation_mode']:13} {r['weight_scale_mode']:16} "
              f"worst={r['worst_nrmse_pct']:.5f}% layer={r['worst_layer']:2} "
              f"{'PASS' if r['worst_nrmse_pct'] <= args.target else 'FAIL'}")


if __name__ == "__main__":
    main()
