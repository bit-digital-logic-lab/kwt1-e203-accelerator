#!/usr/bin/env python3
"""Single-16-bit-container FFN1 headroom comparison and INT32 bound map.

Contest scope: only the first FFN1 linear layer of each of the 12 encoder
layers is quantized, ``99x64 @ 64x256 + bias -> 99x256``.  Everything else
(the audio front end and the full FP32 model that produces each layer's
``99x64`` input) stays FP32 and is treated as an offline reference.

Two questions are answered with one frozen parameter set:

1. Which (activation bits, weight bits) points inside a *single 16-bit word
   per operand* give the best verification-split NRMSE, i.e. is the current
   A12W10 choice actually limited by activation bits or by weight bits?
2. For which (A, W) does the conservative signed-INT32 accumulator bound hold,
   i.e. where is the real overflow edge of the container?

Rules honoured here:

* Calibration uses the training split only (per-layer activation max and
  per-output-channel weight max from the frozen FP32 weights).
* The validation split is only *evaluated*.  No scale search, no gradients,
  no retraining and no test split are used.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from calibrate_blackbox_ffn1 import (base_scales, black_box_error, encode_params,
                                     sha256)
from quantize_ffn1 import collect_activations
from run import KWT1, PREFIX, ROOT

LAYERS, M, K, N = 12, 99, 64, 256
INT32_MAX = (1 << 31) - 1
# A12W10 and A11W10 are the already published 40/class baselines; they are
# re-run here as in-pipeline controls, not as new schemes.
DEFAULT_SCHEMES = ("A11W10", "A12W10", "A12W12", "A16W10", "A14W12")
# Grid for the analytic conservative-bound map (no GEMM, just scale math).
BOUND_GRID_A = (8, 10, 11, 12, 13, 14, 15, 16)
BOUND_GRID_W = (8, 9, 10, 11, 12, 13, 14, 15, 16)


def parse_scheme(name):
    if not name.startswith("A") or "W" not in name:
        raise argparse.ArgumentTypeError(f"bad scheme {name!r}, expected e.g. A12W10")
    a_text, w_text = name[1:].split("W", 1)
    a_bits, w_bits = int(a_text), int(w_text)
    if not (2 <= a_bits <= 16 and 2 <= w_bits <= 16):
        raise argparse.ArgumentTypeError("bit widths must be in 2..16")
    return a_bits, w_bits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path,
                        default=ROOT / ".build/quant_samples_12labels_40.npz")
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/kwt1_fp32.npz")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/ffn1_container_headroom_12labels_40")
    parser.add_argument("--schemes", nargs="+", type=parse_scheme,
                        default=[parse_scheme(s) for s in DEFAULT_SCHEMES])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if len({f"A{a}W{w}" for a, w in args.schemes}) != len(args.schemes):
        parser.error("--schemes contains duplicates")

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    with np.load(args.samples, allow_pickle=False) as data:
        features, partitions, paths, words = (data["features"], data["partitions"],
                                              data["paths"], data["words"])
    if features.dtype != np.float32 or features.ndim != 3 or features.shape[1:] != (98, 40):
        raise ValueError("expected finite FP32 MFCC [samples,98,40]")
    if not np.isfinite(features).all():
        raise ValueError("features contain NaN or infinity")
    if np.any((partitions != "training") & (partitions != "validation")):
        raise ValueError("testing/other partitions are forbidden")
    train_ids = np.flatnonzero(partitions == "training")
    val_ids = np.flatnonzero(partitions == "validation")
    if not len(train_ids) or not len(val_ids):
        raise ValueError("insufficient train or validation samples")
    if set(paths[train_ids]).intersection(paths[val_ids]):
        raise ValueError("train/validation sample paths overlap")

    model = KWT1(args.weights, device)
    activations = collect_activations(features, model)
    if activations.shape != (len(features), LAYERS, M, K):
        raise ValueError(f"unexpected FFN1 input tensor {activations.shape}")

    records, bound_grid = [], []
    for layer in range(LAYERS):
        prefix = f"{PREFIX}/enc_layers/{layer}/ffn/layer_with_weights-0"
        weight = model.weight(prefix + "/kernel").cpu().numpy()
        bias = model.weight(prefix + "/bias").cpu().numpy()
        train_x = activations[train_ids, layer]
        val_x = activations[val_ids, layer]
        reference = val_x @ weight + bias  # FP32 reference, evaluation only.

        for a_bits, w_bits in args.schemes:
            qmax_a = (1 << (a_bits - 1)) - 1
            qmax_w = (1 << (w_bits - 1)) - 1
            sx, sw = base_scales(train_x, weight, a_bits, w_bits)
            result = black_box_error(val_x, reference, weight, bias, sx, sw,
                                     a_bits, w_bits, args.batch_size)
            _, qb = encode_params(weight, bias, sx, sw, a_bits, w_bits)
            bound = K * qmax_a * qmax_w + int(np.max(np.abs(qb)))
            if bound > INT32_MAX:
                raise OverflowError(f"layer {layer+1} A{a_bits}W{w_bits} "
                                    f"conservative bound {bound} exceeds INT32")
            # Clipping is evidence about the frozen scale, not a design knob.
            qx_all = np.rint(val_x.astype(np.float64) / sx)
            x_clip_pct = 100.0 * np.count_nonzero(np.abs(qx_all) > qmax_a) / qx_all.size
            records.append({
                "scheme": f"A{a_bits}W{w_bits}", "a_bits": a_bits, "w_bits": w_bits,
                "layer": layer + 1,
                "validation_nrmse_pct": result["nrmse_pct"],
                "observed_accumulator_peak": result["max_abs_accumulator"],
                "observed_accumulator_min": result["min_accumulator"],
                "observed_accumulator_max": result["max_accumulator"],
                "worst_case_abs_accumulator_bound": bound,
                "bound_headroom_over_int32": bound / (INT32_MAX + 1),
                "activation_scale": sx, "bias_abs_max": int(np.max(np.abs(qb))),
                "activation_clip_pct_val": x_clip_pct,
                "packed_weight_kib_12layers": LAYERS * K * N * w_bits / 8 / 1024,
                "int16_container_weight_bytes_12layers": LAYERS * K * N * 2,
                "mac_input_bits": f"{a_bits}x{w_bits}",
            })
        print(f"layer={layer+1:02} done", flush=True)

    # Analytic conservative-bound map: same frozen scales, no GEMM.
    for a_bits in BOUND_GRID_A:
        for w_bits in BOUND_GRID_W:
            qmax_a = (1 << (a_bits - 1)) - 1
            qmax_w = (1 << (w_bits - 1)) - 1
            worst_bound, worst_layer = 0, 0
            for layer in range(LAYERS):
                prefix = f"{PREFIX}/enc_layers/{layer}/ffn/layer_with_weights-0"
                weight = model.weight(prefix + "/kernel").cpu().numpy()
                bias = model.weight(prefix + "/bias").cpu().numpy()
                sx, sw = base_scales(activations[train_ids, layer], weight, a_bits, w_bits)
                _, qb = encode_params(weight, bias, sx, sw, a_bits, w_bits)
                bound = K * qmax_a * qmax_w + int(np.max(np.abs(qb)))
                if bound > worst_bound:
                    worst_bound, worst_layer = bound, layer + 1
            bound_grid.append({"a_bits": a_bits, "w_bits": w_bits,
                               "worst_layer": worst_layer,
                               "worst_case_abs_accumulator_bound": worst_bound,
                               "fits_int32": worst_bound <= INT32_MAX,
                               "packed_weight_kib_12layers": LAYERS * K * N * w_bits / 8 / 1024})

    summary = []
    for a_bits, w_bits in args.schemes:
        scheme = f"A{a_bits}W{w_bits}"
        subset = [r for r in records if r["scheme"] == scheme]
        worst = max(subset, key=lambda r: r["validation_nrmse_pct"])
        summary.append({
            "scheme": scheme, "worst_layer": worst["layer"],
            "worst_validation_nrmse_pct": worst["validation_nrmse_pct"],
            "max_observed_accumulator": max(r["observed_accumulator_peak"] for r in subset),
            "max_abs_accumulator_bound": max(r["worst_case_abs_accumulator_bound"]
                                             for r in subset),
            "max_bound_headroom_over_int32": max(r["bound_headroom_over_int32"]
                                                 for r in subset),
            "packed_weight_kib_12layers": worst["packed_weight_kib_12layers"],
            "int16_container_weight_bytes_12layers": worst["int16_container_weight_bytes_12layers"],
            "mac_input_bits": worst["mac_input_bits"],
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with args.output.with_name(args.output.name + "_bounds").with_suffix(".csv").open(
            "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bound_grid[0]))
        writer.writeheader()
        writer.writerows(bound_grid)
    meta = {
        "experiment": "single-16-bit-container FFN1 headroom, no-gradient training-only scales",
        "scope": "only the 12 FFN1 first linear layers 99x64 @ 64x256 + bias",
        "metric": ("per-layer RMS(error)/max(RMS(FP32),1e-8) over all validation samples "
                   "and 99x256 outputs; worst of the 12 layers"),
        "calibration": ("per-layer activation max over the training split; per-output-channel "
                        "weight max over the frozen FP32 weights; zero point 0, symmetric"),
        "validation_used_for_calibration": False,
        "test_partition_used": False,
        "rounding": "nearest, ties-to-even; clip to [-qmax,+qmax]",
        "arithmetic": ("signed integer A/W GEMM, INT32 bias, INT32 accumulate; FP64 decode "
                       "only to measure error"),
        "sample_archive": str(args.samples.resolve()), "sample_sha256": sha256(args.samples),
        "weight_archive": str(args.weights.resolve()), "weight_sha256": sha256(args.weights),
        "training_count": int(len(train_ids)), "validation_count": int(len(val_ids)),
        "schemes": [f"A{a}W{w}" for a, w in args.schemes],
        "summary": summary, "layers": records, "int32_bound_grid": bound_grid,
    }
    args.output.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    for item in summary:
        print(f"{item['scheme']:6} worst={item['worst_validation_nrmse_pct']:.6f}% "
              f"layer={item['worst_layer']:2} bound={item['max_abs_accumulator_bound']:>12} "
              f"headroom={100*item['max_bound_headroom_over_int32']:.2f}% "
              f"peak={item['max_observed_accumulator']:>12}")
    print(f"PASS wrote {args.output.with_suffix('.csv')} and {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
