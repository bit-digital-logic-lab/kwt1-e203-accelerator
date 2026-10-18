"""A12W10 training-split calibration-method comparison for the contest FFN1.

Frozen hardware/numerical container (identical to the verified A12W10 design):
signed-symmetric zero-point-0 activation A=12 bits, signed-symmetric weight
W=10 bits per output channel, INT32 bias, nearest ties-to-even rounding and
clip-saturation to ``[-qmax, qmax]``, INT64 host accumulation checked to fit
INT32.  Same rounding/saturation rule for every calibration method.

Only the *scale calibration rule* changes.  Three families are compared:

1. ``max``        -- per-layer activation abs-max; per-channel weight abs-max.
2. ``percentile`` -- per-layer / per-channel magnitude percentile with clipping;
                     the percentile is selected on the training split only.
3. ``err_min``    -- derivative-free coordinate search of activation/weight
                     scale factors that minimises the training NRMSE.

No gradients, no weight update, no retraining, no dynamic input scan.
Calibration sees the training partition only; the validation partition is used
only to *evaluate* the frozen scales and is never read during selection.
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SOFTWARE = HERE.parent
sys.path.insert(0, str(SOFTWARE))

from calibrate_blackbox_ffn1 import (  # noqa: E402
    LAYERS, K, M, N, base_scales, encode_params, quantize_symmetric, sha256,
)
from quantize_ffn1 import collect_activations  # noqa: E402
from run import KWT1, PREFIX, ROOT  # noqa: E402


PERCENTILES = (99.0, 99.5, 99.9, 99.99)
ACT_FACTORS = (0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3)
WGT_FACTORS = (0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3)
METHODS = ("max", "percentile", "err_min")


def scale_percentile(train_x, weight, a_bits, w_bits, percentile):
    qmax_a = (1 << (a_bits - 1)) - 1
    qmax_w = (1 << (w_bits - 1)) - 1
    limit_x = float(np.percentile(np.abs(train_x.astype(np.float64)), percentile))
    limit_w = np.percentile(np.abs(weight.astype(np.float64)), percentile, axis=0)
    sx = max(limit_x / qmax_a, 1e-12)
    sw = np.maximum(limit_w / qmax_w, 1e-12)
    return sx, sw


def evaluate(x, reference, weight, bias, sx, sw, a_bits, w_bits, batch_size=64):
    """Genuine integer GEMM + INT32 bias; only final decode is FP64."""
    qw, qb = encode_params(weight, bias, sx, sw, a_bits, w_bits)
    qmax_a = (1 << (a_bits - 1)) - 1
    qmax_w = (1 << (w_bits - 1)) - 1
    output_scale = sx * sw
    error_square = 0.0
    reference_square = 0.0
    max_abs_error = 0.0
    peak = 0
    min_acc, max_acc = 0, 0
    x_clip = 0
    for begin in range(0, len(x), batch_size):
        end = min(begin + batch_size, len(x))
        block = x[begin:end]
        qx = quantize_symmetric(block, sx, qmax_a)
        x_clip += int(np.count_nonzero(
            np.abs(block.astype(np.float64)) > sx * qmax_a))
        acc = (qx.reshape(-1, K) @ qw).reshape(end - begin, M, N) + qb
        min_acc = min(min_acc, int(acc.min()))
        max_acc = max(max_acc, int(acc.max()))
        peak = max(peak, int(np.max(np.abs(acc))))
        approximate = acc.astype(np.float64) * output_scale
        ref = reference[begin:end].astype(np.float64)
        error_square += float(np.sum((approximate - ref) ** 2))
        reference_square += float(np.sum(ref ** 2))
        max_abs_error = max(max_abs_error, float(np.max(np.abs(approximate - ref))))
    if min_acc < np.iinfo(np.int32).min or max_acc > np.iinfo(np.int32).max:
        raise OverflowError("integer MAC plus bias exceeds signed INT32")
    count = x.size // K * N
    denom = max(np.sqrt(reference_square / count), 1e-8)
    qw_abs = np.abs(weight.astype(np.float64))
    return {
        "nrmse_pct": 100 * np.sqrt(error_square / count) / denom,
        "max_abs_error": max_abs_error,
        "observed_peak": peak,
        "min_accumulator": min_acc,
        "max_accumulator": max_acc,
        "qb_abs_max": int(np.max(np.abs(qb))),
        "x_clip_pct": 100 * x_clip / x.size,
        "w_clip_pct": 100 * np.count_nonzero(qw_abs > sw * qmax_w) / weight.size,
    }


def worst_case_bound(a_bits, w_bits, qb_abs_max):
    qmax_a = (1 << (a_bits - 1)) - 1
    qmax_w = (1 << (w_bits - 1)) - 1
    return K * qmax_a * qmax_w + qb_abs_max


def select_max(train_x, weight, probe_x, probe_ref, bias, a_bits, w_bits):
    sx, sw = base_scales(train_x, weight, a_bits, w_bits)
    return sx, sw, {"method_detail": "abs-max training scale"}


def select_percentile(train_x, weight, probe_x, probe_ref, bias, a_bits, w_bits):
    probes = []
    for p in PERCENTILES:
        sx, sw = scale_percentile(train_x, weight, a_bits, w_bits, p)
        result = evaluate(probe_x, probe_ref, weight, bias, sx, sw, a_bits, w_bits)
        probes.append({"percentile": p, "train_nrmse_pct": result["nrmse_pct"]})
    winner = min(probes, key=lambda r: (r["train_nrmse_pct"], r["percentile"]))
    sx, sw = scale_percentile(train_x, weight, a_bits, w_bits, winner["percentile"])
    return sx, sw, {"percentile": winner["percentile"], "candidates": probes}


def select_err_min(train_x, weight, probe_x, probe_ref, bias, a_bits, w_bits):
    sx0, sw0 = base_scales(train_x, weight, a_bits, w_bits)

    def probe(af, wf):
        return evaluate(probe_x, probe_ref, weight, bias,
                        sx0 * af, sw0 * wf, a_bits, w_bits)["nrmse_pct"]

    trace = []
    activation = [(f, probe(f, 1.0)) for f in ACT_FACTORS]
    af = min(activation, key=lambda t: (t[1], abs(t[0] - 1.0)))[0]
    for f, score in activation:
        trace.append({"stage": "activation", "activation_factor": f,
                      "weight_factor": 1.0, "train_nrmse_pct": score})
    weights = [(f, probe(af, f)) for f in WGT_FACTORS]
    wf = min(weights, key=lambda t: (t[1], abs(t[0] - 1.0)))[0]
    for f, score in weights:
        trace.append({"stage": "weight", "activation_factor": af,
                      "weight_factor": f, "train_nrmse_pct": score})
    return sx0 * af, sw0 * wf, {"activation_factor": af, "weight_factor": wf,
                                "search": trace}


SELECTORS = {"max": select_max, "percentile": select_percentile,
             "err_min": select_err_min}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path,
                        default=ROOT / ".build/quant_samples_12labels_40.npz")
    parser.add_argument("--weights", type=Path,
                        default=ROOT / "weights/kwt1_fp32.npz")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "quant_experiments/ffn1_a12w10_methods")
    parser.add_argument("--activation-bits", type=int, default=12)
    parser.add_argument("--weight-bits", type=int, default=10)
    parser.add_argument("--train-select-per-class", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-layers", type=int, default=LAYERS)
    args = parser.parse_args()
    a_bits, w_bits = args.activation_bits, args.weight_bits
    if (a_bits, w_bits) != (12, 10):
        parser.error("this experiment is frozen to the verified A12W10 container")
    if args.max_layers < 1 or args.max_layers > LAYERS:
        parser.error("max-layers out of range")

    with np.load(args.samples, allow_pickle=False) as data:
        features = data["features"]
        partitions = data["partitions"]
        paths = data["paths"]
        words = data["words"]
    if features.ndim != 3 or features.shape[1:] != (98, 40) or not np.isfinite(features).all():
        raise ValueError("expected finite FP32 MFCC [samples,98,40]")
    if np.any((partitions != "training") & (partitions != "validation")):
        raise ValueError("testing/other partitions are forbidden")
    train_ids = np.flatnonzero(partitions == "training")
    val_ids = np.flatnonzero(partitions == "validation")
    if not len(train_ids) or not len(val_ids):
        raise ValueError("insufficient train or validation samples")
    if set(paths[train_ids]).intersection(paths[val_ids]):
        raise ValueError("train/validation paths overlap")
    if len(set(paths.tolist())) != len(paths):
        raise ValueError("duplicate sample paths")
    train_labels = Counter(words[train_ids].tolist())
    val_labels = Counter(words[val_ids].tolist())

    # Train-only selection subset: first N samples of each training label.
    selected = []
    for label in sorted(train_labels):
        label_ids = train_ids[words[train_ids] == label]
        if len(label_ids) < args.train_select_per_class:
            raise ValueError(f"too few training samples for {label}")
        selected.extend(label_ids[:args.train_select_per_class].tolist())
    selected = np.array(sorted(selected), dtype=np.int64)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    model = KWT1(args.weights, device)
    activations = collect_activations(features, model)
    if activations.shape != (len(features), LAYERS, M, K):
        raise ValueError(f"unexpected FFN1 inputs: {activations.shape}")

    records = []
    selection_trace = []
    for layer in range(args.max_layers):
        p = f"{PREFIX}/enc_layers/{layer}/ffn/layer_with_weights-0"
        weight = model.weight(p + "/kernel").cpu().numpy()
        bias = model.weight(p + "/bias").cpu().numpy()
        if weight.shape != (K, N) or bias.shape != (N,):
            raise ValueError(f"unexpected layer {layer + 1} parameter shape")
        train_x = activations[train_ids, layer]
        probe_x = activations[selected, layer]
        probe_ref = probe_x @ weight + bias
        val_x = activations[val_ids, layer]
        val_ref = val_x @ weight + bias
        for method in METHODS:
            sx, sw, detail = SELECTORS[method](train_x, weight, probe_x, probe_ref,
                                               bias, a_bits, w_bits)
            # Scales are now frozen on training data; validation only evaluates.
            train_result = evaluate(probe_x, probe_ref, weight, bias, sx, sw,
                                    a_bits, w_bits)
            val_result = evaluate(val_x, val_ref, weight, bias, sx, sw,
                                  a_bits, w_bits)
            bound = worst_case_bound(a_bits, w_bits, val_result["qb_abs_max"])
            if bound > np.iinfo(np.int32).max:
                raise OverflowError("worst-case accumulator bound exceeds INT32")
            scheme = f"A{a_bits}W{w_bits}"
            records.append({
                "scheme": scheme, "method": method, "layer": layer + 1,
                "activation_scale": float(sx),
                "weight_scale_min": float(np.min(sw)),
                "weight_scale_max": float(np.max(sw)),
                "train_subset_nrmse_pct": train_result["nrmse_pct"],
                "val_nrmse_pct": val_result["nrmse_pct"],
                "val_max_abs_error": val_result["max_abs_error"],
                "val_x_clip_pct": val_result["x_clip_pct"],
                "val_w_clip_pct": val_result["w_clip_pct"],
                "observed_peak": val_result["observed_peak"],
                "min_accumulator": val_result["min_accumulator"],
                "max_accumulator": val_result["max_accumulator"],
                "qb_abs_max": val_result["qb_abs_max"],
                "worst_case_bound": bound,
                "int32_headroom_pct": 100 * bound / np.iinfo(np.int32).max,
            })
            selection_trace.append({"layer": layer + 1, "method": method, **detail})
            if method == "percentile":
                # Post-hoc diagnostics only: fixed-percentile validation numbers.
                # The percentile choice above never read these values.
                sweep = []
                for percentile in PERCENTILES:
                    sx_p, sw_p = scale_percentile(train_x, weight, a_bits,
                                                  w_bits, percentile)
                    val_p = evaluate(val_x, val_ref, weight, bias, sx_p, sw_p,
                                     a_bits, w_bits)
                    sweep.append({
                        "percentile": percentile,
                        "val_nrmse_pct": val_p["nrmse_pct"],
                        "val_max_abs_error": val_p["max_abs_error"],
                        "val_w_clip_pct": val_p["w_clip_pct"],
                        "selection_input": False,
                    })
                selection_trace[-1]["candidate_val"] = sweep
        print(f"layer={layer + 1:02} done", flush=True)

    summary = []
    for method in METHODS:
        subset = [r for r in records if r["method"] == method]
        worst_nrmse = max(subset, key=lambda r: r["val_nrmse_pct"])
        worst_err = max(subset, key=lambda r: r["val_max_abs_error"])
        peak = max(subset, key=lambda r: r["observed_peak"])
        bound = max(subset, key=lambda r: r["worst_case_bound"])
        summary.append({
            "scheme": f"A{a_bits}W{w_bits}", "method": method,
            "layers": len(subset),
            "worst_layer": worst_nrmse["layer"],
            "worst_val_nrmse_pct": worst_nrmse["val_nrmse_pct"],
            "max_output_abs_error": worst_err["val_max_abs_error"],
            "max_output_abs_error_layer": worst_err["layer"],
            "observed_peak": peak["observed_peak"],
            "observed_peak_layer": peak["layer"],
            "worst_case_bound": bound["worst_case_bound"],
            "int32_headroom_pct": bound["int32_headroom_pct"],
            "max_val_x_clip_pct": max(r["val_x_clip_pct"] for r in subset),
            "max_val_w_clip_pct": max(r["val_w_clip_pct"] for r in subset),
        })

    packed_weights_kib = 12 * K * N * w_bits / 8 / 1024
    container_weights_kib = 12 * K * N * 16 / 8 / 1024
    capacity = {
        "packed_weight_kib_12layers": packed_weights_kib,
        "container_int16_weight_kib_12layers": container_weights_kib,
        "activation_scale_bytes_12layers": 12 * 8,
        "weight_scale_bytes_12layers": 12 * N * 8,
        "int32_bias_kib_12layers": 12 * N * 4 / 1024,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    meta = {
        "experiment": "A12W10 training-split calibration-method comparison",
        "container": "signed-symmetric A12/W10, zero-point 0, INT32 bias, ties-to-even rounding, clip saturation",
        "metric": "per layer RMS(error)/max(RMS(FP32),1e-8) over all validation samples and 99x256 outputs; worst of 12 layers",
        "calibration_methods": {
            "max": "training abs-max activation / per-channel abs-max weight",
            "percentile": f"training magnitude percentile from {PERCENTILES}, percentile chosen on training subset",
            "err_min": "derivative-free coordinate search of activation/weight scale factors minimising training NRMSE",
        },
        "gradient_used": False,
        "sample_archive": str(args.samples.resolve()),
        "sample_sha256": sha256(args.samples),
        "weight_archive": str(args.weights.resolve()),
        "weight_sha256": sha256(args.weights),
        "training_count": len(train_ids), "validation_count": len(val_ids),
        "training_labels": dict(sorted(train_labels.items())),
        "validation_labels": dict(sorted(val_labels.items())),
        "training_selection_count": len(selected),
        "training_selection_per_class": args.train_select_per_class,
        "activation_bits": a_bits, "weight_bits": w_bits,
        "device": str(device), "capacity": capacity,
        "summary": summary, "selection_trace": selection_trace, "layers": records,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    for item in summary:
        print(f"{item['method']:11} worst_val_nrmse={item['worst_val_nrmse_pct']:.6f}% "
              f"layer={item['worst_layer']} max_abs_err={item['max_output_abs_error']:.4e} "
              f"peak={item['observed_peak']} headroom={item['int32_headroom_pct']:.2f}%")
    print(f"PASS wrote {args.output.with_suffix('.csv')} and {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
