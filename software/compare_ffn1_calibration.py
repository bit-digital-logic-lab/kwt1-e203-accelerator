"""Train-only, no-gradient scale comparison for KWT-1 FFN1 integer GEMM."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from calibrate_blackbox_ffn1 import base_scales, black_box_error, encode_params, sha256
from quantize_ffn1 import collect_activations
from run import KWT1, PREFIX, ROOT


BITS = ((8, 8), (11, 10), (12, 9), (12, 10))
FACTORS = ((1.0, 1.0), (0.8, 1.0), (0.9, 1.0), (1.1, 1.0),
           (1.2, 1.0), (1.0, 0.9), (1.0, 1.1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path,
                        default=ROOT / ".build/quant_samples_12labels_20.npz")
    parser.add_argument("--weights", type=Path,
                        default=ROOT / "weights/kwt1_fp32.npz")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/ffn1_multi_scale_12labels_20")
    parser.add_argument("--train-select-per-class", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.train_select_per_class < 1:
        parser.error("train-select-per-class must be positive")

    with np.load(args.samples, allow_pickle=False) as data:
        features, partitions, paths, words = (data["features"], data["partitions"],
                                              data["paths"], data["words"])
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
    if torch.cuda.is_available() and args.device == "auto":
        device = "cuda"
    else:
        device = "cpu" if args.device == "auto" else args.device
    model = KWT1(args.weights, torch.device(device))
    activations = collect_activations(features, model)
    selected_train_ids = []
    for label in sorted(set(words[train_ids].tolist())):
        label_ids = train_ids[words[train_ids] == label]
        if len(label_ids) < args.train_select_per_class:
            raise ValueError(f"too few training samples for {label}")
        selected_train_ids.extend(label_ids[:args.train_select_per_class].tolist())
    selected_train_ids = np.array(selected_train_ids, dtype=np.int64)
    records, candidate_trace, artifacts = [], [], {}

    for layer in range(12):
        p = f"{PREFIX}/enc_layers/{layer}/ffn/layer_with_weights-0"
        weight = model.weight(p + "/kernel").cpu().numpy()
        bias = model.weight(p + "/bias").cpu().numpy()
        all_train_x = activations[train_ids, layer]
        select_x = activations[selected_train_ids, layer]
        validation_x = activations[val_ids, layer]
        select_reference = select_x @ weight + bias
        validation_reference = validation_x @ weight + bias
        for a_bits, w_bits in BITS:
            sx_base, sw_base = base_scales(all_train_x, weight, a_bits, w_bits)
            probes = []
            for af, wf in FACTORS:
                result = black_box_error(select_x, select_reference, weight, bias,
                                         sx_base * af, sw_base * wf, a_bits, w_bits)
                probes.append({"activation_factor": af, "weight_factor": wf,
                               "training_nrmse_pct": result["nrmse_pct"]})
            winner = min(probes, key=lambda x: (x["training_nrmse_pct"],
                                                abs(x["activation_factor"] - 1.0) +
                                                abs(x["weight_factor"] - 1.0)))
            scheme = f"A{a_bits}W{w_bits}"
            candidate_trace.append({"layer": layer + 1, "scheme": scheme,
                                    "candidates": probes})
            for name, af, wf in (("baseline", 1.0, 1.0),
                                 ("train_selected", winner["activation_factor"],
                                  winner["weight_factor"])):
                sx, sw = sx_base * af, sw_base * wf
                result = black_box_error(validation_x, validation_reference,
                                         weight, bias, sx, sw, a_bits, w_bits)
                qw, qb = encode_params(weight, bias, sx, sw, a_bits, w_bits)
                bound = 64 * ((1 << (a_bits - 1)) - 1) * ((1 << (w_bits - 1)) - 1)
                bound += int(np.max(np.abs(qb)))
                if bound > np.iinfo(np.int32).max:
                    raise OverflowError("worst-case accumulator bound exceeds INT32")
                records.append({"scheme": scheme, "layer": layer + 1,
                                "calibration": name, "activation_factor": af,
                                "weight_factor": wf,
                                "validation_nrmse_pct": result["nrmse_pct"],
                                "observed_accumulator_peak": result["max_abs_accumulator"],
                                "worst_case_abs_accumulator_bound": bound,
                                "packed_weight_kib_12layers": 196608 * w_bits / 8 / 1024})
                if name == "train_selected":
                    key = f"{scheme}_layer{layer + 1:02}"
                    artifacts[key + "_weight"] = qw.astype(np.int16)
                    artifacts[key + "_bias"] = qb.astype(np.int32)
                    artifacts[key + "_activation_scale"] = np.array(sx)
                    artifacts[key + "_weight_scales"] = sw
        print(f"layer={layer + 1:02} complete", flush=True)

    summary = []
    for a_bits, w_bits in BITS:
        for name in ("baseline", "train_selected"):
            scheme = f"A{a_bits}W{w_bits}"
            subset = [r for r in records if r["scheme"] == scheme and
                      r["calibration"] == name]
            worst = max(subset, key=lambda r: r["validation_nrmse_pct"])
            summary.append({"scheme": scheme, "calibration": name,
                            "worst_layer": worst["layer"],
                            "worst_nrmse_pct": worst["validation_nrmse_pct"],
                            "max_observed_accumulator": max(r["observed_accumulator_peak"]
                                                            for r in subset),
                            "max_abs_accumulator_bound": max(r["worst_case_abs_accumulator_bound"]
                                                             for r in subset),
                            "packed_weight_kib_12layers": worst["packed_weight_kib_12layers"]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    np.savez_compressed(args.output.with_suffix(".npz"), **artifacts)
    meta = {"experiment": "no-gradient train-only seven-pair scale selection per layer and bit-width",
            "metric": "per-layer RMS(error)/max(RMS(FP32),1e-8), all validation samples and 99x256 outputs, worst of 12 layers",
            "sample_archive": str(args.samples.resolve()), "sample_sha256": sha256(args.samples),
            "weight_archive": str(args.weights.resolve()), "weight_sha256": sha256(args.weights),
            "training_count": len(train_ids), "validation_count": len(val_ids),
            "training_selection_count": len(selected_train_ids),
            "training_selection_per_class": args.train_select_per_class,
            "training_selection_labels": words[selected_train_ids].tolist(),
            "training_selection_paths": paths[selected_train_ids].tolist(),
            "factors": FACTORS, "rounding": "nearest ties-to-even",
            "arithmetic": "signed symmetric zero-point-0 integer A/W, INT32 bias and accumulator; FP64 final scale for error only",
            "summary": summary, "candidate_trace": candidate_trace, "layers": records}
    args.output.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    for item in summary:
        print(f"{item['scheme']} {item['calibration']} "
              f"worst={item['worst_nrmse_pct']:.6f}% layer={item['worst_layer']} "
              f"bound={item['max_abs_accumulator_bound']}")


if __name__ == "__main__":
    main()
