#!/usr/bin/env python3
"""Verify frozen FFN1 quantization on real samples with one Verilator build."""

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

from run_row_broadcast import build, run_case

REPO = Path(__file__).resolve().parents[2]
SOFTWARE = REPO / "software"
sys.path.insert(0, str(SOFTWARE))
from quantize_ffn1 import collect_activations  # noqa: E402
from run import KWT1, PREFIX  # noqa: E402

M, K, N, LAYERS = 99, 64, 256, 12
DEFAULT_PARAMS = {
    "A11W10": SOFTWARE / "results/ffn1_blackbox_a11w10.npz",
    "A12W10": SOFTWARE / "results/ffn1_a12w10_static_sym.npz",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_validation(data, per_label):
    validation = np.flatnonzero(data["partitions"] == "validation")
    words = data["words"]
    if per_label == 0:
        return validation
    selected = []
    for word in sorted(set(words[validation].tolist())):
        candidates = validation[words[validation] == word]
        if len(candidates) < per_label:
            raise ValueError(f"only {len(candidates)} validation samples for {word}")
        selected.extend(candidates[:per_label].tolist())
    return np.asarray(selected, dtype=np.int64)


def read_actual(path):
    actual = np.empty((M, N), dtype=np.uint32)
    for line in path.read_text().splitlines():
        row, col, value = line.split()
        actual[int(row), int(col)] = int(value, 16)
    return actual.view(np.int32).astype(np.int64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path,
                        default=SOFTWARE / ".build/quant_samples_12labels_20.npz")
    parser.add_argument("--scheme", default="A11W10",
                        help="signed symmetric format, for example A8W8 or A12W10")
    parser.add_argument("--activation-bits", type=int, default=11)
    parser.add_argument("--params", type=Path,
                        help="frozen parameter NPZ; defaults to the artifact for --scheme")
    parser.add_argument("--weights", type=Path,
                        default=SOFTWARE / "weights/kwt1_fp32.npz")
    parser.add_argument("--per-label", type=int, default=1,
                        help="first N validation samples per label; 0 means all")
    parser.add_argument("--rows", type=int, choices=(8, 16), default=16)
    parser.add_argument("--rtl-data-width", type=int, choices=(8, 16), default=16,
                        help="physical A/W operand width in the RTL compute core")
    parser.add_argument("--staged", action="store_true",
                        help="include RTL output staging")
    parser.add_argument("--keep-cases", action="store_true",
                        help="retain per-task hex, output and Verilator logs")
    parser.add_argument("--out", type=Path,
                        default=Path("/tmp/kwt-ffn1-batch-rtl"))
    args = parser.parse_args()
    if args.per_label < 0:
        parser.error("--per-label must be nonnegative")
    match = re.fullmatch(r"A(\d+)W(\d+)", args.scheme)
    if match is None or not (2 <= int(match.group(1)) <= 15 and
                            2 <= int(match.group(2)) <= 15):
        parser.error("--scheme must be A<bits>W<bits> with bit widths in 2..15")
    if int(match.group(1)) != args.activation_bits:
        parser.error("--activation-bits must match --scheme")
    if max(int(match.group(1)), int(match.group(2))) > args.rtl_data_width:
        parser.error("format bit widths exceed --rtl-data-width")
    if args.rtl_data_width == 8 and not args.staged:
        parser.error("8-bit physical RTL path requires --staged")
    if args.params is None:
        if args.scheme not in DEFAULT_PARAMS:
            parser.error("--params is required for this scheme")
        args.params = DEFAULT_PARAMS[args.scheme]
    with np.load(args.samples, allow_pickle=False) as dataset:
        chosen = select_validation(dataset, args.per_label)
        features = dataset["features"][chosen].copy()
        labels = dataset["words"][chosen].tolist()
        paths = dataset["paths"][chosen].tolist()
    if not len(chosen) or features.shape[1:] != (98, 40):
        raise ValueError("no valid FP32 MFCC validation samples")
    if len(set(paths)) != len(paths):
        raise ValueError("selected validation paths are not unique")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KWT1(args.weights, device)
    inputs = collect_activations(features, model)
    if inputs.shape != (len(chosen), LAYERS, M, K):
        raise ValueError(f"unexpected FFN1 input shape {inputs.shape}")
    args.out.mkdir(parents=True, exist_ok=True)
    executable = build(args.out, trace=False, rows=args.rows, staged=args.staged,
                       data_width=args.rtl_data_width)
    totals = [{"layer": layer, "error_sq": 0.0, "reference_sq": 0.0,
               "checked_outputs": 0, "cycles": 0, "max_abs_accumulator": 0}
              for layer in range(1, LAYERS + 1)]
    sample_records = []
    with np.load(args.params, allow_pickle=False) as parameters:
        frozen = []
        for layer in range(1, LAYERS + 1):
            stem = f"layer{layer:02}"
            prefix = f"{PREFIX}/enc_layers/{layer-1}/ffn/layer_with_weights-0"
            sx = float(parameters[stem + "_activation_scale"])
            sw = parameters[stem + "_weight_scales"].astype(np.float64)
            qw = parameters[stem + "_weight"].astype(np.int64)
            qb = parameters[stem + "_bias"].astype(np.int64)
            weight = model.weight(prefix + "/kernel").cpu().numpy()
            bias = model.weight(prefix + "/bias").cpu().numpy()
            if qw.shape != (K, N) or qb.shape != (N,) or sw.shape != (N,):
                raise ValueError(f"layer {layer} parameter shape mismatch")
            frozen.append((sx, sw, qw, qb, weight, bias))

        for sample_no, (index, label, path) in enumerate(zip(chosen, labels, paths)):
            sample_layers = []
            for layer, (sx, sw, qw, qb, weight, bias) in enumerate(frozen, 1):
                x = inputs[sample_no, layer-1]
                qmax_a = (1 << (args.activation_bits - 1)) - 1
                qx = np.clip(np.rint(x.astype(np.float64) / sx),
                             -qmax_a, qmax_a).astype(np.int64)
                expected = qx @ qw + qb
                if expected.min() < np.iinfo(np.int32).min or expected.max() > np.iinfo(np.int32).max:
                    raise OverflowError(f"sample {index} layer {layer} signed INT32 overflow")
                name = f"sample{sample_no:03d}_layer{layer:02d}"
                case = {"name": name, "format": "int16", "a": qx.tolist(),
                        "b": qw.tolist(), "bias": qb.tolist(),
                        "expected": (expected & 0xFFFFFFFF).tolist(), "gaps": 0}
                metrics = run_case(executable, case, args.out, trace=False,
                                   rows=args.rows, staged=args.staged)
                actual = read_actual(args.out / name / "actual.txt")
                if not np.array_equal(actual, expected):
                    raise AssertionError(f"sample {index} layer {layer}: RTL differs from integer model")
                reference = x @ weight + bias
                decoded = actual.astype(np.float64) * (sx * sw)
                error = decoded - reference.astype(np.float64)
                entry = totals[layer-1]
                entry["error_sq"] += float(np.sum(error * error))
                entry["reference_sq"] += float(np.sum(reference.astype(np.float64) ** 2))
                entry["checked_outputs"] += M*N
                entry["cycles"] += metrics["cycles"]
                entry["max_abs_accumulator"] = max(entry["max_abs_accumulator"],
                                                     int(np.max(np.abs(actual))))
                sample_layers.append(layer)
                if not args.keep_cases:
                    shutil.rmtree(args.out / name)
            sample_records.append({"validation_index": int(index), "label": label,
                                   "path": path, "layers": sample_layers})
            print(f"PASS sample {sample_no+1}/{len(chosen)} index={index} label={label} "
                  f"outputs={LAYERS*M*N}", flush=True)
            (args.out / "progress.json").write_text(json.dumps({
                "completed_samples": len(sample_records), "total_samples": len(chosen),
                "samples": sample_records}, ensure_ascii=False, indent=2) + "\n")

    layers = []
    for entry in totals:
        n = entry["checked_outputs"]
        denominator = max(np.sqrt(entry["reference_sq"] / n), 1e-8)
        nrmse = 100.0 * np.sqrt(entry["error_sq"] / n) / denominator
        layers.append({"layer": entry["layer"], "nrmse_pct": nrmse,
                       "checked_outputs": n, "cycles": entry["cycles"],
                       "max_abs_accumulator": entry["max_abs_accumulator"]})
    report = {
        "samples": len(chosen), "selection": ("all validation" if args.per_label == 0 else
                                              f"first {args.per_label} validation per label"),
        "sample_records": sample_records,
        "rows": args.rows, "output_staged": args.staged,
        "rtl_data_width": args.rtl_data_width,
        "scheme": args.scheme, "activation_bits": args.activation_bits,
        "source_sha256": {"samples": sha256(args.samples), "params": sha256(args.params),
                          "weights": sha256(args.weights)},
        "metric": "per-layer pooled RMS(actual decoded - FP32) / max(RMS(FP32),1e-8); max across 12 layers",
        "rtl_check": "each output bit-exact against independent signed INT32 GEMM plus bias",
        "hardware_boundary": "Verilator compute tile/output stage; A/W/bias and shared bus are timed testbench models",
        "layers": layers,
        "worst_nrmse_pct": max(item["nrmse_pct"] for item in layers),
        "checked_outputs": sum(item["checked_outputs"] for item in layers),
        "total_cycles": sum(item["cycles"] for item in layers),
        "pass_3pct": all(item["nrmse_pct"] <= 3 for item in layers),
    }
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"PASS {len(chosen)} samples x {LAYERS} layers, {report['checked_outputs']} outputs; "
          f"worst NRMSE={report['worst_nrmse_pct']:.5f}%; report={args.out/'report.json'}")
    if not report["pass_3pct"]:
        raise SystemExit("FAIL: one or more layers exceed 3% NRMSE on this subset")


if __name__ == "__main__":
    main()
