#!/usr/bin/env python3
"""Run frozen A11W10 validation operands through the actual row-broadcast RTL."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from run_row_broadcast import build, run_case

SOFTWARE = Path(__file__).resolve().parents[2] / "software"
import sys
sys.path.insert(0, str(SOFTWARE))
from quantize_ffn1 import collect_activations  # noqa: E402
from run import KWT1, PREFIX  # noqa: E402


def nrmse_pct(actual, reference):
    difference = actual.astype(np.float64) - reference.astype(np.float64)
    denominator = max(np.sqrt(np.mean(reference.astype(np.float64) ** 2)), 1e-8)
    return 100.0 * np.sqrt(np.mean(difference ** 2)) / denominator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path,
                        default=SOFTWARE / ".build/quant_samples_12labels_20.npz")
    parser.add_argument("--params", type=Path,
                        default=SOFTWARE / "results/ffn1_blackbox_a11w10.npz")
    parser.add_argument("--weights", type=Path,
                        default=SOFTWARE / "weights/kwt1_fp32.npz")
    parser.add_argument("--validation-index", type=int, default=0,
                        help="zero-based index within validation partition")
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(1, 13)),
                        help="one-based FFN1 layers to run")
    parser.add_argument("--rows", type=int, choices=(8, 16), default=8,
                        help="parallel A rows / CACC banks")
    parser.add_argument("--staged", action="store_true",
                        help="use RTL output stage and overlap writeback")
    parser.add_argument("--out", type=Path, default=Path("/tmp/kwt-quantized-rtl"))
    args = parser.parse_args()
    if args.validation_index < 0 or any(not 1 <= layer <= 12 for layer in args.layers):
        parser.error("invalid validation index or layer")
    with np.load(args.samples, allow_pickle=False) as data:
        validation = np.flatnonzero(data["partitions"] == "validation")
        if args.validation_index >= len(validation):
            parser.error("validation index exceeds available samples")
        item = validation[args.validation_index]
        feature = data["features"][item:item + 1].copy()
        sample_path = str(data["paths"][item])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KWT1(args.weights, device)
    inputs = collect_activations(feature, model)
    if inputs.shape != (1, 12, 99, 64):
        raise ValueError(f"unexpected FFN1 activation shape {inputs.shape}")
    args.out.mkdir(parents=True, exist_ok=True)
    executable = build(args.out, trace=False, rows=args.rows, staged=args.staged)
    summary = []
    with np.load(args.params, allow_pickle=False) as parameters:
        for layer in args.layers:
            stem = f"layer{layer:02}"
            x = inputs[0, layer - 1]
            sx = float(parameters[stem + "_activation_scale"])
            sw = parameters[stem + "_weight_scales"].astype(np.float64)
            qw = parameters[stem + "_weight"].astype(np.int64)
            qb = parameters[stem + "_bias"].astype(np.int64)
            qx = np.clip(np.rint(x.astype(np.float64) / sx), -1023, 1023).astype(np.int64)
            expected = qx @ qw + qb
            if expected.min() < -(1 << 31) or expected.max() >= (1 << 31):
                raise OverflowError(f"layer {layer} signed INT32 accumulator overflow")
            case = {"name": stem, "format": "int16", "a": qx.tolist(),
                    "b": qw.tolist(), "bias": qb.tolist(),
                    "expected": (expected & 0xFFFFFFFF).tolist(), "gaps": 0}
            metrics = run_case(executable, case, args.out, trace=False,
                               rows=args.rows, staged=args.staged)
            actual = np.empty((99, 256), dtype=np.uint32)
            for line in (args.out / stem / "actual.txt").read_text().splitlines():
                row, col, value = line.split()
                actual[int(row), int(col)] = int(value, 16)
            signed_actual = actual.view(np.int32).astype(np.int64)
            if not np.array_equal(signed_actual, expected):
                raise AssertionError(f"layer {layer}: captured RTL output differs from integer model")
            prefix = f"{PREFIX}/enc_layers/{layer-1}/ffn/layer_with_weights-0"
            weight = model.weight(prefix + "/kernel").cpu().numpy()
            bias = model.weight(prefix + "/bias").cpu().numpy()
            reference = x @ weight + bias
            decoded = signed_actual.astype(np.float64) * (sx * sw)
            record = {"layer": layer, "nrmse_pct": nrmse_pct(decoded, reference),
                      "cycles": metrics["cycles"], "shared_bytes": metrics["shared_bytes"],
                      "checked_outputs": metrics["checked_outputs"]}
            summary.append(record)
            print(f"RTL layer={layer:02} cycles={record['cycles']} "
                  f"outputs={record['checked_outputs']} "
                  f"NRMSE={record['nrmse_pct']:.5f}%", flush=True)
    report = {"sample_path": sample_path, "validation_index": args.validation_index,
              "rows": args.rows, "output_staged": args.staged,
              "source": "frozen quantized parameters and real validation FFN1 input",
              "hardware": ("Verilator kwt_row_broadcast_tile with RTL output stage; "
                           "A/W/bias and shared bus timed testbench model" if args.staged else
                           "Verilator kwt_row_broadcast_tile; task data movement timed testbench model"),
              "layers": summary,
              "worst_nrmse_pct": max(item["nrmse_pct"] for item in summary),
              "total_cycles": sum(item["cycles"] for item in summary)}
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"PASS {len(summary)} RTL layers, worst={report['worst_nrmse_pct']:.5f}%, "
          f"sum_cycles={report['total_cycles']}; report={args.out/'report.json'}")


if __name__ == "__main__":
    main()
