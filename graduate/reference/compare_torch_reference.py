"""Second independent comparison: existing PyTorch KWT-1 runner vs the golden.

``software/run.py`` is an independent PyTorch re-implementation of the same
official TensorFlow architecture.  It is used here as a *second* reference so
that the golden is cross-checked against two unrelated implementations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REFERENCE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REFERENCE_DIR))
sys.path.insert(0, str(REFERENCE_DIR.parents[1] / "software"))

import kwt1_fp32_golden as golden  # noqa: E402
import run as torch_runner  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-dir", type=Path, default=golden.ENCODER_SAMPLES_DEFAULT)
    parser.add_argument("--weights", type=Path, default=golden.WEIGHTS_DEFAULT)
    parser.add_argument("--partition", default="validation")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--output", type=Path,
                        default=REFERENCE_DIR / "results/torch_reference_comparison.json")
    args = parser.parse_args()

    weights = golden.load_weights(args.weights)
    assert len(weights) == 162, f"expected 162 tensors, got {len(weights)}"
    model = torch_runner.KWT1(args.weights, torch.device("cpu"))
    manifest = json.loads((args.samples_dir / "manifest.json").read_text())
    samples = [s for s in manifest["samples"] if args.partition in (None, "all", s["split"])]

    records = []
    worst_overall = 0.0
    for sample in samples:
        encoder = np.load(args.samples_dir / sample["encoder_file"], allow_pickle=False)
        golden_logits, golden_final, _ = golden.forward(encoder, weights)
        golden_arrays = []
        golden.forward(encoder, weights, arrays=golden_arrays)
        with torch.inference_mode():
            torch_logits, torch_final, ffn_inputs = model.forward(
                torch.from_numpy(encoder), capture_ffn=True)
        console = torch_logits.cpu().numpy()
        per_layer = []
        for layer in range(golden.EXPECTED_LAYERS):
            per_layer.append(float(np.max(np.abs(
                ffn_inputs[layer].numpy() - golden_arrays[layer]["ln1"]))))
        final_error = float(np.max(np.abs(torch_final.numpy() - golden_final)))
        logits_error = float(np.max(np.abs(console - golden_logits)))
        worst = max([*per_layer, final_error, logits_error])
        worst_overall = max(worst_overall, worst)
        records.append({
            "archive_member": sample["archive_member"], "split": sample["split"],
            "label": sample["label"], "encoder_sha256": sample["encoder_sha256"],
            "max_abs_error_per_layer_ln1": per_layer,
            "max_abs_error_final_encoder": final_error,
            "max_abs_error_logits": logits_error,
            "worst": worst,
        })
        print(f"{sample['split']:10s} {sample['label']:9s} worst={worst:.3e} "
              f"final={final_error:.3e} logits={logits_error:.3e}")

    passed = worst_overall <= args.tolerance
    output = {
        "reference": "graduate/reference/kwt1_fp32_golden.py",
        "partner": "software/run.py (independent PyTorch KWT-1)",
        "torch_version": torch.__version__,
        "partition": args.partition,
        "sample_count": len(records),
        "tolerance": args.tolerance,
        "worst_max_abs_error": worst_overall,
        "pass": bool(passed),
        "samples": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=1) + "\n")
    print(f"worst_max_abs_error={worst_overall:.6g} tolerance={args.tolerance:g} "
          f"passed={passed} wrote={args.output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
