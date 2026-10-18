#!/usr/bin/env python3
"""Independent software golden check for the KWT-1 layer-0 encoder input.

The design's ``software/offline_encoder_input.py`` project the 98x40 MFCC into a
99x64 encoder input.  This script recomputes the same quantity with a separate,
straight-line NumPy expression and compares three independent sources:

    design convert()            -> what the submitted model exports
    independent numpy reference -> recomputed here, not read from the design
    stored TF-checkpoint golden -> sample_00_encoder.npy written by the
                                   TensorFlow checkpoint path (cross-source)

It prints a machine-readable contract line so ``run_stages.py`` can only PASS
after a real comparison, and exits non-zero when any tolerance is violated.

``--corrupt-golden`` deliberately adds +1.0 to the stored golden; the check must
then fail, which is how the fault-injection test proves it is not a no-op.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ffn1_reference import max_abs_error, nrmse_percent  # noqa: E402

TOLERANCE = 1e-5
PREFIX = "layer_with_weights-0/"
TENSORS = ("patch_proj/kernel", "patch_proj/bias", "class_emb", "pos_emb")


def load_module(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location("design_offline_encoder_input", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def independent_reference(mfcc: np.ndarray, weights: dict) -> np.ndarray:
    """Linear projection + CLS token + positional embedding, in one expression."""
    tokens = np.empty((99, 64), dtype=np.float32)
    tokens[0] = weights[PREFIX + "class_emb"][0, 0]
    tokens[1:] = (mfcc @ weights[PREFIX + "patch_proj/kernel"]
                  + weights[PREFIX + "patch_proj/bias"])
    return tokens + weights[PREFIX + "pos_emb"][0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-root", type=pathlib.Path, required=True)
    parser.add_argument("--sample", default="real_input_chain_20261017",
                        help="validation directory under software/validation")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--corrupt-golden", action="store_true",
                        help="add +1.0 to the stored golden (fault injection)")
    parser.add_argument("--prefix", default="encoder_input")
    args = parser.parse_args()

    root = args.design_root.resolve()
    validation = root / "software/validation" / args.sample
    mfcc_path = validation / f"sample_{args.index:02d}_mfcc.npy"
    golden_path = validation / f"sample_{args.index:02d}_encoder.npy"
    weights_path = root / "software/weights/kwt1_fp32.npz"

    for path in (mfcc_path, golden_path, weights_path,
                 root / "software/offline_encoder_input.py"):
        if not path.exists():
            print(f"SKIP: missing {path}")
            return 0

    design = load_module(root / "software/offline_encoder_input.py")
    mfcc = np.load(mfcc_path, allow_pickle=False)
    stored = np.load(golden_path, allow_pickle=False).astype(np.float64)
    if args.corrupt_golden:
        stored = stored + 1.0
    with np.load(weights_path, allow_pickle=False) as archive:
        weights = {PREFIX + suffix: archive[PREFIX + suffix] for suffix in TENSORS}

    reference = independent_reference(mfcc, weights)
    converted = design.convert(mfcc, weights)

    checks = {
        "converted_vs_independent": max_abs_error(converted, reference),
        "converted_vs_golden": max_abs_error(converted, stored),
        "independent_vs_golden": max_abs_error(reference, stored),
    }
    nrmse = {
        "converted_vs_golden_nrmse_pct": nrmse_percent(converted, stored),
        "independent_vs_golden_nrmse_pct": nrmse_percent(reference, stored),
    }
    print(f"sample={mfcc_path.name} shape={tuple(converted.shape)} dtype={converted.dtype}")
    print(f"golden={golden_path} corrupt={bool(args.corrupt_golden)}")
    for name, value in checks.items():
        print(f"  max_abs {name}={value:.9g}")
    for name, value in nrmse.items():
        print(f"  {name}={value:.9g}")

    checked = int(converted.size) * len(checks)
    failed = [name for name, value in checks.items() if value > TOLERANCE]
    print(json.dumps({"checks": checks, "tolerance": TOLERANCE}))
    if failed:
        print(f"FAIL software golden tolerance violated: {', '.join(failed)}")
        print(f"KWT-GRAD-VERIFY {args.prefix} checked={checked} failed={len(failed)}")
        return 1
    print("PASS software golden: design convert, independent numpy and stored TF "
          "golden agree within tolerance")
    print(f"KWT-GRAD-VERIFY {args.prefix} checked={checked} failed=0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
