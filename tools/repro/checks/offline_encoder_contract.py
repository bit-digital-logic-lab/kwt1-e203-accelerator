#!/usr/bin/env python3
"""Numerical replay of the offline encoder-input contract with numpy only.

For every frozen sample in ``software/validation/real_input_chain_20261017``
this check calls the unchanged ``software/offline_encoder_input.convert`` on
the stored ``98x40`` MFCC and compares the result with the stored ``99x64``
encoder input.  A result comparison (values), not file existence, decides the
outcome.

The content hashes recorded in the sample manifest are compared as
informational metadata only: at this commit the committed ``.npy`` files do
not match the manifest's ``*_sha256`` fields while still agreeing numerically
with ``convert``.  That is reported but does not gate the check.

This is a small software regression check, not competition acceptance.  It
does not exercise the PyTorch/TensorFlow path and it does not cover the
per-layer FFN1 inputs required by the contest.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SOFTWARE = ROOT / "software"
sys.path.insert(0, str(SOFTWARE))

from offline_encoder_input import PREFIX, convert  # noqa: E402

CASE_DIR = SOFTWARE / "validation/real_input_chain_20261017"
WEIGHTS = SOFTWARE / "weights/kwt1_fp32.npz"
TOLERANCE = 1e-5
WEIGHT_TENSORS = ("patch_proj/kernel", "patch_proj/bias", "class_emb", "pos_emb")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    manifest = json.loads((CASE_DIR / "manifest.json").read_text())
    with np.load(WEIGHTS, allow_pickle=False) as archive:
        weights = {PREFIX + suffix: archive[PREFIX + suffix] for suffix in WEIGHT_TENSORS}

    failures: list[str] = []
    hash_mismatches: list[str] = []
    samples: list[dict] = []
    for entry in manifest["samples"]:
        mfcc_path = CASE_DIR / entry["mfcc_file"]
        encoder_path = CASE_DIR / entry["encoder_file"]
        for path, expected in ((mfcc_path, entry["mfcc_sha256"]),
                               (encoder_path, entry["encoder_sha256"])):
            actual = sha256(path)
            if actual != expected:
                hash_mismatches.append(f"{path.name}: {actual} != manifest {expected}")
        mfcc = np.load(mfcc_path, allow_pickle=False)
        expected_encoder = np.load(encoder_path, allow_pickle=False)
        output = convert(mfcc, weights)
        if output.shape != expected_encoder.shape:
            failures.append(f"{encoder_path.name}: shape {output.shape} != {expected_encoder.shape}")
            continue
        error = float(np.max(np.abs(output.astype(np.float64)
                                   - expected_encoder.astype(np.float64))))
        if error > TOLERANCE:
            failures.append(f"{encoder_path.name}: max_abs_error {error:g} > {TOLERANCE:g}")
        samples.append({"encoder_file": encoder_path.name, "max_abs_error": error,
                        "shape": list(output.shape)})

    worst = max((s["max_abs_error"] for s in samples), default=None)
    print(f"checked_samples={len(samples)} worst_max_abs_error={worst} tolerance={TOLERANCE}")
    print(f"manifest_hash_mismatches={len(hash_mismatches)} (informational)")
    print("RESULT_JSON:" + json.dumps({
        "checked_samples": len(samples),
        "worst_max_abs_error": worst,
        "tolerance": TOLERANCE,
        "failures": failures,
        "manifest_hash_mismatches": hash_mismatches,
        "samples": samples,
    }, ensure_ascii=False))
    if failures:
        print("FAIL:", *failures, sep="\n  ", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
