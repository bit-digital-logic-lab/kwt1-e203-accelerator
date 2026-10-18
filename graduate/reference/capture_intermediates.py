"""Capture FP32 golden intermediate tensors for real validation samples.

Runs the pure-NumPy golden (:mod:`kwt1_fp32_golden`) over real KWT-1 encoder
inputs taken from the official Speech Commands V2 validation split and writes a
small JSON summary: per-stage shape, value range and SHA-256 of every
intermediate tensor. Full tensors are deliberately not written to the
repository, only their hashes and scalar statistics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import kwt1_fp32_golden as golden


def select_samples(manifest: dict, partition: str | None):
    samples = manifest["samples"]
    if partition is not None:
        samples = [s for s in samples if s["split"] == partition]
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-dir", type=Path, default=golden.ENCODER_SAMPLES_DEFAULT)
    parser.add_argument("--weights", type=Path, default=golden.WEIGHTS_DEFAULT)
    parser.add_argument("--partition", default="validation",
                        help="Restrict to one official split ('validation', 'training' or 'all')")
    parser.add_argument("--output", type=Path,
                        default=golden.REFERENCE_DIR / "results/golden_intermediates.json")
    args = parser.parse_args()

    partition = None if args.partition == "all" else args.partition
    manifest = json.loads((args.samples_dir / "manifest.json").read_text())
    samples = select_samples(manifest, partition)
    if not samples:
        raise SystemExit(f"No samples for partition {args.partition!r}")

    flags = golden.check_flags(golden.load_flags())
    weights = golden.load_weights(args.weights)
    signatures = golden.layer_parameter_signatures(weights)

    records = []
    for sample in samples:
        encoder = np.load(args.samples_dir / sample["encoder_file"], allow_pickle=False)
        if list(encoder.shape) != [99, 64] or encoder.dtype != np.float32:
            raise ValueError(f"Bad encoder input {sample['encoder_file']}: {encoder.shape} {encoder.dtype}")
        actual_sha = golden.sha256(encoder)
        if actual_sha != sample["encoder_sha256"]:
            raise ValueError(f"Encoder SHA mismatch for {sample['encoder_file']}")
        logits, final, stages = golden.forward(encoder, weights, capture=True)
        top1 = int(np.argmax(logits))
        records.append({
            "archive_member": sample["archive_member"],
            "label": sample["label"],
            "split": sample["split"],
            "encoder_file": sample["encoder_file"],
            "encoder_sha256": actual_sha,
            "encoder_min": float(encoder.min()),
            "encoder_max": float(encoder.max()),
            "logits": [float(v) for v in logits],
            "top1": top1,
            "stages": stages,
        })

    output = {
        "reference": "graduate/reference/kwt1_fp32_golden.py",
        "weights": str(args.weights),
        "weights_sha256": golden.sha256(np.frombuffer(Path(args.weights).read_bytes(), dtype=np.uint8)),
        "semantics": flags,
        "partition": args.partition,
        "sample_count": len(records),
        "layer_parameter_sha256": signatures,
        "distinct_layer_parameters": len(set(signatures)) == len(signatures),
        "samples": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote={args.output} samples={len(records)} bytes={args.output.stat().st_size}")
    for record in records:
        print(f"  {record['split']:10s} {record['label']:9s} top1={record['top1']:2d} "
              f"logit_max={max(record['logits']):.4f} sha={record['encoder_sha256'][:12]}")
    print(f"distinct_layer_parameters={output['distinct_layer_parameters']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
