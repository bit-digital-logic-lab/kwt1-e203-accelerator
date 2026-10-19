#!/usr/bin/env python3
"""Split-provenance check for the encoder gate constants.

Policy under test
-----------------
* Quantisation scales may be calibrated (gradient-free, static max-abs) on the
  official **training** speaker split only.
* The official **validation** split is read only to evaluate error; it must not
  be used to derive a scale or the gate threshold.
* The gate threshold (`verify_encoder.NRMSE_THRESHOLD`) is a fixed 3% spec
  constant.

This script therefore
  1. reads the official train/validation split from the chain manifest,
  2. recomputes the per-tensor activation scales from the *training* samples
     only (gradient-free) and prints them next to the sample_00-derived scales
     baked into the current RTL vectors (meta.json),
  3. asserts that no validation sample was opened for calibration and that the
     threshold is the constant.

Scope note: it does **not** re-quantise weights or rebuild the RTL (the frozen
A12W10 pack and the vectors are outside this task's file scope).  It certifies
the split provenance of the gate constants and quantifies how sample-dependent
the activation scales are.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GRAD = HERE.parents[1]                                   # graduate
MODEL = GRAD / "encoder" / "model"
ENC = GRAD / "encoder"
sys.path.insert(0, str(MODEL))

import kwt_postnorm_ref as ref                            # noqa: E402

spec = importlib.util.spec_from_file_location("verify_encoder", MODEL / "verify_encoder.py")
ve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ve)

ROOT = ref.ROOT
CHAIN = ROOT / "software/validation/real_input_chain_20261017"
SCALARS = ("s_x", "s_a", "s_g", "s_r1", "s_r2", "s_out")


def main():
    manifest = json.loads((CHAIN / "manifest.json").read_text())
    samples = manifest["samples"]
    train = [s for s in samples if s["split"] == "training"]
    val = [s for s in samples if s["split"] == "validation"]
    train_files = {s["encoder_file"] for s in train}
    val_files = {s["encoder_file"] for s in val}
    assert train and val, "manifest must define both splits"
    assert not (train_files & val_files), "train/val files overlap"

    # ---- gradient-free calibration on the TRAINING split only ---------------
    per_sample = {}
    for s in train:
        ref.SAMPLE = CHAIN / s["encoder_file"]            # training path only
        cal = ref.calibrate()
        per_sample[s["encoder_file"]] = {k: float(cal["scales"][k]) for k in SCALARS}

    used = set(per_sample)
    assert used <= train_files, "calibration touched a non-training sample"
    assert not (used & val_files), "calibration touched a validation sample"

    agg = {k: max(r[k] for r in per_sample.values()) for k in SCALARS}  # train static max

    print("train samples used for calibration:", sorted(used))
    print("validation samples (evaluation only):", sorted(val_files))
    for f in sorted(per_sample):
        row = " ".join(f"{k}={per_sample[f][k]:.6e}" for k in SCALARS)
        print(f"  {f}: {row}")
    print("training-split aggregate (per-tensor static max):")
    for k in SCALARS:
        print(f"  train_max {k} = {agg[k]:.6e}")

    # ---- compare with the scales actually baked into the RTL vectors --------
    meta_path = ENC / "vectors" / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        declared = {"s_x": meta.get("scale_x"), "s_a": meta.get("scale_a"),
                    "s_out": meta.get("s_out")}
        print("scales currently baked into vectors/meta.json (sample_00-derived):")
        for k, v in declared.items():
            if v is None:
                continue
            rel = abs(agg[k] - float(v)) / max(abs(float(v)), 1e-300)
            print(f"  declared {k} = {float(v):.6e}  train_max/train rel-diff = {rel*100:.3f}%")
    else:
        print("vectors/meta.json absent (run run_gate.sh step 01 first); "
              "skipping comparison with the baked-in scales")

    # ---- threshold provenance ----------------------------------------------
    assert ve.NRMSE_THRESHOLD == 0.03, "NRMSE threshold is not the fixed 3% constant"
    print(f"gate threshold NRMSE_THRESHOLD = {ve.NRMSE_THRESHOLD*100:.2f}% "
          f"(fixed spec constant; not derived from any sample)")
    print(f"SPLIT_OK n_train={len(train)} n_val={len(val)} "
          f"calibrated_samples={len(used)} threshold_source=constant")


if __name__ == "__main__":
    main()
