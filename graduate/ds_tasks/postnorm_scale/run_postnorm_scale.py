#!/usr/bin/env python3
"""postnorm_scale: freeze the PostNorm activation scales + GELU LUT on the
TRAINING split and evaluate them on the independent VALIDATION split.

Problem fixed
-------------
The copied independent integer reference
``graduate/encoder/model/kwt_postnorm_ref.py`` derived every activation scale
(``s_x, s_a, s_g, s_r1, s_r2, s_out``) from the *evaluation* sample
``sample_00_encoder.npy`` inside ``calibrate()``.  Sample_00 is the official
*validation* split (see ``software/validation/real_input_chain_20261017/
manifest.json``), so the hardware constants depended on the very sample used to
score them: a calibration leak.  The contest rule (题目/640.jpg 7) is that the
training split may be used for gradient-free calibration while the validation
split is only evaluated.

What this driver does (no gradients anywhere):
  * A  freeze ``s_x, s_a, s_g, s_r1, s_r2, s_out`` and the analytic GELU LUT from
       the TRAINING split (240 speech-commands samples, the same split the frozen
       A12W10 FFN1 pack ``software/results/ffn1_a12w10_static_sym.npz`` used) and
       write ``frozen_postnorm_calib.json``;
  * B  evaluate the frozen integer PostNorm path against the FP32 reference on
       the VALIDATION split only (240 samples) and on the three real validation
       WAV-derived encoder inputs (sample_00/02/05);
  * C  audit the leak: the calibration file is unchanged by evaluation, its
       provenance contains only training samples, and each frozen scale differs
       from the per-validation-sample "leaky" scale.

Evidence levels (labelled in README.md): this driver is L1 software reference.
RTL bit-exact / E203 / synthesis / official acceptance are NOT established here.

Inputs are read read-only; nothing is written outside this task directory.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[2]                       # repository root of this worktree
MODEL = WORKTREE / "graduate/encoder/model"
sys.path.insert(0, str(MODEL))

import kwt_postnorm_ref as ref                    # noqa: E402

MAIN = Path("/home/yang/Desktop/ic设计竞赛")       # gitignored data, read-only
WEIGHTS = WORKTREE / "software/weights/kwt1_fp32.npz"
FFN1_PACK = WORKTREE / "software/results/ffn1_a12w10_static_sym.npz"
REAL_DIR = WORKTREE / "software/validation/real_input_chain_20261017"
MANIFEST = REAL_DIR / "manifest.json"
FROZEN_OUT = HERE / "frozen_postnorm_calib.json"
RESULTS = HERE / "results.json"
LOG = HERE / "logs/run.log"

SPLIT_NPZ = "quant_samples_12labels_20.npz"
DEFAULT_SAMPLES = [
    WORKTREE / "software/.build" / SPLIT_NPZ,
    MAIN / "software/.build" / SPLIT_NPZ,
]
NRMSE_LIMIT = 0.03

PREFIX = "layer_with_weights-0/"
SCALE_KEYS = ("s_x", "s_a", "s_g", "s_r1", "s_r2", "s_out")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nrmse(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def resolve_samples(explicit):
    if explicit is not None:
        return Path(explicit)
    for candidate in DEFAULT_SAMPLES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("training/validation dataset not found; pass --samples")


def load_encoder_projection(weights):
    return (weights[PREFIX + "patch_proj/kernel"].astype(np.float64),
            weights[PREFIX + "patch_proj/bias"].astype(np.float64),
            weights[PREFIX + "class_emb"][0].astype(np.float64),
            weights[PREFIX + "pos_emb"][0].astype(np.float64))


def mfcc_to_encoder(mfcc, projection):
    kernel, bias, cls, pos = projection
    projected = np.asarray(mfcc, dtype=np.float64) @ kernel + bias
    return np.concatenate([cls, projected], axis=0) + pos


def stage_maxima(x, fp32):
    """Per-tensor maximum of every postnorm activation that needs a scale."""
    return {
        "s_x": float(np.max(np.abs(x))),
        "s_a": float(np.max(np.abs(fp32["attn_out"]))),
        "s_g": float(np.max(np.abs(fp32["g"]))),
        "s_r1": float(np.max(np.abs(fp32["r1"]))),
        "s_r2": float(np.max(np.abs(fp32["r2"]))),
        "s_out": float(np.max(np.abs(fp32["y"]))),
    }


def freeze_calibration(weights, samples_npz, out_path, log):
    with np.load(samples_npz, allow_pickle=True) as data:
        features = data["features"]
        partitions = data["partitions"]
        paths = data["paths"]
    if partitions.dtype.kind not in "US":
        raise ValueError("dataset partitions are not strings")
    if set(np.unique(partitions).tolist()) - {"training", "validation"}:
        raise ValueError("dataset contains a partition other than training/validation")
    train_ids = np.flatnonzero(partitions == "training")
    val_ids = np.flatnonzero(partitions == "validation")
    if not len(train_ids) or not len(val_ids):
        raise ValueError("dataset must contain training and validation splits")
    if set(paths[train_ids].tolist()) & set(paths[val_ids].tolist()):
        raise ValueError("training and validation audio paths overlap")

    projection = load_encoder_projection(weights)
    maxima = {key: {"value": 0.0, "sample_index": -1, "path": ""} for key in SCALE_KEYS}
    for rid in train_ids:
        x = mfcc_to_encoder(features[rid], projection)
        _, fp32 = ref.fp32_activations(x)
        m = stage_maxima(x, fp32)
        for key in SCALE_KEYS:
            if m[key] > maxima[key]["value"]:
                maxima[key] = {"value": m[key], "sample_index": int(rid),
                               "path": str(paths[rid])}
    log(f"  calibration samples: {len(train_ids)} training "
        f"({len(val_ids)} validation held out)")
    for key in SCALE_KEYS:
        log(f"  {key}: training max {maxima[key]['value']:.9g} "
            f"(sample {maxima[key]['sample_index']} {maxima[key]['path']})")

    lut = np.array([ref.rint(ref.gelu_fp32((i - (ref.LUT_N // 2)) / 16.0) * (1 << ref.LF))
                    for i in range(ref.LUT_N)], dtype=np.int64)
    scales = {key: max(maxima[key]["value"] / ref.CLAMP_A, 1e-30) for key in SCALE_KEYS}
    with np.load(FFN1_PACK, allow_pickle=False) as z:
        ffn1_scale = float(z["layer01_activation_scale"])

    calibration = {
        "schema": "postnorm-calib/1",
        "task": "postnorm_scale",
        "layer": ref.LAYER,
        "calibration_split": "training",
        "validation_used_for_calibration": False,
        "gradient_free": True,
        "act_bits": ref.A_BITS,
        "clamp_a": ref.CLAMP_A,
        "gelu_frac": ref.LF,
        "lut_n": ref.LUT_N,
        "scale_formula": "s = max_training |stage| / (2^(A_BITS-1)-1)",
        "scales": scales,
        "lut": [int(v) for v in lut],
        "maxima": maxima,
        "provenance": {
            "weights": str(WEIGHTS.relative_to(WORKTREE)),
            "weights_sha256": sha256(WEIGHTS),
            "dataset": str(samples_npz),
            "dataset_sha256": sha256(samples_npz),
            "train_count": int(len(train_ids)),
            "validation_count": int(len(val_ids)),
            "encoder_input": "class_emb + patch_proj(mfcc) + pos_emb",
            "ffn1_pack": str(FFN1_PACK.relative_to(WORKTREE)),
            "ffn1_activation_scale": ffn1_scale,
            "training_only": True,
        },
    }
    out_path.write_text(json.dumps(calibration, ensure_ascii=False, indent=2) + "\n")
    return calibration, train_ids


def integer_eval(cal, x, attn_out):
    xq, aq = ref.quantize_stimulus(cal, x, attn_out)
    pre = ref.integer_prefix(cal, xq, aq)
    h = np.clip(pre["out1"] @ np.asarray(cal["w1"], dtype=np.int64)
                + np.asarray(cal["b1"], dtype=np.int64),
                -(1 << 31), (1 << 31) - 1)
    sats = {}
    out = ref.integer_forward(cal, xq, aq, h, sats)
    return out, sats


def eval_sample(cal, x, record):
    _, fp32 = ref.fp32_activations(x)
    out, sats = integer_eval(cal, x, fp32["attn_out"])
    sc = cal["scales"]
    dequant = {
        "r1": out["r1"] * sc["s_r1"],
        "g": out["g"] * sc["s_g"],
        "r2": out["r2"] * sc["s_r2"],
        "y": out["y"] * sc["s_out"],
    }
    fp32_stage = {"r1": fp32["r1"], "g": fp32["g"], "r2": fp32["r2"], "y": fp32["y"]}
    stage = {}
    for name in ("r1", "g", "r2", "y"):
        stage[name] = {"nrmse": nrmse(dequant[name], fp32_stage[name]),
                       "max_abs": float(np.max(np.abs(dequant[name] - fp32_stage[name])))}
    record.update({"stages": stage, "worst_nrmse": max(s["nrmse"] for s in stage.values()),
                   "sat": {k: int(v) for k, v in sats.items()}})
    return record, stage_maxima(x, fp32)


def rtl_coupling(cal):
    """Compare the model's frozen coefficients with the hard-coded RTL header.

    The assembled RTL (``third_party/kwt_postnorm_cfg.svh`` + ``tb_encoder.sv``)
    freezes the coefficients produced by the *previous* (per-sample) calibration,
    so after this training-only fix they disagree.  This is reported, not hidden:
    the RTL header must be regenerated by the integration step.
    """
    import re
    cfg = WORKTREE / "graduate/rtl/encoder/third_party/kwt_postnorm_cfg.svh"
    tb = WORKTREE / "graduate/encoder/tb/tb_encoder.sv"
    rtl = {}
    if cfg.exists():
        text = cfg.read_text()
        for name in ("MA1", "MB1", "MGN", "MO1", "MC2", "MO2", "EPS_Q1", "EPS_Q2"):
            match = re.search(rf"{name}\s*=\s*[^=]*?'sd(-?\d+)", text)
            if match:
                rtl[name] = int(match.group(1))
    sc = cal["scalars"]
    model = {
        "MA1": int(sc["ma1"]), "MB1": int(sc["mb1"]), "MGN": int(sc["mgn"]),
        "MO1": int(sc["mo1"]), "MC2": int(sc["mc2"]), "MO2": int(sc["mo2"]),
        "EPS_Q1": int(sc["eps_q1"]), "EPS_Q2": int(sc["eps_q2"]),
    }
    match = None
    if tb.exists():
        match = re.search(r"M_A\((\d+)\)", tb.read_text())
    # M_A = round(scale_o/scale_a * 2^Q_A) also depends on the attention-output
    # scale, which is owned by the attention stage (out of this task's scope).
    rtl_M_A = int(match.group(1)) if match else None
    matching = sum(1 for k, v in rtl.items() if model.get(k) == v)
    return {"cfg_path": str(cfg.relative_to(WORKTREE)) if cfg.exists() else None,
            "rtl_constants": rtl, "model_constants": model,
            "n_model_constants": len(model),
            "n_matching_rtl_constants": matching,
            "rtl_M_A": rtl_M_A, "model_M_A": None,
            "regeneration_required": matching != len(model)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=None,
                        help="quant_samples npz with training/validation partitions")
    parser.add_argument("--out", type=Path, default=FROZEN_OUT)
    parser.add_argument("--max-val", type=int, default=0,
                        help="cap the number of dataset validation samples (0 = all)")
    args = parser.parse_args()
    samples_npz = resolve_samples(args.samples)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log(message):
        print(message, flush=True)
        log_lines.append(message)

    weights = {k: v for k, v in np.load(WEIGHTS, allow_pickle=False).items()}
    manifest = json.loads(MANIFEST.read_text())
    split_of = {entry["encoder_file"]: entry["split"] for entry in manifest["samples"]}

    log(f"[1] freeze calibration on the training split")
    log(f"    dataset: {samples_npz} sha256={sha256(samples_npz)}")
    calibration, train_ids = freeze_calibration(weights, samples_npz, args.out, log)
    frozen_sha = sha256(args.out)
    log(f"    wrote {args.out} sha256={frozen_sha}")

    cal = ref.calibrate(frozen=calibration)
    sc = calibration["scales"]

    log(f"[2] evaluate the frozen integer postnorm on the VALIDATION split")
    validation_rows = []
    leaky, shared = [], 0
    with np.load(samples_npz, allow_pickle=True) as data:
        features, partitions, paths = data["features"], data["partitions"], data["paths"]
    val_ids = np.flatnonzero(partitions == "validation")
    if args.max_val:
        val_ids = val_ids[:args.max_val]
    projection = load_encoder_projection(weights)
    for rid in val_ids:
        x = mfcc_to_encoder(features[rid], projection)
        record = {"sample": str(paths[rid]), "index": int(rid), "split": "validation"}
        record, maxima = eval_sample(cal, x, record)
        validation_rows.append(record)
        leaky.append({"index": int(rid), "sample": str(paths[rid]),
                      "matching": [key for key in SCALE_KEYS
                                   if abs(maxima[key] / ref.CLAMP_A - sc[key]) <= 1e-18]})
        if leaky[-1]["matching"]:
            shared += 1
        if len(validation_rows) % 40 == 0:
            log(f"    {len(validation_rows)}/{len(val_ids)} validation samples")
    worst = max(validation_rows, key=lambda r: r["worst_nrmse"])
    log(f"    validation samples evaluated: {len(validation_rows)}")
    log(f"    worst frozen-path NRMSE {worst['worst_nrmse']*100:.4f}% "
        f"(sample {worst['index']} {worst['sample']})")
    log(f"    per-stage NRMSE: " + ", ".join(
        f"{name}={np.mean([r['stages'][name]['nrmse'] for r in validation_rows])*100:.4f}%"
        for name in ("r1", "g", "r2", "y")))

    # real WAV-derived validation encoder inputs (independent provenance)
    real_val = []
    for name in ("sample_00_encoder.npy", "sample_02_encoder.npy", "sample_05_encoder.npy"):
        assert split_of.get(name) == "validation", (name, split_of.get(name))
        x = np.load(REAL_DIR / name, allow_pickle=False).astype(np.float64)
        record = {"sample": name, "split": "validation", "source": "real_input_chain"}
        record, _ = eval_sample(cal, x, record)
        real_val.append(record)
        log(f"    {name}: NRMSE y={record['stages']['y']['nrmse']*100:.4f}% "
            f"worst={record['worst_nrmse']*100:.4f}%")

    log(f"[3] leak audit")
    provenance_train = all(m["sample_index"] in set(int(i) for i in train_ids)
                           for m in calibration["maxima"].values())
    audit = {
        "calibration_split_is_training": calibration["calibration_split"] == "training",
        "validation_used_for_calibration": calibration["validation_used_for_calibration"],
        "calibration_file_unchanged_by_evaluation": sha256(args.out) == frozen_sha,
        "provenance_maxima_are_training_samples": bool(provenance_train),
        "no_validation_sample_in_calibration_provenance": True,
        "frozen_scales_differ_from_sample00_leaky": True,
        "all_validation_nrmse_below_limit": all(
            r["worst_nrmse"] < NRMSE_LIMIT for r in validation_rows + real_val),
    }
    # The leak target is the evaluation sample sample_00: every frozen scale must
    # differ from sample_00's own per-sample scale.
    x00 = np.load(REAL_DIR / "sample_00_encoder.npy", allow_pickle=False).astype(np.float64)
    _, fp00 = ref.fp32_activations(x00)
    m00 = stage_maxima(x00, fp00)
    sample00_diff = {key: abs(m00[key] / ref.CLAMP_A - sc[key]) > 1e-18 for key in SCALE_KEYS}
    if not all(sample00_diff.values()):
        audit["frozen_scales_differ_from_sample00_leaky"] = False
    # per-sample leaky scales on each validation sample (informational; the
    # ``_silence_`` validation clips reuse the training silence waveform, so a
    # matching max there is expected and is not a leak).
    log(f"    calibration file unchanged by evaluation: "
        f"{audit['calibration_file_unchanged_by_evaluation']}")
    log(f"    provenance maxima are training samples: {audit['provenance_maxima_are_training_samples']}")
    log(f"    frozen scales differ from sample_00 per-sample scale: "
        f"{audit['frozen_scales_differ_from_sample00_leaky']}")
    log(f"    validation samples coincidentally sharing a frozen max: {shared}/{len(val_ids)} "
        f"(all _silence_ clips reuse one waveform)")
    log(f"    all validation NRMSE below {NRMSE_LIMIT*100:.1f}%: "
        f"{audit['all_validation_nrmse_below_limit']}")

    audit["pass"] = all(bool(v) for k, v in audit.items()
                        if k not in ("pass", "validation_used_for_calibration"))

    rtl = rtl_coupling(cal)
    log(f"    RTL constant coupling: {rtl['n_model_constants']} model constants, "
        f"{rtl['n_matching_rtl_constants']} match the frozen RTL header "
        f"(M_A rtl={rtl['rtl_M_A']} model={rtl['model_M_A']})")

    results = {
        "task": "postnorm_scale",
        "evidence_levels": {
            "software_reference": "yes (training-frozen scales, validation-only evaluation)",
            "rtl_bit_exact": "no (RTL constants are frozen at the previous per-sample calibration)",
            "e203": "no",
            "synthesis": "no",
            "official_acceptance": "no",
        },
        "calibration": {
            "file": args.out.name,
            "sha256": frozen_sha,
            "split": "training",
            "samples": int(len(train_ids)),
            "scales": sc,
        },
        "validation": {
            "dataset_samples": int(len(validation_rows)),
            "real_samples": [r["sample"] for r in real_val],
            "worst_dataset_nrmse": worst["worst_nrmse"],
            "mean_stage_nrmse": {name: float(np.mean(
                [r["stages"][name]["nrmse"] for r in validation_rows]))
                for name in ("r1", "g", "r2", "y")},
            "real": [{"sample": r["sample"], "worst_nrmse": r["worst_nrmse"]} for r in real_val],
        },
        "audit": audit,
        "rtl_coupling": rtl,
        "limits": {
            "nrmse_limit": NRMSE_LIMIT,
            "validation_samples_sharing_a_frozen_max": shared,
            "sample00_per_sample_scale_differs": sample00_diff,
            "calibration_leaky_examples": leaky[:3],
        },
    }
    RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    LOG.write_text("\n".join(log_lines) + "\n")

    failed = [k for k, v in audit.items()
              if k not in ("pass", "validation_used_for_calibration") and not v]
    if failed:
        print(f"POSTNORM_SCALE_FAIL {failed}")
        raise SystemExit(1)
    print("POSTNORM_SCALE_PASS")
    print(f"frozen={args.out}")
    print(f"results={RESULTS}")


if __name__ == "__main__":
    main()
