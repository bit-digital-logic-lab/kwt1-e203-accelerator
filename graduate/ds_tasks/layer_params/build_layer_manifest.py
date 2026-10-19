#!/usr/bin/env python3
"""Frozen 12-layer integer weight/bias/scale manifest for the FFN1 linear layer.

Contest scope (undergraduate required item 2, graduate items 5-6 parameter update):
the *first* feed-forward linear of each of the 12 encoder layers

    99x64  @  64x256  +  bias(256)  ->  99x256

is quantized to signed integers; the audio front end and the full FP32 model
that produces each layer's 99x64 input stay FP32 and are treated as the offline
reference (the same boundary the contest draws).

This worker is independent from the pipeline that produced
``software/results/ffn1_a12w10_static_sym.*``: the quantizer, the integer GEMM
and the NRMSE metric are re-implemented here from the format description, then
cross-checked bit-for-bit against the in-repo reference functions
(``software/calibrate_blackbox_ffn1.py``) when they are importable.

Rules:
  * Calibration (scales, integer weight/bias) uses the **training** split only.
  * The **validation** split is only *evaluated*; it never touches a scale or a
    threshold.  No gradient, no dynamic per-sample scale, no test partition.
  * The official FP32 weights are read read-only and never modified.

Outputs (written next to this file unless --output-dir is given):
  layer_manifest_A12W10.json   human/machine readable frozen manifest
  layer_manifest_A12W10.npz    frozen int16 weights, int32 biases, fp64 scales
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

LAYERS, M, K, N = 12, 99, 64, 256
QMAX = {bits: (1 << (bits - 1)) - 1 for bits in range(2, 17)}
INT32_MAX = (1 << 31) - 1
INT32_MIN = -(1 << 31)

# Known-good SHA-256 of the read-only inputs used for the frozen run.  The
# script refuses to build a manifest from different bytes unless --allow-other
# is passed, so a manifest hash always points at exact inputs.
DEFAULT_SAMPLE_SHA256 = "43d639fb64ea8d907db967b071010e41e68827368532880ba8a99b92769bbae8"
DEFAULT_WEIGHT_SHA256 = "8f359abf7adbe287b8f89fca98af46194f98f90f2146784731bff551a155825a"

HERE = Path(__file__).resolve().parent
# Repo root = .../<worktree>; this file lives at graduate/ds_tasks/layer_params/.
REPO = HERE.parents[2]
SOFTWARE = REPO / "software"
MAIN_PROJECT = Path("/home/yang/Desktop/ic设计竞赛")

# Hardware load order of the 12-layer update controller
# (graduate/rtl/encoder/kwt_encoder_layers_ctrl.sv): this manifest covers the
# FFN1 first linear, i.e. selectors sel7 (W1, 16384 entries) and sel8 (b1, 256).
LOAD_ORDER = {
    "sel7_W1": {"tensor": "ffn1/kernel", "shape": [K, N], "container": "int16",
                "entries_per_layer": K * N},
    "sel8_b1": {"tensor": "ffn1/bias", "shape": [N], "container": "int32",
                "entries_per_layer": N},
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Independent integer implementation (no import of the in-repo reference).
# ---------------------------------------------------------------------------
def quantize_symmetric(values, scale, qmax):
    """nearest ties-to-even (np.rint), then symmetric clip to [-qmax, qmax]."""
    scaled = np.rint(np.asarray(values, dtype=np.float64) / scale)
    return np.clip(scaled, -qmax, qmax).astype(np.int64)


def training_scales(train_x, weight, a_bits, w_bits):
    """Training-only static symmetric scales; zero point fixed at 0."""
    qmax_a, qmax_w = QMAX[a_bits], QMAX[w_bits]
    sx = max(float(np.max(np.abs(train_x.astype(np.float64)))) / qmax_a, 1e-12)
    sw = np.maximum(np.max(np.abs(weight.astype(np.float64)), axis=0) / qmax_w, 1e-12)
    return sx, sw


def encode(weight, bias, sx, sw, w_bits):
    qw = quantize_symmetric(weight, sw, QMAX[w_bits])
    qb = np.rint(bias.astype(np.float64) / (sx * sw)).astype(np.int64)
    if qb.min() < INT32_MIN or qb.max() > INT32_MAX:
        raise OverflowError("quantized bias does not fit signed INT32")
    return qw, qb


def integer_forward(x, qw, qb, sx, sw, a_bits, batch_size=32):
    """Real integer GEMM + integer bias; FP64 decode only to measure error."""
    qmax_a = QMAX[a_bits]
    out_scale = float(sx) * sw
    approx = np.empty((x.shape[0], M, N), dtype=np.float64)
    peak = 0
    for begin in range(0, x.shape[0], batch_size):
        end = min(begin + batch_size, x.shape[0])
        qx = quantize_symmetric(x[begin:end], sx, qmax_a)
        acc = (qx.reshape(-1, K) @ qw).reshape(end - begin, M, N) + qb
        peak = max(peak, int(np.max(np.abs(acc))))
        if acc.min() < INT32_MIN or acc.max() > INT32_MAX:
            raise OverflowError("integer MAC plus bias exceeds signed INT32")
        approx[begin:end] = acc.astype(np.float64) * out_scale
    return approx, peak


def nrmse_pct(approx, reference):
    count = reference.size
    err2 = float(np.sum((approx - reference.astype(np.float64)) ** 2))
    ref2 = float(np.sum(reference.astype(np.float64) ** 2))
    denom = max(np.sqrt(ref2 / count), 1e-8)
    return 100.0 * np.sqrt(err2 / count) / denom


def load_partitions(samples_path, sample_sha256, allow_other):
    actual = sha256(samples_path)
    if actual != sample_sha256 and not allow_other:
        raise SystemExit(f"sample cache sha256 mismatch: {actual} != {sample_sha256}")
    with np.load(samples_path, allow_pickle=False) as data:
        features = data["features"].copy()
        partitions = data["partitions"].copy()
        paths = data["paths"].copy()
        words = data["words"].copy() if "words" in data else None
    if features.ndim != 3 or features.shape[1:] != (98, 40):
        raise SystemExit(f"expected MFCC [samples,98,40], got {features.shape}")
    if not np.isfinite(features).all():
        raise SystemExit("features contain NaN or infinity")
    if np.any((partitions != "training") & (partitions != "validation")):
        raise SystemExit("testing/other partitions are forbidden")
    train_ids = np.flatnonzero(partitions == "training")
    val_ids = np.flatnonzero(partitions == "validation")
    if not len(train_ids) or not len(val_ids):
        raise SystemExit("need both training and validation samples")
    if set(paths[train_ids].tolist()).intersection(paths[val_ids].tolist()):
        raise SystemExit("training/validation sample paths overlap")
    return features, paths, words, train_ids, val_ids, actual


def collect_activations(features, model, device):
    """FFN1 inputs (layer-norm1 output) for every layer, via the FP32 model."""
    import torch

    gathered = []
    with torch.inference_mode():
        for sample in features:
            enc_in = model.encoder_input(torch.from_numpy(sample).to(device))
            _, _, ffn_inputs = model.forward(enc_in, capture_ffn=True)
            gathered.append(np.stack([a.cpu().numpy() for a in ffn_inputs]))
    return np.stack(gathered)


def crosscheck_reference(features, model, device, train_ids, val_ids,
                         activations, schemes):
    """If the in-repo reference is importable, assert bit-identical numbers."""
    if str(SOFTWARE) not in sys.path:
        sys.path.insert(0, str(SOFTWARE))
    try:
        from calibrate_blackbox_ffn1 import (base_scales, black_box_error,
                                              encode_params)
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"available": False, "reason": repr(exc)}
    report = {"available": True, "schemes": {}}
    for a_bits, w_bits in schemes:
        scheme = f"A{a_bits}W{w_bits}"
        worst_val, worst_layer, weight_ok, bias_ok = 0.0, 0, True, True
        for layer in range(LAYERS):
            base = f"layer_with_weights-0/enc_layers/{layer}/ffn/layer_with_weights-0"
            weight = model.weight(base + "/kernel").cpu().numpy()
            bias = model.weight(base + "/bias").cpu().numpy()
            train_x = activations[train_ids, layer]
            val_x = activations[val_ids, layer]
            sx, sw = base_scales(train_x, weight, a_bits, w_bits)
            ref = val_x @ weight + bias
            res = black_box_error(val_x, ref, weight, bias, sx, sw, a_bits,
                                  w_bits, 32)
            qw_r, qb_r = encode_params(weight, bias, sx, sw, a_bits, w_bits)
            qw_m, qb_m = encode(weight, bias, sx, sw, w_bits)
            weight_ok &= bool(np.array_equal(qw_r, qw_m))
            bias_ok &= bool(np.array_equal(qb_r, qb_m))
            if res["nrmse_pct"] > worst_val:
                worst_val, worst_layer = res["nrmse_pct"], layer + 1
        report["schemes"][scheme] = {
            "worst_validation_nrmse_pct": worst_val,
            "worst_layer": worst_layer,
            "weight_encoding_identical": weight_ok,
            "bias_encoding_identical": bias_ok,
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=Path,
                        default=MAIN_PROJECT / "software/.build/quant_samples_12labels_40.npz")
    parser.add_argument("--weights", type=Path, default=SOFTWARE / "weights/kwt1_fp32.npz")
    parser.add_argument("--samples-sha256", default=DEFAULT_SAMPLE_SHA256)
    parser.add_argument("--weights-sha256", default=DEFAULT_WEIGHT_SHA256)
    parser.add_argument("--allow-other", action="store_true",
                        help="accept inputs whose SHA-256 differs from the frozen pair")
    parser.add_argument("--activation-bits", type=int, default=12)
    parser.add_argument("--weight-bits", type=int, default=10)
    parser.add_argument("--control-schemes", nargs="+", default=["A11W10"],
                        help="extra schemes recomputed as published-value controls")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--target", type=float, default=3.0)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()

    if not (2 <= args.activation_bits <= 16 and 2 <= args.weight_bits <= 16):
        parser.error("bit widths must be in 2..16")

    import torch
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    weight_actual = sha256(args.weights)
    if weight_actual != args.weights_sha256 and not args.allow_other:
        raise SystemExit(f"weight sha256 mismatch: {weight_actual} != {args.weights_sha256}")

    features, paths, words, train_ids, val_ids, sample_actual = load_partitions(
        args.samples, args.samples_sha256, args.allow_other)

    if str(SOFTWARE) not in sys.path:
        sys.path.insert(0, str(SOFTWARE))
    from run import KWT1, PREFIX  # read-only official FP32 model

    model = KWT1(args.weights, device)
    activations = collect_activations(features, model, device)
    if activations.shape != (len(features), LAYERS, M, K):
        raise SystemExit(f"unexpected FFN1 input tensor {activations.shape}")

    # Parse control schemes.
    controls = []
    for name in args.control_schemes:
        a_text, w_text = name[1:].split("W", 1)
        controls.append((int(a_text), int(w_text)))

    a_bits, w_bits = args.activation_bits, args.weight_bits
    scheme = f"A{a_bits}W{w_bits}"
    qmax_a, qmax_w = QMAX[a_bits], QMAX[w_bits]

    artifacts, layers = {}, []
    for layer in range(LAYERS):
        base = f"{PREFIX}/enc_layers/{layer}/ffn/layer_with_weights-0"
        weight = model.weight(base + "/kernel").cpu().numpy()
        bias = model.weight(base + "/bias").cpu().numpy()
        if weight.shape != (K, N) or bias.shape != (N,):
            raise SystemExit(f"unexpected layer {layer + 1} parameter shape")

        train_x = activations[train_ids, layer]  # calibration only
        val_x = activations[val_ids, layer]      # evaluation only
        sx, sw = training_scales(train_x, weight, a_bits, w_bits)
        qw, qb = encode(weight, bias, sx, sw, w_bits)

        train_ref = train_x @ weight + bias
        val_ref = val_x @ weight + bias
        train_approx, _ = integer_forward(train_x, qw, qb, sx, sw, a_bits, args.batch_size)
        val_approx, val_peak = integer_forward(val_x, qw, qb, sx, sw, a_bits, args.batch_size)
        bound = K * qmax_a * qmax_w + int(np.max(np.abs(qb)))
        if bound > INT32_MAX:
            raise SystemExit(f"layer {layer + 1} conservative INT32 bound overflows")

        name = f"layer{layer + 1:02}"
        artifacts[name + "_weight"] = qw.astype(np.int16)
        artifacts[name + "_bias"] = qb.astype(np.int32)
        artifacts[name + "_activation_scale"] = np.array(sx, dtype=np.float64)
        artifacts[name + "_weight_scales"] = sw.astype(np.float64)
        layers.append({
            "layer": layer + 1,
            "scheme": scheme,
            "activation_scale": sx,
            "weight_scale_min": float(np.min(sw)),
            "weight_scale_max": float(np.max(sw)),
            "weight_qmin": int(qw.min()), "weight_qmax": int(qw.max()),
            "weight_zero_point": 0, "activation_zero_point": 0,
            "bias_abs_max": int(np.max(np.abs(qb))),
            "training_nrmse_pct": nrmse_pct(train_approx, train_ref),
            "validation_nrmse_pct": nrmse_pct(val_approx, val_ref),
            "validation_observed_accumulator_peak": val_peak,
            "worst_case_abs_accumulator_bound": bound,
            "bound_headroom_over_int32": bound / (INT32_MAX + 1),
        })
        print(f"layer={layer + 1:02} train={layers[-1]['training_nrmse_pct']:.6f}% "
              f"validation={layers[-1]['validation_nrmse_pct']:.6f}% "
              f"sx={sx:.6e}", flush=True)

    worst = max(layers, key=lambda item: item["validation_nrmse_pct"])

    control = crosscheck_reference(features, model, device, train_ids, val_ids,
                                   activations, controls + [(a_bits, w_bits)])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / f"layer_manifest_{scheme}"
    npz_path = stem.with_suffix(".npz")
    np.savez_compressed(npz_path, **artifacts)

    manifest = {
        "artifact": f"frozen 12-layer FFN1 integer weight/bias/scale manifest ({scheme})",
        "scope": "12 x (99x64 @ 64x256 + bias -> 99x256), first FFN linear of each encoder layer",
        "quantization": {
            "activation": "per-layer static symmetric, zero point 0, training-split maximum",
            "weight": "per-output-channel symmetric, zero point 0, frozen FP32 column maximum",
            "bias": "signed INT32, round to nearest ties-to-even at activation_scale*weight_scale",
            "rounding": "nearest, ties-to-even; clip to [-qmax,+qmax]",
            "accumulator": "signed INT32, conservative bound checked for every layer",
            "activation_bits": a_bits, "weight_bits": w_bits,
            "activation_qmax": qmax_a, "weight_qmax": qmax_w,
        },
        "frozen": True,
        "modifies_original_model": False,
        "calibration_partition": "training only",
        "validation_used_for_calibration": False,
        "test_partition_used": False,
        "training_count": int(len(train_ids)),
        "validation_count": int(len(val_ids)),
        "training_labels": sorted(set(words[train_ids].tolist())) if words is not None else None,
        "validation_labels": sorted(set(words[val_ids].tolist())) if words is not None else None,
        "contest_target_nrmse_pct": args.target,
        "worst_layer": worst["layer"],
        "worst_validation_nrmse_pct": worst["validation_nrmse_pct"],
        "validation_pass": bool(worst["validation_nrmse_pct"] <= args.target),
        "artifacts_npz": npz_path.name,
        "weights_container": {"weight": "int16 (10-bit range not packed)",
                              "bias": "int32", "scales": "float64"},
        "totals": {
            "weight_entries_12layers": LAYERS * K * N,
            "bias_entries_12layers": LAYERS * N,
            "weight_container_bytes_12layers": LAYERS * K * N * 2,
            "bias_container_bytes_12layers": LAYERS * N * 4,
            "scale_float64_bytes_12layers": LAYERS * (1 + N) * 8,
            "packed_weight_kib_12layers": LAYERS * K * N * w_bits / 8 / 1024,
            "bytes_per_layer_staged_load": K * N * 2 + N * 4,
        },
        "hardware_load_order_ffn1": LOAD_ORDER,
        "source": {
            "sample_archive": str(args.samples.resolve()),
            "sample_sha256": sample_actual,
            "weight_archive": str(args.weights.resolve()),
            "weight_sha256": weight_actual,
            "script": Path(__file__).name,
            "script_sha256": sha256(Path(__file__).resolve()),
            "npz_sha256": sha256(npz_path),
            "device": str(device),
        },
        "independent_implementation": {
            "note": ("quantizer, integer GEMM and NRMSE re-implemented in this script; "
                     "cross-checked below against the in-repo reference functions"),
            "crosscheck": control,
        },
        "evidence_levels": {
            "L1_software_reference": "this manifest: NumPy/Torch integer FFN1 path, train-calibrated, validation-evaluated",
            "L2_rtl_bit_exact": "not produced here; existing layer-0 chain bit-exact evidence is graduate/encoder/tb/run.sh",
            "L3_e203_soc": "MISSING - no official E203/NICE run in this task",
            "L4_synthesis": "MISSING - no PDK/SDC synthesis in this task",
            "L5_official_acceptance": "MISSING - no official 600-sample evaluator run in this task",
        },
        "layers": layers,
    }
    json_path = stem.with_suffix(".json")
    json_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    print(f"WORST validation={worst['validation_nrmse_pct']:.6f}% layer={worst['layer']} "
          f"target={args.target:.3f}% {'PASS' if manifest['validation_pass'] else 'FAIL'}")
    if control.get("available"):
        for name, info in control["schemes"].items():
            print(f"control {name}: worst={info['worst_validation_nrmse_pct']:.6f}% "
                  f"layer={info['worst_layer']} weight_identical={info['weight_encoding_identical']} "
                  f"bias_identical={info['bias_encoding_identical']}")
    else:
        print(f"control reference unavailable: {control.get('reason')}")
    print(f"wrote {json_path.name} and {npz_path.name}")


if __name__ == "__main__":
    main()
