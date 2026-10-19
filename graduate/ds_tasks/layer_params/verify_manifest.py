#!/usr/bin/env python3
"""Re-verify a frozen layer manifest end-to-end from the artifact itself.

This is a *gate*, not a builder: it reloads the frozen NPZ/JSON, re-collects the
FP32 FFN1 inputs, and checks that

  1. every activation scale equals ``max|training_x| / qmax_a`` (training only),
  2. every weight scale equals ``max|W[:, c]| / qmax_w`` (frozen FP32 weights),
  3. every integer weight/bias equals the manifest encoding,
  4. every layer's validation NRMSE recomputes to the recorded value, within a
     documented cross-device tolerance, and stays <= 3%,
  5. no validation sample was used to derive a scale.

Hashes, weight scales and integer tensors are checked *bit-exactly* (equality,
not tolerance). Two floating-point quantities -- the NRMSE and the provenance of
the activation scale -- carry a documented cross-device tolerance, because both
depend on FP32 reductions over the FFN1 activations: the NRMSE's reference term
(``val_x @ W + b``) is a CPU BLAS reduction, and the scale's ``max|train_x|``
drifts between CUDA and CPU kernels. See NRMSE_RTOL and SCALE_RTOL below for the
measured bounds. The 3% gate and the worst-layer fields are asserted
independently on the measured values.

Exit code 0 only when all checks pass.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

LAYERS, M, K, N = 12, 99, 64, 256
INT32_MAX = (1 << 31) - 1
INT32_MIN = -(1 << 31)
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SOFTWARE = REPO / "software"

# Cross-device re-verification tolerance for the floating-point NRMSE only.
#
# ``nrmse`` compares the dequantized integer result against the FP32 reference
# ``val_x @ W + b``. That reference is a CPU BLAS reduction, so its summation
# order -- and therefore the recorded percentage -- depends on the BLAS/OMP
# thread count (even at a fixed device) and on the device that produced the
# FFN1 inputs (the recorded manifest was frozen on CUDA). The value is thus
# reproducible to ~1e-5 relative, never to the 1e-9 absolute that a naive gate
# would demand.
#
# Measured worst-case deviation from the recorded values, torch 2.5.1+cu121 /
# NumPy 2.1.3 (see logs/verify_auto.log, logs/verify_cpu.log):
#   * same device (CUDA activations), OpenBLAS/OMP threads 1/2/8/12 vs the
#     recorded thread-4 ordering: max |d| ~ 2.0e-8 pp, rel ~ 1.3e-7;
#   * cross device (CUDA-frozen manifest, CPU re-verify via --device cpu):
#     max |d| ~ 1.5e-6 pp, rel ~ 1.4e-5 (the largest relative error sits on the
#     small-NRMSE layer 2).
# We allow rtol = 1e-3 -- ~70x the worst observed cross-device deviation, while
# still ~3e2x tighter than the gap between candidate schemes (A11W10 vs A12W10
# differ by ~3.5e-2 pp) and ~3e4x tighter than the 3% contest gate -- plus a
# 1e-6 percentage-point absolute floor.
NRMSE_RTOL = 1e-3
NRMSE_ATOL_PCT = 1e-6

# Cross-device tolerance for the *provenance* of the activation scale only.
#
# ``sx`` is recomputed from the training activations produced by the current
# device: ``max|train_x| / qmax_a``. Those activations are the FP32 FFN1 inputs
# and therefore drift slightly between CUDA (cuDNN) and CPU kernels, so the
# CUDA-frozen ``sx`` is not bit-identical when re-derived on CPU. The check is
# still a provenance gate -- it must reject a scale derived from the *validation*
# split -- and the tolerance is far below that distinction:
#   * measured CUDA-frozen vs CPU-recomputed scale drift (this run): rel <= 2.7e-7;
#   * train-derived vs validation-derived scale gap: rel >= 1.14e-2 (smallest on
#     layer 6). SCALE_RTOL = 1e-5 sits ~40x above the observed drift and ~1e3x
#     below the provenance gap, so using validation data would still fail.
# The frozen NPZ scale itself is never altered; the integer weights/biases and
# all hashes are still checked bit-exactly.
SCALE_RTOL = 1e-5


def nrmse_tolerance(recorded):
    return NRMSE_ATOL_PCT + NRMSE_RTOL * abs(recorded)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def qsym(values, scale, qmax):
    return np.clip(np.rint(np.asarray(values, dtype=np.float64) / scale),
                   -qmax, qmax).astype(np.int64)


def nrmse(approx, reference):
    count = reference.size
    err2 = float(np.sum((approx - reference.astype(np.float64)) ** 2))
    ref2 = float(np.sum(reference.astype(np.float64) ** 2))
    return 100.0 * np.sqrt(err2 / count) / max(np.sqrt(ref2 / count), 1e-8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=HERE / "layer_manifest_A12W10.json")
    parser.add_argument("--samples", type=Path,
                        default=Path("/home/yang/Desktop/ic设计竞赛/software/.build/"
                                     "quant_samples_12labels_40.npz"))
    parser.add_argument("--weights", type=Path, default=SOFTWARE / "weights/kwt1_fp32.npz")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    npz_path = args.manifest.with_name(manifest["artifacts_npz"])
    src = manifest["source"]
    failures = []

    def check(condition, message):
        if not condition:
            failures.append(message)
            print(f"FAIL {message}")
        else:
            print(f"ok   {message}")

    check(sha256(args.samples) == src["sample_sha256"], "sample cache sha256")
    check(sha256(args.weights) == src["weight_sha256"], "weight archive sha256")
    check(sha256(npz_path) == src["npz_sha256"], "frozen npz sha256")
    check(manifest["validation_used_for_calibration"] is False,
          "manifest declares no validation calibration")
    check(manifest["test_partition_used"] is False, "manifest declares no test partition")

    with np.load(args.samples, allow_pickle=False) as data:
        features, partitions, paths = (data["features"], data["partitions"], data["paths"])
    train_ids = np.flatnonzero(partitions == "training")
    val_ids = np.flatnonzero(partitions == "validation")
    check(len(train_ids) == manifest["training_count"], "training sample count")
    check(len(val_ids) == manifest["validation_count"], "validation sample count")
    check(not set(paths[train_ids]).intersection(paths[val_ids]), "no train/val path overlap")

    import torch
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if str(SOFTWARE) not in sys.path:
        sys.path.insert(0, str(SOFTWARE))
    from run import KWT1, PREFIX

    model = KWT1(args.weights, device)
    acts = []
    with torch.inference_mode():
        for sample in features:
            enc_in = model.encoder_input(torch.from_numpy(sample).to(device))
            _, _, ffn_inputs = model.forward(enc_in, capture_ffn=True)
            acts.append(np.stack([a.cpu().numpy() for a in ffn_inputs]))
    acts = np.stack(acts)
    check(acts.shape == (len(features), LAYERS, M, K), f"activation shape {acts.shape}")

    q = manifest["quantization"]
    a_bits, w_bits = q["activation_bits"], q["weight_bits"]
    qmax_a, qmax_w = (1 << (a_bits - 1)) - 1, (1 << (w_bits - 1)) - 1
    check(q["activation_qmax"] == qmax_a and q["weight_qmax"] == qmax_w, "qmax fields")

    frozen = np.load(npz_path, allow_pickle=False)
    recomputed_worst = 0.0
    worst_nrmse_dev = 0.0
    for layer in range(LAYERS):
        name = f"layer{layer + 1:02}"
        base = f"{PREFIX}/enc_layers/{layer}/ffn/layer_with_weights-0"
        weight = model.weight(base + "/kernel").cpu().numpy()
        bias = model.weight(base + "/bias").cpu().numpy()
        train_x, val_x = acts[train_ids, layer], acts[val_ids, layer]

        sx_expect = max(float(np.max(np.abs(train_x.astype(np.float64)))) / qmax_a, 1e-12)
        sw_expect = np.maximum(np.max(np.abs(weight.astype(np.float64)), axis=0) / qmax_w, 1e-12)
        sx = float(frozen[name + "_activation_scale"])
        sw = frozen[name + "_weight_scales"].astype(np.float64)
        qw_raw = frozen[name + "_weight"]
        qb_raw = frozen[name + "_bias"]
        qw = qw_raw.astype(np.int64)
        qb = qb_raw.astype(np.int64)
        check(abs(sx - sx_expect) <= SCALE_RTOL * abs(sx_expect),
              f"L{layer+1} scale from training only "
              f"(sx={sx:.6e} train_max/{qmax_a}={sx_expect:.6e}, "
              f"rel={abs(sx - sx_expect)/abs(sx_expect):.2e} <= {SCALE_RTOL:g})")
        check(np.array_equal(sw, sw_expect), f"L{layer+1} weight scales from frozen FP32")
        check(np.array_equal(qw, qsym(weight, sw, qmax_w)), f"L{layer+1} integer weights")
        check(np.array_equal(qb, np.rint(bias.astype(np.float64) / (sx * sw)).astype(np.int64)),
              f"L{layer+1} integer bias")
        check(qw_raw.shape == (K, N) and qw_raw.dtype == np.int16, f"L{layer+1} weight shape/dtype")
        check(qb_raw.shape == (N,) and qb_raw.dtype == np.int32, f"L{layer+1} bias shape/dtype")
        check(int(qw.min()) >= -qmax_w and int(qw.max()) <= qmax_w, f"L{layer+1} weight range")

        acc = (qsym(val_x, sx, qmax_a).reshape(-1, K) @ qw).reshape(-1, M, N) + qb
        check(acc.min() >= INT32_MIN and acc.max() <= INT32_MAX, f"L{layer+1} INT32 accumulate")
        approx = acc.astype(np.float64) * (sx * sw)
        measured = nrmse(approx, val_x @ weight + bias)
        recorded = manifest["layers"][layer]["validation_nrmse_pct"]
        deviation = abs(measured - recorded)
        worst_nrmse_dev = max(worst_nrmse_dev, deviation)
        tol = nrmse_tolerance(recorded)
        check(deviation <= tol,
              f"L{layer+1} validation NRMSE {measured:.6f}% "
              f"(recorded {recorded:.6f}%, |d|={deviation:.2e} <= {tol:.2e})")
        check(measured <= manifest["contest_target_nrmse_pct"],
              f"L{layer+1} below {manifest['contest_target_nrmse_pct']}% target")
        recomputed_worst = max(recomputed_worst, measured)

    worst_tol = nrmse_tolerance(manifest["worst_validation_nrmse_pct"])
    check(abs(recomputed_worst - manifest["worst_validation_nrmse_pct"]) <= worst_tol,
          f"worst validation NRMSE {recomputed_worst:.6f}% "
          f"(recorded {manifest['worst_validation_nrmse_pct']:.6f}%, "
          f"|d|={abs(recomputed_worst - manifest['worst_validation_nrmse_pct']):.2e} "
          f"<= {worst_tol:.2e})")
    check(manifest["worst_layer"] == max(manifest["layers"],
                                         key=lambda x: x["validation_nrmse_pct"])["layer"],
          "worst layer field")

    if failures:
        print(f"VERIFY_FAIL {len(failures)} check(s)")
        return 1
    print(f"VERIFY_PASS samples={len(train_ids)}/{len(val_ids)} "
          f"worst={recomputed_worst:.6f}% layer={manifest['worst_layer']} "
          f"nrmse_rtol={NRMSE_RTOL:g} max_dev={worst_nrmse_dev:.2e}pp")
    return 0


if __name__ == "__main__":
    sys.exit(main())
