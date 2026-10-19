#!/usr/bin/env python3
"""r2_full_numeric -- 12-layer, multi-sample numeric check on frozen round-1 assets.

This runner deliberately does NOT recalibrate anything.  It consumes the
round-1 frozen pack produced by ``encoder_12numeric``:

    graduate/ds_tasks/encoder_12numeric/frozen_12layer_calib.json
    graduate/ds_tasks/encoder_12numeric/frozen_12layer_int.npz

and evaluates the frozen integer 12-layer chain on the *validation* partition
(240 dataset samples + 6 real WAV encoder inputs = 246 samples).  Training data
is never touched, and no scale / shift / threshold / LUT is fitted here.

Metrics
-------
* pooled NRMSE per layer and stage over all evaluated samples:
      sqrt(sum((int_dequant - fp32_ref)^2) / sum(fp32_ref^2))
  with denominator floor 1e-8.  This is the closest local analogue of the
  competition's "pool all samples" wording (it is NOT the official 600-sample
  evaluator).
* per-sample max NRMSE, kept for transparency.

Evidence levels emitted by this script: L1 (software reference only).
L2/RTL, L3/E203, L4/synthesis, L5/official acceptance are not produced here.

Usage:
    OPENBLAS_NUM_THREADS=4 python3 model/run_full_numeric.py
    ... --frozen-dir /path/to/encoder_12numeric --limit 12
Exit codes:
    0  all 12 layers satisfy the 3% pooled-NRMSE gate
    2  frozen-asset contract violation (configuration conflict)
    3  numeric gate failed (>= 3% on at least one layer)
    1  unexpected runtime failure
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TASK = HERE.parent
sys.path.insert(0, str(HERE))
import chain as C                                  # noqa: E402
import ref12 as R                                  # noqa: E402
import kwt1_fp32_golden as golden                  # noqa: E402

NRMSE_LIMIT = 0.03
STAGES = ["a", "r1", "g", "f", "r2", "y"]
GOLDEN_KEY = {"a": "attn_projected", "r1": "residual1", "g": "gelu",
              "f": "ffn_out", "r2": "residual2", "y": "layer_output"}
DEFAULT_FROZEN = Path("/home/yang/kwt-worktrees/ds-grad-encoder_12numeric/"
                      "graduate/ds_tasks/encoder_12numeric")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_frozen(calib_json: Path, int_npz: Path):
    """Reconstruct the integer chain dict from the round-1 frozen pack."""
    cal = json.loads(calib_json.read_text())
    z = np.load(int_npz)
    layers = []
    for L in cal["layers"]:
        k = L["layer"]
        pre = f"layer{k + 1:02d}"
        sc = L["scales"]
        sw1 = np.asarray(L["ffn1_sw"], dtype=np.float64)
        sw2 = np.asarray(L["ffn2_sw"], dtype=np.float64)
        attn = dict(
            params={kk: vv for kk, vv in L["attn_params"].items()},
            lut=np.asarray(cal["attn_lut"], dtype=np.int64),
            wq=z[f"{pre}_attn_wq"].astype(np.int64),
            wk=z[f"{pre}_attn_wk"].astype(np.int64),
            wv=z[f"{pre}_attn_wv"].astype(np.int64),
            wo=z[f"{pre}_attn_wo"].astype(np.int64),
            bias=z[f"{pre}_attn_bias"].astype(np.int64),
        )
        layers.append(dict(
            layer=k, M_A=int(L["M_A"]), scales=sc, attn=attn,
            ffn1=dict(w=z[f"{pre}_w1"].astype(np.int64),
                      b=z[f"{pre}_b1"].astype(np.int64),
                      sw=sw1, sh=sc["sx_ffn1"] * sw1),
            ffn2=dict(w=z[f"{pre}_w2"].astype(np.int64),
                      b=z[f"{pre}_b2"].astype(np.int64),
                      sw=sw2, sh2=sc["s_g"] * sw2),
            coeff={kk: (np.asarray(vv, dtype=np.int64) if isinstance(vv, list) else vv)
                   for kk, vv in L["coeff"].items()},
        ))
    return dict(schema=cal["schema"], layers=layers, lut=cal["lut"],
                attn_lut=cal["attn_lut"]), cal


def contract_check(calib, cal_json):
    """Check the frozen pack is internally usable before doing any numeric run."""
    problems = []
    if len(calib["layers"]) != 12:
        problems.append(f"layer_count={len(calib['layers'])} != 12")
    for L in calib["layers"]:
        k = L["layer"]
        sc = L["scales"]
        for name in ("s_x", "s_a", "sx_ffn1", "s_g", "s_r1", "s_r2", "s_out"):
            if not (isinstance(sc[name], (int, float)) and math.isfinite(sc[name])
                    and sc[name] > 0):
                problems.append(f"layer{k}: scale {name}={sc[name]!r} invalid")
        if not (0 < L["M_A"] < (1 << 20)):
            problems.append(f"layer{k}: M_A={L['M_A']} out of range")
        if k + 1 < len(calib["layers"]):
            nxt = calib["layers"][k + 1]["scales"]["s_x"]
            if abs(nxt - sc["s_out"]) > 1e-12 * max(1.0, abs(sc["s_out"])):
                problems.append(
                    f"layer{k}->{k + 1}: s_out={sc['s_out']:.12g} != "
                    f"next s_x={nxt:.12g}")
    # sanity of the frozen JSON's own claims (asset provenance, not calibration)
    if cal_json.get("calibration_split") != "training":
        problems.append("frozen calib did not declare training split")
    if cal_json.get("validation_used_for_calibration") is not False:
        problems.append("frozen calib does not declare validation unused")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen-dir", default=str(DEFAULT_FROZEN),
                    help="round-1 encoder_12numeric task directory")
    ap.add_argument("--limit", type=int, default=12,
                    help="number of layers to evaluate (default 12)")
    ap.add_argument("--samples", default="all",
                    help="all|comma list of real-sample indices for the real group")
    args = ap.parse_args()

    t0 = time.time()
    frozen_dir = Path(args.frozen_dir)
    calib_json = frozen_dir / "frozen_12layer_calib.json"
    int_npz = frozen_dir / "frozen_12layer_int.npz"
    out_json = TASK / "results.json"
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    log("TASK r2_full_numeric: frozen round-1 12-layer pack, validation-only")
    log(f"ASSET frozen_dir={frozen_dir}")
    if not calib_json.exists() or not int_npz.exists():
        log("CONFLICT missing frozen asset(s)")
        (TASK / "logs").mkdir(exist_ok=True)
        (TASK / "logs" / "run_full_numeric.log").write_text("\n".join(log_lines) + "\n")
        return 2
    calib, cal_json = load_frozen(calib_json, int_npz)
    log(f"ASSET calib={calib_json.name} sha256={sha256(calib_json)}")
    log(f"ASSET int_pack={int_npz.name} sha256={sha256(int_npz)}")

    problems = contract_check(calib, cal_json)
    if problems:
        log("CONFLICT frozen-repack contract violation:")
        for p in problems:
            log(f"  - {p}")
        out_json.write_text(json.dumps({
            "task": "r2_full_numeric", "state": "CONFIG_CONFLICT",
            "problems": problems,
        }, indent=2) + "\n")
        (TASK / "logs").mkdir(exist_ok=True)
        (TASK / "logs" / "run_full_numeric.log").write_text("\n".join(log_lines) + "\n")
        return 2
    log("CONTRACT frozen pack internal scales/M_A consistent (12 layers)")

    weights_path = C.WORKTREE / "software/weights/kwt1_fp32.npz"
    if not weights_path.exists():
        weights_path = C.MAIN / "software/weights/kwt1_fp32.npz"
    weights = golden.load_weights(weights_path)
    split_path = C.find_split_npz()
    real_dir = C.find_real_dir()
    features, partitions, paths, words, tr, va = C.load_split(split_path)
    log(f"DATA split={split_path} sha256={sha256(split_path)}")
    log(f"DATA weights={weights_path} sha256={sha256(weights_path)}")
    log(f"DATA training={len(tr)} (NOT used here) validation={len(va)}")
    log(f"DATA real_dir={real_dir}")

    X0_val = [C.mfcc_to_encoder(features[i], weights) for i in va]
    real_files = sorted(real_dir.glob("sample_*_encoder.npy"))
    if args.samples == "all":
        real_idx = list(range(len(real_files)))
    else:
        real_idx = [int(x) for x in args.samples.split(",")]
    real = [(real_files[i].name, np.load(real_files[i]).astype(np.float32))
            for i in real_idx]
    log(f"DATA validation dataset={len(X0_val)} real={len(real)} total={len(X0_val)+len(real)}")

    nlayers = min(args.limit, len(calib["layers"]))
    groups = [("dataset_val", [paths[i] for i in va], X0_val),
              ("real_val", [n for n, _ in real], [x for _, x in real])]

    # accumulators: pooled sums and per-sample maxima
    pooled = {g: {s: np.zeros(nlayers) for s in STAGES} for g, _, _ in groups}
    pool_ref = {g: {s: np.zeros(nlayers) for s in STAGES} for g, _, _ in groups}
    per_sample_max = {g: {s: np.zeros(nlayers) for s in STAGES} for g, _, _ in groups}
    sat_tot = {"r1": 0, "r2": 0, "y": 0}
    finite = True
    n_eval = 0
    for group, names, X0s in groups:
        for name, x0 in zip(names, X0s):
            arrs = []
            golden.forward(x0, weights, 12, arrays=arrs)
            res = C.integer_chain(calib, x0, want_stages=False)
            for k in range(nlayers):
                for st in STAGES:
                    got = np.asarray(res["deq"][st][k], dtype=np.float64)
                    ref = np.asarray(arrs[k][GOLDEN_KEY[st]], dtype=np.float64)
                    if got.shape != ref.shape or not np.isfinite(got).all():
                        finite = False
                        continue
                    err = got - ref
                    pooled[group][st][k] += float(np.sum(err * err))
                    pool_ref[group][st][k] += float(np.sum(ref * ref))
                    e = C.nrmse(got, ref)
                    per_sample_max[group][st][k] = max(per_sample_max[group][st][k], e)
            for key in sat_tot:
                sat_tot[key] += int(res["sat"][key])
            n_eval += 1
        log(f"EVAL {group}: {len(X0s)} samples x {nlayers} layers done")

    def pooled_nrmse(g, st, k):
        return math.sqrt(pooled[g][st][k] / max(pool_ref[g][st][k], 1e-16))

    # official-like metric: pool all local samples (dataset + real) per layer
    combined_pooled = {}
    for st in STAGES:
        combined_pooled[st] = [
            math.sqrt((pooled["dataset_val"][st][k] + pooled["real_val"][st][k]) /
                      max(pool_ref["dataset_val"][st][k] +
                          pool_ref["real_val"][st][k], 1e-16))
            for k in range(nlayers)]
    dataset_pooled = {st: [pooled_nrmse("dataset_val", st, k)
                           for k in range(nlayers)] for st in STAGES}
    real_pooled = {st: [pooled_nrmse("real_val", st, k)
                        for k in range(nlayers)] for st in STAGES}

    # Conservative tail metric: per-sample maximum, kept separate from the
    # official-like pooled metric above.
    tail_combined = {st: np.maximum(per_sample_max["dataset_val"][st],
                                    per_sample_max["real_val"][st])
                     for st in STAGES}
    worst_layer = int(np.argmax(combined_pooled["y"]))
    worst_y = float(combined_pooled["y"][worst_layer])
    tail_layer = int(np.argmax(tail_combined["y"]))
    tail_y = float(tail_combined["y"][tail_layer])
    below = all(v < NRMSE_LIMIT for v in combined_pooled["y"])
    tail_below = all(v < NRMSE_LIMIT for v in tail_combined["y"])
    if not finite:
        state = "NONFINITE"
    elif not below:
        state = "NUMERIC_GATE_FAIL"
    elif not tail_below:
        state = "PASS_POOLED_TAIL_ABOVE_LIMIT"
    else:
        state = "PASS"

    log("RESULT per-layer pooled y NRMSE (dataset / real / combined):")
    for k in range(nlayers):
        log(f"  layer {k:2d}: {dataset_pooled['y'][k]*100:8.4f}% / "
            f"{real_pooled['y'][k]*100:8.4f}% / {combined_pooled['y'][k]*100:8.4f}%")
    log("RESULT per-layer per-sample-max y NRMSE (dataset / real / combined):")
    for k in range(nlayers):
        log(f"  layer {k:2d}: {per_sample_max['dataset_val']['y'][k]*100:8.4f}% / "
            f"{per_sample_max['real_val']['y'][k]*100:8.4f}% / "
            f"{tail_combined['y'][k]*100:8.4f}%")
    log(f"RESULT worst combined pooled layer={worst_layer} "
        f"y_nrmse={worst_y*100:.4f}% limit={NRMSE_LIMIT*100:.1f}%")
    log(f"RESULT worst per-sample-max layer={tail_layer} "
        f"y_nrmse={tail_y*100:.4f}% limit={NRMSE_LIMIT*100:.1f}% "
        f"tail_below_limit={tail_below} state={state}")
    log(f"RESULT samples={n_eval} dataset={len(X0_val)} real={len(real)} "
        f"layers={nlayers} finite={finite}")
    log(f"RESULT saturation_counts={sat_tot}")

    results = {
        "task": "r2_full_numeric",
        "state": state,
        "evidence_levels": {
            "software_reference": "L1: this script, frozen round-1 assets, 246 local validation samples",
            "rtl_bit_exact": "not run by this task (round-1 L2 claim is not re-verified here)",
            "e203": "no",
            "synthesis": "no",
            "official_acceptance": "no",
        },
        "frozen_assets": {
            "calib": str(calib_json), "calib_sha256": sha256(calib_json),
            "int_pack": str(int_npz), "int_pack_sha256": sha256(int_npz),
            "split": str(split_path), "split_sha256": sha256(split_path),
            "weights": str(weights_path), "weights_sha256": sha256(weights_path),
            "calibration_split": cal_json.get("calibration_split"),
            "validation_used_for_calibration": cal_json.get("validation_used_for_calibration"),
        },
        "samples": {
            "dataset_validation": len(X0_val),
            "real_validation": len(real),
            "evaluated": n_eval,
            "training_used": 0,
        },
        "nrmse_limit": NRMSE_LIMIT,
        "per_layer_y_nrmse_dataset": dataset_pooled["y"],
        "per_layer_y_nrmse_real": real_pooled["y"],
        "per_layer_y_nrmse_combined": combined_pooled["y"],
        "per_layer_stage_nrmse_combined": combined_pooled,
        "per_layer_y_nrmse_dataset_persample_max":
            [float(v) for v in per_sample_max["dataset_val"]["y"]],
        "per_layer_y_nrmse_real_persample_max":
            [float(v) for v in per_sample_max["real_val"]["y"]],
        "per_layer_y_nrmse_combined_persample_max":
            [float(v) for v in tail_combined["y"]],
        "worst_layer": worst_layer,
        "worst_y_nrmse": worst_y,
        "worst_persample_layer": tail_layer,
        "worst_persample_y_nrmse": tail_y,
        "all_layers_below_limit": below,
        "all_layers_below_limit_persample_max": tail_below,
        "finite": finite,
        "saturation": sat_tot,
        "runtime_seconds": time.time() - t0,
    }
    out_json.write_text(json.dumps(results, indent=2) + "\n")
    (TASK / "logs").mkdir(exist_ok=True)
    (TASK / "logs" / "run_full_numeric.log").write_text("\n".join(log_lines) + "\n")
    return 0 if (below and finite) else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # keep an explicit runtime-failure exit code
        print(f"RUNTIME_FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
