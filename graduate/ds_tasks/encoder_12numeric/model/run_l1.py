#!/usr/bin/env python3
"""encoder_12numeric  --  L1 independent 12-layer numeric-chain evaluation.

Runs the frozen 12-layer integer encoder chain (12 DISTINCT weight sets and 12
DISTINCT scale sets) over the independent VALIDATION split and reports the
per-layer NRMSE against the FP32 golden.  Calibration uses the TRAINING split
only; validation samples are never used to choose a scale, shift, LUT or
threshold.

Evidence levels produced by THIS script: L1 (software reference).  RTL bit
exactness is produced by ``tb/run_rtl_chain.sh`` + ``chain_verify.py``;
E203/synthesis/official acceptance are NOT produced here.

Usage:
    OPENBLAS_NUM_THREADS=4 python3 model/run_l1.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TASK = HERE.parent
sys.path.insert(0, str(HERE))
import chain as C                                 # noqa: E402
import ref12 as R                                 # noqa: E402
import kwt1_fp32_golden as golden                 # noqa: E402

NRMSE_LIMIT = 0.03
STAGES = ["a", "r1", "g", "f", "r2", "y"]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def locate_weights():
    for cand in (C.WORKTREE / "software/weights/kwt1_fp32.npz",
                 C.MAIN / "software/weights/kwt1_fp32.npz"):
        if cand.exists():
            return cand
    raise FileNotFoundError("kwt1_fp32.npz not found")


def self_test():
    """Vectorised primitives must reproduce the scalar contract exactly."""
    rng = np.random.default_rng(12345)
    ok = True
    # layernorm
    x = rng.integers(-2047, 2048, size=(3, C.D))
    eps_q = 31173
    g = rng.integers(-40000, 40000, size=C.D)
    b = rng.integers(-3000, 3000, size=C.D)
    mo = 410
    vec = C.layernorm_rows(x, eps_q, g, b, mo)
    for i in range(3):
        row, _ = R.layernorm(list(x[i]), eps_q, g, b, mo)
        assert np.array_equal(vec[i], np.array(row)), f"layernorm row {i}"
    # residual
    a = rng.integers(-2047, 2048, size=(4, C.D))
    bb = rng.integers(-2047, 2048, size=(4, C.D))
    mb = rng.integers(-40000, 40000, size=C.D)
    vec = C.residual_rows(a, 61302, bb, mb)
    for i in range(4):
        row, _ = R.residual_add(list(a[i]), 61302, list(bb[i]), list(mb), C.QB)
        assert np.array_equal(vec[i], np.array(row)), f"residual row {i}"
    # gelu
    h = rng.integers(-(1 << 20), 1 << 20, size=(3, C.HID))
    mg = rng.integers(1, 120000, size=C.HID)
    lut = R.lut_from_gelu()
    vec = C.gelu_rows(h, mg, 267644, lut)
    for i in range(3):
        for j in range(C.HID):
            assert vec[i, j] == R.gelu_step(h[i, j], mg[j], 267644, lut), f"gelu {i},{j}"
    # attention vectorised vs scalar
    ap = dict(RQ=6, RK=7, RV=7, RC=14, RO=11, KC=6680, SH_KC=38, F_BITS=6,
              M_LUT=14, G_SH=20, lut=np.array(
                  [int(round((2.0 ** (f / 64)) * (1 << 14))) for f in range(64)], dtype=np.int64),
              wq=rng.integers(-512, 513, size=(C.D, C.D)),
              wk=rng.integers(-512, 513, size=(C.D, C.D)),
              wv=rng.integers(-512, 513, size=(C.D, C.D)),
              wo=rng.integers(-512, 513, size=(C.D, C.D)),
              bias=rng.integers(-1000, 1000, size=C.D))
    X = rng.integers(-2047, 2048, size=(C.T, C.D))
    a1 = C.attention_rows(X, ap)
    a2 = C.attention_scalar(X, ap)
    assert np.array_equal(a1, a2), "attention vectorised != scalar"
    return {"layernorm": "bit-exact", "residual": "bit-exact", "gelu": "bit-exact",
            "attention": "bit-exact", "samples": 1}


def main():
    t0 = time.time()
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    weights_path = locate_weights()
    weights = golden.load_weights(weights_path)
    split_path = C.find_split_npz()
    real_dir = C.find_real_dir()
    features, partitions, paths, words, tr, va = C.load_split(split_path)
    log(f"DATA split={split_path} train={len(tr)} val={len(va)}")
    log(f"DATA weights={weights_path} sha256={sha256(weights_path)[:16]}...")
    log(f"DATA real_dir={real_dir}")

    X0_train = [C.mfcc_to_encoder(features[i], weights) for i in tr]
    X0_val = [C.mfcc_to_encoder(features[i], weights) for i in va]
    real_files = sorted(real_dir.glob("sample_*_encoder.npy"))
    real = [(p.name, np.load(p).astype(np.float32)) for p in real_files]
    log(f"DATA training encoder inputs={len(X0_train)} validation={len(X0_val)} "
        f"real={len(real)}")

    st = self_test()
    log(f"SELFTEST scalar-vs-vector {st}")

    log("CALIB calibrating on the TRAINING split only ...")
    calib = C.build_calibration(weights, X0_train)
    # JSON-safe: strip numpy arrays into lists
    def enc(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, dict):
            return {k: enc(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [enc(v) for v in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return o

    calib_path = TASK / "frozen_12layer_calib.json"
    prov = dict(schema=calib["schema"], calibration_split="training",
                validation_used_for_calibration=False, gradient_free=True,
                weights=str(weights_path), weights_sha256=sha256(weights_path),
                split_npz=str(split_path), split_npz_sha256=sha256(split_path),
                train_count=int(len(tr)), validation_count=int(len(va)),
                real_samples=[n for n, _ in real],
                scale_formula="s = max_training |stage| / (2^(A_BITS-1)-1)",
                max_x0=calib["max_x0"],
                maxima={k: calib["maxima"][k] for k in calib["maxima"]},
                layers=[{
                    "layer": L["layer"], "M_A": L["M_A"], "scales": L["scales"],
                    "attn_params": L["attn"]["params"],
                    "coeff": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                              for k, v in L["coeff"].items()},
                    "ffn1_sw": L["ffn1"]["sw"].tolist(),
                    "ffn2_sw": L["ffn2"]["sw"].tolist(),
                } for L in calib["layers"]],
                lut=[int(v) for v in calib["lut"]],
                attn_lut=[int(v) for v in calib["attn_lut"]])
    calib_path.write_text(json.dumps(prov, indent=2) + "\n")
    calib_sha = sha256(calib_path)
    # frozen integer tensors needed by the RTL vector generator
    int_arrays = {}
    for L in calib["layers"]:
        k = L["layer"]
        int_arrays[f"layer{k+1:02d}_attn_wq"] = np.asarray(L["attn"]["wq"], dtype=np.int16)
        int_arrays[f"layer{k+1:02d}_attn_wk"] = np.asarray(L["attn"]["wk"], dtype=np.int16)
        int_arrays[f"layer{k+1:02d}_attn_wv"] = np.asarray(L["attn"]["wv"], dtype=np.int16)
        int_arrays[f"layer{k+1:02d}_attn_wo"] = np.asarray(L["attn"]["wo"], dtype=np.int16)
        int_arrays[f"layer{k+1:02d}_attn_bias"] = np.asarray(L["attn"]["bias"], dtype=np.int32)
        int_arrays[f"layer{k+1:02d}_w1"] = np.asarray(L["ffn1"]["w"], dtype=np.int16)
        int_arrays[f"layer{k+1:02d}_b1"] = np.asarray(L["ffn1"]["b"], dtype=np.int32)
        int_arrays[f"layer{k+1:02d}_w2"] = np.asarray(L["ffn2"]["w"], dtype=np.int16)
        int_arrays[f"layer{k+1:02d}_b2"] = np.asarray(L["ffn2"]["b"], dtype=np.int32)
    int_npz = TASK / "frozen_12layer_int.npz"
    np.savez_compressed(int_npz, **int_arrays)
    log(f"CALIB frozen integer tensors -> {int_npz.name} sha256={sha256(int_npz)[:16]}...")
    log(f"CALIB frozen -> {calib_path.name} sha256={calib_sha[:16]}...")
    for L in calib["layers"]:
        sc = L["scales"]
        log(f"CALIB layer {L['layer']:2d} M_A={L['M_A']:6d} "
            f"s_x={sc['s_x']:.6e} s_a={sc['s_a']:.6e} sx={sc['sx_ffn1']:.6e} "
            f"s_g={sc['s_g']:.6e} s_r1={sc['s_r1']:.6e} s_r2={sc['s_r2']:.6e} "
            f"s_out={sc['s_out']:.6e}")

    # ---- validation evaluation -------------------------------------------
    envs = [("dataset_val", f"{C.WORKTREE}/software/.build/{split_path.name}",
             [paths[i] for i in va], X0_val)]
    envs.append(("real_val", str(real_dir), [n for n, _ in real], [x for _, x in real]))

    val_records = []
    worst = {"nrmse": 0.0, "layer": 0, "stage": "", "sample": ""}
    total_evals = 0
    layer_y = np.zeros(C.NLAYERS)
    layer_y_real = np.zeros(C.NLAYERS)
    stage_max = {s: 0.0 for s in STAGES}
    for group, src, names, X0s in envs:
        for name, x0 in zip(names, X0s):
            _logits, _x, _stages, arrs = C._golden_arrays(weights, x0)
            res = C.integer_chain(calib, x0, want_stages=False)
            stage_nrmse = {}
            for si in range(C.NLAYERS):
                for st_name in STAGES:
                    e = C.nrmse(res["deq"][st_name][si], arrs[si][{
                        "a": "attn_projected", "r1": "residual1", "g": "gelu",
                        "f": "ffn_out", "r2": "residual2", "y": "layer_output"}[st_name]])
                    stage_nrmse[f"L{si}_{st_name}"] = e
                    stage_max[st_name] = max(stage_max[st_name], e)
                    if e > worst["nrmse"]:
                        worst = {"nrmse": e, "layer": si, "stage": st_name,
                                 "sample": f"{group}:{name}"}
                if group == "real_val":
                    layer_y_real[si] = max(layer_y_real[si], stage_nrmse[f"L{si}_y"])
                else:
                    layer_y[si] = max(layer_y[si], stage_nrmse[f"L{si}_y"])
            val_records.append(dict(group=group, sample=name, stage_nrmse=stage_nrmse))
            total_evals += 1
        log(f"EVAL {group}: {len(X0s)} samples evaluated")

    worst_layer = int(np.argmax(np.maximum(layer_y, layer_y_real)))
    worst_y = float(max(layer_y[worst_layer], layer_y_real[worst_layer]))

    calib_unchanged = sha256(calib_path) == calib_sha
    results = {
        "task": "encoder_12numeric",
        "evidence_levels": {
            "software_reference": "yes (12 distinct layers, distinct scales; "
                                  "train-calibrated, validation-only evaluation)",
            "rtl_bit_exact": "see chain_verify.py (run separately)",
            "e203": "no",
            "synthesis": "no",
            "official_acceptance": "no",
        },
        "calibration": {
            "file": calib_path.name, "sha256": calib_sha,
            "split": "training", "samples": int(len(tr)),
            "gradient_free": True,
        },
        "validation": {
            "dataset_samples": int(len(va)),
            "real_samples": [n for n, _ in real],
            "samples_evaluated": int(total_evals),
            "per_layer_y_nrmse_dataset": layer_y.tolist(),
            "per_layer_y_nrmse_real": layer_y_real.tolist(),
            "worst_layer": worst_layer,
            "worst_y_nrmse": worst_y,
            "worst_stage_overall": worst,
            "stage_max_nrmse": stage_max,
            "nrmse_limit": NRMSE_LIMIT,
            "all_layers_below_limit": bool(worst_y < NRMSE_LIMIT),
        },
        "audit": {
            "calibration_split_is_training": True,
            "validation_used_for_calibration": False,
            "calibration_file_unchanged_by_evaluation": bool(calib_unchanged),
            "layers_have_distinct_scales": bool(
                len({round(L["scales"]["s_x"], 12) for L in calib["layers"]}) == C.NLAYERS),
            "layers_have_distinct_weights": bool(
                len({hashlib.sha256(np.ascontiguousarray(
                    L["ffn1"]["w"], dtype=np.int64).tobytes()).hexdigest()
                    for L in calib["layers"]}) == C.NLAYERS),
        },
        "selftest": st,
        "runtime_seconds": round(time.time() - t0, 2),
    }
    (TASK / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    log("EVAL per-layer y NRMSE (dataset val): " +
        " ".join(f"L{i}={layer_y[i]*100:.4f}%" for i in range(C.NLAYERS)))
    log("EVAL per-layer y NRMSE (real val):    " +
        " ".join(f"L{i}={layer_y_real[i]*100:.4f}%" for i in range(C.NLAYERS)))
    log(f"EVAL worst layer={worst_layer} worst_y={worst_y*100:.4f}% "
        f"(limit {NRMSE_LIMIT*100:.1f}%)")
    log(f"EVAL worst stage overall: {worst}")
    log(f"L1_DONE samples={total_evals} runtime={results['runtime_seconds']}s")
    (TASK / "logs").mkdir(exist_ok=True)
    (TASK / "logs" / "run_l1.log").write_text("\n".join(log_lines) + "\n")
    if not results["audit"]["calibration_file_unchanged_by_evaluation"]:
        raise SystemExit("calibration file changed during evaluation")
    print("L1_PASS" if results["validation"]["all_layers_below_limit"] else "L1_FAIL_LIMIT")


if __name__ == "__main__":
    main()
