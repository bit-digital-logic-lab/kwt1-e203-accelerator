#!/usr/bin/env python3
"""r2_full_numeric -- 12-layer, multi-sample numeric check on frozen round-1 assets.

This runner deliberately does NOT recalibrate anything.  It consumes the
round-1 frozen pack produced by ``encoder_12numeric``:

    graduate/ds_tasks/encoder_12numeric/frozen_12layer_calib.json
    graduate/ds_tasks/encoder_12numeric/frozen_12layer_int.npz

Partitions (fixed by the real manifest, not by this script)
-----------------------------------------------------------
The local npz split has 240 ``training`` + 240 ``validation`` samples.  The
6 real-WAV encoder inputs in ``software/validation/real_input_chain_20261017``
are **not** all validation: ``manifest.json`` marks 3 as ``validation`` and 3
as ``training``.  This script therefore:

  * computes the **formal local metric only on the 240 npz ``validation``
    samples**, pooled per layer;
  * reports the 3 real ``validation`` samples separately (they are the same
    archive members as 3 of the 240 npz validation rows, so they add **0**
    unique samples -- see ``partitions.dedup``);
  * reports the 3 real ``training`` samples only as a non-validation
    reference, never as validation;
  * never touches the npz ``training`` rows for fitting.

Metrics
-------
* Formal local metric: per-layer pooled NRMSE over the 240 npz validation
  samples:
      sqrt(sum((int_dequant - fp32_ref)^2) / sum(fp32_ref^2))
  with denominator floor 1e-8.  The competition's official gate is the same
  per-layer pooled form over its own 600-sample evaluation set; it is **not**
  a per-sample gate.  This local 240-sample number is not the official score.
* Tail diagnostic: per-sample maximum NRMSE, kept for transparency.  It is
  reported but does not gate the run and does not change the exit code.

Evidence levels emitted by this script: L1 (software reference only).
L2/RTL, L3/E203, L4/synthesis, L5/official acceptance are not produced here.

Usage:
    OPENBLAS_NUM_THREADS=4 python3 model/run_full_numeric.py
    ... --frozen-dir /path/to/encoder_12numeric --limit 12
Exit codes:
    0  formal gate pass: 12/12 layers pooled NRMSE < 3% on the 240 npz
       validation samples (tail diagnostic may still exceed 3%)
    2  frozen-asset / partition contract violation
    3  formal numeric gate failed (>= 3% on at least one layer) or non-finite
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


def norm_path(p: str) -> str:
    """Normalise the npz path form ('./four/...wav') to manifest form."""
    return p[2:] if p.startswith("./") else p


def array_sha256(arr) -> str:
    """SHA-256 of the C-contiguous array bytes (the manifest's convention)."""
    a = np.ascontiguousarray(arr)
    return hashlib.sha256(a.tobytes()).hexdigest()


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

    def write_log():
        (TASK / "logs").mkdir(exist_ok=True)
        (TASK / "logs" / "run_full_numeric.log").write_text(
            "\n".join(log_lines) + "\n")

    log("TASK r2_full_numeric: frozen round-1 12-layer pack, npz-validation formal metric")
    log(f"ASSET frozen_dir={frozen_dir}")
    if not calib_json.exists() or not int_npz.exists():
        log("CONFLICT missing frozen asset(s)")
        write_log()
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
        write_log()
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
    log(f"DATA npz partitions: training={len(tr)} validation={len(va)} "
        f"(npz training rows are NOT used here)")

    # ---- partition evidence: real manifest split vs npz partition ----------
    if len(va) != 240:
        problems.append(f"npz validation count={len(va)} != expected 240")
    real_manifest = json.loads((real_dir / "manifest.json").read_text())
    npz_norm = np.array([norm_path(str(p)) for p in paths])
    real_rows = []
    for s in real_manifest["samples"]:
        member = s["archive_member"]
        hits = np.flatnonzero(npz_norm == member)
        if len(hits) != 1:
            problems.append(
                f"real sample {s['encoder_file']}: archive_member={member!r} "
                f"has {len(hits)} npz matches (expected 1)")
            npz_idx = None
            npz_part = None
            npz_raw = None
        else:
            npz_idx = int(hits[0])
            npz_part = str(partitions[npz_idx])
            npz_raw = str(paths[npz_idx])
            if npz_part != s["split"]:
                problems.append(
                    f"real sample {s['encoder_file']}: manifest split="
                    f"{s['split']} but npz partition={npz_part} "
                    f"(npz index {npz_idx})")
        enc_file = real_dir / s["encoder_file"]
        enc_arr = np.load(enc_file)
        enc_sha = array_sha256(enc_arr)
        if enc_sha != s.get("encoder_sha256"):
            problems.append(
                f"real sample {s['encoder_file']}: array sha256={enc_sha} != "
                f"manifest encoder_sha256={s.get('encoder_sha256')}")
        mfcc_sha = array_sha256(np.load(real_dir / s["mfcc_file"]))
        if mfcc_sha != s.get("mfcc_sha256"):
            problems.append(
                f"real sample {s['mfcc_file']}: array sha256={mfcc_sha} != "
                f"manifest mfcc_sha256={s.get('mfcc_sha256')}")
        real_rows.append(dict(
            file=s["encoder_file"], path=str(enc_file),
            manifest_split=s["split"], label=s.get("label"),
            archive_member=member, encoder_sha256=enc_sha,
            mfcc_sha256=mfcc_sha,
            npz_index=npz_idx, npz_partition=npz_part,
            npz_path_raw=npz_raw, npz_path=member,
            duplicate_of_npz=(npz_idx is not None
                              and s["split"] == "validation"
                              and npz_part == "validation"),
            counted_in_formal_metric=False,
        ))

    # encoder-equivalence check: same archive member -> same encoder input,
    # up to the documented MFCC re-extraction error.
    for row in real_rows:
        if row["npz_index"] is None:
            row["encoder_max_abs_diff_vs_npz"] = None
            continue
        ref_x = C.mfcc_to_encoder(features[row["npz_index"]], weights)
        got_x = np.load(row["path"]).astype(np.float32)
        row["encoder_max_abs_diff_vs_npz"] = float(
            np.max(np.abs(got_x.astype(np.float64) - ref_x.astype(np.float64))))
        row["encoder_shape"] = list(got_x.shape)

    real_val = [r for r in real_rows if r["manifest_split"] == "validation"]
    real_train = [r for r in real_rows if r["manifest_split"] == "training"]
    dup_val = [r for r in real_val if r["duplicate_of_npz"]]
    if len(real_val) != 3 or len(real_train) != 3:
        problems.append(
            f"real manifest split counts validation={len(real_val)} "
            f"training={len(real_train)} != expected 3/3")
    if len(dup_val) != len(real_val):
        problems.append(
            f"only {len(dup_val)}/{len(real_val)} real validation samples "
            f"deduplicated against npz validation")
    if any(r["npz_partition"] == "validation" for r in real_train):
        problems.append("a real training sample is marked validation in npz")

    if problems:
        log("CONFLICT partition contract violation:")
        for p in problems:
            log(f"  - {p}")
        out_json.write_text(json.dumps({
            "task": "r2_full_numeric", "state": "PARTITION_CONFLICT",
            "problems": problems,
            "real_manifest": real_rows,
        }, indent=2) + "\n")
        write_log()
        return 2

    log("DATA real manifest (path/partition evidence):")
    for r in real_rows:
        diff = r["encoder_max_abs_diff_vs_npz"]
        diff_txt = "n/a" if diff is None else f"{diff:.3e}"
        log(f"  {r['file']}: manifest_split={r['manifest_split']} "
            f"archive_member={r['archive_member']} "
            f"npz_index={r['npz_index']} npz_partition={r['npz_partition']} "
            f"npz_path={r['npz_path_raw']!r} "
            f"duplicate_of_npz={r['duplicate_of_npz']} "
            f"counted_in_formal_metric={r['counted_in_formal_metric']} "
            f"encoder_max_abs_diff_vs_npz={diff_txt}")

    formal_names = [norm_path(str(paths[i])) for i in va]
    formal_x0 = [C.mfcc_to_encoder(features[i], weights) for i in va]
    log(f"DATA formal metric scope = npz validation only: {len(formal_x0)} samples")
    log(f"DATA real validation reported separately: {len(real_val)} samples "
        f"({len(dup_val)} duplicate of npz validation -> +0 unique)")
    log(f"DATA real training reference only (NOT validation): {len(real_train)} samples")

    nlayers = min(args.limit, len(calib["layers"]))

    def new_acc():
        return {s: np.zeros(nlayers) for s in STAGES}

    # formal (240 npz validation) accumulators
    formal_pool = new_acc()
    formal_ref = new_acc()
    formal_psmax = new_acc()
    # separate real groups, never pooled into the formal metric
    real_val_pool = new_acc()
    real_val_ref = new_acc()
    real_val_psmax = new_acc()
    real_train_pool = new_acc()
    real_train_ref = new_acc()
    real_train_psmax = new_acc()

    def new_stats():
        return {"evaluated": 0, "sat": {"r1": 0, "r2": 0, "y": 0}}

    formal_stats = new_stats()
    real_val_stats = new_stats()
    real_train_stats = new_stats()
    finite = True

    def evaluate(x0, pool, ref, psmax, stats):
        nonlocal finite
        arrs = []
        golden.forward(x0, weights, 12, arrays=arrs)
        res = C.integer_chain(calib, x0, want_stages=False)
        per_layer = {}
        for k in range(nlayers):
            per_layer[k] = {}
            for st in STAGES:
                got = np.asarray(res["deq"][st][k], dtype=np.float64)
                refv = np.asarray(arrs[k][GOLDEN_KEY[st]], dtype=np.float64)
                if got.shape != refv.shape or not np.isfinite(got).all():
                    finite = False
                    continue
                err = got - refv
                pool[st][k] += float(np.sum(err * err))
                ref[st][k] += float(np.sum(refv * refv))
                e = C.nrmse(got, refv)
                psmax[st][k] = max(psmax[st][k], e)
                per_layer[k][st] = e
        for key in stats["sat"]:
            stats["sat"][key] += int(res["sat"][key])
        stats["evaluated"] += 1
        return per_layer

    for name, x0 in zip(formal_names, formal_x0):
        evaluate(x0, formal_pool, formal_ref, formal_psmax, formal_stats)
    log(f"EVAL formal npz validation: {len(formal_x0)} samples x {nlayers} layers done")

    for r in real_val:
        per_layer = evaluate(np.load(r["path"]).astype(np.float32),
                             real_val_pool, real_val_ref, real_val_psmax,
                             real_val_stats)
        r["y_nrmse_per_layer"] = [per_layer[k]["y"] for k in range(nlayers)]
    log(f"EVAL real validation (separate): {len(real_val)} samples done")

    for r in real_train:
        per_layer = evaluate(np.load(r["path"]).astype(np.float32),
                             real_train_pool, real_train_ref, real_train_psmax,
                             real_train_stats)
        r["y_nrmse_per_layer"] = [per_layer[k]["y"] for k in range(nlayers)]
    log(f"EVAL real training reference (not validation): {len(real_train)} samples done")

    def pooled_nrmse(pool, ref, st, k):
        return math.sqrt(pool[st][k] / max(ref[st][k], 1e-16))

    formal_pooled = {st: [pooled_nrmse(formal_pool, formal_ref, st, k)
                          for k in range(nlayers)] for st in STAGES}
    real_val_pooled = {st: [pooled_nrmse(real_val_pool, real_val_ref, st, k)
                            for k in range(nlayers)] for st in STAGES}
    real_train_pooled = {st: [pooled_nrmse(real_train_pool, real_train_ref, st, k)
                              for k in range(nlayers)] for st in STAGES}

    worst_layer = int(np.argmax(formal_pooled["y"]))
    worst_y = float(formal_pooled["y"][worst_layer])
    tail_layer = int(np.argmax(formal_psmax["y"]))
    tail_y = float(formal_psmax["y"][tail_layer])
    below = all(v < NRMSE_LIMIT for v in formal_pooled["y"])
    tail_below = all(v < NRMSE_LIMIT for v in formal_psmax["y"])
    if not finite:
        state = "NONFINITE"
    elif not below:
        state = "NUMERIC_GATE_FAIL"
    elif not tail_below:
        state = "PASS_FORMAL_POOLED_TAIL_ABOVE_LIMIT"
    else:
        state = "PASS"

    log("RESULT formal per-layer pooled y NRMSE (240 npz validation only):")
    for k in range(nlayers):
        log(f"  layer {k:2d}: {formal_pooled['y'][k]*100:8.4f}%")
    log("RESULT real validation y NRMSE (3 samples, separate, NOT added to 240):")
    for k in range(nlayers):
        log(f"  layer {k:2d}: pooled {real_val_pooled['y'][k]*100:8.4f}% "
            f"per-sample-max {real_val_psmax['y'][k]*100:8.4f}%")
    log("RESULT real training reference y NRMSE (3 samples, NOT validation):")
    for k in range(nlayers):
        log(f"  layer {k:2d}: pooled {real_train_pooled['y'][k]*100:8.4f}% "
            f"per-sample-max {real_train_psmax['y'][k]*100:8.4f}%")
    log("RESULT tail diagnostic on formal 240 (per-sample max, NOT the gate):")
    for k in range(nlayers):
        log(f"  layer {k:2d}: {formal_psmax['y'][k]*100:8.4f}%")
    log(f"RESULT worst formal pooled layer={worst_layer} "
        f"y_nrmse={worst_y*100:.4f}% limit={NRMSE_LIMIT*100:.1f}%")
    log(f"RESULT worst per-sample tail layer={tail_layer} "
        f"y_nrmse={tail_y*100:.4f}% (diagnostic only, tail_below_limit={tail_below})")
    log(f"RESULT formal_samples=240 real_validation={len(real_val)} "
        f"real_training_reference={len(real_train)} "
        f"unique_validation_total=240 evaluated_units={formal_stats['evaluated']}"
        f"+{real_val_stats['evaluated']}+{real_train_stats['evaluated']} "
        f"layers={nlayers} finite={finite}")
    log(f"RESULT saturation_counts formal={formal_stats['sat']} "
        f"real_validation={real_val_stats['sat']} "
        f"real_training_reference={real_train_stats['sat']} state={state}")
    log("RESULT gate semantics: official-style gate is per-layer pooled over the "
        "evaluation set (600 samples officially); per-sample values are not a gate")

    results = {
        "task": "r2_full_numeric",
        "state": state,
        "evidence_levels": {
            "software_reference": (
                "L1: this script, frozen round-1 assets, formal metric on 240 "
                "npz validation samples; 3 real validation samples reported "
                "separately (deduplicated), 3 real training samples not validation"),
            "rtl_bit_exact": "not run by this task (round-1 L2 claim is not re-verified here)",
            "e203": "no",
            "synthesis": "no",
            "official_acceptance": "no",
        },
        "formal_metric": {
            "scope": "npz validation partition only (manifest.split == 'validation')",
            "samples": len(formal_x0),
            "pooling": "per-layer pooled NRMSE over all 240 samples",
            "nrmse_formula": "sqrt(sum((int_dequant - fp32_ref)^2) / sum(fp32_ref^2)), denominator floor 1e-8",
            "gate": "every layer < 3% pooled; per-sample values are diagnostic only",
            "official_analogue": "official gate pools its own 600-sample evaluation set per layer; this is not that set",
        },
        "frozen_assets": {
            "calib": str(calib_json), "calib_sha256": sha256(calib_json),
            "int_pack": str(int_npz), "int_pack_sha256": sha256(int_npz),
            "split": str(split_path), "split_sha256": sha256(split_path),
            "weights": str(weights_path), "weights_sha256": sha256(weights_path),
            "calibration_split": cal_json.get("calibration_split"),
            "validation_used_for_calibration": cal_json.get("validation_used_for_calibration"),
        },
        "partitions": {
            "npz": {
                "training": int(len(tr)),
                "validation": int(len(va)),
                "path_form": "npz paths are './<label>/<file>.wav'; manifest uses '<label>/<file>.wav'",
                "real_training_used_for_fitting": 0,
            },
            "formal_validation": {
                "source": "npz partitions == 'validation'",
                "count": len(formal_x0),
            },
            "real_wav": {
                "dir": str(real_dir),
                "manifest": str(real_dir / "manifest.json"),
                "manifest_sha256": sha256(real_dir / "manifest.json"),
                "validation": len(real_val),
                "training": len(real_train),
                "validation_files": [r["file"] for r in real_val],
                "training_files": [r["file"] for r in real_train],
            },
            "dedup": {
                "key": "archive_member (real manifest) == npz path without leading './'",
                "overlap_count": len(dup_val),
                "overlap_files": [r["file"] for r in dup_val],
                "real_validation_counted_additionally": 0,
                "unique_validation_total": len(formal_x0),
                "combined_count_used": len(formal_x0),
                "note": (
                    "the 3 real validation WAVs are the same archive members as "
                    "3 rows already inside the 240 npz validation rows, so they "
                    "are reported separately and add 0 unique samples; the 3 real "
                    "training WAVs are never counted as validation"),
            },
        },
        "samples": {
            "formal_validation": len(formal_x0),
            "real_validation_reported_separately": len(real_val),
            "real_training_reference_only": len(real_train),
            "counted_in_formal_metric": len(formal_x0),
            "evaluated_units": {
                "formal_validation": formal_stats["evaluated"],
                "real_validation": real_val_stats["evaluated"],
                "real_training_reference": real_train_stats["evaluated"],
            },
            "training_used_in_formal_metric": 0,
        },
        "real_samples": real_rows,
        "nrmse_limit": NRMSE_LIMIT,
        "per_layer_y_nrmse_formal_validation": formal_pooled["y"],
        "per_layer_stage_nrmse_formal_validation": formal_pooled,
        "per_layer_y_nrmse_formal_persample_max": [float(v) for v in formal_psmax["y"]],
        "per_layer_y_nrmse_real_validation": real_val_pooled["y"],
        "per_layer_stage_nrmse_real_validation": real_val_pooled,
        "per_layer_y_nrmse_real_validation_persample_max":
            [float(v) for v in real_val_psmax["y"]],
        "per_layer_y_nrmse_real_training_reference": real_train_pooled["y"],
        "per_layer_y_nrmse_real_training_persample_max":
            [float(v) for v in real_train_psmax["y"]],
        "worst_layer": worst_layer,
        "worst_y_nrmse": worst_y,
        "worst_persample_layer": tail_layer,
        "worst_persample_y_nrmse": tail_y,
        "all_layers_below_limit": below,
        "all_layers_below_limit_persample_max": tail_below,
        "persample_max_is_gate": False,
        "finite": finite,
        "saturation": {
            "formal_validation": formal_stats["sat"],
            "real_validation": real_val_stats["sat"],
            "real_training_reference": real_train_stats["sat"],
        },
        "runtime_seconds": time.time() - t0,
    }
    out_json.write_text(json.dumps(results, indent=2) + "\n")
    write_log()
    return 0 if (below and finite) else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # keep an explicit runtime-failure exit code
        print(f"RUNTIME_FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
