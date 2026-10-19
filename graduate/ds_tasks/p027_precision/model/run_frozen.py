#!/usr/bin/env python3
"""p027_precision -- independent frozen-config rerun with NO calibration.

Reads only:

  * ``frozen/MANIFEST.json`` + ``frozen/frozen_config.json`` +
    ``frozen/frozen_calib.npz`` (SHA-256 verified before use),
  * the real checkpoint (to build the FP32 golden reference),
  * the 240-channel validation split (evaluation only).

It never calls ``build_calibration`` / ``fp32_stage_maxima`` and never reads the
training split, so the rerun cannot fit anything.  It reports per-layer pooled
NRMSE, integer-output SHA-256 hashes, saturation counts and exit status.

Usage:
    python3 model/run_frozen.py --pack frozen --out frozen_results.json
Exit codes:
    0  rerun matches the frozen measurement within 1e-12 AND target met
    3  target not met (honest failure propagation)
    2  frozen-pack contract / hash mismatch
    1  runtime failure
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
WORKTREE = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(WORKTREE / "graduate/encoder/reference"))
import prec_ref as P                                   # noqa: E402
import kwt1_fp32_golden as golden                      # noqa: E402
import run_search as S                                 # noqa: E402
import freeze_config as F                              # noqa: E402

TARGET = 0.0027
STAGES = P.STAGES
GOLDEN_KEY = {"a": "attn_projected", "r1": "residual1", "g": "gelu",
              "f": "ffn_out", "r2": "residual2", "y": "layer_output"}
CALIBRATION_CALLS = 0          # by construction: no calibration in this script


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=str(TASK / "frozen"))
    ap.add_argument("--out", default=str(TASK / "frozen_results.json"))
    args = ap.parse_args()
    t0 = time.time()
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    packdir = Path(args.pack)
    try:
        calib, manifest = F.load_pack(packdir)
    except Exception as exc:
        print(f"PACK_FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    cfg = P.cfg_norm(calib["cfg"])
    log(f"PACK config_sha256={manifest['config_sha256']} "
        f"calib_sha256={manifest['calib_sha256']}")
    log(f"PACK layer_count={len(calib['layers'])} "
        f"stage_bits={cfg['stage_bits']} w=({cfg['w_attn']},{cfg['w1']},{cfg['w2']})")
    log("CALIBRATION_CALLS=0 (frozen pack loaded; training split not read)")

    data = S.load_data(want_train=False)
    weights = data["weights"]
    log(f"DATA split={data['split_path']} sha256={S.sha256(data['split_path'])} "
        f"validation={data['n_va']} training_rows_loaded={len(data['X0_tr'])}")
    X0 = data["X0_va"]
    gold = S.golden_val_cache(weights, X0)

    nlayers = len(calib["layers"])
    pool = {st: np.zeros(nlayers) for st in STAGES}
    ref = {st: np.zeros(nlayers) for st in STAGES}
    psmax = {st: np.zeros(nlayers) for st in STAGES}
    sat = {"a": 0, "r1": 0, "g": 0, "r2": 0, "y": 0}
    acc = {}
    int_hash = {k: hashlib.sha256() for k in range(nlayers)}
    finite = True
    for x0, gcache in zip(X0, gold):
        res = P.integer_chain(calib, x0, want_stages=True)
        for k in range(nlayers):
            int_hash[k].update(np.ascontiguousarray(res["int"]["y"][k]).tobytes())
            for st in STAGES:
                got = np.asarray(res["deq"][st][k], dtype=np.float64)
                refv = gcache[k][st]
                if got.shape != refv.shape or not np.isfinite(got).all():
                    finite = False
                    continue
                err = got - refv
                pool[st][k] += float(np.sum(err * err))
                ref[st][k] += float(np.sum(refv * refv))
                psmax[st][k] = max(psmax[st][k], P.nrmse(got, refv))
        for key in sat:
            sat[key] += int(res["sat"][key])
        for key, v in res["acc_bits"].items():
            acc[key] = max(acc.get(key, 0), int(v))
    pooled = {st: [math.sqrt(pool[st][k] / max(ref[st][k], 1e-16))
                   for k in range(nlayers)] for st in STAGES}
    worst = int(np.argmax(pooled["y"]))
    worst_v = float(pooled["y"][worst])
    log("RESULT per-layer pooled y NRMSE (240 validation samples, %):")
    for k in range(nlayers):
        log(f"  layer {k:2d}: {pooled['y'][k]*100:8.4f}")
    log(f"RESULT worst layer={worst} y={worst_v*100:.4f}% "
        f"target={TARGET*100:.2f}% met={worst_v <= TARGET} finite={finite}")
    log(f"RESULT saturation={sat} accumulator_bits={acc}")
    log("RESULT integer y SHA-256 per layer:")
    for k in range(nlayers):
        log(f"  layer {k:2d}: {int_hash[k].hexdigest()}")

    # comparison against the search-time measurement (must be identical)
    cmp_ok, cmp_detail = None, None
    try:
        prev = json.loads((TASK / "results.json").read_text())
        cand = next(r for r in prev["candidates"] if r["name"] == calib.get("name"))
        y_prev = cand["per_layer_pooled_nrmse"]["y"]
        deltas = [abs(a - b) for a, b in zip(pooled["y"], y_prev)]
        cmp_ok = max(deltas) <= 1e-12
        cmp_detail = dict(max_abs_delta=float(max(deltas)),
                          search_worst_y_nrmse=float(max(y_prev)),
                          search_config_sha256=cand.get("config_sha256"))
        log(f"COMPARE search worst={max(y_prev)*100:.6f}% "
            f"frozen worst={worst_v*100:.6f}% max|delta|={max(deltas):.3e} "
            f"identical={cmp_ok}")
    except Exception as exc:
        log(f"COMPARE unavailable: {type(exc).__name__}: {exc}")

    out = dict(
        task="p027_precision_frozen",
        state=("PASS" if (worst_v <= TARGET and finite and cmp_ok is not False)
               else "TARGET_FAIL"),
        name=calib.get("name"),
        config_sha256=manifest["config_sha256"],
        config_canonical_sha256=manifest["config_canonical_sha256"],
        calib_sha256=manifest["calib_sha256"],
        calibration_calls=CALIBRATION_CALLS,
        training_split_read_for_calibration=False,
        samples=len(X0), layers=nlayers,
        split_sha256=S.sha256(data["split_path"]),
        weights_sha256=S.sha256(data["weights_path"]),
        per_layer_pooled_y_nrmse=pooled["y"],
        per_layer_pooled_y_nrmse_pct=[round(v * 100, 6) for v in pooled["y"]],
        per_layer_pooled_stage_nrmse=pooled,
        per_layer_persample_max_y_pct=[round(v * 100, 6) for v in psmax["y"]],
        worst_layer=worst, worst_y_nrmse=worst_v,
        worst_y_nrmse_pct=round(worst_v * 100, 6),
        target=TARGET, target_met=bool(worst_v <= TARGET),
        finite=finite, saturation=sat, measured_accumulator_bits=acc,
        integer_y_sha256_per_layer=[int_hash[k].hexdigest() for k in range(nlayers)],
        comparison=cmp_detail, comparison_identical=cmp_ok,
        evidence_levels=dict(software_reference="frozen integer pack, no calibration",
                             rtl_bit_exact="no", e203="no", synthesis="no",
                             official_acceptance="no"),
        runtime_seconds=round(time.time() - t0, 1),
    )
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    (TASK / "logs").mkdir(exist_ok=True)
    (TASK / "logs" / "run_frozen.log").write_text("\n".join(log_lines) + "\n")
    return 0 if out["state"] == "PASS" else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(f"RUNTIME_FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
