#!/usr/bin/env python3
"""r2_full_numeric -- reproduce the round-1 scale-contract conflict.

Round-1 produced two *layer-0* frozen assets that were intended to be composed
into the assembled encoder:

    attn_scale/attn_calib_train.json          (attention input scale)
    postnorm_scale/frozen_postnorm_calib.json (residual / LayerNorm scales)

The assembled RTL quantises X once for attention and then feeds the *same*
integer X into the PostNorm residual, whose coefficients are derived from the
PostNorm input scale.  Therefore the two assets must agree on s_x.  They do not:
attn s_x = 1.961475e-3 (3 training samples) vs postnorm s_x = 6.737881e-3
(240 training samples).  This script:

  1. evaluates the required invariant and stops at the earliest blocking point
     (layer-0 input-scale contract);
  2. nevertheless runs the *naive* first-round composition through attention
     and the first residual on real validation samples, to quantify the damage;
  3. exits 2 (configuration conflict) and writes conflict.json.

This is intentionally not a "fix": no scale is re-fitted, no validation sample
is used to choose a constant.  It only demonstrates that the two round-1
assets cannot be composed as-is.

Usage:
    OPENBLAS_NUM_THREADS=4 python3 model/run_conflict_witness.py
Exit codes:
    0  no conflict (unexpected)
    2  round-1 scale contract conflict reproduced
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
sys.path.insert(0, str(HERE))
import chain as C                                  # noqa: E402
import ref12 as R                                  # noqa: E402
import kwt1_fp32_golden as golden                  # noqa: E402

ATTN = Path("/home/yang/kwt-worktrees/ds-grad-attn_scale/"
            "graduate/ds_tasks/attn_scale/attn_calib_train.json")
POSTNORM = Path("/home/yang/kwt-worktrees/ds-grad-postnorm_scale/"
                "graduate/ds_tasks/postnorm_scale/frozen_postnorm_calib.json")
FROZEN12 = Path("/home/yang/kwt-worktrees/ds-grad-encoder_12numeric/"
                "graduate/ds_tasks/encoder_12numeric/frozen_12layer_calib.json")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rs(x, q):
    x = np.asarray(x, dtype=np.int64)
    if q <= 0:
        return x
    return (x + (1 << (q - 1))) >> q


def attention_first_round(X_int, attn, weights):
    """Run the round-1 attn asset on X_int, using its frozen constants.

    Integer weights are reconstructed from the official FP32 weights with the
    asset's own per-tensor weight scales; nothing is fitted here.
    """
    p = f"{golden.PREFIX}/enc_layers/0/att"
    ws = attn["weight_scales"]
    wq = np.clip(R.rint(weights[f"{p}/query_dense/kernel"] / ws["wq"]),
                 -R.QMAX_W, R.QMAX_W).astype(np.int64)
    wk = np.clip(R.rint(weights[f"{p}/key_dense/kernel"] / ws["wk"]),
                 -R.QMAX_W, R.QMAX_W).astype(np.int64)
    wv = np.clip(R.rint(weights[f"{p}/value_dense/kernel"] / ws["wv"]),
                 -R.QMAX_W, R.QMAX_W).astype(np.int64)
    wo = np.clip(R.rint(weights[f"{p}/combine_heads/kernel"] / ws["wo"]),
                 -R.QMAX_W, R.QMAX_W).astype(np.int64)

    RQ, RK, RV = attn["RQ"], attn["RK"], attn["RV"]
    RC, RO = attn["RC"], attn["RO"]
    KC, SH_KC = attn["KC"], attn["SH_KC"]
    F_BITS, M_LUT, G_SH = attn["f_bits"], attn["m_lut"], attn["g_sh"]

    Qi = np.clip(rs(X_int @ wq, RQ), -32768, 32767)
    Ki = np.clip(rs(X_int @ wk, RK), -32768, 32767)
    Vi = np.clip(rs(X_int @ wv, RV), -32768, 32767)

    lut = np.array([int(round((2.0 ** (fr / (1 << F_BITS))) * (1 << M_LUT)))
                    for fr in range(1 << F_BITS)], dtype=np.int64)
    S = Qi @ Ki.T
    A = rs((S - S.max(axis=1, keepdims=True)) * KC, SH_KC)
    ka = A >> F_BITS
    frac = A - (ka << F_BITS)
    pos = np.clip(-ka, 0, 31)
    E = np.where(-ka >= 32, 0, lut[frac] >> pos)
    sumE = E.sum(axis=1, keepdims=True)
    Rr = ((1 << (15 + G_SH)) + (sumE >> 1)) // sumE
    P = np.clip((E * Rr) >> G_SH, 0, 32767)

    Ci = np.clip(rs(P @ Vi, RC), -32768, 32767)
    Oa = Ci @ wo + np.asarray(attn["bias_int"], dtype=np.int64)[None, :]
    Oi = np.clip(rs(Oa, RO), -32768, 32767)
    return Oi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="0,2,5",
                    help="real validation sample indices used for the witness")
    args = ap.parse_args()
    t0 = time.time()
    lines = []

    def log(m):
        print(m, flush=True)
        lines.append(m)

    attn = json.loads(ATTN.read_text())
    pn = json.loads(POSTNORM.read_text())
    f12 = json.loads(FROZEN12.read_text())

    sx_attn = float(attn["scale_x"])
    sx_pn = float(pn["scales"]["s_x"])
    diff = abs(sx_attn - sx_pn) / sx_pn
    log("TASK r2_full_numeric: round-1 scale-contract witness")
    log(f"ASSET attn_scale  {ATTN} sha256={sha256(ATTN)}")
    log(f"ASSET postnorm    {POSTNORM} sha256={sha256(POSTNORM)}")
    log(f"CHECK attn.scale_x={sx_attn:.12g} postnorm.s_x={sx_pn:.12g} "
        f"relative_diff={diff*100:.4f}%")

    invariants = []
    required_equal = diff <= 1e-12
    invariants.append({
        "name": "layer0_input_scale_equality",
        "required": "attn.scale_x == postnorm.s_x (same integer X enters residual)",
        "attn": sx_attn, "postnorm": sx_pn, "relative_diff": diff,
        "ok": bool(required_equal),
    })
    # The same 240-sample training split is used by both assets; the attn asset
    # is a 3-sample subset.  Check whether the 3-sample scale even covers it.
    # This is a static range check only (no validation sample is involved).
    max_x_train = float(f12.get("max_x0", float("nan")))
    attn_covers = sx_attn * 2047 >= max_x_train
    invariants.append({
        "name": "attn_scale_covers_training_range",
        "required": "scale_x * 2047 >= max_train|X|",
        "attn_scale_x": sx_attn, "max_train_abs_x": max_x_train,
        "ok": bool(attn_covers),
    })
    log(f"CHECK max_train|X|={max_x_train:.9g} attn_scale*2047="
        f"{sx_attn*2047:.9g} covers={attn_covers}")

    # The first blocking invariant is the scale equality.  Report it and do the
    # witness run anyway, but stop at layer-0 r1: a full chain would be built on
    # a contract that is already false.
    failed = [x for x in invariants if not x["ok"]]
    log(f"CONFLICT earliest_block=layer0.x_scale_contract failed_invariants="
        f"{[x['name'] for x in failed]}")

    weights_path = C.WORKTREE / "software/weights/kwt1_fp32.npz"
    if not weights_path.exists():
        weights_path = C.MAIN / "software/weights/kwt1_fp32.npz"
    weights = golden.load_weights(weights_path)
    real_dir = C.find_real_dir()
    files = sorted(real_dir.glob("sample_*_encoder.npy"))
    idx = [int(x) for x in args.samples.split(",")]

    s_a = float(pn["scales"]["s_a"])
    s_r1 = float(pn["scales"]["s_r1"])
    ma1 = int(R.rint(sx_pn / s_r1 * (1 << R.QB)))
    mb1 = int(R.rint(s_a / s_r1 * (1 << R.QB)))
    M_A = int(R.rint(float(attn["scale_o"]) / s_a * (1 << R.Q_A)))
    log(f"WITNESS M_A=round(scale_o/s_a*2^16)={M_A} ma1={ma1} mb1={mb1}")

    rows = []
    for si in idx:
        x0 = np.load(files[si]).astype(np.float32)
        X_int = np.clip(R.rint(x0 / sx_attn), -R.CLAMP_A, R.CLAMP_A).astype(np.int64)
        Oi = attention_first_round(X_int, attn, weights)
        a_int = R.requant_o_to_a(Oi, M_A)
        r1 = C.residual_rows(X_int, ma1, a_int, mb1)
        deq_r1 = r1 * s_r1
        arrs = []
        golden.forward(x0, weights, 12, arrays=arrs)
        ref_r1 = arrs[0]["residual1"]
        err = C.nrmse(deq_r1, ref_r1)
        # causal control: same attention output, but X re-expressed on the
        # PostNorm scale before the residual.  This is a diagnostic only.
        X_fix = np.clip(R.rint(x0 / sx_pn), -R.CLAMP_A, R.CLAMP_A).astype(np.int64)
        r1_fix = C.residual_rows(X_fix, ma1, a_int, mb1)
        err_fix = C.nrmse(r1_fix * s_r1, ref_r1)
        sat = int(np.sum(np.abs(r1) == R.CLAMP_A))
        rows.append({
            "sample": files[si].name,
            "naive_r1_nrmse": err,
            "naive_r1_saturated_elements": sat,
            "control_requant_x_r1_nrmse": err_fix,
            "x_int_max_abs": int(np.abs(X_int).max()),
        })
        log(f"WITNESS {files[si].name}: naive layer0 r1 NRMSE={err*100:.4f}% "
            f"sat={sat}/6336 control={err_fix*100:.4f}% "
            f"max|X_int|={int(np.abs(X_int).max())} (limit 2047)")
        log("STOP earliest_block=layer0.r1 (no further layers executed)")

    result = {
        "task": "r2_full_numeric",
        "state": "CONFIG_CONFLICT",
        "evidence_level": "L1 software witness (integer attention + residual, FP32 golden); no RTL",
        "assets": {
            "attn_scale": {"path": str(ATTN), "sha256": sha256(ATTN)},
            "postnorm_scale": {"path": str(POSTNORM), "sha256": sha256(POSTNORM)},
            "frozen_12layer": {"path": str(FROZEN12), "sha256": sha256(FROZEN12)},
        },
        "earliest_block": "layer0.x_scale_contract",
        "invariants": invariants,
        "witness_samples": rows,
        "samples": len(rows),
        "nrmse_limit": 0.03,
        "runtime_seconds": time.time() - t0,
    }
    (TASK / "conflict.json").write_text(json.dumps(result, indent=2) + "\n")
    (TASK / "logs").mkdir(exist_ok=True)
    (TASK / "logs" / "run_conflict_witness.log").write_text(
        "\n".join(lines) + "\n")
    return 2 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(f"RUNTIME_FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
