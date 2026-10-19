#!/usr/bin/env python3
"""attn_scale: freeze attention calibration on the TRAINING split, evaluate on the
independent VALIDATION split.

Problem fixed
-------------
The copied attention integer model derived its input activation scale (`sx`) and
every requant shift (`RQ/RK/RV/RC/RO`) from the *evaluation* sample itself, so the
hardware constants depended on the validation data (a calibration leak). Here:

  * calibration (no gradients) reads only `split == "training"` samples and
    freezes `scale_x, RQ, RK, RV, RC, RO, KC, SH_KC, bias_int, scale_o` into
    `attn_calib_train.json`;
  * evaluation reads that frozen file and quantizes each validation sample with
    the *frozen* constants; no scale is re-fitted from validation data;
  * this driver re-implements the whole integer chain independently (own LUT,
    own rounding) so the frozen path and each per-sample "leaky" path can be
    compared, and the model's output is cross-checked bit for bit.

Evidence levels are labelled in README.md: this driver is L1 (software reference)
only; RTL bit-exact / E203 / synthesis / official acceptance are NOT claimed.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL = HERE.parents[1] / "encoder" / "model"          # graduate/encoder/model
sys.path.insert(0, str(MODEL))
import attn_int_model as m                              # noqa: E402

ACT_BITS, F_BITS, M_LUT, G_SH = m.ACT_BITS, m.F_BITS, m.M_LUT, m.G_SH
A12_HI = (1 << (ACT_BITS - 1)) - 1
LUT = np.array([int(round((2.0 ** (f / (1 << F_BITS))) * (1 << M_LUT)))
                for f in range(1 << F_BITS)], dtype=np.int64)
NRMSE_LIMIT = 0.03


# --- independent integer primitives (separate from attn_int_model) ---------
def rq(v, s):
    return v if s == 0 else (v + (1 << (s - 1))) >> s


def clamp16(v):
    return np.clip(v, -32768, 32767)


def fit_shift(acc):
    mx = int(np.abs(acc).max())
    s = 0
    while (mx >> s) > 32767:
        s += 1
    return s


def fit_constants(Xs, wf, verbose=False):
    """Freeze every data-dependent constant from the given float inputs."""
    wi, sw = {}, {}
    for k in ("wq", "wk", "wv", "wo"):
        mx = float(np.abs(wf[k]).max())
        sw[k] = mx / ((1 << (m.W_BITS - 1)) - 1)
        wi[k] = np.rint(wf[k] / sw[k]).astype(np.int64)
    act_max = max(float(np.abs(x).max()) for x in Xs)
    sx = act_max / A12_HI

    def xi(x):
        return np.clip(np.rint(x / sx), -32768, 32767).astype(np.int64)

    shifts = {}
    for tag, key in (("RQ", "wq"), ("RK", "wk"), ("RV", "wv")):
        shifts[tag] = max(fit_shift(xi(x) @ wi[key]) for x in Xs)
    sq = sx * sw["wq"] * (2 ** shifts["RQ"])
    sk = sx * sw["wk"] * (2 ** shifts["RK"])
    sv = sx * sw["wv"] * (2 ** shifts["RV"])
    kc = sq * sk / 8.0 * float(np.log2(np.e)) * (1 << F_BITS)
    sh_kc = 16
    KC = int(round(kc * (1 << sh_kc)))
    while KC < 4096 and sh_kc < 40:
        sh_kc += 1
        KC = int(round(kc * (1 << sh_kc)))
    shifts.update(KC=KC, SH_KC=sh_kc)

    def ctx_of(x):
        st = _softmax_ctx(xi(x), wi, shifts)
        return st["ctx"], st["Vi"]

    ctxs = [ctx_of(x)[0] for x in Xs]
    RC = max(fit_shift(c) for c in ctxs)
    sct = sv * (2 ** (RC - 15))
    bias = np.rint(wf["bias"] / (sct * sw["wo"])).astype(np.int64)
    Oas = [clamp16(rq(c, RC)) @ wi["wo"] + bias for c in ctxs]
    RO = max(fit_shift(o) for o in Oas)
    so = sct * sw["wo"] * (2 ** RO)

    c = dict(scale_x=float(sx), weight_scales={k: float(v) for k, v in sw.items()},
             RQ=shifts["RQ"], RK=shifts["RK"], RV=shifts["RV"],
             KC=KC, SH_KC=sh_kc, RC=RC, RO=RO,
             bias_int=[int(v) for v in bias], scale_o=float(so),
             scale_q=float(sq), scale_k=float(sk), scale_v=float(sv),
             scale_ct=float(sct))
    if verbose:
        print(f"  fit: sx={sx:.9e} RQ={c['RQ']} RK={c['RK']} RV={c['RV']} "
              f"RC={RC} RO={RO} KC={KC} SH_KC={sh_kc}")
    return c, wi


def _softmax_ctx(Xi, wi, sh):
    Qi = clamp16(rq(Xi @ wi["wq"], sh["RQ"]))
    Ki = clamp16(rq(Xi @ wi["wk"], sh["RK"]))
    Vi = clamp16(rq(Xi @ wi["wv"], sh["RV"]))
    S = Qi @ Ki.T
    diff = S - S.max(axis=1, keepdims=True)
    A = rq(diff * sh["KC"], sh["SH_KC"])
    ka = A >> F_BITS
    frac = A - (ka << F_BITS)
    E = np.where(-ka >= 32, 0, LUT[frac] >> np.clip(-ka, 0, 31))
    sumE = E.sum(axis=1, keepdims=True)
    R = ((1 << (15 + G_SH)) + (sumE >> 1)) // sumE
    P = np.clip((E * R) >> G_SH, 0, 32767)
    return {"P": P, "Vi": Vi, "ctx": P @ Vi}


def int_forward(Xf, c, wi):
    Xi = np.clip(np.rint(Xf / c["scale_x"]), -32768, 32767).astype(np.int64)
    st = _softmax_ctx(Xi, wi, c)
    Ci = clamp16(rq(st["ctx"], c["RC"]))
    Oa = Ci @ wi["wo"] + np.asarray(c["bias_int"], dtype=np.int64)
    Oi = clamp16(rq(Oa, c["RO"]))
    return Oi, st["P"], Ci


def fp32_forward(Xf, wf):
    a = (Xf @ wf["wq"]) @ (Xf @ wf["wk"]).T / 8.0
    a = a - a.max(axis=-1, keepdims=True)
    e = np.exp(a)
    P = e / e.sum(axis=-1, keepdims=True)
    return P @ (Xf @ wf["wv"]) @ wf["wo"] + wf["bias"]


def nrmse(Oi, so, Of):
    err = Oi * so - Of
    return float(np.sqrt((err ** 2).mean()) / np.sqrt((Of ** 2).mean())), float(np.abs(err).max())


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calib", type=Path, default=m.DEFAULT_CALIB)
    ap.add_argument("--results", type=Path, default=HERE / "results.json")
    args = ap.parse_args()

    print("== attn_scale: training-frozen calibration, independent validation ==")
    split = m.load_split()
    train, val = split["training"], split["validation"]
    print(f"samples: training={len(train)} validation={len(val)}")
    print(f"training  : {[p.name for p in train]}")
    print(f"validation: {[p.name for p in val]}")

    wf = m.load_weights()

    # 1. freeze constants from the training split (through the model API)
    print("\n[1] calibration on training split (no gradients)")
    model_cal = m.calibrate()
    model_cal_2 = m.calibrate()          # determinism check
    if model_cal != model_cal_2:
        print("  calibration is not deterministic")
        raise SystemExit(3)
    print("  determinism: two independent calibrations are identical")
    m.save_calibration(model_cal, args.calib)
    calib_sha_frozen = sha256(args.calib)

    # 2. independent re-derivation of the same constants (own implementation)
    print("\n[2] independent re-derivation + cross-check")
    Xtr = [np.load(p).astype(np.float64) for p in train]
    drv_cal, wi = fit_constants(Xtr, wf, verbose=True)
    keys = ["scale_x", "RQ", "RK", "RV", "KC", "SH_KC", "RC", "RO", "scale_o",
            "scale_ct"]
    mism = {k: (model_cal[k], drv_cal[k]) for k in keys if model_cal[k] != drv_cal[k]}
    bias_ok = model_cal["bias_int"] == drv_cal["bias_int"]
    print(f"  constant cross-check: {'MATCH' if not mism and bias_ok else 'MISMATCH'}")
    if mism or not bias_ok:
        print(f"  mismatches: {mism} bias_equal={bias_ok}")
        raise SystemExit(2)

    # 3. evaluate each validation sample with the frozen constants
    print("\n[3] validation evaluation (frozen constants; no re-fitting)")
    frozen = m.load_calibration(args.calib)
    rows = []
    for p in val:
        Xf = np.load(p).astype(np.float64)
        Of = fp32_forward(Xf, wf)
        # driver path
        Oi_d, P_d, Ci_d = int_forward(Xf, frozen, wi)
        nd, mad = nrmse(Oi_d, frozen["scale_o"], Of)
        # model path (must agree bit for bit)
        stats, out_m, Of_m = m.evaluate_sample(p, frozen, wi, wf)
        oi_match = bool(np.array_equal(Oi_d, out_m["Oi"]))
        p_match = bool(np.array_equal(P_d, out_m["P"]))
        rows.append(dict(sample=p.name, nrmse=nd, max_abs=mad,
                         bit_exact_driver_vs_model=bool(oi_match and p_match),
                         P_rowsum=[int(P_d.sum(axis=1).min()), int(P_d.sum(axis=1).max())],
                         O_int_max=int(np.abs(Oi_d).max())))
        print(f"  {p.name}: NRMSE {nd*100:.4f}%  max_abs {mad:.3e}  "
              f"driver==model {oi_match and p_match}")

    # 4. training-split reference error + per-sample leaky diagnostic
    print("\n[4] training reference + leaky (per-sample scale) diagnostic")
    train_rows = []
    for p in train:
        Xf = np.load(p).astype(np.float64)
        Of = fp32_forward(Xf, wf)
        Oi, _, _ = int_forward(Xf, frozen, wi)
        n, mad = nrmse(Oi, frozen["scale_o"], Of)
        train_rows.append(dict(sample=p.name, nrmse=n, max_abs=mad))
        print(f"  train {p.name}: NRMSE {n*100:.4f}%")

    leaky = []
    for p in val:
        Xf = np.load(p).astype(np.float64)
        c_l, _ = fit_constants([Xf], wf)
        Oi, _, _ = int_forward(Xf, c_l, wi)
        n, _mad = nrmse(Oi, c_l["scale_o"], fp32_forward(Xf, wf))
        leaky.append(dict(sample=p.name, nrmse=n, scale_x=c_l["scale_x"],
                          RQ=c_l["RQ"], RK=c_l["RK"], RV=c_l["RV"],
                          RC=c_l["RC"], RO=c_l["RO"], KC=c_l["KC"],
                          SH_KC=c_l["SH_KC"]))
        print(f"  leaky(baseline) {p.name}: NRMSE {n*100:.4f}%  "
              f"sx={c_l['scale_x']:.6e} RQ/RK/RV={c_l['RQ']}/{c_l['RK']}/{c_l['RV']} "
              f"KC={c_l['KC']}")

    # 5. leak audit
    print("\n[5] leak audit")
    calib_sha_after = sha256(args.calib)
    prov = {s["file"] for s in frozen["calibration_samples"]}
    valnames = {p.name for p in val}
    checks = {
        "calibration_file_unchanged_by_eval": calib_sha_frozen == calib_sha_after,
        "calibration_split_is_training": frozen["calibration_split"] == "training",
        "no_validation_sample_in_calibration_provenance": not (prov & valnames),
        "frozen_sx_differs_from_each_validation_sx":
            all(abs(frozen["scale_x"] - l["scale_x"]) > 0 for l in leaky),
        "driver_matches_model_bit_exact":
            all(r["bit_exact_driver_vs_model"] for r in rows),
        "all_validation_nrmse_below_limit":
            all(r["nrmse"] < NRMSE_LIMIT for r in rows),
    }
    for k, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {k}")

    results = dict(
        task="attn_scale",
        calibration=dict(split="training", frozen=True,
                         samples=[p.name for p in train],
                         calib_file=args.calib.name, calib_sha256=calib_sha_frozen,
                         constants={k: frozen[k] for k in keys},
                         bias_int_len=len(frozen["bias_int"]),
                         repeat_deterministic=True),
        validation=dict(split="validation", samples=[p.name for p in val], rows=rows),
        training_reference=train_rows,
        leaky_baseline=leaky,
        leak_audit=checks,
        nrmse_limit=NRMSE_LIMIT,
        evidence_levels=dict(
            software_reference="provided (run_attn_scale.py + attn_int_model.py, this results file)",
            rtl_bit_exact="provided (rtl_check.sh: attention RTL 41481/41481 per validation sample)",
            e203="not evidence (no real E203 SoC in this task)",
            synthesis="not evidence",
            official_acceptance="not evidence (no official 600-sample evaluator available)",
        ),
    )
    args.results.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nRESULT written: {args.results}")

    if not all(checks.values()):
        print("SCALE_AUDIT_FAIL")
        raise SystemExit(1)
    print("SCALE_AUDIT_PASS")


if __name__ == "__main__":
    main()
