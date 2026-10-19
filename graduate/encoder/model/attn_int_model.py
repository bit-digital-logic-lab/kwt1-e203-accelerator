#!/usr/bin/env python3
"""Independent integer reference model for the single-head attention RTL prototype.

Scope: layer-0 single-head attention of ARM KWT-1 (d_model=64, heads=1, T=99).
    Q = X @ Wq                     (99x64 int16)
    K = X @ Wk
    V = X @ Wv
    S = Q @ K^T / sqrt(64)         -> kept as scaled score, scaled by 1/8 in softmax
    P = softmax_rows(S)            (99x99, Q15)
    C = P @ V                      (99x64 int16)
    O = C @ Wo + bias              (99x64 int16)

This module is the *independent numeric model* the RTL is compared against
bit-for-bit. It also emits:
  - rtl/attn_params.svh : compile-time localparams (shifts, exp LUT) for the RTL
  - vectors/*.txt       : inputs + expected integer tensors + FP32 reference

It does not import the RTL; the two are independent implementations of the same
documented integer contract. The FP32 (float64 here, treated as the FP32
reference) error is reported separately from the bit-exact integer comparison.

Calibration boundary (leak fix, 2026-10-19)
-------------------------------------------
Every *data-dependent* constant is a hardware compile-time constant and MUST be
frozen from the official **training split** only (no gradients):

    scale_x (input activation scale), RQ/RK/RV (Q/K/V requant shifts),
    KC/SH_KC (softmax exponent coefficient), RC (context shift),
    bias_int (output bias), RO (output shift), scale_o.

The **validation split** is used for evaluation only. It never enters
`calibrate()` and `integer_forward()` never re-derives a scale from its input.
`calibrate()` reads the split labels from the official manifest and asserts that
only `split == "training"` files are consumed. The frozen constants are written
to a JSON file (`graduate/ds_tasks/attn_scale/attn_calib_train.json`) which is
then loaded for inference; the per-row softmax maximum (`mmax`) is the only
input-dependent quantity and is a runtime operation inside the algorithm, not a
frozen calibration threshold.

Usage:
    python3 attn_int_model.py                       # eval validation sample 0
    python3 attn_int_model.py --sample 2            # eval validation sample 2
    python3 attn_int_model.py --recalibrate         # (re)freeze from training split
"""

from pathlib import Path
import argparse
import hashlib
import json

import numpy as np

HERE = Path(__file__).resolve().parent
ATT = HERE.parent                       # graduate/encoder
RTL_DIR = ATT / "rtl"
VEC = ATT / "vectors"

# --- frozen inputs (real KWT-1 layer-0 attention) --------------------------
# [encoder-assembly copy] only the repository-root resolution is adjusted so the
# file works at the shallower graduate/encoder/model/ path; numeric code below is
# byte-identical to the grad-attention source (see PROVENANCE.md).
def _find_root(p):
    for a in [p, *p.parents]:
        if (a / "software" / "weights" / "kwt1_fp32.npz").exists():
            return a
    raise FileNotFoundError("repository root with software/weights/kwt1_fp32.npz not found")
_ROOT = _find_root(HERE)
WSRC = _ROOT / "software" / "weights" / "kwt1_fp32.npz"
XSRC = _ROOT / "software" / "validation" / "real_input_chain_20261017" / "sample_00_encoder.npy"
SPLIT_MANIFEST = (_ROOT / "software" / "validation" / "real_input_chain_20261017"
                  / "manifest.json")
DEFAULT_CALIB = _ROOT / "graduate" / "ds_tasks" / "attn_scale" / "attn_calib_train.json"
PREFIX = "layer_with_weights-0/enc_layers/0/att/"

T = 99
D = 64
ACT_BITS = 12     # activation container use (symmetric, zero point 0)
W_BITS = 10       # weight container use (symmetric, zero point 0)
F_BITS = 6        # exp LUT fraction bits  -> 64 entries
M_LUT = 14        # exp LUT Q format (values in [2^14, 2^15))
G_SH = 20         # reciprocal fractional bits

# design constant ROM, independent of any data
EXP_LUT = np.array([int(round((2.0 ** (f / (1 << F_BITS))) * (1 << M_LUT)))
                    for f in range(1 << F_BITS)], dtype=np.int64)


def round_half_up_shift(x, s):
    """(x + 2^(s-1)) >> s with floor shift on the integer domain (matches RTL)."""
    return x if s == 0 else (x + (1 << (s - 1))) >> s


def symmetrize(t, bits):
    m = float(np.abs(t).max())
    scale = m / ((1 << (bits - 1)) - 1)
    return np.rint(t / scale).astype(np.int64), scale


def fit_shift(acc):
    """smallest right-shift so that max|acc>>s| <= 32767."""
    m = int(np.abs(acc).max())
    s = 0
    while (m >> s) > 32767:
        s += 1
    return s


def clip16(v):
    return np.clip(v, -32768, 32767)


def clip_q15(v):
    return np.clip(v, 0, 32767)


def quantize_activation(t, sx):
    """Round to the int16 container with a *frozen* scale (no re-fitting)."""
    return np.clip(np.rint(t / sx), -32768, 32767).astype(np.int64)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# official train / validation split (read from the manifest, not guessed)
# --------------------------------------------------------------------------

def load_manifest():
    man = json.loads(SPLIT_MANIFEST.read_text())
    return man, SPLIT_MANIFEST.parent


def load_split():
    man, root = load_manifest()
    out = {"training": [], "validation": []}
    for s in man["samples"]:
        out[s["split"]].append((root / s["encoder_file"]).resolve())
    return out


def load_weights():
    d = np.load(WSRC)
    return {
        "wq": d[PREFIX + "query_dense/kernel"].astype(np.float64),
        "wk": d[PREFIX + "key_dense/kernel"].astype(np.float64),
        "wv": d[PREFIX + "value_dense/kernel"].astype(np.float64),
        "wo": d[PREFIX + "combine_heads/kernel"].astype(np.float64),
        "bias": d[PREFIX + "combine_heads/bias"].astype(np.float64),
    }


def quantize_weights(wf):
    """Weight-only quantization: depends only on the frozen checkpoint, no split."""
    wi, sw = {}, {}
    for k in ("wq", "wk", "wv", "wo"):
        wi[k], sw[k] = symmetrize(wf[k], W_BITS)
    return wi, sw


# --------------------------------------------------------------------------
# softmax + context (shared by calibration and evaluation)
# --------------------------------------------------------------------------

def _softmax_ctx(Xi, wi, shifts):
    RQ, RK, RV = shifts["RQ"], shifts["RK"], shifts["RV"]
    Qi = clip16(round_half_up_shift(Xi @ wi["wq"], RQ))
    Ki = clip16(round_half_up_shift(Xi @ wi["wk"], RK))
    Vi = clip16(round_half_up_shift(Xi @ wi["wv"], RV))
    S = Qi @ Ki.T                                   # raw integer scores
    assert np.abs(S).max() < 2 ** 62

    mmax = S.max(axis=1, keepdims=True)             # runtime per-row max
    diff = S - mmax                                 # <= 0
    A = round_half_up_shift(diff * shifts["KC"], shifts["SH_KC"])   # <= 0
    ka = A >> F_BITS
    frac = A - (ka << F_BITS)
    pos = np.clip(-ka, 0, 31)
    E = np.where(-ka >= 32, 0, EXP_LUT[frac] >> pos)
    sumE = E.sum(axis=1, keepdims=True)
    assert (sumE > 0).all()

    R = ((1 << (15 + G_SH)) + (sumE >> 1)) // sumE
    P = clip_q15((E * R) >> G_SH)                   # Q15 probabilities
    assert np.abs(E * R).max() < 2 ** 62
    ctx = P @ Vi
    return dict(Qi=Qi, Ki=Ki, Vi=Vi, S=S, P=P, ctx=ctx)


def _kcoef(scale_q, scale_k):
    """Softmax exponent coefficient KC with SH_KC fractional bits (frozen)."""
    kc = scale_q * scale_k / 8.0 * float(np.log2(np.e)) * (1 << F_BITS)
    SH_KC = 16
    KC = int(round(kc * (1 << SH_KC)))
    while KC < 4096 and SH_KC < 40:
        SH_KC += 1
        KC = int(round(kc * (1 << SH_KC)))
    assert abs(kc * (1 << SH_KC) - KC) < 0.5
    return KC, SH_KC


# --------------------------------------------------------------------------
# calibration (training split only, no gradients)
# --------------------------------------------------------------------------

def calibrate(verbose=True):
    split = load_split()
    train = split["training"]
    val = split["validation"]
    assert train, "training split is empty"
    # hard guard: no validation file may ever be used to fit a constant
    train_set = {str(p) for p in train}
    val_set = {str(p) for p in val}
    assert train_set.isdisjoint(val_set)

    wf = load_weights()
    wi, sw = quantize_weights(wf)
    man, root = load_manifest()
    recs = {s["encoder_file"]: s for s in man["samples"]}

    # provenance: verify each training input against the official manifest hash
    train_records = []
    for p in train:
        rec = recs[p.name]
        assert rec["split"] == "training", f"{p.name} is not a training sample"
        arr = np.load(p)
        data_sha = hashlib.sha256(arr.tobytes()).hexdigest()
        assert data_sha == rec["encoder_sha256"], \
            f"{p.name} data hash {data_sha} != manifest {rec['encoder_sha256']}"
        train_records.append({"file": p.name, "split": "training",
                              "data_sha256": data_sha})
    Xs = [np.load(p).astype(np.float64) for p in train]

    # 1. input activation scale = max over the training split
    act_max = max(float(np.abs(x).max()) for x in Xs)
    sx = act_max / ((1 << (ACT_BITS - 1)) - 1)

    def xi(x):
        return quantize_activation(x, sx)

    # 2. per-tensor static requant shifts fitted on the training split
    RQ = max(fit_shift(xi(x) @ wi["wq"]) for x in Xs)
    RK = max(fit_shift(xi(x) @ wi["wk"]) for x in Xs)
    RV = max(fit_shift(xi(x) @ wi["wv"]) for x in Xs)
    sq = sx * sw["wq"] * (2 ** RQ)
    sk = sx * sw["wk"] * (2 ** RK)
    sv = sx * sw["wv"] * (2 ** RV)

    # 3. softmax coefficient (design constant derived from frozen scales)
    KC, SH_KC = _kcoef(sq, sk)
    shifts = dict(RQ=RQ, RK=RK, RV=RV, KC=KC, SH_KC=SH_KC)

    # 4. context shift from the training split
    ctxs = [_softmax_ctx(xi(x), wi, shifts)["ctx"] for x in Xs]
    RC = max(fit_shift(c) for c in ctxs)
    sct = sv * (2 ** (RC - 15))

    # 5. output bias (frozen), then output shift from the training split
    bias_int = np.rint(wf["bias"] / (sct * sw["wo"])).astype(np.int64)
    Roas = []
    for c in ctxs:
        Ci = clip16(round_half_up_shift(c, RC))
        Roas.append(Ci @ wi["wo"] + bias_int)
    RO = max(fit_shift(o) for o in Roas)
    so = sct * sw["wo"] * (2 ** RO)

    cal = {
        "schema": "attn-int-calib/1",
        "layer": 0,
        "act_bits": ACT_BITS, "w_bits": W_BITS, "f_bits": F_BITS,
        "m_lut": M_LUT, "g_sh": G_SH,
        "calibration_split": "training",
        "calibration_samples": train_records,
        "validation_samples": [p.name for p in val],
        "weights_sha256": sha256_file(WSRC),
        "scale_x": float(sx),
        "weight_scales": {k: float(v) for k, v in sw.items()},
        "RQ": int(RQ), "RK": int(RK), "RV": int(RV),
        "RC": int(RC), "RO": int(RO),
        "KC": int(KC), "SH_KC": int(SH_KC),
        "scale_q": float(sq), "scale_k": float(sk), "scale_v": float(sv),
        "scale_ct": float(sct), "scale_o": float(so),
        "bias_int": [int(v) for v in bias_int],
    }
    if verbose:
        print(f"CALIB_OK split=training samples={len(train)} "
              f"(validation={len(val)} never used) sx={sx:.9e} "
              f"RQ={RQ} RK={RK} RV={RV} RC={RC} RO={RO} KC={KC} SH_KC={SH_KC}")
    return cal


def save_calibration(cal, path=DEFAULT_CALIB):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cal, indent=2, sort_keys=True) + "\n")
    return path


def load_calibration(path=DEFAULT_CALIB):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"frozen calibration not found: {path}\n"
            "run `python3 attn_int_model.py --recalibrate` (training split) first")
    cal = json.loads(path.read_text())
    # guard: a calibration file may only carry training-split provenance
    assert cal.get("calibration_split") == "training", "calibration is not training-frozen"
    used = {s["file"] for s in cal.get("calibration_samples", [])}
    if used & set(cal.get("validation_samples", [])):
        raise AssertionError("validation sample present in calibration provenance")
    return cal


# --------------------------------------------------------------------------
# evaluation (validation split; constants are read, never re-fitted)
# --------------------------------------------------------------------------

def integer_forward(Xf, cal, wi, bias_int=None):
    """Full integer attention forward with frozen constants."""
    if bias_int is None:
        bias_int = np.asarray(cal["bias_int"], dtype=np.int64)
    sx = cal["scale_x"]
    Xi = quantize_activation(Xf, sx)
    st = _softmax_ctx(Xi, wi, cal)
    Ci = clip16(round_half_up_shift(st["ctx"], cal["RC"]))
    Oa = Ci @ wi["wo"] + bias_int
    assert np.abs(Oa).max() < 2 ** 62
    Oi = clip16(round_half_up_shift(Oa, cal["RO"]))
    return dict(Xi=Xi, Ci=Ci, Oi=Oi, Qi=st["Qi"], Ki=st["Ki"], Vi=st["Vi"],
                P=st["P"], ctx=st["ctx"])


def fp32_forward(Xf, wf):
    """Independent float reference for the same attention contract."""
    def softmax_rows(a):
        a = a - a.max(axis=-1, keepdims=True)
        e = np.exp(a)
        return e / e.sum(axis=-1, keepdims=True)
    att = softmax_rows((Xf @ wf["wq"]) @ (Xf @ wf["wk"]).T / 8.0)
    return att @ (Xf @ wf["wv"]) @ wf["wo"] + wf["bias"]


def evaluate_sample(sample_path, cal, wi, wf):
    Xf = np.load(sample_path).astype(np.float64)
    out = integer_forward(Xf, cal, wi)
    Of = fp32_forward(Xf, wf)
    Oi, so = out["Oi"], cal["scale_o"]
    Odec = Oi * so
    err = Odec - Of
    rms_ref = float(np.sqrt((Of ** 2).mean()))
    nrmse = float(np.sqrt((err ** 2).mean()) / rms_ref)
    rx = out["Xi"]
    return dict(
        sample=Path(sample_path).name, shape=list(Xf.shape),
        scale_x=cal["scale_x"], scale_o=so,
        x_int_max=int(np.abs(rx).max()),
        x_int_sat_int16=int((np.abs(rx) >= 32767).sum()),
        x_int_over_a12=int((np.abs(rx) > ((1 << (ACT_BITS - 1)) - 1)).sum()),
        nrmse_fp32=nrmse, max_abs_err_fp32=float(np.abs(err).max()),
        rms_ref_fp32=rms_ref,
        P_rowsum_min=int(out["P"].sum(axis=1).min()),
        P_rowsum_max=int(out["P"].sum(axis=1).max()),
        O_int_max=int(np.abs(Oi).max()),
    ), out, Of


# --------------------------------------------------------------------------
# vector / params emission (consumed by vectors/gen_vectors.py)
# --------------------------------------------------------------------------

def emit(out, Of, cal, wi, sample_index):
    Xi, Oi, P = out["Xi"], out["Oi"], out["P"]
    bq = np.asarray(cal["bias_int"], dtype=np.int64)

    # --- emit RTL params ---------------------------------------------------
    RTL_DIR.mkdir(parents=True, exist_ok=True)
    (RTL_DIR / "attn_params.svh").write_text(
        "// AUTO-GENERATED by model/attn_int_model.py -- do not edit by hand.\n"
        "// Constants frozen from the training split (see ds_tasks/attn_scale).\n"
        "localparam int ATT_T     = %d;\n"
        "localparam int ATT_D     = %d;\n"
        "localparam int F_BITS    = %d;\n"
        "localparam int M_LUT     = %d;\n"
        "localparam int G_SH      = %d;\n"
        "localparam int KC        = %d;\n"
        "localparam int SH_KC     = %d;\n"
        "localparam int RQ        = %d;\n"
        "localparam int RK        = %d;\n"
        "localparam int RV        = %d;\n"
        "localparam int RC        = %d;\n"
        "localparam int RO        = %d;\n"
        % (T, D, F_BITS, M_LUT, G_SH, cal["KC"], cal["SH_KC"], cal["RQ"],
           cal["RK"], cal["RV"], cal["RC"], cal["RO"]))

    # --- emit vectors ------------------------------------------------------
    VEC.mkdir(parents=True, exist_ok=True)
    np.savetxt(VEC / "x.txt", Xi.reshape(-1), fmt="%d")
    for name, key in (("wq", "wq"), ("wk", "wk"), ("wv", "wv"), ("wo", "wo")):
        np.savetxt(VEC / f"{name}.txt", wi[key].reshape(-1), fmt="%d")
    np.savetxt(VEC / "bias.txt", bq.reshape(-1), fmt="%d")
    np.savetxt(VEC / "lut.txt", EXP_LUT.reshape(-1), fmt="%d")
    for name, arr in (("q", out["Qi"]), ("k", out["Ki"]), ("v", out["Vi"]),
                      ("c", out["Ci"]), ("o", Oi)):
        np.savetxt(VEC / f"exp_{name}.txt", arr.reshape(-1), fmt="%d")
    np.savetxt(VEC / "exp_p.txt", P.reshape(-1), fmt="%d")
    np.savetxt(VEC / "fp32_o.txt", Of.reshape(-1), fmt="%.9g")


def build_stats(sample_path, stats, cal):
    st = dict(stats)
    st.update({
        "T": T, "D": D, "act_bits": ACT_BITS, "w_bits": W_BITS,
        "F_BITS": F_BITS, "M_LUT": M_LUT, "G_SH": G_SH,
        "calibration_split": "training",
        "calibration_samples": [s["file"] for s in cal["calibration_samples"]],
        "sample_index": stats.get("sample_index"),
        "input_file": Path(sample_path).name,
        "RQ": cal["RQ"], "RK": cal["RK"], "RV": cal["RV"],
        "RC": cal["RC"], "RO": cal["RO"],
        "KC": cal["KC"], "SH_KC": cal["SH_KC"],
    })
    return st


def main():
    ap = argparse.ArgumentParser(description="integer attention reference model")
    ap.add_argument("--sample", type=int, default=0,
                    help="validation sample index (0..N-1)")
    ap.add_argument("--calib", type=Path, default=DEFAULT_CALIB,
                    help="frozen calibration JSON (training-split constants)")
    ap.add_argument("--recalibrate", action="store_true",
                    help="recompute the frozen constants from the training split")
    ap.add_argument("--write-vectors", dest="write_vectors", action="store_true",
                    default=True)
    ap.add_argument("--no-write-vectors", dest="write_vectors", action="store_false")
    args = ap.parse_args()

    if args.recalibrate:
        cal = calibrate()
        save_calibration(cal, args.calib)
    else:
        cal = load_calibration(args.calib)

    split = load_split()
    samples = split["validation"]
    if not (0 <= args.sample < len(samples)):
        raise SystemExit(f"--sample {args.sample} outside validation range "
                         f"0..{len(samples)-1}")
    xsrc = samples[args.sample]

    wf = load_weights()
    wi, _sw = quantize_weights(wf)
    stats, out, Of = evaluate_sample(xsrc, cal, wi, wf)
    stats["sample_index"] = args.sample
    stats = build_stats(xsrc, stats, cal)

    if args.write_vectors:
        emit(out, Of, cal, wi, args.sample)
        (VEC / "params.json").write_text(json.dumps(stats, indent=2))

    print(json.dumps(stats, indent=2))
    print("\ninput X=int16 (99*64), weights int16 (64*64), bias int32 (64)")
    print("expected: q,k,v,c,o int16 (99*64); p Q15 (99*99)")


if __name__ == "__main__":
    main()
