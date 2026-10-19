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

Usage: python3 attn_int_model.py
"""

from pathlib import Path
import argparse
import json

import numpy as np

HERE = Path(__file__).resolve().parent
ATT = HERE.parent                       # graduate/rtl/attention
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
PREFIX = "layer_with_weights-0/enc_layers/0/att/"

T = 99
D = 64
ACT_BITS = 12     # activation container use (symmetric, zero point 0)
W_BITS = 10       # weight container use (symmetric, zero point 0)
F_BITS = 6        # exp LUT fraction bits  -> 64 entries
M_LUT = 14        # exp LUT Q format (values in [2^14, 2^15))
G_SH = 20         # reciprocal fractional bits


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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=0,
                    help="real encoder-input sample index (0..5)")
    args = ap.parse_args()

    d = np.load(WSRC)
    xsrc = XSRC.parent / f"sample_{args.sample:02d}_encoder.npy"
    Xf = np.load(xsrc).astype(np.float64)
    Wqf = d[PREFIX + "query_dense/kernel"].astype(np.float64)
    Wkf = d[PREFIX + "key_dense/kernel"].astype(np.float64)
    Wvf = d[PREFIX + "value_dense/kernel"].astype(np.float64)
    Wof = d[PREFIX + "combine_heads/kernel"].astype(np.float64)
    bf = d[PREFIX + "combine_heads/bias"].astype(np.float64)

    # --- quantization (calibration, PC side, allowed by the task) ----------
    Xi, sx = symmetrize(Xf, ACT_BITS)
    Wqi, swq = symmetrize(Wqf, W_BITS)
    Wki, swk = symmetrize(Wkf, W_BITS)
    Wvi, swv = symmetrize(Wvf, W_BITS)
    Woi, swo = symmetrize(Wof, W_BITS)

    Qa = Xi @ Wqi
    RQ = fit_shift(Qa)
    Qi = clip16(round_half_up_shift(Qa, RQ))
    sq = sx * swq * (2 ** RQ)

    Ka = Xi @ Wki
    RK = fit_shift(Ka)
    Ki = clip16(round_half_up_shift(Ka, RK))
    sk = sx * swk * (2 ** RK)

    Va = Xi @ Wvi
    RV = fit_shift(Va)
    Vi = clip16(round_half_up_shift(Va, RV))
    sv = sx * swv * (2 ** RV)

    assert np.abs(Qa).max() < 2 ** 62 and np.abs(Ka).max() < 2 ** 62

    # --- scores + row softmax (fused) --------------------------------------
    S = Qi @ Ki.T                                   # raw integer scores
    assert np.abs(S).max() < 2 ** 62

    lut = np.array([int(round((2.0 ** (f / (1 << F_BITS))) * (1 << M_LUT)))
                    for f in range(1 << F_BITS)], dtype=np.int64)

    ss = sq * sk / 8.0                              # real scale of scaled score
    kc = ss * float(np.log2(np.e)) * (1 << F_BITS)  # real units of A per integer diff
    SH_KC = 16
    KC = int(round(kc * (1 << SH_KC)))
    while KC < 4096 and SH_KC < 40:
        SH_KC += 1
        KC = int(round(kc * (1 << SH_KC)))
    assert abs(kc * (1 << SH_KC) - KC) < 0.5

    mmax = S.max(axis=1, keepdims=True)
    diff = S - mmax                                 # <= 0
    A = round_half_up_shift(diff * KC, SH_KC)       # <= 0
    ka = A >> F_BITS
    frac = A - (ka << F_BITS)
    pos = np.clip(-ka, 0, 31)
    E = np.where(-ka >= 32, 0, lut[frac] >> pos)
    sumE = E.sum(axis=1, keepdims=True)
    assert (sumE > 0).all()

    R = ((1 << (15 + G_SH)) + (sumE >> 1)) // sumE
    P = clip_q15((E * R) >> G_SH)                   # Q15 probabilities
    assert np.abs(E * R).max() < 2 ** 62

    # --- context + output projection ---------------------------------------
    ctx = P @ Vi
    RC = fit_shift(ctx)
    Ci = clip16(round_half_up_shift(ctx, RC))
    sct = sv * (2 ** (RC - 15))

    bq = np.rint(bf / (sct * swo)).astype(np.int64)
    Oa = Ci @ Woi + bq
    assert np.abs(Oa).max() < 2 ** 62
    RO = fit_shift(Oa)
    Oi = clip16(round_half_up_shift(Oa, RO))
    so = sct * swo * (2 ** RO)

    # --- FP32 reference (independent float path) ---------------------------
    def softmax_rows(a):
        a = a - a.max(axis=-1, keepdims=True)
        e = np.exp(a)
        return e / e.sum(axis=-1, keepdims=True)

    att = softmax_rows((Xf @ Wqf) @ (Xf @ Wkf).T / 8.0)
    Of = att @ (Xf @ Wvf) @ Wof + bf
    Odec = Oi * so
    err = Odec - Of
    rms_ref = float(np.sqrt((Of ** 2).mean()))
    nrmse = float(np.sqrt((err ** 2).mean()) / rms_ref)
    # per-row NRMSE for first/last row reporting
    row0 = float(np.sqrt(((Odec[0] - Of[0]) ** 2).mean()) / max(np.sqrt((Of[0] ** 2).mean()), 1e-8))
    rowL = float(np.sqrt(((Odec[-1] - Of[-1]) ** 2).mean()) / max(np.sqrt((Of[-1] ** 2).mean()), 1e-8))

    stats = {
        "T": T, "D": D, "act_bits": ACT_BITS, "w_bits": W_BITS,
        "scale_x": sx, "scale_wq": swq, "scale_wk": swk, "scale_wv": swv, "scale_wo": swo,
        "scale_q": sq, "scale_k": sk, "scale_v": sv, "scale_ct": sct, "scale_o": so,
        "RQ": int(RQ), "RK": int(RK), "RV": int(RV), "RC": int(RC), "RO": int(RO),
        "F_BITS": F_BITS, "M_LUT": M_LUT, "G_SH": G_SH, "KC": int(KC), "SH_KC": int(SH_KC),
        "sample_index": args.sample, "input_file": xsrc.name,
        "nrmse_fp32": nrmse, "nrmse_row0_fp32": row0, "nrmse_row_last_fp32": rowL,
        "max_abs_err_fp32": float(np.abs(err).max()), "rms_ref_fp32": rms_ref,
        "P_rowsum_min": int(P.sum(axis=1).min()), "P_rowsum_max": int(P.sum(axis=1).max()),
        "O_int_max": int(np.abs(Oi).max()), "P_int_max": int(np.abs(P).max()),
    }

    # --- emit RTL params ---------------------------------------------------
    RTL_DIR.mkdir(parents=True, exist_ok=True)
    (RTL_DIR / "attn_params.svh").write_text(
        "// AUTO-GENERATED by model/attn_int_model.py -- do not edit by hand.\n"
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
        % (T, D, F_BITS, M_LUT, G_SH, KC, SH_KC, int(RQ), int(RK), int(RV),
           int(RC), int(RO)))

    # --- emit vectors ------------------------------------------------------
    VEC.mkdir(parents=True, exist_ok=True)
    np.savetxt(VEC / "x.txt", Xi.reshape(-1), fmt="%d")
    np.savetxt(VEC / "wq.txt", Wqi.reshape(-1), fmt="%d")
    np.savetxt(VEC / "wk.txt", Wki.reshape(-1), fmt="%d")
    np.savetxt(VEC / "wv.txt", Wvi.reshape(-1), fmt="%d")
    np.savetxt(VEC / "wo.txt", Woi.reshape(-1), fmt="%d")
    np.savetxt(VEC / "bias.txt", bq.reshape(-1), fmt="%d")
    np.savetxt(VEC / "lut.txt", lut.reshape(-1), fmt="%d")
    for name, arr in (("q", Qi), ("k", Ki), ("v", Vi), ("c", Ci), ("o", Oi)):
        np.savetxt(VEC / f"exp_{name}.txt", arr.reshape(-1), fmt="%d")
    np.savetxt(VEC / "exp_p.txt", P.reshape(-1), fmt="%d")
    np.savetxt(VEC / "fp32_o.txt", Of.reshape(-1), fmt="%.9g")
    (VEC / "params.json").write_text(json.dumps(stats, indent=2))

    print(json.dumps(stats, indent=2))
    print("\ninput X=int16 (99*64), weights int16 (64*64), bias int32 (64)")
    print("expected: q,k,v,c,o int16 (99*64); p Q15 (99*99)")


if __name__ == "__main__":
    main()
