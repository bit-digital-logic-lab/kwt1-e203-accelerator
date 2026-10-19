#!/usr/bin/env python3
"""12-layer calibration + independent integer chain for the encoder_12numeric task.

Design (frozen, gradient-free):

  * calibration split  = the *training* partition of
    ``software/.build/quant_samples_12labels_20.npz`` (240 MFCC samples);
    encoder inputs are built from the frozen checkpoint projection.
  * every activation scale is a per-tensor static maximum over the TRAINING
    split only:  s = max_train |stage| / (2^(A_BITS-1)-1).
  * the layer-to-layer input scale is continuous:  s_x[k] = s_out[k-1], so the
    integer output of layer k-1 is *literally* the integer input of layer k
    (this is exactly the RTL controller's ``y -> x_buf`` copy).
  * validation samples never influence any scale, threshold, LUT or shift.
  * attention shift/scale constants (RQ,RK,RV,RC,RO,KC,SH_KC, M_A) are likewise
    fixed from the TRAINING split.

The integer chain is an independent re-implementation of the documented RTL
contract (see ref12.py header); the FP32 golden is the read-only graduate
module ``graduate/encoder/reference/kwt1_fp32_golden.py``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TASK = HERE.parent
WORKTREE = HERE.parents[3]                       # task/model/.. -> repo root

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(WORKTREE / "graduate/encoder/reference"))
import ref12 as R                                # noqa: E402
import kwt1_fp32_golden as golden                # noqa: E402

D, HID, T, NLAYERS = R.D, R.HID, R.T, R.NLAYERS
CLAMP_A, QB, G, GO, K, VAF, QG, QGN, LF, LUT_N, EPS = (
    R.CLAMP_A, R.QB, R.G, R.GO, R.K, R.VAF, R.QG, R.QGN, R.LF, R.LUT_N, R.EPS)
Q_A = R.Q_A
PREFIX = golden.PREFIX
MAIN = Path("/home/yang/Desktop/ic设计竞赛")        # gitignored data, read-only

STAGE_KEYS = ["attn_projected", "residual1", "ln1", "gelu", "residual2", "layer_output"]


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------
def find_split_npz():
    for cand in (WORKTREE / "software/.build/quant_samples_12labels_20.npz",
                 MAIN / "software/.build/quant_samples_12labels_20.npz"):
        if cand.exists():
            return cand
    raise FileNotFoundError("quant_samples_12labels_20.npz not found")


def find_real_dir():
    for cand in (WORKTREE / "software/validation/real_input_chain_20261017",
                 MAIN / "software/validation/real_input_chain_20261017"):
        if cand.exists():
            return cand
    raise FileNotFoundError("real_input_chain_20261017 not found")


def load_split(npz_path):
    with np.load(npz_path, allow_pickle=True) as z:
        features = z["features"].astype(np.float64)
        partitions = z["partitions"]
        paths = z["paths"]
        words = z["words"]
    if set(np.unique(partitions).tolist()) - {"training", "validation"}:
        raise ValueError("unexpected partition label")
    tr = np.flatnonzero(partitions == "training")
    va = np.flatnonzero(partitions == "validation")
    if set(paths[tr].tolist()) & set(paths[va].tolist()):
        raise ValueError("training/validation paths overlap")
    return features, partitions, paths, words, tr, va


def mfcc_to_encoder(features, weights):
    kernel = weights[f"{PREFIX}/patch_proj/kernel"]
    bias = weights[f"{PREFIX}/patch_proj/bias"]
    cls = weights[f"{PREFIX}/class_emb"][0]
    pos = weights[f"{PREFIX}/pos_emb"][0]
    proj = features @ kernel + bias
    return (np.concatenate([cls, proj], axis=0) + pos).astype(np.float32)


# --------------------------------------------------------------------------
# vectorised integer primitives (same contract as ref12 scalar helpers)
# --------------------------------------------------------------------------
def mul_sr_arr(a, m, q):
    a = np.asarray(a, dtype=np.int64)
    m = np.asarray(m, dtype=np.int64)
    return (a * m + np.int64(1 << (q - 1))) >> np.int64(q)


def _isqrt_arr(v):
    v = np.asarray(v, dtype=np.int64).copy()
    res = np.zeros_like(v)
    bpos = np.full_like(v, np.int64(1) << np.int64(62))
    n = v.copy()
    for _ in range(32):
        cond = n >= res + bpos
        n = np.where(cond, n - (res + bpos), n)
        res = np.where(cond, (res >> 1) + bpos, res >> 1)
        bpos >>= np.int64(2)
    return res


def layernorm_rows(x, eps_q, gcoef, bcoef, mo):
    x = np.asarray(x, dtype=np.int64)
    mean = x.sum(axis=1) >> 6
    d = x - mean[:, None]
    ss = (d * d).sum(axis=1)
    var = ss >> 6
    vq = (var << VAF) + np.int64(eps_q)
    sq = _isqrt_arr(vq)
    num = np.int64(1 << (K + VAF // 2))
    isq = np.where(sq == 0, np.int64(R.ISQ_MAX),
                   np.minimum((num + (sq >> 1)) // np.maximum(sq, 1), np.int64(R.ISQ_MAX)))
    t = d * gcoef[None, :]
    u = mul_sr_arr(t, isq[:, None], K)
    v = mul_sr_arr(u, np.int64(mo), GO) + bcoef[None, :]
    return np.clip(v, -CLAMP_A, CLAMP_A)


def residual_rows(a, ma, b, mb):
    sa = mul_sr_arr(a, np.int64(ma), QB)
    sb = mul_sr_arr(b, np.asarray(mb, dtype=np.int64), QB)
    return np.clip(sa + sb, -CLAMP_A, CLAMP_A)


def gelu_rows(h, mg, mgn, lut):
    vq = (h * mg[None, :] + np.int64(1 << (QG - 1))) >> np.int64(QG)
    v = np.clip(vq, -(1 << (LF + 3)), (1 << (LF + 3)) - 1)
    idx = v + (1 << (LF + 3))
    li = idx >> 4
    frac = idx & 15
    l0 = lut[li]
    l1 = lut[np.minimum(li + 1, LUT_N - 1)]
    g8 = l0 + (((l1 - l0) * frac) >> 4)
    return np.clip(mul_sr_arr(g8, np.int64(mgn), QGN), -CLAMP_A, CLAMP_A)


def ffn2_rows(g, w2, b2):
    return g @ w2 + b2[None, :]


# ---- attention, vectorised, on frozen per-layer params -------------------
def attention_proj_max(X_int, W):
    return int(np.abs(X_int @ W).max())


def fit_shift(m):
    s = 0
    while (m >> s) > 32767:
        s += 1
    return s


def attention_rows(X_int, ap):
    """Vectorised attention using the frozen params dict `ap`."""
    RQ, RK, RV, RC, RO = ap["RQ"], ap["RK"], ap["RV"], ap["RC"], ap["RO"]
    KC, SH_KC, F_BITS, G_SH = ap["KC"], ap["SH_KC"], ap["F_BITS"], ap["G_SH"]
    lut = np.asarray(ap["lut"], dtype=np.int64)

    Qa = X_int @ ap["wq"]
    Qi = np.clip((Qa + (1 << (RQ - 1))) >> RQ, -32768, 32767) if RQ else np.clip(Qa, -32768, 32767)
    Ka = X_int @ ap["wk"]
    Ki = np.clip((Ka + (1 << (RK - 1))) >> RK, -32768, 32767) if RK else np.clip(Ka, -32768, 32767)
    Va = X_int @ ap["wv"]
    Vi = np.clip((Va + (1 << (RV - 1))) >> RV, -32768, 32767) if RV else np.clip(Va, -32768, 32767)

    S = Qi @ Ki.T
    mmax = S.max(axis=1, keepdims=True)
    A = (S - mmax) * KC
    A = (A + (1 << (SH_KC - 1))) >> SH_KC
    ka = A >> F_BITS
    frac = A - (ka << F_BITS)
    pos = np.clip(-ka, 0, 31)
    E = np.where(-ka >= 32, 0, lut[frac] >> pos)
    sumE = E.sum(axis=1, keepdims=True)
    Rr = ((1 << (15 + G_SH)) + (sumE >> 1)) // sumE
    P = np.clip((E * Rr) >> G_SH, 0, 32767)

    ctx = P @ Vi
    Ci = np.clip((ctx + (1 << (RC - 1))) >> RC, -32768, 32767) if RC else np.clip(ctx, -32768, 32767)
    Oa = Ci @ ap["wo"] + ap["bias"][None, :]
    Oi = np.clip((Oa + (1 << (RO - 1))) >> RO, -32768, 32767) if RO else np.clip(Oa, -32768, 32767)
    return Oi


def attention_scalar(X_int, ap):
    """Per-row scalar attention on frozen params (bit-exactness self-test)."""
    RQ, RK, RV, RC, RO = ap["RQ"], ap["RK"], ap["RV"], ap["RC"], ap["RO"]
    KC, SH_KC, F_BITS, G_SH = ap["KC"], ap["SH_KC"], ap["F_BITS"], ap["G_SH"]
    lut = ap["lut"]
    Tn = X_int.shape[0]
    Qi = np.clip((X_int @ ap["wq"] + (1 << (RQ - 1))) >> RQ if RQ else X_int @ ap["wq"],
                 -32768, 32767)
    Ki = np.clip((X_int @ ap["wk"] + (1 << (RK - 1))) >> RK if RK else X_int @ ap["wk"],
                 -32768, 32767)
    Vi = np.clip((X_int @ ap["wv"] + (1 << (RV - 1))) >> RV if RV else X_int @ ap["wv"],
                 -32768, 32767)
    out = np.zeros((Tn, D), dtype=np.int64)
    for i in range(Tn):
        srow = [int(np.dot(Qi[i], Ki[j])) for j in range(Tn)]
        mmax = max(srow)
        erow = []
        for j in range(Tn):
            a = ((srow[j] - mmax) * KC + (1 << (SH_KC - 1))) >> SH_KC
            ka = a >> F_BITS
            fr = a - (ka << F_BITS)
            sh = -ka
            if -ka >= 32:
                erow.append(0)
            else:
                erow.append(int(lut[fr]) >> sh if sh >= 0 else int(lut[fr]) << 0)
        sumE = sum(erow)
        Rr = ((1 << (15 + G_SH)) + (sumE >> 1)) // sumE
        cacc = []
        for d in range(D):
            acc = 0
            for j in range(Tn):
                p = (erow[j] * Rr) >> G_SH
                p = 32767 if p > 32767 else p
                acc += p * int(Vi[j][d])
            cacc.append(acc)
        Ci = np.clip(np.array([(v + (1 << (RC - 1))) >> RC if RC else v for v in cacc]),
                     -32768, 32767)
        Oa = Ci @ ap["wo"] + ap["bias"]
        out[i] = np.clip((Oa + (1 << (RO - 1))) >> RO if RO else Oa, -32768, 32767)
    return out


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------
def _golden_arrays(weights, x0):
    """Run the FP32 golden and return (logits, x, stages, per-layer arrays)."""
    arrs = []
    logits, x, stages = golden.forward(x0, weights, NLAYERS, arrays=arrs)
    return logits, x, stages, arrs


def fp32_pass(weights, X0_list):
    """One pass over the calibration split.

    Returns per-layer maxima of the scaled activations and, for every layer, the
    stacked FP32 input X_k (needed to freeze the attention shifts/scale).
    """
    mx = {k: np.zeros(NLAYERS) for k in STAGE_KEYS}
    mx_x0 = 0.0
    Xk = [[] for _ in range(NLAYERS)]
    for x0 in X0_list:
        mx_x0 = max(mx_x0, float(np.max(np.abs(x0))))
        _logits, _x, _stages, arrs = _golden_arrays(weights, x0)
        for k in range(NLAYERS):
            for key in STAGE_KEYS:
                mx[key][k] = max(mx[key][k], float(np.max(np.abs(arrs[k][key]))))
            Xk[k].append(x0 if k == 0 else arrs[k - 1]["layer_output"])
    return mx, mx_x0, [np.concatenate(v, axis=0) for v in Xk]


def build_calibration(weights, X0_train):
    mx, mx_x0, Xk_stack = fp32_pass(weights, X0_train)

    s_x = np.zeros(NLAYERS)
    s_a = np.maximum(mx["attn_projected"], 1e-30) / CLAMP_A
    s_r1 = np.maximum(mx["residual1"], 1e-30) / CLAMP_A
    s_r2 = np.maximum(mx["residual2"], 1e-30) / CLAMP_A
    s_out = np.maximum(mx["layer_output"], 1e-30) / CLAMP_A
    sx_ffn1 = np.maximum(mx["ln1"], 1e-30) / CLAMP_A
    s_g = np.maximum(mx["gelu"], 1e-30) / CLAMP_A
    s_x[0] = max(mx_x0, 1e-30) / CLAMP_A
    s_x[1:] = s_out[:-1]                          # continuous layer input scale

    lut = R.lut_from_gelu()
    attn_lut = np.array([int(round((2.0 ** (f / (1 << R.F_BITS))) * (1 << R.M_LUT)))
                         for f in range(1 << R.F_BITS)], dtype=np.int64)
    layers = []
    for k in range(NLAYERS):
        p = f"{PREFIX}/enc_layers/{k}"
        Wqf = weights[f"{p}/att/query_dense/kernel"].astype(np.float64)
        Wkf = weights[f"{p}/att/key_dense/kernel"].astype(np.float64)
        Wvf = weights[f"{p}/att/value_dense/kernel"].astype(np.float64)
        Wof = weights[f"{p}/att/combine_heads/kernel"].astype(np.float64)
        bf = weights[f"{p}/att/combine_heads/bias"].astype(np.float64)

        Wqi, swq = R.symmetrize(Wqf, R.W_BITS)
        Wki, swk = R.symmetrize(Wkf, R.W_BITS)
        Wvi, swv = R.symmetrize(Wvf, R.W_BITS)
        Woi, swo = R.symmetrize(Wof, R.W_BITS)

        Xint_tr = np.clip(R.rint(Xk_stack[k] / s_x[k]), -CLAMP_A, CLAMP_A).astype(np.int64)
        RQ = fit_shift(attention_proj_max(Xint_tr, Wqi))
        RK = fit_shift(attention_proj_max(Xint_tr, Wki))
        RV = fit_shift(attention_proj_max(Xint_tr, Wvi))
        sq = s_x[k] * swq * (2 ** RQ)
        sk = s_x[k] * swk * (2 ** RK)
        sv = s_x[k] * swv * (2 ** RV)

        ss = sq * sk / 8.0
        kc = ss * float(np.log2(np.e)) * (1 << R.F_BITS)
        SH_KC = 16
        KC = int(round(kc * (1 << SH_KC)))
        while KC < 4096 and SH_KC < 40:
            SH_KC += 1
            KC = int(round(kc * (1 << SH_KC)))
        if not abs(kc * (1 << SH_KC) - KC) < 0.5:
            raise ValueError(f"layer {k}: KC rounding too coarse")

        ap = dict(RQ=int(RQ), RK=int(RK), RV=int(RV), RC=0, RO=0,
                  KC=int(KC), SH_KC=int(SH_KC), F_BITS=R.F_BITS, M_LUT=R.M_LUT,
                  G_SH=R.G_SH, lut=attn_lut, wq=Wqi, wk=Wki, wv=Wvi, wo=Woi)

        # context scale / output shift / bias / scale_o are data driven: run the
        # attention on the *training* stack with the projection shifts fixed.
        Qa = Xint_tr @ Wqi
        Qi = np.clip((Qa + (1 << (RQ - 1))) >> RQ, -32768, 32767) if RQ else np.clip(Qa, -32768, 32767)
        Ka = Xint_tr @ Wki
        Ki = np.clip((Ka + (1 << (RK - 1))) >> RK, -32768, 32767) if RK else np.clip(Ka, -32768, 32767)
        Va = Xint_tr @ Wvi
        Vi = np.clip((Va + (1 << (RV - 1))) >> RV, -32768, 32767) if RV else np.clip(Va, -32768, 32767)
        # The context scale and the output shift are data driven; sweep the
        # TRAINING stack once, accumulating the raw (unshifted) context and
        # output-projection magnitudes, then freeze RC/RO from those maxima.
        ctxmax = 0
        for start in range(0, Xint_tr.shape[0], 99):
            S = Qi[start:start + 99] @ Ki[start:start + 99].T
            mmax = S.max(axis=1, keepdims=True)
            A = ((S - mmax) * KC + (1 << (SH_KC - 1))) >> SH_KC
            ka = A >> R.F_BITS
            frac = A - (ka << R.F_BITS)
            pos = np.clip(-ka, 0, 31)
            E = np.where(-ka >= 32, 0, attn_lut[frac] >> pos)
            sumE = E.sum(axis=1, keepdims=True)
            Rr = ((1 << (15 + R.G_SH)) + (sumE >> 1)) // sumE
            P = np.clip((E * Rr) >> R.G_SH, 0, 32767)
            ctxraw = P.astype(np.int64) @ Vi[start:start + 99].astype(np.int64)
            ctxmax = max(ctxmax, int(np.abs(ctxraw).max()))
        RC = fit_shift(ctxmax)
        sct = sv * (2 ** (RC - 15))
        bq = np.rint(bf / (sct * swo)).astype(np.int64)
        oamax = 0
        for start in range(0, Xint_tr.shape[0], 99):
            S = Qi[start:start + 99] @ Ki[start:start + 99].T
            mmax = S.max(axis=1, keepdims=True)
            A = ((S - mmax) * KC + (1 << (SH_KC - 1))) >> SH_KC
            ka = A >> R.F_BITS
            frac = A - (ka << R.F_BITS)
            pos = np.clip(-ka, 0, 31)
            E = np.where(-ka >= 32, 0, attn_lut[frac] >> pos)
            sumE = E.sum(axis=1, keepdims=True)
            Rr = ((1 << (15 + R.G_SH)) + (sumE >> 1)) // sumE
            P = np.clip((E * Rr) >> R.G_SH, 0, 32767)
            ctx = P.astype(np.int64) @ Vi[start:start + 99].astype(np.int64)
            Ci = np.clip((ctx + (1 << (RC - 1))) >> RC if RC else ctx, -32768, 32767)
            oamax = max(oamax, int(np.abs(Ci @ Woi + bq[None, :]).max()))
        RO = fit_shift(oamax)
        so = sct * swo * (2 ** RO)
        ap.update(RC=int(RC), RO=int(RO), bias=bq,
                  scale_x=float(s_x[k]), scale_o=float(so),
                  scale_q=float(sq), scale_k=float(sk), scale_v=float(sv),
                  scale_ct=float(sct), swq=float(swq), swk=float(swk),
                  swv=float(swv), swo=float(swo))

        M_A = int(R.rint(so / s_a[k] * (1 << Q_A)))
        if not (0 < M_A < (1 << 20)):
            raise ValueError(f"layer {k}: requant coefficient out of range: {M_A}")

        W1f = weights[f"{p}/ffn/layer_with_weights-0/kernel"].astype(np.float64)
        b1f = weights[f"{p}/ffn/layer_with_weights-0/bias"].astype(np.float64)
        w1i, b1i, sw1, sh = R.quantize_w1(W1f, b1f, sx_ffn1[k])

        W2f = weights[f"{p}/ffn/layer_with_weights-1/kernel"].astype(np.float64)
        b2f = weights[f"{p}/ffn/layer_with_weights-1/bias"].astype(np.float64)
        w2i, b2i, sw2, sh2 = R.quantize_w2(W2f, b2f, s_g[k])

        mg = R.rint(sh * (1 << (LF + QG))).astype(np.int64)
        mgn = int(R.rint((1 << QGN) / ((1 << LF) * s_g[k])))

        ma1 = int(R.rint(s_x[k] / s_r1[k] * (1 << QB)))
        mb1 = int(R.rint(s_a[k] / s_r1[k] * (1 << QB)))
        eps_q1 = int(R.rint(EPS / (s_r1[k] ** 2) * (1 << VAF)))
        gamma1 = weights[f"{p}/layernorm1/gamma"].astype(np.float64)
        beta1 = weights[f"{p}/layernorm1/beta"].astype(np.float64)
        g1f = R.rint(gamma1 * (1 << G)).astype(np.int64)
        b1fk = R.rint(beta1 / sx_ffn1[k]).astype(np.int64)
        mo1 = int(R.rint((1 << GO) / ((1 << G) * sx_ffn1[k])))

        mc2 = int(R.rint(sx_ffn1[k] / s_r2[k] * (1 << QB)))
        md2 = R.rint(sh2 / s_r2[k] * (1 << QB)).astype(np.int64)
        eps_q2 = int(R.rint(EPS / (s_r2[k] ** 2) * (1 << VAF)))
        gamma2 = weights[f"{p}/layernorm2/gamma"].astype(np.float64)
        beta2 = weights[f"{p}/layernorm2/beta"].astype(np.float64)
        g2f = R.rint(gamma2 * (1 << G)).astype(np.int64)
        b2fk = R.rint(beta2 / s_out[k]).astype(np.int64)
        mo2 = int(R.rint((1 << GO) / ((1 << G) * s_out[k])))

        for nm, val in (("ma1", ma1), ("mb1", mb1), ("eps_q1", eps_q1), ("mo1", mo1),
                        ("mc2", mc2), ("eps_q2", eps_q2), ("mo2", mo2), ("mgn", mgn),
                        ("M_A", M_A), ("KC", KC), ("SH_KC", SH_KC)):
            if abs(val) > (1 << 31) - 1:
                raise OverflowError(f"layer {k}: coefficient {nm} exceeds INT32")
        for nm, arr in (("g1f", g1f), ("b1f", b1fk), ("g2f", g2f), ("b2f", b2fk),
                        ("md2", md2), ("mg", mg), ("bias", bq)):
            if np.any(np.abs(arr) > (1 << 31) - 1):
                raise OverflowError(f"layer {k}: coefficient {nm} exceeds INT32")

        layers.append(dict(
            layer=k, M_A=M_A,
            scales=dict(s_x=float(s_x[k]), s_a=float(s_a[k]), sx_ffn1=float(sx_ffn1[k]),
                        s_g=float(s_g[k]), s_r1=float(s_r1[k]), s_r2=float(s_r2[k]),
                        s_out=float(s_out[k])),
            attn=dict(params={kk: vv for kk, vv in ap.items()
                              if kk not in ("lut", "wq", "wk", "wv", "wo", "bias")},
                      lut=[int(v) for v in attn_lut],
                      wq=Wqi, wk=Wki, wv=Wvi, wo=Woi, bias=bq),
            ffn1=dict(w=w1i, b=b1i, sw=sw1, sh=sh),
            ffn2=dict(w=w2i, b=b2i, sw=sw2, sh2=sh2),
            coeff=dict(ma1=ma1, mb1=mb1, eps_q1=eps_q1, mo1=mo1, mgn=mgn,
                       ma2=mc2, eps_q2=eps_q2, mo2=mo2,
                       mg=mg, g1f=g1f, b1f=b1fk, g2f=g2f, b2f=b2fk, md2=md2),
        ))

    calib = dict(schema="encoder12-calib/1", layers=layers,
                 lut=[int(v) for v in lut],
                 attn_lut=[int(v) for v in attn_lut],
                 maxima={k: [float(v) for v in mx[k]] for k in STAGE_KEYS},
                 max_x0=float(mx_x0))
    return calib


# --------------------------------------------------------------------------
# integer forward chain
# --------------------------------------------------------------------------
def ap_of(calib, L):
    ap = dict(L["attn"]["params"])
    ap["lut"] = np.asarray(calib.get("attn_lut", calib["lut"]), dtype=np.int64)
    for kk in ("wq", "wk", "wv", "wo", "bias"):
        ap[kk] = L["attn"][kk]
    return ap


def integer_chain(calib, X0_fp32, want_stages=True):
    layers = calib["layers"]
    s_x0 = layers[0]["scales"]["s_x"]
    x_int = np.clip(R.rint(X0_fp32 / s_x0), -CLAMP_A, CLAMP_A).astype(np.int64)

    out = {k: [] for k in ("a", "r1", "out1", "g", "f", "r2", "y")}
    deq = {k: [] for k in ("a", "r1", "g", "f", "r2", "y")}
    sats = {"r1": 0, "r2": 0, "y": 0}
    for L in layers:
        sc = L["scales"]
        ap = ap_of(calib, L)
        o_int = attention_rows(x_int, ap)
        a_int = np.clip((o_int * L["M_A"] + (1 << (Q_A - 1))) >> Q_A, -32768, 32767)
        c = L["coeff"]
        r1 = residual_rows(x_int, c["ma1"], a_int, c["mb1"])
        out1 = layernorm_rows(r1, c["eps_q1"], c["g1f"], c["b1f"], c["mo1"])
        h = np.clip(out1 @ L["ffn1"]["w"] + L["ffn1"]["b"][None, :],
                    -(1 << 31), (1 << 31) - 1)
        g = gelu_rows(h, c["mg"], c["mgn"], np.asarray(calib["lut"], dtype=np.int64))
        f = np.clip(ffn2_rows(g, L["ffn2"]["w"], L["ffn2"]["b"]),
                    -(1 << 31), (1 << 31) - 1)
        r2 = residual_rows(out1, c["ma2"], f, c["md2"])
        y = layernorm_rows(r2, c["eps_q2"], c["g2f"], c["b2f"], c["mo2"])

        s_h2 = L["ffn2"]["sh2"]
        if want_stages:
            out["a"].append(a_int); out["r1"].append(r1); out["out1"].append(out1)
            out["g"].append(g); out["f"].append(f); out["r2"].append(r2); out["y"].append(y)
        deq["a"].append(a_int * sc["s_a"])
        deq["r1"].append(r1 * sc["s_r1"])
        deq["g"].append(g * sc["s_g"])
        deq["f"].append(f * s_h2[None, :])
        deq["r2"].append(r2 * sc["s_r2"])
        deq["y"].append(y * sc["s_out"])
        sats["r1"] += int(np.sum(np.abs(r1) == CLAMP_A))
        sats["r2"] += int(np.sum(np.abs(r2) == CLAMP_A))
        sats["y"] += int(np.sum(np.abs(y) == CLAMP_A))
        x_int = y
    return dict(int={k: (np.array(v) if want_stages else None) for k, v in out.items()},
                deq={k: np.array(v) for k, v in deq.items()}, sat=sats)


def nrmse(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))
