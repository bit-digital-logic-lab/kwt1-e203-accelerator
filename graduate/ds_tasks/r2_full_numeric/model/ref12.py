#!/usr/bin/env python3
"""Independent integer + FP32 reference for the full 12-layer KWT-1 encoder.

This module is written for the ``encoder_12numeric`` task.  It is a *separate*
implementation of the documented integer contract already frozen by the RTL
(`graduate/rtl/encoder/third_party/*.sv`); it never reads an RTL dump.  The
integer datapath primitives below mirror the specification text:

    mul_sr(a,m,q) = (a*m + 2^(q-1)) >>> q          (round-half-up on signed)
    activations saturate symmetrically to +/- CLAMP_A = 2047
    LayerNorm: population variance, eps_q = round(eps/s^2 * 2^VAR_FRAC),
               isq = round(2^ISQ_FRAC / sqrt(var+eps)), integer sqrt.

Unlike the layer-0 prototype, everything here is parameterised by layer index
and by a *frozen*, training-derived calibration dictionary, so the 12 layers use
DIFFERENT weights AND DIFFERENT scales.

The FP32 golden is imported read-only from
`graduate/encoder/reference/kwt1_fp32_golden.py` (official PostNorm order).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# fixed container / fractional-bit constants (identical to the RTL contract)
# --------------------------------------------------------------------------
D = 64
HID = 256
T = 99
NLAYERS = 12

ACT_BITS = 12
W_BITS = 10
W1_BITS = 10
W2_BITS = 10
CLAMP_A = (1 << (ACT_BITS - 1)) - 1            # 2047
QMAX_W = (1 << (W_BITS - 1)) - 1               # 1023
QMAX_W1 = (1 << (W1_BITS - 1)) - 1
QMAX_W2 = (1 << (W2_BITS - 1)) - 1

QB = 16
G = 16
GO = 16
K = 16
VAF = 16
QG = 28
QGN = 16
LF = 8
LUT_N = 2 * 8 * 16 + 1                          # 257
ISQ_MAX = (1 << (K + 8)) - 1
EPS = 1e-6

# attention integer model constants
F_BITS = 6
M_LUT = 14
G_SH = 20
Q_A = 16


def rint(x):
    """Round-half-to-even on scalars/arrays (numpy rint)."""
    return np.rint(x)


def mul_sr(a, m, q):
    p = int(a) * int(m)
    return (p + (1 << (q - 1))) >> q


def isqrt_hw(v):
    n = int(v)
    res = 0
    bpos = 1 << 62
    for _ in range(32):
        if n >= res + bpos:
            n -= res + bpos
            res = (res >> 1) + bpos
        else:
            res >>= 1
        bpos >>= 2
    assert res == math.isqrt(int(v))
    return res


def _sat(v, clamp=CLAMP_A):
    if v > clamp:
        return clamp, True
    if v < -clamp:
        return -clamp, True
    return v, False


def residual_add(a_row, ma, b_row, mb, qb=QB, clamp=CLAMP_A):
    out = []
    sat = 0
    for i in range(len(a_row)):
        sa = mul_sr(a_row[i], ma, qb)
        sb = mul_sr(b_row[i], mb[i], qb)
        v, s = _sat(sa + sb, clamp)
        out.append(v)
        sat += int(s)
    return out, sat


def layernorm(x_row, eps_q, gcoef, bcoef, mo, clamp=CLAMP_A):
    shift = int(math.log2(D))
    mean = sum(int(v) for v in x_row) >> shift
    ss = sum((int(v) - mean) ** 2 for v in x_row)
    var = ss >> shift
    vq = (var << VAF) + int(eps_q)
    sq = isqrt_hw(vq)
    num = 1 << (K + VAF // 2)
    isq = ISQ_MAX if sq == 0 else min((num + (sq >> 1)) // sq, ISQ_MAX)
    out = []
    sat = 0
    for i in range(len(x_row)):
        d = int(x_row[i]) - mean
        t = d * int(gcoef[i])
        u = mul_sr(t, isq, K)
        v = mul_sr(u, int(mo), GO) + int(bcoef[i])
        v, s = _sat(v, clamp)
        out.append(v)
        sat += int(s)
    return out, sat


def gelu_step(h, mg, mgn, lut, clamp=CLAMP_A):
    qg, qgn, lf = QG, QGN, LF
    vq = (int(h) * int(mg) + (1 << (qg - 1))) >> qg
    v = min(max(vq, -(1 << (lf + 3))), (1 << (lf + 3)) - 1)
    idx = v + (1 << (lf + 3))
    li = idx >> 4
    frac = idx & 15
    g8 = int(lut[li]) + (((int(lut[li + 1]) - int(lut[li])) * frac) >> 4)
    return _sat(mul_sr(g8, int(mgn), qgn), clamp)[0]


def ffn2(g_row, w2, b2):
    out = []
    for k in range(len(b2)):
        acc = 0
        for j in range(len(g_row)):
            acc += int(g_row[j]) * int(w2[j][k])
        if abs(acc) >= (1 << 46):
            raise OverflowError("FFN2 accumulator exceeds 48-bit RTL width")
        out.append(acc + int(b2[k]))
    return out


# --------------------------------------------------------------------------
# FP32 helpers (used only for calibration maxima and for the NRMSE reference)
# --------------------------------------------------------------------------
def gelu_fp32(x):
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def lut_from_gelu():
    return np.array([rint(gelu_fp32((i - 8 * 16) / 16.0) * (1 << LF))
                     for i in range(LUT_N)], dtype=np.int64)


def symmetrize(t, bits):
    m = float(np.abs(t).max())
    scale = m / ((1 << (bits - 1)) - 1)
    return np.rint(t / scale).astype(np.int64), scale


def fit_shift(acc):
    m = int(np.abs(acc).max())
    s = 0
    while (m >> s) > 32767:
        s += 1
    return s


# --------------------------------------------------------------------------
# independent single-layer attention integer model (parameterised by layer)
# --------------------------------------------------------------------------
def attention_int_full(Xf, X_int, Wqf, Wkf, Wvf, Wof, bf, s_x):
    """Full independent attention model.

    Reproduces `attn_int_model.py` exactly but with a *frozen* input scale s_x
    instead of the per-sample symmetrize, and for an arbitrary layer.
    Returns (O_int, so, params).
    """
    Wqi, swq = symmetrize(Wqf, W_BITS)
    Wki, swk = symmetrize(Wkf, W_BITS)
    Wvi, swv = symmetrize(Wvf, W_BITS)
    Woi, swo = symmetrize(Wof, W_BITS)

    Qa = X_int @ Wqi
    RQ = fit_shift(Qa)
    Qi = np.clip((Qa + (1 << (RQ - 1))) >> RQ, -32768, 32767) if RQ else \
        np.clip(Qa, -32768, 32767)
    Ka = X_int @ Wki
    RK = fit_shift(Ka)
    Ki = np.clip((Ka + (1 << (RK - 1))) >> RK, -32768, 32767) if RK else \
        np.clip(Ka, -32768, 32767)
    Va = X_int @ Wvi
    RV = fit_shift(Va)
    Vi = np.clip((Va + (1 << (RV - 1))) >> RV, -32768, 32767) if RV else \
        np.clip(Va, -32768, 32767)
    sq = s_x * swq * (2 ** RQ)
    sk = s_x * swk * (2 ** RK)
    sv = s_x * swv * (2 ** RV)

    S = Qi @ Ki.T
    lut = np.array([int(round((2.0 ** (f / (1 << F_BITS))) * (1 << M_LUT)))
                    for f in range(1 << F_BITS)], dtype=np.int64)

    ss = sq * sk / 8.0
    kc = ss * float(np.log2(np.e)) * (1 << F_BITS)
    SH_KC = 16
    KC = int(round(kc * (1 << SH_KC)))
    while KC < 4096 and SH_KC < 40:
        SH_KC += 1
        KC = int(round(kc * (1 << SH_KC)))
    assert abs(kc * (1 << SH_KC) - KC) < 0.5

    mmax = S.max(axis=1, keepdims=True)
    diff = S - mmax
    A = (diff * KC + (1 << (SH_KC - 1))) >> SH_KC
    ka = A >> F_BITS
    frac = A - (ka << F_BITS)
    pos = np.clip(-ka, 0, 31)
    E = np.where(-ka >= 32, 0, lut[frac] >> pos)
    sumE = E.sum(axis=1, keepdims=True)
    assert (sumE > 0).all()
    R = ((1 << (15 + G_SH)) + (sumE >> 1)) // sumE
    P = np.clip((E * R) >> G_SH, 0, 32767)

    ctx = P @ Vi
    RC = fit_shift(ctx)
    Ci = np.clip((ctx + (1 << (RC - 1))) >> RC, -32768, 32767) if RC else \
        np.clip(ctx, -32768, 32767)
    sct = sv * (2 ** (RC - 15))

    bq = np.rint(bf / (sct * swo)).astype(np.int64)
    Oa = Ci @ Woi + bq
    RO = fit_shift(Oa)
    Oi = np.clip((Oa + (1 << (RO - 1))) >> RO, -32768, 32767) if RO else \
        np.clip(Oa, -32768, 32767)
    so = sct * swo * (2 ** RO)

    params = dict(RQ=int(RQ), RK=int(RK), RV=int(RV), RC=int(RC), RO=int(RO),
                  KC=int(KC), SH_KC=int(SH_KC), F_BITS=F_BITS, M_LUT=M_LUT,
                  G_SH=G_SH, scale_x=s_x, scale_q=sq, scale_k=sk, scale_v=sv,
                  scale_ct=sct, scale_o=so, swq=swq, swk=swk, swv=swv, swo=swo,
                  lut=lut, wq=Wqi, wk=Wki, wv=Wvi, wo=Woi, bias=bq)
    return Oi, so, params


def requant_o_to_a(O_int, M_A, q_a=Q_A):
    return np.clip((O_int * M_A + (1 << (q_a - 1))) >> q_a, -32768, 32767)


# --------------------------------------------------------------------------
# integer weight post-processing for the postnorm blocks
# --------------------------------------------------------------------------
def quantize_w1(W1f, b1f, sx):
    """W1f:(64,256) FP32, b1f:(256,) -> int weights/bias at activation scale sx."""
    sw1 = np.maximum(np.max(np.abs(W1f), axis=0) / QMAX_W1, 1e-30)
    w1i = np.clip(rint(W1f / sw1[None, :]), -QMAX_W1, QMAX_W1).astype(np.int64)
    b1i = rint(b1f / (sw1 * sx)).astype(np.int64)
    sh = sx * sw1
    return w1i, b1i, sw1, sh


def quantize_w2(W2f, b2f, sg):
    sw2 = np.maximum(np.max(np.abs(W2f), axis=0) / QMAX_W2, 1e-30)
    w2i = np.clip(rint(W2f / sw2[None, :]), -QMAX_W2, QMAX_W2).astype(np.int64)
    b2i = rint(b2f / (sg * sw2)).astype(np.int64)
    sh2 = sg * sw2
    return w2i, b2i, sw2, sh2
