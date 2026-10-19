#!/usr/bin/env python3
"""p027_precision -- parameterised integer reference for the 12-layer KWT-1 encoder.

This is an EXPERIMENTAL reference.  It does not modify ``encoder_12numeric`` or
the RTL baseline; it re-implements the same documented integer contract with
the bit widths as explicit parameters, so that hardware-representable precision
schemes can be compared on the same 240 training / 240 validation split.

Contract (unchanged from the frozen baseline unless a parameter says otherwise):

  * per-tensor static symmetric activation scale from TRAINING maxima only:
        s = max_train|stage| / (2^(A_stage-1) - 1),  zero point 0
  * layer-to-layer scale continuity: s_x[k] = s_out[k-1], so the integer output
    of layer k-1 is literally the integer input of layer k (no requant).
  * integer primitives:
        mul_sr(a,m,q) = (a*m + 2^(q-1)) >> q        (round-half-up, signed)
        activations saturate symmetrically
        LayerNorm: population variance, eps_q = round(eps/s^2 * 2^VAF),
                   isq = round(2^(K+VAF/2) / isqrt(var<<VAF + eps_q))
        GELU: 257-node piecewise-linear LUT over [-8,8), Q(LF) output
        attention: per-tensor Q/K/V/O shifts, exp2 LUT softmax, requant M_A/2^Q_A
  * attention internal registers are 16-bit (QKV_BITS / CTX_BITS); this is a
    real hardware bound and is reported rather than hidden.

Widths that can be searched: activation bits per stage, attention/FFN weight
bits, softmax LUT fractional bits (with optional linear interpolation), GELU
output fractional bits.  Accumulators are int64 in software; the *required*
signed width is measured and reported (the RTL baseline used INT48 for FFN2).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

D, HID, T, NLAYERS = 64, 256, 99, 12
STAGES = ["a", "r1", "g", "f", "r2", "y"]

QB = 16          # residual coefficient fraction
Q_A = 16         # attention -> activation requant fraction
G = 16           # LayerNorm gamma fraction
GO = 16          # LayerNorm output coefficient fraction
K = 16           # LayerNorm inverse-sqrt fraction
VAF = 16         # LayerNorm variance fraction
QG = 28          # GELU input rescale fraction
QGN = 16         # GELU output rescale fraction
EPS = 1e-6
LUT_N = 257      # GELU nodes at the baseline (LF=8, 16-LSB node spacing)

# default = the frozen A12W10 baseline
DEFAULT_CFG = dict(
    stage_bits=dict(x=12, a=12, r1=12, ln1=12, g=12, r2=12, y=12),
    w_attn=10, w1=10, w2=10,
    f_table=6, f_interp=0, m_lut=14, g_sh=20,
    lf=8, qkv_bits=16, ctx_bits=16, out_bits=16,
    qb=16, headroom=1.0, gelu_shift=4,
    prob_bits=15, softmax_round=False, qg=None, isq_frac=None, kc_min=4096,
    a_clip_bits=None,
)


def cfg_norm(cfg: dict) -> dict:
    c = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULT_CFG.items()}
    for k, v in cfg.items():
        if k == "stage_bits":
            c["stage_bits"].update(v)
        else:
            c[k] = v
    # y feeds the next layer's x directly: the integer width must be identical
    c["stage_bits"]["y"] = c["stage_bits"]["x"]
    return c


def clamp_of(bits: int) -> int:
    return (1 << (bits - 1)) - 1


def rint(x):
    return np.rint(x)


# --------------------------------------------------------------------------
# integer primitives
# --------------------------------------------------------------------------
def mul_sr(a, m, q):
    a = np.asarray(a, dtype=np.int64)
    m = np.asarray(m, dtype=np.int64)
    return (a * m + np.int64(1 << (q - 1))) >> np.int64(q)


def isqrt_arr(v):
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


ISQ_REG_BITS = 31          # stored inverse-sqrt register cap (hardware bound)


def isq_max(k_frac):
    return (1 << ISQ_REG_BITS) - 1


def layernorm_rows(x, eps_q, gcoef, bcoef, mo, clamp, k_frac=K):
    """LayerNorm with an explicit inverse-sqrt fraction width.

    ``k_frac`` must grow with the activation width: ``var`` scales as
    2^(2A) while ``isq = 2^(k_frac+VAF/2)/sqrt(var<<VAF)`` is a *stored
    integer*.  Keeping ``k_frac = 16`` at A=16 would quantise isq to a 2-bit
    value (a real fixed-point bug, not a data effect), so the LN reciprocal
    width is a first-class search parameter.
    """
    x = np.asarray(x, dtype=np.int64)
    mean = x.sum(axis=1) >> 6
    d = x - mean[:, None]
    ss = (d * d).sum(axis=1)
    var = ss >> 6
    vq = (var << VAF) + np.int64(eps_q)
    sq = isqrt_arr(vq)
    imax = np.int64(isq_max(k_frac))
    num = np.int64(1 << (k_frac + VAF // 2))
    isq = np.where(sq == 0, imax,
                   np.minimum((num + (sq >> 1)) // np.maximum(sq, 1), imax))
    t = d * gcoef[None, :]
    prod = t * isq[:, None]
    if np.any(np.abs(prod) >= (1 << 62)):
        raise OverflowError("LayerNorm product exceeds int64; reduce isq_frac")
    u = (prod + np.int64(1 << (k_frac - 1))) >> np.int64(k_frac)
    v = mul_sr(u, np.int64(mo), GO) + bcoef[None, :]
    return np.clip(v, -clamp, clamp)


def residual_rows(a, ma, b, mb, clamp, qb=QB):
    sa = mul_sr(a, np.int64(ma), qb)
    sb = mul_sr(b, np.asarray(mb, dtype=np.int64), qb)
    return np.clip(sa + sb, -clamp, clamp)


def gelu_rows(h, mg, mgn, lut, lf, clamp, shift=4, qg=QG):
    """GELU with a piecewise-linear LUT.

    ``shift`` = log2(node spacing in LSB).  The real-domain node spacing is
    2^shift / 2^lf, so raising ``lf`` alone sharpens the output quantization
    while raising the node count requires a smaller ``shift``.
    """
    lut_n = lut.size
    vq = (h * mg[None, :] + np.int64(1 << (qg - 1))) >> np.int64(qg)
    lim = 1 << (lf + 3)
    v = np.clip(vq, -lim, lim - 1)
    idx = v + lim
    li = idx >> shift
    frac = idx & ((1 << shift) - 1)
    l0 = lut[li]
    l1 = lut[np.minimum(li + 1, lut_n - 1)]
    g8 = l0 + (((l1 - l0) * frac) >> shift)
    return np.clip(mul_sr(g8, np.int64(mgn), QGN), -clamp, clamp)


def gelu_qg(cfg):
    """GELU input-rescale fraction.

    ``mg = round(sx*sw * 2^(LF+QG))`` and ``sx*sw`` scales as 2^-A, so QG must
    grow with the activation width or the stored multiplier loses relative
    precision (at A=20 a fixed QG=28 gives mg~134, i.e. 0.7% multiplier noise).
    """
    if cfg.get("qg"):
        return int(cfg["qg"])
    return QG + max(int(v) for v in cfg["stage_bits"].values()) - 12


def lut_from_gelu(lf, shift=4):
    """LUT over the real domain [-8, 8) with node spacing 2^shift / 2^lf."""
    n = (1 << (lf + 4 - shift)) + 1
    xs = (np.arange(n) * (1 << shift) - (1 << (lf + 3))) / float(1 << lf)
    g = 0.5 * xs * (1.0 + np.vectorize(math.erf)(xs / math.sqrt(2.0)))
    return np.rint(g * (1 << lf)).astype(np.int64)


def exp2_lut(f_table, m_lut):
    return np.array([int(round((2.0 ** (f / (1 << f_table))) * (1 << m_lut)))
                     for f in range(1 << f_table)], dtype=np.int64)


def symmetrize(t, bits):
    m = float(np.abs(t).max())
    scale = max(m / ((1 << (bits - 1)) - 1), 1e-30)
    return np.rint(t / scale).astype(np.int64), scale


def fit_shift(acc, bits):
    m = int(np.abs(acc).max())
    lim = (1 << (bits - 1)) - 1
    s = 0
    while (m >> s) > lim:
        s += 1
    return s


def _shift_round(x, s, bits):
    lim = (1 << (bits - 1)) - 1
    if s:
        x = (x + (1 << (s - 1))) >> s
    return np.clip(x, -lim - 1, lim)


# --------------------------------------------------------------------------
# batched attention (identical integer contract, one sample = 99 rows)
# --------------------------------------------------------------------------
def attention_batch(X_int, ap, cfg):
    """X_int: (N,T,D) int64.  Returns (O_int (N,T,D), stats dict)."""
    f_bits = cfg["f_table"] + cfg["f_interp"]
    f_int = cfg["f_interp"]
    lut = ap["lut"]
    qb = cfg["qkv_bits"]
    cb = cfg["ctx_bits"]
    ob = cfg["out_bits"]

    Qa = X_int @ ap["wq"]
    Qi = _shift_round(Qa, ap["RQ"], qb)
    Ka = X_int @ ap["wk"]
    Ki = _shift_round(Ka, ap["RK"], qb)
    Va = X_int @ ap["wv"]
    Vi = _shift_round(Va, ap["RV"], qb)

    S = Qi @ Ki.transpose(0, 2, 1)
    mmax = S.max(axis=-1, keepdims=True)
    A = ((S - mmax) * np.int64(ap["KC"]) + np.int64(1 << (ap["SH_KC"] - 1))) >> np.int64(ap["SH_KC"])
    ka = A >> f_bits
    frac = A - (ka << f_bits)
    if f_int:
        hi = frac >> f_int
        lo = frac - (hi << f_int)
        e0 = (lut[hi] * ((1 << f_int) - lo) + lut[np.minimum(hi + 1, lut.size - 1)] * lo
              + (1 << (f_int - 1))) >> f_int
    else:
        e0 = lut[frac]
    pos = np.clip(-ka, 0, 31)
    if cfg.get("softmax_round"):
        e0 = (e0 + (np.int64(1) << np.maximum(pos - 1, 0))) >> pos
        E = np.where(-ka >= 32, np.int64(0), e0)
    else:
        E = np.where(-ka >= 32, np.int64(0), e0 >> pos)
    sumE = E.sum(axis=-1, keepdims=True)
    pb = int(cfg.get("prob_bits", 15))
    Rr = ((np.int64(1) << (pb + cfg["g_sh"])) + (sumE >> 1)) // sumE
    P = np.clip((E * Rr) >> cfg["g_sh"], 0, (1 << pb) - 1)

    ctx = P @ Vi
    Ci = _shift_round(ctx, ap["RC"], cb)
    Oa = Ci @ ap["wo"] + ap["bias"][None, None, :]
    Oi = _shift_round(Oa, ap["RO"], ob)
    stats = dict(
        qkv_acc_bits=max(int(np.abs(Qa).max()), int(np.abs(Ka).max()),
                         int(np.abs(Va).max())).bit_length() + 1,
        score_acc_bits=int(np.abs(S).max()).bit_length() + 1,
        ctx_acc_bits=int(np.abs(ctx).max()).bit_length() + 1,
        out_acc_bits=int(np.abs(Oa).max()).bit_length() + 1,
        exp_entries=lut.size,
    )
    return Oi, stats


# --------------------------------------------------------------------------
# calibration (training split only)
# --------------------------------------------------------------------------
def fp32_stage_maxima(weights, X0_list, golden, nlayers=NLAYERS):
    """FP32 stage maxima and the per-layer FP32 input stacks (training only)."""
    keys = ["attn_projected", "residual1", "ln1", "gelu", "residual2", "layer_output"]
    mx = {k: np.zeros(nlayers) for k in keys}
    mx_x0 = 0.0
    Xk = [[] for _ in range(nlayers)]
    for x0 in X0_list:
        mx_x0 = max(mx_x0, float(np.max(np.abs(x0))))
        arrs = []
        golden.forward(x0, weights, nlayers, arrays=arrs)
        for k in range(nlayers):
            for key in keys:
                mx[key][k] = max(mx[key][k], float(np.max(np.abs(arrs[k][key]))))
            Xk[k].append(x0 if k == 0 else arrs[k - 1]["layer_output"])
    return mx, mx_x0, [np.concatenate(v, axis=0) for v in Xk]


def build_calibration(weights, maxima, Xk_stack, cfg, nlayers=NLAYERS):
    """Freeze scales/coefficients from TRAINING maxima.  Validation is untouched."""
    cfg = cfg_norm(cfg)
    sb = cfg["stage_bits"]
    mx, mx_x0 = maxima
    cl = {k: clamp_of(sb[k]) for k in ("x", "a", "r1", "ln1", "g", "r2", "y")}
    cl["y"] = cl["x"]

    s_x = np.zeros(nlayers)
    hr = float(cfg.get("headroom", 1.0))
    s_a = np.maximum(mx["attn_projected"], 1e-30) * hr / cl["a"]
    s_r1 = np.maximum(mx["residual1"], 1e-30) * hr / cl["r1"]
    s_r2 = np.maximum(mx["residual2"], 1e-30) * hr / cl["r2"]
    s_out = np.maximum(mx["layer_output"], 1e-30) * hr / cl["y"]
    sx_ffn1 = np.maximum(mx["ln1"], 1e-30) * hr / cl["ln1"]
    s_g = np.maximum(mx["gelu"], 1e-30) * hr / cl["g"]
    s_x[0] = max(mx_x0, 1e-30) * hr / cl["x"]
    s_x[1:] = s_out[:-1]                      # continuous layer input scale

    lut = lut_from_gelu(cfg["lf"], cfg["gelu_shift"])
    attn_lut = exp2_lut(cfg["f_table"], cfg["m_lut"])
    f_bits = cfg["f_table"] + cfg["f_interp"]

    layers = []
    for k in range(nlayers):
        p = f"layer_with_weights-0/enc_layers/{k}"
        Wqf = weights[f"{p}/att/query_dense/kernel"].astype(np.float64)
        Wkf = weights[f"{p}/att/key_dense/kernel"].astype(np.float64)
        Wvf = weights[f"{p}/att/value_dense/kernel"].astype(np.float64)
        Wof = weights[f"{p}/att/combine_heads/kernel"].astype(np.float64)
        bf = weights[f"{p}/att/combine_heads/bias"].astype(np.float64)

        Wqi, swq = symmetrize(Wqf, cfg["w_attn"])
        Wki, swk = symmetrize(Wkf, cfg["w_attn"])
        Wvi, swv = symmetrize(Wvf, cfg["w_attn"])
        Woi, swo = symmetrize(Wof, cfg["w_attn"])

        Xint_tr = np.clip(rint(Xk_stack[k] / s_x[k]), -cl["x"] - 1,
                          cl["x"]).astype(np.int64)
        RQ = fit_shift(attention_proj_max(Xint_tr, Wqi), cfg["qkv_bits"])
        RK = fit_shift(attention_proj_max(Xint_tr, Wki), cfg["qkv_bits"])
        RV = fit_shift(attention_proj_max(Xint_tr, Wvi), cfg["qkv_bits"])
        sq = s_x[k] * swq * (2 ** RQ)
        sk = s_x[k] * swk * (2 ** RK)
        sv = s_x[k] * swv * (2 ** RV)

        ss = sq * sk / 8.0
        kc = ss * float(np.log2(np.e)) * (1 << f_bits)
        SH_KC = 16
        KC = int(round(kc * (1 << SH_KC)))
        # KC must stay a wide integer: the shift budget is 40 bits total.
        while KC < int(cfg["kc_min"]) and SH_KC < 40:
            SH_KC += 1
            KC = int(round(kc * (1 << SH_KC)))
        if not abs(kc * (1 << SH_KC) - KC) < 0.5:
            raise ValueError(f"layer {k}: KC rounding too coarse")

        ap = dict(RQ=int(RQ), RK=int(RK), RV=int(RV), RC=0, RO=0,
                  KC=int(KC), SH_KC=int(SH_KC), lut=attn_lut,
                  wq=Wqi, wk=Wki, wv=Wvi, wo=Woi)

        # data-driven context/output shifts, swept over the TRAINING stack
        Qa = Xint_tr @ Wqi
        Qi = _shift_round(Qa, RQ, cfg["qkv_bits"])
        Ka = Xint_tr @ Wki
        Ki = _shift_round(Ka, RK, cfg["qkv_bits"])
        Va = Xint_tr @ Wvi
        Vi = _shift_round(Va, RV, cfg["qkv_bits"])

        ctxmax = 0
        for start in range(0, Xint_tr.shape[0], T):
            n = min(T, Xint_tr.shape[0] - start)
            P = _softmax_probs(Qi[start:start + n], Ki[start:start + n], KC, SH_KC,
                               f_bits, cfg["f_interp"], attn_lut, cfg["g_sh"],
                               cfg["prob_bits"], cfg["softmax_round"])
            ctxraw = P @ Vi[start:start + n]
            ctxmax = max(ctxmax, int(np.abs(ctxraw).max()))
        RC = fit_shift(np.array([ctxmax]), cfg["ctx_bits"])
        # P is probability scaled by 2^prob_bits, so ctx/2^RC has scale
        # sv * 2^(RC - prob_bits); this is exact for the baseline (15 bits).
        sct = sv * (2 ** (RC - int(cfg["prob_bits"])))
        bq = np.rint(bf / (sct * swo)).astype(np.int64)
        oamax = 0
        for start in range(0, Xint_tr.shape[0], T):
            n = min(T, Xint_tr.shape[0] - start)
            P = _softmax_probs(Qi[start:start + n], Ki[start:start + n], KC, SH_KC,
                               f_bits, cfg["f_interp"], attn_lut, cfg["g_sh"],
                               cfg["prob_bits"], cfg["softmax_round"])
            ctx = P @ Vi[start:start + n]
            Ci = _shift_round(ctx, RC, cfg["ctx_bits"])
            oamax = max(oamax, int(np.abs(Ci @ Woi + bq[None, :]).max()))
        RO = fit_shift(np.array([oamax]), cfg["out_bits"])
        so = sct * swo * (2 ** RO)
        ap.update(RC=int(RC), RO=int(RO), bias=bq,
                  scale_x=float(s_x[k]), scale_o=float(so),
                  scale_q=float(sq), scale_k=float(sk), scale_v=float(sv),
                  scale_ct=float(sct), swq=float(swq), swk=float(swk),
                  swv=float(swv), swo=float(swo))

        M_A = int(rint(so / s_a[k] * (1 << Q_A)))
        # M_A scales as 2^(A-12): it is a constant requant multiplier, so the
        # bound is a multiplier-width bound, not a fixed 20-bit field.
        if not (0 < M_A < (1 << 28)):
            raise ValueError(f"layer {k}: M_A out of range: {M_A}")

        W1f = weights[f"{p}/ffn/layer_with_weights-0/kernel"].astype(np.float64)
        b1f = weights[f"{p}/ffn/layer_with_weights-0/bias"].astype(np.float64)
        w1i, b1i, sw1, sh = quantize_w1(W1f, b1f, sx_ffn1[k], cfg["w1"])
        W2f = weights[f"{p}/ffn/layer_with_weights-1/kernel"].astype(np.float64)
        b2f = weights[f"{p}/ffn/layer_with_weights-1/bias"].astype(np.float64)
        w2i, b2i, sw2, sh2 = quantize_w2(W2f, b2f, s_g[k], cfg["w2"])

        qg = gelu_qg(cfg)
        mg = rint(sh * (1 << (cfg["lf"] + qg))).astype(np.int64)
        mgn = int(rint((1 << QGN) / ((1 << cfg["lf"]) * s_g[k])))

        qb = int(cfg["qb"])
        ma1 = int(rint(s_x[k] / s_r1[k] * (1 << qb)))
        mb1 = int(rint(s_a[k] / s_r1[k] * (1 << qb)))
        eps_q1 = int(rint(EPS / (s_r1[k] ** 2) * (1 << VAF)))
        gamma1 = weights[f"{p}/layernorm1/gamma"].astype(np.float64)
        beta1 = weights[f"{p}/layernorm1/beta"].astype(np.float64)
        g1f = rint(gamma1 * (1 << G)).astype(np.int64)
        b1fk = rint(beta1 / sx_ffn1[k]).astype(np.int64)
        mo1 = int(rint((1 << GO) / ((1 << G) * sx_ffn1[k])))

        mc2 = int(rint(sx_ffn1[k] / s_r2[k] * (1 << qb)))
        md2 = rint(sh2 / s_r2[k] * (1 << qb)).astype(np.int64)
        eps_q2 = int(rint(EPS / (s_r2[k] ** 2) * (1 << VAF)))
        gamma2 = weights[f"{p}/layernorm2/gamma"].astype(np.float64)
        beta2 = weights[f"{p}/layernorm2/beta"].astype(np.float64)
        g2f = rint(gamma2 * (1 << G)).astype(np.int64)
        b2fk = rint(beta2 / s_out[k]).astype(np.int64)
        mo2 = int(rint((1 << GO) / ((1 << G) * s_out[k])))

        layers.append(dict(
            layer=k, M_A=M_A,
            scales=dict(s_x=float(s_x[k]), s_a=float(s_a[k]),
                        sx_ffn1=float(sx_ffn1[k]), s_g=float(s_g[k]),
                        s_r1=float(s_r1[k]), s_r2=float(s_r2[k]),
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

    return dict(schema="p027-precision/1", cfg=cfg,
                isq_frac={st: ln_k({"cfg": cfg}, st) for st in ("ln1", "y")},
                layers=layers,
                lut=[int(v) for v in lut], attn_lut=[int(v) for v in attn_lut],
                maxima={k: [float(v) for v in mx[k]] for k in mx},
                max_x0=float(mx_x0))


def attention_proj_max(X_int, W):
    return int(np.abs(X_int @ W).max())


def _softmax_probs(Qi, Ki, KC, SH_KC, f_bits, f_int, lut, g_sh,
                   prob_bits=15, softmax_round=False):
    S = Qi @ Ki.transpose(0, 2, 1) if Qi.ndim == 3 else Qi @ Ki.T
    mmax = S.max(axis=-1, keepdims=True)
    A = ((S - mmax) * np.int64(KC) + np.int64(1 << (SH_KC - 1))) >> np.int64(SH_KC)
    ka = A >> f_bits
    frac = A - (ka << f_bits)
    if f_int:
        hi = frac >> f_int
        lo = frac - (hi << f_int)
        e0 = (lut[hi] * ((1 << f_int) - lo) + lut[np.minimum(hi + 1, lut.size - 1)] * lo
              + (1 << (f_int - 1))) >> f_int
    else:
        e0 = lut[frac]
    pos = np.clip(-ka, 0, 31)
    if softmax_round:
        e0 = (e0 + (np.int64(1) << np.maximum(pos - 1, 0))) >> pos
        E = np.where(-ka >= 32, np.int64(0), e0)
    else:
        E = np.where(-ka >= 32, np.int64(0), e0 >> pos)
    sumE = E.sum(axis=-1, keepdims=True)
    Rr = ((np.int64(1) << (prob_bits + g_sh)) + (sumE >> 1)) // sumE
    return np.clip((E * Rr) >> g_sh, 0, (1 << prob_bits) - 1)


def quantize_w1(W1f, b1f, sx, bits):
    qmax = (1 << (bits - 1)) - 1
    sw1 = np.maximum(np.max(np.abs(W1f), axis=0) / qmax, 1e-30)
    w1i = np.clip(rint(W1f / sw1[None, :]), -qmax, qmax).astype(np.int64)
    b1i = rint(b1f / (sw1 * sx)).astype(np.int64)
    return w1i, b1i, sw1, sx * sw1


def quantize_w2(W2f, b2f, sg, bits):
    qmax = (1 << (bits - 1)) - 1
    sw2 = np.maximum(np.max(np.abs(W2f), axis=0) / qmax, 1e-30)
    w2i = np.clip(rint(W2f / sw2[None, :]), -qmax, qmax).astype(np.int64)
    b2i = rint(b2f / (sg * sw2)).astype(np.int64)
    return w2i, b2i, sw2, sg * sw2


def ln_k(calib, stage):
    """Inverse-sqrt fraction bits for a LayerNorm stage.

    ``isq = 2^K / sqrt(var)`` is a *stored* integer whose relative precision is
    ``sqrt(var)/2^K``.  ``var`` grows as 4^(A-12), so the frozen K=16 loses
    ~0.8% of the reciprocal per activation bit.  The default rule keeps the
    baseline at A=12 (K=16) and grows K by 2 per extra activation bit, plus
    ``ln_extra`` bits of accuracy headroom (default 8).
    """
    cfg = calib["cfg"]
    if cfg.get("isq_frac"):
        return int(cfg["isq_frac"])
    a = int(cfg["stage_bits"][stage])
    extra = int(cfg.get("ln_extra", 8))
    return 16 + 2 * (a - 12) + (extra if a > 12 else 0)


def ap_of(L):
    ap = dict(L["attn"]["params"])
    ap["lut"] = np.asarray(L["attn"]["lut"], dtype=np.int64)
    for kk in ("wq", "wk", "wv", "wo", "bias"):
        ap[kk] = L["attn"][kk]
    return ap


# --------------------------------------------------------------------------
# integer forward chain
# --------------------------------------------------------------------------
def integer_chain(calib, X0_fp32, want_stages=True):
    cfg = cfg_norm(calib["cfg"])
    sb = cfg["stage_bits"]
    cl = {k: clamp_of(sb[k]) for k in ("x", "a", "r1", "ln1", "g", "r2", "y")}
    cl["y"] = cl["x"]
    layers = calib["layers"]
    s_x0 = layers[0]["scales"]["s_x"]
    X = np.asarray(X0_fp32, dtype=np.float64)[None, :, :]
    x_int = np.clip(rint(X / s_x0), -cl["x"] - 1, cl["x"]).astype(np.int64)

    out = {k: [] for k in ("a", "r1", "out1", "g", "f", "r2", "y")}
    deq = {k: [] for k in ("a", "r1", "g", "f", "r2", "y")}
    sats = {"a": 0, "r1": 0, "g": 0, "r2": 0, "y": 0}
    acc_bits = {"qkv": 0, "score": 0, "ctx": 0, "attn_out": 0,
                "ffn1": 0, "ffn2": 0}
    for L in layers:
        sc = L["scales"]
        ap = ap_of(L)
        o_int, st = attention_batch(x_int, ap, cfg)
        acc_bits["qkv"] = max(acc_bits["qkv"], st["qkv_acc_bits"])
        acc_bits["score"] = max(acc_bits["score"], st["score_acc_bits"])
        acc_bits["ctx"] = max(acc_bits["ctx"], st["ctx_acc_bits"])
        acc_bits["attn_out"] = max(acc_bits["attn_out"], st["out_acc_bits"])
        # the attention-output register is int16 in the frozen RTL; schemes with
        # A>16 widen it to A so the requantised activation is not re-clipped
        # Two headroom bits absorb validation samples above the training max:
        # a_int = O_int * M_A / 2^16 can overshoot 2^(A-1) by the ratio
        # oamax/max_a even when O_int itself is not clipped.
        ab = int(cfg.get("a_clip_bits") or max(16, sb["a"] + 2))
        alim = (1 << (ab - 1)) - 1
        a_int = np.clip((o_int * np.int64(L["M_A"]) + (1 << (Q_A - 1))) >> Q_A,
                        -alim - 1, alim)
        c = L["coeff"]
        r1 = residual_rows(x_int[0], c["ma1"], a_int[0], c["mb1"], cl["r1"],
                           int(cfg["qb"]))
        out1 = layernorm_rows(r1, c["eps_q1"], c["g1f"], c["b1f"], c["mo1"], cl["ln1"],
                              ln_k(calib, "ln1"))
        h = out1 @ L["ffn1"]["w"] + L["ffn1"]["b"][None, :]
        acc_bits["ffn1"] = max(acc_bits["ffn1"], int(np.abs(h).max()).bit_length() + 1)
        g = gelu_rows(h, c["mg"], c["mgn"], np.asarray(calib["lut"], dtype=np.int64),
                      cfg["lf"], cl["g"], cfg["gelu_shift"], gelu_qg(cfg))
        f = g @ L["ffn2"]["w"] + L["ffn2"]["b"][None, :]
        acc_bits["ffn2"] = max(acc_bits["ffn2"], int(np.abs(f).max()).bit_length() + 1)
        r2 = residual_rows(out1, c["ma2"], f, c["md2"], cl["r2"],
                           int(cfg["qb"]))
        y = layernorm_rows(r2, c["eps_q2"], c["g2f"], c["b2f"], c["mo2"], cl["y"],
                           ln_k(calib, "y"))

        s_h2 = L["ffn2"]["sh2"]
        sats["a"] += int(np.sum(np.abs(a_int) >= alim))
        sats["r1"] += int(np.sum(np.abs(r1) >= cl["r1"]))
        sats["g"] += int(np.sum(np.abs(g) >= cl["g"]))
        sats["r2"] += int(np.sum(np.abs(r2) >= cl["r2"]))
        sats["y"] += int(np.sum(np.abs(y) >= cl["y"]))
        if want_stages:
            out["a"].append(a_int[0]); out["r1"].append(r1); out["out1"].append(out1)
            out["g"].append(g); out["f"].append(f); out["r2"].append(r2); out["y"].append(y)
        deq["a"].append(a_int[0] * sc["s_a"])
        deq["r1"].append(r1 * sc["s_r1"])
        deq["g"].append(g * sc["s_g"])
        deq["f"].append(f * s_h2[None, :])
        deq["r2"].append(r2 * sc["s_r2"])
        deq["y"].append(y * sc["s_out"])
        x_int = y[None, :, :]
    return dict(int={k: (np.array(v) if want_stages else None) for k, v in out.items()},
                deq={k: np.array(v) for k, v in deq.items()}, sat=sats,
                acc_bits=acc_bits)


def nrmse(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


# --------------------------------------------------------------------------
# cost proxies
# --------------------------------------------------------------------------
def cost_model(cfg, nlayers=NLAYERS):
    """Analytic storage / multiplier / accumulator proxies for one scheme."""
    cfg = cfg_norm(cfg)
    sb = cfg["stage_bits"]
    a_xy, a_a, a_r1, a_ln1, a_g, a_r2 = (sb["x"], sb["a"], sb["r1"],
                                          sb["ln1"], sb["g"], sb["r2"])
    wa, w1, w2 = cfg["w_attn"], cfg["w1"], cfg["w2"]
    f_bits = cfg["f_table"] + cfg["f_interp"]
    pb = int(cfg["prob_bits"])
    qkv = int(cfg["qkv_bits"])
    qb = int(cfg["qb"])
    qg = gelu_qg(cfg)
    isq = ln_k({"cfg": cfg}, "y")

    # per-layer parameter storage
    wbits = (4 * D * D * wa + D * HID * w1 + HID * D * w2
             + (4 * D + HID + D) * 16            # biases (int16)
             + 4 * D * 16                        # 2 x (gamma,beta) int16
             + 4 * D * 16)                       # misc LN coefficients
    wbytes = wbits / 8.0

    # stored activations per layer (peak is the 99x256 GELU tensor)
    act_elems = dict(x=T * D, a=T * D, r1=T * D, out1=T * D, g=T * HID,
                     r2=T * D, y=T * D)
    a_reg = int(cfg.get("a_clip_bits") or max(16, a_a + 2))
    act_bits = dict(x=a_xy, a=a_reg, r1=a_r1, out1=a_ln1, g=a_g, r2=a_r2, y=a_xy)
    act_total_bits = sum(act_elems[k] * act_bits[k] for k in act_elems)
    act_peak_bits = max(act_elems[k] * act_bits[k] for k in act_elems)

    # MAC counts and bit products (per layer)
    macs = dict(attn_proj=4 * T * D * D, qk=T * T * D, pv=T * T * D,
                ffn1=T * D * HID, ffn2=T * HID * D)
    bitprod = dict(
        attn_proj=macs["attn_proj"] * a_xy * wa,
        qk=macs["qk"] * qkv * qkv,
        pv=macs["pv"] * pb * qkv,
        ffn1=macs["ffn1"] * a_ln1 * w1,
        ffn2=macs["ffn2"] * a_g * w2,
    )
    acc_bits = dict(
        attn_proj=a_xy + wa + int(math.ceil(math.log2(D))),
        qk=2 * qkv + int(math.ceil(math.log2(T))),
        pv=pb + qkv + int(math.ceil(math.log2(T))),
        ffn1=a_ln1 + w1 + int(math.ceil(math.log2(D))),
        ffn2=a_g + w2 + int(math.ceil(math.log2(HID))),
        requant=a_reg + int(cfg["qb"]),        # M_A ~ 2^(A-12), Q_A=16
        residual=qb + max(a_xy, a_a, a_g) + 1,
        layernorm=a_ln1 + 16 + 1 + 16,          # d*gamma, then *isq
    )
    return dict(
        per_layer=dict(
            weight_bytes=wbytes,
            weight_bits=int(wbits),
            activation_bytes=act_total_bits / 8.0,
            activation_peak_bytes=act_peak_bits / 8.0,
            activation_bits_by_stage=act_bits,
            attention_output_register_bits=a_reg,
            macs=macs,
            mac_bit_products=bitprod,
            mac_bit_products_total=sum(bitprod.values()),
            multiplier_bits=dict(attn=a_xy + wa, ffn1=a_ln1 + w1, ffn2=a_g + w2,
                                 qk=2 * qkv, pv=pb + qkv,
                                 requant=a_reg + 16, layernorm=a_ln1 + 16),
            accumulator_bits=acc_bits,
            exp_lut_entries=int(2 ** cfg["f_table"]),
            exp_lut_frac_bits=f_bits,
            exp_mul_frac_bits=cfg["m_lut"],
            exp_prob_bits=pb,
            kc_bits=int(math.ceil(math.log2(max(1, int(cfg["kc_min"]))))) + 1,
            gelu_lut_entries=int((1 << (cfg["lf"] + 4 - cfg["gelu_shift"])) + 1),
            gelu_frac_bits=cfg["lf"],
            gelu_input_frac_bits=qg,
            residual_frac_bits=qb,
            isq_frac_bits=isq,
            isq_register_bits=ISQ_REG_BITS,
        ),
        twelve_layers=dict(
            weight_bytes=wbytes * nlayers,
            weight_kib=wbytes * nlayers / 1024.0,
            activation_bytes=act_total_bits * nlayers / 8.0,
            macs=sum(macs.values()) * nlayers,
            mac_bit_products_total=sum(bitprod.values()) * nlayers,
        ),
    )
