#!/usr/bin/env python3
"""Independent integer and FP32 reference for the PostNorm FFN/GELU/LayerNorm RTL.

The integer datapath below is written from the specification text, not from the
RTL source: same operations, same rounding, same saturations, but a separate
implementation.  `verify.py` then requires the RTL dumps to match it bit for bit.

Rounding
  mul_sr(a,m,q) = (a*m + 2^(q-1)) >> q          (arithmetic shift == floor)
Saturation
  activations are clamped to +/-CLAMP_A in every block that produces one.
"""

import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# [encoder-assembly copy] only the repository-root resolution is adjusted so the
# file works at the shallower graduate/encoder/model/ path; numeric code below is
# byte-identical to the grad-nonlinear source (see PROVENANCE.md).
def _find_root(p):
    for a in [p, *p.parents]:
        if (a / "software" / "weights" / "kwt1_fp32.npz").exists():
            return a
    raise FileNotFoundError("repository root with software/weights/kwt1_fp32.npz not found")
ROOT = _find_root(HERE)
WEIGHTS = ROOT / "software/weights/kwt1_fp32.npz"
SAMPLE = ROOT / "software/validation/real_input_chain_20261017/sample_00_encoder.npy"
FROZEN = ROOT / "software/results/ffn1_a12w10_static_sym.npz"

D = 64
HID = 256
LAYER = 0
A_BITS = 12
W1_BITS = 10
W2_BITS = 10
CLAMP_A = (1 << (A_BITS - 1)) - 1
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
LUT_N = 2 * 8 * 16 + 1
ISQ_MAX = (1 << (K + 8)) - 1
EPS = 1e-6


def rint(x):
    """Round half to even, implemented without numpy."""
    f = math.floor(x)
    d = x - f
    if d > 0.5:
        return f + 1
    if d < 0.5:
        return f
    return f if f % 2 == 0 else f + 1


def rint_arr(values):
    return np.array([rint(float(v)) for v in np.asarray(values).ravel()]).reshape(np.shape(values))


def mul_sr(a, m, q):
    """Round-half-up signed multiply-shift; mirrors the RTL helper exactly."""
    p = int(a) * int(m)
    if abs(p) >= (1 << 62):
        raise OverflowError(f"mul_sr product overflow: {a}*{m}")
    return (p + (1 << (q - 1))) >> q


def isqrt_hw(v):
    """Binary restoring integer square root, same loop as the RTL function."""
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
    if res * res > int(v) or (res + 1) * (res + 1) <= int(v):
        raise AssertionError("isqrt_hw disagrees with floor(sqrt)")
    assert res == math.isqrt(int(v)), (res, math.isqrt(int(v)))
    return res


def _sat(v, clamp):
    if v > clamp:
        return clamp, True
    if v < -clamp:
        return -clamp, True
    return v, False


def residual_add(a_row, ma, b_row, mb, qb, clamp=CLAMP_A):
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
    if sq == 0:
        isq = ISQ_MAX
    else:
        isq = min((num + (sq >> 1)) // sq, ISQ_MAX)
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
            raise OverflowError("FFN2 accumulator exceeds the 48-bit RTL width")
        out.append(acc + int(b2[k]))
    return out


# --------------------------------------------------------------------------
# calibration (recomputed independently from the frozen checkpoint + sample)
# --------------------------------------------------------------------------

def gelu_fp32(x):
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def layer_norm_fp32(x, gamma, beta):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + EPS) * gamma + beta


def fp32_reference():
    w = {k: v.astype(np.float64) for k, v in np.load(WEIGHTS, allow_pickle=False).items()}
    p = f"layer_with_weights-0/enc_layers/{LAYER}"
    x = np.load(SAMPLE, allow_pickle=False).astype(np.float64)
    a = p + "/att"
    q = x @ w[a + "/query_dense/kernel"]
    k = x @ w[a + "/key_dense/kernel"]
    v = x @ w[a + "/value_dense/kernel"]
    score = (q @ k.T) / math.sqrt(float(D))
    e = np.exp(score - score.max(axis=-1, keepdims=True))
    attn = (e / e.sum(axis=-1, keepdims=True)) @ v
    attn_out = attn @ w[a + "/combine_heads/kernel"] + w[a + "/combine_heads/bias"]
    r1 = x + attn_out
    out1 = layer_norm_fp32(r1, w[p + "/layernorm1/gamma"], w[p + "/layernorm1/beta"])
    h = out1 @ w[p + "/ffn/layer_with_weights-0/kernel"] + w[p + "/ffn/layer_with_weights-0/bias"]
    g = gelu_fp32(h)
    f = g @ w[p + "/ffn/layer_with_weights-1/kernel"] + w[p + "/ffn/layer_with_weights-1/bias"]
    r2 = out1 + f
    y = layer_norm_fp32(r2, w[p + "/layernorm2/gamma"], w[p + "/layernorm2/beta"])
    return w, x, dict(attn_out=attn_out, r1=r1, out1=out1, h=h, g=g, f=f, r2=r2, y=y)


def calibrate():
    """Recompute every frozen coefficient from the checkpoint and the sample."""
    w, x0, ref = fp32_reference()
    p = f"layer_with_weights-0/enc_layers/{LAYER}"
    with np.load(FROZEN, allow_pickle=False) as z:
        w1_int = z["layer01_weight"].astype(np.int64)
        b1_int = z["layer01_bias"].astype(np.int64)
        sx = float(z["layer01_activation_scale"])
        sw1 = z["layer01_weight_scales"].astype(np.float64)

    # stimulus quantization scales (per-tensor static maximum, A12)
    s_x = max(float(np.max(np.abs(x0))) / CLAMP_A, 1e-30)
    s_a = max(float(np.max(np.abs(ref["attn_out"]))) / CLAMP_A, 1e-30)
    xq = np.clip(rint_arr(x0 / s_x), -CLAMP_A, CLAMP_A).astype(np.int64)
    aq = np.clip(rint_arr(ref["attn_out"] / s_a), -CLAMP_A, CLAMP_A).astype(np.int64)

    lut = np.array([rint(gelu_fp32((i - 128) / 16.0) * (1 << LF)) for i in range(LUT_N)],
                   dtype=np.int64)
    sh = sx * sw1
    mg = np.array([rint(v * (1 << (LF + QG))) for v in sh], dtype=np.int64)
    sg = max(float(np.max(np.abs(ref["g"]))) / CLAMP_A, 1e-30)
    mgn = rint((1 << QGN) / ((1 << LF) * sg))

    w2f = w[p + "/ffn/layer_with_weights-1/kernel"]
    b2f = w[p + "/ffn/layer_with_weights-1/bias"]
    sw2 = np.maximum(np.max(np.abs(w2f), axis=0) / QMAX_W2, 1e-30)
    w2_int = np.clip(rint_arr(w2f / sw2[None, :]), -QMAX_W2, QMAX_W2).astype(np.int64)
    b2_int = np.array([rint(float(b) / (sg * s)) for b, s in zip(b2f, sw2)], dtype=np.int64)
    sh2 = sg * sw2

    sr1 = max(float(np.max(np.abs(ref["r1"]))) / CLAMP_A, 1e-30)
    ma1 = rint(s_x / sr1 * (1 << QB))
    mb1 = rint(s_a / sr1 * (1 << QB))
    epsq1 = rint(EPS / (sr1 * sr1) * (1 << VAF))
    g1f = rint_arr(w[p + "/layernorm1/gamma"] * (1 << G)).astype(np.int64)
    b1f = rint_arr(w[p + "/layernorm1/beta"] / sx).astype(np.int64)
    mo1 = rint((1 << GO) / ((1 << G) * sx))

    sr2 = max(float(np.max(np.abs(ref["r2"]))) / CLAMP_A, 1e-30)
    mc2 = rint(sx / sr2 * (1 << QB))
    md2 = np.array([rint(v / sr2 * (1 << QB)) for v in sh2], dtype=np.int64)
    epsq2 = rint(EPS / (sr2 * sr2) * (1 << VAF))
    s_out = max(float(np.max(np.abs(ref["y"]))) / CLAMP_A, 1e-30)
    g2f = rint_arr(w[p + "/layernorm2/gamma"] * (1 << G)).astype(np.int64)
    b2f = rint_arr(w[p + "/layernorm2/beta"] / s_out).astype(np.int64)
    mo2 = rint((1 << GO) / ((1 << G) * s_out))

    return dict(
        scales=dict(s_x=s_x, s_a=s_a, sx=sx, s_g=sg, s_r1=sr1, s_r2=sr2, s_out=s_out,
                    s_h=sh, s_h2=sh2),
        scalars=dict(ma1=ma1, mb1=mb1, mgn=mgn, mo1=mo1, mc2=mc2, mo2=mo2,
                     eps_q1=epsq1, eps_q2=epsq2),
        arrays=dict(mg=mg, g1f=g1f, b1f=b1f, g2f=g2f, b2f=b2f, md2=md2, lut=lut),
        w1=w1_int, b1=b1_int, w2=w2_int, b2=b2_int,
        xq=xq, aq=aq, ref=ref)


def integer_prefix(cal, x_rows, a_rows):
    """Residual 1 + LayerNorm 1: the FFN1 input, in integers."""
    sc, ar = cal["scalars"], cal["arrays"]
    r1, out1 = [], []
    sat_r1 = sat_o1 = 0
    for r in range(len(x_rows)):
        rr, s1 = residual_add(x_rows[r], sc["ma1"], a_rows[r], [sc["mb1"]] * D, QB)
        oo, so = layernorm(rr, sc["eps_q1"], ar["g1f"], ar["b1f"], sc["mo1"])
        r1.append(rr)
        out1.append(oo)
        sat_r1 += s1
        sat_o1 += so
    return dict(r1=np.array(r1, dtype=np.int64), out1=np.array(out1, dtype=np.int64),
                sat_r1=sat_r1, sat_o1=sat_o1)


def integer_forward(cal, x_rows, a_rows, h_rows, sats=None):
    """Full PostNorm fragment in integers; returns per-stage arrays."""
    sc, ar = cal["scalars"], cal["arrays"]
    pre = integer_prefix(cal, x_rows, a_rows)
    out = dict(r1=[], out1=[], g=[], f=[], r2=[], y=[])
    sat_r2 = sat_y = 0
    for r in range(len(x_rows)):
        r1 = pre["r1"][r]
        out1 = pre["out1"][r]
        g = [gelu_step(h_rows[r][j], ar["mg"][j], sc["mgn"], ar["lut"]) for j in range(HID)]
        f = ffn2(g, cal["w2"], cal["b2"])
        r2, s2 = residual_add(out1, sc["mc2"], f, ar["md2"], QB)
        y, sy = layernorm(r2, sc["eps_q2"], ar["g2f"], ar["b2f"], sc["mo2"])
        out["r1"].append(r1)
        out["out1"].append(out1)
        out["g"].append(g)
        out["f"].append(f)
        out["r2"].append(r2)
        out["y"].append(y)
        sat_r2 += s2
        sat_y += sy
    if sats is not None:
        sats.update(sat_r1=pre["sat_r1"], sat_o1=pre["sat_o1"], sat_r2=sat_r2, sat_y=sat_y)
    return {k: np.array(v, dtype=np.int64) for k, v in out.items()}
