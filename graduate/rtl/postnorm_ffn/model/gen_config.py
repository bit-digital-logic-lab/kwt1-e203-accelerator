#!/usr/bin/env python3
"""Calibrate the PostNorm FFN/GELU/residual/LayerNorm prototype and dump vectors.

Reads the frozen KWT-1 FP32 checkpoint (`software/weights/kwt1_fp32.npz`) and one
real encoder input (`software/validation/real_input_chain_20261017/sample_00_encoder.npy`),
reproduces the official *PostNorm* single-layer order of `software/run.py` in NumPy,
then freezes integer coefficients and test vectors for the RTL prototype.

This script deliberately does NOT implement the integer datapath.  The integer
outputs are produced only by `kwt_postnorm_ref.py` (the independent reference
model) and by the RTL testbench.  This file owns calibration + stimulus only.

Outputs (all inside this directory):
  rtl/kwt_postnorm_cfg.svh     coefficients + GELU LUT as SystemVerilog localparams
  rtl/kwt_ffn2_w2.hex          INT16 FFN2 weights, $readmemh order [k][j]
  model/vectors/*.hex          x / a / h stimulus, one row per line
  model/vectors/meta.json      coefficients actually emitted (cross-checked by verify.py)
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]  # worktree root: graduate/rtl/postnorm_ffn/model -> root
WEIGHTS = ROOT / "software/weights/kwt1_fp32.npz"
SAMPLE = ROOT / "software/validation/real_input_chain_20261017/sample_00_encoder.npy"
FROZEN = ROOT / "software/results/ffn1_a12w10_static_sym.npz"

D = 64
H = 256
LAYER = 0                 # enc_layers/0
A_BITS = 12
W1_BITS = 10
W2_BITS = 10
QMAX_A = (1 << (A_BITS - 1)) - 1        # 2047
QMAX_W1 = (1 << (W1_BITS - 1)) - 1      # 1023
QMAX_W2 = (1 << (W2_BITS - 1)) - 1      # 1023
CLAMP_A = QMAX_A                        # activation container effective range

QB = 16      # residual scale-coefficient fractional bits
G = 16       # LayerNorm gamma fractional bits
GO = 16      # LayerNorm affine/output fractional bits
K = 16       # LayerNorm 1/sqrt fractional bits
VAF = 16     # variance fractional bits
QG = 28      # GELU input rescale fractional bits
QGN = 16     # GELU output rescale fractional bits
LF = 8       # nonlinear-domain fractional bits (Q8)
LUT_STEP = 1 << (LF - 4)                # LUT granularity: 1/16 -> 16 Q8 units
LUT_N = 2 * 8 * 16 + 1                  # real range [-8, 8), 257 nodes
ISQ_MAX = (1 << (K + 8)) - 1

EPS = 1e-6


def rint(x):
    """Round half to even (matches the project's A12W10 calibration)."""
    return np.rint(x)


def gelu(x):
    """Exact (erf) GELU, matching `F.gelu(..., approximate='none')` in run.py."""
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def layer_norm(x, gamma, beta):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + EPS) * gamma + beta


def attention(x, w):
    p = f"layer_with_weights-0/enc_layers/{LAYER}/att"
    q = x @ w[p + "/query_dense/kernel"]
    k = x @ w[p + "/key_dense/kernel"]
    v = x @ w[p + "/value_dense/kernel"]
    score = (q @ k.T) / math.sqrt(float(D))          # heads=1, d_k=64 (see flags.json)
    attn = np.exp(score - score.max(axis=-1, keepdims=True))
    attn = attn / attn.sum(axis=-1, keepdims=True)
    return attn @ v @ w[p + "/combine_heads/kernel"] + w[p + "/combine_heads/bias"]


def fp32_reference(x, w):
    """Official PostNorm single-layer order (software/run.py, prenorm=False)."""
    p = f"layer_with_weights-0/enc_layers/{LAYER}"
    attn_out = attention(x, w)
    r1 = x + attn_out
    out1 = layer_norm(r1, w[p + "/layernorm1/gamma"], w[p + "/layernorm1/beta"])
    h = out1 @ w[p + "/ffn/layer_with_weights-0/kernel"] + w[p + "/ffn/layer_with_weights-0/bias"]
    g = gelu(h)
    f = g @ w[p + "/ffn/layer_with_weights-1/kernel"] + w[p + "/ffn/layer_with_weights-1/bias"]
    r2 = out1 + f
    y = layer_norm(r2, w[p + "/layernorm2/gamma"], w[p + "/layernorm2/beta"])
    return dict(attn_out=attn_out, r1=r1, out1=out1, h=h, g=g, f=f, r2=r2, y=y)


def quantize_activations(real, clamp=CLAMP_A):
    peak = float(np.max(np.abs(real)))
    scale = max(peak / clamp, 1e-30)
    q = np.clip(rint(real / scale), -clamp, clamp).astype(np.int64)
    return q, scale


def hex_line(values, width):
    mask = (1 << width) - 1
    return "".join(f"{int(v) & mask:0{width // 4}x}" for v in values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE)
    args = ap.parse_args()
    out = args.out
    (out / "vectors").mkdir(parents=True, exist_ok=True)
    (out.parent / "rtl").mkdir(parents=True, exist_ok=True)

    w = {k: v.astype(np.float64) for k, v in np.load(WEIGHTS, allow_pickle=False).items()}
    p = f"layer_with_weights-0/enc_layers/{LAYER}"
    x0 = np.load(SAMPLE, allow_pickle=False).astype(np.float64)
    if x0.shape != (99, D):
        raise ValueError(f"expected encoder input (99,{D}), got {x0.shape}")

    ref = fp32_reference(x0, w)

    # ---- frozen A12W10 FFN1 pack (must agree with a local re-quantization) ----
    with np.load(FROZEN, allow_pickle=False) as z:
        w1_int = z["layer01_weight"].astype(np.int64)          # (64,256) [k][n]
        b1_int = z["layer01_bias"].astype(np.int64)            # (256,)
        sx = float(z["layer01_activation_scale"])              # FFN1 activation scale
        sw1 = z["layer01_weight_scales"].astype(np.float64)    # (256,)
    w1_fp32 = w[p + "/ffn/layer_with_weights-0/kernel"]
    b1_fp32 = w[p + "/ffn/layer_with_weights-0/bias"]
    if w1_int.shape != (D, H):
        raise ValueError(f"unexpected frozen FFN1 weight shape {w1_int.shape}")
    re_w1 = np.clip(rint(w1_fp32 / sw1[None, :]), -QMAX_W1, QMAX_W1).astype(np.int64)
    if not np.array_equal(re_w1, w1_int):
        raise ValueError("frozen FFN1 weights disagree with a local re-quantization")
    re_b1 = rint(b1_fp32 / (sw1 * sx)).astype(np.int64)
    if not np.array_equal(re_b1, b1_int):
        raise ValueError("frozen FFN1 bias disagrees with a local re-quantization")

    # ---- stimulus: 99 real rows + extreme/variance rows ----
    # The FFN1 input is the LayerNorm1 output, so the residual/LN1 stage must be
    # calibrated before the hidden vectors can be produced.  Doing it with the
    # reference model is exactly what the hardware does: LN1 -> FFN1 -> this block.
    import kwt_postnorm_ref as refmodel
    xq, s_x = quantize_activations(x0)
    aq, s_a = quantize_activations(ref["attn_out"])

    sh = sx * sw1                                             # (256,) real scale of h
    mg = rint(sh * (1 << (LF + QG))).astype(np.int64)
    if np.any(np.abs(mg) > (1 << 31) - 1):
        raise OverflowError("GELU input coefficient exceeds INT32")
    sg = max(float(np.max(np.abs(ref["g"]))) / QMAX_A, 1e-30)  # FFN2 activation scale
    mgn = int(rint((1 << QGN) / ((1 << LF) * sg)))

    sr1 = max(float(np.max(np.abs(ref["r1"]))) / CLAMP_A, 1e-30)
    ma1 = int(rint(s_x / sr1 * (1 << QB)))
    mb1 = int(rint(s_a / sr1 * (1 << QB)))
    epsq1 = int(rint(EPS / (sr1 * sr1) * (1 << VAF)))
    g1f = rint(w[p + "/layernorm1/gamma"] * (1 << G)).astype(np.int64)
    b1f = rint(w[p + "/layernorm1/beta"] / sx).astype(np.int64)
    mo1 = int(rint((1 << GO) / ((1 << G) * sx)))

    prefix_cal = dict(scalars=dict(ma1=ma1, mb1=mb1, eps_q1=epsq1, mo1=mo1),
                      arrays=dict(g1f=g1f, b1f=b1f))
    prefix = refmodel.integer_prefix(prefix_cal, xq, aq)
    out1_int = prefix["out1"]

    x_rows, a_rows, h_rows, kinds = [], [], [], []
    for i in range(99):
        x_rows.append(xq[i]); a_rows.append(aq[i])
        h_rows.append(np.clip(out1_int[i] @ w1_int + b1_int, -(1 << 31), (1 << 31) - 1))
        kinds.append("real")

    def add(kind, xv, av, hv):
        x_rows.append(np.asarray(xv, dtype=np.int64))
        a_rows.append(np.asarray(av, dtype=np.int64))
        h_rows.append(np.asarray(hv, dtype=np.int64))
        kinds.append(kind)

    ones = np.ones(D, dtype=np.int64)
    add("x_max_pos", QMAX_A * ones, QMAX_A * ones, np.full(H, QMAX_A))
    add("x_max_neg", -QMAX_A * ones, -QMAX_A * ones, np.full(H, -QMAX_A))
    add("x_mixed", QMAX_A * ones, -QMAX_A * ones, np.full(H, 0))
    add("var_zero", 7 * ones, -3 * ones, np.full(H, QMAX_A))
    add("all_zero", np.zeros(D, dtype=np.int64), np.zeros(D, dtype=np.int64), np.zeros(H, dtype=np.int64))
    add("h_int32_max", QMAX_A * ones, np.zeros(D, dtype=np.int64),
        np.full(H, (1 << 31) - 1))
    add("h_int32_min", -QMAX_A * ones, np.zeros(D, dtype=np.int64),
        np.full(H, -(1 << 31)))
    alt = np.array([QMAX_A if i % 2 == 0 else -QMAX_A for i in range(D)], dtype=np.int64)
    add("x_alt", alt, -alt, np.full(H, QMAX_A // 2))
    add("x_sat16", np.full(D, 32767), np.full(D, 32767), np.full(H, 32767))
    # Near-zero but non-zero variance: one lane differs by 24 LSBs from a
    # constant row. After the residual scale alignment this leaves r1 with
    # var_q ~ 7 (checked by `verify.py`), far below the eps term (eps_q1 ~ 3e4)
    # and the full-scale variance (~4e6), yet not the sq == 0 special case of
    # the exact-zero `var_zero` row.
    tiny = np.full(D, 1000, dtype=np.int64)
    tiny[0] = 1024
    add("var_tiny", tiny, np.zeros(D, dtype=np.int64), np.zeros(H, dtype=np.int64))

    # ---- GELU LUT (independent of the datapath) ----
    lut = []
    for i in range(LUT_N):
        real = (i - 8 * 16) / 16.0
        lut.append(int(rint(gelu(real) * (1 << LF))))
    lut = np.asarray(lut, dtype=np.int64)

    # ---- FFN1 hidden scale, GELU coefficients ----
    w2_fp32 = w[p + "/ffn/layer_with_weights-1/kernel"]        # (256,64)
    b2_fp32 = w[p + "/ffn/layer_with_weights-1/bias"]          # (64,)
    sw2 = np.maximum(np.max(np.abs(w2_fp32), axis=0) / QMAX_W2, 1e-30)   # (64,)
    w2_int = np.clip(rint(w2_fp32 / sw2[None, :]), -QMAX_W2, QMAX_W2).astype(np.int64)
    b2_int = rint(b2_fp32 / (sg * sw2)).astype(np.int64)
    if np.any(np.abs(b2_int) > (1 << 31) - 1):
        raise OverflowError("FFN2 bias exceeds INT32")
    sh2 = sg * sw2                                             # (64,) real scale of f

    # ---- residual1 / LayerNorm1 coefficients already computed above ----

    # ---- residual2 / LayerNorm2 ----
    sr2 = max(float(np.max(np.abs(ref["r2"]))) / CLAMP_A, 1e-30)
    mc2 = int(rint(sx / sr2 * (1 << QB)))
    md2 = rint(sh2 / sr2 * (1 << QB)).astype(np.int64)
    epsq2 = int(rint(EPS / (sr2 * sr2) * (1 << VAF)))
    s_out = max(float(np.max(np.abs(ref["y"]))) / CLAMP_A, 1e-30)
    g2f = rint(w[p + "/layernorm2/gamma"] * (1 << G)).astype(np.int64)
    b2f = rint(w[p + "/layernorm2/beta"] / s_out).astype(np.int64)
    mo2 = int(rint((1 << GO) / ((1 << G) * s_out)))

    for name, coeff in [("mg", mg), ("g1f", g1f), ("b1f", b1f), ("g2f", g2f),
                        ("b2f", b2f), ("md2", md2)]:
        if np.any(np.abs(coeff) > (1 << 31) - 1):
            raise OverflowError(f"coefficient {name} exceeds INT32")
    for name, value in [("ma1", ma1), ("mb1", mb1), ("mc2", mc2), ("mgn", mgn),
                        ("mo1", mo1), ("mo2", mo2), ("epsq1", epsq1), ("epsq2", epsq2)]:
        if abs(value) > (1 << 31) - 1:
            raise OverflowError(f"coefficient {name}={value} exceeds INT32")

    h_rows = [np.clip(r, -(1 << 31), (1 << 31) - 1) for r in h_rows]

    # ---- emit SystemVerilog configuration ----
    # Icarus 11 does not accept unpacked-array parameters/localparams, so every
    # table is emitted as an indexed function (also a natural ROM for synthesis).
    def func(name, values, width):
        fmt = {64: "64'sd", 32: "32'sd", 16: "16'sd"}[width]
        lines = [f"function automatic logic signed [{width - 1}:0] {name}(input int unsigned i);\n",
                 "    case (i)\n"]
        for i, v in enumerate(values):
            lit = f"-{fmt}{abs(int(v))}" if int(v) < 0 else f"{fmt}{int(v)}"
            lines.append(f"        {len(values)}'d{i}: {name} = {lit};\n")
        lines.append(f"        default: {name} = {width}'sd0;\n")
        lines.append("    endcase\nendfunction\n")
        return "".join(lines)

    cfg = []
    cfg.append("// Generated by model/gen_config.py - do not edit by hand.\n")
    cfg.append("// KWT-1 PostNorm FFN/GELU/residual/LayerNorm prototype coefficients (layer 1).\n")
    cfg.append("// No include guard: include this file once per module scope.\n")
    for k, v in [("D", D), ("HID", H), ("CLAMP_A", CLAMP_A), ("QB", QB), ("GAMMA_FRAC", G),
                 ("GOUT_FRAC", GO), ("ISQ_FRAC", K), ("VAR_FRAC", VAF), ("QG", QG),
                 ("QGN", QGN), ("GELU_FRAC", LF), ("LUT_STEP", LUT_STEP),
                 ("LUT_N", LUT_N), ("ISQ_MAX", ISQ_MAX)]:
        cfg.append(f"localparam integer {k} = {v};\n")
    cfg.append(f"localparam logic signed [47:0] EPS_Q1 = 48'sd{epsq1};\n")
    cfg.append(f"localparam logic signed [47:0] EPS_Q2 = 48'sd{epsq2};\n")
    cfg.append(f"localparam logic signed [31:0] MA1 = 32'sd{ma1};\n")
    cfg.append(f"localparam logic signed [31:0] MB1 = 32'sd{mb1};\n")
    cfg.append(f"localparam logic signed [31:0] MGN = 32'sd{mgn};\n")
    cfg.append(f"localparam logic signed [31:0] MO1 = 32'sd{mo1};\n")
    cfg.append(f"localparam logic signed [31:0] MC2 = 32'sd{mc2};\n")
    cfg.append(f"localparam logic signed [31:0] MO2 = 32'sd{mo2};\n")
    cfg.append(func("MG_F", mg, 32))
    cfg.append(func("G1F_F", g1f, 32))
    cfg.append(func("B1F_F", b1f, 32))
    cfg.append(func("G2F_F", g2f, 32))
    cfg.append(func("B2F_F", b2f, 32))
    cfg.append(func("MD2_F", md2, 32))
    cfg.append(func("GELU_LUT_F", lut, 16))
    (out.parent / "rtl" / "kwt_postnorm_cfg.svh").write_text("".join(cfg))

    vec = out / "vectors"

    def dump(path, matrix, width):
        # one token per line: Icarus' $readmemh is line oriented for flat arrays
        with path.open("w") as fh:
            for row in matrix:
                for v in row:
                    fh.write(hex_line([v], width) + "\n")

    dump(vec / "x.hex", x_rows, 16)
    dump(vec / "a.hex", a_rows, 16)
    dump(vec / "h.hex", h_rows, 32)
    dump(vec / "w2.hex", [[w2_int[j, k] for j in range(H)] for k in range(D)], 16)
    dump(vec / "b2.hex", [b2_int], 32)
    np.savez(vec / "stimulus.npz",
             x=np.asarray(x_rows, dtype=np.int64), a=np.asarray(a_rows, dtype=np.int64),
             h=np.asarray(h_rows, dtype=np.int64), w1=w1_int, b1=b1_int,
             w2=w2_int, b2=b2_int, lut=lut, kinds=np.asarray(kinds))

    meta = {
        "layer": LAYER + 1, "rows": len(x_rows), "d_model": D, "mlp_dim": H,
        "scheme": {"activation_bits": A_BITS, "ffn1_weight_bits": W1_BITS,
                   "ffn2_weight_bits": W2_BITS},
        "scales": {"s_x": s_x, "s_a": s_a, "sx_ffn1_input_frozen": sx, "s_g": sg,
                   "s_r1": sr1, "s_r2": sr2, "s_out": s_out,
                   "s_h": sh.tolist(), "s_h2": sh2.tolist()},
        "coefficients": {"ma1": ma1, "mb1": mb1, "mgn": mgn, "mo1": mo1, "mc2": mc2,
                         "mo2": mo2, "eps_q1": epsq1, "eps_q2": epsq2,
                         "mg": mg.tolist(), "g1f": g1f.tolist(), "b1f": b1f.tolist(),
                         "g2f": g2f.tolist(), "b2f": b2f.tolist(), "md2": md2.tolist(),
                         "lut": lut.tolist()},
        "frac_bits": {"QB": QB, "G": G, "GO": GO, "K": K, "VAF": VAF, "QG": QG,
                      "QGN": QGN, "LF": LF},
        "rounding": "coefficients np.rint (ties-to-even); datapath round-half-up arithmetic shift",
        "kinds": kinds,
        "note": "FP32 reference is a NumPy re-implementation of software/run.py (torch absent).",
    }
    (vec / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"PASS gen_config rows={len(x_rows)} sx={sx:.6g} sg={sg:.6g} sr1={sr1:.6g} "
          f"sr2={sr2:.6g} s_out={s_out:.6g}")


if __name__ == "__main__":
    main()
