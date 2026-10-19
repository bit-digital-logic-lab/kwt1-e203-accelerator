#!/usr/bin/env python3
"""Level-by-level verification of the assembled encoder RTL.

Inputs (all produced by this directory, no RTL output feeds the model):
  vectors/  : X, attention params, FFN1/FFN2 params, exp_o (attention integer
              model output), meta.json
  build/    : RTL dumps a/r1/g/f/r2/y

Checks:
  1. requant(RTL attention output) == requant(attention integer model output)
     -- this is the a-stage, and it pins the attention RTL to its own model.
  2. composed independent integer model (attention model -> requant -> postnorm
     ref -> FFN1) reproduces every RTL stage a/r1/g/f/r2/y bit for bit.
  3. FP32 error of the dequantised RTL stages against the graduate FP32 golden
     (`reference/kwt1_fp32_golden.py`, item 5), NRMSE per stage; the worst-stage
     NRMSE is gated against NRMSE_THRESHOLD (3%, a fixed spec/acceptance
     constant, not fitted from any sample).

Exit code 0 only when every bit-exact check passes AND the worst-stage NRMSE is
at or below NRMSE_THRESHOLD.  `--negative-inject {bitexact,nrmse}` injects a
controlled fault so that the gate itself can be shown to fire (test only).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent          # graduate/encoder/model
ENC = HERE.parent                               # graduate/encoder
VEC = ENC / "vectors"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ENC / "reference"))

import kwt_postnorm_ref as ref                  # noqa: E402
import kwt1_fp32_golden as golden               # noqa: E402

Q_A = 16

# Acceptance gate: the worst-stage dequantised NRMSE must be <= 3%.  This is a
# fixed spec constant, deliberately NOT derived from any calibration/validation
# sample (see graduate/ds_tasks/encoder_gate/README.md on the split policy).
NRMSE_THRESHOLD = 0.03

# `--negative-inject nrmse` perturbs only the FP32 dequantisation path (stage
# label -> multiplicative factor) so the NRMSE gate is exercised without
# touching the bit-exact integer arrays.
NRMSE_INJECT = {"y  (layer_output)": 1.10}


def read_hex(path, rows, cols, width):
    vals = []
    lines = [ln for ln in Path(path).read_text().splitlines() if ln.strip()]
    if len(lines) != rows:
        raise SystemExit(f"VERIFY_FAIL {path}: {len(lines)} lines, expected {rows}")
    per = width // 4
    sign = 1 << (width - 1)
    full = 1 << width
    for ln in lines:
        if len(ln) != cols * per:
            raise SystemExit(f"VERIFY_FAIL {path}: line width {len(ln)} != {cols*per}")
        row = []
        for i in range(0, len(ln), per):
            v = int(ln[i:i + per], 16)
            row.append(v - full if v >= sign else v)
        vals.append(row)
    return np.array(vals, dtype=np.int64)


def requant(o, m, q):
    return np.clip((o.astype(np.int64) * m + (1 << (q - 1))) >> q, -32768, 32767)


def nrmse(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def nrmse_ok(worst, threshold=NRMSE_THRESHOLD):
    """Gate predicate: fail strictly when worst-stage NRMSE exceeds the ceiling.

    Kept as a standalone function so the boundary can be unit-tested without a
    full Verilator rebuild (see graduate/ds_tasks/encoder_gate/gate_selfcheck.py).
    """
    return float(worst) <= float(threshold)


def cmp_bit(name, rtl, mdl, ok_flag):
    if rtl.shape != mdl.shape:
        print(f"  {name}: SHAPE rtl={rtl.shape} model={mdl.shape}")
        return False
    bad = np.argwhere(rtl != mdl)
    if len(bad):
        r, i = bad[0]
        print(f"  {name}: {len(bad)} mismatches; first row={r} idx={i} "
              f"rtl={rtl[r, i]} model={mdl[r, i]}")
        return False
    print(f"  {name}: {rtl.shape[0]}x{rtl.shape[1]} bit-exact")
    return ok_flag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, default=ENC / "tb/build")
    ap.add_argument("--vectors", type=Path, default=VEC)
    ap.add_argument("--nrmse-threshold", type=float, default=NRMSE_THRESHOLD,
                    help="worst-stage NRMSE ceiling (spec constant; not fitted)")
    ap.add_argument("--negative-inject", choices=("none", "bitexact", "nrmse"),
                    default="none",
                    help="inject a controlled fault to prove the gate fires")
    args = ap.parse_args()
    build = args.build
    meta = json.loads((args.vectors / "meta.json").read_text())
    T, D, HID = meta["T"], meta["D"], meta["HID"]
    M_A = int(meta["M_A"])

    # ---- attention integer model output + composed oracle ----
    o_model = np.loadtxt(args.vectors / "exp_o.txt").astype(np.int64).reshape(T, D)
    a_model = requant(o_model, M_A, Q_A)

    cal = ref.calibrate()
    x_rows = np.asarray(cal["xq"], dtype=np.int64)
    pre = ref.integer_prefix(cal, x_rows, a_model)
    w1 = np.asarray(cal["w1"], dtype=np.int64)
    b1 = np.asarray(cal["b1"], dtype=np.int64)
    h_rows = np.clip(pre["out1"] @ w1 + b1, -(1 << 31), (1 << 31) - 1)
    model = ref.integer_forward(cal, x_rows, a_model, h_rows)

    # ---- RTL dumps ----
    a_rtl = read_hex(build / "a.hex", T, D, 16)
    r1 = read_hex(build / "r1.hex", T, D, 16)
    g = read_hex(build / "g.hex", T, HID, 16)
    f = read_hex(build / "f.hex", T, D, 32)
    r2 = read_hex(build / "r2.hex", T, D, 16)
    y = read_hex(build / "y.hex", T, D, 16)

    neg = args.negative_inject
    if neg == "bitexact":
        y = y.copy()
        y[0, 0] += 1
        print("[neg-inject] bitexact: y[0,0] += 1 (one RTL stage cell corrupted)")

    ok = True
    ok = cmp_bit("a  (requant attn)", a_rtl, a_model, ok)
    ok = cmp_bit("r1 (resid1)", r1, model["r1"], ok)
    ok = cmp_bit("g  (gelu)", g, model["g"], ok)
    ok = cmp_bit("f  (ffn2)", f, model["f"], ok)
    ok = cmp_bit("r2 (resid2)", r2, model["r2"], ok)
    ok = cmp_bit("y  (ln2)", y, model["y"], ok)

    # ---- FP32 error vs the graduate golden ----
    sc = cal["scales"]
    x_fp32 = np.load(ref.ROOT / "software/validation/real_input_chain_20261017/"
                     "sample_00_encoder.npy").astype(np.float64)
    wts = golden.load_weights()
    arrays = {}
    golden.encoder_layer(x_fp32, wts, 0, arrays=arrays)

    s_h2 = np.asarray(sc["s_h2"], dtype=np.float64)
    fp = {
        "a  (attn_projected)": (a_rtl * sc["s_a"], arrays["attn_projected"]),
        "r1 (residual1)":      (r1 * sc["s_r1"], arrays["residual1"]),
        "g  (gelu)":           (g * sc["s_g"], arrays["gelu"]),
        "f  (ffn_out)":        (f * s_h2[None, :], arrays["ffn_out"]),
        "r2 (residual2)":      (r2 * sc["s_r2"], arrays["residual2"]),
        "y  (layer_output)":   (y * sc["s_out"], arrays["layer_output"]),
    }
    if neg == "nrmse":
        print(f"[neg-inject] nrmse: {NRMSE_INJECT} applied to the dequantised RTL path")

    print("FP32 NRMSE vs graduate golden (dequantised RTL vs FP32):")
    worst = 0.0
    worst_name = ""
    nrmse_fail = {}
    for name, (approx, exact) in fp.items():
        if neg == "nrmse":
            approx = approx * NRMSE_INJECT.get(name, 1.0)
        e = nrmse(approx, exact)
        if e > worst:
            worst, worst_name = e, name
        if not nrmse_ok(e, args.nrmse_threshold):
            nrmse_fail[name] = e
        flag = "  <-- OVER THRESHOLD" if name in nrmse_fail else ""
        print(f"  {name}: NRMSE {e*100:.4f}%  max_abs {np.max(np.abs(approx-exact)):.3e}{flag}")
    print(f"  worst-stage NRMSE {worst*100:.4f}% [{worst_name.strip()}] "
          f"(threshold {args.nrmse_threshold*100:.2f}%)")

    reasons = []
    if not ok:
        reasons.append("bit_exact")
    if not nrmse_ok(worst, args.nrmse_threshold):
        reasons.append("nrmse")
    print(f"GATE bit_exact={'PASS' if ok else 'FAIL'} "
          f"nrmse={'PASS' if not nrmse_fail else 'FAIL'} "
          f"rows={T} stages={len(fp)} worst={worst*100:.4f}% "
          f"threshold={args.nrmse_threshold*100:.2f}%")

    if reasons:
        print("VERIFY_FAIL reason=" + ",".join(reasons))
        raise SystemExit(1)
    print("VERIFY_PASS")


if __name__ == "__main__":
    main()
