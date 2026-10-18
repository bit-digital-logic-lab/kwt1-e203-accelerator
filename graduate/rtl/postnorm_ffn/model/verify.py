#!/usr/bin/env python3
"""Bit-exact verification of the PostNorm FFN/GELU/LayerNorm RTL prototype.

Checks, in order:
  1. the coefficients recomputed here match the ones baked into the RTL config;
  2. the FFN1 hidden vector fed to the RTL matches this model's own FFN1 integer
     evaluation (the vector generator and the reference are separate code);
  3. every RTL snapshot (r1, out1, g, f, r2, y) matches the independent integer
     reference bit for bit, on all 109 rows;
  4. FP32 error of the integer result against the official PostNorm order
     (reported separately, real rows only).

Exit code 0 only when every bit-exact check passes.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

import kwt_postnorm_ref as ref
from kwt_postnorm_ref import D, HID, integer_forward, calibrate

DEFAULT = Path(__file__).resolve().parent


def le_hex(values, width):
    mask = (1 << width) - 1
    return "".join(f"{int(v) & mask:0{width // 4}x}" for v in values)


def read_hex_i(path, count, width):
    lines = [ln for ln in Path(path).read_text().splitlines() if ln.strip()]
    if len(lines) != count:
        raise SystemExit(f"VERIFY_FAIL {path}: {len(lines)} lines, expected {count}")
    out = []
    sign = 1 << (width - 1)
    full = 1 << width
    for ln in lines:
        row = []
        for i in range(0, len(ln), width // 4):
            v = int(ln[i:i + width // 4], 16)
            row.append(v - full if v >= sign else v)
        out.append(row)
    return np.array(out, dtype=np.int64)


def compare(name, rtl, model, rows):
    if rtl.shape != model.shape:
        print(f"  {name}: SHAPE rtl={rtl.shape} model={model.shape}")
        return False
    bad = np.argwhere(rtl != model)
    if len(bad):
        r, i = bad[0]
        print(f"  {name}: {len(bad)} mismatched values; first row={r} idx={i} "
              f"rtl={rtl[r, i]} model={model[r, i]}")
        return False
    print(f"  {name}: {rtl.shape[0]} rows x {rtl.shape[1]} lanes bit-exact")
    return True


def nrmse(approx, exact):
    approx = np.asarray(approx, dtype=np.float64)
    exact = np.asarray(exact, dtype=np.float64)
    return float(np.linalg.norm(approx - exact) / max(np.linalg.norm(exact), 1e-300))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, default=DEFAULT.parent / "build")
    ap.add_argument("--vectors", type=Path, default=DEFAULT / "vectors")
    ap.add_argument("--meta", type=Path, default=None)
    args = ap.parse_args()
    build = args.build
    meta_path = args.meta or (args.vectors / "meta.json")
    meta = json.loads(Path(meta_path).read_text())
    stim = np.load(args.vectors / "stimulus.npz")
    rows = int(meta["rows"])
    real = int(np.sum(np.asarray(stim["kinds"]) == "real"))

    print(f"runtime_ref rows={rows} real_rows={real} layer={meta['layer']}")

    cal = calibrate()

    # ---- 1. coefficients: recomputed here vs baked into the RTL ----
    coef_ok = True
    for key, value in cal["scalars"].items():
        got = meta["coefficients"][key]
        if int(got) != int(value):
            print(f"  coefficient {key}: RTL config={got} model={value}")
            coef_ok = False
    for key, values in cal["arrays"].items():
        got = np.array(meta["coefficients"][key], dtype=np.int64)
        if got.shape != values.shape or not np.array_equal(got, values):
            print(f"  array coefficient {key}: RTL config and model differ")
            coef_ok = False
    print(f"  coefficients: {'match' if coef_ok else 'MISMATCH'}")

    # ---- 2. FFN1 hidden vector: generator vs this model ----
    x_rows = np.asarray(stim["x"], dtype=np.int64)
    a_rows = np.asarray(stim["a"], dtype=np.int64)
    h_given = np.asarray(stim["h"], dtype=np.int64)
    h_model = np.array([np.clip(x @ cal["w1"] + cal["b1"], -(1 << 31), (1 << 31) - 1)
                        for x in ref.integer_prefix(cal, x_rows, a_rows)["out1"]])
    # the extreme rows carry synthetic INT32 hidden vectors, so only the real
    # rows can be cross-checked against an FFN1 evaluation of the LN1 output
    ff = bool(np.array_equal(h_model[:real], h_given[:real]))
    print(f"  FFN1 hidden from LN1 output ({real} real rows): {'match' if ff else 'MISMATCH'}")

    # ---- 3. integer reference vs RTL ----
    sats = {}
    model = integer_forward(cal, x_rows, a_rows, h_given, sats)
    dumps = {
        "r1": (build / "r1.hex", 16),
        "out1": (build / "o1.hex", 16),
        "g": (build / "g.hex", 16),
        "f": (build / "f.hex", 32),
        "r2": (build / "r2.hex", 16),
        "y": (build / "y.hex", 16),
    }
    exact = True
    widths = {"r1": D, "out1": D, "g": HID, "f": D, "r2": D, "y": D}
    for name, (path, w) in dumps.items():
        rtl = read_hex_i(path, rows, w)
        if rtl.shape[1] != widths[name]:
            raise SystemExit(f"VERIFY_FAIL {path}: width {rtl.shape[1]}")
        exact &= compare(name, rtl, model[name], rows)

    # ---- 4. FP32 error (integer result decoded, real rows only) ----
    sc, refv = cal["scales"], cal["ref"]
    r1_dec = model["r1"][:real].astype(np.float64) * sc["s_r1"]
    out1_dec = model["out1"][:real].astype(np.float64) * sc["sx"]
    g_dec = model["g"][:real].astype(np.float64) * sc["s_g"]
    f_dec = model["f"][:real].astype(np.float64) * sc["s_h2"][None, :]
    r2_dec = model["r2"][:real].astype(np.float64) * sc["s_r2"]
    y_dec = model["y"][:real].astype(np.float64) * sc["s_out"]
    print("fp32_error (integer datapath decoded vs official PostNorm FP32, real rows):")
    for name, approx, exactref in [
        ("r1 ", r1_dec, refv["r1"][:real]),
        ("out1", out1_dec, refv["out1"][:real]),
        ("g  ", g_dec, refv["g"][:real]),
        ("f  ", f_dec, refv["f"][:real]),
        ("r2 ", r2_dec, refv["r2"][:real]),
        ("y  ", y_dec, refv["y"][:real]),
    ]:
        err = np.abs(approx - exactref)
        print(f"  {name}: NRMSE={nrmse(approx, exactref) * 100:.4f}%  "
              f"max_abs={err.max():.6g}  rms_abs={math.sqrt(float((err ** 2).mean())):.6g}")

    # GELU approximation in isolation: compare the integer GELU against exact
    # GELU of the *same* quantized hidden vector, so the FFN1 quantization error
    # is excluded from the activation-approximation number.
    h_dec = h_given[:real].astype(np.float64) * sc["s_h"][None, :]
    g_iso = ref.gelu_fp32(h_dec)
    print(f"gelu_approx_error (vs exact GELU of the same quantized h): "
          f"NRMSE={nrmse(g_dec, g_iso) * 100:.4f}%  "
          f"max_abs={np.abs(g_dec - g_iso).max():.6g}  (s_g={sc['s_g']:.6g})")
    lut = np.array(meta["coefficients"]["lut"], dtype=np.int64)
    smp = np.arange(-(1 << (ref.LF + 3)), (1 << (ref.LF + 3))) / float(1 << ref.LF)
    approx = []
    for x in smp:
        v = int(round(x * (1 << ref.LF)))
        idx = v + (1 << (ref.LF + 3))
        i, fr = idx >> 4, idx & 15
        approx.append((lut[i] + (((lut[i + 1] - lut[i]) * fr) >> 4)) / float(1 << ref.LF))
    lut_err = np.abs(np.array(approx) - ref.gelu_fp32(smp))
    print(f"gelu_lut_error  (pure LUT, [-8,8), 257 nodes): "
          f"max_abs={lut_err.max():.6g}  rms_abs={math.sqrt(float((lut_err ** 2).mean())):.6g}")

    print(f"saturated_lanes model: r1={sats['sat_r1']} out1={sats['sat_o1']} "
          f"r2={sats['sat_r2']} y={sats['sat_y']} "
          f"(of {rows * D} lanes each)")

    # ---- 5. coverage guard: the two variance rows really hit the eps-dominated
    # LayerNorm branch, one with var_q == 0 and one with 0 < var_q << eps_q ----
    kinds = np.asarray(stim["kinds"]).astype(str)
    var_ok = True
    for kind, want_zero in (("var_zero", True), ("var_tiny", False)):
        if kind not in kinds:
            print(f"  coverage {kind}: MISSING")
            var_ok = False
            continue
        row = int(np.argmax(kinds == kind))
        r1 = model["r1"][row]
        mean = int(r1.sum()) >> 6
        var_q = int(((r1 - mean) ** 2).sum()) >> 6
        good = (var_q == 0) if want_zero else (0 < var_q < 64)
        print(f"  coverage {kind}: var_q={var_q} -> {'ok' if good else 'WRONG'}")
        var_ok &= good

    ok = coef_ok and ff and exact and var_ok
    print("VERIFY_PASS" if ok else "VERIFY_FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
