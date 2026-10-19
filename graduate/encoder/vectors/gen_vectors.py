#!/usr/bin/env python3
"""Generate all hex stimulus/parameter files for the encoder-RTL testbench.

Everything is derived inside this repository from the frozen KWT-1 checkpoint
(`software/weights/kwt1_fp32.npz`), the real encoder-input sample
(`software/validation/real_input_chain_20261017/sample_00_encoder.npy`) and the
frozen A12W10 FFN1 pack (`software/results/ffn1_a12w10_static_sym.npz`).

Two existing independent models are *reused* (not modified) to produce the
attention vectors and the postnorm integer coefficients:
  * model/attn_int_model.py        (copied from grad-attention)
  * model/kwt_postnorm_ref.py      (copied from grad-nonlinear)
No RTL output is used as an input here.

Outputs (graduate/encoder/vectors/):
  sel0_x.hex   6336 int16   X            (row-major 99x64)
  sel1_wq.hex  4096 int16   Wq           (row-major 64x64)
  sel2_wk.hex  4096 int16
  sel3_wv.hex  4096 int16
  sel4_wo.hex  4096 int16
  sel5_bias.hex 64 int32    combine_heads bias
  sel6_lut.hex  64 int16    exp LUT (design constant)
  sel7_w1.hex  16384 int16  FFN1 W1, index c*D+j  (= w1_int[j][c])
  sel8_b1.hex  256 int32    FFN1 integer bias b1_int[c]
  sel9_w2.hex  16384 int16  FFN2 W2, index k*HID+j (= w2_int[j][k])
  sel10_b2.hex 64 int32     FFN2 integer bias b2_int[k]
  meta.json                scales, M_A/Q_A, row counts
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

# Reporting precision for the diagnostic floats stored in meta.json.
#
# The FP32-reference metrics (attention NRMSE, quantisation scales taken from
# BLAS-produced tensors) are computed with numpy matrix products whose
# reduction order - and therefore the last ULPs - depend on the BLAS build and
# thread count. Two interpreters on the same machine already disagree:
#   /usr/bin/python3   (numpy 1.21.5, netlib BLAS)  -> nrmse 0.0015886401079165293
#   project .venv      (numpy 2.1.3, OpenBLAS)       -> nrmse 0.0015886401079165228
# Embedding those raw doubles made the meta.json sha256 non-reproducible.
# Rounding every reported float to 9 significant decimal digits keeps ~10^6x
# more precision than the report needs (and than the observed ULP noise) and
# makes the file byte-identical regardless of the interpreter/BLAS. The integer
# fields and the RTL-facing coefficients (hex files, M_A, Q_A) are untouched.
CANON_SIGDIG = 9


def canon_float(x):
    """Canonicalise a diagnostic float to a fixed significant-digit count."""
    return float(f"{float(x):.{CANON_SIGDIG}g}")

HERE = Path(__file__).resolve().parent          # graduate/encoder/vectors
ENC = HERE.parent                               # graduate/encoder
MODEL = ENC / "model"
ROOT = HERE.parents[2]                          # repository root
WEIGHTS = ROOT / "software/weights/kwt1_fp32.npz"
SAMPLE = ROOT / "software/validation/real_input_chain_20261017/sample_00_encoder.npy"
FROZEN = ROOT / "software/results/ffn1_a12w10_static_sym.npz"

T, D, HID = 99, 64, 256
Q_A = 16


def hx(v, width):
    return f"{int(v) & ((1 << width) - 1):0{width // 4}x}"


def write_hex(path: Path, values, width):
    with path.open("w") as f:
        for v in np.asarray(values).ravel():
            f.write(hx(v, width) + "\n")


def main():
    # ---- 1. attention vectors via the (copied) independent attention model ----
    subprocess.run([sys.executable, str(MODEL / "attn_int_model.py"), "--sample", "0"],
                   check=True, cwd=str(MODEL))
    VEC = ENC / "vectors"
    att = {n: np.loadtxt(VEC / f"{n}.txt") for n in
           ["x", "wq", "wk", "wv", "wo", "bias", "lut", "exp_o"]}
    att_params = json.loads((VEC / "params.json").read_text())

    x = att["x"].astype(np.int64).reshape(T, D)
    wq = att["wq"].astype(np.int64).reshape(D, D)
    wk = att["wk"].astype(np.int64).reshape(D, D)
    wv = att["wv"].astype(np.int64).reshape(D, D)
    wo = att["wo"].astype(np.int64).reshape(D, D)

    # ---- 2. postnorm integer coefficients via the (copied) independent ref ----
    sys.path.insert(0, str(MODEL))
    import kwt_postnorm_ref as ref
    cal = ref.calibrate()
    w1 = np.asarray(cal["w1"], dtype=np.int64)     # (D,HID) [j][c]
    b1 = np.asarray(cal["b1"], dtype=np.int64)     # (HID,)
    w2 = np.asarray(cal["w2"], dtype=np.int64)     # (HID,D) [j][k]
    b2 = np.asarray(cal["b2"], dtype=np.int64)     # (D,)

    # ---- 3. requantisation coefficient o(scale_o) -> a(scale_a) ----
    s_o = float(att_params["scale_o"])
    s_a = float(cal["scales"]["s_a"])
    m_a = int(round(s_o / s_a * (1 << Q_A)))
    if not (0 < m_a < (1 << 20)):
        raise SystemExit(f"unexpected requant coefficient {m_a}")

    # ---- 4. write the hex files ----
    write_hex(VEC / "sel0_x.hex", x, 16)
    write_hex(VEC / "sel1_wq.hex", wq, 16)
    write_hex(VEC / "sel2_wk.hex", wk, 16)
    write_hex(VEC / "sel3_wv.hex", wv, 16)
    write_hex(VEC / "sel4_wo.hex", wo, 16)
    write_hex(VEC / "sel5_bias.hex", att["bias"], 32)
    write_hex(VEC / "sel6_lut.hex", att["lut"], 16)
    # W1 index c*D+j  ->  w1[j][c]
    write_hex(VEC / "sel7_w1.hex", w1.T.reshape(-1), 16)
    write_hex(VEC / "sel8_b1.hex", b1, 32)
    # W2 index k*HID+j ->  w2[j][k]
    write_hex(VEC / "sel9_w2.hex", w2.T.reshape(-1), 16)
    write_hex(VEC / "sel10_b2.hex", b2, 32)

    meta = dict(
        sample="sample_00_encoder.npy", layer=0, T=T, D=D, HID=HID,
        scale_x=canon_float(att_params["scale_x"]), scale_o=canon_float(s_o),
        scale_a=canon_float(s_a), sx=canon_float(cal["scales"]["sx"]),
        s_out=canon_float(cal["scales"]["s_out"]),
        M_A=m_a, Q_A=Q_A,
        attn_nrmse_fp32=canon_float(att_params["nrmse_fp32"]),
    )
    (VEC / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"GEN_OK M_A={m_a} Q_A={Q_A} scale_o={s_o:.6e} scale_a={s_a:.6e}")
    print(f"out_dir={VEC}")


if __name__ == "__main__":
    main()
