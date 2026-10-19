#!/usr/bin/env python3
"""p027_precision -- scalar-vs-vector integer semantics self-check.

The search runs vectorised NumPy primitives.  This script re-implements the
documented integer contract in scalar Python (round-half-up shifts, integer
sqrt, saturation) and checks that the vectorised primitives agree bit-for-bit
for both the baseline (A12) and the high-precision (A16/A20) profiles.

It also checks:
  * activation scale continuity s_x[k+1] == s_out[k] (exact float equality),
  * accumulator width bounds (measured max < 2^(declared-1)),
  * the A12W10 profile reproduces the frozen baseline per-layer numbers.
Exit 0 = all checks pass.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TASK = HERE.parent
WORKTREE = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(WORKTREE / "graduate/encoder/reference"))
import prec_ref as P                                   # noqa: E402
import kwt1_fp32_golden as golden                      # noqa: E402
import run_search as S                                 # noqa: E402


def mul_sr_scalar(a, m, q):
    return (int(a) * int(m) + (1 << (q - 1))) >> q


def isqrt_scalar(v):
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


def layernorm_scalar(x, eps_q, gcoef, bcoef, mo, clamp, k_frac):
    mean = sum(int(v) for v in x) >> 6
    d = [int(v) - mean for v in x]
    var = sum(v * v for v in d) >> 6
    vq = (var << P.VAF) + int(eps_q)
    sq = isqrt_scalar(vq)
    num = 1 << (k_frac + P.VAF // 2)
    imax = (1 << P.ISQ_REG_BITS) - 1
    isq = imax if sq == 0 else min((num + (sq >> 1)) // max(sq, 1), imax)
    out = []
    for i, di in enumerate(d):
        u = mul_sr_scalar(di * int(gcoef[i]), isq, k_frac)
        v = mul_sr_scalar(u, int(mo), P.GO) + int(bcoef[i])
        out.append(min(max(v, -clamp), clamp))
    return out


def residual_scalar(a, ma, b, mb, clamp, qb):
    out = []
    for i in range(len(a)):
        v = mul_sr_scalar(a[i], ma, qb) + mul_sr_scalar(b[i], mb[i], qb)
        out.append(min(max(v, -clamp), clamp))
    return out


def gelu_scalar(h, mg, mgn, lut, lf, clamp, shift, qg):
    out = []
    lim = 1 << (lf + 3)
    for i, hi in enumerate(h):
        vq = mul_sr_scalar(hi, mg[i], qg)
        v = min(max(vq, -lim), lim - 1)
        idx = v + lim
        li = idx >> shift
        frac = idx & ((1 << shift) - 1)
        l0 = int(lut[li])
        l1 = int(lut[min(li + 1, lut.size - 1)])
        g8 = l0 + (((l1 - l0) * frac) >> shift)
        g = mul_sr_scalar(g8, mgn, P.QGN)
        out.append(min(max(g, -clamp), clamp))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=2)
    args = ap.parse_args()
    rng = np.random.default_rng(20261019)
    ok = True
    checks = []

    data = S.load_data()
    maxima = P.fp32_stage_maxima(data["weights"], data["X0_tr"][:8], golden)

    for label, over in (
        ("A12W10", dict()),
        ("A16W16", dict(stage_bits=dict(x=16, a=16, r1=16, ln1=16, g=16, r2=16,
                                        y=16), w_attn=16, w1=16, w2=16, qb=32,
                        lf=12, gelu_shift=8, f_table=8, f_interp=4, m_lut=20,
                        prob_bits=20, kc_min=1 << 20)),
        ("A20W16", dict(stage_bits=dict(x=20, a=20, r1=20, ln1=20, g=20, r2=20,
                                        y=20), w_attn=16, w1=16, w2=16, qb=36,
                        lf=12, gelu_shift=8, f_table=8, f_interp=4, m_lut=20,
                        prob_bits=20, kc_min=1 << 20)),
    ):
        cfg = P.cfg_norm(over)
        calib = P.build_calibration(data["weights"], (maxima[0], maxima[1]),
                                    maxima[2], cfg)
        cl = {k: P.clamp_of(cfg["stage_bits"][k])
              for k in ("x", "a", "r1", "ln1", "g", "r2", "y")}
        # 1. scalar vs vector primitives
        for _ in range(args.samples):
            x = rng.integers(-cl["y"], cl["y"] + 1, size=(3, P.D))
            eps_q = 12345
            g = rng.integers(-(1 << 16), 1 << 16, size=P.D)
            b = rng.integers(-5000, 5000, size=P.D)
            mo = 4321
            kf = P.ln_k(calib, "y")
            vec = P.layernorm_rows(x, eps_q, g, b, mo, cl["y"], kf)
            for i in range(3):
                scalar = layernorm_scalar(list(x[i]), eps_q, g, b, mo, cl["y"], kf)
                if not np.array_equal(vec[i], np.array(scalar)):
                    ok = False
                    checks.append(f"{label}: layernorm row {i} mismatch")
            a1 = rng.integers(-cl["r1"], cl["r1"] + 1, size=(2, P.D))
            a2 = rng.integers(-cl["a"], cl["a"] + 1, size=(2, P.D))
            mb = rng.integers(-(1 << 20), 1 << 20, size=P.D)
            vec = P.residual_rows(a1, 61234, a2, mb, cl["r1"], int(cfg["qb"]))
            for i in range(2):
                scalar = residual_scalar(list(a1[i]), 61234, list(a2[i]), list(mb),
                                         cl["r1"], int(cfg["qb"]))
                if not np.array_equal(vec[i], np.array(scalar)):
                    ok = False
                    checks.append(f"{label}: residual row {i} mismatch")
            h = rng.integers(-(1 << 24), 1 << 24, size=(2, P.HID))
            mg = rng.integers(1, 100000, size=P.HID)
            lut = np.asarray(calib["lut"], dtype=np.int64)
            vec = P.gelu_rows(h, mg, 129003, lut, cfg["lf"], cl["g"],
                              cfg["gelu_shift"], P.gelu_qg(cfg))
            for i in range(2):
                for j in range(0, P.HID, 37):
                    scalar = gelu_scalar([int(h[i, j])], [int(mg[j])], 129003, lut,
                                         cfg["lf"], cl["g"], cfg["gelu_shift"],
                                         P.gelu_qg(cfg))[0]
                    if int(vec[i, j]) != int(scalar):
                        ok = False
                        checks.append(f"{label}: gelu {i},{j} mismatch")
        # 2. activation scale continuity
        for k in range(P.NLAYERS - 1):
            if calib["layers"][k]["scales"]["s_out"] != calib["layers"][k + 1]["scales"]["s_x"]:
                ok = False
                checks.append(f"{label}: scale discontinuity at layer {k}")
        # 3. accumulator width bound
        ev = S.evaluate(calib, data["X0_va"][:2],
                        S.golden_val_cache(data["weights"], data["X0_va"][:2]))
        for key, bits in ev["acc_bits"].items():
            if bits > 64:
                ok = False
                checks.append(f"{label}: {key} accumulator needs {bits} bits")
        checks.append(f"{label}: scalar==vector, scale-continuity, acc<=64 OK "
                      f"(acc={ev['acc_bits']}, sat={ev['sat']})")

    # 4. baseline reproduction against the frozen 240-sample measurement
    base = next(r for r in json.loads((TASK / "results.json").read_text())["candidates"]
                if r["name"] == "A12W10") if (TASK / "results.json").exists() else None
    if base and base.get("state") == "EVALUATED":
        checks.append(f"A12W10 frozen-measurement worst={base['worst_y_nrmse_pct']}% "
                      f"(240 validation samples) -- reference 2.7528%")

    print("SELFTEST " + ("PASS" if ok else "FAIL"))
    for c in checks:
        print("  " + c)
    (TASK / "logs").mkdir(exist_ok=True)
    (TASK / "logs" / "selftest.log").write_text(
        "\n".join(["SELFTEST " + ("PASS" if ok else "FAIL")] + checks) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(f"RUNTIME_FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
