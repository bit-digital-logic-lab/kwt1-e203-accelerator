#!/usr/bin/env python3
"""Boundary self-check for the encoder NRMSE gate (no Verilator needed).

The production gate lives in ../../encoder/model/verify_encoder.py.  This script
re-uses its `nrmse`, `nrmse_ok` and `cmp_bit` helpers and pins the boundary:

  * a stage at exactly the 3% ceiling passes   (spec says "fail when OVER 3%");
  * a stage just over the ceiling fails;
  * NRMSE(approx, exact) is 0 for identical tensors and equals the exact
    relative factor for a strictly scaled tensor;
  * cmp_bit() flags a single corrupted cell.

It is a *software-level* unit test of the gate predicate; the end-to-end
bitexact/nrmse injections are exercised by run_gate.sh against real RTL dumps.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent                      # graduate/ds_tasks/encoder_gate
MODEL = HERE.parents[1] / "encoder" / "model"               # graduate/encoder/model
sys.path.insert(0, str(MODEL))

spec = importlib.util.spec_from_file_location("verify_encoder", MODEL / "verify_encoder.py")
ve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ve)

checks = 0


def expect(label, got, want):
    global checks
    checks += 1
    if got != want:
        print(f"SELFCHECK_FAIL {label}: got {got!r}, want {want!r}")
        raise SystemExit(1)


def main():
    thr = ve.NRMSE_THRESHOLD
    expect("threshold is the fixed 3% spec constant", thr, 0.03)

    # --- predicate boundary: strictly greater than the ceiling fails ---------
    expect("nrmse_ok(0.0)", ve.nrmse_ok(0.0, thr), True)
    expect("nrmse_ok(2.99%)", ve.nrmse_ok(0.0299, thr), True)
    expect("nrmse_ok(3.00%) passes (not over)", ve.nrmse_ok(0.03, thr), True)
    expect("nrmse_ok(3.01%) fails", ve.nrmse_ok(0.0301, thr), False)
    expect("nrmse_ok(10%) fails", ve.nrmse_ok(0.10, thr), False)

    # --- nrmse numerically exact on strictly scaled tensors ------------------
    ones = np.ones((4, 8), dtype=np.float64)
    expect("nrmse identical == 0", ve.nrmse(ones, ones), 0.0)
    expect("nrmse 1.03125x == 0.03125", ve.nrmse(ones * 1.03125, ones), 0.03125)
    expect("nrmse 2x == 1.0", ve.nrmse(ones * 2.0, ones), 1.0)
    expect("nrmse_ok(1.03125x) fails", ve.nrmse_ok(ve.nrmse(ones * 1.03125, ones), thr), False)

    # --- bit-exact comparator flags a single corrupted cell -----------------
    rtl = np.arange(12, dtype=np.int64).reshape(3, 4)
    expect("cmp_bit identical stays True", ve.cmp_bit("t", rtl, rtl.copy(), True), True)
    corrupted = rtl.copy()
    corrupted[1, 2] += 1
    expect("cmp_bit one-cell corruption -> False", ve.cmp_bit("t", corrupted, rtl, True), False)

    print(f"SELFCHECK_PASS checks={checks} threshold={thr*100:.2f}%")


if __name__ == "__main__":
    main()
