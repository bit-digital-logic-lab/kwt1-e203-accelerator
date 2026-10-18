"""Independent numeric references for the KWT-1 FFN1 graduate verification.

Two things live here:

* an exactly-specified integer GEMM (``int_gemm_python``) written as plain
  Python integer arithmetic, plus a second, structurally different
  implementation on top of NumPy int64 (``int_gemm_numpy``).  They are used as
  a differential golden pair: a stage may only trust an integer accelerator
  when both agree and a perturbed golden is actually rejected.
* a small FP32 NRMSE helper matching the contest convention (RMS error over an
  RMS denominator with a 1e-8 floor).

Nothing here reads a DUT output; the references are computed from the inputs.
"""

from __future__ import annotations

import numpy as np

NRMSE_FLOOR = 1e-8


def int_gemm_python(a, w, bias=None):
    """Pure-Python exact integer GEMM: a[M,K] @ w[K,N] + bias[N]."""
    m = len(a)
    k = len(a[0]) if m else 0
    n = len(w[0]) if k else 0
    if k and len(w) != k:
        raise ValueError("K dimension mismatch")
    out = []
    for i in range(m):
        row = []
        for j in range(n):
            acc = 0
            for t in range(k):
                acc += int(a[i][t]) * int(w[t][j])
            if bias is not None:
                acc += int(bias[j])
            row.append(acc)
        out.append(row)
    return out


def int_gemm_numpy(a, w, bias=None):
    """NumPy int64 GEMM; a structurally different implementation of the same math."""
    a64 = np.asarray(a, dtype=np.int64)
    w64 = np.asarray(w, dtype=np.int64)
    acc = a64 @ w64
    if bias is not None:
        acc = acc + np.asarray(bias, dtype=np.int64)
    return acc


def nrmse_percent(actual, reference):
    """Contest-style NRMSE in percent with the 1e-8 denominator floor."""
    actual = np.asarray(actual, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if actual.shape != reference.shape:
        raise ValueError(f"shape mismatch {actual.shape} vs {reference.shape}")
    difference = actual - reference
    denominator = max(np.sqrt(np.mean(reference ** 2)), NRMSE_FLOOR)
    return 100.0 * np.sqrt(np.mean(difference ** 2)) / denominator


def max_abs_error(actual, reference):
    actual = np.asarray(actual, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    return float(np.max(np.abs(actual - reference)))


def quantize_activation(x, scale, bits=11):
    """Symmetric integer activation quantisation used by the frozen A11W10 scheme."""
    limit = (1 << (bits - 1)) - 1
    return np.clip(np.rint(np.asarray(x, dtype=np.float64) / float(scale)),
                   -limit, limit).astype(np.int64)


def decode_int32(q, scale):
    return np.asarray(q, dtype=np.float64) * float(scale)
