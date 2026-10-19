#!/usr/bin/env python3
"""Write/refresh the frozen 12-layer *integer* tensor pack for the RTL step.

Rebuilds the deterministic calibration (same code path as model/run_l1.py) and
writes ``frozen_12layer_int.npz``.  It asserts that every scale and M_A matches
the frozen ``frozen_12layer_calib.json`` so the RTL step cannot silently use a
different configuration.

Usage:  OPENBLAS_NUM_THREADS=4 python3 model/freeze_int.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TASK = HERE.parent
sys.path.insert(0, str(HERE))
import chain as C                                 # noqa: E402
import kwt1_fp32_golden as golden                 # noqa: E402


def main():
    weights = golden.load_weights(C.WORKTREE / "software/weights/kwt1_fp32.npz")
    features, partitions, paths, words, tr, va = C.load_split(C.find_split_npz())
    X0_train = [C.mfcc_to_encoder(features[i], weights) for i in tr]
    calib = C.build_calibration(weights, X0_train)

    frozen = json.loads((TASK / "frozen_12layer_calib.json").read_text())
    assert frozen["schema"] == calib["schema"]
    for L, F in zip(calib["layers"], frozen["layers"]):
        assert L["layer"] == F["layer"]
        for key in ("s_x", "s_a", "sx_ffn1", "s_g", "s_r1", "s_r2", "s_out"):
            a, b = L["scales"][key], F["scales"][key]
            assert abs(a - b) <= 1e-15 * max(1.0, abs(b)), (key, a, b)
        assert L["M_A"] == F["M_A"], (L["M_A"], F["M_A"])
        assert L["attn"]["params"] == F["attn_params"]

    arrays = {}
    for L in calib["layers"]:
        k = L["layer"]
        pre = f"layer{k+1:02d}"
        arrays[f"{pre}_attn_wq"] = np.asarray(L["attn"]["wq"], dtype=np.int16)
        arrays[f"{pre}_attn_wk"] = np.asarray(L["attn"]["wk"], dtype=np.int16)
        arrays[f"{pre}_attn_wv"] = np.asarray(L["attn"]["wv"], dtype=np.int16)
        arrays[f"{pre}_attn_wo"] = np.asarray(L["attn"]["wo"], dtype=np.int16)
        arrays[f"{pre}_attn_bias"] = np.asarray(L["attn"]["bias"], dtype=np.int32)
        arrays[f"{pre}_w1"] = np.asarray(L["ffn1"]["w"], dtype=np.int16)
        arrays[f"{pre}_b1"] = np.asarray(L["ffn1"]["b"], dtype=np.int32)
        arrays[f"{pre}_w2"] = np.asarray(L["ffn2"]["w"], dtype=np.int16)
        arrays[f"{pre}_b2"] = np.asarray(L["ffn2"]["b"], dtype=np.int32)
    out = TASK / "frozen_12layer_int.npz"
    np.savez_compressed(out, **arrays)
    print(f"FREEZE_OK layers={len(calib['layers'])} -> {out}")


if __name__ == "__main__":
    main()
