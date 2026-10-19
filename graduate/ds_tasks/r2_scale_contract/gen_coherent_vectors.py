#!/usr/bin/env python3
"""Emit RTL vectors for the COHERENT attention calibration (r2_scale_contract).

Reads the round-1 attention model READ-ONLY (attn_scale worktree) and the
candidate coherent calibration produced by ``run_cross_check.py``, then writes
``attn_params.svh``, the validation vectors and ``params.json`` into a build
directory (default /tmp/opencode/r2_scale_contract/rtl_check).  Nothing is
written into any other worktree.

The emitted vectors are checked bit-for-bit by tb/tb_attn.cpp under Verilator
(see rtl_check_coherent.sh).  This is L2 evidence for the ATTENTION stage only.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ATTN_WT = Path("/home/yang/kwt-worktrees/ds-grad-attn_scale")
ATTN_MODEL = ATTN_WT / "graduate/encoder/model/attn_int_model.py"
CAND = HERE / "candidate/coherent_frozen_calib.json"
REAL_DIR = ROOT / "software/validation/real_input_chain_20261017"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", type=Path,
                    default=Path("/tmp/opencode/r2_scale_contract/rtl_check"))
    args = ap.parse_args()
    rtl_dir = args.build / "rtl"
    vec_dir = args.build / "vectors"
    rtl_dir.mkdir(parents=True, exist_ok=True)
    vec_dir.mkdir(parents=True, exist_ok=True)

    A = load_module("r1_attn_model_coh", ATTN_MODEL)
    cand = json.loads(CAND.read_text())
    a = cand["attention"]
    cal = {
        "schema": "attn-int-calib/1", "layer": 0,
        "calibration_split": "training",
        "calibration_samples": [{"file": f, "split": "training"}
                                for f in ("coherent_240_training_split",)],
        "validation_samples": [],
        "scale_x": a["scale_x"], "scale_o": a["scale_o"],
        "RQ": a["RQ"], "RK": a["RK"], "RV": a["RV"],
        "RC": a["RC"], "RO": a["RO"], "KC": a["KC"], "SH_KC": a["SH_KC"],
        "bias_int": a["bias_int"],
    }
    # emit() writes attn_params.svh + vectors; redirect it into the build dir.
    A.RTL_DIR = rtl_dir
    A.VEC = vec_dir

    wf = A.load_weights()
    wi, _sw = A.quantize_weights(wf)

    samples = ["sample_00_encoder.npy", "sample_02_encoder.npy", "sample_05_encoder.npy"]
    for idx, name in enumerate(samples):
        Xf = np.load(REAL_DIR / name, allow_pickle=False).astype(np.float64)
        out = A.integer_forward(Xf, cal, wi)
        Of = A.fp32_forward(Xf, wf)
        stats, _out, _Of = A.evaluate_sample(REAL_DIR / name, cal, wi, wf)
        stats["sample_index"] = idx
        stats = A.build_stats(REAL_DIR / name, stats, cal)
        A.emit(out, Of, cal, wi, idx)
        (vec_dir / "params.json").write_text(json.dumps(stats, indent=2))
        # keep a per-sample copy so the shell driver can rebuild the vectors
        sample_dir = vec_dir / f"sample_{idx}"
        sample_dir.mkdir(exist_ok=True)
        for f in vec_dir.iterdir():
            if f.is_file():
                (sample_dir / f.name).write_bytes(f.read_bytes())
        print(f"EMIT_OK sample{idx} {name} scale_x={cal['scale_x']:.9g} "
              f"KC={cal['KC']} SH_KC={cal['SH_KC']} O_int_max={stats['O_int_max']}")
    print(f"GEN_COHERENT_VECTORS_OK dir={vec_dir}")


if __name__ == "__main__":
    main()
