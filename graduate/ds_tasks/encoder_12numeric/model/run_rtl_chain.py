#!/usr/bin/env python3
"""RTL bit-exact + FP32 test of the 12-layer numerical chain (encoder_12numeric).

One Verilator binary is built per encoder layer with that layer's own
``attn_params.svh`` / ``kwt_postnorm_cfg.svh`` (distinct attention shifts and
distinct residual/LayerNorm/GELU/FFN2 coefficients) and its own requantisation
parameter ``M_A``.  The binary is then re-used for several samples: only the
stimulus directory changes.

The 12 layers are chained *through the RTL*: layer k's integer output y is
written as layer k+1's X stimulus (the calibration guarantees s_x[k+1] =
s_out[k], so this is an exact copy, exactly like the on-chip controller).

For every layer and sample the script checks
  * RTL y / a / r1 / g / f / r2  ==  independent integer reference (bit-exact);
  * dequantised RTL y vs the FP32 golden  -> layuser NRMSE.

Evidence levels: L2 (RTL bit-exact).  L1 is `model/run_l1.py`; E203,
synthesis and official acceptance are NOT established here.

Usage:
    python3 model/run_rtl_chain.py                 # all 6 real validation samples
    python3 model/run_rtl_chain.py --samples 0     # only sample_00
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TASK = HERE.parent
sys.path.insert(0, str(HERE))
import chain as C                                 # noqa: E402
import gen_rtl_vectors as GV                      # noqa: E402
import kwt1_fp32_golden as golden                 # noqa: E402

RTL = C.WORKTREE / "graduate/rtl/encoder"
TP = RTL / "third_party"
TB = TASK / "tb/tb_chain_layer.sv"
SOURCES = [
    TP / "kwt_postnorm_pkg.sv",
    TP / "kwt_residual_add.sv",
    TP / "kwt_layernorm.sv",
    TP / "kwt_gelu.sv",
    TP / "kwt_ffn2.sv",
    TP / "kwt_postnorm_ffn.sv",
    TP / "attn_single_head.sv",
    RTL / "kwt_ffn1_row.sv",
    RTL / "kwt_encoder_layer.sv",
    TB,
]
D, HID, T, NLAYERS = 64, 256, 99, 12
STAGE_FILES = {"a": ("a.hex", 16, D), "r1": ("r1.hex", 16, D),
               "g": ("g.hex", 16, HID), "f": ("f.hex", 32, D),
               "r2": ("r2.hex", 16, D), "y": ("y.hex", 16, D)}
STAGE_REF = {"a": "a", "r1": "r1", "g": "g", "f": "f", "r2": "r2", "y": "y"}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_frozen(calib_json: Path, int_npz: Path):
    cal = json.loads(calib_json.read_text())
    z = np.load(int_npz)
    layers = []
    for L in cal["layers"]:
        k = L["layer"]
        pre = f"layer{k+1:02d}"
        sc = L["scales"]
        sw1 = np.asarray(L["ffn1_sw"], dtype=np.float64)
        sw2 = np.asarray(L["ffn2_sw"], dtype=np.float64)
        attn = dict(params={kk: vv for kk, vv in L["attn_params"].items()},
                    lut=np.asarray(cal["attn_lut"], dtype=np.int64),
                    wq=z[f"{pre}_attn_wq"].astype(np.int64),
                    wk=z[f"{pre}_attn_wk"].astype(np.int64),
                    wv=z[f"{pre}_attn_wv"].astype(np.int64),
                    wo=z[f"{pre}_attn_wo"].astype(np.int64),
                    bias=z[f"{pre}_attn_bias"].astype(np.int64))
        layers.append(dict(
            layer=k, M_A=int(L["M_A"]), scales=sc,
            attn=attn,
            ffn1=dict(w=z[f"{pre}_w1"].astype(np.int64), b=z[f"{pre}_b1"].astype(np.int64),
                      sw=sw1, sh=sc["sx_ffn1"] * sw1),
            ffn2=dict(w=z[f"{pre}_w2"].astype(np.int64), b=z[f"{pre}_b2"].astype(np.int64),
                      sw=sw2, sh2=sc["s_g"] * sw2),
            coeff={kk: (np.asarray(vv, dtype=np.int64) if isinstance(vv, list) else vv)
                   for kk, vv in L["coeff"].items()},
        ))
    return dict(schema=cal["schema"], layers=layers, lut=cal["lut"],
                attn_lut=cal["attn_lut"]), cal


def read_hex(path: Path, width: int, cols: int) -> np.ndarray:
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    per = width // 4
    sign = 1 << (width - 1)
    full = 1 << width
    rows = []
    for ln in lines:
        assert len(ln) == cols * per, f"{path}: bad line width {len(ln)}"
        rows.append([int(ln[i:i + per], 16) for i in range(0, len(ln), per)])
    vals = np.array(rows, dtype=np.int64)
    return np.where(vals >= sign, vals - full, vals)


def write_x_hex(path: Path, X_int: np.ndarray):
    with Path(path).open("w") as fh:
        for v in np.asarray(X_int, dtype=np.int64).ravel():
            fh.write(GV.hx(v, 16) + "\n")


def build_layer(layer, build_root: Path, log):
    pre = f"layer{layer['layer'] + 1:02d}"
    bdir = build_root / pre
    (bdir / "obj").mkdir(parents=True, exist_ok=True)
    (bdir / "out").mkdir(parents=True, exist_ok=True)
    cmd = ["verilator", "--binary", "--timing", "-j", "4", "-Wno-fatal",
           f"-I{bdir / 'include'}", "--top-module", "tb_chain_layer",
           "-o", "sim_layer", "--Mdir", str(bdir / "obj"),
           f"-GM_A={layer['M_A']}", "-GQ_A=16"] + [str(s) for s in SOURCES]
    t = time.time()
    with (bdir / "verilator.log").open("w") as fh:
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    dt = time.time() - t
    log(f"BUILD {pre} exit={r.returncode} {dt:.1f}s -> {bdir/'verilator.log'}")
    return r.returncode, bdir


def run_layer(layer, bdir: Path, vec: Path, log):
    cmd = [str(bdir / "obj" / "sim_layer"), f"+VEC={vec}", f"+OUT={bdir/'out'}"]
    t = time.time()
    with (bdir / "sim.log").open("w") as fh:
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    dt = time.time() - t
    txt = (bdir / "sim.log").read_text()
    ok = r.returncode == 0 and "TB_PASS" in txt
    log(f"RUN   layer{layer['layer']:02d} exit={r.returncode} {dt:.1f}s pass={ok}")
    return r.returncode, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="all",
                    help="all|comma list of real sample indices")
    ap.add_argument("--limit-layers", type=int, default=NLAYERS)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    lines = []

    def log(m):
        print(m, flush=True)
        lines.append(m)

    calib_json = TASK / "frozen_12layer_calib.json"
    int_npz = TASK / "frozen_12layer_int.npz"
    if not calib_json.exists() or not int_npz.exists():
        raise SystemExit("run model/run_l1.py first (frozen calibration missing)")
    calib, cal_json = load_frozen(calib_json, int_npz)
    log(f"FROZEN {calib_json.name} sha256={sha256(calib_json)[:16]}... "
        f"{int_npz.name} sha256={sha256(int_npz)[:16]}...")

    weights = golden.load_weights(C.WORKTREE / "software/weights/kwt1_fp32.npz")
    real_dir = C.find_real_dir()
    real_files = sorted(real_dir.glob("sample_*_encoder.npy"))
    if args.samples == "all":
        idx = list(range(len(real_files)))
    else:
        idx = [int(x) for x in args.samples.split(",")]

    build_root = TASK / "build"
    nlayers = min(args.limit_layers, NLAYERS)

    # ---- build one binary per layer --------------------------------------
    bids = {}
    for k in range(nlayers):
        bdir = build_root / f"layer{k+1:02d}"
        rc = 0
        if args.rebuild or not (bdir / "obj" / "sim_layer").exists():
            # the include files must exist before verilator runs; the actual
            # stimulus is (re)written per sample in the run phase below.
            GV.write_layer(bdir, calib["layers"][k], calib, np.zeros((T, D), dtype=np.int64))
            rc, bdir = build_layer(calib["layers"][k], build_root, log)
        else:
            log(f"BUILD layer{k+1:02d} cached")
        if rc != 0:
            raise SystemExit(f"verilator build failed for layer {k}")
        bids[k] = bdir

    # ---- run every sample through the RTL chain ---------------------------
    records = []
    bit_ok_all = True
    for si in idx:
        name = real_files[si].name
        X0 = np.load(real_files[si]).astype(np.float32)
        arrs = []
        golden.forward(X0, weights, NLAYERS, arrays=arrs)
        ref = C.integer_chain(calib, X0, want_stages=True)

        s_x0 = calib["layers"][0]["scales"]["s_x"]
        x_int = np.clip(np.rint(X0 / s_x0), -C.CLAMP_A, C.CLAMP_A).astype(np.int64)
        rec = {"sample": name, "layers": [], "bit_exact": True}
        for k in range(nlayers):
            pre = f"layer{k+1:02d}"
            ldir = build_root / pre
            vec = ldir / "vectors"
            GV.write_layer(ldir, calib["layers"][k], calib, x_int)
            rc, ok = run_layer(calib["layers"][k], bids[k], vec, log)
            if not (rc == 0 and ok):
                rec["bit_exact"] = False
                bit_ok_all = False
                rec["layers"].append({"layer": k, "run_exit": rc, "tb_pass": ok})
                break
            y_rtl = read_hex(ldir / "out" / "y.hex", 16, D)
            stage_cmp = {}
            lay_ok = True
            for st, (fname, width, cols) in STAGE_FILES.items():
                rtl_st = read_hex(ldir / "out" / fname, width, cols)
                ref_st = ref["int"][STAGE_REF[st]][k]
                eq = np.array_equal(rtl_st, ref_st)
                stage_cmp[st] = "exact" if eq else int(np.sum(rtl_st != ref_st))
                lay_ok &= eq
            y_nrmse = C.nrmse(y_rtl * calib["layers"][k]["scales"]["s_out"],
                              arrs[k]["layer_output"])
            # chain the RTL output into the next layer (exact copy semantics)
            x_int = y_rtl
            rec["layers"].append({"layer": k, "stages": stage_cmp,
                                  "y_nrmse_fp32": y_nrmse,
                                  "y_rows": int(y_rtl.shape[0]),
                                  "bit_exact": bool(lay_ok)})
            if not lay_ok:
                rec["bit_exact"] = False
                bit_ok_all = False
        records.append(rec)
        log(f"SAMPLE {name} bit_exact={rec['bit_exact']} worst_y_nrmse="
            f"{max(l['y_nrmse_fp32'] for l in rec['layers'])*100:.4f}%")

    results = {
        "task": "encoder_12numeric",
        "evidence_levels": {
            "software_reference": "L1: model/run_l1.py",
            "rtl_bit_exact": "L2: this script (per-layer elaborated RTL)",
            "e203": "no",
            "synthesis": "no",
            "official_acceptance": "no",
        },
        "layers": nlayers,
        "samples": [real_files[i].name for i in idx],
        "bit_exact_all": bool(bit_ok_all),
        "records": records,
        "worst_y_nrmse": max((l["y_nrmse_fp32"] for r in records for l in r["layers"]),
                             default=None),
        "runtime_seconds": round(time.time() - t0, 2),
    }
    (TASK / "rtl_results.json").write_text(json.dumps(results, indent=2) + "\n")
    (TASK / "logs").mkdir(exist_ok=True)
    (TASK / "logs" / "run_rtl_chain.log").write_text("\n".join(lines) + "\n")
    log(f"RTL_DONE bit_exact_all={bit_ok_all} runtime={results['runtime_seconds']}s")
    print("RTL_PASS" if bit_ok_all else "RTL_FAIL")


if __name__ == "__main__":
    main()
