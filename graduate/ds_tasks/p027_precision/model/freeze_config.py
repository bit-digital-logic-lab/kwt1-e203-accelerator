#!/usr/bin/env python3
"""p027_precision -- freeze one candidate into a self-contained config pack.

The frozen pack contains every integer tensor and coefficient needed to run the
12-layer chain: no calibration code path, no training data, no FP32 maxima.
``run_frozen.py`` consumes only this pack.

Usage:
    python3 model/freeze_config.py --name SEL_A20xy_W12
"""

from __future__ import annotations

import argparse
import hashlib
import json
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


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_pack(calib, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    meta_layers = []
    for L in calib["layers"]:
        k = L["layer"]
        pre = f"layer{k:02d}"
        arrays[f"{pre}_wq"] = L["attn"]["wq"].astype(np.int32)
        arrays[f"{pre}_wk"] = L["attn"]["wk"].astype(np.int32)
        arrays[f"{pre}_wv"] = L["attn"]["wv"].astype(np.int32)
        arrays[f"{pre}_wo"] = L["attn"]["wo"].astype(np.int32)
        arrays[f"{pre}_attn_bias"] = L["attn"]["bias"].astype(np.int64)
        arrays[f"{pre}_w1"] = L["ffn1"]["w"].astype(np.int32)
        arrays[f"{pre}_b1"] = L["ffn1"]["b"].astype(np.int64)
        arrays[f"{pre}_sw1"] = L["ffn1"]["sw"].astype(np.float64)
        arrays[f"{pre}_sh1"] = L["ffn1"]["sh"].astype(np.float64)
        arrays[f"{pre}_w2"] = L["ffn2"]["w"].astype(np.int32)
        arrays[f"{pre}_b2"] = L["ffn2"]["b"].astype(np.int64)
        arrays[f"{pre}_sw2"] = L["ffn2"]["sw"].astype(np.float64)
        arrays[f"{pre}_sh2"] = L["ffn2"]["sh2"].astype(np.float64)
        meta_layers.append(dict(
            layer=k, M_A=int(L["M_A"]), scales=L["scales"],
            attn_params=L["attn"]["params"],
            alt_key=dict(ma1=L["coeff"]["ma1"], mb1=L["coeff"]["mb1"],
                         ma2=L["coeff"]["ma2"]),
            coeff={kk: (np.asarray(vv).tolist() if isinstance(vv, np.ndarray) else vv)
                   for kk, vv in L["coeff"].items()},
        ))
    np.savez_compressed(outdir / "frozen_calib.npz", **arrays)
    config = dict(name=calib.get("name", ""), cfg=calib["cfg"],
                  isq_frac=calib["isq_frac"], schema=calib["schema"],
                  lut=[int(v) for v in calib["lut"]],
                  attn_lut=[int(v) for v in calib["attn_lut"]],
                  layers=meta_layers)
    (outdir / "frozen_config.json").write_text(json.dumps(config, indent=2) + "\n")
    manifest = dict(
        config="frozen_config.json", config_sha256=sha256(outdir / "frozen_config.json"),
        calib="frozen_calib.npz", calib_sha256=sha256(outdir / "frozen_calib.npz"),
        config_canonical_sha256=S.cfg_sha(P.cfg_norm(calib["cfg"])),
    )
    (outdir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_pack(outdir: Path):
    manifest = json.loads((outdir / "MANIFEST.json").read_text())
    if sha256(outdir / manifest["config"]) != manifest["config_sha256"]:
        raise ValueError("frozen_config.json sha256 mismatch")
    if sha256(outdir / manifest["calib"]) != manifest["calib_sha256"]:
        raise ValueError("frozen_calib.npz sha256 mismatch")
    config = json.loads((outdir / manifest["config"]).read_text())
    z = np.load(outdir / manifest["calib"])
    layers = []
    for m in config["layers"]:
        k = m["layer"]
        pre = f"layer{k:02d}"
        coeff = {}
        for kk, vv in m["coeff"].items():
            coeff[kk] = (np.asarray(vv, dtype=np.int64)
                         if isinstance(vv, list) else vv)
        layers.append(dict(
            layer=k, M_A=int(m["M_A"]), scales=m["scales"],
            attn=dict(params=m["attn_params"],
                      lut=np.asarray(config["attn_lut"], dtype=np.int64),
                      wq=z[f"{pre}_wq"].astype(np.int64),
                      wk=z[f"{pre}_wk"].astype(np.int64),
                      wv=z[f"{pre}_wv"].astype(np.int64),
                      wo=z[f"{pre}_wo"].astype(np.int64),
                      bias=z[f"{pre}_attn_bias"].astype(np.int64)),
            ffn1=dict(w=z[f"{pre}_w1"].astype(np.int64),
                      b=z[f"{pre}_b1"].astype(np.int64),
                      sw=z[f"{pre}_sw1"], sh=z[f"{pre}_sh1"]),
            ffn2=dict(w=z[f"{pre}_w2"].astype(np.int64),
                      b=z[f"{pre}_b2"].astype(np.int64),
                      sw=z[f"{pre}_sw2"], sh2=z[f"{pre}_sh2"]),
            coeff=coeff,
        ))
    return dict(schema=config["schema"], cfg=config["cfg"], name=config.get("name", ""),
                isq_frac=config["isq_frac"], layers=layers,
                lut=np.asarray(config["lut"], dtype=np.int64),
                attn_lut=np.asarray(config["attn_lut"], dtype=np.int64)), manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None,
                    help="candidate name from results.json (default: selected)")
    ap.add_argument("--outdir", default=str(TASK / "frozen"))
    args = ap.parse_args()
    res = json.loads((TASK / "results.json").read_text())
    name = args.name or res.get("selected")
    if not name:
        raise SystemExit("no selected candidate and --name not given")
    cand = next(r for r in res["candidates"] if r["name"] == name)
    if cand.get("state") != "EVALUATED":
        raise SystemExit(f"candidate {name} was not evaluated")
    cfg = P.cfg_norm(cand["cfg"])

    data = S.load_data()
    maxima = P.fp32_stage_maxima(data["weights"], data["X0_tr"], golden)
    calib = P.build_calibration(data["weights"], (maxima[0], maxima[1]),
                                maxima[2], cfg)
    calib["name"] = name
    manifest = save_pack(calib, Path(args.outdir))
    print(f"FREEZE_OK name={name} config_sha256={manifest['config_sha256']} "
          f"calib_sha256={manifest['calib_sha256']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(f"FREEZE_FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
