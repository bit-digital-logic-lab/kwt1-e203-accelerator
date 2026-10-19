#!/usr/bin/env python3
"""r2_scale_contract: cross-check the round-1 attn_scale / postnorm_scale
training samples, input scales, quantisation boundaries and RTL constants, and
decide whether the two frozen artefacts can be composed into ONE frozen config.

Round-1 artefacts (READ-ONLY, other worktrees):
  * attn_scale   : graduate/encoder/model/attn_int_model.py
                   graduate/ds_tasks/attn_scale/attn_calib_train.json
  * postnorm_scale: graduate/encoder/model/kwt_postnorm_ref.py
                   graduate/ds_tasks/postnorm_scale/frozen_postnorm_calib.json

What this driver does (all gradient-free):
  [1] provenance      : splits, sample lists, weight/dataset hashes
  [2] split overlap   : are the round-1 sample sets the same official split?
  [3] input scale     : the encoder input X is quantised twice (attention QKV
                        input and the postnorm residual).  Compare the two frozen
                        values and their training maxima.
  [4] boundary        : A12/2047 hard clamp vs int16 container; attention input
                        integer range under each scale.
  [5] RTL constants   : third_party/*.svh + tb_encoder.sv M_A vs the frozen model
                        constants.
  [6] conflict proof  : run the naive merge (attention scale_x for x, postnorm
                        s_x for the residual multiplier) on the VALIDATION split
                        and show the residual blows past the 3% gate.
  [7] coherent config : freeze ONE encoder-input scale on the shared TRAINING
                        split (attn 3 samples are a subset of the postnorm 240),
                        refit the attention constants on that 240-sample training
                        set, keep the postnorm constants, derive M_A; evaluate the
                        whole chain on the VALIDATION split only.

The training split is used for calibration; the validation split is evaluated
only.  No validation sample ever fits a scale or a threshold.

Evidence levels printed at the end: L1 software reference (this file) and RTL
bit-exact (only if the separate rtl_check_coherent.sh passes).
"""

import argparse
import hashlib
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                                   # this worktree root

ATTN_WT = Path("/home/yang/kwt-worktrees/ds-grad-attn_scale")
POST_WT = Path("/home/yang/kwt-worktrees/ds-grad-postnorm_scale")
ATTN_MODEL = ATTN_WT / "graduate/encoder/model/attn_int_model.py"
ATTN_CALIB = ATTN_WT / "graduate/ds_tasks/attn_scale/attn_calib_train.json"
POST_MODEL = POST_WT / "graduate/encoder/model/kwt_postnorm_ref.py"
POST_CALIB = POST_WT / "graduate/ds_tasks/postnorm_scale/frozen_postnorm_calib.json"

ATTN_HDR_ROUND1 = ATTN_WT / "graduate/encoder/rtl/attn_params.svh"     # round-1 (3-sample)
ATTN_HDR_BASE = ROOT / "graduate/rtl/encoder/third_party/attn_params.svh"   # grad-encoder baseline
POST_HDR = ROOT / "graduate/rtl/encoder/third_party/kwt_postnorm_cfg.svh"
TB_ENCODER = ROOT / "graduate/encoder/tb/tb_encoder.sv"

WEIGHTS = ROOT / "software/weights/kwt1_fp32.npz"
FFN1_PACK = ROOT / "software/results/ffn1_a12w10_static_sym.npz"
REAL_DIR = ROOT / "software/validation/real_input_chain_20261017"
MANIFEST = REAL_DIR / "manifest.json"

SPLIT_NPZ_CANDIDATES = [
    ROOT / "software/.build/quant_samples_12labels_20.npz",
    Path("/home/yang/Desktop/ic设计竞赛/software/.build/quant_samples_12labels_20.npz"),
]
NRMSE_LIMIT = 0.03
PREFIX = "layer_with_weights-0/"
SCALE_KEYS = ("s_x", "s_a", "s_g", "s_r1", "s_r2", "s_out")

LOG_LINES = []


def log(msg=""):
    print(msg, flush=True)
    LOG_LINES.append(str(msg))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def resolve_npz():
    for c in SPLIT_NPZ_CANDIDATES:
        if c.exists():
            return c
    raise FileNotFoundError("quant_samples_12labels_20.npz not found")


def nrmse(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


# --------------------------------------------------------------------------
# dataset helpers (encoder input projection, identical to postnorm_scale)
# --------------------------------------------------------------------------

def load_projection(weights):
    return (weights[PREFIX + "patch_proj/kernel"].astype(np.float64),
            weights[PREFIX + "patch_proj/bias"].astype(np.float64),
            weights[PREFIX + "class_emb"][0].astype(np.float64),
            weights[PREFIX + "pos_emb"][0].astype(np.float64))


def mfcc_to_encoder(mfcc, projection):
    kernel, bias, cls, pos = projection
    projected = np.asarray(mfcc, dtype=np.float64) @ kernel + bias
    return np.concatenate([cls, projected], axis=0) + pos


# --------------------------------------------------------------------------
# attention refit on an arbitrary TRAINING set (replicates attn_int_model.calibrate)
# --------------------------------------------------------------------------

def attn_refit(A, Xs, sx, wf, wi, sw):
    def xi(x):
        return A.quantize_activation(x, sx)

    RQ = max(A.fit_shift(xi(x) @ wi["wq"]) for x in Xs)
    RK = max(A.fit_shift(xi(x) @ wi["wk"]) for x in Xs)
    RV = max(A.fit_shift(xi(x) @ wi["wv"]) for x in Xs)
    sq = sx * sw["wq"] * (2 ** RQ)
    sk = sx * sw["wk"] * (2 ** RK)
    sv = sx * sw["wv"] * (2 ** RV)
    KC, SH_KC = A._kcoef(sq, sk)
    shifts = dict(RQ=RQ, RK=RK, RV=RV, KC=KC, SH_KC=SH_KC)
    ctxs = [A._softmax_ctx(xi(x), wi, shifts)["ctx"] for x in Xs]
    RC = max(A.fit_shift(c) for c in ctxs)
    sct = sv * (2 ** (RC - 15))
    bias_int = np.rint(wf["bias"] / (sct * sw["wo"])).astype(np.int64)
    Roas = []
    for c in ctxs:
        Ci = A.clip16(A.round_half_up_shift(c, RC))
        Roas.append(Ci @ wi["wo"] + bias_int)
    RO = max(A.fit_shift(o) for o in Roas)
    so = sct * sw["wo"] * (2 ** RO)
    return dict(scale_x=float(sx), RQ=int(RQ), RK=int(RK), RV=int(RV),
                RC=int(RC), RO=int(RO), KC=int(KC), SH_KC=int(SH_KC),
                scale_q=float(sq), scale_k=float(sk), scale_v=float(sv),
                scale_ct=float(sct), scale_o=float(so),
                bias_int=[int(v) for v in bias_int])


# --------------------------------------------------------------------------
# numpy FFN2 (numerically identical to the pure-python ref, much faster)
# --------------------------------------------------------------------------

def vectorise_ffn2(P):
    def ffn2_np(g_row, w2, b2):
        g = np.asarray(g_row, dtype=np.int64)
        w = np.asarray(w2, dtype=np.int64)
        acc = w.T @ g
        if np.abs(acc).max() >= (1 << 46):
            raise OverflowError("FFN2 accumulator exceeds the 48-bit RTL width")
        return (acc + np.asarray(b2, dtype=np.int64)).tolist()
    P.ffn2 = ffn2_np


def eval_chain(A, P, calA, calP, x, wi, use_requant, requant):
    """Run the integer chain for one sample; return dequantised stage NRMSEs.

    use_requant=False -> postnorm `a` is quantised from the FP32 attention output
                         (pure software-reference chain, as postnorm_scale does).
    use_requant=True  -> postnorm `a` is the RTL requant of the integer attention
                         output (hardware-faithful chain).
    """
    w, ref = P.fp32_activations(x)
    sc = calP["scales"]
    s_x = sc["s_x"]
    xq_raw = P.rint_arr(x / s_x)
    x_clip = int((np.abs(xq_raw) > P.CLAMP_A).sum())
    xq = np.clip(xq_raw, -P.CLAMP_A, P.CLAMP_A).astype(np.int64)

    outA = A.integer_forward(x, calA, wi)
    Oi = outA["Oi"]
    if use_requant:
        M_A, Q_A = requant
        rq = (Oi.astype(np.int64) * M_A + (1 << (Q_A - 1))) >> Q_A
        a_rows = np.clip(rq, -32768, 32767).astype(np.int64)
    else:
        a_rows = np.clip(P.rint_arr(ref["attn_out"] / sc["s_a"]),
                         -P.CLAMP_A, P.CLAMP_A).astype(np.int64)

    pre = P.integer_prefix(calP, xq, a_rows)
    h = np.clip(pre["out1"] @ np.asarray(calP["w1"], dtype=np.int64)
                + np.asarray(calP["b1"], dtype=np.int64),
                -(1 << 31), (1 << 31) - 1)
    sats = {}
    out = P.integer_forward(calP, xq, a_rows, h, sats)
    dequant = {"r1": out["r1"] * sc["s_r1"], "g": out["g"] * sc["s_g"],
               "r2": out["r2"] * sc["s_r2"], "y": out["y"] * sc["s_out"]}
    stages = {n: nrmse(dequant[n], ref[n]) for n in ("r1", "g", "r2", "y")}
    return {"stages": stages,
            "worst_nrmse": max(stages.values()),
            "attn_nrmse": nrmse(Oi * calA["scale_o"], ref["attn_out"]),
            "x_int_max": int(np.abs(np.clip(A.quantize_activation(x, calA["scale_x"]),
                                            -32768, 32767)).max()),
            "a_int_max": int(np.abs(a_rows).max()),
            "x_clip": x_clip,
            "sat": {k: int(v) for k, v in sats.items()}}


def main():
    t0 = time.time()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-val", type=int, default=0,
                    help="cap dataset validation samples (0 = all 240)")
    ap.add_argument("--out", type=Path, default=HERE / "results.json")
    args = ap.parse_args()

    log("=" * 78)
    log("r2_scale_contract — cross-check attn_scale x postnorm_scale")
    log("=" * 78)

    A = load_module("r1_attn_model", ATTN_MODEL)
    P = load_module("r1_postnorm_model", POST_MODEL)
    vectorise_ffn2(P)

    calA = json.loads(ATTN_CALIB.read_text())
    calP = json.loads(POST_CALIB.read_text())
    frP = P.load_frozen_calibration(POST_CALIB)

    results = {"task": "r2_scale_contract"}

    # ---------------------------------------------------------------- [1]
    log("\n[1] provenance")
    log(f"    attn calib    : {ATTN_CALIB}")
    log(f"      sha256={sha256_file(ATTN_CALIB)}  split={calA['calibration_split']}"
        f"  train={len(calA['calibration_samples'])}"
        f"  val={len(calA['validation_samples'])}")
    log(f"    postnorm calib: {POST_CALIB}")
    log(f"      sha256={sha256_file(POST_CALIB)}  split={calP['calibration_split']}"
        f"  train={calP['provenance']['train_count']}"
        f"  val={calP['provenance']['validation_count']}")
    log(f"    weights sha    : attn={calA['weights_sha256'][:16]}…"
        f"  post={calP['provenance']['weights_sha256'][:16]}…"
        f"  local={sha256_file(WEIGHTS)[:16]}…")
    same_weights = (calA["weights_sha256"] == calP["provenance"]["weights_sha256"]
                    == sha256_file(WEIGHTS))
    log(f"    same checkpoint: {same_weights}")
    results["provenance"] = {
        "attn_calib_sha256": sha256_file(ATTN_CALIB),
        "postnorm_calib_sha256": sha256_file(POST_CALIB),
        "attn_train": len(calA["calibration_samples"]),
        "attn_val": len(calA["validation_samples"]),
        "postnorm_train": calP["provenance"]["train_count"],
        "postnorm_val": calP["provenance"]["validation_count"],
        "same_weights_sha256": bool(same_weights),
    }

    # ---------------------------------------------------------------- [2]
    log("\n[2] split overlap (official manifests)")
    npz = resolve_npz()
    with np.load(npz, allow_pickle=True) as z:
        features = z["features"]
        partitions = z["partitions"]
        paths = np.array([str(p) for p in z["paths"]])
    def norm_paths(seq):
        return {str(p).lstrip("./") for p in seq}
    npz_train_paths = norm_paths(paths[partitions == "training"])
    npz_val_paths = norm_paths(paths[partitions == "validation"])
    manifest = json.loads(MANIFEST.read_text())
    attn_train_files = [s["file"] for s in calA["calibration_samples"]]
    man_by_file = {s["encoder_file"]: s for s in manifest["samples"]}
    attn_train_wavs = {man_by_file[f]["archive_member"].lstrip("./") for f in attn_train_files}
    attn_val_wavs = {man_by_file[f]["archive_member"].lstrip("./") for f in calA["validation_samples"]}

    log(f"    npz split: training={len(npz_train_paths)} validation={len(npz_val_paths)}")
    log(f"    attn training wavs ⊂ npz training : "
        f"{attn_train_wavs <= npz_train_paths}  ({sorted(attn_train_wavs)})")
    log(f"    attn validation wavs ⊂ npz validation : "
        f"{attn_val_wavs <= npz_val_paths}  ({sorted(attn_val_wavs)})")
    log(f"    train/val disjoint in npz : {not (npz_train_paths & npz_val_paths)}")
    same_split = (attn_train_wavs <= npz_train_paths) and (attn_val_wavs <= npz_val_paths)
    results["split_overlap"] = {
        "attn_train_subset_of_npz_train": bool(attn_train_wavs <= npz_train_paths),
        "attn_val_subset_of_npz_val": bool(attn_val_wavs <= npz_val_paths),
        "same_official_split": bool(same_split),
        "npz_train": len(npz_train_paths), "npz_val": len(npz_val_paths),
    }

    # encoder inputs for the training / validation splits
    weights = {k: v for k, v in np.load(WEIGHTS, allow_pickle=False).items()}
    proj = load_projection(weights)
    train_ids = np.flatnonzero(partitions == "training")
    val_ids = np.flatnonzero(partitions == "validation")
    if args.max_val:
        val_ids = val_ids[:args.max_val]

    X_train = [mfcc_to_encoder(features[i], proj) for i in train_ids]
    train_max = max(float(np.abs(x).max()) for x in X_train)
    post_sx = calP["scales"]["s_x"]
    log(f"\n[3] encoder-input (X) scale — the SAME tensor feeds attention QKV and")
    log("    the postnorm residual")
    log(f"    postnorm training set : {len(train_ids)} samples  max|X|={train_max:.9g}"
        f"  -> s_x      = {post_sx:.9g}")
    log(f"    attn    training set : {len(attn_train_files)} samples"
        f" (subset)  max|X|={calA['scale_x']*2047:.9g}  -> scale_x = {calA['scale_x']:.9g}")
    ratio = post_sx / calA["scale_x"]
    log(f"    CONFLICT: scale_x != s_x  (ratio {ratio:.6f}x, i.e. {np.log2(ratio):.3f} bits)")
    # verify the 3 attn training samples' encoder input agrees between sources
    e2e = []
    for f in attn_train_files:
        wav = man_by_file[f]["archive_member"]
        idx = int(np.flatnonzero(paths == "./" + wav)[0])
        x_npz = mfcc_to_encoder(features[idx], proj)
        x_npy = np.load(REAL_DIR / f, allow_pickle=False).astype(np.float64)
        e2e.append(float(np.abs(x_npz - x_npy).max()))
    log(f"    cross-source X agreement (npz-projected vs real_input_chain npy): "
        f"max|Δ|={max(e2e):.3g}")
    results["input_scale"] = {
        "postnorm_s_x": post_sx, "postnorm_train_n": int(len(train_ids)),
        "postnorm_train_max_abs": train_max,
        "attn_scale_x": calA["scale_x"], "attn_train_n": len(attn_train_files),
        "attn_train_max_abs": calA["scale_x"] * 2047,
        "ratio_post_over_attn": ratio, "conflict": bool(abs(ratio - 1) > 1e-6),
        "cross_source_max_abs_diff": max(e2e),
    }

    # ---------------------------------------------------------------- [4]
    log("\n[4] quantisation boundaries")
    log("    attn  : ACT_BITS=12 but quantize_activation clamps to the int16")
    log("            container ±32767/±32768 -> values |Xi|>2047 are legal")
    log("    post  : CLAMP_A=2047 hard clamp in quantize_stimulus/residual/LN/GELU")
    log("    RTL   : requant o->a saturates at int16 ±32768, not at 2047")
    oor = {"attn_scale_elems": 0, "attn_scale_samples": 0, "coherent_samples": 0}
    for x in X_train:
        xi_a = A.quantize_activation(x, calA["scale_x"])
        bad = int((np.abs(xi_a) > 2047).sum())
        oor["attn_scale_elems"] += bad
        oor["attn_scale_samples"] += int(bad > 0)
        if int(np.abs(A.quantize_activation(x, post_sx)).max()) > 2047:
            oor["coherent_samples"] += 1
    log(f"    under the attn 3-sample scale, training samples exceeding A12: "
        f"{oor['attn_scale_samples']}/{len(X_train)}"
        f" ({oor['attn_scale_elems']} element(s) with |Xi|>2047)")
    log(f"    under the coherent scale {post_sx:.6g}: "
        f"{oor['coherent_samples']}/{len(X_train)} training samples exceed A12")
    results["boundaries"] = {
        "attn_container": "int16 (±32768)", "attn_nominal": 2047,
        "postnorm_clamp_a": 2047,
        "attn_scale_train_samples_over_2047": int(oor["attn_scale_samples"]),
        "attn_scale_train_elems_over_2047": int(oor["attn_scale_elems"]),
        "coherent_scale_train_samples_over_2047": int(oor["coherent_samples"]),
        "coherent_scale_train_n": int(len(X_train)),
    }

    # ---------------------------------------------------------------- [5]
    log("\n[5] RTL constants vs frozen models")
    def parse_hdr(path):
        d = {}
        if not path.exists():
            return d
        for m in re.finditer(r"localparam int (\w+)\s*=\s*(-?\d+)", path.read_text()):
            d[m.group(1)] = int(m.group(2))
        return d
    hdr_base = parse_hdr(ATTN_HDR_BASE)
    hdr_r1 = parse_hdr(ATTN_HDR_ROUND1)
    attn_keys = ["KC", "SH_KC", "RQ", "RK", "RV", "RC", "RO"]
    attn_rtl_diff = {k: [hdr_base.get(k), calA[k]] for k in attn_keys
                     if hdr_base.get(k) != calA[k]}
    log(f"    attn baseline header {ATTN_HDR_BASE.name}: "
        f"KC={hdr_base.get('KC')} SH_KC={hdr_base.get('SH_KC')} (round-1 header: "
        f"KC={hdr_r1.get('KC')} SH_KC={hdr_r1.get('SH_KC')})")
    log(f"    frozen attn model KC={calA['KC']} SH_KC={calA['SH_KC']}; "
        f"stale fields vs baseline header: {attn_rtl_diff}")
    m = re.search(r"M_A\((\d+)\)", TB_ENCODER.read_text())
    rtl_MA = int(m.group(1)) if m else None
    log(f"    tb_encoder.sv M_A = {rtl_MA}")

    # postnorm integer constants: baseline header vs the training-frozen model
    calP_d = P.calibrate(frozen=frP)          # postnorm integer coefficients
    post_keys = ["MA1", "MB1", "MGN", "MO1", "MC2", "MO2", "EPS_Q1", "EPS_Q2"]
    hdr_post = {}
    if POST_HDR.exists():
        for m in re.finditer(r"localparam logic signed \[\d+:\d+\] (\w+)\s*=\s*(?:48|32)'sd(-?\d+);",
                             POST_HDR.read_text()):
            hdr_post[m.group(1)] = int(m.group(2))
    model_post = {k: int(calP_d["scalars"][k.lower()]) for k in post_keys}
    post_same = [k for k in post_keys if hdr_post.get(k) == model_post[k]]
    post_stale = {k: [hdr_post.get(k), model_post[k]] for k in post_keys
                  if hdr_post.get(k) != model_post[k]}
    log(f"    postnorm header {POST_HDR.name} vs training-frozen model: "
        f"{len(post_same)}/{len(post_keys)} match {post_same}; stale: {post_stale}")

    # naming trap: the vector generator reads a same-named-but-different scale
    genv = ROOT / "graduate/encoder/vectors/gen_vectors.py"
    genv_reads_sx = ('cal["scales"]["sx"]' in genv.read_text()) if genv.exists() else None
    sx_alt = float(P.calibrate()["scales"]["sx"])          # what that key really holds
    ffn1_act = float(np.load(FFN1_PACK, allow_pickle=False)["layer01_activation_scale"])
    log(f"    naming trap: {genv.name} emits meta.sx from cal[\"scales\"][\"sx\"] "
        f"(key present in source: {genv_reads_sx})")
    log(f"      that key = {sx_alt:.9g} != encoder-input s_x = {post_sx:.9g}; it is the "
        f"FFN1-pack activation scale {ffn1_act:.9g} (equal: {sx_alt == ffn1_act})")

    results["rtl_constants"] = {
        "attn_header_baseline": {k: hdr_base.get(k) for k in attn_keys},
        "attn_header_round1": {k: hdr_r1.get(k) for k in attn_keys},
        "attn_model_frozen": {k: calA[k] for k in attn_keys},
        "attn_stale_vs_baseline": attn_rtl_diff,
        "postnorm_header": {k: hdr_post.get(k) for k in post_keys},
        "postnorm_model_frozen": model_post,
        "postnorm_matching": post_same,
        "postnorm_n_matching": len(post_same),
        "naming_trap_gen_vectors_sx": {
            "gen_vectors_reads_cal_scales_sx": genv_reads_sx,
            "key_value": sx_alt, "encoder_input_s_x": post_sx,
            "ffn1_pack_activation_scale": ffn1_act, "equals_ffn1_pack": bool(sx_alt == ffn1_act),
        },
        "tb_M_A": rtl_MA,
    }

    # ---------------------------------------------------------------- [6]
    log("\n[6] conflict proof — naive merge of the two round-1 artefacts")
    log("    The RTL feeds the attention-quantised X into the postnorm residual")
    log("    (kwt_encoder_layer x_buf <- attn rd_sel 0) while MA1 is derived from")
    log("    the postnorm s_x.  With scale_x != s_x the residual x-term is")
    log(f"    multiplied by s_x/scale_x = {ratio:.4f}.")
    wfA = A.load_weights()
    wiA, swA = A.quantize_weights(wfA)

    mism_rows = []
    for name in ("sample_00_encoder.npy", "sample_02_encoder.npy", "sample_05_encoder.npy"):
        x = np.load(REAL_DIR / name, allow_pickle=False).astype(np.float64)
        _, ref = P.fp32_activations(x)
        sc = calP["scales"]
        # hardware: x_buf at attn scale_x, a at requant(attn Oi); multipliers from s_x
        outA = A.integer_forward(x, calA, wiA)
        M_A = 5981                             # stale tb constant (informational)
        a_rows = np.clip((outA["Oi"].astype(np.int64) * M_A + (1 << 15)) >> 16,
                         -32768, 32767).astype(np.int64)
        x_buf = A.quantize_activation(x, calA["scale_x"])
        pre = P.integer_prefix(calP_d, x_buf, a_rows)
        r1 = pre["r1"] * sc["s_r1"]
        nrm = nrmse(r1, ref["r1"])
        mism_rows.append({"sample": name, "r1_nrmse": nrm})
        log(f"    {name}: mismatched residual r1 NRMSE = {nrm*100:.3f}%  "
            f"(limit {NRMSE_LIMIT*100:.0f}%)")
    worst_mismatch = max(r["r1_nrmse"] for r in mism_rows)
    log(f"    worst mismatched r1 NRMSE = {worst_mismatch*100:.3f}%  "
        f"-> {'INCOMPATIBLE' if worst_mismatch > NRMSE_LIMIT else 'ok'}")
    results["conflict"] = {
        "reason": "attention scale_x drives x_buf but MA1 is derived from postnorm s_x",
        "ratio_post_over_attn": ratio, "stale_tb_M_A": 5981,
        "rows": mism_rows, "worst_r1_nrmse": worst_mismatch,
        "exceeds_limit": bool(worst_mismatch > NRMSE_LIMIT),
    }

    # ---------------------------------------------------------------- [7]
    log("\n[7] coherent candidate — ONE encoder-input scale on the shared TRAINING split")
    s_x_coh = train_max / 2047.0
    log(f"    union training max|X| = {train_max:.9g}  ->  s_x_coh = {s_x_coh:.9g}")
    log(f"    (attn's 3 training samples are a subset; their max "
        f"{calA['scale_x']*2047:.6g} < {train_max:.6g})")
    calA_coh = attn_refit(A, X_train, s_x_coh, wfA, wiA, swA)
    log("    attention refit on 240 training samples:")
    log(f"      scale_x={calA_coh['scale_x']:.9g} RQ={calA_coh['RQ']} RK={calA_coh['RK']}"
        f" RV={calA_coh['RV']} RC={calA_coh['RC']} RO={calA_coh['RO']}"
        f" KC={calA_coh['KC']} SH_KC={calA_coh['SH_KC']} scale_o={calA_coh['scale_o']:.9g}")
    s_a = calP["scales"]["s_a"]
    M_A_coh = int(round(calA_coh["scale_o"] / s_a * (1 << 16)))
    log(f"    postnorm scales kept (s_x already {s_x_coh:.9g}); M_A = "
        f"round(scale_o/s_a*2^16) = {M_A_coh}")

    # evaluate the coherent hardware-faithful chain on the VALIDATION split
    val_coh, val_sw = [], []
    for i in val_ids:
        x = mfcc_to_encoder(features[i], proj)
        val_coh.append(eval_chain(A, P, calA_coh, calP_d, x, wiA, True, (M_A_coh, 16)))
        val_sw.append(eval_chain(A, P, calA_coh, calP_d, x, wiA, False, None))
    real_coh = []
    for name in ("sample_00_encoder.npy", "sample_02_encoder.npy", "sample_05_encoder.npy"):
        x = np.load(REAL_DIR / name, allow_pickle=False).astype(np.float64)
        r = eval_chain(A, P, calA_coh, calP_d, x, wiA, True, (M_A_coh, 16))
        r["sample"] = name
        real_coh.append(r)
    worst_coh = max([r["worst_nrmse"] for r in val_coh] + [r["worst_nrmse"] for r in real_coh])
    worst_sw = max(r["worst_nrmse"] for r in val_sw)
    mean_stage = {n: float(np.mean([r["stages"][n] for r in val_coh]))
                  for n in ("r1", "g", "r2", "y")}
    log(f"    validation (dataset {len(val_coh)} samples + 3 real):")
    log(f"      coherent hardware chain worst NRMSE = {worst_coh*100:.4f}%")
    log(f"      coherent software chain worst NRMSE = {worst_sw*100:.4f}%")
    log(f"      mean stage (hw chain) r1={mean_stage['r1']*100:.4f}% g={mean_stage['g']*100:.4f}%"
        f" r2={mean_stage['r2']*100:.4f}% y={mean_stage['y']*100:.4f}%")
    xmax = max(r["x_int_max"] for r in val_coh + real_coh)
    amax = max(r["a_int_max"] for r in val_coh + real_coh)
    xclip = sum(r["x_clip"] for r in val_coh + real_coh)
    log(f"      max |X_int|={xmax} (<=2047? {xmax <= 2047})   max |a_int|={amax}")
    log(f"      validation X-quantisation clips at ±2047: {xclip} element(s)")
    coh_ok = worst_coh < NRMSE_LIMIT and xmax <= 2047
    results["coherent_candidate"] = {
        "s_x_coh": s_x_coh, "train_n": int(len(X_train)), "train_max_abs": train_max,
        "attn": {k: calA_coh[k] for k in ("scale_x", "RQ", "RK", "RV", "RC", "RO",
                                          "KC", "SH_KC", "scale_o")},
        "postnorm_scales": calP["scales"], "M_A": M_A_coh, "Q_A": 16,
        "validation_n": len(val_coh), "real_n": len(real_coh),
        "hw_chain_worst_nrmse": worst_coh, "sw_chain_worst_nrmse": worst_sw,
        "mean_stage_nrmse": mean_stage,
        "val_x_int_max": xmax, "val_a_int_max": amax,
        "val_x_clip_2047": int(xclip),
        "below_limit": bool(worst_coh < NRMSE_LIMIT),
    }
    # persist the candidate attention header + frozen json (proposal only)
    cand = HERE / "candidate"
    cand.mkdir(exist_ok=True)
    (cand / "attn_params.svh").write_text(
        "// PROPOSED coherent attention constants (r2_scale_contract).\n"
        "// Frozen on the 240-sample TRAINING split (gradient-free); NOT yet\n"
        "// integrated into third_party/ and NOT yet RTL-reverified for the\n"
        "// full encoder chain.\n"
        f"localparam int ATT_T  = 99;\nlocalparam int ATT_D  = 64;\n"
        f"localparam int F_BITS = 6;\nlocalparam int M_LUT  = 14;\n"
        f"localparam int G_SH   = 20;\n"
        f"localparam int KC    = {calA_coh['KC']};\n"
        f"localparam int SH_KC = {calA_coh['SH_KC']};\n"
        f"localparam int RQ    = {calA_coh['RQ']};\nlocalparam int RK = {calA_coh['RK']};\n"
        f"localparam int RV    = {calA_coh['RV']};\nlocalparam int RC = {calA_coh['RC']};\n"
        f"localparam int RO    = {calA_coh['RO']};\n")
    (cand / "coherent_frozen_calib.json").write_text(json.dumps({
        "schema": "r2-coherent-calib/1",
        "calibration_split": "training",
        "validation_used_for_calibration": False,
        "gradient_free": True,
        "shared_input_scale": "s_x == attention scale_x (union training max/2047)",
        "attention": calA_coh,
        "postnorm_scales": calP["scales"],
        "M_A": M_A_coh, "Q_A": 16,
        "provenance": {
            "attn_calib": str(ATTN_CALIB), "postnorm_calib": str(POST_CALIB),
            "dataset": str(npz), "dataset_sha256": sha256_file(npz),
            "train_count": int(len(train_ids)),
        },
    }, ensure_ascii=False, indent=2) + "\n")
    log(f"    wrote {cand/'attn_params.svh'} and {cand/'coherent_frozen_calib.json'}")

    # ---------------------------------------------------------------- gate
    gate = {
        "both_train_frozen": calA["calibration_split"] == "training"
                             and calP["calibration_split"] == "training",
        "same_weights": bool(same_weights),
        "same_official_split": bool(same_split),
        "no_val_in_attn_calib": not (set(attn_train_files) & set(calA["validation_samples"])),
        "scale_conflict_present": bool(abs(ratio - 1) > 1e-6),
        "naive_merge_fails": bool(worst_mismatch > NRMSE_LIMIT),
        "coherent_candidate_ok": bool(coh_ok),
    }
    log("\n[gate]")
    for k, v in gate.items():
        log(f"    {'PASS' if v else 'FAIL'}  {k}")
    ok = all(gate.values())
    results["gate"] = gate
    results["evidence_levels"] = {
        "software_reference": "yes (this driver, results.json)",
        "rtl_bit_exact": "no in this driver; see rtl_check_coherent.sh (attention only)",
        "e203": "no",
        "synthesis": "no",
        "official_acceptance": "no",
    }
    results["runtime_s"] = round(time.time() - t0, 2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    (HERE / "logs").mkdir(exist_ok=True)
    (HERE / "logs/cross_check.log").write_text("\n".join(LOG_LINES) + "\n")

    verdict = "COMPOSABLE_ON_SHARED_TRAINING_SPLIT" if ok else "NOT_COMPOSABLE"
    print(f"\nSCALE_CONTRACT_{'PASS' if ok else 'FAIL'}  verdict={verdict}")
    print(f"results={args.out}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
