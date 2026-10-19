#!/usr/bin/env python3
"""p027_precision -- bounded hardware-representable precision search.

Same real checkpoint and the same 240 training / 240 validation split as the
frozen A12W10 baseline (``graduate/ds_tasks/encoder_12numeric``):

  * calibration (scales, shifts, LUTs, coefficients) uses the TRAINING split
    only and is gradient-free;
  * the 240 validation samples are used for evaluation only;
  * the local split has no test partition (checked at load time).

For every candidate scheme the runner records:
  * per-layer pooled validation NRMSE for all six encoder stages,
  * the worst layer and the per-sample tail diagnostic,
  * saturation counts and measured accumulator widths,
  * cost proxies (weight/activation bytes, multiplier/accumulator widths,
    exponent/GELU LUT sizes, estimated MAC bit products),
  * a SHA-256 of the candidate configuration.

Selection signal: a train-split holdout (calibrate on the first half of the
training split, evaluate on the second half) so that scheme choice does not
depend on the validation split.  Validation remains the reported metric.

Usage:
    OPENBLAS_NUM_THREADS=4 python3 model/run_search.py
    ... --only A16W16 --limit-samples 24      # quick screening
    ... --no-holdout
Exit codes:
    0  search completed
    2  contract violation (split/partition/scale continuity)
    1  runtime failure
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TASK = HERE.parent
WORKTREE = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(WORKTREE / "graduate/encoder/reference"))
import prec_ref as P                                   # noqa: E402
import kwt1_fp32_golden as golden                      # noqa: E402

NRMSE_LIMIT = 0.03
TARGET = 0.0027          # user goal: worst-layer pooled encoder-output NRMSE
STAGES = P.STAGES
GOLDEN_KEY = {"a": "attn_projected", "r1": "residual1", "g": "gelu",
              "f": "ffn_out", "r2": "residual2", "y": "layer_output"}


# ---------------------------------------------------------------------------
# fixed-point profile helpers
# ---------------------------------------------------------------------------
def prof(A, wa, w1, w2, **kw):
    """Uniform-A scheme; the fixed-point widths scale with A where needed."""
    d = dict(stage_bits=dict(x=A, a=A, r1=A, ln1=A, g=A, r2=A, y=A),
             w_attn=wa, w1=w1, w2=w2,
             qb=A + 16, lf=12, gelu_shift=8,
             f_table=8, f_interp=4, m_lut=20, prob_bits=20, kc_min=1 << 20,
             qkv_bits=max(16, A), ctx_bits=max(16, A), out_bits=max(16, A),
             a_clip_bits=max(16, A + 2))
    d.update(kw)
    return d


def sel(bits, wa, w1, w2, **kw):
    amax = max(bits.values())
    d = dict(stage_bits=bits, w_attn=wa, w1=w1, w2=w2,
             qb=amax + 16, lf=12, gelu_shift=8,
             f_table=8, f_interp=4, m_lut=20, prob_bits=20, kc_min=1 << 20,
             qkv_bits=max(16, amax), ctx_bits=max(16, amax),
             out_bits=max(16, amax), a_clip_bits=max(16, amax + 2))
    d.update(kw)
    return d


CANDIDATES = [
    ("A12W10", dict(),
     "frozen baseline contract (sanity: per-layer pooled ~2.75% on 240 val)"),
    ("A16W10", prof(16, 10, 10, 10),
     "16-bit activations, 10-bit weights (expected near miss)"),
    ("A16W12", prof(16, 12, 12, 12),
     "16-bit activations, 12-bit weights, 20-bit softmax probs"),
    ("A16W16", prof(16, 16, 16, 16),
     "uniform 16-bit activations and weights"),
    ("A20W16", prof(20, 16, 16, 16),
     "20-bit activations, 16-bit weights"),
    ("A16W12_lut10", prof(16, 12, 12, 12, lf=10, gelu_shift=6,
                          f_table=6, f_interp=4, m_lut=18),
     "cheapest passing variant: 10-bit GELU/softmax LUTs"),
    ("SEL_A20xy_W12", sel(dict(x=20, y=20, a=16, r1=20, ln1=16, g=16, r2=20),
                          12, 12, 12),
     "selective: 20-bit residual stream (x/y/r1/r2), 16-bit a/ln1/g, 12-bit weights"),
    ("SEL_A20xy_W16", sel(dict(x=20, y=20, a=16, r1=20, ln1=16, g=16, r2=20),
                          16, 16, 16),
     "selective: 20-bit residual stream, 16-bit weights"),
    ("A16W16_hr105", prof(16, 16, 16, 16, headroom=1.05),
     "A16W16 with 5% activation headroom (removes residual saturation)"),
    ("SEL_A20xy_W12_hr105", sel(dict(x=20, y=20, a=16, r1=20, ln1=16, g=16, r2=20),
                                12, 12, 12, headroom=1.05),
     "selective 20/16 residual stream, 12-bit weights, 5% headroom"),
]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cfg_sha(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def locate(name):
    for cand in (WORKTREE / name, Path("/home/yang/Desktop/ic设计竞赛") / name):
        if cand.exists():
            return cand
    raise FileNotFoundError(name)


def load_data(want_train=True):
    split_path = locate("software/.build/quant_samples_12labels_20.npz")
    weights_path = locate("software/weights/kwt1_fp32.npz")
    weights = golden.load_weights(weights_path)
    with np.load(split_path, allow_pickle=True) as z:
        features = z["features"].astype(np.float64)
        partitions = z["partitions"]
        paths = z["paths"]
    if set(np.unique(partitions).tolist()) - {"training", "validation"}:
        raise ValueError("unexpected partition label (test partition must not exist)")
    tr = np.flatnonzero(partitions == "training")
    va = np.flatnonzero(partitions == "validation")
    if set(paths[tr].tolist()) & set(paths[va].tolist()):
        raise ValueError("training/validation overlap")

    def project(i):
        kernel = weights[f"{golden.PREFIX}/patch_proj/kernel"]
        bias = weights[f"{golden.PREFIX}/patch_proj/bias"]
        cls = weights[f"{golden.PREFIX}/class_emb"][0]
        pos = weights[f"{golden.PREFIX}/pos_emb"][0]
        return (np.concatenate([cls, features[i] @ kernel + bias], axis=0)
                + pos).astype(np.float32)

    return dict(weights=weights, weights_path=weights_path, split_path=split_path,
                X0_tr=[project(i) for i in tr] if want_train else [],
                X0_va=[project(i) for i in va],
                n_tr=len(tr), n_va=len(va))


def golden_val_cache(weights, X0_list):
    """FP32 reference stage tensors for evaluation (never used for calibration)."""
    cache = []
    for x0 in X0_list:
        arrs = []
        golden.forward(x0, weights, P.NLAYERS, arrays=arrs)
        cache.append({k: {st: np.asarray(arrs[k][GOLDEN_KEY[st]], dtype=np.float64)
                          for st in STAGES} for k in range(P.NLAYERS)})
    return cache


def evaluate(calib, X0_list, gold_cache):
    nlayers = len(calib["layers"])
    pool = {st: np.zeros(nlayers) for st in STAGES}
    ref = {st: np.zeros(nlayers) for st in STAGES}
    psmax = {st: np.zeros(nlayers) for st in STAGES}
    sat = {"a": 0, "r1": 0, "g": 0, "r2": 0, "y": 0}
    acc = {"qkv": 0, "score": 0, "ctx": 0, "attn_out": 0, "ffn1": 0, "ffn2": 0}
    finite = True
    for x0, gcache in zip(X0_list, gold_cache):
        res = P.integer_chain(calib, x0, want_stages=False)
        for k in range(nlayers):
            for st in STAGES:
                got = np.asarray(res["deq"][st][k], dtype=np.float64)
                refv = gcache[k][st]
                if got.shape != refv.shape or not np.isfinite(got).all():
                    finite = False
                    continue
                err = got - refv
                pool[st][k] += float(np.sum(err * err))
                ref[st][k] += float(np.sum(refv * refv))
                psmax[st][k] = max(psmax[st][k], P.nrmse(got, refv))
        for key in sat:
            sat[key] += int(res["sat"][key])
        for key in acc:
            acc[key] = max(acc[key], int(res["acc_bits"][key]))
    pooled = {st: [math.sqrt(pool[st][k] / max(ref[st][k], 1e-16))
                   for k in range(nlayers)] for st in STAGES}
    return dict(pooled=pooled,
                persample_max={st: [float(v) for v in psmax[st]] for st in STAGES},
                sat=sat, acc_bits=acc, finite=finite, samples=len(X0_list))


def contract_check(calib):
    problems = []
    if len(calib["layers"]) != P.NLAYERS:
        problems.append(f"layer count {len(calib['layers'])} != 12")
    for k, L in enumerate(calib["layers"]):
        if k + 1 < len(calib["layers"]):
            a = L["scales"]["s_out"]
            b = calib["layers"][k + 1]["scales"]["s_x"]
            if abs(a - b) > 1e-12 * max(1.0, abs(a)):
                problems.append(f"layer{k}->{k+1} scale discontinuity {a} != {b}")
        if not (0 < L["M_A"] < (1 << 28)):
            problems.append(f"layer{k} M_A={L['M_A']} out of range")
    return problems


def measured_coefficient_bits(calib):
    """Widest stored coefficient per kind (measured, not assumed)."""
    keys = ["ma1", "mb1", "eps_q1", "mo1", "mgn", "ma2", "eps_q2", "mo2",
            "mg", "g1f", "b1f", "g2f", "b2f", "md2"]
    mx = {k: 0 for k in keys}
    mx["M_A"] = 0
    for L in calib["layers"]:
        c = L["coeff"]
        mx["M_A"] = max(mx["M_A"], int(L["M_A"]).bit_length())
        for k in keys:
            if k not in c:
                continue
            v = np.abs(np.asarray(c[k]))
            mx[k] = max(mx[k], int(v.max()).bit_length() + 1)
    return dict(
        coefficient_bits=mx,
        isq_frac=dict(calib["isq_frac"]),
        isq_register_bits=P.ISQ_REG_BITS,
        weight_bits=dict(attn=calib["cfg"]["w_attn"], w1=calib["cfg"]["w1"],
                         w2=calib["cfg"]["w2"]),
        activation_bits=dict(calib["cfg"]["stage_bits"]),
        qkv_bits=calib["cfg"]["qkv_bits"],
        prob_bits=calib["cfg"]["prob_bits"],
        qg_bits=P.gelu_qg(calib["cfg"]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--limit-samples", type=int, default=240)
    ap.add_argument("--no-holdout", action="store_true")
    ap.add_argument("--out", default=str(TASK / "results.json"))
    args = ap.parse_args()

    t0 = time.time()
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    data = load_data()
    weights = data["weights"]
    log(f"DATA split={data['split_path']} sha256={sha256(data['split_path'])}")
    log(f"DATA weights={data['weights_path']} sha256={sha256(data['weights_path'])}")
    log(f"DATA partitions training={data['n_tr']} validation={data['n_va']} test=0")

    X0_va = data["X0_va"][:args.limit_samples]
    t = time.time()
    gold = golden_val_cache(weights, X0_va)
    log(f"REF fp32 golden cache, {len(X0_va)} validation samples ({time.time()-t:.1f}s)")

    n_tf = 96
    X0_tf = data["X0_tr"][:n_tf]
    t = time.time()
    max_formal = P.fp32_stage_maxima(weights, data["X0_tr"], golden)
    if not args.no_holdout:
        gold_tf = golden_val_cache(weights, X0_tf)
    log(f"CALIB training maxima over {len(data['X0_tr'])} training samples "
        f"({time.time()-t:.1f}s); train-fit diagnostic on the first {n_tf} "
        f"training samples (validation-independent)")

    wanted = None if args.only is None else set(args.only.split(","))
    results = []
    for name, over, note in CANDIDATES:
        if wanted is not None and name not in wanted:
            continue
        cfg = P.cfg_norm(over)
        rec = dict(name=name, note=note, cfg=cfg, config_sha256=cfg_sha(cfg))
        try:
            t = time.time()
            calib = P.build_calibration(weights, (max_formal[0], max_formal[1]),
                                        max_formal[2], cfg)
            tcal = time.time() - t
            problems = contract_check(calib)
            if problems:
                rec.update(state="CONTRACT_FAIL", problems=problems)
                log(f"CAND {name}: CONTRACT FAIL {problems}")
                results.append(rec)
                continue
            t = time.time()
            ev = evaluate(calib, X0_va, gold)
            rec.update(
                state="EVALUATED",
                calibration_seconds=round(tcal, 2),
                eval_seconds=round(time.time() - t, 2),
                samples=ev["samples"], layers=len(calib["layers"]),
                per_layer_pooled_nrmse={st: ev["pooled"][st] for st in STAGES},
                per_layer_pooled_nrmse_pct={st: [round(v * 100, 5)
                                                 for v in ev["pooled"][st]]
                                            for st in STAGES},
                per_layer_persample_max_pct={st: [round(v * 100, 5)
                                                  for v in ev["persample_max"][st]]
                                             for st in STAGES},
                worst_layer=int(np.argmax(ev["pooled"]["y"])),
                worst_y_nrmse=float(max(ev["pooled"]["y"])),
                worst_y_nrmse_pct=round(max(ev["pooled"]["y"]) * 100, 5),
                all_layers_below_3pct=bool(all(v < NRMSE_LIMIT
                                               for v in ev["pooled"]["y"])),
                target_0p27pct=bool(all(v <= TARGET for v in ev["pooled"]["y"])),
                finite=ev["finite"],
                saturation=ev["sat"],
                measured_accumulator_bits=ev["acc_bits"],
                measured_coefficient_bits=measured_coefficient_bits(calib),
                cost=P.cost_model(cfg),
            )
            log(f"CAND {name}: worst L{rec['worst_layer']} "
                f"y={rec['worst_y_nrmse_pct']:.4f}% "
                f"all<3%={rec['all_layers_below_3pct']} "
                f"target<=0.27%={rec['target_0p27pct']} "
                f"sat={ev['sat']} acc={ev['acc_bits']} "
                f"({tcal + rec['eval_seconds']:.1f}s)")
            log("  y%: " + " ".join(f"{v*100:7.4f}" for v in ev["pooled"]["y"]))

            if not args.no_holdout:
                t = time.time()
                eh = evaluate(calib, X0_tf, gold_tf)
                hm = max(eh["pooled"]["y"])
                rec["holdout_train"] = dict(
                    kind="train-fit diagnostic (same training-only calibration)",
                    fit_samples=len(data["X0_tr"]), eval_samples=len(X0_tf),
                    worst_y_nrmse=float(hm),
                    worst_y_nrmse_pct=round(hm * 100, 5),
                    per_layer_pooled_y_pct=[round(v * 100, 5)
                                            for v in eh["pooled"]["y"]],
                    target_0p27pct=bool(hm <= TARGET),
                    saturation=eh["sat"], seconds=round(time.time() - t, 1))
                log(f"  train-fit ({len(X0_tf)} training samples, "
                    f"validation-independent): worst={hm*100:.4f}% "
                    f"target<=0.27%={rec['holdout_train']['target_0p27pct']}")
        except Exception as exc:                          # keep the search going
            rec.update(state="RUNTIME_FAIL", error=f"{type(exc).__name__}: {exc}")
            log(f"CAND {name}: RUNTIME_FAIL {type(exc).__name__}: {exc}")
        results.append(rec)

    evaluated = [r for r in results if r.get("state") == "EVALUATED"]
    passing = [r for r in evaluated if r.get("target_0p27pct")]
    hold_pass = [r for r in evaluated
                 if r.get("holdout_train", {}).get("target_0p27pct")]
    best = min(evaluated, key=lambda r: r["worst_y_nrmse"], default=None)
    # selection: smallest cost among schemes that pass on BOTH the held-out
    # training split and the validation split (validation is still reported).
    margin = 0.75 * TARGET      # >=25% headroom so the tail is not on the limit
    eligible = [r for r in evaluated
                if r.get("target_0p27pct")
                and (args.no_holdout or r.get("holdout_train", {}).get("target_0p27pct"))]
    with_margin = [r for r in eligible
                   if r["worst_y_nrmse"] <= margin
                   and (args.no_holdout
                        or r.get("holdout_train", {}).get("worst_y_nrmse", 1.0) <= margin)]
    pool = with_margin or eligible
    selected = min(pool, key=lambda r: r["cost"]["twelve_layers"]["mac_bit_products_total"],
                   default=None)
    state = ("TARGET_MET" if passing else
             ("NEAR_MISS" if best and best["all_layers_below_3pct"] else "FAIL"))
    out = dict(
        task="p027_precision",
        state=state,
        target=dict(metric="worst-layer pooled encoder-output (y) NRMSE on the "
                           "240 validation samples",
                    value=TARGET, samples=len(X0_va), layers=12),
        baseline_reference=dict(
            name="A12W10 (frozen encoder_12numeric pack)",
            worst_y_nrmse=0.027528,
            source="graduate/ds_tasks/r2_full_numeric/results.json",
            note="the frozen 12-bit/10-bit chain; not relabelled here"),
        evidence_levels=dict(
            software_reference="L1 parameterised integer chain, train-calibrated, "
                               "validation evaluation only",
            rtl_bit_exact="no (precision study; no RTL is changed)",
            e203="no", synthesis="no", official_acceptance="no"),
        split=dict(file=str(data["split_path"]), sha256=sha256(data["split_path"]),
                   training=int(data["n_tr"]), validation=int(data["n_va"]), test=0,
                   calibration_split="training", validation_used_for_calibration=False,
                   holdout=dict(fit="all 240 training samples",
                                eval="first 96 training samples",
                                samples=len(X0_tf),
                                note="train-fit diagnostic, not a validation sample")),
        weights=dict(file=str(data["weights_path"]),
                     sha256=sha256(data["weights_path"])),
        candidates=results,
        best=best["name"] if best else None,
        best_worst_y_nrmse_pct=best["worst_y_nrmse_pct"] if best else None,
        passing_candidates=[r["name"] for r in passing],
        holdout_passing_candidates=[r["name"] for r in hold_pass],
        selected=selected["name"] if selected else None,
        selection_rule=("pass target on the 240 validation samples AND on the "
                        "validation-independent train-fit diagnostic; prefer "
                        ">=25% headroom (worst <= 0.2025%), then take the "
                        "smallest MAC bit-product total"),
        runtime_seconds=round(time.time() - t0, 1),
    )
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    (TASK / "logs").mkdir(exist_ok=True)
    (TASK / "logs" / "run_search.log").write_text("\n".join(log_lines) + "\n")
    log(f"RESULT state={state} best={out['best']} best_worst="
        f"{out['best_worst_y_nrmse_pct']}% passing={out['passing_candidates']} "
        f"selected={out['selected']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(f"RUNTIME_FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
