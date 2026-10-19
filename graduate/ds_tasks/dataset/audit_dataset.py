#!/usr/bin/env python3
"""Dataset audit for the graduate KWT-1 tasks: split / hash / overlap / calibration leakage.

Scope of this worker (independent worktree only, no sub-agents, no writes outside
``graduate/ds_tasks/dataset/``):

  A. Train/validation split is re-derived from the *official* rule in the
     specified ARM commit (``kws_streaming/data/input_data.py: which_set``:
     sha1 of the basename with ``_nohash_.*`` stripped, 10% validation / 10%
     testing) and compared with the split labels stored in every cached sample
     pool and in the real-input-chain manifest.
  B. Hashes are recomputed: the read-only weight pack, ``flags.json``, the
     dataset archive, the six real-input ``.npy``/``.wav`` files, and the frozen
     FFN1 parameter pack.
  C. Sample overlap is checked between and inside the pools (path sets and WAV
     content hashes).
  D. Calibration leakage: which partition did each calibration step actually
     read? The FFN1 pack claims training-only; the encoder assembly
     (``attn_int_model.py`` / ``kwt_postnorm_ref.calibrate``) is inspected
     statically *and* re-run on a validation vs a training sample to show that
     its frozen scales depend on the sample.

Rules being enforced (contest 题目/640.jpg 5 item 3, 640.jpg 7 item 3):
  * training partition  -> derivative-free calibration allowed
  * validation partition -> evaluation only, never a scale/threshold
  * testing partition    -> never used for weights, quantization or numeric config

Evidence tier of everything produced here: **软件参考 (software reference, CPU
NumPy)**.  It says nothing about RTL bit-exactness, E203 SoC integration,
synthesis or the official 600-sample acceptance.

Exit codes: 0 all checks pass; 1 rule violations detected; 2 infrastructure error.
"""

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import re
import sys
import tarfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                       # worktree root
DEFAULT_MAIN = Path("/home/yang/Desktop/ic设计竞赛")
DATA = ROOT / "software/validation/real_input_chain_20261017"
ENC_MODEL = ROOT / "graduate/encoder/model"
FROZEN_FFN1 = ROOT / "software/results/ffn1_a12w10_static_sym.npz"

MAX_NUM_WAVS_PER_CLASS = 2 ** 27 - 1
VALIDATION_PERCENTAGE = 10
TESTING_PERCENTAGE = 10

# The contest does not publish a SHA-256 for speech_commands_v0.02.tar.gz, so the
# archive is recorded (hash + member count + the six manifest WAV hashes) rather
# than compared against an external value.


def sha256_file(path, block=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def official_split(name):
    """Return (split, percentage_hash) exactly as the specified commit does."""
    base = Path(str(name)).name
    hash_name = re.sub(r"_nohash_.*$", "", base)
    hashed = int(hashlib.sha1(hash_name.encode()).hexdigest(), 16)
    pct = (hashed % (MAX_NUM_WAVS_PER_CLASS + 1)) * (100.0 / MAX_NUM_WAVS_PER_CLASS)
    if pct < VALIDATION_PERCENTAGE:
        return "validation", pct
    if pct < TESTING_PERCENTAGE + VALIDATION_PERCENTAGE:
        return "testing", pct
    return "training", pct


def norm_path(p):
    return str(p).lstrip("./")


# ---------------------------------------------------------------------------
# A. split labels
# ---------------------------------------------------------------------------
def audit_splits(main):
    """Re-derive the split for every stored sample.

    ``_silence_/*.wav`` rows are synthetic 1-second zero waveforms generated
    *per partition* (contest rules: silence follows the official per-split
    silence_percentage); they have no archive membership, so the SHA1 speaker
    rule does not apply and they are audited separately.
    """
    pools = {}
    mismatches = []
    synthetic = {}
    build = main / "software/.build"
    for npz in sorted(build.glob("quant_samples*.npz")):
        with np.load(npz, allow_pickle=False) as d:
            parts = d["partitions"].astype(str)
            paths = d["paths"].astype(str)
        counts = {}
        for p in parts:
            counts[p] = counts.get(p, 0) + 1
        bad = []
        synth = {}
        for path, stored in zip(paths, parts):
            if path.startswith("_silence_"):
                synth[stored] = synth.get(stored, 0) + 1
                continue
            recomputed, _ = official_split(path)
            if recomputed != stored:
                bad.append({"path": path, "stored": stored, "official": recomputed})
        pools[npz.name] = {
            "path": str(npz),
            "sha256": sha256_file(npz),
            "total": int(len(parts)),
            "counts": counts,
            "split_mismatches": len(bad),
            "synthetic_silence_per_partition": synth,
            "examples": bad[:5],
        }
        synthetic[npz.name] = synth
        mismatches.extend({"pool": npz.name, **b} for b in bad)

    manifest_path = DATA / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_mismatches = []
    for s in manifest["samples"]:
        recomputed, _ = official_split(s["archive_member"])
        if recomputed != s["split"]:
            manifest_mismatches.append({"path": s["archive_member"],
                                        "stored": s["split"], "official": recomputed})
    return pools, manifest, mismatches, manifest_mismatches


# ---------------------------------------------------------------------------
# B. hashes
# ---------------------------------------------------------------------------
def audit_hashes(main, manifest, check_archive):
    out = {"ok": True, "items": {}, "errors": []}

    def record(name, path, expected):
        actual = None
        if Path(path).exists():
            actual = sha256_file(path)
        ok = expected is None or actual == expected
        out["items"][name] = {"path": str(path), "expected": expected,
                              "actual": actual, "ok": bool(ok)}
        if not ok:
            out["ok"] = False
            out["errors"].append(f"{name}: {actual} != {expected}")

    record("weights", ROOT / "software/weights/kwt1_fp32.npz", manifest["weights_sha256"])
    record("flags", main / "资料/ARM_KWT-1_237e4097/models_data_v2_12_labels/kwt1/flags.json",
           manifest["flags_sha256"])
    for i, s in enumerate(manifest["samples"]):
        record(f"sample_{i:02d}_encoder", DATA / s["encoder_file"], s["encoder_sha256"])
        record(f"sample_{i:02d}_mfcc", DATA / s["mfcc_file"], s["mfcc_sha256"])

    frozen = json.loads((ROOT / "software/results/ffn1_a12w10_static_sym.json").read_text())
    record("ffn1_frozen_npz", FROZEN_FFN1, frozen["parameter_sha256"])
    record("ffn1_weights_ref", ROOT / "software/weights/kwt1_fp32.npz", frozen["weight_sha256"])

    if check_archive:
        archive = main / "资料/数据集/speech_commands_v0.02.tar.gz"
        out["items"]["dataset_archive"] = {"path": str(archive)}
        if archive.exists():
            out["items"]["dataset_archive"]["sha256"] = sha256_file(archive)
            out["items"]["dataset_archive"]["bytes"] = archive.stat().st_size
            wanted = {norm_path(s["archive_member"]): s for s in manifest["samples"]}
            found = {}
            n_members = 0
            n_wavs = 0
            with tarfile.open(archive, "r|gz") as tar:
                for member in tar:
                    n_members += 1
                    name = norm_path(member.name)
                    if name.endswith(".wav"):
                        n_wavs += 1
                    if name in wanted and name not in found:
                        content = tar.extractfile(member).read()
                        found[name] = {
                            "wav_bytes": len(content),
                            "wav_sha256": sha256_bytes(content),
                        }
            out["items"]["dataset_archive"]["members"] = n_members
            out["items"]["dataset_archive"]["wav_members"] = n_wavs
            for name, s in wanted.items():
                got = found.get(name)
                ok = bool(got and got["wav_sha256"] == s["wav_sha256"]
                          and got["wav_bytes"] == s["wav_bytes"])
                out["items"][f"wav::{name}"] = {"expected": s["wav_sha256"],
                                                "actual": got, "ok": ok}
                if not ok:
                    out["ok"] = False
                    out["errors"].append(f"wav {name}: {got} does not match manifest")
        else:
            out["ok"] = False
            out["errors"].append("dataset archive missing")
    return out


# ---------------------------------------------------------------------------
# B2. numeric consistency of the real-input .npy chain (hash record vs content)
# ---------------------------------------------------------------------------
def audit_npy_content(manifest):
    """Independent 40->64 projection: is the committed .npy content still right?

    This separates "stale hash record" from "corrupted tensor": the manifest
    stores ``encoder_independent_max_abs_error`` per sample, so the recomputed
    projection error is compared against it.
    """
    out = {"ok": True, "samples": [], "errors": []}
    try:
        spec = importlib.util.spec_from_file_location(
            "offline_encoder_input", ROOT / "software/offline_encoder_input.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with np.load(ROOT / "software/weights/kwt1_fp32.npz", allow_pickle=False) as w:
            weights = {"layer_with_weights-0/" + s: w["layer_with_weights-0/" + s]
                       for s in ("patch_proj/kernel", "patch_proj/bias",
                                 "class_emb", "pos_emb")}
    except Exception as exc:  # pragma: no cover
        out["ok"] = False
        out["errors"].append(f"{type(exc).__name__}: {exc}")
        return out
    for i, s in enumerate(manifest["samples"]):
        mfcc = np.load(DATA / s["mfcc_file"], allow_pickle=False)
        encoder = np.load(DATA / s["encoder_file"], allow_pickle=False)
        got = mod.convert(mfcc, weights)
        diff = float(np.max(np.abs(got - encoder)))
        recorded = float(s.get("encoder_independent_max_abs_error", -1))
        ok = bool(got.shape == encoder.shape == (99, 64) and diff <= max(recorded, 1e-6) + 1e-9)
        out["samples"].append({"file": s["encoder_file"], "max_abs_diff": diff,
                               "recorded_independent_max_abs_error": recorded, "ok": ok})
        if not ok:
            out["ok"] = False
            out["errors"].append(f"{s['encoder_file']}: content diff {diff} > recorded {recorded}")
    return out


# ---------------------------------------------------------------------------
# C. overlap
# ---------------------------------------------------------------------------
def audit_overlap(main, pools, manifest):
    out = {"ok": True, "details": {}, "errors": []}
    sets = {}
    for name, info in pools.items():
        with np.load(info["path"], allow_pickle=False) as d:
            parts = d["partitions"].astype(str)
            paths = np.array([norm_path(p) for p in d["paths"].astype(str)])
            features = d["features"]
        sets[name] = {
            "training": set(paths[parts == "training"]),
            "validation": set(paths[parts == "validation"]),
            "testing": set(paths[parts == "testing"]),
        }
        # exact duplicate feature tensors across partitions (excluding synthetic silence)
        def fp_hash(i):
            return hashlib.sha256(np.ascontiguousarray(features[i]).tobytes()).hexdigest()
        tr = {fp_hash(i) for i in np.flatnonzero(parts == "training")
              if not paths[i].startswith("_silence_")}
        va = {fp_hash(i) for i in np.flatnonzero(parts == "validation")
              if not paths[i].startswith("_silence_")}
        out["details"].setdefault("duplicate_features", {})[name] = len(tr & va)
        if tr & va:
            out["ok"] = False
            out["errors"].append(f"{name}: {len(tr & va)} identical feature tensors in "
                                 "both training and validation")
    for name, s in sets.items():
        inter = s["training"] & s["validation"]
        inter_t = s["training"] & s["testing"]
        inter_v = s["validation"] & s["testing"]
        out["details"][name] = {"train_val": len(inter), "train_test": len(inter_t),
                                "val_test": len(inter_v)}
        if inter or inter_t or inter_v:
            out["ok"] = False
            out["errors"].append(f"{name}: split overlap train/val={len(inter)} "
                                 f"train/test={len(inter_t)} val/test={len(inter_v)}")

    enc = norm_path("four/750e3e75_nohash_0.wav")   # sample_00 used by encoder
    used_by_encoder = {enc}
    for name, s in sets.items():
        in_train = enc in s["training"]
        out["details"].setdefault("encoder_sample00", {})[name] = {
            "in_training_of_this_pool": in_train}
        if in_train:
            out["ok"] = False
            out["errors"].append(f"encoder sample_00 appears in training pool {name}")

    # cross-pool: does the 6-sample real chain overlap any pooled training sample?
    chain = {norm_path(s["archive_member"]) for s in manifest["samples"]}
    for name, s in sets.items():
        ov = chain & s["training"]
        out["details"].setdefault("real_chain_vs_pool_training", {})[name] = sorted(ov)
    return out


# ---------------------------------------------------------------------------
# D. calibration leakage
# ---------------------------------------------------------------------------
def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def attention_scales(sample_index, tmpdir):
    """Run the copied attention integer model on one real sample; return its scale set."""
    module = _load_module(f"attn_int_model_{sample_index}", ENC_MODEL / "attn_int_model.py")
    module.RTL_DIR = tmpdir / f"attn{sample_index}/rtl"
    module.VEC = tmpdir / f"attn{sample_index}/vectors"
    old_argv = sys.argv
    sys.argv = ["attn_int_model.py", "--sample", str(sample_index)]
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            module.main()
    finally:
        sys.argv = old_argv
    return json.loads((module.VEC / "params.json").read_text())


def postnorm_scales(sample_index):
    """Recompute the postnorm integer coefficient set for one real sample."""
    module = _load_module(f"kwt_postnorm_ref_{sample_index}",
                          ENC_MODEL / "kwt_postnorm_ref.py")
    module.SAMPLE = module.SAMPLE.parent / f"sample_{sample_index:02d}_encoder.npy"
    cal = module.calibrate()
    scales = {k: float(v) for k, v in cal["scales"].items() if np.isscalar(v)}
    scalars = {k: (int(v) if np.isscalar(v) else [int(x) for x in np.ravel(v)])
               for k, v in cal["scalars"].items()}
    return {"scales": scales, "scalars": scalars}


def audit_calibration(main, manifest, pools, tmpdir):
    npy_names = {i: s["archive_member"] for i, s in enumerate(manifest["samples"])}
    npy_splits = {i: s["split"] for i, s in enumerate(manifest["samples"])}
    out = {"encoder_source_sample_index": 0,
           "encoder_source_member": npy_names[0],
           "encoder_source_split": npy_splits[0],
           "violations": [], "checks": {}}

    # --- static scan of the encoder model: which sample path is frozen in ---
    src = (ENC_MODEL / "attn_int_model.py").read_text()
    ref_src = (ENC_MODEL / "kwt_postnorm_ref.py").read_text()
    out["checks"]["attn_hardcoded_sample00"] = "sample_00_encoder.npy" in src
    out["checks"]["postnorm_hardcoded_sample00"] = "sample_00_encoder.npy" in ref_src

    # --- FFN1 frozen pack provenance (must be training-only) ---
    frozen = json.loads((ROOT / "software/results/ffn1_a12w10_static_sym.json").read_text())
    out["checks"]["ffn1_validation_used_for_calibration"] = frozen.get(
        "validation_used_for_calibration")
    if frozen.get("validation_used_for_calibration") is not False:
        out["violations"].append("FFN1 frozen pack does not assert training-only calibration")

    # --- blackbox calibration record: training objective, validation only eval ---
    bb_path = ROOT / "software/results/ffn1_blackbox_a11w10.json"
    if bb_path.exists():
        bb = json.loads(bb_path.read_text())
        tr, va = set(bb["training_paths"]), set(bb["validation_paths"])
        out["checks"]["blackbox_training"] = len(tr)
        out["checks"]["blackbox_validation"] = len(va)
        out["checks"]["blackbox_disjoint"] = not (tr & va)
        factors = [(r["activation_factor"], r["weight_factor"]) for r in bb["layers"]]
        out["checks"]["blackbox_selected_factors"] = factors
        if tr & va:
            out["violations"].append("FFN1 blackbox training/validation path overlap")

    # --- dynamic: does the frozen scale set depend on the sample partition? ---
    validation_index, training_index = 0, 1
    assert npy_splits[validation_index] == "validation"
    assert npy_splits[training_index] == "training"
    att_val = attention_scales(validation_index, tmpdir)
    att_tr = attention_scales(training_index, tmpdir)
    pn_val = postnorm_scales(validation_index)
    pn_tr = postnorm_scales(training_index)

    att_keys = ["scale_x", "scale_wq", "scale_wk", "scale_wv", "scale_wo",
                "scale_o", "RQ", "RK", "RV", "RC", "RO", "KC", "SH_KC"]
    att_cmp = {}
    for k in att_keys:
        v, t = att_val[k], att_tr[k]
        if isinstance(v, float):
            rel = abs(v - t) / max(abs(t), 1e-30)
        else:
            rel = None
        att_cmp[k] = {"validation_sample00": v, "training_sample01": t,
                      "rel_diff": rel}
    pn_cmp = {}
    for k, v in pn_val["scales"].items():
        t = pn_tr["scales"][k]
        pn_cmp[k] = {"validation_sample00": v, "training_sample01": t,
                     "rel_diff": abs(v - t) / max(abs(t), 1e-30)}
    scalar_cmp = {}
    for k, v in pn_val["scalars"].items():
        t = pn_tr["scalars"][k]
        if isinstance(v, list):
            d = int(np.max(np.abs(np.array(v) - np.array(t)))) if v else 0
        else:
            d = int(abs(v - t))
        scalar_cmp[k] = {"validation_sample00": v, "training_sample01": t,
                         "max_abs_diff": d}

    att_dependent = any(c["rel_diff"] is not None and c["rel_diff"] > 1e-12
                        for c in att_cmp.values())
    pn_dependent = any(c["rel_diff"] > 1e-12 for c in pn_cmp.values())
    out["attention"] = att_cmp
    out["postnorm_scales"] = pn_cmp
    out["postnorm_scalars"] = scalar_cmp
    out["checks"]["attention_scales_depend_on_sample"] = att_dependent
    out["checks"]["postnorm_scales_depend_on_sample"] = pn_dependent

    if npy_splits[validation_index] == "validation" and att_dependent:
        out["violations"].append(
            "encoder attention static scales are derived from validation sample_00")
    if npy_splits[validation_index] == "validation" and pn_dependent:
        out["violations"].append(
            "encoder postnorm static scales/LUT/output scale are derived from "
            "validation sample_00")

    # --- documentation-level contamination note (documented in results/README) ---
    dev_note = (ROOT / "software/results/README.md").read_text()
    out["checks"]["results_readme_admits_test_probe"] = (
        "用少量测试划分音频试跑程序" in dev_note)
    testing_rows = pools.get("quant_samples.npz", {}).get("counts", {}).get("testing", 0)
    out["checks"]["legacy_pool_quant_samples_testing_rows"] = testing_rows
    if testing_rows:
        out["violations"].append(
            f"legacy pool quant_samples.npz still contains {testing_rows} testing-partition "
            "rows that were probed during development (documented contamination)")
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--main", type=Path, default=DEFAULT_MAIN,
                    help="main project root holding the gitignored dataset (read-only)")
    ap.add_argument("--logs", type=Path, default=HERE / "logs")
    ap.add_argument("--skip-archive", action="store_true",
                    help="skip the 2.3 GB archive scan (weights/npy hashes still checked)")
    args = ap.parse_args()
    args.logs.mkdir(parents=True, exist_ok=True)
    tmpdir = args.logs / "_tmp_calibration"

    report = {"evidence_tier": "software-reference (CPU NumPy)",
              "main_project": str(args.main), "worktree": str(ROOT)}
    try:
        pools, manifest, split_mismatches, manifest_mismatches = audit_splits(args.main)
        report["pools"] = pools
        report["split_mismatches"] = split_mismatches
        report["manifest_split_mismatches"] = manifest_mismatches
        report["hashes"] = audit_hashes(args.main, manifest, not args.skip_archive)
        report["npy_content_consistency"] = audit_npy_content(manifest)
        report["overlap"] = audit_overlap(args.main, pools, manifest)
        report["calibration"] = audit_calibration(args.main, manifest, pools, tmpdir)
    except Exception as exc:  # pragma: no cover - infrastructure failures
        report["error"] = f"{type(exc).__name__}: {exc}"
        (args.logs / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"AUDIT_ERROR {report['error']}")
        return 2

    violations = []
    if split_mismatches or manifest_mismatches:
        violations.append(f"{len(split_mismatches)} pooled + {len(manifest_mismatches)} "
                          "manifest split-label mismatches")
    if not report["hashes"]["ok"]:
        violations.append("hash mismatches: " + "; ".join(report["hashes"]["errors"]))
    if not report["npy_content_consistency"]["ok"]:
        violations.append("npy content inconsistency: "
                          + "; ".join(report["npy_content_consistency"]["errors"]))
    if not report["overlap"]["ok"]:
        violations.append("sample overlap: " + "; ".join(report["overlap"]["errors"]))
    violations.extend(report["calibration"]["violations"])
    report["violations"] = violations
    report["verdict"] = "FAIL (rule violations)" if violations else "PASS"

    (args.logs / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    print("== split labels ==")
    for name, info in pools.items():
        print(f"  {name}: total={info['total']} counts={info['counts']} "
              f"mismatches={info['split_mismatches']} "
              f"silence={info['synthetic_silence_per_partition']}")
    print(f"  manifest samples: {len(manifest['samples'])} "
          f"mismatches={len(manifest_mismatches)}")
    print("== hashes ==")
    print(f"  ok={report['hashes']['ok']} items={len(report['hashes']['items'])}")
    for e in report["hashes"]["errors"]:
        print(f"  ERROR {e}")
    print("== npy content consistency (independent projection) ==")
    print(f"  ok={report['npy_content_consistency']['ok']}")
    for s in report["npy_content_consistency"]["samples"]:
        print(f"  {s['file']}: max_abs_diff={s['max_abs_diff']:.3e} "
              f"recorded={s['recorded_independent_max_abs_error']:.3e} ok={s['ok']}")
    print("== overlap ==")
    print(f"  ok={report['overlap']['ok']} errors={report['overlap']['errors']}")
    print("== calibration ==")
    for k, v in report["calibration"]["checks"].items():
        print(f"  {k} = {v}")
    if "attention" in report["calibration"]:
        a = report["calibration"]["attention"]
        print(f"  attention scale_x sample00={a['scale_x']['validation_sample00']:.6e} "
              f"sample01={a['scale_x']['training_sample01']:.6e} "
              f"rel={a['scale_x']['rel_diff']:.4e}")
        print(f"  attention scale_o sample00={a['scale_o']['validation_sample00']:.6e} "
              f"sample01={a['scale_o']['training_sample01']:.6e} "
              f"rel={a['scale_o']['rel_diff']:.4e}")
    print(f"VERDICT {report['verdict']}")
    for v in violations:
        print(f"  VIOLATION {v}")
    print(f"report: {args.logs / 'audit.json'}")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
