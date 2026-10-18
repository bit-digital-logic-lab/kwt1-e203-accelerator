"""Broaden the local validation of the real-audio -> 12-layer ``X[layer]`` chain.

Contest context (``题目/README.md``, ``doc/本科FFN1数据与硬件边界核对_20261017.md``)
and the user analysis (P1-5): the twelve hardware FFN1 inputs are the per-layer
post-attention, post-residual, *first* LayerNorm ``99x64`` tensors produced
offline from a 1 s, 16 kHz, mono clip.  The organiser runs the submitted
software numeric model over 600 clips chosen after the freeze; those clips are
not released.  This script does **not** rebuild that official set.  It builds a
larger *local*, reproducible validation set from the pinned Speech Commands V2
split rule and verifies the offline chain against an independent reference.

What this measures (no golden is ever read back from the implementation under
test):

* real WAV -> ``98x40`` MFCC: the project RDFT recipe vs an independent NumPy
  FFT recipe (E2);
* ``98x40 -> 99x64`` first encoder input: ``offline_encoder_input.convert`` vs
  an independent float64 projection (E2);
* ``X[layer]`` for all 12 layers: ``run.KWT1.forward(..., capture_ffn=True)``
  vs an independent float64 forward, with wrong-boundary candidates (pre-LN1
  residual, attention combination, layer input, LN2) for discrimination (E2);
* shapes, dtype, finiteness and 12-way layer distinctness (E4);
* affine/token metamorphic invariants of the projection and capture (E3).

The sample split uses ``prepare_quant_samples.split_for`` (the official
10/10/80 validation/testing/training hash rule).  Only the **validation** split
is sampled; the testing split is never used to tune anything.

Outputs default to the git-ignored ``software/.build/large_scale`` (arrays and
the full manifest); the committed artifact is a compact summary written to this
directory.  Run with the PyTorch interpreter on CPU::

    OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
      /home/yang/miniconda3/bin/python \
      software/validation/large_scale/build_validation_inputs.py --per-label 50
"""

import argparse
from collections import defaultdict, OrderedDict
import hashlib
import io
import json
import platform
import sys
import tarfile
import time
from pathlib import Path
import wave

import numpy as np


HERE = Path(__file__).resolve().parent
SOFTWARE = HERE.parent.parent
sys.path.insert(0, str(SOFTWARE))

# Reuse the already-audited independent float64 reference and its constants.
from audit_ffn1_input_boundary import (  # noqa: E402
    ARCHIVE, SHARED_CACHE, WEIGHTS, N_LAYERS, D_MODEL, N_TOKENS,
    TOL_MFCC, TOL_ENCODER, TOL_FFN1, TOL_METAMORPHIC, MARGIN,
    independent_mfcc, independent_encode, independent_forward,
    load_weights, projection_weights, metamorphic_checks, sha256, sha256_file,
)

FULL_OUT = SOFTWARE / ".build/large_scale"
SUMMARY_OUT = HERE / "manifest_summary.json"

# The 12 KWT-1 label directories, in the frozen ``labels.txt`` order.
LABELS = ["_silence_", "_unknown_", "yes", "no", "up", "down", "left",
          "right", "on", "off", "stop", "go"]
REAL_LABELS = [label for label in LABELS if label != "_silence_"]


def load_flags():
    flags_path = (SOFTWARE / ".." / "资料/ARM_KWT-1_237e4097/"
                  "models_data_v2_12_labels/kwt1/flags.json")
    return json.loads(flags_path.read_text())


def wav_float(content: bytes) -> np.ndarray:
    """Mono 16 kHz PCM16 -> clipped/padded 16000-sample float32 (official rule)."""
    with wave.open(io.BytesIO(content)) as handle:
        if (handle.getnchannels(), handle.getframerate(),
                handle.getsampwidth()) != (1, 16000, 2):
            raise ValueError("Expected mono 16 kHz PCM16 WAV")
        samples = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    audio = np.zeros(16000, dtype=np.float32)
    n = min(len(samples), 16000)
    audio[:n] = samples[:n].astype(np.float32) / 32768.0
    return audio


def select_real(per_label, archive=ARCHIVE, wanted=None):
    """One streaming pass: keep the ``per_label`` lowest-SHA validation clips.

    Selection is deterministic (top-N by ``sha256(member name)``) and matches
    ``prepare_quant_samples``' rank/partition conventions.  Only validation-split
    members are eligible; ``_background_noise_`` is never a sample.
    """
    from prepare_quant_samples import split_for
    wanted = wanted or set(load_flags()["wanted_words"].split(","))
    buf = defaultdict(list)
    scanned = 0
    valid = defaultdict(int)
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            parts = Path(member.name).parts
            if not member.isfile() or len(parts) != 2 or not parts[1].endswith(".wav"):
                continue
            if parts[0] == "_background_noise_":
                continue
            scanned += 1
            label = parts[0] if parts[0] in wanted else "_unknown_"
            if split_for(member.name) != "validation":
                continue
            valid[label] += 1
            rank = hashlib.sha256(member.name.encode()).hexdigest()
            entries = buf[label]
            if len(entries) >= per_label and rank >= entries[-1][0]:
                continue
            content = tar.extractfile(member).read()
            entries.append((rank, member.name, content))
            entries.sort(key=lambda item: item[0])
            del entries[per_label:]
    selected = OrderedDict()
    for label in REAL_LABELS:
        entries = sorted(buf.get(label, []), key=lambda item: item[0])[:per_label]
        selected[label] = entries
    return selected, {"wav_members_scanned": scanned,
                      "validation_candidates": {k: valid[k] for k in sorted(valid)}}


def silence_entries(per_label):
    """The official synthetic ``_silence_`` special label (zeros, not real audio)."""
    return [("0" * 64, f"_silence_/local_zero_{i:03d}.wav",
             np.zeros(16000, dtype=np.float32)) for i in range(per_label)]


def evaluate(name, label, audio, w, torch, model, convert, matrices, mfcc_batch):
    """Chain + boundary record for one clip.  Independent reference is float64."""
    from audit_ffn1_input_boundary import check_sample_metadata
    failures = []
    mfcc_ref = independent_mfcc(audio)                    # independent golden
    check_sample_metadata(mfcc_ref, name)
    mfcc_proj = mfcc_batch(audio[None, :], matrices)[0]   # project candidate
    mfcc_err = float(np.max(np.abs(mfcc_proj - mfcc_ref)))
    if mfcc_err > TOL_MFCC:
        failures.append(f"mfcc:{mfcc_err:.3g}")

    golden_encoder = independent_encode(mfcc_ref, w)
    cand_encoder = convert(mfcc_ref, projection_weights(w))
    if cand_encoder.shape != (N_TOKENS, D_MODEL) or cand_encoder.dtype != np.float32:
        failures.append(f"encoder_shape:{cand_encoder.shape}/{cand_encoder.dtype}")
    encoder_err = float(np.max(np.abs(cand_encoder - golden_encoder)))
    if encoder_err > TOL_ENCODER:
        failures.append(f"encoder:{encoder_err:.3g}")

    golden = independent_forward(golden_encoder, w)
    with torch.inference_mode():
        logits, _, caps = model.forward(
            torch.from_numpy(cand_encoder).to(torch.device("cpu")), capture_ffn=True)
    if len(caps) != N_LAYERS:
        raise AssertionError(f"{name}: expected {N_LAYERS} captures, got {len(caps)}")

    layers = []
    for i in range(N_LAYERS):
        captured = caps[i].numpy()
        if captured.shape != (N_TOKENS, D_MODEL) or captured.dtype != np.float32:
            failures.append(f"layer{i}_shape:{captured.shape}/{captured.dtype}")
        ffn1_err = float(np.max(np.abs(captured - golden["ffn1"][i])))
        wrong = {k: float(np.max(np.abs(captured - golden[k][i])))
                 for k in ("residual", "combined", "layer_input", "ln2")}
        wrong_min = min(wrong.values())
        if ffn1_err > TOL_FFN1:
            failures.append(f"layer{i}_ffn1:{ffn1_err:.3g}")
        if ffn1_err + MARGIN > wrong_min:
            failures.append(f"layer{i}_boundary:{ffn1_err:.3g}>={wrong_min:.3g}")
        layers.append({"layer": i,
                       "capture_shape": list(captured.shape),
                       "capture_dtype": str(captured.dtype),
                       "capture_finite": bool(np.isfinite(captured).all()),
                       "capture_sha256": sha256(captured.tobytes()),
                       "ffn1_max_abs_error": ffn1_err,
                       "wrong_boundary_max_abs_error": wrong,
                       "wrong_boundary_min_max_abs_error": wrong_min})

    finite = lambda a: bool(np.isfinite(a).all())  # noqa: E731
    distinct = min(float(np.max(np.abs(caps[i].numpy() - caps[j].numpy())))
                   for i in range(N_LAYERS) for j in range(i + 1, N_LAYERS))
    if distinct <= 0.0:
        failures.append("layers_not_distinct")

    # End-to-end real chain: project MFCC into the encoder, then the same forward.
    chain_encoder = convert(mfcc_proj, projection_weights(w))
    with torch.inference_mode():
        _, _, caps_full = model.forward(
            torch.from_numpy(chain_encoder).to(torch.device("cpu")), capture_ffn=True)
    full_chain_err = max(
        float(np.max(np.abs(caps_full[i].numpy() - golden["ffn1"][i])))
        for i in range(N_LAYERS))

    scores = logits.numpy()
    argmax = int(scores.argmax())
    return {
        "sample": name,
        "label": label,
        "partition": "validation",
        "mfcc_shape": list(mfcc_ref.shape),
        "mfcc_dtype": str(mfcc_ref.dtype),
        "mfcc_finite": finite(mfcc_ref),
        "mfcc_sha256": sha256(mfcc_proj.tobytes()),
        "mfcc_independent_vs_project_max_abs_error": mfcc_err,
        "encoder_shape": list(cand_encoder.shape),
        "encoder_dtype": str(cand_encoder.dtype),
        "encoder_finite": finite(cand_encoder),
        "encoder_sha256": sha256(cand_encoder.tobytes()),
        "encoder_independent_max_abs_error": encoder_err,
        "ffn1_inputs_shape": [list(c.shape) for c in caps],
        "ffn1_inputs_finite": all(finite(c.numpy()) for c in caps),
        "ffn1_distinct_min_max_abs": distinct,
        "layers": layers,
        "full_chain_ffn1_max_abs_error": full_chain_err,
        "logits_argmax": argmax,
        "logits_label": LABELS[argmax] if argmax < len(LABELS) else str(argmax),
        "logits_min": float(scores.min()),
        "logits_max": float(scores.max()),
        "pass": not failures,
        "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-label", type=int, default=50,
                        help="Validation clips per label (11 real + 1 synthetic silence)")
    parser.add_argument("--full-out", type=Path, default=FULL_OUT)
    parser.add_argument("--summary-out", type=Path, default=SUMMARY_OUT)
    parser.add_argument("--seed", type=int, default=20261018)
    args = parser.parse_args()

    import torch
    sys.path.insert(0, str(SOFTWARE))
    from offline_encoder_input import convert
    from run import KWT1
    from prepare_quant_samples import mfcc_batch, mfcc_matrices

    t0 = time.time()
    args.full_out.mkdir(parents=True, exist_ok=True)
    w = load_weights()
    model = KWT1(WEIGHTS, torch.device("cpu"))
    matrices = mfcc_matrices()

    selected, scan_stats = select_real(args.per_label)
    silence = silence_entries(args.per_label)

    records = []
    t_select = time.time()
    for label in LABELS:
        entries = silence if label == "_silence_" else [(r, n, c) for r, n, c in selected[label]]
        for rank, name, content in entries:
            audio = content if isinstance(content, np.ndarray) else wav_float(content)
            record = evaluate(name, label, audio, w, torch, model, convert,
                              matrices, mfcc_batch)
            record["wav_sha256"] = (None if label == "_silence_"
                                    else sha256(content))
            record["wav_bytes"] = (None if label == "_silence_" else len(content))
            record["select_rank_sha256"] = rank
            records.append(record)
    t_eval = time.time()

    real = [r for r in records if r["label"] != "_silence_"]
    failures = [r for r in records if not r["pass"]]

    def worst(key, rows):
        return max((row[key] for row in rows), default=None)

    metamorphic = metamorphic_checks(
        independent_mfcc(wav_float(selected[REAL_LABELS[0]][0][2])),
        independent_mfcc(wav_float(selected[REAL_LABELS[1]][0][2])),
        w, torch, model, convert)

    by_label = OrderedDict(
        (label, sum(1 for r in records if r["label"] == label)) for label in LABELS)
    aggregate = {
        "samples": len(records),
        "real_audio_samples": len(real),
        "synthetic_silence_samples": len(records) - len(real),
        "by_label": by_label,
        "independent_reference": "NumPy float64; never reads implementation output",
        "mfcc_independent_vs_project_max_abs_error": worst(
            "mfcc_independent_vs_project_max_abs_error", records),
        "encoder_independent_max_abs_error": worst(
            "encoder_independent_max_abs_error", records),
        "ffn1_independent_max_abs_error": max(
            (l["ffn1_max_abs_error"] for r in records for l in r["layers"]), default=None),
        "wrong_boundary_min_max_abs_error": min(
            (l["wrong_boundary_min_max_abs_error"] for r in records for l in r["layers"]),
            default=None),
        "ffn1_distinct_min_max_abs": min(
            (r["ffn1_distinct_min_max_abs"] for r in records), default=None),
        "full_chain_ffn1_max_abs_error": worst(
            "full_chain_ffn1_max_abs_error", records),
        "failed_samples": len(failures),
        "failed_sample_ids": [r["sample"] for r in failures][:50],
        "tolerances": {"mfcc": TOL_MFCC, "encoder": TOL_ENCODER, "ffn1": TOL_FFN1,
                       "metamorphic": TOL_METAMORPHIC, "boundary_margin": MARGIN},
        "metamorphic": metamorphic,
    }

    report = {
        "scope": "local validation of real audio -> 98x40 MFCC -> 99x64 -> 12-layer X[layer]",
        "official_600": False,
        "notice": ("Locally selected Speech Commands V2 *validation*-split clips "
                   "(top-N per label by member-name SHA-256).  This is NOT the "
                   "organiser's hidden 600-clip evaluation set and cannot replace "
                   "the official accuracy gate.  The testing split is never used."),
        "weights": str(WEIGHTS),
        "weights_sha256": sha256_file(WEIGHTS),
        "archive": {"path": str(ARCHIVE), "bytes": ARCHIVE.stat().st_size},
        "source": {"selection_rule": "prepare_quant_samples.split_for == validation",
                   "resolve": "sha256(member name) ascending, cap per label",
                   "cache_read_only": str(SHARED_CACHE), **scan_stats},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "machine": platform.machine(), "numpy": np.__version__,
                        "torch": torch.__version__, "device_used": "cpu",
                        "torch_cuda_available": bool(torch.cuda.is_available())},
        "timing_seconds": {"select": round(t_select - t0, 3),
                           "evaluate": round(t_eval - t_select, 3),
                           "total": round(time.time() - t0, 3)},
        "aggregate": aggregate,
        "samples": records,
    }

    args.full_out.mkdir(parents=True, exist_ok=True)
    (args.full_out / "manifest_full.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    # Compact, committed summary: no large arrays, no per-layer SHA blob.
    summary = dict(report)
    summary["samples"] = [{
        "sample": r["sample"],
        "label": r["label"],
        "partition": r["partition"],
        "wav_sha256": r["wav_sha256"],
        "wav_bytes": r["wav_bytes"],
        "mfcc_shape": r["mfcc_shape"],
        "encoder_shape": r["encoder_shape"],
        "encoder_sha256": r["encoder_sha256"],
        "ffn1_inputs_finite": r["ffn1_inputs_finite"],
        "ffn1_distinct_min_max_abs": r["ffn1_distinct_min_max_abs"],
        "mfcc_independent_vs_project_max_abs_error":
            r["mfcc_independent_vs_project_max_abs_error"],
        "encoder_independent_max_abs_error": r["encoder_independent_max_abs_error"],
        "ffn1_max_abs_error": max(l["ffn1_max_abs_error"] for l in r["layers"]),
        "wrong_boundary_min_max_abs_error": min(
            l["wrong_boundary_min_max_abs_error"] for l in r["layers"]),
        "full_chain_ffn1_max_abs_error": r["full_chain_ffn1_max_abs_error"],
        "logits_argmax": r["logits_argmax"],
        "logits_label": r["logits_label"],
        "pass": r["pass"],
        "failures": r["failures"],
    } for r in records]
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    print(f"samples={len(records)} real={len(real)} failed={len(failures)}")
    print(f"mfcc_max={aggregate['mfcc_independent_vs_project_max_abs_error']:.6g} "
          f"encoder_max={aggregate['encoder_independent_max_abs_error']:.6g} "
          f"ffn1_max={aggregate['ffn1_independent_max_abs_error']:.6g}")
    print(f"wrong_boundary_min={aggregate['wrong_boundary_min_max_abs_error']:.6g} "
          f"full_chain_ffn1_max={aggregate['full_chain_ffn1_max_abs_error']:.6g}")
    print(f"metamorphic={json.dumps(metamorphic)}")
    print(f"full={args.full_out / 'manifest_full.json'}")
    print(f"summary={args.summary_out}")
    print(f"timing={report['timing_seconds']}")
    if failures:
        print("FAILED SAMPLES: " + ", ".join(r["sample"] for r in failures))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
