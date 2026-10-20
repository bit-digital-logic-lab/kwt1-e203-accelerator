"""Evaluate a frozen integer encoder with the FP32 PC classifier head.

Only the 12-layer encoder is quantized. The WAV/MFCC/projection and classifier
head stay on the PC. --pack points to a frozen p027_precision parameter pack.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from prepare_quant_samples import ARCHIVE, mfcc_batch, mfcc_matrices, wav_float
from run import KWT1, ROOT, SOURCE
from validate_audio_to_label import REAL, read_wavs


OUT = ROOT / "validation/audio_to_label_20261020"
HEAD = "layer_with_weights-1/layer_with_weights-0"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=ROOT / ".build/quant_samples_12labels_40.npz")
    parser.add_argument("--limit", type=int, default=0, help="Validation cache limit; 0 means all")
    parser.add_argument("--rebuild-cache-wavs", action="store_true",
                        help="Rebuild every cached validation feature from its WAV")
    args = parser.parse_args()
    model_dir = args.pack.parent / "model"
    if not (model_dir / "prec_ref.py").is_file():
        parser.error("Expected prec_ref.py beside the frozen pack directory")
    sys.path.insert(0, str(model_dir))
    import prec_ref as P
    import freeze_config as F

    calib, manifest = F.load_pack(args.pack)
    if len(calib["layers"]) != 12:
        raise AssertionError("Frozen pack does not contain 12 encoder layers")
    model = KWT1(ROOT / "weights/kwt1_fp32.npz", torch.device("cpu"))
    labels = (SOURCE / "labels.txt").read_text().splitlines()
    kernel = model.weight(HEAD + "/kernel").numpy()
    bias = model.weight(HEAD + "/bias").numpy()
    real_rows = json.loads((REAL / "manifest.json").read_text())["samples"]
    wavs = read_wavs({r["archive_member"] for r in real_rows})
    matrices = mfcc_matrices()
    entries = []
    for row in real_rows:
        raw = wavs[row["archive_member"]]
        if hashlib.sha256(raw).hexdigest() != row["wav_sha256"]:
            raise AssertionError("WAV hash mismatch")
        feature = mfcc_batch(wav_float(raw)[None], matrices)[0]
        entries.append(("wav", row["archive_member"], row["label"], feature))
    with np.load(args.cache, allow_pickle=False) as z:
        selected = np.flatnonzero(z["partitions"] == "validation")
        if args.limit:
            selected = selected[:args.limit]
        paths = z["paths"][selected].astype(str)
        truth = z["words"][selected].astype(str)
        stored_features = z["features"][selected]
    if args.rebuild_cache_wavs:
        names = {path.removeprefix("./") for path in paths
                 if not path.startswith("_silence_/")}
        source_wavs = read_wavs(names)
        cache_features = np.empty_like(stored_features)
        for i, path in enumerate(paths):
            audio = (np.zeros(16000, dtype=np.float32) if path.startswith("_silence_/")
                     else wav_float(source_wavs[path.removeprefix("./")]))
            cache_features[i] = mfcc_batch(audio[None], matrices)[0]
        cache_mfcc_max_abs = float(np.max(np.abs(cache_features - stored_features)))
        if cache_mfcc_max_abs > 1e-3:
            raise AssertionError(f"Cache/WAV MFCC mismatch: {cache_mfcc_max_abs}")
    else:
        cache_features = stored_features
        cache_mfcc_max_abs = None
    for path, label, feature in zip(paths, truth, cache_features):
        entries.append(("cache", path, label, feature))
    output = []
    err2 = ref2 = 0.0
    flips = []
    with torch.inference_mode():
        for n, (source, path, true_label, feature) in enumerate(entries):
            x = model.encoder_input(torch.from_numpy(feature)).numpy()
            fp_logits, _, _ = model.forward(torch.from_numpy(x))
            fp = fp_logits.numpy()
            integer = P.integer_chain(calib, x, want_stages=False)
            y = integer["deq"]["y"][-1]
            quant = y[0] @ kernel + bias
            if not np.isfinite(quant).all():
                raise AssertionError(f"Nonfinite quantized logits: {path}")
            fp_label, quant_label = labels[int(fp.argmax())], labels[int(quant.argmax())]
            if fp_label != quant_label:
                flips.append(path)
            err2 += float(np.sum((quant - fp) ** 2))
            ref2 += float(np.sum(fp ** 2))
            output.append(dict(source=source, path=path, true_label=true_label,
                               fp32=fp_label, quantized=quant_label,
                               fp32_correct=fp_label == true_label,
                               quantized_correct=quant_label == true_label,
                               logits_max_abs=float(np.max(np.abs(quant - fp))),
                               fp32_margin=float(np.sort(fp)[-1] - np.sort(fp)[-2]),
                               quantized_margin=float(np.sort(quant)[-1] - np.sort(quant)[-2])))
            if (n + 1) % 40 == 0:
                print(f"processed={n+1}/{len(entries)}", flush=True)
    real = [r for r in output if r["source"] == "wav"]
    cache = [r for r in output if r["source"] == "cache"]
    report = dict(status="pass", scope="PC WAV/MFCC/projection + frozen integer encoder + PC FP32 head",
                  pack_config_sha256=manifest["config_sha256"],
                  pack_calib_sha256=manifest["calib_sha256"],
                  weights_sha256=sha256(ROOT / "weights/kwt1_fp32.npz"),
                  cache_sha256=sha256(args.cache),
                  cache_features_source="wav_rebuilt" if args.rebuild_cache_wavs else "stored_mfcc",
                  cache_mfcc_max_abs=cache_mfcc_max_abs,
                  cache_raw_wav_count=int(np.sum([not p.startswith("_silence_/") for p in paths]))
                  if args.rebuild_cache_wavs else 0,
                  cache_synthetic_silence_count=int(np.sum([p.startswith("_silence_/") for p in paths]))
                  if args.rebuild_cache_wavs else 0,
                  real_wav_count=len(real), cache_validation_count=len(cache),
                  fp32_cache_correct=sum(r["fp32_correct"] for r in cache),
                  quantized_cache_correct=sum(r["quantized_correct"] for r in cache),
                  fp32_real_correct=sum(r["fp32_correct"] for r in real),
                  quantized_real_correct=sum(r["quantized_correct"] for r in real),
                  top1_flips=len(flips), flipped_paths=flips,
                  pooled_logit_nrmse=float(np.sqrt(err2 / max(ref2, 1e-16))),
                  max_logit_abs=max(r["logits_max_abs"] for r in output),
                  rows=output,
                  limitations=["Integer encoder is a software numerical model; this is not RTL or SoC output.",
                               "Local validation cache is not the official hidden contest set."])
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "quantized_results.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"real={len(real)} fp32_correct={report['fp32_real_correct']} "
          f"quant_correct={report['quantized_real_correct']}")
    print(f"cache={len(cache)} fp32_correct={report['fp32_cache_correct']} "
          f"quant_correct={report['quantized_cache_correct']} top1_flips={len(flips)} "
          f"pooled_logit_nrmse={report['pooled_logit_nrmse']:.6g}")
    print(f"results={path}")


if __name__ == "__main__":
    main()
