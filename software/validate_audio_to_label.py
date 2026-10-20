"""Check real WAV -> MFCC -> KWT-1 -> 12-way label and a local validation cache.

The six WAVs are read afresh from the Speech Commands archive. TensorFlow
reads the original checkpoint for an independent full-model comparison.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile

import numpy as np
import torch

from prepare_quant_samples import ARCHIVE, mfcc_batch, mfcc_matrices, wav_float
from run import KWT1, PREFIX, ROOT, SOURCE


OUT = ROOT / "validation/audio_to_label_20261020"
REAL = ROOT / "validation/real_input_chain_20261017"
CHECKPOINT = SOURCE / "best_weights"
SUFFIX = "/.ATTRIBUTES/VARIABLE_VALUE"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_wavs(names):
    found = {}
    with tarfile.open(ARCHIVE, "r|gz") as archive:
        for member in archive:
            name = member.name.removeprefix("./")
            if name in names and member.isfile():
                found[name] = archive.extractfile(member).read()
                if len(found) == len(names):
                    break
    if found.keys() != names:
        raise RuntimeError(f"WAVs missing from archive: {sorted(names - found.keys())}")
    return found


def tf_logits(features, reader):
    """TensorFlow math with tensors loaded from the original checkpoint."""
    import tensorflow as tf

    def w(name):
        return tf.constant(reader.get_tensor(name + SUFFIX))

    def linear(x, name, bias=True):
        y = tf.linalg.matmul(x, w(name + "/kernel"))
        return y + w(name + "/bias") if bias else y

    def norm(x, name):
        mean, var = tf.nn.moments(x, axes=[-1], keepdims=True)
        return (x - mean) * tf.math.rsqrt(var + 1e-6) * w(name + "/gamma") + w(name + "/beta")

    x = linear(tf.constant(features), PREFIX + "/patch_proj")
    x = tf.concat((w(PREFIX + "/class_emb")[0], x), axis=0) + w(PREFIX + "/pos_emb")[0]
    for layer in range(12):
        p = f"{PREFIX}/enc_layers/{layer}"
        a = p + "/att"
        q = linear(x, a + "/query_dense", False)
        k = linear(x, a + "/key_dense", False)
        v = linear(x, a + "/value_dense", False)
        score = tf.matmul(q, k, transpose_b=True) / tf.sqrt(tf.constant(64.0))
        attn = tf.matmul(tf.nn.softmax(score, axis=-1), v)
        out1 = norm(x + linear(attn, a + "/combine_heads"), p + "/layernorm1")
        h = tf.nn.gelu(linear(out1, p + "/ffn/layer_with_weights-0"), approximate=False)
        x = norm(out1 + linear(h, p + "/ffn/layer_with_weights-1"), p + "/layernorm2")
    return linear(x[0:1], "layer_with_weights-1/layer_with_weights-0")[0].numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / ".build/quant_samples_12labels_40.npz")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--limit", type=int, default=0, help="Limit cached validation samples; 0 means all")
    parser.add_argument("--rebuild-cache-wavs", action="store_true",
                        help="Rebuild each cached validation MFCC from its original WAV")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    import tensorflow as tf

    labels = (SOURCE / "labels.txt").read_text().splitlines()
    if len(labels) != 12 or len(set(labels)) != 12:
        raise AssertionError("Expected 12 distinct labels")
    flags = json.loads((SOURCE / "flags.json").read_text())
    wanted = flags["wanted_words"].split(",")
    if labels != ["_silence_", "_unknown_"] + wanted:
        raise AssertionError("Checkpoint label order differs from flags")
    weights_path = ROOT / "weights/kwt1_fp32.npz"
    model = KWT1(weights_path, torch.device(args.device))
    reader = tf.train.load_checkpoint(str(CHECKPOINT))
    records = json.loads((REAL / "manifest.json").read_text())["samples"]
    wavs = read_wavs({r["archive_member"] for r in records})
    matrices = mfcc_matrices()
    real_results = []
    max_mfcc, max_enc, max_logits = 0.0, 0.0, 0.0
    with torch.inference_mode():
        for row in records:
            name = row["archive_member"]
            raw = wavs[name]
            if hashlib.sha256(raw).hexdigest() != row["wav_sha256"]:
                raise AssertionError(f"WAV SHA mismatch: {name}")
            features = mfcc_batch(wav_float(raw)[None, :], matrices)[0]
            prior = np.load(REAL / row["mfcc_file"], allow_pickle=False)
            mfcc_error = float(np.max(np.abs(features - prior)))
            x = model.encoder_input(torch.from_numpy(features).to(model.device))
            prior_x = np.load(REAL / row["encoder_file"], allow_pickle=False)
            encoder_error = float(np.max(np.abs(x.cpu().numpy() - prior_x)))
            logits, _, _ = model.forward(x)
            torch_logits = logits.cpu().numpy()
            reference = tf_logits(features, reader)
            error = float(np.max(np.abs(reference - torch_logits)))
            predicted = labels[int(torch_logits.argmax())]
            reference_predicted = labels[int(reference.argmax())]
            if predicted != reference_predicted or error > 1e-3 or mfcc_error > 1e-3 or encoder_error > 1e-4:
                raise AssertionError(f"Real WAV chain mismatch: {name}: logits={error} "
                                     f"mfcc={mfcc_error} encoder={encoder_error} "
                                     f"torch={predicted} tf={reference_predicted}")
            real_results.append(dict(path=name, true_label=row["label"], predicted=predicted,
                                     correct=predicted == row["label"], split=row["split"],
                                     top1_margin=float(np.sort(torch_logits)[-1] - np.sort(torch_logits)[-2]),
                                     mfcc_prior_max_abs=mfcc_error,
                                     encoder_prior_max_abs=encoder_error,
                                     tensorflow_logits_max_abs=error))
            max_mfcc = max(max_mfcc, mfcc_error)
            max_enc = max(max_enc, encoder_error)
            max_logits = max(max_logits, error)
    with np.load(args.cache, allow_pickle=False) as cache:
        selected = np.flatnonzero(cache["partitions"] == "validation")
        if args.limit:
            selected = selected[:args.limit]
        features = cache["features"][selected]
        truth = cache["words"][selected].astype(str)
        paths = cache["paths"][selected].astype(str)
    if len(selected) == 0 or any(label not in labels for label in truth):
        raise AssertionError("Empty validation split or invalid label")
    cache_mfcc_max_abs = None
    cache_raw_wav_count = 0
    cache_synthetic_silence_count = 0
    if args.rebuild_cache_wavs:
        names = {name.removeprefix("./") for name in paths
                 if not name.startswith("_silence_/")}
        source_wavs = read_wavs(names)
        rebuilt = np.empty_like(features)
        for i, name in enumerate(paths):
            if name.startswith("_silence_/"):
                audio = np.zeros(16000, dtype=np.float32)
                cache_synthetic_silence_count += 1
            else:
                audio = wav_float(source_wavs[name.removeprefix("./")])
                cache_raw_wav_count += 1
            rebuilt[i] = mfcc_batch(audio[None, :], matrices)[0]
        cache_mfcc_max_abs = float(np.max(np.abs(rebuilt - features)))
        if cache_mfcc_max_abs > 1e-3:
            raise AssertionError(f"Validation cache differs from rebuilt WAV MFCC: {cache_mfcc_max_abs}")
        features = rebuilt
    predictions = []
    margins = []
    stratified_tf_count = 0
    stratified_tf_logits_max_abs = 0.0
    checked_labels = set()
    with torch.inference_mode():
        for item, true_label in zip(features, truth):
            x = model.encoder_input(torch.from_numpy(item).to(model.device))
            logits, _, _ = model.forward(x)
            scores = logits.cpu().numpy()
            predictions.append(labels[int(scores.argmax())])
            margins.append(float(np.sort(scores)[-1] - np.sort(scores)[-2]))
            if true_label not in checked_labels:
                tf_scores = tf_logits(item, reader)
                difference = float(np.max(np.abs(scores - tf_scores)))
                if difference > 1e-3 or int(scores.argmax()) != int(tf_scores.argmax()):
                    raise AssertionError(f"Stratified TensorFlow mismatch for {true_label}: {difference}")
                checked_labels.add(true_label)
                stratified_tf_count += 1
                stratified_tf_logits_max_abs = max(stratified_tf_logits_max_abs, difference)
    predictions = np.asarray(predictions)
    cache_rows = [dict(path=path, true_label=label, predicted=pred,
                       correct=bool(pred == label), top1_margin=margin)
                  for path, label, pred, margin in zip(paths, truth, predictions, margins)]
    by_label = {label: dict(count=int(np.sum(truth == label)),
                            correct=int(np.sum((truth == label) & (predictions == label))))
                for label in labels}
    report = dict(status="pass", scope="FP32 software only", device=args.device,
                  weights_sha256=digest(weights_path), cache_sha256=digest(args.cache),
                  flags_sha256=digest(SOURCE / "flags.json"),
                  raw_wav_count=len(real_results), raw_wav_correct=sum(r["correct"] for r in real_results),
                  max_mfcc_prior_abs=max_mfcc, max_encoder_prior_abs=max_enc,
                  max_tensorflow_logits_abs=max_logits, raw_wavs=real_results,
                  stratified_tf_count=stratified_tf_count,
                  stratified_tf_logits_max_abs=stratified_tf_logits_max_abs,
                  cache_validation_count=len(selected), cache_correct=int(np.sum(predictions == truth)),
                  cache_accuracy=float(np.mean(predictions == truth)), cache_by_label=by_label,
                  cache_features_source="wav_rebuilt" if args.rebuild_cache_wavs else "stored_mfcc",
                  cache_raw_wav_count=cache_raw_wav_count,
                  cache_synthetic_silence_count=cache_synthetic_silence_count,
                  cache_mfcc_max_abs=cache_mfcc_max_abs,
                  cache_rows=cache_rows,
                  limitations=["Local validation cache is not the official hidden contest set.",
                               "This run does not exercise RTL, SoC, or the quantized encoder."])
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "results.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"real_wav={len(real_results)} correct={report['raw_wav_correct']} "
          f"tf_logits_max_abs={max_logits:.9g}")
    print(f"cache_validation={len(selected)} correct={report['cache_correct']} "
          f"accuracy={100*report['cache_accuracy']:.3f}%")
    print(f"results={path}")


if __name__ == "__main__":
    sys.exit(main())
