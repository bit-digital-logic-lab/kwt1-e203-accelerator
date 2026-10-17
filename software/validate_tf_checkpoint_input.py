"""Independently compare exported inputs with TensorFlow checkpoint math."""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "validation/real_input_chain_20261017"
CHECKPOINT = ROOT.parent / "资料/ARM_KWT-1_237e4097/models_data_v2_12_labels/kwt1/best_weights"
SUFFIX = "/.ATTRIBUTES/VARIABLE_VALUE"
PREFIX = "layer_with_weights-0/"


def main():
    manifest = json.loads((OUT / "manifest.json").read_text())
    reader = tf.train.load_checkpoint(str(CHECKPOINT))
    def weight(name):
        return tf.constant(reader.get_tensor(PREFIX + name + SUFFIX))

    kernel = weight("patch_proj/kernel")
    bias = weight("patch_proj/bias")
    cls = weight("class_emb")
    pos = weight("pos_emb")
    maximum = 0.0
    for row in manifest["samples"]:
        mfcc = np.load(OUT / row["mfcc_file"], allow_pickle=False)
        exported = np.load(OUT / row["encoder_file"], allow_pickle=False)
        projected = tf.linalg.matmul(tf.constant(mfcc)[None], kernel) + bias
        reference = tf.concat([cls, projected], axis=1) + pos
        reference = reference.numpy()[0]
        error = float(np.max(np.abs(exported - reference)))
        row["encoder_tf_checkpoint_max_abs_error"] = error
        maximum = max(maximum, error)
        if reference.shape != (99, 64) or not np.isfinite(reference).all() or error > 2e-5:
            raise AssertionError(f"TensorFlow checkpoint mismatch: {row['archive_member']}: {error}")
        print(f"{row['archive_member']} max_abs_error={error:.9g}")
    manifest["tensorflow_version"] = tf.__version__
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"samples={len(manifest['samples'])} max_tf_checkpoint_error={maximum:.9g}")


if __name__ == "__main__":
    main()
