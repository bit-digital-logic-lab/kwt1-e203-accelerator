"""Compare the pure-NumPy golden with the official Arm KWT-1 TensorFlow code.

The official ``transformer_utils.TransformerBlock`` (postnorm branch) and
``MultiHeadSelfAttention`` are loaded and executed verbatim from the repository
snapshot, so the comparison guards the exact operation order rather than a
re-implementation.

Boundary: this snapshot is written for TF 2.4 / tensorflow_addons and the
candidate environment ships TF 2.19 without ``tensorflow_addons`` and without
``keras.layers.experimental.preprocessing``.  Two *import-only* shims are
installed (``tensorflow_addons.layers.GELU`` -> ``tf.nn.gelu`` exact, and
``experimental.preprocessing.Rescaling`` -> ``keras.layers.Rescaling``); neither
is used by the executed code path.  The executed math is the official code.

Run with an interpreter that has TensorFlow, e.g.::

    /home/yang/Desktop/ic设计竞赛/software/.venv/bin/python \
        graduate/reference/compare_official_tf.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np

REFERENCE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REFERENCE_DIR))

import kwt1_fp32_golden as golden  # noqa: E402

OFFICIAL_FILE = (golden.REPO_ROOT / "资料/ARM_KWT-1_237e4097/kws_streaming/models"
                 / "transformer_utils.py")
STAGES = ("softmax", "attended_v", "attn_projected", "ln1", "ffn_hidden", "gelu",
          "ffn_out", "layer_output")


def load_official(tf):
    """Import the official module with import-only shims for the newer TF."""
    addons = types.ModuleType("tensorflow_addons")

    class GELU(tf.keras.layers.Layer):
        def __init__(self, approximate=False, **kwargs):
            super().__init__(**kwargs)
            self.approximate = approximate

        def call(self, inputs):
            return tf.nn.gelu(inputs, approximate=self.approximate)

    addons.layers = types.SimpleNamespace(GELU=GELU)
    sys.modules["tensorflow_addons"] = addons
    preprocessing = types.ModuleType("tensorflow.keras.layers.experimental.preprocessing")
    preprocessing.Rescaling = tf.keras.layers.Rescaling
    sys.modules["tensorflow.keras.layers.experimental"] = types.ModuleType(
        "tensorflow.keras.layers.experimental")
    sys.modules["tensorflow.keras.layers.experimental.preprocessing"] = preprocessing
    spec = importlib.util.spec_from_file_location("kwt1_official_transformer_utils", OFFICIAL_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assign_block_weights(block, weights, layer):
    prefix = f"{golden.PREFIX}/enc_layers/{layer}"
    block.att.query_dense.set_weights([weights[f"{prefix}/att/query_dense/kernel"]])
    block.att.key_dense.set_weights([weights[f"{prefix}/att/key_dense/kernel"]])
    block.att.value_dense.set_weights([weights[f"{prefix}/att/value_dense/kernel"]])
    block.att.combine_heads.set_weights([weights[f"{prefix}/att/combine_heads/kernel"],
                                         weights[f"{prefix}/att/combine_heads/bias"]])
    block.ffn.layers[0].set_weights([weights[f"{prefix}/ffn/layer_with_weights-0/kernel"],
                                     weights[f"{prefix}/ffn/layer_with_weights-0/bias"]])
    block.ffn.layers[2].set_weights([weights[f"{prefix}/ffn/layer_with_weights-1/kernel"],
                                     weights[f"{prefix}/ffn/layer_with_weights-1/bias"]])
    block.layernorm1.set_weights([weights[f"{prefix}/layernorm1/gamma"],
                                  weights[f"{prefix}/layernorm1/beta"]])
    block.layernorm2.set_weights([weights[f"{prefix}/layernorm2/gamma"],
                                  weights[f"{prefix}/layernorm2/beta"]])


def official_layer(block, tf, x):
    """Run one official postnorm block and return raw intermediates."""
    x_tf = tf.constant(x[None, ...], dtype=tf.float32)
    query = block.att.query_dense(x_tf)
    key = block.att.key_dense(x_tf)
    value = block.att.value_dense(x_tf)
    head_q = block.att.separate_heads(query, 1)
    head_k = block.att.separate_heads(key, 1)
    head_v = block.att.separate_heads(value, 1)
    attended, probs = block.att.attention(head_q, head_k, head_v)
    attended = tf.transpose(attended, perm=[0, 2, 1, 3])
    attended = tf.reshape(attended, (1, -1, block.att.embed_dim))
    projected = block.att.combine_heads(attended)
    residual1 = x_tf + projected
    out1 = block.layernorm1(residual1)
    hidden = block.ffn.layers[0](out1)
    activated = block.ffn.layers[1](hidden)
    ffn_out = block.ffn.layers[2](activated)
    output = block.layernorm2(out1 + ffn_out)
    assert tf.reduce_all(tf.math.equal(output, block(x_tf, training=False)[0])).numpy()
    return {
        "softmax": probs.numpy().reshape(99, 99),
        "attended_v": attended.numpy()[0],
        "attn_projected": projected.numpy()[0],
        "ln1": out1.numpy()[0],
        "ffn_hidden": hidden.numpy()[0],
        "gelu": activated.numpy()[0],
        "ffn_out": ffn_out.numpy()[0],
        "layer_output": output.numpy()[0],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-dir", type=Path, default=golden.ENCODER_SAMPLES_DEFAULT)
    parser.add_argument("--weights", type=Path, default=golden.WEIGHTS_DEFAULT)
    parser.add_argument("--partition", default="validation")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--output", type=Path,
                        default=REFERENCE_DIR / "results/official_tf_comparison.json")
    args = parser.parse_args()

    import tensorflow as tf

    official = load_official(tf)
    weights = golden.load_weights(args.weights)
    manifest = json.loads((args.samples_dir / "manifest.json").read_text())
    samples = [s for s in manifest["samples"] if args.partition in (None, "all", s["split"])]

    block = official.TransformerBlock(embed_dim=64, num_heads=1, ff_dim=256, dropout=0.0,
                                      prenorm=False, approximate_gelu=False)
    _ = block(tf.zeros([1, 99, 64], dtype=tf.float32), training=False)  # build sublayers
    head = tf.keras.layers.Dense(12)
    head.build((None, 64))
    head.set_weights([weights[f"{golden.HEAD}/kernel"], weights[f"{golden.HEAD}/bias"]])

    records = []
    worst_overall = 0.0
    for sample in samples:
        encoder = np.load(args.samples_dir / sample["encoder_file"], allow_pickle=False)
        golden_logits, golden_final, _ = golden.forward(encoder, weights)
        golden_arrays = []
        golden.forward(encoder, weights, arrays=golden_arrays)

        x = encoder.copy()
        per_stage = {stage: 0.0 for stage in STAGES}
        for layer in range(golden.EXPECTED_LAYERS):
            assign_block_weights(block, weights, layer)
            ref = official_layer(block, tf, x)
            for stage in STAGES:
                error = float(np.max(np.abs(ref[stage] - golden_arrays[layer][stage])))
                per_stage[stage] = max(per_stage[stage], error)
            x = ref["layer_output"]
        ref_logits = head(tf.constant(x[0][None, ...]))[0].numpy()
        logits_error = float(np.max(np.abs(ref_logits - golden_logits)))
        worst = max([*per_stage.values(), logits_error])
        worst_overall = max(worst_overall, worst)
        records.append({
            "archive_member": sample["archive_member"], "split": sample["split"],
            "label": sample["label"], "encoder_sha256": sample["encoder_sha256"],
            "max_abs_error_per_stage": per_stage,
            "max_abs_error_logits": logits_error,
            "worst": worst,
        })
        print(f"{sample['split']:10s} {sample['label']:9s} worst={worst:.3e} "
              f"ln1={per_stage['ln1']:.3e} layer_output={per_stage['layer_output']:.3e} "
              f"logits={logits_error:.3e}")

    passed = worst_overall <= args.tolerance
    output = {
        "reference": "graduate/reference/kwt1_fp32_golden.py",
        "partner": "official Arm KWT-1 transformer_utils (verbatim, TF)",
        "tensorflow_version": tf.__version__,
        "official_file": str(OFFICIAL_FILE),
        "partition": args.partition,
        "sample_count": len(records),
        "tolerance": args.tolerance,
        "worst_max_abs_error": worst_overall,
        "pass": bool(passed),
        "shims": ["tensorflow_addons.layers.GELU->tf.nn.gelu(exact)",
                  "experimental.preprocessing.Rescaling->keras.layers.Rescaling"],
        "samples": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=1) + "\n")
    print(f"worst_max_abs_error={worst_overall:.6g} tolerance={args.tolerance:g} "
          f"passed={passed} wrote={args.output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
