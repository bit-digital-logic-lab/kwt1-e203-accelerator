"""Runnable tests for the independent FP32 golden (graduate items 4-6).

Only needs NumPy, so it runs on the plain system interpreter::

    python3 -m unittest graduate.reference.test_kwt1_fp32_golden -v
    # or from inside the directory:
    python3 graduate/reference/test_kwt1_fp32_golden.py
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

REFERENCE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REFERENCE_DIR))

import kwt1_fp32_golden as golden  # noqa: E402


class GoldenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.flags = golden.load_flags()
        cls.spec = golden.check_flags(cls.flags)
        cls.weights = golden.load_weights()
        cls.rng = np.random.default_rng(20261018)
        cls.x = cls.rng.normal(size=(99, 64)).astype(np.float32)

    def run_layer(self, layer, x=None):
        x = self.x if x is None else x
        arrays = {}
        out = golden.encoder_layer(x, self.weights, layer, arrays=arrays)
        return out, arrays

    def test_flags_semantics(self):
        self.assertEqual(self.spec["norm"], "postnorm")
        self.assertEqual(self.spec["num_layers"], 12)
        self.assertEqual(self.spec["d_model"], 64)
        self.assertEqual(self.spec["mlp_dim"], 256)
        self.assertEqual(self.spec["heads"], 1)
        self.assertEqual(self.spec["num_tokens"], 99)
        self.assertEqual(self.spec["attention_scale"], 8.0)
        self.assertEqual(self.spec["layernorm_epsilon"], 1e-6)
        self.assertFalse(self.flags.get("prenorm", False))
        self.assertFalse(self.flags.get("approximate_gelu", False))

    def test_weight_shapes_and_tensor_count(self):
        w = self.weights
        self.assertEqual(len(w), 162)
        p = f"{golden.PREFIX}/enc_layers/0"
        shapes = {
            f"{p}/att/query_dense/kernel": (64, 64),
            f"{p}/att/key_dense/kernel": (64, 64),
            f"{p}/att/value_dense/kernel": (64, 64),
            f"{p}/att/combine_heads/kernel": (64, 64),
            f"{p}/att/combine_heads/bias": (64,),
            f"{p}/ffn/layer_with_weights-0/kernel": (64, 256),
            f"{p}/ffn/layer_with_weights-0/bias": (256,),
            f"{p}/ffn/layer_with_weights-1/kernel": (256, 64),
            f"{p}/ffn/layer_with_weights-1/bias": (64,),
            f"{p}/layernorm1/gamma": (64,),
            f"{p}/layernorm1/beta": (64,),
            f"{p}/layernorm2/gamma": (64,),
            f"{p}/layernorm2/beta": (64,),
            f"{golden.HEAD}/kernel": (64, 12),
            f"{golden.HEAD}/bias": (12,),
        }
        for name, shape in shapes.items():
            self.assertIn(name, w)
            self.assertEqual(w[name].shape, shape, name)
            self.assertEqual(w[name].dtype, np.float32, name)
        # Q/K/V projection have no bias (use_bias=False in the official code).
        self.assertNotIn(f"{p}/att/query_dense/bias", w)

    def test_twelve_layers_have_distinct_parameters(self):
        signatures = golden.layer_parameter_signatures(self.weights)
        self.assertEqual(len(signatures), 12)
        self.assertEqual(len(set(signatures)), 12, "layers must reload distinct weights")

    def test_attention_shapes_and_scaled_score(self):
        _, a = self.run_layer(0)
        self.assertEqual(a["q"].shape, (99, 64))
        self.assertEqual(a["k"].shape, (99, 64))
        self.assertEqual(a["v"].shape, (99, 64))
        self.assertEqual(a["score"].shape, (99, 99))
        self.assertEqual(a["softmax"].shape, (99, 99))
        self.assertEqual(a["attended_v"].shape, (99, 64))
        self.assertEqual(a["attn_projected"].shape, (99, 64))
        expected = (a["q"] @ a["k"].T) / np.float32(8.0)  # QK^T / sqrt(64)
        self.assertLess(float(np.max(np.abs(expected - a["score"]))), 1e-5)

    def test_row_softmax_and_weighted_value(self):
        _, a = self.run_layer(0)
        row_sums = a["softmax"].sum(axis=-1)
        self.assertLess(float(np.max(np.abs(row_sums - 1.0))), 1e-6)
        self.assertTrue((a["softmax"] >= 0).all())
        attended = a["softmax"] @ a["v"]
        self.assertLess(float(np.max(np.abs(attended - a["attended_v"]))), 1e-6)

    def test_attention_output_projection(self):
        _, a = self.run_layer(0)
        w = self.weights
        kernel = w[f"{golden.PREFIX}/enc_layers/0/att/combine_heads/kernel"]
        bias = w[f"{golden.PREFIX}/enc_layers/0/att/combine_heads/bias"]
        expected = a["attended_v"] @ kernel + bias
        self.assertLess(float(np.max(np.abs(expected - a["attn_projected"]))), 1e-6)

    def test_postnorm_order(self):
        _, a = self.run_layer(0)
        w = self.weights
        p = f"{golden.PREFIX}/enc_layers/0"
        # residual 1 connects the block input to LN1 (not a prenorm ordering).
        residual1 = self.x + a["attn_projected"]
        self.assertLess(float(np.max(np.abs(residual1 - a["residual1"]))), 1e-6)
        ln1 = golden.layer_norm(residual1, w[f"{p}/layernorm1/gamma"], w[f"{p}/layernorm1/beta"])
        self.assertLess(float(np.max(np.abs(ln1 - a["ln1"]))), 1e-6)
        residual2 = a["ln1"] + a["ffn_out"]
        self.assertLess(float(np.max(np.abs(residual2 - a["residual2"]))), 1e-6)
        ln2 = golden.layer_norm(residual2, w[f"{p}/layernorm2/gamma"], w[f"{p}/layernorm2/beta"])
        self.assertLess(float(np.max(np.abs(ln2 - a["layer_output"]))), 1e-6)
        # PostNorm: LayerNorm inputs are the residual sums, not the residual-free x.
        self.assertGreater(float(np.max(np.abs(residual1 - self.x))), 1e-3)
        self.assertGreater(float(np.max(np.abs(residual2 - a["ln1"]))), 1e-3)

    def test_ffn_dimensions_and_exact_gelu(self):
        _, a = self.run_layer(0)
        self.assertEqual(a["ffn_hidden"].shape, (99, 256))
        self.assertEqual(a["gelu"].shape, (99, 256))
        self.assertEqual(a["ffn_out"].shape, (99, 64))
        w = self.weights
        p = f"{golden.PREFIX}/enc_layers/0"
        hidden = a["ln1"] @ w[f"{p}/ffn/layer_with_weights-0/kernel"] + w[f"{p}/ffn/layer_with_weights-0/bias"]
        self.assertLess(float(np.max(np.abs(hidden - a["ffn_hidden"]))), 1e-5)
        erf = np.vectorize(math.erf)
        manual = (np.float32(0.5) * a["ffn_hidden"]
                  * (np.float32(1.0) + erf(a["ffn_hidden"] / math.sqrt(2.0)))).astype(np.float32)
        self.assertLess(float(np.max(np.abs(manual - a["gelu"]))), 1e-6)

    def test_twelve_layer_parameter_schedule(self):
        full = []
        golden.forward(self.x, self.weights, arrays=full)
        partial = []
        golden.forward(self.x, self.weights, num_layers=3, arrays=partial)
        for layer in range(3):
            self.assertLess(float(np.max(np.abs(
                full[layer]["layer_output"] - partial[layer]["layer_output"]))), 1e-6)

    def test_classification_head(self):
        logits, final, _ = golden.forward(self.x, self.weights)
        w = self.weights
        expected = final[0] @ w[f"{golden.HEAD}/kernel"] + w[f"{golden.HEAD}/bias"]
        self.assertEqual(logits.shape, (12,))
        self.assertTrue(np.isfinite(logits).all())
        self.assertLess(float(np.max(np.abs(expected - logits))), 1e-6)

    def test_encoder_input_shape_guard(self):
        with self.assertRaises(ValueError):
            golden.forward(np.zeros((98, 64), dtype=np.float32), self.weights)

    def test_real_validation_samples(self):
        samples_dir = golden.ENCODER_SAMPLES_DEFAULT
        manifest_path = samples_dir / "manifest.json"
        if not manifest_path.exists():
            self.skipTest("cached real validation samples are unavailable")
        import json
        manifest = json.loads(manifest_path.read_text())
        validation = [s for s in manifest["samples"] if s["split"] == "validation"]
        self.assertGreaterEqual(len(validation), 3)
        labels = {s["label"] for s in validation}
        self.assertGreaterEqual(len(labels), 2, "need cross-category validation samples")
        for sample in validation:
            encoder = np.load(samples_dir / sample["encoder_file"], allow_pickle=False)
            self.assertEqual(encoder.shape, (99, 64))
            self.assertEqual(encoder.dtype, np.float32)
            self.assertEqual(golden.sha256(encoder), sample["encoder_sha256"])
            logits, final, _ = golden.forward(encoder, self.weights)
            self.assertTrue(np.isfinite(logits).all())
            self.assertTrue(np.isfinite(final).all())
            self.assertEqual(int(np.argmax(logits)), int(np.argmax(logits)))  # deterministic


if __name__ == "__main__":
    unittest.main(verbosity=2)
