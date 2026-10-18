"""Unittest wrapper for :mod:`audit_ffn1_input_boundary`.

Runs the independent reference and metamorphic checks on two different real
categories from the read-only shared sample cache.  The heavy raw-WAV audit is
executed by ``software/audit_ffn1_input_boundary.py`` directly.
"""

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import audit_ffn1_input_boundary as audit


class FFN1InputBoundaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:  # pragma: no cover - environment dependent
            raise unittest.SkipTest("PyTorch is unavailable")
        if not audit.SHARED_CACHE.exists():
            raise unittest.SkipTest(f"Shared sample cache is unavailable: {audit.SHARED_CACHE}")
        cls.torch = torch
        cls.w = audit.load_weights()
        from offline_encoder_input import convert
        from run import KWT1
        cls.convert = staticmethod(convert)
        cls.model = KWT1(audit.WEIGHTS, torch.device("cpu"))
        cls.samples = [(path, category, mfcc.astype(np.float32))
                       for path, category, mfcc in audit.read_cache_samples(["yes", "go"])]

    def test_two_categories_match_independent_reference(self):
        self.assertEqual({category for _, category, _ in self.samples}, {"yes", "go"})
        for path, category, mfcc in self.samples:
            record = audit.run_sample(path, category, mfcc, None, self.w, self.torch,
                                      self.model, self.convert)
            self.assertEqual(record["encoder_shape"], [audit.N_TOKENS, audit.D_MODEL])
            self.assertLess(record["encoder_independent_max_abs_error"], audit.TOL_ENCODER)
            self.assertEqual(len(record["layers"]), audit.N_LAYERS)
            for layer in record["layers"]:
                self.assertLess(layer["ffn1_max_abs_error"], audit.TOL_FFN1)
                self.assertGreater(min(layer["wrong_boundary_max_abs_error"].values()),
                                   layer["ffn1_max_abs_error"] + audit.MARGIN)

    def test_metamorphic_invariants(self):
        result = audit.metamorphic_checks(self.samples[0][2], self.samples[1][2],
                                          self.w, self.torch, self.model, self.convert)
        self.assertTrue(result["class_token_row_identical"])
        self.assertLess(result["projection_additivity_max_abs_error"], audit.TOL_METAMORPHIC)
        self.assertLess(result["projection_homogeneity_max_abs_error"], audit.TOL_METAMORPHIC)
        self.assertLess(result["ffn1_token_permutation_max_abs_error"], audit.TOL_METAMORPHIC)

    def test_cache_samples_are_distinct_categories(self):
        first, second = self.samples[0], self.samples[1]
        self.assertNotEqual(first[1], second[1])
        self.assertEqual(first[2].shape, (audit.N_FRAMES, audit.N_MFCC))


if __name__ == "__main__":
    unittest.main()
