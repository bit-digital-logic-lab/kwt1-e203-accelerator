"""Independent PyTorch reference and ordering checks for offline export."""

import unittest

import numpy as np
import torch

from offline_encoder_input import PREFIX, ROOT, convert
from run import KWT1


class OfflineEncoderInputTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / "weights/kwt1_fp32.npz"
        with np.load(cls.path, allow_pickle=False) as archive:
            cls.weights = {PREFIX + suffix: archive[PREFIX + suffix] for suffix in
                           ("patch_proj/kernel", "patch_proj/bias", "class_emb", "pos_emb")}

    def test_matches_model_and_token_order(self):
        model = KWT1(self.path, torch.device("cpu"))
        for seed in (0, 42):
            mfcc = np.random.default_rng(seed).normal(size=(98, 40)).astype(np.float32)
            actual = convert(mfcc, self.weights)
            with torch.inference_mode():
                reference = model.encoder_input(torch.from_numpy(mfcc)).numpy()
            self.assertEqual(actual.shape, (99, 64))
            np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=2e-5)
            np.testing.assert_allclose(actual[0],
                                       self.weights[PREFIX + "class_emb"][0, 0]
                                       + self.weights[PREFIX + "pos_emb"][0, 0], rtol=0, atol=0)
            projected_first = (mfcc[0] @ self.weights[PREFIX + "patch_proj/kernel"]
                               + self.weights[PREFIX + "patch_proj/bias"])
            np.testing.assert_allclose(actual[1], projected_first
                                       + self.weights[PREFIX + "pos_emb"][0, 1], rtol=1e-5, atol=2e-5)

    def test_rejects_invalid_features(self):
        for mfcc in (np.zeros((98, 39), np.float32), np.zeros((98, 40), np.float64),
                     np.full((98, 40), np.nan, np.float32)):
            with self.assertRaises(ValueError):
                convert(mfcc, self.weights)


if __name__ == "__main__":
    unittest.main()
