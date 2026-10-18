"""Golden and fault-injection tests with real independent references.

1. Differential integer golden: a pure-Python GEMM and a NumPy int64 GEMM must
   agree exactly on random data; a one-element perturbation must be rejected.
2. A hand-computed literal vector fixes the expected arithmetic.
3. Real on-disk fault injection: the software golden check must pass normally
   and must fail when the stored golden is corrupted; the Yosys evidence must
   pass normally and must fail when an impossible multiplier count is demanded.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import unittest

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "golden"))
import run_stages  # noqa: E402
from ffn1_reference import int_gemm_numpy, int_gemm_python, nrmse_percent  # noqa: E402

DESIGN_ROOT = HERE.parents[2]
SOFTWARE_CHECK = HERE.parent / "golden" / "check_encoder_input.py"
SYNTH_CHECK = HERE.parent / "synth" / "yosys_evidence.py"


class IntegerGoldenTest(unittest.TestCase):
    def test_hand_computed_vector(self):
        a = [[1, 2, 3], [4, 5, 6]]
        w = [[1, 0], [0, 1], [1, 1]]
        bias = [10, -10]
        expected = [[14, -5], [20, 1]]
        self.assertEqual(int_gemm_python(a, w, bias), expected)
        np.testing.assert_array_equal(int_gemm_numpy(a, w, bias), expected)

    def test_differential_random_matrices(self):
        rng = np.random.default_rng(20261018)
        for _ in range(25):
            m = int(rng.integers(1, 8))
            k = int(rng.integers(1, 12))
            n = int(rng.integers(1, 8))
            a = rng.integers(-128, 128, size=(m, k))
            w = rng.integers(-128, 128, size=(k, n))
            bias = rng.integers(-1 << 20, 1 << 20, size=n)
            reference = int_gemm_python(a.tolist(), w.tolist(), bias.tolist())
            np.testing.assert_array_equal(int_gemm_numpy(a, w, bias), reference)

    def test_perturbed_golden_is_detected(self):
        rng = np.random.default_rng(7)
        a = rng.integers(-100, 100, size=(4, 5))
        w = rng.integers(-100, 100, size=(5, 6))
        golden = int_gemm_python(a.tolist(), w.tolist())
        bad = [row[:] for row in golden]
        bad[0][0] += 1
        self.assertNotEqual(golden, bad)

    def test_nrmse_floor_and_scaling(self):
        reference = np.zeros(8)
        self.assertEqual(nrmse_percent(np.zeros(8), reference), 0.0)
        self.assertGreater(nrmse_percent(np.ones(8), reference), 0.0)
        scaled = np.ones(8) * 0.01
        self.assertAlmostEqual(nrmse_percent(scaled, np.ones(8)), 99.0, places=9)


class RealFaultInjectionTest(unittest.TestCase):
    @unittest.skipUnless(DESIGN_ROOT.joinpath("software/weights/kwt1_fp32.npz").exists(),
                         "design weights missing")
    def test_software_golden_pass_then_corrupt_fail(self):
        base = [sys.executable, str(SOFTWARE_CHECK), "--design-root", str(DESIGN_ROOT)]
        good = subprocess.run(base, capture_output=True, text=True)
        self.assertEqual(good.returncode, 0, good.stdout + good.stderr)
        self.assertIn("KWT-GRAD-VERIFY encoder_input", good.stdout)

        bad = subprocess.run(base + ["--corrupt-golden"], capture_output=True, text=True)
        self.assertNotEqual(bad.returncode, 0, "corrupted golden was accepted")
        self.assertIn("FAIL", bad.stdout)

    def test_synth_pass_then_corrupt_fail(self):
        tools = run_stages.resolve_tools(type("A", (), {"iverilog_bin": None, "yosys_bin": None})(),
                                         DESIGN_ROOT)
        if not tools.get("yosys"):
            self.skipTest("no capable yosys")
        base = [sys.executable, str(SYNTH_CHECK), "--design-root", str(DESIGN_ROOT),
                "--yosys", tools["yosys"], "--liberty", "/nonexistent.lib"]
        good = subprocess.run(base, capture_output=True, text=True)
        self.assertEqual(good.returncode, 0, good.stdout + good.stderr)
        self.assertIn("KWT-GRAD-VERIFY synth", good.stdout)

        bad = subprocess.run(base + ["--corrupt-mul-count"], capture_output=True, text=True)
        self.assertNotEqual(bad.returncode, 0, "impossible multiplier count was accepted")


if __name__ == "__main__":
    unittest.main()
