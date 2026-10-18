"""Fault-injection tests for the staged runner's classification.

These tests do not need a design checkout: they exercise the pure classifier
that ``run_stages.py`` uses, proving that

* a command that exits 0 but prints nothing is FAIL, not PASS;
* a forbidden marker (FAIL/MISMATCH) is FAIL;
* a contract with ``checked=0`` or ``failed>0`` is FAIL;
* a ratio marker that is not full and non-empty is FAIL;
* only a truthful contract/marker yields PASS.

The real end-to-end faults (corrupted golden on disk, wrong multiplier count,
corrupted software golden) live in ``test_golden_faults.py``.
"""

from __future__ import annotations

import pathlib
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import run_stages  # noqa: E402


def make_stage(validators, forbid=None):
    return run_stages.Stage(
        name="unit", kind="rtl", gate=[], harness=[], command=["true"],
        sample="unit", compare="unit", golden="unit",
        validators=validators, forbid=forbid or [],
    )


class HarnessFaultTest(unittest.TestCase):
    def test_exit_zero_without_output_is_fail(self):
        stage = make_stage([run_stages.contract_validator("unit")])
        status, reason = run_stages.classify(0, "", stage)
        self.assertEqual(status, "FAIL")
        self.assertIn("contract line", reason)

    def test_nonzero_exit_is_fail(self):
        stage = make_stage([run_stages.contract_validator("unit")])
        status, reason = run_stages.classify(1, "KWT-GRAD-VERIFY unit checked=5 failed=0", stage)
        self.assertEqual(status, "FAIL")
        self.assertIn("exited 1", reason)

    def test_forbidden_marker_is_fail(self):
        stage = make_stage([run_stages.result_ratio_validator()], forbid=[r"\bFAIL\b"])
        output = "result: 1/1 passed\nFAIL something"
        status, reason = run_stages.classify(0, output, stage)
        self.assertEqual(status, "FAIL")
        self.assertIn("forbidden", reason)

    def test_contract_zero_checked_is_fail(self):
        stage = make_stage([run_stages.contract_validator("unit")])
        status, reason = run_stages.classify(0, "KWT-GRAD-VERIFY unit checked=0 failed=0", stage)
        self.assertEqual(status, "FAIL")

    def test_contract_with_failures_is_fail(self):
        stage = make_stage([run_stages.contract_validator("unit")])
        status, reason = run_stages.classify(0, "KWT-GRAD-VERIFY unit checked=9 failed=2", stage)
        self.assertEqual(status, "FAIL")
        self.assertIn("failed=2", reason)

    def test_wrong_stage_contract_is_fail(self):
        stage = make_stage([run_stages.contract_validator("unit")])
        status, reason = run_stages.classify(0, "KWT-GRAD-VERIFY other checked=9 failed=0", stage)
        self.assertEqual(status, "FAIL")

    def test_partial_ratio_is_fail(self):
        stage = make_stage([run_stages.result_ratio_validator()])
        status, reason = run_stages.classify(0, "result: 1/2 passed", stage)
        self.assertEqual(status, "FAIL")
        self.assertIn("1/2", reason)

    def test_truthful_contract_is_pass(self):
        stage = make_stage([run_stages.contract_validator("unit")])
        status, reason = run_stages.classify(0, "KWT-GRAD-VERIFY unit checked=128 failed=0", stage)
        self.assertEqual(status, "PASS")

    def test_full_ratio_is_pass(self):
        stage = make_stage([run_stages.result_ratio_validator()])
        status, reason = run_stages.classify(0, "result: 2/2 passed", stage)
        self.assertEqual(status, "PASS")


class SkipSemanticsTest(unittest.TestCase):
    """A missing module must be SKIP, and module-without-harness must be FAIL."""

    def test_missing_module_is_skip(self):
        root = pathlib.Path("/nonexistent-design-root")
        stage = run_stages.Stage(
            name="attention", kind="rtl",
            gate=[root / "hardware/rtl/attention/kwt_attention.sv"],
            harness=[root / "hardware/rtl/attention/tb/run.sh"],
            command=["true"], sample="s", compare="c", golden="g", validators=[])
        tools = {"python": sys.executable}
        record = run_stages.run_stage(stage, tools, pathlib.Path("/tmp/kwt-grad-skip-test"))
        self.assertEqual(record["status"], "SKIP")
        self.assertIn("design module(s) absent", record["reason"])
        self.assertIsNone(record["exit_code"])

    def test_module_without_harness_is_fail(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            module = root / "hardware/rtl/attention/kwt_attention.sv"
            module.parent.mkdir(parents=True)
            module.write_text("module kwt_attention; endmodule\n")
            stage = run_stages.Stage(
                name="attention", kind="rtl", gate=[module],
                harness=[root / "hardware/rtl/attention/tb/run.sh"],
                command=["true"], sample="s", compare="c", golden="g", validators=[])
            tools = {"python": sys.executable}
            record = run_stages.run_stage(stage, tools, pathlib.Path(tmp) / "out")
            self.assertEqual(record["status"], "FAIL")
            self.assertIn("harness missing", record["reason"])


if __name__ == "__main__":
    unittest.main()
