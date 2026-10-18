#!/usr/bin/env python3
"""Executable acceptance test for the frozen KWT1-FFN1-A12W10-SYM-v1 model.

Checks, in order:
  1. config record SHA-256 for the frozen params/weights/samples archive;
  2. existing E203/RTL golden vectors are reproduced bit for bit by an
     independent NumPy INT64 GEMM + INT32 wrap (12 layers);
  3. negative injections (wrong weight, wrong bias, wrong rounding) are
     detected by the same reference;
  4. two real validation samples from different labels, 12 layers, produce
     INT32 expected32 and a worst-layer NRMSE inside the 3% contest gate.

Exit code 0 only if every executable check passes.  A check whose external
resource is unavailable is reported as SKIP, never silently as PASS.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CLI = HERE / "ffn1_reference.py"
RESULTS = HERE / "results"
SUMMARY = RESULTS / "selftest_summary.json"
CONFIG_ID = "KWT1-FFN1-A12W10-SYM-v1"

DEFAULT_VECTORS = Path(
    "/home/yang/Desktop/ic设计竞赛/hardware/architecture/results/"
    "real_e203_validation0_vectors")
VECTORS = Path(os.environ.get("KWT_VECTORS_DIR", DEFAULT_VECTORS))
SAMPLES = REPO / "software/.build/quant_samples_12labels_20.npz"
BUILD = REPO / "software/.build/submission_model/test"


def run(*arguments):
    completed = subprocess.run(
        [sys.executable, str(CLI), *map(str, arguments)],
        capture_output=True, text=True)
    return completed.returncode, completed.stdout + completed.stderr


def main():
    checks = []

    def record(name, passed, detail):
        checks.append({"name": name, "passed": passed, "detail": detail})

    # 1. config record -----------------------------------------------------
    code, output = run("config", "--out", BUILD)
    config = json.loads((BUILD / "config_record.json").read_text()) if code == 0 else {}
    pinned = {
        "params": "a77f60fb7c8560adbca73db3cf64dcd5501ec0108b96a1131a00aa722be1918e",
        "weights": "8f359abf7adbe287b8f89fca98af46194f98f90f2146784731bff551a155825a",
        "samples": "ee04e849bc407149618e4ecd5ff99575687586d4e57c0ade6a0b3f7b66c8c149",
    }
    record("config_shas",
           code == 0 and config.get("sha256") == pinned,
           {"exit": code, "sha256": config.get("sha256")})

    # 2. RTL golden vectors, bit exact ------------------------------------
    if VECTORS.is_dir():
        code, output = run("verify-vectors", "--vectors", VECTORS, "--out", BUILD)
        verify = json.loads((BUILD / "verify_vectors.json").read_text()) if code == 0 else {}
        record("rtl_vectors_bit_exact",
               code == 0 and verify.get("all_bit_exact") is True,
               {"exit": code, "all_bit_exact": verify.get("all_bit_exact"),
                "vectors_dir": str(VECTORS)})
    else:
        record("rtl_vectors_bit_exact", None, f"vectors missing: {VECTORS}")

    # 3. negative injections ----------------------------------------------
    if VECTORS.is_dir():
        code, output = run("selftest", "--vectors", VECTORS, "--out", BUILD)
        self_test = json.loads((BUILD / "selftest.json").read_text()) if code == 0 else {}
        record("negative_injection_detected",
               code == 0 and self_test.get("result") == "PASS",
               {"exit": code, "checks": self_test.get("checks")})
    else:
        record("negative_injection_detected", None, f"vectors missing: {VECTORS}")

    # 4. two real samples, 12 layers, NRMSE gate --------------------------
    if SAMPLES.is_file():
        code, output = run("emit", "--sample-count", "2", "--out", BUILD / "emit")
        manifest = (BUILD / "emit/manifest.json").read_text()
        manifest = json.loads(manifest) if code == 0 else {}
        worst = manifest.get("worst_nrmse_pct")
        labels = manifest.get("sample_labels")
        record("two_samples_12_layers",
               code == 0 and worst is not None and worst <= 3.0
               and len(set(labels or [])) == 2,
               {"exit": code, "worst_nrmse_pct": worst, "labels": labels})
    else:
        record("two_samples_12_layers", None, f"samples missing: {SAMPLES}")

    failed = [c["name"] for c in checks if c["passed"] is False]
    skipped = [c["name"] for c in checks if c["passed"] is None]
    summary = {"config_id": CONFIG_ID, "checks": checks,
               "result": "PASS" if not failed else "FAIL",
               "failed": failed, "skipped": skipped}
    RESULTS.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    for check in checks:
        tag = {True: "PASS", False: "FAIL", None: "SKIP"}[check["passed"]]
        print(f"{tag} {check['name']}: {json.dumps(check['detail'], ensure_ascii=False)}")
    print(f"{summary['result']} test_submission_model -> {SUMMARY}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
