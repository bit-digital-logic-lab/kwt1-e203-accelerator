#!/usr/bin/env python3
"""Prove the testbenchplus comparator rejects an independent golden.

The vector harness supplies `expected` itself and never derives it from DUT
output.  This self-check runs each shipped example twice through the unchanged
`run_vectors.py`: once with the original golden (must pass) and once with a
single `expected` element corrupted (must fail).  A clean run therefore shows
both that the vectors pass and that a wrong golden is actually detected.
"""

import json
import pathlib
import subprocess
import sys
import tempfile

PLUS = pathlib.Path(__file__).resolve().parents[1]
RUN_VECTORS = pathlib.Path(__file__).resolve().parent / "run_vectors.py"


def run(case_path):
    result = subprocess.run([sys.executable, str(RUN_VECTORS), str(case_path)],
                            text=True, capture_output=True)
    return result.returncode, result.stdout + result.stderr


def check_format(source):
    case = json.loads(source.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="kwt-neg-inject-") as temp:
        good_dir = pathlib.Path(temp) / "good"
        good_dir.mkdir()
        good_path = good_dir / "case.json"
        good_path.write_text(json.dumps(case), encoding="utf-8")
        good_rc, good_log = run(good_path)
        if good_rc != 0 or "result: 1/1 passed" not in good_log:
            raise SystemExit(f"{source}: original golden did not pass\n{good_log}")

        bad_case = json.loads(json.dumps(case))
        first_row = bad_case["expected"][0]
        if case["format"] == "fp32":
            first_row[0] = float(first_row[0]) + 1.0
        else:
            first_row[0] = int(first_row[0]) + 1
        bad_dir = pathlib.Path(temp) / "bad"
        bad_dir.mkdir()
        bad_path = bad_dir / "case.json"
        bad_path.write_text(json.dumps(bad_case), encoding="utf-8")
        bad_rc, bad_log = run(bad_path)
        if bad_rc == 0 or "FAIL" not in bad_log:
            raise SystemExit(
                f"{source}: corrupted golden was not detected rc={bad_rc}\n{bad_log}")
    return source.parent.name, bad_rc


def main():
    cases = sorted((PLUS / "vectors").glob("*/case.json"))
    if not cases:
        raise SystemExit("no example vectors found")
    for source in cases:
        name, bad_rc = check_format(source)
        print(f"PASS negative injection {name}: detected with exit={bad_rc}")
    print(f"PASS testbenchplus negative-injection self-check ({len(cases)} vectors)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
