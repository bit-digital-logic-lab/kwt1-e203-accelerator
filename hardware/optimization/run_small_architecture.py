#!/usr/bin/env python3
"""Run independent tile/microkernel checks concurrently through quiet hooks."""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
HOOK = REPO / "hardware/optimization/run_hook.py"
STATUS = Path("/tmp/kwt-optimization-hooks")
CASES = {
    "tiling_experiment": (20, ["python3", "hardware/optimization/tiling_experiment.py",
                               "--out", str(STATUS / "tiling_results.json")]),
    "tiling_k16_spill": (20, ["python3", "hardware/optimization/tiling_experiment.py",
                               "--tiles", "8x16", "--k-block", "16", "--spill-psum",
                               "--out", str(STATUS / "tiling_k16_spill_results.json")]),
    "matrix_stride": (20, ["python3", "hardware/optimization/matrix_stride_experiment.py"]),
    "kernel_collision": (120, ["bash", "hardware/optimization/kernel_collision.sh"]),
    "kernel_os": (120, ["bash", "hardware/optimization/kernel_os.sh"]),
    "kernel_structural": (30, ["bash", "hardware/optimization/kernel_structural.sh"]),
    "kernel_row16": (90, ["bash", "hardware/architecture/run_mac_continuous.sh"]),
}


def run(name: str, timeout: int, command: list[str]) -> tuple[str, int, str]:
    result = subprocess.run(
        [sys.executable, str(HOOK), name, "--timeout", str(timeout), "--", *command],
        cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        check=False,
    )
    status = STATUS / f"{name}.json"
    if not status.exists():
        return name, result.returncode or 1, result.stdout.strip()
    payload = json.loads(status.read_text())
    return name, payload["exit_code"], result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    results = {}
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(run, name, *spec): name for name, spec in CASES.items()}
        for future in as_completed(futures):
            name, code, message = future.result()
            results[name] = (code, message)
    for name in CASES:
        code, message = results[name]
        print(f"{'PASS' if code == 0 else 'FAIL'} {name}: {message}")
    failures = [name for name, (code, _) in results.items() if code]
    if failures:
        print(f"FAIL small architecture checks: {', '.join(failures)}", file=sys.stderr)
        return 1
    print(f"PASS {len(CASES)} small architecture checks; no E203/full-layer simulation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
