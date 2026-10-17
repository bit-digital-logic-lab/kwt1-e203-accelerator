#!/usr/bin/env python3
"""Run one regression quietly; publish a small status file and keep full logs."""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="unique short name for this run")
    parser.add_argument("--out", type=Path, default=Path("/tmp/kwt-optimization-hooks"))
    parser.add_argument("--timeout", type=int, default=0, help="seconds; 0 disables")
    if "--" not in sys.argv[1:]:
        parser.error("missing -- before command")
    separator = sys.argv.index("--", 1)
    args = parser.parse_args(sys.argv[1:separator])
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.name):
        parser.error("name must contain only letters, numbers, underscores or dashes")
    command = sys.argv[separator + 1:]
    if not command:
        parser.error("missing command after --")
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / f"{args.name}.log"
    status_path = args.out / f"{args.name}.json"
    started = time.monotonic()
    timed_out = False
    with log_path.open("w") as log:
        try:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                exit_code = process.wait(timeout=args.timeout or None)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                exit_code = 124
        except OSError as exc:
            print(f"cannot start command: {exc}", file=log)
            exit_code = 127
    status = {
        "name": args.name,
        "command": command,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "elapsed_s": round(time.monotonic() - started, 3),
        "log": str(log_path.resolve()),
    }
    temp_path = status_path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    temp_path.replace(status_path)
    outcome = "PASS" if exit_code == 0 else "FAIL"
    print(f"{outcome} {args.name} rc={exit_code} elapsed={status['elapsed_s']}s "
          f"log={log_path} status={status_path}")
    if exit_code:
        print("--- last 20 log lines ---", file=sys.stderr)
        for line in log_path.read_text(errors="replace").splitlines()[-20:]:
            print(line, file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
