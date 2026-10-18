#!/usr/bin/env python3
"""Record interpreter and simulator versions as a machine-readable result."""

import json
import platform
import shutil
import subprocess
import sys


def probe(name: str, command: list[str]) -> str | None:
    if not shutil.which(command[0]):
        return None
    try:
        proc = subprocess.run(command, text=True, capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
        return f"error: {exc}"
    text = (proc.stdout + proc.stderr).strip()
    return text.splitlines()[0] if text else ""


def main() -> int:
    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "iverilog": probe("iverilog", ["iverilog", "-V"]),
        "vvp": probe("vvp", ["vvp", "-V"]),
        "verilator": probe("verilator", ["verilator", "--version"]),
        "yosys": probe("yosys", ["yosys", "-V"]),
    }
    print("RESULT_JSON:" + json.dumps(versions, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
