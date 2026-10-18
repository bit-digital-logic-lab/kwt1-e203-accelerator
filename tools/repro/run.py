#!/usr/bin/env python3
"""Reproduction entry point for the KWT-1 undergraduate accelerator worktree.

Two levels:

* ``smoke`` (default) runs only small, fast, dependency-light checks that are
  expected to pass in a clean checkout.
* ``full`` (must be requested explicitly with ``--level full``) additionally
  runs the heavier simulation checks, including the checks that are currently
  known to hang or fail under this environment.  It never silently omits them.

Every check is executed for real with a hard timeout.  A check is reported as
PASS only when the process exits with the expected code inside the timeout;
the mere existence of an output file is never treated as a pass.  Results are
written as machine-readable JSON under ``tools/repro/results/<run-id>/``.

This harness is a *reproduction aid*, not competition acceptance.  See
``tools/repro/README.md`` for dependencies and coverage boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPRO = ROOT / "tools/repro"
SCHEMA = "kwt-repro/1"

# ---------------------------------------------------------------------------
# Check manifest.
#
# gate=True  -> the check is part of the pass/fail gate for its level.
# gate=False -> the check is executed and its real outcome recorded, but a
#               non-passing result does not change the overall verdict.  These
#               document current environment limits / known-broken tests.
# enabled=False -> not selected by a level; only runnable with ``--only``.
# prereq     -> list of repo-relative paths that must exist, otherwise SKIP.
# inputs     -> repo-relative globs whose sha256 is recorded for the check.
# ---------------------------------------------------------------------------
CHECKS: list[dict] = [
    {
        "id": "env.versions",
        "level": "smoke",
        "gate": True,
        "desc": "record interpreter and simulator versions",
        "cmd": ["python3", "tools/repro/checks/env_versions.py"],
        "timeout": 60,
        "inputs": [],
    },
    {
        "id": "sw.offline_encoder_contract",
        "level": "smoke",
        "gate": True,
        "desc": "numpy-only replay of offline_encoder_input.convert against frozen artifacts",
        "cmd": ["python3", "tools/repro/checks/offline_encoder_contract.py"],
        "timeout": 120,
        "inputs": [
            "software/offline_encoder_input.py",
            "software/weights/kwt1_fp32.npz",
            "software/validation/real_input_chain_20261017/*.npy",
            "software/validation/real_input_chain_20261017/manifest.json",
        ],
    },
    {
        "id": "rtl.dma_iverilog",
        "level": "smoke",
        "gate": True,
        "desc": "DMA load/writeback, tails, bounds, invalid descriptors (Icarus)",
        "cmd": ["bash", "hardware/rtl/dma/tb/run.sh"],
        "timeout": 120,
        "inputs": [
            "hardware/rtl/dma/tb/run.sh",
            "hardware/rtl/dma/tb/tb_kwt_dma.sv",
            "hardware/rtl/dma/kwt_dma.sv",
            "hardware/rtl/dma/kwt_ext_memory_model.sv",
            "hardware/rtl/sram/kwt_tile_buffers.sv",
        ],
    },
    {
        "id": "rtl.dma_2d_verilator",
        "level": "smoke",
        "gate": True,
        "desc": "2D stride DMA prototype (Verilator)",
        "cmd": ["bash", "hardware/rtl/dma/tb/run_2d_verilator.sh"],
        "timeout": 180,
        "inputs": [
            "hardware/rtl/dma/tb/run_2d_verilator.sh",
            "hardware/rtl/dma/tb/tb_kwt_dma_2d.sv",
            "hardware/rtl/dma/kwt_dma.sv",
            "hardware/rtl/dma/kwt_dma_2d.sv",
            "hardware/rtl/dma/kwt_ext_memory_model.sv",
            "hardware/rtl/sram/kwt_tile_buffers.sv",
        ],
    },
    {
        "id": "rtl.control_verilator",
        "level": "smoke",
        "gate": True,
        "desc": "single-tile system model incl. ping-pong CACC (Verilator)",
        "cmd": ["bash", "hardware/rtl/control/tb/run_verilator.sh"],
        "timeout": 240,
        "inputs": [
            "hardware/rtl/control/tb/run_verilator.sh",
            "hardware/rtl/control/tb/tb_kwt_tile_system_model.sv",
            "hardware/rtl/control/kwt_tile_system_model.sv",
            "hardware/rtl/pulse_array/*.sv",
            "hardware/rtl/sram/*.sv",
            "hardware/rtl/dma/*.sv",
        ],
    },
    # ------- full-only checks -------
    {
        "id": "vectors.int16",
        "level": "full",
        "gate": False,
        "desc": "independent INT16 vector; known partial mismatch under Icarus here",
        "cmd": [
            "python3",
            "hardware/testbenchplus/common/run_vectors.py",
            "hardware/testbenchplus/vectors/layer0_int16/case.json",
        ],
        "timeout": 120,
        "inputs": [
            "hardware/testbenchplus/common/run_vectors.py",
            "hardware/testbenchplus/common/tb_array_vector.sv",
            "hardware/testbenchplus/vectors/layer0_int16/case.json",
            "hardware/rtl/pulse_array/kwt_pe.sv",
            "hardware/rtl/pulse_array/kwt_pulse_array.sv",
        ],
    },
    {
        "id": "rtl.pingpong_cacc_verilator",
        "level": "full",
        "gate": True,
        "desc": "256 INT32 results of the ping-pong CACC engine (Verilator)",
        "cmd": ["bash", "hardware/rtl/pulse_array/tb/run_pingpong_cacc_verilator.sh"],
        "timeout": 300,
        "inputs": [
            "hardware/rtl/pulse_array/tb/run_pingpong_cacc_verilator.sh",
            "hardware/rtl/pulse_array/tb/tb_kwt_pingpong_cacc_engine.sv",
            "hardware/rtl/pulse_array/*.sv",
            "hardware/rtl/sram/kwt_operand_pingpong.sv",
        ],
    },
    {
        "id": "vectors.all",
        "level": "full",
        "gate": False,
        "desc": "all testbenchplus vectors; FP32 case currently does not terminate",
        "cmd": ["python3", "hardware/testbenchplus/common/run_vectors.py"],
        "timeout": 60,
        "inputs": [
            "hardware/testbenchplus/common/run_vectors.py",
            "hardware/testbenchplus/common/tb_array_vector.sv",
            "hardware/testbenchplus/vectors/*/case.json",
            "hardware/rtl/pulse_array/kwt_pe.sv",
            "hardware/rtl/pulse_array/kwt_pulse_array.sv",
        ],
    },
    {
        "id": "rtl.pulse_array_iverilog",
        "level": "full",
        "gate": False,
        "desc": "array timing/integration regression (Icarus); known TIMING FAIL here",
        "cmd": ["bash", "hardware/rtl/pulse_array/tb/run.sh"],
        "timeout": 180,
        "inputs": [
            "hardware/rtl/pulse_array/tb/run.sh",
            "hardware/rtl/pulse_array/tb/tb_kwt_pulse_array.sv",
            "hardware/rtl/pulse_array/*.sv",
        ],
    },
    {
        "id": "rtl.control_iverilog",
        "level": "full",
        "gate": False,
        "desc": "system model under Icarus; known elaboration failure here",
        "cmd": ["bash", "hardware/rtl/control/tb/run.sh"],
        "timeout": 180,
        "inputs": [
            "hardware/rtl/control/tb/run.sh",
            "hardware/rtl/control/tb/tb_kwt_tile_system_model.sv",
            "hardware/rtl/control/kwt_tile_system_model.sv",
            "hardware/rtl/pulse_array/*.sv",
            "hardware/rtl/sram/*.sv",
            "hardware/rtl/dma/*.sv",
        ],
    },
    {
        "id": "rtl.engine_iverilog",
        "level": "full",
        "gate": False,
        "desc": "legacy single-tile engine (Icarus); known compile failure here",
        "cmd": ["bash", "hardware/rtl/pulse_array/tb/run_engine.sh"],
        "timeout": 180,
        "inputs": [
            "hardware/rtl/pulse_array/tb/run_engine.sh",
            "hardware/rtl/pulse_array/tb/tb_kwt_tile_engine.sv",
            "hardware/rtl/pulse_array/*.sv",
            "hardware/rtl/sram/kwt_tile_buffers.sv",
        ],
    },
    {
        "id": "e203.full_layer_208tile",
        "level": "full",
        "gate": False,
        "desc": "208-tile FFN1 full layer on the real E203 core (resource-gated)",
        "cmd": ["bash", "hardware/rtl/control/tb/run_e203.sh", "--full-layer"],
        "timeout": 3600,
        "prereq": ["资料/e203_hbirdv2_full"],
        "prereq_tools": ["riscv64-linux-gnu-gcc", "riscv64-linux-gnu-objcopy", "xxd"],
        "inputs": [
            "hardware/rtl/control/tb/run_e203.sh",
            "hardware/rtl/control/tb/e203_kwt_full_layer.S",
            "hardware/rtl/control/tb/tb_kwt_full_layer_e203.sv",
        ],
    },
    # ------- probe checks: disabled by default, used for harness self-test -------
    {
        "id": "fault.probe_exit7",
        "level": "smoke",
        "gate": True,
        "enabled": False,
        "desc": "controlled-failure probe: exits 7 (harness failure-propagation test)",
        "cmd": ["python3", "-c", "print('probe'); raise SystemExit(7)"],
        "timeout": 30,
        "inputs": [],
    },
    {
        "id": "fault.probe_timeout",
        "level": "smoke",
        "gate": True,
        "enabled": False,
        "desc": "controlled-failure probe: sleeps past a 2s timeout",
        "cmd": ["python3", "-c", "import time; time.sleep(30)"],
        "timeout": 2,
        "inputs": [],
    },
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_inputs(patterns: list[str]) -> list[dict]:
    files: list[Path] = []
    for pattern in patterns:
        matches = sorted(ROOT.glob(pattern))
        if not matches:
            files.append(ROOT / pattern)  # record the miss explicitly
        files.extend(matches)
    records: list[dict] = []
    for path in files:
        if path.is_file():
            records.append({
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            })
        else:
            records.append({"path": str(path.relative_to(ROOT)),
                            "sha256": None, "missing": True})
    return records


def tool_versions() -> dict:
    commands = {
        "python3": ["python3", "--version"],
        "iverilog": ["iverilog", "-V"],
        "verilator": ["verilator", "--version"],
        "yosys": ["yosys", "-V"],
        "riskv_gcc": ["riscv64-linux-gnu-gcc", "--version"],
    }
    versions: dict[str, str | None] = {}
    for name, command in commands.items():
        if not shutil.which(command[0]):
            versions[name] = None
            continue
        try:
            proc = subprocess.run(command, text=True, capture_output=True, timeout=30)
            text = (proc.stdout + proc.stderr).strip()
            versions[name] = text.splitlines()[0] if text else ""
        except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
            versions[name] = f"error: {exc}"
    return versions


def git_state() -> dict:
    def run(args: list[str]) -> str:
        try:
            proc = subprocess.run(["git", *args], cwd=ROOT, text=True,
                                  capture_output=True, timeout=30)
            return proc.stdout.strip() if proc.returncode == 0 else ""
        except (OSError, subprocess.SubprocessError):
            return ""

    return {
        "commit": run(["rev-parse", "HEAD"]),
        "branch": run(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(run(["status", "--porcelain"])),
    }


def hosts() -> dict:
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }


def run_check(check: dict, run_dir: Path) -> dict:
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    missing = [path for path in check.get("prereq", []) if not (ROOT / path).exists()]
    missing_tools = [tool for tool in check.get("prereq_tools", [])
                     if not shutil.which(tool)]
    record: dict = {
        "id": check["id"],
        "level": check["level"],
        "gate": check.get("gate", True),
        "desc": check["desc"],
        "command": check["cmd"],
        "cwd": str(ROOT),
        "started_at": started_at,
        "prereq_missing": missing,
        "prereq_tools_missing": missing_tools,
    }
    if missing or missing_tools:
        parts = []
        if missing:
            parts.append("missing prerequisite path(s): " + ", ".join(missing))
        if missing_tools:
            parts.append("missing tool(s): " + ", ".join(missing_tools))
        record.update({
            "status": "SKIP",
            "exit_code": None,
            "timed_out": False,
            "duration_s": round(time.monotonic() - started, 3),
            "reason": "; ".join(parts),
        })
        print(f"SKIP {check['id']}: {record['reason']}")
        return record

    stdout_path = run_dir / f"{check['id']}.stdout.log"
    stderr_path = run_dir / f"{check['id']}.stderr.log"
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    timed_out = False
    exit_code: int | None = None
    with stdout_path.open("w") as out, stderr_path.open("w") as err:
        try:
            proc = subprocess.run(check["cmd"], cwd=ROOT, env=env,
                                  stdout=out, stderr=err,
                                  timeout=check["timeout"])
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
        except OSError as exc:
            err.write(f"failed to launch: {exc}\n")

    expected = check.get("expect_exit", 0)
    if timed_out:
        status = "TIMEOUT"
    elif exit_code is None:
        status = "ERROR"
    elif exit_code == expected:
        status = "PASS"
    else:
        status = "FAIL"

    stdout_text = stdout_path.read_text(errors="replace")
    parsed = None
    for line in stdout_text.splitlines():
        if line.startswith("RESULT_JSON:"):
            try:
                parsed = json.loads(line[len("RESULT_JSON:"):])
            except json.JSONDecodeError:
                parsed = {"error": "unparseable RESULT_JSON line"}
    record.update({
        "status": status,
        "exit_code": exit_code,
        "expected_exit": expected,
        "timed_out": timed_out,
        "duration_s": round(time.monotonic() - started, 3),
        "timeout_s": check["timeout"],
        "inputs": resolve_inputs(check.get("inputs", [])),
        "stdout_log": str(stdout_path.relative_to(ROOT)),
        "stderr_log": str(stderr_path.relative_to(ROOT)),
        "stdout_tail": stdout_text.strip().splitlines()[-5:],
        "stderr_tail": stderr_path.read_text(errors="replace").strip().splitlines()[-5:],
        "parsed_result": parsed,
    })
    print(f"{status:7s} {check['id']} (exit={exit_code}, {record['duration_s']}s)")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--level", choices=("smoke", "full"), default="smoke",
                        help="smoke (default) or explicit full")
    parser.add_argument("--only", action="append", default=[], metavar="ID",
                        help="run only these check ids (also enables disabled probes)")
    parser.add_argument("--out", type=Path, default=None,
                        help="override the results directory")
    parser.add_argument("--list", action="store_true", help="list checks and exit")
    args = parser.parse_args()

    if args.list:
        for check in CHECKS:
            print(f"{check['level']:5s} gate={int(check.get('gate', True))} "
                  f"{check['id']}: {check['desc']}")
        return 0

    if args.only:
        selected = [c for c in CHECKS if c["id"] in args.only]
        unknown = set(args.only) - {c["id"] for c in selected}
        if unknown:
            parser.error(f"unknown check id(s): {', '.join(sorted(unknown))}")
    else:
        selected = [c for c in CHECKS
                    if c.get("enabled", True) and c["level"] == args.level]
        if args.level == "full":
            # full is the union: smoke checks first, then full-only checks.
            selected = [c for c in CHECKS if c.get("enabled", True)]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{args.level}"
    run_dir = args.out or (REPRO / "results" / run_id)
    run_dir = run_dir if run_dir.is_absolute() else (ROOT / run_dir)
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"level={args.level} checks={len(selected)} results={run_dir}")
    records = [run_check(check, run_dir) for check in selected]

    gate = [r for r in records if r.get("gate", True)]
    counts = {status: sum(1 for r in records if r["status"] == status)
              for status in ("PASS", "FAIL", "TIMEOUT", "SKIP", "ERROR")}
    if gate:
        ok = all(r["status"] == "PASS" for r in gate)
    else:
        # Only non-gate checks selected (e.g. a resource-gated probe): a SKIP
        # is acceptable, a real failure/timeout is not.
        ok = not any(r["status"] in ("FAIL", "TIMEOUT", "ERROR") for r in records)
    summary = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "level": args.level,
        "run_id": run_dir.name,
        "git": git_state(),
        "environment": {"tools": tool_versions(), "host": hosts()},
        "counts": counts,
        "gate_ok": ok,
        "gate_ids": [r["id"] for r in gate],
        "checks": records,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    non_gate_failures = [r["id"] for r in records
                         if not r.get("gate", True)
                         and r["status"] in ("FAIL", "TIMEOUT", "ERROR")]
    print(f"summary: {summary_path}")
    print(f"counts={counts} gate_ok={ok}"
          + (f" non_gate_failures={non_gate_failures}" if non_gate_failures else ""))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
