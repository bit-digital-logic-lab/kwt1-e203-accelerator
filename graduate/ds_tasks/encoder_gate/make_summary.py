#!/usr/bin/env python3
"""Assemble summary.json for the encoder_gate task from the run_gate.sh TSV."""

import json
import re
import sys
from pathlib import Path

TRUTHY = {"OK": True, "BAD": False}


def parse_gate_line(text):
    """Extract the GATE sentence printed by verify_encoder.py, if present."""
    m = re.search(
        r"GATE bit_exact=(?P<be>PASS|FAIL) nrmse=(?P<nr>PASS|FAIL) "
        r"rows=(?P<rows>\d+) stages=(?P<stages>\d+) "
        r"worst=(?P<worst>[\d.]+)% threshold=(?P<thr>[\d.]+)%", text)
    return m.groupdict() if m else {}


def main():
    tsv, pos_log, out = (Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(),
                         Path(sys.argv[3]).resolve())
    commands = []
    for line in tsv.read_text().splitlines():
        if not line.strip():
            continue
        name, expect, rc, log, verdict, why = line.split("\t")
        root = out.parents[3]                                # repository root
        try:
            rel_log = str(Path(log).relative_to(root))
        except ValueError:
            rel_log = log
        commands.append({
            "name": name,
            "expect": "exit 0" if expect == "0" else "non-zero exit",
            "exit_code": int(rc),
            "verdict": verdict,
            "log": rel_log,
            "note": why,
        })

    gate = parse_gate_line(pos_log.read_text()) if pos_log.exists() else {}
    summary = {
        "task": "encoder_gate",
        "scope": [
            "gate the per-stage FP32 NRMSE at 3% (worst stage) in verify_encoder.py",
            "negative injection to prove the gate fires",
            "training-split-only gradient-free calibration; validation = evaluation only",
        ],
        "commands": commands,
        "all_ok": all(c["verdict"] == "OK" for c in commands),
        "positive_run": gate,
        "evidence_levels": {
            "E0_software_reference": {
                "level": "software reference (NumPy integer model + FP32 golden)",
                "status": "supported",
                "where": "graduate/encoder/model/verify_encoder.py (bit-exact oracle, NRMSE golden)",
            },
            "E1_rtl_bit_exact": {
                "level": "RTL functional simulation, bit-exact (Verilator 5.030)",
                "status": "supported",
                "where": "6 stages x 99 rows bit-exact; see logs/02_verify_positive.log",
            },
            "E2_e203_integration": {
                "level": "real E203 instruction stream / System Bus / NICE / SRAM macros",
                "status": "not covered",
                "where": "E203-like transaction model only (SOFTWARE_BOUNDARY.md)",
            },
            "E3_synthesis_sta": {
                "level": "logic synthesis / 150 MHz STA / area / power",
                "status": "not covered",
                "where": "not run in this task",
            },
            "E4_official_acceptance": {
                "level": "official 600-sample evaluation set, NRMSE / accuracy",
                "status": "not covered",
                "where": "no official evaluation set/script available here",
            },
        },
    }
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"SUMMARY_WRITTEN {out} all_ok={summary['all_ok']}")


if __name__ == "__main__":
    main()
