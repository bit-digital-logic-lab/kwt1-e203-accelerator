#!/usr/bin/env python3
"""Require 12 clean full-layer E203 runs and preserve their provenance."""

import argparse
import hashlib
import json
import re
from pathlib import Path

PATTERN = re.compile(
    r"^PASS frozen real-sample E203 FFN1 tiles: "
    r"start=(\d+) count=(\d+) outputs=(\d+) padding=(\d+) "
    r"task_cycles=(\d+) total_cycles=(\d+)$", re.MULTILINE)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.vectors / "manifest.json").read_text())
    records = []
    if [entry["layer"] for entry in manifest["layers"]] != list(range(1, 13)):
        raise ValueError("vector manifest must contain layers 1..12 once each")
    for entry in manifest["layers"]:
        layer = entry["layer"]
        stem = f"layer{layer:02}"
        for file_record in entry["files"].values():
            path = Path(file_record["path"])
            if sha256(path) != file_record["sha256"]:
                raise ValueError(f"{stem} vector SHA mismatch: {path}")
        log_path = args.logs / f"{stem}.log"
        exit_path = args.logs / f"{stem}.exit"
        if not log_path.is_file() or not exit_path.is_file() or exit_path.read_text().strip() != "0":
            raise ValueError(f"{stem} missing log or clean exit")
        matches = PATTERN.findall(log_path.read_text())
        if len(matches) != 1:
            raise ValueError(f"{stem} must have exactly one full-layer PASS")
        start, count, outputs, padding, task_cycles, total_cycles = map(int, matches[0])
        if (start, count, outputs, padding) != (0, 208, 25344, 1280):
            raise ValueError(f"{stem} did not verify the full layer")
        records.append({"layer": layer, "task_cycles": task_cycles,
                        "total_cycles": total_cycles, "checked_outputs": outputs,
                        "checked_padding": padding, "log_sha256": sha256(log_path)})
    report = {"sample_path": manifest["sample_path"], "label": manifest["label"],
              "format": manifest["format"],
              "sample_archive_sha256": manifest["sample_archive_sha256"],
              "parameter_archive_sha256": manifest["parameter_archive_sha256"],
              "fp32_weight_archive_sha256": manifest["fp32_weight_archive_sha256"],
              "worst_layer_nrmse_pct": manifest["worst_layer_nrmse_pct"],
              "hardware_boundary": "Icarus E203 plus provisional NICE-like system, DMA, old 8x16 engine, simulated 512KiB shared RAM",
              "layers": records, "checked_outputs": sum(r["checked_outputs"] for r in records),
              "checked_padding": sum(r["checked_padding"] for r in records),
              "total_task_cycles": sum(r["task_cycles"] for r in records)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"PASS {len(records)} full E203 tasks, {report['checked_outputs']} outputs, "
          f"{report['checked_padding']} padding positions; {args.out}")


if __name__ == "__main__":
    main()
