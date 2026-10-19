#!/usr/bin/env python3
"""Streaming MAC experiment: true baseline (single-CACC staged task) vs a
dual-CACC dataflow that overlaps CACC export with the next tile's compute.

Both engines use the SAME arithmetic RTL (kwt_row_broadcast_tile.sv, unchanged)
and the SAME output stage (kwt_tile_output_stage.sv, unchanged) and the SAME
16-bit-word bus / memory timing models. The only difference is the task-level
schedule: the baseline exports a tile's accumulator and waits (serial CACC
drain); the dual dataflow alternates two accumulator pages so the export of
tile T runs under the compute window of tile T+1.

This script:
  * builds and runs BOTH Verilator binaries (baseline binary is the existing,
    unmodified hardware/architecture TB),
  * checks every INT32 output against an independent numpy reference,
  * checks architectural access counts,
  * writes results.json and per-case logs under logs/.

It is a functional/task-level simulation, NOT an E203 run, NOT a 150 MHz STA.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PA = ROOT / "hardware" / "rtl" / "pulse_array"
CORE = PA / "kwt_row_broadcast_tile.sv"
STAGE = PA / "kwt_tile_output_stage.sv"
BASE_TB = ROOT / "hardware" / "architecture" / "tb" / "tb_kwt_row_broadcast_staged_task.sv"
DUAL_TB = HERE / "tb" / "tb_kwt_streaming_task.sv"
UNIT_TB = HERE / "tb" / "tb_kwt_dualacc_unit.sv"

MAX_M, MAX_K, MAX_N = 99, 64, 256
DBL_FULL_BASELINE = 154240   # committed staged_task_row16_summary.json "full"


def wrap32(number):
    return number & 0xFFFFFFFF


def make_case(name):
    if name == "full":
        m, k, n = 99, 64, 256
        a = [[((row * 3 + kk * 5) % 15) - 7 for kk in range(k)] for row in range(m)]
        w = [[((kk * 7 + col * 2) % 13) - 6 for col in range(n)] for kk in range(k)]
        bias = [(col % 17) - 8 for col in range(n)]
        gaps = 0
    elif name == "overflow":
        m, k, n = 8, 64, 16
        a = [[-32768 if (row + kk) & 1 else 32767 for kk in range(k)] for row in range(m)]
        w = [[-32768 if (kk + col) & 1 else 32767 for col in range(n)] for kk in range(k)]
        bias = [0x7FFFFFFF if col & 1 else -0x80000000 for col in range(n)]
        gaps = 1
    elif name == "twotile":
        m, k, n = 32, 8, 16
        rng = np.random.default_rng(101)
        a = rng.integers(-2000, 2000, size=(m, k)).tolist()
        w = rng.integers(-2000, 2000, size=(k, n)).tolist()
        bias = [int(v) for v in rng.integers(-10 ** 6, 10 ** 6, size=n)]
        gaps = 1
    else:
        m, k, n, seed, gaps = ((3, 4, 5, 17, 0) if name == "tiny"
                               else (10, 7, 19, 29, 1))
        rng = np.random.default_rng(seed)
        a = rng.integers(-100, 100, size=(m, k)).tolist()
        w = rng.integers(-100, 100, size=(k, n)).tolist()
        bias = [int(v) for v in rng.integers(-1000, 1000, size=n)]
        a[0][0] = -32768
        w[0][0] = 32767
    return {"name": name, "format": "int16", "a": a, "b": w,
            "bias": bias, "gaps": gaps}


def check_case(case):
    a, w, bias = case["a"], case["b"], case["bias"]
    m, k, n = len(a), len(a[0]), len(w[0])
    if not (1 <= m <= MAX_M and 1 <= k <= MAX_K and 1 <= n <= MAX_N):
        raise ValueError("shape exceeds 99x64 * 64x256")
    if any(len(r) != k for r in a) or len(w) != k or any(len(r) != n for r in w):
        raise ValueError("ragged operand matrix")
    if len(bias) != n:
        raise ValueError("bias length differs from N")
    if any(not -32768 <= int(v) <= 32767 for row in a + w for v in row):
        raise ValueError("int16 operand out of range")
    ref = (np.asarray(a, dtype=np.int64) @ np.asarray(w, dtype=np.int64)
           + np.asarray(bias, dtype=np.int64)) & 0xFFFFFFFF
    case["expected"] = ref.tolist()
    return m, k, n


def write_hex(path, values, count, width):
    if len(values) > count:
        raise ValueError(f"{path.name} exceeds memory size")
    mask = (1 << width) - 1
    path.write_text("".join(
        f"{int(v) & mask:0{width // 4}x}\n" for v in list(values) + [0] * (count - len(values))))


def build(workdir, top, rtl_files, extra_args=()):
    workdir.mkdir(parents=True, exist_ok=True)
    cmd = ["verilator", "--binary", "--timing", "--assert", "-Wno-fatal",
           "-j", "4", "--top-module", top, "--Mdir", str(workdir / "obj")]
    cmd += list(extra_args) + [str(f) for f in rtl_files]
    log = workdir / f"build_{top}.log"
    with log.open("w") as fh:
        res = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    if res.returncode:
        raise RuntimeError(f"Verilator build failed for {top}: see {log}")
    return workdir / "obj" / f"V{top}"


def run_case(exe, case, workdir, rows, extra=()):
    m, k, n = check_case(case)
    d = workdir / case["name"]
    d.mkdir(parents=True, exist_ok=True)
    write_hex(d / "a.hex", [v for row in case["a"] for v in row], MAX_M * MAX_K, 16)
    write_hex(d / "w.hex", [v for row in case["b"] for v in row], MAX_K * MAX_N, 16)
    write_hex(d / "bias.hex", case["bias"], MAX_N, 32)
    write_hex(d / "expected.hex", [v for row in case["expected"] for v in row], MAX_M * MAX_N, 32)
    cmd = [str(exe), f"+A={d/'a.hex'}", f"+W={d/'w.hex'}", f"+BIAS={d/'bias.hex'}",
           f"+EXPECTED={d/'expected.hex'}", f"+OUTPUT={d/'actual.txt'}",
           f"+M={m}", f"+K={k}", f"+N={n}", f"+GAPS={int(case['gaps'])}"]
    cmd += list(extra)
    res = subprocess.run(cmd, text=True, capture_output=True)
    (d / "run.log").write_text(res.stdout + res.stderr)
    if res.returncode or "PASS " not in res.stdout:
        raise RuntimeError(f"{case['name']} run failed; see {d/'run.log'}\n"
                           + res.stdout[-1500:] + res.stderr[-1500:])
    metric_line = next(l for l in res.stdout.splitlines() if l.startswith("ARCH_METRICS "))
    metrics = {kk: int(v) for kk, v in re.findall(r"(\w+)=(\d+)", metric_line)}
    expect = {"a_load_words": m * k, "w_load_words": k * n, "bias_load_words": 2 * n,
              "out_write_words": 2 * m * n,
              "a_local_reads": m * k * ((n + 15) // 16),
              "w_local_reads": ((m + rows - 1) // rows) * ((n + 15) // 16) * k * 16,
              "checked_outputs": m * n}
    for key, val in expect.items():
        if metrics.get(key) != val:
            raise ValueError(f"{case['name']} {key}={metrics.get(key)} != {val}")
    seen = set()
    for line in (d / "actual.txt").read_text().splitlines():
        row_t, col_t, val_t = line.split()
        row, col = int(row_t), int(col_t)
        if not (0 <= row < m and 0 <= col < n) or (row, col) in seen:
            raise ValueError(f"{case['name']} duplicate/invalid output ({row},{col})")
        seen.add((row, col))
        if int(val_t, 16) != case["expected"][row][col]:
            raise ValueError(f"{case['name']} mismatch ({row},{col})")
    if len(seen) != m * n:
        raise ValueError(f"{case['name']} captured {len(seen)}/{m*n}")
    metrics["shared_bytes"] = 2 * (metrics["a_load_words"] + metrics["w_load_words"]
                                   + metrics["bias_load_words"] + metrics["out_write_words"])
    metrics["valid_outputs"] = metrics["checked_outputs"]
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--out", type=Path, default=HERE)
    ap.add_argument("--quick", action="store_true", help="skip the 99x64x256 case")
    args = ap.parse_args()
    if not shutil.which("verilator"):
        ap.error("verilator is not installed")

    out = args.out.resolve()
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="kwt-streaming-", dir="/tmp/opencode"))

    case_names = ["tiny", "ragged", "twotile", "overflow"]
    if not args.quick:
        case_names.append("full")
    cases = {name: make_case(name) for name in case_names}

    results = {"meta": {
        "schema": "kwt-streaming-mac/1",
        "description": "single-CACC staged task (baseline) vs dual-CACC streaming task",
        "core_rtl": "hardware/rtl/pulse_array/kwt_row_broadcast_tile.sv (unchanged)",
        "stage_rtl": "hardware/rtl/pulse_array/kwt_tile_output_stage.sv (unchanged)",
        "baseline_tb": "hardware/architecture/tb/tb_kwt_row_broadcast_staged_task.sv (unchanged)",
        "dual_tb": "hardware/experiments/streaming_mac/tb/tb_kwt_streaming_task.sv",
        "rows": args.rows, "clock_hz": 150_000_000,
        "bus_model": "one 16-bit word/cycle load; output accepted every 2 cycles (2 slots/result)",
        "scope": "RTL functional/task-level simulation; NOT E203, NOT 150 MHz STA",
    }, "commands": [], "cases": [], "checks": [], "model": None, "unit": None}

    # Unit microtest (dual-CACC primitive: overlap, reset, back-pressure).
    res = subprocess.run(["verilator", "--binary", "--timing", "--assert", "-Wno-fatal",
                          "-j", "4", "--top-module", "tb_kwt_dualacc_unit",
                          "--Mdir", str(work / "unit"),
                          str(CORE), str(UNIT_TB)],
                         capture_output=True, text=True)
    (logs / "unit_build.log").write_text(res.stdout + res.stderr)
    if res.returncode:
        raise RuntimeError("unit build failed")
    unit_run = subprocess.run([str(work / "unit" / "Vtb_kwt_dualacc_unit")],
                              capture_output=True, text=True)
    (logs / "unit_run.log").write_text(unit_run.stdout + unit_run.stderr)
    results["unit"] = {"dualacc": "PASS" if (unit_run.returncode == 0 and
                                             "PASS dual-cacc unit" in unit_run.stdout)
                       else "FAIL"}
    results["commands"].append({"unit_build": "verilator --binary --timing --assert "
                                f"--top-module tb_kwt_dualacc_unit {CORE} {UNIT_TB}",
                                "exit": res.returncode})
    results["commands"].append({"unit_run": str(work / "unit" / "Vtb_kwt_dualacc_unit"),
                                "exit": unit_run.returncode})
    if results["unit"]["dualacc"] != "PASS":
        raise RuntimeError("dual-CACC unit test failed")

    base_exe = build(work / "base", "tb_kwt_row_broadcast_staged_task",
                     [CORE, STAGE, BASE_TB], [f"-GROWS={args.rows}"])
    dual_exe = build(work / "dual", "tb_kwt_streaming_task",
                     [CORE, STAGE, DUAL_TB], [f"-GROWS={args.rows}"])
    results["commands"].append({"baseline_build": f"verilator ... -GROWS={args.rows} "
                                f"{CORE} {STAGE} {BASE_TB}", "exit": 0})
    results["commands"].append({"dual_build": f"verilator ... -GROWS={args.rows} "
                                f"{CORE} {STAGE} {DUAL_TB}", "exit": 0})

    for name in case_names:
        case = cases[name]
        base = run_case(base_exe, case, work / "base", args.rows)
        dual = run_case(dual_exe, case, work / "dual", args.rows)
        m, k, n = len(case["a"]), len(case["a"][0]), len(case["b"][0])
        row_tiles = (m + args.rows - 1) // args.rows
        col_tiles = (n + 15) // 16
        pure_mac = row_tiles * k * n
        entry = {
            "name": name, "m": m, "k": k, "n": n,
            "tiles": row_tiles * col_tiles, "pure_mac_compute_cycles": pure_mac,
            "bytes": {"shared_read_bytes": base["shared_bytes"],
                      "a_words": base["a_load_words"], "w_words": base["w_load_words"],
                      "bias_slots": base["bias_load_words"],
                      "out_slots": base["out_write_words"]},
            "baseline": base, "dual": dual,
            "delta": {
                "cycles_saved": base["cycles"] - dual["cycles"],
                "speedup": round(base["cycles"] / dual["cycles"], 4),
                "baseline_drain_pct": round(100.0 * base.get("drain_cycles", 0) / base["cycles"], 2),
                "dual_overlapped_captures": dual.get("overlapped_captures", 0),
                "dual_stall_cycles": dual.get("stall_cycles", 0),
            },
        }
        results["cases"].append(entry)
        results["checks"].append({
            "case": name, "baseline_int32_exact": True, "dual_int32_exact": True,
            "dual_access_counts_exact": True, "dual_outputs": dual["checked_outputs"]})
        print(f"PASS {name}: baseline={base['cycles']} dual={dual['cycles']} "
              f"saved={base['cycles']-dual['cycles']} ({entry['delta']['speedup']}x)")

    # Deep back-pressure robustness (dual only; correctness must hold).
    bp_cases = []
    for name, period in (("tiny", 4), ("ragged", 3)):
        case = cases[name]
        d = work / "dual"
        metrics = run_case(dual_exe, case, d, args.rows, [f"+BACKPRESSURE={period}"])
        bp_cases.append({"case": name, "backpressure_period": period,
                         "dual_cycles": metrics["cycles"],
                         "stall_cycles": metrics.get("stall_cycles", 0),
                         "int32_exact": True})
    results["backpressure"] = bp_cases

    # Modeled 12-layer whole-task projection (clearly separated from RTL).
    full = next((c for c in results["cases"] if c["name"] == "full"), None)
    if full is None:
        per_layer_base = per_layer_dual = None
    else:
        per_layer_base = full["baseline"]["cycles"]
        per_layer_dual = full["dual"]["cycles"]
    event_model = None
    if per_layer_base:
        model_script = HERE / "model" / "event_schedule.py"
        model_out = HERE / "model" / "model_12layer.json"
        mres = subprocess.run([sys.executable, str(model_script),
                               "--baseline-rtl", str(per_layer_base),
                               "--dual-rtl", str(per_layer_dual),
                               "--out", str(model_out)],
                              capture_output=True, text=True, cwd=str(ROOT))
        (logs / "event_schedule.log").write_text(mres.stdout + mres.stderr)
        results["commands"].append({"event_schedule_model": f"{model_script} "
                                    f"--baseline-rtl {per_layer_base} --dual-rtl {per_layer_dual}",
                                    "exit": mres.returncode})
        if mres.returncode == 0 and model_out.exists():
            event_model = json.loads(model_out.read_text())
    results["model"] = {
        "label": "MODEL-ONLY (software event schedule) - NOT RTL, NOT measured",
        "assumption": "12 identical FFN1 layers run back-to-back; per-layer stage "
                      "cycle counts are the RTL-measured full-case counts; no "
                      "inter-layer E203/bus overhead is added (set to 0).",
        "layers": 12,
        "rtl_measured_one_layer": {"baseline": per_layer_base, "dual": per_layer_dual},
        "modeled_12_layer_total_cycles": (
            {"baseline": 12 * per_layer_base, "dual": 12 * per_layer_dual}
            if per_layer_base else None),
        "modeled_12_layer_ms_at_150mhz": (
            {"baseline": round(12 * per_layer_base / 150e6 * 1e3, 4),
             "dual": round(12 * per_layer_dual / 150e6 * 1e3, 4)}
            if per_layer_base else None),
        "modeled_speedup": (round(per_layer_base / per_layer_dual, 4)
                            if per_layer_base else None),
        "ideal_pure_mac_one_layer_cycles_16rows": 114688,
        "event_schedule": event_model,
    }

    # 16-bit x 128/1024/4096 SRAM macro / port table (required by the statement).
    results["macro_port_table"] = {
        "budget": "16-bit x 128/1024/4096 single/dual-port macros only; no extra DDR",
        "a_resident_16bit_99x64": {"bytes": 12672, "needs_4096x16_macros": 2,
                                   "ports": "1R compute + 1W load; dual-port or 2 macros"},
        "w_pingpong_16bit_64x16x2": {"bytes": 4096, "needs_1024x16_macros": 2,
                                     "ports": "compute reads active page; DMA writes spare page"},
        "cacc_dual_16x16x32x2": {"bytes": 2048,
                                 "note": "regs or 16-bit macros x2 per 32-bit word; 2 pages",
                                 "sram_macros_if_macro_mapped": 4,
                                 "ports": "1R dump page + 1W accumulate page per row"},
        "output_stage_256x32": {"bytes": 1024, "sram_macros_if_macro_mapped": 2,
                                "ports": "1W capture + 1R drain"},
        "note": "macro counts are capacity-based; 32-bit words require 2 x 16-bit macros "
                "or width-cascading. Port/bank arbitration and 150 MHz STA are NOT proven here.",
    }

    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    # Copy the per-case logs next to results for the record.
    for sub in ("base", "dual"):
        for d in (work / sub).glob("*/run.log"):
            shutil.copy(d, logs / f"{sub}_{d.parent.name}_run.log")
        for f in (work / sub).glob("build_*.log"):
            shutil.copy(f, logs / f"{sub}_{f.name}")
    print(f"WROTE {out/'results.json'}")
    print(f"PASS streaming-mac experiment: {len(results['cases'])} cases, "
          f"unit={results['unit']['dualacc']}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        sys.exit(1)
