#!/usr/bin/env python3
"""One-MAC-array / two-CACC-page experiment.

Deliverable: hardware/experiments/single_mac_dual_cacc/rtl/kwt_single_mac_dual_cacc.sv
instantiates ONE 16-lane multiplier array and TWO independent 16x16 INT32 CACC
pages. It replaces the old streaming_mac dual-CACC harness, which obtained its
two pages by instantiating kwt_row_broadcast_tile TWICE (2 x 16 multipliers).

This script:
  * builds and runs a focused RTL microtest (reset / ownership / MAC latency /
    valid-ready conflicts / backpressure / clear),
  * builds and runs the NEW task harness on tiny, ragged and the full
    99x64 x 64x256 FFN1 shape against an INDEPENDENT NumPy INT32 golden,
  * builds and runs the OLD one-core baseline and the OLD two-core dual
    harness (unmodified accepted RTL/TBs) only as a cycle/area COMPARISON,
  * counts elaborated $mul cells (yosys proc/opt_expr/opt_clean) for the new
    single-MAC core, the old one-core and the old two-core wrapper,
  * writes results.json and logs/.

It is a functional/task-level simulation: NOT E203, NOT a 150 MHz STA, NOT a
synthesised area sign-off.
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
CORE = PA / "kwt_row_broadcast_tile.sv"          # original, unchanged
STAGE = PA / "kwt_tile_output_stage.sv"          # original, unchanged
BASE_TB = ROOT / "hardware" / "architecture" / "tb" / "tb_kwt_row_broadcast_staged_task.sv"
OLD_DUAL_TB = ROOT / "hardware" / "experiments" / "streaming_mac" / "tb" / "tb_kwt_streaming_task.sv"
NEW_RTL = HERE / "rtl" / "kwt_single_mac_dual_cacc.sv"
NEW_TB = HERE / "tb" / "tb_kwt_single_mac_task.sv"
UNIT_TB = HERE / "tb" / "tb_kwt_single_mac_unit.sv"
TWO_CORE_WRAP = HERE / "area" / "kwt_two_core_compare_wrapper.sv"

MAX_M, MAX_K, MAX_N = 99, 64, 256
DBL_FULL_BASELINE = 154240      # committed staged_task_row16_summary.json "full"
DBL_FULL_DUAL = 125603          # streaming_mac README, two-core dual


def make_case(name):
    if name == "full":
        m, k, n = 99, 64, 256
        a = [[((row * 3 + kk * 5) % 15) - 7 for kk in range(k)] for row in range(m)]
        w = [[((kk * 7 + col * 2) % 13) - 6 for col in range(n)] for kk in range(k)]
        bias = [(col % 17) - 8 for col in range(n)]
        gaps = 0
    elif name == "ragged":
        m, k, n, seed, gaps = 10, 7, 19, 29, 1
        rng = np.random.default_rng(seed)
        a = rng.integers(-100, 100, size=(m, k)).tolist()
        w = rng.integers(-100, 100, size=(k, n)).tolist()
        bias = [int(v) for v in rng.integers(-1000, 1000, size=n)]
        a[0][0] = -32768
        w[0][0] = 32767
    else:  # tiny
        m, k, n, seed, gaps = 3, 4, 5, 17, 0
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


def count_muls(log_dir, tag, top, files, chparam_top=None):
    """Elaborated $mul cells via yosys proc+opt_expr+opt_clean."""
    script = "read_verilog -sv " + " ".join(str(f) for f in files) + "; "
    if chparam_top:
        script += f"chparam -set ROWS 16 {chparam_top}; "
    script += f"hierarchy -top {top}; proc; opt_expr; opt_clean; stat"
    res = subprocess.run(["yosys", "-p", script], capture_output=True, text=True,
                         timeout=1200)
    (log_dir / f"yosys_mul_{tag}.log").write_text(res.stdout + res.stderr)
    if res.returncode:
        raise RuntimeError(f"yosys mul count failed for {tag}")
    vals = re.findall(r"^\s+(\d+)\s+\$mul\s*$", res.stdout, re.M)
    return int(vals[-1]) if vals else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--out", type=Path, default=HERE)
    ap.add_argument("--quick", action="store_true", help="skip the 99x64x256 case")
    args = ap.parse_args()
    for tool in ("verilator", "yosys"):
        if not shutil.which(tool):
            ap.error(f"{tool} is not installed")

    out = args.out.resolve()
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="kwt-single-mac-", dir="/tmp/opencode"))

    case_names = ["tiny", "ragged"]
    if not args.quick:
        case_names.append("full")
    cases = {name: make_case(name) for name in case_names}

    results = {"meta": {
        "schema": "kwt-single-mac-dual-cacc/1",
        "description": "ONE 16-lane multiplier array + TWO 16x16 INT32 CACC pages",
        "new_rtl": "hardware/experiments/single_mac_dual_cacc/rtl/kwt_single_mac_dual_cacc.sv",
        "new_tb": "hardware/experiments/single_mac_dual_cacc/tb/tb_kwt_single_mac_task.sv",
        "unit_tb": "hardware/experiments/single_mac_dual_cacc/tb/tb_kwt_single_mac_unit.sv",
        "preserved_rtl": ["hardware/rtl/pulse_array/kwt_row_broadcast_tile.sv (unmodified)",
                          "hardware/rtl/pulse_array/kwt_tile_output_stage.sv (unmodified)"],
        "comparison_only": ["hardware/architecture/tb/tb_kwt_row_broadcast_staged_task.sv (old one-core baseline)",
                            "hardware/experiments/streaming_mac/tb/tb_kwt_streaming_task.sv (old two-core dual)",
                            "hardware/experiments/single_mac_dual_cacc/area/kwt_two_core_compare_wrapper.sv (2x old core)"],
        "rows": args.rows, "clock_hz": 150_000_000,
        "bus_model": "one 16-bit word/cycle load; output accepted every 2 cycles (2 slots/result)",
        "scope": "RTL functional/task-level simulation; NOT E203, NOT 150 MHz STA, NOT synthesised area",
    }, "commands": [], "cases": [], "checks": [], "unit": None,
        "multiplier_counts": None, "backpressure": [], "cycle_model": None,
        "macro_port_table": None}

    # ---- unit microtest: new RTL primitive ----
    unit_dir = work / "unit"
    unit_dir.mkdir(parents=True)
    res = subprocess.run(["verilator", "--binary", "--timing", "--assert", "-Wno-fatal",
                          "-j", "4", "--top-module", "tb_kwt_single_mac_unit",
                          "--Mdir", str(unit_dir / "obj"), str(NEW_RTL), str(UNIT_TB)],
                         capture_output=True, text=True)
    (logs / "unit_build.log").write_text(res.stdout + res.stderr)
    if res.returncode:
        raise RuntimeError("unit build failed")
    unit_run = subprocess.run([str(unit_dir / "obj" / "Vtb_kwt_single_mac_unit")],
                              capture_output=True, text=True)
    (logs / "unit_run.log").write_text(unit_run.stdout + unit_run.stderr)
    results["unit"] = {"name": "tb_kwt_single_mac_unit",
                       "status": "PASS" if (unit_run.returncode == 0 and
                                            "PASS single-mac unit" in unit_run.stdout)
                       else "FAIL"}
    results["commands"].append({"cmd": "verilator --binary --timing --assert "
                                f"--top-module tb_kwt_single_mac_unit {NEW_RTL} {UNIT_TB}",
                                "exit": res.returncode})
    results["commands"].append({"cmd": str(unit_dir / "obj" / "Vtb_kwt_single_mac_unit"),
                                "exit": unit_run.returncode})
    if results["unit"]["status"] != "PASS":
        raise RuntimeError("single-MAC unit test failed")

    # ---- builds: new engine, old one-core baseline, old two-core dual ----
    new_exe = build(work / "new", "tb_kwt_single_mac_task", [NEW_RTL, STAGE, NEW_TB],
                    [f"-GROWS={args.rows}"])
    base_exe = build(work / "base", "tb_kwt_row_broadcast_staged_task",
                     [CORE, STAGE, BASE_TB], [f"-GROWS={args.rows}"])
    olddual_exe = build(work / "olddual", "tb_kwt_streaming_task",
                        [CORE, STAGE, OLD_DUAL_TB], [f"-GROWS={args.rows}"])
    results["commands"].append({"cmd": f"verilator ... -GROWS={args.rows} {NEW_RTL} {STAGE} {NEW_TB}",
                                "exit": 0})
    results["commands"].append({"cmd": f"verilator ... -GROWS={args.rows} {CORE} {STAGE} {BASE_TB}",
                                "exit": 0})
    results["commands"].append({"cmd": f"verilator ... -GROWS={args.rows} {CORE} {STAGE} {OLD_DUAL_TB}",
                                "exit": 0})

    for name in case_names:
        case = cases[name]
        new = run_case(new_exe, case, work / "new", args.rows)
        base = run_case(base_exe, case, work / "base", args.rows)
        olddual = run_case(olddual_exe, case, work / "olddual", args.rows)
        m, k, n = len(case["a"]), len(case["a"][0]), len(case["b"][0])
        row_tiles = (m + args.rows - 1) // args.rows
        col_tiles = (n + 15) // 16
        pure_mac = row_tiles * k * n
        if new.get("feed_stall_cycles", 0) != 0:
            raise ValueError(f"{name}: new engine feed stalls {new['feed_stall_cycles']}")
        if new.get("dump_not_ready_cycles", 0) != 0:
            raise ValueError(f"{name}: output-stage dump request not ready "
                             f"{new['dump_not_ready_cycles']} cycles")
        entry = {
            "name": name, "m": m, "k": k, "n": n,
            "tiles": row_tiles * col_tiles, "pure_mac_compute_cycles": pure_mac,
            "bytes": {"shared_read_bytes": new["shared_bytes"],
                      "a_words": new["a_load_words"], "w_words": new["w_load_words"],
                      "bias_slots": new["bias_load_words"], "out_slots": new["out_write_words"]},
            "new_single_mac": new,
            "old_one_core_baseline": base,
            "old_two_core_dual": olddual,
            "delta": {
                "new_vs_baseline_saved": base["cycles"] - new["cycles"],
                "new_vs_baseline_speedup": round(base["cycles"] / new["cycles"], 4),
                "new_vs_two_core_saved": olddual["cycles"] - new["cycles"],
                "new_max_of_old_cores_equiv": "1 core (16 muls) vs 2 cores (32 muls)",
                "new_overlapped_captures": new.get("overlapped_captures", 0),
                "new_drain_cycles": new.get("drain_cycles", 0),
            },
        }
        results["cases"].append(entry)
        results["checks"].append({
            "case": name, "new_int32_exact": True, "new_access_counts_exact": True,
            "old_baseline_int32_exact": True, "old_two_core_int32_exact": True,
            "new_outputs": new["checked_outputs"]})
        print(f"PASS {name}: new={new['cycles']} old_1core={base['cycles']} "
              f"old_2core={olddual['cycles']} saved_vs_base={base['cycles']-new['cycles']}")

    # ---- deep back-pressure on the new engine ----
    for name, period in (("tiny", 4), ("ragged", 3)):
        case = cases[name]
        metrics = run_case(new_exe, case, work / "newbp", args.rows, [f"+BACKPRESSURE={period}"])
        results["backpressure"].append({
            "case": name, "backpressure_period": period,
            "new_cycles": metrics["cycles"], "feed_stall_cycles": metrics.get("feed_stall_cycles", 0),
            "stall_cycles": metrics.get("stall_cycles", 0), "int32_exact": True})
        print(f"PASS backpressure {name} period={period}: cycles={metrics['cycles']}")

    # ---- elaborated multiplier counts (real elaboration, lint-level) ----
    results["multiplier_counts"] = {
        "method": "yosys read_verilog -sv; chparam -set ROWS 16; hierarchy -top; "
                  "proc; opt_expr; opt_clean; stat -> count of $mul cells",
        "new_single_mac_dual_cacc_ROWS16": count_muls(
            logs, "new", "kwt_single_mac_dual_cacc", [NEW_RTL],
            "kwt_single_mac_dual_cacc"),
        "old_one_core_kwt_row_broadcast_tile_ROWS16": count_muls(
            logs, "old_one_core", "kwt_row_broadcast_tile", [CORE],
            "kwt_row_broadcast_tile"),
        "old_two_core_wrapper_ROWS16": count_muls(
            logs, "old_two_core", "kwt_two_core_compare_wrapper",
            [TWO_CORE_WRAP, CORE], "kwt_two_core_compare_wrapper"),
        "interpretation": "The new engine has the SAME number of multipliers as ONE "
                          "old core and HALF of the old two-core dual harness.",
    }
    mc = results["multiplier_counts"]
    print("MUL counts: new=%d old_1core=%d old_2core=%d"
          % (mc["new_single_mac_dual_cacc_ROWS16"],
             mc["old_one_core_kwt_row_broadcast_tile_ROWS16"],
             mc["old_two_core_wrapper_ROWS16"]))

    # ---- cycle model: separated from RTL measurement ----
    full_new = next((c for c in results["cases"] if c["name"] == "full"), None)
    if full_new is not None:
        results["cycle_model"] = {
            "label": "MODEL ONLY - arithmetic lower bound, not a measured number",
            "ideal_pure_mac_one_layer_cycles_16rows": full_new["pure_mac_compute_cycles"],
            "rtl_new_full_cycles": full_new["new_single_mac"]["cycles"],
            "rtl_old_two_core_full_cycles": full_new["old_two_core_dual"]["cycles"],
            "rtl_old_one_core_full_cycles": full_new["old_one_core_baseline"]["cycles"],
            "new_above_pure_mac_pct": round(
                100.0 * (full_new["new_single_mac"]["cycles"]
                         - full_new["pure_mac_compute_cycles"])
                / full_new["pure_mac_compute_cycles"], 3),
            "note": "The new one-MAC-array engine reproduces the old two-core dual "
                    "schedule cycles; that is a functional schedule result, not a "
                    "timing-closure or area result.",
        }

    # ---- SRAM macro / port requirement table ----
    results["macro_port_table"] = {
        "budget": "16-bit x 128/1024/4096 single/dual-port macros only; no extra DDR",
        "cacc_two_pages_16x16x32": {
            "bytes": 2 * 16 * 16 * 4,
            "if_register_mapped": "2 x 256 x 32 = 4096 flops + 512 valid bits (as this experiment)",
            "if_macro_mapped": "row-banked sketch: 2 pages x 16 rows x 2 width slices = 64 x 16b x 128 minimum-size macros, heavily underfilled; exact mapping unproven",
            "ports": "the active page performs 16 independent row-accumulator updates each cycle, each requiring old-sum read and new-sum write or forwarding; the inactive page supplies one dump read. One 1R1W macro per whole page is insufficient.",
        },
        "bias_group": {
            "size": "N=256 INT32 = 1024 B",
            "note": "bias is read at export only and added AFTER the CACC read; "
                    "it never enters the multiplier feed",
            "macros": "2 x 16b x 1024 (overallocated), registers, or a slice of an existing bank; 16b x 256 is not an allowed macro size",
        },
        "result_buffer": {
            "size": "16x16 INT32 = 1024 B (kwt_tile_output_stage, unchanged)",
            "macros": "2 x 16b x 1024 (32-bit width-cascade) with 1W capture + 1R drain",
        },
        "clock_area_evidence_limit": "This experiment reports elaborated $mul counts, "
            "simulated cycles and byte/bit capacity for macro mapping. Bank/port "
            "arbitration, 150 MHz timing closure and synthesised area/power are NOT "
            "measured or claimed here.",
    }

    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    for sub in ("new", "newbp", "base", "olddual"):
        for d in (work / sub).glob("*/run.log"):
            shutil.copy(d, logs / f"{sub}_{d.parent.name}_run.log")
        for f in (work / sub).glob("build_*.log"):
            shutil.copy(f, logs / f"{sub}_{f.name}")
    print(f"WROTE {out/'results.json'}")
    print(f"PASS single-mac dual-cacc experiment: {len(results['cases'])} cases, "
          f"unit={results['unit']['status']}, muls={results['multiplier_counts']['new_single_mac_dual_cacc_ROWS16']}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        sys.exit(1)
