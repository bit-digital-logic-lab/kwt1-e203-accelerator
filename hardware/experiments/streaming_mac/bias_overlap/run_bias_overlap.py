#!/usr/bin/env python3
"""Build/run the bias-overlap experiment.

What is measured (RTL, Verilator):
  * accepted baseline single-CACC task (unchanged TB + unchanged core/stage)
  * accepted dual-CACC task (unchanged TB + unchanged core/stage)
  * new one-descriptor bias-overlap task (new TB + new arbiter/bias-bank;
    unchanged core/stage)
  * new bias/arbiter unit microtest (normal, delayed, back-pressure, reset)

What is modelled (Python, clearly separated):
  * hardware/experiments/streaming_mac/bias_overlap/model/bus_event_model.py
    reproduces the accepted full-FFN1 RTL 154240 / 125603 and predicts the
    bias-overlap schedule cycle by cycle over one shared 16-bit bus.

All outputs are checked against an independent numpy INT32 reference.
The script writes results.json and logs/ next to itself.
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

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
PA = ROOT / "hardware" / "rtl" / "pulse_array"
CORE = PA / "kwt_row_broadcast_tile.sv"
STAGE = PA / "kwt_tile_output_stage.sv"
BASE_TB = ROOT / "hardware" / "architecture" / "tb" / "tb_kwt_row_broadcast_staged_task.sv"
DUAL_TB = ROOT / "hardware" / "experiments" / "streaming_mac" / "tb" / "tb_kwt_streaming_task.sv"
BO_TB = HERE / "tb" / "tb_kwt_bias_overlap_task.sv"
UNIT_TB = HERE / "tb" / "tb_kwt_bias_arbiter_unit.sv"
ARB = HERE / "rtl" / "kwt_shared_bus_arbiter.sv"
BIAS = HERE / "rtl" / "kwt_bias_bank.sv"

MAX_M, MAX_K, MAX_N = 99, 64, 256
ACCEPTED_BASE_FULL = 154240
ACCEPTED_DUAL_FULL = 125603


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
    ref = (np.asarray(a, dtype=np.int64) @ np.asarray(w, dtype=np.int64)
           + np.asarray(bias, dtype=np.int64)) & 0xFFFFFFFF
    case["expected"] = ref.tolist()
    return m, k, n


def write_hex(path, values, count, width):
    if len(values) > count:
        raise ValueError(f"{path.name} exceeds memory size")
    mask = (1 << width) - 1
    path.write_text("".join(
        f"{int(v) & mask:0{width // 4}x}\n"
        for v in list(values) + [0] * (count - len(values))))


def build(workdir, top, rtl_files, extra_args=(), logs=None):
    workdir.mkdir(parents=True, exist_ok=True)
    cmd = ["verilator", "--binary", "--timing", "--assert", "-Wno-fatal",
           "-j", "4", "--top-module", top, "--Mdir", str(workdir / "obj")]
    cmd += list(extra_args) + [str(f) for f in rtl_files]
    log = workdir / f"build_{top}.log"
    with log.open("w") as fh:
        res = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    if logs is not None:
        shutil.copy(log, logs / f"build_{top}.log")
    if res.returncode:
        raise RuntimeError(f"Verilator build failed for {top}: see {log}")
    return workdir / "obj" / f"V{top}", cmd, res.returncode


def run_case(exe, case, workdir, rows, extra=(), tag=None, padded_bias=False):
    m, k, n = check_case(case)
    d = workdir / (tag or case["name"])
    d.mkdir(parents=True, exist_ok=True)
    write_hex(d / "a.hex", [v for row in case["a"] for v in row], MAX_M * MAX_K, 16)
    write_hex(d / "w.hex", [v for row in case["b"] for v in row], MAX_K * MAX_N, 16)
    write_hex(d / "bias.hex", case["bias"], MAX_N, 32)
    write_hex(d / "expected.hex", [v for row in case["expected"] for v in row],
              MAX_M * MAX_N, 32)
    cmd = [str(exe), f"+A={d/'a.hex'}", f"+W={d/'w.hex'}",
           f"+BIAS={d/'bias.hex'}", f"+EXPECTED={d/'expected.hex'}",
           f"+OUTPUT={d/'actual.txt'}", f"+M={m}", f"+K={k}", f"+N={n}",
           f"+GAPS={int(case['gaps'])}"] + list(extra)
    res = subprocess.run(cmd, text=True, capture_output=True)
    (d / "run.log").write_text(res.stdout + res.stderr)
    if res.returncode or "PASS " not in res.stdout:
        raise RuntimeError(f"{case['name']} run failed; see {d/'run.log'}\n"
                           + res.stdout[-2000:] + res.stderr[-2000:])
    metric_line = next(l for l in res.stdout.splitlines()
                       if l.startswith("ARCH_METRICS "))
    metrics = {kk: int(v) for kk, v in re.findall(r"(\w+)=(\d+)", metric_line)}
    # The new bias bank fills a whole 16-INT32 column group (32 slots), so a
    # partial column tile reads padded bias values. The accepted TBs read
    # exactly n values (2*n slots).
    col_tiles = (n + 15) // 16
    bias_slots = 2 * col_tiles * 16 if padded_bias else 2 * n
    expect = {"a_load_words": m * k, "w_load_words": k * n,
              "bias_load_words": bias_slots, "out_write_words": 2 * m * n,
              "a_local_reads": m * k * ((n + 15) // 16),
              "w_local_reads": ((m + rows - 1) // rows) * ((n + 15) // 16) * k * 16,
              "checked_outputs": m * n}
    for key, val in expect.items():
        if metrics.get(key) != val:
            raise ValueError(f"{case['name']} {key}={metrics.get(key)} != {val}")
    if "out_slots" in metrics and metrics["out_slots"] != 2 * m * n:
        raise ValueError(f"{case['name']} out_slots={metrics['out_slots']}")
    if "bus_grant_out" in metrics:
        if metrics["bus_grant_out"] != 2 * m * n:
            raise ValueError(f"{case['name']} bus_grant_out mismatch")
        w_bus = metrics.get("w_fetch_words", metrics["w_load_words"])
        bus_total = (metrics["bus_grant_a"] + metrics["bus_grant_w"] +
                     metrics["bus_grant_bias"] + metrics["bus_grant_out"])
        data_total = (metrics["a_load_words"] + w_bus +
                      metrics["bias_load_words"] + metrics["out_write_words"])
        if bus_total != data_total:
            raise ValueError(f"{case['name']} shared-bus grant accounting mismatch "
                             f"{bus_total} != {data_total}")
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
    w_bus = metrics.get("w_fetch_words", metrics["w_load_words"])
    metrics["shared_bytes"] = 2 * (metrics["a_load_words"] + w_bus
                                   + metrics["bias_load_words"] + metrics["out_write_words"])
    metrics["valid_outputs"] = metrics["checked_outputs"]
    if tag:
        shutil.copy(d / "run.log", HERE / "logs" / f"{tag}_run.log")
    return metrics, cmd, res.returncode


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
    work = Path(tempfile.mkdtemp(prefix="kwt-bias-overlap-", dir="/tmp/opencode"))
    cases = {name: make_case(name) for name in
             ("tiny", "ragged", "twotile", "overflow", "full")}
    case_names = ["tiny", "ragged", "twotile", "overflow"]
    if not args.quick:
        case_names.append("full")

    results = {
        "meta": {
            "schema": "kwt-bias-overlap/1",
            "description": "one-descriptor A/W/bias concurrent fill with bias at CACC export",
            "core_rtl": "hardware/rtl/pulse_array/kwt_row_broadcast_tile.sv (unchanged)",
            "stage_rtl": "hardware/rtl/pulse_array/kwt_tile_output_stage.sv (unchanged)",
            "baseline_tb": "hardware/architecture/tb/tb_kwt_row_broadcast_staged_task.sv (unchanged)",
            "accepted_dual_tb": "hardware/experiments/streaming_mac/tb/tb_kwt_streaming_task.sv (unchanged)",
            "bias_overlap_tb": "hardware/experiments/streaming_mac/bias_overlap/tb/tb_kwt_bias_overlap_task.sv",
            "new_rtl": ["bias_overlap/rtl/kwt_shared_bus_arbiter.sv",
                         "bias_overlap/rtl/kwt_bias_bank.sv"],
            "rows": args.rows, "clock_hz": 150_000_000,
            "bus_model": "one 16-bit slot/cycle; result needs 2 slots; "
                         "OUT>BIAS>A>W fixed priority; no simultaneous bus transfers",
            "scope": "RTL functional/task-level simulation; NOT E203, NOT 150 MHz STA",
        },
        "commands": [], "cases": [], "checks": [],
        "comparison": {}, "model": None, "unit": None,
    }

    # Unit microtest.
    unit_exe, unit_cmd, unit_build_exit = build(
        work / "unit", "tb_kwt_bias_arbiter_unit", [ARB, BIAS, UNIT_TB], logs=logs)
    results["commands"].append({"unit_build": " ".join(map(str, unit_cmd)),
                                "exit": unit_build_exit})
    unit_run = subprocess.run([str(unit_exe)], capture_output=True, text=True)
    (logs / "unit_run.log").write_text(unit_run.stdout + unit_run.stderr)
    results["commands"].append({"unit_run": str(unit_exe), "exit": unit_run.returncode})
    results["unit"] = {"status": "PASS" if (unit_run.returncode == 0 and
                                            "PASS bias/arbiter unit" in unit_run.stdout)
                       else "FAIL"}
    if results["unit"]["status"] != "PASS":
        raise RuntimeError("bias/arbiter unit test failed")

    # Build accepted baseline, accepted dual, and new bias-overlap task.
    base_exe, base_cmd, base_exit = build(
        work / "base", "tb_kwt_row_broadcast_staged_task",
        [CORE, STAGE, BASE_TB], [f"-GROWS={args.rows}"], logs=logs)
    dual_exe, dual_cmd, dual_exit = build(
        work / "dual", "tb_kwt_streaming_task",
        [CORE, STAGE, DUAL_TB], [f"-GROWS={args.rows}"], logs=logs)
    bo_exe, bo_cmd, bo_exit = build(
        work / "bo", "tb_kwt_bias_overlap_task",
        [CORE, STAGE, ARB, BIAS, BO_TB], [f"-GROWS={args.rows}"], logs=logs)
    for name, cmd, code in (("baseline_build", base_cmd, base_exit),
                            ("accepted_dual_build", dual_cmd, dual_exit),
                            ("bias_overlap_build", bo_cmd, bo_exit)):
        results["commands"].append({name: " ".join(map(str, cmd)), "exit": code})

    # Accepted baseline/dual on all cases (reproduction check).
    for name in case_names:
        case = cases[name]
        base, bcmd, bexit = run_case(base_exe, case, work / "base", args.rows,
                                     tag=f"base_{name}")
        dual, dcmd, dexit = run_case(dual_exe, case, work / "dual", args.rows,
                                     tag=f"dual_{name}")
        results["cases"].append({"name": name, "m": len(case["a"]),
                                 "k": len(case["a"][0]), "n": len(case["b"][0]),
                                 "baseline": base, "dual": dual})
        results["commands"].append({"baseline_run": " ".join(map(str, bcmd)), "exit": bexit})
        results["commands"].append({"accepted_dual_run": " ".join(map(str, dcmd)), "exit": dexit})
        results["checks"].append({"case": name, "baseline_int32_exact": True,
                                  "dual_int32_exact": True})

    # New bias-overlap task, normal cases.
    bo_cases = []
    for name in case_names:
        case = cases[name]
        m = len(case["a"])
        rt = (m + args.rows - 1) // args.rows
        pre = min(3, rt)
        extra = [f"+A_PRELOAD_TILES={pre}", "+BIAS_DELAY=0", "+OUT_BP=1", "+BUS_GAP=0"]
        metrics, cmd, code = run_case(bo_exe, case, work / "bo", args.rows,
                                      extra=extra, tag=f"bo_{name}",
                                      padded_bias=True)
        bo_cases.append({"name": name, "m": m, "k": len(case["a"][0]),
                         "n": len(case["b"][0]), "a_preload_tiles": pre,
                         "metrics": metrics})
        results["commands"].append({"bias_overlap_run": " ".join(map(str, cmd)),
                                    "exit": code})
        results["checks"].append({"case": name, "bias_overlap_int32_exact": True,
                                  "bias_overlap_bus_slots_exact": True})
    results["bias_overlap_cases"] = bo_cases

    # Full-shape robustness: preload sweep, delayed bias, back-pressure, bus gaps.
    if "full" in cases:
        full = cases["full"]
        sweep = []
        for pre in range(1, 8):
            extra = [f"+A_PRELOAD_TILES={pre}", "+BIAS_DELAY=0", "+OUT_BP=1", "+BUS_GAP=0"]
            metrics, cmd, code = run_case(bo_exe, full, work / "bo", args.rows,
                                          extra=extra, tag=f"bo_full_pre{pre}",
                                          padded_bias=True)
            sweep.append({"a_preload_tiles": pre, **metrics})
            results["commands"].append({"bias_overlap_preload_run":
                                        " ".join(map(str, cmd)), "exit": code})
        results["preload_sweep"] = sweep

        robustness = []
        for label, extra in (
            ("delayed_bias_6000", ["+A_PRELOAD_TILES=3", "+BIAS_DELAY=6000",
                                   "+OUT_BP=1", "+BUS_GAP=0"]),
            ("delayed_bias_20000", ["+A_PRELOAD_TILES=3", "+BIAS_DELAY=20000",
                                    "+OUT_BP=1", "+BUS_GAP=0"]),
            ("out_backpressure_3", ["+A_PRELOAD_TILES=3", "+BIAS_DELAY=0",
                                    "+OUT_BP=3", "+BUS_GAP=0"]),
            ("out_backpressure_6", ["+A_PRELOAD_TILES=3", "+BIAS_DELAY=0",
                                    "+OUT_BP=6", "+BUS_GAP=0"]),
            ("bus_gap_7", ["+A_PRELOAD_TILES=3", "+BIAS_DELAY=0",
                           "+OUT_BP=1", "+BUS_GAP=7"]),
            ("delay_bp_gap", ["+A_PRELOAD_TILES=3", "+BIAS_DELAY=20000",
                              "+OUT_BP=3", "+BUS_GAP=11"]),
        ):
            metrics, cmd, code = run_case(bo_exe, full, work / "bo", args.rows,
                                          extra=extra, tag=f"bo_{label}",
                                          padded_bias=True)
            robustness.append({"case": label, **metrics})
            results["commands"].append({"bias_overlap_robustness_run":
                                        " ".join(map(str, cmd)), "exit": code})
        results["robustness"] = robustness

    # Comparison: actual RTL vs accepted constants.
    by_name = {c["name"]: c for c in results["cases"]}
    base_full = by_name.get("full", {}).get("baseline", {}).get("cycles")
    dual_full = by_name.get("full", {}).get("dual", {}).get("cycles")
    bo_full = None
    if "full" in cases:
        bo_full = next(c["metrics"]["cycles"] for c in bo_cases if c["name"] == "full")
    results["comparison"] = {
        "baseline_full": base_full,
        "accepted_dual_full": dual_full,
        "bias_overlap_full": bo_full,
        "accepted_committed": {"baseline_full": ACCEPTED_BASE_FULL,
                               "dual_full": ACCEPTED_DUAL_FULL},
        "baseline_matches_committed": base_full == ACCEPTED_BASE_FULL,
        "accepted_dual_matches_committed": dual_full == ACCEPTED_DUAL_FULL,
        "bias_overlap_vs_accepted_dual_cycles":
            (dual_full - bo_full) if (dual_full and bo_full) else None,
        "bias_overlap_vs_accepted_dual_pct":
            round(100.0 * (dual_full - bo_full) / dual_full, 3)
            if (dual_full and bo_full) else None,
        "bias_overlap_vs_baseline_cycles":
            (base_full - bo_full) if (base_full and bo_full) else None,
        "bias_overlap_vs_baseline_pct":
            round(100.0 * (base_full - bo_full) / base_full, 3)
            if (base_full and bo_full) else None,
    }
    if "full" in cases:
        rob = {r["case"]: r for r in results["robustness"]}
        results["comparison"]["bias_overlap_delayed"] = rob["delayed_bias_6000"]["cycles"]
        results["comparison"]["bias_overlap_delayed_20k"] = rob["delayed_bias_20000"]["cycles"]
        results["comparison"]["bias_overlap_out_bp"] = rob["out_backpressure_3"]["cycles"]

    # Macro/area/port accounting (capacity only; no macro mapping or STA).
    results["area_port_table"] = {
        "budget": "16-bit x 128/1024/4096 macros; total accelerator+SRAM <= 2.25 mm^2",
        "bias_bank_256x32": {
            "bytes": 1024,
            "sram_macros_if_macro_mapped": "2 x 16b x 1024 (or 512 x 16-bit registers)",
            "ports": "1 x 16-bit fill write + 1 x 32-bit export read",
            "note": "dedicated bank; reuses the bias storage the accepted TB already had",
        },
        "arbiter": {
            "size": "4 requesters, combinational fixed priority, no buffering",
            "note": "one 16-bit slot/cycle; no extra data macro",
        },
        "accepted_buffers_unchanged": {
            "a_resident_99x64x16": "12672 B, 2 x 16b x 4096, 1R compute + 1W fill",
            "w_pingpong_2x64x16x16": "4096 B, 2 x 16b x 1024",
            "cacc_dual_2x16x16x32": "2048 B (4 x 16b x 1024 if macro-mapped)",
            "output_stage_256x32": "1024 B, 2 x 16b x 1024",
        },
        "unresolved": "150 MHz closure, exact macro bank/port arbitration and 2.25 mm^2 fit are NOT proven",
    }

    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    # Run the cycle model, then attach its output to results.json.
    model_script = HERE / "model" / "bus_event_model.py"
    model_out = HERE / "model" / "model_results.json"
    mres = subprocess.run([sys.executable, str(model_script),
                           "--out", str(model_out),
                           "--rtl-results", str(out / "results.json")],
                          capture_output=True, text=True, cwd=str(ROOT))
    (logs / "model_run.log").write_text(mres.stdout + mres.stderr)
    results["commands"].append({"model_run": f"{model_script} --out {model_out} "
                                f"--rtl-results {out/'results.json'}",
                                "exit": mres.returncode})
    if mres.returncode != 0:
        raise RuntimeError("cycle model failed; see logs/model_run.log")
    model_data = json.loads(model_out.read_text())
    results["model"] = model_data
    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    # Copy run logs from the temp work tree.
    for sub in ("base", "dual", "bo"):
        for d in (work / sub).glob("*/run.log"):
            shutil.copy(d, logs / f"{sub}_{d.parent.name}_run.log")
        for f in (work / sub).glob("build_*.log"):
            shutil.copy(f, logs / f"{sub}_{f.name}")

    print(f"baseline full RTL      = {base_full}")
    print(f"accepted dual full RTL = {dual_full}")
    print(f"bias-overlap full RTL  = {bo_full}")
    print(f"WROTE {out/'results.json'}")
    print(f"PASS bias-overlap experiment: {len(case_names)} normal cases + "
          f"{len(results.get('robustness', []))} robustness cases, "
          f"unit={results['unit']['status']}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        sys.exit(1)
