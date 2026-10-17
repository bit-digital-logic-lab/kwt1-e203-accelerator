#!/usr/bin/env python3
"""Compile the unchanged pulse array and compare its outputs with vector goldens."""
import argparse
import json
import math
import pathlib
import shutil
import struct
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
PLUS = ROOT / "testbenchplus"
FORMATS = {"int8": (8, 32, 0), "int16": (16, 32, 0),
           "int32": (32, 64, 0), "fp32": (32, 32, 1)}


def bits(value, name):
    if name == "fp32":
        return int.from_bytes(struct.pack("<f", float(value)), "little")
    width = FORMATS[name][0]
    value = int(value)
    if not -(1 << (width - 1)) <= value < (1 << (width - 1)):
        raise ValueError(f"{name} input out of range: {value}")
    return value & ((1 << width) - 1)


def decode(value, name):
    if name == "fp32":
        return struct.unpack("<f", value.to_bytes(4, "little"))[0]
    width = FORMATS[name][1]
    return value - (1 << width) if value & (1 << (width - 1)) else value


def write_operands(case, directory):
    name = case["format"]
    a, b = case["a"], case["b"]
    m, k = len(a), len(a[0])
    n = len(b[0])
    if not 1 <= m <= 8 or not 1 <= n <= 16 or not 1 <= k <= 64:
        raise ValueError("supported shape: 1<=M<=8, 1<=N<=16, 1<=K<=64")
    if any(len(row) != k for row in a) or len(b) != k or any(len(row) != n for row in b):
        raise ValueError("ragged A or B matrix")
    if len(case["expected"]) != m or any(len(row) != n for row in case["expected"]):
        raise ValueError("expected must have shape M x N")
    for filename, count, matrix in (("a.hex", 8, a), ("b.hex", 16, b)):
        words = []
        for kk in range(k):
            for lane in range(count):
                value = matrix[lane][kk] if filename == "a.hex" and lane < m else (
                    matrix[kk][lane] if filename == "b.hex" and lane < n else 0)
                words.append(f"{bits(value, name):x}")
        (directory / filename).write_text("\n".join(words) + "\n")
    return m, n, k


def compile_format(name, directory):
    dw, aw, fmt = FORMATS[name]
    image = directory / name
    cmd = ["iverilog", "-g2012", "-s", "tb_array_vector",
           f"-Ptb_array_vector.DATA_W={dw}", f"-Ptb_array_vector.ACC_W={aw}",
           f"-Ptb_array_vector.FORMAT={fmt}", "-o", str(image),
           str(ROOT / "rtl/pulse_array/kwt_pe.sv"),
           str(ROOT / "rtl/pulse_array/kwt_pulse_array.sv"),
           str(PLUS / "common/tb_array_vector.sv")]
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return image


def compare(case, output):
    name = case["format"]
    expected = [x for row in case["expected"] for x in row]
    raw = [int(line, 16) for line in output.read_text().splitlines()]
    if len(raw) != len(expected):
        raise ValueError(f"output count {len(raw)} != expected {len(expected)}")
    got = [decode(x, name) for x in raw]
    if name == "fp32":
        if any(not math.isfinite(x) for x in got):
            raise ValueError("FP32 output contains NaN or infinity")
        errors = [float(x) - float(y) for x, y in zip(got, expected)]
        denominator = sum(float(y)**2 for y in expected)
        nrmse = math.sqrt(sum(e*e for e in errors) / denominator) if denominator else (
            0.0 if not any(errors) else math.inf)
        maximum = max(abs(e) for e in errors)
        limit = float(case.get("max_nrmse", 0.03))
        print(f"  FP32 NRMSE={nrmse:.8%} max_abs={maximum:.8g} limit={limit:.2%}")
        if nrmse > limit:
            raise ValueError("NRMSE exceeds limit")
    else:
        mismatch = [(i, x, y) for i, (x, y) in enumerate(zip(got, expected)) if x != int(y)]
        print(f"  integer exact={len(expected)-len(mismatch)}/{len(expected)} "
              f"accuracy={(len(expected)-len(mismatch))/len(expected):.2%}")
        if mismatch:
            raise ValueError(f"first mismatch flat_index/got/expected={mismatch[0]}")


def main():
    parser = argparse.ArgumentParser(description="Run testbenchplus vectors")
    parser.add_argument("cases", nargs="*", type=pathlib.Path,
                        help="case.json paths; default: vectors/*/case.json")
    parser.add_argument("--dump", action="store_true", help="write VCD under --out")
    parser.add_argument("--out", type=pathlib.Path, help="retain outputs and counters")
    args = parser.parse_args()
    cases = args.cases or sorted((PLUS / "vectors").glob("*/case.json"))
    if not cases:
        parser.error("no vector cases found")
    for tool in ("iverilog", "vvp"):
        if not shutil.which(tool):
            parser.error(f"{tool} not installed")
    with tempfile.TemporaryDirectory(prefix="testbenchplus-") as tmp:
        work = pathlib.Path(tmp)
        compiled = {}
        failures = 0
        for path in cases:
            try:
                case = json.loads(path.read_text())
                name = case["format"]
                if name not in FORMATS:
                    raise ValueError(f"unsupported format {name}")
                case_dir = work / path.parent.name
                case_dir.mkdir(exist_ok=True)
                m, n, k = write_operands(case, case_dir)
                if name not in compiled:
                    compiled[name] = compile_format(name, work)
                cmd = ["vvp", str(compiled[name]), f"+A={case_dir/'a.hex'}",
                       f"+B={case_dir/'b.hex'}", f"+OUT={case_dir/'result.hex'}",
                       f"+PERF={case_dir/'perf.txt'}", f"+ROWS={m}",
                       f"+COLS={n}", f"+K={k}"]
                if args.dump:
                    cmd.append(f"+DUMP={case_dir/'state.vcd'}")
                result = subprocess.run(cmd, text=True, capture_output=True)
                if result.returncode:
                    raise RuntimeError(result.stdout + result.stderr)
                perf = dict(line.split("=", 1) for line in
                            (case_dir / "perf.txt").read_text().splitlines())
                if perf["status"] != "2":
                    raise ValueError(f"unexpected status {perf['status']}")
                print(f"{path.parent.name}: {m}x{k} * {k}x{n} {name}")
                compare(case, case_dir / "result.hex")
                print("  " + " ".join(f"{key}={value}" for key, value in perf.items()))
                if args.out:
                    destination = args.out.resolve() / path.parent.name
                    destination.mkdir(parents=True, exist_ok=True)
                    for item in case_dir.iterdir():
                        shutil.copy2(item, destination / item.name)
            except (OSError, ValueError, KeyError, RuntimeError) as exc:
                failures += 1
                print(f"FAIL {path}: {exc}", file=sys.stderr)
        print(f"result: {len(cases)-failures}/{len(cases)} passed")
        return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
