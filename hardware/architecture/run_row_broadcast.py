#!/usr/bin/env python3
"""Build one Verilator DUT and check reusable FFN1 integer task vectors."""
import argparse
import json
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import tempfile

HARDWARE = Path(__file__).resolve().parents[1]
ARCH = HARDWARE / "architecture"
RTL = HARDWARE / "rtl" / "pulse_array" / "kwt_row_broadcast_tile.sv"
TB = ARCH / "tb" / "tb_kwt_row_broadcast_task.sv"
STAGE_RTL = HARDWARE / "rtl" / "pulse_array" / "kwt_tile_output_stage.sv"
STAGE_TB = ARCH / "tb" / "tb_kwt_row_broadcast_staged_task.sv"
MAX_M, MAX_K, MAX_N = 99, 64, 256


def wrap32(number):
    return number & 0xFFFFFFFF


def make_case(name):
    if name == "full":
        m, k, n = 99, 64, 256
        a = [[((row * 3 + kk * 5) % 15) - 7 for kk in range(k)]
             for row in range(m)]
        w = [[((kk * 7 + col * 2) % 13) - 6 for col in range(n)]
             for kk in range(k)]
        bias = [(col % 17) - 8 for col in range(n)]
        gaps = 0
    elif name == "overflow":
        m, k, n = 8, 64, 16
        a = [[-32768 if (row + kk) & 1 else 32767 for kk in range(k)]
             for row in range(m)]
        w = [[-32768 if (kk + col) & 1 else 32767 for col in range(n)]
             for kk in range(k)]
        bias = [0x7FFFFFFF if col & 1 else -0x80000000 for col in range(n)]
        gaps = 1
    else:
        m, k, n, seed, gaps = ((3, 4, 5, 17, 0) if name == "tiny" else
                               (10, 7, 19, 29, 1))
        rng = random.Random(seed)
        a = [[rng.randint(-100, 100) for _ in range(k)] for _ in range(m)]
        w = [[rng.randint(-100, 100) for _ in range(n)] for _ in range(k)]
        bias = [rng.randint(-1000, 1000) for _ in range(n)]
        a[0][0], w[0][0] = -32768, 32767
    expected = [[wrap32(bias[col] + sum(a[row][kk] * w[kk][col]
                                             for kk in range(k)))
                 for col in range(n)] for row in range(m)]
    return {"name": name, "format": "int16", "a": a, "b": w,
            "bias": bias, "expected": expected, "gaps": gaps}


def load_case(path):
    case = json.loads(path.read_text())
    case.setdefault("name", path.stem)
    case.setdefault("bias", [0] * len(case["b"][0]))
    case.setdefault("gaps", 0)
    return case


def check_case(case):
    if case.get("format") != "int16":
        raise ValueError("row-broadcast RTL currently supports int16 only")
    a, w, bias = case["a"], case["b"], case["bias"]
    m, k, n = len(a), len(a[0]), len(w[0])
    if not (1 <= m <= MAX_M and 1 <= k <= MAX_K and 1 <= n <= MAX_N):
        raise ValueError("shape exceeds 99x64 * 64x256")
    if any(len(row) != k for row in a) or len(w) != k or any(len(row) != n for row in w):
        raise ValueError("ragged operand matrix")
    if len(bias) != n:
        raise ValueError("bias length differs from N")
    if any(not -32768 <= int(v) <= 32767 for row in a+w for v in row):
        raise ValueError("int16 operand out of range")
    reference = [[wrap32(int(bias[col]) + sum(int(a[row][kk]) * int(w[kk][col])
                                                   for kk in range(k)))
                  for col in range(n)] for row in range(m)]
    if "expected" in case:
        if len(case["expected"]) != m or any(len(row) != n for row in case["expected"]):
            raise ValueError("expected shape differs from MxN")
        if any(wrap32(int(given)) != want for row_g, row_w in
               zip(case["expected"], reference) for given, want in zip(row_g, row_w)):
            raise ValueError("supplied expected matrix differs from independent INT32 MAC")
    case["expected"] = reference
    return m, k, n


def write_hex(path, values, count, width):
    if len(values) > count:
        raise ValueError(f"{path.name} exceeds memory size")
    mask = (1 << width) - 1
    words = values + [0] * (count - len(values))
    path.write_text("".join(f"{int(value) & mask:0{width//4}x}\n" for value in words))


def build(workdir, trace, rows=8, staged=False):
    top = "tb_kwt_row_broadcast_staged_task" if staged else "tb_kwt_row_broadcast_task"
    cmd = ["verilator", "--binary", "--timing", "--assert", "-Wno-fatal",
           "-j", "4", "--top-module", top,
           "--Mdir", str(workdir / "obj"), f"-GROWS={rows}"]
    if trace:
        cmd.append("--trace")
    cmd += [str(RTL), str(STAGE_RTL), str(STAGE_TB)] if staged else [str(RTL), str(TB)]
    with (workdir / "build.log").open("w") as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Verilator build failed; see {workdir/'build.log'}")
    return workdir / "obj" / f"V{top}"


def run_case(executable, case, workdir, trace, rows=8, staged=False):
    m, k, n = check_case(case)
    case_dir = workdir / case["name"]
    case_dir.mkdir(parents=True, exist_ok=True)
    write_hex(case_dir / "a.hex", [v for row in case["a"] for v in row],
              MAX_M*MAX_K, 16)
    write_hex(case_dir / "w.hex", [v for row in case["b"] for v in row],
              MAX_K*MAX_N, 16)
    write_hex(case_dir / "bias.hex", case["bias"], MAX_N, 32)
    write_hex(case_dir / "expected.hex", [v for row in case["expected"] for v in row],
              MAX_M*MAX_N, 32)
    cmd = [str(executable), f"+A={case_dir/'a.hex'}", f"+W={case_dir/'w.hex'}",
           f"+BIAS={case_dir/'bias.hex'}", f"+EXPECTED={case_dir/'expected.hex'}",
           f"+OUTPUT={case_dir/'actual.txt'}", f"+M={m}", f"+K={k}",
           f"+N={n}", f"+GAPS={int(case['gaps'])}"]
    if trace:
        cmd.append(f"+VCD={case_dir/'trace.vcd'}")
    result = subprocess.run(cmd, text=True, capture_output=True)
    (case_dir / "run.log").write_text(result.stdout + result.stderr)
    pass_marker = "PASS row-broadcast staged task" if staged else "PASS row-broadcast task"
    if result.returncode or pass_marker not in result.stdout:
        raise RuntimeError(f"{case['name']} failed; see {case_dir/'run.log'}\n"
                           + result.stdout[-1200:] + result.stderr[-1200:])
    metric_line = next(line for line in result.stdout.splitlines()
                       if line.startswith("ARCH_METRICS "))
    metrics = {key: int(value) for key, value in re.findall(r"(\w+)=(\d+)", metric_line)}
    expected_counts = {"a_load_words": m*k, "w_load_words": k*n,
                       "bias_load_words": 2*n, "out_write_words": 2*m*n,
                       "a_local_reads": m*k*((n+15)//16),
                       "w_local_reads": ((m+rows-1)//rows)*((n+15)//16)*k*16,
                       "checked_outputs": m*n}
    for key, value in expected_counts.items():
        if metrics.get(key) != value:
            raise ValueError(f"{case['name']} {key}={metrics.get(key)} != {value}")
    seen = set()
    for line in (case_dir / "actual.txt").read_text().splitlines():
        row_text, col_text, value_text = line.split()
        row, col = int(row_text), int(col_text)
        if not (0 <= row < m and 0 <= col < n) or (row, col) in seen:
            raise ValueError(f"{case['name']} duplicate or invalid output ({row},{col})")
        seen.add((row, col))
        if int(value_text, 16) != case["expected"][row][col]:
            raise ValueError(f"{case['name']} output mismatch at ({row},{col})")
    if len(seen) != m*n:
        raise ValueError(f"{case['name']} captured {len(seen)} of {m*n} outputs")
    metrics.update({"name": case["name"], "m": m, "k": k, "n": n,
                    "shared_bytes": 2*(metrics["a_load_words"]+metrics["w_load_words"]+
                                      metrics["bias_load_words"]+metrics["out_write_words"])})
    print(f"PASS {case['name']}: {m}x{k} * {k}x{n}, "
          f"cycles={metrics['cycles']}, shared_bytes={metrics['shared_bytes']}, "
          f"outputs={metrics['checked_outputs']}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, action="append",
                        help="optional int16 case.json with a[M][K], b[K][N], bias[N]")
    parser.add_argument("--quick", action="store_true", help="skip 99x64x256 case")
    parser.add_argument("--rows", type=int, choices=(8, 16), default=8,
                        help="parallel A rows / CACC banks")
    parser.add_argument("--trace", action="store_true", help="write VCD files")
    parser.add_argument("--staged", action="store_true",
                        help="use RTL CACC output staging and overlap writeback")
    parser.add_argument("--out", type=Path, help="retain build, vectors, and logs")
    args = parser.parse_args()
    if not shutil.which("verilator"):
        parser.error("verilator is not installed")
    cases = [load_case(path) for path in args.case] if args.case else [
        make_case("tiny"), make_case("ragged"), make_case("overflow")]
    if not args.case and not args.quick:
        cases.append(make_case("full"))
    if args.out:
        workdir = args.out.resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        cleanup = None
    else:
        cleanup = tempfile.TemporaryDirectory(prefix="kwt-arch-sim-")
        workdir = Path(cleanup.name)
    try:
        executable = build(workdir, args.trace, args.rows, args.staged)
        summary = [run_case(executable, case, workdir, args.trace, args.rows, args.staged)
                   for case in cases]
        (workdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"PASS {len(summary)}/{len(summary)} architecture cases; logs={workdir}")
        return 0
    finally:
        if cleanup:
            cleanup.cleanup()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        sys.exit(1)
