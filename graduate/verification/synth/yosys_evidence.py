#!/usr/bin/env python3
"""Yosys structural synthesis evidence for the FFN1 compute core.

This is a *synthesis* evidence class: it elaborates the frozen RTL, runs
``proc/opt/memory_collect/check -assert`` and counts structural cells.  It is
deliberately not a timing, power or area sign-off — the mapping (when a library
is available) only produces an example-library area for relative comparison.

The script prints a ``KWT-GRAD-VERIFY synth checked=N failed=0`` contract line
so the staged runner can only PASS after the real Yosys run, never from file
existence.  ``--corrupt-mul-count`` lowers the expected multiplier count below
the observed one so the self-check must fail (used by the fault tests).
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

DEFAULT_LIBERTY = "/home/yang/yosys-sta/pdk/nangate45/lib/Nangate45_typ.lib"


def run_yosys(yosys: str, script: str, timeout: int = 600) -> tuple[int, str]:
    proc = subprocess.run([yosys, "-Q", "-T", "-p", script],
                          capture_output=True, text=True, timeout=timeout)
    return proc.returncode, proc.stdout + proc.stderr


def count_cells(output: str, cell: str) -> int | None:
    match = re.search(r"^\s*(\d+)\s+" + re.escape(cell) + r"\s*$", output, re.MULTILINE)
    return int(match.group(1)) if match else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-root", type=pathlib.Path, required=True)
    parser.add_argument("--yosys", default="yosys")
    parser.add_argument("--liberty", default=DEFAULT_LIBERTY)
    parser.add_argument("--want-mul", type=int, default=1)
    parser.add_argument("--corrupt-mul-count", action="store_true",
                        help="require one more multiplier than the design has (fault injection)")
    parser.add_argument("--prefix", default="synth")
    args = parser.parse_args()

    root = args.design_root.resolve()
    rtl = {
        "product_array": root / "hardware/rtl/pulse_array/kwt_product_array.sv",
        "cacc": root / "hardware/rtl/pulse_array/kwt_cacc.sv",
        "row_broadcast_tile": root / "hardware/rtl/pulse_array/kwt_row_broadcast_tile.sv",
    }
    missing = [str(p) for p in rtl.values() if not p.exists()]
    if missing:
        print("SKIP: missing RTL: " + ", ".join(missing))
        return 0

    top = "kwt_row_broadcast_tile"
    files = " ".join(str(p) for p in rtl.values())
    coarse_script = (
        f"read_verilog -sv {files}; "
        f"hierarchy -check -top {top}; proc; opt; memory_collect; check -assert; stat"
    )

    checks = 0
    failures = 0
    rc, output = run_yosys(args.yosys, coarse_script)
    open("/tmp/kwt-grad-synth-coarse.log", "w", encoding="utf-8").write(output)
    if rc != 0:
        print(f"FAIL yosys coarse exited {rc}")
        print(f"KWT-GRAD-VERIFY {args.prefix} checked=0 failed=1")
        return 1
    if "Found and reported 0 problems" not in output:
        print("FAIL check -assert did not report a clean design")
        print(f"KWT-GRAD-VERIFY {args.prefix} checked={checks + 1} failed=1")
        return 1
    checks += 1
    print(f"PASS elaboration + check -assert clean for top {top}")

    multipliers = count_cells(output, "$mul")
    memories = count_cells(output, "$mem_v2")
    adds = count_cells(output, "$add")
    print(f"  structural counts: $mul={multipliers} $mem_v2={memories} $add={adds}")

    want_mul = args.want_mul + (1000 if args.corrupt_mul_count else 0)
    if multipliers is None or multipliers < want_mul:
        print(f"FAIL expected >= {want_mul} $mul, observed {multipliers}")
        failures += 1
    else:
        checks += 1
        print(f"PASS $mul count {multipliers} >= {want_mul}")
    if memories is None or memories < 1:
        print(f"FAIL expected >= 1 $mem_v2, observed {memories}")
        failures += 1
    else:
        checks += 1
        print(f"PASS $mem_v2 count {memories} >= 1")

    liberty = pathlib.Path(args.liberty)
    if liberty.exists():
        mapped_script = (
            f"read_liberty -lib -ignore_miss_func {liberty}; "
            f"read_verilog -sv {rtl['cacc']}; synth -top kwt_cacc; "
            f"dfflibmap -liberty {liberty}; abc -liberty {liberty}; check -assert; "
            f"stat -liberty {liberty}"
        )
        mrc, moutput = run_yosys(args.yosys, mapped_script)
        open("/tmp/kwt-grad-synth-cacc.log", "w", encoding="utf-8").write(moutput)
        area = re.search(r"Chip area for module '\\?kwt_cacc':\s*([\d.eE+-]+)", moutput)
        if mrc == 0 and area:
            checks += 1
            print(f"PASS kwt_cacc mapped with example library {liberty.name}: "
                  f"Chip area={area.group(1)} (NOT a sign-off area)")
        else:
            failures += 1
            print(f"FAIL optional kwt_cacc mapping rc={mrc}")
    else:
        print(f"NOTE liberty {liberty} absent; skipped mapped-area evidence")

    status = "failed=0" if failures == 0 else f"failed={failures}"
    print(f"KWT-GRAD-VERIFY {args.prefix} checked={checks} {status}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
