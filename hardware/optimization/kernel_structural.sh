#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "$0")/../.." && pwd)"
out_dir="${KERNEL_STRUCT_OUT:-/tmp/kwt-kernel-structural}"
mkdir -p "$out_dir"
cd "$repo_dir"
for variant in cacc os; do
    if [[ "$variant" == cacc ]]; then
        rtl=hardware/rtl/pulse_array/kwt_cacc.sv
        top=kwt_cacc
    else
        rtl=hardware/optimization/kernel_os_accum.sv
        top=kernel_os_accum
    fi
    yosys -Q -T -p "read_verilog -sv $rtl; chparam -set ROWS 8 -set COLS 16 $top; hierarchy -top $top; proc; opt; memory_collect; stat" \
        > "$out_dir/$variant.log" 2>&1
done
python3 - "$out_dir" <<'PY'
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
for variant in ('cacc', 'os'):
    report = (root / f'{variant}.log').read_text()
    counts = {kind: int(value) for value, kind in re.findall(
        r'^\s+(\d+)\s+(\$(?:add|adffe|mem_v2))\s*$', report, re.M)}
    print(f'{variant}: add={counts.get("$add", 0)} '
          f'adffe={counts.get("$adffe", 0)} mem_v2={counts.get("$mem_v2", 0)}')
    if counts.get('$add') != (8 if variant == 'cacc' else 128):
        raise SystemExit(f'unexpected adder count: {variant}')
print('PASS kernel structural comparison')
PY
