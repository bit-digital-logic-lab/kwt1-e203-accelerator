#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/../.." && pwd)"
out_dir="${MAC_VERIFY_OUT:-/tmp/kwt-mac-pipe3-verify}"
mkdir -p "$out_dir"
cd "$repo_dir"

verilator --binary --timing --assert -Wno-fatal -j 4 \
    --top-module tb_kwt_mac_pipe3 --Mdir "$out_dir/obj" \
    hardware/rtl/pulse_array/kwt_row_broadcast_tile.sv \
    hardware/rtl/pulse_array/kwt_row_broadcast_tile_pipe3.sv \
    hardware/rtl/pulse_array/tb/tb_kwt_mac_pipe3.sv \
    > "$out_dir/build.log" 2>&1
"$out_dir/obj/Vtb_kwt_mac_pipe3" > "$out_dir/sim.log" 2>&1
rg -q 'PASS pipe3 compute-only II=1 regression' "$out_dir/sim.log"

for variant in base pipe3; do
    if [[ "$variant" == base ]]; then
        top=kwt_row_broadcast_tile
        rtl=hardware/rtl/pulse_array/kwt_row_broadcast_tile.sv
    else
        top=kwt_row_broadcast_tile_pipe3
        rtl=hardware/rtl/pulse_array/kwt_row_broadcast_tile_pipe3.sv
    fi
    yosys -Q -T -p \
        "read_verilog -sv $rtl; chparam -set ROWS 16 $top; synth -top $top; ltp -noff; stat" \
        > "$out_dir/$variant.synth.log" 2>&1
done

python3 - "$out_dir" <<'PY'
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
for variant in ('base', 'pipe3'):
    log = (root / f'{variant}.synth.log').read_text()
    depth = re.search(r'Longest topological path.*length=(\d+)', log)
    cells = re.findall(r'^\s+(\d+) cells$', log, re.M)
    if not depth or not cells:
        raise SystemExit(f'missing synthesis metrics for {variant}')
    print(f'{variant}: generic cells={cells[-1]}, logic levels={depth.group(1)}')
print('PASS MAC pipe3 verification')
PY
