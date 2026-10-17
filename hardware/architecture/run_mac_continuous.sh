#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/../.." && pwd)"
output_dir="${MAC_CONTINUOUS_OUT:-/tmp/kwt-mac-continuous}"
mkdir -p "$output_dir"
cd "$repo_dir"

verilator --binary --timing --assert -Wno-fatal -j 4 \
    --top-module tb_kwt_mac_continuous --Mdir "$output_dir/obj" \
    hardware/rtl/pulse_array/kwt_row_broadcast_tile.sv \
    hardware/architecture/tb/tb_kwt_mac_continuous.sv \
    > "$output_dir/build.log" 2>&1
"$output_dir/obj/Vtb_kwt_mac_continuous" | tee "$output_dir/run.log"

yosys -Q -T -p \
    'read_verilog -sv hardware/rtl/pulse_array/kwt_row_broadcast_tile.sv; chparam -set ROWS 16 kwt_row_broadcast_tile; synth -top kwt_row_broadcast_tile; ltp -noff' \
    > "$output_dir/logic_depth.log"
rg -n 'Longest topological path|^   ff:' "$output_dir/logic_depth.log" | tail -2
echo "PASS compute-only continuous MAC; logs=$output_dir"
