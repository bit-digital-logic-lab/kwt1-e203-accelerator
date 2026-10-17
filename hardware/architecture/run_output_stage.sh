#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/../.." && pwd)"
output_dir="${OUTPUT_STAGE_OUT:-/tmp/kwt-output-stage-test}"
mkdir -p "$output_dir"
cd "$repo_dir"

verilator --binary --timing --assert -Wno-fatal -j 4 \
    --top-module tb_kwt_tile_output_stage --Mdir "$output_dir/obj" \
    hardware/rtl/pulse_array/kwt_row_broadcast_tile.sv \
    hardware/rtl/pulse_array/kwt_tile_output_stage.sv \
    hardware/architecture/tb/tb_kwt_tile_output_stage.sv \
    > "$output_dir/build.log" 2>&1
"$output_dir/obj/Vtb_kwt_tile_output_stage" | tee "$output_dir/run.log"

yosys -Q -T -p \
    'read_verilog -sv hardware/rtl/pulse_array/kwt_tile_output_stage.sv; hierarchy -top kwt_tile_output_stage; proc; opt; memory_collect; check -assert; stat' \
    > "$output_dir/synth.log"
rg -n 'Found and reported 0 problems|\$mem_v2' "$output_dir/synth.log" | tail -2
echo "PASS output-stage Verilator and Yosys; logs=$output_dir"
