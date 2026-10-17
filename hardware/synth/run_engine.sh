#!/usr/bin/env bash
set -euo pipefail

hardware_dir="$(cd "$(dirname "$0")/.." && pwd)"
repo_dir="$(cd "$hardware_dir/.." && pwd)"
output_dir="${SYNTH_OUT:-/tmp/kwt-engine-synth}"
liberty="${LIBERTY:-/home/yang/yosys-sta/pdk/nangate45/lib/Nangate45_typ.lib}"
full_map="${1:-}"

if [[ "$full_map" != "" && "$full_map" != "--full-flop-map" ]]; then
    echo "Usage: bash hardware/synth/run_engine.sh [--full-flop-map]" >&2
    exit 2
fi
if [[ ! -f "$liberty" ]]; then
    echo "Missing Liberty library: $liberty" >&2
    exit 2
fi
mkdir -p "$output_dir"
cd "$repo_dir"

rtl_files='hardware/rtl/pulse_array/kwt_product_array.sv hardware/rtl/pulse_array/kwt_cacc.sv hardware/rtl/sram/kwt_operand_pingpong.sv hardware/rtl/pulse_array/kwt_pingpong_cacc_engine.sv'

# Preserve inferred SRAMs to inspect bank count, arithmetic units, and ports.
yosys -Q -T -p "read_verilog -sv $rtl_files; hierarchy -check -top kwt_pingpong_cacc_engine; proc; opt; memory_collect; check -assert; stat" \
    > "$output_dir/engine_coarse.log" 2>&1

# A standard-cell-only reference maps the CACC banks to flip-flops.
yosys -Q -T -p "read_liberty -lib -ignore_miss_func $liberty; read_verilog -sv hardware/rtl/pulse_array/kwt_cacc.sv; synth -top kwt_cacc; dfflibmap -liberty $liberty; abc -liberty $liberty; check -assert; stat -liberty $liberty" \
    > "$output_dir/cacc_flopmapped.log" 2>&1

if [[ "$full_map" == "--full-flop-map" ]]; then
    yosys -Q -T -p "read_liberty -lib -ignore_miss_func $liberty; read_verilog -sv $rtl_files; synth -top kwt_pingpong_cacc_engine -flatten; dfflibmap -liberty $liberty; abc -liberty $liberty; check -assert; stat -liberty $liberty" \
        > "$output_dir/engine_flopmapped.log" 2>&1
fi

echo "Synthesis logs: $output_dir"
logs=("$output_dir/engine_coarse.log" "$output_dir/cacc_flopmapped.log")
if [[ "$full_map" == "--full-flop-map" ]]; then
    logs+=("$output_dir/engine_flopmapped.log")
fi
rg -n '^=== design hierarchy ===|^=== kwt_cacc ===|^=== kwt_pingpong_cacc_engine ===|^[[:space:]]+[0-9]+[[:space:]]+\$mem_v2|^[[:space:]]+[0-9]+[[:space:]]+\$mul|^[[:space:]]+[0-9]+[[:space:]]+\$add|^[[:space:]]+Chip area|^Found and reported 0 problems' \
    "${logs[@]}" | tail -40
