#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/../.." && pwd)"
output_dir="${SYNTH_OUT:-/tmp/kwt-architecture-synth}"
liberty="${LIBERTY:-/home/yang/yosys-sta/pdk/nangate45/lib/Nangate45_typ.lib}"
mode="${1:-}"
row_broadcast_rows="${ROW_BROADCAST_ROWS:-8}"
if [[ "$mode" != "" && "$mode" != "--coarse-only" ]]; then
    echo "Usage: bash hardware/architecture/run_compare_synth.sh [--coarse-only]" >&2
    exit 2
fi
if [[ "$row_broadcast_rows" != "8" && "$row_broadcast_rows" != "16" ]]; then
    echo "ROW_BROADCAST_ROWS must be 8 or 16" >&2
    exit 2
fi
if [[ "$mode" != "--coarse-only" && ! -f "$liberty" ]]; then
    echo "Missing Liberty library: $liberty" >&2
    exit 2
fi
mkdir -p "$output_dir"
cd "$repo_dir"

legacy_rtl='hardware/rtl/pulse_array/kwt_product_array.sv hardware/rtl/pulse_array/kwt_cacc.sv hardware/rtl/pulse_array/kwt_systolic_cacc_tile.sv'
row_rtl='hardware/rtl/pulse_array/kwt_row_broadcast_tile.sv'

for design in systolic row_broadcast; do
    if [[ "$design" == systolic ]]; then
        rtl="$legacy_rtl"
        top=kwt_systolic_cacc_tile
        parameterize=''
    else
        rtl="$row_rtl"
        top=kwt_row_broadcast_tile
        parameterize="chparam -set ROWS $row_broadcast_rows $top;"
    fi
    yosys -Q -T -p "read_verilog -sv $rtl; $parameterize hierarchy -check -top $top; proc; opt; memory_collect; check -assert; stat" \
        > "$output_dir/${design}_coarse.log" 2>&1
    if [[ "$mode" != "--coarse-only" ]]; then
        yosys -Q -T -p "read_liberty -lib -ignore_miss_func $liberty; read_verilog -sv $rtl; $parameterize synth -top $top -flatten; dfflibmap -liberty $liberty; abc -liberty $liberty; check -assert; stat -liberty $liberty" \
            > "$output_dir/${design}_flopmapped.log" 2>&1
    fi
done

echo "Compute-only synthesis logs: $output_dir (row broadcast ROWS=$row_broadcast_rows)"
for design in systolic row_broadcast; do
    echo "== $design, memories preserved =="
    rg -n '^=== design hierarchy ===|^=== kwt_row_broadcast_tile ===|^[[:space:]]+[0-9]+[[:space:]]+\$mul|^[[:space:]]+[0-9]+[[:space:]]+\$add|^[[:space:]]+[0-9]+[[:space:]]+\$mem_v2|^Found and reported 0 problems' \
        "$output_dir/${design}_coarse.log" | tail -12
    if [[ "$mode" != "--coarse-only" ]]; then
        echo "== $design, Nangate45 with psums mapped to flops =="
        rg -n '^=== kwt_systolic_cacc_tile ===|^=== kwt_row_broadcast_tile ===|^[[:space:]]+[0-9]+.*cells|^[[:space:]]+Chip area|^Found and reported 0 problems' \
            "$output_dir/${design}_flopmapped.log" | tail -8
    fi
done
