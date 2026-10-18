#!/usr/bin/env bash
# 16-row adapter regression. Drives the same real A12W10 vectors used by the
# legacy E203 path (layer??_a16/w16/bias32/expected32.hex) through the 16-row
# adapter + shared DMA/bus + external-memory model.
#
# Usage: bash run_row16.sh CASE_ROOT LAYER  [--verilator]
#   CASE_ROOT: dir containing layerNN_a16.hex etc.
#   LAYER:     two decimal digits, e.g. 01
set -euo pipefail
tb_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
rtl_dir="$(cd -- "$tb_dir/.." && pwd)"
mem_dir="$(cd -- "$rtl_dir/../sram" && pwd)"
dma_dir="$(cd -- "$rtl_dir/../dma" && pwd)"
pulse_dir="$(cd -- "$rtl_dir/../pulse_array" && pwd)"

if [[ $# -lt 2 ]]; then
    echo "Usage: bash run_row16.sh CASE_ROOT LAYER [--verilator]" >&2
    exit 2
fi
case_root="$1"
layer="$2"
engine="${3:-iverilog}"
if [[ ! -d "$case_root" ]]; then echo "no such CASE_ROOT: $case_root" >&2; exit 2; fi

tmp_base="${KWT_E203_TMP_BASE:-${TMPDIR:-/tmp}}"
if [[ "$tmp_base" != /* || "$tmp_base" == *[!\ -~]* ]]; then tmp_base=/tmp; fi
build_dir="$(mktemp -d "$tmp_base/kwt_row16.XXXXXX")"
trap 'rm -rf -- "$build_dir"' EXIT

stage="$build_dir/cases"
mkdir -p "$stage"
for suffix in a16 w16 bias32 expected32; do
    src="$case_root/layer${layer}_${suffix}.hex"
    [[ -f "$src" ]] || { echo "missing $src" >&2; exit 2; }
    cp -f -- "$src" "$stage/layer${layer}_${suffix}.hex"
done

rtl_files=(
    "$pulse_dir/kwt_row_broadcast_tile.sv"
    "$dma_dir/kwt_dma.sv"
    "$dma_dir/kwt_ext_memory_model.sv"
    "$rtl_dir/kwt_row16_tile_adapter.sv"
    "$tb_dir/tb_kwt_row16_adapter.sv"
)

if [[ "$engine" == "--verilator" ]]; then
    verilator --binary --timing --assert -Wno-fatal --top-module tb_kwt_row16_adapter \
        -I"$mem_dir" --Mdir "$build_dir/obj" "${rtl_files[@]}"
    exec "$build_dir/obj/Vtb_kwt_row16_adapter" \
        "+A=$stage/layer${layer}_a16.hex" "+W=$stage/layer${layer}_w16.hex" \
        "+BIAS=$stage/layer${layer}_bias32.hex" \
        "+EXPECTED=$stage/layer${layer}_expected32.hex"
fi

iverilog -g2012 -gno-assertions -I"$mem_dir" -s tb_kwt_row16_adapter \
    -o "$build_dir/row16.vvp" "${rtl_files[@]}"
exec vvp "$build_dir/row16.vvp" \
    "+A=$stage/layer${layer}_a16.hex" "+W=$stage/layer${layer}_w16.hex" \
    "+BIAS=$stage/layer${layer}_bias32.hex" \
    "+EXPECTED=$stage/layer${layer}_expected32.hex"
