#!/usr/bin/env bash
set -euo pipefail
tb_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
rtl_dir="$(dirname -- "$tb_dir")"
build_dir="$(mktemp -d)"
trap 'rm -rf -- "$build_dir"' EXIT
iverilog -g2012 -I "$rtl_dir/../sram" -s tb_kwt_tile_system_model \
    -o "$build_dir/model" \
    "$rtl_dir/../pulse_array/kwt_pe.sv" \
    "$rtl_dir/../pulse_array/kwt_pulse_array.sv" \
    "$rtl_dir/../pulse_array/kwt_postprocess.sv" \
    "$rtl_dir/../pulse_array/kwt_tile_engine.sv" \
    "$rtl_dir/../pulse_array/kwt_product_array.sv" \
    "$rtl_dir/../pulse_array/kwt_cacc.sv" \
    "$rtl_dir/../pulse_array/kwt_pingpong_cacc_engine.sv" \
    "$rtl_dir/../sram/kwt_operand_pingpong.sv" \
    "$rtl_dir/../sram/kwt_tile_buffers.sv" \
    "$rtl_dir/../dma/kwt_dma.sv" \
    "$rtl_dir/../dma/kwt_ext_memory_model.sv" \
    "$rtl_dir/kwt_tile_system_model.sv" \
    "$tb_dir/tb_kwt_tile_system_model.sv" \
    2>"$build_dir/compile.log" || { cat "$build_dir/compile.log" >&2; exit 1; }
vvp "$build_dir/model"
