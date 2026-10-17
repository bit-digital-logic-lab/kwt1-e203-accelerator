#!/usr/bin/env bash
set -euo pipefail
tb_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
rtl_dir="$(dirname -- "$tb_dir")"
build_dir="$(mktemp -d /tmp/kwt-system-verilator.XXXXXX)"
trap 'rm -rf -- "$build_dir"' EXIT

verilator --binary --timing --assert -Wno-fatal --top-module tb_kwt_tile_system_model \
    -I"$rtl_dir/../sram" --Mdir "$build_dir/obj" \
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
    "$tb_dir/tb_kwt_tile_system_model.sv"
"$build_dir/obj/Vtb_kwt_tile_system_model"
