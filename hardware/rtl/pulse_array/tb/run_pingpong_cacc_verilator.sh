#!/usr/bin/env bash
set -euo pipefail

tb_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
rtl_dir="$(dirname -- "$tb_dir")"
repo_dir="$(cd -- "$rtl_dir/../../.." && pwd)"
build_dir="$(mktemp -d /tmp/kwt-pingpong-cacc.XXXXXX)"
trap 'rm -rf -- "$build_dir"' EXIT

verilator --binary --timing --assert -Wall -Wno-fatal \
    --top-module tb_kwt_pingpong_cacc_engine \
    --Mdir "$build_dir/obj" \
    "$rtl_dir/kwt_product_array.sv" \
    "$rtl_dir/kwt_cacc.sv" \
    "$repo_dir/hardware/rtl/sram/kwt_operand_pingpong.sv" \
    "$rtl_dir/kwt_pingpong_cacc_engine.sv" \
    "$tb_dir/tb_kwt_pingpong_cacc_engine.sv"
"$build_dir/obj/Vtb_kwt_pingpong_cacc_engine"
