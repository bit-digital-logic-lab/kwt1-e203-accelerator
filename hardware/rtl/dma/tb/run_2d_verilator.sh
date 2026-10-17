#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "$0")/../../../.." && pwd)"
build_dir="$(mktemp -d /tmp/kwt-dma-2d.XXXXXX)"
trap 'rm -rf "$build_dir"' EXIT
cd "$repo_dir"
verilator --binary --timing --top-module tb_kwt_dma_2d \
    -Wno-fatal -Ihardware/rtl/sram \
    --Mdir "$build_dir/obj" \
    hardware/rtl/dma/kwt_dma.sv \
    hardware/rtl/dma/kwt_dma_2d.sv \
    hardware/rtl/dma/kwt_ext_memory_model.sv \
    hardware/rtl/sram/kwt_tile_buffers.sv \
    hardware/rtl/dma/tb/tb_kwt_dma_2d.sv \
    > "$build_dir/build.log" 2>&1 || { cat "$build_dir/build.log"; exit 1; }
"$build_dir/obj/Vtb_kwt_dma_2d"
