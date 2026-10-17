#!/usr/bin/env bash
set -euo pipefail
base_dir="$(cd "$(dirname "$0")/../.." && pwd)"
sim_out="$(mktemp)"
trap 'rm -f "$sim_out"' EXIT
iverilog -g2012 -I "$base_dir/sram" -s tb_kwt_dma -o "$sim_out" \
  "$base_dir/dma/kwt_dma.sv" \
  "$base_dir/dma/kwt_ext_memory_model.sv" \
  "$base_dir/sram/kwt_tile_buffers.sv" \
  "$base_dir/dma/tb/tb_kwt_dma.sv"
vvp "$sim_out"
