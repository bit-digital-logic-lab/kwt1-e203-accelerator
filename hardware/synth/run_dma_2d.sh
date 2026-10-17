#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "$0")/../.." && pwd)"
output="${SYNTH_LOG:-/tmp/kwt-dma-2d-synth.log}"
cd "$repo_dir"
yosys -Q -T -p \
    'read_verilog -sv -Ihardware/rtl/sram hardware/rtl/dma/kwt_dma.sv hardware/rtl/dma/kwt_dma_2d.sv; hierarchy -check -top kwt_dma_2d; proc; opt; check -assert; stat' \
    > "$output" 2>&1
rg 'Found and reported 0 problems|=== design hierarchy ===|\$mul|\$div|\$mod' "$output" | tail -20
echo "Synthesis log: $output"
