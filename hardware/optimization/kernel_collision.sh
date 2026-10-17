#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "$0")/../.." && pwd)"
out_dir="${KERNEL_COLLISION_OUT:-/tmp/kwt-kernel-collision}"
mkdir -p "$out_dir"
cd "$repo_dir"
verilator --binary --timing --assert -Wno-fatal \
  --top-module kernel_collision_tb --Mdir "$out_dir/obj" \
  hardware/rtl/pulse_array/kwt_product_array.sv \
  hardware/rtl/pulse_array/kwt_cacc.sv \
  hardware/optimization/kernel_collision_tb.sv > "$out_dir/build.log" 2>&1
"$out_dir/obj/Vkernel_collision_tb"
