#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "$0")/../.." && pwd)"
out_dir="${KERNEL_OS_OUT:-/tmp/kwt-kernel-os}"
mkdir -p "$out_dir"
cd "$repo_dir"
verilator --binary --timing --assert -Wno-fatal \
  --top-module kernel_os_tb --Mdir "$out_dir/obj" \
  hardware/rtl/pulse_array/kwt_product_array.sv \
  hardware/optimization/kernel_os_accum.sv \
  hardware/optimization/kernel_os_tb.sv > "$out_dir/build.log" 2>&1
"$out_dir/obj/Vkernel_os_tb"
