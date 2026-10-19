#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../../.." && pwd)"
python3 "$here/make_variant.py"
verilator --binary --timing --top-module tb_zero_gate -Wno-fatal \
  --Mdir "$here/.build" \
  "$root/hardware/rtl/pulse_array/kwt_row_broadcast_tile.sv" \
  "$here/kwt_row_broadcast_tile_zero_gate.sv" "$here/tb_zero_gate.sv" \
  > "$here/build.log" 2>&1
"$here/.build/Vtb_zero_gate" | tee "$here/sim.log"
