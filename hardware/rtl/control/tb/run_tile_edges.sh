#!/usr/bin/env bash
# Tile-level edge-case regression: last-tile padding, continuous commands,
# in-flight reset, and a negative-injection check of the independent golden.
#
# Requires an Icarus Verilog that can elaborate the frozen RTL's unpacked
# arrays (Icarus 14 from oss-cad-suite is tested; Ubuntu's Icarus 11 cannot).
# Override the tools with IVERILOG=/path/iverilog VVP=/path/vvp.
set -euo pipefail
tb_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 -u "$tb_dir/run_tile_edge_cases.py" "$@"
