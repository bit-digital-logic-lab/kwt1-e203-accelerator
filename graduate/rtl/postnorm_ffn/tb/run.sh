#!/usr/bin/env bash
# Build stimulus, run the RTL simulation, and check it bit-exactly against the
# independent integer reference model.
#
#   SIM=verilator (default when available) or SIM=iverilog
#   BUILD_DIR=<dir> overrides the artifact directory
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
vectors="$here/model/vectors"
build="${BUILD_DIR:-$here/build}"
sim="${SIM:-verilator}"
mkdir -p "$build"

python3 "$here/model/gen_config.py" --out "$here/model" >/dev/null

rtl=(
    "$here/rtl/kwt_postnorm_pkg.sv"
    "$here/rtl/kwt_residual_add.sv"
    "$here/rtl/kwt_layernorm.sv"
    "$here/rtl/kwt_gelu.sv"
    "$here/rtl/kwt_ffn2.sv"
    "$here/rtl/kwt_postnorm_ffn.sv"
)

if [ "$sim" = verilator ] && command -v verilator >/dev/null; then
    verilator --binary --timing -j 4 -Wno-fatal -I"$here/rtl" \
        --top-module tb_kwt_postnorm_ffn -o sim_vl --Mdir "$build/obj_vl" \
        "${rtl[@]}" "$here/tb/tb_kwt_postnorm_ffn.sv" > "$build/build.log" 2>&1
    bin="$build/obj_vl/sim_vl"
else
    iverilog -g2012 -I "$here/rtl" -o "$build/sim.out" "${rtl[@]}" \
        "$here/tb/tb_kwt_postnorm_ffn.sv" 2> "$build/build.log"
    bin="vvp $build/sim.out"
fi

$bin "+XHEX=$vectors/x.hex" "+AHEX=$vectors/a.hex" "+HHEX=$vectors/h.hex" \
    "+W2HEX=$vectors/w2.hex" "+B2HEX=$vectors/b2.hex" "+OUT=$build" \
    | tee "$build/sim.log"
grep -q "TB_PASS" "$build/sim.log"

python3 "$here/model/verify.py" --build "$build" --vectors "$vectors" \
    | tee "$build/verify.log"
grep -q "VERIFY_PASS" "$build/verify.log"
