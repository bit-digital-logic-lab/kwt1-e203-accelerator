#!/usr/bin/env bash
# Positive-run cross-check of the NICE-like protocol boundary suite with
# Verilator. The fault-injection negative runs live in run_nice_protocol.sh
# (Icarus); this only re-runs the clean suite to confirm the RTL behaves the
# same under a second simulator.
set -uo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
rtl="$(cd -- "$here/../../rtl" && pwd)"
tb="$here/tb_nice_protocol_boundaries.sv"
build="$(mktemp -d /tmp/kwt-nice-verilator.XXXXXX)"
trap 'rm -rf -- "$build"' EXIT

VERILATOR="${VERILATOR:-}"
if [ -z "$VERILATOR" ] || ! command -v "$VERILATOR" >/dev/null 2>&1; then
    if command -v verilator >/dev/null 2>&1; then VERILATOR=verilator
    else echo "verilator not found" >&2; exit 2; fi
fi
echo "verilator: $VERILATOR ($("$VERILATOR" --version 2>&1 | head -1))"

"$VERILATOR" --binary --timing -Wno-fatal --top-module tb_nice_protocol_boundaries \
    -I"$rtl/sram" --Mdir "$build/obj" \
    "$rtl/pulse_array/kwt_pe.sv" \
    "$rtl/pulse_array/kwt_pulse_array.sv" \
    "$rtl/pulse_array/kwt_postprocess.sv" \
    "$rtl/pulse_array/kwt_tile_engine.sv" \
    "$rtl/pulse_array/kwt_product_array.sv" \
    "$rtl/pulse_array/kwt_cacc.sv" \
    "$rtl/pulse_array/kwt_pingpong_cacc_engine.sv" \
    "$rtl/sram/kwt_operand_pingpong.sv" \
    "$rtl/sram/kwt_tile_buffers.sv" \
    "$rtl/dma/kwt_dma.sv" \
    "$rtl/dma/kwt_ext_memory_model.sv" \
    "$rtl/control/kwt_tile_system_model.sv" \
    "$rtl/control/kwt_nice_system_model.sv" \
    "$tb" >"$build/compile.log" 2>&1 || { echo "[COMPILE-FAIL]"; cat "$build/compile.log"; exit 1; }

"$build/obj/Vtb_nice_protocol_boundaries" 2>&1 | tee "$build/run.log"
rc=${PIPESTATUS[0]}
echo "VERILATOR_RUN_EXIT=$rc"
if [ "$rc" -eq 0 ] && grep -q "PASS nice_protocol_boundaries" "$build/run.log"; then
    echo "VERILATOR_RESULT PASS"
    exit 0
fi
echo "VERILATOR_RESULT FAIL"
exit 1
