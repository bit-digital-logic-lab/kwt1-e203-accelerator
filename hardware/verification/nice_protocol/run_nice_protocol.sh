#!/usr/bin/env bash
# Build and run the NICE-like protocol boundary suite with Icarus Verilog.
#
# Positive run must exit 0 with the PASS banner. Two fault-injection negative
# runs must exit non-zero and be caught by a named checker. The script itself
# only reports what the simulators actually returned (exit codes + log gates).
set -uo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
rtl="$(cd -- "$here/../../rtl" && pwd)"
tb="$here/tb_nice_protocol_boundaries.sv"
build="$(mktemp -d /tmp/kwt-nice-protocol.XXXXXX)"
trap 'rm -rf -- "$build"' EXIT

find_tool() {
    local name="$1"
    local override="${2:-}"
    if [ -n "$override" ] && command -v "$override" >/dev/null 2>&1; then
        command -v "$override"; return 0
    fi
    if [ -x "/home/yang/yosys-sta/oss-cad-suite/bin/$name" ]; then
        echo "/home/yang/yosys-sta/oss-cad-suite/bin/$name"; return 0
    fi
    command -v "$name"
}

IVERILOG="$(find_tool iverilog "${IVERILOG:-}")" || { echo "iverilog not found" >&2; exit 2; }
VVP="$(find_tool vvp "${VVP:-}")" || { echo "vvp not found" >&2; exit 2; }

SOURCES=(
    "$rtl/pulse_array/kwt_pe.sv"
    "$rtl/pulse_array/kwt_pulse_array.sv"
    "$rtl/pulse_array/kwt_postprocess.sv"
    "$rtl/pulse_array/kwt_tile_engine.sv"
    "$rtl/pulse_array/kwt_product_array.sv"
    "$rtl/pulse_array/kwt_cacc.sv"
    "$rtl/pulse_array/kwt_pingpong_cacc_engine.sv"
    "$rtl/sram/kwt_operand_pingpong.sv"
    "$rtl/sram/kwt_tile_buffers.sv"
    "$rtl/dma/kwt_dma.sv"
    "$rtl/dma/kwt_ext_memory_model.sv"
    "$rtl/control/kwt_tile_system_model.sv"
    "$rtl/control/kwt_nice_system_model.sv"
    "$tb"
)

echo "iverilog : $IVERILOG ($("$IVERILOG" -V 2>&1 | head -1))"
echo "vvp      : $VVP"

run_case() {
    local name="$1" define="$2" expect_exit="$3" gate="$4"
    local defargs=()
    [ -n "$define" ] && defargs=("-D$define")
    local log="$build/$name.log"
    local bin="$build/$name.bin"
    if ! "$IVERILOG" -g2012 -I "$rtl/sram" -s tb_nice_protocol_boundaries \
            "${defargs[@]}" -o "$bin" "${SOURCES[@]}" >"$build/$name.compile.log" 2>&1; then
        echo "[COMPILE-FAIL] $name"
        cat "$build/$name.compile.log"
        return 1
    fi
    "$VVP" "$bin" >"$log" 2>&1
    local rc=$?
    local gate_ok=0
    if [ -n "$gate" ] && grep -q "$gate" "$log"; then gate_ok=1; fi
    local pass=0
    if [ "$expect_exit" = "nonzero" ]; then
        [ "$rc" -ne 0 ] && [ "$gate_ok" -eq 1 ] && pass=1
    else
        [ "$rc" -eq 0 ] && [ "$gate_ok" -eq 1 ] && pass=1
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$rc" "$expect_exit" "$gate" "$gate_ok" "$pass" >>"$build/results.tsv"
    echo "----- $name (exit=$rc, expect=$expect_exit, gate='$gate' found=$gate_ok) -----"
    tail -n 5 "$log"
    [ "$pass" -eq 1 ]
}

: >"$build/results.tsv"
overall=0

run_case "positive"          ""               "zero"    "PASS nice_protocol_boundaries" || overall=1
run_case "negative_rsp_fault" "NICE_FAULT_RSP" "nonzero" "RSP_VALUE_MISMATCH"            || overall=1
run_case "negative_bus_fault" "NICE_FAULT_BUS" "nonzero" "BUS_STABILITY_VIOLATION"       || overall=1

echo
echo "==== summary (name/exit/expect/gate/gate_found/pass) ===="
cat "$build/results.tsv"
if [ "$overall" -eq 0 ]; then
    echo "SUITE_RESULT PASS"
else
    echo "SUITE_RESULT FAIL"
fi
exit "$overall"
