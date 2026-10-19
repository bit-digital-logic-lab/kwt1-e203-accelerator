#!/usr/bin/env bash
# r2_pack_format — one-click reproduction.
#
#   L1  Python reference pack/unpack roundtrip + structural/value negative controls
#   L2  Verilator byte-lane RTL bit-exact check (little-endian + PACK12)
#   L4* generic yosys synthesis of the byte-lane RTL (NOT PDK / NOT 150 MHz)
#
# Writes only to this task directory and /tmp/opencode/r2_pack_format/.
# Missing evidence levels (E203 SoC, real PDK synthesis, official 600-sample
# acceptance) are NOT produced here and must not be inferred.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
BUILD="${BUILD_DIR:-/tmp/opencode/r2_pack_format}"
RTL_BUILD="$BUILD/rtl"
LOGS="$HERE/logs"
mkdir -p "$LOGS" "$RTL_BUILD"

PY="${PYTHON:-python3}"
SUMMARY="$LOGS/summary.tsv"
: > "$SUMMARY"
fail=0

run() {  # run <name> <required 0|1> <cmd...>
    local name="$1" required="$2"; shift 2
    echo "### $name: $*" >&2
    "$@" > "$LOGS/$name.log" 2>&1
    local rc=$?
    printf '%s\t%s\t%s\n' "$name" "$rc" "$( [ "$rc" -eq 0 ] && echo PASS || echo FAIL )" >> "$SUMMARY"
    if [ "$rc" -ne 0 ]; then
        echo "!! $name exit=$rc (log: logs/$name.log)" >&2
        tail -n 12 "$LOGS/$name.log" >&2
        [ "$required" -eq 1 ] && fail=1
    fi
    return $rc
}

cd "$ROOT"

# ---- L1: python reference -------------------------------------------------
run l1_run_pack 1 "$PY" "$HERE/run_pack.py" --build-dir "$BUILD"

# ---- L2: Verilator byte-lane ---------------------------------------------
run l2_emit_vectors 1 "$PY" "$HERE/emit_rtl_vectors.py" "$RTL_BUILD"

verilator --binary --timing -Wno-fatal -Wno-TIMESCALEMOD -Wno-WIDTHTRUNC \
    --top-module tb_byte_lane -Mdir "$RTL_BUILD/obj_dir" \
    "$HERE/rtl/kwt_pack_lane.sv" "$HERE/tb/tb_byte_lane.sv" \
    > "$LOGS/l2_verilator_build.log" 2>&1
rc=$?
printf 'l2_verilator_build\t%s\t%s\n' "$rc" "$( [ $rc -eq 0 ] && echo PASS || echo FAIL )" >> "$SUMMARY"
if [ $rc -ne 0 ]; then fail=1; echo "!! verilator build failed" >&2; tail -n 10 "$LOGS/l2_verilator_build.log" >&2; fi

if [ $rc -eq 0 ]; then
    ( cd "$RTL_BUILD" && timeout 120 ./obj_dir/Vtb_byte_lane ) \
        > "$LOGS/l2_verilator_run.log" 2>&1
    rc=$?
    printf 'l2_verilator_run\t%s\t%s\n' "$rc" "$( [ $rc -eq 0 ] && echo PASS || echo FAIL )" >> "$SUMMARY"
    if [ $rc -ne 0 ]; then fail=1; echo "!! byte-lane sim failed" >&2; tail -n 10 "$LOGS/l2_verilator_run.log" >&2; fi
fi

# ---- L4*: generic yosys synthesis (optional binary) ----------------------
YOSYS=""
for cand in \
    /home/yang/yosys-sta/oss-cad-suite/bin/yosys \
    "$(command -v yosys 2>/dev/null)"; do
    [ -n "$cand" ] && [ -x "$cand" ] && YOSYS="$cand" && break
done
if [ -n "$YOSYS" ]; then
    {
        echo "# yosys: $YOSYS ($($YOSYS -V 2>&1 | tail -1))"
        for top in kwt_le_unpack kwt_le_pack kwt_bit_unpack; do
            echo "===== synth -top $top ====="
            "$YOSYS" -p "read_verilog -sv $HERE/rtl/kwt_pack_lane.sv; \
                         hierarchy -top $top; proc; opt; synth -top $top; stat" 2>&1 \
                | grep -iE "error|warning|Found and reported|cells|wire bits|memories"
            echo "synth_exit=$?"
        done
    } > "$LOGS/l4_yosys_synth.log" 2>&1
    rc=$?
    printf 'l4_yosys_synth\t%s\t%s\n' "$rc" "$( [ $rc -eq 0 ] && echo PASS || echo FAIL )" >> "$SUMMARY"
else
    printf 'l4_yosys_synth\t127\tSKIP\n' >> "$SUMMARY"
fi

echo "---- summary (logs/summary.tsv) ----"
cat "$SUMMARY"
if [ "$fail" -eq 0 ]; then
    echo "R2_PACK_FORMAT_SUITE_PASS"
else
    echo "R2_PACK_FORMAT_SUITE_FAIL"
fi
exit "$fail"
