#!/usr/bin/env bash
# Regenerate the integer model + vectors, build the RTL with Verilator, run the
# bit-exact / FP32 comparison. Prints an explicit exit code summary.
set -u
HERE="$(cd "$(dirname "$0")/.." && pwd)"      # .../graduate/rtl/attention
cd "$HERE"

echo "== [1/3] integer model + vectors =="
python3 model/attn_int_model.py || { echo "MODEL EXIT=$?"; exit 1; }

echo "== [2/3] verilator build =="
verilator --cc --exe --build -j 4 -O2 -Wno-fatal --top-module attn_single_head \
    -Irtl rtl/attn_single_head.sv tb/tb_attn.cpp --Mdir obj_dir -o tb_attn
rc=$?
echo "VERILATOR_BUILD_EXIT=$rc"
[ $rc -ne 0 ] && exit $rc

echo "== [3/3] run =="
./obj_dir/tb_attn vectors
rc=$?
echo "TB_EXIT=$rc"
exit $rc
