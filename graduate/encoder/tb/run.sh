#!/usr/bin/env bash
# One-click reproduction of the single-layer PostNorm encoder assembly:
#   generate vectors -> build/run Verilator testbench -> bit-exact + FP32 verify.
#
#   bash graduate/encoder/tb/run.sh
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"          # graduate/encoder
root="$(cd "$here/../.." && pwd)"                 # repository root
rtl="$root/graduate/rtl/encoder"
vec="$here/vectors"
build="${BUILD_DIR:-$here/tb/build}"
mkdir -p "$build"

python3 "$vec/gen_vectors.py"

verilator --binary --timing -j 4 -Wno-fatal -I"$rtl/third_party" \
    --top-module tb_encoder -o sim_enc --Mdir "$build/obj_vl" \
    "$rtl/third_party/kwt_postnorm_pkg.sv" \
    "$rtl/third_party/kwt_residual_add.sv" \
    "$rtl/third_party/kwt_layernorm.sv" \
    "$rtl/third_party/kwt_gelu.sv" \
    "$rtl/third_party/kwt_ffn2.sv" \
    "$rtl/third_party/kwt_postnorm_ffn.sv" \
    "$rtl/third_party/attn_single_head.sv" \
    "$rtl/kwt_ffn1_row.sv" \
    "$rtl/kwt_encoder_layer.sv" \
    "$here/tb/tb_encoder.sv" > "$build/build.log" 2>&1

"$build/obj_vl/sim_enc" "+VEC=$vec" "+OUT=$build" | tee "$build/sim.log"
grep -q "TB_PASS" "$build/sim.log"

python3 "$here/model/verify_encoder.py" --build "$build" --vectors "$vec" \
    | tee "$build/verify.log"
grep -q "VERIFY_PASS" "$build/verify.log"
