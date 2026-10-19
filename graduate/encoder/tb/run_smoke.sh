#!/usr/bin/env bash
# One-click 12-layer scheduling smoke test for the encoder control path.
#
#   bash graduate/encoder/tb/run_smoke.sh
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
root="$(cd "$here/../.." && pwd)"
rtl="$root/graduate/rtl/encoder"
vec="$here/vectors"
build="${SMOKE_BUILD_DIR:-$here/tb/build_smoke}"
mkdir -p "$build"

# vectors must already exist (run.sh generates them); regenerate cheaply for
# standalone use.
python3 "$vec/gen_vectors.py" >/dev/null

verilator --binary --timing -j 4 -Wno-fatal -I"$rtl/third_party" \
    --top-module tb_layers_smoke -o sim_smoke --Mdir "$build/obj_vl" \
    "$rtl/third_party/kwt_postnorm_pkg.sv" \
    "$rtl/third_party/kwt_residual_add.sv" \
    "$rtl/third_party/kwt_layernorm.sv" \
    "$rtl/third_party/kwt_gelu.sv" \
    "$rtl/third_party/kwt_ffn2.sv" \
    "$rtl/third_party/kwt_postnorm_ffn.sv" \
    "$rtl/third_party/attn_single_head.sv" \
    "$rtl/kwt_ffn1_row.sv" \
    "$rtl/kwt_encoder_layer.sv" \
    "$rtl/kwt_encoder_layers_ctrl.sv" \
    "$rtl/kwt_encoder_top.sv" \
    "$here/tb/tb_layers_smoke.sv" > "$build/build.log" 2>&1

"$build/obj_vl/sim_smoke" "+VEC=$vec" | tee "$build/smoke.log"
grep -q "SMOKE_PASS" "$build/smoke.log"
