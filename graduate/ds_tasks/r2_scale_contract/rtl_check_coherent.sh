#!/usr/bin/env bash
# L2 evidence: attention RTL bit-exact check under the COHERENT training-frozen
# calibration (r2_scale_contract).
#
# Reuses the round-1 attention RTL (graduate/rtl/encoder/third_party/
# attn_single_head.sv) and the round-1 Verilator testbench
# (ds-grad-attn_scale/graduate/ds_tasks/attn_scale/tb/tb_attn.cpp), both read-only.
# The RTL params header + vectors are regenerated for the coherent calibration by
# gen_coherent_vectors.py and written only under the build dir.  Scope: the
# ATTENTION stage only; the full encoder chain is NOT re-verified here.
#
#   bash graduate/ds_tasks/r2_scale_contract/rtl_check_coherent.sh
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../../.." && pwd)"
attn_wt="/home/yang/kwt-worktrees/ds-grad-attn_scale"
build="${BUILD_DIR:-/tmp/opencode/r2_scale_contract/rtl_check}"
logs="$here/logs"
mkdir -p "$build/rtl" "$build/vectors" "$logs"

rtl="$root/graduate/rtl/encoder/third_party/attn_single_head.sv"
tb="$attn_wt/graduate/ds_tasks/attn_scale/tb/tb_attn.cpp"

echo "== emit coherent vectors =="
python3 "$here/gen_coherent_vectors.py" --build "$build" > "$logs/rtl_coh_gen.log" 2>&1
echo "GEN_EXIT=$?"
cp "$rtl" "$build/rtl/"
cp "$tb" "$build/tb_attn.cpp"
sha256sum "$rtl" "$tb" | tee "$logs/rtl_coh_sources.sha256"

echo "== verilator build =="
verilator --cc --exe --build -j 4 -O2 -Wno-fatal --top-module attn_single_head \
    -I"$build/rtl" "$build/rtl/attn_single_head.sv" "$build/tb_attn.cpp" \
    --Mdir "$build/obj_dir" -o tb_attn > "$logs/rtl_coh_build.log" 2>&1
echo "RTL_BUILD_EXIT=$?"

rc=0
for S in 0 1 2; do
    if "$build/obj_dir/tb_attn" "$build/vectors/sample_$S" > "$logs/rtl_coh_val_sample$S.log" 2>&1; then
        st=0
    else
        st=1
        rc=1
    fi
    summary=$(grep -E 'bit-exact total|FP32 NRMSE|RESULT' "$logs/rtl_coh_val_sample$S.log" | tr '\n' ' ')
    echo "coherent sample$S exit=$st  $summary"
done
echo "RTL_CHECK_EXIT=$rc"
exit $rc
