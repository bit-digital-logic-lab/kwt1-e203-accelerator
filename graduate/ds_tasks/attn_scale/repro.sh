#!/usr/bin/env bash
# One-click reproduction of the attn_scale task (L1 software + L2 RTL bit-exact).
#
#   bash graduate/ds_tasks/attn_scale/repro.sh
#
# Steps:
#   1. freeze calibration from the training split (no gradients)
#   2. multi-sample validation evaluation + leak audit          (L1)
#   3. attention RTL bit-exact check per validation sample      (L2)
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../../.." && pwd)"
logs="$here/logs"
mkdir -p "$logs"
status=0

echo "=== [1] freeze calibration from training split ==="
python3 "$root/graduate/encoder/model/attn_int_model.py" --recalibrate --no-write-vectors \
    > "$logs/recalibrate.log" 2>&1
rc=$?
echo "RECALIBRATE_EXIT=$rc"
[ $rc -ne 0 ] && status=1

echo "=== [2] L1 validation evaluation + leak audit ==="
python3 "$here/run_attn_scale.py" > "$logs/run.log" 2>&1
rc=$?
echo "RUN_ATTN_SCALE_EXIT=$rc"
[ $rc -ne 0 ] && status=1

echo "=== [3] L2 attention RTL bit-exact (train-frozen constants) ==="
bash "$here/rtl_check.sh" > "$logs/rtl_check.log" 2>&1
rc=$?
echo "RTL_CHECK_EXIT=$rc"
[ $rc -ne 0 ] && status=1
cat "$logs/rtl_check.log"

echo "REPRO_EXIT=$status"
exit $status
