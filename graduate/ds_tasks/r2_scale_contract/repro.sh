#!/usr/bin/env bash
# One-shot reproduction for r2_scale_contract.
#
#   bash graduate/ds_tasks/r2_scale_contract/repro.sh
#
# Runs the L1 software cross-check and the L2 attention RTL bit-exact check,
# records every exit code to logs/ and fails if any step fails.  No file outside
# this task directory (and the approved /tmp/opencode build dir) is written.
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
logs="$here/logs"
mkdir -p "$logs" /tmp/opencode/r2_scale_contract
rc=0

echo "[1/2] L1 software cross-check"
python3 "$here/run_cross_check.py" > "$logs/cross_check_console.log" 2>&1
e1=$?
echo "CROSS_CHECK_EXIT=$e1"
[ "$e1" -eq 0 ] || rc=1

echo "[2/2] L2 attention RTL bit-exact (coherent constants)"
bash "$here/rtl_check_coherent.sh" > "$logs/rtl_coh_console.log" 2>&1
e2=$?
echo "RTL_CHECK_SCRIPT_EXIT=$e2"
[ "$e2" -eq 0 ] || rc=1

echo "REPRO_EXIT=$rc"
exit $rc
