#!/usr/bin/env bash
# r2_full_numeric -- reproduce both the round-1 scale-contract conflict and the
# frozen 12-layer multi-sample numeric check.
#
# Run from anywhere; logs and exit codes land in graduate/ds_tasks/r2_full_numeric/logs/.
#
# Expected results (2026-10-19, this worktree):
#   phase 1 conflict witness : exit 2  (round-1 attn/postnorm s_x conflict)
#   phase 2 RTL config check : exit 2  (assembled RTL headers != frozen pack)
#   phase 3 full numeric     : exit 0  (pooled local NRMSE < 3% on all 12 layers;
#                                       per-sample tail layer 11 = 3.78% > 3%)
# The script exits 0 only when all phases match those expected codes.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS="$HERE/logs"
mkdir -p "$LOGS"

export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4

python3 "$HERE/model/run_conflict_witness.py" \
    > "$LOGS/run_conflict_witness.console.log" 2>&1
conflict_rc=$?

python3 "$HERE/model/check_rtl_config.py" \
    > "$LOGS/check_rtl_config.console.log" 2>&1
rtl_rc=$?

python3 "$HERE/model/run_full_numeric.py" \
    > "$LOGS/run_full_numeric.console.log" 2>&1
numeric_rc=$?

{
    echo "# r2_full_numeric -- real command exit codes ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
    echo "[1] python3 model/run_conflict_witness.py"
    echo "    exit: $conflict_rc   (expected 2: round-1 layer0 x-scale contract conflict)"
    echo "    log : logs/run_conflict_witness.console.log"
    echo "[2] python3 model/check_rtl_config.py"
    echo "    exit: $rtl_rc   (expected 2: assembled RTL headers != frozen layer-0 pack)"
    echo "    log : logs/check_rtl_config.console.log"
    echo "[3] python3 model/run_full_numeric.py"
    echo "    exit: $numeric_rc   (expected 0: pooled local NRMSE < 3% on all 12 layers)"
    echo "    log : logs/run_full_numeric.console.log"
    echo "    note: conservative per-sample tail is 3.78% at layer 11; see results.json"
} > "$LOGS/exit_codes.txt"

cat "$LOGS/exit_codes.txt"

if [[ "$conflict_rc" -eq 2 && "$rtl_rc" -eq 2 && "$numeric_rc" -eq 0 ]]; then
    echo "REPRO_OK conflict_exit=$conflict_rc rtl_exit=$rtl_rc numeric_exit=$numeric_rc"
    exit 0
fi
echo "REPRO_UNEXPECTED conflict_exit=$conflict_rc rtl_exit=$rtl_rc numeric_exit=$numeric_rc"
exit 1
