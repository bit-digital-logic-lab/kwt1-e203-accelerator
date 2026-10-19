#!/usr/bin/env bash
# p027_precision -- reproduce the bounded precision search, freeze the selected
# scheme, then rerun the frozen pack with NO calibration.
#
#   bash graduate/ds_tasks/p027_precision/repro.sh
#
# Exit codes:
#   0  search completed; frozen rerun exit code is 0 (target met) or 3 (target
#      not met -- propagated, never masked)
#   1  unexpected failure
#   3  target not met in the independent frozen rerun
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
LOGS="$HERE/logs"
mkdir -p "$LOGS"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
cd "$ROOT"

echo "== stage 1: precision search on the 240 training / 240 validation split =="
python3 "$HERE/model/run_search.py" > "$LOGS/run_search.console.log" 2>&1
rc_search=$?
echo "EXIT run_search=$rc_search  (log: logs/run_search.console.log)"

selected="$(python3 - "$HERE/results.json" <<'PY' 2>/dev/null || true
import json, sys
try:
    print(json.load(open(sys.argv[1])).get("selected") or "")
except Exception:
    print("")
PY
)"
echo "SELECTED candidate: ${selected:-<none>}"

rc_freeze=99
rc_frozen=99
if [[ -n "$selected" ]]; then
    echo "== stage 2: freeze the selected scheme (training-only calibration) =="
    python3 "$HERE/model/freeze_config.py" --name "$selected" \
        > "$LOGS/freeze_config.console.log" 2>&1
    rc_freeze=$?
    echo "EXIT freeze_config=$rc_freeze"

    echo "== stage 3: independent frozen rerun (no calibration, no training split) =="
    python3 "$HERE/model/run_frozen.py" --pack "$HERE/frozen" \
        --out "$HERE/frozen_results.json" > "$LOGS/run_frozen.console.log" 2>&1
    rc_frozen=$?
    echo "EXIT run_frozen=$rc_frozen  (log: logs/run_frozen.console.log)"
fi

{
    echo "# p027_precision -- real command exit codes ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
    echo "[1] python3 model/run_search.py"
    echo "    exit: $rc_search   (0 = search completed)"
    echo "    log : logs/run_search.console.log"
    echo "[2] python3 model/freeze_config.py --name $selected"
    echo "    exit: $rc_freeze   (0 = frozen pack written)"
    echo "    log : logs/freeze_config.console.log"
    echo "[3] python3 model/run_frozen.py --pack frozen"
    echo "    exit: $rc_frozen   (0 = target met in the frozen rerun; 3 = target not met)"
    echo "    log : logs/run_frozen.console.log"
} > "$LOGS/exit_codes.txt"
cat "$LOGS/exit_codes.txt"

if [[ "$rc_search" -ne 0 ]]; then
    echo "REPRO_FAIL search_exit=$rc_search"
    exit 1
fi
if [[ -n "$selected" && "$rc_freeze" -ne 0 ]]; then
    echo "REPRO_FAIL freeze_exit=$rc_freeze"
    exit 1
fi
if [[ "$rc_frozen" -eq 3 ]]; then
    echo "REPRO_TARGET_FAIL"
    exit 3
fi
if [[ "$rc_frozen" -ne 0 ]]; then
    echo "REPRO_FAIL frozen_exit=$rc_frozen"
    exit 1
fi
echo "REPRO_OK selected=$selected"
