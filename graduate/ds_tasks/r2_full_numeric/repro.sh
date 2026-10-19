#!/usr/bin/env bash
# r2_full_numeric -- reproduce the round-1 scale-contract conflicts and the
# frozen 12-layer numeric check.
#
# Run from anywhere; logs and exit codes land in graduate/ds_tasks/r2_full_numeric/logs/.
#
# Command semantics (not all phases are supposed to exit 0):
#   phase 1 conflict witness : expected 2  -- a reproduced *conflict*, not a
#                                            failure of the tooling
#   phase 2 RTL config check : expected 2  -- assembled RTL constants != frozen
#                                            layer-0 pack (static comparison)
#   phase 3 full numeric     : expected 0  -- formal gate: per-layer pooled
#                                            NRMSE < 3% on the 240 npz
#                                            validation samples only.
#                                            Real WAVs: 3 validation reported
#                                            separately (duplicates of npz rows,
#                                            +0 unique) and 3 training used as
#                                            a reference, never as validation.
#                                            The per-sample tail (layer 11) is a
#                                            diagnostic, not the gate.
#
# Exit codes of this script:
#   0  all three commands returned their expected codes
#   1  at least one command returned an unexpected code (or failed to run)
#   3  the formal numeric gate itself failed (propagated, never masked as 0)
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS="$HERE/logs"
mkdir -p "$LOGS"

export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4

run_phase() {   # run_phase <logname> <command...>
    local logname="$1"; shift
    "$@" > "$LOGS/$logname.console.log" 2>&1
    echo $?
}

conflict_rc=$(run_phase run_conflict_witness python3 "$HERE/model/run_conflict_witness.py")
rtl_rc=$(run_phase check_rtl_config python3 "$HERE/model/check_rtl_config.py")
numeric_rc=$(run_phase run_full_numeric python3 "$HERE/model/run_full_numeric.py")

numeric_state="$(python3 - "$HERE/results.json" <<'PY' 2>/dev/null || true
import json, sys
try:
    r = json.load(open(sys.argv[1]))
except Exception:
    print("unreadable"); raise SystemExit
print(r.get("state", "unknown"))
PY
)"

{
    echo "# r2_full_numeric -- real command exit codes ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
    echo "# semantics: phase 1/2 exit 2 = reproduced conflict (expected), phase 3"
    echo "# exit 0 = formal pooled gate pass on 240 npz validation samples only."
    echo "[1] python3 model/run_conflict_witness.py"
    echo "    exit: $conflict_rc   (expected 2: round-1 layer0 x-scale contract conflict)"
    echo "    log : logs/run_conflict_witness.console.log"
    echo "[2] python3 model/check_rtl_config.py"
    echo "    exit: $rtl_rc   (expected 2: assembled RTL headers != frozen layer-0 pack)"
    echo "    log : logs/check_rtl_config.console.log"
    echo "[3] python3 model/run_full_numeric.py"
    echo "    exit: $numeric_rc   (expected 0: 12/12 layers pooled NRMSE < 3% on 240 npz validation)"
    echo "    state: $numeric_state"
    echo "    log : logs/run_full_numeric.console.log"
    echo "    metric: formal = per-layer pooled over the 240 npz validation samples;"
    echo "            real validation 3 reported separately (0 unique added);"
    echo "            real training 3 reference only; per-sample tail is diagnostic."
} > "$LOGS/exit_codes.txt"

cat "$LOGS/exit_codes.txt"

if [[ "$conflict_rc" -eq 2 && "$rtl_rc" -eq 2 && "$numeric_rc" -eq 0 ]]; then
    echo "REPRO_OK conflict_exit=$conflict_rc rtl_exit=$rtl_rc numeric_exit=$numeric_rc state=$numeric_state"
    exit 0
fi
echo "REPRO_UNEXPECTED conflict_exit=$conflict_rc rtl_exit=$rtl_rc numeric_exit=$numeric_rc state=$numeric_state"
if [[ "$numeric_rc" -eq 3 ]]; then
    exit 3
fi
exit 1
