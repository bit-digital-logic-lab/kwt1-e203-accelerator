#!/usr/bin/env bash
# Shared EXIT-trap helper for long-job finalizers.
#
# Guarantees a terminal check-in state is recorded even when the finalizer
# exits abnormally, so the 15-minute scan does not keep reporting a task as
# "service_stopped" after the work actually finished (or genuinely died).
#
# Usage inside a finalizer:
#   source "$repo/.opencode/skills/quarter-hour-check/scripts/finalize_hook.sh"
#   check_finalize_init "$check_id" "$check_py"
#   ...
#   check_finalize_state complete "$report" "12 层完成"   # only on success
#
# If the script ends without calling check_finalize_state, the trap records
# "failed" with an abnormal-exit note.  The exit status of the finalizer is
# preserved, and a registration failure never changes it.
#
# State root follows check.py: --state-dir / QUARTER_HOUR_STATE_DIR.

check_finalize_init() {
    CHECK_FINALIZE_ID="${1:?check id required}"
    CHECK_FINALIZE_PY="${2:?path to check.py required}"
    CHECK_FINALIZE_STATE="failed"
    CHECK_FINALIZE_RESULT=""
    CHECK_FINALIZE_NOTE="finalizer 异常结束，未登记正常完成"
    trap check_finalize_run EXIT
    # Turn signals into a normal exit so the EXIT trap always runs.
    trap 'exit 129' HUP
    trap 'exit 130' INT
    trap 'exit 143' TERM
}

check_finalize_state() {
    CHECK_FINALIZE_STATE="${1:-failed}"
    CHECK_FINALIZE_RESULT="${2:-}"
    CHECK_FINALIZE_NOTE="${3:-}"
}

check_finalize_run() {
    local rc=$?
    local -a argv=(update --id "$CHECK_FINALIZE_ID" --state "$CHECK_FINALIZE_STATE")
    if [[ -n "$CHECK_FINALIZE_RESULT" ]]; then
        argv+=(--result "$CHECK_FINALIZE_RESULT")
    fi
    argv+=(--note "$CHECK_FINALIZE_NOTE")
    if [[ -n "${QUARTER_HOUR_STATE_DIR:-}" ]]; then
        argv+=(--state-dir "$QUARTER_HOUR_STATE_DIR")
    fi
    python3 "$CHECK_FINALIZE_PY" "${argv[@]}" >/dev/null 2>&1 || true
    exit "$rc"
}
