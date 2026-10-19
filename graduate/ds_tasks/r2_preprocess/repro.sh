#!/usr/bin/env bash
# r2_preprocess — one-shot reproduction of the offline-path independent verification.
#
# Runs the independent verifier, captures stdout/stderr to logs/verify_preprocess.log,
# records the exit code to logs/exit_codes.txt, and writes a source-hash manifest.
# Read-only over software/ and the main-project archive; writes only under
# graduate/ds_tasks/r2_preprocess/.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
LOGS="$HERE/logs"
PY="${R2_PYTHON:-/home/yang/miniconda3/bin/python}"

mkdir -p "$LOGS"
cd "$ROOT"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
: > "$LOGS/exit_codes.txt"

echo "== r2_preprocess repro $(date -Iseconds) =="
echo "root=$ROOT python=$PY"

"$PY" -V
echo "python_V_exit=$?" >> "$LOGS/exit_codes.txt"

"$PY" graduate/ds_tasks/r2_preprocess/verify_preprocess.py > "$LOGS/verify_preprocess.log" 2>&1
code=$?
echo "verify_preprocess.py=$code" >> "$LOGS/exit_codes.txt"
echo "--- tail(verify_preprocess.log) ---"
tail -n 8 "$LOGS/verify_preprocess.log"

sha256sum \
  graduate/ds_tasks/r2_preprocess/verify_preprocess.py \
  graduate/ds_tasks/r2_preprocess/repro.sh \
  graduate/ds_tasks/r2_preprocess/results.json \
  > "$LOGS/sources.sha256" 2>/dev/null
echo "sources_sha256=$LOGS/sources.sha256"

echo "R2_PREPROCESS_EXIT=$code"
exit "$code"
