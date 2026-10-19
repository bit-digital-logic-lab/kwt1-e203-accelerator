#!/usr/bin/env bash
# Reproduce the encoder_12numeric task end to end (software reference + RTL chain).
#
#   bash graduate/ds_tasks/encoder_12numeric/run_all.sh
#
# Writes logs/ and the results JSON files next to this script.  Every command's
# exit code is echoed.  Nothing is written outside this task directory.
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../../.." && pwd)"
py="${PYTHON:-python3}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
cd "$root"

echo "== stage 1: L1 independent 12-layer integer reference (train cal / val eval) =="
"$py" "$here/model/run_l1.py"; echo "EXIT run_l1=$?"

echo "== stage 2: freeze integer tensor pack (deterministic re-calibration check) =="
"$py" "$here/model/freeze_int.py"; echo "EXIT freeze_int=$?"

echo "== stage 3: L2 RTL bit-exact 12-layer chain (per-layer elaborated RTL) =="
"$py" "$here/model/run_rtl_chain.py" --samples all; echo "EXIT run_rtl_chain=$?"
