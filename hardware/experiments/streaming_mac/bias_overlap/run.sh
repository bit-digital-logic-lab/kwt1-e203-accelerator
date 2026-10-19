#!/usr/bin/env bash
# One-command reproducible run of the bias-overlap experiment.
#
#   bash hardware/experiments/streaming_mac/bias_overlap/run.sh
#
# Builds: accepted baseline TB, accepted dual-CACC TB, new bias-overlap task TB,
#         new bias-bank/arbiter unit microtest.
# Runs:   tiny/ragged/twotile/overflow/full, preload sweep, delayed bias,
#         output back-pressure, global bus gaps, plus the cycle model.
# Writes: hardware/experiments/streaming_mac/bias_overlap/results.json, logs/,
#         model/model_results.json.
# Prior streaming_mac artifacts are read-only here and are not modified.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
cd "$repo"

if ! command -v verilator >/dev/null 2>&1; then
    echo "FAIL verilator not found" >&2
    exit 1
fi

python3 "$here/run_bias_overlap.py" --rows 16 --out "$here"
