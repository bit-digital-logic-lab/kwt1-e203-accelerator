#!/usr/bin/env bash
# One-command reproducible run of the streaming-MAC experiment.
#
#   bash hardware/experiments/streaming_mac/run.sh
#
# Builds two Verilator binaries (baseline = existing unmodified staged TB,
# dual = new dual-CACC TB) plus the dual-CACC unit microtest, runs tiny/ragged/
# two-tile/overflow/full FFN1 shapes, checks every INT32 output against an
# independent numpy reference, and writes results.json + logs/ + the 12-layer
# event-schedule model. Simulated task model, NOT E203, NOT 150 MHz STA.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
cd "$repo"

if ! command -v verilator >/dev/null 2>&1; then
    echo "FAIL verilator not found" >&2
    exit 1
fi

python3 "$here/run_streaming_mac.py" --rows 16 --out "$here"
