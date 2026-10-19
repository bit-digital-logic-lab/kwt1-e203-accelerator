#!/usr/bin/env bash
# One-command reproducible run of the ONE-MAC-ARRAY / TWO-CACC-PAGE experiment.
#
#   bash hardware/experiments/single_mac_dual_cacc/run.sh
#
# Builds a Verilator microtest for the new RTL primitive and Verilator task
# binaries for the new engine, the old one-core baseline and the old two-core
# dual harness; runs tiny / ragged / full 99x64x256 FFN1 shapes against an
# independent NumPy INT32 golden; counts elaborated $mul cells with yosys; and
# writes results.json + logs/. Simulated task model, NOT E203, NOT 150 MHz STA.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
cd "$repo"

if ! command -v verilator >/dev/null 2>&1; then
    echo "FAIL verilator not found" >&2
    exit 1
fi
if ! command -v yosys >/dev/null 2>&1; then
    echo "FAIL yosys not found" >&2
    exit 1
fi

python3 "$here/run_single_mac_dual_cacc.py" --rows 16 --out "$here"
