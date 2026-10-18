#!/usr/bin/env bash
# Reproducible entry point for the extended Verilator FFN1 regression.
# Usage: bash hardware/verification/verilator_extended/run_extended.sh [OUT_DIR]
set -euo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
out="${1:-/tmp/kwt-verilator-extended}"
exec python3 -u "$here/run_extended_verilator.py" --out "$out"
