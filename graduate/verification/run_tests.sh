#!/usr/bin/env bash
# Run the graduate verification unit + fault-injection tests, then the staged
# runner on the current worktree.  Exit code is non-zero if any test or stage
# FAILs (SKIPs are allowed unless --strict is passed through).
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
verification="$here"
root="$(dirname -- "$(dirname -- "$verification")")"

python3 -m unittest discover -s "$verification/faults" -t "$verification/faults" -v
python3 "$verification/run_stages.py" --design-root "$root" "$@"
