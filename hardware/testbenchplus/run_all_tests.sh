#!/usr/bin/env bash
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root="$(dirname -- "$here")"
mode=default
if [[ "${1:-}" == "--vectors-only" ]]; then
    mode=vectors
    shift
elif [[ "${1:-}" == "--full" ]]; then
    mode=full
    shift
fi

python3 "$here/common/run_vectors.py" "$@"
if [[ "$mode" == vectors ]]; then
    exit 0
fi

for suite in \
    rtl/pulse_array/tb/run.sh \
    rtl/pulse_array/tb/run_engine.sh \
    rtl/dma/tb/run.sh \
    rtl/control/tb/run.sh \
    rtl/control/tb/run_nice.sh; do
    printf '\n== %s ==\n' "$suite"
    bash "$root/$suite"
done

if [[ "$mode" == full ]]; then
    printf '\n== E203 NICE ==\n'
    bash "$root/rtl/control/tb/run_e203.sh"
    printf '\n== E203 full layer ==\n'
    bash "$root/rtl/control/tb/run_e203.sh" --full-layer
fi
printf '\nPASS testbenchplus regression (%s)\n' "$mode"
