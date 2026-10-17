#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$repo_dir"
parallel="${E203_CHUNK_PARALLEL:-4}"
if [[ ! "$parallel" =~ ^[1-9][0-9]*$ ]]; then
    echo 'E203_CHUNK_PARALLEL must be a positive integer' >&2
    exit 2
fi

seq 0 16 192 | xargs -P "$parallel" -I{} \
    bash hardware/rtl/control/tb/run_e203.sh --tiles {} 16
echo 'PASS E203 all 13 independent 16-tile chunks'
