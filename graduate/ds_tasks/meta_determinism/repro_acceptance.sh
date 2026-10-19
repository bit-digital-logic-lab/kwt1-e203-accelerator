#!/usr/bin/env bash
# Independent re-run of the grad-encoder acceptance contract from a *clean* Git
# tree, i.e. exactly the way the orchestrator's verify_worker.py does it:
#
#   git archive HEAD  ->  throwaway export  ->  bash <each declared command>
#   ->  sha256 every declared artifact  ->  compare with ACCEPTANCE.json
#
# Nothing in the working tree is touched: the export is a copy of the committed
# HEAD only, so untracked/generated files from the developer checkout cannot
# leak in.
#
# Usage:
#   bash graduate/ds_tasks/meta_determinism/repro_acceptance.sh [--python <interp>] [--out <dir>]
#
#   --python <interp>  run the export with a specific python3 (default: whatever
#                      `python3` resolves to). Used to prove the meta.json digest
#                      is now stable across interpreters/BLAS builds.
#   --out <dir>        where to write logs + summary.json (default: ./repro-out).
#
# Exit code: 0 only if every command exits with its expected code AND every
# declared artifact sha256 matches.
set -uo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../../.." && pwd)"     # repository root (worktree)
python=""
out="$here/repro-out"
while [ $# -gt 0 ]; do
    case "$1" in
        --python) python="$2"; shift 2 ;;
        --out) out="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

mkdir -p "$out"
shadow="$(mktemp -d "${out%/}/clean-tree-XXXXXX")"
status=0

echo "# export HEAD ($(git -C "$root" rev-parse HEAD)) -> $shadow" | tee "$out/repro.log"
git -C "$root" archive --format=tar HEAD | tar -x -C "$shadow" || { echo "archive failed" >&2; exit 2; }

# Optionally force a specific interpreter (run.sh calls `python3`).
if [ -n "$python" ]; then
    bindir="$shadow/.interp"
    mkdir -p "$bindir"
    ln -sf "$(command -v "$python")" "$bindir/python3"
    export PATH="$bindir:$PATH"
fi
echo "# python3 -> $(command -v python3)" | tee -a "$out/repro.log"

commands=(
    "bash graduate/encoder/tb/run.sh"
    "bash graduate/encoder/tb/run_smoke.sh"
)
exit_codes=""
i=0
for cmd in "${commands[@]}"; do
    i=$((i + 1))
    log="$out/cmd${i}.log"
    {
        echo ""
        echo "\$ (cd $shadow && $cmd)"
    } | tee -a "$out/repro.log"
    ( cd "$shadow" && eval "$cmd" ) >"$log" 2>&1
    code=$?
    exit_codes="$exit_codes $code"
    echo "# exit=$code (log: $log)" | tee -a "$out/repro.log"
    [ "$code" -eq 0 ] || status=1
done

echo "" | tee -a "$out/repro.log"
echo "# artifact digests (computed in the clean export)" | tee -a "$out/repro.log"

art_args=()
while read -r p declared; do
    [ -n "$p" ] || continue
    actual="$(sha256sum "$shadow/$p" 2>/dev/null | cut -d' ' -f1)"
    [ -n "$actual" ] || actual="MISSING"
    if [ "$actual" = "$declared" ]; then
        echo "OK   $p" | tee -a "$out/repro.log"
    else
        echo "FAIL $p declared=$declared actual=$actual" | tee -a "$out/repro.log"
        status=1
    fi
    art_args+=(--artifact "$p" "$declared" "$actual")
done < <(python3 - "$shadow/ACCEPTANCE.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
for a in c["artifacts"]:
    print(a["path"], a.get("sha256", ""))
PY
)

python3 "$here/summarize.py" --out "$out/summary.json" \
    --head "$(git -C "$root" rev-parse HEAD)" \
    --exit-codes $exit_codes "${art_args[@]}"

echo "" | tee -a "$out/repro.log"
echo "# exit codes:$exit_codes" | tee -a "$out/repro.log"
if [ "$status" -eq 0 ]; then
    echo "# ACCEPTANCE_REPRO_OK" | tee -a "$out/repro.log"
else
    echo "# ACCEPTANCE_REPRO_FAIL" | tee -a "$out/repro.log"
fi
exit "$status"
