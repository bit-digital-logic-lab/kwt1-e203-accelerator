#!/usr/bin/env bash
# Negative control / root-cause demo.
#
# Runs the PRE-FIX gen_vectors.py (blob taken from Git, not the working tree)
# in a clean export of HEAD with two interpreters that are both present on this
# machine, and shows that meta.json - and therefore its sha256 - differed:
#
#   /usr/bin/python3   numpy 1.21.5, netlib reference BLAS   (what the worker used)
#   project .venv      numpy 2.1.3,  scipy-openblas           (what verify_worker used)
#
# With the fix in place the same two interpreters agree byte for byte; run
# repro_acceptance.sh to see that.
#
# Usage:
#   bash graduate/ds_tasks/meta_determinism/demo_baseline_nondeterminism.sh [--out <dir>]
set -uo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../../.." && pwd)"
out="$here/evidence/baseline"
while [ $# -gt 0 ]; do
    case "$1" in
        --out) out="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
mkdir -p "$out"

# Pre-fix gen_vectors.py straight out of Git history (parent of the fix commit).
git -C "$root" show HEAD^:graduate/encoder/vectors/gen_vectors.py > "$out/gen_vectors.old.py"

tree="$(mktemp -d "${out%/}/clean-tree-XXXXXX")"
git -C "$root" archive --format=tar HEAD | tar -x -C "$tree"
cp "$out/gen_vectors.old.py" "$tree/graduate/encoder/vectors/gen_vectors.py"

echo "# pre-fix code, interpreter A: $(python3 -c 'import sys,numpy;print(sys.version.split()[0], numpy.__version__)' 2>&1)"
python3 "$tree/graduate/encoder/vectors/gen_vectors.py" >/dev/null 2>&1
cp "$tree/graduate/encoder/vectors/meta.json" "$out/meta.sys.json"
A="$(sha256sum "$out/meta.sys.json" | cut -d' ' -f1)"

venv_py="${VENV_PY:-/home/yang/miniconda3/bin/python3}"
if [ -x "$venv_py" ]; then
    echo "# pre-fix code, interpreter B: $("$venv_py" -c 'import sys,numpy;print(sys.version.split()[0], numpy.__version__)' 2>&1)"
    rm -f "$tree/graduate/encoder/vectors/meta.json" "$tree/graduate/encoder/vectors/params.json"
    "$venv_py" "$tree/graduate/encoder/vectors/gen_vectors.py" >/dev/null 2>&1
    cp "$tree/graduate/encoder/vectors/meta.json" "$out/meta.venv.json"
    B="$(sha256sum "$out/meta.venv.json" | cut -d' ' -f1)"
else
    echo "# interpreter B ($venv_py) missing; using B=A"
    B="$A"; cp "$out/meta.sys.json" "$out/meta.venv.json"
fi

{
    echo "interpreter A ($(python3 -c 'import numpy;print(numpy.__version__)')): $A"
    echo "interpreter B ($("$venv_py" -c 'import numpy;print(numpy.__version__)' 2>/dev/null)): $B"
    echo "--- diff ---"
    diff "$out/meta.sys.json" "$out/meta.venv.json" || true
} | tee "$out/diff.txt"

rm -rf "$tree"
if [ "$A" = "$B" ]; then
    echo "BASELINE_UNEXPECTEDLY_STABLE"
    exit 1
fi
echo "BASELINE_NONDETERMINISM_REPRODUCED"
