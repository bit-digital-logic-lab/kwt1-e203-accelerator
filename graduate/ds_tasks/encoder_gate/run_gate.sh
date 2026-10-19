#!/usr/bin/env bash
# encoder_gate acceptance driver.
#
# Proves, with real exit codes and logs, that:
#   01  the full positive run (gen vectors -> Verilator -> verify) passes
#   02  verify_encoder.py still passes with the 3% NRMSE gate armed
#   03  a bit-exact corruption makes it fail for reason=bit_exact
#   04  an over-threshold dequantisation makes it fail for reason=nrmse
#   05  the gate predicate boundary is correct (software unit test)
#   06  gate constants are training-split calibrated; threshold is not fitted
#
# Usage:  bash graduate/ds_tasks/encoder_gate/run_gate.sh
# Logs:   graduate/ds_tasks/encoder_gate/logs/*.log   (exit codes in gate_results.tsv)
# Exit:   0 only if every case behaves as expected.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
MODEL="$ROOT/graduate/encoder/model"
LOGS="$HERE/logs"
RES="$HERE/gate_results.tsv"
mkdir -p "$LOGS"
: > "$RES"

PASS=0
FAIL=0

# check <name> <expect 0|nonzero> <rc> <log> <must-grep|-> <must-not-grep|->
check() {
  local name="$1" expect="$2" rc="$3" log="$4" must="$5" mustnot="$6"
  local verdict="OK" why=""
  if [ "$expect" = "0" ]; then
    [ "$rc" -eq 0 ] || { verdict="BAD"; why="exit=$rc expected 0"; }
  else
    [ "$rc" -ne 0 ] || { verdict="BAD"; why="exit=0 expected non-zero"; }
  fi
  if [ "$verdict" = "OK" ] && [ "$must" != "-" ] && ! grep -q -- "$must" "$log"; then
    verdict="BAD"; why="log missing '$must'"
  fi
  if [ "$verdict" = "OK" ] && [ "$mustnot" != "-" ] && grep -q -- "$mustnot" "$log"; then
    verdict="BAD"; why="log unexpectedly contains '$mustnot'"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$expect" "$rc" "$log" "$verdict" "$why" >> "$RES"
  if [ "$verdict" = "OK" ]; then
    PASS=$((PASS + 1)); echo "OK   $name (exit=$rc)"
  else
    FAIL=$((FAIL + 1)); echo "BAD  $name (exit=$rc) -- $why"
  fi
}

echo "== 01 positive: full run (gen vectors -> Verilator -> verify) =="
bash "$ROOT/graduate/encoder/tb/run.sh" > "$LOGS/01_run_all.log" 2>&1; rc=$?
check 01_run_all 0 "$rc" "$LOGS/01_run_all.log" "VERIFY_PASS" "VERIFY_FAIL"

echo "== 02 positive: verify with the NRMSE gate armed =="
python3 "$MODEL/verify_encoder.py" > "$LOGS/02_verify_positive.log" 2>&1; rc=$?
check 02_verify_positive 0 "$rc" "$LOGS/02_verify_positive.log" "VERIFY_PASS" "VERIFY_FAIL"

echo "== 02b negative: clean data against a too-tight threshold (gate is data-driven) =="
python3 "$MODEL/verify_encoder.py" --nrmse-threshold 0.001 \
    > "$LOGS/02b_clean_tight_threshold.log" 2>&1; rc=$?
check 02b_clean_tight_threshold nonzero "$rc" "$LOGS/02b_clean_tight_threshold.log" \
    "reason=nrmse" "VERIFY_PASS"

echo "== 03 negative: bit-exact corruption (y[0,0] += 1) =="
python3 "$MODEL/verify_encoder.py" --negative-inject bitexact \
    > "$LOGS/03_neg_bitexact.log" 2>&1; rc=$?
check 03_neg_bitexact nonzero "$rc" "$LOGS/03_neg_bitexact.log" "reason=bit_exact" "VERIFY_PASS"

echo "== 04 negative: over-threshold dequant (y dequant x1.10) =="
python3 "$MODEL/verify_encoder.py" --negative-inject nrmse \
    > "$LOGS/04_neg_nrmse.log" 2>&1; rc=$?
check 04_neg_nrmse nonzero "$rc" "$LOGS/04_neg_nrmse.log" "reason=nrmse" "VERIFY_PASS"

echo "== 05 gate boundary self-check (software) =="
python3 "$HERE/gate_selfcheck.py" > "$LOGS/05_gate_selfcheck.log" 2>&1; rc=$?
check 05_gate_selfcheck 0 "$rc" "$LOGS/05_gate_selfcheck.log" "SELFCHECK_PASS" "-"

echo "== 06 training-split calibration / threshold provenance =="
python3 "$HERE/split_calibration.py" > "$LOGS/06_split_calibration.log" 2>&1; rc=$?
check 06_split_calibration 0 "$rc" "$LOGS/06_split_calibration.log" "SPLIT_OK" "-"

python3 "$HERE/make_summary.py" "$RES" "$LOGS/02_verify_positive.log" "$HERE/summary.json" \
    > "$LOGS/07_summary.log" 2>&1; rc=$?
echo "     make_summary exit=$rc"

echo
echo "PASS=$PASS FAIL=$FAIL"
if [ "$FAIL" -ne 0 ]; then
  echo "GATE_DRIVER_FAIL"
  exit 1
fi
echo "GATE_DRIVER_PASS"
