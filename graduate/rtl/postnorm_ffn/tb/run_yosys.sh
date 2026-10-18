#!/usr/bin/env bash
# Synthesizability check for the PostNorm FFN/GELU/residual/LayerNorm fragment.
# Yosys `check -assert` plus a generic elaboration.  This is a *structural*
# check (does it read as synthesizable RTL), NOT a timing/area/power sign-off:
# the target PDK, SRAM macros and SDC are not available in this repository.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
build="${SYNTH_OUT:-$here/build/yosys}"
mkdir -p "$build"

# Prefer the OSS CAD Suite Yosys (0.6x); the distro Yosys 0.9 cannot parse
# functions in packages.
YOSYS="${YOSYS:-$(command -v yosys)}"
if [ -x /home/yang/yosys-sta/oss-cad-suite/bin/yosys ]; then
    YOSYS=/home/yang/yosys-sta/oss-cad-suite/bin/yosys
fi
echo "using $YOSYS ($($YOSYS -V 2>&1 | head -1))"

"$YOSYS" -p "
  read_verilog -sv -I$here/rtl \
    $here/rtl/kwt_postnorm_pkg.sv \
    $here/rtl/kwt_residual_add.sv \
    $here/rtl/kwt_layernorm.sv \
    $here/rtl/kwt_gelu.sv \
    $here/rtl/kwt_ffn2.sv \
    $here/rtl/kwt_postnorm_ffn.sv
  hierarchy -check -top kwt_postnorm_ffn
  proc
  opt
  check -assert
  stat
" 2>&1 | tee "$build/yosys_check.log"

if grep -qE "^ERROR:" "$build/yosys_check.log"; then
    echo "YOSYS_FAIL"
    exit 1
fi
echo "YOSYS_PASS"
