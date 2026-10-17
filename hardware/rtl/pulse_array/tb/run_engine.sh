#!/usr/bin/env bash
set -euo pipefail

# 编译产物存放在临时目录，结束后清理。
tb_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
rtl_dir="$(dirname -- "$tb_dir")"
build_dir="$(mktemp -d)"
trap 'rm -rf -- "$build_dir"' EXIT

# 以完整 8×16 阵列运行单块输入到结果 SRAM 的仿真。
iverilog -g2012 -I "$rtl_dir/../sram" -s tb_kwt_tile_engine -o "$build_dir/engine" \
    "$rtl_dir/kwt_pe.sv" \
    "$rtl_dir/kwt_pulse_array.sv" \
    "$rtl_dir/kwt_postprocess.sv" \
    "$rtl_dir/kwt_tile_engine.sv" \
    "$rtl_dir/../sram/kwt_tile_buffers.sv" \
    "$tb_dir/tb_kwt_tile_engine.sv" \
    2>"$build_dir/compile.log" || { cat "$build_dir/compile.log" >&2; exit 1; }

# Icarus 对 PE 中 always_comb 的位选敏感表会产生重复提示。
if [[ -s "$build_dir/compile.log" ]]; then
    printf 'NOTE: Icarus 编译提示 %s 行（首条：' "$(wc -l < "$build_dir/compile.log")"
    head -n 1 "$build_dir/compile.log"
    printf ')\n'
fi
vvp "$build_dir/engine"
