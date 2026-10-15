#!/usr/bin/env bash
set -euo pipefail

# 从任意目录运行，编译产物放入临时目录并在退出时清理。
tb_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
rtl_dir="$(dirname -- "$tb_dir")"
build_dir="$(mktemp -d)"
trap 'rm -rf -- "$build_dir"' EXIT

# 四种配置共用同一份阵列和测试台。
run_case() {
    local name="$1"
    local data_w="$2"
    local acc_w="$3"
    local format="$4"

    if ! iverilog -g2012 -s tb_kwt_pulse_array \
        -Ptb_kwt_pulse_array.DATA_W="$data_w" \
        -Ptb_kwt_pulse_array.ACC_W="$acc_w" \
        -Ptb_kwt_pulse_array.FORMAT="$format" \
        -o "$build_dir/$name" \
        "$rtl_dir/kwt_pe.sv" \
        "$rtl_dir/kwt_pulse_array.sv" \
        "$tb_dir/tb_kwt_pulse_array.sv" \
        2>"$build_dir/$name.compile.log"; then
        cat "$build_dir/$name.compile.log" >&2
        return 1
    fi

    # Icarus 对 always_comb 的位选敏感表会为每个 PE 重复提示。
    if [[ -s "$build_dir/$name.compile.log" ]]; then
        printf 'NOTE %s: Icarus 编译提示 %s 行（首条：' \
            "$name" "$(wc -l < "$build_dir/$name.compile.log")"
        head -n 1 "$build_dir/$name.compile.log"
        printf ')\n'
    fi

    vvp "$build_dir/$name"
}

run_case int8 8 32 0
run_case int16 16 32 0
run_case int32 32 64 0
run_case fp32 32 32 1
