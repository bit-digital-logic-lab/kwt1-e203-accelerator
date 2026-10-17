#!/usr/bin/env bash
set -euo pipefail
tb_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
rtl_dir="$(dirname -- "$tb_dir")"
upstream="$(cd -- "$tb_dir/../../../../资料/e203_hbirdv2_full" && pwd)"
build_dir="$(mktemp -d)"
trap 'rm -rf -- "$build_dir"' EXIT

tb_top=tb_kwt_real_e203
program_source="$tb_dir/e203_kwt_program.S"
define_args=()
run_args=()
if [[ "${1:-}" == "--full-layer" ]]; then
    tb_top=tb_kwt_full_layer_e203
    program_source="$tb_dir/e203_kwt_full_layer.S"
    define_args=(-DKWT_E203_EXT_MEM_BYTES=524288)
elif [[ "${1:-}" == "--tiles" ]]; then
    tile_start="${2:-}"
    tile_count="${3:-}"
    if [[ ! "$tile_start" =~ ^[0-9]+$ || ! "$tile_count" =~ ^[0-9]+$ ]] ||
       (( 10#$tile_count < 1 || 10#$tile_start + 10#$tile_count > 208 )); then
        echo "Usage: bash rtl/control/tb/run_e203.sh --tiles START COUNT (within 0..207)" >&2
        exit 2
    fi
    tile_start=$((10#$tile_start))
    tile_count=$((10#$tile_count))
    tb_top=tb_kwt_full_layer_e203
    program_source="$tb_dir/e203_kwt_full_layer.S"
    define_args=(-DKWT_E203_EXT_MEM_BYTES=524288
                 "-DKWT_TILE_START=$tile_start" "-DKWT_TILE_COUNT=$tile_count")
    run_args=("+TILE_START=$tile_start" "+TILE_COUNT=$tile_count")
elif [[ $# -ne 0 ]]; then
    echo "Usage: bash rtl/control/tb/run_e203.sh [--full-layer | --tiles START COUNT]" >&2
    exit 2
fi

riscv64-linux-gnu-gcc "${define_args[@]}" -march=rv32imac -mabi=ilp32 -nostdlib -nostartfiles \
    -Wl,-Ttext=0x80000000 -Wl,--no-relax \
    -o "$build_dir/program.elf" "$program_source"
riscv64-linux-gnu-objcopy -j .text -O binary "$build_dir/program.elf" "$build_dir/program.bin"
xxd -p -c 1 "$build_dir/program.bin" > "$build_dir/program.hex"
program_size="$(stat -c %s "$build_dir/program.bin")"
for ((byte_index=program_size; byte_index<64; byte_index++)); do
    printf '00\n' >> "$build_dir/program.hex"
done

mapfile -t core_files < <(find "$upstream/rtl/e203/core" "$upstream/rtl/e203/general" \
    -maxdepth 1 -name '*.v' ! -name 'config.v' ! -name 'e203_defines.v' | sort)
iverilog -g2012 -gno-assertions "${define_args[@]}" \
    -I "$upstream/rtl/e203/core" -I "$upstream/rtl/e203/general" \
    -I "$rtl_dir/../sram" -I "$tb_dir" -s "$tb_top" -o "$build_dir/e203" \
    "${core_files[@]}" \
    "$rtl_dir/../pulse_array/kwt_pe.sv" \
    "$rtl_dir/../pulse_array/kwt_pulse_array.sv" \
    "$rtl_dir/../pulse_array/kwt_postprocess.sv" \
    "$rtl_dir/../pulse_array/kwt_tile_engine.sv" \
    "$rtl_dir/../pulse_array/kwt_product_array.sv" \
    "$rtl_dir/../pulse_array/kwt_cacc.sv" \
    "$rtl_dir/../pulse_array/kwt_pingpong_cacc_engine.sv" \
    "$rtl_dir/../sram/kwt_operand_pingpong.sv" \
    "$rtl_dir/../sram/kwt_tile_buffers.sv" \
    "$rtl_dir/../dma/kwt_dma.sv" \
    "$rtl_dir/../dma/kwt_ext_memory_model.sv" \
    "$rtl_dir/kwt_tile_system_model.sv" \
    "$rtl_dir/kwt_nice_system_model.sv" \
    "$tb_dir/e203_kwt_nice_core.sv" \
    "$tb_dir/tb_kwt_real_e203.sv" \
    "$tb_dir/tb_kwt_full_layer_e203.sv" \
    2>"$build_dir/compile.log" || { cat "$build_dir/compile.log" >&2; exit 1; }
vvp "$build_dir/e203" "+PROGRAM=$build_dir/program.hex" "${run_args[@]}"
