#!/usr/bin/env bash
set -euo pipefail
tb_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
rtl_dir="$(dirname -- "$tb_dir")"
upstream="$(cd -- "$tb_dir/../../../../资料/e203_hbirdv2_full" && pwd)"
# Icarus `$readmemh` rejects path bytes outside printable ASCII (it warns about
# non-printable characters and loads X). The project path is Chinese, so force
# an ASCII-only temp root instead of asking the user to prepare a /tmp alias.
tmp_base="${KWT_E203_TMP_BASE:-${TMPDIR:-/tmp}}"
if [[ "$tmp_base" != /* || "$tmp_base" == *[!\ -~]* ]]; then
    tmp_base=/tmp
fi
build_dir="$(mktemp -d "$tmp_base/kwt_e203.XXXXXX")"
trap 'rm -rf -- "$build_dir"' EXIT

tb_top=tb_kwt_real_e203
program_source="$tb_dir/e203_kwt_program.S"
define_args=()
run_args=()
real_cases_root=""
if [[ "${1:-}" == "--full-layer" ]]; then
    tb_top=tb_kwt_full_layer_e203
    program_source="$tb_dir/e203_kwt_full_layer.S"
    define_args=(-DKWT_E203_EXT_MEM_BYTES=524288)
elif [[ "${1:-}" == "--real-layers" ]]; then
    if [[ ( $# -ne 2 && $# -ne 4 ) || ! -d "$2" ]]; then
        echo "Usage: bash rtl/control/tb/run_e203.sh --real-layers CASE_ROOT [TILE_START TILE_COUNT]" >&2
        exit 2
    fi
    tb_top=tb_kwt_real_layer_e203
    program_source="$tb_dir/e203_kwt_full_layer.S"
    define_args=(-DKWT_E203_EXT_MEM_BYTES=524288)
    real_cases_root="$2"
    if [[ $# -eq 4 ]]; then
        if [[ ! "$3" =~ ^[0-9]+$ || ! "$4" =~ ^[0-9]+$ ]] ||
           (( 10#$4 < 1 || 10#$3 + 10#$4 > 208 )); then
            echo "Real-layer tile range must be within 0..207" >&2
            exit 2
        fi
        tile_start=$((10#$3))
        tile_count=$((10#$4))
        define_args+=("-DKWT_TILE_START=$tile_start" "-DKWT_TILE_COUNT=$tile_count")
        run_args=("+TILE_START=$tile_start" "+TILE_COUNT=$tile_count")
    fi
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
    echo "Usage: bash rtl/control/tb/run_e203.sh [--full-layer | --real-layers CASE_ROOT [START COUNT] | --tiles START COUNT]" >&2
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
    "$tb_dir/tb_kwt_real_layer_e203.sv" \
    2>"$build_dir/compile.log" || { cat "$build_dir/compile.log" >&2; exit 1; }
if [[ -n "$real_cases_root" ]]; then
    shopt -s nullglob
    ascii_cases_dir="$build_dir/real_cases"
    mkdir -p "$ascii_cases_dir"
    real_cases=("$real_cases_root"/layer??_a16.hex)
    if (( ${#real_cases[@]} == 0 )); then
        echo "No layer??_a16.hex cases in $real_cases_root" >&2
        exit 2
    fi
    for a_file in "${real_cases[@]}"; do
        case_stem="${a_file%_a16.hex}"
        for filename in "$a_file" "${case_stem}_w16.hex" "${case_stem}_bias32.hex" "${case_stem}_expected32.hex"; do
            if [[ ! -f "$filename" ]]; then
                echo "Missing $filename" >&2
                exit 2
            fi
        done
        expected_lines=(6656 16384 256 25344)
        case_files=("$a_file" "${case_stem}_w16.hex" "${case_stem}_bias32.hex" "${case_stem}_expected32.hex")
        for i in 0 1 2 3; do
            actual_lines="$(wc -l < "${case_files[i]}")"
            if [[ "$actual_lines" -ne "${expected_lines[i]}" ]]; then
                echo "${case_files[i]} has $actual_lines lines, expected ${expected_lines[i]}" >&2
                exit 2
            fi
        done
        # Stage ASCII copies before handing paths to Icarus: CASE_ROOT may sit
        # under a Chinese project path, and `$readmemh` would otherwise load X.
        stage_stem="$ascii_cases_dir/$(basename "$case_stem")"
        cp -f -- "$a_file" "${stage_stem}_a16.hex"
        cp -f -- "${case_stem}_w16.hex" "${stage_stem}_w16.hex"
        cp -f -- "${case_stem}_bias32.hex" "${stage_stem}_bias32.hex"
        cp -f -- "${case_stem}_expected32.hex" "${stage_stem}_expected32.hex"
        echo "Running E203 $(basename "$case_stem")"
        vvp "$build_dir/e203" "+PROGRAM=$build_dir/program.hex" \
            "+A=${stage_stem}_a16.hex" "+W=${stage_stem}_w16.hex" \
            "+BIAS=${stage_stem}_bias32.hex" "+EXPECTED=${stage_stem}_expected32.hex" \
            "${run_args[@]}"
    done
else
    vvp "$build_dir/e203" "+PROGRAM=$build_dir/program.hex" "${run_args[@]}"
fi
