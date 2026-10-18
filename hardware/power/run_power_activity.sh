#!/usr/bin/env bash
# Reproducible activity/task-cycle evidence for the frozen A12W10 8x16 FFN1
# E203 path. Runs real validation-sample layer01 operands through the
# NICE-like system model and dumps a bounded VCD for tile 0 and the last tile.
#
# This produces switching activity, NOT a power or energy number. No PDK,
# no interconnect model, and no SRAM macro power model are available here.
set -euo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd -- "$here/../.." && pwd)"
hw="$root/hardware"
tb_dir="$hw/power/tb"
rtl_dir="$hw/rtl"
upstream="$(cd -- "$hw/../资料/e203_hbirdv2_full" && pwd)"

tool_bin="${KWT_E203_TOOLCHAIN:-/home/yang/yosys-sta/oss-cad-suite/bin}"
iverilog="$tool_bin/iverilog"
vvp="$tool_bin/vvp"
if [[ ! -x "$iverilog" || ! -x "$vvp" ]]; then
    echo "Need oss-cad-suite iverilog/vvp (set KWT_E203_TOOLCHAIN). Tried: $tool_bin" >&2
    exit 3
fi
command -v riscv64-linux-gnu-gcc >/dev/null || { echo "missing riscv64-linux-gnu-gcc" >&2; exit 3; }
command -v riscv64-linux-gnu-objcopy >/dev/null || { echo "missing riscv64-linux-gnu-objcopy" >&2; exit 3; }
command -v xxd >/dev/null || { echo "missing xxd" >&2; exit 3; }

vectors="${1:-$hw/power/vectors/real_e203_validation0_layer01}"
results="${2:-$hw/power/results}"
if [[ ! -d "$vectors" ]]; then
    echo "vector directory not found: $vectors" >&2
    exit 2
fi
mkdir -p "$results"

# Icarus $readmemh rejects path bytes outside printable ASCII; force an
# ASCII-only temp root like run_e203.sh does.
tmp_base="${KWT_E203_TMP_BASE:-${TMPDIR:-/tmp}}"
if [[ "$tmp_base" != /* || "$tmp_base" == *[!\ -~]* ]]; then
    tmp_base=/tmp
fi
build_dir="$(mktemp -d "$tmp_base/kwt_power.XXXXXX")"
trap 'rm -rf -- "$build_dir"' EXIT

program_source="$rtl_dir/control/tb/e203_kwt_full_layer.S"
define_args=(-DKWT_E203_EXT_MEM_BYTES=524288)

# Stage vectors into the ASCII build dir (they may also come from a Chinese path).
ascii_cases="$build_dir/cases"
mkdir -p "$ascii_cases"
shopt -s nullglob
case_stems=()
for a_file in "$vectors"/*_a16.hex; do
    stem="$(basename "${a_file%_a16.hex}")"
    for suffix in a16 w16 bias32 expected32; do
        src="$vectors/${stem}_${suffix}.hex"
        [[ -f "$src" ]] || { echo "missing $src" >&2; exit 2; }
        cp -f -- "$src" "$ascii_cases/${stem}_${suffix}.hex"
    done
    case_stems+=("$stem")
done
if (( ${#case_stems[@]} == 0 )); then
    echo "no layer??_a16.hex in $vectors" >&2
    exit 2
fi
shopt -u nullglob

echo "== building E203+RTL model =="
mapfile -t core_files < <(find "$upstream/rtl/e203/core" "$upstream/rtl/e203/general" \
    -maxdepth 1 -name '*.v' ! -name 'config.v' ! -name 'e203_defines.v' | sort)
"$iverilog" -g2012 -gno-assertions "${define_args[@]}" \
    -I "$upstream/rtl/e203/core" -I "$upstream/rtl/e203/general" \
    -I "$rtl_dir/sram" -I "$rtl_dir/control/tb" -I "$tb_dir" \
    -s tb_kwt_power_activity -o "$build_dir/e203_power" \
    "${core_files[@]}" \
    "$rtl_dir/pulse_array/kwt_pe.sv" \
    "$rtl_dir/pulse_array/kwt_pulse_array.sv" \
    "$rtl_dir/pulse_array/kwt_postprocess.sv" \
    "$rtl_dir/pulse_array/kwt_tile_engine.sv" \
    "$rtl_dir/pulse_array/kwt_product_array.sv" \
    "$rtl_dir/pulse_array/kwt_cacc.sv" \
    "$rtl_dir/pulse_array/kwt_pingpong_cacc_engine.sv" \
    "$rtl_dir/sram/kwt_operand_pingpong.sv" \
    "$rtl_dir/sram/kwt_tile_buffers.sv" \
    "$rtl_dir/dma/kwt_dma.sv" \
    "$rtl_dir/dma/kwt_ext_memory_model.sv" \
    "$rtl_dir/control/kwt_tile_system_model.sv" \
    "$rtl_dir/control/kwt_nice_system_model.sv" \
    "$rtl_dir/control/tb/e203_kwt_nice_core.sv" \
    "$tb_dir/tb_kwt_power_activity.sv" \
    2>"$results/compile.log" || { cat "$results/compile.log" >&2; exit 1; }

status=0
for stem in "${case_stems[@]}"; do
    for tile in 0 207; do
        tag="$(printf '%s_tile%03d' "$stem" "$tile")"
        vcd="$results/${tag}.vcd"
        log="$results/${tag}.log"
        echo "== running $stem tile $tile =="
        # The RV32 submit loop is compiled per tile: the program only submits
        # KWT_TILE_COUNT descriptors starting at KWT_TILE_START.
        riscv64-linux-gnu-gcc "${define_args[@]}" \
            "-DKWT_TILE_START=$tile" "-DKWT_TILE_COUNT=1" \
            -march=rv32imac -mabi=ilp32 -nostdlib -nostartfiles \
            -Wl,-Ttext=0x80000000 -Wl,--no-relax \
            -o "$build_dir/program_${tile}.elf" "$program_source"
        riscv64-linux-gnu-objcopy -j .text -O binary "$build_dir/program_${tile}.elf" \
            "$build_dir/program_${tile}.bin"
        xxd -p -c 1 "$build_dir/program_${tile}.bin" > "$build_dir/program_${tile}.hex"
        program_size="$(stat -c %s "$build_dir/program_${tile}.bin")"
        for ((byte_index=program_size; byte_index<64; byte_index++)); do
            printf '00\n' >> "$build_dir/program_${tile}.hex"
        done
        if "$vvp" "$build_dir/e203_power" "+PROGRAM=$build_dir/program_${tile}.hex" \
            "+A=$ascii_cases/${stem}_a16.hex" "+W=$ascii_cases/${stem}_w16.hex" \
            "+BIAS=$ascii_cases/${stem}_bias32.hex" \
            "+EXPECTED=$ascii_cases/${stem}_expected32.hex" \
            "+TILE_START=$tile" "+TILE_COUNT=1" "+VCD=$vcd" >"$log" 2>&1; then
            echo "   sim exit 0"
        else
            rc=$?
            echo "   sim FAILED rc=$rc (see $log)" >&2
            status=1
        fi
    done
done

echo "== parsing VCDs =="
python3 "$here/parse_vcd.py" --results "$results" || status=1

exit "$status"
