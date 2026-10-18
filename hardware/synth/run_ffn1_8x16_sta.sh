#!/usr/bin/env bash
# 依现性综合 + 示例库时序脚本：本科必做 8×16 FFN1 计算核
#
# 对象（证据层级 = 仅计算核，不含存储/DMA/接口/控制）：
#   kwt_systolic_cacc_tile = kwt_product_array(8×16, 128 个 16×16 有符号乘法)
#                          + kwt_cacc(8 个 1R1W 部分和 bank，每 bank 16×INT32)
# 这是 hardware/synth/run_engine.sh 里 8×16 路径的算术主体；本脚本单独、可复现地
# 记录它的资源计数、示例库面积，并尝试给出一个给定时钟下的 iSTA 时序。
#
# 边界（勿越过）：
#   不含 A/W 操作数 SRAM、输出暂存、DMA、E203/System Bus/NICE 接口和任务控制器。
#   因此本脚本的数字既不是整机 2.25 mm²，也不是整机 150 MHz 证明；示例库
#   （Nangate45 / icsprout55）不是组委会工艺库，也不包含题目规定的 16 bit SRAM 宏。
#   示例库面积（Chip area for module）只能做同库相对比较，禁止换算成竞赛 2.25 mm²。
#
# iSTA 现状（重要，勿误读）：
#   - 手工 `write_verilog`（含 assign/hierarchical alias）的网表会让 iSTA 在
#     link_design 阶段 abort（SIGABRT，退出码 134，见 Sta.cc:834
#     "the left port is not exist."）。
#   - 本脚本改用 `write_verilog -noexpr` 生成无 assign 的网表，但截至最后一次
#     实跑，noexpr 网表只被"生成"过，尚未成功跑通 iSTA。在 iSTA 真正通过之前，
#     不得声称该计算核满足 150 MHz。
#   - 因此 iSTA 软件非零退出会直接让本脚本失败（不再 `|| true` 吞掉）。
#
# 依赖与版本（默认值可用环境变量覆盖）：
#   YOSYS        建议 >= 0.48（系统 /usr/bin/yosys 0.9 不支持参数化 SystemVerilog）
#   OSS_YOSYS    默认 /home/yang/yosys-sta/oss-cad-suite/bin/yosys (0.63)，优先于 YOSYS
#   YOSYS_STA    默认 /home/yang/yosys-sta（只读引用其示例 PDK 与 STA 脚本）
#   IEDA         默认 $YOSYS_STA/bin/iEDA（iSTA）
#
# 用法：
#   bash hardware/synth/run_ffn1_8x16_sta.sh [--coarse-only] [--no-sta] [--skip-nangate]
#   CLK_FREQ_MHZ=150 SYNTH_OUT=/tmp/kwt-ffn1-8x16 bash hardware/synth/run_ffn1_8x16_sta.sh
#
# 退出码：
#   0  所选步骤全部成功（--coarse-only 只跑粗粒度，必然不涉及 iSTA）
#   非0 任一步骤失败；iSTA 失败会原样透传其退出码（例如 134 = SIGABRT）
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/../.." && pwd)"
out_dir="${SYNTH_OUT:-/tmp/kwt-ffn1-8x16-synth-sta}"
clock_mhz="${CLK_FREQ_MHZ:-150}"

YOSYS_STA="${YOSYS_STA:-/home/yang/yosys-sta}"
if [[ -z "${YOSYS:-}" && -x "$YOSYS_STA/oss-cad-suite/bin/yosys" ]]; then
    YOSYS="$YOSYS_STA/oss-cad-suite/bin/yosys"
else
    YOSYS="${YOSYS:-yosys}"
fi
IEDA="${IEDA:-$YOSYS_STA/bin/iEDA}"
N45_LIB="${N45_LIB:-$YOSYS_STA/pdk/nangate45/lib/Nangate45_typ.lib}"
ICS_LIB="${ICS_LIB:-$YOSYS_STA/pdk/icsprout55/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/liberty/ics55_LLSC_H7CL_ss_rcworst_1p08_125_nldm.lib}"
STA_TCL="${STA_TCL:-$YOSYS_STA/scripts/sta.tcl}"

do_coarse=1
do_nangate=1
do_sta=1
for arg in "$@"; do
    case "$arg" in
        --coarse-only) do_nangate=0; do_sta=0 ;;
        --no-sta)      do_sta=0 ;;
        --skip-nangate) do_nangate=0 ;;
        *) echo "Usage: bash hardware/synth/run_ffn1_8x16_sta.sh [--coarse-only] [--no-sta] [--skip-nangate]" >&2; exit 2 ;;
    esac
done

top=kwt_systolic_cacc_tile
rtl=(hardware/rtl/pulse_array/kwt_product_array.sv
     hardware/rtl/pulse_array/kwt_cacc.sv
     hardware/rtl/pulse_array/kwt_systolic_cacc_tile.sv)
period_ns="$(awk -v f="$clock_mhz" 'BEGIN{printf "%.4f", 1000.0/f}')"

mkdir -p "$out_dir"
cd "$repo_dir"

# 记录真实使用到的工具版本，避免事后推断。
{
    echo "date: $(date -Iseconds)"
    echo "clock_mhz: $clock_mhz (period ${period_ns} ns) [target only, not a proven result]"
    echo "yosys: $("$YOSYS" -V 2>&1 | head -1)"
    echo "iEDA:  $(echo exit | "$IEDA" -v 2>&1 | head -1)"
    echo "n45_lib: $N45_LIB"
    echo "ics_lib: $ICS_LIB"
    echo "rtl: ${rtl[*]}"
} | tee "$out_dir/versions.txt"

# 1) 粗粒度：保留推断存储，记录乘法/加法/bank 计数。
if [[ "$do_coarse" == 1 ]]; then
    echo "== coarse (memories preserved) =="
    "$YOSYS" -Q -T -p \
        "read_verilog -sv ${rtl[*]}; hierarchy -check -top $top; proc; opt; memory_collect; check -assert; stat" \
        > "$out_dir/${top}_coarse.log" 2>&1
    grep -E '^=== |^[[:space:]]+[0-9]+[[:space:]]+\$mul|^[[:space:]]+[0-9]+[[:space:]]+\$add|^[[:space:]]+[0-9]+[[:space:]]+\$mem_v2|^[[:space:]]+[0-9]+[[:space:]]+cells|Found and reported 0 problems' \
        "$out_dir/${top}_coarse.log"
fi

# 2) Nangate45 typical：把部分和 bank 与内部存储全部映射为触发器后的计算核面积。
#    仅用于同库相对比较，不是赛题面积；数字不得换算成竞赛 2.25 mm²。
if [[ "$do_nangate" == 1 ]]; then
    if [[ ! -f "$N45_LIB" ]]; then
        echo "Missing Nangate45 Liberty: $N45_LIB" >&2; exit 2
    fi
    echo "== Nangate45 flop-mapped compute-core area (relative reference only) =="
    "$YOSYS" -Q -T -p \
        "read_liberty -lib -ignore_miss_func $N45_LIB; read_verilog -sv ${rtl[*]}; synth -top $top -flatten; dfflibmap -liberty $N45_LIB; abc -liberty $N45_LIB; opt_clean; check -assert; stat -liberty $N45_LIB" \
        > "$out_dir/${top}_nangate45.log" 2>&1
    grep -E "Chip area for module|Found and reported" "$out_dir/${top}_nangate45.log" | tail -3
fi

# 3) icsprout55（ss_rcworst 1.08V 125C）映射 + iSTA @ 目标时钟。
#    仍然需要目标 SRAM 宏/PDK/接口后才能外推赛题结论；示例库面积严禁换算成 2.25 mm²。
if [[ "$do_sta" == 1 ]]; then
    if [[ ! -f "$ICS_LIB" ]]; then
        echo "Missing icsprout55 Liberty: $ICS_LIB" >&2; exit 2
    fi
    if [[ ! -x "$IEDA" ]]; then
        echo "Missing iSTA binary: $IEDA" >&2; exit 2
    fi
    echo "== icsprout55 mapped netlist (ss_rcworst) =="
    # 同时导出两种网表：
    #   *_icsprout55.v        含 assign 的默认网表（面积/网表归档用，iSTA 会 abort）
    #   *_ics_noexpr.v        无 assign 的网表（供 iSTA 尝试；尚未验证能跑通）
    "$YOSYS" -Q -T -p \
        "read_liberty -lib -ignore_miss_func $ICS_LIB; read_verilog -sv ${rtl[*]}; synth -top $top -flatten; dfflibmap -liberty $ICS_LIB; abc -liberty $ICS_LIB; hilomap -singleton -hicell TIEHIH7L Z -locell TIELOH7L Z; setundef -zero; opt_clean; check -assert; stat -liberty $ICS_LIB; write_verilog -noattr $out_dir/${top}_icsprout55.v; splitnets -format __v; opt_clean -purge; write_verilog -noattr -noexpr -nodec -defparam $out_dir/${top}_ics_noexpr.v" \
        > "$out_dir/${top}_icsprout55_map.log" 2>&1
    grep -E "Chip area for module|Found and reported" "$out_dir/${top}_icsprout55_map.log" | tail -3

    netlist_noexpr="$out_dir/${top}_ics_noexpr.v"
    if [[ ! -s "$netlist_noexpr" ]]; then
        echo "iSTA netlist was not produced: $netlist_noexpr" >&2
        exit 3
    fi
    # iSTA 对含 hierarchical assign 别名的网表会 abort；此处仅作前置警示，不代替真跑。
    assign_count="$(grep -c 'assign' "$netlist_noexpr" || true)"
    echo "assigns in $netlist_noexpr: $assign_count"

    cat > "$out_dir/${top}_${clock_mhz}mhz.sdc" <<EOF
# FFN1 8x16 compute core: ${clock_mhz} MHz TARGET (period ${period_ns} ns)
# NOTE: this is a requested constraint, not a proven achievable frequency.
create_clock -name clk -period ${period_ns} [get_ports clk]
set_clock_uncertainty 0.05 [get_clocks clk]
EOF
    echo "== iSTA @ ${clock_mhz} MHz (target, result not yet demonstrated) =="
    set +e
    "$IEDA" -script "$STA_TCL" \
        "$out_dir/${top}_${clock_mhz}mhz.sdc" \
        "$netlist_noexpr" "$top" icsprout55 \
        > "$out_dir/${top}_${clock_mhz}mhz_sta.log" 2>&1
    sta_rc=$?
    set -e
    if [[ "$sta_rc" -ne 0 ]]; then
        echo "iSTA FAILED with exit code $sta_rc (see $out_dir/${top}_${clock_mhz}mhz_sta.log)" >&2
        echo "Do NOT claim ${clock_mhz} MHz: no timing report was produced." >&2
        exit "$sta_rc"
    fi
    # 只有 iSTA 成功退出后才抽取时序报告。
    if ! grep -iE 'slack|critical path|arrival|required|report_timing|Setup|max_path' \
        "$out_dir/${top}_${clock_mhz}mhz_sta.log" | tail -40; then
        echo "iSTA exited 0 but produced no recognizable timing lines; inspect the log." >&2
        exit 4
    fi
fi

echo "Logs: $out_dir"
