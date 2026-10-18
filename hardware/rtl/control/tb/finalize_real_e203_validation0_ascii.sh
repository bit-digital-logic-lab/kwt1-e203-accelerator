#!/usr/bin/env bash
set -uo pipefail

repo="/home/yang/Desktop/ic设计竞赛"
results="$repo/hardware/architecture/results"
vectors="$results/real_e203_validation0_vectors"
logs="$results/real_e203_validation0_ascii_logs"
summary="$results/real_e203_validation0_ascii_summary.json"
report="$results/real_e203_validation0_ascii_report.md"
status="$results/real_e203_validation0_ascii_finalizer.status"
opencode="/home/yang/.nvm/versions/node/v24.18.0/bin/opencode"
check_id="${KWT_CHECK_ID:-real_e203_validation0_ascii}"
check_py="$repo/.opencode/skills/quarter-hour-check/scripts/check.py"

# Record a terminal check-in even on abnormal exit, otherwise the 15-minute
# scan keeps flagging service_stopped after the 12-layer run already finished.
if [[ -r "$repo/.opencode/skills/quarter-hour-check/scripts/finalize_hook.sh" ]]; then
    source "$repo/.opencode/skills/quarter-hour-check/scripts/finalize_hook.sh"
    check_finalize_init "$check_id" "$check_py"
else
    echo "warning: finalize_hook.sh missing, terminal check-in disabled" >&2
    check_finalize_init() { :; }
    check_finalize_state() { :; }
fi

cd "$repo" || exit 1
flow_ok=1
printf 'WAITING\n' > "$status"
for ((attempt=0; attempt<180; attempt++)); do
    count=0
    for ((layer=1; layer<=12; layer++)); do
        printf -v stem 'layer%02d' "$layer"
        [[ -f "$logs/$stem.exit" ]] && ((count+=1))
    done
    ((count==12)) && break
    sleep 30
done
if ((count!=12)); then
    printf 'TIMEOUT %d/12\n' "$count" > "$status"
    flow_ok=0
    check_finalize_state failed "$status" "12 层日志未齐：$count/12"
else
    python3 hardware/rtl/control/tb/summarize_real_e203.py \
        --vectors "$vectors" --logs "$logs" --out "$summary" \
        > "$results/real_e203_validation0_ascii_summary_run.log" 2>&1
    summarize_rc=$?
    printf 'SUMMARIZE_EXIT %d\n' "$summarize_rc" > "$status"
    if ((summarize_rc!=0)); then
        flow_ok=0
        check_finalize_state failed "$status" "summarize 退出码 $summarize_rc"
    fi
fi

prompt="单独处理例行结果整理，不调用子智能体。检查 $status、$results/real_e203_validation0_ascii_summary_run.log、12 层日志与退出码，以及 $vectors/manifest.json。仅据实际记录写一份简短中文报告到 $report，说明通过/失败层数、有效输出/填充校验、周期、数值误差、CPU 向量复现范围，以及这只是 E203 + NICE-like 仿真模型而非组委会正式 SoC/PDK。若测试失败，写明具体失败层和原因，不得写 PASS。完成后直接退出，不等待对话。"
"$opencode" run --standalone --auto --model opencode-go/deepseek-v4.1-flash "$prompt" \
    > "$results/real_e203_validation0_ascii_deepseek.log" 2>&1
model_rc=$?
printf 'MODEL_EXIT %d\n' "$model_rc" >> "$status"
[[ -f "$report" ]] || printf 'REPORT_MISSING\n' >> "$status"

# Only reached with a clean summarize; require the report and a clean model exit.
if ((flow_ok==1)); then
    if ((model_rc==0)) && [[ -f "$report" ]]; then
        check_finalize_state complete "$report" "12 层完成，报告与汇总已核验"
    else
        printf 'FINALIZE_FAILED model_rc=%d report=%s\n' "$model_rc" \
            "$([[ -f "$report" ]] && printf present || printf missing)" >> "$status"
        check_finalize_state failed "$status" "报告整理未完成 model_rc=$model_rc"
    fi
fi
