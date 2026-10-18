# WORKER_REPORT — FFN1 8×16 综合/时序事实收束

- 分支：`codex/kwt-synth`
- 工作树：`/home/yang/kwt-worktrees/synth`
- 模型：`opencode-go/deepseek-v4.1-flash`
- 时间：2026-10-18（约 10 分钟内完成，未重跑完整映射或 STA）
- 首次运行日志：`/home/yang/kwt-worktrees/orchestration/logs/synth-first.log`
- 既有产物目录：`/tmp/kwt-8x16-sta/`

## 目标

把首次运行（超时中断）留下的事实收束为可复现、不误导的最小交付：修正未跟踪脚本
`hardware/synth/run_ffn1_8x16_sta.sh` 的关键缺陷，如实记录综合/面积/iSTA 现状，
不夸大 150 MHz 与竞赛 2.25 mm²。

## 执行命令、退出码与结果

| # | 命令 | 退出码 | 结果 |
| --- | --- | --- | --- |
| 1 | `git status --porcelain -uall` | 0 | 唯一未跟踪文件：`hardware/synth/run_ffn1_8x16_sta.sh` |
| 2 | `bash -n hardware/synth/run_ffn1_8x16_sta.sh` | 0 | 语法 OK |
| 3 | `SYNTH_OUT=/tmp/kwt-ffn1-8x16-verify bash hardware/synth/run_ffn1_8x16_sta.sh --coarse-only` | 0 | 1.98s；128 `$mul` / 9 `$add` / 8 `$mem_v2` / 2270 cells；0 problems |
| 4 | iSTA 失败码透传自测（stub 退出 134） | 134 | 脚本正确透传 134 并打印「Do NOT claim 150 MHz」 |
| 5 | 解析 `coarse.log` / `map_*.log` / `sta_*.log` | 0 | 见下方事实 |
| 6 | `git add` 指定文件 + `git commit` | 0 | 见提交信息 |

## 修正的脚本缺陷（原 `run_ffn1_8x16_sta.sh`）

1. **iSTA 失败被吞掉**：原脚本
   `"$IEDA" ... > log 2>&1 || true`，即使 iSTA `abort`（退出码 134）脚本仍会走完
   并打印 `Logs: ...`，读者会误以为时序通过。
   → 现改为 `set +e` 捕获退出码：非零即打印失败原因并 `exit "$sta_rc"` 透传 134。
2. **无条件解析不存在的时序报告**：原脚本在 iSTA 已崩溃时仍 `grep ... || true`。
   → 现仅在 iSTA 退出码为 0 后解析，且解析不到时序行时以退出码 4 失败。
3. **noexpr 网表缺失**：原脚本只产出含 706 条 `assign` 的网表（iSTA 会 abort）。
   → 现额外用 `splitnets -format __v; opt_clean -purge; write_verilog -noexpr` 产出
   0 条 `assign` 的 `*_ics_noexpr.v`，并在文档/注释中注明**已生成但尚未经过 STA**。
4. **措辞纠偏**：脚本头部、SDC 注释、版本记录统一标注 `150 MHz` 为「目标，非结论」，
   并声明示例库面积不得换算成竞赛 2.25 mm²。

## 失败原因（真实、未修复）

iSTA（iEDA `071226fa`）在 `link_design` 解析含层次化 `assign` 别名的网表时崩溃：

```
F20261018 14:49:09.838889 42757 Sta.cc:834] the left port is not exist.
*** Check failure stack trace: ***
timeout: the monitored command dumped core
```

- Nangate45 网表：`/tmp/kwt-8x16-sta/sta_nangate45.log` → `EXIT=134`
- icsprout55 网表：`/tmp/kwt-8x16-sta/sta_icsprout55.log` → `EXIT=134`
- 两份日志 `slack`/`report_timing` 匹配行数均为 **0**：**未产生**任何时序结论。

## 已确认事实

- 工具：Yosys `0.63+97`、iEDA/iSTA `071226fa6d52375d3a365605751a3413ea51cd41`。
- 粗粒度：128 `$mul`、8 `$mem_v2`、2270 cells、0 problems（退出码 0）。
- 已产出网表：`systolic_nangate45.v`、`systolic_icsprout55.v`（706 assign）、
  `systolic_ics_noexpr.v`（0 assign）。
- 示例库面积（仅同库相对参考）：Nangate45 `355178.628000`；
  icsprout55 `542054.800000`。

## 剩余工作

1. 用 `systolic_ics_noexpr.v` 实跑 iSTA，确认 noexpr 网表能否解析；若仍 abort，需继续
   排查（例如改用 `write_verilog` 的其他选项或调整 iSTA 输入）。
2. 取得任何**成功**的 iSTA 报告后，才能谈论 150 MHz；在此之前不得声称达标。
3. 面积结论需在**组委会工艺库**、含题目规定 16 bit SRAM 宏、整机（含 E203/总线/
   NICE/控制器）条件下重测，方可与 2.25 mm² 比较；当前数字不可换算。
4. 若后续确需长耗时映射/STA，请在具备 >15 分钟预算时运行并登记结果。

## 交付物

- `hardware/synth/run_ffn1_8x16_sta.sh`（已修正）
- `hardware/synth/FFN1_8x16_synthesis_status.md`（如实状态）
- `WORKER_REPORT.md`（本文件）
