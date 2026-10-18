# WORKER_REPORT — FFN1 8×16 `noexpr` iSTA 复核

- 分支：`codex/kwt-sta-followup`（基于 synth 快照）
- 工作树：`/home/yang/kwt-worktrees/sta-followup`
- 模型：`opencode-go/deepseek-v4.1-flash`
- 时间：2026-10-18（两次 bounded iSTA，各 < 240 s）
- 边界：只改本工作树；未 push；未开子智能体；未改主目录/其他分支；
  未修改已有综合脚本或 RTL 数值

## 任务

验证上轮生成的 assign-free `systolic_ics_noexpr.v` 能否被本机 iSTA 解析，
是否产生真实存在的 slack/report_timing，时钟端口是否生效；不得声称 150 MHz 达标。

## 执行命令、退出码与结果

| # | 命令 | 退出码 | 结果 |
| --- | --- | --- | --- |
| 1 | 读 `hardware/synth/FFN1_8x16_synthesis_status.md`、`run_ffn1_8x16_sta.sh`、`/tmp/kwt-8x16-sta` 产物 | 0 | 确认上轮默认网表含 706 `assign`，iSTA SIGABRT 134；noexpr 网表 0 `assign` 但尚未跑 STA |
| 2 | `timeout 240 iEDA -script sta.tcl ffn1_8x16_150mhz.sdc systolic_ics_noexpr.v kwt_systolic_cacc_tile icsprout55` | **0** | 约 147 s；`link_design` 成功；报告 `kwt_systolic_cacc_tile.rpt` 生成 |
| 3 | 新增并运行 `bash hardware/synth/run_sta_noexpr_followup.sh`（`OUT=/tmp/kwt-8x16-sta-followup`） | **0** | 约 175 s；脚本自检退出码/报告/时钟/slack 全部通过 |
| 4 | `bash -n hardware/synth/run_sta_noexpr_followup.sh` | 0 | 语法 OK |
| 5 | `git add` 指定文件 + `git commit` | 0 | 见本文件末尾 |

## 实测结论

- **iSTA 能解析 noexpr 网表**：`inst num: 238537`、`net num: 238989`、`port num: 486`。
- **时钟端口生效**：日志 `create clock clk for pin/port: clk`；报告时钟组 `clk`，
  setup required `6.596 ns`。
- **slack / report_timing 真实存在**：报告含 `Slack`、`TNS`、逐级 path 表、
  `slack (VIOLATED)` / `slack (MET)`；`The timing engine run success.`。
- **时序结果（探索性）**：worst setup slack `-12.298 ns`，arrival `18.894 ns`，
  required `6.596 ns`，TNS `-2770.397 ns`，worst hold `+0.092 ns`，报告 `52.732 MHz`。
- **不能声称 150 MHz 达标**：该约束下 setup 明确 VIOLATED。
- **不可当作签核**：理想时钟、`path net delay = 0.000`、无布局/CTS/寄生、
  示例库、`10` 次库表外插值警告、`416` fanout 主导路径、无 I/O 约束、对象仅计算核。

## 证据（绝对路径 + SHA256）

| 产物 | 路径 | SHA256 |
| --- | --- | --- |
| noexpr 网表 | `/tmp/kwt-8x16-sta/systolic_ics_noexpr.v` | `a983c2b09595f41f18e7aee7096bf0aa54400803aeb7293c27360f18b5871cd8` |
| SDC | `/tmp/kwt-8x16-sta/ffn1_8x16_150mhz.sdc` | `cbc3759efd91ba0609492ec1849be696273ca1ab2e6ae3b351bc6bc949c53fdc` |
| 第一次 iSTA 日志 | `/tmp/kwt-8x16-sta-followup/noexpr_sta.log` | `3260d7c528c74277fcc751bf6475807cc144dfc6e40198aca31bc6d6cba27fe0` |
| 第一次 iSTA 报告 | `/tmp/kwt-8x16-sta/kwt_systolic_cacc_tile.rpt` | `55dd5d9b1809a41ccba68efeae9fb4562c6ac2c0a1884d891351cef40ff90eeb` |
| 脚本 iSTA 日志 | `/tmp/kwt-8x16-sta-followup/kwt_systolic_cacc_tile_150mhz_noexpr_sta.log` | `611fec58002a858726dd8df4ead073747234d638ce8135f6de5c30fc8d75b599` |
| 脚本 iSTA 报告 | `/tmp/kwt-8x16-sta-followup/kwt_systolic_cacc_tile.rpt` | `24c64ef532e72b6c96039faecb25d543200981f3db8dc36bb099b4ad881bfc7d` |
| 脚本版本记录 | `/tmp/kwt-8x16-sta-followup/kwt_systolic_cacc_tile_noexpr_versions.txt` | `218d95b88a918d480d7fbc42d6c5fad79be243285a311e59421cb73307c5aaea` |
| 脚本 stdout | `/tmp/kwt-8x16-sta-followup/run_script_stdout.txt` | `8c774b9df16593b233d0392cae010f4c7039c9f47081ccf697fedbca5686cf6b` |

大网表/日志不提交，只在本机 `/tmp` 引用。

## 交付物

- 新增 `hardware/synth/STA_FOLLOWUP_20261018.md`
- 新增 `hardware/synth/run_sta_noexpr_followup.sh`
- 更新 `WORKER_REPORT.md`（本文件）

未修改 `hardware/synth/run_ffn1_8x16_sta.sh`、其他综合脚本或 RTL。

## 剩余工作 / 不得越界

1. 仍需正式 PDK、题目 16 bit SRAM 宏和整机（E203/总线/NICE/控制器）后，才有资格
   讨论竞赛 150 MHz 与 2.25 mm²。
2. 示例库面积 `542054.800000` 与本次 slack 数字均不得换算或外推。
3. 若要在正式条件下做长耗时 STA，需在 >15 分钟预算下运行并登记。
