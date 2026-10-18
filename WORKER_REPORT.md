# WORKER_REPORT — 冻结 A12W10/8×16 活动与任务周期证据（非功耗）

- 分支：`codex/kwt-power-activity`
- 工作树：`/home/yang/kwt-worktrees/power-activity`
- 基线提交（父）：`fcb4b72`
- 范围：只新增 `hardware/power/`（runner、新 testbench、解析脚本、报告、向量副本）。
  未改数值核、`run_e203.sh`、他人的文件；未开子智能体；未 push。

## 结论（证据层级）

在**同一冻结 A12W10、8×16** 的现有 E203/NICE-like 系统模型上，真实产生了
**真实验证样本 layer01**（validation-index 0）**tile 0 与末 tile 207** 的 VCD，
并用脚本核验了波形、时钟、关键模块翻转、窗口起止与装载/计算/写回相位、任务周期。

- **证据层级：NICE-like E203 行为仿真 + 128-bit 仿真总线 + 行为 SRAM 模型**，
  **不是**组委会正式 SoC/PDK，**不是**综合网表，**不是** 150 MHz 或功耗结论。
- **功耗/单任务能耗：`not_computable`**。无 PDK、无互连功耗模型、无 SRAM 宏功耗模型；
  未标注电路与未建模 SRAM 不按 0 处理。
- 任务周期为**仿真观察值**：两 tile 均 `task_cycles=4025`（首 NICE 请求 → 末 NICE 响应）。

## 实际执行的命令与真实退出码

| # | 命令 | 退出码 | 结果 |
| --- | --- | --- | --- |
| 1 | `bash hardware/power/run_power_activity.sh` | **0** | 2 个 tile 仿真 `sim exit 0`，解析 `verdict=PASS` |
| 2 | `python3 hardware/power/parse_vcd.py --results hardware/power/results` | **0** | 所有检查 TRUE |

运行日志（`hardware/power/results/`）：

```
layer01_tile000.log: PASS ... outputs=128 padding=0 task_cycles=4025 total_cycles=4038 first_req_cycle=7 last_rsp_cycle=4031
layer01_tile207.log: PASS ... outputs=48 padding=80 task_cycles=4025 total_cycles=4039 first_req_cycle=8 last_rsp_cycle=4032
```

## 波形核验结果（`hardware/power/results/summary.json`）

| 指标 | tile 000 | tile 207 |
| --- | ---: | ---: |
| VCD 字节 | 822166 | 581868 |
| 时钟翻转数 | 8097 | 8099 |
| 时钟周期（时间单位） | 10 | 10 |
| 任务周期（波形推导） | 4025 | 4025 |
| 任务周期（日志） | 4025 | 4025 |
| 所选信号覆盖率 | 100% | 100% |
| 窗口起点/终点 | 175 / 40415 | 185 / 40425 |
| 装载相位（u） | 19991 | 19991 |
| 计算相位（u） | 16961 | 16961 |
| 写回相位（u） | 3221 | 3221 |

关键模块切换计数（tile 000 / tile 207）：
E203 NICE FSM 11060/10728，tile 系统 11034/10702，计算引擎 6611/6279，
乘积阵列 1290/1285，CACC 518/518，DMA 4349/4349。

检查项（VCD 非空、时钟翻转、关键模块翻转、窗口起点对应首 NICE 请求、
窗口终点对应末 NICE 响应、装载/计算/写回相位存在且顺序落在窗口内、周期数与日志一致、
日志 PASS）在两 tile 上全部为 TRUE。

## 改动文件

| 文件 | 说明 |
| --- | --- |
| `hardware/power/tb/tb_kwt_power_activity.sv` | 新 testbench：同 `tb_kwt_real_layer_e203` 功能，增加受限 VCD dump 与窗口标记 |
| `hardware/power/run_power_activity.sh` | runner：逐 tile 编译提交程序、编译 E203，跑 tile 0/207 并调用解析 |
| `hardware/power/parse_vcd.py` | VCD 解析与核验，输出 `summary.json` |
| `hardware/power/vectors/real_e203_validation0_layer01/` | 真实样本 A12W10 layer01 向量副本 + manifest |
| `hardware/power/FROZEN_CONFIG.md` | 冻结配置、输入哈希、工具版本、频率假设 |
| `hardware/power/README.md` | 口径、运行方式与边界 |
| `hardware/power/results/` | `layer01_tile000/207.log`、`compile.log`、`summary.json`（VCD 被 `.gitignore` 排除） |

## 覆盖率与未知部分（诚实边界）

- `toggled_coverage_pct=100%` 只覆盖**脚本精选的控制/总线信号**；大行为 SRAM 阵列与
  E203 数据通路内部**未 dump**，因此这不是全设计覆盖率。
- VCD `$timescale 1s` 为 Icarus 默认（上游 E203 文件无统一 timescale，显式 1ns 会造成
  跨 timescale 语义漂移，实测会挂起，故未设）。周期数与时间单位无关。
- 频率假设：题目 150 MHz 目标；`4025 cycles × 6.6667 ns ≈ 26.83 µs/tile-task` 仅为
  **算术换算**，未做 STA、无目标库/SRAM 宏/布局布线，**不是已达成频率**。
- 无正式 System Bus/NICE 适配、无目标 SRAM 宏、无综合网表功耗、无隐藏 600 条评测。
- 向量来自 `_12labels_40`（480/480）标定的 A12W10 参数包，与本分支 240/240 的
  `ffn1_a12w10_static_sym.npz` 不同（两者不可混用），见 `FROZEN_CONFIG.md`。
- VCD 含 `$date`，跨运行字节哈希会变；提交的是脚本、日志与 `summary.json`。

## 未证事项

- 未跑 layer02–12 的活动 VCD（本次只对真实 layer01 做了 tile 0 与 tile 207）。
- 未做示例库功耗估算；即使要做，也只能作为独立探索性口径，需明确无 SRAM/互连模型。
- 未在组委会正式 SoC、总线或 NICE 接口上联调。
