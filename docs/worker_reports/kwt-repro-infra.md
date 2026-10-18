# WORKER_REPORT — repro-infra（项目复现入口 tools/repro）

- 分支：`codex/kwt-repro-infra`（worktree `/home/yang/kwt-worktrees/repro-infra`，基线 `56b8a39`）
- 任务：只在新目录 `tools/repro/` 实现项目复现入口，两级真实运行（默认 `smoke`、显式 `--level full`），逐项保存版本、命令、退出码、耗时、输入哈希与结果 JSON；超时显式失败；文件存在不等于 PASS；smoke 不跑 208 块整层全仿真与大批量模型采样。
- 约束遵守：未开启子智能体；未改核心 RTL、主测试脚本、钩子或其他代理目录；未 push/merge/rebase；未改主目录与其他分支。仅新增 `tools/repro/**` 与本报告。

## 改动（全部新增）

- `tools/repro/run.py`：运行器与检查清单（level / gate / 超时 / 前置条件 / 输入哈希 / 真实退出码判定 / `RESULT_JSON` 解析 / `summary.json`）。
- `tools/repro/checks/env_versions.py`：记录工具版本。
- `tools/repro/checks/offline_encoder_contract.py`：纯 numpy 回放冻结的 `offline_encoder_input.convert`，逐值比较。
- `tools/repro/README.md`：用法、结果 schema、依赖与覆盖边界。
- `tools/repro/results/{smoke,full,fault-exit7,fault-timeout,e203-skip}/`：本报告对应的真实运行证据。
- 本文件 `WORKER_REPORT.md`。

未改动任何 RTL（`hardware/**`）、主测试脚本（`hardware/testbenchplus/run_all_tests.sh`、`hardware/rtl/**/tb/run*.sh`）、`software/**` 源码、`.opencode` 钩子或其他 worktree。

## 实际执行的命令与退出码

| 命令 | 退出码 | 结果 |
| --- | ---: | --- |
| `python3 tools/repro/run.py`（= `--level smoke`，--out results/smoke） | **0** | `counts={PASS:5,FAIL:0,TIMEOUT:0,SKIP:0,ERROR:0} gate_ok=True` |
| `python3 tools/repro/run.py --level full`（--out results/full） | 0 | `counts={PASS:6,FAIL:4,TIMEOUT:1,SKIP:1} gate_ok=True`；gate 全 PASS，非 gate 失败如实记录 |
| `python3 tools/repro/run.py --only fault.probe_exit7` | **1** | 受控故障：`FAIL exit_code=7`，`gate_ok=False` |
| `python3 tools/repro/run.py --only fault.probe_timeout` | **1** | 受控故障：`TIMEOUT exit_code=None timed_out=True`，`gate_ok=False` |
| `python3 tools/repro/run.py --only e203.full_layer_208tile` | 0 | `SKIP`，原因 `missing prerequisite path(s): 资料/e203_hbirdv2_full` |

## smoke 逐项真实结果（results/smoke/summary.json）

| 检查 | 状态 | 退出码 | 耗时(s) | 输入文件数(含 sha256) |
| --- | --- | ---: | ---: | ---: |
| env.versions | PASS | 0 | 0.11 | 0 |
| sw.offline_encoder_contract | PASS | 0 | 0.17 | 15 |
| rtl.dma_iverilog | PASS | 0 | 0.03 | 5 |
| rtl.dma_2d_verilator | PASS | 0 | 0.42 | 6 |
| rtl.control_verilator | PASS | 0 | 2.48 | 19 |

软件检查的真实指标（非文件存在性）：6 条冻结样本，`worst_max_abs_error=9.54e-7`，容差 `1e-5`，`failures=[]`。
环境：Python 3.10.12 / Icarus Verilog 11.0 / Verilator 5.030 / Yosys 0.9；git `56b8a39`。

## 失败传播与机器可读性验证（受控故障）

- `fault.probe_exit7` 用 `python3 -c "...SystemExit(7)"` 制造受控非零退出：运行器记 `status=FAIL`、`exit_code=7`、整体退出 1。
- `fault.probe_timeout` 用 `sleep 30` 配 2 秒硬超时：记 `status=TIMEOUT`、`exit_code=None`、`timed_out=True`、整体退出 1。
- 两者默认 `enabled=false`，只在 `--only` 下启用，不污染 smoke。
- 全部证据写入 `summary.json`，可用 `json.load` 直接读取字段 `status/exit_code/timed_out/duration_s/inputs[].sha256/gate_ok`。

## 发现的现存问题（如实记录，不在本任务范围内修复）

- `hardware/testbenchplus` FP32 向量用例在本环境不终止：`vectors.all` 60 秒硬超时记 TIMEOUT。
- `vectors.int16` 在 Icarus 下值不匹配（`got/expected=(0,12)`）。
- Icarus 路径失败：`rtl.pulse_array_iverilog`（TIMING FAIL）、`rtl.control_iverilog`（无法展开 `kwt_product_array` 的线网数组）、`rtl.engine_iverilog`（`Cannot return from tasks`）；Verilator 路径为当前可复现路径。
- `software/validation/real_input_chain_20261017/manifest.json` 的内容哈希与提交的 `.npy` 文件不一致，但数值与 `convert` 一致；本检查记为 `manifest_hash_mismatches`（信息项，不 gate）。
- 208 块整层 E203 仿真因缺 `资料/e203_hbirdv2_full` 而 SKIP；`riscv64-linux-gnu-gcc` 实际存在。

## 覆盖边界（重要）

`smoke` 只证明上述小检查在当前工作树可复现，**不是比赛验收**：不覆盖本科必做的 12 层 FFN1 数值门槛（3% NRMSE）、RTL 逐位一致、SoC 正式总线/NICE 接入、150 MHz 时序、面积/功耗，也不覆盖官方 600 条评测集。smoke 不运行 208 块整层全仿真，也不做大批量模型采样。依赖：`python3+numpy`、`iverilog/vvp`、`verilator`；full 的 E203 项另需组委会/上游资料与 RISC-V 工具链。

## 工作树状态

只新增 `tools/repro/**` 与本报告，随后提交；提交后 `git status --short` 为空。
