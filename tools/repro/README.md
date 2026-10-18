# tools/repro — 项目复现入口

本目录是 KWT-1 本科加速器工程的**可复现入口**，把当前工作树里已经存在的稳定小测试
组织成两级真实运行，并保存机器可读证据。它不替代任何核心 RTL、主测试脚本或比赛验收。

```
smoke  默认。只跑小型、快、依赖少的检查，预期在干净检出中通过（约 5–15 秒）。
full   必须显式 `--level full`。在 smoke 之上追加较重的仿真、当前会超时/失败的检查
       和资源受限时的跳过项；失败也会被如实记录。
```

## 用法

```bash
# 默认 smoke（在仓库根目录执行）
python3 tools/repro/run.py

# 显式全量
python3 tools/repro/run.py --level full

# 列出检查
python3 tools/repro/run.py --list

# 只跑指定检查（也会启用默认关闭的受控故障探针）
python3 tools/repro/run.py --only fault.probe_exit7
python3 tools/repro/run.py --only fault.probe_timeout

# 指定结果目录（缺省 tools/repro/results/<UTC时间戳>_<level>/）
python3 tools/repro/run.py --level smoke --out tools/repro/results/smoke
```

退出码：当所有 **gate** 检查为 PASS 时为 0，否则为 1。SKIP 不影响 gate（资源缺失时
记录原因，例如 208 块整层 E203 仿真所需的 `资料/e203_hbirdv2_full` 与 RISC-V 工具链）。

## 结果格式（机器可读）

每次运行写入一个目录：

- `summary.json` —— 顶层：`schema`、`generated_at`、`level`、`run_id`、`git`（commit/
  branch/dirty）、`environment`（工具版本、主机）、`counts`、`gate_ok`、`gate_ids`、
  `checks`。
- 每个检查：`id`、`level`、`gate`、`desc`、`command`、`cwd`、`started_at`、`status`
  （PASS/FAIL/TIMEOUT/SKIP/ERROR）、`exit_code`、`expected_exit`、`timed_out`、
  `duration_s`、`timeout_s`、`inputs`（每个输入文件的路径、sha256、字节数）、
  `stdout_log`、`stderr_log`、`stdout_tail`、`stderr_tail`、`parsed_result`。
- `<check>.stdout.log` / `<check>.stderr.log` —— 原始输出。

约定：**PASS 只看进程真实退出码**。超时单独记为 `TIMEOUT`（也属于失败），输出文件
存在本身不构成 PASS。检查脚本可通过一行 `RESULT_JSON:{...}` 追加结构化指标，运行器
解析进 `parsed_result`。

## 本目录包含

| 路径 | 说明 |
| --- | --- |
| `run.py` | 运行器与检查清单（manifest 内嵌，含 level/gate/超时/输入哈希/前置条件）。 |
| `checks/env_versions.py` | 记录 python/iverilog/vvp/verilator/yosys 版本。 |
| `checks/offline_encoder_contract.py` | 纯 numpy 回放 `software/offline_encoder_input.convert`，与冻结样本逐值比较。 |
| `results/` | 本报告对应的真实运行证据（smoke、full、受控故障、E203 跳过）。 |

## 检查覆盖

smoke（gate）：

1. `env.versions` —— 工具链版本快照。
2. `sw.offline_encoder_contract` —— 用不变更的 `offline_encoder_input.convert` 对
   `software/validation/real_input_chain_20261017` 的 6 条冻结样本回放，逐值比较最大
   绝对误差（容差 1e-5）。不是文件存在性检查。
3. `rtl.dma_iverilog` —— DMA 装载/写回、尾拍、边界、非法描述符（Icarus）。
4. `rtl.dma_2d_verilator` —— 二维步长 DMA 原型（Verilator）。
5. `rtl.control_verilator` —— 单块系统模型（含乘积阵列 / CACC / ping-pong 引擎），
   Verilator。

full 追加：

6. `rtl.pingpong_cacc_verilator` —— 256 个 INT32 结果的逐项核对（gate）。
7–11. 当前失败或超时的检查，`gate=false`，但仍真实运行并记录：
   `vectors.int16`（值不匹配）、`vectors.all`（FP32 用例不终止 → TIMEOUT）、
   `rtl.pulse_array_iverilog`、`rtl.control_iverilog`、`rtl.engine_iverilog`。
12. `e203.full_layer_208tile` —— 依赖缺失时 SKIP，否则按 3600 秒超时运行 208 块整层。

受控故障探针（默认关闭，仅 `--only` 启用，用于验证失败传播）：
`fault.probe_exit7`（退出码 7）、`fault.probe_timeout`（超时）。

## 依赖

- `python3` + `numpy`（smoke 的软件检查只需 numpy；不需要 torch/tensorflow）。
- `iverilog` + `vvp`（DMA Icarus 检查）。
- `verilator`（系统模型、DMA 2D、ping-pong CACC 检查）。
- full 的 208 块整层检查另需 `资料/e203_hbirdv2_full` 与 `riscv64-linux-gnu-gcc/
  objcopy/xxd`；缺失即 SKIP，不算通过。

两级检查都不访问网络、不训练模型、不做大批量采样。

## 覆盖边界（重要）

- **smoke 不是比赛验收。** 它只证明所选小检查在当前工作树上可复现；不覆盖本科必做
  的 12 层 FFN1 数值门槛（3% NRMSE）、RTL 逐位一致、SoC 正式总线/NICE 接入、
  150 MHz 时序、面积/功耗或官方 600 条评测集。
- smoke 不运行 208 块整层 E203 全仿真，也不做大批量模型采样；这些都是 full 或工作树
  之外的产物。
- Icarus 下的阵列/系统测试在本快照与工具链组合中失败（`Cannot return from tasks`、
  `Unable to assign words of unresolved wire array`、TIMING FAIL）；Verilator 路径才是
  当前可复现路径。full 会如实报告这些失败，`gate=false`。
- `hardware/testbenchplus` 的 FP32 向量用例在本环境不终止（60 秒超时），因此
  `vectors.all` 记 TIMEOUT；`vectors.int16` 值不匹配。
- `sw.offline_encoder_contract` 的 manifest 内容哈希与提交的 `.npy` 文件不一致
  （数值仍一致，报告为 `manifest_hash_mismatches`，不 gate）。
- 所有版本、退出码、耗时与输入 sha256 均来自真实运行，见 `results/`。
