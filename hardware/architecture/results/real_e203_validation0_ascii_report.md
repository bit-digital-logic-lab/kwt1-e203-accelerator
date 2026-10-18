# real_e203_validation0（ASCII 路径重跑）结果整理报告

**结论：12 层全部通过（通过 12/12，失败 0/12）。**

依据文件（均为实际产物）：

- `hardware/architecture/results/real_e203_validation0_ascii_finalizer.status` → `SUMMARIZE_EXIT 0`
- `hardware/architecture/results/real_e203_validation0_ascii_summary_run.log` → `PASS 12 full E203 tasks, 304128 outputs, 15360 padding positions`
- `hardware/architecture/results/real_e203_validation0_ascii_logs/layer01..12.{log,exit}`
- `hardware/architecture/results/real_e203_validation0_ascii_summary.json`
- `hardware/architecture/results/real_e203_validation0_vectors/manifest.json`

## 通过 / 失败层数

| 层 | 退出码 | 日志结果 |
| --- | --- | --- |
| layer01–layer12（12/12） | 全部为 `0` | 每层恰好一条全层 `PASS` 行 |

12 个 `.exit` 文件内容全为 `0`。每层日志末行的 `PASS` 记录完全一致：

```
PASS frozen real-sample E203 FFN1 tiles: start=0 count=208 outputs=25344 padding=1280 task_cycles=838028 total_cycles=838041
```

12 层各覆盖 208 个 tile。汇总脚本据此产出 `real_e203_validation0_ascii_summary.json`，`finalizer.status` 记为 `SUMMARIZE_EXIT 0`。无失败层，无 `FATAL`/`mismatch`/`error` 记录。

## 有效输出 / 填充校验

- 每层：`outputs=25344`、`padding=1280`。
- 12 层合计：有效输出 **304128**，填充位 **15360**；与 `summary.json` 中 `checked_outputs=304128`、`checked_padding=15360` 一致。
- TB 源为 `hardware/rtl/control/tb/tb_kwt_real_layer_e203.sv`（请求于 cycle 7 开始，进度打印至 `192/208 selected tiles completed`）。

## 周期

- 每层：`task_cycles=838028`、`total_cycles=838041`（与数据无关的固定调度，12 层数值完全相同）。
- 12 层 `task_cycles` 合计 **10056336**（`summary.json` 的 `total_task_cycles`）。
- 每层日志观察到的 tile 完成时刻：tile 0/1/2/3 = cycle 4031 / 8060 / 12089 / 16118，仿真正式结束于 `Time: 8380510`。

## 数值误差

- 本次为 DUT 输出与期望 INT32 的逐元素精确比对（`expected32`），12 层全部通过，即实测误差为 0。
- manifest 另记录生成侧（CPU，FP64 解码仅用于 NRMSE）指标，属参考值：12 层 NRMSE 为 **0.1017%–0.1349%**，最差层 **layer11 = 0.13493030006895154%**（与 `summary.json` 的 `worst_layer_nrmse_pct` 一致）。
- 数值协议：A12W10、对称零点 0、rint 偶数舍入、有符号 INT32 GEMM+bias。
- `max_abs_expected_int32` 最大 4593325（layer01）；保守累加器界最大 67780536（layer04），均远小于 INT32 上限，无溢出风险。

## CPU 向量复现范围

- `generation_device = "cpu"`，`validation_index = 0`，样本 `./four/750e3e75_nohash_0.wav`，`label = _unknown_`，训练/验证计数各 480。
- 张量形状：X 99×64、W 64×256、bias 256、Y 99×256。
- 12 层逐层向量，每层 4 个文件：a16 6656 元素（13×64×8）、w16 16384 元素（16×64×16）、bias32 256 元素、expected32 25344 元素；manifest 记录各文件 SHA-256，以及样本/参数/fp32 权重归档哈希，可作可追溯的复现依据。
- 本次只复现“逐层全层”一种范围（每层 208 个 tile），未覆盖多样本批量复现，也未做多层端到端流水复现。

## 边界说明

本结果来自 Icarus 仿真下的 **E203 + 临时 NICE-like 加速器模型**（模拟 512 KiB 共享 RAM、旧 8×16 计算引擎与仿真 DMA/总线），**不是组委会正式 SoC 或 PDK**。没有任何综合、时序、面积、功耗或正式平台数据；这些数字不能作为对组委会评估平台的结论。

另：本次为 ASCII 路径重跑，`summary.json` 的 `log_sha256` 记录了各层日志哈希；未出现上一轮非 ASCII 运行中的路径非打印字符问题。

## 备注（仅记录，未执行）

- 通过与失败的差别仅在于本轮使用 ASCII 路径后输出写回正常；上一轮失败记录见 `real_e203_validation0_report.md`（那轮的 `finalizer.status` 为 `SUMMARIZE_EXIT 1`，DUT 输出为 `x`）。
