# real_e203_validation0 结果整理报告

**结论：失败。12 层全部失败（通过 0/12），不构成 PASS。**

依据文件（均为实际产物）：

- `hardware/architecture/results/real_e203_validation0_finalizer.status` → `SUMMARIZE_EXIT 1`
- `hardware/architecture/results/real_e203_validation0_summary_run.log` → `ValueError: layer01 missing log or clean exit`（`summarize_real_e203.py:40`）
- `hardware/architecture/results/real_e203_validation0_logs/layer01..12.{log,exit}`
- `hardware/architecture/results/real_e203_validation0_vectors/manifest.json`

## 通过 / 失败层数

| 层 | 退出码 | 日志结果 |
| --- | --- | --- |
| layer01–layer12（12/12） | 全部为 `1` | 每层一条 `FATAL`，无 `PASS` 行 |

12 个 `.exit` 文件内容全为 `1`。12 份日志除层名与向量路径外内容完全一致，说明失败与具体层数据无关，是系统性问题。

汇总脚本因此未产出 `real_e203_validation0_summary.json`（该文件不存在），`finalizer.status` 记为 `SUMMARIZE_EXIT 1`。

## 具体失败层与原因

每层均在第一个被检查的输出元素上中止：

```
FATAL: hardware/rtl/control/tb/tb_kwt_real_layer_e203.sv:51:
       layer output [0,0] got=xxxxxxxx expected=00000000
       Time: 8380510  Scope: tb_kwt_real_layer_e203.check_tile
```

- 失败位置：tile 0 的输出 `[row=0,col=0]`，即全部 12 层的第一个校验点。
- 直接原因：读出值 `xxxxxxxx`，示例共享 RAM 中输出区（`OUT_BASE=0x10000`）为未定义态，说明权重/描述符路径虽被驱动，但加速器/DMA 的结果写回没有在仿真结束前把有效数据落到输出缓冲区。
- 触发 `$fatal` 后仿真立即结束，后续元素不再校验。

## 有效输出 / 填充校验

- 计划值（来自向量 manifest 与 TB 常量）：每层 25344 个有效输出 + 1280 个填充位；12 层合计 304128 个有效输出、15360 个填充位。
- 实际值：**0**。每层都在第一个有效输出处中止，日志中没有 `outputs=... padding=...` 统计，校验计数全部未完成。

## 周期

- 日志显示每层首个 NICE 请求在 cycle 7，tile 0/1/2/3 完成于 cycle 4031 / 8060 / 12089 / 16118，进度打印到达 `192/208`，随后中止于 `Time: 8380510`（约 838051 个时钟周期）。
- 12 层打印的周期数完全相同（固定调度、与数据无关）。
- 由于没有 `PASS` 行，`task_cycles` / `total_cycles` 未被记录；只能确认每层计算阶段约跑到 83.8 万周期。

## 数值误差

向量 manifest 记录的是生成侧（CPU，FP64 解码）指标，不是 DUT 实测值，仅供参考：

- 12 层 NRMSE 为 0.1017%–0.1349%，最差层（layer11）0.13493030006895154%。
- 数值协议：A12W10、对称零点 0、rint 偶数舍入、有符号 INT32 GEMM+bias。
- 每层 `max_abs_expected_int32` 最大 4593325（layer01），保守累加器界最大 67780536，均远小于 INT32 上限。
- 本次 DUT 输出为 `x`，因此没有任何实测误差可报告。

## CPU 向量复现范围

- `validation_index = 0`，样本 `./four/750e3e75_nohash_0.wav`，`label = _unknown_`，训练/验证计数各 480。
- 张量形状：X 99×64、W 64×256、bias 256、Y 99×256；12 层逐层向量，每层 4 个文件：a16 6656 元素、w16 16384 元素、bias32 256 元素、expected32 25344 元素。
- 向量由 CPU 生成，manifest 记录了各文件 SHA-256 及样本/参数/fp32 权重归档哈希，可作为可追溯的复现依据。
- 本次运行只复现了“逐层全层”这一种范围（每层 208 个 tile），未覆盖多样本批量或多层端到端流水复现。

## 边界说明

本结果来自 Icarus 仿真下的 **E203 + 临时 NICE-like 加速器模型**（模拟共享 RAM、旧 8×16 计算引擎与仿真 DMA/总线），**不是组委会正式 SoC 或 PDK**。没有任何综合、时序、面积、功耗或正式平台数据；这些数字不能作为对组委会评估平台的结论。

另注：每层日志有 4 条 `$readmemh` 路径含非打印字符的警告（路径中 `设计竞赛` 段以 `\377` 显示），日志中未出现文件打开失败错误。

## 后续建议（仅记录，未执行）

1. 定位输出写回路径（tile_system 的 STORE_OUT → DMA writeback）为何未写入 `OUT_BASE`。
2. 修复后重跑，需 12 层退出码均为 0 且每层恰好一条全层 `PASS`，汇总脚本才会通过。
