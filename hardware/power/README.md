# hardware/power — 活动/任务周期证据（非功耗估算）

本目录只新增可复现的 runner/解析脚本、新 testbench 和报告，用于在**同一冻结
A12W10、8×16 基线**上产生真实的切换活动（VCD）与任务周期证据。它不修改数值核、
`run_e203.sh` 或任何他人文件。

## 结论口径

- 产物是**切换活动与任务周期**，**不是**功耗/能耗数字。
- 无组委会 PDK、无互连功耗模型、无 SRAM 宏功耗模型，因此
  **功耗/单任务能耗一律标记 `not_computable`**。
- 未标注电路与未建模 SRAM **不能按零活动/零功耗处理**；本目录不把它们算 0。
- 示例库（Nangate45 typical）综合面积只能作相对趋势，不能用于比赛面积或功耗结论。

## 运行

```bash
bash hardware/power/run_power_activity.sh
```

- 默认向量：`hardware/power/vectors/real_e203_validation0_layer01/`
  （A12W10 layer01，来自真实验证样本 validation-index 0）。
- 会在 `hardware/power/results/` 生成 `layer01_tile000.vcd`、
  `layer01_tile207.vcd`、对应 `.log`、`compile.log` 和 `summary.json`。
- 默认跑 **tile 0 与末 tile 207**（8 行 tile 网格共 208 个），每个 tile 一个任务。
- 需要 oss-cad-suite 的 `iverilog`/`vvp` 14.0（默认
  `/home/yang/yosys-sta/oss-cad-suite/bin`）、`riscv64-linux-gnu-gcc/objcopy`、`xxd`、`python3`。
- 退出码：任何仿真的真实退出码被保留；解析任一检查不过返回非零。

## 脚本核验（`parse_vcd.py`）

对每个 VCD：

1. 波形非空、有档值段、时钟翻转；
2. 关键模块（E203 NICE FSM、tile 装载/计算/写回、计算引擎、乘积阵列、CACC、DMA）
   均有翻转，并打印每层切换计数；
3. **窗口起点**与首个 NICE 请求握手、**窗口终点**与最后一个 NICE 响应握手对应
   （由 tb 的 `marker_first_req`/`marker_last_rsp` 与原始 `nice_req/rsp` 双向核对）；
4. 从波形推导的任务周期数与仿真日志 `task_cycles` 一致；
5. 由 `tile_system.state` 提取**装载/计算/写回**三个相位窗口并核对顺序落在任务窗口内。

解析失败会写出 `verdict=FAIL` 并以非零退出，不写 PASS。

## 已知边界（未知部分）

- VCD 只 dump 精选控制/总线信号；**大行为 SRAM 阵列和 E203 数据通路内部不在 dump 内**，
  因此 `toggled_coverage_pct` 只代表“所选信号”的覆盖率，**不是全设计覆盖率**。
- 仿真时钟为 tb 的 10 时间单位周期（Icarus 默认 `timescale 1s`，因上游 E203 文件无统一
  `timescale`，显式 1ns 会导致跨 timescale 语义漂移，故不设）。周期数与频率无关；
  150 MHz 只是题目目标假设，本文只做“周期 × 1/150MHz”的算术换算，**不是已达成频率**。
- VCD 中含 `$date`，跨运行字节哈希会变；`summary.json` 内的哈希对应本次产物。
- 无正式总线/NICE 适配、无目标 SRAM 宏、无 SDC/STA、无布局布线、无隐藏 600 条评测。
- 本目录的 VCD 被 `.gitignore` 排除（`*.vcd`），提交的是脚本、日志与 `summary.json`；
  VCD 可随时用 runner 复现。
