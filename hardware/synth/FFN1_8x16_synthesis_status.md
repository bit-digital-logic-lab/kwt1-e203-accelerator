# FFN1 8×16 计算核综合 / 时序状态（截至 2026-10-18）

本文件只记录**已实际跑出的事实**。凡未跑通、未产出报告的项目一律标注为「未证」。
证据目录：`/tmp/kwt-8x16-sta/`（首次实跑产物），脚本：
`hardware/synth/run_ffn1_8x16_sta.sh`。

## 1. 实测工具与库（来自真实版本输出）

| 项目 | 实测值 |
| --- | --- |
| Yosys | `Yosys 0.63+97 (git sha1 1111a401f, clang++ 18.1.8 -fPIC -O3)`，路径 `/home/yang/yosys-sta/oss-cad-suite/bin/yosys` |
| iSTA (iEDA) | `Git version: 071226fa6d52375d3a365605751a3413ea51cd41`，路径 `/home/yang/yosys-sta/bin/iEDA` |
| Nangate45 库 | `Nangate45_typ.lib`（示例库，**非**组委会工艺库） |
| icsprout55 库 | `ics55_LLSC_H7CL_ss_rcworst_1p08_125_nldm.lib`（示例库，ss/rcworst/1.08V/125C，**非**组委会工艺库） |
| STA 脚本 | `/home/yang/yosys-sta/scripts/sta.tcl`（第三方示例脚本） |

顶层对象：`kwt_systolic_cacc_tile`
RTL：`kwt_product_array.sv` + `kwt_cacc.sv` + `kwt_systolic_cacc_tile.sv`

## 2. 粗粒度资源计数（已产出，`coarse.log`，退出码 0）

`read_verilog -sv; hierarchy -check; proc; opt; memory_collect; check -assert; stat`

| 指标 | 计数 |
| --- | --- |
| `kwt_product_array` 乘法器 `$mul` | 128 |
| `kwt_cacc` `$mem_v2`（部分和 bank） | 8 |
| 设计层次合计 cells | 2270 |
| 合计 `$mul` / `$add` / `$mem_v2` | 128 / 9 / 8 |
| `check -assert` | Found and reported 0 problems |

以上为「保留推断存储」的算术/存储结构计数，**不**代表时钟频率。

## 3. 已产出的网表与面积日志（示例库）

| 产物 | 大小 | 说明 |
| --- | --- | --- |
| `systolic_nangate45.v` | 29,444,768 B | Nangate45 触发器映射网表 |
| `systolic_icsprout55.v` | 28,455,863 B | icsprout55 映射网表，含 706 条 `assign` |
| `systolic_ics_noexpr.v` | 28,799,682 B | icsprout55 映射、`write_verilog -noexpr`，`assign` 数 = 0 |
| `map_nangate45.log` | — | `Chip area for module '\kwt_systolic_cacc_tile': 355178.628000` |
| `map_icsprout55.log` | — | `Chip area for module '\kwt_systolic_cacc_tile': 542054.800000` |
| `map_ics_noexpr.log` | — | 同一映射（面积同上，542054.800000） |

面积口径说明：

- 这些是**示例库**的 `Chip area for module`（含把 CACC 部分和 bank 与内部存储
  全部映射成触发器后的门级面积），只可用于**同库相对比较**。
- **严禁**把 355178.628 / 542054.800 换算成竞赛要求的 2.25 mm²：库不同、缺少题目
  规定的 16 bit SRAM 宏、且本对象只是计算核（不含 E203/总线/NICE/控制器）。
- 面积单位为示例库 Liberty 的面积单位，未经工艺换算，不构成物理实现结论。

## 4. iSTA 失败原文（未证事项的核心）

两次 iSTA 尝试均以 **SIGABRT（退出码 134）** 结束，**没有**产生任何时序报告：

- Nangate45 网表：`sta_nangate45.log`，`EXIT=134`
- icsprout55 网表：`sta_icsprout55.log`，`EXIT=134`

关键失败行（`sta_icsprout55.log`）：

```
E20261018 14:48:23.297080 42757 Sta.cc:810] the right port is not exist.
...
F20261018 14:49:09.838889 42757 Sta.cc:834] the left port is not exist.
*** Check failure stack trace: ***
timeout: the monitored command dumped core
```

- 起因：`write_verilog`（默认、含层次化 `assign` 别名）的网表在 iSTA `link_design`
  阶段解析 `assign` 时崩溃（`Sta.cc:810/818/834`）。
- 两份日志中 `slack`/`report_timing` 匹配行数均为 **0**：**没有**得到任何 slack 或
  关键路径数据。

## 5. 未证事项（不得越界声称）

1. **未证明 150 MHz**：没有任何成功的 iSTA 时序报告，`6.6667 ns` 约束只是输入目标，
   不是结论。
2. **noexpr 网表已生成，但尚未经过 STA**：`systolic_ics_noexpr.v`（0 条 `assign`）
   已产出，但截至本文件，**尚未**用它成功跑通 iSTA。是否可跑通需后续实测确认。
3. **未产出竞赛面积**：无 2.25 mm² 相关测量；无目标工艺库、无 SRAM 宏、无整机。
4. **对象范围仅计算核**：不含 A/W 操作数 SRAM、输出暂存、DMA、E203/System Bus/NICE
   接口与任务控制器。

## 6. 如何复现（仅快速步骤）

```bash
# 只跑粗粒度（秒级，退出码 0）
bash hardware/synth/run_ffn1_8x16_sta.sh --coarse-only

# 含示例库映射与 iSTA 尝试（耗时数分钟；iSTA 若仍 abort 会让脚本以非零退出）
SYNTH_OUT=/tmp/kwt-8x16-sta bash hardware/synth/run_ffn1_8x16_sta.sh
```

脚本行为约定：

- iSTA 非零退出会**透传退出码**并使脚本失败，**不会**用 `|| true` 吞掉；
- iSTA 失败时明确打印「Do NOT claim 150 MHz」；
- 只有 iSTA 退出码为 0 才解析时序报告。
