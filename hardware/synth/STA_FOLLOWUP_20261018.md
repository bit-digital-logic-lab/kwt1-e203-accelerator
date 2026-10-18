# FFN1 8×16 `noexpr` 网表 iSTA 复核（2026-10-18）

本文件只记录**本轮真实执行**的退出码、日志和报告内容。结论边界：示例库、
计算核、探索性时序；**不是** 150 MHz 达标证明，也**不是**正式 PDK/整机签核。

## 1. 结论先行

| 问题 | 实测答案 |
| --- | --- |
| iSTA 能否解析 `systolic_ics_noexpr.v` | **能**。`link_design` 完成，`inst num: 238537`、`net num: 238989`、`port num: 486` |
| 时钟端口是否生效 | **是**。日志有 `create clock clk for pin/port: clk`，报告中时钟组为 `clk`，setup required 为 `6.596 ns` |
| 是否真的产生 slack / report_timing | **是**。报告文件存在且含 `Slack`、`TNS`、逐级 path 表、`slack (VIOLATED)` / `slack (MET)` |
| iSTA 退出码 | **0**（两次 bounded 运行均为 0） |
| 150 MHz 是否达标 | **不能这样声称**。该约束下 worst setup slack = **-12.298 ns**，报告估计 **52.732 MHz** |
| 报告性质 | **示例库/计算核探索性 STA**；非签核、非物理实现结论 |

在最坏 setup 路径上：

- data arrival time：`18.894 ns`
- data required time：`6.596 ns`（`6.6667 ns` 周期减 `0.070 ns` library setup）
- slack：`-12.298 ns`（VIOLATED）
- TNS（max）：`-2770.397 ns`
- worst hold slack：`+0.092 ns`
- 报告 `Freq(MHz)` 列：`52.732`

这些数字只说明该示例库、该扁平化计算核、该理想时钟约束下的静态计算结果，
不能外推到竞赛 150 MHz、2.25 mm² 或整机性能。

## 2. 对象与边界

- 顶层：`kwt_systolic_cacc_tile`
- RTL：`kwt_product_array.sv` + `kwt_cacc.sv` + `kwt_systolic_cacc_tile.sv`
- 范围：仅 8×16 脉动乘加计算核（128 个 16×16 有符号乘法 + 8 个部分和 bank）
- **不含**：A/W 操作数 SRAM、输出暂存、DMA、E203/System Bus/NICE、任务控制器
- 工艺库：`icsprout55` 示例库（ss/rcworst/1.08V/125C），**非**组委会正式 PDK
- 面积数字（`542054.800000`）仍是示例库 `Chip area for module`，不得换算成 2.25 mm²

## 3. 输入、工具与 SHA256

| 项目 | 值 |
| --- | --- |
| 网表 | `/tmp/kwt-8x16-sta/systolic_ics_noexpr.v` |
| 网表 SHA256 | `a983c2b09595f41f18e7aee7096bf0aa54400803aeb7293c27360f18b5871cd8` |
| 网表 `assign` 数 | `0` |
| SDC | `/tmp/kwt-8x16-sta/ffn1_8x16_150mhz.sdc` |
| SDC SHA256 | `cbc3759efd91ba0609492ec1849be696273ca1ab2e6ae3b351bc6bc949c53fdc` |
| iSTA 二进制 | `/home/yang/yosys-sta/bin/iEDA` |
| `iEDA -v` | `Git version: 071226fa6d52375d3a365605751a3413ea51cd41` |
| STA Tcl | `/home/yang/yosys-sta/scripts/sta.tcl`（第三方示例脚本） |
| PDK 选择 | `icsprout55` |

注意：报告头写的是 `GitVersion: aa25008b3778a76ae1975c839d4cc459f7f7b3b7`，
与 `iEDA -v` 的 `071226fa...` 不一致。本报告同时保留两个值，不替工具做版本归一化。

## 4. 实际运行

### 4.1 手工 bounded 运行（第一次）

```bash
timeout 240 /home/yang/yosys-sta/bin/iEDA -script /home/yang/yosys-sta/scripts/sta.tcl \
  /tmp/kwt-8x16-sta/ffn1_8x16_150mhz.sdc \
  /tmp/kwt-8x16-sta/systolic_ics_noexpr.v \
  kwt_systolic_cacc_tile icsprout55
```

- 墙钟：约 147 s（15:21:25 → 15:23:52）
- 退出码：`0`（`/tmp/kwt-8x16-sta-followup/noexpr_sta.exit` 记录 `EXIT=0`）
- 日志：`/tmp/kwt-8x16-sta-followup/noexpr_sta.log`
- 日志 SHA256：`3260d7c528c74277fcc751bf6475807cc144dfc6e40198aca31bc6d6cba27fe0`
- 报告：`/tmp/kwt-8x16-sta/kwt_systolic_cacc_tile.rpt`
- 报告 SHA256：`55dd5d9b1809a41ccba68efeae9fb4562c6ac2c0a1884d891351cef40ff90eeb`

### 4.2 可复现脚本运行（第二次，用于验证脚本）

```bash
OUT=/tmp/kwt-8x16-sta-followup \
  bash hardware/synth/run_sta_noexpr_followup.sh
```

- 墙钟：约 175 s（15:24:46 → 15:27:41），脚本内 `timeout 240`
- 退出码：`0`
- 脚本 stdout：`/tmp/kwt-8x16-sta-followup/run_script_stdout.txt`
- 日志：`/tmp/kwt-8x16-sta-followup/kwt_systolic_cacc_tile_150mhz_noexpr_sta.log`
- 日志 SHA256：`611fec58002a858726dd8df4ead073747234d638ce8135f6de5c30fc8d75b599`
- 报告：`/tmp/kwt-8x16-sta-followup/kwt_systolic_cacc_tile.rpt`
- 报告 SHA256：`24c64ef532e72b6c96039faecb25d543200981f3db8dc36bb099b4ad881bfc7d`
- 版本记录：`/tmp/kwt-8x16-sta-followup/kwt_systolic_cacc_tile_noexpr_versions.txt`
  SHA256 `218d95b88a918d480d7fbc42d6c5fad79be243285a311e59421cb73307c5aaea`

两次运行的 slack、required/arrival、TNS 数值一致；报告文件 SHA 不同是因为
最坏路径端点排序不同，不是时序数值变化。

## 5. 关键证据原文

### 5.1 解析、端口与时钟

```text
link design kwt_systolic_cacc_tile end
design kwt_systolic_cacc_tile inst num: 238537
design kwt_systolic_cacc_tile net num: 238989
design kwt_systolic_cacc_tile port num: 486
create clock clk for pin/port: clk
The timing engine run success.
```

### 5.2 报告头部（脚本运行）

```text
| Endpoint   | Clock Group | Delay Type | Path Delay | Path Required | CPPR  | Slack   | Freq(MHz) |
| _450413_:D | clk         | max        | 18.894r    | 6.596         | 0.000 | -12.298 | 52.732    |
| _461198_:D | clk         | min        | 0.131r     | 0.039         | 0.000 | 0.092   | NA        |
| Clock | Delay Type | TNS       |
| clk   | max        | -2770.397 |
| clk   | min        | 0.000     |
```

### 5.3 最坏 setup 路径

```text
path cell delay              18.894(100.000%)
path net delay                0.000(0.000%)
data require time             6.596
data arrival time            18.894
slack (VIOLATED)            -12.298
```

路径起点 `_458554_:CK (DFFRQX2H7L)`，终点 `_450413_:D (DFFQX1H7L)`。其中
`_426714_:B0 → _426714_:Y (AOI21X0P5H7L)` 单级增量约 `14.084 ns`，其输入 net `_129096_`
fanout 为 `416`、transition 为 `23.788 ns`；这是扁平化后未做缓冲/布局的
高扇出结构，不是已物理实现的布线结果。

## 6. 为什么“能跑”仍不等于“可签核”

报告是 iSTA 真实产生的，但只在本报告的限定范围内有意义，原因如下：

1. **理想时钟、零线延迟**：日志明确为 `clock network delay (ideal)`；路径
   `path net delay = 0.000`。没有布局、时钟树、寄生参数或布线。
2. **示例库而非正式 PDK**：icsprout55 只作为可运行示例；没有题目规定的
   16 bit SRAM 宏，也没有整机环境。
3. **库表外插值警告**：日志出现 `10` 次
   `Warning: val outside table ranges: val = 0; min_val = 0.0121961; max_val = 0.795659`。
   零输入 transition 落到库表范围外，单元延迟存在外插不确定性。
4. **高扇出主导路径**：`416` fanout 的 net 造成 `23.788 ns` transition，单级
   `AOI21` 增量约 `14.084 ns`；这更像“未缓冲的全局控制/使能网”，不能当作
   物理关键路径结论。
5. **约束不完整**：只有 `create_clock` 和 `set_clock_uncertainty`，没有输入/输出
   延迟、驱动/负载、false path、多时钟或 I/O 约束。setup 路径的 required
   `6.596 ns` 是 `6.6667 - 0.070`（library setup），报告中该 max 路径没有
   单独出现 `0.05 ns` clock uncertainty 行；因此不能把该数当作保守签核值。
6. **对象不完整**：仅是计算核，不含 SRAM/DMA/E203/总线/控制器，不能代表整机。

因此：**iSTA 真实解析成功，报告真实存在，timing 真实为负；但只能标记为
示例库/计算核探索性时序，不能声称 150 MHz 达标，也不能作为正式 PDK 签核。**

## 7. 与上轮失败的关系

上轮默认 `write_verilog` 网表含层次化 `assign` 别名（706 条），iSTA 在
`link_design` 解析时 SIGABRT（退出码 134，`Sta.cc:810/834`）。本轮改用
`write_verilog -noexpr` 的 `systolic_ics_noexpr.v`（`assign` 数 0），同一 iSTA
二进制可完整解析并产出报告。结论只适用于 **noexpr 网表**，不回溯证明默认网表
可用。

## 8. 复现方式

```bash
# 前提：noexpr 网表与 SDC 已由既有综合脚本产出（本轮直接复用，未重综合）
OUT=/tmp/kwt-8x16-sta-followup \
  bash hardware/synth/run_sta_noexpr_followup.sh
```

脚本行为：

- 默认输入 `/tmp/kwt-8x16-sta/systolic_ics_noexpr.v` 与
  `/tmp/kwt-8x16-sta/ffn1_8x16_150mhz.sdc`，可用 `NETLIST` / `SDC` 覆盖；
- 强制 `timeout 240`，不重综合、不无界运行；
- 若网表仍含 `assign` 则拒绝运行（退出码 3）；
- iSTA 非零退出原样透传；超时返回 124；
- 只有退出码 0、日志出现 `The timing engine run success.`、报告存在、且日志出现
  `create clock clk for pin/port: clk` 才解析 slack；
- 即使 slack 为负，脚本也以 0 表示“分析完成”，并明确打印 `TIMING_MET=no`，
  避免把“工具跑通”误读成“时序达标”。

若网表确实缺失，才用既有综合脚本在有界预算内重建：

```bash
SYNTH_OUT=/tmp/kwt-8x16-sta timeout 240 \
  bash hardware/synth/run_ffn1_8x16_sta.sh --skip-nangate
```

## 9. 产物与提交边界

以下大文件/日志**不提交**，仅保留在本机 `/tmp` 供复核：

- `/tmp/kwt-8x16-sta/systolic_ics_noexpr.v`（28,799,682 B）
- `/tmp/kwt-8x16-sta-followup/noexpr_sta.log`（2,593,618 B）
- `/tmp/kwt-8x16-sta-followup/kwt_systolic_cacc_tile_150mhz_noexpr_sta.log`（2,593,696 B）
- `/tmp/kwt-8x16-sta-followup/kwt_systolic_cacc_tile.rpt`（53,462 B）
- `/tmp/kwt-8x16-sta-followup/kwt_systolic_cacc_tile.v`（iEDA 输出，24,974,517 B）

本轮代码改动只有：

- 新增 `hardware/synth/run_sta_noexpr_followup.sh`（有界 iSTA 复核脚本）
- 新增 `hardware/synth/STA_FOLLOWUP_20261018.md`（本文件）
- 更新 `WORKER_REPORT.md`

未修改任何已有综合脚本或 RTL 数值逻辑。
