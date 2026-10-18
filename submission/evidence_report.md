# 本科必做 1-3 项证据核对报告（自动生成）

- 生成时间（UTC）：2026-10-18T06:22:53.375964+00:00
- 分支 / 提交：`codex/kwt-packaging` @ `d94fb6f`
- 主目录只读参考：`/home/yang/Desktop/ic设计竞赛`
- PASS 仅用于已上机且产物齐备的条目；ENTRY/PARTIAL/FAIL/GAP 均不算通过。

## 汇总

| 状态 | 条目数 |
| --- | ---: |
| ENTRY | 2 |
| FAIL | 1 |
| GAP | 6 |
| PARTIAL | 2 |
| PASS | 7 |

## 按证据层级

### L1_num_core_verilator — 已上机数值核（Verilator RTL 计算核 + 输出暂存）

| 条目 | 状态 | 说明 |
| --- | --- | --- |
| `l1_240_a11w10` | PASS | 证据文件齐备（数值/结论层级见 note） |
| `l1_240_a12w10` | PASS | 证据文件齐备（数值/结论层级见 note） |
| `l1_real_sample_12layers` | PASS | 证据文件齐备（数值/结论层级见 note） |
| `l1_row16_staged_cycles` | PASS | 证据文件齐备（数值/结论层级见 note） |
| `l1_quant_format_compare` | PARTIAL | 有产物但无该次运行的退出码/日志 |
| `gap_sample_archive` | GAP | 样本归档被 .gitignore 排除，提交树内不存在 |

### L2_nice_like_e203 — NICE-like E203 / SoC 仿真（非组委会正式 SoC）

| 条目 | 状态 | 说明 |
| --- | --- | --- |
| `l2_run_e203_entry` | ENTRY | 仅有可执行入口，未附运行结果 |
| `l2_nice_system_model` | PASS | 证据文件齐备（数值/结论层级见 note） |
| `l2_synthetic_full_layer` | PARTIAL | 声称已跑但无提交日志（不计 PASS） |
| `l2_real_layer_run` | FAIL | 由只读外部证据判定 |

### L3_driver_host — C 驱动宿主测试 / RV32 交叉编译（非 SoC 联调）

| 条目 | 状态 | 说明 |
| --- | --- | --- |
| `l3_driver_source` | PASS | 证据文件齐备（数值/结论层级见 note） |
| `l3_host_test_fresh` | PASS | 本次运行退出码 0 |

### L4_example_lib_synth — 示例库综合（Yosys + Nangate45 示例 Liberty）

| 条目 | 状态 | 说明 |
| --- | --- | --- |
| `l4b_synth_entry` | ENTRY | 仅有可执行入口，未附运行结果 |

### GAP_official — 未获得的组委会材料 / 目标库 / 宏 / SDC / 评测集

| 条目 | 状态 | 说明 |
| --- | --- | --- |
| `gap_组委会_System_Bus_正式_SoC_地址` | GAP | 未获得组委会材料（预期缺失） |
| `gap_目标_PDK_标准单元库_非示例_Nangate` | GAP | 未获得组委会材料（预期缺失） |
| `gap_指定_16bit_128_1024_4096_S` | GAP | 未获得组委会材料（预期缺失） |
| `gap_目标_SDC_约束文件` | GAP | 未获得组委会材料（预期缺失） |
| `gap_官方_600_条评测集_评测脚本` | GAP | 未获得组委会材料（预期缺失） |

## 工具

| 工具 | 存在 | 版本 |
| --- | --- | --- |
| python3 | yes | Python 3.10.12 |
| git | yes | git version 2.34.1 |
| sha256sum | yes | sha256sum (GNU coreutils) 8.32 |
| cc | yes | cc (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0 |
| gcc | yes | gcc (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0 |
| verilator | yes | Verilator 5.030 2024-10-27 rev v5.030 |
| iverilog | yes | /usr/bin/iverilog: invalid option -- '-' |
| vvp | yes | /usr/bin/vvp: invalid option -- '-' |
| yosys | yes | /usr/bin/yosys: invalid option -- '-' |
| riscv64-linux-gnu-gcc | yes | riscv64-linux-gnu-gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 |
| riscv64-linux-gnu-objcopy | yes | GNU objcopy (GNU Binutils for Ubuntu) 2.38 |

## 官方来源完整性（git blob SHA）

| 文件 | 存在 | 一致 |
| --- | --- | --- |
| `资料/ARM_KWT-1_237e4097/models_data_v2_12_labels/kwt1/labels.txt` | True | True |
| `资料/ARM_KWT-1_237e4097/models_data_v2_12_labels/kwt1/best_weights.data-00000-of-00001` | True | True |
| `资料/ARM_KWT-1_237e4097/models_data_v2_12_labels/kwt1/flags.json` | True | True |
| `资料/ARM_KWT-1_237e4097/kws_streaming/layers/speech_features.py` | True | True |
| `资料/E203_Hummingbirdv2_9f3e34f/rtl/e203/core/e203_exu_nice.v` | True | True |

## 参数 / 权重哈希核对

| 文件 | 期望 | 实际 | 一致 |
| --- | --- | --- | --- |
| `software/weights/kwt1_fp32.npz` | `8f359abf7adb…` | `8f359abf7adb…` | True |
| `software/results/ffn1_blackbox_a11w10.npz` | `bc7aff59fadf…` | `bc7aff59fadf…` | True |
| `software/results/ffn1_a12w10_static_sym.npz` | `a77f60fb7c85…` | `a77f60fb7c85…` | True |
| `software/weights/kwt1_fp32.npz` | `8f359abf7adb…` | `8f359abf7adb…` | True |

## 样本归档

- 记录哈希：`ee04e849bc407149618e4ecd5ff99575687586d4e57c0ade6a0b3f7b66c8c149`
- 状态：**GAP** — 仓库内 software/.build 被 .gitignore 排除，实际样本归档缺失；主目录有同哈希副本，但不在提交树中，复现依赖外部文件。

## 未获得的官方材料扫描（预期缺失）

| 材料 | 状态 | 命中文件 |
| --- | --- | --- |
| 组委会 System Bus / 正式 SoC 地址图 | ABSENT_AS_EXPECTED | （无） |
| 目标 PDK / 标准单元库（非示例 Nangate45） | ABSENT_AS_EXPECTED | （无） |
| 指定 16bit×128/1024/4096 SRAM 宏（liberty/verilog 宏模型） | ABSENT_AS_EXPECTED | （无） |
| 目标 SDC 约束文件 | ABSENT_AS_EXPECTED | （无） |
| 官方 600 条评测集 / 评测脚本 | ABSENT_AS_EXPECTED | （无） |

## 快速宿主测试

| 测试 | 状态 | 退出码 |
| --- | --- | --- |
| `c_driver_host_test` | PASS | 0 |
| `c_driver_rv32_compile` | PASS | 0 |

```text
# c_driver_host_test
PASS provisional NICE FFN1 descriptor layout and bounds
```

