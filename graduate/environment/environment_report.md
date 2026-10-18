# 研究生组环境检测报告（自动生成）

- 生成时间（UTC）：2026-10-18T07:47:02.041245+00:00
- 仓库根：`/home/yang/kwt-worktrees/grad-env`
- 宿主：Linux-5.15.0-164-generic-x86_64-with-glibc2.35 / x86_64 / python 3.10.12
- 结论上限：在缺官方 SoC/PDK/SRAM 宏/SDC/评测集的前提下，本环境最高只能给出 L1 功能仿真与 L2 代理工艺粗综合证据；不得据此宣称真实 SoC 或 150 MHz。

## 1. 软件工具链

| 工具 | 类别 | 存在 | 路径 | 版本/输出 | 退出码 | 备注 |
| --- | --- | :-: | --- | --- | :-: | --- |
| `python3` | python | 是 | `/usr/bin/python3` | Python 3.10.12 | 0 |  |
| `verilator` | hdl_sim | 是 | `/usr/local/bin/verilator` | Verilator 5.030 2024-10-27 rev v5.030 | 0 |  |
| `iverilog` | hdl_sim | 是 | `/usr/bin/iverilog` | Icarus Verilog version 11.0 (stable) () | 0 |  |
| `yosys_system` | synthesis | 是 | `/usr/bin/yosys` | Yosys 0.9 (git sha1 1979e0b) | 0 |  |
| `yosys_oss_cad` | synthesis | 是 | `/home/yang/yosys-sta/oss-cad-suite/bin/yosys` | Yosys 0.63+97 (git sha1 1111a401f, clang++ 18.1.8 -fPIC -O3) | 0 |  |
| `sta_ieda` | sta | 是 | `/home/yang/yosys-sta/bin/iEDA` | — | 0 | 命令退出码 0 但无版本字符串输出 |
| `sta_opensta` | sta | 否 | — | — | — | 未找到可执行文件（PATH 与已知候选均无） |
| `rv32_gcc` | rv32 | 是 | `/usr/bin/riscv64-linux-gnu-gcc` | riscv64-linux-gnu-gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 | 0 |  |
| `objdump_rv` | rv32 | 是 | `/usr/bin/riscv64-linux-gnu-objdump` | GNU objdump (GNU Binutils for Ubuntu) 2.38 | 0 |  |
| `gtkwave` | waveform | 是 | `/usr/bin/gtkwave` | GTKWave Analyzer v3.3.104 (w)1999-2020 BSI | 0 |  |

## 2. Python 运行时与模型库

| 运行时 | 路径 | 版本 | tensorflow | torch | numpy | onnx |
| --- | --- | --- | :-: | :-: | :-: | :-: |
| system | `/usr/bin/python3` | Python 3.10.12 | 否 | 否 | 1.21.5 | 否 |
| conda_base | `/home/yang/miniconda3/bin/python3` | Python 3.12.12 | 否 | 2.5.1+cu121 | 2.5.2 | 否 |
| conda_eda | `/home/yang/miniconda3/envs/eda/bin/python3` | Python 3.14.2 | 否 | 否 | 2.4.1 | 否 |
| conda_rag | `/home/yang/miniconda3/envs/rag/bin/python3` | Python 3.10.14 | 否 | 2.5.1+cu121 | 1.26.4 | 否 |
| conda_qpanda | `/home/yang/miniconda3/envs/qpanda/bin/python3` | Python 3.10.19 | 否 | 否 | 1.26.4 | 否 |

## 3. 官方资料哈希（对照 `资料/来源校验.json`）

- 哈希算法：git blob sha1 (sha1 of 'blob <size>\0' + content)（与 `资料/补全大文件.sh` 的 `git hash-object` 口径一致）
- 来源提交：{"ARM_KWT-1_237e4097": "237e4097afd4f6851e0b82f0a8d65c2dfe8eae30", "E203_Hummingbirdv2_9f3e34f": "9f3e34f337f5d0a1236a285871ffa2032491617a"}
- 检查 25 个：match=25，mismatch=0，missing=0

| 文件 | 期望 blob 哈希 | 实际 blob 哈希 | 状态 |
| --- | --- | --- | :-: |
| (全部一致) | | | ✅ |

## 4. 真实资产与接口约定

| 资产 | 存在 | 来源 | 说明 |
| --- | :-: | --- | --- |
| 官方模型源码 (237e4097) | 是 | in-repo | 比赛锁定提交的源码快照（非 git 仓库），只读参考。 |
| KWT-1 best_weights | 是 | in-repo | TF checkpoint 权重数据文件 |
| FP32 导出 npz | 是 | in-repo | 由 TF checkpoint 导出的全 162 张量 FP32 基线，不是官方评测输入。 |
| Speech Commands v0.02 | 否 | external-local/missing | Google Speech Commands v0.02；被 .gitignore 排除，需按资料/补全大文件.sh --dataset 下载。 |
| E203 子集 (9f3e34f) | 是 | in-repo | 官方锁定的 Hummingbird E203 关键文件子集（含 nice_core），用于对齐接口。 |
| E203 完整上游 | 是 | external-local | 本机独立管理的完整上游仓库（symlink 到 /home/yang/Desktop/ic设计竞赛/资料），不属于本 worktree。 |
| 官方 E203 SoC 平台 | 否 | official-missing | 委员会打包的 E203 SoC 平台（固定地址映射、共享 512KB SRAM、总线/NICE 接线与外设）不在仓库内。 |
| Nangate45 PDK | 是 | external-local | 学术 Nangate45 库，仅作通用逻辑映射/粗综合代理，不是赛题指定工艺。 |
| 官方 PDK | 否 | official-missing | 赛题指定 PDK 缺失。 |
| fakeram 通用宏 | 是 | external-local | fakeram 通用宏（45nm 学术库），不等于题目指定的 16bit×128/1024/4096。 |
| 指定 SRAM 宏 16bit×128/1024/4096 | 否 | official-missing | 题目指定的内部 SRAM 宏缺失，无法据此核算面积/端口/冲突。 |
| 官方 150MHz SDC | 否 | official-missing | 官方 150 MHz SDC 缺失；仓库内没有任何 .sdc。 |

## 5. 官方平台缺失项

| 需求 | 存在 | 说明 |
| --- | :-: | --- |
| 组委会官方 E203 SoC 平台（System Bus / NICE 接线、地址映射、共享 512KB SRAM） | 否 | 第 3 项 SoC 集成、第 6 项十二层调度、第 7 项综合与性能都以官方平台为验收对象。 |
| 组委会指定工艺 PDK 与标准单元库（用于 150 MHz / 面积签核） | 否 | 官方给出的 150 MHz 与 ≤2.25 mm² 只有在指定工艺库下才有意义。 |
| 指定内部 SRAM 宏：16 bits × 128 / 1024 / 4096 单口或双口 | 否 | 面积、端口冲突和每拍读写能力必须按指定宏核算，不能拿通用 fakeram 代替。 |
| 官方 150 MHz 时序约束（SDC）与综合脚本 | 否 | 没有官方时钟/输入输出延迟定义就无法做可比的 150 MHz 时序检查。 |
| 官方统一评测集（600 条样本）与官方评测脚本 | 否 | 本科/研究生数值门槛与准确率必须按官方 600 条口径复算。 |

## 6. 可验证到哪一层

| 层级 | 能力 | 状态 | 依据 |
| --- | --- | :-: | --- |
| L1_kernel_functional | 计算核/输出级功能仿真（Verilator/Icarus，逐位对照软件整数模型） | achievable | 不依赖官方 SoC/PDK/SDC。 |
| L2_module_synthesis_proxy | 模块级逻辑综合（Yosys 通用库，或 iEDA + Nangate45 代理库） | achievable | 结果为代理工艺下的相对参考，不是赛题指定 PDK 的签核数据。 |
| L3_e203_rtl_simulation | E203 RTL 仿真接入（Hummingbird E203 固定提交，含 NICE/总线仿真模型） | partial | 缺少委员会官方 SoC 封装（地址映射、共享 SRAM、外设），不等于官方平台仿真。 |
| L4_official_timing_signoff | 150 MHz 时序签核 | blocked | 代理库/示例 SDC 下的频率数字不能当作 150 MHz 达标。 |
| L5_area_power_signoff | ≤2.25 mm² 面积与功耗/能耗签核 | blocked | 不能用通用 fakeram 面积或不同配置拼出面积/功耗结论。 |
| L6_official_eval_accuracy | 官方 600 条评测集上的 NRMSE / 准确率 | blocked | 480 软件 / 240 RTL / 官方 600 是三个不同统计，不可混用或外推。 |

## 7. 口径声明

**可以宣称：**
- 在 Verilator/Icarus 上的模块级与计算核级功能仿真结果，以及逐位对照软件整数模型的一致性。
- 在 Yosys 通用库或 iEDA+Nangate45 代理库下的相对逻辑综合/面积量级，注明为代理工艺。
- 官方锁定提交（KWT-1 237e4097 / E203 9f3e34f）源码快照与权重的哈希一致性。
- Hummingbird E203（固定上游）RTL 级 CPU 仿真中的指令流/接口行为。

**不能宣称：**
- 在官方 E203 SoC 平台完成 SoC 集成仿真——官方 SoC 封装缺失。
- 达到 150 MHz 时序——无官方 PDK/库/SDC，代理约束不成立。
- 满足 ≤2.25 mm² 面积与功耗签核——无指定 SRAM 宏与官方工艺。
- 官方 600 条评测集上的 NRMSE/准确率——评测集未发布。
- 任何基于模拟模型/代理库的数字代表真实 SoC 或真实工艺结果。

