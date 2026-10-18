# WORKER_REPORT — codex/kwt-grad-env（研究生组第 4-7 项：环境与接口约定）

- 分支：`codex/kwt-grad-env`（worktree `/home/yang/kwt-worktrees/grad-env`）
- 范围：为研究生组第 4-7 项（单头注意力、单层编码器、十二层编码器与参数调度、编码器
  加速器综合与性能）建立**环境与接口约定**：检测 Python/TF/PyTorch、HDL 仿真/综合/STA/RV32
  工具、官方资料哈希、真实 E203/PDK/SRAM 宏/SDC 是否存在；输出机器可读 JSON 与简短文档；
  实际运行 smoke test；对缺失项给可读状态。
- 约束：只改 `graduate/environment/` 与根 `WORKER_REPORT.md`；未开子智能体；未 push /
  未 merge/rebase/cherry-pick；未碰主目录与其他 worktree；未改核心数值 RTL。

## 交付物

| 文件 | 说明 |
| --- | --- |
| `graduate/environment/check_environment.py` | 只读盘点工具链/运行时/哈希/资产，写 `environment_report.json`+`.md` |
| `graduate/environment/smoke_toolchain.py` | 真实 smoke test，写 `smoke_test.json`+`.md`+`.log` |
| `graduate/environment/run_all.sh` | 顺序运行上面两个脚本 |
| `graduate/environment/environment_report.json` / `.md` | 机器可读清单 + 简短文档 |
| `graduate/environment/smoke_test.json` / `.md` / `.log` | 工具链真实退出码与原始日志 |
| `graduate/environment/README.md` | 验证边界与第 4-7 项接口约定 |

## 实跑命令与退出码（真实记录）

| 命令 | 退出码 | 结果 |
| --- | ---: | --- |
| `python3 graduate/environment/check_environment.py` | **0** | 工具 9/10 present；哈希 25/25 match，0 mismatch；官方缺失 5 项 |
| `python3 graduate/environment/smoke_toolchain.py` | **0** | pass=7 fail=0 skipped_missing_tool=0 |
| `bash graduate/environment/run_all.sh` | **0** | 两脚本顺序通过 |
| `python3 -m py_compile graduate/environment/*.py` | **0** | 编译通过 |

smoke test 明细（7/7 pass，均来自真实退出码）：

| 用例 | 解释器 | 退出码 |
| --- | --- | --- |
| Verilator 最小计数器编译+仿真 | `verilator --binary --timing` | 0 / 0 |
| Verilator 对仓库 SRAM RTL lint | `--lint-only hardware/rtl/sram/kwt_operand_pingpong.sv` | 0 |
| Icarus 最小计数器编译+仿真 | `iverilog -g2012` + `vvp` | 0 / 0 |
| Yosys 最小计数器综合 | oss-cad-suite `yosys -p "synth -top"` | 0 |
| Yosys 读项目 SystemVerilog RTL | `read_verilog -sv` + `hierarchy -check; proc` | 0 |
| iEDA 可执行 | `/home/yang/yosys-sta/bin/iEDA -version` | 0（无版本字符串，如实记录） |
| RV32 交叉编译 + 架构核对 | `riscv64-linux-gnu-gcc -march=rv32imac -mabi=ilp32` + `objdump -f` → `riscv:rv32` | 0 / 0 |

## 环境事实

- 存在：Verilator 5.030、Icarus 11.0、Yosys 0.9（系统）与 0.63+97（oss-cad-suite）、
  iEDA、riscv64-linux-gnu-gcc 11.4、objdump 2.38、gtkwave。缺独立 `OpenSTA`（STA 由 iEDA 承担）。
- Python：系统 3.10.12 仅 numpy；`conda_base`/`conda_rag` 有 torch 2.5.1+cu121；
  **任何运行时都没有 TensorFlow**。
- 官方资料哈希：`资料/来源校验.json` 25 文件 **25/25 一致**。该文件是 **Git blob 哈希**
  （`git hash-object` 口径），不是裸 SHA-1；用裸 SHA-1 会得到 25/25 全 mismatch 的错误结论。
- 真实资产：官方模型源码快照、KWT-1 checkpoint、FP32 npz、E203 子集与完整上游存在；
  Speech Commands v0.02 tar 未下载；Nangate45/icsprout55 学术 PDK 与 fakeram 通用宏仅作
  代理；仓库内**无任何 `.sdc`**。

## 软硬件边界（能否宣称真实 SoC / 150 MHz）

- **可做**：L1 计算核/输出级功能仿真（Verilator/Icarus，逐位对照软件整数模型）；
  L2 模块级综合（Yosys 通用库或 iEDA+Nangate45 **代理库**）。
- **部分**：L3 E203 RTL 仿真接入（Hummingbird 固定提交，含 NICE 仿真模型），不等于官方
  SoC 封装仿真。
- **阻塞**：L4 150 MHz 时序签核、L5 ≤2.25 mm² 面积与功耗签核、L6 官方 600 条评测集精度。
- **不能宣称**：在官方 E203 SoC 平台完成 SoC 集成仿真；达到 150 MHz；满足面积/功耗签核；
  官方评测集精度。**任何模拟模型或代理库数字都不能代表真实 SoC 或真实工艺结果。**

## 阻塞项

1. 官方 SoC 平台、指定 PDK、SRAM 宏（16bit×128/1024/4096）、150 MHz SDC、官方 600 条评测集
   全部缺失 → L4/L5/L6 无法签核。
2. 本机无 TensorFlow → 不能在本机复算官方 checkpoint；依赖既有 `kwt1_fp32.npz`。
3. Speech Commands v0.02 数据集未下载（被 `.gitignore` 排除）。
4. iEDA 可执行但不打印版本，且**未**跑完整 STA 流程；独立 `OpenSTA` 二进制缺失。

## 未证事项

- 本轮只证明工具链**可用**与本机资产清单，不证明研究生组第 4-7 项任何功能/性能指标。
- L2 代理库综合结果、L3 E203 仿真行为均不得外推为官方口径。
