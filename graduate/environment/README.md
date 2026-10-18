# 研究生组第 4-7 项：环境与接口约定

本目录只做**环境盘点与验证边界**，不代表研究生组第 4-7 项（注意力、单层编码器、十二层
调度、编码器加速器综合与性能）已经完成。它回答两个问题：

1. 当前机器上能以哪一层证据支撑第 4-7 项？
2. 哪些验收项因**官方资产缺失**而无法宣称？

## 1. 文件

| 文件 | 作用 |
| --- | --- |
| `check_environment.py` | 只读盘点：Python/TF/PyTorch、HDL 仿真/综合/STA/RV32 工具、官方资料哈希、真实资产是否存在。写 `environment_report.json` + `environment_report.md`。 |
| `smoke_toolchain.py` | 真实跑最小用例与项目 RTL：Verilator/Icarus/Yosys/iEDA/RV32，记录真实退出码。写 `smoke_test.json` + `smoke_test.md` + `smoke_test.log`。 |
| `run_all.sh` | 依次运行上面两个脚本并打印摘要。 |
| `environment_report.json/.md` | 资产清单与能力边界（机器可读 + 人类可读）。 |
| `smoke_test.json/.md/.log` | 工具链 smoke test 结果与原始日志。 |

## 2. 运行

```bash
cd /home/yang/kwt-worktrees/grad-env
bash graduate/environment/run_all.sh
# 或分开跑
python3 graduate/environment/check_environment.py
python3 graduate/environment/smoke_toolchain.py
```

脚本不安装、不下载、不改仓库；缺失项在 JSON 里是 `present=false` / `official-missing`，
工具缺失时 smoke 记为 `skipped_missing_tool`，**不会写 PASS**。

## 3. 本机实测结论（2026-10-18）

- 工具链 9/10 存在：Verilator 5.030、Icarus 11.0、Yosys 0.9（系统）与 0.63（oss-cad-suite）、
  iEDA（`/home/yang/yosys-sta/bin/iEDA`）、riscv64-linux-gnu-gcc 11.4、objdump、gtkwave。
  缺 `OpenSTA` 独立可执行（由 iEDA 承担 STA/综合流程）。
- 官方资料哈希：`资料/来源校验.json` 的 25 个文件 **25/25 一致**。注意该文件用的是
  **Git blob 哈希**（`git hash-object` 口径），不是裸文件 SHA-1。
- smoke test：7/7 pass（Verilator 独立仿真 + 项目 SRAM RTL lint；Icarus 仿真；Yosys 独立综合
  与读项目 SV；iEDA 可执行；RV32 编译并核对 `riscv:rv32`）。
- Python：系统 `python3` 3.10.12 只有 numpy 1.21.5；conda 运行时里 `conda_base`（3.12）与
  `conda_rag`（3.10）装有 `torch 2.5.1+cu121`，但**所有运行时都没有 TensorFlow**。因此本机
  无法复算官方 TF checkpoint，只能依赖既有 `software/weights/kwt1_fp32.npz`（162 张量 FP32 导出）。

## 4. 官方平台缺失时能验证到哪一层

| 层级 | 能力 | 状态 | 说明 |
| --- | --- | :-: | --- |
| L1 | 计算核/输出级功能仿真，逐位对照软件整数模型 | 可做 | 只用仓库内 RTL + Verilator/Icarus |
| L2 | 模块级逻辑综合（Yosys 通用库 / iEDA+Nangate45 代理库） | 可做 | 结果是**代理工艺**相对参考，非签核 |
| L3 | E203 RTL 仿真接入（Hummingbird 固定提交，含 NICE 仿真模型） | 部分 | 是开源核仿真，不等于官方 SoC 封装 |
| L4 | 150 MHz 时序签核 | **阻塞** | 缺官方 PDK/单元库/SDC |
| L5 | ≤2.25 mm² 面积与功耗签核 | **阻塞** | 缺指定 SRAM 宏与官方工艺 |
| L6 | 官方 600 条评测集 NRMSE/准确率 | **阻塞** | 评测集未发布 |

## 5. 口径声明（写进报告时必须遵守）

**可以宣称**：Verilator/Icarus 下的模块级/计算核级功能仿真；Yosys/iEDA+Nangate45 代理库下的
相对综合量级（注明代理工艺）；官方锁定提交与权重的哈希一致性；Hummingbird E203 RTL 级
CPU 仿真行为。

**不能宣称**：在官方 E203 SoC 平台完成 SoC 集成仿真；达到 150 MHz；满足 ≤2.25 mm² /
功耗签核；官方 600 条评测集上的精度。**任何模拟模型或代理库的数字都不能代表真实 SoC
或真实工艺结果。**

## 6. 第 4-7 项接口约定（工程约定，非题目唯一答案）

第 4-7 项共享本科第 3 项已定的接入方式，这里补充研究生组自己必须冻结的约定：

- **接入接口**：System Bus 或 NICE 二选一。仓库现有 `hardware/rtl/control/kwt_nice_system_model.sv`
  与 `software/driver/kwt_nice_driver.*` 走 NICE 仿真模型；正式选择需与 SoC 驱动寄存器/命令
  布局一致并冻结。
- **数值协议**：注意力分数矩阵 `99×99`、单头特征维 64；Q/K/V 投影、缩放、逐行 Softmax、
  概率加权、输出投影；`64→256→64` FFN + GELU + 两处残差 + 两处 LayerNorm（PostNorm）。
  位宽、尺度、零点、舍入/饱和由队伍选定并冻结；GELU、注意力近似误差须单独记录。
- **层间数据与参数调度**：12 层顺序执行，E203 下发命令并搬运，编码器内数值计算由加速器
  执行；逐层参数更新与层间张量传递必须在同一冻结配置下验证。
- **性能/面积口径**：运行周期、吞吐、平均功耗、单任务能耗、SRAM 访问量与参数加载开销
  必须来自**同一设计配置、同一综合网表、规定测试条件**；不能拼接不同配置的最优值。

## 7. 阻塞项（本轮未解决）

1. 官方 SoC 平台、PDK、指定 SRAM 宏（16bit×128/1024/4096）、150 MHz SDC、官方 600 条评测集
   全部缺失 → L4/L5/L6 无法签核。
2. 本机无 TensorFlow/PyTorch 运行时 → 不能在本机复算官方 checkpoint；依赖既有 `kwt1_fp32.npz`。
3. Speech Commands v0.02 数据集 tar 未下载（被 `.gitignore` 排除，按 `资料/补全大文件.sh --dataset` 获取）。
4. iEDA 可执行但不打印版本，也**未**跑完整 STA 流程；`OpenSTA` 独立二进制缺失。
