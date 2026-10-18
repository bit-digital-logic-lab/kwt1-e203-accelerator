# 冻结配置与输入哈希（power-activity 分支）

本文固定 `hardware/power/` 活动证据的数值/结构配置、向量来源、工具版本与频率假设，
使 `run_power_activity.sh` 的产物可复现到同一冻结基线。

## 1. 冻结数值/结构配置

| 项目 | 取值 |
| --- | --- |
| 量化方案 | A12W10 静态对称，零点 0，舍入 nearest ties-to-even |
| 激活 | 12 bit 有符号（Qmax 2047），每层一个训练集最大绝对值尺度 |
| 权重 | 10 bit 有符号（Qmax 511），按输出通道独立尺度 |
| 容器 | INT16 字（W10 放在 INT16 容器，物理 10 bit 打包未实现） |
| 累加/bias/输出 | 有符号 INT32；output layout 低 16 位在前 |
| 阵列 | 8×16 PE（`kwt_tile_system_model` 单块 8×16；整层 13×16=208 tile） |
| GELU | 旁路（本科必做只到第一处前馈线性层） |
| 基准任务 | `X[99,64] @ W[64,256] + b[256] -> Y[99,256]` |

## 2. 向量来源与哈希

证据使用**真实验证样本**的首层 FFN1 A12W10 参数包（validation-index 0，
`four/750e3e75_nohash_0.wav`，label `_unknown_`），由
`hardware/rtl/control/tb/prepare_real_e203_vectors.py` 生成，源向量目录位于主工作树：

`/home/yang/Desktop/ic设计竞赛/hardware/architecture/results/real_e203_validation0_vectors/`
（已复制 layer01 到本分支 `hardware/power/vectors/real_e203_validation0_layer01/`）。

本分支内文件 sha256：

| 文件 | sha256 |
| --- | --- |
| `layer01_a16.hex` | `2077cb910cdc338fc12f5fbd0379940873f97bdfe83efadf89d35c15531026b8` |
| `layer01_w16.hex` | `0fa421247d828b5f6d97dfb359447e2af6cb4c4592587442dbb9d0fd61948ae0` |
| `layer01_bias32.hex` | `e5b6acfb7537c060891d8c8e4ba1b8bdc8adbbd0cafb1cd6b6faac8c1ce1a3df` |
| `layer01_expected32.hex` | `97c14656bd26804987235c235d20ec99942959208e3bc2d622e6b906dd616e35` |
| `manifest.json` | `343692199b82b60c0739e2e7e7b0cfa6abc98fb75f3f4e250dc54bf41fbb1bd7` |

manifest 记录的源哈希：

| 源 | sha256 |
| --- | --- |
| 样本归档 `quant_samples_12labels_40.npz` | `43d639fb64ea8d907db967b071010e41e68827368532880ba8a99b92769bbae8` |
| A12W10 参数归档 `ffn1_a12w10_static_sym_12labels_40.npz` | `10c481c7a79d0ee8063ad807f8c0f685769c2c8e1dbcdb5f33079919dfeef367` |
| FP32 权重 `kwt1_fp32.npz` | `8f359abf7adbe287b8f89fca98af46194f98f90f2146784731bff551a155825a` |

layer01 该向量：NRMSE `0.10170796%`，`max_abs_expected_int32=4593325`，
保守累加上界 `67389605 < INT32_MAX`。

**重要边界**：真实验证样本用的是 `_12labels_40`（480/480）标定的 A12W10 参数包
（`10c481…`）。本分支 `software/results/ffn1_a12w10_static_sym.npz`
（`a77f60…`，240/240 标定，对应 `ffn1_a12w10_static_sym.json` `b3a8908…`）是另一份
A12W10 参数；两者数值方案相同但标定样本集不同，**不可混用**。本目录活动证据使用前者，
因为它是与 E203 真实样本向量配套的冻结包。

## 3. 工具与上游

| 项目 | 版本/路径 |
| --- | --- |
| Icarus Verilog / vvp | 14.0（oss-cad-suite s20251012），`/home/yang/yosys-sta/oss-cad-suite/bin` |
| RISC-V 工具链 | `riscv64-linux-gnu-gcc` 11.4.0 (Ubuntu 22.04) |
| Python | 3.10.12 |
| 上游 E203 RTL | `/home/yang/Desktop/ic设计竞赛/资料/e203_hbirdv2_full`（未改动） |
| 本分支基线提交 | `fcb4b72` |

## 4. 频率与时间假设

- 仿真时钟：tb `always #5 clk=~clk`，周期 10 时间单位；VCD `$timescale 1s`（Icarus
  默认，见 `README.md` 边界说明）。**周期数与时间单位无关。**
- 题目系统时钟目标 150 MHz（周期 6.666… ns）。若把本证据的任务周期数按 150 MHz 换算：
  `4025 cycles × 6.6667 ns ≈ 26.83 µs/tile-task`（**仅算术换算**）。
- **未验证** 150 MHz：无组委会目标库、无 SRAM 宏时序、无 SDC/STA、无布局布线。

## 5. 功耗/能耗

`not_computable`。理由：无 PDK、无互连功耗模型、无 SRAM 宏功耗模型；未标注电路与未建模
SRAM 不能按 0 处理。示例库 Nangate45 综合面积（见
`hardware/architecture/量化格式_硬件成本与周期对照_20261017.md`）仅为相对趋势，不能用于
面积或功耗结论。
