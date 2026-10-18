# WORKER_REPORT（codex/kwt-transport-abi）

## 任务与边界
冻结并逐字节校对 **PC → 片上消费端（题目称 “RCL”）的 FFN1 第一线性层数据传输 ABI**：
PC 端离线做 MFCC `98×40` 与首层编码器输入 `99×64`，片上只执行 `99×64 · 64×256 + 256 → 99×256`。
只新增 `software/transport/`（代码/测试/文档）与根目录本报告；**未改**既有预处理脚本、驱动、
RTL。判据来自真实接口，不臆造未定义的 SoC ABI。

术语澄清：仓库内无 “RCL” 定义；本报告把 ABI 限定为仓库真实存在的 **(1) 外存/SRAM 字节流布局**
与 **(2) 32 字节 tile 描述符**，不假设任何 pin/寄存器映射。

## 结论（证据层级）
- **V1 冻结格式经真实样本 pack→unpack→独立数值比对全部一致，29/29 检查通过（EXIT=0）。**
  证据层级：**宿主 Python 字节/整数级** + 与既有 RTL 常量逐条对齐；**不是** SoC/总线联调，
  **不是** 综合/时序/功耗。
- **发现 2 项“现状不支持”的字段**（§发现），已在隔离的候选 V2 中显式版本化，未改现状。

## 冻结格式（依据 RTL/驱动实接口）
- 小端；A/W 有符号 INT16 容器，bias/Y 有符号 INT32。
- A `99×64` 平铺 `(rt,k,r)` 13×64×8 = 13312 B，stride `rt*1024`；
  W `64×256` 平铺 `(ct,k,c)` 16×64×16 = 32768 B，stride `ct*2048`；
  bias 256×4 = 1024 B，stride `ct*64`；Y `99×256` 平铺 `(tile,r,c)` = 106496 B，stride `tile*512`。
- 所有 tile 起始/长度 16 字节对齐，总域不越 512 KiB；末行块 3 行有效，`row_mask=0x07`。
- 描述符 32 B：字节 0/4/8/12=A/B/bias/out 地址；字节 16=`K|ReLU<<7`（K=64→0x40）；字节 20=行掩码，21-22=列掩码。
- **无饱和**：INT32 回绕加法；安全性来自导出期保守上界（本层 67,389,605 < 2^31）。

## 实际执行的命令与退出码
| # | 命令 | 退出码 | 结果 |
| --- | --- | --- | --- |
| 1 | `python3 software/transport/run_transport_check.py` | **0** | `RESULT PASS (29 checks)`；见下 |

关键检查（真实数据：`sample_00_encoder.npy` 99×64 FP32 sha256 `489dc7c2…4a27c058`；
参数 `ffn1_a12w10_static_sym.npz` sha256 `a77f60fb…be1918e`，与 `.json` `parameter_sha256` 一致）：
- A/W/bias/Y pack→unpack→repack 逐字节相等；端序 `A[0,0]=-195` → `3d ff`。
- 独立 INT64 GEMM 与逐 tile INT32 累加逐位相等（25,344 输出，max|diff|=0）。
- 208 描述符往返一致；out 区末址 `0x2a000 < 0x80000`；越界/未对齐被拒。
- FP64 解码 vs 冻结 FP32 权重参考：**NRMSE=0.119933%**；本样本激活零截断。
- 无饱和演示：`INT32_MAX+1` 回绕为 `-2147483648` 且容器保留。
- 负例：单字节翻转→SHA 变化+元素改变；截断 1 字节→`ValueError`；V2 载荷翻转→CRC 拒绝。

blob sha256（写 `/tmp/opencode/transport_abi/`，不入库）：
A `b24dcaba…0ca5b538`、W `2ad48c26…b759d1faa4`、BIAS `a4c7207e…e4333497`、
Y `3a443acf…c6063154`、V2 `9d324a92…5daa4b859b`。
结果 JSON：`software/transport/results/transport_abi_v1_layer01.json`。
15 分钟检查登记：`.ai-checks/checkins/transport-abi.json`（complete）。

## 发现（现状不支持，已在候选 V2 中区分，未改现状）
1. **W 非 10 位物理存储**：`.npz` 用 INT16 容器装 W10（32768 B/层），文档称 240 KiB 只是
   紧密 10 位打包理论容量，打包/解包逻辑未实现。
2. **传输流不自描述**：无 magic/版本/dtype/尺度/长度/CRC；尺度仅在旁路 `.json`/`.npz`，
   且 `.json` 的 `parameter_archive` 为陈旧绝对路径。
候选 **V2**（`transport_abi.StreamV2`，仅 PC 侧提案，RTL 未实现）：64 B 头+48 B/节目录+
  载荷 CRC32，新增 `SCALEX`/`SCALEW` 尺度节与可选紧密 10 位 `W10` 节（20480 B vs 32768 B，往返逐位一致）。

## 未证事项
- 只覆盖 **layer01 × 1 条真实样本**；未跑 12 层 × 多样本。
- X 用仓库已提交的真实 **首层编码器输入 `99×64`**；真正“注意力/残差/LayerNorm 之后”的 FFN1
  激活需完整 FP32 前向，本工作树无 torch，未重算——本层验证的是搬运/布局/整数数值契约，不是输入语义。
- 未接组委会正式 SoC/System Bus/NICE；未综合；无时序/面积/功耗/吞吐结论。
- 未在 HDL 仿真中重放 V1 字节流（既有 `hardware/rtl/control/tb/` 向量回归覆盖，本目录不重复）。
