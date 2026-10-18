# 16 行计算核接入 NICE-like E203/DMA 系统模型（M1 设计冻结）

> 范围：本科必做第 1–3 项中的 FFN1 线性层 `99x64 @ 64x256 + bias`。本文冻结 16 行核
> 如何接入与旧 8×16 路径相同的 NICE-like 命令/`128-bit` 仿真总线/16-bit SRAM/DMA 系统模型，
> 并逐条列出 `208→112` tile 控制差异与尚未闭合的接口缺口。
> 证据层级：RTL 行为仿真（Icarus / Verilator），**不是**组委会正式 SoC/System Bus/NICE 平台，
> **不是**综合网表，**不是** 150 MHz 时序或面积/功耗结论。

## 1. 为什么必须是 16 行核（而不是 8×16）

- 精度/性能证据（`quantized_validation240_rtl.md` 等）来自 `kwt_row_broadcast_tile`（16 行），
  但 `run_e203.sh` 编译的是旧的 8 行 `kwt_pulse_array` / `kwt_pingpong_cacc_engine`。
- “验证过的核”和“上板的核”必须是同一个。16 行核接入系统模型是消除该差距的前提。
- 16 行把行 tile 数从 `ceil(99/8)=13` 降到 `ceil(99/16)=7`，tile 总数 `208→112`。

## 2. 16 行核端口（`kwt_row_broadcast_tile`, ROWS=16, COLS=16, DATA_W=16, ACC_W=32）

| 端口 | 方向 | 宽度 | 说明 |
| --- | --- | --- | --- |
| `clk` / `rst_n` | in | 1 | 时钟/异步低复位 |
| `clear` | in | 1 | 清部分和有效位与 `product_valid_q`（不清 SRAM 数据） |
| `in_valid` | in | 1 | 每拍一个 `(k, col)` 乘积有效 |
| `row_mask` | in | **16** | 行 lane 有效掩码（旧 8 行路径为 8 位） |
| `a_rows` | in | **256** | 16 行 × 16-bit A 值，同一 k 的 16 行并行输入 |
| `b_word` | in | 16 | 当前 B 元素（列内广播） |
| `in_col` | in | 4 | 当前列 0..15 |
| `dump_req` | in | 1 | 部分和读出请求 |
| `dump_row` / `dump_col` | in | 4 / 4 | 读出坐标 |
| `dump_bias` | in | 32 | 该列 bias（与 `dump_req` 同拍给出） |
| `dump_valid` / `dump_data` | out | 1 / 32 | 读出结果（`dump_req` 后一拍有效）= 部分和 + bias |

计算时序：`k_len × COLS = 64 × 16 = 1024` 拍完成一个 16×16 tile 的 MAC；
读出 `ROWS × COLS = 256` 拍（每拍一发一收）。

## 3. 适配器命令接口（NICE-like，单 tile）

`hardware/rtl/row16_integration/kwt_row16_tile_adapter.sv`

| 信号 | 宽度 | 变化 vs 8 行模型 |
| --- | --- | --- |
| `cmd_a_addr/b_addr/bias_addr/out_addr` | 32 | 同（16 字节对齐） |
| `cmd_k_len` | 7 | 同（1..64） |
| `cmd_row_mask` | **16** | 8 行模型的 8 位 → 16 位（关键变化） |
| `cmd_col_mask` | 16 | 同 |
| `cmd_relu_en` | 1 | 接口保留；**16 行核未实现 relu（缺口 G5）** |
| `rsp_status` | 2 | 0=OK, 1=命令错, 2=DMA 错 |

DMA 搬运（复用未修改的 `kwt_dma` 与 `kwt_ext_memory_model`）：

| 阶段 | bank | 外部字节地址 | SRAM 字地址 | 字数 |
| --- | --- | --- | --- | --- |
| A 装载 | 0 | `cmd_a_addr` | 0 | `k_len×16` |
| W 装载 | 1 | `cmd_b_addr` | 0 | `64×16 = 1024` |
| bias 装载 | 2 | `cmd_bias_addr` | 512 | `16×2 = 32`（32-bit 拆两个 16-bit 字） |
| 输出写回 | 2 | `cmd_out_addr` | 0 | `256×2 = 512` |

内部 SRAM 布局（均 16-bit 字）：
- A：`a_mem[k*16 + row]`（tile 内 16 行 × 64 k）。
- W：`w_mem[k*16 + col]`（64 k × 16 列）。
- 输出/bias：`out_mem[2*(row*16+col)]`、`+1` 为 32-bit 结果低/高 16 位；
  `out_mem[512+2*col]`、`+1` 为 bias 低/高 16 位。

## 4. 地址图（适配器测试台冻结值）

| 区域 | 基址 | tile 步长 | 单 tile 大小 |
| --- | ---: | ---: | ---: |
| A | `0x1000` | 2048 B | `64×16×2` |
| W | `0x4000` | 2048 B | `64×16×2` |
| bias | `0x8000` | 64 B | `16×4` |
| 输出 | `0x10000` | 1024 B | `256×4` |
| 外部内存 | 512 KB | — | — |

## 5. `208 → 112` tile 控制变化

| 项 | 8 行路径 | 16 行路径 |
| --- | --- | --- |
| 行 tile 数 | `ceil(99/8)=13` | `ceil(99/16)=7` |
| 列 tile 数 | 16 | 16 |
| tile 总数 | 208 | **112** |
| tile 序号 | `rt*16+ct`，`rt∈0..12` | `rt*16+ct`，`rt∈0..6` |
| 满 tile `row_mask` | `8'hFF` | `16'hFFFF` |
| 尾 tile | `rt=12`，`row_mask=8'h07`（行 96..98） | `rt=6`，`row_mask=16'h0007`（行 96..98） |
| 单 tile 输出 | `8×16=128` 个 32-bit（512 B） | `16×16=256` 个 32-bit（1024 B） |
| 输出字节步长 | 512 B | **1024 B** |

## 6. 与既有 E203 路径的接口缺口（M2 只闭合单 tile，下述为仍需改造处）

- **G1 描述符 row_mask 位宽**：`kwt_nice_system_model.sv` FETCH1 把 `row_mask` 取在
  `bus_rdata[39:32]`（8 位）。16 行需 16 位，`col_mask`（现 `[55:40]`）需重新排布并扩展到 16 位。
- **G2 控制模型硬编码**：`kwt_tile_system_model.sv` 的 `cmd_row_mask[7:0]`、输出写回
  固定 256 字（8×16）与 tile 输出步长 512 B 都是 8 行口径，不能直接驱动 16 行核。
- **G3 核实例**：`kwt_tile_system_model.sv` 实例化的是 `kwt_pingpong_cacc_engine`，
  与 16 行核端口不一致；需要新增 16 行控制路径（本次以新文件 `kwt_row16_tile_adapter.sv` 给出）。
- **G4 A 读端口结构**：16 行核每拍并行读 16 个 A 值。适配器用寄存器阵列 16 路组合读实现，
  与赛题“16-bit 单口/双口宏”口径不符；正式实现需 16 字并行读（多宏拼宽或分行 bank），
  端口数与面积待证。
- **G5 relu 未实现**：16 行核无 relu；`cmd_relu_en` 仅保留接口，本次向量 `relu=0`。
- **G6 回归脚本**：`run_e203.sh` 的文件清单与 `tb_kwt_real_layer_e203.sv` 的 tile 表
  均按 8 行固定（13/208/512B/`row_mask=0xFF,0x07`）。**本次不修改**这些既有文件，
  16 行验证走独立 runner `hardware/rtl/row16_integration/tb/run_row16.sh`。

## 7. 验证边界

- M2 只覆盖 **单 tile**：tile0（`rt=0,ct=0`，全掩码）与尾 tile（`rt=6,ct=15`，`0x0007`）。
- 冻结 A12W10 真实向量：与旧 E203 路径同一 `layerNN_a16/w16/bias32/expected32.hex`；
  `expected32` 由软件独立生成，测试台不预计算 `XW+b`、不用 golden 当 DUT 输出。
- 12 层全量与索引进系统模型（DMA/驱动真实搬运）未做（M3，时间允许再做）。
- 周期数是该 NICE-like 仿真模型的观察值，不能当吞吐/能耗/频率结论；
  150 MHz 与面积达标需目标库/宏，未证。
