# 脉冲矩阵原型

当前面向本科组的第一前馈线性层：`X[99,64] × W[64,256] + b[256]`。阵列为 8×16 PE，A 从左侧进入并向右传播，B 从上侧进入并向下传播；valid 随数据传播。当前系统路径使用乘积脉冲阵列和阵列外的行分 bank CACC；每个 PE 不再存放自己的跨 K 累加器。原 `kwt_pulse_array`/`kwt_tile_engine` 保留作第一版与格式实验。

针对完整 FFN1 任务的数据复用、8 行广播计算核、可复用 Verilator 向量和同边界综合比较，见 [`architecture/README.md`](architecture/README.md)。

## 模块

| 文件 | 作用 |
| --- | --- |
| [`rtl/sram/kwt_tile_buffers.sv`](rtl/sram/kwt_tile_buffers.sv) | A、B、输出三块独立的 16 bit×1024 同步缓存行为模型 |
| [`rtl/pulse_array/kwt_pulse_array.sv`](rtl/pulse_array/kwt_pulse_array.sv) | 参数化二维阵列及边界错拍，供整数与浮点共用 |
| [`rtl/pulse_array/kwt_pe.sv`](rtl/pulse_array/kwt_pe.sv) | 最低层计算模块；`FORMAT=0` 为有符号整数乘加，`FORMAT=1` 为 FP32 乘加 |
| [`rtl/pulse_array/kwt_tile_engine.sv`](rtl/pulse_array/kwt_tile_engine.sv) | 第一版单块控制器，连接三块 SRAM、8×16 INT16 阵列和结果写回 |
| [`rtl/pulse_array/kwt_product_array.sv`](rtl/pulse_array/kwt_product_array.sv) | 8×16 乘积脉冲阵列，只传播操作数并产生带坐标的乘积，不保留部分和 |
| [`rtl/pulse_array/kwt_cacc.sv`](rtl/pulse_array/kwt_cacc.sv) | 集中式行分 bank 累加器；每行每拍接收一个乘积，按输出列读改写 INT32 部分和 |
| [`rtl/pulse_array/kwt_pingpong_cacc_engine.sv`](rtl/pulse_array/kwt_pingpong_cacc_engine.sv) | 双页 A/B 缓存、乘积阵列、CACC 和 INT32 结果写回组成的新单块引擎 |
| [`rtl/sram/kwt_operand_pingpong.sv`](rtl/sram/kwt_operand_pingpong.sv) | A、B 各自两页的同步缓存行为模型；计算读活动页时可写另一页 |
| [`rtl/pulse_array/kwt_postprocess.sv`](rtl/pulse_array/kwt_postprocess.sv) | INT32 偏置加法及可选 ReLU |
| [`rtl/nonlinear/`](rtl/nonlinear/) | GELU、Softmax、LayerNorm 的独立定点功能原型 |

阵列参数为 `ROWS`、`COLS`、`DATA_W`、`ACC_W`、`FORMAT`。整数 PE 可用 `DATA_W=8/16/32`；FP32 PE 要求 `DATA_W=ACC_W=32`。整数结果按 `ACC_W` 截断。FP32 使用乘法后加法的两次舍入，对次正规数清零。数据宽度变化由阵列总线参数处理，具体数值规则只在 PE 中。

| 格式 | 乘法流水 | 加法流水 | 同一 PE 累加项的最小发射间隔 |
| --- | ---: | ---: | ---: |
| INT8 | 2 拍 | 1 拍 | 3 拍 |
| INT16 | 4 拍 | 2 拍 | 6 拍 |
| INT32 / FP32 | 8 拍 | 4 拍 | 12 拍 |

整数乘法按 4 bit 乘数片段逐级累加部分积；FP32 将 24 bit 尾数乘法分成六段，再做规格化与舍入。加法也分段并注册中间结果。`acc_update` 表示对应 PE 的结果在该拍写入。单 PE 只有一份部分和，发射控制必须遵守上表间隔，`acc_clear` 应在阵列流水排空后使用。

缓存保持竞赛指定的 16 bit 字宽。INT32/FP32 均需两个字保存一个数；INT16 为一个字。INT8 的存储打包、各格式的输入装载与输出写回尚待控制器确定。当前缓存数组是单口、同步读一拍的 SRAM 行为模型；是否在读后增加寄存器，要等赛题存储宏和装载控制接入后看时序，不能据此计算宏面积。

## 单块控制器

主机空闲时将 A 按 `A[k][r]` 写入地址 `8k+r`，将 B 按 `B[k][c]` 写入地址 `16k+c`。`start` 与 `k_len`、行列掩码、偏置和 `relu_en` 同时给出；本版要求这些配置在 `busy` 期间保持稳定。`k_len` 范围为 0～64。控制器逐个 K 读出输入、发射脉冲，等阵列流水排空后读取 128 个累加结果。结果按 `out[r][c]` 写入结果 SRAM 的 `2(16r+c)` 和 `2(16r+c)+1` 地址，低 16 bit 在前。`done` 为一个周期；主机仅在 `busy=0` 时访问缓存。

输出为有符号 INT32 回绕累加及偏置加法；可选 ReLU。KWT-1 前馈层实际使用 GELU，当前没有定义定点缩放和 GELU 近似，因此这个后处理模块尚不能用于完整模型精度验收。控制器当前只连接 INT16 输入和 INT32 累加；PE/阵列的其他格式仍可独立仿真。输入 SRAM 是单口，读取一组 A/B 边界向量需要 17 拍，随后逐个 K 等待流水排空，尚未做重叠搬运。

## 脉冲矩阵仿真

运行 `bash hardware/rtl/pulse_array/tb/run.sh`。测试台位于 [`rtl/pulse_array/tb`](rtl/pulse_array/tb/)，分别运行 INT8、INT16、INT32 和 FP32 四种配置。每种配置逐 PE 检查传播、结果、完成拍数、掩码、清零、复位和最小间隔连续发射；四种配置的 Icarus Verilog 仿真均通过。Verilator 对 FP32 阵列的 lint 无错误，仍有特殊值判断未使用符号位的提示。

运行 `bash hardware/rtl/pulse_array/tb/run_engine.sh` 可仿真第一版单页引擎。新引擎用 `bash hardware/rtl/pulse_array/tb/run_pingpong_cacc_verilator.sh` 运行 Verilator 测试：逐项核对两页共 256 个 INT32 结果，覆盖 K=2 部分和累加、CACC 清空与排空、活动页写保护、计算期间预装另一页。`iverilog -g2012 -s tb_kwt_cacc_forward -o /tmp/kwt-cacc-forward.vvp hardware/rtl/pulse_array/kwt_cacc.sv hardware/rtl/pulse_array/tb/tb_kwt_cacc_forward.sv && vvp /tmp/kwt-cacc-forward.vvp` 单独验证同列连续乘积的转发和清空。CACC 是每行一个读改写通道（8 个行 bank），不是每个 PE 一个累加器；乘法器仍为 128 个，跨 K 的 32 bit 加法通道为 8 个，并保存 128 个 INT32 输出部分和。当前按行列地址读改写，尚未实现物理 shift-ring；尚未映射到赛题 SRAM 宏或验证宏的读延迟/端口配置。

单口 A/B 存储器每拍读取一个 lane；控制器在发出下一 lane 地址时捕获上一 lane，把每个 K 的读取阶段由逐 lane 的两拍缩成连续读取。相邻 K 的启动间隔为 18 拍，不再在每个 K 后等待整阵列排空；只在最后一个 K 后排空。真实 E203 的 K=64 首、尾块仿真均通过，单块任务耗时由 7067 降至 4280 拍，减少约 39.4%。这一周期数包含结果写回和当前简化 DMA/总线等待，不是目标 SoC 的最终性能。

## 非线性单元功能原型

运行 `bash hardware/rtl/nonlinear/tb/run.sh`。三个模块各自接受一组 `in_valid` 输入并在下一个时钟边沿更新输出和 `out_valid`，目前未连接进单块控制器。GELU 使用 `x·sigmoid(1.703125x)` 和间隔 0.25 的指数查找表；Softmax 先减最大值，再查同一张表并做整数除法归一化；LayerNorm 逐向量计算均值、方差、整数平方根及 gamma/beta 仿射。

输入和 GELU、LayerNorm 输出均为有符号 Q8.8；Softmax 输出为无符号 Q0.15。Softmax 默认 16 元素，LayerNorm 默认 64 元素，均可改 `LANES` 参数。LayerNorm 的 epsilon 取 Q16.16 可表示的最小正数 `1/65536`，与 KWT-1 源码中的 `1e-6` 不同。查表、除法、平方根当前是功能写法，尚未流水切分，也没有综合时序和面积结果；这些格式及近似仍不能证明整模型误差满足 3%。

目前完成的是缓存、计算阵列和 INT16 单块控制路径；DMA 已在下述系统仿真模型中接入，并有真实开源 E203 CPU 执行 NICE 指令的测试，但没有竞赛平台的正式总线接口和提交用的软件数值模型。单块仿真通过，不代表 3% 精度已验收；150 MHz 时序和整机面积也尚未评估。浮点 PE 的结构仅供当前 FP32 方案起步，最终是否保留 128 个并行 FP32 MAC 要看综合结果。

## DMA 与外部内存

[`rtl/dma/kwt_dma.sv`](rtl/dma/kwt_dma.sv) 接收一次搬运描述符：`start`、方向 `writeback`、目标缓存 `bank`（0=A、1=B、2=输出）、16 字节对齐的 `ext_byte_addr`、缓存起始字地址 `sram_word_addr` 和 16 bit 字数 `word_count`。装载允许三个缓存；写回只允许输出缓存。`busy` 表示搬运中，`done` 是一拍完成脉冲，`error` 在非法描述符完成时置位，并由下次 `start` 清除。启动时会检查对齐、缓存和外部内存范围；长度为零会报错。工作期间保持描述符稳定，且不要同时让计算控制器访问同一块单口 SRAM。

外部总线一次传送 128 bit，读请求由 `bus_valid && bus_ready && !bus_write` 接受，随后以 `bus_rvalid` 返回数据；写请求带逐字节 `bus_wstrb`，可处理不足 8 个 16 bit 字的尾拍。缓存端直接给出三组 `en/we/addr/wdata/rdata` 信号，无额外 SRAM 命令包。单口 SRAM 每拍只能读或写一个字，搬运的有效速率因此低于 128 bit/拍。`rtl/dma/kwt_ext_memory_model.sv` 是仿真用 128 bit 存储模型，读响应在请求被接受后的下一拍有效。

默认容量和宽度在 [`rtl/sram/kwt_memory_defs.svh`](rtl/sram/kwt_memory_defs.svh) 中：外部内存按 **512 KiB（512×1024 字节）** 解释，缓存按 **16 bit×1024 字** 建模，总线为 **128 bit**。这些宏可在编译时覆盖，模块参数也可逐实例覆盖；DMA 测试台把外部内存缩到 256 字节以加快仿真。运行 `bash rtl/dma/tb/run.sh` 检查三个缓存装载、输出写回、尾拍和越界。系统模型现在把 DMA 接到新引擎的 host 端口；系统层仍按 A、B、偏置、计算、写回顺序运行。双页并行预装在引擎模块测试中覆盖，系统 DMA 还没有安排下一块数据与当前计算重叠。当前行为模型未对接赛题提供的实际存储宏及 E203 总线。

## CPU 命令与数据总线仿真模型

赛题原图第 3 页写的是“AI Engine 通过 System Bus **或** NICE Interface 接入，参赛队选择一种方式完成集成”；第 3 页还说明总线和外设采用组委会平台工程。现有题目图片没有 System Bus 的信号定义、协议、实际数据位宽或仲裁规则，工作区也没有组委会平台工程。因此 [`rtl/control/kwt_tile_system_model.sv`](rtl/control/kwt_tile_system_model.sv) 使用**抽象 CPU 命令口**与现有 **128 bit DMA 仿真总线**，用于先验证控制链；它不是 NICE 或竞赛 System Bus 的引脚实现。“NICE 传控制、System Bus 搬数据”的双通路想法也尚未证明符合赛题“选择一种方式”的集成要求，正式方案须在平台文件到手后确认。

模型每次执行一个 8×16、INT16 输入和 INT32 输出的矩阵块。CPU 在 `cmd_valid && cmd_ready` 时提交并锁存：A、B、偏置和输出的**外存字节地址**，`k_len`（1～64）、8 位行掩码、16 位列掩码、`relu_en`。四个地址均要求 16 字节对齐；配置错误返回 `rsp_status=1`。`rsp_valid` 保持到 `rsp_ready`；`rsp_status=0` 表示成功，`2` 表示 DMA 描述符错误。该模型未定义 CPU 指令编码、描述符内存格式或中断。

| 步骤 | 搬运或计算 | 片上位置 |
| --- | --- | --- |
| 装 A | `8 × k_len` 个 16 位字 | A SRAM 字地址 0 起，按 `A[k][r]` |
| 装 B | `16 × k_len` 个 16 位字 | B SRAM 字地址 0 起，按 `B[k][c]` |
| 装偏置 | 16 个 INT32，即 32 个 16 位字 | 输出 SRAM 字地址 256～287，低字在前 |
| 计算 | 先把偏置读入 512 位寄存器，再运行单块控制器 | 输出 SRAM 字地址 0～255 |
| 写回 | 256 个 16 位字，即 128 个 INT32 | 从输出 SRAM 字地址 0 起写至外存 |

系统模型采用顺序搬运和独占 SRAM，不重叠 DMA 与计算。A/B/偏置在外存中必须已经按上表连续排好；当前 DMA 不支持从普通行主序大矩阵中按步长抽块。新引擎为 DMA 保留输入写端口和输出缓存主机端口，供 DMA 装载、偏置读入及结果写回。仿真外存为 `kwt_ext_memory_model`，默认 `ready=1`、读请求后一拍返回；实际系统的等待、错误、总线宽度转换和仲裁尚待平台适配。

在 `hardware` 目录运行 `bash rtl/control/tb/run.sh`。测试把外存缩为 2 KiB，验证两项 K 累加、偏置、掩码、ReLU、128 个 INT32 写回结果，以及命令对齐错误和 DMA 越界错误。`bash rtl/control/tb/run_verilator.sh` 用 Verilator 对同一条 DMA—新引擎—写回路径运行同一测试台。此测试只证明上述仿真模型的单块路径，不代表 NICE、System Bus、整层 208 个块、GELU 或综合验收。

### NICE 指针命令的粗集成

[`rtl/control/kwt_nice_system_model.sv`](rtl/control/kwt_nice_system_model.sv) 在上述单块模型前增加一层 NICE 风格的请求/响应和描述符读取。仿真约定 Custom-0、`funct3=110`、`funct7=0` 表示 `SUBMIT`；`nice_req_rs1` 是**共享 RAM 中 16 字节对齐的描述符字节地址**，`nice_req_rs2` 暂不使用。收到请求后，控制器先通过同一条简化 128 bit 总线连续读两个 beat，再把字段交给单块模型。描述符大小固定 32 字节，小端字布局如下：

| 字节偏移 | 内容 |
| ---: | --- |
| `0, 4, 8, 12` | A、B、偏置、输出的外存字节地址，均为 32 bit |
| `16` | `k_len` 在 bit `[6:0]`，`relu_en` 在 bit `7` |
| `20` | 行掩码在 bit `[7:0]`，列掩码在 bit `[23:8]` |
| `24, 28` | 保留，写零 |

`nice_req_valid && nice_req_ready` 接收一条请求；本版**等整个块的结果写回后**才发出响应，`nice_rsp_valid` 保持到 `nice_rsp_ready`。`nice_rsp_data[1:0]` 的 `0/1/2/3` 依次表示成功、块配置错误、DMA 错误、指令或描述符地址错误；非零时 `nice_rsp_err=1`。这是阻塞式命令，CPU 在等待这条指令完成时不能借此模型并行调度下一块。描述符读取和 DMA 独占同一仿真总线，仿真 RAM 总是可接受请求、读数据在下一拍返回；没有真实总线仲裁、超时、访存错误和 32/128 bit 宽度桥接。真实 E203 仿真使用了此模型作为测试专用 NICE 单元，但没有证明竞赛平台的双接口集成满足要求。

运行 `bash rtl/control/tb/run_nice.sh`：测试从 2 KiB 仿真 RAM 读取描述符，计算并核对 128 个输出，同时检查非法指令、描述符越界、块配置错误、DMA 越界及响应背压。此处的指令编码仅是模型约定，拿到组委会平台后仍须核对 NICE 译码、共享 SRAM 地址映射和最终选择的集成方式。

### 真实开源 E203 CPU 测试

运行 `bash rtl/control/tb/run_e203.sh`。脚本使用上级目录 `资料/e203_hbirdv2_full/` 中的**原版开源 E203 CPU RTL**，以 RV32 汇编编译 [`e203_kwt_program.S`](rtl/control/tb/e203_kwt_program.S)，装入 CPU 内部 ITCM。CPU 执行 `li a0, 0x500` 和一条 Custom-0 指令，以 `rs1=a0` 提交共享 RAM 内的 32 字节描述符，并用 `rd=a1` 接收完成状态。测试台把 A、B、偏置和描述符预置到 2 KiB 模拟共享 RAM，检查实际 NICE 请求/响应各一次、`a1` 从预置值变为返回的零状态，以及 128 个 INT32 写回结果。

[`e203_kwt_nice_core.sv`](rtl/control/tb/e203_kwt_nice_core.sv) 是测试专用的同名 NICE 单元，替换开源工程自带的示例协处理器；**上游 CPU RTL 没有改动**。它把 CPU 的真实 NICE 引脚接到本项目的控制模型，数据侧仍使用 `kwt_ext_memory_model` 的简化 128 bit 总线。测试程序和输入数据由 testbench 预装，没有验证 CPU 经系统总线写入描述符、真实共享 SRAM 仲裁、NICE 访存通道或组委会提供的 SoC 平台。脚本需要 `iverilog`、`vvp`、`riscv64-linux-gnu-gcc`、`riscv64-linux-gnu-objcopy` 和 `xxd`。

### FFN1 整层功能仿真

在 `hardware` 目录运行 `bash rtl/control/tb/run_e203.sh --full-layer`。该测试仍以 Icarus Verilog 编译 SystemVerilog，并运行原版 E203 CPU RTL。CPU 的 RV32 程序连续提交 208 个 NICE 描述符；行方向为 `ceil(99/8)=13` 块，列方向为 `256/16=16` 块，每块完成后检查指令返回状态。测试台在 512 KiB 仿真 RAM 中预置 A、B、偏置、描述符，将 `99×64` 与 `64×256` 乘加，并在 CPU 完成后用独立的整数循环核对 25344 个有效 INT32 输出和 1280 个填充位置。

调试时可运行 `bash rtl/control/tb/run_e203.sh --tiles START COUNT`，只让 CPU 执行从 0 开始编号的指定连续块，并逐项检查该区间的输出。例如 `--tiles 0 1` 验证首块，`--tiles 207 1` 验证最后一个含填充行的块，`--tiles 16 16` 验证第二组行块。本轮首、尾单块在 Icarus 下各运行约 3 秒；将 208 块拆成多个进程并行可减少多核机器上的墙钟时间，但总仿真工作量不会减少。平时先跑阵列/单块 Verilator 测试、E203 首尾块及修改涉及的块，最终版本再跑完整 208 块。

本轮在仓库根目录用 `seq 0 16 192 | xargs -P 8 -I{} bash hardware/rtl/control/tb/run_e203.sh --tiles {} 16` 八路并行跑完 13 个区间，全部通过；合计核对 25344 个有效输出和 1280 个填充值。各区间独立启动 CPU，因而该结果覆盖所有块的数值与区间内连续指令，却不覆盖区间边界的连续执行状态；最终仍需单次运行 `--full-layer`。

原 PE 本地累加版本的 Icarus Verilog 运行结果：`PASS real E203 full FFN1 layer: tiles=208 outputs=25344 padding=1280 task_cycles=1257772 total_cycles=1257785`。该周期数从第一条 NICE 请求算至最后一条响应，基于顺序搬运、单口 SRAM 和总是就绪的简化总线，不是目标 SoC 的性能估计。当前集中 CACC 和新流水版本重新用 13 个独立区间覆盖全部 208 块，共核对 25,344 个有效输出与 1,280 个填充位置，全部通过；每个 16 块区间的 `task_cycles=68540`。这不是单次连续运行整层，不能直接把区间周期数相加视为整层实测耗时。

赛题本科组 FFN1 的矩阵尺寸为 `99×64 × 64×256 + bias`。本测试使用可复现的小整数合成数据，覆盖该尺寸和 208 次搬运、计算、写回；它没有使用赛题的预训练权重和真实层输入，也没有计算 GELU、12 层 NRMSE 或验证 150 MHz。描述符和输入仍由 testbench 预装，数据总线仍为简化模型。

### 初步综合与路径分析

本机有 Yosys 0.63 和 `Nangate45_typ.lib` 示例单元库。在 `hardware` 目录下可先对不含仿真 RAM 的 DMA 运行：

```sh
yosys -Q -T -p 'read_verilog -sv -I rtl/sram rtl/dma/kwt_dma.sv; hierarchy -check -top kwt_dma; synth -top kwt_dma; ltp -noff; dfflibmap -liberty /home/yang/yosys-sta/pdk/nangate45/lib/Nangate45_typ.lib; abc -liberty /home/yang/yosys-sta/pdk/nangate45/lib/Nangate45_typ.lib; stat -liberty /home/yang/yosys-sta/pdk/nangate45/lib/Nangate45_typ.lib'
```

本次试跑通过 `check`，映射后为 1615 个单元、库面积 3406.396（只含 DMA）。`ltp -noff` 在通用门级网表里发现 32 级组合路径，从 `word_count[3]` 到 `lane[1]`，经过启动条件和控制选择逻辑；这只表示逻辑层数，不是纳秒延迟，也不能据此判定 150 MHz 达标。真正的关键路径应为最终顶层指定 6.667 ns 时钟、输入输出延迟和存储宏时序后，用静态时序分析读取起点、终点、逐级延迟及 slack；布局布线后还要重跑。当前未找到本机 OpenSTA/OpenROAD 可执行文件，也没有竞赛平台的目标工艺库和 SRAM 宏，所以示例库面积不用于比赛面积结论。

新引擎可在仓库根目录运行 `bash hardware/synth/run_engine.sh`；脚本把粗综合与 CACC 标准单元映射日志写入 `/tmp/kwt-engine-synth/`。粗综合保留存储器：8×16 阵列有 128 个乘法器，CACC 有 8 个 INT32 加法器和 8 个 1R1W 部分和 bank；完整单块引擎共识别 11 个存储器。执行 `bash hardware/synth/run_engine.sh --full-flop-map` 可额外把整个引擎的存储器展开成寄存器，用 Nangate45 示例库做门级映射。此次 `check -assert` 无问题；CACC 映射为 23,403 个标准单元、库面积 47,861.114，完整引擎映射为 529,875 个标准单元、库面积 1,007,428.45。完整引擎里有 86,320 个普通 DFF 和 12,383 个带复位 DFF，主要来自将 A/B、输出和部分和存储器展开；没有 SRAM 宏时，这个面积数字不能作为最终芯片面积或频率结论。150 MHz 仍需目标库、存储宏与时序约束下的 STA 验证。
