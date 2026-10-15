# 脉冲矩阵原型

当前面向本科组的第一前馈线性层：`X[99,64] × W[64,256] + b[256]`。阵列为 8×16 PE，A 从左侧进入并向右传播，B 从上侧进入并向下传播；valid 随数据传播。各 PE 累加自己对应的输出元素。

## 模块

| 文件 | 作用 |
| --- | --- |
| [`rtl/sram/kwt_tile_buffers.sv`](rtl/sram/kwt_tile_buffers.sv) | A、B、输出三块独立的 16 bit×1024 同步缓存行为模型 |
| [`rtl/pulse_array/kwt_pulse_array.sv`](rtl/pulse_array/kwt_pulse_array.sv) | 参数化二维阵列及边界错拍，供整数与浮点共用 |
| [`rtl/pulse_array/kwt_pe.sv`](rtl/pulse_array/kwt_pe.sv) | 最低层计算模块；`FORMAT=0` 为有符号整数乘加，`FORMAT=1` 为 FP32 乘加 |

阵列参数为 `ROWS`、`COLS`、`DATA_W`、`ACC_W`、`FORMAT`。整数 PE 可用 `DATA_W=8/16/32`；FP32 PE 要求 `DATA_W=ACC_W=32`。整数结果按 `ACC_W` 截断。FP32 使用乘法后加法的两次舍入，对次正规数清零。数据宽度变化由阵列总线参数处理，具体数值规则只在 PE 中。

| 格式 | 乘法流水 | 加法流水 | 同一 PE 累加项的最小发射间隔 |
| --- | ---: | ---: | ---: |
| INT8 | 2 拍 | 1 拍 | 3 拍 |
| INT16 | 4 拍 | 2 拍 | 6 拍 |
| INT32 / FP32 | 8 拍 | 4 拍 | 12 拍 |

整数乘法按 4 bit 乘数片段逐级累加部分积；FP32 将 24 bit 尾数乘法分成六段，再做规格化与舍入。加法也分段并注册中间结果。`acc_update` 表示对应 PE 的结果在该拍写入。单 PE 只有一份部分和，发射控制必须遵守上表间隔，`acc_clear` 应在阵列流水排空后使用。

缓存保持竞赛指定的 16 bit 字宽。INT32/FP32 均需两个字保存一个数；INT16 为一个字。INT8 的存储打包、各格式的输入装载与输出写回尚待控制器确定。当前缓存数组是单口、同步读一拍的 SRAM 行为模型；是否在读后增加寄存器，要等赛题存储宏和装载控制接入后看时序，不能据此计算宏面积。

## 脉冲矩阵仿真

运行 `bash hardware/rtl/pulse_array/tb/run.sh`。测试台位于 [`rtl/pulse_array/tb`](rtl/pulse_array/tb/)，分别运行 INT8、INT16、INT32 和 FP32 四种配置。每种配置逐 PE 检查传播、结果、完成拍数、掩码、清零、复位和最小间隔连续发射；四种配置的 Icarus Verilog 仿真均通过。Verilator 对 FP32 阵列的 lint 无错误，仍有特殊值判断未使用符号位的提示。

目前完成的是缓存与计算阵列原型，尚无把缓存接到阵列的控制器、共享 SRAM 搬运、E203 接口和提交用的软件数值模型。阵列单元仿真通过，不代表整层计算或 3% 精度已验收；150 MHz 时序和面积也尚未综合评估。浮点 PE 的结构仅供当前 FP32 方案起步，最终是否保留 128 个并行 FP32 MAC 要看综合结果。
